// Paths of the Woses: direct card movement, with the pinned source's cargo,
// identity, siege and overflow behavior. Ordinary movement is not fabricated.
bool epw_supported(const Snapshot& t,uint64_t card){return card!=missing&&text_is(t,obs_attr(t,card,"card_name"),"Paths of the Woses");}
bool epw_pending(const Snapshot& s){
    const auto p=obs_attr(s,s.root,"pending_card_option",false);
    return !ctx_none(s,p)&&text_is(s,field(s,p,"card_name",false),"Paths of the Woses");
}
bool cdm_enter(const Snapshot& s,const Snapshot& t,uint64_t u,std::string_view to);
bool cdm_fp_guard(const Snapshot& s,const Snapshot& t,uint64_t army,const std::string& to,const std::string& via);
uint64_t epw_options(Snapshot& s,const Snapshot& t){
    std::vector<uint64_t> rows;std::set<std::pair<std::string,std::string>> seen;
    for(const auto& choice:evt_paths_options(s,t))if(seen.insert(choice).second){
        const auto army=hr_army(s,choice.first);int64_t own=0,regular=0,elite=0;
        for(auto u:child_edges(s,obs_attr(s,army,"units")))if(ctx_unit_side(s,u)==0){++own;if(cdm_enter(s,t,u,choice.second))(enum_is(s,obs_attr(s,u,"kind"),"game_state.UnitKind","ELITE")?elite:regular)++;}
        if(regular+elite>0&&regular+elite==own&&cdm_fp_guard(s,t,army,choice.second,"")){
            for(int64_t r=0;r<=regular;++r)for(int64_t e=0;e<=elite;++e){
                if((r+e==0&&own>1)||(r==regular&&e==elite))continue;
                rows.push_back(ev_dict(s,{{"from_region",rr_string(s,choice.first)},{"to_region",rr_string(s,choice.second)},{"n_regular",rr_scalar(s,WOTR_INT,r)},{"n_elite",rr_scalar(s,WOTR_INT,e)}}));
            }
        }else rows.push_back(ev_dict(s,{{"from_region",rr_string(s,choice.first)},{"to_region",rr_string(s,choice.second)}}));
    }
    return cp_container(s,WOTR_LIST,rows);
}
int64_t cdm_move(Snapshot& s,const Snapshot& t,const std::string& from,const std::string& to,int side,bool border_rule,const std::string& via,int64_t regular,int64_t elite,bool allow_overstack,uint64_t cargo,bool defer_overstack);
int64_t epw_move(Snapshot& s,const Snapshot& t,const std::string& from,const std::string& to,int64_t regular=-1,int64_t elite=-1){
    return cdm_move(s,t,from,to,0,true,"",regular,elite,true,missing,false);
}
void epw_effect(Snapshot& s,const Snapshot& t,uint64_t choice){
    const auto options=evt_paths_options(s,t);
    const auto raw_from=field(s,choice,"from_region",false),raw_to=field(s,choice,"to_region",false);
    std::string from,to;
    if(ctx_none(s,raw_from)||ctx_none(s,raw_to)){
        const auto offered=child_edges(s,epw_options(s,t));if(offered.empty())return;choice=offered.front();from=text(s,field(s,choice,"from_region"));to=text(s,field(s,choice,"to_region"));
    }else{from=sanitize_name(text(s,raw_from));to=sanitize_name(text(s,raw_to));}
    if(std::find(options.begin(),options.end(),std::make_pair(from,to))!=options.end()){const auto r=field(s,choice,"n_regular",false),e=field(s,choice,"n_elite",false);epw_move(s,t,from,to,ctx_none(s,r)?-1:typed(s,r,WOTR_INT,"Woses regular count must be integer").integer,ctx_none(s,e)?-1:typed(s,e,WOTR_INT,"Woses elite count must be integer").integer);}
    // The source handler returns a completed zero-effect play for an invalid
    // destination; it does not refund or make a different automatic move.
}
fm_Result apply_woses_event(const Snapshot& source,const Snapshot& t,const void* random_source,int32_t side,int64_t index,std::string_view id){
    require(side==0,"Woses belongs to FP");quiet_action_gate(source,t,false,false,false,false,false,false,false,false,false,false,false,true);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==0,"Woses violates retained Ring actor");
    const auto card=ev_card(t,id);require(epw_supported(t,card)&&contains_string(source,tc_hand(source,0),id),"Woses card absent");require(ev_requirement(source,t,card),"Woses requirement false");
    const auto die=ev_payment(source,t,0,index,card);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace")),deck(ev_lower(text(t,obs_attr(t,card,"deck"))));
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;am_Builder events;std::vector<uint64_t> rows;
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));rr_set_attr(s,ev_payment(s,t,0,index,card),"used",rr_scalar(s,WOTR_BOOL,1));
    const auto options=epw_options(s,t);const auto choices=child_edges(s,options);
    if(choices.size()>1){
        rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,"Paths of the Woses")},{"side",rr_string(s,"fp")},{"decider",rr_string(s,"fp")},{"options",options},{"die_face",rr_string(s,face)},{"deck",rr_string(s,deck)}}));tc_open_hand_limit_discard(s);
    }else{if(!choices.empty())epw_effect(s,t,choices.front());er_commit(s,t,id,deck,face);}
    cau_finish(s,t,result.random.get(),-1,events,rows);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
std::unique_ptr<Snapshot> native_woses_choice_actions(const Snapshot& source,int32_t side){
    require(side==0&&epw_pending(source),"No FP Woses choice pending");auto out=std::make_unique<Snapshot>(source);auto& s=*out;const auto p=obs_attr(s,s.root,"pending_card_option");
    require(text_is(s,field(s,p,"side",false),"fp")&&text_is(s,field(s,p,"decider",false),"fp"),"Woses pending belongs to FP");
    std::vector<uint64_t> rows;int64_t index=0;
    for(auto choice:child_edges(s,field(s,p,"options")))rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",choice}})}));
    s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
fm_Result apply_woses_choice(const Snapshot& source,const Snapshot& t,const void* random_source,int32_t side,int64_t index){
    require(side==0&&epw_pending(source),"No FP Woses choice pending");const auto p=obs_attr(source,source.root,"pending_card_option");const auto options=child_edges(source,field(source,p,"options"));require(index>=0&&static_cast<uint64_t>(index)<options.size(),"Woses choice index exceeds menu");
    const std::string id(text(source,field(source,p,"card_id"))),deck(text(source,field(source,p,"deck"))),face(text(source,field(source,p,"die_face")));require(epw_supported(t,ev_card(t,id))&&contains_string(source,tc_hand(source,0),id),"Pending Woses card absent");
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;am_Builder events;std::vector<uint64_t> rows;
    const auto choice=child_edges(s,field(s,obs_attr(s,s.root,"pending_card_option"),"options"))[index],marker=obs_attr(s,s.root,"_battle_resolved_attacker",false);const int marker_side=ctx_none(s,marker)?-1:ctx_side(s,marker);
    rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));quiet_action_gate(s,t,false,false,false,false,false,false,false,false,false,false,false,true);ring_waiver(s,0);ring_waiver(s,1);
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    // play_event_card checks current requirements again after clearing the
    // pending. A newly false requirement keeps the card without a die refund.
    if(ev_requirement(s,t,ev_card(t,id))){epw_effect(s,t,choice);er_commit(s,t,id,deck,face);}
    cau_finish(s,t,result.random.get(),marker_side,events,rows);
    events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
