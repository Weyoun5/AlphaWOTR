// Faramir: player selects target, RNG strikes, Shadow allocates losses, then
// FP selects recruitment. No casualty continuation can repeat the strike.
bool eff_supported(const Snapshot& t,uint64_t card){return card!=missing && text_is(t,obs_attr(t,card,"card_name"),"Faramir's Rangers");}
bool eff_army(const Snapshot& s,std::string_view region,int side){const auto army=mu_army(s,region);if(army==missing)return false;for(auto u:child_edges(s,obs_attr(s,army,"units")))if(ctx_unit_side(s,u)==side)return true;return false;}
std::vector<std::string> eff_targets(const Snapshot& s){std::vector<std::string> result;for(const char* key:{"osgiliath","south ithilien","north ithilien"})if(eff_army(s,key,1))result.emplace_back(key);return result;}
bool eff_requirement(const Snapshot& s){return !eff_targets(s).empty() || eff_army(s,"osgiliath",0);}
bool eff_pending(const Snapshot& s){const auto p=obs_attr(s,s.root,"pending_card_option",false);return !ctx_none(s,p) && !truth(s,field(s,p,"card_casualty",false)) && text_is(s,field(s,p,"card_name",false),"Faramir's Rangers") && text_is(s,field(s,p,"side",false),"fp") && text_is(s,field(s,p,"decider",false),"fp");}
uint64_t eff_recruit_options(Snapshot& s,const Snapshot& t){
    std::vector<uint64_t> options;if(!eff_army(s,"osgiliath",0))return cp_container(s,WOTR_LIST,options);
    for(const char* kind:{"regular","elite"})if(mu_available(s,t,"GONDOR",kind)>0)options.push_back(ev_dict(s,{{"_far_recruit",rr_scalar(s,WOTR_BOOL,1)},{"unit_type",rr_string(s,kind)}}));
    if(options.empty())options.push_back(ev_dict(s,{{"_far_recruit",rr_scalar(s,WOTR_BOOL,1)}}));return cp_container(s,WOTR_LIST,options);
}
void eff_recruit(Snapshot& s,const Snapshot& t,std::string_view id,uint64_t choice=missing){
    if(!eff_army(s,"osgiliath",0))return;const auto options=eff_recruit_options(s,t);const auto values=child_edges(s,options);
    if(choice==missing && values.size()>1){rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,"Faramir's Rangers")},{"side",rr_string(s,"fp")},{"decider",rr_string(s,"fp")},{"direct",rr_scalar(s,WOTR_BOOL,1)},{"faramir_recruit",rr_scalar(s,WOTR_BOOL,1)},{"options",options}}));return;}
    if(choice==missing && values.size()==1)choice=values.front();
    const auto selected=choice==missing?missing:field(s,choice,"unit_type",false);const std::string kind=ctx_none(s,selected)?er_default_kind(s,t,"GONDOR"):std::string(text(s,selected));
    er_place(s,t,"GONDOR","osgiliath",kind);er_place(s,t,"GONDOR","osgiliath","leader");
}
void eff_casualty_resume(Snapshot& s,const Snapshot& t,std::string_view region,void* rng,am_Builder& events,std::vector<uint64_t>& rows){
    // This is exactly the _far_cas_resumed leg: region records the previous
    // strike, but the printed recruitment location remains Osgiliath.
    eff_recruit(s,t,"FREE_PEOPLES_Strategy_Faramir's_Rangers");
}
void eff_effect(Snapshot& s,const Snapshot& t,std::string_view id,uint64_t choice,void* rng,am_Builder& events,std::vector<uint64_t>& rows){
    if(truth(s,field(s,choice,"_far_recruit",false))){eff_recruit(s,t,id,choice);return;}
    const auto target=field(s,choice,"target_region",false);const std::string region=ctx_none(s,target)?"osgiliath":sanitize_name(text(s,target));
    if(eff_army(s,region,1)){
        std::array<uint32_t,3> rolls{};hf_roll(rng,rolls.data(),rolls.size(),"card effect");
        int64_t hits=0;std::vector<uint64_t> faces;for(auto roll:rolls){hits+=roll>=5;faces.push_back(events.integer(roll));}
        rows.push_back(events.row(-1,"dice_roll",{events.str("faces"),events.container(WOTR_LIST,faces),events.str("ctx"),events.str("Faramir's Rangers roll"),events.str("side"),rr_scalar(events.out,WOTR_NONE)}));
        if(hits>0 && !ecc_route(s,t,region,1,hits,id,"Faramir's Rangers",rng))return;
    }
    eff_recruit(s,t,id);
}
fm_Result apply_faramir_event(const Snapshot& source,const Snapshot& t,const void* random_source,int side,int64_t index,std::string_view id){
    require(side==0,"Faramir belongs to FP");quiet_stable_action_gate(source,t);ring_waiver(source,0);ring_waiver(source,1);const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained) || ctx_side(source,retained)==0,"Faramir violates retained Ring actor");
    const auto card=ev_card(t,id);require(eff_supported(t,card) && ctx_side(t,obs_attr(t,card,"side"))==0 && contains_string(source,tc_hand(source,0),id),"Faramir event absent or invalid");require(eff_requirement(source),"Faramir needs Shadow target or FP Osgiliath army");
    const auto die=ev_payment(source,t,0,index,card);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace")),deck(ev_lower(text(t,obs_attr(t,card,"deck"))));
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;am_Builder events;std::vector<uint64_t> rows;
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));rr_set_attr(s,ev_payment(s,t,0,index,card),"used",rr_scalar(s,WOTR_BOOL,1));
    const auto targets=eff_targets(s);std::vector<uint64_t> options;for(const auto& target:targets)options.push_back(ev_dict(s,{{"target_region",rr_string(s,target)}}));
    if(options.size()>1){rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,"Faramir's Rangers")},{"side",rr_string(s,"fp")},{"decider",rr_string(s,"fp")},{"options",cp_container(s,WOTR_LIST,options)},{"die_face",rr_string(s,face)},{"deck",rr_string(s,deck)}}));tc_open_hand_limit_discard(s);}
    else{eff_effect(s,t,id,options.empty()?ev_dict(s,{}):options.front(),result.random.get(),events,rows);er_commit(s,t,id,deck,face);}
    cau_finish(s,t,result.random.get(),-1,events,rows);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
std::unique_ptr<Snapshot> native_faramir_choice_actions(const Snapshot& source,const Snapshot& t,int side){
    require(side==0 && eff_pending(source),"No FP Faramir choice pending");const auto p0=obs_attr(source,source.root,"pending_card_option");require(eff_supported(t,ev_card(t,text(source,field(source,p0,"card_id")))),"Faramir pending ID invalid");
    auto out=std::make_unique<Snapshot>(source);auto& s=*out;const auto p=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t index=0;
    for(auto choice:child_edges(s,field(s,p,"options")))rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",choice}})}));
    s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
fm_Result apply_faramir_choice(const Snapshot& source,const Snapshot& t,const void* random_source,int side,int64_t index){
    require(side==0 && eff_pending(source),"No FP Faramir choice pending");const auto old=obs_attr(source,source.root,"pending_card_option");const std::string id(text(source,field(source,old,"card_id")));require(eff_supported(t,ev_card(t,id)),"Faramir pending ID invalid");
    const auto options=child_edges(source,field(source,old,"options"));require(index>=0 && static_cast<uint64_t>(index)<options.size(),"Faramir choice index invalid");const bool recruit=truth(source,field(source,old,"faramir_recruit",false)),direct=truth(source,field(source,old,"direct",false));
    std::string deck,face;if(!direct){require(contains_string(source,tc_hand(source,0),id),"Faramir uncommitted card missing");deck=std::string(text(source,field(source,old,"deck")));face=std::string(text(source,field(source,old,"die_face")));}
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;am_Builder events;std::vector<uint64_t> rows;
    const auto p=obs_attr(s,s.root,"pending_card_option"),choice=child_edges(s,field(s,p,"options"))[index];bool valid=false;
    if(recruit){const auto live=eff_recruit_options(s,t);for(auto option:child_edges(s,live))valid|=tail_equal(s,option,choice);require(direct,"Faramir recruit must be paid continuation");}
    else{const auto target=field(s,choice,"target_region",false);const auto targets=eff_targets(s);valid=typed(s,choice,WOTR_DICT,"Faramir choice must be dict").count==2 && !ctx_none(s,target) && std::find(targets.begin(),targets.end(),sanitize_name(text(s,target)))!=targets.end();}
    require(valid,"Faramir choice is not currently available");const auto marker=obs_attr(s,s.root,"_battle_resolved_attacker",false);const int marker_side=ctx_none(s,marker)?-1:ctx_side(s,marker);
    rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));quiet_action_gate(s,t,false,false,false,false,false,false,false,false,false,false,true,true);ring_waiver(s,0);ring_waiver(s,1);
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));eff_effect(s,t,id,choice,result.random.get(),events,rows);
    if(!direct)er_commit(s,t,id,deck,face);else su_resolve_overstacks(s,t);cau_finish(s,t,result.random.get(),marker_side,events,rows);
    events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
