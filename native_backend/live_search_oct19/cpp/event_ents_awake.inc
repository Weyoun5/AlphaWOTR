// Ents trilogy, shared owner casualties and explicit free Character funding.
bool eea_supported(const Snapshot& t,uint64_t card){
    if(card==missing)return false;const auto name=obs_attr(t,card,"card_name");
    for(const char* option:{"The Ents Awake: Treebeard","The Ents Awake: Huorns","The Ents Awake: Entmoot"})if(text_is(t,name,option))return true;
    return false;
}
bool eea_pending(const Snapshot& s){
    const auto p=obs_attr(s,s.root,"pending_card_option",false);
    if(ctx_none(s,p)||truth(s,field(s,p,"card_casualty",false)))return false;
    for(const char* name:{"The Ents Awake: Treebeard","The Ents Awake: Huorns","The Ents Awake: Entmoot"})if(text_is(s,field(s,p,"card_name",false),name))return true;
    return false;
}
bool eea_white_here(const Snapshot& s,const Snapshot& t){
    const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));
    for(size_t i=0;i<armies.size();i+=2){
        bool found=false;for(auto figure:child_edges(s,obs_attr(s,armies[i+1],"leaders")))found|=text_is(s,obs_attr(s,figure,"name",false),"Gandalf the White");
        if(!found)continue;
        const std::string key=sanitize_name(text(s,armies[i]));
        // Source stops after the FIRST matching army even if out of range.
        return key=="fangorn"||text_is(t,field(t,field(t,t.root,"board_region_parent_names"),key.c_str(),false),"rohan");
    }return false;
}
void eea_chain(Snapshot& s,const Snapshot& t,std::string_view id,std::string_view name){
    const auto old=obs_attr(s,s.root,"_ents_chain_n",false);
    const auto n=crs_sum(ctx_none(s,old)?0:typed(s,old,WOTR_INT,"Ents chain count must be integer").integer,1);
    rr_set_attr(s,s.root,"_ents_chain_n",rr_scalar(s,WOTR_INT,n));
    if(n<4&&eea_white_here(s,t)){
        std::vector<uint64_t> options;
        for(auto card:child_edges(s,tc_hand(s,0))){
            if(s.nodes[card].tag!=WOTR_STRING)continue;const auto candidate=text(s,card);
            if(candidate==id||candidate.find("_Character_")==std::string_view::npos)continue;
            // Enumerate ALL source-legal Character cards, without narrowing to
            // current native executors. Unsupported execution remains explicit.
            const auto definition=ev_card(t,candidate);
            if(definition!=missing&&ev_requirement(s,t,definition))options.push_back(ev_dict(s,{{"ents_chain_play",card}}));
        }
        if(!options.empty()){
            options.push_back(ev_dict(s,{{"_ents_chain_done",rr_scalar(s,WOTR_BOOL,1)}}));
            rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,name)},{"side",rr_string(s,"fp")},{"decider",rr_string(s,"fp")},{"direct",rr_scalar(s,WOTR_BOOL,1)},{"options",cp_container(s,WOTR_LIST,options)}}));return;
        }
    }rr_set_attr(s,s.root,"_ents_chain_n",rr_scalar(s,WOTR_INT,0));
}
void eea_effect(Snapshot& s,const Snapshot& t,std::string_view id,bool resumed,void* rng,am_Builder& events,std::vector<uint64_t>& rows){
    const auto card=ev_card(t,id);require(eea_supported(t,card),"Ents effect card identity invalid");const std::string name(text(t,obs_attr(t,card,"card_name")));
    int64_t hits=0;
    if(!resumed){
        std::array<uint32_t,3> rolls{};hf_roll(rng,rolls.data(),rolls.size(),"card effect");
        std::vector<uint64_t> faces;for(auto roll:rolls){hits+=roll>=4;faces.push_back(events.integer(roll));}
        // This exact Entmoot label is also used for Treebeard and Huorns.
        rows.push_back(events.row(-1,"dice_roll",{events.str("faces"),events.container(WOTR_LIST,faces),events.str("ctx"),events.str("The Ents Awake: Entmoot roll"),events.str("side"),rr_scalar(events.out,WOTR_NONE)}));
    }
    auto army=mu_army(s,"orthanc");bool shadow=false;
    if(army!=missing)for(auto unit:child_edges(s,obs_attr(s,army,"units")))shadow|=ctx_unit_side(s,unit)==1;
    if(shadow||resumed){
        if(!resumed){
            std::set<std::string> nations;for(auto unit:child_edges(s,obs_attr(s,army,"units")))if(ctx_unit_side(s,unit)==1)nations.emplace(ctx_nation(s,obs_attr(s,unit,"nation")));
            for(const auto& nation:nations)wt_defender_advance(s,nation,"orthanc");
        }
        // Reuse the shared allocator and its explicit Ents resume callback.
        if(hits>0&&!ecc_route(s,t,"orthanc",1,hits,id,name,rng))return;
        army=mu_army(s,"orthanc");shadow=false;
        if(army!=missing)for(auto unit:child_edges(s,obs_attr(s,army,"units")))shadow|=ctx_unit_side(s,unit)==1;
        if(army!=missing&&!shadow){
            cbe_reconcile(s,"orthanc",1,rng);
            const auto remaining=mu_army(s,"orthanc");
            if(remaining!=missing&&contains_string(s,obs_attr(s,s.root,"besieged_strongholds"),"orthanc")){
                bool fp=false;for(auto unit:child_edges(s,obs_attr(s,remaining,"units")))fp|=ctx_unit_side(s,unit)==0;
                if(fp){me_set_member(s,obs_attr(s,s.root,"besieged_strongholds"),"orthanc",false);crs_capture(s,t,"orthanc",0);}
            }
        }
    }else if(army!=missing){
        std::vector<uint64_t> removed,remaining;
        for(auto leader:child_edges(s,obs_attr(s,army,"leaders")))(am_saruman(s,leader)?removed:remaining).push_back(leader);
        if(!removed.empty()){rr_set_attr(s,army,"leaders",cp_container(s,WOTR_LIST,remaining));crs_book_figures(s,removed,rng);}
    }
    eea_chain(s,t,id,name);
}
void ef_execute_free_event(Snapshot& s,const Snapshot& t,void* rng,std::string_view id,am_Builder& events,std::vector<uint64_t>& rows);
void apply_ents_event_inplace(Snapshot& s,const Snapshot& t,void* rng,int side,int64_t index,std::string_view id,bool free_ents,am_Builder& events,std::vector<uint64_t>& rows){
    ef_validate_free(s,t,side,index,id,free_ents);require(side==0,"Ents belongs to FP");
    quiet_action_gate(s,t,false,false,false,false,false,false,false,false,false,false,free_ents,true);ring_waiver(s,0);ring_waiver(s,1);
    const auto marker=ctx_side(s,obs_attr(s,s.root,"_battle_resolved_attacker",false));const auto unused=ef_unused_dice(s);
    const auto retained=obs_attr(s,s.root,"_ring_free_side",false);require(ctx_none(s,retained)||ctx_side(s,retained)==0,"Ents violates retained Ring actor");
    const auto card=ev_card(t,id);require(eea_supported(t,card)&&contains_string(s,tc_hand(s,0),id),"Ents card absent");require(ev_requirement(s,t,card),"Ents requirement false");
    const auto die=free_ents?missing:ev_payment(s,t,0,index,card);
    const std::string face=free_ents?std::string{}:std::string(enum_member(s,obs_attr(s,die,"current"),"game_state.ActionDieFace"));
    rr_set_attr(s,s.root,"_casualty_operations",cp_container(s,WOTR_LIST,{}));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));if(!free_ents)rr_set_attr(s,die,"used",rr_scalar(s,WOTR_BOOL,1));
    eea_effect(s,t,id,false,rng,events,rows);er_commit(s,t,id,"character",face);
    cau_finish(s,t,rng,free_ents?ef_final_marker(s,marker,unused):-1,events,rows);
}
fm_Result apply_ents_event(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index,std::string_view id){
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());
    am_Builder events;std::vector<uint64_t> rows;apply_ents_event_inplace(*result.state,t,result.random.get(),side,index,id,false,events,rows);
    events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
uint64_t eea_validate(const Snapshot& s,const Snapshot& t){
    require(eea_pending(s),"No Ents chain pending");const auto p=obs_attr(s,s.root,"pending_card_option");
    const auto card=ev_card(t,text(s,field(s,p,"card_id")));require(eea_supported(t,card)&&text_is(s,field(s,p,"card_name"),text(t,obs_attr(t,card,"card_name"))),"Ents pending card identity invalid");
    require(text_is(s,field(s,p,"side"),"fp")&&text_is(s,field(s,p,"decider"),"fp")&&truth(s,field(s,p,"direct")),"Ents chain belongs to FP");
    const auto choices=child_edges(s,field(s,p,"options"));require(choices.size()>=2,"Ents chain requires a card and Done");
    for(size_t i=0;i<choices.size();++i){
        require(typed(s,choices[i],WOTR_DICT,"Ents choice must be dictionary").count==2,"Ents stored choice shape invalid");
        if(i+1==choices.size())require(typed(s,field(s,choices[i],"_ents_chain_done"),WOTR_BOOL,"Ents Done must be Boolean").integer==1,"Ents Done must be true");
        else require(text(s,field(s,choices[i],"ents_chain_play")).find("_Character_")!=std::string_view::npos,"Ents choice must be Character card");
    }return p;
}
std::unique_ptr<Snapshot> native_ents_choices(const Snapshot& source,const Snapshot& t,int side){
    require(side==0,"Ents choices belong to FP");eea_validate(source,t);Snapshot s(source);const auto p=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t index=0;
    for(auto choice:child_edges(s,field(s,p,"options")))rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",choice}})}));s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
fm_Result apply_ents_choice(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index){
    require(side==0,"Ents chain belongs to FP");const auto p=eea_validate(source,t);const auto options=child_edges(source,field(source,p,"options"));require(index>=0&&static_cast<uint64_t>(index)<options.size(),"Ents chain index invalid");
    const auto choice=options[index];const bool done=truth(source,field(source,choice,"_ents_chain_done",false));const std::string id=done?std::string{}:std::string(text(source,field(source,choice,"ents_chain_play")));
    const auto marker=ctx_side(source,obs_attr(source,source.root,"_battle_resolved_attacker",false));
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;
    rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));quiet_action_gate(s,t,false,false,false,false,false,false,false,false,false,false,true,true);ring_waiver(s,0);ring_waiver(s,1);
    rr_set_attr(s,s.root,"_casualty_operations",cp_container(s,WOTR_LIST,{}));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));am_Builder events;std::vector<uint64_t> rows;
    if(done)rr_set_attr(s,s.root,"_ents_chain_n",rr_scalar(s,WOTR_INT,0));else ef_execute_free_event(s,t,result.random.get(),id,events,rows);
    su_resolve_overstacks(s,t);cau_finish(s,t,result.random.get(),marker,events,rows);
    events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
