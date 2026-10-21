// House of the Stewards, using existing physical recruitment and draw.
bool ehs_supported(const Snapshot& t,uint64_t card){return card!=missing&&text_is(t,obs_attr(t,card,"card_name"),"House of the Stewards");}
bool ehs_pending(const Snapshot& s){const auto p=obs_attr(s,s.root,"pending_card_option",false);return !ctx_none(s,p)&&text_is(s,field(s,p,"card_name",false),"House of the Stewards");}
uint64_t ehs_options(Snapshot& s,const Snapshot& t){
    const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));std::vector<uint64_t> options;
    for(size_t i=0;i<armies.size();i+=2){
        bool boromir=false;for(auto f:child_edges(s,obs_attr(s,armies[i+1],"leaders")))boromir|=object_type(s,f)=="game_state.Companion"&&text_is(s,obs_attr(s,f,"name",false),"Boromir");
        if(!boromir)continue;const auto region=sanitize_name(text(s,armies[i]));
        if(!enum_is(t,ctx_get(t,field(t,t.root,"region_nations"),region),"game_state.Nation","GONDOR"))break;
        if(er_region_open(s,region,0,true))for(const char* kind:{"regular","elite"})if(mu_available(s,t,"GONDOR",kind)>0)options.push_back(ev_dict(s,{{"boromir_region",rr_string(s,region)},{"unit_type",rr_string(s,kind)}}));
        if(options.empty())options.push_back(ev_dict(s,{{"boromir_region",rr_string(s,region)}}));break;
    }return cp_container(s,WOTR_LIST,options);
}
void ehs_effect(Snapshot& s,const Snapshot& t,uint64_t choice,am_Builder& events,std::vector<uint64_t>& rows){
    const std::string region(text(s,field(s,choice,"boromir_region")));const auto selected=field(s,choice,"unit_type",false);
    er_place(s,t,"GONDOR",region,ctx_none(s,selected)?er_default_kind(s,t,"GONDOR"):std::string(text(s,selected)));
    for(int i=0;i<2&&truth(s,obs_attr(s,s.root,"fp_strat_deck",false));++i){const auto card=tc_draw_event_card(s,0,"strategy");bc_settle_draw(s,0,"strategy");rows.push_back(events.row(-1,"card_drawn",{events.str("deck"),events.str("strategy"),events.str("card_id"),events.str(text(s,card)),events.str("side"),events.str("FP")}));}
}
void apply_house_stewards_event_inplace(Snapshot& s,const Snapshot& t,void* rng,int side,int64_t index,std::string_view id,bool free_ents,am_Builder& ef_events,std::vector<uint64_t>& ef_annotations){
    const auto& source=s;ef_validate_free(source,t,side,index,id,free_ents);
    const int ef_marker=ctx_side(source,obs_attr(source,source.root,"_battle_resolved_attacker",false));const auto ef_unused=ef_unused_dice(source);

    require(side==0,"House of the Stewards belongs to FP");quiet_action_gate(source,t,false,false,false,false,false,false,false,false,false,false,(false||free_ents),true);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==side,"House of the Stewards violates retained Ring actor");
    const auto card=ev_card(t,id);require(ehs_supported(t,card)&&ctx_side(t,obs_attr(t,card,"side"))==0&&contains_string(source,tc_hand(source,0),id),"House of the Stewards absent from FP hand");require(ev_requirement(source,t,card),"House of the Stewards requirement false");
    const auto die=free_ents?missing:ev_payment(source,t,0,index,card);const std::string face(free_ents?std::string_view{}:enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace"));
    
    rr_set_attr(s,s.root,"_casualty_operations",cp_container(s,WOTR_LIST,{}));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));if(!free_ents)rr_set_attr(s,ev_payment(s,t,0,index,card),"used",rr_scalar(s,WOTR_BOOL,1));
    const auto options=ehs_options(s,t);const auto choices=child_edges(s,options);require(!choices.empty(),"House of the Stewards needs board Boromir in Gondor");
    if(choices.size()>1){
        rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,"House of the Stewards")},{"side",rr_string(s,"fp")},{"decider",rr_string(s,"fp")},{"options",options},{"die_face",ef_pending_face_node(s,face)},{"deck",rr_string(s,"character")}}));tc_open_hand_limit_discard(s);
    }else{ehs_effect(s,t,choices.front(),ef_events,ef_annotations);er_commit(s,t,id,"character",face);}
    cau_finish(s,t,rng,free_ents?ef_final_marker(s,ef_marker,ef_unused):-1,ef_events,ef_annotations);
}
fm_Result apply_house_stewards_event(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index,std::string_view id){
    require(side==0,"House of the Stewards belongs to FP");quiet_stable_action_gate(source,t);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==side,"House of the Stewards violates retained Ring actor");
    const auto card=ev_card(t,id);require(ehs_supported(t,card)&&ctx_side(t,obs_attr(t,card,"side"))==0&&contains_string(source,tc_hand(source,0),id),"House of the Stewards absent from FP hand");require(ev_requirement(source,t,card),"House of the Stewards requirement false");
    const auto die=ev_payment(source,t,0,index,card);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace"));
    
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());
    am_Builder ef_events;std::vector<uint64_t> ef_annotations;
    apply_house_stewards_event_inplace(*result.state,t,result.random.get(),side,index,id,false,ef_events,ef_annotations);
    ef_events.out.root=ef_events.container(WOTR_LIST,ef_annotations);result.events=std::make_unique<Snapshot>(std::move(ef_events.out));return result;
}
uint64_t ehs_validate_pending(const Snapshot& s,const Snapshot& t){
    require(ehs_pending(s),"No House of the Stewards choice pending");const auto p=obs_attr(s,s.root,"pending_card_option"),card=ev_card(t,text(s,field(s,p,"card_id")));
    require(ehs_supported(t,card)&&text_is(s,field(s,p,"side"),"fp")&&text_is(s,field(s,p,"decider"),"fp")&&!truth(s,field(s,p,"direct",false)),"House of the Stewards pending identity or owner mismatch");
    require(text_is(s,field(s,p,"deck"),"character")&&(ef_pending_is_free(s,p)||text_is(s,field(s,p,"die_face"),"CHARACTER")||text_is(s,field(s,p,"die_face"),"EVENT")||text_is(s,field(s,p,"die_face"),"WILL_OF_WEST")),"House of the Stewards pending funding invalid");
    const auto choices=child_edges(s,field(s,p,"options"));require(choices.size()==2,"House of the Stewards modal options invalid");
    for(size_t i=0;i<choices.size();++i)require(typed(s,choices[i],WOTR_DICT,"House recruit choice must be dictionary").count==4&&text_is(s,field(s,choices[i],"unit_type"),i==0?"regular":"elite")&&text_is(s,field(s,choices[i],"boromir_region"),text(s,field(s,choices[0],"boromir_region"))),"House of the Stewards stored choice malformed");return p;
}
std::unique_ptr<Snapshot> native_house_stewards_choices(const Snapshot& source,const Snapshot& t,int side){
    require(side==0,"House of the Stewards choice belongs to FP");ehs_validate_pending(source,t);Snapshot s(source);const auto p=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t index=0;
    for(auto choice:child_edges(s,field(s,p,"options")))rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",choice}})}));s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
fm_Result apply_house_stewards_choice(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index){
    const bool free_ents=ef_pending_is_free(source,obs_attr(source,source.root,"pending_card_option",false));

    require(side==0,"House of the Stewards choice belongs to FP");const auto pending=ehs_validate_pending(source,t),options=field(source,pending,"options");require(index>=0&&index<2,"House of the Stewards choice index invalid");
    Snapshot view(source);const auto current=ehs_options(view,t);const auto selected=child_edges(source,options)[index];bool available=false;
    for(auto choice:child_edges(view,current))available|=ap_equal(ap_freeze(source,selected),ap_freeze(view,choice));require(available,"House of the Stewards choice is no longer available");
    const std::string id(text(source,field(source,pending,"card_id"))),face(ef_pending_face(source,pending));const int marker=ctx_side(source,obs_attr(source,source.root,"_battle_resolved_attacker",false));
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;const auto p=obs_attr(s,s.root,"pending_card_option"),choice=child_edges(s,field(s,p,"options"))[index];
    rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));quiet_action_gate(s,t,false,false,false,false,false,false,false,false,false,false,(false||free_ents),true);ring_waiver(s,0);ring_waiver(s,1);rr_set_attr(s,s.root,"_casualty_operations",cp_container(s,WOTR_LIST,{}));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    am_Builder events;std::vector<uint64_t> rows;ehs_effect(s,t,choice,events,rows);er_commit(s,t,id,"character",face);cau_finish(s,t,result.random.get(),marker,events,rows);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
