// Heal first, then the source's optional Gollum action.
bool eaw_supported(const Snapshot& t,uint64_t card){
    return card!=missing && text_is(t,obs_attr(t,card,"card_name"),"There is Another Way");
}
bool eaw_pending(const Snapshot& s){
    const auto p=obs_attr(s,s.root,"pending_card_option",false);
    return !ctx_none(s,p)&&text_is(s,field(s,p,"card_name",false),"There is Another Way");
}
bool eaw_gollum(const Snapshot& s){
    const auto fs=obs_attr(s,s.root,"fellowship"),guide=obs_attr(s,fs,"guide");
    return !ctx_none(s,guide)&&text(s,obs_attr(s,guide,"name")).find("Gollum")!=std::string_view::npos;
}
bool eaw_stable_siege_scope(const Snapshot& s){
    // Proved public Hunt/placement continuations preserve stable sieges.
    (void)s;return true;
}
uint64_t eaw_options(Snapshot& s){
    std::vector<uint64_t> options;
    if(eaw_gollum(s))options.push_back(ev_dict(s,{{"action",rr_string(s,
        truth(s,obs_attr(s,obs_attr(s,s.root,"fellowship"),"hidden"))?"move":"hide")}}));
    options.push_back(ev_dict(s,{{"action",rr_string(s,"heal_only")}}));
    return cp_container(s,WOTR_LIST,options);
}
void eaw_effect(Snapshot& s,const Snapshot& t,void* rng,std::string_view action,am_Builder& events,std::vector<uint64_t>& rows){
    const auto fs=obs_attr(s,s.root,"fellowship");
    rr_set_attr(s,fs,"corruption",rr_scalar(s,WOTR_INT,std::max<int64_t>(0,ht_integer(s,fs,"corruption")-1)));
    if(!eaw_gollum(s))return;
    const bool hidden=truth(s,obs_attr(s,fs,"hidden"));
    if(action=="hide"){
        if(!hidden){rr_set_attr(s,fs,"hidden",rr_scalar(s,WOTR_BOOL,1));rr_set_attr(s,s.root,"fellowship_acted_in_mordor",rr_scalar(s,WOTR_BOOL,1));}
        return;
    }
    if(action!="move"||!hidden)return;
    // The event invokes move_fellowship, not the paid action wrapper: no
    // extra die, no Hunt-box debt, no newly opened Crebain reaction, and no
    // ordinary progress12 cap. Existing Hunt modifiers remain effective.
    rr_set_attr(s,s.root,"fellowship_acted_in_mordor",rr_scalar(s,WOTR_BOOL,1));
    if(truth(s,obs_attr(s,fs,"mordor_entered"))){
        const auto step=ht_integer(s,fs,"mordor_step");if(step>=5)return;
        rr_set_attr(s,fs,"mordor_step",rr_scalar(s,WOTR_INT,step+1));
        const auto tile=ht_resolve(s,t,rng,0,true,true,false,events,rows);
        if(tile.stop)rr_set_attr(s,fs,"mordor_step",rr_scalar(s,WOTR_INT,step));
    }else{
        rr_set_attr(s,fs,"progress",rr_scalar(s,WOTR_INT,ht_integer(s,fs,"progress")+1));
        const auto params=hr_parameters(s,t);
        const auto hits=params.dice?hr_roll(s,t,rng,params.dice,6,params.fp,events,rows):0;
        if(hits)ht_resolve(s,t,rng,hits,false,false,false,events,rows);
    }
}
void apply_another_way_event_inplace(Snapshot& s,const Snapshot& t,void* rng,int side,int64_t index,std::string_view id,bool free_ents,am_Builder& ef_events,std::vector<uint64_t>& ef_annotations){
    const auto& source=s;ef_validate_free(source,t,side,index,id,free_ents);
    const int ef_marker=ctx_side(source,obs_attr(source,source.root,"_battle_resolved_attacker",false));const auto ef_unused=ef_unused_dice(source);

    require(side==0,"There is Another Way belongs to FP");
    quiet_action_gate(source,t,false,false,false,false,false,false,false,false,true,false,(false||free_ents),eaw_stable_siege_scope(source));ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);
    require(ctx_none(source,retained)||ctx_side(source,retained)==0,"Another Way violates retained Ring actor");
    const auto card=ev_card(t,id);
    require(eaw_supported(t,card)&&contains_string(source,tc_hand(source,0),id),"Another Way card absent");
    require(ev_requirement(source,t,card),"Another Way requirement false");
    const auto die=free_ents?missing:ev_payment(source,t,0,index,card);
    const std::string face(free_ents?std::string_view{}:enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace"));
    const std::string deck=ev_lower(text(t,obs_attr(t,card,"deck")));
    
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    if(!free_ents)rr_set_attr(s,ev_payment(s,t,0,index,card),"used",rr_scalar(s,WOTR_BOOL,1));
    const auto options=eaw_options(s);
    if(s.nodes[options].count>1){
        rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,"There is Another Way")},
            {"side",rr_string(s,"fp")},{"decider",rr_string(s,"fp")},{"options",options},{"die_face",ef_pending_face_node(s,face)},{"deck",rr_string(s,deck)}}));
        tc_open_hand_limit_discard(s);
    }else{
        eaw_effect(s,t,rng,"heal_only",ef_events,ef_annotations);
        er_commit(s,t,id,deck,face);
    }
    ehd_finish(s,t,rng,ef_events,ef_annotations);cau_finish(s,t,rng,free_ents?ef_final_marker(s,ef_marker,ef_unused):-1,ef_events,ef_annotations);
    
}
fm_Result apply_another_way_event(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index,std::string_view id){
    require(side==0,"There is Another Way belongs to FP");
    quiet_action_gate(source,t,false,false,false,false,false,false,false,false,true,false,false,eaw_stable_siege_scope(source));ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);
    require(ctx_none(source,retained)||ctx_side(source,retained)==0,"Another Way violates retained Ring actor");
    const auto card=ev_card(t,id);
    require(eaw_supported(t,card)&&contains_string(source,tc_hand(source,0),id),"Another Way card absent");
    require(ev_requirement(source,t,card),"Another Way requirement false");
    const auto die=ev_payment(source,t,0,index,card);
    const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace"));
    const std::string deck=ev_lower(text(t,obs_attr(t,card,"deck")));
    
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());
    am_Builder ef_events;std::vector<uint64_t> ef_annotations;
    apply_another_way_event_inplace(*result.state,t,result.random.get(),side,index,id,false,ef_events,ef_annotations);
    ef_events.out.root=ef_events.container(WOTR_LIST,ef_annotations);result.events=std::make_unique<Snapshot>(std::move(ef_events.out));return result;
}
uint64_t eaw_validate(const Snapshot& s,const Snapshot& t){
    require(eaw_pending(s),"No Another Way option pending");const auto p=obs_attr(s,s.root,"pending_card_option");
    require(text_is(s,field(s,p,"side"),"fp")&&text_is(s,field(s,p,"decider"),"fp")&&!truth(s,field(s,p,"direct",false)),"Another Way pending owner/stage invalid");
    const auto id=text(s,field(s,p,"card_id"));
    require(eaw_supported(t,ev_card(t,id)),"Another Way pending card identity invalid");/* package R: hand membership is not a menu precondition (frozen card_option menu, ai.py 7764-7770) */
    require(text_is(s,field(s,p,"deck"),"character"),"Another Way pending deck invalid");
    const auto face=ef_pending_face(s,p);
    require(face.empty()||face=="CHARACTER"||face=="EVENT"||face=="WILL_OF_WEST","Another Way pending payment invalid");
    const auto options=child_edges(s,field(s,p,"options"));
    require(options.size()==2,"Another Way pending options must retain two choices");
    for(auto option:options){
        require(typed(s,option,WOTR_DICT,"Another Way option must be a dictionary").count==2,"Another Way option shape invalid");
        const auto action=text(s,field(s,option,"action"));require(action=="move"||action=="hide"||action=="heal_only","Another Way option action invalid");
    }
    const auto first=text(s,field(s,options[0],"action"));
    require((first=="move"||first=="hide")&&text_is(s,field(s,options[1],"action"),"heal_only"),"Another Way stored choice order invalid");
    return p;
}
std::unique_ptr<Snapshot> native_another_way_choices(const Snapshot& source,const Snapshot& t,int side){
    require(side==0,"Another Way choices belong to FP");eaw_validate(source,t);
    auto result=std::make_unique<Snapshot>(source);auto& s=*result;
    const auto p=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t index=0;
    for(auto choice:child_edges(s,field(s,p,"options")))rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),
        ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",choice}})}));
    s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
fm_Result apply_another_way_choice(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index){
    const bool free_ents=ef_pending_is_free(source,obs_attr(source,source.root,"pending_card_option",false));

    require(side==0,"Another Way choice belongs to FP");const auto p=eaw_validate(source,t);const auto options=child_edges(source,field(source,p,"options"));
    require(index>=0&&static_cast<uint64_t>(index)<options.size(),"Another Way choice index exceeds menu");
    const std::string id(text(source,field(source,p,"card_id"))),face(ef_pending_face(source,p)),deck(text(source,field(source,p,"deck"))),action(text(source,field(source,options[index],"action")));
    const auto oldmarker=obs_attr(source,source.root,"_battle_resolved_attacker",false);const int marker=ctx_none(source,oldmarker)?-1:ctx_side(source,oldmarker);
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));
    if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;
    rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));quiet_action_gate(s,t,false,false,false,false,false,false,false,false,true,false,(false||free_ents),eaw_stable_siege_scope(source));ring_waiver(s,0);ring_waiver(s,1);
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    am_Builder events;std::vector<uint64_t> rows;
    if(ev_requirement(s,t,ev_card(t,id))){eaw_effect(s,t,result.random.get(),action,events,rows);er_commit(s,t,id,deck,face);}
    ehd_finish(s,t,result.random.get(),events,rows);cau_finish(s,t,result.random.get(),marker,events,rows);
    events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
