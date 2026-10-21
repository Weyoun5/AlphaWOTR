// Cruel Weather moves only the Fellowship figure. It does not roll Hunt,
// change progress/visibility, or activate the nation at its destination.
bool ecw_supported(const Snapshot& t,uint64_t card){
    return card!=missing && text_is(t,obs_attr(t,card,"card_name"),"Cruel Weather");
}
bool ecw_pending(const Snapshot& s){
    const auto p=obs_attr(s,s.root,"pending_card_option",false);
    return !ctx_none(s,p) && text_is(s,field(s,p,"card_name",false),"Cruel Weather");
}
std::vector<std::string> ecw_targets(const Snapshot& s){
    const auto board=obs_attr(s,s.root,"board"),adj=obs_attr(s,board,"_adj");
    auto key=glb_fellowship_key(s,obs_attr(s,s.root,"fellowship"));
    auto edges=ctx_get(s,adj,key);
    if(edges==missing){
        const auto alias=key.substr(0,4)=="the "?key.substr(4):"the "+key;
        edges=ctx_get(s,adj,alias);
    }
    std::vector<std::string> ordered,targets;
    if(edges!=missing)for(auto value:child_edges(s,edges))ordered.emplace_back(text(s,value));
    std::sort(ordered.begin(),ordered.end());
    for(const auto& value:ordered)targets.push_back(sanitize_name(value));
    return targets;
}
uint64_t ecw_options(Snapshot& s,const std::vector<std::string>& targets){
    std::vector<uint64_t> options;
    for(const auto& target:targets)options.push_back(ev_dict(s,{{"target_region",rr_string(s,target)}}));
    return cp_container(s,WOTR_LIST,options);
}
void ecw_move(Snapshot& s,std::string_view target){
    if(target.empty())return;
    const auto regions=obs_attr(s,obs_attr(s,s.root,"board"),"_regions");
    auto region=ctx_get(s,regions,target);
    if(region==missing){const std::string key(target);region=ctx_get(s,regions,key.substr(0,4)=="the "?key.substr(4):"the "+key);}
    require(region!=missing,"Cruel Weather target has no Board region");
    rr_set_attr(s,obs_attr(s,s.root,"fellowship"),"region",region);
}
uint64_t ecw_validate_pending(const Snapshot& s,const Snapshot& t){
    require(ecw_pending(s),"No Cruel Weather choice pending");
    const auto p=obs_attr(s,s.root,"pending_card_option"),card=ev_card(t,text(s,field(s,p,"card_id")));
    require(ecw_supported(t,card)&&ctx_side(t,obs_attr(t,card,"side"))==1,"Cruel Weather pending card mismatch");
    require(text_is(s,field(s,p,"side"),"sh")&&text_is(s,field(s,p,"decider"),"sh"),"Cruel Weather pending owner invalid");
    require(!truth(s,field(s,p,"direct",false)),"Cruel Weather does not have a direct continuation");
    require(text_is(s,field(s,p,"deck"),"character"),"Cruel Weather pending deck invalid");
    const auto face=text(s,field(s,p,"die_face"));
    require(face=="CHARACTER"||face=="EVENT","Cruel Weather pending payment invalid");
    /* package R: hand membership is not a menu precondition (frozen card_option menu, ai.py 7764-7770) */
    const auto targets=ecw_targets(s);const auto options=child_edges(s,field(s,p,"options"));
    require(options.size()>=2&&options.size()==targets.size(),"Cruel Weather pending options stale");
    for(size_t i=0;i<options.size();++i)
        require(typed(s,options[i],WOTR_DICT,"Cruel Weather option must be dict").count==2
            &&text_is(s,field(s,options[i],"target_region"),targets[i]),"Cruel Weather pending options stale");
    return p;
}
std::unique_ptr<Snapshot> native_cruel_weather_choices(const Snapshot& source,const Snapshot& t,int side){
    require(side==1,"Cruel Weather choices belong to Shadow");ecw_validate_pending(source,t);
    auto result=std::make_unique<Snapshot>(source);auto& s=*result;
    const auto p=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t index=0;
    for(auto choice:child_edges(s,field(s,p,"options")))rows.push_back(cp_container(s,WOTR_TUPLE,
        {rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},
        {"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",choice}})}));
    s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
fm_Result apply_cruel_weather(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index,std::string_view id){
    require(side==1,"Cruel Weather belongs to Shadow");quiet_stable_action_gate(source,t);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);
    require(ctx_none(source,retained)||ctx_side(source,retained)==1,"Cruel Weather violates retained Ring actor");
    const auto card=ev_card(t,id);require(ecw_supported(t,card)&&ctx_side(t,obs_attr(t,card,"side"))==1,"Unsupported Cruel Weather card");
    require(contains_string(source,tc_hand(source,1),id),"Cruel Weather absent from hand");
    require(ev_requirement(source,t,card),"Cruel Weather requirement false");
    const auto die=ev_payment(source,t,1,index,card);
    const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace"));
    const auto targets=ecw_targets(source);
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));
    if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    rr_set_attr(s,ev_payment(s,t,1,index,card),"used",rr_scalar(s,WOTR_BOOL,1));
    if(targets.size()>=2){
        rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,"Cruel Weather")},
            {"side",rr_string(s,"sh")},{"decider",rr_string(s,"sh")},{"options",ecw_options(s,targets)},
            {"die_face",rr_string(s,face)},{"deck",rr_string(s,"character")}}));
        tc_open_hand_limit_discard(s);
    }else{ecw_move(s,targets.empty()?std::string_view{}:std::string_view(targets.front()));ed_commit(s,1,id,"character",face);}
    am_Builder events;std::vector<uint64_t> rows;cau_finish(s,t,result.random.get(),-1,events,rows);
    events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
fm_Result apply_cruel_weather_choice(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index){
    require(side==1,"Cruel Weather choice belongs to Shadow");const auto p=ecw_validate_pending(source,t);
    const auto targets=ecw_targets(source);require(index>=0&&static_cast<uint64_t>(index)<targets.size(),"Cruel Weather choice out of range");
    const std::string id(text(source,field(source,p,"card_id"))),face(text(source,field(source,p,"die_face")));
    const auto oldmarker=obs_attr(source,source.root,"_battle_resolved_attacker",false);
    const int marker=ctx_none(source,oldmarker)?-1:ctx_side(source,oldmarker);
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));
    if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;
    rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));quiet_stable_action_gate(s,t);ring_waiver(s,0);ring_waiver(s,1);
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    ecw_move(s,targets[index]);ed_commit(s,1,id,"character",face);
    am_Builder events;std::vector<uint64_t> rows;cau_finish(s,t,result.random.get(),marker,events,rows);
    events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
