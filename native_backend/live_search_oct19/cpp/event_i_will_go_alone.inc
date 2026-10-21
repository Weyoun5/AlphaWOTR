// Card-specific group ordering and aliases; shared movement and arrival bodies.
bool eal_supported(const Snapshot& t,uint64_t card){return card!=missing&&text_is(t,obs_attr(t,card,"card_name"),"I Will Go Alone");}
bool eal_pending(const Snapshot& s){const auto p=obs_attr(s,s.root,"pending_card_option",false);return !ctx_none(s,p)&&text_is(s,field(s,p,"card_name",false),"I Will Go Alone");}
uint64_t eal_options(Snapshot& s,const Snapshot& t){
    const auto fs=obs_attr(s,s.root,"fellowship");const auto members=cmp_separable(s);const auto start=glb_fellowship_key(s,fs);
    const bool mordor=truth(s,obs_attr(s,fs,"mordor_entered")),siege=contains_string(s,obs_attr(s,s.root,"besieged_strongholds"),start)&&at_controller(s,t,start)==0;
    std::vector<uint64_t> rows;std::set<std::pair<std::vector<std::string>,std::string>> seen;
    std::map<int64_t,std::set<std::string>> by_range;
    auto emit=[&](const std::vector<uint64_t>& group,bool shared){
        int64_t highest=0;std::vector<std::string> names;for(auto comp:group){names.emplace_back(text(s,obs_attr(s,comp,"name")));highest=std::max(highest,ht_integer(s,comp,"level"));}
        const auto distance=std::max<int64_t>(0,ht_integer(s,fs,"progress")+highest+1);auto found=by_range.find(distance);
        if(found==by_range.end()){
            std::set<std::string> targets;
            if(mordor)targets.insert("__remove_from_play__");else if(siege)targets.insert(start);
            else{targets=cmp_reachable(s,t,start,distance);targets.insert(start);}
            found=by_range.emplace(distance,std::move(targets)).first;
        }
        auto encode=[&](){std::vector<uint64_t> values;for(const auto& name:names)values.push_back(rr_string(s,name));return cp_container(s,WOTR_LIST,values);};
        const auto shared_names=shared?encode():missing;
        for(const auto& destination:found->second){
            if(!seen.insert({names,destination}).second)continue;
            rows.push_back(ev_dict(s,{{"companions",shared?shared_names:encode()},{"destination",rr_string(s,destination)}}));
        }
    };
    for(auto comp:members)emit({comp},false);
    for(size_t size=2;size<=members.size();++size)cmp_combinations(members,size,[&](const std::vector<uint64_t>& group){emit(group,true);});
    return cp_container(s,WOTR_LIST,rows);
}
bool eal_effect(Snapshot& s,const Snapshot& t,uint64_t option){
    const auto fs=obs_attr(s,s.root,"fellowship"),list=obs_attr(s,fs,"companions");std::vector<std::string> names;std::string destination;
    if(option!=missing){for(auto name:child_edges(s,field(s,option,"companions")))names.emplace_back(text(s,name));destination=text(s,field(s,option,"destination"));}
    if((names.empty()||destination.empty())&&!cmp_separable(s).empty())return false;
    if(!names.empty()&&!destination.empty()){
        const bool remove=destination=="__remove_from_play__"&&truth(s,obs_attr(s,fs,"mordor_entered"));
        const auto target=sanitize_name(destination),start=glb_fellowship_key(s,fs);std::vector<uint64_t> picked;int64_t highest=0;
        for(auto comp:child_edges(s,list))if(std::find(names.begin(),names.end(),text(s,obs_attr(s,comp,"name")))!=names.end()){highest=picked.empty()?ht_integer(s,comp,"level"):std::max(highest,ht_integer(s,comp,"level"));picked.push_back(comp);}
        if(!remove&&target!=start&&!cmp_reachable(s,t,start,std::max<int64_t>(0,ht_integer(s,fs,"progress")+highest+1)).count(target))return false;
        std::vector<std::string> separated;
        for(auto comp:picked){
            auto remaining=child_edges(s,list);const auto found=std::find_if(remaining.begin(),remaining.end(),[&](uint64_t other){return ft_companion_equal(s,other,comp);});
            require(found!=remaining.end(),"Alone companion disappeared during separation");remaining.erase(found);set_edges(s,list,remaining);
            if(!remove)hp_place(s,t,comp,target);separated.emplace_back(text(s,obs_attr(s,comp,"name")));
        }
        if(!separated.empty()){
            ft_guide(s); // Any composition change permits a Guide choice (p37).
            cbp_companion_death(s,nullptr,false); // Separation expires required table cards without a Hunt casualty.
        }
    }
    rr_set_attr(s,fs,"corruption",rr_scalar(s,WOTR_INT,std::max<int64_t>(0,ht_integer(s,fs,"corruption")-1)));return true;
}
void apply_alone_event_inplace(Snapshot& s,const Snapshot& t,void* rng,int side,int64_t index,std::string_view id,bool free_ents,am_Builder& ef_events,std::vector<uint64_t>& ef_annotations){
    const auto& source=s;ef_validate_free(source,t,side,index,id,free_ents);
    const int ef_marker=ctx_side(source,obs_attr(source,source.root,"_battle_resolved_attacker",false));const auto ef_unused=ef_unused_dice(source);

    require(side==0,"I Will Go Alone belongs to FP");quiet_action_gate(source,t,false,false,false,false,false,false,false,false,true,false,(false||free_ents),true);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==0,"Alone violates retained Ring actor");
    const auto card=ev_card(t,id);require(eal_supported(t,card)&&contains_string(source,tc_hand(source,0),id),"Alone card absent");require(ev_requirement(source,t,card),"Alone requirement false");
    const auto die=free_ents?missing:ev_payment(source,t,0,index,card);const std::string face(free_ents?std::string_view{}:enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace"));
    
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));if(!free_ents)rr_set_attr(s,ev_payment(s,t,0,index,card),"used",rr_scalar(s,WOTR_BOOL,1));
    const auto options=eal_options(s,t);const auto rows=child_edges(s,options);
    if(rows.size()>1){rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,"I Will Go Alone")},{"side",rr_string(s,"fp")},{"decider",rr_string(s,"fp")},{"options",options},{"die_face",ef_pending_face_node(s,face)},{"deck",rr_string(s,"character")}}));tc_open_hand_limit_discard(s);}
    else if(eal_effect(s,t,rows.empty()?missing:rows.front())||rows.empty())er_commit(s,t,id,"character",face);
    cau_finish(s,t,rng,free_ents?ef_final_marker(s,ef_marker,ef_unused):-1,ef_events,ef_annotations);
}
fm_Result apply_alone_event(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index,std::string_view id){
    require(side==0,"I Will Go Alone belongs to FP");quiet_action_gate(source,t,false,false,false,false,false,false,false,false,true,false,false,true);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==0,"Alone violates retained Ring actor");
    const auto card=ev_card(t,id);require(eal_supported(t,card)&&contains_string(source,tc_hand(source,0),id),"Alone card absent");require(ev_requirement(source,t,card),"Alone requirement false");
    const auto die=ev_payment(source,t,0,index,card);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace"));
    
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());
    am_Builder ef_events;std::vector<uint64_t> ef_annotations;
    apply_alone_event_inplace(*result.state,t,result.random.get(),side,index,id,false,ef_events,ef_annotations);
    ef_events.out.root=ef_events.container(WOTR_LIST,ef_annotations);result.events=std::make_unique<Snapshot>(std::move(ef_events.out));return result;
}
uint64_t eal_validate(const Snapshot& s,const Snapshot& t){
    require(eal_pending(s),"No Alone option pending");const auto p=obs_attr(s,s.root,"pending_card_option");
    require(text_is(s,field(s,p,"side"),"fp")&&text_is(s,field(s,p,"decider"),"fp")&&!truth(s,field(s,p,"direct",false)),"Alone pending owner/stage invalid");
    const auto id=text(s,field(s,p,"card_id"));require(eal_supported(t,ev_card(t,id)),"Alone pending card identity invalid");/* package R: hand membership is not a menu precondition (frozen card_option menu, ai.py 7764-7770) */
    require(text_is(s,field(s,p,"deck"),"character"),"Alone pending deck invalid");const auto face=ef_pending_face(s,p);
    require(face.empty()||face=="CHARACTER"||face=="EVENT"||face=="WILL_OF_WEST","Alone pending payment invalid");
    const auto options=child_edges(s,field(s,p,"options"));require(options.size()>1,"Alone paid pending requires multiple stored choices");
    for(auto option:options){require(typed(s,option,WOTR_DICT,"Alone option must be dictionary").count==4,"Alone option shape invalid");const auto names=field(s,option,"companions");require(typed(s,names,WOTR_LIST,"Alone companion names must be list").count>0,"Alone choice has no companions");for(auto name:child_edges(s,names))text(s,name);text(s,field(s,option,"destination"));}
    return p;
}
std::unique_ptr<Snapshot> native_alone_choices(const Snapshot& source,const Snapshot& t,int side){
    require(side==0,"Alone choices belong to FP");eal_validate(source,t);auto result=std::make_unique<Snapshot>(source);auto& s=*result;
    const auto p=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t index=0;
    for(auto choice:child_edges(s,field(s,p,"options")))rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",choice}})}));
    s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
fm_Result apply_alone_choice(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index){
    const bool free_ents=ef_pending_is_free(source,obs_attr(source,source.root,"pending_card_option",false));

    require(side==0,"Alone choice belongs to FP");const auto p=eal_validate(source,t);const auto options=child_edges(source,field(source,p,"options"));require(index>=0&&static_cast<uint64_t>(index)<options.size(),"Alone choice index exceeds menu");
    const std::string id(text(source,field(source,p,"card_id"))),face(ef_pending_face(source,p));const auto marker=obs_attr(source,source.root,"_battle_resolved_attacker",false);const int oldside=ctx_none(source,marker)?-1:ctx_side(source,marker);
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;
    const auto choice=child_edges(s,field(s,obs_attr(s,s.root,"pending_card_option"),"options"))[index];rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));
    quiet_action_gate(s,t,false,false,false,false,false,false,false,false,true,false,(false||free_ents),true);ring_waiver(s,0);ring_waiver(s,1);rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    if(ev_requirement(s,t,ev_card(t,id))){const bool played=eal_effect(s,t,choice);if(played||!truth(s,eal_options(s,t)))er_commit(s,t,id,"character",face);}
    am_Builder events;std::vector<uint64_t> rows;cau_finish(s,t,result.random.get(),oldside,events,rows);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
