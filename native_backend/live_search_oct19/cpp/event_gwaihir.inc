// Gwaihir's full single/group lattice and one card-funded companion flight.
bool egw_supported(const Snapshot& t,uint64_t card){return card!=missing&&text_is(t,obs_attr(t,card,"card_name"),"Gwaihir the Windlord");}
bool egw_pending(const Snapshot& s){
    const auto p=obs_attr(s,s.root,"pending_card_option",false);
    return !ctx_none(s,p)&&text_is(s,field(s,p,"card_name",false),"Gwaihir the Windlord")&&text_is(s,field(s,p,"side",false),"fp")&&text_is(s,field(s,p,"decider",false),"fp");
}
bool egw_destination(const Snapshot& s,const Snapshot& t,std::string_view start,std::string_view target,int64_t steps){
    if(start==target)return true;
    const auto sieges=obs_attr(s,s.root,"besieged_strongholds");
    // Gwaihir and Swifter permit entering, not leaving, a friendly siege.
    if(contains_string(s,sieges,start)&&at_controller(s,t,start)==0)return false;
    if(cmp_reachable(s,t,start,std::max<int64_t>(0,steps)).count(std::string(target)))return true;
    if(!contains_string(s,sieges,target))return false;
    std::set<std::string> seen{std::string(start)},frontier{std::string(start)};
    for(int64_t depth=0;depth<steps&&!frontier.empty();++depth){std::set<std::string> next;
        for(const auto& from:frontier)for(const auto& region:fr_neighbors(s,from)){
            const auto dest=sanitize_name(region);if(!seen.insert(dest).second)continue;
            if(dest==target)return true;
            if(contains_string(s,sieges,dest)||hp_shadow_controlled_stronghold(s,t,dest))continue;
            next.insert(dest);
        }frontier=std::move(next);
    }return false;
}
std::set<std::string> egw_destinations(const Snapshot& s,const Snapshot& t,std::string_view start,int64_t movement,bool separation){
    const auto fs=obs_attr(s,s.root,"fellowship"),sieges=obs_attr(s,s.root,"besieged_strongholds");
    if(separation&&truth(s,obs_attr(s,fs,"mordor_entered")))return {"__remove_from_play__"};
    if(separation&&contains_string(s,sieges,start)&&at_controller(s,t,start)==0)return {std::string(start)};
    auto result=cmp_reachable(s,t,start,std::max<int64_t>(0,movement));
    if(separation)result.insert(std::string(start));else result.erase(std::string(start));
    for(auto region:child_edges(s,sieges)){
        const auto dest=sanitize_name(text(s,region));
        if((separation||dest!=start)&&!result.count(dest)&&egw_destination(s,t,start,dest,std::max<int64_t>(0,movement)))result.insert(dest);
    }return result;
}
uint64_t egw_options(Snapshot& s,const Snapshot& t,bool level_based=false){
    std::vector<uint64_t> choices;std::set<std::tuple<std::string,std::vector<std::string>,std::string>> seen;
    auto emit=[&](const std::string& action,const std::vector<uint64_t>& group,const std::set<std::string>& destinations,bool named){
        std::vector<std::string> names;if(named)for(auto comp:group)names.emplace_back(text(s,obs_attr(s,comp,"name")));
        auto encode_names=[&](){std::vector<uint64_t> values;for(const auto& name:names)values.push_back(rr_string(s,name));return cp_container(s,WOTR_LIST,values);};
        // The separation spec builds one list outside its destination loop;
        // the board-group spec builds a fresh list for each destination.
        const auto shared_names=named&&action=="group"?encode_names():missing;
        for(const auto& dest:destinations){
            if(!seen.insert({action,names,dest}).second)continue;
            if(named){const auto encoded=shared_names==missing?encode_names():shared_names;choices.push_back(ev_dict(s,{{"companion_action",rr_string(s,action)},{"companions",encoded},{"destination",rr_string(s,dest)}}));}
            else choices.push_back(ev_dict(s,{{"companion_action",rr_string(s,action)},{"destination",rr_string(s,dest)}}));
        }
    };
    const auto fs=obs_attr(s,s.root,"fellowship");const auto members=cmp_separable(s);const auto separated=egw_destinations(s,t,glb_fellowship_key(s,fs),ht_integer(s,fs,"progress")+4,true);
    auto extra=[&](const std::vector<uint64_t>& group){return cm_allowance(s,group)+2;};
    for(auto member:members)emit("sep_"+std::string(text(s,obs_attr(s,member,"name"))),{member},level_based?egw_destinations(s,t,glb_fellowship_key(s,fs),ht_integer(s,fs,"progress")+extra({member}),true):separated,false);
    for(size_t count=2;count<=members.size();++count)cmp_combinations(members,count,[&](const std::vector<uint64_t>& group){emit("group",group,level_based?egw_destinations(s,t,glb_fellowship_key(s,fs),ht_integer(s,fs,"progress")+extra(group),true):separated,true);});
    const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));
    for(size_t i=0;i+1<armies.size();i+=2){
        const auto here=cm_companions(s,armies[i+1]);if(here.empty())continue;
        const auto destinations=egw_destinations(s,t,sanitize_name(text(s,armies[i])),4,false);
        for(auto member:here)emit("move_"+std::string(text(s,obs_attr(s,member,"name"))),{member},level_based?egw_destinations(s,t,sanitize_name(text(s,armies[i])),extra({member}),false):destinations,false);
        for(size_t count=2;count<=here.size();++count)cmp_combinations(here,count,[&](const std::vector<uint64_t>& group){emit("move_group",group,level_based?egw_destinations(s,t,sanitize_name(text(s,armies[i])),extra(group),false):destinations,true);});
    }return cp_container(s,WOTR_LIST,choices);
}
void egw_effect(Snapshot& s,const Snapshot& t,uint64_t choice){
    const std::string action(text(s,field(s,choice,"companion_action"))),destination(text(s,field(s,choice,"destination")));const auto target=sanitize_name(destination);
    const auto fs=obs_attr(s,s.root,"fellowship"),list=obs_attr(s,fs,"companions");
    const bool group=action=="group",sep=action.rfind("sep_",0)==0,movegroup=action=="move_group",move=action.rfind("move_",0)==0;
    require(group||sep||movegroup||move,"Unsupported Gwaihir companion choice");std::vector<std::string> names;
    if(group||movegroup)for(auto value:child_edges(s,field(s,choice,"companions")))names.emplace_back(text(s,value));
    else names.push_back(action.substr(sep?4:5));
    if(group||sep){
        const bool remove=destination=="__remove_from_play__"&&truth(s,obs_attr(s,fs,"mordor_entered"));
        std::vector<uint64_t> picked;for(auto comp:child_edges(s,list))if(std::find(names.begin(),names.end(),text(s,obs_attr(s,comp,"name")))!=names.end()){picked.push_back(comp);if(sep&&!remove)break;}
        require(!picked.empty()&&(!group||picked.size()==names.size()),"Gwaihir separation member absent");
        require(remove||egw_destination(s,t,glb_fellowship_key(s,fs),target,ht_integer(s,fs,"progress")+4),"Gwaihir separation destination unreachable");
        for(auto comp:picked){auto remaining=child_edges(s,list);const auto found=std::find_if(remaining.begin(),remaining.end(),[&](uint64_t other){return ft_companion_equal(s,other,comp);});require(found!=remaining.end(),"Gwaihir separation member disappeared");remaining.erase(found);set_edges(s,list,remaining);if(!remove)hp_place(s,t,comp,target);}
        ft_guide(s); // Any composition change permits a Guide choice (p37).
        cbp_companion_death(s,nullptr,false); // Board and Mordor separation both expire required table cards.
        return;
    }
    const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));uint64_t source_army=missing;std::string start;
    for(size_t i=0;i+1<armies.size();i+=2){
        const auto here=cm_companions(s,armies[i+1]);bool all=!names.empty();
        for(const auto& name:names){bool found=false;for(auto comp:here)found|=text_is(s,obs_attr(s,comp,"name"),name);all&=found;}
        if(all){source_army=armies[i+1];start=sanitize_name(text(s,armies[i]));break;}
    }
    require(source_army!=missing,"Gwaihir moving group is not co-located");require(egw_destination(s,t,start,target,4),"Gwaihir board destination unreachable");
    std::vector<uint64_t> picked;for(auto comp:cm_companions(s,source_army))if(std::find(names.begin(),names.end(),text(s,obs_attr(s,comp,"name")))!=names.end())picked.push_back(comp);
    cm_transfer(s,t,start,target,picked);
}
void apply_gwaihir_event_inplace(Snapshot& s,const Snapshot& t,void* rng,int side,int64_t index,std::string_view id,bool free_ents,am_Builder& ef_events,std::vector<uint64_t>& ef_annotations){
    const auto& source=s;ef_validate_free(source,t,side,index,id,free_ents);
    const int ef_marker=ctx_side(source,obs_attr(source,source.root,"_battle_resolved_attacker",false));const auto ef_unused=ef_unused_dice(source);

    require(side==0,"Gwaihir belongs to FP");quiet_action_gate(source,t,false,false,false,false,false,false,false,false,false,true,(false||free_ents),true);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==0,"Gwaihir violates retained Ring actor");
    const auto card=ev_card(t,id);require(egw_supported(t,card)&&contains_string(source,tc_hand(source,0),id),"Gwaihir absent from acting hand");require(ev_requirement(source,t,card),"Gwaihir requirement false");
    const auto die=free_ents?missing:ev_payment(source,t,0,index,card);const std::string face(free_ents?std::string_view{}:enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace")),deck(ev_lower(text(t,obs_attr(t,card,"deck"))));
    
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));if(!free_ents)rr_set_attr(s,ev_payment(s,t,0,index,card),"used",rr_scalar(s,WOTR_BOOL,1));
    const auto options=egw_options(s,t);const auto choices=child_edges(s,options);
    if(choices.size()>1){
        rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,"Gwaihir the Windlord")},{"side",rr_string(s,"fp")},{"decider",rr_string(s,"fp")},{"options",options},{"die_face",ef_pending_face_node(s,face)},{"deck",rr_string(s,deck)}}));tc_open_hand_limit_discard(s);
    }else{if(!choices.empty())egw_effect(s,t,choices.front());er_commit(s,t,id,deck,face);}
    cau_finish(s,t,rng,free_ents?ef_final_marker(s,ef_marker,ef_unused):-1,ef_events,ef_annotations);
}
fm_Result apply_gwaihir_event(const Snapshot& source,const Snapshot& t,const void* random_source,int32_t side,int64_t index,std::string_view id){
    require(side==0,"Gwaihir belongs to FP");quiet_action_gate(source,t,false,false,false,false,false,false,false,false,false,true,false,true);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==0,"Gwaihir violates retained Ring actor");
    const auto card=ev_card(t,id);require(egw_supported(t,card)&&contains_string(source,tc_hand(source,0),id),"Gwaihir absent from acting hand");require(ev_requirement(source,t,card),"Gwaihir requirement false");
    const auto die=ev_payment(source,t,0,index,card);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace")),deck(ev_lower(text(t,obs_attr(t,card,"deck"))));
    
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());
    am_Builder ef_events;std::vector<uint64_t> ef_annotations;
    apply_gwaihir_event_inplace(*result.state,t,result.random.get(),side,index,id,false,ef_events,ef_annotations);
    ef_events.out.root=ef_events.container(WOTR_LIST,ef_annotations);result.events=std::make_unique<Snapshot>(std::move(ef_events.out));return result;
}
std::unique_ptr<Snapshot> native_gwaihir_choice_actions(const Snapshot& source,int32_t side){
    require(side==0&&egw_pending(source),"No FP Gwaihir choice pending");auto out=std::make_unique<Snapshot>(source);auto& s=*out;const auto p=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t index=0;
    for(auto choice:child_edges(s,field(s,p,"options")))rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",choice}})}));
    s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
fm_Result apply_gwaihir_choice(const Snapshot& source,const Snapshot& t,const void* random_source,int32_t side,int64_t index){
    const bool free_ents=ef_pending_is_free(source,obs_attr(source,source.root,"pending_card_option",false));

    require(side==0&&egw_pending(source),"No FP Gwaihir choice pending");const auto p=obs_attr(source,source.root,"pending_card_option");const auto options=child_edges(source,field(source,p,"options"));require(index>=0&&static_cast<uint64_t>(index)<options.size(),"Gwaihir choice index exceeds menu");
    const std::string id(text(source,field(source,p,"card_id"))),deck(text(source,field(source,p,"deck"))),face(ef_pending_face(source,p));require(egw_supported(t,ev_card(t,id))&&contains_string(source,tc_hand(source,0),id),"Pending Gwaihir card absent");
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;am_Builder events;std::vector<uint64_t> rows;
    const auto choice=child_edges(s,field(s,obs_attr(s,s.root,"pending_card_option"),"options"))[index],marker=obs_attr(s,s.root,"_battle_resolved_attacker",false);const int marker_side=ctx_none(s,marker)?-1:ctx_side(s,marker);
    rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));quiet_action_gate(s,t,false,false,false,false,false,false,false,false,false,true,(false||free_ents),true);ring_waiver(s,0);ring_waiver(s,1);
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));egw_effect(s,t,choice);er_commit(s,t,id,deck,face);cau_finish(s,t,result.random.get(),marker_side,events,rows);
    events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
