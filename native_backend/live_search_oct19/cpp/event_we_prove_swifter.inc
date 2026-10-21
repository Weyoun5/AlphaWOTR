// Exact per-member ranges; common flight topology and placement primitives.
uint64_t egw_options(Snapshot&,const Snapshot&,bool);
bool egw_destination(const Snapshot&,const Snapshot&,std::string_view,std::string_view,int64_t);
bool esw_supported(const Snapshot& t,uint64_t card){return card!=missing&&text_is(t,obs_attr(t,card,"card_name"),"We Prove Swifter");}
bool esw_pending(const Snapshot& s){const auto p=obs_attr(s,s.root,"pending_card_option",false);return !ctx_none(s,p)&&text_is(s,field(s,p,"card_name",false),"We Prove Swifter");}
uint64_t esw_options(Snapshot& s,const Snapshot& t){return egw_options(s,t,true);}
bool esw_effect(Snapshot& s,const Snapshot& t,uint64_t option){
    if(option==missing)return false;
    const std::string action(text(s,field(s,option,"companion_action"))),destination(text(s,field(s,option,"destination"))),target=sanitize_name(destination);
    const bool group=action=="group",sep=action.rfind("sep_",0)==0,movegroup=action=="move_group",move=action.rfind("move_",0)==0;
    require(group||sep||movegroup||move,"Swifter companion action invalid");
    std::vector<std::string> names;
    if(group||movegroup)for(auto name:child_edges(s,field(s,option,"companions")))names.emplace_back(text(s,name));
    else names.push_back(action.substr(sep?4:5));
    const auto fs=obs_attr(s,s.root,"fellowship"),list=obs_attr(s,fs,"companions");
    if(group||sep){
        const bool remove=destination=="__remove_from_play__"&&truth(s,obs_attr(s,fs,"mordor_entered"));
        std::vector<uint64_t> picked;int64_t highest=0;
        for(auto comp:child_edges(s,list))if(std::find(names.begin(),names.end(),text(s,obs_attr(s,comp,"name")))!=names.end()){
            highest=picked.empty()?ht_integer(s,comp,"level"):std::max(highest,ht_integer(s,comp,"level"));picked.push_back(comp);if(sep&&!remove)break;
        }
        if(group&&(names.empty()||picked.size()!=names.size()))return false;
        if(!remove&&!egw_destination(s,t,glb_fellowship_key(s,fs),target,ht_integer(s,fs,"progress")+highest+2))return false;
        for(auto comp:picked){
            auto remaining=child_edges(s,list);const auto found=std::find_if(remaining.begin(),remaining.end(),[&](uint64_t other){return ft_companion_equal(s,other,comp);});
            require(found!=remaining.end(),"Swifter companion disappeared during separation");remaining.erase(found);set_edges(s,list,remaining);if(!remove)hp_place(s,t,comp,target);
        }
        if(!picked.empty()){
            ft_guide(s); // Any composition change permits a Guide choice (p37).
            cbp_companion_death(s,nullptr,false); // Separation is not a Hunt casualty.
        }
        return true;
    }
    // Find the co-located moving group; Shadowfax applies to its composition.
    const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));uint64_t origin=missing;std::string start;
    for(size_t i=0;i<armies.size();i+=2){
        const auto here=cm_companions(s,armies[i+1]);bool all=!names.empty();
        for(const auto& name:names){bool found=false;for(auto comp:here)found|=text_is(s,obs_attr(s,comp,"name"),name);all&=found;}
        if(all&&origin==missing){origin=armies[i+1];start=sanitize_name(text(s,armies[i]));}
    }
    if(origin==missing)return false;
    std::vector<uint64_t> picked;for(auto comp:cm_companions(s,origin))if(std::find(names.begin(),names.end(),text(s,obs_attr(s,comp,"name")))!=names.end())picked.push_back(comp);
    if(!egw_destination(s,t,start,target,cm_allowance(s,picked)+2))return false;
    cm_transfer(s,t,start,target,picked);return true;
}
bool esw_resolve(Snapshot& s,const Snapshot& t,uint64_t choice){
    if(esw_effect(s,t,choice))return true;
    const auto options=child_edges(s,esw_options(s,t));if(options.empty())return true;
    if(choice==missing)return false;
    // A refunded handler still commits when the selected choice remains a
    // current public option.
    for(auto option:options){bool matches=true;const auto fields=child_edges(s,option);
        for(size_t i=0;i<fields.size();i+=2){const std::string key(text(s,fields[i]));if(!key.empty()&&key.front()=='_')continue;const auto actual=field(s,choice,key.c_str(),false);
            if(actual==missing){matches=false;break;}
            if(s.nodes[fields[i+1]].tag==WOTR_STRING&&s.nodes[actual].tag==WOTR_STRING)matches&=sanitize_name(text(s,fields[i+1]))==sanitize_name(text(s,actual));
            else matches&=ap_equal(ap_freeze(s,fields[i+1]),ap_freeze(s,actual));
        }if(matches)return true;
    }return false;
}
void apply_swifter_event_inplace(Snapshot& s,const Snapshot& t,void* rng,int side,int64_t index,std::string_view id,bool free_ents,am_Builder& ef_events,std::vector<uint64_t>& ef_annotations){
    const auto& source=s;ef_validate_free(source,t,side,index,id,free_ents);
    const int ef_marker=ctx_side(source,obs_attr(source,source.root,"_battle_resolved_attacker",false));const auto ef_unused=ef_unused_dice(source);

    require(side==0,"We Prove Swifter belongs to FP");quiet_action_gate(source,t,false,false,false,false,false,false,false,false,true,false,(false||free_ents),true);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==0,"Swifter violates retained Ring actor");
    const auto card=ev_card(t,id);require(esw_supported(t,card)&&contains_string(source,tc_hand(source,0),id),"Swifter card absent");require(ev_requirement(source,t,card),"Swifter requirement false");
    const auto die=free_ents?missing:ev_payment(source,t,0,index,card);const std::string face(free_ents?std::string_view{}:enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace"));
    
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));if(!free_ents)rr_set_attr(s,ev_payment(s,t,0,index,card),"used",rr_scalar(s,WOTR_BOOL,1));
    const auto options=esw_options(s,t);const auto rows=child_edges(s,options);
    if(rows.size()>1){rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,"We Prove Swifter")},{"side",rr_string(s,"fp")},{"decider",rr_string(s,"fp")},{"options",options},{"die_face",ef_pending_face_node(s,face)},{"deck",rr_string(s,"character")}}));tc_open_hand_limit_discard(s);}
    else if(esw_resolve(s,t,rows.empty()?missing:rows.front()))er_commit(s,t,id,"character",face);
    cau_finish(s,t,rng,free_ents?ef_final_marker(s,ef_marker,ef_unused):-1,ef_events,ef_annotations);
}
fm_Result apply_swifter_event(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index,std::string_view id){
    require(side==0,"We Prove Swifter belongs to FP");quiet_action_gate(source,t,false,false,false,false,false,false,false,false,true,false,false,true);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==0,"Swifter violates retained Ring actor");
    const auto card=ev_card(t,id);require(esw_supported(t,card)&&contains_string(source,tc_hand(source,0),id),"Swifter card absent");require(ev_requirement(source,t,card),"Swifter requirement false");
    const auto die=ev_payment(source,t,0,index,card);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace"));
    
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());
    am_Builder ef_events;std::vector<uint64_t> ef_annotations;
    apply_swifter_event_inplace(*result.state,t,result.random.get(),side,index,id,false,ef_events,ef_annotations);
    ef_events.out.root=ef_events.container(WOTR_LIST,ef_annotations);result.events=std::make_unique<Snapshot>(std::move(ef_events.out));return result;
}
uint64_t esw_validate(const Snapshot& s,const Snapshot& t){
    require(esw_pending(s),"No Swifter option pending");const auto p=obs_attr(s,s.root,"pending_card_option");
    require(text_is(s,field(s,p,"side"),"fp")&&text_is(s,field(s,p,"decider"),"fp")&&!truth(s,field(s,p,"direct",false)),"Swifter pending owner/stage invalid");
    const auto id=text(s,field(s,p,"card_id"));require(esw_supported(t,ev_card(t,id)),"Swifter pending card identity invalid");/* package R: hand membership is not a menu precondition (frozen card_option menu, ai.py 7764-7770) */
    require(text_is(s,field(s,p,"deck"),"character"),"Swifter pending deck invalid");const auto face=ef_pending_face(s,p);
    require(face.empty()||face=="CHARACTER"||face=="EVENT"||face=="WILL_OF_WEST","Swifter pending payment invalid");
    const auto options=child_edges(s,field(s,p,"options"));require(options.size()>1,"Swifter paid pending requires multiple stored choices");
    for(auto option:options){
        const auto count=typed(s,option,WOTR_DICT,"Swifter option must be dictionary").count;
        const auto kind=text(s,field(s,option,"companion_action"));const bool group=kind=="group"||kind=="move_group";
        require(group||kind.rfind("sep_",0)==0||kind.rfind("move_",0)==0,"Swifter pending action invalid");
        require(count==(group?6:4),"Swifter option shape invalid");
        if(group){const auto names=field(s,option,"companions");require(typed(s,names,WOTR_LIST,"Swifter names must be list").count>0,"Swifter group empty");for(auto name:child_edges(s,names))text(s,name);}
        text(s,field(s,option,"destination"));
    }
    return p;
}
std::unique_ptr<Snapshot> native_swifter_choices(const Snapshot& source,const Snapshot& t,int side){
    require(side==0,"Swifter choices belong to FP");esw_validate(source,t);auto result=std::make_unique<Snapshot>(source);auto& s=*result;
    const auto p=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t index=0;
    for(auto choice:child_edges(s,field(s,p,"options")))rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",choice}})}));
    s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
fm_Result apply_swifter_choice(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index){
    const bool free_ents=ef_pending_is_free(source,obs_attr(source,source.root,"pending_card_option",false));

    require(side==0,"Swifter choice belongs to FP");const auto p=esw_validate(source,t);const auto options=child_edges(source,field(source,p,"options"));require(index>=0&&static_cast<uint64_t>(index)<options.size(),"Swifter choice index exceeds menu");
    const std::string id(text(source,field(source,p,"card_id"))),face(ef_pending_face(source,p));const auto marker=obs_attr(source,source.root,"_battle_resolved_attacker",false);const int oldside=ctx_none(source,marker)?-1:ctx_side(source,marker);
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;
    const auto choice=child_edges(s,field(s,obs_attr(s,s.root,"pending_card_option"),"options"))[index];rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));
    quiet_action_gate(s,t,false,false,false,false,false,false,false,false,true,false,(false||free_ents),true);ring_waiver(s,0);ring_waiver(s,1);rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    if(ev_requirement(s,t,ev_card(t,id))){if(esw_resolve(s,t,choice))er_commit(s,t,id,"character",face);}
    am_Builder events;std::vector<uint64_t> rows;cau_finish(s,t,result.random.get(),oldside,events,rows);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
