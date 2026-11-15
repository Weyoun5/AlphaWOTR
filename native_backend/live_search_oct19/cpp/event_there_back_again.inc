// Card-specific group ordering and aliases; shared movement and arrival bodies.
bool etb_supported(const Snapshot& t,uint64_t card){return card!=missing&&text_is(t,obs_attr(t,card,"card_name"),"There and Back Again");}
bool etb_pending(const Snapshot& s){const auto p=obs_attr(s,s.root,"pending_card_option",false);return !ctx_none(s,p)&&text_is(s,field(s,p,"card_name",false),"There and Back Again");}
uint64_t etb_options(Snapshot& s,const Snapshot& t){
    const auto guide=obs_attr(s,obs_attr(s,s.root,"fellowship"),"guide",false);
    if(!ctx_none(s,guide)&&text_is(s,obs_attr(s,guide,"name"),"Gollum")&&cmp_separable(s).empty())
        return cp_container(s,WOTR_LIST,{ev_dict(s,{{"_taba_done",rr_scalar(s,WOTR_BOOL,1)}})});
    return eal_options(s,t); // Exact shared _separation_card(1, True) specification.
}
void etb_politics(Snapshot& s,bool trigger=false){
    const auto controls=ctx_side_value(s,obs_attr(s,s.root,"enemy_controlled"),1);
    const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));
    for(size_t i=0;i<armies.size();i+=2){
        const auto region=sanitize_name(text(s,armies[i]));
        if((region!="dale"&&region!="erebor"&&region!="woodland realm"&&region!="woodlandrealm"&&region!="woodland_realm")||contains_string(s,controls,region))continue;
        for(auto leader:child_edges(s,obs_attr(s,armies[i+1],"leaders")))
            if(object_type(s,leader)=="game_state.Companion"&&(text_is(s,obs_attr(s,leader,"name"),"Gimli")||text_is(s,obs_attr(s,leader,"name"),"Legolas")))trigger=true;
    }
    if(!trigger)return;
    for(const char* nation:{"DWARVES","THE_NORTH"})me_activate(s,nation);
    const auto boxes=obs_attr(s,obs_attr(s,s.root,"political"),"boxes");
    for(const char* nation:{"DWARVES","ELVES","THE_NORTH"}){
        const auto value=typed(s,ctx_nation_value(s,boxes,nation),WOTR_INT,"Back Again political position must be integer").integer;
        const auto next=std::min<int64_t>(wt_active(s,nation)?3:2,value+1);
        if(next>value)me_nation_assign(s,boxes,nation,rr_scalar(s,WOTR_INT,next));
    }
}
bool etb_effect(Snapshot& s,const Snapshot& t,uint64_t option){
    const auto fs=obs_attr(s,s.root,"fellowship"),list=obs_attr(s,fs,"companions");std::vector<std::string> names;std::string destination;
    if(option!=missing){const auto selected=field(s,option,"companions",false);if(selected!=missing)for(auto name:child_edges(s,selected))names.emplace_back(text(s,name));const auto dest=field(s,option,"destination",false);if(dest!=missing)destination=text(s,dest);}
    const bool remove=!names.empty()&&(destination=="__remove_from_play__"||truth(s,obs_attr(s,fs,"mordor_entered")));bool arrival_trigger=false;
    if(remove||(!names.empty()&&!destination.empty())){
        const auto target=sanitize_name(destination),start=glb_fellowship_key(s,fs);std::vector<uint64_t> picked;int64_t highest=0;
        for(auto comp:child_edges(s,list))if(std::find(names.begin(),names.end(),text(s,obs_attr(s,comp,"name")))!=names.end()){highest=picked.empty()?ht_integer(s,comp,"level"):std::max(highest,ht_integer(s,comp,"level"));picked.push_back(comp);}
        if(!remove&&target!=start&&!cmp_reachable(s,t,start,std::max<int64_t>(0,ht_integer(s,fs,"progress")+highest+1)).count(target))return false;
        std::vector<std::string> separated;
        for(auto comp:picked){
            auto remaining=child_edges(s,list);const auto found=std::find_if(remaining.begin(),remaining.end(),[&](uint64_t other){return ft_companion_equal(s,other,comp);});
            require(found!=remaining.end(),"Back Again companion disappeared during separation");remaining.erase(found);set_edges(s,list,remaining);
            if(!remove)hp_place(s,t,comp,target);separated.emplace_back(text(s,obs_attr(s,comp,"name")));
        }
        if(!remove&&(target=="dale"||target=="erebor"||target=="woodland realm"||target=="woodlandrealm"||target=="woodland_realm")&&!contains_string(s,ctx_side_value(s,obs_attr(s,s.root,"enemy_controlled"),1),target))
            arrival_trigger=std::find(separated.begin(),separated.end(),"Gimli")!=separated.end()||std::find(separated.begin(),separated.end(),"Legolas")!=separated.end();
        if(!separated.empty()){
            ft_guide(s); // Any composition change permits a Guide choice (p37).
            cbp_companion_death(s,nullptr,false); // Separation expires required table cards without a Hunt casualty.
        }
    }else{
        const auto guide=obs_attr(s,fs,"guide",false);
        if(ctx_none(s,guide)||!text_is(s,obs_attr(s,guide,"name"),"Gollum")||!cmp_separable(s).empty())return false;
    }
    etb_politics(s,arrival_trigger);return true;
}
void apply_back_again_event_inplace(Snapshot& s,const Snapshot& t,void* rng,int side,int64_t index,std::string_view id,bool free_ents,am_Builder& ef_events,std::vector<uint64_t>& ef_annotations){
    const auto& source=s;ef_validate_free(source,t,side,index,id,free_ents);
    const int ef_marker=ctx_side(source,obs_attr(source,source.root,"_battle_resolved_attacker",false));const auto ef_unused=ef_unused_dice(source);

    require(side==0,"There and Back Again belongs to FP");quiet_action_gate(source,t,false,false,false,false,false,false,false,false,true,false,(false||free_ents),true);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==0,"Back Again violates retained Ring actor");
    const auto card=ev_card(t,id);require(etb_supported(t,card)&&contains_string(source,tc_hand(source,0),id),"Back Again card absent");require(ev_requirement(source,t,card),"Back Again requirement false");
    const auto die=free_ents?missing:ev_payment(source,t,0,index,card);const std::string face(free_ents?std::string_view{}:enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace"));
    
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));if(!free_ents)rr_set_attr(s,ev_payment(s,t,0,index,card),"used",rr_scalar(s,WOTR_BOOL,1));
    const auto options=etb_options(s,t);const auto rows=child_edges(s,options);
    if(rows.size()>1){rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,"There and Back Again")},{"side",rr_string(s,"fp")},{"decider",rr_string(s,"fp")},{"options",options},{"die_face",ef_pending_face_node(s,face)},{"deck",rr_string(s,"character")}}));tc_open_hand_limit_discard(s);}
    else if(etb_effect(s,t,rows.empty()?missing:rows.front())||rows.empty())er_commit(s,t,id,"character",face);
    cau_finish(s,t,rng,free_ents?ef_final_marker(s,ef_marker,ef_unused):-1,ef_events,ef_annotations);
}
fm_Result apply_back_again_event(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index,std::string_view id){
    require(side==0,"There and Back Again belongs to FP");quiet_action_gate(source,t,false,false,false,false,false,false,false,false,true,false,false,true);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==0,"Back Again violates retained Ring actor");
    const auto card=ev_card(t,id);require(etb_supported(t,card)&&contains_string(source,tc_hand(source,0),id),"Back Again card absent");require(ev_requirement(source,t,card),"Back Again requirement false");
    const auto die=ev_payment(source,t,0,index,card);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace"));
    
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());
    am_Builder ef_events;std::vector<uint64_t> ef_annotations;
    apply_back_again_event_inplace(*result.state,t,result.random.get(),side,index,id,false,ef_events,ef_annotations);
    ef_events.out.root=ef_events.container(WOTR_LIST,ef_annotations);result.events=std::make_unique<Snapshot>(std::move(ef_events.out));return result;
}
uint64_t etb_validate(const Snapshot& s,const Snapshot& t){
    require(etb_pending(s),"No Back Again option pending");const auto p=obs_attr(s,s.root,"pending_card_option");
    require(text_is(s,field(s,p,"side"),"fp")&&text_is(s,field(s,p,"decider"),"fp")&&!truth(s,field(s,p,"direct",false)),"Back Again pending owner/stage invalid");
    const auto id=text(s,field(s,p,"card_id"));require(etb_supported(t,ev_card(t,id)),"Back Again pending card identity invalid");/* package R: hand membership is not a menu precondition (frozen card_option menu, ai.py 7764-7770) */
    require(text_is(s,field(s,p,"deck"),"character"),"Back Again pending deck invalid");const auto face=ef_pending_face(s,p);
    require(face.empty()||face=="CHARACTER"||face=="EVENT"||face=="WILL_OF_WEST","Back Again pending payment invalid");
    const auto options=child_edges(s,field(s,p,"options"));require(options.size()>1,"Back Again paid pending requires multiple stored choices");
    for(auto option:options){require(typed(s,option,WOTR_DICT,"Back Again option must be dictionary").count==4,"Back Again option shape invalid");const auto names=field(s,option,"companions");require(typed(s,names,WOTR_LIST,"Back Again companion names must be list").count>0,"Back Again choice has no companions");for(auto name:child_edges(s,names))text(s,name);text(s,field(s,option,"destination"));}
    return p;
}
std::unique_ptr<Snapshot> native_back_again_choices(const Snapshot& source,const Snapshot& t,int side){
    require(side==0,"Back Again choices belong to FP");etb_validate(source,t);auto result=std::make_unique<Snapshot>(source);auto& s=*result;
    const auto p=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t index=0;
    for(auto choice:child_edges(s,field(s,p,"options")))rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",choice}})}));
    s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
fm_Result apply_back_again_choice(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index){
    const bool free_ents=ef_pending_is_free(source,obs_attr(source,source.root,"pending_card_option",false));

    require(side==0,"Back Again choice belongs to FP");const auto p=etb_validate(source,t);const auto options=child_edges(source,field(source,p,"options"));require(index>=0&&static_cast<uint64_t>(index)<options.size(),"Back Again choice index exceeds menu");
    const std::string id(text(source,field(source,p,"card_id"))),face(ef_pending_face(source,p));const auto marker=obs_attr(source,source.root,"_battle_resolved_attacker",false);const int oldside=ctx_none(source,marker)?-1:ctx_side(source,marker);
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;
    const auto choice=child_edges(s,field(s,obs_attr(s,s.root,"pending_card_option"),"options"))[index];rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));
    quiet_action_gate(s,t,false,false,false,false,false,false,false,false,true,false,(false||free_ents),true);ring_waiver(s,0);ring_waiver(s,1);rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    if(ev_requirement(s,t,ev_card(t,id))){const bool played=etb_effect(s,t,choice);if(played||!truth(s,etb_options(s,t)))er_commit(s,t,id,"character",face);}
    am_Builder events;std::vector<uint64_t> rows;cau_finish(s,t,result.random.get(),oldside,events,rows);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
