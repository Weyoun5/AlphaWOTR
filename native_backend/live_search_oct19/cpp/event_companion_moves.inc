// Book of Mazarbul / Fear! Fire! Foes!: one paid event, learned groups,
// optional Done, and a final board-wide political condition.
bool efm_supported(const Snapshot& t,uint64_t card){
    if(card==missing)return false;const auto name=obs_attr(t,card,"card_name");return text_is(t,name,"Book of Mazarbul") || text_is(t,name,"Fear! Fire! Foes!");
}
bool efm_pending(const Snapshot& s){
    const auto p=obs_attr(s,s.root,"pending_card_option",false);if(ctx_none(s,p))return false;
    const auto name=field(s,p,"card_name",false);return (text_is(s,name,"Book of Mazarbul") || text_is(s,name,"Fear! Fire! Foes!")) && text_is(s,field(s,p,"side",false),"fp") && text_is(s,field(s,p,"decider",false),"fp");
}
uint64_t efm_options(Snapshot& s,const Snapshot& t,const std::set<std::string>& moved,bool continuation){
    const auto ordinary=cm_chain_options(s,t,moved);auto choices=child_edges(s,ordinary);
    for(auto option:choices){
        const auto group=ev_dict(s,{{"from",field(s,option,"from_region")},{"to",field(s,option,"to_region")},{"companions",field(s,option,"companions")}});
        const auto already=field(s,option,"_comp_moved");std::vector<uint64_t> entries{rr_string(s,"move_group"),group};
        if(continuation){entries.push_back(rr_string(s,"_ffm_moved"));entries.push_back(already);}set_edges(s,option,entries);
    }
    if(!choices.empty() || !continuation)choices.push_back(ev_dict(s,{{"_ffm_done",rr_scalar(s,WOTR_BOOL,1)}}));
    set_edges(s,ordinary,choices);return ordinary;
}
void efm_war(Snapshot& s,std::string_view name){
    const bool book=name=="Book of Mazarbul";const std::set<std::string> targets=book?std::set<std::string>{"erebor","ered luin","eredluin","ered_luin"}:std::set<std::string>{"the shire","shire","bree"};
    const auto entries=child_edges(s,obs_attr(s,s.root,"armies"));const auto shadow=ctx_side_value(s,obs_attr(s,s.root,"enemy_controlled"),1);
    for(size_t i=0;i<entries.size();i+=2){const auto key=sanitize_name(text(s,entries[i]));if(!targets.count(key) || contains_string(s,shadow,key))continue;
        for(auto leader:child_edges(s,obs_attr(s,entries[i+1],"leaders")))if(object_type(s,leader)=="game_state.Companion"){
            const char* nation=book?"DWARVES":"THE_NORTH";const auto political=obs_attr(s,s.root,"political");
            me_nation_assign(s,obs_attr(s,political,"boxes"),nation,rr_scalar(s,WOTR_INT,3));me_nation_assign(s,obs_attr(s,political,"active"),nation,rr_scalar(s,WOTR_BOOL,1));return;
        }
    }
}
void efm_effect(Snapshot& s,const Snapshot& t,std::string_view id,std::string_view name,uint64_t choice){
    if(truth(s,field(s,choice,"_ffm_done",false))){efm_war(s,name);return;}
    const auto group=field(s,choice,"move_group");const std::string from(text(s,field(s,group,"from"))),to(text(s,field(s,group,"to")));std::vector<std::string> names;std::set<std::string> moved;
    for(auto value:child_edges(s,field(s,group,"companions")))names.push_back(std::string(text(s,value)));
    const auto old=field(s,choice,"_ffm_moved",false);if(!ctx_none(s,old))for(auto value:child_edges(s,old))moved.insert(std::string(text(s,value)));
    const auto comps=cm_select(s,from,names);require(from==to || cmp_reachable(s,t,from,cm_allowance(s,comps)).count(to),"Companion event destination no longer reachable");
    // Ordinary arrival activation also applies to event movement; the final
    // Then clause adds the separate Dwarven/North war effect.
    cm_transfer(s,t,from,to,comps);for(const auto& value:names)moved.insert(value);
    const auto options=efm_options(s,t,moved,true);if(!truth(s,options)){efm_war(s,name);return;}
    rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,name)},{"side",rr_string(s,"fp")},{"decider",rr_string(s,"fp")},{"direct",rr_scalar(s,WOTR_BOOL,1)},{"options",options}}));
}
fm_Result apply_companion_event(const Snapshot& source,const Snapshot& t,const void* random_source,int32_t side,int64_t index,std::string_view id){
    require(side==0,"Companion event belongs to FP");quiet_stable_action_gate(source,t);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained) || ctx_side(source,retained)==0,"Companion event violates retained Ring actor");
    const auto card=ev_card(t,id);require(efm_supported(t,card) && ctx_side(t,obs_attr(t,card,"side"))==0,"Unsupported companion event");require(contains_string(source,tc_hand(source,0),id),"Companion event absent from hand");require(ev_requirement(source,t,card),"Companion event requirement false");
    const auto die=ev_payment(source,t,side,index,card);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace")),deck(ev_lower(text(t,obs_attr(t,card,"deck")))),name(text(t,obs_attr(t,card,"card_name")));
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;am_Builder events;std::vector<uint64_t> rows;
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));rr_set_attr(s,ev_payment(s,t,0,index,card),"used",rr_scalar(s,WOTR_BOOL,1));
    const auto options=efm_options(s,t,{},false);if(s.nodes[options].count>=2){
        rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,name)},{"side",rr_string(s,"fp")},{"decider",rr_string(s,"fp")},{"options",options},{"die_face",rr_string(s,face)},{"deck",rr_string(s,deck)}}));
        tc_open_hand_limit_discard(s);
    }else{efm_effect(s,t,id,name,child_edges(s,options).front());er_commit(s,t,id,deck,face);}
    cau_finish(s,t,result.random.get(),-1,events,rows);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
std::unique_ptr<Snapshot> native_companion_event_choice_actions(const Snapshot& source,int32_t side){
    require(side==0,"Companion event choices belong to FP");require(efm_pending(source),"No companion event choice pending");auto out=std::make_unique<Snapshot>(source);auto& s=*out;const auto p=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t index=0;
    for(auto choice:child_edges(s,field(s,p,"options")))rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",choice}})}));
    s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
fm_Result apply_companion_event_choice(const Snapshot& source,const Snapshot& t,const void* random_source,int32_t side,int64_t index){
    require(side==0,"Companion event choice belongs to FP");require(efm_pending(source),"No companion event choice pending");const auto p=obs_attr(source,source.root,"pending_card_option");const auto options=child_edges(source,field(source,p,"options"));
    require(index>=0 && static_cast<uint64_t>(index)<options.size(),"Companion event option index out of range");const std::string id(text(source,field(source,p,"card_id"))),name(text(source,field(source,p,"card_name")));
    const auto card=ev_card(t,id);require(efm_supported(t,card) && text_is(t,obs_attr(t,card,"card_name"),name),"Companion event pending card mismatch");const bool direct=truth(source,field(source,p,"direct",false));
    std::string deck,face;if(!direct){require(contains_string(source,tc_hand(source,0),id),"Uncommitted companion event absent from hand");deck=std::string(text(source,field(source,p,"deck")));face=std::string(text(source,field(source,p,"die_face")));}
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;am_Builder events;std::vector<uint64_t> rows;
    const auto choice=child_edges(s,field(s,obs_attr(s,s.root,"pending_card_option"),"options"))[index],marker=obs_attr(s,s.root,"_battle_resolved_attacker",false);const int marker_side=ctx_none(s,marker)?-1:ctx_side(s,marker);
    rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));quiet_action_gate(s,t,false,false,false,false,false,false,false,false,false,false,true,true);ring_waiver(s,0);ring_waiver(s,1);
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));efm_effect(s,t,id,name,choice);
    if(!direct)er_commit(s,t,id,deck,face);cau_finish(s,t,result.random.get(),marker_side,events,rows);
    events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
