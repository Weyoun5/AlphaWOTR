// FP relief battle producer; existing combat machinery owns rounds and casualties.
bool ehuf_supported(const Snapshot& t,uint64_t card){return card!=missing&&text_is(t,obs_attr(t,card,"card_name"),"Help Unlooked For");}
bool ehuf_pending(const Snapshot& s){const auto p=obs_attr(s,s.root,"pending_card_option",false);return !ctx_none(s,p)&&text_is(s,field(s,p,"card_name",false),"Help Unlooked For");}

uint64_t ehuf_options(Snapshot& s,const Snapshot& t){
    std::vector<uint64_t> options;std::set<std::string> sieges;
    for(auto region:child_edges(s,obs_attr(s,s.root,"besieged_strongholds")))sieges.emplace(text(s,region));
    for(const auto& target:sieges){if(c9_garrison(s,t,target)!=0||!evb_units(s,hr_army(s,target),1))continue;
        for(const auto& raw:esg_neighbors(s,target)){const auto from=sanitize_name(raw);const auto army=hr_army(s,from);if(ctx_none(s,army)||at_formation(s,army,0).units.empty())continue;
            options.push_back(ev_dict(s,{{"stronghold",rr_string(s,target)},{"attack_from",rr_string(s,from)}}));}}
    return cp_container(s,WOTR_LIST,options);
}

void ehuf_effect(crr_Result& owned,const Snapshot& t,uint64_t choice,am_Builder& events,std::vector<uint64_t>& rows){
    auto& s=*owned.state;const std::string from=sanitize_name(text(s,field(s,choice,"attack_from"))),target=sanitize_name(text(s,field(s,choice,"stronghold")));
    const auto army=hr_army(s,from);if(ctx_none(s,army)||at_formation(s,army,0).units.empty())return;
    const auto defender=hr_army(s,target);int64_t reduction=0;if(!ctx_none(s,defender))for(auto unit:child_edges(s,obs_attr(s,defender,"units")))reduction+=ctx_unit_side(s,unit)==0;
    const auto pc=ev_dict(s,{{"attacker_region",rr_string(s,from)},{"defender_region",rr_string(s,target)},{"is_siege",rr_scalar(s,WOTR_BOOL,0)},{"stronghold_region",rr_scalar(s,WOTR_NONE)},{"phase",rr_string(s,"combat_setup")},{"deciding_side",mu_enum(s,"game_state.Side","FREE_PEOPLES")},{"attacker_side",mu_enum(s,"game_state.Side","FREE_PEOPLES")},{"defender_side",mu_enum(s,"game_state.Side","SHADOW")},{"rounds_fought",rr_scalar(s,WOTR_INT,0)},{"def_dice_reduction",rr_scalar(s,WOTR_INT,reduction)}});
    rr_set_attr(s,s.root,"pending_combat",pc);cbb_run(owned,t,0,events,rows);
}
fm_Result ehuf_finish(crr_Result owned,const Snapshot& t,int marker,am_Builder& events,std::vector<uint64_t>& rows){
    cau_finish(*owned.state,t,owned.random.get(),marker,events,rows);fm_Result result;result.state=ac_compact(*owned.state);result.random=std::move(owned.random);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
fm_Result apply_help_unlooked_event(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index,std::string_view id){
    require(side==0,"Help Unlooked For belongs to FP");quiet_stable_action_gate(source,t);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==side,"Help Unlooked For violates retained Ring actor");
    const auto card=ev_card(t,id);require(ehuf_supported(t,card)&&contains_string(source,tc_hand(source,side),id),"Help Unlooked For card absent");require(ev_requirement(source,t,card),"Help Unlooked For requirement false");
    const auto die=ev_payment(source,t,side,index,card);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace")),deck(ev_lower(text(t,obs_attr(t,card,"deck"))));
    crr_Result owned;owned.state=action_clone(source);owned.random.reset(wotr_random_clone(rng));if(!owned.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*owned.state;am_Builder events;std::vector<uint64_t> rows;
    rr_set_attr(s,s.root,"_casualty_operations",cp_container(s,WOTR_LIST,{}));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));ev_consume_payment(s,t,side,index,card,false);
    const auto options=ehuf_options(s,t);const auto choices=child_edges(s,options);
    if(choices.size()>1){rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,"Help Unlooked For")},{"side",rr_string(s,"fp")},{"decider",rr_string(s,"fp")},{"options",options},{"die_face",rr_string(s,face)},{"deck",rr_string(s,deck)}}));tc_open_hand_limit_discard(s);}
    else{if(!choices.empty())ehuf_effect(owned,t,choices.front(),events,rows);er_commit(*owned.state,t,id,deck,face);}
    return ehuf_finish(std::move(owned),t,-1,events,rows);
}
uint64_t ehuf_validate(const Snapshot& s,const Snapshot& t,int side){
    require(side==0&&ehuf_pending(s),"No FP Help Unlooked For choice pending");const auto p=obs_attr(s,s.root,"pending_card_option");
    require(ehuf_supported(t,ev_card(t,text(s,field(s,p,"card_id"))))&&text_is(s,field(s,p,"side",false),"fp")&&text_is(s,field(s,p,"decider",false),"fp")&&!truth(s,field(s,p,"direct",false)),"Help Unlooked For pending identity or owner invalid");
    /* package R: hand membership is not a menu precondition (frozen card_option menu, ai.py 7764-7770) */
    const auto face=text(s,field(s,p,"die_face"));require((face=="EVENT"||face=="ARMY"||face=="MUSTER_ARMY"||face=="WILL_OF_WEST")&&text_is(s,field(s,p,"deck"),"strategy"),"Help Unlooked For stored funding invalid");return p;
}
std::unique_ptr<Snapshot> native_help_unlooked_choice_actions(const Snapshot& source,const Snapshot& t,int side){
    ehuf_validate(source,t,side);auto out=std::make_unique<Snapshot>(source);auto& s=*out;const auto p=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t index=0;
    for(auto choice:child_edges(s,field(s,p,"options")))rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",choice}})}));s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
fm_Result apply_help_unlooked_choice(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index){
    const auto pending=ehuf_validate(source,t,side);const auto options=child_edges(source,field(source,pending,"options"));require(index>=0&&static_cast<uint64_t>(index)<options.size(),"Help Unlooked For choice index outside menu");
    const std::string id(text(source,field(source,pending,"card_id"))),face(text(source,field(source,pending,"die_face"))),deck(text(source,field(source,pending,"deck")));const int marker=ctx_side(source,obs_attr(source,source.root,"_battle_resolved_attacker",false));
    crr_Result owned;owned.state=action_clone(source);owned.random.reset(wotr_random_clone(rng));if(!owned.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*owned.state;am_Builder events;std::vector<uint64_t> rows;
    const auto choice=child_edges(s,field(s,obs_attr(s,s.root,"pending_card_option"),"options"))[index];rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));quiet_stable_action_gate(s,t);ring_waiver(s,0);ring_waiver(s,1);
    rr_set_attr(s,s.root,"_casualty_operations",cp_container(s,WOTR_LIST,{}));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    // Source checks current requirements again, but executes the stored target
    // without rebuilding its options. False requirements retain the card.
    if(ev_requirement(s,t,ev_card(t,id))){ehuf_effect(owned,t,choice,events,rows);er_commit(*owned.state,t,id,deck,face);}
    return ehuf_finish(std::move(owned),t,marker,events,rows);
}
