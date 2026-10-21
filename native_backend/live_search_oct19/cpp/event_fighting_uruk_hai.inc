// Funded Fighting Uruk-hai event producer. The existing combat machinery
// owns mandatory rounds, card timing, casualties and normal siege continuation.
bool euh_supported(const Snapshot& t,uint64_t card){return card!=missing&&text_is(t,obs_attr(t,card,"card_name"),"The Fighting Uruk-hai");}
bool euh_pending(const Snapshot& s){const auto p=obs_attr(s,s.root,"pending_card_option",false);return !ctx_none(s,p)&&text_is(s,field(s,p,"card_name",false),"The Fighting Uruk-hai");}
uint64_t euh_options(Snapshot& s,const Snapshot& t){
    std::vector<uint64_t> options;std::set<std::string> seen;const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));const auto sieges=obs_attr(s,s.root,"besieged_strongholds");
    for(size_t i=0;i<armies.size();i+=2){
        const auto key=sanitize_name(text(s,armies[i]));if(!contains_string(s,sieges,key)||c9_garrison(s,t,key)!=0)continue;
        bool isengard=false;for(auto unit:child_edges(s,obs_attr(s,armies[i+1],"units")))isengard|=enum_is(s,obs_attr(s,unit,"nation"),"game_state.Nation","ISENGARD");
        if(isengard&&seen.insert(key).second)options.push_back(ev_dict(s,{{"attack_from",rr_string(s,key)},{"attack_target",rr_string(s,key)}}));
    }return cp_container(s,WOTR_LIST,options);
}
void euh_effect(crr_Result& owned,const Snapshot& t,uint64_t choice,am_Builder& events,std::vector<uint64_t>& rows){
    auto& s=*owned.state;const std::string from=sanitize_name(text(s,field(s,choice,"attack_from"))),target=sanitize_name(text(s,field(s,choice,"attack_target")));
    // A valid public choice protected by Power is a paid zero-effect play:
    // the source event wrapper converts the handler's refund in this case.
    if(ept_blocks(s,target))return;
    const auto defender=mu_army(s,target);require(defender!=missing,"Uruk-hai target army absent");std::set<std::string> nations;
    for(auto unit:child_edges(s,obs_attr(s,defender,"units")))if(ctx_unit_side(s,unit)==0)nations.emplace(ctx_nation(s,obs_attr(s,unit,"nation")));
    political_table_gate(s);for(const auto& nation:nations)wt_defender_advance(s,nation,target);
    const int defender_side=c9_garrison(s,t,target);require(defender_side>=0,"Uruk-hai target has no garrison owner");
    const auto pc=ev_dict(s,{{"attacker_region",rr_string(s,from)},{"defender_region",rr_string(s,target)},{"is_siege",rr_scalar(s,WOTR_BOOL,1)},{"stronghold_region",rr_string(s,target)},{"phase",rr_string(s,"combat_setup")},{"deciding_side",mu_enum(s,"game_state.Side","SHADOW")},{"attacker_side",mu_enum(s,"game_state.Side","SHADOW")},{"defender_side",mu_enum(s,"game_state.Side",defender_side==0?"FREE_PEOPLES":"SHADOW")},{"rounds_fought",rr_scalar(s,WOTR_INT,0)},{"mandatory_rounds",rr_scalar(s,WOTR_INT,3)},{"no_def_card_r1_unless_companion",rr_scalar(s,WOTR_BOOL,1)}});
    rr_set_attr(s,s.root,"pending_combat",pc);cbb_run(owned,t,1,events,rows);
}
fm_Result euh_finish(crr_Result owned,const Snapshot& t,int marker,am_Builder& events,std::vector<uint64_t>& rows){
    cau_finish(*owned.state,t,owned.random.get(),marker,events,rows);fm_Result result;result.state=ac_compact(*owned.state);result.random=std::move(owned.random);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
fm_Result apply_uruk_event(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index,std::string_view id,bool mouth_muster=false){
    require(side==1,"Fighting Uruk-hai belongs to Shadow");quiet_stable_action_gate(source,t);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==side,"Uruk-hai violates retained Ring actor");
    const auto card=ev_card(t,id);require(euh_supported(t,card)&&contains_string(source,tc_hand(source,side),id),"Uruk-hai card absent");require(ev_requirement(source,t,card),"Uruk-hai requirement false");
    const auto die=ev_payment(source,t,side,index,card,mouth_muster);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace")),deck(ev_lower(text(t,obs_attr(t,card,"deck"))));
    crr_Result owned;owned.state=action_clone(source);owned.random.reset(wotr_random_clone(rng));if(!owned.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*owned.state;am_Builder events;std::vector<uint64_t> rows;
    rr_set_attr(s,s.root,"_casualty_operations",cp_container(s,WOTR_LIST,{}));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));ev_consume_payment(s,t,side,index,card,mouth_muster);
    rr_set_attr(s,s.root,"_battle_resolved_attacker",rr_scalar(s,WOTR_NONE)); // Package W (P8): a real die-spending action landed - the frozen tail clears the marker even when the new battle stays pending (ai.py:9905-9915; army_attack_action.inc:25 pattern)
    const auto options=euh_options(s,t);const auto choices=child_edges(s,options);
    if(choices.size()>1){rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,"The Fighting Uruk-hai")},{"side",rr_string(s,"sh")},{"decider",rr_string(s,"sh")},{"options",options},{"die_face",rr_string(s,face)},{"deck",rr_string(s,deck)}}));tc_open_hand_limit_discard(s);}
    else if(choices.size()==1){euh_effect(owned,t,choices.front(),events,rows);ed_commit(*owned.state,side,id,deck,face);}
    else ed_commit(*owned.state,side,id,deck,face); // Package W (P2): no qualifying target = the frozen zero-effect play (ruling 1.230)
    return euh_finish(std::move(owned),t,-1,events,rows);
}
uint64_t euh_validate(const Snapshot& s,const Snapshot& t,int side){
    require(side==1&&euh_pending(s),"No Shadow Uruk-hai choice pending");const auto p=obs_attr(s,s.root,"pending_card_option");
    require(euh_supported(t,ev_card(t,text(s,field(s,p,"card_id"))))&&text_is(s,field(s,p,"side",false),"sh")&&text_is(s,field(s,p,"decider",false),"sh")&&!truth(s,field(s,p,"direct",false)),"Uruk-hai pending identity or owner invalid");
    /* package R: hand membership is not a menu precondition (frozen card_option menu, ai.py 7764-7770) */
    const auto face=text(s,field(s,p,"die_face"));require((face=="EVENT"||face=="ARMY"||face=="MUSTER_ARMY"||face=="MUSTER")&&text_is(s,field(s,p,"deck"),"strategy"),"Uruk-hai stored funding invalid");return p;
}
std::unique_ptr<Snapshot> native_uruk_choice_actions(const Snapshot& source,const Snapshot& t,int side){
    euh_validate(source,t,side);auto out=std::make_unique<Snapshot>(source);auto& s=*out;const auto p=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t index=0;
    for(auto choice:child_edges(s,field(s,p,"options")))rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",choice}})}));s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
fm_Result apply_uruk_choice(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index){
    const auto pending=euh_validate(source,t,side);const auto options=child_edges(source,field(source,pending,"options"));require(index>=0&&static_cast<uint64_t>(index)<options.size(),"Uruk-hai choice index outside menu");
    const std::string id(text(source,field(source,pending,"card_id"))),face(text(source,field(source,pending,"die_face"))),deck(text(source,field(source,pending,"deck")));const int marker=ctx_side(source,obs_attr(source,source.root,"_battle_resolved_attacker",false));
    crr_Result owned;owned.state=action_clone(source);owned.random.reset(wotr_random_clone(rng));if(!owned.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*owned.state;am_Builder events;std::vector<uint64_t> rows;
    const auto choice=child_edges(s,field(s,obs_attr(s,s.root,"pending_card_option"),"options"))[index];rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));quiet_stable_action_gate(s,t);ring_waiver(s,0);ring_waiver(s,1);
    rr_set_attr(s,s.root,"_casualty_operations",cp_container(s,WOTR_LIST,{}));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    // Source checks current requirements again, but executes the stored target
    // without rebuilding its options. False requirements retain the card.
    if(ev_requirement(s,t,ev_card(t,id))){euh_effect(owned,t,choice,events,rows);ed_commit(*owned.state,side,id,deck,face);}
    return euh_finish(std::move(owned),t,marker,events,rows);
}
