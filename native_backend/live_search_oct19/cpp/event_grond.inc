// Funded Grond event producer. The existing combat machinery
// owns mandatory rounds, card timing, casualties and normal siege continuation.
bool egr_supported(const Snapshot& t,uint64_t card){return card!=missing&&text_is(t,obs_attr(t,card,"card_name"),"Grond Hammer of the Underworld");}
bool egr_pending(const Snapshot& s){const auto p=obs_attr(s,s.root,"pending_card_option",false);return !ctx_none(s,p)&&text_is(s,field(s,p,"card_name",false),"Grond Hammer of the Underworld");}
uint64_t egr_options(Snapshot& s,const Snapshot& t){
    std::vector<uint64_t> options;const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));
    for(size_t i=0;i<armies.size();i+=2){
        const auto key=sanitize_name(text(s,armies[i]));
        if(evb_grond_target(s,t,key,armies[i+1]))options.push_back(ev_dict(s,{{"attack_from",rr_string(s,key)},{"attack_target",rr_string(s,key)}}));
    }return cp_container(s,WOTR_LIST,options);
}
fm_Result apply_grond_event(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index,std::string_view id,bool mouth_muster=false){
    require(side==1,"Grond belongs to Shadow");quiet_stable_action_gate(source,t);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==side,"Grond violates retained Ring actor");
    const auto card=ev_card(t,id);require(egr_supported(t,card)&&contains_string(source,tc_hand(source,side),id),"Grond card absent");require(ev_requirement(source,t,card),"Grond requirement false");
    const auto die=ev_payment(source,t,side,index,card,mouth_muster);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace")),deck(ev_lower(text(t,obs_attr(t,card,"deck"))));
    crr_Result owned;owned.state=action_clone(source);owned.random.reset(wotr_random_clone(rng));if(!owned.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*owned.state;am_Builder events;std::vector<uint64_t> rows;
    rr_set_attr(s,s.root,"_casualty_operations",cp_container(s,WOTR_LIST,{}));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));ev_consume_payment(s,t,side,index,card,mouth_muster);
    rr_set_attr(s,s.root,"_battle_resolved_attacker",rr_scalar(s,WOTR_NONE)); // Package W2 (P11): a real die-spending action landed - the frozen tail clears the marker even when the new siege battle stays pending (ai.py:9905-9915; W's P8 pattern, army_attack_action.inc:25)
    const auto options=egr_options(s,t);const auto choices=child_edges(s,options);
    if(choices.size()>1){rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,"Grond Hammer of the Underworld")},{"side",rr_string(s,"sh")},{"decider",rr_string(s,"sh")},{"options",options},{"die_face",rr_string(s,face)},{"deck",rr_string(s,deck)}}));tc_open_hand_limit_discard(s);}
    else{if(!choices.empty())euh_effect(owned,t,choices.front(),events,rows);ed_commit(*owned.state,side,id,deck,face);}
    return euh_finish(std::move(owned),t,-1,events,rows);
}
uint64_t egr_validate(const Snapshot& s,const Snapshot& t,int side){
    require(side==1&&egr_pending(s),"No Shadow Grond choice pending");const auto p=obs_attr(s,s.root,"pending_card_option");
    require(egr_supported(t,ev_card(t,text(s,field(s,p,"card_id"))))&&text_is(s,field(s,p,"side",false),"sh")&&text_is(s,field(s,p,"decider",false),"sh")&&!truth(s,field(s,p,"direct",false)),"Grond pending identity or owner invalid");
    /* package R: hand membership is not a menu precondition (frozen card_option menu, ai.py 7764-7770) */
    const auto face=text(s,field(s,p,"die_face"));require((face=="EVENT"||face=="CHARACTER")&&text_is(s,field(s,p,"deck"),"character"),"Grond stored funding invalid");return p;
}
std::unique_ptr<Snapshot> native_grond_choice_actions(const Snapshot& source,const Snapshot& t,int side){
    egr_validate(source,t,side);auto out=std::make_unique<Snapshot>(source);auto& s=*out;const auto p=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t index=0;
    for(auto choice:child_edges(s,field(s,p,"options")))rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",choice}})}));s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
fm_Result apply_grond_choice(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index){
    const auto pending=egr_validate(source,t,side);const auto options=child_edges(source,field(source,pending,"options"));require(index>=0&&static_cast<uint64_t>(index)<options.size(),"Grond choice index outside menu");
    const std::string id(text(source,field(source,pending,"card_id"))),face(text(source,field(source,pending,"die_face"))),deck(text(source,field(source,pending,"deck")));const int marker=ctx_side(source,obs_attr(source,source.root,"_battle_resolved_attacker",false));
    crr_Result owned;owned.state=action_clone(source);owned.random.reset(wotr_random_clone(rng));if(!owned.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*owned.state;am_Builder events;std::vector<uint64_t> rows;
    const auto choice=child_edges(s,field(s,obs_attr(s,s.root,"pending_card_option"),"options"))[index];rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));quiet_stable_action_gate(s,t);ring_waiver(s,0);ring_waiver(s,1);
    rr_set_attr(s,s.root,"_casualty_operations",cp_container(s,WOTR_LIST,{}));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    // Source checks current requirements again, but executes the stored target
    // without rebuilding its options. False requirements retain the card.
    if(ev_requirement(s,t,ev_card(t,id))){euh_effect(owned,t,choice,events,rows);ed_commit(*owned.state,side,id,deck,face);}
    return euh_finish(std::move(owned),t,marker,events,rows);
}
