// Pinned persistent_effects.can_activate_nation / game_logic.activate_nation.
// The raw me_activate setter deliberately remains separate: event_effects'
// activation helper does not execute the Rohan discard callback.
uint64_t wt_first(const Snapshot& s){
    const auto view=political_view(s,"ROHAN",false);
    for(auto list:card_lists(s,view))for(auto card:child_edges(s,list)){
        const auto kind=obs_attr(s,card,"card_type",false);
        if(enum_is(s,kind,"game_state.PersistentCardType","WORMTONGUE"))return card;
    }
    return missing;
}
bool wt_can_activate(const Snapshot& s,std::string_view nation,std::string_view method,std::string_view region={}){
    if(nation!="ROHAN" || wt_first(s)==missing)return true;
    if(method=="companion_ability" || method=="fellowship_declared")return true;
    if(method!="companion_move" && method!="attack")return false;
    const auto key=sanitize_name(region);
    return key=="edoras" || key=="helm s deep";
}
bool wt_active(const Snapshot& s,std::string_view nation){
    return active_nation(s,political_view(s,nation,false),nation,false);
}
void wt_activate_checked(Snapshot& s,std::string_view nation,std::string_view method,std::string_view region={}){
    require(wt_can_activate(s,nation,method,region),"Wormtongue prevents nation activation");
    me_activate(s,nation);
    if(nation!="ROHAN")return;
    const auto card=wt_first(s);if(card==missing)return;
    const auto view=political_view(s,nation,false);
    const std::string id(card_id(s,card));
    const auto removed=remove_card(s,view,id);
    discard_card(s,view,removed==missing?card:removed);
}
// Call sites which catch a blocked activation and only call while inactive.
// An already-active Rohan leaves Wormtongue on the table, exactly as source.
bool wt_activate_if_inactive(Snapshot& s,std::string_view nation,std::string_view method,std::string_view region={}){
    if(wt_active(s,nation))return false;
    if(!wt_can_activate(s,nation,method,region))return false;
    wt_activate_checked(s,nation,method,region);return true;
}
// Source advance_defenders_political: a denied activation does not prohibit
// advancement to box two. Reuse the existing method-sensitive Threats logic.
void wt_defender_advance(Snapshot& s,std::string_view nation,std::string_view region){
    if(nation=="ROHAN"&&!wt_active(s,nation)&&!wt_can_activate(s,nation,"attack",region)){
        const auto view=political_view(s,nation,true);
        const auto previous=obs_attr(s,s.root,"_wormtongue_attack_pending",false);
        const int64_t debt=ctx_none(s,previous)?0:s.nodes[previous].integer;
        rr_set_attr(s,s.root,"_wormtongue_attack_pending",rr_scalar(s,WOTR_INT,std::max<int64_t>(debt,view.box==2?2:1)));
    }
    wt_activate_if_inactive(s,nation,"attack",region);
    const auto view=political_view(s,nation,true);
    if(!can_advance(s,view,nation,"attack"))return;
    // Keep the existing direct-write fast path when no discard can occur.
    // This helper runs on the attack's unpublished child snapshot.
    if(fp_nation(nation) && threats_card(s,view)!=missing)advance(s,nation,"attack");
    else assign_box(s,view,view.box+1);
}
// Almanac p30: reconsider a reaction blocked by Wormtongue after the entire
// attack, including retreat/advance and card-started battle continuations.
void wt_settle_attack(Snapshot& s){
    const auto debt=obs_attr(s,s.root,"_wormtongue_attack_pending",false);
    if(!truth(s,debt))return;
    for(const char* key:{"pending_combat","pending_card_option","pending_card_followup","pending_discard"})
        if(!ctx_none(s,obs_attr(s,s.root,key,false)))return;
    if(truth(s,obs_attr(s,s.root,"_overstack_regions",false)))return;
    const auto owed=typed(s,debt,WOTR_INT,"Invalid Wormtongue attack reaction").integer;
    require(owed==1||owed==2,"Invalid Wormtongue attack reaction");
    rr_set_attr(s,s.root,"_wormtongue_attack_pending",rr_scalar(s,WOTR_NONE));
    if(wt_first(s)!=missing)return;
    wt_activate_checked(s,"ROHAN","attack");
    if(owed==2){
        const auto view=political_view(s,"ROHAN",true);
        if(can_advance(s,view,"ROHAN","attack"))advance(s,"ROHAN","attack");
        else if(view.box==3){
            const auto card=threats_card(s,view);
            if(card!=missing){const std::string id(card_id(s,card));const auto removed=remove_card(s,view,id);discard_card(s,view,removed==missing?card:removed);}
        }
    }
}
