// Native equivalent of the frozen pending-owner cascade and _next_acting_side.
// Read-only routing does not imply the corresponding transition is implemented.
int ar_side(const Snapshot& s,uint64_t value){
    if(ctx_none(s,value))return -1;
    require(enum_is(s,value,"game_state.Side","FREE_PEOPLES") || enum_is(s,value,"game_state.Side","SHADOW"),"routing owner must be Side or None");
    return ctx_side(s,value);
}
int ar_option_owner(const Snapshot& s){
    for(const char* key:{"pending_card_option","pending_card_followup"}){
        const auto p=obs_attr(s,s.root,key,false);if(!truth(s,p))continue;
        const auto tag=field(s,p,"side",false);if(!truth(s,tag))continue;
        std::string name(text(s,tag));for(auto& c:name)if(c>='A' && c<='Z')c=static_cast<char>(c-'A'+'a');
        if(name=="fp")return 0;if(name=="sh")return 1;
    }return -1;
}
struct ar_Pending {std::string kind;int side=-1;};
ar_Pending native_pending_actor(const Snapshot& s,int fallback){
    require(fallback>=-1 && fallback<=1,"routing fallback outside Side domain");
    const auto combat=obs_attr(s,s.root,"pending_combat");
    if(!ctx_none(s,combat)){
        const auto phase=field(s,combat,"phase",false),owner=field(s,combat,"deciding_side",false);
        return {std::string("combat:")+(phase==missing?"?":std::string(text(s,phase))),owner==missing?fallback:ar_side(s,owner)};
    }
    if(!ctx_none(s,obs_attr(s,s.root,"pending_guide_change")))return {"guide_change",0};
    if(!ctx_none(s,obs_attr(s,s.root,"pending_hunt")))return {"hunt",0};
    if(truth(s,obs_attr(s,s.root,"_reveal_place_due",false)) &&
       ctx_none(s,obs_attr(s,s.root,"pending_card_option",false)) && !truth(s,obs_attr(s,s.root,"_pending_hunt_damage",false)))
        return {"reveal_place",0};
    const auto discard=obs_attr(s,s.root,"pending_discard",false);
    if(!ctx_none(s,discard)){
        const int owner=ar_side(s,field(s,discard,"side"));
        if(ar_option_owner(s)!=owner)return {"discard",owner};
    }
    if(truth(s,obs_attr(s,s.root,"_declare_phase",false)))return {"declare",0};
    if(truth(s,obs_attr(s,s.root,"_eye_allocation_phase",false)))return {"eye",1};
    const auto follow=obs_attr(s,s.root,"pending_card_followup",false);
    if(!ctx_none(s,follow))return {"card_followup",text_is(s,field(s,follow,"side"),"fp")?0:1};
    const auto option=obs_attr(s,s.root,"pending_card_option",false);
    if(!ctx_none(s,option)){
        const auto owner=field(s,option,"side"),decider=field(s,option,"decider",false);
        return {"card_option",text_is(s,decider==missing?owner:decider,"fp")?0:1};
    }
    return {};
}
bool ar_has_unused(const Snapshot& s,int side){
    const auto dice=obs_attr(s,s.root,"dice");const auto pool=ctx_side_value(s,dice,side);
    if(ctx_none(s,pool))return false;
    for(auto die:child_edges(s,pool))
        if(!truth(s,obs_attr(s,die,"used")) && !ctx_none(s,obs_attr(s,die,"current")))return true;
    return false;
}
int native_next_actor(const Snapshot& s,int just_acted){
    require(just_acted==0 || just_acted==1,"next actor requires Side");
    const auto pending=native_pending_actor(s,just_acted);if(pending.side>=0)return pending.side;
    const auto retained=ar_side(s,obs_attr(s,s.root,"_ring_free_side",false));
    if(retained>=0 && ar_has_unused(s,retained))return retained;
    if(!ctx_none(s,obs_attr(s,s.root,"pending_army_move_die")) || !ctx_none(s,obs_attr(s,s.root,"pending_muster_die")))return just_acted;
    const auto attacker=ar_side(s,obs_attr(s,s.root,"_battle_resolved_attacker",false));
    const int effective=attacker<0?just_acted:attacker;
    if(ar_has_unused(s,1-effective))return 1-effective;
    if(ar_has_unused(s,effective))return effective;
    return -1;
}
