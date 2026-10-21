// Complete ordinary advance_political action, including admitted stable-point upkeep.
bool tc_open_hand_limit_discard(Snapshot& s);
void su_cleanup(Snapshot& s,const Snapshot& t);
void pa_common_upkeep(Snapshot& s,const Snapshot& tables){
    // Match _execute_action: hand-limit obligation precedes siege/control and
    // overflow reconciliation. Remaining pending/RNG phases are still gated.
    tc_open_hand_limit_discard(s);
    su_cleanup(s,tables);
}
uint64_t pa_pool(const Snapshot& s,int32_t side){
    require(side==0 || side==1,"political action side must be FP or SH");
    const auto dice=obs_attr(s,s.root,"dice");
    const auto edge=enum_value_edge(s,dice,"game_state.Side",side==0?"FREE_PEOPLES":"SHADOW");
    require(edge!=missing,"political action side has no dice");
    const auto pool=s.edges[edge];typed(s,pool,WOTR_LIST,"political action dice must be a list");return pool;
}
uint64_t pa_payment_die(const Snapshot& s,int32_t side,int64_t index){
    const auto pool=pa_pool(s,side);const auto& list=s.nodes[pool];
    require(index>=0 && static_cast<uint64_t>(index)<list.count,"political payment die index is out of range");
    const auto die=s.edges[list.offset+static_cast<uint64_t>(index)];
    require(object_type(s,die)=="game_state.ActionDie","political payment requires ActionDie");
    require(enum_is(s,obs_attr(s,die,"side"),"game_state.Side",side==0?"FREE_PEOPLES":"SHADOW"),"political payment die has wrong owner");
    require(!truth(s,obs_attr(s,die,"used")),"political payment die is already used");
    const auto face=obs_attr(s,die,"current");
    require(enum_is(s,face,"game_state.ActionDieFace","MUSTER") ||
            enum_is(s,face,"game_state.ActionDieFace","MUSTER_ARMY") ||
            enum_is(s,face,"game_state.ActionDieFace","WILL_OF_WEST"),"political payment needs an ordinary muster-capable face");
    return die;
}
std::unique_ptr<Snapshot> apply_political_action(const Snapshot& s,const Snapshot& tables,
                                               int32_t side,int64_t index,std::string_view nation){
    require(side==0 || side==1,"political action side must be FP or SH");
    require((fp_nation(nation)?0:1)==side,"political nation belongs to the other side");
    quiet_stable_action_gate(s,tables,false,true);
    const auto retained=obs_attr(s,s.root,"_ring_free_side",false);
    require(ctx_none(s,retained) || ctx_side(s,retained)==side,"political action belongs to the retained ring actor");
    ring_waiver(s,0);ring_waiver(s,1);
    pa_payment_die(s,side,index);
    const auto view=political_view(s,nation,true);
    require(view.box<3 && can_advance(s,view,nation,"muster_die"),"political action is not a legal advancement");
    auto child=action_clone(s);auto& out=*child;
    // Ordinary action cloning preserves every gate predicate (it only resets
    // the already-false hand flags and materializes inactive defaults).
    // The unpublished child owns its buffers, so it needs no second transaction.
    rr_set_attr(out,out.root,"_casualty_operations",rr_scalar(out,WOTR_LIST));
    rr_set_attr(out,out.root,"_ring_free_side",rr_scalar(out,WOTR_NONE));
    const auto die=pa_payment_die(out,side,index);
    rr_set_attr(out,die,"used",rr_scalar(out,WOTR_BOOL,1));
    const auto child_view=political_view(out,nation,true);
    assign_box(out,child_view,child_view.box+1); // muster advancement never discards admitted T&P
    pa_common_upkeep(out,tables);
    rr_set_attr(out,out.root,"_battle_resolved_attacker",rr_scalar(out,WOTR_NONE));
    return child;
}
bool pa_has_unused(const Snapshot& s,int32_t side){
    for(auto die:child_edges(s,pa_pool(s,side)))
        if(!truth(s,obs_attr(s,die,"used")) && s.nodes[obs_attr(s,die,"current")].tag!=WOTR_NONE)return true;
    return false;
}
uint64_t am_die(const Snapshot& s,int side,int64_t index,bool character,bool muster=false);
int dc_owner(const Snapshot& s);
int next_quiet_actor(const Snapshot& s,const Snapshot& tables,int32_t just_acted){
    require(just_acted==0 || just_acted==1,"quiet routing side must be FP or SH");
    quiet_action_gate(s,tables,true,true,false,true,true);
    if(!ctx_none(s,obs_attr(s,s.root,"pending_discard",false)))return dc_owner(s);
    const auto retained=ctx_side(s,obs_attr(s,s.root,"_ring_free_side",false));
    if(retained>=0 && pa_has_unused(s,retained))return retained;
    const auto army_pending=obs_attr(s,s.root,"pending_army_move_die");
    if(!ctx_none(s,army_pending)){
        const auto index=typed(s,army_pending,WOTR_INT,"army routing requires integer die index").integer;
        const auto owner=ctx_side(s,obs_attr(s,s.root,"pending_army_move_side"));
        require(owner>=0,"army routing requires owner");am_die(s,owner,index,false,true);
        return just_acted;
    }
    const auto pending=obs_attr(s,s.root,"pending_muster_die");
    if(!ctx_none(s,pending)){
        const auto index=typed(s,pending,WOTR_INT,"muster routing requires integer die index").integer;
        const auto owner=ctx_side(s,obs_attr(s,s.root,"pending_muster_side"));
        require(owner>=0,"muster routing requires owner");pa_payment_die(s,owner,index);
        return just_acted;
    }
    const auto attacker=ctx_side(s,obs_attr(s,s.root,"_battle_resolved_attacker",false));
    const auto effective=attacker<0?just_acted:attacker;
    for(auto side:{1-effective,effective})if(pa_has_unused(s,side))return side;
    return 2;
}
