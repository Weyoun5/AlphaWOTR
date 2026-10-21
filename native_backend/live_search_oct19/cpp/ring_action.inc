// Complete elven_ring action envelope in the same restricted quiet-state scope.
// Physical conversion remains the independently verified ring_rules helper.
std::unique_ptr<Snapshot> apply_elven_ring_action(const Snapshot& s,const Snapshot& tables,
                                                int32_t side,int64_t index,std::string_view face){
    require(side==0 || side==1,"Elven Ring action side must be FP or SH");
    quiet_stable_action_gate(s,tables);
    const auto retained=ctx_side(s,obs_attr(s,s.root,"_ring_free_side",false));
    require(retained<0 || retained==side,"Elven Ring action belongs to the retained actor");
    ring_waiver(s,0);ring_waiver(s,1);
    const auto pool=pa_pool(s,side);const auto& list=s.nodes[pool];
    require(index>=0 && static_cast<uint64_t>(index)<list.count,"Elven Ring action index is out of range");
    const auto die=s.edges[list.offset+static_cast<uint64_t>(index)];
    require(object_type(s,die)=="game_state.ActionDie","Elven Ring action requires ActionDie");
    require(enum_is(s,obs_attr(s,die,"side"),"game_state.Side",side==0?"FREE_PEOPLES":"SHADOW"),"Elven Ring action die has wrong owner");
    auto child=action_clone(s);auto& out=*child;
    rr_set_attr(out,out.root,"_casualty_operations",rr_scalar(out,WOTR_LIST));
    use_ring(out,side,index,face);
    auto marker=rr_scalar(out,WOTR_NONE);
    if(pa_has_unused(out,side)){
        const auto edge=enum_value_edge(out,obs_attr(out,out.root,"dice"),"game_state.Side",side==0?"FREE_PEOPLES":"SHADOW");
        require(edge!=missing,"Elven Ring action lost the side's dice");
        marker=out.edges[edge-1];
    }
    rr_set_attr(out,out.root,"_ring_free_side",marker);
    pa_common_upkeep(out,tables);
    // Only a Shadow Eye conversion spends a die. Other conversions preserve
    // a prior resolved-battle marker, just as the common Python wrapper does.
    if(side==1 && face=="EYE")rr_set_attr(out,out.root,"_battle_resolved_attacker",rr_scalar(out,WOTR_NONE));
    return child;
}
