// Frozen ai.get_legal_actions eye-allocation branch and complete supported
// ai.apply_action envelope. Eye count remains an explicitly chosen action.
std::pair<int64_t,int64_t> ea_range(const Snapshot& s){
    attributes(s,s.root,"game_state.GameState");
    require(truth(s,obs_attr(s,s.root,"_eye_allocation_phase",false)),"Eye allocation phase is inactive");
    // These higher-priority menus must not be replaced by allocation counts.
    for(auto name:{"pending_combat","pending_guide_change","pending_hunt","pending_discard"})
        require(ctx_none(s,obs_attr(s,s.root,name,false)),"Eye allocation rejects a higher-priority pending decision");
    require(!truth(s,obs_attr(s,s.root,"_declare_phase",false)),"Declaration precedes eye allocation");
    require(!truth(s,obs_attr(s,s.root,"_reveal_place_due",false)),"Reveal placement precedes eye allocation");
    const auto fellowship=obs_attr(s,s.root,"fellowship");
    const auto companions=obs_attr(s,fellowship,"companions");
    const auto total=typed(s,companions,WOTR_LIST,"Eye allocation requires companion list").count;
    const auto dice=obs_attr(s,s.root,"dice");
    const auto edge=enum_value_edge(s,dice,"game_state.Side","SHADOW");
    uint64_t unused=0;
    if(edge!=missing){
        const auto pool=s.edges[edge];typed(s,pool,WOTR_LIST,"Eye allocation requires dice list");
        for(auto die:child_edges(s,pool)){
            require(object_type(s,die)=="game_state.ActionDie","Eye allocation requires ActionDie");
            if(!truth(s,obs_attr(s,die,"used")))++unused;
        }
    }
    const auto maximum=std::min(std::max<uint64_t>(total,1),unused);
    require(maximum<=uint64_t(std::numeric_limits<int64_t>::max()),"Eye allocation count exceeds native integer range");
    const int64_t minimum=truth(s,obs_attr(s,s.root,"_fp_used_hunt_box_last_turn",false))?std::min<uint64_t>(1,maximum):0;
    return {minimum,static_cast<int64_t>(maximum)};
}
std::unique_ptr<Snapshot> eye_allocation_actions(const Snapshot& s,int32_t side){
    require(side==0 || side==1,"Eye menu side must be FP or SH");
    // Original raw menu intentionally returns this same list for either side;
    // the episode/search router owns the fact that Shadow takes the decision.
    const auto bounds=ea_range(s);am_Builder b;std::vector<uint64_t> rows;
    for(int64_t count=bounds.first;count<=bounds.second;++count)
        rows.push_back(b.row(-1,"eye_allocate",{b.str("count"),b.integer(count)}));
    b.out.root=b.container(WOTR_LIST,rows);return std::make_unique<Snapshot>(std::move(b.out));
}
uint64_t ea_unused_rolled(const Snapshot& s){
    const auto dice=obs_attr(s,s.root,"dice");uint64_t count=0;
    for(auto side:{"FREE_PEOPLES","SHADOW"}){
        const auto edge=enum_value_edge(s,dice,"game_state.Side",side);if(edge==missing)continue;
        for(auto die:child_edges(s,s.edges[edge]))
            if(!truth(s,obs_attr(s,die,"used")) && s.nodes[obs_attr(s,die,"current")].tag!=WOTR_NONE)++count;
    }
    return count;
}
void ea_allocate_inner(Snapshot& changed,int64_t count){
    if(count>0){
        const auto dice=obs_attr(changed,changed.root,"dice");
        const auto edge=enum_value_edge(changed,dice,"game_state.Side","SHADOW");
        require(edge!=missing,"Shadow dice are missing for allocation");const auto pool=changed.edges[edge];
        std::vector<uint64_t> selected;selected.reserve(static_cast<size_t>(count));
        for(auto die:child_edges(changed,pool))
            if(!truth(changed,obs_attr(changed,die,"used")) && selected.size()<static_cast<size_t>(count))selected.push_back(die);
        require(selected.size()==static_cast<size_t>(count),"Insufficient unused Shadow dice");
        const auto box=obs_attr(changed,changed.root,"hunt_box");typed(changed,box,WOTR_LIST,"Hunt Box must be a list");
        const auto eye=rr_enum(changed,"EYE");
        for(auto die:selected){
            require(!truth(changed,obs_attr(changed,die,"used")),"Selected allocation die is already used");
            rr_set_attr(changed,die,"current",eye);rr_set_attr(changed,die,"used",rr_scalar(changed,WOTR_BOOL,1));
            auto entries=child_edges(changed,box);entries.push_back(die);set_edges(changed,box,entries);
        }
        const auto current=obs_attr(changed,changed.root,"eyes_allocated");const auto number=changed.nodes[current];uint64_t updated;
        if(number.tag==WOTR_FLOAT)updated=rr_scalar(changed,WOTR_FLOAT,0,number.real+double(count));
        else{
            require(number.tag==WOTR_INT || number.tag==WOTR_BOOL,"Allocated-eye count must be numeric");
            require(number.integer<=std::numeric_limits<int64_t>::max()-count,"Allocated-eye count exceeds int64 storage");
            updated=rr_scalar(changed,WOTR_INT,number.integer+count);
        }
        rr_set_attr(changed,changed.root,"eyes_allocated",updated);
    }
    // Even the zero decision completes the phase. It does not add a missing
    // eyes_allocated field or alter its scalar type/value.
    rr_set_attr(changed,changed.root,"_eye_allocation_phase",rr_scalar(changed,WOTR_BOOL,0));
}
std::unique_ptr<Snapshot> apply_eye_allocation(const Snapshot& s,const Snapshot& tables,int64_t count){
    const auto bounds=ea_range(s);
    require(count>=bounds.first && count<=bounds.second,"Eye count is outside the legal allocation menu");
    // Package M: the frozen source carries no siege clause on eye allocation (ai.py:7691-7697 menu, :12248-12256
    // executor, game_logic.allocate_eyes :250-275); a stable siege is admitted with the same structural validation
    // the other turn-start executors (hand-limit discard, declaration, guide) already pass.
    quiet_action_gate(s,tables,false,false,true,false,false,false,false,false,false,false,false,true);
    auto child=action_clone(s);auto& changed=*child;const auto before_unused=ea_unused_rolled(changed);
    rr_set_attr(changed,changed.root,"_casualty_operations",rr_scalar(changed,WOTR_LIST));
    rr_set_attr(changed,changed.root,"_ring_free_side",rr_scalar(changed,WOTR_NONE));
    ea_allocate_inner(changed,count);
    if(ea_unused_rolled(changed)<before_unused)
        rr_set_attr(changed,changed.root,"_battle_resolved_attacker",rr_scalar(changed,WOTR_NONE));
    return child;
}
