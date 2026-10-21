// _get_fellowship_actions raw per-die family and ordinary funded hide action.
std::unique_ptr<Snapshot> native_fellowship_actions(const Snapshot& s,int64_t index){
    const auto fs=obs_attr(s,s.root,"fellowship");am_Builder b;std::vector<uint64_t> rows;
    if(truth(s,obs_attr(s,fs,"hidden"))){
        const bool mordor=truth(s,obs_attr(s,fs,"mordor_entered"));
        if(mordor?ht_integer(s,fs,"mordor_step")<5:ht_integer(s,fs,"progress")<12)rows.push_back(b.row(index,"fellowship_move",{}));
    }else rows.push_back(b.row(index,"fellowship_hide",{}));
    b.out.root=b.container(WOTR_LIST,rows);return std::make_unique<Snapshot>(std::move(b.out));
}
uint64_t fh_die(const Snapshot& s,int64_t index,bool strider){
    if(!strider)return fm_payment_die(s,index);
    const auto dice=child_edges(s,pa_pool(s,0));require(index>=0 && static_cast<uint64_t>(index)<dice.size(),"Hide payment die index out of range");const auto die=dice[index];
    require(object_type(s,die)=="game_state.ActionDie" && enum_is(s,obs_attr(s,die,"side"),"game_state.Side","FREE_PEOPLES") && !truth(s,obs_attr(s,die,"used")),"Hide payment requires unused FP die");
    rr_face_index(enum_member(s,obs_attr(s,die,"current"),"game_state.ActionDieFace"));return die;
}
std::unique_ptr<Snapshot> apply_fellowship_hide(const Snapshot& source,const Snapshot& tables,int64_t index,bool strider){
    quiet_stable_action_gate(source,tables);ring_waiver(source,0);ring_waiver(source,1);fh_die(source,index,strider);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained) || ctx_side(source,retained)==0,"Hide belongs to retained ring actor");
    const auto fellowship=obs_attr(source,source.root,"fellowship");require(!truth(source,obs_attr(source,fellowship,"hidden")),"Fellowship is already hidden");
    if(strider){const auto guide=obs_attr(source,fellowship,"guide");require(!ctx_none(source,guide) && text(source,obs_attr(source,guide,"name")).find("Strider")!=std::string_view::npos,"Any-die hide requires Strider as Guide");}
    auto child=action_clone(source);auto& s=*child;
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    rr_set_attr(s,fh_die(s,index,strider),"used",rr_scalar(s,WOTR_BOOL,1));
    rr_set_attr(s,obs_attr(s,s.root,"fellowship"),"hidden",rr_scalar(s,WOTR_BOOL,1));
    rr_set_attr(s,s.root,"fellowship_acted_in_mordor",rr_scalar(s,WOTR_BOOL,1));rr_set_attr(s,s.root,"_battle_resolved_attacker",rr_scalar(s,WOTR_NONE));return child;
}
