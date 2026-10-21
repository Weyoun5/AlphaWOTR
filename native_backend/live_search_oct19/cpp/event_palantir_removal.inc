// ai._get_table_kill_actions / discard_table_card: FP's deliberate Palantir
// removal. Ring payment is independent of the once-per-turn die conversion.
std::unique_ptr<Snapshot> native_palantir_kill_for_die(const Snapshot& s,int side,int64_t index){
    am_Builder b;std::vector<uint64_t> rows;const auto card=hpr_card(s,"PALANTIR_OF_ORTHANC");
    if(side==0 && card!=missing){
        const auto dice=child_edges(s,pa_pool(s,0));require(index>=0&&static_cast<uint64_t>(index)<dice.size(),"Palantir removal die index invalid");
        const bool will=enum_is(s,obs_attr(s,dice[index],"current"),"game_state.ActionDieFace","WILL_OF_WEST");
        if(will || !rr_nonpositive(s,obs_attr(s,s.root,"fp_elven_rings")))
            rows.push_back(b.row(index,"discard_table_card",{b.str("card_id"),b.str(text(s,obs_attr(s,card,"card_id"))),b.str("card_name"),b.str(text(s,obs_attr(s,card,"card_name"))),b.str("payment"),b.str(will?"will":"ring")}));
    }
    b.out.root=b.container(WOTR_LIST,rows);return std::make_unique<Snapshot>(b.out);
}
uint64_t epa_kill_die(const Snapshot& s,int64_t index,std::string_view payment){
    const auto dice=child_edges(s,pa_pool(s,0));require(index>=0&&static_cast<uint64_t>(index)<dice.size(),"Palantir removal die index invalid");
    const auto die=dice[index];require(object_type(s,die)=="game_state.ActionDie"&&ctx_side(s,obs_attr(s,die,"side"))==0&&!truth(s,obs_attr(s,die,"used")),"Palantir removal die unavailable");
    const auto face=enum_member(s,obs_attr(s,die,"current"),"game_state.ActionDieFace");rr_face_index(face);
    require(payment=="will" || payment=="ring","Palantir removal payment invalid");
    if(payment=="will")require(face=="WILL_OF_WEST","Palantir removal requires a Will die");
    else require(face!="WILL_OF_WEST"&&!rr_nonpositive(s,obs_attr(s,s.root,"fp_elven_rings")),"Palantir ring removal is not currently offered");
    return die;
}
std::unique_ptr<Snapshot> apply_palantir_kill(const Snapshot& source,const Snapshot& t,int side,int64_t index,std::string_view id,std::string_view payment){
    require(side==0,"Palantir removal belongs to FP");quiet_stable_action_gate(source,t);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==0,"Palantir removal belongs to retained Ring actor");
    const auto card=hpr_card(source,"PALANTIR_OF_ORTHANC");require(card!=missing&&text_is(source,obs_attr(source,card,"card_id"),id),"Palantir removal target absent");epa_kill_die(source,index,payment);
    auto child=action_clone(source);auto& s=*child;rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    rr_set_attr(s,epa_kill_die(s,index,payment),"used",rr_scalar(s,WOTR_BOOL,1));
    if(payment=="ring"){
        rr_set_attr(s,s.root,"fp_elven_rings",rr_adjust_counter(s,obs_attr(s,s.root,"fp_elven_rings"),-1));
        rr_set_attr(s,s.root,"sh_elven_rings",rr_adjust_counter(s,obs_attr(s,s.root,"sh_elven_rings"),1));
    }
    const auto view=political_view(s,"ROHAN",false);const auto removed=remove_card(s,view,id);discard_card(s,view,removed);
    su_cleanup(s,t);tc_open_hand_limit_discard(s);cau_gandalf(s);rr_set_attr(s,s.root,"_battle_resolved_attacker",rr_scalar(s,WOTR_NONE));return child;
}
