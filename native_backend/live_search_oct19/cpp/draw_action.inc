// Ordinary Event/Will die card draws. The deck tail is already sampled in
// state: drawing does not consume RNG and does not trigger event-play bonuses.
uint64_t da_payment(const Snapshot& s,int32_t side,int64_t index){
    const auto dice=child_edges(s,pa_pool(s,side));
    require(index>=0 && static_cast<uint64_t>(index)<dice.size(),"draw payment die index is out of range");
    const auto die=dice[index];require(object_type(s,die)=="game_state.ActionDie","draw payment requires ActionDie");
    require(ctx_side(s,obs_attr(s,die,"side"))==side,"draw payment die belongs to another side");
    require(!truth(s,obs_attr(s,die,"used")),"draw payment die is already used");
    const auto face=obs_attr(s,die,"current");
    require(enum_is(s,face,"game_state.ActionDieFace","EVENT") || enum_is(s,face,"game_state.ActionDieFace","WILL_OF_WEST"),
            "draw payment requires Event or Will of the West");
    return die;
}
const char* da_deck_name(int32_t side,std::string_view deck){
    require(deck=="character" || deck=="strategy","draw requires character or strategy deck");
    return side==0?(deck=="character"?"fp_char_deck":"fp_strat_deck"):(deck=="character"?"sh_char_deck":"sh_strat_deck");
}
std::unique_ptr<Snapshot> native_draw_card_actions(const Snapshot& s,const Snapshot& t,int32_t side){
    require(side==0 || side==1,"draw menu side must be FP or SH");quiet_stable_action_gate(s,t);ring_waiver(s,side);
    am_Builder b;std::vector<uint64_t> rows;std::array<bool,7> seen{};
    const auto dice=child_edges(s,pa_pool(s,side));
    for(size_t i=0;i<dice.size();++i){
        const auto die=dice[i];require(object_type(s,die)=="game_state.ActionDie","draw menu requires ActionDie");
        const auto face=obs_attr(s,die,"current");if(truth(s,obs_attr(s,die,"used")) || ctx_none(s,face))continue;
        const auto index=rr_face_index(enum_member(s,face,"game_state.ActionDieFace"));if(seen[index])continue;seen[index]=true;
        if(index!=3 && index!=6)continue;
        for(const auto deck:{"character","strategy"}){
            const auto pile=obs_attr(s,s.root,da_deck_name(side,deck),false);
            if(!truth(s,pile))continue;typed(s,pile,WOTR_LIST,"draw menu deck must be a list");
            rows.push_back(b.row(i,"draw_card",{b.str("deck"),b.str(deck),b.str("side"),b.str(side==0?"fp":"sh")}));
        }
    }
    b.out.root=b.container(WOTR_LIST,rows);return std::make_unique<Snapshot>(b.out);
}
std::unique_ptr<Snapshot> apply_draw_card_action(const Snapshot& s,const Snapshot& t,int32_t side,int64_t index,std::string_view deck){
    require(side==0 || side==1,"draw action side must be FP or SH");const auto name=da_deck_name(side,deck);
    quiet_stable_action_gate(s,t);ring_waiver(s,0);ring_waiver(s,1);
    const auto retained=obs_attr(s,s.root,"_ring_free_side",false);
    require(ctx_none(s,retained) || ctx_side(s,retained)==side,"draw action belongs to retained Ring actor");
    da_payment(s,side,index);
    const auto pile=obs_attr(s,s.root,name,false);
    if(truth(s,pile))typed(s,pile,WOTR_LIST,"draw deck must be a list");
    auto child=action_clone(s);auto& out=*child;
    rr_set_attr(out,out.root,"_casualty_operations",rr_scalar(out,WOTR_LIST));
    rr_set_attr(out,out.root,"_ring_free_side",rr_scalar(out,WOTR_NONE));
    rr_set_attr(out,da_payment(out,side,index),"used",rr_scalar(out,WOTR_BOOL,1));
    // The frozen executor catches an empty-deck RuntimeError AFTER payment.
    // Preserve that behavior for a stale caller, while the menu omits it.
    if(truth(out,obs_attr(out,out.root,name,false)))tc_draw_event_card(out,side,deck);
    tc_open_hand_limit_discard(out);
    rr_set_attr(out,out.root,"_battle_resolved_attacker",rr_scalar(out,WOTR_NONE));
    return child;
}
