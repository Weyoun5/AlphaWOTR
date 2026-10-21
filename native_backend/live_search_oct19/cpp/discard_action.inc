// Learned hand-limit discards; included after turn_cards and quiet action gate.
// Human card choices remain legal; no automatic choice policy is introduced.
void cau_gandalf(Snapshot& s);
std::vector<std::string> fr_destinations(const Snapshot& s,const Snapshot& t);
int dc_owner(const Snapshot& s);
// Package W (P9a): the hand-limit discard family mirrors the frozen menu branch order (ai.get_legal_actions, ai.py:7690-7711)
// instead of the quiet gate's scope box: a discard is served once no combat, guide change or Hunt decision is pending and no
// reveal placement is due first (game_logic._reveal_hunt_pending + ai._get_reveal_placement_actions), to the pending side,
// unless that side's OWN card option / follow-up is still resolving (ai._pending_option_owner, ai.py:13184-13202).  Every
// other pending or phase flag (the other side's die pair, a paid Fellowship die, the declare / eye phases, deferred draws,
// bookkeeping stashes) is tolerated exactly as the frozen menu tolerates it.
void dc_gate(const Snapshot& s,const Snapshot& tables,int side){
    for(const char* key:{"pending_combat","pending_guide_change","pending_hunt"})
        require(ctx_none(s,obs_attr(s,s.root,key,false)),"hand-limit discard waits for the pending decision");
    if(truth(s,obs_attr(s,s.root,"_reveal_place_due",false))){
        const bool hunt_pending=!ctx_none(s,obs_attr(s,s.root,"pending_card_option",false))||truth(s,obs_attr(s,s.root,"_pending_hunt_damage",false)); // pending_hunt / pending_guide_change are absent here
        require(hunt_pending || fr_destinations(s,tables).empty(),"hand-limit discard waits for the reveal placement");
    }
    require(dc_owner(s)==side,"hand-limit discard belongs to the other side");
    for(const char* key:{"pending_card_option","pending_card_followup"}){
        const auto p=obs_attr(s,s.root,key,false);if(ctx_none(s,p))continue;
        const auto tag=field(s,p,"side",false);if(tag==missing||ctx_none(s,tag))continue;
        if(text_is(s,tag,"fp")){require(side!=0,"hand-limit discard yields to the owner's card option");break;}
        if(text_is(s,tag,"sh")){require(side!=1,"hand-limit discard yields to the owner's card option");break;}
    }
}
int dc_owner(const Snapshot& s){
    const auto pending=obs_attr(s,s.root,"pending_discard",false);
    require(!ctx_none(s,pending),"no pending hand-limit discard");
    const int owner=tc_pending_side(s,pending);
    require(owner>=0,"pending hand-limit discard needs a canonical owner");return owner;
}
std::unique_ptr<Snapshot> discard_card_actions(const Snapshot& s,const Snapshot& tables,int32_t side){
    require(side==0 || side==1,"discard menu side must be FP or SH");
    am_Builder b;std::vector<uint64_t> rows;std::set<std::string> seen;
    if(dc_owner(s)==side){
        dc_gate(s,tables,side); // Only the owner enters the discard branch; other-side menu is empty.
        for(auto card:child_edges(s,tc_hand(s,side))){
        const std::string id(text(s,card));
        if(seen.insert(id).second)rows.push_back(b.row(-1,"discard_card",{
            b.str("card_id"),b.str(id),b.str("side"),b.str(side==0?"fp":"sh")}));
        }
    }
    b.out.root=b.container(WOTR_LIST,rows);return std::make_unique<Snapshot>(b.out);
}
void dc_discard_inner(Snapshot& s,int side,std::string_view card){
    const auto player=tc_player(s,side),hand=tc_hand(s,side);auto cards=child_edges(s,hand);
    const auto found=std::find_if(cards.begin(),cards.end(),[&](uint64_t n){return text_is(s,n,card);});
    require(found!=cards.end(),"discard card vanished during action clone");
    const auto id=*found;cards.erase(found);set_edges(s,hand,cards);
    const auto pile=obs_attr(s,player,"discarded_face_down");auto discarded=child_edges(s,pile);
    discarded.push_back(id);set_edges(s,pile,discarded);
    rr_set_attr(s,player,"needs_discard",rr_scalar(s,WOTR_BOOL,s.nodes[hand].count>6));
    rr_set_attr(s,s.root,"pending_discard",rr_scalar(s,WOTR_NONE));
    tc_open_hand_limit_discard(s); // shared primitive's mandatory re-derivation
}
std::unique_ptr<Snapshot> apply_discard_card(const Snapshot& source,const Snapshot& tables,int32_t side,std::string_view card){
    require(side==0 || side==1,"discard side must be FP or SH");
    dc_gate(source,tables,side); // the redundant owner check below keeps its refusal text
    require(dc_owner(source)==side,"hand-limit discard belongs to the other side");
    require(contains_string(source,tc_hand(source,side),card),"discard card is not in owner's hand");
    auto child=action_clone(source);auto& s=*child;
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));
    rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    dc_discard_inner(s,side,card);
    tc_open_hand_limit_discard(s); // apply_action's finally block checks again
    cau_gandalf(s); // An earned draw waits until mandatory hand-limit work ends.
    return child;
}
