// Exact draw/hand-limit state primitives from game_logic.py:3963 and :406.
// Call on an unpublished child/transaction. No dice payment, grant_draw debt,
// persistent-card trigger, policy choice, RNG or Python trace callback occurs.
uint64_t tc_player(const Snapshot& s,int32_t side){
    require(side==0 || side==1,"card side must be FP or SH");
    const auto hands=obs_attr(s,s.root,"hands");
    const auto edge=enum_value_edge(s,hands,"game_state.Side",side==0?"FREE_PEOPLES":"SHADOW");
    require(edge!=missing,"card state is missing side's hand");
    const auto player=s.edges[edge];require(object_type(s,player)=="game_state.PlayerInfo","card hand must be PlayerInfo");return player;
}
uint64_t tc_hand(const Snapshot& s,int32_t side){
    const auto hand=obs_attr(s,tc_player(s,side),"hand");typed(s,hand,WOTR_LIST,"card hand must be a list");return hand;
}
uint64_t tc_draw_event_card(Snapshot& s,int32_t side,std::string_view deck){
    const auto player=tc_player(s,side),hand=tc_hand(s,side);
    const bool character=deck=="character";
    const char* name=side==0?(character?"fp_char_deck":"fp_strat_deck"):(character?"sh_char_deck":"sh_strat_deck");
    const auto pile=obs_attr(s,s.root,name,false);
    if(!truth(s,pile))throw std::runtime_error(std::string(side==0?"FP ":"SH ")+(character?"character":"strategy")+" deck empty");
    const auto& list=typed(s,pile,WOTR_LIST,"card deck must be a mutable list");
    require(list.count>0,"card deck has no tail");
    const auto card=s.edges[list.offset+list.count-1];
    auto remaining=child_edges(s,pile);remaining.pop_back();set_edges(s,pile,remaining);
    // Fetch after pop: an aliased deck/hand list has already changed, matching
    // Python's sequential list operations rather than a stale copied hand.
    auto cards=child_edges(s,hand);cards.push_back(card);set_edges(s,hand,cards);
    rr_set_attr(s,player,"needs_discard",rr_scalar(s,WOTR_BOOL,cards.size()>6));
    return card;
}
int tc_pending_side(const Snapshot& s,uint64_t pending){
    if(s.nodes[pending].tag!=WOTR_DICT)return -1;
    const auto side=ctx_get(s,pending,"side");if(ctx_none(s,side))return -1;
    if(enum_is(s,side,"game_state.Side","FREE_PEOPLES"))return 0;
    if(enum_is(s,side,"game_state.Side","SHADOW"))return 1;
    const auto tag=s.nodes[side].tag;
    require(tag!=WOTR_LIST && tag!=WOTR_DICT && tag!=WOTR_SET && tag!=WOTR_OBJECT,
            "pending discard side is not a hashable canonical key");
    return -1; // Other scalar/enum keys are absent from canonical Side-keyed hands.
}
bool tc_open_hand_limit_discard(Snapshot& s){
    const auto pending=obs_attr(s,s.root,"pending_discard",false);
    if(!ctx_none(s,pending)){
        const int owner=tc_pending_side(s,pending);
        if(owner>=0){
            const auto hand=tc_hand(s,owner);
            if(s.nodes[hand].count>6)return true; // Preserve pending object and flags exactly.
        }
        rr_set_attr(s,s.root,"pending_discard",rr_scalar(s,WOTR_NONE));
        if(owner>=0)rr_set_attr(s,tc_player(s,owner),"needs_discard",rr_scalar(s,WOTR_BOOL));
    }
    for(int side=0;side<2;++side){
        const auto hand=tc_hand(s,side);const bool over=s.nodes[hand].count>6;
        rr_set_attr(s,tc_player(s,side),"needs_discard",rr_scalar(s,WOTR_BOOL,over));
        if(over){
            const auto hands=obs_attr(s,s.root,"hands");
            const auto edge=enum_value_edge(s,hands,"game_state.Side",side==0?"FREE_PEOPLES":"SHADOW");
            require(edge!=missing,"discard owner absent from hand dictionary");
            const auto key=rr_string(s,"side"),owner=s.edges[edge-1],next=rr_scalar(s,WOTR_DICT);
            set_edges(s,next,{key,owner});rr_set_attr(s,s.root,"pending_discard",next);return true;
        }
    }
    return false;
}
