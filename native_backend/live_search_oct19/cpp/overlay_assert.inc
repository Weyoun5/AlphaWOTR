// Only the source _ov_ok-filtered opponent hand is needed by the assertion.
std::unique_ptr<Snapshot> native_overlay_assert_hand(const Snapshot& s,int side){
    require(side==0||side==1,"Overlay assertion side must be FP or Shadow");
    sm_Builder b(s);std::vector<std::string> cards;
    auto hands=obs_attr(s,s.root,"hands");auto entry=enum_value_edge(s,hands,"game_state.Side",side==0?"FREE_PEOPLES":"SHADOW");
    auto info=entry==missing?missing:s.edges[entry];
    if(truth(s,info)){
        auto hand=obs_attr(s,info,"hand");
        for(auto card:child_edges(s,hand))if(s.nodes[card].tag==WOTR_STRING){auto value=text(s,card);if(value.substr(0,9)!="_unknown_")cards.emplace_back(value);}
    }
    std::sort(cards.begin(),cards.end());std::vector<uint64_t> values;for(auto& card:cards)values.push_back(b.str(card));
    b.out.root=b.container(WOTR_LIST,values);return std::make_unique<Snapshot>(b.out);
}
