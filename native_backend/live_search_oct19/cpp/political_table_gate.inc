// Political-only Threats and Promises may coexist with the five verified
// Hunt cards, which have no political modifier or nation-advance discard.
bool hpr_card_supported(const Snapshot& s,uint64_t card);
bool political_table_gate(const Snapshot& s){
    const auto pc=obs_attr(s,s.root,"persistent_cards");attributes(s,pc,"game_state.PersistentCardState");
    const auto fp=obs_attr(s,pc,"fp_cards"),sh=obs_attr(s,pc,"sh_cards");
    typed(s,fp,WOTR_LIST,"political table requires FP list");typed(s,sh,WOTR_LIST,"political table requires SH list");
    for(auto card:child_edges(s,fp))require(hpr_card_supported(s,card),"political table has an unsupported FP effect");
    std::vector<uint64_t> other;for(auto card:child_edges(s,sh))if(!hpr_card_supported(s,card))other.push_back(card);
    size_t threats=0;std::vector<std::string> unported;
    for(auto card:other){
        attributes(s,card,"game_state.PersistentCard");
        const auto type=obs_attr(s,card,"card_type");
        const bool claims_threats=enum_is(s,type,"game_state.PersistentCardType","THREATS_AND_PROMISES") ||
            text_is(s,obs_attr(s,card,"card_id"),"SHADOW_Strategy_Threats_and_Promises") ||
            text_is(s,obs_attr(s,card,"card_name"),"Threats and Promises");
        if(claims_threats){
            require(text_is(s,obs_attr(s,card,"card_id"),"SHADOW_Strategy_Threats_and_Promises") &&
                    text_is(s,obs_attr(s,card,"card_name"),"Threats and Promises") &&
                    enum_is(s,obs_attr(s,card,"side"),"game_state.Side","SHADOW") &&
                    enum_is(s,type,"game_state.PersistentCardType","THREATS_AND_PROMISES") &&
                    ctx_none(s,obs_attr(s,card,"requires_companion",false)),
                    "political table supports only canonical Threats and Promises");
            ++threats;continue;
        }
        const std::string kind(enum_member(s,type,"game_state.PersistentCardType"));
        if(kind=="PALANTIR_OF_ORTHANC"){
            require(text_is(s,obs_attr(s,card,"card_id"),"SHADOW_Character_The_Palant\xc3\xadr_of_Orthanc") &&
                    text_is(s,obs_attr(s,card,"card_name"),"The Palant\xc3\xadr of Orthanc") &&
                    enum_is(s,obs_attr(s,card,"side"),"game_state.Side","SHADOW") &&
                    ctx_none(s,obs_attr(s,card,"requires_companion",false)),
                    "political table has malformed Palantir of Orthanc");
        }
        if(kind=="DENETHORS_FOLLY"){
            require(text_is(s,obs_attr(s,card,"card_id"),"SHADOW_Strategy_Denethor's_Folly")&&text_is(s,obs_attr(s,card,"card_name"),"Denethor's Folly")&&enum_is(s,obs_attr(s,card,"side"),"game_state.Side","SHADOW")&&ctx_none(s,obs_attr(s,card,"requires_companion",false)),"political table has malformed Denethor's Folly");
            continue;
        }
        if(kind=="WORMTONGUE"){
            require(text_is(s,obs_attr(s,card,"card_id"),"SHADOW_Character_Wormtongue") &&
                    text_is(s,obs_attr(s,card,"card_name"),"Wormtongue") &&
                    enum_is(s,obs_attr(s,card,"side"),"game_state.Side","SHADOW") &&
                    ctx_none(s,obs_attr(s,card,"requires_companion",false)),
                    "political table has malformed Wormtongue");
            continue;
        }
        unported.push_back(kind);
    }
    // Count actual canonical Threats, not unrelated unported effects. Inspect
    // all Threats metadata before an unsupported-domain return so that a bad
    // card or duplicate remains a hard failure even beside a legal Palantir.
    require(threats<=1,"political table supports at most one Threats and Promises");
    if(!unported.empty())throw std::invalid_argument("political table has an unported Shadow effect: "+unported.front());
    return threats==1;
}
