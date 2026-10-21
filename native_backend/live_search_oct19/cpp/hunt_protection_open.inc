// Hunt protection timing shared by ordinary, Mordor, and reveal-route draws.
void ew_open(Snapshot& s,void* rng);
bool ec_open(Snapshot& s,int64_t index);
uint64_t hpr_card(const Snapshot& s,std::string_view type){
    const auto persistent=obs_attr(s,s.root,"persistent_cards",false);if(ctx_none(s,persistent))return missing;
    for(const char* side:{"fp_cards","sh_cards"})for(auto card:child_edges(s,obs_attr(s,persistent,side)))
        if(enum_is(s,obs_attr(s,card,"card_type"),"game_state.PersistentCardType",type))return card;
    return missing;
}
bool hpr_card_supported(const Snapshot& s,uint64_t card){
    const auto type=obs_attr(s,card,"card_type");bool supported=false;
    // PersistentCard.card_type=None is generic tracking in Python: typed
    // rule lookups ignore it. Preserve the record without inventing an effect.
    if(ctx_none(s,type))return object_type(s,card)=="game_state.PersistentCard";
    if(enum_is(s,type,"game_state.PersistentCardType","PALANTIR_OF_ORTHANC"))
        return object_type(s,card)=="game_state.PersistentCard" &&
            text_is(s,obs_attr(s,card,"card_id"),"SHADOW_Character_The_Palant\xc3\xadr_of_Orthanc") &&
            text_is(s,obs_attr(s,card,"card_name"),"The Palant\xc3\xadr of Orthanc") &&
            enum_is(s,obs_attr(s,card,"side"),"game_state.Side","SHADOW") &&
            ctx_none(s,obs_attr(s,card,"requires_companion",false));
    if(enum_is(s,type,"game_state.PersistentCardType","POWER_TOO_GREAT"))
        return object_type(s,card)=="game_state.PersistentCard" &&
            text_is(s,obs_attr(s,card,"card_id"),"FREE_PEOPLES_Strategy_A_Power_too_Great") &&
            text_is(s,obs_attr(s,card,"card_name"),"A Power too Great") &&
            enum_is(s,obs_attr(s,card,"side"),"game_state.Side","FREE_PEOPLES") &&
            ctx_none(s,obs_attr(s,card,"requires_companion",false));
    if(enum_is(s,type,"game_state.PersistentCardType","TOM_BOMBADIL"))
        return object_type(s,card)=="game_state.PersistentCard" &&
            text_is(s,obs_attr(s,card,"card_id"),"FREE_PEOPLES_Strategy_The_Power_of_Tom_Bombadil") &&
            text_is(s,obs_attr(s,card,"card_name"),"The Power of Tom Bombadil") &&
            enum_is(s,obs_attr(s,card,"side"),"game_state.Side","FREE_PEOPLES") &&
            ctx_none(s,obs_attr(s,card,"requires_companion",false));
    if(enum_is(s,type,"game_state.PersistentCardType","LAST_BATTLE"))
        return object_type(s,card)=="game_state.PersistentCard" &&
            text_is(s,obs_attr(s,card,"card_id"),"FREE_PEOPLES_Strategy_The_Last_Battle") &&
            text_is(s,obs_attr(s,card,"card_name"),"The Last Battle") &&
            enum_is(s,obs_attr(s,card,"side"),"game_state.Side","FREE_PEOPLES") &&
            ctx_none(s,obs_attr(s,card,"requires_companion",false));
    for(auto name:{"WIZARD_STAFF","MITHRIL_COAT","AXE_AND_BOW","HORN_OF_GONDOR"})supported|=enum_is(s,type,"game_state.PersistentCardType",name);
    const bool shadow=enum_is(s,type,"game_state.PersistentCardType","BALROG_OF_MORIA") || enum_is(s,type,"game_state.PersistentCardType","WORN_WITH_SORROW") || enum_is(s,type,"game_state.PersistentCardType","FLOCKS_OF_CREBAIN");
    return (supported || shadow) && enum_is(s,obs_attr(s,card,"side"),"game_state.Side",shadow?"SHADOW":"FREE_PEOPLES");
}
void hpr_card_gate(const Snapshot& s){
    political_table_gate(s);
}
uint64_t hpr_dict(Snapshot& s,const std::vector<uint64_t>& entries){const auto out=rr_scalar(s,WOTR_DICT);set_edges(s,out,entries);return out;}
bool hpr_offer_wizard(Snapshot& s,uint64_t stash){
    const auto card=hpr_card(s,"WIZARD_STAFF");if(card==missing || !ctx_none(s,obs_attr(s,s.root,"pending_card_option",false)))return false;
    rr_set_attr(s,s.root,"_wizardstaff_stash",stash);
    const auto yes=hpr_dict(s,{rr_string(s,"wizardstaff_prevent"),rr_scalar(s,WOTR_BOOL,1)}),no=hpr_dict(s,{rr_string(s,"wizardstaff_prevent"),rr_scalar(s,WOTR_BOOL)});
    const auto options=rr_scalar(s,WOTR_LIST);set_edges(s,options,{yes,no});
    const auto pending=hpr_dict(s,{rr_string(s,"card_id"),obs_attr(s,card,"card_id"),rr_string(s,"card_name"),rr_string(s,"Wizard's Staff"),rr_string(s,"side"),rr_string(s,"fp"),rr_string(s,"decider"),rr_string(s,"fp"),rr_string(s,"direct"),rr_scalar(s,WOTR_BOOL,1),rr_string(s,"options"),options});
    rr_set_attr(s,s.root,"pending_card_option",pending);return true;
}
bool hpr_offer_mithril(Snapshot& s,uint64_t stash){
    const auto card=hpr_card(s,"MITHRIL_COAT");if(card==missing || !ctx_none(s,obs_attr(s,s.root,"pending_card_option",false)))return false;
    rr_set_attr(s,s.root,"_mithril_stash",stash);
    const auto yes=hpr_dict(s,{rr_string(s,"mithril_redraw"),rr_scalar(s,WOTR_BOOL,1)}),no=hpr_dict(s,{rr_string(s,"mithril_redraw"),rr_scalar(s,WOTR_BOOL)});
    const auto options=rr_scalar(s,WOTR_LIST);set_edges(s,options,{yes,no});
    const auto pending=hpr_dict(s,{rr_string(s,"card_id"),obs_attr(s,card,"card_id"),rr_string(s,"card_name"),rr_string(s,"Mithril Coat and Sting"),rr_string(s,"side"),rr_string(s,"fp"),rr_string(s,"decider"),rr_string(s,"fp"),rr_string(s,"direct"),rr_scalar(s,WOTR_BOOL,1),rr_string(s,"options"),options});
    rr_set_attr(s,s.root,"pending_card_option",pending);return true;
}
bool hpr_offer_reduction(Snapshot& s){
    const auto damage=obs_attr(s,s.root,"_pending_hunt_damage",false);
    if(ctx_none(s,damage) || typed(s,damage,WOTR_INT,"Hunt damage must be integer").integer<=0 || !ctx_none(s,obs_attr(s,s.root,"pending_card_option",false)))return false;
    auto card=hpr_card(s,"AXE_AND_BOW");if(card==missing)card=hpr_card(s,"HORN_OF_GONDOR");if(card==missing)return false;
    std::vector<uint64_t> rows;
    for(const char* kind:{"AXE_AND_BOW","HORN_OF_GONDOR"}){const auto payer=hpr_card(s,kind);if(payer!=missing)rows.push_back(hpr_dict(s,{rr_string(s,"use_hunt_reduction"),rr_scalar(s,WOTR_INT,1),rr_string(s,"card_id"),obs_attr(s,payer,"card_id")}));}
    rows.push_back(hpr_dict(s,{rr_string(s,"use_hunt_reduction"),rr_scalar(s,WOTR_INT)}));
    const auto options=rr_scalar(s,WOTR_LIST);set_edges(s,options,rows);
    const auto pending=hpr_dict(s,{rr_string(s,"card_id"),obs_attr(s,card,"card_id"),rr_string(s,"card_name"),obs_attr(s,card,"card_name"),rr_string(s,"side"),rr_string(s,"fp"),rr_string(s,"decider"),rr_string(s,"fp"),rr_string(s,"direct"),rr_scalar(s,WOTR_BOOL,1),rr_string(s,"hunt_reduction"),rr_scalar(s,WOTR_BOOL,1),rr_string(s,"options"),options});
    rr_set_attr(s,s.root,"pending_card_option",pending);return true;
}
bool hpr_return_mithril(Snapshot& s,void* rng){
    const auto stash=obs_attr(s,s.root,"_mithril_stash",false);if(!truth(s,stash) || !truth(s,field(s,stash,"return_pending",false)))return false;
    for(auto key:{"pending_combat","pending_hunt","pending_guide_change","pending_card_option","pending_card_followup","pending_discard","pending_army_move_die","pending_muster_die"})
        if(!ctx_none(s,obs_attr(s,s.root,key,false)))return false;
    for(auto key:{"_pending_hunt_damage","_hunt_reveal_pending","_reveal_place_due","_pending_reveal_strongholds","_fellowship_route_event","_hobbit_place_queue","_overstack_regions","_wwsat_owed"})
        if(truth(s,obs_attr(s,s.root,key,false)))return false;
    const auto pool=obs_attr(s,s.root,"hunt_pool");auto tiles=child_edges(s,pool);tiles.push_back(field(s,stash,"tile"));
    rr_set_attr(s,s.root,"_mithril_stash",rr_scalar(s,WOTR_NONE));
    std::vector<uint64_t> permutation(tiles.size()),shuffled(tiles.size());if(hf_shuffle(rng,permutation.data(),permutation.size()))throw std::invalid_argument(wotr_random_last_error());
    for(size_t i=0;i<tiles.size();++i)shuffled[i]=tiles[permutation[i]];set_edges(s,pool,shuffled);return true;
}
