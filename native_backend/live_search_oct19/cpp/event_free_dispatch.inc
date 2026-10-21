// Included only after the shared action tail and all FP Character in-place
// executors. The caller has validated/consumed a complete Ents stored row.
void ef_finish_free(Snapshot& s,const Snapshot& t,void* rng,int marker,int64_t before,am_Builder& events,std::vector<uint64_t>& rows){cau_finish(s,t,rng,ef_final_marker(s,marker,before),events,rows);}
void ef_execute_free_event(Snapshot& s,const Snapshot& t,void* rng,std::string_view id,am_Builder& events,std::vector<uint64_t>& rows){
    const int marker=ctx_side(s,obs_attr(s,s.root,"_battle_resolved_attacker",false));const auto unused=ef_unused_dice(s);
    rr_set_attr(s,s.root,"_casualty_operations",cp_container(s,WOTR_LIST,{}));cau_dedupe(s);rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    if(truth(s,obs_attr(s,s.root,"_declare_phase",false)))rr_set_attr(s,s.root,"_declare_phase",rr_scalar(s,WOTR_BOOL,0));
    // Stored choices may become impossible. Source's ordinary play has both
    // the phantom-hand guard and the requirement's retained-card no-op.
    if(!contains_string(s,tc_hand(s,0),id)){ef_finish_free(s,t,rng,marker,unused,events,rows);return;}
    ef_validate_free(s,t,0,-1,id,true);const auto card=ev_card(t,id);
    if(!ev_requirement(s,t,card)){ef_finish_free(s,t,rng,marker,unused,events,rows);return;}
    if(eea_supported(t,card))return apply_ents_event_inplace(s,t,rng,0,-1,id,true,events,rows);
    if(eaw_supported(t,card))return apply_another_way_event_inplace(s,t,rng,0,-1,id,true,events,rows);
    if(eal_supported(t,card))return apply_alone_event_inplace(s,t,rng,0,-1,id,true,events,rows);
    if(eck_supported(t,card))return apply_challenge_event_inplace(s,t,rng,0,-1,id,true,events,rows);
    if(egw_supported(t,card))return apply_gwaihir_event_inplace(s,t,rng,0,-1,id,true,events,rows);
    if(esw_supported(t,card))return apply_swifter_event_inplace(s,t,rng,0,-1,id,true,events,rows);
    if(etb_supported(t,card))return apply_back_again_event_inplace(s,t,rng,0,-1,id,true,events,rows);
    if(eeg_supported(t,card))return apply_eagles_event_inplace(s,t,rng,0,-1,id,true,events,rows);
    if(ehs_supported(t,card))return apply_house_stewards_event_inplace(s,t,rng,0,-1,id,true,events,rows);
    if(egc_supported(t,card))return apply_grey_company_event_inplace(s,t,rng,0,-1,id,true,events,rows);
    if(edm_supported(t,card))return apply_dead_men_event_inplace(s,t,rng,0,-1,id,true,events,rows);
    if(eh_template(t,card)!=missing)return apply_special_tile_event_action_inplace(s,t,rng,0,-1,id,true,events,rows);
    if(eht_type(t,card))return apply_hunt_table_event_inplace(s,t,rng,0,-1,id,true,events,rows);
    if(em_supported(t,card))return apply_misc_event_action_inplace(s,t,rng,0,-1,id,true,events,rows);
    if(ed_supported(t,card))return apply_dice_event_action_inplace(s,t,rng,0,-1,id,true,events,rows);
    if(ev_simple_supported(t,card))return apply_simple_event_action_inplace(s,t,rng,0,-1,id,true,events,rows);
    throw std::invalid_argument("Ents free Character event effect is not ported");
}
