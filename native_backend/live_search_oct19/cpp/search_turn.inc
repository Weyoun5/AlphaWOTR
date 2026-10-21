// Self-play/search uses ai._safe_next_turn, not game_logic.next_turn.
// Terminal inactivity corruption stops before any reset or card draws.
std::vector<std::string> st_blockers(const Snapshot& s,const Snapshot& t){
    std::vector<std::string> result;
    for(const char* key:{"pending_combat","pending_hunt","pending_guide_change","pending_card_option","pending_card_followup","pending_discard","pending_army_move_die","pending_muster_die","_fellowship_move_die_pending"})
        if(!ctx_none(s,obs_attr(s,s.root,key,false)))result.emplace_back(key);
    for(const char* key:{"_pending_hunt_damage","_hunt_reveal_pending","_reveal_place_due","_pending_reveal_strongholds","_fellowship_route_event","_hobbit_place_queue","_mithril_stash","_pending_reveal_strongholds","_fellowship_route_event","_wwsat_owed","_palantir_draw_pending","_gandalf_draw_pending","_owed_draws","_declare_phase","_eye_allocation_phase","_fellowship_tail_phase"})
        if(truth(s,obs_attr(s,s.root,key,false)))result.emplace_back(key);
    if(!result.empty())return result;
    for(int side=0;side<2;++side){
        size_t unused=0;for(auto die:child_edges(s,pa_pool(s,side)))unused+=!truth(s,obs_attr(s,die,"used"))&&!ctx_none(s,obs_attr(s,die,"current"));
        if(!unused)continue;
        auto probe=action_clone(s);auto raw=native_full_actions(*probe,t,side);auto policy=project_native_ai_actions(*probe,*raw,side);
        if(truth(*policy.actions,policy.actions->root))result.push_back(std::string("unused_action_dice:")+(side==0?"FREE_PEOPLES":"SHADOW")+"="+std::to_string(unused));
    }
    return result;
}
bool st_safe_next_turn(Snapshot& s,const Snapshot& t,am_Builder& events,std::vector<uint64_t>& rows){
    const auto blockers=st_blockers(s,t);
    if(!blockers.empty()){std::string error="unfinished_turn: refusing rollover with ";for(size_t i=0;i<blockers.size();++i){if(i)error+=", ";error+=blockers[i];}throw std::invalid_argument(error);}
    if(tr_settle(s,t))return true;
    const auto turn=typed(s,obs_attr(s,s.root,"turn"),WOTR_INT,"Turn must be integer").integer;require(turn<std::numeric_limits<int64_t>::max(),"Next turn exceeds int64");
    tr_reset_physical(s,turn+1,false);
    for(const char* key:{"pending_combat","pending_hunt","pending_guide_change","pending_army_move_die","pending_army_move_from","pending_army_move_side","pending_army_move_preexist","pending_muster_die","pending_muster_first_type","pending_muster_region","pending_muster_side","pending_card_option","_gandalf_draw_pending"})rr_set_attr(s,s.root,key,rr_scalar(s,WOTR_NONE));
    for(const char* key:{"pending_card_followup","pending_discard"})if(obs_attr(s,s.root,key,false)!=missing)rr_set_attr(s,s.root,key,rr_scalar(s,WOTR_NONE));
    rr_set_attr(s,s.root,"_pending_hunt_damage",rr_scalar(s,WOTR_INT,0));
    rr_set_attr(s,s.root,"_reveal_place_due",rr_scalar(s,WOTR_BOOL,0));
    for(const char* key:{"_hobbit_place_queue","_wwsat_owed"})rr_set_attr(s,s.root,key,rr_scalar(s,WOTR_LIST));
    rr_set_attr(s,s.root,"_palantir_draw_pending",rr_scalar(s,WOTR_BOOL,0));
    // Failures are surfaced atomically instead of writing a violation and
    // continuing with an incompletely reset child.
    su_cleanup(s,t);tr_bonus_dice(s);nt_deal(s,events,rows);return false;
}
