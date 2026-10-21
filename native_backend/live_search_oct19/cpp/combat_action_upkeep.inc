// Shared original _execute_action bookkeeping for combat continuations.
// Included after Fellowship/Hunt completion helpers.
void cau_dedupe(Snapshot& s){
    const auto armies=obs_attr(s,s.root,"armies");const auto entries=child_edges(s,armies);std::map<std::string,uint64_t> seen;std::vector<std::pair<uint64_t,uint64_t>> duplicates;
    for(size_t i=0;i<entries.size();i+=2){const auto key=sanitize_name(text(s,entries[i]));const auto found=seen.find(key);if(found==seen.end())seen.emplace(key,entries[i+1]);else duplicates.emplace_back(found->second,entries[i]);}
    for(const auto& [keep,extra_key]:duplicates){const std::string key(text(s,extra_key));const auto extra=field(s,armies,key.c_str());for(const char* attribute:{"units","leaders"}){const auto target=obs_attr(s,keep,attribute);auto values=child_edges(s,target);const auto additional=child_edges(s,obs_attr(s,extra,attribute));values.insert(values.end(),additional.begin(),additional.end());set_edges(s,target,values);}cp_erase(s,armies,key);}
}
uint64_t cau_persistent(const Snapshot& s,std::string_view kind){
    const auto persistent=obs_attr(s,s.root,"persistent_cards",false);if(ctx_none(s,persistent))return missing;
    for(const char* side:{"fp_cards","sh_cards"})for(auto card:child_edges(s,obs_attr(s,persistent,side)))if(enum_is(s,obs_attr(s,card,"card_type"),"game_state.PersistentCardType",kind))return card;return missing;
}
void cau_last_battle(Snapshot& s,const Snapshot& tables){
    const auto last=cau_persistent(s,"LAST_BATTLE"),persistent=obs_attr(s,s.root,"persistent_cards",false);
    if(last!=missing){bool allowed=truth(s,obs_attr(s,obs_attr(s,s.root,"fellowship"),"hidden"));const auto id=obs_attr(s,last,"card_id"),card=field(tables,field(tables,tables.root,"cards"),std::string(text(s,id)).c_str(),false);allowed&=!ctx_none(tables,card);
        if(allowed)allowed=ev_requirement(s,tables,card);
        if(!allowed)cbp_discard_table(s,cbp_remove_card(s,persistent,last));
    }
    const auto denethor=cau_persistent(s,"DENETHORS_FOLLY");if(denethor!=missing&&!contains_string(s,obs_attr(s,s.root,"besieged_strongholds"),"minas tirith"))cbp_discard_table(s,cbp_remove_card(s,persistent,denethor));
}
void cau_wwsat(Snapshot& s,void* rng){
    const auto owed=obs_attr(s,s.root,"_wwsat_owed",false);if(!truth(s,owed))return;auto debts=child_edges(s,owed);debts.pop_back();set_edges(s,owed,debts);
    const auto player=tc_player(s,0),hand=obs_attr(s,player,"hand"),persistent=obs_attr(s,s.root,"persistent_cards",false);std::vector<uint64_t> chars,table;
    for(auto id:child_edges(s,hand)){const auto name=text(s,id);if(name.find("_Character_")!=std::string_view::npos||(name.substr(0,3)=="HU_"&&name.find("_CHAR_")!=std::string_view::npos))chars.push_back(id);}
    if(!ctx_none(s,persistent))for(auto card:child_edges(s,obs_attr(s,persistent,"fp_cards")))if(text(s,obs_attr(s,card,"card_id")).find("Character")!=std::string_view::npos)table.push_back(card);
    if(chars.empty()&&table.empty())return;
    std::vector<uint64_t> options;
    for(auto card:table)options.push_back(cp_container(s,WOTR_DICT,{rr_string(s,"wwsat_source"),rr_string(s,"table"),rr_string(s,"card_id"),obs_attr(s,card,"card_id")}));
    if(!chars.empty())options.push_back(cp_container(s,WOTR_DICT,{rr_string(s,"wwsat_source"),rr_string(s,"hand")}));
    options.push_back(cp_container(s,WOTR_DICT,{rr_string(s,"wwsat_source"),rr_string(s,"decline")}));
    rr_set_attr(s,s.root,"pending_card_option",cp_container(s,WOTR_DICT,{rr_string(s,"card_id"),rr_string(s,"Worn with Sorrow and Toil"),rr_string(s,"card_name"),rr_string(s,"__wwsat_discard__"),rr_string(s,"side"),rr_string(s,"sh"),rr_string(s,"decider"),rr_string(s,"sh"),rr_string(s,"direct"),rr_scalar(s,WOTR_BOOL,1),rr_string(s,"options"),cp_container(s,WOTR_LIST,options)}));
}
void cau_gandalf(Snapshot& s){
    // Shared completion hook also serves hand-limit and overstack tails
    // which can outlive the final combat choice.
    wt_settle_attack(s);
    const auto queued=obs_attr(s,s.root,"_gandalf_draw_pending",false);if(ctx_none(s,queued))return;
    for(const char* key:{"pending_combat","pending_hunt","pending_guide_change","pending_card_option","pending_card_followup","pending_discard","_pending_hunt_damage","_hunt_reveal_pending","_reveal_place_due","_pending_reveal_strongholds","_fellowship_route_event","_hobbit_place_queue","_overstack_regions","_wwsat_owed"})if(truth(s,obs_attr(s,s.root,key,false)))return;
    if(tc_open_hand_limit_discard(s))return;const auto fellowship=obs_attr(s,s.root,"fellowship"),guide=obs_attr(s,fellowship,"guide"),deck=field(s,queued,"deck");const bool eligible=!ctx_none(s,guide)&&hc_guide_member(s)&&text_is(s,obs_attr(s,guide,"name"),"Gandalf the Grey");const bool drawable=truth(s,obs_attr(s,s.root,text_is(s,deck,"character")?"fp_char_deck":"fp_strat_deck",false));rr_set_attr(s,s.root,"_gandalf_draw_pending",rr_scalar(s,WOTR_NONE));if(!eligible||!drawable)return;
    rr_set_attr(s,s.root,"pending_card_option",cp_container(s,WOTR_DICT,{rr_string(s,"card_id"),rr_string(s,"__gandalf_draw__"),rr_string(s,"card_name"),rr_string(s,"Gandalf guide draw"),rr_string(s,"side"),rr_string(s,"fp"),rr_string(s,"decider"),rr_string(s,"fp"),rr_string(s,"direct"),rr_scalar(s,WOTR_BOOL,1),rr_string(s,"gandalf_draw"),rr_scalar(s,WOTR_BOOL,1),rr_string(s,"source_card_id"),field(s,queued,"source_card_id"),rr_string(s,"deck"),deck,rr_string(s,"die_face"),field(s,queued,"die_face"),rr_string(s,"options"),cp_container(s,WOTR_LIST,{cp_container(s,WOTR_DICT,{rr_string(s,"gandalf_draw"),rr_scalar(s,WOTR_BOOL,1)}),cp_container(s,WOTR_DICT,{rr_string(s,"gandalf_draw"),rr_scalar(s,WOTR_BOOL,0)})})}));
}
void cau_finish(Snapshot& s,const Snapshot& tables,void* rng,int attacker_side,am_Builder& events,std::vector<uint64_t>& rows){
    cau_dedupe(s);tc_open_hand_limit_discard(s);
    auto no_combat=[&](){return ctx_none(s,obs_attr(s,s.root,"pending_combat"));};auto quiet=[&](){return no_combat()&&ctx_none(s,obs_attr(s,s.root,"pending_card_option",false))&&ctx_none(s,obs_attr(s,s.root,"pending_hunt"));};
    if(truth(s,obs_attr(s,s.root,"_palantir_draw_pending",false))&&quiet())cbe_palantir(s);
    if(no_combat()){su_cleanup(s,tables);cau_last_battle(s,tables);}
    if(quiet()&&truth(s,obs_attr(s,s.root,"_wwsat_owed",false)))cau_wwsat(s,rng);
    fr_open(s,tables,rng,events,rows);const auto fellowship=obs_attr(s,s.root,"fellowship");
    if(truth(s,obs_attr(s,fellowship,"companions"))&&ctx_none(s,obs_attr(s,s.root,"pending_guide_change",false))&&!hc_guide_member(s))ft_guide(s);
    hpr_return_mithril(s,rng);fm_finish(s,tables);wt_settle_attack(s);epa_maybe(s);cau_gandalf(s);
    if(no_combat())rr_set_attr(s,s.root,"_battle_resolved_attacker",attacker_side<0?rr_scalar(s,WOTR_NONE):mu_enum(s,"game_state.Side",attacker_side==0?"FREE_PEOPLES":"SHADOW"));
}
fm_Result apply_combat_round_choice(const Snapshot& source,const Snapshot& tables,const void* random_source,int64_t selection_index){
    require(selection_index>=0,"Combat choice index must be nonnegative");auto child=action_clone(source);auto& s=*child;cau_dedupe(s);rr_set_attr(s,s.root,"_casualty_operations",cp_container(s,WOTR_LIST,{}));
    const auto pc=obs_attr(s,s.root,"pending_combat"),effect=field(s,pc,"effect_choice");require(text_is(s,field(s,pc,"phase"),"combat_effect_choice")&&text_is(s,field(s,effect,"stage"),"round"),"Combat round action requires an observed-stage owner choice");
    const auto side=field(s,pc,"attacker_side");const int attacker=ctx_none(s,side)?-1:enum_is(s,side,"game_state.Side","FREE_PEOPLES")?0:1;
    rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));if(truth(s,obs_attr(s,s.root,"_declare_phase",false)))rr_set_attr(s,s.root,"_declare_phase",rr_scalar(s,WOTR_BOOL,0));rr_set_attr(s,s.root,"_last_round_phase_call",rr_scalar(s,WOTR_BOOL,1));
    am_Builder events;std::vector<uint64_t> rows;crr_Result owned;owned.state=std::move(child);owned.random.reset(wotr_random_clone(random_source));if(!owned.random)throw std::invalid_argument(wotr_random_last_error());auto resumed=combat_resume_settle_owned(std::move(owned),tables,selection_index,&events,&rows);fm_Result result;result.state=std::move(resumed.state);result.random=std::move(resumed.random);cau_finish(*result.state,tables,result.random.get(),attacker,events,rows);result.state=ac_compact(*result.state);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
