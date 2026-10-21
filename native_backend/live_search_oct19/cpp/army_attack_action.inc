// Complete ordinary funded attack: freeze compulsory guards before politics.
// Ordinary/Mouth die funding and validated free card grants share this attack.
fm_Result apply_army_attack(const Snapshot& source,const Snapshot& tables,const void* random_source,int side,int64_t index,std::string_view from,std::string_view to,bool mouth_muster=false,bool card_followup=false){
    require(side==0||side==1,"Army attack side must be FP or SH");
    crr_Result owned;owned.state=action_clone(source);owned.random.reset(wotr_random_clone(random_source));if(!owned.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*owned.state;
    cau_dedupe(s);nt_enforce_control(s,tables);
    require(!card_followup || (side==1&&index==-1&&!mouth_muster),"Invalid free card attack funding");
    if(card_followup)af_scope(s,tables);else quiet_action_gate(s,tables,false,false,false,false,false,false,false,false,false,true);
    ring_waiver(s,0);ring_waiver(s,1);
    const auto retained=obs_attr(s,s.root,"_ring_free_side",false);require(ctx_none(s,retained)||ctx_side(s,retained)==side,"Army attack belongs to retained ring actor");
    const auto army=mu_army(s,sanitize_name(from)),defender=mu_army(s,sanitize_name(to));require(army!=missing&&defender!=missing,"Army attack requires both armies");
    const auto formation=at_formation(s,army,side);require(!formation.units.empty(),"No attacking unit remains after compulsory guards");
    bool enemy=false;for(auto unit:child_edges(s,obs_attr(s,defender,"units")))enemy|=ctx_unit_side(s,unit)!=side;
    require(enemy,"Army attack requires an enemy defender");
    require(!mouth_muster || am_mouth_available(s,side),"Mouth army ability is unavailable");
    const auto die=card_followup?missing:am_die(s,side,index,formation.character,mouth_muster);
    require(!mouth_muster || enum_is(s,obs_attr(s,die,"current"),"game_state.ActionDieFace","MUSTER"),"Mouth stamp requires a physical Muster die");
    // Validate current reachability and siege direction before spending the die.
    auto legal=native_army_attack_for_die(s,tables,side,index,!card_followup&&enum_is(s,obs_attr(s,die,"current"),"game_state.ActionDieFace","CHARACTER"));bool found=false;
    for(auto row:child_edges(*legal,legal->root)){
        const auto parts=child_edges(*legal,row);const auto params=parts[2];
        found|=sanitize_name(text(*legal,field(*legal,params,"from")))==sanitize_name(from)&&sanitize_name(text(*legal,field(*legal,params,"to")))==sanitize_name(to);
    }
    require(found,"Army attack is not currently legal");
    rr_set_attr(s,s.root,"_casualty_operations",cp_container(s,WOTR_LIST,{}));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));if(!card_followup){rr_set_attr(s,die,"used",rr_scalar(s,WOTR_BOOL,1));rr_set_attr(s,s.root,"_battle_resolved_attacker",rr_scalar(s,WOTR_NONE));}else rr_set_attr(s,s.root,"pending_card_followup",rr_scalar(s,WOTR_NONE));
    if(mouth_muster)rr_set_attr(s,s.root,"sh_mouth_muster_used_this_turn",rr_scalar(s,WOTR_BOOL,1));
    const auto target=sanitize_name(to);const auto meta=ctx_get(tables,field(tables,tables.root,"board_regions"),target);
    const bool stronghold=truth(tables,ctx_get(tables,meta,"is_stronghold")),besieged=contains_string(s,obs_attr(s,s.root,"besieged_strongholds"),target);
    const bool sortie=besieged&&nt_controller(s,tables,target)==side;
    const auto pc=cp_container(s,WOTR_DICT,{rr_string(s,"attacker_region"),rr_string(s,from),rr_string(s,"defender_region"),rr_string(s,to),rr_string(s,"is_siege"),rr_scalar(s,WOTR_BOOL,besieged&&!sortie),rr_string(s,"stronghold_region"),rr_scalar(s,WOTR_BOOL,stronghold),rr_string(s,"phase"),rr_string(s,"attacker_rearguard"),rr_string(s,"deciding_side"),mu_enum(s,"game_state.Side",side==0?"FREE_PEOPLES":"SHADOW"),rr_string(s,"attacker_side"),mu_enum(s,"game_state.Side",side==0?"FREE_PEOPLES":"SHADOW"),rr_string(s,"defender_side"),mu_enum(s,"game_state.Side",side==0?"SHADOW":"FREE_PEOPLES"),rr_string(s,"rounds_fought"),rr_scalar(s,WOTR_INT,0)});
    rr_set_attr(s,s.root,"pending_combat",pc);cbb_freeze(s,pc);
    std::set<std::string> nations;for(auto unit:child_edges(s,obs_attr(s,defender,"units")))if(ctx_unit_side(s,unit)!=side)nations.emplace(ctx_nation(s,obs_attr(s,unit,"nation")));
    political_table_gate(s);
    for(const auto& nation:nations)wt_defender_advance(s,nation,target);
    am_Builder events;std::vector<uint64_t> rows;
    if(besieged&&!sortie)cp_put(s,pc,"phase",rr_string(s,"attacker_siege_decision"));
    else if(stronghold&&at_controller(s,tables,target)==1-side){cp_put(s,pc,"phase",rr_string(s,"defender_siege_choice"));cp_put(s,pc,"deciding_side",field(s,pc,"defender_side"));}
    else cbb_run(owned,tables,-1,events,rows);
    cau_finish(*owned.state,tables,owned.random.get(),card_followup?ctx_side(source,obs_attr(source,source.root,"_battle_resolved_attacker",false)):-1,events,rows);fm_Result result;result.state=ac_compact(*owned.state);result.random=std::move(owned.random);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
