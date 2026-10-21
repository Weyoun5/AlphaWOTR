// Execute a resident descriptor without constructing a Python Action/params.
// This dispatches supported families; it is not a complete legal-menu promise.
void ad_keys(const Snapshot& a,uint64_t params,std::initializer_list<std::string_view> allowed){
    const auto keys=child_edges(a,params);
    for(size_t i=0;i<keys.size();i+=2){
        const auto name=text(a,keys[i]);
        require(std::find(allowed.begin(),allowed.end(),name)!=allowed.end(),"native descriptor has unsupported parameters");
    }
}
int64_t ad_int(const Snapshot& a,uint64_t params,const char* key,int64_t fallback=-1){
    const auto value=field(a,params,key,false);
    return value==missing?fallback:typed(a,value,WOTR_INT,"native descriptor count/index must be integer").integer;
}
std::string ad_text(const Snapshot& a,uint64_t params,const char* key){return std::string(text(a,field(a,params,key)));}
std::unique_ptr<Snapshot> apply_power_payment_pair(const Snapshot&,const Snapshot&,int,const std::string&,const std::string&);   // Package W2 (P10): event_power.inc
void ad_side(const Snapshot& a,uint64_t params,int side){
    const auto value=field(a,params,"side",false);
    if(value!=missing)require(text_is(a,value,side==0?"fp":"sh"),"native descriptor belongs to a different actor");
}
// package R (engine_alone): the frozen card_option executor (r4 ai.py 11967-12046) resolves a NON-direct optioned play through
// event_effects.play_event_card, which answers {"success": False, "message": "Card not in hand"} WITHOUT raising when the card is
// no longer in the owner's hand (event_effects.py 1003-1005; in the search the opponent's hand is the per-simulation overlay
// sample, ai.py 15012-15070 = search_overlay.inc so_patch): the pending clears, no effect, no discard, the spent die stays spent,
// _grant_event_play_draw (ai.py 10144-10195) then schedules the same-deck grant exactly as after a completed play (the card IS out
// of the hand), and _execute_action's common tail (9726-9860) runs.  The frozen card_option MENU (7764-7770 / 13503-13509) reads
// no hand at all.  Both dispatchers route such a descriptor here, after validating it against the complete menu exactly as every
// card branch validates it against the card's own menu; sentinel / direct / casualty / payment pendings never match.
void cau_finish(Snapshot& s,const Snapshot& tables,void* rng,int attacker_side,am_Builder& events,std::vector<uint64_t>& rows);
void cau_dedupe(Snapshot& s);void cau_gandalf(Snapshot& s);void su_cleanup(Snapshot& s,const Snapshot& t);
std::unique_ptr<Snapshot> fm_native_full_actions_impl(const Snapshot& s,const Snapshot& t,int32_t side);
bool ef_pending_card_absent(const Snapshot& s,const Snapshot& t){
    const auto p=obs_attr(s,s.root,"pending_card_option",false);if(ctx_none(s,p)||s.nodes[p].tag!=WOTR_DICT)return false;
    for(const char* key:{"direct","card_casualty","gandalf_draw","table_kill_payment","palantir_draw","hunt_reduction","foul_guide","bof_separation"})if(truth(s,field(s,p,key,false)))return false;
    if(!ctx_none(s,field(s,p,"resume_action",false)))return false;
    const auto deck=field(s,p,"deck",false),id=field(s,p,"card_id",false),owner=field(s,p,"side",false);
    if(deck==missing||s.nodes[deck].tag!=WOTR_STRING||id==missing||s.nodes[id].tag!=WOTR_STRING||owner==missing||s.nodes[owner].tag!=WOTR_STRING)return false;
    if(ev_card(t,text(s,id))==missing)return false;
    return !contains_string(s,tc_hand(s,text_is(s,owner,"fp")?0:1),text(s,id));
}
void ef_absent_pending_inplace(Snapshot& s,const Snapshot& t){
    const auto p=obs_attr(s,s.root,"pending_card_option");
    const std::string id(text(s,field(s,p,"card_id"))),deck(text(s,field(s,p,"deck")));const int owner=text_is(s,field(s,p,"side"),"fp")?0:1;
    const auto face_node=field(s,p,"die_face",false);const std::string face(face_node==missing||s.nodes[face_node].tag!=WOTR_STRING?std::string():std::string(text(s,face_node)));
    // _execute_action 9734-9773: the operations list, the army dedupe, the ring-free marker; the inner branch clears the pending (11827)
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));cau_dedupe(s);rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));
    // _grant_event_play_draw with the pending's recorded deck / funding face: FP queues the Gandalf grant on an EVENT / WILL play
    // (the card is not in the hand, 10164-10175; the queue write is er_commit's), the Shadow runs the Palantir rule (10178-10195)
    if(owner==0){if((face=="EVENT"||face=="WILL_OF_WEST")&&(deck=="character"||deck=="strategy"))rr_set_attr(s,s.root,"_gandalf_draw_pending",ev_dict(s,{{"source_card_id",rr_string(s,id)},{"deck",rr_string(s,deck)},{"die_face",rr_string(s,face)}}));}
    else epa_after_event(s,1,id,deck,face);
}
std::unique_ptr<Snapshot> ef_absent_pending_apply(const Snapshot& source,const Snapshot& t){
    auto child=action_clone(source);auto& s=*child;ef_absent_pending_inplace(s,t);
    // the no-RNG card_option executors' common tail (er_commit / esr_commit without the hand removal): hand limit, siege cleanup, the queued grant
    tc_open_hand_limit_discard(s);if(ctx_none(s,obs_attr(s,s.root,"pending_combat")))su_cleanup(s,t);cau_gandalf(s);return child;
}
fm_Result ef_absent_pending_apply_random(const Snapshot& source,const Snapshot& t,const void* rng){
    const auto marker=obs_attr(source,source.root,"_battle_resolved_attacker",false);const int oldside=ctx_none(source,marker)?-1:ctx_side(source,marker);
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;
    ef_absent_pending_inplace(s,t);
    am_Builder events;std::vector<uint64_t> rows;cau_finish(s,t,result.random.get(),oldside,events,rows);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
bool edf_pending(const Snapshot&);
std::unique_ptr<Snapshot> native_denethor_choices(const Snapshot&,const Snapshot&,int);
std::unique_ptr<Snapshot> apply_native_descriptor_inner(const Snapshot& s,const Snapshot& t,const Snapshot& actions,
                                                 int32_t side,int64_t row_index){
    require(side==0 || side==1,"native descriptor actor must be FP or SH");
    const auto& list=typed(actions,actions.root,WOTR_LIST,"native descriptor menu must be a list");
    require(row_index>=0 && static_cast<uint64_t>(row_index)<list.count,"native descriptor index out of range");
    const auto row=actions.edges[list.offset+row_index];
    const auto& tuple=typed(actions,row,WOTR_TUPLE,"native action descriptor must be tuple");
    require(tuple.count==3,"native action descriptor must have three fields");
    const auto index=typed(actions,actions.edges[tuple.offset],WOTR_INT,"native action die index must be integer").integer;
    const auto kind=text(actions,actions.edges[tuple.offset+1]);const auto p=actions.edges[tuple.offset+2];
    typed(actions,p,WOTR_DICT,"native action params must be dictionary");ad_side(actions,p,side);
    if(kind=="advance_political"){
        ad_keys(actions,p,{"nation","companion_ability"});
        const auto nation=enum_member(actions,field(actions,p,"nation"),"game_state.Nation");
        const auto companion=field(actions,p,"companion_ability",false);
        if(companion!=missing)return apply_companion_political_action(s,t,side,index,nation,text(actions,companion));
        return apply_political_action(s,t,side,index,nation);
    }
    if(kind=="muster_recruit" || kind=="muster_elite"){
        ad_keys(actions,p,{"nation","region","type","saruman","placements"});
        const auto nation=enum_member(actions,field(actions,p,"nation"),"game_state.Nation");
        const auto special=field(actions,p,"saruman",false);
        if(special!=missing){
            const auto mode=text(actions,special);
            require(nation=="ISENGARD" && ad_text(actions,p,"region")=="orthanc","invalid Saruman descriptor location/nation");
            require((mode=="settlements" && kind=="muster_recruit") || (mode=="swap2" && kind=="muster_elite"),"Saruman descriptor type mismatch");
            if(mode=="settlements"){
                require(ad_text(actions,p,"type")=="regular","Saruman settlements recruit regulars");
                const auto places=field(actions,p,"placements");const auto& values=typed(actions,places,WOTR_LIST,"Saruman placements must be list");
                require(values.count==3 && text_is(actions,actions.edges[values.offset],"orthanc") &&
                    text_is(actions,actions.edges[values.offset+1],"north dunland") && text_is(actions,actions.edges[values.offset+2],"south dunland"),"unsupported Saruman placement descriptor");
            }
            return apply_saruman_muster_action(s,t,side,index,mode);
        }
        require(field(actions,p,"placements",false)==missing,"ordinary muster cannot carry placement override");
        return apply_muster_action(s,t,side,index,nation,ad_text(actions,p,"region"),kind=="muster_elite"?"elite":ad_text(actions,p,"type"));
    }
    if(!ctx_none(s,obs_attr(s,s.root,"pending_card_followup",false))&&(kind=="army_move"||kind=="army_attack"||kind=="followup_done"))throw std::invalid_argument("Card army follow-up requires owned RNG");
    if(kind=="army_move"){
        ad_keys(actions,p,{"from","to","side","n_regular","n_elite","n_leaders","n_nazgul","take_leaders","mouth_muster"});
        ArmyMoveRequest req;req.side=side;req.die_index=index;req.from=ad_text(actions,p,"from");req.to=ad_text(actions,p,"to");
        req.regular=ad_int(actions,p,"n_regular");req.elite=ad_int(actions,p,"n_elite");
        req.leaders=ad_int(actions,p,"n_leaders");req.nazgul=ad_int(actions,p,"n_nazgul");
        const auto take=field(actions,p,"take_leaders",false);
        if(take!=missing)req.take_leaders=typed(actions,take,WOTR_BOOL,"take_leaders must be Boolean").integer!=0;
        const auto mouth=field(actions,p,"mouth_muster",false);
        if(mouth!=missing)req.mouth_muster=typed(actions,mouth,WOTR_BOOL,"mouth_muster must be Boolean").integer!=0;
        return apply_army_move(s,t,req);
    }
    if(kind=="elven_ring"){
        ad_keys(actions,p,{"target_die_idx","new_face","old_face","side"});
        const auto value=ad_text(actions,p,"new_face");int face=-1;
        for(size_t i=0;i<rr_face_values.size();++i)if(value==rr_face_values[i])face=static_cast<int>(i);
        require(face>=0,"native ring descriptor has invalid result");
        return apply_elven_ring_action(s,t,side,ad_int(actions,p,"target_die_idx"),rr_face_names[face]);
    }
    if(kind=="eye_allocate"){
        ad_keys(actions,p,{"count"});require(side==1,"Eye allocation belongs to Shadow");
        return apply_eye_allocation(s,t,ad_int(actions,p,"count",0));
    }
    if(kind=="discard_card"){
        ad_keys(actions,p,{"card_id","side"});return apply_discard_card(s,t,side,ad_text(actions,p,"card_id"));
    }
    if(kind=="draw_card"){
        ad_keys(actions,p,{"deck","side"});return apply_draw_card_action(s,t,side,index,ad_text(actions,p,"deck"));
    }
    if(kind=="play_event"){
        const auto id=ad_text(actions,p,"card_id");const auto mouth=field(actions,p,"mouth_muster",false);
        const bool mode=mouth!=missing && typed(actions,mouth,WOTR_BOOL,"mouth_muster must be Boolean").integer!=0;
        require(!esm_supported(t,ev_card(t,id))&&!emk_supported(t,ev_card(t,id)),"Misty Mountains and Many Kings events require an explicit owned RNG");
        require(!eaw_supported(t,ev_card(t,id)),"Another Way event requires an explicit owned RNG");
        require(!esw_supported(t,ev_card(t,id)),"Swifter event requires an explicit owned RNG");
        require(!eck_supported(t,ev_card(t,id)),"Challenge event requires an explicit owned RNG");
        require(!eea_supported(t,ev_card(t,id)),"Ents play requires an explicit owned RNG");
        require(!esc_supported(t,ev_card(t,id)),"Stormcrow event requires an explicit owned RNG");
        require(!etsm_supported(t,ev_card(t,id)),"Shadow is Moving event requires an explicit owned RNG");
        require(!etd_supported(t,ev_card(t,id)),"Through Day event requires an explicit owned RNG");
        require(!eco_supported(t,ev_card(t,id)),"Corsairs event requires owned RNG");
        require(!ehuf_supported(t,ev_card(t,id)),"Help Unlooked For event requires an explicit owned RNG");
        require(!egr_supported(t,ev_card(t,id)),"Grond event requires an explicit owned RNG");
        require(!eal_supported(t,ev_card(t,id)),"Alone event requires an explicit owned RNG");
        require(!etb_supported(t,ev_card(t,id)),"Back Again event requires an explicit owned RNG");
        require(!ehe_supported(t,ev_card(t,id)),"Horde event requires an explicit owned RNG");
        require(!erg_supported(t,ev_card(t,id)),"Rage event requires an explicit owned RNG");
        require(!euh_supported(t,ev_card(t,id)),"Uruk-hai event requires an explicit owned RNG");
        require(!epw_supported(t,ev_card(t,id)),"Woses requires an explicit owned RNG");
        require(!edm_supported(t,ev_card(t,id)),"Dead Men event requires owned RNG");
        require(!erv_supported(t,ev_card(t,id)),"Return to Valinor requires owned RNG");
        require(!eeg_supported(t,ev_card(t,id)),"Eagles event requires owned RNG");
        require(!eds_supported(t,ev_card(t,id)),"Dreadful event requires owned RNG");
        require(!esp_supported(t,ev_card(t,id)),"Spirit event requires owned RNG");
        require(!egc_supported(t,ev_card(t,id)),"Grey Company event requires owned RNG");
        require(!ehs_supported(t,ev_card(t,id)),"House of the Stewards event requires owned RNG");
        require(!ebc_supported(t,ev_card(t,id)),"Black Captain event requires owned RNG");
        require(!erw_supported(t,ev_card(t,id)),"Ringwraiths event requires owned RNG");
        require(!egw_supported(t,ev_card(t,id)),"Gwaihir requires an explicit owned RNG");
        if(epr_supported(t,ev_card(t,id))){ad_keys(actions,p,{"card_id","side"});return apply_pits_event(s,t,side,index,id);}
        require(!eff_supported(t,ev_card(t,id)),"Faramir requires an explicit owned RNG");
        if(ept_supported(t,ev_card(t,id))){ad_keys(actions,p,{"card_id","side"});return apply_power_event(s,t,side,index,id);}
        if(ert_kind(t,ev_card(t,id))){ad_keys(actions,p,{"card_id","side"});return apply_target_recruit_event(s,t,side,index,id);}
        if(esr_kind(t,ev_card(t,id))){ad_keys(actions,p,{"card_id","side"});return apply_shadow_recruit_event(s,t,side,index,id);}
        if(eho_supported(t,ev_card(t,id))){ad_keys(actions,p,{"card_id","side","mouth_muster"});return apply_half_orcs_event(s,t,side,index,id,mode);}
        if(esg_supported(t,ev_card(t,id))){ad_keys(actions,p,{"card_id","side","mouth_muster"});return apply_shadows_gather_event(s,t,side,index,id,mode);}
        if(ehi_supported(t,ev_card(t,id))){ad_keys(actions,p,{"card_id","side","mouth_muster"});return apply_hill_trolls_event(s,t,side,index,id,mode);}
        if(er_supported(t,ev_card(t,id))){ad_keys(actions,p,{"card_id","side"});return apply_recruit_event_action(s,t,side,index,id);}
        ad_keys(actions,p,{"card_id","side","mouth_muster"});
        if(ed_supported(t,ev_card(t,id)))return apply_dice_event_action(s,t,side,index,id);
        return apply_simple_event_action(s,t,side,index,id,mode);
    }
    // package R: a NON-direct optioned play whose card left its owner's hand resolves as the frozen executor does (no effect; see the helpers above)
    if(kind=="card_option"&&ef_pending_card_absent(s,t)){
        require(index==-1,"Card option continuation spends no die");ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});
        auto menu=fm_native_full_actions_impl(s,t,side);bool legal=false;
        for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
        require(legal,"Card option descriptor is stale");return ef_absent_pending_apply(s,t);
    }
    if(kind=="card_option"&&esra_pending(s)){
        require(index==-1,"Allocation continuation spends no second die");ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});auto menu=native_shadow_allocation_choices(s,t,side);bool legal=false;
        for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
        require(legal,"Allocation descriptor is not current");return apply_shadow_allocation_choice(s,t,side,ad_int(actions,p,"idx"));
    }
    if(kind=="card_option")require(!edf_pending(s),"Denethor choice requires an explicit owned RNG");
    if(kind=="card_option")require(!esm_pending(s)&&!emk_pending(s),"Misty Mountains and Many Kings choices require an explicit owned RNG");
    if(kind=="card_option")require(!eaw_pending(s),"Another Way choice requires an explicit owned RNG");
    if(kind=="card_option")require(!esw_pending(s),"Swifter choice requires an explicit owned RNG");
    if(kind=="card_option")require(!eea_pending(s),"Ents choice requires an explicit owned RNG");
    if(kind=="card_option")require(!esc_pending(s),"Stormcrow choice requires an explicit owned RNG");
    if(kind=="card_option")require(!etsm_pending(s),"Shadow is Moving choice requires an explicit owned RNG");
    if(kind=="card_option")require(!etd_pending(s),"Through Day choice requires an explicit owned RNG");
    if(kind=="card_option")require(!eco_pending(s),"Corsairs choice requires owned RNG");
    if(kind=="card_option")require(!ehuf_pending(s),"Help Unlooked For choice requires an explicit owned RNG");
    if(kind=="card_option")require(!egr_pending(s),"Grond choice requires an explicit owned RNG");
    if(kind=="card_option")require(!eal_pending(s),"Alone choice requires an explicit owned RNG");
    if(kind=="card_option")require(!etb_pending(s),"Back Again choice requires an explicit owned RNG");
    if(kind=="card_option")require(!ehe_pending(s),"Horde choice requires an explicit owned RNG");
    if(kind=="card_option")require(!erg_pending(s),"Rage choices require an explicit owned RNG");
    if(kind=="card_option")require(!edm_pending(s),"Dead Men choice requires owned RNG");
    if(kind=="card_option")require(!eeg_pending(s),"Eagles choice requires owned RNG");
    if(kind=="card_option")require(!eds_pending(s),"Dreadful target requires owned RNG");
    if(kind=="card_option")require(!esp_pending(s),"Spirit target requires owned RNG");
    if(kind=="card_option")require(!egc_pending(s),"Grey Company choice requires owned RNG");
    if(kind=="card_option")require(!ehs_pending(s),"House of the Stewards choice requires owned RNG");
    if(kind=="card_option")require(!ebc_pending(s)&&!erw_pending(s),"Card-funded Nazgul choices require owned RNG");
    if(kind=="card_option"&&euh_pending(s))throw std::invalid_argument("Uruk-hai choice requires an explicit owned RNG");
    if(kind=="card_option")require(!ecc_pending(s) && !eff_pending(s) && !egw_pending(s) && !epw_pending(s),"Faramir, Gwaihir and card casualty choices require explicit owned RNG");
    if(kind=="card_option" && text_is(actions,field(actions,p,"card_name",false),"Pits of Mordor")){
        require(index==-1,"Pits continuation spends no second die");ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});auto menu=native_pits_choice_actions(s,t,side);bool legal=false;
        for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
        require(legal,"Pits descriptor is not current");return apply_pits_choice(s,t,side,ad_int(actions,p,"idx"));
    }
    if(kind=="discard_table_card" && text_is(actions,field(actions,p,"card_name",false),"Denethor's Folly")){
        ad_keys(actions,p,{"card_id","card_name","payment"});return apply_denethor_kill(s,t,side,index,ad_text(actions,p,"card_id"),ad_text(actions,p,"payment"));
    }
    if(kind=="discard_table_card" && text_is(actions,field(actions,p,"card_name",false),"The Palant\xc3\xadr of Orthanc")){
        ad_keys(actions,p,{"card_id","card_name","payment"});return apply_palantir_kill(s,t,side,index,ad_text(actions,p,"card_id"),ad_text(actions,p,"payment"));
    }
    if(kind=="discard_table_card" && (text_is(actions,field(actions,p,"card_name",false),"A Power too Great") || text_is(actions,field(actions,p,"card_name",false),"The Power of Tom Bombadil"))){
        require(side==1,"Power removal belongs to Shadow");ad_keys(actions,p,{"card_id","card_name"});
        auto menu=native_power_kill_for_die(s,t,side,index);bool legal=false;
        for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
        require(legal,"Power removal descriptor is not current");return apply_power_kill(s,t,side,index,ad_text(actions,p,"card_id"));
    }
    if(kind=="card_option" && (text_is(actions,field(actions,p,"card_name",false),"A Power too Great") || text_is(actions,field(actions,p,"card_name",false),"The Power of Tom Bombadil"))){
        require(side==1 && index==-1,"Power payment is a free Shadow card-pair choice");ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});
        // Package W2 (P10): the frozen executor pays the pair named by `choices` (ai.py 11855-11861 -> pay_table_card_kill) and reads neither
        // the row index nor the menu: a learned-draw payload row keeps the idx of the world that created it (action_policy_key drops idx),
        // while the pairs are recomputed from the sampled hand - the menu-membership require refused rows the frozen engine accepts.
        const auto choices=field(actions,p,"choices");return apply_power_payment_pair(s,t,side,ad_text(actions,choices,"army_card"),ad_text(actions,choices,"character_card"));
    }
    if(kind=="card_option" && (text_is(actions,field(actions,p,"card_name",false),"Cirdan's Ships") || text_is(actions,field(actions,p,"card_name",false),"Riders of Th\xC3\xA9oden") || text_is(actions,field(actions,p,"card_name",false),"\xC3\x89omer son of \xC3\x89omund"))){
        require(index==-1,"Targeted recruitment continuation spends no second die");ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});
        auto menu=native_target_recruit_choice_actions(s,t,side);bool legal=false;
        for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
        require(legal,"Targeted recruitment descriptor is not current");return apply_target_recruit_choice(s,t,side,ad_int(actions,p,"idx"));
    }
    if(kind=="card_option" && text_is(actions,field(actions,p,"card_name",false),"Shadows Gather")){
        require(index==-1,"Shadows Gather continuation spends no second die");ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});
        auto menu=native_shadows_gather_choice_actions(s,t,side);bool legal=false;
        for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
        require(legal,"Shadows Gather descriptor is not current");return apply_shadows_gather_choice(s,t,side,ad_int(actions,p,"idx"));
    }
    if(kind=="card_option" && (text_is(actions,field(actions,p,"card_name",false),"Half-orcs and Goblin-men") || text_is(actions,field(actions,p,"card_name",false),"Olog-hai"))){
        require(index==-1,"Half-orcs continuation spends no second die");ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});
        auto menu=native_half_orcs_choice_actions(s,t,side);bool legal=false;
        for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
        require(legal,"Half-orcs descriptor is not current");return apply_half_orcs_choice(s,t,side,ad_int(actions,p,"idx"));
    }
    if(kind=="card_option" && text_is(actions,field(actions,p,"card_name",false),"Hill-trolls")){
        require(index==-1,"Hill-trolls continuation spends no second die");ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});
        auto menu=native_hill_trolls_choice_actions(s,t,side);bool legal=false;
        for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
        require(legal,"Hill-trolls descriptor is not current");return apply_hill_trolls_choice(s,t,side,ad_int(actions,p,"idx"));
    }
    if(kind=="card_option" && text_is(actions,field(actions,p,"card_name",false),"The Lidless Eye")){
        ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});
        auto current=native_dice_event_choice_actions(s,side);bool legal=false;
        for(auto candidate:child_edges(*current,current->root))legal|=ap_equal(ap_freeze(*current,candidate),ap_freeze(actions,row));
        require(legal,"Lidless descriptor is not a current legal choice");return apply_dice_event_choice(s,t,side,ad_int(actions,p,"idx"));
    }
    if(kind=="card_option" && text_is(actions,field(actions,p,"card_name",false),"Wisdom of Elrond")){
        require(index==-1,"Wisdom continuation spends no second die");ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});
        auto current=native_misc_event_choice_actions(s,side);bool legal=false;
        for(auto candidate:child_edges(*current,current->root))legal|=ap_equal(ap_freeze(*current,candidate),ap_freeze(actions,row));
        require(legal,"Wisdom descriptor is not a current legal choice");return apply_misc_event_choice(s,t,side,ad_int(actions,p,"idx"));
    }
    if(kind=="card_option" && (text_is(actions,field(actions,p,"card_name",false),"Guards of the Citadel")||text_is(actions,field(actions,p,"card_id",false),"__overstack__"))){
        require(index==-1,"Recruitment continuation spends no second die");ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});
        const bool over=text_is(actions,field(actions,p,"card_id",false),"__overstack__");
        auto menu=over?native_overstack_choice_actions(s,side):native_recruit_event_choice_actions(s,t,side);bool legal=false;
        for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
        require(legal,"Recruitment descriptor is not current");return over?apply_overstack_choice(s,t,side,ad_int(actions,p,"idx")):apply_recruit_event_choice(s,t,side,ad_int(actions,p,"idx"));
    }
    if(kind=="card_option"&&text_is(actions,field(actions,p,"card_id",false),"__gandalf_draw__")){
        ad_keys(actions,p,{"idx","card_id","card_name","side","choices","source_card_id","deck"});
        auto current=native_gandalf_draw_actions(s,side);bool legal=false;
        for(auto candidate:child_edges(*current,current->root))legal|=ap_equal(ap_freeze(*current,candidate),ap_freeze(actions,row));
        require(legal,"Gandalf descriptor is not a current legal choice");const auto choice=field(actions,p,"choices");
        return apply_gandalf_draw_action(s,t,side,typed(actions,field(actions,choice,"gandalf_draw"),WOTR_BOOL,"Gandalf decision must be bool").integer!=0);
    }
    if(kind=="mouth_move"){
        ad_keys(actions,p,{"from","to","side"});return apply_mouth_move_action(s,t,side,index,ad_text(actions,p,"from"),ad_text(actions,p,"to"));
    }
    if(kind=="nazgul_move"){
        ad_keys(actions,p,{"from","to","count","include_wk"});const auto witch=field(actions,p,"include_wk",false);
        return apply_nazgul_move_action(s,t,side,index,ad_text(actions,p,"from"),ad_text(actions,p,"to"),ad_int(actions,p,"count",-1),witch!=missing && typed(actions,witch,WOTR_BOOL,"Witch-king flag must be bool").integer!=0);
    }
    if(kind=="card_option" && text_is(actions,field(actions,p,"card_id",false),"__nazgul_moves__")){
        require(side==1 && index==-1,"Nazgul continuation belongs to Shadow and spends no die");ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});
        require(text_is(actions,field(actions,p,"card_name"),"__nazgul_moves__"),"Nazgul descriptor card name mismatch");
        const auto pending=nz_pending(s),options=field(s,pending,"options");const auto chosen=ad_int(actions,p,"idx");
        require(chosen>=0 && static_cast<uint64_t>(chosen)<s.nodes[options].count,"Nazgul descriptor choice index invalid");
        require(ap_equal(ap_freeze(actions,field(actions,p,"choices")),ap_freeze(s,s.edges[s.nodes[options].offset+chosen])),"Nazgul descriptor choice is stale");
        return apply_nazgul_chain_action(s,t,side,chosen);
    }
    if(kind=="transform_gandalf" || kind=="crown_aragorn"){
        if(kind=="transform_gandalf"){ad_keys(actions,p,{"region"});return apply_will_special_action(s,t,side,index,true,ad_text(actions,p,"region"));}
        ad_keys(actions,p,{});return apply_will_special_action(s,t,side,index,false,"");
    }
    if(kind=="muster_minion"){
        ad_keys(actions,p,{"minion","region"});return apply_minion_muster_action(s,t,side,index,ad_text(actions,p,"minion"),ad_text(actions,p,"region"));
    }
    if(kind=="companion_separate"||kind=="companion_move"){
        require(side==0,"Companion actions belong to Free Peoples");std::vector<std::string> names;const auto list=field(actions,p,"companions",false);
        if(list!=missing){typed(actions,list,WOTR_LIST,"Companion names must be list");for(auto name:child_edges(actions,list))names.emplace_back(text(actions,name));}
        if(kind=="companion_separate"){
            ad_keys(actions,p,{"companion","companions","to_region"});const auto one=field(actions,p,"companion",false);if(one!=missing){require(names.empty(),"Ambiguous companion selection");names.emplace_back(text(actions,one));}
            return apply_companion_separation(s,t,index,names,ad_text(actions,p,"to_region"));
        }
        ad_keys(actions,p,{"from","to","companions"});return apply_companion_move(s,t,index,ad_text(actions,p,"from"),ad_text(actions,p,"to"),names);
    }
    if(kind=="card_option" && text_is(actions,field(actions,p,"card_id",false),"__companion_moves__")){
        require(side==0,"Companion continuation belongs to Free Peoples");ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});
        cm_validate_chain(s);const auto pending=obs_attr(s,s.root,"pending_card_option"),options=field(s,pending,"options");const auto chosen=ad_int(actions,p,"idx");
        require(chosen>=0&&static_cast<uint64_t>(chosen)<s.nodes[options].count,"Companion continuation index invalid");
        require(ap_equal(ap_freeze(actions,field(actions,p,"choices")),ap_freeze(s,s.edges[s.nodes[options].offset+chosen])),"Companion descriptor is stale");
        return apply_companion_move_option(s,t,chosen);
    }
    if(kind=="fellowship_hide"){
        ad_keys(actions,p,{"strider_ability"});require(side==0,"Fellowship hide belongs to Free Peoples");
        const auto ability=field(actions,p,"strider_ability",false);
        return apply_fellowship_hide(s,t,index,ability!=missing&&truth(actions,ability));
    }
    if(kind=="fellowship_declare"){
        ad_keys(actions,p,{"region"});require(side==0,"Fellowship declaration belongs to Free Peoples");
        return apply_fellowship_declare(s,t,ad_text(actions,p,"region"));
    }
    if(kind=="choose_guide"){
        ad_keys(actions,p,{"companion","side"});require(side==0,"Guide choice belongs to Free Peoples");
        return apply_guide_choice(s,t,ad_text(actions,p,"companion"));
    }
    if(kind=="skip"){
    // Rulebook p21: discard the named unused non-Eye die without effect.
    ad_keys(actions,p,{"face","side"});ad_side(actions,p,side);
    const std::string face(ad_text(actions,p,"face"));require(rr_face_index(face)!=5,"die discard requires a non-Eye face");
    quiet_stable_action_gate(s,t);ring_waiver(s,0);ring_waiver(s,1);
    const auto retained=obs_attr(s,s.root,"_ring_free_side",false);
    require(ctx_none(s,retained) || ctx_side(s,retained)==side,"die kill belongs to retained Ring actor");
    const auto dice=child_edges(s,pa_pool(s,side));
    require(index>=0 && static_cast<uint64_t>(index)<dice.size(),"die kill index is out of range");
    const auto die=dice[index];require(object_type(s,die)=="game_state.ActionDie","die kill requires ActionDie");
    require(!truth(s,obs_attr(s,die,"used")),"die kill die is already used");
    require(enum_is(s,obs_attr(s,die,"current"),"game_state.ActionDieFace",face),"die discard requires the named unused face");
    auto child=action_clone(s);auto& out=*child;
    rr_set_attr(out,out.root,"_casualty_operations",rr_scalar(out,WOTR_LIST));
    rr_set_attr(out,out.root,"_ring_free_side",rr_scalar(out,WOTR_NONE));
    rr_set_attr(out,child_edges(out,pa_pool(out,side))[index],"used",rr_scalar(out,WOTR_BOOL,1));
    rr_set_attr(out,out.root,"_battle_resolved_attacker",rr_scalar(out,WOTR_NONE));
    return child;
    }
    if(kind=="pass"){
        ad_keys(actions,p,{});
        if(truth(s,obs_attr(s,s.root,"_declare_phase",false))){
            require(side==0,"Fellowship declaration belongs to Free Peoples");return apply_fellowship_declare(s,t,"",true);
        }
        if(!ctx_none(s,obs_attr(s,s.root,"pending_muster_die")))return apply_muster_action(s,t,side,0,"","","pass");
        if(!ctx_none(s,obs_attr(s,s.root,"pending_army_move_die"))){ArmyMoveRequest req;req.side=side;req.die_index=0;req.pass=true;return apply_army_move(s,t,req);}
        quiet_stable_action_gate(s,t);require(index==-1,"ordinary pass cannot spend an indexed die");
        std::array<size_t,2> counts{};
        for(int owner=0;owner<2;++owner)for(auto die:child_edges(s,pa_pool(s,owner)))
            if(!truth(s,obs_attr(s,die,"used")) && !ctx_none(s,obs_attr(s,die,"current")))++counts[owner];
        require(counts[side]>0 && counts[side]<counts[1-side],"ordinary pass requires fewer dice than opponent");
        auto child=action_clone(s);
        rr_set_attr(*child,child->root,"_casualty_operations",rr_scalar(*child,WOTR_LIST));
        rr_set_attr(*child,child->root,"_ring_free_side",rr_scalar(*child,WOTR_NONE));
        rr_set_attr(*child,child->root,"_battle_resolved_attacker",rr_scalar(*child,WOTR_NONE));
        return child;
    }
    throw std::invalid_argument("native descriptor action family is not implemented");
}

// Public descriptor execution has the same stale-empty-discard context as its
// full-menu query. Primitive executors retain their stricter preconditions.
bool fm_stale_discard_cleanup_due(const Snapshot& s,const Snapshot& t,int side);
std::unique_ptr<Snapshot> fm_prepare_menu_context(const Snapshot& source,const Snapshot& t,int side);
// Source public-action persistent upkeep; never run during an unresolved combat.
std::unique_ptr<Snapshot> apply_native_descriptor(const Snapshot& s,const Snapshot& t,const Snapshot& actions,
                                                 int32_t side,int64_t row_index){
    if(!ctx_none(s,obs_attr(s,s.root,"pending_discard",false)) && fm_stale_discard_cleanup_due(s,t,side)){
        auto normalized=fm_prepare_menu_context(s,t,side);
        return apply_native_descriptor(*normalized,t,actions,side,row_index);
    }
    auto result=apply_native_descriptor_inner(s,t,actions,side,row_index);
    auto& child=*result;
    if(ctx_none(child,obs_attr(child,child.root,"pending_combat",false)) && cau_persistent(child,"LAST_BATTLE")!=missing)cau_last_battle(child,t);
    return result;
}
