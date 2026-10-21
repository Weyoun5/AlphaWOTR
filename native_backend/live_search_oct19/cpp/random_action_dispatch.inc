// Resident action descriptors with owned RNG. No Python Action/state decoding.
fm_Result apply_native_random_descriptor_inner(const Snapshot& s,const Snapshot& t,const Snapshot& actions,const void* rng,int32_t side,int64_t row_index){
    require(side==0||side==1,"Native action actor must be FP or SH");const auto& list=typed(actions,actions.root,WOTR_LIST,"Native menu must be a list");
    require(row_index>=0&&static_cast<uint64_t>(row_index)<list.count,"Native action row out of range");
    const auto row=actions.edges[list.offset+row_index];const auto& tuple=typed(actions,row,WOTR_TUPLE,"Native action must be tuple");require(tuple.count==3,"Native action needs three fields");
    const auto index=typed(actions,actions.edges[tuple.offset],WOTR_INT,"Action die index must be integer").integer;
    const auto kind=text(actions,actions.edges[tuple.offset+1]);const auto p=actions.edges[tuple.offset+2];typed(actions,p,WOTR_DICT,"Action parameters must be dictionary");
    // Breaking and Stormcrow record the Shadow card owner while FP owns a choice.
    // That branch validates both the actual actor and the complete current row.
    if(!(kind=="card_option"&&(ehd_pending(s)||el_pending(s)||eeg_destination(s)||edm_cross_side(s)||esc_pending(s))))ad_side(actions,p,side);
    const auto combat=obs_attr(s,s.root,"pending_combat",false);
    if(!ctx_none(s,combat))return apply_native_combat_descriptor(s,t,actions,rng,side,row);
    if(kind=="use_token"||kind=="dwarven_ring"){
        require(index==-1,"Bid compensation is a free action");auto menu=kind=="use_token"?native_token_actions(s,t,side):native_dwarven_ring_actions(s,side);bool legal=false;
        for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
        require(legal,"Bid descriptor is stale or illegal");
        if(kind=="dwarven_ring")return apply_dwarven_ring_action(s,t,rng,side,ad_int(actions,p,"die_idx"));
        const auto mode=ad_text(actions,p,"token_type");const std::string value=mode=="muster"?std::string(enum_member(actions,field(actions,p,"nation"),"game_state.Nation")):ad_text(actions,p,"deck");
        return apply_token_action(s,t,rng,side,mode.c_str(),value.c_str());
    }
    if(!ctx_none(s,obs_attr(s,s.root,"pending_card_followup",false))&&(kind=="army_move"||kind=="army_attack"||kind=="followup_done"))return apply_card_army_followup_descriptor(s,t,actions,rng,side,row);
    if(kind=="army_attack"){
        ad_keys(actions,p,{"from","to","side","siege","mouth_muster"});const auto mouth=field(actions,p,"mouth_muster",false);
        const bool mode=mouth!=missing && typed(actions,mouth,WOTR_BOOL,"mouth_muster must be Boolean").integer!=0;
        return apply_army_attack(s,t,rng,side,index,ad_text(actions,p,"from"),ad_text(actions,p,"to"),mode);
    }
    if(kind=="fellowship_declare"){
        require(side==0,"Declaration belongs to FP");ad_keys(actions,p,{"region"});return apply_fellowship_declare_with_rng(s,t,rng,ad_text(actions,p,"region"));
    }
    if(kind=="pass"&&truth(s,obs_attr(s,s.root,"_declare_phase",false))){
        require(side==0,"Declaration pass belongs to FP");ad_keys(actions,p,{});return apply_fellowship_declare_with_rng(s,t,rng,"",true);
    }
    if(kind=="fellowship_move"){
        require(side==0,"Fellowship movement belongs to FP");ad_keys(actions,p,{});return apply_fellowship_move(s,t,rng,index);
    }
    if(kind=="play_event"){
        ad_keys(actions,p,{"card_id","side","mouth_muster"});const auto id=ad_text(actions,p,"card_id");
        const auto mouth=field(actions,p,"mouth_muster",false);
        const bool mode=mouth!=missing && typed(actions,mouth,WOTR_BOOL,"mouth_muster must be Boolean").integer!=0;
        if(esm_supported(t,ev_card(t,id)))return apply_misty_event(s,t,rng,side,index,id);
        if(emk_supported(t,ev_card(t,id)))return apply_many_kings_event(s,t,rng,side,index,id);
        if(eaw_supported(t,ev_card(t,id)))return apply_another_way_event(s,t,rng,side,index,id);
        if(esw_supported(t,ev_card(t,id)))return apply_swifter_event(s,t,rng,side,index,id);
        if(eck_supported(t,ev_card(t,id)))return apply_challenge_event(s,t,rng,side,index,id);
        if(eea_supported(t,ev_card(t,id)))return apply_ents_event(s,t,rng,side,index,id);
        if(esc_supported(t,ev_card(t,id)))return apply_stormcrow_event(s,t,rng,side,index,id);
        if(eal_supported(t,ev_card(t,id)))return apply_alone_event(s,t,rng,side,index,id);
        if(etb_supported(t,ev_card(t,id)))return apply_back_again_event(s,t,rng,side,index,id);
        if(ehe_supported(t,ev_card(t,id)))return apply_horde_event(s,t,rng,side,index,id);
        if(erg_supported(t,ev_card(t,id)))return apply_rage_event(s,t,rng,side,index,id);
        if(eco_supported(t,ev_card(t,id)))return apply_corsairs_event(s,t,rng,side,index,id,mode);
        if(ehuf_supported(t,ev_card(t,id)))return apply_help_unlooked_event(s,t,rng,side,index,id);
        if(egr_supported(t,ev_card(t,id)))return apply_grond_event(s,t,rng,side,index,id);
        if(euh_supported(t,ev_card(t,id)))return apply_uruk_event(s,t,rng,side,index,id,mode);
        if(etd_supported(t,ev_card(t,id)))return apply_through_day_event(s,t,rng,side,index,id);
        if(epw_supported(t,ev_card(t,id)))return apply_woses_event(s,t,rng,side,index,id);
        if(egw_supported(t,ev_card(t,id)))return apply_gwaihir_event(s,t,rng,side,index,id);
        if(edm_supported(t,ev_card(t,id)))return apply_dead_men_event(s,t,rng,side,index,id);
        if(erv_supported(t,ev_card(t,id)))return apply_return_valinor_event(s,t,rng,side,index,id,mode);
        if(eeg_supported(t,ev_card(t,id)))return apply_eagles_event(s,t,rng,side,index,id);
        if(eds_supported(t,ev_card(t,id)))return apply_dreadful_event(s,t,rng,side,index,id);
        if(esp_supported(t,ev_card(t,id)))return apply_spirit_event(s,t,rng,side,index,id);
        if(egc_supported(t,ev_card(t,id)))return apply_grey_company_event(s,t,rng,side,index,id);
        if(ehs_supported(t,ev_card(t,id)))return apply_house_stewards_event(s,t,rng,side,index,id);
        if(ebc_supported(t,ev_card(t,id)))return apply_black_captain_event(s,t,rng,side,index,id);
        if(erw_supported(t,ev_card(t,id)))return apply_ringwraiths_event(s,t,rng,side,index,id);
        if(ecw_supported(t,ev_card(t,id)))return apply_cruel_weather(s,t,rng,side,index,id);
        if(eff_supported(t,ev_card(t,id)))return apply_faramir_event(s,t,rng,side,index,id);
        if(etsm_supported(t,ev_card(t,id)))return apply_shadow_is_moving_event(s,t,rng,side,index,id,mode);
        if(esl_supported(t,ev_card(t,id)))return apply_shadow_lengthens_event(s,t,rng,side,index,id,mode);
        if(!enh_kind(t,ev_card(t,id)).empty())return apply_nazgul_hunt_event(s,t,rng,side,index,id);
        if(efm_supported(t,ev_card(t,id)))return apply_companion_event(s,t,rng,side,index,id);
        if(eh_template(t,ev_card(t,id))!=missing)return apply_special_tile_event_action(s,t,rng,side,index,id);
        if(!ehd_kind(t,ev_card(t,id)).empty())return apply_hunt_event_action(s,t,rng,side,index,id);
        if(em_supported(t,ev_card(t,id)))return apply_misc_event_action(s,t,rng,side,index,id);
        if(el_supported(t,ev_card(t,id)))return apply_lure_event_action(s,t,rng,side,index,id);
        if(eht_type(t,ev_card(t,id)))return apply_hunt_table_event(s,t,rng,side,index,id,mode);
        if(er_supported(t,ev_card(t,id)))return apply_recruit_event_traced(s,t,rng,side,index,id);
    }
    if(kind=="hunt_take_corruption"||kind=="hunt_sacrifice_guide"||kind=="hunt_sacrifice_random"||kind=="hunt_gollum_reveal"||kind=="hunt_gollum_no_reveal"||kind=="hunt_guide_separate"){
        require(side==0,"Hunt decisions belong to FP");
        if(kind=="hunt_guide_separate"){
            ad_keys(actions,p,{"companion"});const auto guide=obs_attr(s,obs_attr(s,s.root,"fellowship"),"guide");
            require(text(s,obs_attr(s,guide,"name"))==ad_text(actions,p,"companion"),"Hunt companion descriptor is stale");
        }else ad_keys(actions,p,{});
        return apply_hunt_choice(s,t,rng,kind);
    }
    if(kind=="fellowship_reveal_place"){
        require(side==0,"Fellowship reveal belongs to FP");ad_keys(actions,p,{"region"});return apply_fellowship_reveal(s,t,rng,ad_text(actions,p,"region"));
    }
    const auto pending_hunt=obs_attr(s,s.root,"pending_hunt",false);
    if(kind=="companion_separate"&&(truth(s,obs_attr(s,s.root,"_hobbit_place_queue",false))||(!ctx_none(s,pending_hunt)&&text_is(s,field(s,pending_hunt,"phase",false),"hobbit_place")))){
        require(side==0,"Hobbit placement belongs to FP");ad_keys(actions,p,{"companion","to_region"});
        auto menu=native_hobbit_place_actions(s,t);bool found=false;
        for(auto candidate:child_edges(*menu,menu->root))found|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
        require(found,"Hobbit placement descriptor is not a current legal action");return apply_hobbit_placement(s,t,rng,ad_text(actions,p,"to_region"));
    }
    bool hunt_guide=!ctx_none(s,obs_attr(s,s.root,"_fellowship_move_die_pending",false))||!ctx_none(s,obs_attr(s,s.root,"pending_hunt",false));
    hunt_guide|=ehd_pending(s)||ew_pending(s)||epa_waiting(s);
    for(const char* key:{"_pending_hunt_damage","_reveal_place_due","_pending_reveal_strongholds","_fellowship_route_event","_hobbit_place_queue","_mithril_stash"})hunt_guide|=truth(s,obs_attr(s,s.root,key,false));
    if(kind=="choose_guide"&&hunt_guide){
        require(side==0,"Guide choice belongs to FP");ad_keys(actions,p,{"companion","side"});return apply_hunt_guide_choice(s,t,rng,ad_text(actions,p,"companion"));
    }
    if(kind=="card_option"){
        // package R: a NON-direct optioned play whose card left its owner's hand resolves as the frozen executor does (no effect;
        // action_dispatch.inc ef_absent_pending_*), after the descriptor is validated against the complete menu as every branch below does
        if(ef_pending_card_absent(s,t)){
            require(index==-1,"Card option continuation spends no die");ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});
            auto menu=fm_native_full_actions_impl(s,t,side);bool legal=false;
            for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
            require(legal,"Card option descriptor is stale");return ef_absent_pending_apply_random(s,t,rng);
        }
        if(ecw_pending(s)){
            require(side==1 && index==-1,"Cruel Weather continuation spends no second die");
            ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});
            auto menu=native_cruel_weather_choices(s,t,side);bool legal=false;
            for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
            require(legal,"Cruel Weather descriptor is stale");return apply_cruel_weather_choice(s,t,rng,side,ad_int(actions,p,"idx"));
        }
        const auto pal=obs_attr(s,s.root,"pending_card_option",false);
        if(!ctx_none(s,pal)&&text_is(s,field(s,pal,"card_id",false),"__palantir_draw__")){
            require(side==1&&index==-1,"Palantir draw is a free Shadow choice");ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});
            auto menu=native_palantir_draw_actions(s,side);bool legal=false;
            for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
            require(legal,"Palantir draw descriptor is stale");return apply_palantir_draw(s,t,rng,side,ad_int(actions,p,"idx"));
        }
        if(edf_pending(s)){
            require(side==1&&index==-1,"Denethor continuation spends no second die");ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});
            auto menu=native_denethor_choices(s,t,side);bool legal=false;
            for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
            require(legal,"Denethor descriptor is stale");return apply_denethor_choice(s,t,rng,side,ad_int(actions,p,"idx"));
        }
        if(esm_pending(s)){
            require(side==1&&index==-1,"Misty Mountains continuation spends no second die");ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});auto menu=native_misty_choices(s,t,side);bool legal=false;
            for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
            require(legal,"Misty Mountains descriptor is stale");return apply_misty_choice(s,t,rng,side,ad_int(actions,p,"idx"));
        }
        if(esra_pending(s)){
            require(side==1&&index==-1,"Allocation continuation spends no second die");ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});auto menu=native_shadow_allocation_choices(s,t,side);bool legal=false;
            for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
            require(legal,"Allocation descriptor is stale");fm_Result result;result.state=apply_shadow_allocation_choice(s,t,side,ad_int(actions,p,"idx"));result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());
            am_Builder events;std::vector<uint64_t> rows;const auto marker=obs_attr(s,s.root,"_battle_resolved_attacker",false);cau_finish(*result.state,t,result.random.get(),ctx_none(s,marker)?-1:ctx_side(s,marker),events,rows);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
        }
        if(emk_pending(s)){
            require(side==1&&index==-1,"Many Kings continuation spends no second die");ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});auto menu=native_many_kings_choices(s,t,side);bool legal=false;
            for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
            require(legal,"Many Kings descriptor is stale");return apply_many_kings_choice(s,t,rng,side,ad_int(actions,p,"idx"));
        }
        if(eaw_pending(s)){
            require(side==0&&index==-1,"Another Way continuation spends no second die");
            ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});
            auto menu=native_another_way_choices(s,t,side);bool legal=false;
            for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
            require(legal,"Another Way descriptor is stale");return apply_another_way_choice(s,t,rng,side,ad_int(actions,p,"idx"));
        }
        if(esw_pending(s)){
            require(side==0&&index==-1,"Swifter continuation spends no second die");
            ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});
            auto menu=native_swifter_choices(s,t,side);bool legal=false;
            for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
            require(legal,"Swifter descriptor is stale");return apply_swifter_choice(s,t,rng,side,ad_int(actions,p,"idx"));
        }
        if(eea_pending(s)){
            require(side==0&&index==-1,"Ents continuation spends no second die");
            ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});
            auto menu=native_ents_choices(s,t,side);bool legal=false;
            for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
            require(legal,"Ents descriptor is stale");return apply_ents_choice(s,t,rng,side,ad_int(actions,p,"idx"));
        }
        if(esc_pending(s)){
            require(index==-1,"Stormcrow continuation spends no second die");
            ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});
            auto menu=native_stormcrow_choices(s,t,side);bool legal=false;
            for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
            require(legal,"Stormcrow descriptor is stale");return apply_stormcrow_choice(s,t,rng,side,ad_int(actions,p,"idx"));
        }
        if(eal_pending(s)){
            require(side==0&&index==-1,"Alone continuation spends no second die");
            ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});
            auto menu=native_alone_choices(s,t,side);bool legal=false;
            for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
            require(legal,"Alone descriptor is stale");return apply_alone_choice(s,t,rng,side,ad_int(actions,p,"idx"));
        }
        if(etb_pending(s)){
            require(side==0&&index==-1,"Back Again continuation spends no second die");
            ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});
            auto menu=native_back_again_choices(s,t,side);bool legal=false;
            for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
            require(legal,"Back Again descriptor is stale");return apply_back_again_choice(s,t,rng,side,ad_int(actions,p,"idx"));
        }
        if(ehe_pending(s)){
            require(side==1&&index==-1,"Horde continuation spends no second die");ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});auto menu=native_horde_choices(s,t,side);bool legal=false;
            for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
            require(legal,"Horde descriptor is stale");return apply_horde_choice(s,t,rng,side,ad_int(actions,p,"idx"));
        }
        if(erg_pending(s)){
            require(side==1&&index==-1,"Rage continuation spends no second die");ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});auto menu=native_rage_choices(s,t,side);bool legal=false;
            for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
            require(legal,"Rage descriptor is stale");return apply_rage_choice(s,t,rng,side,ad_int(actions,p,"idx"));
        }
        if(eco_pending(s)){
            require(index==-1,"Corsairs continuation spends no second die");ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});auto menu=native_corsairs_choices(s,t,side);bool legal=false;
            for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
            require(legal,"Corsairs descriptor is stale");return apply_corsairs_choice(s,t,rng,side,ad_int(actions,p,"idx"));
        }
        if(ehuf_pending(s)){
            require(index==-1,"Help Unlooked For continuation spends no second die");
            ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});
            auto menu=native_help_unlooked_choice_actions(s,t,side);bool legal=false;
            for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
            require(legal,"Help Unlooked For descriptor is stale");return apply_help_unlooked_choice(s,t,rng,side,ad_int(actions,p,"idx"));
        }
        if(egr_pending(s)){
            require(side==1&&index==-1,"Grond continuation spends no second die");ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});auto menu=native_grond_choice_actions(s,t,side);bool legal=false;
            for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
            require(legal,"Grond descriptor is stale");return apply_grond_choice(s,t,rng,side,ad_int(actions,p,"idx"));
        }
        if(euh_pending(s)){
            require(side==1&&index==-1,"Uruk-hai continuation spends no second die");ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});auto menu=native_uruk_choice_actions(s,t,side);bool legal=false;
            for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
            require(legal,"Uruk-hai descriptor is stale");return apply_uruk_choice(s,t,rng,side,ad_int(actions,p,"idx"));
        }
        if(etd_pending(s)){
            require(index==-1,"Through Day continuation spends no second die");
            ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});
            auto menu=native_through_day_choice_actions(s,side);bool legal=false;
            for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
            require(legal,"Through Day descriptor is stale");return apply_through_day_choice(s,t,rng,side,ad_int(actions,p,"idx"));
        }
        if(epw_pending(s)){
            require(side==0&&index==-1,"Woses continuation spends no second die");ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});auto menu=native_woses_choice_actions(s,side);bool legal=false;
            for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
            require(legal,"Woses descriptor is stale");return apply_woses_choice(s,t,rng,side,ad_int(actions,p,"idx"));
        }
        if(edm_pending(s)){
            require(index==-1,"Dead Men continuation spends no second die");ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});auto menu=native_dead_men_choices(s,t,side);bool legal=false;
            for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
            require(legal,"Dead Men descriptor is stale");return apply_dead_men_choice(s,t,rng,side,ad_int(actions,p,"idx"));
        }
        if(eeg_pending(s)){
            require(index==-1,"Eagles continuation spends no second die");ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});auto menu=native_eagles_choices(s,t,side);bool legal=false;
            for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
            require(legal,"Eagles descriptor is stale");return apply_eagles_choice(s,t,rng,side,ad_int(actions,p,"idx"));
        }
        if(eds_pending(s)){
            require(side==1&&index==-1,"Dreadful target continuation is free");ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});auto menu=native_dreadful_choices(s,t,side);bool legal=false;
            for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
            require(legal,"Dreadful descriptor is stale");return apply_dreadful_choice(s,t,rng,side,ad_int(actions,p,"idx"));
        }
        if(esp_pending(s)){
            require(side==0&&index==-1,"Spirit target continuation is free");ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});auto menu=native_spirit_choices(s,t,side);bool legal=false;
            for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
            require(legal,"Spirit descriptor is stale");return apply_spirit_choice(s,t,rng,side,ad_int(actions,p,"idx"));
        }
        if(egc_pending(s)){
            require(side==0&&index==-1,"Grey Company continuation is free");ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});auto menu=native_grey_company_choices(s,t,side);bool legal=false;
            for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
            require(legal,"Grey Company descriptor is stale");return apply_grey_company_choice(s,t,rng,side,ad_int(actions,p,"idx"));
        }
        if(ehs_pending(s)){
            require(side==0&&index==-1,"House of the Stewards continuation is free");ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});auto menu=native_house_stewards_choices(s,t,side);bool legal=false;
            for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
            require(legal,"House of the Stewards descriptor is stale");return apply_house_stewards_choice(s,t,rng,side,ad_int(actions,p,"idx"));
        }
        if(ebc_pending(s)){
            require(side==1&&index==-1,"Black Captain modal choice is free");ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});auto menu=native_black_captain_choices(s,t,side);bool legal=false;
            for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
            require(legal,"Black Captain descriptor is stale");return apply_black_captain_choice(s,t,rng,side,ad_int(actions,p,"idx"));
        }
        if(erw_pending(s)){
            require(side==1&&index==-1,"Ringwraiths chain is a free Shadow choice");ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});auto menu=native_ringwraiths_choice_actions(s,t,side);bool legal=false;
            for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
            require(legal,"Ringwraiths descriptor is stale");return apply_ringwraiths_choice(s,t,rng,side,ad_int(actions,p,"idx"));
        }
        if(egw_pending(s)){
            require(side==0 && index==-1,"Gwaihir continuation spends no second die");ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});auto menu=native_gwaihir_choice_actions(s,side);bool legal=false;
            for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
            require(legal,"Gwaihir descriptor is stale");return apply_gwaihir_choice(s,t,rng,side,ad_int(actions,p,"idx"));
        }
        if(ecc_pending(s) || eff_pending(s)){
            require(index==-1,"Faramir and card losses spend no second die");ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});
            const bool casualty=ecc_pending(s);auto menu=casualty?native_card_casualty_actions(s,t,side):native_faramir_choice_actions(s,t,side);bool legal=false;
            for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
            require(legal,"Faramir or card casualty descriptor is stale");
            return casualty?apply_card_casualty_choice(s,t,rng,side,ad_int(actions,p,"idx")):apply_faramir_choice(s,t,rng,side,ad_int(actions,p,"idx"));
        }
        if(etsm_pending(s)){
            require(index==-1,"Shadow is Moving continuation spends no second die");
            ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});
            auto menu=native_shadow_is_moving_choice_actions(s,side);bool legal=false;
            for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
            require(legal,"Shadow is Moving descriptor is stale");return apply_shadow_is_moving_choice(s,t,rng,side,ad_int(actions,p,"idx"));
        }
        if(esl_pending(s)){
            require(side==1 && index==-1,"Shadow Lengthens continuation spends no second die");ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});
            auto menu=native_shadow_lengthens_choice_actions(s,side);bool legal=false;
            for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
            require(legal,"Shadow Lengthens descriptor is stale");return apply_shadow_lengthens_choice(s,t,rng,side,ad_int(actions,p,"idx"));
        }
        if(enh_pending(s)){
            require(side==1 && index==-1,"Nazgul event continuation spends no second die");ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});
            auto menu=native_nazgul_hunt_choice_actions(s,side);bool legal=false;
            for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
            require(legal,"Nazgul event descriptor is stale");return apply_nazgul_hunt_choice(s,t,rng,side,ad_int(actions,p,"idx"));
        }
        if(ec_pending(s)){
            require(side==1&&index==-1,"Crebain is a free Shadow use-or-keep choice");ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});
            auto menu=native_crebain_choice_actions(s,side);bool legal=false;
            for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
            require(legal,"Crebain descriptor is stale");return apply_crebain_choice(s,t,rng,side,ad_int(actions,p,"idx"));
        }
        if(efm_pending(s)){
            require(side==0&&index==-1,"Companion event continuation spends no second die");ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});
            auto menu=native_companion_event_choice_actions(s,side);bool legal=false;
            for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
            require(legal,"Companion event descriptor is stale");return apply_companion_event_choice(s,t,rng,side,ad_int(actions,p,"idx"));
        }
        if(ew_pending(s)){
            require(side==1&&index==-1,"Worn discard is a free Shadow source choice");ad_keys(actions,p,{"idx","card_id","card_name","side","choices"});
            auto menu=native_worn_choice_actions(s,side);bool legal=false;
            for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
            require(legal,"Worn descriptor is stale");return apply_worn_choice(s,t,rng,side,ad_int(actions,p,"idx"));
        }
        if(el_pending(s)){
            require(side==0,"Lure choice belongs to FP");auto menu=native_lure_choice_actions(s);bool legal=false;
            for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
            require(legal,"Lure descriptor is stale");return apply_lure_event_choice(s,t,rng,ad_int(actions,p,"idx"));
        }
        if(ehd_pending(s)){
            require(side==0,"Hunt event continuation belongs to FP");
            auto menu=native_hunt_event_choice_actions(s);bool legal=false;
            for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
            require(legal,"Hunt event descriptor is stale");return apply_hunt_event_choice(s,t,rng,ad_int(actions,p,"idx"));
        }
        const auto choice=field(actions,p,"choices",false);
        if(choice!=missing&&(field(actions,choice,"wizardstaff_prevent",false)!=missing||field(actions,choice,"mithril_redraw",false)!=missing||field(actions,choice,"use_hunt_reduction",false)!=missing||field(actions,choice,"balrog_declare_draw",false)!=missing)){
            require(side==(field(actions,choice,"balrog_declare_draw",false)!=missing?1:0),"Hunt protection descriptor belongs to another actor");auto menu=native_hunt_protection_actions(s);bool legal=false;
            for(auto candidate:child_edges(*menu,menu->root))legal|=ap_equal(ap_freeze(*menu,candidate),ap_freeze(actions,row));
            require(legal,"Hunt protection descriptor is stale");return apply_hunt_protection(s,t,rng,ad_int(actions,p,"idx"));
        }
    }
    fm_Result result;result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());
    result.state=apply_native_descriptor(s,t,actions,side,row_index);am_Builder events;std::vector<uint64_t> annotations;
    const bool gandalf=kind=="card_option"&&text_is(actions,field(actions,p,"card_id",false),"__gandalf_draw__")&&truth(actions,field(actions,field(actions,p,"choices"),"gandalf_draw"));
    if(kind=="draw_card"||gandalf){
        const auto deck=ad_text(actions,p,"deck");const auto pile=obs_attr(s,s.root,da_deck_name(side,deck),false);
        if(truth(s,pile)){const auto card=s.edges[s.nodes[pile].offset+s.nodes[pile].count-1];
            annotations.push_back(events.row(-1,"card_drawn",{events.str("deck"),events.str(deck),events.str("card_id"),events.str(text(s,card)),events.str("side"),events.str(side==0?"FP":"SH")}));}
    }
    events.out.root=events.container(WOTR_LIST,annotations);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}

// Source public-action persistent upkeep; never run during an unresolved combat.
fm_Result apply_native_random_descriptor(const Snapshot& s,const Snapshot& t,const Snapshot& actions,const void* rng,int32_t side,int64_t row_index){
    if(!ctx_none(s,obs_attr(s,s.root,"pending_discard",false)) && fm_stale_discard_cleanup_due(s,t,side)){
        auto normalized=fm_prepare_menu_context(s,t,side);
        return apply_native_random_descriptor(*normalized,t,actions,rng,side,row_index);
    }
    auto result=apply_native_random_descriptor_inner(s,t,actions,rng,side,row_index);
    auto& child=*result.state;
    if(ctx_none(child,obs_attr(child,child.root,"pending_combat",false)) && cau_persistent(child,"LAST_BATTLE")!=missing)cau_last_battle(child,t);
    return result;
}
