// Complete raw menus only for proved supported branches. A refusal means that
// completeness is unproved; callers must never replace it with a partial menu.
bool eaw_stable_siege_scope(const Snapshot&);
std::unique_ptr<Snapshot> fm_empty(){am_Builder b;b.out.root=b.container(WOTR_LIST,{});return std::make_unique<Snapshot>(b.out);}
int fm_option_owner(const Snapshot& s){
    for(const auto key:{"pending_card_option","pending_card_followup"}){
        const auto pending=obs_attr(s,s.root,key,false);if(!truth(s,pending))continue;
        const auto owner=field(s,pending,"side",false);if(!truth(s,owner))continue;
        auto tag=std::string(text(s,owner));for(auto& c:tag)if(c>='A' && c<='Z')c=c-'A'+'a';
        if(tag=="fp")return 0;if(tag=="sh")return 1;
    }
    return -1;
}
// Exact state effect of Python get_legal_actions when its stale-empty discard
// branch is reached. This does not consume dice, fix flags, or skip earlier decisions.
bool fm_stale_discard_cleanup_due(const Snapshot& s,const Snapshot& t,int side){
    require(side==0 || side==1,"menu cleanup actor must be FP or SH");
    for(const char* key:{"pending_combat","pending_guide_change","pending_hunt"})
        if(!ctx_none(s,obs_attr(s,s.root,key,false)))return false;
    if(truth(s,obs_attr(s,s.root,"_reveal_place_due",false)) && !fr_pending(s)){
        auto reveal=native_reveal_actions(s,t);if(reveal->nodes[reveal->root].count)return false;
    }
    const auto pending=obs_attr(s,s.root,"pending_discard",false);
    return !ctx_none(s,pending) && dc_owner(s)==side && fm_option_owner(s)!=side && s.nodes[tc_hand(s,side)].count==0;
}
void fm_prepare_menu_context_inplace(Snapshot& s,const Snapshot& t,int side){
    if(fm_stale_discard_cleanup_due(s,t,side))rr_set_attr(s,s.root,"pending_discard",rr_scalar(s,WOTR_NONE));
}
std::unique_ptr<Snapshot> fm_prepare_menu_context(const Snapshot& source,const Snapshot& t,int side){
    auto state=std::make_unique<Snapshot>(source);fm_prepare_menu_context_inplace(*state,t,side);return state;
}
// Import one descriptor graph once, preserving all of its aliases. Row IDs
// are retained in order for later per-die interleaving, not concatenated.
std::vector<uint64_t> fm_import(Snapshot& out,const Snapshot& source){
    const auto& root=typed(source,source.root,WOTR_LIST,"complete menu family must return a list");
    const uint64_t nodes=out.nodes.size(),edges=out.edges.size(),bytes=out.bytes.size();
    am_grow(out.nodes,source.nodes.size());am_grow(out.edges,source.edges.size());am_grow(out.bytes,source.bytes.size());   // speed pass (growth policy): same appends; the capacity grows to the need or by 3/2, never exact-fit per import
    out.bytes.insert(out.bytes.end(),source.bytes.begin(),source.bytes.end());
    for(const auto edge:source.edges)out.edges.push_back(edge+nodes);
    for(auto value:source.nodes){
        if(value.tag==WOTR_STRING || value.tag==WOTR_BYTES || value.tag==WOTR_BIGINT)value.offset+=bytes;
        else if(value.tag>=WOTR_LIST && value.tag<=WOTR_ENUM)value.offset+=edges;
        if(value.flags)value.integer+=static_cast<int64_t>(nodes);
        out.nodes.push_back(value);
    }
    std::vector<uint64_t> rows;rows.reserve(root.count);for(uint64_t i=0;i<root.count;++i)rows.push_back(source.edges[root.offset+i]+nodes);return rows;
}
int64_t fm_die(const Snapshot& s,uint64_t row){const auto& n=s.nodes[row];return s.nodes[s.edges[n.offset]].integer;}
std::string_view fm_kind(const Snapshot& s,uint64_t row){return text(s,s.edges[s.nodes[row].offset+1]);}
// Speed pass: hand the assembled menu out without copy-constructing it (identical nodes, edges, bytes, root and
// context_prepared; the builder is discarded by every caller).
std::unique_ptr<Snapshot> fm_take(am_Builder& b){auto out=std::make_unique<Snapshot>();out->nodes.swap(b.out.nodes);out->edges.swap(b.out.edges);out->bytes.swap(b.out.bytes);out->root=b.out.root;out->context_prepared=b.out.context_prepared;return out;}
// Speed pass (growth policy): the ten head family menus of the complete menu are built first in the certified order - fm_stage
// repeats fm_import's root check at the same point of the sequence, so the first throwing operation is unchanged - then the
// complete builder is reserved once for their sum and they are imported in that same order.  The node/edge/byte sequence of
// the complete menu is identical; the head imports cost one allocation instead of one reallocation each.
std::unique_ptr<Snapshot> fm_stage(std::unique_ptr<Snapshot> menu){typed(*menu,menu->root,WOTR_LIST,"complete menu family must return a list");return menu;}
void fm_reserve_staged(am_Builder& b,const std::array<std::unique_ptr<Snapshot>,10>& staged){size_t nodes=0,edges=0,bytes=0;for(const auto& m:staged)if(m){nodes+=m->nodes.size();edges+=m->edges.size();bytes+=m->bytes.size();}b.grow(nodes,edges,bytes);}
// Speed pass (per-call memos): within ONE complete-menu build the event family is asked once per (die, category) and the attack
// family once per (die, require_leader); everything either certified builder reads is independent of the die except the die
// value stored in each row.  fm_EventMemo runs the event builder's gate, ring waiver, hand read, card lookup and printed-
// requirement evaluation the first time each is needed - in the certified order, lazily per card in hand order, so the
// sequence of throwing operations is unchanged - and reuses the results on the later calls of the same menu; every row is
// emitted by the certified expressions.  fm_AttackMemo keeps the first family built per require_leader value (certified
// builder) and imports that graph again for a later die, rewriting the die integer of every imported row (the only
// die-dependent bytes; each row's die node is a fresh INT node shared with nothing).
#define WOTR_NATIVE_FM_MEMO_AVAILABLE 1
uint64_t fm_memo_event_reuse=0,fm_memo_attack_reuse=0,fm_memo_attack_rows_rewritten=0;
struct fm_EventMemo{
    const Snapshot& s;const Snapshot& t;const int32_t side;bool gated=false;
    struct Card{uint64_t id;uint64_t card=0;bool resolved=false;int8_t requirement=-1;};std::vector<Card> cards;
    fm_EventMemo(const Snapshot& s_,const Snapshot& t_,int32_t side_):s(s_),t(t_),side(side_){}
    std::unique_ptr<Snapshot> menu(int64_t index,std::string_view category){
        require(side==0 || side==1,"event menu side must be FP or SH");
        if(gated)++fm_memo_event_reuse;
        else{quiet_action_gate(s,t,false,false,false,false,false,false,false,false,false,true);ring_waiver(s,side);gated=true;
            for(auto id:child_edges(s,tc_hand(s,side)))cards.push_back({id});}
        am_Builder b;std::vector<uint64_t> rows;
        for(auto& c:cards){
            if(!c.resolved){c.card=ev_card(t,text(s,c.id));c.resolved=true;}
            if(c.card==missing || !ev_category(t,c.card,category))continue;
            if(c.requirement<0)c.requirement=ev_requirement(s,t,c.card)?1:0;
            if(c.requirement)rows.push_back(b.row(index,"play_event",{b.str("card_id"),b.str(text(s,c.id)),b.str("side"),b.str(side==0?"fp":"sh")}));
        }
        b.out.root=b.container(WOTR_LIST,rows);return fm_take(b);
    }
};
struct fm_AttackMemo{
    const Snapshot& s;const Snapshot& t;const int side;std::unique_ptr<Snapshot> built[2];
    fm_AttackMemo(const Snapshot& s_,const Snapshot& t_,int side_):s(s_),t(t_),side(side_){}
    // == fm_import(out,*native_army_attack_for_die(s,t,side,index,require_leader)): first call per require_leader builds
    // through the certified builder; a later call re-imports that graph and sets every imported row's die integer to `index`.
    std::vector<uint64_t> import(Snapshot& out,int64_t index,bool require_leader){
        auto& cached=built[require_leader?1:0];
        if(!cached){cached=native_army_attack_for_die(s,t,side,index,require_leader);return fm_import(out,*cached);}
        auto rows=fm_import(out,*cached);++fm_memo_attack_reuse;
        for(auto row:rows){out.nodes[out.edges[out.nodes[row].offset]].integer=index;++fm_memo_attack_rows_rewritten;}
        return rows;
    }
};
std::unique_ptr<Snapshot> fm_native_full_actions_impl(const Snapshot& s,const Snapshot& t,int32_t side){
    if(fm_stale_discard_cleanup_due(s,t,side)){auto normalized=fm_prepare_menu_context(s,t,side);return fm_native_full_actions_impl(*normalized,t,side);}
    require(side==0 || side==1,"complete menu side must be FP or SH");attributes(s,s.root,"game_state.GameState");
    const auto combat=obs_attr(s,s.root,"pending_combat");
    if(!ctx_none(s,combat)){
        const auto owner=field(s,combat,"deciding_side",false);
        if(!ctx_none(s,owner) && ctx_side(s,owner)!=side)return fm_empty();
        return native_combat_pending_actions(s,t,side);
    }
    if(!ctx_none(s,obs_attr(s,s.root,"pending_guide_change")))return native_guide_actions(s,side);
    const auto hunt=obs_attr(s,s.root,"pending_hunt");
    if(!ctx_none(s,hunt)){
        if(text_is(s,field(s,hunt,"phase"),"hobbit_place"))return native_hobbit_place_actions(s,t);
        return native_hunt_damage_actions(s);
    }
    // Frozen raw menus return Hunt and reveal rows independently of the
    // caller's side; the driver owns FP routing. Preserve that exact behavior.
    if(truth(s,obs_attr(s,s.root,"_reveal_place_due",false)) && !fr_pending(s)){
        auto reveal=native_reveal_actions(s,t);if(reveal->nodes[reveal->root].count)return reveal;
    }
    const auto discard=obs_attr(s,s.root,"pending_discard",false);
    if(!ctx_none(s,discard) && dc_owner(s)==side && fm_option_owner(s)!=side){
        const auto hand=tc_hand(s,side); // Package W: an empty hand yields the frozen empty discard menu
        am_Builder b;std::vector<uint64_t> rows;std::set<std::string> seen;
        for(auto card:child_edges(s,hand)){
            const std::string id(text(s,card));if(seen.insert(id).second)rows.push_back(b.row(-1,"discard_card",{
                b.str("card_id"),b.str(id),b.str("side"),b.str(side==0?"fp":"sh")}));
        }
        b.out.root=b.container(WOTR_LIST,rows);return std::make_unique<Snapshot>(b.out);
    }
    if(truth(s,obs_attr(s,s.root,"_declare_phase",false)))return native_fellowship_declare_actions(s,t,side);
    if(truth(s,obs_attr(s,s.root,"_eye_allocation_phase",false)))return eye_allocation_actions(s,side);
    if(!ctx_none(s,obs_attr(s,s.root,"pending_card_followup",false)))return native_card_army_followup_actions(s,t,side);
    const auto option=obs_attr(s,s.root,"pending_card_option",false);
    if(ecc_pending(s))return native_card_casualty_actions(s,t,side);
    if(edm_pending(s))return native_dead_men_choices(s,t,side);
    if(eeg_pending(s))return native_eagles_choices(s,t,side);
    if(eds_pending(s))return native_dreadful_choices(s,t,side);
    if(esp_pending(s))return native_spirit_choices(s,t,side);
    if(eff_pending(s))return native_faramir_choice_actions(s,t,side);
    if(esm_pending(s))return native_misty_choices(s,t,side);
    if(esra_pending(s))return native_shadow_allocation_choices(s,t,side);
    if(emk_pending(s))return native_many_kings_choices(s,t,side);
    if(edf_pending(s))return native_denethor_choices(s,t,side);
    if(eaw_pending(s))return native_another_way_choices(s,t,side);
    if(esw_pending(s))return native_swifter_choices(s,t,side);
    if(eea_pending(s))return native_ents_choices(s,t,side);
    if(esc_pending(s))return native_stormcrow_choices(s,t,side);
    if(eal_pending(s))return native_alone_choices(s,t,side);
    if(etb_pending(s))return native_back_again_choices(s,t,side);
    if(ehe_pending(s))return native_horde_choices(s,t,side);
    if(erg_pending(s))return native_rage_choices(s,t,side);
    if(eco_pending(s))return native_corsairs_choices(s,t,side);
    if(ehuf_pending(s))return native_help_unlooked_choice_actions(s,t,side);
    if(egr_pending(s))return native_grond_choice_actions(s,t,side);
    if(euh_pending(s))return native_uruk_choice_actions(s,t,side);
    if(etd_pending(s))return native_through_day_choice_actions(s,side);
    if(epw_pending(s))return native_woses_choice_actions(s,side);
    if(egw_pending(s))return native_gwaihir_choice_actions(s,side);
    if(ecw_pending(s))return native_cruel_weather_choices(s,t,side);
    if(!ctx_none(s,option)&&text_is(s,field(s,option,"card_id",false),"__palantir_draw__"))return native_palantir_draw_actions(s,side);
    if(!ctx_none(s,option) && text_is(s,field(s,option,"card_name",false),"Pits of Mordor"))return native_pits_choice_actions(s,t,side);
    if(!ctx_none(s,option) && text_is(s,field(s,option,"card_id",false),"__gandalf_draw__"))return native_gandalf_draw_actions(s,side);
    if(!ctx_none(s,option) && text_is(s,field(s,option,"card_id",false),"__companion_moves__")){
        require(side==0,"complete companion continuation belongs to FP");return native_companion_chain_actions(s);
    }
    if(egc_pending(s))return native_grey_company_choices(s,t,side);
    if(ehs_pending(s))return native_house_stewards_choices(s,t,side);
    if(ebc_pending(s))return native_black_captain_choices(s,t,side);
    if(erw_pending(s))return native_ringwraiths_choice_actions(s,t,side);
    if(enh_pending(s))return native_nazgul_hunt_choice_actions(s,side);
    if(!ctx_none(s,option) && text_is(s,field(s,option,"card_id",false),"__nazgul_moves__"))return native_nazgul_chain_actions(s,side);
    if(!ctx_none(s,option) && text_is(s,field(s,option,"card_name",false),"The Lidless Eye"))return native_dice_event_choice_actions(s,side);
    if(!ctx_none(s,option) && text_is(s,field(s,option,"card_name",false),"Wisdom of Elrond"))return native_misc_event_choice_actions(s,side);
    if(ehd_pending(s)){require(side==0,"Hunt event choice belongs to FP");return native_hunt_event_choice_actions(s);}
    if(el_pending(s)){require(side==0,"Lure choice belongs to FP");return native_lure_choice_actions(s);}
    if(ew_pending(s))return native_worn_choice_actions(s,side);
    if(ec_pending(s))return native_crebain_choice_actions(s,side);
    if(efm_pending(s))return native_companion_event_choice_actions(s,side);
    if(etsm_pending(s))return native_shadow_is_moving_choice_actions(s,side);
    if(esl_pending(s))return native_shadow_lengthens_choice_actions(s,side);
    if(!ctx_none(s,option) && (text_is(s,field(s,option,"card_name",false),"A Power too Great") || text_is(s,field(s,option,"card_name",false),"The Power of Tom Bombadil")))return native_power_payment_actions(s,t,side);
    if(!ctx_none(s,option) && (text_is(s,field(s,option,"card_name",false),"Cirdan's Ships") || text_is(s,field(s,option,"card_name",false),"Riders of Th\xC3\xA9oden") || text_is(s,field(s,option,"card_name",false),"\xC3\x89omer son of \xC3\x89omund")))return native_target_recruit_choice_actions(s,t,side);
    if(!ctx_none(s,option) && (text_is(s,field(s,option,"card_name",false),"Half-orcs and Goblin-men") || text_is(s,field(s,option,"card_name",false),"Olog-hai")))return native_half_orcs_choice_actions(s,t,side);
    if(!ctx_none(s,option) && text_is(s,field(s,option,"card_name",false),"Shadows Gather"))return native_shadows_gather_choice_actions(s,t,side);
    if(!ctx_none(s,option)&&text_is(s,field(s,option,"card_name",false),"Hill-trolls"))return native_hill_trolls_choice_actions(s,t,side);
    if(!ctx_none(s,option)&&text_is(s,field(s,option,"card_name",false),"Guards of the Citadel"))return native_recruit_event_choice_actions(s,t,side);
    if(!ctx_none(s,option)&&text_is(s,field(s,option,"card_id",false),"__overstack__"))return native_overstack_choice_actions(s,side);
    if(!ctx_none(s,option) && (truth(s,field(s,option,"hunt_reduction",false)) ||
        text_is(s,field(s,option,"card_name",false),"Wizard's Staff") || text_is(s,field(s,option,"card_name",false),"Mithril Coat and Sting") || text_is(s,field(s,option,"card_name",false),"Balrog of Moria"))){
        const auto family=hpr_family(s);require(side==(family=="balrog"?1:0),"Hunt protection menu requested for wrong deciding side");return native_hunt_protection_actions(s);
    }
    require(ctx_none(s,obs_attr(s,s.root,"pending_card_option",false)),"complete card-option menu is not ported");
    const auto army=obs_attr(s,s.root,"pending_army_move_die"),army_owner=obs_attr(s,s.root,"pending_army_move_side",false);
    if(!ctx_none(s,army) && (ctx_none(s,army_owner) || ctx_side(s,army_owner)==side))return native_army_move_actions(s,t,side);
    const auto muster=obs_attr(s,s.root,"pending_muster_die"),muster_owner=obs_attr(s,s.root,"pending_muster_side",false);
    if(!ctx_none(s,muster) && (ctx_none(s,muster_owner) || ctx_side(s,muster_owner)==side))return native_muster_actions(s,t,side);

    const bool stable_siege_menu=truth(s,obs_attr(s,s.root,"besieged_strongholds",false));
    // Package P: a quiet turn-start state before the action roll (no pending decision, no phase flag, dice_rolled false -
    // the child of the last hand-limit discard inside a discard search, ai.py:16995-17010 / 17060-17072). The frozen
    // get_legal_actions (ai.py:13497-13607) enumerates nothing die-funded there (_get_unused_dice is empty), no elven_ring
    // / dwarven_ring row (both need a rolled face) and no pass; only the bid-compensation tokens (_get_token_actions,
    // die_idx -1) remain. The same quiet-board proof applies with the roll clause waived (allow_unrolled).
    if(!truth(s,obs_attr(s,s.root,"dice_rolled"))){
        if(stable_siege_menu)quiet_stable_action_gate(s,t,false,false,false,true);else quiet_action_gate(s,t,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,true);
        ring_waiver(s,side);
        return native_token_actions(s,t,side);
    }
    if(stable_siege_menu)quiet_stable_action_gate(s,t);else quiet_action_gate(s,t);
    ring_waiver(s,side);
    const auto dice=child_edges(s,pa_pool(s,side));std::vector<std::pair<size_t,int>> unused;bool army_face=false,muster_face=false,character_face=false,will_face=false;
    for(size_t i=0;i<dice.size();++i){
        const auto die=dice[i];require(object_type(s,die)=="game_state.ActionDie","complete menu requires ActionDie");
        const auto current=obs_attr(s,die,"current");if(truth(s,obs_attr(s,die,"used")) || ctx_none(s,current))continue;
        const int face=rr_face_index(enum_member(s,current,"game_state.ActionDieFace"));
        require(face!=6 || side==0,"complete menu does not admit nonphysical Shadow Will dice");
        unused.emplace_back(i,face);army_face|=face==0 || face==1 || face==4 || face==6;muster_face|=face==2 || face==4 || face==6;character_face|=face==0 || face==6;will_face|=face==6;
    }
    const auto fs=obs_attr(s,s.root,"fellowship"),guide=obs_attr(s,fs,"guide");
    const bool strider_hide=side==0 && !truth(s,obs_attr(s,fs,"hidden")) && !ctx_none(s,guide) &&
        std::string(text(s,obs_attr(s,guide,"name"))).find("Strider")!=std::string::npos;
    const bool mouth_funding=am_mouth_available(s,side);
    am_Builder b;std::vector<uint64_t> rows;
    auto import=[&](std::unique_ptr<Snapshot> menu){return fm_import(b.out,*menu);};
    // speed pass (growth policy): build the head families first (certified order, same conditions), reserve once, import in that order - see fm_stage
    std::array<std::unique_ptr<Snapshot>,10> staged;
    staged[0]=fm_stage(political_ring_actions(s,t,side));
    staged[1]=fm_stage(native_muster_actions(s,t,side));
    if(army_face || mouth_funding)staged[2]=fm_stage(native_army_move_actions(s,t,side));
    if(side==1 && muster_face)staged[3]=fm_stage(saruman_muster_actions(s,t,side));
    if(side==1 && muster_face)staged[4]=fm_stage(native_minion_muster_actions(s,t,side));
    if(side==0)staged[5]=fm_stage(native_companion_political_actions(s,t,side));
    staged[6]=fm_stage(native_draw_card_actions(s,t,side));
    if(character_face && side==1)staged[7]=fm_stage(native_mouth_move_actions(s,t,side));
    if(character_face && side==1)staged[8]=fm_stage(native_nazgul_move_actions(s,t,side));
    if(will_face)staged[9]=fm_stage(native_will_special_actions(s,t,side));
    fm_reserve_staged(b,staged);
    auto take_staged=[&](size_t i){return staged[i]?fm_import(b.out,*staged[i]):std::vector<uint64_t>{};};
    const auto political=take_staged(0);const auto recruits=take_staged(1);const auto moves=take_staged(2);const auto saruman=take_staged(3);const auto minions=take_staged(4);
    const auto companions=take_staged(5);const auto draws=take_staged(6);const auto mouth=take_staged(7);const auto nazgul=take_staged(8);const auto special=take_staged(9);
    auto append=[&](const std::vector<uint64_t>& family,size_t die){for(auto row:family)if(fm_die(b.out,row)==static_cast<int64_t>(die))rows.push_back(row);};
    fm_EventMemo event_memo(s,t,side);fm_AttackMemo attack_memo(s,t,side);   // speed pass (per-call memos), see fm_EventMemo / fm_AttackMemo
    auto events=[&](size_t die,const char* category,bool mouth_mode=false){
        auto menu=event_memo.menu(static_cast<int64_t>(die),category);
        for(auto row:child_edges(*menu,menu->root)){
            const auto parts=child_edges(*menu,row);const auto id=field(*menu,parts[2],"card_id");
            const auto card=ev_card(t,text(*menu,id));
            require(ctx_side(t,obs_attr(t,card,"side"))==side,"complete menu has a wrong-side card in acting hand");
        }
        const auto added=import(std::move(menu));
        for(auto row:added){
            if(mouth_mode){const auto p=child_edges(b.out,row)[2];auto edges=child_edges(b.out,p);edges.push_back(b.str("mouth_muster"));edges.push_back(rr_scalar(b.out,WOTR_BOOL,1));set_edges(b.out,p,edges);}
            rows.push_back(row);
        }
    };
    std::array<bool,7> seen{};const bool power_kill=side==1 && ept_first(s)!=missing;
    const bool palantir_kill=side==0 && hpr_card(s,"PALANTIR_OF_ORTHANC")!=missing;
    auto attacks=[&](size_t index,bool require_leader){
        const auto family=attack_memo.import(b.out,static_cast<int64_t>(index),require_leader);rows.insert(rows.end(),family.begin(),family.end());
    };
    for(const auto& entry:unused){
        const auto index=entry.first;const int face=entry.second;if(seen[face])continue;seen[face]=true;
        if(face==1 || face==4 || face==6){append(moves,index);attacks(index,false);if(face!=6)events(index,"army");}
        if(face==2 || face==4 || face==6){append(political,index);append(recruits,index);append(saruman,index);append(minions,index);if(face!=6)events(index,"muster");}
        if(face==2 && mouth_funding){
            append(moves,index);const auto family=attack_memo.import(b.out,static_cast<int64_t>(index),false);
            for(auto row:family){const auto p=child_edges(b.out,row)[2];auto edges=child_edges(b.out,p);edges.push_back(b.str("mouth_muster"));edges.push_back(rr_scalar(b.out,WOTR_BOOL,1));set_edges(b.out,p,edges);rows.push_back(row);}
            events(index,"army",true);
        }
        if(power_kill){const auto removals=import(native_power_kill_for_die(s,t,side,index));rows.insert(rows.end(),removals.begin(),removals.end());}
        if(side==0&&hpr_card(s,"DENETHORS_FOLLY")!=missing){const auto removals=import(native_denethor_kill_for_die(s,side,index));rows.insert(rows.end(),removals.begin(),removals.end());}
        if(palantir_kill){const auto removals=import(native_palantir_kill_for_die(s,side,index));rows.insert(rows.end(),removals.begin(),removals.end());}
        append(companions,index);
        if(face==0 || face==6){
            if(side==0){
                const auto fellowship=import(native_fellowship_actions(s,index));rows.insert(rows.end(),fellowship.begin(),fellowship.end());
                const auto separate=import(native_companion_separation_actions(s,t,index));rows.insert(rows.end(),separate.begin(),separate.end());
                const auto figures=import(native_companion_move_actions(s,t,index));rows.insert(rows.end(),figures.begin(),figures.end());
            }else{append(nazgul,index);append(mouth,index);}
            if(face==0){append(moves,index);attacks(index,true);events(index,"character");}
        }
        if(face==3 || face==6){append(draws,index);events(index,"event");}
        if(face==6)append(special,index);
    }
    if(strider_hide){
        std::array<bool,7> seen_hide{};
        for(const auto& entry:unused){const auto face=entry.second;if(face==0 || face==6 || seen_hide[face])continue;seen_hide[face]=true;
            rows.push_back(b.row(entry.first,"fellowship_hide",{b.str("strider_ability"),rr_scalar(b.out,WOTR_BOOL,1)}));
        }
    }
    for(auto row:political)if(fm_kind(b.out,row)=="elven_ring")rows.push_back(row);
    for(auto row:import(native_token_actions(s,t,side)))rows.push_back(row);
    for(auto row:import(native_dwarven_ring_actions(s,side)))rows.push_back(row);
    size_t other_unused=0;for(auto die:child_edges(s,pa_pool(s,1-side)))
        if(!truth(s,obs_attr(s,die,"used")) && !ctx_none(s,obs_attr(s,die,"current")))++other_unused;
    // 1.272 mirror (static_r5): ai.get_legal_actions - USER 2026-09-10 "if the decks are empty, and the player only has palantirs and nothing playable, allow the option to kill the die for no benefit" (rulebook 1678-1680 restored in exactly this situation; 1.67 stands elsewhere): every unused die shows the Event face, both own decks are empty, no event row for that die -> ONE skip row for the first unused Event die
    if(!unused.empty()){bool all_event=true;for(const auto& entry:unused)all_event&=entry.second==3;
        if(all_event){const size_t kd=unused.front().first;
            const bool decks_empty=!truth(s,obs_attr(s,s.root,side==0?"fp_char_deck":"sh_char_deck",false)) && !truth(s,obs_attr(s,s.root,side==0?"fp_strat_deck":"sh_strat_deck",false));
            if(decks_empty){auto kill_menu=event_memo.menu(static_cast<int64_t>(kd),"event");
                if(child_edges(*kill_menu,kill_menu->root).empty())rows.push_back(b.row(static_cast<int64_t>(kd),"skip",{b.str("face"),b.str("EVENT"),b.str("side"),b.str(side==0?"fp":"sh")}));}
        }
    }
    // MC6 mirror (production turn_loop_repair_r1 2026-09-10, ai.py get_legal_actions AND its compiled twin _segment_get_legal_actions, the only ai.py
    // difference static_r5 -> src/; package Z 2026-09-11): "once the opponent has no legal action, this side plays its remaining actions consecutively.
    // A pass cannot hand control back to an empty menu forever" - the ordinary pass row is offered ONLY when the opponent's COMPLETE menu of this
    // same state is non-empty (Python: get_legal_actions(clone_state(state), other_side); the native builder never mutates the graph, so this
    // copy's own complete-menu function is probed on the same snapshot - a probe refusal propagates, an unproved opponent menu leaves the pass
    // row unproved).  Recursion stops here exactly as in Python: the opponent holds MORE unused dice, so its own pass condition is false.
    if(!unused.empty() && unused.size()<other_unused){
        auto opponent_menu=fm_native_full_actions_impl(s,t,1-side);
        if(!child_edges(*opponent_menu,opponent_menu->root).empty())rows.push_back(b.row(-1,"pass",{}));
    }
    // Rulebook p21: if no ordinary action exists, spend one non-Eye die
    // without effect. Pending decisions returned before this ordinary menu.
    if(rows.empty())for(const auto& entry:unused)if(entry.second!=5){
        rows.push_back(b.row(entry.first,"skip",{b.str("face"),b.str(rr_face_names[entry.second]),b.str("side"),b.str(side==0?"fp":"sh")}));break;
    }
    // Compose the original complete raw graph first. Refuse the whole menu
    // if any offered action has no proved stable-siege executor. Never filter
    // such rows out, or infer executor support from raw-menu enumeration.
    if(stable_siege_menu)for(auto row:rows){
        const auto kind=fm_kind(b.out,row);
        const bool ordinary=kind=="advance_political"||kind=="elven_ring"||
            kind=="muster_recruit"||kind=="muster_elite"||kind=="muster_minion"||
            kind=="army_move"||kind=="army_attack"||kind=="draw_card"||
            kind=="fellowship_move"||kind=="fellowship_hide"||
            kind=="companion_separate"||kind=="companion_move"||kind=="nazgul_move"||
            kind=="mouth_move"||kind=="use_token"||kind=="dwarven_ring"||
            kind=="crown_aragorn"||kind=="transform_gandalf"||kind=="pass"||kind=="skip";   // 1.272 mirror (static_r5): a die kill is an ordinary (proved) executor
        if(ordinary)continue;
        // Paths has independent mixed-siege origin/partial-departure and
        // exact saved494 public proof, including its deferred choice phase.
        if(kind=="play_event"){
            const auto params=child_edges(b.out,row)[2];
            const auto definition=ev_card(t,text(b.out,field(b.out,params,"card_id")));
            if(side==0&&(efm_supported(t,definition)||eck_supported(t,definition)||ehs_supported(t,definition)||egc_supported(t,definition)||ept_supported(t,definition)||text_is(t,obs_attr(t,definition,"card_name"),"Mithril Coat and Sting") || text_is(t,obs_attr(t,definition,"card_name"),"Axe and Bow")||etd_supported(t,definition)||etb_supported(t,definition)||text_is(t,obs_attr(t,definition,"card_name"),"Celeborn's Galadhrim") || text_is(t,obs_attr(t,definition,"card_name"),"Grimbeorn the Old Son of Beorn") || text_is(t,obs_attr(t,definition,"card_name"),"King Brand's Men") || text_is(t,obs_attr(t,definition,"card_name"),"Swords in Eriador") || text_is(t,obs_attr(t,definition,"card_name"),"Elven Rope") || text_is(t,obs_attr(t,definition,"card_name"),"\x53\x6d\xc3\xa9\x61\x67\x6f\x6c\x20\x48\x65\x6c\x70\x73\x20\x4e\x69\x63\x65\x20\x4d\x61\x73\x74\x65\x72") || text_is(t,obs_attr(t,definition,"card_name"),"Bilbo's Song") || text_is(t,obs_attr(t,definition,"card_name"),"Mirror of Galadriel")||(eaw_supported(t,definition)&&eaw_stable_siege_scope(s))||text_is(t,obs_attr(t,definition,"card_name"),"Imrahil of Dol Amroth")||text_is(t,obs_attr(t,definition,"card_name"),"Guards of the Citadel")||(text_is(t,obs_attr(t,definition,"card_name"),"Dain Ironfoot's Guard") || text_is(t,obs_attr(t,definition,"card_name"),"The Red Arrow") || text_is(t,obs_attr(t,definition,"card_name"),"Thranduil's Archers"))||text_is(t,obs_attr(t,definition,"card_name"),"Horn of Gondor")||text_is(t,obs_attr(t,definition,"card_name"),"We Prove Swifter")||text_is(t,obs_attr(t,definition,"card_name"),"Kindred of Glorfindel")||eal_supported(t,definition)||text_is(t,obs_attr(t,definition,"card_name"),"The Last Battle")||eea_supported(t,definition)||ehuf_supported(t,definition)||edm_supported(t,definition)||eeg_supported(t,definition)||epw_supported(t,definition)||esp_supported(t,definition)||text_is(t,obs_attr(t,definition,"card_name"),"Phial of Galadriel")||text_is(t,obs_attr(t,definition,"card_name"),"Wizard's Staff")||text_is(t,obs_attr(t,definition,"card_name"),"Elven Cloaks")||em_supported(t,definition)||eff_supported(t,definition)||egw_supported(t,definition)||ert_kind(t,definition)==1||ert_kind(t,definition)==2||ert_kind(t,definition)==3))continue;
            if(side==1&&(text_is(t,obs_attr(t,definition,"card_name"),"Orc Patrol")||text_is(t,obs_attr(t,definition,"card_name"),"Isildur's Bane")||text_is(t,obs_attr(t,definition,"card_name"),"The Shadow is Moving")||text_is(t,obs_attr(t,definition,"card_name"),"Hill-trolls")||text_is(t,obs_attr(t,definition,"card_name"),"The Lidless Eye")||(text_is(t,obs_attr(t,definition,"card_name"),"Worn with Sorrow and Toil") || text_is(t,obs_attr(t,definition,"card_name"),"Flocks of Crebain") || text_is(t,obs_attr(t,definition,"card_name"),"Balrog of Moria") || text_is(t,obs_attr(t,definition,"card_name"),"Wormtongue"))||!enh_kind(t,definition).empty()||text_is(t,obs_attr(t,definition,"card_name"),"Threats and Promises")||(esr_kind(t,definition)>0 && esr_kind(t,definition)<5)||(text_is(t,obs_attr(t,definition,"card_name"),"Shelob's Lair") || text_is(t,obs_attr(t,definition,"card_name"),"The Ring is Mine!") || text_is(t,obs_attr(t,definition,"card_name"),"On On They Went") || text_is(t,obs_attr(t,definition,"card_name"),"Morgul Wound") || text_is(t,obs_attr(t,definition,"card_name"),"The Day Without Dawn") || text_is(t,obs_attr(t,definition,"card_name"),"Stormcrow") || text_is(t,obs_attr(t,definition,"card_name"),"Rage of the Dunlendings") || text_is(t,obs_attr(t,definition,"card_name"),"Return of the Witch-king") || text_is(t,obs_attr(t,definition,"card_name"),"Olog-hai") || text_is(t,obs_attr(t,definition,"card_name"),"Many Kings to the Service of Mordor") || text_is(t,obs_attr(t,definition,"card_name"),"The King is Revealed") || text_is(t,obs_attr(t,definition,"card_name"),"Shadows on the Misty Mountains") || text_is(t,obs_attr(t,definition,"card_name"),"Horde from the East") || text_is(t,obs_attr(t,definition,"card_name"),"Pits of Mordor"))||esl_supported(t,definition)||erw_supported(t,definition)||ebc_supported(t,definition)||esg_supported(t,definition)||text_is(t,obs_attr(t,definition,"card_name"),"The Palant\xc3\xadr of Orthanc")||text_is(t,obs_attr(t,definition,"card_name"),"Half-orcs and Goblin-men")||text_is(t,obs_attr(t,definition,"card_name"),"Give it to Uss!")||ehd_kind(t,definition)=="foul_thing"||ehd_kind(t,definition)=="breaking"||el_supported(t,definition)||egr_supported(t,definition)||eco_supported(t,definition)||erv_supported(t,definition)||eds_supported(t,definition)||euh_supported(t,definition)||ecw_supported(t,definition)||text_is(t,obs_attr(t,definition,"card_name"),"Denethor's Folly")))continue;
        }
        if(kind=="discard_table_card"){
            const auto params=child_edges(b.out,row)[2];const auto folly=hpr_card(s,"DENETHORS_FOLLY");if(side==0&&folly!=missing&&text_is(b.out,field(b.out,params,"card_id"),text(s,obs_attr(s,folly,"card_id"))))continue;const auto card=hpr_card(s,"PALANTIR_OF_ORTHANC");
            if(side==0&&card!=missing&&text_is(b.out,field(b.out,params,"card_id"),text(s,obs_attr(s,card,"card_id"))))continue;
            // Both removal entry and its learned pair-payment continuation
            // have independent public mixed-siege execution proof.
            bool proved_removal=false;
            if(side==1)for(const char* type:{"POWER_TOO_GREAT","TOM_BOMBADIL"}){
                const auto target=hpr_card(s,type);
                proved_removal|=target!=missing&&text_is(b.out,field(b.out,params,"card_id"),text(s,obs_attr(s,target,"card_id")));
            }
            if(proved_removal)continue;
        }
        (void)kind; // Package W: the complete stable-siege allowlist is lifted (package L: 0 firings / 333,668 rows)
    }
    b.out.root=b.container(WOTR_LIST,rows);return fm_take(b);
}
// ---------------------------------------------------------------------------------------------------------------
// WOTR_NATIVE_FULLMENU_CHECK reference builder.  From here to fm_check_compare: the certified text of this fragment
// (ROOT + the upstream package layers = engine_patches/upstream/full_action_menu.inc.certified.patch + army_move_menu.inc) VERBATIM with every
// column-0 definition renamed `_ref` (controller/ref_copy_tool.py, assembled by controller/fm_draft_tool.py; renamed:
// amg_append, fm_die, fm_empty, fm_import, fm_kind, fm_option_owner, native_army_move_actions, native_full_actions).  The family builders it calls are the shared engine functions until one
// of them is optimized, at which point that fragment joins the reference set.  `fm_check_active` is set by
// search_controller.inc from WOTR_NATIVE_FULLMENU_CHECK=1 (never in timed runs); `native_full_actions` below then
// builds every complete menu twice and throws on the first row-count, row, graph-identity or exception-text difference.
#define WOTR_NATIVE_FULLMENU_CHECK_AVAILABLE 1
int fm_check_active=0;uint64_t fm_check_menus_checked=0,fm_check_rows_checked=0;
// Raw _get_army_move_actions/get_legal_actions descriptors. No AI projection.
bool at_shadow_blocked(const Snapshot& s,std::string_view key);
void amg_append_ref(am_Builder& b,std::vector<uint64_t>& rows,const Snapshot& s,const Snapshot& t,
                int side,int64_t die,bool require_leader,std::string_view exclude){
    const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));
    const auto adj=obs_attr(s,obs_attr(s,s.root,"board"),"_adj");
    const auto pre=obs_attr(s,s.root,"pending_army_move_preexist",false);
    const auto sieges=obs_attr(s,s.root,"besieged_strongholds",false);
    bool saruman=false;
    for(size_t i=1;i<armies.size();i+=2)for(auto l:child_edges(s,obs_attr(s,armies[i],"leaders")))saruman|=am_saruman(s,l);
    for(size_t i=0;i<armies.size();i+=2){
        const auto akey=text(s,armies[i]);const auto army=armies[i+1];
        const auto units=child_edges(s,obs_attr(s,army,"units")),leaders=child_edges(s,obs_attr(s,army,"leaders"));
        std::vector<uint64_t> own;for(auto u:units)if(ctx_unit_side(s,u)==side)own.push_back(u);
        if(own.empty())continue;
        bool figure=false;for(auto l:leaders)figure|=am_leader_side(s,l)==side;
        if(saruman)for(auto u:own)figure|=enum_is(s,obs_attr(s,u,"nation"),"game_state.Nation","ISENGARD") && enum_is(s,obs_attr(s,u,"kind"),"game_state.UnitKind","ELITE");
        if(require_leader && !figure)continue;
        const auto from=sanitize_name(akey);const bool capped=!exclude.empty() && from==exclude;
        const bool source_garrison=contains_string(s,sieges,from) && at_controller(s,t,from)==side;
        auto cap=[&](const char* name){const auto n=ctx_get(s,pre,name);return ctx_none(s,n)?int64_t(0):typed(s,n,WOTR_INT,"army menu preexisting count must be integer").integer;};
        if(capped && (!truth(s,pre) || cap("regular")+cap("elite")<=0))continue;
        int64_t plain=0,must=0,nazgul=0;
        for(auto l:leaders){
            plain+=object_type(s,l)!="game_state.Nazgul" && !am_saruman(s,l) && am_leader_side(s,l)==side;
            must+=am_plain(s,l) && am_leader_side(s,l)==side;
            nazgul+=side==1 && object_type(s,l)=="game_state.Nazgul";
        }
        const int64_t plain_all=plain;   // D2 mirror (static_r4): the UNCAPPED non-Nazgul count = the figures a take_leaders row really moves
        if(capped){plain=std::min(plain,cap("leaders"));must=std::min(must,plain);}
        const auto political=obs_attr(s,s.root,"political"),boxes=obs_attr(s,political,"boxes");
        auto atwar=[&](uint64_t n){return ctx_num(s,ctx_nation_value(s,boxes,ctx_nation(s,n)))>=3;};
        bool passive=false;for(auto u:own)passive|=!atwar(obs_attr(s,u,"nation"));
        for(auto l:leaders)if(am_plain(s,l) && am_leader_side(s,l)==side)passive|=!atwar(obs_attr(s,l,"nation"));
        std::vector<std::string> neighbors;if(!source_garrison)for(auto n:child_edges(s,ctx_get(s,adj,from)))neighbors.emplace_back(text(s,n));
        if(side==0 && me_companion_activation_available(s,t,from,leaders))neighbors.push_back(from);
        std::sort(neighbors.begin(),neighbors.end());
        for(const auto& to:neighbors){
            const bool stationary=to==from;
            if(side==1 && at_shadow_blocked(s,to))continue;
            if(!stationary && contains_string(s,sieges,to) && at_controller(s,t,to)==side)continue;
            const auto nation=ctx_get(t,field(t,t.root,"region_nations"),to);
            const bool border=!stationary && passive && !ctx_none(t,nation);
            auto eligible=[&](uint64_t n){return !border || atwar(n) || ctx_nation(s,n)==ctx_nation(t,nation);};
            std::vector<uint64_t> pool;for(auto u:own)if(eligible(obs_attr(s,u,"nation")))pool.push_back(u);
            if(pool.empty())continue;
            int64_t pl=plain,mg=must,pl_all=plain_all;
            if(border){
                pl=mg=0;
                for(auto l:leaders)if(am_leader_side(s,l)==side){
                    const bool crosses=!am_plain(s,l) || eligible(obs_attr(s,l,"nation"));
                    pl+=object_type(s,l)!="game_state.Nazgul" && !am_saruman(s,l) && crosses;
                    mg+=am_plain(s,l) && crosses;
                }
                pl_all=pl;
                if(capped){pl=std::min(pl,cap("leaders"));mg=std::min(mg,pl);}
                if(require_leader && pl==0 && plain>0)continue;
            }
            bool held_plain=false;for(auto l:leaders)held_plain|=am_plain(s,l) && am_leader_side(s,l)==side && !eligible(obs_attr(s,l,"nation"));
            const bool stay=mg==0 && pl>0;
            const auto dst=mu_army(s,to);int64_t existing=0;bool enemy=false;
            if(dst!=missing)for(auto u:child_edges(s,obs_attr(s,dst,"units"))){if(ctx_unit_side(s,u)==side)++existing;else enemy=true;}
            if(enemy && !existing)continue;
            if(stationary)existing=0;
            int64_t nr=0,ne=0;bool isengard_elite=false;
            for(auto u:pool){
                const auto kind=obs_attr(s,u,"kind");nr+=enum_is(s,kind,"game_state.UnitKind","REGULAR");ne+=enum_is(s,kind,"game_state.UnitKind","ELITE");
                isengard_elite|=side==1 && saruman && enum_is(s,kind,"game_state.UnitKind","ELITE") && enum_is(s,obs_attr(s,u,"nation"),"game_state.Nation","ISENGARD");
            }
            if(capped){nr=std::min(nr,cap("regular"));ne=std::min(ne,cap("elite"));}
            const bool take_ok=!capped || cap("leaders")>0;
            const int64_t capL=capped?cap("leaders"):int64_t(-1);   // D2 mirror (static_r4): every leader figure a row moves (Nazgul included) must fit the pre-existing count - the executor rule of army_move.inc (rulebook p.25 "no figure is moved twice using the same action")
            for(int64_t mr=0;mr<=nr;++mr)for(int64_t me=0;me<=ne;++me){
                if(mr+me==0 || existing+mr+me>10 || (held_plain && mr+me==static_cast<int64_t>(own.size())))continue;
                for(bool take:{true,false}){
                    if(take ? !take_ok : (require_leader || !(stationary || mr+me<static_cast<int64_t>(own.size()) || stay)))continue;
                    auto parameters=[&](int64_t nl,int64_t nn,bool late_naz){
                        if(stationary){
                            int64_t remaining=nl>=0?nl:(take?static_cast<int64_t>(leaders.size()):0);
                            std::vector<uint64_t> selected;
                            for(bool is_plain:{true,false})for(auto leader:leaders)
                                if(remaining>0 && am_plain(s,leader)==is_plain && !am_saruman(s,leader) && am_leader_side(s,leader)==side){
                                    selected.push_back(leader);--remaining;
                                }
                            if(!me_companion_activation_available(s,t,from,selected))return;
                        }
                        std::vector<uint64_t> p={b.str("from"),b.str(akey),b.str("to"),b.str(to),b.str("n_regular"),b.integer(mr),b.str("n_elite"),b.integer(me),b.str("take_leaders"),rr_scalar(b.out,WOTR_BOOL,take)};
                        if(nl>=0){p.push_back(b.str("n_leaders"));p.push_back(b.integer(nl));}
                        if(nn>=0 && !late_naz){p.push_back(b.str("n_nazgul"));p.push_back(b.integer(nn));}
                        p.push_back(b.str("side"));p.push_back(b.str(side==0?"fp":"sh"));
                        if(nn>=0 && late_naz){p.push_back(b.str("n_nazgul"));p.push_back(b.integer(nn));}
                        rows.push_back(b.row(die,"army_move",p));
                    };
                    if(capL<0 || (take?pl_all+nazgul:int64_t(0))<=capL)parameters(-1,-1,false);
                    if(nazgul>0)for(int64_t nn=0;nn<=nazgul;++nn){
                        if(nn==(take?nazgul:0))continue;
                        if(require_leader && nn==0 && pl==0 && !(me>0 && isengard_elite))continue;
                        if(capL>=0 && (take?pl_all:int64_t(0))+nn>capL)continue;
                        parameters(-1,nn,false);
                    }
                    if(take && pl>1)for(int64_t nl=1;nl<pl;++nl){
                        if(!(stationary || mr+me<static_cast<int64_t>(own.size()) || nl>=mg))continue;
                        if(capL>=0 && nl>capL)continue;
                        if(nazgul>0)for(int64_t nn=0;nn<=nazgul;++nn){if(capL>=0 && nl+nn>capL)continue;parameters(nl,nn,true);}
                        else parameters(nl,-1,false);
                    }
                }
            }
        }
    }
}
std::unique_ptr<Snapshot> native_army_move_actions_ref(const Snapshot& s,const Snapshot& t,int32_t side){
    require(side==0 || side==1,"army menu side must be FP or SH");quiet_action_gate(s,t,false,true,false,false,false,false,false,false,false,false,false,true);
    am_Builder b;std::vector<uint64_t> rows;
    const auto pending=obs_attr(s,s.root,"pending_army_move_die"),owner=obs_attr(s,s.root,"pending_army_move_side",false);
    if(!ctx_none(s,pending) && (ctx_none(s,owner) || ctx_side(s,owner)==side)){
        const auto die=typed(s,pending,WOTR_INT,"pending army menu index must be integer").integer;
        const auto prev=obs_attr(s,s.root,"pending_army_move_from",false);
        amg_append_ref(b,rows,s,t,side,die,false,truth(s,prev)?sanitize_name(text(s,prev)):std::string{});
        rows.push_back(b.row(die,"pass",{}));
    }else{
        ring_waiver(s,side);
        const bool mouth_mode=am_mouth_available(s,side);
        const auto dice=child_edges(s,pa_pool(s,side));std::array<bool,7> seen{};
        for(size_t i=0;i<dice.size();++i){
            const auto die=dice[i];require(object_type(s,die)=="game_state.ActionDie","army menu requires ActionDie");
            if(truth(s,obs_attr(s,die,"used")))continue;
            const auto current=obs_attr(s,die,"current");if(ctx_none(s,current))continue;
            const int face=rr_face_index(enum_member(s,current,"game_state.ActionDieFace"));
            if(seen[face])continue;seen[face]=true;
            if(face==1 || face==4 || face==6)amg_append_ref(b,rows,s,t,side,i,false,{});
            else if(face==0)amg_append_ref(b,rows,s,t,side,i,true,{});
            else if(face==2 && mouth_mode){
                const size_t begin=rows.size();amg_append_ref(b,rows,s,t,side,i,false,{});
                for(size_t j=begin;j<rows.size();++j){const auto p=child_edges(b.out,rows[j])[2];auto edges=child_edges(b.out,p);edges.push_back(b.str("mouth_muster"));edges.push_back(rr_scalar(b.out,WOTR_BOOL,1));set_edges(b.out,p,edges);}
            }
        }
    }
    b.out.root=b.container(WOTR_LIST,rows);return std::make_unique<Snapshot>(std::move(b.out));
}
// Complete raw menus only for proved supported branches. A refusal means that
// completeness is unproved; callers must never replace it with a partial menu.
bool eaw_stable_siege_scope(const Snapshot&);
std::unique_ptr<Snapshot> fm_empty_ref(){am_Builder b;b.out.root=b.container(WOTR_LIST,{});return std::make_unique<Snapshot>(b.out);}
int fm_option_owner_ref(const Snapshot& s){
    for(const auto key:{"pending_card_option","pending_card_followup"}){
        const auto pending=obs_attr(s,s.root,key,false);if(!truth(s,pending))continue;
        const auto owner=field(s,pending,"side",false);if(!truth(s,owner))continue;
        auto tag=std::string(text(s,owner));for(auto& c:tag)if(c>='A' && c<='Z')c=c-'A'+'a';
        if(tag=="fp")return 0;if(tag=="sh")return 1;
    }
    return -1;
}
// Import one descriptor graph once, preserving all of its aliases. Row IDs
// are retained in order for later per-die interleaving, not concatenated.
std::vector<uint64_t> fm_import_ref(Snapshot& out,const Snapshot& source){
    const auto& root=typed(source,source.root,WOTR_LIST,"complete menu family must return a list");
    const uint64_t nodes=out.nodes.size(),edges=out.edges.size(),bytes=out.bytes.size();
    out.bytes.insert(out.bytes.end(),source.bytes.begin(),source.bytes.end());
    for(const auto edge:source.edges)out.edges.push_back(edge+nodes);
    for(auto value:source.nodes){
        if(value.tag==WOTR_STRING || value.tag==WOTR_BYTES || value.tag==WOTR_BIGINT)value.offset+=bytes;
        else if(value.tag>=WOTR_LIST && value.tag<=WOTR_ENUM)value.offset+=edges;
        if(value.flags)value.integer+=static_cast<int64_t>(nodes);
        out.nodes.push_back(value);
    }
    std::vector<uint64_t> rows;for(uint64_t i=0;i<root.count;++i)rows.push_back(source.edges[root.offset+i]+nodes);return rows;
}
int64_t fm_die_ref(const Snapshot& s,uint64_t row){const auto& n=s.nodes[row];return s.nodes[s.edges[n.offset]].integer;}
std::string_view fm_kind_ref(const Snapshot& s,uint64_t row){return text(s,s.edges[s.nodes[row].offset+1]);}
std::unique_ptr<Snapshot> native_full_actions_ref(const Snapshot& s,const Snapshot& t,int32_t side){
    if(fm_stale_discard_cleanup_due(s,t,side)){auto normalized=fm_prepare_menu_context(s,t,side);return native_full_actions_ref(*normalized,t,side);}
    require(side==0 || side==1,"complete menu side must be FP or SH");attributes(s,s.root,"game_state.GameState");
    const auto combat=obs_attr(s,s.root,"pending_combat");
    if(!ctx_none(s,combat)){
        const auto owner=field(s,combat,"deciding_side",false);
        if(!ctx_none(s,owner) && ctx_side(s,owner)!=side)return fm_empty_ref();
        return native_combat_pending_actions(s,t,side);
    }
    if(!ctx_none(s,obs_attr(s,s.root,"pending_guide_change")))return native_guide_actions(s,side);
    const auto hunt=obs_attr(s,s.root,"pending_hunt");
    if(!ctx_none(s,hunt)){
        if(text_is(s,field(s,hunt,"phase"),"hobbit_place"))return native_hobbit_place_actions(s,t);
        return native_hunt_damage_actions(s);
    }
    // Frozen raw menus return Hunt and reveal rows independently of the
    // caller's side; the driver owns FP routing. Preserve that exact behavior.
    if(truth(s,obs_attr(s,s.root,"_reveal_place_due",false)) && !fr_pending(s)){
        auto reveal=native_reveal_actions(s,t);if(reveal->nodes[reveal->root].count)return reveal;
    }
    const auto discard=obs_attr(s,s.root,"pending_discard",false);
    if(!ctx_none(s,discard) && dc_owner(s)==side && fm_option_owner_ref(s)!=side){
        const auto hand=tc_hand(s,side); // Package W: an empty hand yields the frozen empty discard menu
        am_Builder b;std::vector<uint64_t> rows;std::set<std::string> seen;
        for(auto card:child_edges(s,hand)){
            const std::string id(text(s,card));if(seen.insert(id).second)rows.push_back(b.row(-1,"discard_card",{
                b.str("card_id"),b.str(id),b.str("side"),b.str(side==0?"fp":"sh")}));
        }
        b.out.root=b.container(WOTR_LIST,rows);return std::make_unique<Snapshot>(b.out);
    }
    if(truth(s,obs_attr(s,s.root,"_declare_phase",false)))return native_fellowship_declare_actions(s,t,side);
    if(truth(s,obs_attr(s,s.root,"_eye_allocation_phase",false)))return eye_allocation_actions(s,side);
    if(!ctx_none(s,obs_attr(s,s.root,"pending_card_followup",false)))return native_card_army_followup_actions(s,t,side);
    const auto option=obs_attr(s,s.root,"pending_card_option",false);
    if(ecc_pending(s))return native_card_casualty_actions(s,t,side);
    if(edm_pending(s))return native_dead_men_choices(s,t,side);
    if(eeg_pending(s))return native_eagles_choices(s,t,side);
    if(eds_pending(s))return native_dreadful_choices(s,t,side);
    if(esp_pending(s))return native_spirit_choices(s,t,side);
    if(eff_pending(s))return native_faramir_choice_actions(s,t,side);
    if(esm_pending(s))return native_misty_choices(s,t,side);
    if(esra_pending(s))return native_shadow_allocation_choices(s,t,side);
    if(emk_pending(s))return native_many_kings_choices(s,t,side);
    if(edf_pending(s))return native_denethor_choices(s,t,side);
    if(eaw_pending(s))return native_another_way_choices(s,t,side);
    if(esw_pending(s))return native_swifter_choices(s,t,side);
    if(eea_pending(s))return native_ents_choices(s,t,side);
    if(esc_pending(s))return native_stormcrow_choices(s,t,side);
    if(eal_pending(s))return native_alone_choices(s,t,side);
    if(etb_pending(s))return native_back_again_choices(s,t,side);
    if(ehe_pending(s))return native_horde_choices(s,t,side);
    if(erg_pending(s))return native_rage_choices(s,t,side);
    if(eco_pending(s))return native_corsairs_choices(s,t,side);
    if(ehuf_pending(s))return native_help_unlooked_choice_actions(s,t,side);
    if(egr_pending(s))return native_grond_choice_actions(s,t,side);
    if(euh_pending(s))return native_uruk_choice_actions(s,t,side);
    if(etd_pending(s))return native_through_day_choice_actions(s,side);
    if(epw_pending(s))return native_woses_choice_actions(s,side);
    if(egw_pending(s))return native_gwaihir_choice_actions(s,side);
    if(ecw_pending(s))return native_cruel_weather_choices(s,t,side);
    if(!ctx_none(s,option)&&text_is(s,field(s,option,"card_id",false),"__palantir_draw__"))return native_palantir_draw_actions(s,side);
    if(!ctx_none(s,option) && text_is(s,field(s,option,"card_name",false),"Pits of Mordor"))return native_pits_choice_actions(s,t,side);
    if(!ctx_none(s,option) && text_is(s,field(s,option,"card_id",false),"__gandalf_draw__"))return native_gandalf_draw_actions(s,side);
    if(!ctx_none(s,option) && text_is(s,field(s,option,"card_id",false),"__companion_moves__")){
        require(side==0,"complete companion continuation belongs to FP");return native_companion_chain_actions(s);
    }
    if(egc_pending(s))return native_grey_company_choices(s,t,side);
    if(ehs_pending(s))return native_house_stewards_choices(s,t,side);
    if(ebc_pending(s))return native_black_captain_choices(s,t,side);
    if(erw_pending(s))return native_ringwraiths_choice_actions(s,t,side);
    if(enh_pending(s))return native_nazgul_hunt_choice_actions(s,side);
    if(!ctx_none(s,option) && text_is(s,field(s,option,"card_id",false),"__nazgul_moves__"))return native_nazgul_chain_actions(s,side);
    if(!ctx_none(s,option) && text_is(s,field(s,option,"card_name",false),"The Lidless Eye"))return native_dice_event_choice_actions(s,side);
    if(!ctx_none(s,option) && text_is(s,field(s,option,"card_name",false),"Wisdom of Elrond"))return native_misc_event_choice_actions(s,side);
    if(ehd_pending(s)){require(side==0,"Hunt event choice belongs to FP");return native_hunt_event_choice_actions(s);}
    if(el_pending(s)){require(side==0,"Lure choice belongs to FP");return native_lure_choice_actions(s);}
    if(ew_pending(s))return native_worn_choice_actions(s,side);
    if(ec_pending(s))return native_crebain_choice_actions(s,side);
    if(efm_pending(s))return native_companion_event_choice_actions(s,side);
    if(etsm_pending(s))return native_shadow_is_moving_choice_actions(s,side);
    if(esl_pending(s))return native_shadow_lengthens_choice_actions(s,side);
    if(!ctx_none(s,option) && (text_is(s,field(s,option,"card_name",false),"A Power too Great") || text_is(s,field(s,option,"card_name",false),"The Power of Tom Bombadil")))return native_power_payment_actions(s,t,side);
    if(!ctx_none(s,option) && (text_is(s,field(s,option,"card_name",false),"Cirdan's Ships") || text_is(s,field(s,option,"card_name",false),"Riders of Th\xC3\xA9oden") || text_is(s,field(s,option,"card_name",false),"\xC3\x89omer son of \xC3\x89omund")))return native_target_recruit_choice_actions(s,t,side);
    if(!ctx_none(s,option) && (text_is(s,field(s,option,"card_name",false),"Half-orcs and Goblin-men") || text_is(s,field(s,option,"card_name",false),"Olog-hai")))return native_half_orcs_choice_actions(s,t,side);
    if(!ctx_none(s,option) && text_is(s,field(s,option,"card_name",false),"Shadows Gather"))return native_shadows_gather_choice_actions(s,t,side);
    if(!ctx_none(s,option)&&text_is(s,field(s,option,"card_name",false),"Hill-trolls"))return native_hill_trolls_choice_actions(s,t,side);
    if(!ctx_none(s,option)&&text_is(s,field(s,option,"card_name",false),"Guards of the Citadel"))return native_recruit_event_choice_actions(s,t,side);
    if(!ctx_none(s,option)&&text_is(s,field(s,option,"card_id",false),"__overstack__"))return native_overstack_choice_actions(s,side);
    if(!ctx_none(s,option) && (truth(s,field(s,option,"hunt_reduction",false)) ||
        text_is(s,field(s,option,"card_name",false),"Wizard's Staff") || text_is(s,field(s,option,"card_name",false),"Mithril Coat and Sting") || text_is(s,field(s,option,"card_name",false),"Balrog of Moria"))){
        const auto family=hpr_family(s);require(side==(family=="balrog"?1:0),"Hunt protection menu requested for wrong deciding side");return native_hunt_protection_actions(s);
    }
    require(ctx_none(s,obs_attr(s,s.root,"pending_card_option",false)),"complete card-option menu is not ported");
    const auto army=obs_attr(s,s.root,"pending_army_move_die"),army_owner=obs_attr(s,s.root,"pending_army_move_side",false);
    if(!ctx_none(s,army) && (ctx_none(s,army_owner) || ctx_side(s,army_owner)==side))return native_army_move_actions_ref(s,t,side);
    const auto muster=obs_attr(s,s.root,"pending_muster_die"),muster_owner=obs_attr(s,s.root,"pending_muster_side",false);
    if(!ctx_none(s,muster) && (ctx_none(s,muster_owner) || ctx_side(s,muster_owner)==side))return native_muster_actions(s,t,side);

    const bool stable_siege_menu=truth(s,obs_attr(s,s.root,"besieged_strongholds",false));
    // Package P: a quiet turn-start state before the action roll (no pending decision, no phase flag, dice_rolled false -
    // the child of the last hand-limit discard inside a discard search, ai.py:16995-17010 / 17060-17072). The frozen
    // get_legal_actions (ai.py:13497-13607) enumerates nothing die-funded there (_get_unused_dice is empty), no elven_ring
    // / dwarven_ring row (both need a rolled face) and no pass; only the bid-compensation tokens (_get_token_actions,
    // die_idx -1) remain. The same quiet-board proof applies with the roll clause waived (allow_unrolled).
    if(!truth(s,obs_attr(s,s.root,"dice_rolled"))){
        if(stable_siege_menu)quiet_stable_action_gate(s,t,false,false,false,true);else quiet_action_gate(s,t,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,true);
        ring_waiver(s,side);
        return native_token_actions(s,t,side);
    }
    if(stable_siege_menu)quiet_stable_action_gate(s,t);else quiet_action_gate(s,t);
    ring_waiver(s,side);
    const auto dice=child_edges(s,pa_pool(s,side));std::vector<std::pair<size_t,int>> unused;bool army_face=false,muster_face=false,character_face=false,will_face=false;
    for(size_t i=0;i<dice.size();++i){
        const auto die=dice[i];require(object_type(s,die)=="game_state.ActionDie","complete menu requires ActionDie");
        const auto current=obs_attr(s,die,"current");if(truth(s,obs_attr(s,die,"used")) || ctx_none(s,current))continue;
        const int face=rr_face_index(enum_member(s,current,"game_state.ActionDieFace"));
        require(face!=6 || side==0,"complete menu does not admit nonphysical Shadow Will dice");
        unused.emplace_back(i,face);army_face|=face==0 || face==1 || face==4 || face==6;muster_face|=face==2 || face==4 || face==6;character_face|=face==0 || face==6;will_face|=face==6;
    }
    const auto fs=obs_attr(s,s.root,"fellowship"),guide=obs_attr(s,fs,"guide");
    const bool strider_hide=side==0 && !truth(s,obs_attr(s,fs,"hidden")) && !ctx_none(s,guide) &&
        std::string(text(s,obs_attr(s,guide,"name"))).find("Strider")!=std::string::npos;
    const bool mouth_funding=am_mouth_available(s,side);
    am_Builder b;std::vector<uint64_t> rows;
    auto import=[&](std::unique_ptr<Snapshot> menu){return fm_import_ref(b.out,*menu);};
    const auto political=import(political_ring_actions(s,t,side));
    const auto recruits=import(native_muster_actions(s,t,side));
    const auto moves=(army_face || mouth_funding)?import(native_army_move_actions_ref(s,t,side)):std::vector<uint64_t>{};
    const auto saruman=side==1 && muster_face?import(saruman_muster_actions(s,t,side)):std::vector<uint64_t>{};
    const auto minions=side==1 && muster_face?import(native_minion_muster_actions(s,t,side)):std::vector<uint64_t>{};
    const auto companions=side==0?import(native_companion_political_actions(s,t,side)):std::vector<uint64_t>{};
    const auto draws=import(native_draw_card_actions(s,t,side));
    const auto mouth=character_face && side==1?import(native_mouth_move_actions(s,t,side)):std::vector<uint64_t>{};
    const auto nazgul=character_face && side==1?import(native_nazgul_move_actions(s,t,side)):std::vector<uint64_t>{};
    const auto special=will_face?import(native_will_special_actions(s,t,side)):std::vector<uint64_t>{};
    auto append=[&](const std::vector<uint64_t>& family,size_t die){for(auto row:family)if(fm_die_ref(b.out,row)==static_cast<int64_t>(die))rows.push_back(row);};
    auto events=[&](size_t die,const char* category,bool mouth_mode=false){
        auto menu=native_play_event_for_die(s,t,side,die,category);
        for(auto row:child_edges(*menu,menu->root)){
            const auto parts=child_edges(*menu,row);const auto id=field(*menu,parts[2],"card_id");
            const auto card=ev_card(t,text(*menu,id));
            require(ctx_side(t,obs_attr(t,card,"side"))==side,"complete menu has a wrong-side card in acting hand");
        }
        const auto added=import(std::move(menu));
        for(auto row:added){
            if(mouth_mode){const auto p=child_edges(b.out,row)[2];auto edges=child_edges(b.out,p);edges.push_back(b.str("mouth_muster"));edges.push_back(rr_scalar(b.out,WOTR_BOOL,1));set_edges(b.out,p,edges);}
            rows.push_back(row);
        }
    };
    std::array<bool,7> seen{};const bool power_kill=side==1 && ept_first(s)!=missing;
    const bool palantir_kill=side==0 && hpr_card(s,"PALANTIR_OF_ORTHANC")!=missing;
    auto attacks=[&](size_t index,bool require_leader){
        const auto family=import(native_army_attack_for_die(s,t,side,index,require_leader));rows.insert(rows.end(),family.begin(),family.end());
    };
    for(const auto& entry:unused){
        const auto index=entry.first;const int face=entry.second;if(seen[face])continue;seen[face]=true;
        if(face==1 || face==4 || face==6){append(moves,index);attacks(index,false);if(face!=6)events(index,"army");}
        if(face==2 || face==4 || face==6){append(political,index);append(recruits,index);append(saruman,index);append(minions,index);if(face!=6)events(index,"muster");}
        if(face==2 && mouth_funding){
            append(moves,index);const auto family=import(native_army_attack_for_die(s,t,side,index,false));
            for(auto row:family){const auto p=child_edges(b.out,row)[2];auto edges=child_edges(b.out,p);edges.push_back(b.str("mouth_muster"));edges.push_back(rr_scalar(b.out,WOTR_BOOL,1));set_edges(b.out,p,edges);rows.push_back(row);}
            events(index,"army",true);
        }
        if(power_kill){const auto removals=import(native_power_kill_for_die(s,t,side,index));rows.insert(rows.end(),removals.begin(),removals.end());}
        if(side==0&&hpr_card(s,"DENETHORS_FOLLY")!=missing){const auto removals=import(native_denethor_kill_for_die(s,side,index));rows.insert(rows.end(),removals.begin(),removals.end());}
        if(palantir_kill){const auto removals=import(native_palantir_kill_for_die(s,side,index));rows.insert(rows.end(),removals.begin(),removals.end());}
        append(companions,index);
        if(face==0 || face==6){
            if(side==0){
                const auto fellowship=import(native_fellowship_actions(s,index));rows.insert(rows.end(),fellowship.begin(),fellowship.end());
                const auto separate=import(native_companion_separation_actions(s,t,index));rows.insert(rows.end(),separate.begin(),separate.end());
                const auto figures=import(native_companion_move_actions(s,t,index));rows.insert(rows.end(),figures.begin(),figures.end());
            }else{append(nazgul,index);append(mouth,index);}
            if(face==0){append(moves,index);attacks(index,true);events(index,"character");}
        }
        if(face==3 || face==6){append(draws,index);events(index,"event");}
        if(face==6)append(special,index);
    }
    if(strider_hide){
        std::array<bool,7> seen_hide{};
        for(const auto& entry:unused){const auto face=entry.second;if(face==0 || face==6 || seen_hide[face])continue;seen_hide[face]=true;
            rows.push_back(b.row(entry.first,"fellowship_hide",{b.str("strider_ability"),rr_scalar(b.out,WOTR_BOOL,1)}));
        }
    }
    for(auto row:political)if(fm_kind_ref(b.out,row)=="elven_ring")rows.push_back(row);
    for(auto row:import(native_token_actions(s,t,side)))rows.push_back(row);
    for(auto row:import(native_dwarven_ring_actions(s,side)))rows.push_back(row);
    size_t other_unused=0;for(auto die:child_edges(s,pa_pool(s,1-side)))
        if(!truth(s,obs_attr(s,die,"used")) && !ctx_none(s,obs_attr(s,die,"current")))++other_unused;
    // 1.272 mirror (static_r5): ai.get_legal_actions - USER 2026-09-10 "if the decks are empty, and the player only has palantirs and nothing playable, allow the option to kill the die for no benefit" (rulebook 1678-1680 restored in exactly this situation; 1.67 stands elsewhere): every unused die shows the Event face, both own decks are empty, no event row for that die -> ONE skip row for the first unused Event die
    if(!unused.empty()){bool all_event=true;for(const auto& entry:unused)all_event&=entry.second==3;
        if(all_event){const size_t kd=unused.front().first;
            const bool decks_empty=!truth(s,obs_attr(s,s.root,side==0?"fp_char_deck":"sh_char_deck",false)) && !truth(s,obs_attr(s,s.root,side==0?"fp_strat_deck":"sh_strat_deck",false));
            if(decks_empty){auto kill_menu=native_play_event_for_die(s,t,side,kd,"event");
                if(child_edges(*kill_menu,kill_menu->root).empty())rows.push_back(b.row(static_cast<int64_t>(kd),"skip",{b.str("face"),b.str("EVENT"),b.str("side"),b.str(side==0?"fp":"sh")}));}
        }
    }
    // MC6 mirror (production turn_loop_repair_r1 2026-09-10, ai.py get_legal_actions AND its compiled twin _segment_get_legal_actions, the only ai.py
    // difference static_r5 -> src/; package Z 2026-09-11): "once the opponent has no legal action, this side plays its remaining actions consecutively.
    // A pass cannot hand control back to an empty menu forever" - the ordinary pass row is offered ONLY when the opponent's COMPLETE menu of this
    // same state is non-empty (Python: get_legal_actions(clone_state(state), other_side); the native builder never mutates the graph, so this
    // copy's own complete-menu function is probed on the same snapshot - a probe refusal propagates, an unproved opponent menu leaves the pass
    // row unproved).  Recursion stops here exactly as in Python: the opponent holds MORE unused dice, so its own pass condition is false.
    if(!unused.empty() && unused.size()<other_unused){
        auto opponent_menu=native_full_actions_ref(s,t,1-side);
        if(!child_edges(*opponent_menu,opponent_menu->root).empty())rows.push_back(b.row(-1,"pass",{}));
    }
    // Rulebook p21: if no ordinary action exists, spend one non-Eye die
    // without effect. Pending decisions returned before this ordinary menu.
    if(rows.empty())for(const auto& entry:unused)if(entry.second!=5){
        rows.push_back(b.row(entry.first,"skip",{b.str("face"),b.str(rr_face_names[entry.second]),b.str("side"),b.str(side==0?"fp":"sh")}));break;
    }
    // Compose the original complete raw graph first. Refuse the whole menu
    // if any offered action has no proved stable-siege executor. Never filter
    // such rows out, or infer executor support from raw-menu enumeration.
    if(stable_siege_menu)for(auto row:rows){
        const auto kind=fm_kind_ref(b.out,row);
        const bool ordinary=kind=="advance_political"||kind=="elven_ring"||
            kind=="muster_recruit"||kind=="muster_elite"||kind=="muster_minion"||
            kind=="army_move"||kind=="army_attack"||kind=="draw_card"||
            kind=="fellowship_move"||kind=="fellowship_hide"||
            kind=="companion_separate"||kind=="companion_move"||kind=="nazgul_move"||
            kind=="mouth_move"||kind=="use_token"||kind=="dwarven_ring"||
            kind=="crown_aragorn"||kind=="transform_gandalf"||kind=="pass"||kind=="skip";   // 1.272 mirror (static_r5): a die kill is an ordinary (proved) executor
        if(ordinary)continue;
        // Paths has independent mixed-siege origin/partial-departure and
        // exact saved494 public proof, including its deferred choice phase.
        if(kind=="play_event"){
            const auto params=child_edges(b.out,row)[2];
            const auto definition=ev_card(t,text(b.out,field(b.out,params,"card_id")));
            if(side==0&&(efm_supported(t,definition)||eck_supported(t,definition)||ehs_supported(t,definition)||egc_supported(t,definition)||ept_supported(t,definition)||text_is(t,obs_attr(t,definition,"card_name"),"Mithril Coat and Sting") || text_is(t,obs_attr(t,definition,"card_name"),"Axe and Bow")||etd_supported(t,definition)||etb_supported(t,definition)||text_is(t,obs_attr(t,definition,"card_name"),"Celeborn's Galadhrim") || text_is(t,obs_attr(t,definition,"card_name"),"Grimbeorn the Old Son of Beorn") || text_is(t,obs_attr(t,definition,"card_name"),"King Brand's Men") || text_is(t,obs_attr(t,definition,"card_name"),"Swords in Eriador") || text_is(t,obs_attr(t,definition,"card_name"),"Elven Rope") || text_is(t,obs_attr(t,definition,"card_name"),"\x53\x6d\xc3\xa9\x61\x67\x6f\x6c\x20\x48\x65\x6c\x70\x73\x20\x4e\x69\x63\x65\x20\x4d\x61\x73\x74\x65\x72") || text_is(t,obs_attr(t,definition,"card_name"),"Bilbo's Song") || text_is(t,obs_attr(t,definition,"card_name"),"Mirror of Galadriel")||(eaw_supported(t,definition)&&eaw_stable_siege_scope(s))||text_is(t,obs_attr(t,definition,"card_name"),"Imrahil of Dol Amroth")||text_is(t,obs_attr(t,definition,"card_name"),"Guards of the Citadel")||(text_is(t,obs_attr(t,definition,"card_name"),"Dain Ironfoot's Guard") || text_is(t,obs_attr(t,definition,"card_name"),"The Red Arrow") || text_is(t,obs_attr(t,definition,"card_name"),"Thranduil's Archers"))||text_is(t,obs_attr(t,definition,"card_name"),"Horn of Gondor")||text_is(t,obs_attr(t,definition,"card_name"),"We Prove Swifter")||text_is(t,obs_attr(t,definition,"card_name"),"Kindred of Glorfindel")||eal_supported(t,definition)||text_is(t,obs_attr(t,definition,"card_name"),"The Last Battle")||eea_supported(t,definition)||ehuf_supported(t,definition)||edm_supported(t,definition)||eeg_supported(t,definition)||epw_supported(t,definition)||esp_supported(t,definition)||text_is(t,obs_attr(t,definition,"card_name"),"Phial of Galadriel")||text_is(t,obs_attr(t,definition,"card_name"),"Wizard's Staff")||text_is(t,obs_attr(t,definition,"card_name"),"Elven Cloaks")||em_supported(t,definition)||eff_supported(t,definition)||egw_supported(t,definition)||ert_kind(t,definition)==1||ert_kind(t,definition)==2||ert_kind(t,definition)==3))continue;
            if(side==1&&(text_is(t,obs_attr(t,definition,"card_name"),"Orc Patrol")||text_is(t,obs_attr(t,definition,"card_name"),"Isildur's Bane")||text_is(t,obs_attr(t,definition,"card_name"),"The Shadow is Moving")||text_is(t,obs_attr(t,definition,"card_name"),"Hill-trolls")||text_is(t,obs_attr(t,definition,"card_name"),"The Lidless Eye")||(text_is(t,obs_attr(t,definition,"card_name"),"Worn with Sorrow and Toil") || text_is(t,obs_attr(t,definition,"card_name"),"Flocks of Crebain") || text_is(t,obs_attr(t,definition,"card_name"),"Balrog of Moria") || text_is(t,obs_attr(t,definition,"card_name"),"Wormtongue"))||!enh_kind(t,definition).empty()||text_is(t,obs_attr(t,definition,"card_name"),"Threats and Promises")||(esr_kind(t,definition)>0 && esr_kind(t,definition)<5)||(text_is(t,obs_attr(t,definition,"card_name"),"Shelob's Lair") || text_is(t,obs_attr(t,definition,"card_name"),"The Ring is Mine!") || text_is(t,obs_attr(t,definition,"card_name"),"On On They Went") || text_is(t,obs_attr(t,definition,"card_name"),"Morgul Wound") || text_is(t,obs_attr(t,definition,"card_name"),"The Day Without Dawn") || text_is(t,obs_attr(t,definition,"card_name"),"Stormcrow") || text_is(t,obs_attr(t,definition,"card_name"),"Rage of the Dunlendings") || text_is(t,obs_attr(t,definition,"card_name"),"Return of the Witch-king") || text_is(t,obs_attr(t,definition,"card_name"),"Olog-hai") || text_is(t,obs_attr(t,definition,"card_name"),"Many Kings to the Service of Mordor") || text_is(t,obs_attr(t,definition,"card_name"),"The King is Revealed") || text_is(t,obs_attr(t,definition,"card_name"),"Shadows on the Misty Mountains") || text_is(t,obs_attr(t,definition,"card_name"),"Horde from the East") || text_is(t,obs_attr(t,definition,"card_name"),"Pits of Mordor"))||esl_supported(t,definition)||erw_supported(t,definition)||ebc_supported(t,definition)||esg_supported(t,definition)||text_is(t,obs_attr(t,definition,"card_name"),"The Palant\xc3\xadr of Orthanc")||text_is(t,obs_attr(t,definition,"card_name"),"Half-orcs and Goblin-men")||text_is(t,obs_attr(t,definition,"card_name"),"Give it to Uss!")||ehd_kind(t,definition)=="foul_thing"||ehd_kind(t,definition)=="breaking"||el_supported(t,definition)||egr_supported(t,definition)||eco_supported(t,definition)||erv_supported(t,definition)||eds_supported(t,definition)||euh_supported(t,definition)||ecw_supported(t,definition)||text_is(t,obs_attr(t,definition,"card_name"),"Denethor's Folly")))continue;
        }
        if(kind=="discard_table_card"){
            const auto params=child_edges(b.out,row)[2];const auto folly=hpr_card(s,"DENETHORS_FOLLY");if(side==0&&folly!=missing&&text_is(b.out,field(b.out,params,"card_id"),text(s,obs_attr(s,folly,"card_id"))))continue;const auto card=hpr_card(s,"PALANTIR_OF_ORTHANC");
            if(side==0&&card!=missing&&text_is(b.out,field(b.out,params,"card_id"),text(s,obs_attr(s,card,"card_id"))))continue;
            // Both removal entry and its learned pair-payment continuation
            // have independent public mixed-siege execution proof.
            bool proved_removal=false;
            if(side==1)for(const char* type:{"POWER_TOO_GREAT","TOM_BOMBADIL"}){
                const auto target=hpr_card(s,type);
                proved_removal|=target!=missing&&text_is(b.out,field(b.out,params,"card_id"),text(s,obs_attr(s,target,"card_id")));
            }
            if(proved_removal)continue;
        }
        (void)kind; // Package W: the complete stable-siege allowlist is lifted (package L: 0 firings / 333,668 rows)
    }
    b.out.root=b.container(WOTR_LIST,rows);return std::make_unique<Snapshot>(b.out);
}
bool fm_check_same(const Snapshot& a,uint64_t x,const Snapshot& b,uint64_t y,size_t depth){
    require(depth<512,"WOTR_NATIVE_FULLMENU_CHECK: menu graph too deep");
    const auto& p=a.nodes[x];const auto& q=b.nodes[y];
    if(p.tag!=q.tag||p.flags!=q.flags)return false;
    if(p.flags==1&&!fm_check_same(a,static_cast<uint64_t>(p.integer),b,static_cast<uint64_t>(q.integer),depth+1))return false;
    if(p.tag<=WOTR_FLOAT){if(p.tag==WOTR_FLOAT){uint64_t u,v;std::memcpy(&u,&p.real,8);std::memcpy(&v,&q.real,8);return u==v;}return p.integer==q.integer;}
    if(p.tag==WOTR_STRING||p.tag==WOTR_BYTES||p.tag==WOTR_BIGINT)return p.count==q.count&&(!p.count||std::memcmp(a.bytes.data()+p.offset,b.bytes.data()+q.offset,static_cast<size_t>(p.count))==0);
    if(p.count!=q.count)return false;
    for(uint64_t i=0;i<p.count;++i)if(!fm_check_same(a,a.edges[p.offset+i],b,b.edges[q.offset+i],depth+1))return false;
    return true;
}
std::string fm_check_label(const Snapshot& m,uint64_t row){
    const auto& n=m.nodes[row];if(n.tag!=WOTR_TUPLE||n.count!=3)return "<non-row>";
    const auto die=m.edges[n.offset],kind=m.edges[n.offset+1];std::string out="(";
    out+=m.nodes[die].tag==WOTR_INT?std::to_string(m.nodes[die].integer):std::string("?");out+=", ";
    out+=m.nodes[kind].tag==WOTR_STRING?std::string(text(m,kind)):std::string("?");return out+")";
}
void fm_check_compare(const Snapshot& fast,const Snapshot& ref){
    const auto& fr=fast.nodes[fast.root];const auto& rr=ref.nodes[ref.root];
    if(fr.tag!=WOTR_LIST||rr.tag!=WOTR_LIST||fr.count!=rr.count)throw std::logic_error("WOTR_NATIVE_FULLMENU_CHECK: row count differs: optimized "+std::to_string(fr.count)+" reference "+std::to_string(rr.count));
    for(uint64_t i=0;i<fr.count;++i){const auto x=fast.edges[fr.offset+i],y=ref.edges[rr.offset+i];
        if(!fm_check_same(fast,x,ref,y,0))throw std::logic_error("WOTR_NATIVE_FULLMENU_CHECK: row "+std::to_string(i)+" differs: optimized "+fm_check_label(fast,x)+" reference "+fm_check_label(ref,y));}
    // Graph-level identity (the structural pass above ignores node ids): every speed-pass step claims a byte-identical menu
    // graph, so the node sequence, the edge sequence, the byte store and the root must match exactly.  A content-preserving
    // node-order change (e.g. an intern hoisted out of a row loop) is reported here and nowhere else.
    if(fast.nodes.size()!=ref.nodes.size()||fast.edges.size()!=ref.edges.size()||fast.bytes.size()!=ref.bytes.size()||fast.root!=ref.root)
        throw std::logic_error("WOTR_NATIVE_FULLMENU_CHECK: graph layout differs: optimized nodes/edges/bytes/root "+std::to_string(fast.nodes.size())+"/"+std::to_string(fast.edges.size())+"/"+std::to_string(fast.bytes.size())+"/"+std::to_string(fast.root)+" reference "+std::to_string(ref.nodes.size())+"/"+std::to_string(ref.edges.size())+"/"+std::to_string(ref.bytes.size())+"/"+std::to_string(ref.root));
    for(size_t i=0;i<fast.nodes.size();++i){const auto& p=fast.nodes[i];const auto& q=ref.nodes[i];uint64_t u,v;std::memcpy(&u,&p.real,8);std::memcpy(&v,&q.real,8);
        if(p.tag!=q.tag||p.flags!=q.flags||p.integer!=q.integer||u!=v||p.offset!=q.offset||p.count!=q.count)throw std::logic_error("WOTR_NATIVE_FULLMENU_CHECK: node "+std::to_string(i)+" differs from the reference graph (node order or layout): optimized tag "+std::to_string(p.tag)+" integer "+std::to_string(p.integer)+" reference tag "+std::to_string(q.tag)+" integer "+std::to_string(q.integer));}
    if(fast.edges!=ref.edges)throw std::logic_error("WOTR_NATIVE_FULLMENU_CHECK: edge sequence differs from the reference graph");
    if(fast.bytes!=ref.bytes)throw std::logic_error("WOTR_NATIVE_FULLMENU_CHECK: byte store differs from the reference graph");
    fm_check_menus_checked++;fm_check_rows_checked+=fr.count;
}
std::unique_ptr<Snapshot> native_full_actions(const Snapshot& s,const Snapshot& t,int32_t side){
    if(!fm_check_active)return fm_native_full_actions_impl(s,t,side);
    std::unique_ptr<Snapshot> fast;std::string fast_error;std::exception_ptr pending;
    try{fast=fm_native_full_actions_impl(s,t,side);}catch(const std::exception& e){fast_error=e.what();if(fast_error.empty())fast_error="<empty>";pending=std::current_exception();}
    std::unique_ptr<Snapshot> ref;std::string ref_error;
    try{ref=native_full_actions_ref(s,t,side);}catch(const std::exception& e){ref_error=e.what();if(ref_error.empty())ref_error="<empty>";}
    if(fast_error!=ref_error)throw std::logic_error("WOTR_NATIVE_FULLMENU_CHECK: exception mismatch: optimized='"+fast_error+"' reference='"+ref_error+"'");
    if(pending)std::rethrow_exception(pending);
    fm_check_compare(*fast,*ref);return fast;
}
