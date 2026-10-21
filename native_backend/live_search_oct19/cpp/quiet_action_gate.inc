
// Gate-local storage only: borrowed names remain inside this immutable call.
// The unusual large/noncanonical cases retain owned-set fallback semantics.
template<class T> struct qg_BoundedSeen {
    std::array<T,256> slots{};
    std::array<uint8_t,256> occupied{};
    size_t used=0;
    std::unique_ptr<std::set<T>> overflow;
    template<class F> void each(F f)const{
        if(overflow){for(const auto& value:*overflow)f(value);}
        else for(size_t i=0;i<slots.size();++i)if(occupied[i])f(slots[i]);
    }
    bool insert(const T& value){
        if(overflow)return overflow->insert(value).second;
        auto i=std::hash<T>{}(value)&255;
        while(occupied[i]){if(slots[i]==value)return false;i=(i+1)&255;}
        if(used==128){
            auto owned=std::make_unique<std::set<T>>();
            each([&](const T& old){owned->insert(old);});
            overflow=std::move(owned);return overflow->insert(value).second;
        }
        slots[i]=value;occupied[i]=1;++used;return true;
    }
};
bool qg_ascii_canonical(std::string_view value){
    bool previous_space=true;
    for(unsigned char c:value){
        if((c>='a'&&c<='z')||(c>='0'&&c<='9'))previous_space=false;
        else if(c==' '&&!previous_space)previous_space=true;
        else return false;
    }
    return value.empty()||!previous_space;
}
bool qg_canonical_name(std::string_view value){
    // Noncanonical input still runs the original UTF-8 parser, including errors.
    return qg_ascii_canonical(value)||sanitize_name(value)==value;
}
struct qg_NameSeen {
    qg_BoundedSeen<std::string_view> borrowed;
    std::array<std::array<char,64>,128> normalized_storage;
    size_t normalized_count=0;
    std::unique_ptr<std::set<std::string>> owned;
    bool insert(std::string_view value,bool source_owned){
        if(owned)return owned->emplace(value).second;
        if(source_owned)return borrowed.insert(value);
        if(value.size()<=64&&normalized_count<normalized_storage.size()){
            auto& saved=normalized_storage[normalized_count];
            if(!value.empty())std::memcpy(saved.data(),value.data(),value.size());
            const bool added=borrowed.insert(std::string_view(saved.data(),value.size()));
            if(added)++normalized_count;return added;
        }
        auto saved=std::make_unique<std::set<std::string>>();
        borrowed.each([&](std::string_view old){saved->emplace(old);});
        owned=std::move(saved);return owned->emplace(value).second;
    }
};

// Conservative support gate for one complete ordinary political action.
// Included in snapshot.cpp's private namespace after observation_tails.inc.
// Reads only. Payment, political legality, clone semantics and bookkeeping
// assignments remain the caller's responsibility (see QUIET_ACTION_GATE.md).
void cm_validate_chain(const Snapshot& s);
void hpr_card_gate(const Snapshot& s);
uint64_t eo_pending(const Snapshot& s,int side);
// Package P: allow_unrolled waives ONLY the rolled-dice clause (a quiet turn-start state before the action roll: the
// frozen complete menu is empty there apart from bid-compensation tokens, ai.py:13497-13607, and the frozen use_token
// executor reads no dice); every other clause of the gate applies unchanged.
void quiet_action_gate(const Snapshot& s,const Snapshot& tables,bool allow_muster=false,bool allow_army=false,bool allow_eye=false,bool allow_discard=false,bool allow_political_table=false,bool allow_declare=false,bool allow_guide=false,bool allow_companion_chain=false,bool allow_hunt_protection=false,bool allow_battle_entry=false,bool allow_recruit_tail=false,bool allow_stable_siege=false,bool allow_palantir_deferred=false,bool allow_palantir_reveal=false,bool allow_chained_overflow=false,bool allow_unrolled=false){
    const auto attrs=attributes(s,s.root,"game_state.GameState");
    auto value=[&](const char* name,bool required=false){return field(s,attrs,name,required);};
    auto absent=[&](const char* name,bool required=false){
        const auto n=value(name,required);
        if(!ctx_none(s,n))throw std::invalid_argument(std::string("quiet action rejects pending ")+name);
    };
    auto inactive=[&](const char* name){
        if(truth(s,value(name)))throw std::invalid_argument(std::string("quiet action rejects active ")+name);
    };
    bool overflow_pending=false,overflow_regions_nonempty=false;
    if(allow_recruit_tail){
        const auto pending=value("pending_card_option");
        if(!ctx_none(s,pending)){
            const auto owner=field(s,pending,"side",false);
            require(text_is(s,owner,"fp") || text_is(s,owner,"sh"),"Recruitment tail needs canonical overstack owner");
            eo_pending(s,text_is(s,owner,"fp")?0:1);overflow_pending=true;
        }
        const auto queued=value("_gandalf_draw_pending");
        if(!ctx_none(s,queued)){
            require(typed(s,queued,WOTR_DICT,"Queued Gandalf grant must be dictionary").count==6,"Queued Gandalf grant shape invalid");
            text(s,field(s,queued,"source_card_id"));
            require(text_is(s,field(s,queued,"deck"),"character") || text_is(s,field(s,queued,"deck"),"strategy"),"Queued Gandalf deck invalid");
            require(text_is(s,field(s,queued,"die_face"),"EVENT") || text_is(s,field(s,queued,"die_face"),"WILL_OF_WEST"),"Queued Gandalf funding invalid");
        }
        const auto tracked=value("_overstack_regions");
        if(!ctx_none(s,tracked)){
            typed(s,tracked,WOTR_SET,"Tracked overflow regions must be a set");
            const auto& regions=s.nodes[tracked];
            for(uint64_t i=0;i<regions.count;++i){
                qg_canonical_name(text(s,s.edges[regions.offset+i]));overflow_regions_nonempty=true;
            }
        }
        require(!overflow_regions_nonempty || overflow_pending || allow_chained_overflow,"Tracked overflow needs its explicit pending choice");
    }
    const auto rolled=value("dice_rolled",true);
    require(typed(s,rolled,WOTR_BOOL,"quiet action requires boolean dice_rolled").integer==1 || allow_eye || allow_discard || allow_declare || allow_unrolled,
            "quiet action requires rolled dice");
    if(!allow_declare)inactive("_declare_phase");
    if(!allow_eye)inactive("_eye_allocation_phase");
    if(!allow_guide)inactive("_fellowship_tail_phase");
    for(const char* key:{"pending_combat","pending_hunt"})absent(key,true);
    if(!allow_guide)absent("pending_guide_change",true);
    if(!allow_army)absent("pending_army_move_die",true);
    if(!allow_muster)absent("pending_muster_die",true);
    if(!allow_discard)absent("pending_discard");
    if(allow_companion_chain)cm_validate_chain(s);else if(!allow_recruit_tail)absent("pending_card_option");
    for(const char* key:{"pending_card_followup","_fellowship_move_die_pending"})absent(key);
    if(!allow_recruit_tail){absent("_gandalf_draw_pending");inactive("_overstack_regions");}
    if(!allow_palantir_deferred)inactive("_palantir_draw_pending");
    if(!allow_palantir_reveal)inactive("_reveal_place_due");
    for(const char* key:{"_wwsat_owed","_pending_hunt_damage","_hunt_reveal_pending",
                         "_pending_reveal_strongholds","_fellowship_route_event",
                         "_hobbit_place_queue","_mithril_stash","_wizardstaff_stash"})inactive(key);

    const auto persistent=value("persistent_cards",true);
    attributes(s,persistent,"game_state.PersistentCardState");
    if(allow_hunt_protection)hpr_card_gate(s);
    else if(allow_political_table)political_table_gate(s);
    else hpr_card_gate(s); // Only the five verified Hunt/table effects.
    typed(s,value("besieged_strongholds",true),WOTR_SET,"quiet action requires existing siege set"); // Package W: the stable-siege refusal is lifted (frozen: none)

    if(allow_stable_siege){
        const auto siege_node=value("besieged_strongholds",true);const auto& sieges=s.nodes[siege_node];
        for(uint64_t i=0;i<sieges.count;++i){
            const auto key=text(s,s.edges[sieges.offset+i]);
            require(qg_canonical_name(key),"quiet stable siege requires canonical keys");
        }
    }

    const auto hands=value("hands",true);
    typed(s,hands,WOTR_DICT,"quiet action requires hands dictionary");
    for(int side=0;side<2;++side){
        const auto hand=ctx_side_value(s,hands,side);
        attributes(s,hand,"game_state.PlayerInfo");
        typed(s,obs_attr(s,hand,"hand"),WOTR_LIST,"quiet action requires hand list"); // Package W: over-limit hand / stale discard flag clauses lifted (frozen: none)
        typed(s,obs_attr(s,hand,"needs_discard"),WOTR_BOOL,"quiet action requires boolean needs_discard");
    }
    const auto fellowship=value("fellowship",true);
    attributes(s,fellowship,"game_state.Fellowship");
    const auto companions=obs_attr(s,fellowship,"companions");
    const auto& companion_list=typed(s,companions,WOTR_LIST,"quiet action requires companion list");
    const auto guide=obs_attr(s,fellowship,"guide");
    bool guide_member=false;
    for(uint64_t i=0;i<companion_list.count;++i){
        const auto companion=s.edges[companion_list.offset+i];
        require(object_type(s,companion)=="game_state.Companion","quiet action requires registered companions");
        guide_member|=guide==companion;
    }
    require(!companion_list.count || guide_member || allow_guide,"quiet action rejects guide repair or nonidentical guide");

    typed(tables,tables.root,WOTR_DICT,"quiet action requires observation metadata");
    require(text_is(tables,field(tables,tables.root,"schema"),"wotr_native_observation_tables_v1"),
            "quiet action requires pinned observation metadata");
    const auto board_regions=field(tables,tables.root,"board_regions");
    typed(tables,board_regions,WOTR_DICT,"quiet action requires board region metadata");
    const auto fp_regions=field(tables,tables.root,"fp_region_keys");
    typed(tables,fp_regions,WOTR_SET,"quiet action requires FP-region metadata set");
    const auto controls=value("enemy_controlled",true);
    require(typed(s,controls,WOTR_DICT,"quiet action requires control dictionary").count==4,
            "quiet action requires exactly two control sides");
    for(int side=0;side<2;++side){
        const auto collection=ctx_side_value(s,controls,side);
        const auto& regions=typed(s,collection,WOTR_SET,"quiet action requires control sets");
        for(uint64_t i=0;i<regions.count;++i){
            const auto key=text(s,s.edges[regions.offset+i]);
            require(qg_canonical_name(key) && ctx_get(tables,board_regions,key)!=missing,
                    "quiet action rejects noncanonical or unknown control region");
        }
    }

    const auto armies=value("armies",true);
    const auto& army_dict=typed(s,armies,WOTR_DICT,"quiet action requires army dictionary");
    qg_NameSeen normalized_keys;
    qg_BoundedSeen<uint64_t> army_objects;
    for(uint64_t i=0;i<army_dict.count;i+=2){
        const auto raw_key=text(s,s.edges[army_dict.offset+i]);const bool canonical=qg_ascii_canonical(raw_key);
        std::string normalized;std::string_view key=raw_key;
        if(!canonical){normalized=sanitize_name(raw_key);key=normalized;}
        require(normalized_keys.insert(key,canonical),"quiet action rejects duplicate normalized army keys");
        const auto meta=ctx_get(tables,board_regions,key);
        require(meta!=missing,"quiet action rejects unknown army region");
        typed(tables,meta,WOTR_DICT,"quiet action requires per-region metadata");
        const auto army=s.edges[army_dict.offset+i+1];
        require(army_objects.insert(army),"quiet action rejects shared army object across keys");
        attributes(s,army,"game_state.Army");
        const auto units=obs_attr(s,army,"units");
        const auto& unit_list=typed(s,units,WOTR_LIST,"quiet action requires army unit list");
        typed(s,obs_attr(s,army,"leaders"),WOTR_LIST,"quiet action requires army leader list");
        std::array<uint64_t,2> counts{};
        for(uint64_t j=0;j<unit_list.count;++j){
            const auto unit=s.edges[unit_list.offset+j];
            require(object_type(s,unit)=="game_state.Unit","quiet action requires registered units");
            ++counts[ctx_unit_side(s,unit)];
        }
        const bool stronghold=truth(tables,ctx_get(tables,meta,"is_stronghold"));
        const bool settlement=stronghold || truth(tables,ctx_get(tables,meta,"is_city"))
                                        || truth(tables,ctx_get(tables,meta,"is_town"));
        (void)stronghold;(void)settlement; // Package W: siege-flag / garrison-cap / capture-bookkeeping / stale-flag clauses lifted (frozen: none; the frozen tail re-derives them)
    }
}
// Deliberately opt-in: callers must preserve all armies and stable siege caps,
// or validate every changed army separately. Complete siege menus additionally
// require proved execution scope for every offered action family.
void quiet_stable_action_gate(const Snapshot& s,const Snapshot& t,bool muster=false,bool political=false,bool palantir_reveal=false,bool unrolled=false){
    quiet_action_gate(s,t,muster,false,false,false,political,false,false,false,false,false,false,true,false,palantir_reveal,false,unrolled);
}
