// Palantir event grant timing and learned deck selection.
bool epa_live(const Snapshot& s){return hpr_card(s,"PALANTIR_OF_ORTHANC")!=missing;}
bool epa_self(std::string_view id){return id=="SHADOW_Character_The_Palant\xc3\xadr_of_Orthanc";}
bool epa_waiting(const Snapshot& s){
    const auto p=obs_attr(s,s.root,"pending_card_option",false);
    return !ctx_none(s,p)&&text_is(s,field(s,p,"card_id",false),"__palantir_draw__");
}

void epa_open(Snapshot& s){
    if(cbe_palantir_blocked(s)){
        rr_set_attr(s,s.root,"_palantir_draw_pending",rr_scalar(s,WOTR_BOOL,1));return;
    }
    std::vector<uint64_t> options;
    for(const char* deck:{"character","strategy"}){
        const auto pile=obs_attr(s,s.root,da_deck_name(1,deck),false);
        if(truth(s,pile))options.push_back(ev_dict(s,{{"palantir_deck",rr_string(s,deck)}}));
    }
    if(options.empty())return;
    rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{
        {"card_id",rr_string(s,"__palantir_draw__")},{"card_name",rr_string(s,"Palantir draw")},
        {"side",rr_string(s,"sh")},{"decider",rr_string(s,"sh")},{"direct",rr_scalar(s,WOTR_BOOL,1)},
        {"palantir_draw",rr_scalar(s,WOTR_BOOL,1)},{"options",cp_container(s,WOTR_LIST,options)}}));
}

void epa_after_event(Snapshot& s,int side,std::string_view id,std::string_view deck,std::string_view face){
    // Call only at the real card commit boundary. Initial optioned play must
    // retain its funding metadata until the first card-consuming choice.
    if(side!=1 || (deck!="character"&&deck!="strategy") || face!="EVENT" || epa_self(id) || !epa_live(s))return;
    if(!ctx_none(s,obs_attr(s,s.root,"pending_combat"))){
        rr_set_attr(s,s.root,"_palantir_draw_pending",rr_scalar(s,WOTR_BOOL,1));return;
    }
    epa_open(s);
}

void epa_maybe(Snapshot& s){
    if(truth(s,obs_attr(s,s.root,"_palantir_draw_pending",false)) &&
       ctx_none(s,obs_attr(s,s.root,"pending_combat")) &&
       ctx_none(s,obs_attr(s,s.root,"pending_card_option",false)) &&
       ctx_none(s,obs_attr(s,s.root,"pending_hunt")))cbe_palantir(s);
}

// A paid card can still own cargo/landing decisions when its bonus is queued.
// This validates that specific continuation; unrelated obligations stay rejected.
bool epa_paid_continuation_debt(const Snapshot& s,std::string_view id,bool direct){
    const auto queued=obs_attr(s,s.root,"_palantir_draw_pending",false);
    if(!truth(s,queued))return false;
    require(typed(s,queued,WOTR_BOOL,"Queued Palantir grant must be Boolean").integer==1,
            "Queued Palantir grant must be true");
    require(direct&&epa_live(s)&&!contains_string(s,tc_hand(s,1),id),
            "Card continuation has unrelated queued Palantir");
    const auto discards=obs_attr(s,s.root,"discard_face_up");
    require(contains_string(s,field(s,discards,"sh_strategy"),id)||
            contains_string(s,field(s,discards,"sh_character"),id),
            "Queued Palantir continuation lacks a committed card");
    return true;
}

uint64_t epa_pending(const Snapshot& s,int side){
    require(side==1,"Palantir draw belongs to Shadow");
    const auto p=obs_attr(s,s.root,"pending_card_option",false);
    require(!ctx_none(s,p)&&text_is(s,field(s,p,"card_id",false),"__palantir_draw__")&&
            text_is(s,field(s,p,"card_name",false),"Palantir draw")&&
            text_is(s,field(s,p,"side",false),"sh")&&text_is(s,field(s,p,"decider",false),"sh")&&
            truth(s,field(s,p,"direct",false))&&truth(s,field(s,p,"palantir_draw",false)),
            "No canonical Palantir draw is pending");
    const auto options=child_edges(s,field(s,p,"options"));
    require(!options.empty()&&options.size()<=2,"Palantir pending deck choices invalid");
    std::set<std::string> seen;
    for(auto option:options){
        require(typed(s,option,WOTR_DICT,"Palantir choice must be dictionary").count==2,"Palantir choice shape invalid");
        const std::string deck(text(s,field(s,option,"palantir_deck")));
        require((deck=="character"||deck=="strategy")&&seen.insert(deck).second,"Palantir deck choice invalid or duplicate");
    }
    return p;
}

std::unique_ptr<Snapshot> native_palantir_draw_actions(const Snapshot& source,int side){
    epa_pending(source,side);auto result=std::make_unique<Snapshot>(source);auto& s=*result;
    const auto p=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t index=0;
    for(auto option:child_edges(s,field(s,p,"options"))){
        const auto params=ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},
            {"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",option}});
        rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),params}));
    }
    s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}

void epa_settle_draw_debt(Snapshot& s,std::string_view deck){
    const auto owed=obs_attr(s,s.root,"_owed_draws",false);if(!truth(s,owed))return;
    const auto entries=child_edges(s,owed);
    for(size_t i=0;i<entries.size();i+=2){
        const auto key=entries[i];if(s.nodes[key].tag!=WOTR_TUPLE)continue;
        const auto pair=child_edges(s,key);
        if(pair.size()!=2||!text_is(s,pair[0],"SHADOW")||!text_is(s,pair[1],deck))continue;
        const auto count=typed(s,entries[i+1],WOTR_INT,"Draw debt must be integer").integer;
        if(count>0){auto updated=entries;updated[i+1]=rr_scalar(s,WOTR_INT,count-1);set_edges(s,owed,updated);}return;
    }
}

fm_Result apply_palantir_draw(const Snapshot& source,const Snapshot& t,const void* random_source,int side,int64_t index){
    const auto p=epa_pending(source,side);
    const auto options=child_edges(source,field(source,p,"options"));
    require(index>=0&&static_cast<uint64_t>(index)<options.size(),"Palantir draw choice index out of range");
    const std::string deck(text(source,field(source,options[index],"palantir_deck")));
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(random_source));
    if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;
    rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));
    rr_set_attr(s,s.root,"_palantir_draw_pending",rr_scalar(s,WOTR_BOOL,0));
    // Fresh grants wait for the complete card, including Search reveal routes.
    // Retain compatibility with saved bonus choices that predate this ordering.
    // Return of the Witch-king may also expose tracked overflow only in final
    // deduplication, after a grant was opened; validate that real tracking here.
    const auto tracked=obs_attr(s,s.root,"_overstack_regions",false);const bool overflow=truth(s,tracked);
    if(overflow){
        typed(s,tracked,WOTR_SET,"Palantir overflow tracking must be a set");
        for(auto raw:child_edges(s,tracked)){
            const auto region=sanitize_name(text(s,raw));
            require(ctx_get(t,field(t,t.root,"board_regions"),region)!=missing,"Palantir overflow region is unknown");
            const auto army=mu_army(s,region);require(army!=missing,"Palantir overflow army is absent");std::array<size_t,2> counts{};
            for(auto unit:child_edges(s,obs_attr(s,army,"units")))++counts[ctx_unit_side(s,unit)];
            require(counts[1]>static_cast<size_t>(su_cap(s,t,region,1))&&counts[0]<=static_cast<size_t>(su_cap(s,t,region,0)),"Palantir tracking needs actual Shadow-only excess");
        }
        quiet_action_gate(s,t,false,false,false,false,false,false,false,false,false,false,true,true,false,true,true);
    }else{Snapshot check=s;rr_set_attr(check,check.root,"_mithril_stash",rr_scalar(check,WOTR_NONE));quiet_stable_action_gate(check,t,false,false,true);} // Package W: one path for every Shadow Palantir draw (a retained Mithril tile is not a structural fact)
    ring_waiver(s,0);ring_waiver(s,1);
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    am_Builder events;std::vector<uint64_t> rows;
    const auto pile=obs_attr(s,s.root,da_deck_name(1,deck),false);
    // A saved choice can outlive the final card. Python catches the exhausted
    // grant and clears the pending choice; it does not create another choice.
    if(truth(s,pile)){
        const auto drawn=tc_draw_event_card(s,1,deck);epa_settle_draw_debt(s,deck);
        rows.push_back(events.row(-1,"card_drawn",{events.str("deck"),events.str(deck),events.str("card_id"),events.str(text(s,drawn)),events.str("side"),events.str("SH")}));
    }
    const int marker=ctx_side(s,obs_attr(s,s.root,"_battle_resolved_attacker",false));
    cau_finish(s,t,result.random.get(),marker,events,rows);
    events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
