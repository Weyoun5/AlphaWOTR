// Read-only projections used by the unchanged Python combat-search control flow.
// Never export the round cursor: its _live_state can own an entire GameState.
void csf_small(const Snapshot& s,uint64_t root){
    std::vector<uint64_t> todo{root};std::set<uint64_t> seen;
    while(!todo.empty()){
        auto n=todo.back();todo.pop_back();if(n==missing||!seen.insert(n).second)continue;
        require(seen.size()<=4096,"Combat search metadata exceeds compact bound");
        const auto& v=s.nodes[n];
        if(v.tag==WOTR_OBJECT)require(object_type(s,n)!="game_state.GameState","Combat search metadata must not contain GameState");
        if(v.tag>=WOTR_LIST&&v.tag<=WOTR_ENUM){for(auto c:child_edges(s,n))todo.push_back(c);if(v.flags==1)todo.push_back(v.integer);}
    }
}
std::unique_ptr<Snapshot> native_combat_search_facts(const Snapshot& s){
    sm_Builder b(s);std::vector<uint64_t> result;
    auto copy=[&](uint64_t v){csf_small(s,v);return b.raw(v);};
    auto add=[&](std::vector<uint64_t>& rows,const char* key,uint64_t v){if(v!=missing){rows.push_back(b.str(key));rows.push_back(copy(v));}};
    add(result,"_last_round_call",obs_attr(s,s.root,"_last_round_call",false));
    for(auto k:{"turn","turn_number"})add(result,k,obs_attr(s,s.root,k,false));
    auto pc=obs_attr(s,s.root,"pending_combat",false);
    // GameState's dataclass declares this class default. Deleting the instance
    // field still makes getattr/hasattr see None in the frozen Python engine.
    if(pc==missing){result.push_back(b.str("pending_combat"));result.push_back(b.nil);}
    if(pc!=missing){
        result.push_back(b.str("pending_combat"));
        if(ctx_none(s,pc))result.push_back(b.nil);
        else{
            typed(s,pc,WOTR_DICT,"Combat search pending context must be a dictionary");std::vector<uint64_t> fields;
            for(auto k:{"phase","alloc_stage","atk_hits","def_hits","_p9_done","atk_combat_card","_commit_nonce","rounds_fought","attacker_region","defender_region","attacker_side","defender_side","deciding_side"})add(fields,k,field(s,pc,k,false));
            auto cursor=field(s,pc,"_round_resume",false);
            if(cursor!=missing){fields.push_back(b.str("_round_resume"));fields.push_back(b.boolean(truth(s,cursor)));}
            for(auto k:{"effect_choice","_onslaught_payment"}){
                auto value=field(s,pc,k,false);if(value==missing)continue;
                fields.push_back(b.str(k));
                if(!truth(s,value)){fields.push_back(copy(value));continue;}
                typed(s,value,WOTR_DICT,"Combat search nested context must be a dictionary");std::vector<uint64_t> selected;
                if(std::string_view(k)=="effect_choice")for(auto x:{"stage","effect"})add(selected,x,field(s,value,x,false));
                else add(selected,"paid",field(s,value,"paid",false));
                fields.push_back(b.container(WOTR_DICT,selected));
            }
            result.push_back(b.container(WOTR_DICT,fields));
        }
    }
    b.out.root=b.container(WOTR_DICT,result);return std::make_unique<Snapshot>(b.out);
}
std::unique_ptr<Snapshot> native_combat_search_units(const Snapshot& s,const Snapshot& args){
    const auto region=std::string(text(args,field(args,args.root,"region")));
    const auto side=ctx_side(args,field(args,args.root,"side"));
    const auto kind=field(args,args.root,"kind"),given=field(args,args.root,"pc");
    if(!ctx_none(args,kind))require(enum_is(args,kind,"game_state.UnitKind","REGULAR")||enum_is(args,kind,"game_state.UnitKind","ELITE"),"Combat allocation kind must be Regular, Elite, or None");
    Snapshot work;work.root=rr_scalar(work,WOTR_NONE);std::unordered_map<uint64_t,uint64_t> state_memo,args_memo;
    auto army=cbs_lookup(s,region);auto local_army=kc_copy(work,s,army,state_memo);
    const auto& context_source=ctx_none(args,given)?s:args;
    auto context=ctx_none(args,given)?obs_attr(s,s.root,"pending_combat",false):given;
    auto rear=ctx_none(context_source,context)?missing:field(context_source,context,"attacker_rearguard",false);
    auto& memo=ctx_none(args,given)?state_memo:args_memo;
    csf_small(context_source,rear);
    auto pc=ev_dict(work,{{"attacker_rearguard",kc_copy(work,context_source,rear,memo)}});
    auto units=ca_partition(work,local_army,region,side,pc).units;
    const auto card=obs_attr(s,s.root,"pending_card_option",false);uint64_t eligible=missing;
    if(truth(s,card)&&truth(s,field(s,card,"card_casualty",false))&&enum_is(s,field(s,card,"cas_side",false),"game_state.Side",side==0?"FREE_PEOPLES":"SHADOW")){
        auto at=field(s,card,"cas_region",false);sm_Builder reader(s);
        if(sanitize_name(at==missing?"":reader.string(at))==sanitize_name(region))eligible=field(s,card,"cas_eligible_nations",false);
    }
    sm_Builder out(work);std::vector<uint64_t> kept;
    for(auto unit:units){
        if(!ctx_none(args,kind)&&enum_member(work,obs_attr(work,unit,"kind"),"game_state.UnitKind")!=enum_member(args,kind,"game_state.UnitKind"))continue;
        if(!ctx_none(s,eligible)){
            bool found=false;auto nation=enum_member(work,obs_attr(work,unit,"nation"),"game_state.Nation");
            for(auto n:child_edges(s,eligible))found|=enum_is(s,n,"game_state.Nation",nation);
            if(!found)continue;
        }
        csf_small(work,unit);kept.push_back(out.raw(unit));
    }
    out.out.root=out.container(WOTR_LIST,kept);return std::make_unique<Snapshot>(out.out);
}
