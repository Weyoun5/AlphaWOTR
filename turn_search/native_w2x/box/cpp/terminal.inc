// Frozen game_logic victory predicates, plus ai._settle_new_node marker priority.
// No profile rewards, network values, proof propagation, or state/RNG mutation.
struct TerminalResult {
    int32_t winner=-1; // -1 none, 0 Free Peoples, 1 Shadow
    int32_t kind=0;    // 0 none, 1 military, 2 ring_destroyed, 3 ring_corruption
    int64_t fp_vp=0,sh_vp=0;
};
int64_t terminal_vp(const Snapshot& s,const Snapshot& tables,uint64_t captures,uint64_t settlements){
    const auto& list=tables.nodes[settlements];
    require(list.tag==WOTR_LIST || list.tag==WOTR_TUPLE,"terminal settlements must be sequence");
    // Python only attempts membership when a settlement row exists. Preserve
    // absent-side (), but reject explicit None/unsupported membership objects.
    if(captures!=missing){
        const auto tag=s.nodes[captures].tag;
        require(tag==WOTR_STRING || tag==WOTR_LIST || tag==WOTR_TUPLE || tag==WOTR_SET ||
                tag==WOTR_FROZENSET || tag==WOTR_DICT,"terminal control membership unsupported");
    }
    int64_t total=0;
    for(uint64_t i=0;i<list.count;++i){
        const auto& row=tables.nodes[tables.edges[list.offset+i]];
        require((row.tag==WOTR_LIST || row.tag==WOTR_TUPLE) && row.count==2,"invalid terminal settlement row");
        const auto key=text(tables,tables.edges[row.offset]);
        const auto points=typed(tables,tables.edges[row.offset+1],WOTR_INT,"terminal VP must be integer").integer;
        require(points==1 || points==2,"terminal settlement VP must be one or two");
        if(contains_string(s,captures,key)){
            require(total<=std::numeric_limits<int64_t>::max()-points,"terminal VP total overflow");
            total+=points;
        }
    }
    return total;
}
TerminalResult check_victory(const Snapshot& s,const Snapshot& tables){
    const auto attrs=attributes(s,s.root,"game_state.GameState");
    const auto fellowship=field(s,attrs,"fellowship");
    attributes(s,fellowship,"game_state.Fellowship");
    TerminalResult result;
    const double corruption=ctx_num(s,obs_attr(s,fellowship,"corruption"));
    if(corruption>=12){result.winner=1;result.kind=3;}
    else if(truth(s,obs_attr(s,fellowship,"mordor_entered")) && ctx_num(s,obs_attr(s,fellowship,"mordor_step"))>=5){
        if(corruption<12 && !truth(s,field(s,attrs,"_hunt_reveal_pending",false)) && !truth(s,field(s,attrs,"_pending_hunt_damage",false)) &&
           ctx_none(s,field(s,attrs,"pending_hunt")) &&
           ctx_none(s,field(s,attrs,"_mithril_stash",false)) &&
           ctx_none(s,field(s,attrs,"_wizardstaff_stash",false))){
            result.winner=0;result.kind=2;
        }
    }
    typed(tables,tables.root,WOTR_DICT,"terminal metadata must be dictionary");
    require(text_is(tables,field(tables,tables.root,"schema"),"wotr_native_observation_tables_v1"),
            "terminal requires pinned metadata");
    const auto controls=field(s,attrs,"enemy_controlled");
    typed(s,controls,WOTR_DICT,"terminal control must be dictionary");
    const auto fp=ctx_side_value(s,controls,0),sh=ctx_side_value(s,controls,1);
    result.fp_vp=terminal_vp(s,tables,fp,field(tables,tables.root,"live_vp_sh_settlements"));
    result.sh_vp=terminal_vp(s,tables,sh,field(tables,tables.root,"live_vp_fp_settlements"));
    const auto fp_limit=typed(tables,field(tables,tables.root,"fp_vp_victory"),WOTR_INT,"terminal FP threshold must be integer").integer;
    const auto sh_limit=typed(tables,field(tables,tables.root,"shadow_vp_victory"),WOTR_INT,"terminal SH threshold must be integer").integer;
    require(fp_limit==4 && sh_limit==10,"unsupported terminal victory thresholds");
    if(result.winner<0){
        if(result.sh_vp>=sh_limit){result.winner=1;result.kind=1;}
        else if(result.fp_vp>=fp_limit){result.winner=0;result.kind=1;}
    }
    return result;
}
bool terminal_should_end(const TerminalResult& result,bool end_of_turn){
    return result.winner>=0 && (result.kind==2 || result.kind==3 || end_of_turn);
}
// An end-turn marker is authoritative even if a later caller changes the
// board. Missing/None uses live predicates; malformed markers reject clearly.
TerminalResult search_victory(const Snapshot& s,const Snapshot& tables,bool end_of_turn){
    const auto marker=obs_attr(s,s.root,"_search_endturn_victory",false);
    if(!ctx_none(s,marker)){
        typed(s,marker,WOTR_DICT,"search victory marker must be dictionary");
        TerminalResult result;
        const auto winner=field(s,marker,"victory");
        if(text_is(s,winner,"free_peoples"))result.winner=0;
        else if(text_is(s,winner,"shadow"))result.winner=1;
        else throw std::invalid_argument("search victory marker requires known winner");
        const auto kind=field(s,marker,"victory_type");
        if(text_is(s,kind,"military"))result.kind=1;
        else if(text_is(s,kind,"ring_destroyed") && result.winner==0)result.kind=2;
        else if(text_is(s,kind,"ring_corruption") && result.winner==1)result.kind=3;
        else throw std::invalid_argument("search victory marker requires compatible victory type");
        result.fp_vp=typed(s,field(s,marker,"free_peoples_vp"),WOTR_INT,"search FP VP must be integer").integer;
        result.sh_vp=typed(s,field(s,marker,"shadow_vp"),WOTR_INT,"search SH VP must be integer").integer;
        require(result.fp_vp>=0 && result.sh_vp>=0,"search marker VP cannot be negative");
        return result;
    }
    auto result=check_victory(s,tables);
    if(!terminal_should_end(result,end_of_turn)){result.winner=-1;result.kind=0;}
    return result;
}
