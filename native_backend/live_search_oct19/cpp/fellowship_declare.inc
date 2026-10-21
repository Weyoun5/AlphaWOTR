// Turn-start declaration placement, not reveal and not Mordor entry.
void fd_gate(const Snapshot& s,const Snapshot& t,bool protection=false){
    // Declaration placement preserves both armies and settlement control.
    // Admit only canonical, fully tracked stable sieges with valid garrisons;
    // this does not widen ordinary movement or whole-turn menus.
    quiet_action_gate(s,t,false,false,false,false,false,true,false,false,protection,false,false,true);
    if(!protection){const auto cards=obs_attr(s,s.root,"persistent_cards");for(const char* pile:{"fp_cards","sh_cards"})
        require(!truth(s,obs_attr(s,cards,pile)),"Declaration with table cards requires the explicit RNG route API");}
    require(truth(s,obs_attr(s,s.root,"_declare_phase",false)),"Fellowship declaration phase is inactive");
}
std::set<std::string> fd_reachable(const Snapshot& s,uint64_t fellowship){
    const auto progress=obs_attr(s,fellowship,"progress");
    const auto steps=typed(s,progress,WOTR_INT,"declaration progress must be an integer").integer;
    require(steps>=0,"declaration progress cannot be negative");
    const auto start=glb_fellowship_key(s,fellowship);const auto board=obs_attr(s,s.root,"board");
    const auto regions=obs_attr(s,board,"_regions"),adj=obs_attr(s,board,"_adj");
    require(ctx_get(s,regions,start)!=missing,"declaration start is not a known Board region");
    std::set<std::string> found{start},frontier{start};
    for(int64_t depth=0;depth<steps && !frontier.empty();++depth){
        std::set<std::string> next;
        for(const auto& current:frontier){
            const auto neighbors=ctx_get(s,adj,current);
            require(neighbors!=missing,"declaration adjacency is missing");
            for(auto value:child_edges(s,neighbors)){
                const auto key=sanitize_name(text(s,value));
                if(found.insert(key).second)next.insert(key);
            }
        }
        frontier=std::move(next);
    }
    return found;
}
std::unique_ptr<Snapshot> native_fellowship_declare_actions(const Snapshot& s,const Snapshot& t,int32_t side){
    require(side==0 || side==1,"declaration menu side must be FP or SH");fd_gate(s,t,true);
    am_Builder b;std::vector<uint64_t> rows;const auto fs=obs_attr(s,s.root,"fellowship");
    const auto steps=typed(s,obs_attr(s,fs,"progress"),WOTR_INT,"declaration progress must be integer").integer;
    if(truth(s,obs_attr(s,fs,"hidden")) && !truth(s,obs_attr(s,fs,"mordor_entered")) && steps>=0)
        for(const auto& region:fd_reachable(s,fs))
            rows.push_back(b.row(-1,"fellowship_declare",{b.str("region"),b.str(region)}));
    rows.push_back(b.row(-1,"pass",{}));b.out.root=b.container(WOTR_LIST,rows);return std::make_unique<Snapshot>(std::move(b.out));
}
void fd_declare_inner(Snapshot& out,const Snapshot& t,std::string_view key){
    const auto f=obs_attr(out,out.root,"fellowship"),board=obs_attr(out,out.root,"board");
    const auto region=ctx_get(out,obs_attr(out,board,"_regions"),key);
    require(region!=missing,"declaration destination Board region missing");attributes(out,region,"board.Region");
    require(text_is(out,obs_attr(out,region,"key"),key),"declaration destination Region key is noncanonical");
    rr_set_attr(out,f,"region",region);rr_set_attr(out,f,"progress",rr_scalar(out,WOTR_INT,0));rr_set_attr(out,f,"hidden",rr_scalar(out,WOTR_BOOL,1));
    // Almanac p17: declaration itself discards The Last Battle, even in
    // place or at a Mordor gateway while Aragorn's Army remains eligible.
    const auto table=obs_attr(out,out.root,"persistent_cards");uint64_t last=missing;
    for(const char* pile:{"fp_cards","sh_cards"})for(auto card:child_edges(out,obs_attr(out,table,pile)))
        if(last==missing && enum_is(out,obs_attr(out,card,"card_type"),"game_state.PersistentCardType","LAST_BATTLE"))last=card;
    if(last!=missing)cbp_discard_table(out,cbp_remove_card(out,table,last));
    const auto meta=ctx_get(t,field(t,t.root,"board_regions"),key);require(meta!=missing,"declaration region metadata missing");
    if((truth(t,ctx_get(t,meta,"is_city")) || truth(t,ctx_get(t,meta,"is_stronghold"))) && contains_string(t,field(t,t.root,"fp_region_keys"),key)){
        const auto sh=ctx_side_value(out,obs_attr(out,out.root,"enemy_controlled"),1);
        if(!contains_string(out,sh,key)){
            const auto persistent=obs_attr(out,out.root,"persistent_cards");std::vector<uint64_t> discard;
            for(const char* type:{"WORN_WITH_SORROW","FLOCKS_OF_CREBAIN"}){bool found=false;for(const char* pile:{"fp_cards","sh_cards"})for(auto card:child_edges(out,obs_attr(out,persistent,pile)))
                if(!found && enum_is(out,obs_attr(out,card,"card_type"),"game_state.PersistentCardType",type)){discard.push_back(card);found=true;}}
            for(auto card:discard)cbp_discard_table(out,cbp_remove_card(out,persistent,card));
            const auto value=out.nodes[obs_attr(out,f,"corruption")];
            require(value.tag==WOTR_INT || value.tag==WOTR_BOOL || value.tag==WOTR_FLOAT,"declaration corruption must be numeric");
            const bool zero=value.tag==WOTR_FLOAT?value.real<=1:value.integer<=1;
            const auto healed=zero?rr_scalar(out,WOTR_INT,0):value.tag==WOTR_FLOAT?rr_scalar(out,WOTR_FLOAT,0,value.real-1):rr_scalar(out,WOTR_INT,value.integer-1);
            rr_set_attr(out,f,"corruption",healed);
        }
        const auto nation=ctx_get(t,field(t,t.root,"movement_region_nations"),key);
        if(!ctx_none(t,nation) && fp_nation(ctx_nation(t,nation)))wt_activate_if_inactive(out,ctx_nation(t,nation),"fellowship_declared");
    }
    // The RNG/trace-bearing wrapper registers an admitted Balrog route after
    // this placement. The driver tail owns positional Mordor entry afterward.
    if(!truth(out,obs_attr(out,out.root,"dice_rolled")))rr_set_attr(out,out.root,"_fellowship_tail_phase",rr_scalar(out,WOTR_BOOL,1));
}
std::unique_ptr<Snapshot> apply_fellowship_declare(const Snapshot& s,const Snapshot& t,std::string_view destination,bool pass=false,bool protection=false){
    fd_gate(s,t,protection);const auto fs=obs_attr(s,s.root,"fellowship");const auto key=sanitize_name(destination);
    if(!pass){
        require(truth(s,obs_attr(s,fs,"hidden")),"revealed Fellowship cannot declare");
        require(!truth(s,obs_attr(s,fs,"mordor_entered")),"Mordor Track has no map declaration");
        require(fd_reachable(s,fs).count(key)!=0,"declaration region is not reachable within progress");
    }
    auto child=action_clone(s);auto& out=*child;
    rr_set_attr(out,out.root,"_casualty_operations",rr_scalar(out,WOTR_LIST));
    rr_set_attr(out,out.root,"_ring_free_side",rr_scalar(out,WOTR_NONE));
    rr_set_attr(out,out.root,"_declare_phase",rr_scalar(out,WOTR_BOOL,0));
    if(pass){rr_set_attr(out,out.root,"_battle_resolved_attacker",rr_scalar(out,WOTR_NONE));return child;}
    fd_declare_inner(out,t,key);
    return child;
}
