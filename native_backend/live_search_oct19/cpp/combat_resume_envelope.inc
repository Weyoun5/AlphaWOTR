// resume_combat + _accept_combat_round_result, explicit RNG and battle ledgers.
int64_t crs_sum(int64_t left,int64_t right){
    require((right<=0||left<=std::numeric_limits<int64_t>::max()-right)&&(right>=0||left>=std::numeric_limits<int64_t>::min()-right),"Combat total overflow");return left+right;
}
void crs_book_figures(Snapshot& s,const std::vector<uint64_t>& figures,void* rng){
    const auto context=mu_object(s,"game_state.CombatState",{{"_live_state",s.root}});
    for(auto figure:figures)if(object_type(s,figure)=="game_logic.Minion")cs_minion_bookkeeping(s,context,figure);
    for(auto figure:figures)if(cs_plain_fp(s,figure))cbp_book(s,obs_attr(s,figure,"nation"),"leader");
    for(auto figure:figures)if(object_type(s,figure)=="game_state.Companion")cbp_companion_death(s,rng,false);
}
void crs_drop_empty(Snapshot& s,std::string_view region,uint64_t army){
    if(truth(s,obs_attr(s,army,"units"))||truth(s,obs_attr(s,army,"leaders")))return;
    const auto map=obs_attr(s,s.root,"armies");const auto entries=child_edges(s,map);
    for(size_t i=0;i<entries.size();i+=2)if(entries[i+1]==army&&sanitize_name(text(s,entries[i]))==sanitize_name(region)){const std::string key(text(s,entries[i]));cp_erase(s,map,key);return;}
}
void crs_capture(Snapshot& s,const Snapshot& tables,std::string_view region,int side){
    const auto key=sanitize_name(region);const auto meta=field(tables,field(tables,tables.root,"board_regions"),key.c_str(),false);
    if(!ctx_none(tables,meta)&&(truth(tables,field(tables,meta,"is_city",false))||truth(tables,field(tables,meta,"is_town",false))||truth(tables,field(tables,meta,"is_stronghold",false))))nt_capture(s,tables,key,side);
}
void crs_write_finished(Snapshot& s,const Snapshot& tables,uint64_t combat,uint64_t payload,void* rng){
    const auto args=child_edges(s,field(s,payload,"args"));const std::array<std::string,2> regions={std::string(text(s,args[0])),std::string(text(s,args[1]))};
    const std::array<bool,2> dead={truth(s,obs_attr(s,combat,"attacker_destroyed")),truth(s,obs_attr(s,combat,"defender_destroyed"))};
    const int attacker_side=enum_is(s,obs_attr(s,combat,"attacker_side"),"game_state.Side","FREE_PEOPLES")?0:1;
    const auto parent=obs_attr(s,combat,"coloc_parent",false);
    if(!ctx_none(s,parent)){
        const auto rear=child_edges(s,obs_attr(s,combat,"coloc_rear"));std::array<std::vector<uint64_t>,2> leaders;
        for(int role:{1,0}){const auto army=cp_attr(s,combat,cp_role(role)+"_army");leaders[role]=child_edges(s,obs_attr(s,army,"leaders"));if(dead[role]){crs_book_figures(s,leaders[role],rng);leaders[role].clear();}}
        cbp_join(s,parent,"units",child_edges(s,obs_attr(s,obs_attr(s,combat,"defender_army"),"units")),child_edges(s,obs_attr(s,obs_attr(s,combat,"attacker_army"),"units")),child_edges(s,rear[0]));
        cbp_join(s,parent,"leaders",leaders[1],leaders[0],child_edges(s,rear[1]));crs_drop_empty(s,regions[1],parent);
        if(dead[1]&&(!dead[0]||truth(s,rear[0])))crs_capture(s,tables,regions[1],attacker_side);
        cbp_publish_casualties(s,combat,payload,rng);return;
    }
    std::array<bool,2> split{};
    for(int role=0;role<2;++role){const std::string prefix=role==0?"away_atk":"away_def";const auto away=cp_attr(s,combat,prefix+"_parent",false);split[role]=!ctx_none(s,away);if(!split[role])continue;
        const auto rear=child_edges(s,cp_attr(s,combat,prefix+"_rest"));const auto army=cp_attr(s,combat,cp_role(role)+"_army");auto leaders=child_edges(s,obs_attr(s,army,"leaders"));auto units=child_edges(s,obs_attr(s,army,"units"));
        if(dead[role]){crs_book_figures(s,leaders,rng);leaders.clear();units.clear();}cbp_join(s,away,"units",units,child_edges(s,rear[0]),{});cbp_join(s,away,"leaders",leaders,child_edges(s,rear[1]),{});crs_drop_empty(s,regions[role],away);
    }
    for(int role=0;role<2;++role)if(dead[role]&&!split[role]){cbe_reconcile(s,regions[role],role==0?attacker_side:1-attacker_side,rng);
        if(role==1&&!dead[0]&&sanitize_name(regions[0])==sanitize_name(regions[1]))crs_capture(s,tables,regions[1],attacker_side);
    }
    cbp_publish_casualties(s,combat,payload,rng);
}
void crs_witch_king_draw(Snapshot& s,uint64_t pc,am_Builder* events,std::vector<uint64_t>* rows){
    const auto deck_node=field(s,pc,"_wk_sorcerer_deck",false);cp_erase(s,pc,"_wk_sorcerer_deck");if(ctx_none(s,deck_node))return;
    const auto sheet=obs_attr(s,s.root,"witch_king",false);if(!ctx_none(s,sheet)&&!truth(s,obs_attr(s,sheet,"in_play",false)))return;
    const std::string deck(text(s,deck_node));const auto pile=obs_attr(s,s.root,deck=="character"?"sh_char_deck":"sh_strat_deck",false);
    if(truth(s,pile)){
        const auto drawn=tc_draw_event_card(s,1,deck);const auto owed=obs_attr(s,s.root,"_owed_draws",false);
        if(!ctx_none(s,owed)){const auto entries=child_edges(s,owed);for(size_t i=0;i<entries.size();i+=2){const auto key=entries[i];if(s.nodes[key].tag!=WOTR_TUPLE)continue;const auto pair=child_edges(s,key);if(pair.size()!=2||!text_is(s,pair[0],"SHADOW")||!text_is(s,pair[1],deck))continue;
            const auto count=typed(s,entries[i+1],WOTR_INT,"Draw debt must be integer").integer;if(count>0){auto updated=child_edges(s,owed);updated[i+1]=rr_scalar(s,WOTR_INT,count-1);set_edges(s,owed,updated);}break;}}
        if(events&&rows){rows->push_back(events->row(-1,"card_drawn",{events->str("deck"),events->str(deck),events->str("card_id"),events->str(text(s,drawn)),events->str("side"),events->str("SH")}));rows->push_back(events->row(-1,"wk_sorcerer_draw",{events->str("deck"),events->str(deck)}));}
    }
    tc_open_hand_limit_discard(s);
}
void crs_accept_completed(crr_Result& owned,const Snapshot& tables,uint64_t result,uint64_t attacker_card,uint64_t defender_card,am_Builder* events=nullptr,std::vector<uint64_t>* rows=nullptr){
    auto& s=*owned.state;const auto pc=obs_attr(s,s.root,"pending_combat");rr_set_attr(s,s.root,"_last_battle_result",result);
    for(const char* key:{"_round_resume","_round_cards","effect_choice"})cp_erase(s,pc,key);
    const auto old=field(s,pc,"rounds_fought",false);const int64_t rounds=crs_sum(ctx_none(s,old)?0:typed(s,old,WOTR_INT,"Round count must be integer").integer,1);cp_put(s,pc,"rounds_fought",rr_scalar(s,WOTR_INT,rounds));
    cp_put(s,pc,"_observed_round_completed",cp_container(s,WOTR_DICT,{rr_string(s,"schema"),rr_string(s,"observed_round_v1"),rr_string(s,"round"),rr_scalar(s,WOTR_INT,rounds-1),rr_string(s,"attacker_card"),attacker_card,rr_string(s,"defender_card"),defender_card}));
    if(truth(s,obs_attr(s,result,"casualties_deferred",false))){
        cp_put(s,pc,"def_hits",obs_attr(s,result,"deferred_hits_on_defender"));cp_put(s,pc,"atk_hits",obs_attr(s,result,"deferred_hits_on_attacker"));cp_put(s,pc,"alloc_stage",rr_string(s,"attacker"));
        if(truth(s,attacker_card)||truth(s,defender_card))cp_put(s,pc,"_p9",cp_container(s,WOTR_DICT,{rr_string(s,"atk_card"),attacker_card,rr_string(s,"def_card"),defender_card,rr_string(s,"atk_cancelled"),rr_scalar(s,WOTR_BOOL,truth(s,obs_attr(s,result,"attacker_card_cancelled",false))),rr_string(s,"def_cancelled"),rr_scalar(s,WOTR_BOOL,truth(s,obs_attr(s,result,"defender_card_cancelled",false)))}));else cp_erase(s,pc,"_p9");
        if(!ca_enter(s,tables,pc))cbe_complete_allocated(owned,tables,events,rows);return;
    }
    const auto role_node=obs_attr(s,result,"deferred_effect",false);const int role=text_is(s,role_node,"attacker")?0:text_is(s,role_node,"defender")?1:-1;const bool attacker_dead=truth(s,obs_attr(s,result,"attacker_destroyed")),defender_dead=truth(s,obs_attr(s,result,"defender_destroyed"));
    if(cbe_onslaught(s,pc,role,role==0?attacker_card:defender_card,attacker_dead,defender_dead,events,rows))return;
    if(attacker_dead||defender_dead){cbe_finish(s,tables,pc,attacker_dead,defender_dead,events,rows);return;}
    cbe_post_round(s,pc,events,rows);
}
crr_Result crs_settle_advanced(crr_Result owned,const Snapshot& tables,am_Builder* events,std::vector<uint64_t>* rows){
    auto& s=*owned.state;const auto pc=obs_attr(s,s.root,"pending_combat"),payload=field(s,pc,"_round_resume"),cursor=field(s,payload,"cursor"),combat=obs_attr(s,cursor,"combat"),pending=obs_attr(s,cursor,"pending");
    if(events&&rows)ctr_annotations(s,combat,payload,*events,*rows);
    if(!ctx_none(s,pending)){cbe_accept_pause(s,pc,payload,combat,cbe_descriptor_copy(s,pending),owned.random.get());owned.state=ac_compact(s);return owned;}
    const auto args=field(s,payload,"args");const auto parameters=child_edges(s,args),totals=child_edges(s,field(s,payload,"totals"));require(parameters.size()==7&&totals.size()==4,"Battle resume requires original argument and total tuple");
    require(truth(s,obs_attr(s,combat,"battle_over")),"Native resumed round must finish or defer allocation before another round");
    cp_put(s,payload,"cursor",rr_scalar(s,WOTR_NONE));cbe_annotations(s,combat,payload,truth(s,parameters[4]),text(s,parameters[1]));
    const auto rounds=crs_sum(typed(s,totals[2],WOTR_INT,"Round totals must be integer").integer,1),atk_hits=crs_sum(typed(s,totals[0],WOTR_INT,"Round totals must be integer").integer,cp_number(s,combat,"attacker_final_hits",0,true)),def_hits=crs_sum(typed(s,totals[1],WOTR_INT,"Round totals must be integer").integer,cp_number(s,combat,"defender_final_hits",0,true));
    cp_set_number(s,combat,"siege_round",crs_sum(cp_number(s,combat,"siege_round"),1));cp_put(s,payload,"done",rr_scalar(s,WOTR_BOOL,1));const auto result=cbe_result(s,combat,args,rounds,atk_hits,def_hits);
    rr_set_attr(s,result,"siege_continues",rr_scalar(s,WOTR_BOOL,truth(s,parameters[4])&&!truth(s,obs_attr(s,combat,"battle_over"))));rr_set_attr(s,result,"deferred_effect",obs_attr(s,combat,"deferred_effect"));rr_set_attr(s,result,"round_paused",rr_scalar(s,WOTR_BOOL,0));
    for(const char* key:{"casualties_deferred","deferred_hits_on_defender","deferred_hits_on_attacker","attacker_card_cancelled","defender_card_cancelled"}){auto value=obs_attr(s,combat,key,false);if(value==missing)value=rr_scalar(s,std::string(key).find("hits")!=std::string::npos?WOTR_INT:WOTR_BOOL,0);rr_set_attr(s,result,key,value);}
    crs_write_finished(s,tables,combat,payload,owned.random.get());const auto ids=child_edges(s,field(s,pc,"_round_cards"));require(ids.size()==2,"Published battle must retain both card IDs");crs_accept_completed(owned,tables,result,ids[0],ids[1],events,rows);owned.state=ac_compact(*owned.state);return owned;
}
crr_Result combat_resume_settle_owned(crr_Result owned,const Snapshot& tables,int64_t selection_index,am_Builder* events=nullptr,std::vector<uint64_t>* rows=nullptr){
    auto& s=*owned.state;const auto pc=obs_attr(s,s.root,"pending_combat"),payload=field(s,pc,"_round_resume");require(!truth(s,field(s,payload,"done",false)),"Battle resume is already complete");require(selection_index>=0,"Battle resume requires an explicit pending choice");
    // Original completed-round publication keeps the pre-advance ledgers.
    // Save the small ordered value maps before mutating this owned graph.
    const std::array<const char*,2> keys={"downgrades","reinforcement_baseline"};std::array<std::vector<std::pair<std::string,int64_t>>,2> saved;
    for(size_t k=0;k<keys.size();++k){const auto entries=child_edges(s,field(s,payload,keys[k]));for(size_t i=0;i<entries.size();i+=2)saved[k].emplace_back(std::string(enum_member(s,entries[i],"game_state.Nation")),s.nodes[entries[i+1]].integer);}
    combat_round_stage_inplace(s,owned.random.get(),selection_index,true);const auto cursor=field(s,payload,"cursor");
    if(ctx_none(s,obs_attr(s,cursor,"pending")))for(size_t k=0;k<keys.size();++k){std::vector<uint64_t> entries;for(const auto& value:saved[k]){entries.push_back(mu_enum(s,"game_state.Nation",value.first));entries.push_back(rr_scalar(s,WOTR_INT,value.second));}cp_put(s,payload,keys[k],cp_container(s,WOTR_DICT,entries));}
    return crs_settle_advanced(std::move(owned),tables,events,rows);
}
crr_Result combat_resume_settle(const Snapshot& source,const Snapshot& tables,const void* random_source,int64_t selection_index,am_Builder* events=nullptr,std::vector<uint64_t>* rows=nullptr){
    crr_Result owned;owned.state=std::make_unique<Snapshot>(source);owned.random.reset(wotr_random_clone(random_source));if(!owned.random)throw std::invalid_argument(wotr_random_last_error());return combat_resume_settle_owned(std::move(owned),tables,selection_index,events,rows);
}
