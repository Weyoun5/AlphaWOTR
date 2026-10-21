// Card spending and queued bonuses are shared with search-only protocol normalization.
std::string cbc_deck(std::string_view id){if(id.find("_Character_")!=std::string_view::npos)return "character";if(id.find("_Strategy_")!=std::string_view::npos)return "strategy";if(id.substr(0,3)=="HU_"){if(id.find("_CHAR_")!=std::string_view::npos)return "character";if(id.find("_STRAT_")!=std::string_view::npos)return "strategy";}return {};}
void cbc_spend_hand(Snapshot& s,int side,uint64_t id,const std::string& discard_key){
    const std::string cid(text(s,id));const auto hand=obs_attr(s,ctx_side_value(s,obs_attr(s,s.root,"hands"),side),"hand");auto entries=child_edges(s,hand);auto victim=std::find_if(entries.begin(),entries.end(),[&](uint64_t item){return text_is(s,item,cid);});
    if(victim==entries.end()){const auto deck=cbc_deck(cid);victim=std::find_if(entries.begin(),entries.end(),[&](uint64_t item){if(s.nodes[item].tag!=WOTR_STRING)return false;const auto name=text(s,item);return name.substr(0,9)=="_unknown_"||(name.substr(0,3)=="HU_"&&cbc_deck(name)==deck);});}if(victim!=entries.end()){entries.erase(victim);set_edges(s,hand,entries);}
    if(!discard_key.empty()){const auto discards=obs_attr(s,s.root,"discard_face_up",false);if(!ctx_none(s,discards)&&s.nodes[discards].tag==WOTR_DICT){const auto pile=field(s,discards,discard_key.c_str(),false);if(pile!=missing){auto old=child_edges(s,pile);old.push_back(id);set_edges(s,pile,old);}}}
}
void cbc_complete_commit(Snapshot& s,uint64_t pc,int role,uint64_t id){
    const int side=cbb_side(s,pc,role);
    if(!ctx_none(s,id)){const std::string cid(text(s,id));const auto rounds=field(s,pc,"rounds_fought",false);if(side==1&&(rounds==missing||(s.nodes[rounds].tag==WOTR_INT&&s.nodes[rounds].integer==0))){bool witch=false;const auto region=sanitize_name(text(s,field(s,pc,role==0?"attacker_region":"defender_region")));const auto army=mu_army(s,region);if(!ctx_none(s,army)){const auto participants=ca_partition(s,army,region,side,pc);if(!participants.units.empty())for(auto leader:participants.leaders)witch|=object_type(s,leader)=="game_logic.Minion"&&cp_has(cp_lower(text(s,obs_attr(s,leader,"name"))),"witch");}if(witch){auto deck=cbc_deck(cid);cp_put(s,pc,"_wk_sorcerer_deck",rr_string(s,deck.empty()?"strategy":deck));}}}
    cp_put(s,pc,role==0?"atk_combat_card":"def_combat_card",id);
}
void cbc_commit(Snapshot& s,const Snapshot& tables,uint64_t pc,int role,uint64_t id,bool from_table){
    const int side=cbb_side(s,pc,role);if(!ctx_none(s,id)){
        const std::string cid(text(s,id));if(from_table){
            const auto persistent=obs_attr(s,s.root,"persistent_cards",false);uint64_t found=missing;if(!ctx_none(s,persistent))for(const char* key:{"fp_cards","sh_cards"}){for(auto card:child_edges(s,obs_attr(s,persistent,key)))if(text_is(s,obs_attr(s,card,"card_id"),cid)){found=card;break;}if(found!=missing)break;}if(found!=missing)cbp_discard_table(s,cbp_remove_card(s,persistent,found));
        }else{
            const auto card=field(tables,field(tables,tables.root,"cards"),cid.c_str(),false);std::string key;if(!ctx_none(tables,card))key=std::string(enum_is(tables,obs_attr(tables,card,"side"),"game_state.Side","FREE_PEOPLES")?"fp_":"sh_")+cp_lower(text(tables,obs_attr(tables,card,"deck")));cbc_spend_hand(s,side,id,key);
        }
    }
    cbc_complete_commit(s,pc,role,id);
}
bool cbc_stale(crr_Result& owned,const Snapshot& tables,am_Builder* events=nullptr,std::vector<uint64_t>* rows=nullptr){
    auto& s=*owned.state;const auto pc=obs_attr(s,s.root,"pending_combat");const auto phase=field(s,pc,"phase");if(!text_is(s,phase,"attacker_combat_card")&&!text_is(s,phase,"defender_combat_card")&&!text_is(s,phase,"combat_round_result")&&!text_is(s,phase,"attacker_continue")&&!text_is(s,phase,"defender_retreat")&&!text_is(s,phase,"defender_siege_choice")&&!text_is(s,phase,"attacker_siege_decision"))return false;const std::array<std::string,2> region={std::string(text(s,field(s,pc,"attacker_region"))),std::string(text(s,field(s,pc,"defender_region")))};const std::array<int,2> side={cbb_side(s,pc,0),cbb_side(s,pc,1)};const std::array<bool,2> dead={ca_units(s,region[0],side[0]).empty(),ca_units(s,region[1],side[1]).empty()};if(!dead[0]&&!dead[1])return false;
    for(int role=0;role<2;++role)if(dead[role])cbe_reconcile(s,region[role],side[role],owned.random.get());if(dead[1]&&!dead[0]&&sanitize_name(region[0])==sanitize_name(region[1])&&c9_garrison(s,tables,sanitize_name(region[1]))==side[1])crs_capture(s,tables,region[1],side[0]);cbe_finish(s,tables,pc,dead[0],dead[1],events,rows);return true;
}
fm_Result cbc_action(const Snapshot& source,const Snapshot& tables,const void* random_source,int mode,const char* card_id,bool from_table,int64_t selection){
    crr_Result owned;owned.state=action_clone(source);owned.random.reset(wotr_random_clone(random_source));if(!owned.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*owned.state;rr_set_attr(s,s.root,"_casualty_operations",cp_container(s,WOTR_LIST,{}));cau_dedupe(s);const auto pc=obs_attr(s,s.root,"pending_combat");require(!ctx_none(s,pc),"Combat entry action requires pending battle");const int attacker=cbb_side(s,pc,0);am_Builder events;std::vector<uint64_t> rows;
    if(!cbc_stale(owned,tables,&events,&rows)){
        rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));if(truth(s,obs_attr(s,s.root,"_declare_phase",false)))rr_set_attr(s,s.root,"_declare_phase",rr_scalar(s,WOTR_BOOL,0));
        if(mode==0){const auto phase=field(s,pc,"phase");require(text_is(s,phase,"attacker_combat_card")||text_is(s,phase,"defender_combat_card"),"No combat card commitment is pending");cbc_commit(s,tables,pc,text_is(s,phase,"attacker_combat_card")?0:1,card_id?rr_string(s,card_id):rr_scalar(s,WOTR_NONE),from_table);}
        else if(mode==1){require(text_is(s,field(s,pc,"phase"),"fp_shine")&&(selection==0||selection==1),"No Shine choice is pending");cp_put(s,pc,"gtw_shine",rr_scalar(s,WOTR_BOOL,selection));cp_put(s,pc,"shine_asked",rr_scalar(s,WOTR_BOOL,1));}
        else{
            const auto effect=field(s,pc,"effect_choice");require(text_is(s,field(s,pc,"phase"),"combat_effect_choice")&&text_is(s,field(s,effect,"stage"),"pre")&&text_is(s,field(s,effect,"effect"),"words_of_power"),"No pre-roll Words of Power choice is pending");const auto targets=child_edges(s,field(s,effect,"targets"));require(selection>=0&&static_cast<size_t>(selection)<targets.size(),"Words of Power target index exceeds menu");const bool atk=text_is(s,field(s,effect,"owner"),"atk");cp_erase(s,pc,"effect_choice");cp_put(s,pc,atk?"atk_effect_target":"def_effect_target",targets[selection]);
        }
        cbb_run(owned,tables,-1,events,rows);cau_finish(*owned.state,tables,owned.random.get(),attacker,events,rows);
    }
    fm_Result result;result.state=ac_compact(*owned.state);result.random=std::move(owned.random);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
fm_Result apply_combat_card(const Snapshot& source,const Snapshot& tables,const void* random_source,const char* card_id,bool from_table){return cbc_action(source,tables,random_source,0,card_id,from_table,0);}
fm_Result apply_combat_shine(const Snapshot& source,const Snapshot& tables,const void* random_source,int64_t shine){return cbc_action(source,tables,random_source,1,nullptr,false,shine);}
fm_Result apply_combat_pre_target(const Snapshot& source,const Snapshot& tables,const void* random_source,int64_t index){return cbc_action(source,tables,random_source,2,nullptr,false,index);}
