// Exact _run_combat_round entry, through the next genuine decision boundary.
// crs_settle_advanced shares completion/publication with resumed rounds.
crr_Result crs_settle_advanced(crr_Result owned,const Snapshot& tables,am_Builder* events,std::vector<uint64_t>* rows);
uint64_t cbb_take(Snapshot& s,uint64_t mapping,const char* key){auto value=field(s,mapping,key,false);cp_erase(s,mapping,key);return value==missing?rr_scalar(s,WOTR_NONE):value;}
int cbb_side(const Snapshot& s,uint64_t pc,int role){return enum_is(s,field(s,pc,role==0?"attacker_side":"defender_side"),"game_state.Side","FREE_PEOPLES")?0:1;}
bool cbb_war(const Snapshot& s,uint64_t nation){const auto boxes=obs_attr(s,obs_attr(s,s.root,"political"),"boxes"),box=ctx_nation_value(s,boxes,enum_member(s,nation,"game_state.Nation"));require(!ctx_none(s,box)&&s.nodes[box].tag==WOTR_INT&&s.nodes[box].integer>=0&&s.nodes[box].integer<=3,"Attack formation requires a valid political position");return s.nodes[box].integer==3;}
void cbb_freeze(Snapshot& s,uint64_t pc){
    if(!ctx_none(s,field(s,pc,"attacker_rearguard",false)))return;const auto region=sanitize_name(text(s,field(s,pc,"attacker_region")));const int side=cbb_side(s,pc,0);const auto army=mu_army(s,region);require(!ctx_none(s,army),"Attack formation requires an Army");std::vector<uint64_t> attack,rear_units,rear_leaders;
    auto eligible_units=child_edges(s,obs_attr(s,army,"units")),eligible_leaders=child_edges(s,obs_attr(s,army,"leaders"));
    if(truth(s,field(s,pc,"corsairs_relief_garrison",false))){const auto arriving=ca_partition(s,army,region,side,pc);eligible_units=arriving.units;eligible_leaders=arriving.leaders;}
    for(auto unit:eligible_units)if(ctx_unit_side(s,unit)==side)(cbb_war(s,obs_attr(s,unit,"nation"))?attack:rear_units).push_back(unit);
    for(auto leader:eligible_leaders){if(cp_leader_side(s,leader)!=side)continue;const auto type=object_type(s,leader);if(type=="game_state.Companion"||type=="game_state.Nazgul"||type=="game_logic.Minion")continue;if(!cbb_war(s,obs_attr(s,leader,"nation")))rear_leaders.push_back(leader);}
    if(!rear_leaders.empty()&&rear_units.empty()&&!attack.empty()){
        const auto chosen=std::min_element(attack.begin(),attack.end(),[&](uint64_t a,uint64_t b){const bool ae=!enum_is(s,obs_attr(s,a,"kind"),"game_state.UnitKind","REGULAR"),be=!enum_is(s,obs_attr(s,b,"kind"),"game_state.UnitKind","REGULAR");if(ae!=be)return ae<be;return enum_member(s,obs_attr(s,a,"nation"),"game_state.Nation")<enum_member(s,obs_attr(s,b,"nation"),"game_state.Nation");});rear_units.push_back(*chosen);attack.erase(chosen);
    }
    require(!attack.empty(),"Cannot form attacking army: no eligible unit remains after forced guards");std::map<std::pair<std::string,std::string>,int64_t> units;std::map<std::string,int64_t> leaders;
    for(auto unit:rear_units)++units[{std::string(enum_member(s,obs_attr(s,unit,"nation"),"game_state.Nation")),std::string(enum_member(s,obs_attr(s,unit,"kind"),"game_state.UnitKind"))}];for(auto leader:rear_leaders)++leaders[std::string(enum_member(s,obs_attr(s,leader,"nation"),"game_state.Nation"))];std::vector<uint64_t> us,ls;
    for(const auto& [key,count]:units)us.push_back(cp_container(s,WOTR_TUPLE,{rr_string(s,key.first),rr_string(s,key.second),rr_scalar(s,WOTR_INT,count)}));for(const auto& [nation,count]:leaders)ls.push_back(cp_container(s,WOTR_TUPLE,{rr_string(s,nation),rr_scalar(s,WOTR_INT,count)}));
    cp_put(s,pc,"attacker_rearguard",cp_container(s,WOTR_TUPLE,{rr_string(s,region),rr_string(s,side==0?"FREE_PEOPLES":"SHADOW"),cp_container(s,WOTR_TUPLE,{rr_string(s,"forced_rearguard_v1"),cp_container(s,WOTR_TUPLE,us),cp_container(s,WOTR_TUPLE,ls)})}));
}
bool cbb_shine(Snapshot& s,const Snapshot& tables,uint64_t pc){
    if(truth(s,field(s,pc,"shine_asked",false)))return false;const auto temp=cbs_setup(s,tables,text(s,field(s,pc,"attacker_region")),text(s,field(s,pc,"defender_region")),rr_scalar(s,WOTR_NONE),rr_scalar(s,WOTR_NONE),truth(s,field(s,pc,"is_siege",false)),cbb_side(s,pc,0));bool white=false,nazgul=false;
    for(auto companion:child_edges(s,obs_attr(s,temp,"companions_in_battle")))white|=truth(s,obs_attr(s,companion,"can_cancel_nazgul_leadership",false))||text_is(s,obs_attr(s,companion,"name"),"Gandalf the White");const auto shadow=obs_attr(s,temp,cbb_side(s,pc,0)==1?"attacker_army":"defender_army");for(auto leader:child_edges(s,obs_attr(s,shadow,"leaders"))){std::string name;const auto original=obs_attr(s,leader,"name",false);if(!ctx_none(s,original))for(char c:cp_lower(text(s,original)))if(c>='a'&&c<='z')name+=c;nazgul|=object_type(s,leader)=="game_state.Nazgul"||cp_has(name,"witchking");}
    if(white&&nazgul){cp_put(s,pc,"phase",rr_string(s,"fp_shine"));cp_put(s,pc,"deciding_side",mu_enum(s,"game_state.Side","FREE_PEOPLES"));return true;}cp_put(s,pc,"shine_asked",rr_scalar(s,WOTR_BOOL,1));return false;
}
bool cbb_card_phase(Snapshot& s,uint64_t pc){
    if(field(s,pc,"atk_combat_card",false)==missing){cp_put(s,pc,"phase",rr_string(s,"attacker_combat_card"));cp_put(s,pc,"deciding_side",field(s,pc,"attacker_side"));return true;}
    if(field(s,pc,"def_combat_card",false)==missing){const auto rounds=field(s,pc,"rounds_fought",false);if(truth(s,field(s,pc,"no_def_card_r1_unless_companion",false))&&(ctx_none(s,rounds)||s.nodes[rounds].integer==0)){const auto army=mu_army(s,sanitize_name(text(s,field(s,pc,"defender_region"))));bool companion=false;if(!ctx_none(s,army))for(auto leader:child_edges(s,obs_attr(s,army,"leaders")))companion|=object_type(s,leader)=="game_state.Companion";if(!companion){cp_put(s,pc,"def_combat_card",rr_scalar(s,WOTR_NONE));return false;}}
        cp_put(s,pc,"phase",rr_string(s,"defender_combat_card"));cp_put(s,pc,"deciding_side",field(s,pc,"defender_side"));return true;
    }return false;
}
bool cbb_word_choice(Snapshot& s,const Snapshot& tables,uint64_t pc){
    for(int role=0;role<2;++role){const std::string prefix=role==0?"atk":"def";if(truth(s,field(s,pc,(prefix+"_effect_asked").c_str(),false)))continue;cp_put(s,pc,(prefix+"_effect_asked").c_str(),rr_scalar(s,WOTR_BOOL,1));const auto id=field(s,pc,(prefix+"_combat_card").c_str(),false);if(!truth(s,id))continue;const auto card=field(tables,field(tables,tables.root,"cards"),std::string(text(s,id)).c_str(),false);if(ctx_none(tables,card)||!cp_has(cp_lower(text(tables,obs_attr(tables,card,"combat_name"))),"words of power"))continue;
        const auto temp=cbs_setup(s,tables,text(s,field(s,pc,"attacker_region")),text(s,field(s,pc,"defender_region")),rr_scalar(s,WOTR_NONE),rr_scalar(s,WOTR_NONE),truth(s,field(s,pc,"is_siege",false)),cbb_side(s,pc,0));std::vector<uint64_t> targets;for(auto companion:child_edges(s,obs_attr(s,temp,"companions_in_battle")))targets.push_back(obs_attr(s,companion,"name"));if(targets.size()<=1)continue;
        cp_put(s,pc,"effect_choice",cp_container(s,WOTR_DICT,{rr_string(s,"stage"),rr_string(s,"pre"),rr_string(s,"owner"),rr_string(s,prefix),rr_string(s,"effect"),rr_string(s,"words_of_power"),rr_string(s,"card_id"),id,rr_string(s,"targets"),cp_container(s,WOTR_LIST,targets)}));cp_put(s,pc,"phase",rr_string(s,"combat_effect_choice"));cp_put(s,pc,"deciding_side",field(s,pc,role==0?"attacker_side":"defender_side"));return true;
    }return false;
}
bool cbb_retreat_card(Snapshot& s,const Snapshot& tables,uint64_t pc){
    const auto rounds=field(s,pc,"rounds_fought",false);
    const auto round=ctx_none(s,rounds)?0:typed(s,rounds,WOTR_INT,"Combat rounds fought must be integer").integer;
    const auto offered=field(s,pc,"_card_retreat_round",false);
    if(!ctx_none(s,offered)&&typed(s,offered,WOTR_INT,"Card retreat round must be integer").integer==round)return false;
    uint64_t cards[2]={missing,missing};bool retreat[2]={false,false};
    for(int role=0;role<2;++role){
        const auto id=field(s,pc,role==0?"atk_combat_card":"def_combat_card",false);
        if(truth(s,id))cards[role]=field(tables,field(tables,tables.root,"cards"),std::string(text(s,id)).c_str(),false);
        retreat[role]=!ctx_none(tables,cards[role])&&cp_has(cp_lower(text(tables,obs_attr(tables,cards[role],"combat_effect"))),"retreat your army");
    }
    if(!retreat[0]&&!retreat[1])return false;
    // Preview initiative-zero cancellation on a temporary combat object. No
    // committed card, board figure, pending choice, or random state is changed.
    uint64_t preview_cards[2];
    for(int role=0;role<2;++role){
        const auto id=field(s,pc,role==0?"atk_combat_card":"def_combat_card",false);
        preview_cards[role]=cbs_card(s,tables,truth(s,id)?std::string(text(s,id)).c_str():nullptr);
    }
    const auto preview=cbs_setup(s,tables,text(s,field(s,pc,"attacker_region")),text(s,field(s,pc,"defender_region")),
                                 preview_cards[0],preview_cards[1],truth(s,field(s,pc,"is_siege",false)),cbb_side(s,pc,0));
    ce_begin(s,preview);
    for(int role:{1,0}){
        if(!retreat[role]||truth(s,field(s,pc,"is_siege",false))||
                truth(s,obs_attr(s,preview,role==0?"attacker_card_cancelled":"defender_card_cancelled")))continue;
        cp_put(s,pc,"_card_retreat_round",rr_scalar(s,WOTR_INT,round));
        cp_put(s,pc,"_card_retreat_mandatory",rr_scalar(s,WOTR_BOOL,1));
        cp_put(s,pc,"phase",rr_string(s,role==0?"attacker_card_retreat":"defender_retreat"));
        cp_put(s,pc,"deciding_side",field(s,pc,role==0?"attacker_side":"defender_side"));
        return true;
    }
    return false;
}
uint64_t cbb_cursor(Snapshot& s,uint64_t combat,uint64_t choices){
    const auto cursor=mu_object(s,"combat_round_resume.RoundCursor",{{"combat",combat},{"choices",cp_container(s,WOTR_DICT,child_edges(s,choices))},{"stage",rr_scalar(s,WOTR_INT,0)},{"pending",rr_scalar(s,WOTR_NONE)},{"selection",rr_scalar(s,WOTR_NONE)},{"observed_rolls",cp_container(s,WOTR_SET,{})},{"observed_roll_sides",cp_container(s,WOTR_SET,{})},{"paid_leadership_targets",cp_container(s,WOTR_LIST,{})},{"ra_remaining",rr_scalar(s,WOTR_NONE)},{"ra_paid",rr_scalar(s,WOTR_INT,0)},{"ra_attacker",rr_scalar(s,WOTR_NONE)},{"pre_attack_index",rr_scalar(s,WOTR_INT,0)},{"pre_remaining",rr_scalar(s,WOTR_NONE)},{"pre_victim_attacker",rr_scalar(s,WOTR_NONE)},{"pre_total",rr_scalar(s,WOTR_INT,0)},{"pre_paid",rr_scalar(s,WOTR_INT,0)},{"schema",rr_string(s,"combat_round_cursor_v1")}});return cursor;
}
void cbb_run(crr_Result& owned,const Snapshot& tables,int siege_override,am_Builder& events,std::vector<uint64_t>& rows){
    auto& s=*owned.state;const auto pc=obs_attr(s,s.root,"pending_combat");require(ctx_none(s,field(s,pc,"_round_resume",false)),"Paused combat must resume with its observed-stage choice");cp_erase(s,pc,"_observed_round_completed");const auto fought=field(s,pc,"rounds_fought",false);if(ctx_none(s,field(s,pc,"attacker_rearguard",false))&&!truth(s,fought))cbb_freeze(s,pc);
    const std::array<std::string,2> region={std::string(text(s,field(s,pc,"attacker_region"))),std::string(text(s,field(s,pc,"defender_region")))};const std::array<int,2> side={cbb_side(s,pc,0),cbb_side(s,pc,1)};const std::array<bool,2> dead={ca_units(s,region[0],side[0]).empty(),ca_units(s,region[1],side[1]).empty()};
    if(dead[0]||dead[1]){for(int role=0;role<2;++role)if(dead[role])cbe_reconcile(s,region[role],side[role],owned.random.get());if(dead[1]&&!dead[0]&&sanitize_name(region[0])==sanitize_name(region[1])&&c9_garrison(s,tables,sanitize_name(region[1]))==side[1])crs_capture(s,tables,region[1],side[0]);cbe_finish(s,tables,pc,dead[0],dead[1],&events,&rows);return;}
    if(truth(s,fought))crs_witch_king_draw(s,pc,&events,&rows);
    for(const char* key:{"atk_casualty_policy","def_casualty_policy"})if(field(s,pc,key,false)==missing)cp_put(s,pc,key,rr_string(s,"preserve_elites"));if(siege_override>=0)cp_put(s,pc,"is_siege",rr_scalar(s,WOTR_BOOL,siege_override));
    if(cbb_shine(s,tables,pc)||cbb_card_phase(s,pc))return;if(truth(s,field(s,pc,"atk_hidden_cc",false))||truth(s,field(s,pc,"def_hidden_cc",false))){cp_put(s,pc,"phase",rr_string(s,"combat_card_reveal"));cp_put(s,pc,"deciding_side",rr_scalar(s,WOTR_NONE));return;}if(cbb_word_choice(s,tables,pc)||cbb_retreat_card(s,tables,pc))return;
    const auto ac=cbb_take(s,pc,"atk_combat_card"),dc=cbb_take(s,pc,"def_combat_card"),a_count=cbb_take(s,pc,"atk_effect_count"),d_count=cbb_take(s,pc,"def_effect_count"),a_target=cbb_take(s,pc,"atk_effect_target"),d_target=cbb_take(s,pc,"def_effect_target");cp_erase(s,pc,"atk_effect_asked");cp_erase(s,pc,"def_effect_asked");rr_set_attr(s,s.root,"_last_combat_cards",cp_container(s,WOTR_TUPLE,{ac,dc}));
    const bool siege=truth(s,field(s,pc,"is_siege",false)),disable=!ctx_none(s,fought)&&typed(s,fought,WOTR_INT,"Combat rounds fought must be integer").integer>0;auto optional=[&](const char* key,uint64_t fallback){const auto v=field(s,pc,key,false);return v==missing?fallback:v;};
    const auto choices=cp_container(s,WOTR_DICT,{rr_string(s,"attacker_side"),field(s,pc,"attacker_side"),rr_string(s,"disable_city_defense"),rr_scalar(s,WOTR_BOOL,disable),rr_string(s,"atk_casualty_policy"),field(s,pc,"atk_casualty_policy"),rr_string(s,"def_casualty_policy"),field(s,pc,"def_casualty_policy"),rr_string(s,"atk_effect_count"),a_count,rr_string(s,"def_effect_count"),d_count,rr_string(s,"atk_effect_target"),a_target,rr_string(s,"def_effect_target"),d_target,rr_string(s,"attacker_dice_reduction"),optional("atk_dice_reduction",rr_scalar(s,WOTR_INT,0)),rr_string(s,"defender_dice_reduction"),optional("def_dice_reduction",rr_scalar(s,WOTR_INT,0)),rr_string(s,"defer_casualties"),rr_scalar(s,WOTR_BOOL,1),rr_string(s,"defer_onslaught"),rr_scalar(s,WOTR_BOOL,1),rr_string(s,"defer_round_choices"),rr_scalar(s,WOTR_BOOL,1)});
    rr_set_attr(s,s.root,"_last_round_call",cp_container(s,WOTR_DICT,{rr_string(s,"atk_region"),field(s,pc,"attacker_region"),rr_string(s,"def_region"),field(s,pc,"defender_region"),rr_string(s,"atk_side"),field(s,pc,"attacker_side"),rr_string(s,"def_side"),field(s,pc,"defender_side"),rr_string(s,"atk_card"),ac,rr_string(s,"def_card"),dc,rr_string(s,"attacker_rearguard"),optional("attacker_rearguard",rr_scalar(s,WOTR_NONE)),rr_string(s,"is_siege"),rr_scalar(s,WOTR_BOOL,siege),rr_string(s,"gtw_shine"),optional("gtw_shine",rr_scalar(s,WOTR_NONE)),rr_string(s,"choices"),cp_container(s,WOTR_DICT,child_edges(s,choices))}));
    std::vector<uint64_t> baseline;for(const char* nation:{"DWARVES","ELVES","GONDOR","THE_NORTH","ROHAN","ISENGARD","SAURON","SOUTHRON_EASTERLINGS"}){baseline.push_back(mu_enum(s,"game_state.Nation",nation));baseline.push_back(rr_scalar(s,WOTR_INT,mu_available(s,tables,nation,"regular")));}
    const auto combat=cbs_setup(s,tables,region[0],region[1],cbs_card(s,tables,ctx_none(s,ac)?nullptr:std::string(text(s,ac)).c_str()),cbs_card(s,tables,ctx_none(s,dc)?nullptr:std::string(text(s,dc)).c_str()),siege,side[0]);for(const char* key:{"attacker_dice_reduction","defender_dice_reduction"}){const auto n=field(s,choices,key);cp_set_number(s,combat,key,truth(s,n)?typed(s,n,WOTR_INT,"Battle reduction must be integer").integer:0);}if(disable)rr_set_attr(s,combat,"city_defense",rr_scalar(s,WOTR_BOOL,0));
    const auto payload=cp_container(s,WOTR_DICT,{rr_string(s,"schema"),rr_string(s,"combat_battle_resume_v1"),rr_string(s,"cursor"),cbb_cursor(s,combat,choices),rr_string(s,"args"),cp_container(s,WOTR_TUPLE,{field(s,pc,"attacker_region"),field(s,pc,"defender_region"),ac,dc,rr_scalar(s,WOTR_BOOL,siege),rr_scalar(s,WOTR_INT,1),rr_scalar(s,WOTR_BOOL,0)}),rr_string(s,"choices"),cp_container(s,WOTR_DICT,child_edges(s,choices)),rr_string(s,"published"),cp_container(s,WOTR_LIST,{rr_scalar(s,WOTR_INT,0),rr_scalar(s,WOTR_INT,0)}),rr_string(s,"done"),rr_scalar(s,WOTR_BOOL,0),rr_string(s,"totals"),cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,0),rr_scalar(s,WOTR_INT,0),rr_scalar(s,WOTR_INT,0),rr_scalar(s,WOTR_INT,1)}),rr_string(s,"downgrades"),cp_container(s,WOTR_DICT,{}),rr_string(s,"reinforcement_baseline"),cp_container(s,WOTR_DICT,baseline)});
    cp_put(s,pc,"_round_resume",payload);cp_put(s,pc,"_round_cards",cp_container(s,WOTR_TUPLE,{ac,dc}));owned=combat_round_stage(s,owned.random.get(),-1,true);owned=crs_settle_advanced(std::move(owned),tables,&events,&rows);
}
fm_Result combat_round_begin(const Snapshot& source,const Snapshot& tables,const void* random_source,int siege_override){
    require(siege_override>=-1&&siege_override<=1,"Siege override must be absent or boolean");crr_Result owned;owned.state=std::make_unique<Snapshot>(source);owned.random.reset(wotr_random_clone(random_source));if(!owned.random)throw std::invalid_argument(wotr_random_last_error());am_Builder events;std::vector<uint64_t> rows;cbb_run(owned,tables,siege_override,events,rows);fm_Result result;result.state=ac_compact(*owned.state);result.random=std::move(owned.random);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
