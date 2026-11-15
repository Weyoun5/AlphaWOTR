// Original setup_combat, including virtual siege armies and frozen rearguards.
uint64_t cbs_meta(Snapshot& s,const Snapshot& tables,uint64_t value){
    const auto& node=tables.nodes[value];switch(node.tag){
        case WOTR_NONE:return rr_scalar(s,WOTR_NONE);
        case WOTR_BOOL:case WOTR_INT:return rr_scalar(s,node.tag,node.integer);
        case WOTR_STRING:return rr_string(s,text(tables,value));
        case WOTR_ENUM:{const auto parts=child_edges(tables,value);return mu_enum(s,text(tables,parts[0]),text(tables,parts[1]));}
        case WOTR_LIST:{std::vector<uint64_t> values;for(auto item:child_edges(tables,value))values.push_back(cbs_meta(s,tables,item));return cp_container(s,WOTR_LIST,values);}
        default:throw std::invalid_argument("Unexpected combat-card metadata type");
    }
}
uint64_t cbs_card(Snapshot& s,const Snapshot& tables,const char* id){
    if(!id||!*id)return rr_scalar(s,WOTR_NONE);const auto card=field(tables,field(tables,tables.root,"cards"),id,false);if(ctx_none(tables,card))return rr_scalar(s,WOTR_NONE);
    const auto result=mu_object(s,"game_state.CombatCard",{});
    for(const char* key:{"card_id","side","deck","combat_name","combat_value","combat_req","combat_effect","requires_companion","requires_nazgul","requires_elite","requires_leader","min_nazgul_leadership","requires_fellowship_region","requires_stronghold","requires_fp_nation","region_restrictions","requires_isengard_unit","requires_se_elite","requires_shadow_elite","requires_fp_elite"})rr_set_attr(s,result,key,cbs_meta(s,tables,obs_attr(tables,card,key)));return result;
}
uint64_t cbs_army(Snapshot& s,uint64_t parent,const std::vector<uint64_t>& units,const std::vector<uint64_t>& leaders){return mu_object(s,"game_state.Army",{{"region",obs_attr(s,parent,"region")},{"units",cp_container(s,WOTR_LIST,units)},{"leaders",cp_container(s,WOTR_LIST,leaders)}});}
int cbs_side(const Snapshot& s,uint64_t army){const auto units=child_edges(s,obs_attr(s,army,"units"));return units.empty()?0:ctx_unit_side(s,units.front());}
uint64_t cbs_lookup(const Snapshot& s,std::string_view region){
    const auto map=obs_attr(s,s.root,"armies");auto army=field(s,map,std::string(region).c_str(),false);if(!ctx_none(s,army))return army;army=field(s,map,sanitize_name(region).c_str(),false);return !ctx_none(s,army)?army:mu_army(s,sanitize_name(region));
}
bool cbs_mixed(const Snapshot& s,uint64_t army){bool fp=false,sh=false;for(auto unit:child_edges(s,obs_attr(s,army,"units")))(ctx_unit_side(s,unit)==0?fp:sh)=true;return fp&&sh;}
std::pair<uint64_t,uint64_t> cbs_split(Snapshot& s,uint64_t parent,int side){
    std::vector<uint64_t> units,leaders,rear_units,rear_leaders;for(auto unit:child_edges(s,obs_attr(s,parent,"units")))(ctx_unit_side(s,unit)==side?units:rear_units).push_back(unit);for(auto leader:child_edges(s,obs_attr(s,parent,"leaders")))(cp_leader_side(s,leader)==side?leaders:rear_leaders).push_back(leader);
    return {cbs_army(s,parent,units,leaders),cp_container(s,WOTR_TUPLE,{cp_container(s,WOTR_LIST,rear_units),cp_container(s,WOTR_LIST,rear_leaders)})};
}
int64_t cbs_nazgul_pool(const Snapshot& s,const std::vector<uint64_t>& leaders){int64_t n=0;for(auto leader:leaders){if(object_type(s,leader)=="game_state.Nazgul")++n;else if(object_type(s,leader)=="game_logic.Minion"&&cp_has(cp_lower(text(s,obs_attr(s,leader,"name"))),"witch"))n+=cp_number(s,leader,"leadership",2,true);}return n;}
int64_t cbs_leadership(const Snapshot& s,uint64_t army,bool cancel,bool saruman,uint64_t shined){
    const auto units=child_edges(s,obs_attr(s,army,"units")),leaders=child_edges(s,obs_attr(s,army,"leaders"));const int own=units.empty()?-1:ctx_unit_side(s,units.front());int64_t value=0;
    for(auto leader:leaders){if(own>=0&&cp_leader_side(s,leader)!=own)continue;const auto type=object_type(s,leader);if(type=="game_state.Companion")value+=cp_number(s,leader,"leadership",1,true);else if(type=="game_state.Nazgul")value+=cancel?0:1;else if(type=="game_logic.Minion"){const auto name=cp_lower(text(s,obs_attr(s,leader,"name")));value+=cp_has(name,"witch")?(cancel?0:2):cp_has(name,"mouth")?2:1;}else ++value;}
    if(saruman&&own==1)for(auto unit:units)value+=enum_is(s,obs_attr(s,unit,"nation"),"game_state.Nation","ISENGARD")&&enum_is(s,obs_attr(s,unit,"kind"),"game_state.UnitKind","ELITE");
    if(!ctx_none(s,shined)&&std::find(leaders.begin(),leaders.end(),shined)!=leaders.end())value=std::max<int64_t>(0,value-cp_number(s,shined,"leadership",1,true));return value;
}
uint64_t cbs_setup(Snapshot& s,const Snapshot& tables,std::string_view attacker_region,std::string_view defender_region,uint64_t attacker_card,uint64_t defender_card,bool siege,int requested_side){
    const std::string atk_key=sanitize_name(attacker_region),def_key=sanitize_name(defender_region);auto attacker=cbs_lookup(s,attacker_region),defender=cbs_lookup(s,defender_region);require(!ctx_none(s,attacker)&&!ctx_none(s,defender),"Combat setup requires both region armies");
    auto none=[&](){return rr_scalar(s,WOTR_NONE);};uint64_t coloc=none(),away_attacker=none(),away_defender=none(),away_atk_rest=none(),away_def_rest=none();
    if(attacker==defender){const int garrison=c9_garrison(s,tables,def_key),actor=requested_side<0?1-garrison:requested_side;const auto parent=attacker;const auto [garrison_army,besieger_rest]=cbs_split(s,parent,garrison);const auto rest=child_edges(s,besieger_rest);const auto besieger=cbs_army(s,parent,child_edges(s,rest[0]),child_edges(s,rest[1]));
        auto units=child_edges(s,obs_attr(s,garrison_army,"units"));if(units.size()>5&&contains_string(s,obs_attr(s,s.root,"besieged_strongholds",false),def_key)&&!truth(s,field(s,obs_attr(s,s.root,"pending_combat",false),"corsairs_relief_garrison",false))){const auto excess=units.size()-5;std::vector<uint64_t> removable;for(const char* kind:{"REGULAR","ELITE"})for(auto unit:units)if(enum_is(s,obs_attr(s,unit,"kind"),"game_state.UnitKind",kind))removable.push_back(unit);for(size_t i=0;i<excess;++i){const auto found=std::find(units.begin(),units.end(),removable[i]);require(found!=units.end(),"Garrison cap figure missing");units.erase(found);}set_edges(s,obs_attr(s,garrison_army,"units"),units);}
        attacker=actor==garrison?garrison_army:besieger;defender=actor==garrison?besieger:garrison_army;require(truth(s,obs_attr(s,attacker,"units"))&&truth(s,obs_attr(s,defender,"units")),"Co-located battle needs units on both sides");coloc=parent;
    }
    if(ctx_none(s,coloc)){
        if(cbs_mixed(s,attacker)){const auto split=cbs_split(s,attacker,1-c9_garrison(s,tables,atk_key));if(truth(s,obs_attr(s,split.first,"units"))){away_attacker=attacker;away_atk_rest=split.second;attacker=split.first;}}
        if(cbs_mixed(s,defender)){const auto split=cbs_split(s,defender,1-cbs_side(s,attacker));if(truth(s,obs_attr(s,split.first,"units"))){away_defender=defender;away_def_rest=split.second;defender=split.first;}}
    }
    const auto formation=ca_partition(s,attacker,atk_key,requested_side<0?cbs_side(s,attacker):requested_side);const auto rear=cp_container(s,WOTR_TUPLE,{cp_container(s,WOTR_LIST,formation.rear_units),cp_container(s,WOTR_LIST,formation.rear_leaders)});
    if(!formation.rear_units.empty()||!formation.rear_leaders.empty()){
        if(ctx_none(s,coloc)){if(ctx_none(s,away_attacker)){away_attacker=attacker;away_atk_rest=cp_container(s,WOTR_TUPLE,{cp_container(s,WOTR_LIST,formation.rear_units),cp_container(s,WOTR_LIST,formation.rear_leaders)});}else{const auto prior=child_edges(s,away_atk_rest);auto ru=child_edges(s,prior[0]),rl=child_edges(s,prior[1]);ru.insert(ru.end(),formation.rear_units.begin(),formation.rear_units.end());rl.insert(rl.end(),formation.rear_leaders.begin(),formation.rear_leaders.end());away_atk_rest=cp_container(s,WOTR_TUPLE,{cp_container(s,WOTR_LIST,ru),cp_container(s,WOTR_LIST,rl)});}}
        attacker=cbs_army(s,attacker,formation.units,formation.leaders);
    }
    const int atk_side=cbs_side(s,attacker),def_side=cbs_side(s,defender);require(atk_side!=def_side,"Cannot attack friendly army");const std::array<uint64_t,2> armies={attacker,defender};const std::array<int,2> sides={atk_side,def_side};std::array<std::vector<uint64_t>,2> own,companions,minions;
    for(int role=0;role<2;++role)for(auto leader:child_edges(s,obs_attr(s,armies[role],"leaders")))if(cp_leader_side(s,leader)==sides[role]){own[role].push_back(leader);if(object_type(s,leader)=="game_state.Companion")companions[role].push_back(leader);if(object_type(s,leader)=="game_logic.Minion")minions[role].push_back(obs_attr(s,leader,"name"));}
    if(def_key=="minas tirith"&&cau_persistent(s,"DENETHORS_FOLLY")!=missing)(atk_side==0?attacker_card:defender_card)=none();
    auto companion_order=companions[0];companion_order.insert(companion_order.end(),companions[1].begin(),companions[1].end());uint64_t white=none();for(auto companion:companion_order)if(truth(s,obs_attr(s,companion,"can_cancel_nazgul_leadership",false))||text_is(s,obs_attr(s,companion,"name"),"Gandalf the White")){white=companion;break;}
    const auto pc=obs_attr(s,s.root,"pending_combat",false),choice=ctx_none(s,pc)?missing:field(s,pc,"gtw_shine",false);const bool explicit_false=choice!=missing&&s.nodes[choice].tag==WOTR_BOOL&&!s.nodes[choice].integer,explicit_true=choice!=missing&&s.nodes[choice].tag==WOTR_BOOL&&s.nodes[choice].integer;const bool shines=!ctx_none(s,white)&&!explicit_false;const auto shined=explicit_true&&!ctx_none(s,white)?white:none();bool saruman=false;
    const auto physical=child_edges(s,obs_attr(s,s.root,"armies"));for(size_t i=1;i<physical.size();i+=2)for(auto figure:child_edges(s,obs_attr(s,physical[i],"leaders")))saruman|=object_type(s,figure)=="game_logic.Minion"&&cp_has(cp_lower(text(s,obs_attr(s,figure,"name"))),"saruman");
    const auto meta=field(tables,field(tables,tables.root,"board_regions"),def_key.c_str(),false);auto flag=[&](const char* key){return !ctx_none(tables,meta)&&truth(tables,field(tables,meta,key,false));};
    const auto combat=mu_object(s,"game_state.CombatState",{{"attacker_region",rr_string(s,atk_key)},{"defender_region",rr_string(s,def_key)},{"attacker_army",attacker},{"defender_army",defender},{"attacker_side",mu_enum(s,"game_state.Side",atk_side==0?"FREE_PEOPLES":"SHADOW")},{"defender_side",mu_enum(s,"game_state.Side",def_side==0?"FREE_PEOPLES":"SHADOW")},{"is_siege",rr_scalar(s,WOTR_BOOL,siege)},{"siege_round",rr_scalar(s,WOTR_INT,1)},{"stronghold_defense",rr_scalar(s,WOTR_BOOL,flag("is_stronghold"))},{"city_defense",rr_scalar(s,WOTR_BOOL,(flag("is_city")||flag("is_fort"))&&!siege)},{"attacker_card",attacker_card},{"defender_card",defender_card}});
    for(const char* key:{"attacker_card_cancelled","defender_card_cancelled"})rr_set_attr(s,combat,key,rr_scalar(s,WOTR_BOOL,0));
    cp_set_number(s,combat,"attacker_combat_strength",cp_strength(s,attacker,missing));cp_set_number(s,combat,"defender_combat_strength",cp_strength(s,defender,missing));
    for(const char* key:{"attacker_leadership","defender_leadership","attacker_dice_mod","defender_dice_mod","attacker_reroll_mod","defender_reroll_mod"})cp_set_number(s,combat,key,0);
    for(const char* key:{"attacker_dice_cap","defender_dice_cap"})rr_set_attr(s,combat,key,none());
    for(const char* key:{"attacker_combat_rolls","defender_combat_rolls","attacker_rerolls","defender_rerolls","attacker_pre_combat_rolls","defender_pre_combat_rolls"})rr_set_attr(s,combat,key,cp_container(s,WOTR_LIST,{}));
    for(const char* key:{"attacker_hits","defender_hits","attacker_pre_combat_hits","defender_pre_combat_hits","attacker_bonus_hits","defender_bonus_hits"})cp_set_number(s,combat,key,0);
    for(const char* key:{"attacker_reroll_cancelled","defender_reroll_cancelled"})rr_set_attr(s,combat,key,rr_scalar(s,WOTR_BOOL,0));
    auto all_companions=companions[1];all_companions.insert(all_companions.end(),companions[0].begin(),companions[0].end());rr_set_attr(s,combat,"companions_in_battle",cp_container(s,WOTR_LIST,all_companions));cp_set_number(s,combat,"nazgul_in_battle",0);auto all_minions=minions[0];all_minions.insert(all_minions.end(),minions[1].begin(),minions[1].end());rr_set_attr(s,combat,"minions_in_battle",cp_container(s,WOTR_LIST,all_minions));
    for(const char* key:{"attacker_confusion","defender_confusion"})rr_set_attr(s,combat,key,rr_scalar(s,WOTR_BOOL,0));for(const char* key:{"attacker_hits_cancelled","defender_hits_cancelled"})cp_set_number(s,combat,key,0);rr_set_attr(s,combat,"deferred_effect",none());for(const char* key:{"attacker_casualties","defender_casualties"})rr_set_attr(s,combat,key,cp_container(s,WOTR_LIST,{}));for(const char* key:{"battle_over","attacker_retreated","defender_retreated","attacker_destroyed","defender_destroyed"})rr_set_attr(s,combat,key,rr_scalar(s,WOTR_BOOL,0));
    rr_set_attr(s,combat,"_shined_companion",shined);rr_set_attr(s,combat,"_saruman_in_play",rr_scalar(s,WOTR_BOOL,saruman));rr_set_attr(s,combat,"_wop_cancelled",none());for(const char* key:{"_wop_cancelled_is_attacker","_foul_stench_cancel_atk","_foul_stench_cancel_def"})rr_set_attr(s,combat,key,rr_scalar(s,WOTR_BOOL,0));rr_set_attr(s,combat,"_shine_companion",shines?white:none());cp_set_number(s,combat,"_shine_cancelled_leadership",shines?cbs_nazgul_pool(s,own[atk_side==1?0:1]):0);
    for(int role=0;role<2;++role)cp_set_number(s,combat,cp_role(role)+"_leadership",cbs_leadership(s,armies[role],shines&&sides[role]==1,saruman,shined));cp_set_number(s,combat,"nazgul_in_battle",cbs_nazgul_pool(s,own[0])+cbs_nazgul_pool(s,own[1]));
    rr_set_attr(s,combat,"coloc_parent",coloc);rr_set_attr(s,combat,"coloc_rear",rear);rr_set_attr(s,combat,"away_atk_parent",away_attacker);rr_set_attr(s,combat,"away_atk_rest",away_atk_rest);rr_set_attr(s,combat,"away_def_parent",away_defender);rr_set_attr(s,combat,"away_def_rest",away_def_rest);rr_set_attr(s,combat,"_live_state",s.root);return combat;
}
std::unique_ptr<Snapshot> combat_setup(const Snapshot& source,const Snapshot& tables,const char* attacker_region,const char* defender_region,const char* attacker_card,const char* defender_card,bool siege,int requested_side){
    require(attacker_region&&defender_region&&requested_side>=-1&&requested_side<=1,"Combat setup requires explicit regions and optional actor");auto result=std::make_unique<Snapshot>(source);const auto ac=cbs_card(*result,tables,attacker_card),dc=cbs_card(*result,tables,defender_card);result->root=cbs_setup(*result,tables,attacker_region,defender_region,ac,dc,siege,requested_side);return ac_compact(*result);
}
