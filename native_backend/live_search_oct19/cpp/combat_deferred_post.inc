// Source resume_deferred_post_casualty with defer_onslaught/defer_post_hits:
// no policy decisions are swallowed, and enemy damage returns for allocation.
struct c9_Result{crr_Result owned;bool attacker_destroyed=false,defender_destroyed=false;int64_t role=-1,hits_on_attacker=0,hits_on_defender=0;};
int c9_garrison(const Snapshot& s,const Snapshot& tables,std::string_view region){
    const auto controls=obs_attr(s,s.root,"enemy_controlled");for(int side:{1,0})if(contains_string(s,ctx_side_value(s,controls,side),region))return side;
    return contains_string(tables,field(tables,tables.root,"fp_region_keys"),region)?0:1;
}
void c9_write(Snapshot& s,const Snapshot& tables,std::string_view region,int side,const std::vector<uint64_t>& own){
    const auto army=mu_army(s,sanitize_name(region));if(army==missing)return;
    const auto others=ca_partition(s,army,region,side).rear_units;std::vector<uint64_t> entries;
    if(c9_garrison(s,tables,sanitize_name(region))==side){entries=own;entries.insert(entries.end(),others.begin(),others.end());}
    else{entries=others;entries.insert(entries.end(),own.begin(),own.end());}
    rr_set_attr(s,army,"units",cp_container(s,WOTR_LIST,entries));
}
c9_Result combat_deferred_post(const Snapshot& source,const Snapshot& tables,const void* random_source){
    const auto pc=obs_attr(source,source.root,"pending_combat"),phase9=field(source,pc,"_p9");typed(source,phase9,WOTR_DICT,"Deferred post-casualty phase requires card context");
    const std::array<std::string,2> regions={std::string(text(source,field(source,pc,"attacker_region"))),std::string(text(source,field(source,pc,"defender_region")))};
    std::array<int,2> sides{};for(int role=0;role<2;++role){const auto node=field(source,pc,role==0?"attacker_side":"defender_side");const auto member=enum_member(source,node,"game_state.Side");require(member=="FREE_PEOPLES"||member=="SHADOW","Deferred battle requires actual player sides");sides[role]=member=="FREE_PEOPLES"?0:1;}
    std::array<std::vector<uint64_t>,2> units;for(int role=0;role<2;++role)units[role]=ca_partition(source,mu_army(source,sanitize_name(regions[role])),regions[role],sides[role]).units;
    std::array<uint64_t,2> cards{};std::array<int64_t,2> initiative{};for(int role=0;role<2;++role){
        const auto cid=field(source,phase9,role==0?"atk_card":"def_card",false);cards[role]=ctx_none(source,cid)||!truth(source,cid)?missing:field(tables,field(tables,tables.root,"cards"),std::string(text(source,cid)).c_str(),false);
        if(truth(source,field(source,phase9,role==0?"atk_cancelled":"def_cancelled",false)))cards[role]=missing;
        initiative[role]=cards[role]==missing?99:cp_number(tables,cards[role],"combat_value");
    }
    c9_Result result;result.attacker_destroyed=units[0].empty();result.defender_destroyed=units[1].empty();result.owned.state=std::make_unique<Snapshot>(source);result.owned.random.reset(wotr_random_clone(random_source));if(!result.owned.random)throw std::invalid_argument(wotr_random_last_error());
    const bool finished=result.attacker_destroyed||result.defender_destroyed,defender_first=initiative[1]<=initiative[0];
    for(int step=0;step<2;++step){const int own=(step==0?defender_first:!defender_first)?1:0,enemy=1-own;
        if((finished&&units[own].empty())||cards[own]==missing)continue;
        const auto effect=cp_lower(text(tables,obs_attr(tables,cards[own],"combat_effect")));
        const bool onslaught=(cp_has(effect,"up to four additional hits")&&cp_has(effect,"4+"))||(cp_has(effect,"four additional hits")&&cp_has(effect,"score one hit"));
        if(onslaught){if(!units[own].empty())result.role=own;continue;}
        if(units[enemy].empty())continue;int64_t hits=0;
        if(cp_has(effect,"elite")&&cp_has(effect,"additional")&&cp_has(effect,"after removing casualties")){
            size_t elite=0;for(auto unit:units[own])elite+=enum_is(source,obs_attr(source,unit,"kind"),"game_state.UnitKind","ELITE");elite=std::min<size_t>(5,elite);
            if(elite){std::array<uint32_t,5> dice{};cf_d6(result.owned.random.get(),dice.data(),elite,5,0,true);for(size_t i=0;i<elite;++i)hits+=crr_hit(dice[i],5,0);}
        }
        if(cp_has(effect,"twice as many")&&cp_has(effect,"automatic hit")&&units[own].size()>=2*units[enemy].size())++hits;
        if(own==0)result.hits_on_defender+=hits;else result.hits_on_attacker+=hits;
    }
    auto& state=*result.owned.state;for(int role=0;role<2;++role)c9_write(state,tables,regions[role],sides[role],units[role]);
    if((sides[0]==0||sides[1]==0)&&ctx_none(state,obs_attr(state,state.root,"fp_casualties",false)))rr_set_attr(state,state.root,"fp_casualties",cp_container(state,WOTR_DICT,{}));
    result.owned.state=ac_compact(state);return result;
}
