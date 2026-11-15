// Exact raw per-die attack enumeration, with automatic compulsory rear guards.
// This family does not by itself certify a complete action menu.
struct at_Formation{std::vector<uint64_t> units,leaders;bool character=false;};
bool at_war(const Snapshot& s,uint64_t nation){
    const auto name=ctx_nation(s,nation);validate_nation_name(name);
    const auto box=ctx_nation_value(s,obs_attr(s,obs_attr(s,s.root,"political"),"boxes"),name);
    const auto value=typed(s,box,WOTR_INT,"Attack formation requires integer political position").integer;
    require(value>=0&&value<=3,"Attack formation political position out of range");return value==3;
}
at_Formation at_formation(const Snapshot& s,uint64_t army,int side){
    at_Formation result;bool rear_units=false,rear_leaders=false;
    for(auto unit:child_edges(s,obs_attr(s,army,"units"))){
        require(object_type(s,unit)=="game_state.Unit","Attack requires army units");
        const auto kind=enum_member(s,obs_attr(s,unit,"kind"),"game_state.UnitKind");require(kind=="REGULAR"||kind=="ELITE","Unknown army unit kind");
        if(ctx_unit_side(s,unit)!=side)continue;
        if(at_war(s,obs_attr(s,unit,"nation")))result.units.push_back(unit);else rear_units=true;
    }
    for(auto leader:child_edges(s,obs_attr(s,army,"leaders"))){
        if(cp_leader_side(s,leader)!=side)continue;
        if(!am_plain(s,leader)||at_war(s,obs_attr(s,leader,"nation")))result.leaders.push_back(leader);else rear_leaders=true;
    }
    if(rear_leaders&&!rear_units&&!result.units.empty()){
        auto key=[&](uint64_t unit){return std::pair<bool,std::string>{!enum_is(s,obs_attr(s,unit,"kind"),"game_state.UnitKind","REGULAR"),std::string(ctx_nation(s,obs_attr(s,unit,"nation")))};};
        const auto guard=std::min_element(result.units.begin(),result.units.end(),[&](uint64_t a,uint64_t b){return key(a)<key(b);});result.units.erase(guard);
    }
    bool leader=!result.leaders.empty();
    if(!leader){
        bool isengard_elite=false;for(auto unit:result.units)isengard_elite|=enum_is(s,obs_attr(s,unit,"nation"),"game_state.Nation","ISENGARD")&&enum_is(s,obs_attr(s,unit,"kind"),"game_state.UnitKind","ELITE");
        if(isengard_elite){const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));for(size_t i=1;i<armies.size();i+=2)for(auto figure:child_edges(s,obs_attr(s,armies[i],"leaders")))leader|=object_type(s,figure)=="game_logic.Minion"&&text_is(s,obs_attr(s,figure,"name"),"Saruman");}
    }
    result.character=!result.units.empty()&&leader;return result;
}
int at_controller(const Snapshot& s,const Snapshot& t,std::string_view key){
    const auto meta=ctx_get(t,field(t,t.root,"board_regions"),key);if(ctx_none(t,meta)||!truth(t,ctx_get(t,meta,"is_stronghold")))return -1;
    const auto nation=ctx_get(t,field(t,t.root,"region_nations"),key);if(ctx_none(t,nation))return -1;
    const int native=fp_nation(ctx_nation(t,nation))?0:1;
    return contains_string(s,ctx_side_value(s,obs_attr(s,s.root,"enemy_controlled"),1-native),key)?1-native:native;
}
bool at_shadow_blocked(const Snapshot& s,std::string_view key){
    const auto persistent=obs_attr(s,s.root,"persistent_cards",false);if(ctx_none(s,persistent))return false;
    for(const char* owner:{"fp_cards","sh_cards"})for(auto card:child_edges(s,obs_attr(s,persistent,owner))){
        const auto type=obs_attr(s,card,"card_type");
        if(enum_is(s,type,"game_state.PersistentCardType","POWER_TOO_GREAT")&&(key=="lorien"||key=="rivendell"||key=="grey havens"||key=="the grey havens"))return true;
        if(enum_is(s,type,"game_state.PersistentCardType","TOM_BOMBADIL")&&(key=="the old forest"||key=="old forest"||key=="the shire"||key=="shire"||key=="buckland"))return true;
    }return false;
}
std::unique_ptr<Snapshot> native_army_attack_for_die(const Snapshot& s,const Snapshot& t,int side,int64_t index,bool require_leader){
    require(side==0||side==1,"Attack side must be FP or SH");
    am_Builder b;std::vector<uint64_t> rows;const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));
    std::unordered_map<std::string,std::pair<uint64_t,uint64_t>> targets;
    for(size_t i=0;i<armies.size();i+=2)targets[sanitize_name(text(s,armies[i]))]={armies[i],armies[i+1]};
    const auto adj=obs_attr(s,obs_attr(s,s.root,"board"),"_adj");
    for(size_t i=0;i<armies.size();i+=2){
        const auto army=armies[i+1];bool own=false,enemy=false;
        for(auto unit:child_edges(s,obs_attr(s,army,"units")))(ctx_unit_side(s,unit)==side?own:enemy)=true;
        if(!own)continue;const auto formation=at_formation(s,army,side);if(formation.units.empty()||(require_leader&&!formation.character))continue;
        const auto original=text(s,armies[i]);const auto from=sanitize_name(original);
        if(enemy&&(side!=1||!at_shadow_blocked(s,from)))rows.push_back(b.row(index,"army_attack",{b.str("from"),b.str(original),b.str("to"),b.str(original),b.str("siege"),rr_scalar(b.out,WOTR_BOOL,1),b.str("side"),b.str(side==0?"fp":"sh")}));
        if(contains_string(s,obs_attr(s,s.root,"besieged_strongholds",false),from)&&at_controller(s,t,from)==side)continue;
        const auto adjacent=ctx_get(s,adj,from);if(ctx_none(s,adjacent))continue;
        std::vector<std::string> neighbors;for(auto node:child_edges(s,adjacent))neighbors.emplace_back(text(s,node));std::sort(neighbors.begin(),neighbors.end());
        for(const auto& destination:neighbors){
            const auto found=targets.find(destination);if(found==targets.end()||(side==1&&at_shadow_blocked(s,destination)))continue;
            bool target_own=false,target_enemy=false;for(auto unit:child_edges(s,obs_attr(s,found->second.second,"units")))(ctx_unit_side(s,unit)==side?target_own:target_enemy)=true;
            if(!target_enemy||(target_own&&at_controller(s,t,destination)!=side))continue;
            rows.push_back(b.row(index,"army_attack",{b.str("from"),b.str(original),b.str("to"),b.str(text(s,found->second.first)),b.str("side"),b.str(side==0?"fp":"sh")}));
        }
    }
    b.out.root=b.container(WOTR_LIST,rows);return std::make_unique<Snapshot>(std::move(b.out));
}
