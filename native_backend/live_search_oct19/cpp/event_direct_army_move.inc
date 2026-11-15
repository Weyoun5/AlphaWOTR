// Source _direct_army_move: shared by Through Day and Corsairs.
// Source game_logic.army_stack_cap (E1-05): 5 only for the GARRISON side (combat.garrison_side_for)
// of a besieged entry that still holds the besiegers; a besieger or a stale one-sided flag gets 10.
int cdm_cap(const Snapshot& s,const Snapshot& t,std::string_view region,int side){
    if(!contains_string(s,obs_attr(s,s.root,"besieged_strongholds",false),region))return 10;
    const auto army=mu_army(s,region);if(ctx_none(s,army)||!truth(s,obs_attr(s,army,"units")))return 10;
    if(crl_garrison_side(s,t,region)!=side)return 10;
    for(auto unit:child_edges(s,obs_attr(s,army,"units")))if(ctx_unit_side(s,unit)!=side)return 5;
    return 10;
}
bool cdm_enter(const Snapshot& s,const Snapshot& t,uint64_t u,std::string_view to){
    const auto nation=ctx_get(t,field(t,t.root,"region_nations"),to);const auto own=ctx_nation(s,obs_attr(s,u,"nation"));return ctx_none(t,nation)||political_view(s,own,true).box>=3||own==ctx_nation(t,nation);
}
bool cdm_fp_guard(const Snapshot& s,const Snapshot& t,uint64_t army,const std::string& to,const std::string& via){
    for(auto l:child_edges(s,obs_attr(s,army,"leaders")))if(am_plain(s,l)&&am_leader_side(s,l)==0&&(!cdm_enter(s,t,l,to)||(!via.empty()&&!cdm_enter(s,t,l,via))))return true;return false;
}
// Source event_effects._direct_move_pick: the figures a direct card move of `src` carries - ONE
// body for cdm_move and the v53 Through a Day and a Night roster (encoding_v53.inc). regular /
// elite < 0 = every eligible unit of that kind. R4-TDAAN-SPLIT-UNIT-IDENTITY (D5 E3-02): a
// non-null unit_set NAMES the split's units (army_card_cargo freeze form): exactly those eligible
// units move (first matches in army order, am_pick); a set the pool cannot supply or one
// disagreeing with the counts is refused (Python raises ValueError).
struct cdm_Pick{std::vector<uint64_t> moving,staying,leaders,staying_leaders,eligible;};
cdm_Pick cdm_pick(const Snapshot& s,const Snapshot& t,uint64_t src,const std::string& to,int side,bool border_rule,const std::string& via,int64_t regular,int64_t elite,const am_FigureSet* unit_set){
    cdm_Pick p;
    for(auto u:child_edges(s,obs_attr(s,src,"units"))){const bool allowed=ctx_unit_side(s,u)==side&&(!border_rule||(cdm_enter(s,t,u,to)&&(via.empty()||cdm_enter(s,t,u,via))));(allowed?p.eligible:p.staying).push_back(u);}
    if(unit_set){
        require(regular>=0&&elite>=0,"unit_set needs n_regular and n_elite");
        p.moving=am_pick(s,p.eligible,*unit_set,true);int64_t pr=0,pe=0;
        for(auto u:p.moving)(enum_is(s,obs_attr(s,u,"kind"),"game_state.UnitKind","ELITE")?pe:pr)++;
        require(pr==regular&&pe==elite,"unit_set disagrees with the unit counts");
        const std::set<uint64_t> chosen(p.moving.begin(),p.moving.end());for(auto u:p.eligible)if(!chosen.count(u))p.staying.push_back(u);
    }else for(auto u:p.eligible){const bool e=enum_is(s,obs_attr(s,u,"kind"),"game_state.UnitKind","ELITE");auto& left=e?elite:regular;if(left<0||left>0){p.moving.push_back(u);if(left>0)--left;}else p.staying.push_back(u);}
    for(auto l:child_edges(s,obs_attr(s,src,"leaders")))
        (am_saruman(s,l)||am_leader_side(s,l)!=side||(border_rule&&am_plain(s,l)&&(!cdm_enter(s,t,l,to)||(!via.empty()&&!cdm_enter(s,t,l,via))))?p.staying_leaders:p.leaders).push_back(l);
    return p;
}
int64_t cdm_move_core(Snapshot& s,const Snapshot& t,const std::string& from,const std::string& to,int side,bool border_rule,const std::string& via,int64_t regular,int64_t elite,bool allow_overstack,uint64_t cargo,bool defer_overstack,const am_FigureSet* unit_set){
    if(side==1&&at_shadow_blocked(s,to))return 0;
    const auto armies=obs_attr(s,s.root,"armies"),src=hr_army(s,from);
    if(ctx_none(s,src))return 0;
    require(!unit_set||ctx_none(s,cargo),"A named unit_set cannot mix with Shadow cargo");
    auto picked=cdm_pick(s,t,src,to,side,border_rule,via,regular,elite,unit_set);
    auto& moving=picked.moving;auto& staying=picked.staying;auto& leaders=picked.leaders;auto& staying_leaders=picked.staying_leaders;const auto& eligible=picked.eligible;
    if(!ctx_none(s,cargo)){
        require(side==1&&regular<0&&elite<0,"Exact cargo cannot mix with aggregate splits");const auto exact=cag_selected(s,from,cargo);const std::set<uint64_t> permitted(eligible.begin(),eligible.end());for(auto u:exact.first)require(permitted.count(u),"Cargo cannot enter destination");moving=exact.first;leaders=exact.second;const std::set<uint64_t> chosen(moving.begin(),moving.end()),chosen_leaders(leaders.begin(),leaders.end());staying.clear();staying_leaders.clear();for(auto u:child_edges(s,obs_attr(s,src,"units")))if(!chosen.count(u))staying.push_back(u);for(auto l:child_edges(s,obs_attr(s,src,"leaders")))if(!chosen_leaders.count(l))staying_leaders.push_back(l);
    }
    if(moving.empty())return 0;
    bool needs_guard=false,has_guard=false;for(auto l:staying_leaders)needs_guard|=am_plain(s,l)&&am_leader_side(s,l)==side;for(auto u:staying)has_guard|=ctx_unit_side(s,u)==side;require(!needs_guard||has_guard,"Mandatory plain leader needs a unit left behind");
    // Source lookup prefers an exact key; destination lookup scans aliases.
    const std::string source_key=ctx_get(s,armies,from)==src?from:am_key(s,from);
    auto dst=mu_army(s,to);int64_t own=0;
    if(!ctx_none(s,dst))for(auto u:child_edges(s,obs_attr(s,dst,"units")))own+=ctx_unit_side(s,u)==side;
    const int64_t room=cdm_cap(s,t,to,side)-own;
    if(allow_overstack){if(!defer_overstack&&static_cast<int64_t>(moving.size())>room){auto tracking=obs_attr(s,s.root,"_overstack_regions",false);if(ctx_none(s,tracking)){tracking=cp_container(s,WOTR_SET,{});rr_set_attr(s,s.root,"_overstack_regions",tracking);}me_set_member(s,tracking,to,true);}}
    else if(room<=0)return 0;
    if(!allow_overstack&&static_cast<int64_t>(moving.size())>room){
        const auto overflow=static_cast<size_t>(moving.size()-room);std::vector<uint64_t> keep;
        for(const char* kind:{"REGULAR","ELITE"})for(auto u:moving)
            if(keep.size()<overflow&&enum_is(s,obs_attr(s,u,"kind"),"game_state.UnitKind",kind))keep.push_back(u);
        const std::set<uint64_t> identities(keep.begin(),keep.end());std::vector<uint64_t> accepted;
        for(auto u:moving)(identities.count(u)?staying:accepted).push_back(u);
        moving=std::move(accepted);
    }
    const auto sieges=obs_attr(s,s.root,"besieged_strongholds");
    const int old_garrison=contains_string(s,sieges,from)?c9_garrison(s,t,from):-1;
    am_assign_list(s,src,"units",staying);am_assign_list(s,src,"leaders",staying_leaders);
    if(staying.empty()&&staying_leaders.empty())am_erase(s,source_key);
    if(ctx_none(s,dst)){
        const auto board=obs_attr(s,s.root,"board"),region=ctx_get(s,obs_attr(s,board,"_regions"),to);
        require(region!=missing,"Direct movement destination absent from board");dst=am_new_army(s,region);am_insert(s,to,dst);
    }
    am_extend(s,dst,"units",moving);am_extend(s,dst,"leaders",leaders);
    if(old_garrison==side){
        const auto remainder=hr_army(s,from);
        if(!evb_units(s,remainder,side)){
            me_set_member(s,sieges,from,false);
            if(evb_units(s,remainder,1-side))movement_capture_effects(s,t,from,1-side);
        }
    }
    movement_entry_effects(s,t,to,side,leaders);am_orphans(s);
    if(!evb_units(s,dst,1-side))movement_capture_effects(s,t,to,side);
    return static_cast<int64_t>(moving.size());
}
// The pre-R4 entry point (signature forward-declared in event_paths_woses.inc): no named unit set.
int64_t cdm_move(Snapshot& s,const Snapshot& t,const std::string& from,const std::string& to,int side,bool border_rule,const std::string& via,int64_t regular,int64_t elite,bool allow_overstack,uint64_t cargo=missing,bool defer_overstack=false){
    return cdm_move_core(s,t,from,to,side,border_rule,via,regular,elite,allow_overstack,cargo,defer_overstack,nullptr);
}
