// Saruman's two explicit ordinary muster alternatives, ai.py:6363--6526.
// Included after muster_menu.inc. Human ordinary recruits remain independent;
// the fixed Elite projection must not remove these differently flagged modes.
bool sm_saruman_present(const Snapshot& s){
    const auto entries=child_edges(s,obs_attr(s,s.root,"armies"));
    for(size_t i=1;i<entries.size();i+=2){
        // Preserve source's figure-presence test, not the roster flag.
        const auto army=entries[i];
        for(const auto leader:child_edges(s,obs_attr(s,army,"leaders"))){
            if(s.nodes[leader].tag!=WOTR_OBJECT || object_type(s,leader)!="game_logic.Minion")continue;
            const auto name=obs_attr(s,leader,"name",false);if(ctx_none(s,name))continue;
            std::string lower(text(s,name));
            for(auto& c:lower)if(c>='A' && c<='Z')c=static_cast<char>(c-'A'+'a');
            if(lower.find("saruman")!=std::string::npos)return true;
        }
    }
    return false;
}
bool sm_settlements_eligible(const Snapshot& s,const Snapshot& tables){
    if(!sm_saruman_present(s))return false;
    for(const auto region:{"orthanc","north dunland","south dunland"})
        if(!mm_legal(s,tables,1,"ISENGARD",region,"regular"))return false;
    return mu_available(s,tables,"ISENGARD","regular")>=3;
}
bool sm_swap_eligible(const Snapshot& s,const Snapshot& tables){
    if(!sm_saruman_present(s) || political_view(s,"ISENGARD",false).box<3)return false;
    if(contains_string(s,obs_attr(s,s.root,"besieged_strongholds",false),"orthanc"))return false;
    if(contains_string(s,ctx_side_value(s,obs_attr(s,s.root,"enemy_controlled"),0),"orthanc"))return false;
    const auto army=mu_army(s,"orthanc");if(army==missing)return false;
    size_t regulars=0;
    for(const auto unit:child_edges(s,obs_attr(s,army,"units")))
        if(enum_is(s,obs_attr(s,unit,"nation"),"game_state.Nation","ISENGARD") &&
           enum_is(s,obs_attr(s,unit,"kind"),"game_state.UnitKind","REGULAR"))++regulars;
    return regulars>=2 && mu_available(s,tables,"ISENGARD","elite")>=2;
}
std::unique_ptr<Snapshot> saruman_muster_actions(const Snapshot& s,const Snapshot& tables,int32_t side){
    require(side==0 || side==1,"Saruman menu side must be FP or SH");quiet_stable_action_gate(s,tables,true);
    am_Builder b;std::vector<uint64_t> rows;
    const auto pending=obs_attr(s,s.root,"pending_muster_die"),owner=obs_attr(s,s.root,"pending_muster_side",false);
    const bool pending_menu=!ctx_none(s,pending) && (ctx_none(s,owner) || ctx_side(s,owner)==side);
    if(!pending_menu){
        ring_waiver(s,side);
        if(side==1){
            const bool settlements=sm_settlements_eligible(s,tables),swap=sm_swap_eligible(s,tables);
            const auto pool=pa_pool(s,side);const auto& dice=s.nodes[pool];std::array<bool,7> seen{};
            for(uint64_t i=0;i<dice.count;++i){
                const auto die=s.edges[dice.offset+i];if(truth(s,obs_attr(s,die,"used")))continue;
                const auto current=obs_attr(s,die,"current");if(s.nodes[current].tag==WOTR_NONE)continue;
                const int face=rr_face_index(enum_member(s,current,"game_state.ActionDieFace"));
                if(seen[face])continue;seen[face]=true;if(face!=2 && face!=4 && face!=6)continue;
                if(settlements){
                    const auto places=b.container(WOTR_LIST,{b.str("orthanc"),b.str("north dunland"),b.str("south dunland")});
                    rows.push_back(b.row(static_cast<int64_t>(i),"muster_recruit",{
                        b.str("nation"),b.nation("ISENGARD"),b.str("region"),b.str("orthanc"),
                        b.str("type"),b.str("regular"),b.str("saruman"),b.str("settlements"),b.str("placements"),places}));
                }
                if(swap)rows.push_back(b.row(static_cast<int64_t>(i),"muster_elite",{
                    b.str("nation"),b.nation("ISENGARD"),b.str("region"),b.str("orthanc"),b.str("saruman"),b.str("swap2")}));
            }
        }
    }
    b.out.root=b.container(WOTR_LIST,rows);return std::make_unique<Snapshot>(std::move(b.out));
}
std::unique_ptr<Snapshot> apply_saruman_muster_action(const Snapshot& s,const Snapshot& tables,
                                                   int32_t side,int64_t index,std::string_view mode){
    require(side==1,"Saruman recruitment belongs to Shadow");
    require(mode=="settlements" || mode=="swap2","unknown Saruman recruitment mode");
    quiet_stable_action_gate(s,tables);ring_waiver(s,0);ring_waiver(s,1);
    const auto retained=ctx_side(s,obs_attr(s,s.root,"_ring_free_side",false));
    require(retained<0 || retained==side,"Saruman recruitment belongs to the retained ring actor");
    pa_payment_die(s,side,index);
    require(mode=="settlements"?sm_settlements_eligible(s,tables):sm_swap_eligible(s,tables),
            "Saruman recruitment is not currently legal");
    auto child=action_clone(s);auto& out=*child;
    rr_set_attr(out,out.root,"_casualty_operations",rr_scalar(out,WOTR_LIST));
    rr_set_attr(out,out.root,"_ring_free_side",rr_scalar(out,WOTR_NONE));
    if(mode=="settlements"){
        for(const auto region:{"orthanc","north dunland","south dunland"})mu_place(out,"ISENGARD",region,"regular");
    }else{
        const auto units=obs_attr(out,mu_army(out,"orthanc"),"units");
        std::vector<uint64_t> kept;size_t removed=0;
        for(const auto unit:child_edges(out,units)){
            if(removed<2 && enum_is(out,obs_attr(out,unit,"nation"),"game_state.Nation","ISENGARD") &&
               enum_is(out,obs_attr(out,unit,"kind"),"game_state.UnitKind","REGULAR")){++removed;continue;}
            kept.push_back(unit);
        }
        require(removed==2,"Saruman swap lost its two regular figures");set_edges(out,units,kept);
        mu_place(out,"ISENGARD","orthanc","elite");mu_place(out,"ISENGARD","orthanc","elite");
    }
    rr_set_attr(out,pa_payment_die(out,side,index),"used",rr_scalar(out,WOTR_BOOL,1));
    rr_set_attr(out,out.root,"_battle_resolved_attacker",rr_scalar(out,WOTR_NONE));
    quiet_stable_action_gate(out,tables);return child;
}
