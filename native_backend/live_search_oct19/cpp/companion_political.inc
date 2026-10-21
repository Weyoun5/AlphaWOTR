// Legolas/Gimli/Boromir political ability in quiet states without table cards.
std::string_view cp_nation(std::string_view name){
    if(name=="Legolas")return "ELVES";if(name=="Gimli")return "DWARVES";if(name=="Boromir")return "GONDOR";return {};
}
std::vector<std::pair<std::string,std::string>> cp_available(const Snapshot& s,const Snapshot& t){
    std::vector<std::pair<std::string,std::string>> result;std::set<std::string> seen;
    const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));
    const auto controls=ctx_side_value(s,obs_attr(s,s.root,"enemy_controlled"),1);
    for(size_t i=0;i<armies.size();i+=2){
        const auto key=sanitize_name(text(s,armies[i]));const auto army=armies[i+1];
        for(auto leader:child_edges(s,obs_attr(s,army,"leaders"))){
            if(object_type(s,leader)!="game_state.Companion")continue;
            const auto name=text(s,obs_attr(s,leader,"name")),nation=cp_nation(name);
            if(nation.empty() || seen.count(std::string(nation)))continue;
            const auto meta=ctx_get(t,field(t,t.root,"board_regions"),key);
            if(!truth(t,ctx_get(t,meta,"is_city")) && !truth(t,ctx_get(t,meta,"is_stronghold")))continue;
            if(!enum_is(t,ctx_get(t,field(t,t.root,"region_nations"),key),"game_state.Nation",nation) || contains_string(s,controls,key))continue;
            const auto view=political_view(s,nation,false);if(!can_advance(s,view,nation,"companion_ability"))continue;
            seen.insert(std::string(nation));result.emplace_back(nation,name);
        }
    }
    return result;
}
std::unique_ptr<Snapshot> apply_companion_political_action(const Snapshot& s,const Snapshot& t,int32_t side,
                                                         int64_t index,std::string_view nation,std::string_view companion){
    require(side==0,"companion political ability belongs to Free Peoples");quiet_stable_action_gate(s,t,false,true);
    ring_waiver(s,0);ring_waiver(s,1);
    const auto retained=ctx_side(s,obs_attr(s,s.root,"_ring_free_side",false));
    require(retained<0 || retained==0,"companion political ability belongs to retained ring actor");
    require(cp_nation(companion)==nation && !nation.empty(),"invalid companion political nation/name");
    bool present=false;for(const auto& option:cp_available(s,t))present|=option.first==nation && option.second==companion;
    require(present,"companion political ability lacks eligible board figure or advancement");
    const auto pool=child_edges(s,pa_pool(s,0));require(index>=0 && static_cast<uint64_t>(index)<pool.size(),"companion political die index out of range");
    const auto die=pool[index];require(object_type(s,die)=="game_state.ActionDie","companion political payment requires ActionDie");
    require(ctx_side(s,obs_attr(s,die,"side"))==0 && !truth(s,obs_attr(s,die,"used")),"companion political payment has wrong owner or is used");
    const auto face=obs_attr(s,die,"current");require(!ctx_none(s,face) && rr_face_index(enum_member(s,face,"game_state.ActionDieFace"))!=5,"companion political payment requires a non-Eye action die");
    auto child=action_clone(s);auto& out=*child;
    rr_set_attr(out,out.root,"_casualty_operations",rr_scalar(out,WOTR_LIST));
    rr_set_attr(out,out.root,"_ring_free_side",rr_scalar(out,WOTR_NONE));
    const auto copied_die=child_edges(out,pa_pool(out,0))[index];rr_set_attr(out,copied_die,"used",rr_scalar(out,WOTR_BOOL,1));
    if(political_table_gate(out))advance(out,nation,"companion_ability");
    else{const auto view=political_view(out,nation,true);assign_box(out,view,view.box+1);}
    rr_set_attr(out,out.root,"_battle_resolved_attacker",rr_scalar(out,WOTR_NONE));return child;
}
std::unique_ptr<Snapshot> native_companion_political_actions(const Snapshot& s,const Snapshot& t,int32_t side){
    require(side==0 || side==1,"companion menu side must be FP or SH");quiet_stable_action_gate(s,t,false,true);ring_waiver(s,side);
    am_Builder b;std::vector<uint64_t> rows;
    if(side==0){
        const auto options=cp_available(s,t);const auto dice=child_edges(s,pa_pool(s,0));std::array<bool,7> seen{};
        for(size_t i=0;i<dice.size();++i){
            const auto die=dice[i];require(object_type(s,die)=="game_state.ActionDie","companion menu requires ActionDie");
            if(truth(s,obs_attr(s,die,"used")))continue;const auto current=obs_attr(s,die,"current");if(ctx_none(s,current))continue;
            const int face=rr_face_index(enum_member(s,current,"game_state.ActionDieFace"));if(seen[face])continue;seen[face]=true;
            // No Threats and Promises: ordinary muster advance already covers
            // every eligible companion nation, so raw source removes duplicate.
            if(!political_table_gate(s) && (face==2 || face==4 || face==6))continue;
            for(const auto& option:options)rows.push_back(b.row(i,"advance_political",{b.str("nation"),b.nation(option.first),b.str("companion_ability"),b.str(option.second)}));
        }
    }
    b.out.root=b.container(WOTR_LIST,rows);return std::make_unique<Snapshot>(std::move(b.out));
}
