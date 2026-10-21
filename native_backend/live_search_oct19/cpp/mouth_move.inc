// Mouth of Sauron solo movement: pinned ai.py _mouth_reachable and mouth_move.
// Reachability is read-only and includes sieges. Complete action execution
// retains the quiet-state gate until common siege upkeep is ported.
bool mo_figure(const Snapshot& s,uint64_t leader){
    if(object_type(s,leader)!="game_logic.Minion")return false;
    auto name=std::string(text(s,obs_attr(s,leader,"name")));
    for(auto& c:name)if(c>='A' && c<='Z')c=c-'A'+'a';
    return name.find("mouth")!=std::string::npos;
}
std::string mo_region(const Snapshot& s){
    const auto mouth=obs_attr(s,s.root,"mouth_of_sauron",false);
    if(ctx_none(s,mouth) || !truth(s,obs_attr(s,mouth,"in_play",false)))return {};
    const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));
    for(size_t i=0;i<armies.size();i+=2)for(auto leader:child_edges(s,obs_attr(s,armies[i+1],"leaders")))
        if(mo_figure(s,leader))return sanitize_name(text(s,armies[i]));
    return {};
}
int mo_stronghold_owner(const Snapshot& s,const Snapshot& t,std::string_view raw){
    const auto key=sanitize_name(raw);const auto meta=ctx_get(t,field(t,t.root,"board_regions"),key);
    if(meta==missing || !truth(t,ctx_get(t,meta,"is_stronghold")))return -1;
    const auto nation=ctx_get(t,field(t,t.root,"region_nations"),key);if(ctx_none(t,nation))return -1;
    const int native=fp_nation(ctx_nation(t,nation))?0:1;
    const auto enemy=ctx_side_value(s,obs_attr(s,s.root,"enemy_controlled"),1-native);
    return contains_string(s,enemy,key)?1-native:native;
}
std::set<std::string> mo_reachable(const Snapshot& s,const Snapshot& t,std::string_view start){
    const auto siege=obs_attr(s,s.root,"besieged_strongholds",false);std::set<std::string> besieged,shadow_besieged;
    if(truth(s,siege))for(auto value:child_edges(s,siege)){
        const std::string key(text(s,value));besieged.insert(key);if(mo_stronghold_owner(s,t,key)==1)shadow_besieged.insert(key);
    }
    if(shadow_besieged.count(std::string(start)))return {};
    std::set<std::string> result,frontier{std::string(start)},visited{std::string(start)};
    const auto adjacent=obs_attr(s,obs_attr(s,s.root,"board"),"_adj");
    for(int depth=0;depth<3;++depth){
        std::set<std::string> next;
        for(const auto& key:frontier){
            const auto neighbors=ctx_get(s,adjacent,key);if(neighbors==missing)continue;
            for(auto value:child_edges(s,neighbors)){
                const auto dest=sanitize_name(text(s,value));if(!visited.insert(dest).second)continue;
                if(shadow_besieged.count(dest))continue;
                if(mo_stronghold_owner(s,t,dest)==0){
                    if(besieged.count(dest))result.insert(dest);
                    continue;
                }
                result.insert(dest);next.insert(dest);
            }
        }
        frontier=std::move(next);
    }
    return result;
}
std::unique_ptr<Snapshot> native_mouth_move_actions(const Snapshot& s,const Snapshot& t,int32_t side){
    require(side==0 || side==1,"Mouth menu side must be FP or SH");
    // This is the ordinary die-loop subset, so higher-priority continuations
    // cannot be replaced with a Mouth list. Siege geometry itself is admitted.
    for(const auto key:{"pending_combat","pending_guide_change","pending_hunt","pending_card_option","pending_card_followup","pending_discard","pending_army_move_die","pending_muster_die"})
        require(ctx_none(s,obs_attr(s,s.root,key,false)),"Mouth ordinary menu rejects pending decision");
    for(const auto key:{"_declare_phase","_eye_allocation_phase","_reveal_place_due"})
        require(!truth(s,obs_attr(s,s.root,key,false)),"Mouth ordinary menu rejects phase decision");
    ring_waiver(s,side);am_Builder b;std::vector<uint64_t> rows;
    const auto source=side==1?mo_region(s):std::string{};
    if(!source.empty()){
        const auto destinations=mo_reachable(s,t,source);const auto dice=child_edges(s,pa_pool(s,side));std::array<bool,7> seen{};
        for(size_t i=0;i<dice.size();++i){
            const auto die=dice[i];require(object_type(s,die)=="game_state.ActionDie","Mouth menu requires ActionDie");
            const auto current=obs_attr(s,die,"current");if(truth(s,obs_attr(s,die,"used")) || ctx_none(s,current))continue;
            const int face=rr_face_index(enum_member(s,current,"game_state.ActionDieFace"));if(seen[face])continue;seen[face]=true;
            if(face!=0 && face!=6)continue;
            for(const auto& destination:destinations)rows.push_back(b.row(i,"mouth_move",{
                b.str("from"),b.str(source),b.str("to"),b.str(destination)}));
        }
    }
    b.out.root=b.container(WOTR_LIST,rows);return std::make_unique<Snapshot>(b.out);
}
std::unique_ptr<Snapshot> apply_mouth_move_action(const Snapshot& s,const Snapshot& t,int32_t side,int64_t index,
                                                std::string_view raw_from,std::string_view raw_to){
    require(side==1,"Mouth movement belongs to Shadow");quiet_stable_action_gate(s,t);ring_waiver(s,0);ring_waiver(s,1);
    const auto retained=obs_attr(s,s.root,"_ring_free_side",false);
    require(ctx_none(s,retained) || ctx_side(s,retained)==1,"Mouth movement belongs to retained Ring actor");
    const auto from=sanitize_name(raw_from),to=sanitize_name(raw_to);
    require(!from.empty() && mo_region(s)==from,"Mouth movement source has no in-play Mouth");
    require(mo_reachable(s,t,from).count(to)!=0,"Mouth destination is not legally reachable in three steps");
    const auto dice=child_edges(s,pa_pool(s,1));require(index>=0 && static_cast<uint64_t>(index)<dice.size(),"Mouth payment index out of range");
    const auto die=dice[index];require(object_type(s,die)=="game_state.ActionDie","Mouth payment requires ActionDie");
    require(ctx_side(s,obs_attr(s,die,"side"))==1 && !truth(s,obs_attr(s,die,"used")) &&
        enum_is(s,obs_attr(s,die,"current"),"game_state.ActionDieFace","CHARACTER"),"Mouth movement needs the named unused Character die");
    const auto regions=obs_attr(s,obs_attr(s,s.root,"board"),"_regions"),region=ctx_get(s,regions,to);
    require(region!=missing && object_type(s,region)=="board.Region" && text_is(s,obs_attr(s,region,"key"),to),"Mouth destination requires canonical Board Region");
    const auto metadata=ctx_get(t,field(t,t.root,"board_regions"),to);
    require(metadata!=missing,"Mouth destination has no pinned board metadata");
    const auto original=ctx_get(t,metadata,"orig");const std::string destination_key=original==missing?std::string(raw_to):std::string(text(t,original));
    require(sanitize_name(destination_key)==to,"Mouth destination original name differs from board key");
    auto child=action_clone(s);auto& out=*child;
    rr_set_attr(out,out.root,"_casualty_operations",rr_scalar(out,WOTR_LIST));rr_set_attr(out,out.root,"_ring_free_side",rr_scalar(out,WOTR_NONE));
    rr_set_attr(out,child_edges(out,pa_pool(out,1))[index],"used",rr_scalar(out,WOTR_BOOL,1));
    const auto source_key=am_key(out,from);const auto army=mu_army(out,from),leaders=obs_attr(out,army,"leaders");
    auto kept=child_edges(out,leaders);const auto found=std::find_if(kept.begin(),kept.end(),[&](uint64_t leader){return mo_figure(out,leader);});
    require(found!=kept.end(),"Mouth figure vanished during action clone");const auto mouth=*found;kept.erase(found);set_edges(out,leaders,kept);
    if(kept.empty() && !truth(out,obs_attr(out,army,"units")))am_erase(out,source_key);
    auto destination=mu_army(out,to);
    if(destination==missing){
        const auto copied_region=ctx_get(out,obs_attr(out,obs_attr(out,out.root,"board"),"_regions"),to);
        destination=am_new_army(out,copied_region);am_insert(out,destination_key,destination);
    }
    am_extend(out,destination,"leaders",{mouth});
    tc_open_hand_limit_discard(out);rr_set_attr(out,out.root,"_battle_resolved_attacker",rr_scalar(out,WOTR_NONE));
    return child;
}
