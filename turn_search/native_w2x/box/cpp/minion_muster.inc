// Shadow minion entry: original can_muster_* predicates and muster_minion.
// Global ShadowMinion marker and newly constructed board Minion stay distinct.
const std::array<const char*,3> mi_labels={"Witch King","Saruman","Mouth of Sauron"};
const std::array<const char*,3> mi_attributes={"witch_king","saruman","mouth_of_sauron"};
int mi_index(std::string_view label){
    std::string value(label);for(auto& c:value)if(c>='A' && c<='Z')c=c-'A'+'a';
    if(value=="witch king")return 0;if(value=="saruman")return 1;if(value=="mouth of sauron" || value=="mouth")return 2;
    throw std::invalid_argument("unknown muster minion name");
}
std::vector<std::string> mi_regions(const Snapshot& s,const Snapshot& t,int minion){
    require(minion>=0 && minion<3,"invalid native minion index");
    const auto marker=obs_attr(s,s.root,mi_attributes[minion]);
    require(!ctx_none(s,marker),"minion muster requires its registered global marker");
    if(truth(s,obs_attr(s,marker,"in_play")) || truth(s,obs_attr(s,marker,"was_ever_in_play")))return {};
    std::vector<std::string> regions;
    if(minion==0){
        if(political_view(s,"SAURON",true).box<3)return {};
        bool fp_war=false;for(const auto nation:{"ELVES","DWARVES","GONDOR","THE_NORTH","ROHAN"})fp_war|=political_view(s,nation,true).box>=3;
        if(!fp_war)return {};
        std::set<std::string> seen;const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));
        for(size_t i=0;i<armies.size();i+=2){
            bool sauron=false;for(auto unit:child_edges(s,obs_attr(s,armies[i+1],"units")))sauron|=enum_is(s,obs_attr(s,unit,"nation"),"game_state.Nation","SAURON");
            if(sauron){const std::string key(text(s,armies[i]));if(seen.insert(sanitize_name(key)).second)regions.push_back(key);}
        }
    }else if(minion==1){
        if(political_view(s,"ISENGARD",true).box<3)return {};
        if(contains_string(s,ctx_side_value(s,obs_attr(s,s.root,"enemy_controlled"),0),"orthanc"))return {};
        regions.push_back("orthanc");
    }else{
        bool fp_war=true;for(const auto nation:{"GONDOR","ROHAN","ELVES","DWARVES","THE_NORTH"})fp_war&=political_view(s,nation,true).box>=3;
        if(!fp_war && !truth(s,obs_attr(s,obs_attr(s,s.root,"fellowship"),"mordor_entered")))return {};
        const auto fp=ctx_side_value(s,obs_attr(s,s.root,"enemy_controlled"),0);
        // The shared ordered table contains all Shadow Strongholds, but the
        // Mouth's entry card permits only those belonging to Sauron.
        const auto nations=field(t,t.root,"region_nations");
        for(auto region:child_edges(t,field(t,t.root,"shadow_strongholds"))){
            const std::string key(text(t,region));
            if(!enum_is(t,ctx_get(t,nations,key),"game_state.Nation","SAURON"))continue;
            if(!contains_string(s,fp,key))regions.push_back(key);
        }
    }
    return regions;
}
std::unique_ptr<Snapshot> native_minion_muster_actions(const Snapshot& s,const Snapshot& t,int32_t side){
    require(side==0 || side==1,"minion muster menu side must be FP or SH");quiet_stable_action_gate(s,t);ring_waiver(s,side);
    am_Builder b;std::vector<uint64_t> rows;
    if(side==1){
        std::array<std::vector<std::string>,3> candidates;for(int k=0;k<3;++k)candidates[k]=mi_regions(s,t,k);
        std::array<bool,7> seen{};const auto dice=child_edges(s,pa_pool(s,side));
        for(size_t i=0;i<dice.size();++i){
            const auto die=dice[i];require(object_type(s,die)=="game_state.ActionDie","minion menu requires ActionDie");
            const auto current=obs_attr(s,die,"current");if(truth(s,obs_attr(s,die,"used")) || ctx_none(s,current))continue;
            const int face=rr_face_index(enum_member(s,current,"game_state.ActionDieFace"));if(seen[face])continue;seen[face]=true;
            if(face!=2 && face!=4 && face!=6)continue;
            for(int k=0;k<3;++k)for(const auto& region:candidates[k])rows.push_back(b.row(i,"muster_minion",{
                b.str("minion"),b.str(mi_labels[k]),b.str("region"),b.str(region)}));
        }
    }
    b.out.root=b.container(WOTR_LIST,rows);return std::make_unique<Snapshot>(b.out);
}
std::unique_ptr<Snapshot> apply_minion_muster_action(const Snapshot& s,const Snapshot& t,int32_t side,int64_t index,
                                                   std::string_view name,std::string_view raw_region){
    require(side==1,"minion muster belongs to Shadow");const int kind=mi_index(name);const auto region=sanitize_name(raw_region);
    quiet_stable_action_gate(s,t);ring_waiver(s,0);ring_waiver(s,1);
    const auto retained=obs_attr(s,s.root,"_ring_free_side",false);
    require(ctx_none(s,retained) || ctx_side(s,retained)==1,"minion muster belongs to retained Ring actor");
    const auto regions=mi_regions(s,t,kind);bool legal=false;for(const auto& candidate:regions)legal|=sanitize_name(candidate)==region;
    require(legal,"minion is unavailable or muster region is illegal");
    const auto dice=child_edges(s,pa_pool(s,1));require(index>=0 && static_cast<uint64_t>(index)<dice.size(),"minion muster die index out of range");
    const auto die=dice[index];require(object_type(s,die)=="game_state.ActionDie","minion muster payment requires ActionDie");
    require(ctx_side(s,obs_attr(s,die,"side"))==1 && !truth(s,obs_attr(s,die,"used")),"minion muster payment used or belongs to other side");
    const auto face=obs_attr(s,die,"current");
    require(enum_is(s,face,"game_state.ActionDieFace","MUSTER") || enum_is(s,face,"game_state.ActionDieFace","MUSTER_ARMY"),
            "Shadow minion muster requires named Muster or Muster/Army die");
    auto child=action_clone(s);auto& out=*child;
    rr_set_attr(out,out.root,"_casualty_operations",rr_scalar(out,WOTR_LIST));rr_set_attr(out,out.root,"_ring_free_side",rr_scalar(out,WOTR_NONE));
    rr_set_attr(out,child_edges(out,pa_pool(out,1))[index],"used",rr_scalar(out,WOTR_BOOL,1));
    const auto marker=obs_attr(out,out.root,mi_attributes[kind]);
    rr_set_attr(out,marker,"in_play",rr_scalar(out,WOTR_BOOL,1));rr_set_attr(out,marker,"was_ever_in_play",rr_scalar(out,WOTR_BOOL,1));
    const auto nation=mu_enum(out,"game_state.Nation","SAURON"),marker_name=obs_attr(out,marker,"name");
    const auto figure=mu_object(out,"game_logic.Minion",{{"nation",nation},{"name",marker_name}});
    auto army=mu_army(out,region);
    if(army==missing){
        const auto entry=ctx_get(out,obs_attr(out,obs_attr(out,out.root,"board"),"_regions"),region);
        require(entry!=missing && object_type(out,entry)=="board.Region" && text_is(out,obs_attr(out,entry,"key"),region),"minion muster requires canonical Board region");
        army=am_new_army(out,entry);am_insert(out,region,army);
    }
    am_extend(out,army,"leaders",{figure});
    if(kind==0)for(const auto fp:{"ELVES","DWARVES","GONDOR","THE_NORTH","ROHAN"})wt_activate_if_inactive(out,fp,"witch_king_entry");
    tc_open_hand_limit_discard(out);rr_set_attr(out,out.root,"_battle_resolved_attacker",rr_scalar(out,WOTR_NONE));
    return child;
}
