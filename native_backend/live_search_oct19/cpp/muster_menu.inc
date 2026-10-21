// Raw ordinary muster descriptors. Included after action_menu and muster_action.
// Legality stays with mu_validate; only its declared legal-placement refusals
// omit a candidate. Malformed-state/type errors remain loud.
bool mm_legal(const Snapshot& s,const Snapshot& tables,int32_t side,
              std::string_view nation,std::string_view region,std::string_view kind){
    try{mu_validate(s,tables,side,nation,region,kind);return true;}
    catch(const std::invalid_argument& error){
        const std::string_view message(error.what());
        for(const auto reason:{"muster nation must be At War","muster needs a settlement","muster cannot recruit into besieged stronghold",
            "muster needs the recruiting nation's settlement","muster settlement is enemy controlled",
            "muster stacking limit reached","a lone Free Peoples leader cannot be recruited",
            "Shadow leaders recruit at strongholds","muster reinforcement pool is exhausted"})
            if(message==reason)return false;
        throw;
    }
}
void mm_append(am_Builder& b,std::vector<uint64_t>& rows,const Snapshot& s,const Snapshot& tables,
               int32_t side,int64_t index,bool elite,std::string_view exclude){
    const std::vector<std::string_view> nations=side==0
        ? std::vector<std::string_view>{"ELVES","DWARVES","GONDOR","THE_NORTH","ROHAN"}
        : std::vector<std::string_view>{"ISENGARD","SAURON","SOUTHRON_EASTERLINGS"};
    const auto regions=field(tables,tables.root,"board_regions"),region_nations=field(tables,tables.root,"region_nations");
    const auto& metadata=typed(tables,regions,WOTR_DICT,"muster menu requires region metadata");
    for(const auto nation:nations){
        if(political_view(s,nation,false).box<3)continue;
        for(uint64_t i=0;i<metadata.count;i+=2){
            const auto key=text(tables,tables.edges[metadata.offset+i]);
            const auto info=tables.edges[metadata.offset+i+1];
            if(!truth(tables,ctx_get(tables,info,"is_city")) && !truth(tables,ctx_get(tables,info,"is_stronghold")) &&
               !truth(tables,ctx_get(tables,info,"is_town")))continue;
            if(!enum_is(tables,ctx_get(tables,region_nations,key),"game_state.Nation",nation))continue;
            if(!exclude.empty() && sanitize_name(key)==exclude)continue;
            const std::vector<std::string_view> kinds=elite?std::vector<std::string_view>{"elite"}:std::vector<std::string_view>{"regular","leader"};
            for(const auto kind:kinds){
                if(!mm_legal(s,tables,side,nation,key,kind))continue;
                std::vector<uint64_t> params={b.str("nation"),b.nation(nation),b.str("region"),b.str(key)};
                if(!elite){params.push_back(b.str("type"));params.push_back(b.str(kind));}
                rows.push_back(b.row(index,elite?"muster_elite":"muster_recruit",params));
            }
        }
    }
}
std::unique_ptr<Snapshot> native_muster_actions(const Snapshot& s,const Snapshot& tables,int32_t side){
    require(side==0 || side==1,"muster menu side must be FP or SH");quiet_stable_action_gate(s,tables,true);
    am_Builder b;std::vector<uint64_t> rows;
    const auto pending=obs_attr(s,s.root,"pending_muster_die"),owner=obs_attr(s,s.root,"pending_muster_side",false);
    if(!ctx_none(s,pending) && (ctx_none(s,owner) || ctx_side(s,owner)==side)){
        const auto index=typed(s,pending,WOTR_INT,"pending muster menu index must be integer").integer;
        const auto previous=obs_attr(s,s.root,"pending_muster_region",false);
        const std::string exclude=truth(s,previous)?sanitize_name(text(s,previous)):std::string{};
        mm_append(b,rows,s,tables,side,index,false,exclude);
        rows.push_back(b.row(index,"pass",{}));
    }else{
        ring_waiver(s,side); // This branch follows raw ordinary get_legal_actions.
        const auto pool=pa_pool(s,side);const auto& dice=s.nodes[pool];std::array<bool,7> seen{};
        for(uint64_t i=0;i<dice.count;++i){
            const auto die=s.edges[dice.offset+i];require(object_type(s,die)=="game_state.ActionDie","muster menu requires ActionDie");
            if(truth(s,obs_attr(s,die,"used")))continue;
            const auto current=obs_attr(s,die,"current");if(s.nodes[current].tag==WOTR_NONE)continue;
            const int face=rr_face_index(enum_member(s,current,"game_state.ActionDieFace"));
            if(seen[face])continue;seen[face]=true;
            if(face!=2 && face!=4 && face!=6)continue;
            mm_append(b,rows,s,tables,side,static_cast<int64_t>(i),false,{});
            mm_append(b,rows,s,tables,side,static_cast<int64_t>(i),true,{});
        }
    }
    b.out.root=b.container(WOTR_LIST,rows);return std::make_unique<Snapshot>(std::move(b.out));
}
