// Requirement-only predicates from pinned event_effects.py. These do not
// imply execution support or prune legal zero-effect plays by supply/movement.
bool evt_companion_type(const Snapshot& s,uint64_t army){
    for(auto figure:child_edges(s,obs_attr(s,army,"leaders")))
        if(object_type(s,figure)=="game_state.Companion")return true;
    return false;
}
bool evt_spirit(const Snapshot& s){
    const auto entries=child_edges(s,obs_attr(s,s.root,"armies"));
    for(size_t i=1;i<entries.size();i+=2){
        std::set<std::string_view> nations;
        for(auto unit:child_edges(s,obs_attr(s,entries[i],"units"))){
            const auto nation=enum_member(s,obs_attr(s,unit,"nation"),"game_state.Nation");
            if(nation=="SAURON"||nation=="ISENGARD"||nation=="SOUTHRON_EASTERLINGS")nations.insert(nation);
        }
        if(nations.size()>=2)return true;
    }return false;
}
bool evt_through_day(const Snapshot& s){
    const auto entries=child_edges(s,obs_attr(s,s.root,"armies"));
    for(size_t i=1;i<entries.size();i+=2)
        if(evb_units(s,entries[i],0)&&evt_companion_type(s,entries[i]))return true;
    return false;
}
bool evt_stormcrow_region(const Snapshot& s,const Snapshot& t,std::string_view key){
    const auto meta=ctx_get(t,field(t,t.root,"event_region_metadata"),key);
    if(ctx_none(t,meta))return false;const auto name=ctx_get(t,meta,"nation");
    if(ctx_none(t,name))return false;
    const auto nation=text(t,name);const char* political=nullptr;
    if(nation=="gondor")political="GONDOR";else if(nation=="rohan")political="ROHAN";
    else if(nation=="elves")political="ELVES";else if(nation=="dwarves")political="DWARVES";
    else if(nation=="north"||nation=="the north")political="THE_NORTH";
    return political&&political_view(s,political,true).box<3;
}
bool evt_stormcrow(const Snapshot& s,const Snapshot& t){
    if(evt_stormcrow_region(s,t,glb_fellowship_key(s,obs_attr(s,s.root,"fellowship"))))return true;
    const auto entries=child_edges(s,obs_attr(s,s.root,"armies"));
    for(size_t i=0;i<entries.size();i+=2)
        if(evt_companion_type(s,entries[i+1])&&evt_stormcrow_region(s,t,sanitize_name(text(s,entries[i]))))return true;
    return false;
}
std::vector<std::pair<std::string,std::string>> evt_paths_options(const Snapshot& s,const Snapshot& t){
    const auto sieges=obs_attr(s,s.root,"besieged_strongholds"),controls=obs_attr(s,s.root,"enemy_controlled");
    auto shadow_at=[&](const char* key){return evb_units(s,hr_army(s,key),1);};
    std::vector<std::string> destinations;
    if(!contains_string(s,sieges,"minas tirith")&&!contains_string(s,ctx_side_value(s,controls,1),"minas tirith")&&!shadow_at("minas tirith"))destinations.push_back("minas tirith");
    else for(const char* key:{"druadan forest","osgiliath","lossarnach"})
        if(ctx_get(t,field(t,t.root,"board_region_parent_names"),key)!=missing&&!contains_string(s,sieges,key)&&!shadow_at(key))destinations.push_back(key);
    std::vector<std::pair<std::string,std::string>> options;if(destinations.empty())return options;
    const auto entries=child_edges(s,obs_attr(s,s.root,"armies"));
    for(size_t i=0;i<entries.size();i+=2){
        const auto key=sanitize_name(text(s,entries[i]));
        if(text_is(t,ctx_get(t,field(t,t.root,"board_region_parent_names"),key),"rohan")&&evb_units(s,entries[i+1],0))
            for(const auto& destination:destinations)if(destination!=key)options.emplace_back(key,destination);
    }return options;
}
bool evt_paths(const Snapshot& s,const Snapshot& t){return political_view(s,"ROHAN",true).box>=3&&!evt_paths_options(s,t).empty();}
bool evt_horde_free(const Snapshot& s,const char* key){
    // Exactly the four printed S&E regions. They are never FP national cities
    // or strongholds, so the source's subsequent FP-native restriction is false.
    const auto controls=obs_attr(s,s.root,"enemy_controlled"),sieges=obs_attr(s,s.root,"besieged_strongholds");
    if(contains_string(s,ctx_side_value(s,controls,0),key)&&!contains_string(s,sieges,key))return false;
    const auto entries=child_edges(s,obs_attr(s,s.root,"armies"));
    for(size_t i=0;i<entries.size();i+=2){
        if(sanitize_name(text(s,entries[i]))!=key)continue;
        if(evb_units(s,entries[i+1],0))return false;
    }return true;
}
bool evt_horde(const Snapshot& s){
    for(const char* key:{"east rhun","south rhun","khand","far harad"})if(evt_horde_free(s,key))return true;
    return false;
}
