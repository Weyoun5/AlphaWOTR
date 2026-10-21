// Board-only printed predicates. Execution support remains independently gated.
bool hr_nazgul(const Snapshot& s,uint64_t leader);
uint64_t hr_army(const Snapshot& s,std::string_view key);
bool evb_units(const Snapshot& s,uint64_t army,int side){
    if(ctx_none(s,army))return false;
    for(auto u:child_edges(s,obs_attr(s,army,"units")))if(ctx_unit_side(s,u)==side)return true;return false;
}
bool evb_companion(const Snapshot& s,uint64_t army){
    for(auto c:child_edges(s,obs_attr(s,army,"leaders"))){
        const auto name=obs_attr(s,c,"name",false);if(ctx_none(s,name))continue;
        for(const auto known:{"Gandalf the Grey","Gandalf the White","Strider","Aragorn","Aragorn - Heir to Isildur","Boromir","Legolas","Gimli","Meriadoc","Peregrin"})
            if(text(s,name).find(known)!=std::string_view::npos)return true;
    }return false;
}
bool evb_nazgul_army(const Snapshot& s,uint64_t army){
    if(!evb_units(s,army,1))return false;
    for(auto c:child_edges(s,obs_attr(s,army,"leaders")))if(hr_nazgul(s,c))return true;return false;
}
bool evb_witch_army(const Snapshot& s,uint64_t army){
    for(auto c:child_edges(s,obs_attr(s,army,"leaders"))){
        if(object_type(s,c)=="game_state.Nazgul"){
            if(truth(s,obs_attr(s,c,"is_witch_king",false)))return true;
        }else if(hr_nazgul(s,c))return true;
        // A named ordinary Nazgul also satisfies the source's name fallback.
        const auto name=obs_attr(s,c,"name",false);if(ctx_none(s,name))continue;
        std::string letters;for(auto ch:text(s,name)){
            if(ch>='A'&&ch<='Z')ch=ch-'A'+'a';if(ch>='a'&&ch<='z')letters+=ch;
        }if(letters.find("witchking")!=std::string::npos)return true;
    }return false;
}
int evb_garrison(const Snapshot& s,const Snapshot& t,std::string_view key){
    const auto controls=obs_attr(s,s.root,"enemy_controlled");
    if(contains_string(s,ctx_side_value(s,controls,1),key))return 1;
    if(contains_string(s,ctx_side_value(s,controls,0),key))return 0;
    return contains_string(t,field(t,t.root,"fp_region_keys"),key)?0:1;
}
bool evb_grond_target(const Snapshot& s,const Snapshot& t,std::string_view key,uint64_t army){
    if(!contains_string(s,obs_attr(s,s.root,"besieged_strongholds"),key)||evb_garrison(s,t,key)!=0)return false;
    const auto meta=ctx_get(t,field(t,t.root,"event_region_metadata"),key);
    return !ctx_none(t,meta)&&truth(t,ctx_get(t,meta,"is_stronghold"))
        &&text_is(t,ctx_get(t,meta,"owner"),"free")&&evb_units(s,army,1)&&evb_witch_army(s,army);
}
bool evb_shadow_controlled(const Snapshot& s,const Snapshot& t,std::string_view key){
    const auto controls=obs_attr(s,s.root,"enemy_controlled");
    if(contains_string(s,ctx_side_value(s,controls,1),key))return true;
    if(contains_string(s,ctx_side_value(s,controls,0),key))return false;
    // A besieger may appear first after a partial movement; the Stronghold
    // remains controlled by its garrison throughout the siege (RB31).
    if(contains_string(s,obs_attr(s,s.root,"besieged_strongholds"),key))return evb_garrison(s,t,key)==1;
    // Outside a siege, preserve the source's field-occupancy fallback.
    const auto army=mu_army(s,key);
    if(army!=missing){const auto units=child_edges(s,obs_attr(s,army,"units"));if(!units.empty())return ctx_unit_side(s,units[0])==1;}
    const auto meta=ctx_get(t,field(t,t.root,"event_region_metadata"),key);
    return !ctx_none(t,meta)&&text_is(t,ctx_get(t,meta,"owner"),"shadow");
}
bool evb_requirement(const Snapshot& s,const Snapshot& t,const std::string& req){
    auto has=[&](const char* term){return req.find(term)!=std::string::npos;};
    const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));
    const auto adj=field(t,t.root,"board_adj"),sieges=obs_attr(s,s.root,"besieged_strongholds");
    if(has("a free peoples army containing a companion is adjacent to")&&has("shadow army containing a nazg\xC3\xBBl")){
        bool found=false;
        for(size_t i=0;i+1<armies.size()&&!found;i+=2){
            const auto army=armies[i+1];if(!evb_units(s,army,0)||!evb_companion(s,army))continue;
            const auto key=sanitize_name(text(s,armies[i]));const auto neighbors=ctx_get(t,adj,key);
            if(!ctx_none(t,neighbors))for(auto n:child_edges(t,neighbors))if(evb_nazgul_army(s,hr_army(s,text(t,n)))){found=true;break;}
            if(!found&&evb_nazgul_army(s,army))found=true;
            for(size_t j=0;j+1<armies.size()&&!found;j+=2)
                if(j!=i&&sanitize_name(text(s,armies[j]))==key&&evb_nazgul_army(s,armies[j+1]))found=true;
        }if(!found)return false;
    }
    if(has("a shadow army containing a nazg\xC3\xBBl is adjacent to")&&has("free peoples army")){
        bool found=false;
        for(size_t i=0;i+1<armies.size()&&!found;i+=2){
            if(!evb_nazgul_army(s,armies[i+1]))continue;
            const auto key=sanitize_name(text(s,armies[i]));std::set<std::string> candidates{key};
            const auto neighbors=ctx_get(t,adj,key);if(!ctx_none(t,neighbors))for(auto n:child_edges(t,neighbors))candidates.insert(sanitize_name(text(t,n)));
            for(size_t j=0;j+1<armies.size();j+=2)if(candidates.count(sanitize_name(text(s,armies[j])))&&evb_units(s,armies[j+1],0)){found=true;break;}
        }if(!found)return false;
    }
    if(has("witch-king is in play and is with a shadow army besieging a free peoples stronghold")){
        bool found=false;for(size_t i=0;i+1<armies.size();i+=2){
            if(evb_grond_target(s,t,sanitize_name(text(s,armies[i])),armies[i+1])){found=true;break;}
        }if(!found)return false;
    }
    if(has("minas tirith is besieged by a shadow army")&&(!contains_string(s,sieges,"minas tirith")||evb_garrison(s,t,"minas tirith")!=0))return false;
    if(has("an army containing an isengard unit is besieging a stronghold")){
        bool found=false;for(auto region:child_edges(s,sieges)){
            const auto key=sanitize_name(text(s,region));if(evb_garrison(s,t,key)!=0)continue;
            const auto army=hr_army(s,key);if(ctx_none(s,army))continue;
            for(auto u:child_edges(s,obs_attr(s,army,"units")))if(enum_is(s,obs_attr(s,u,"nation"),"game_state.Nation","ISENGARD")){found=true;break;}
            if(found)break;
        }if(!found)return false;
    }
    if(has("there is a free peoples army in minas tirith")||has("there is a free peoples army in minis tirith")){
        if(!evb_units(s,mu_army(s,"minas tirith"),0))return false;
    }
    if(has("you control at least one elven stronghold")){
        bool found=false;for(const char* key:{"rivendell","lorien","grey havens","woodland realm"})if(evb_shadow_controlled(s,t,key)){found=true;break;}
        if(!found)return false;
    }
    return true;
}
