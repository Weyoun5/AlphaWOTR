// Printed figure predicates, copied from the pinned Python requirement checker.
// Keep its tracker/map distinctions and insertion-order first-match semantics.
bool evf_named(const Snapshot& s,uint64_t figure,const std::vector<std::string_view>& names){
    const auto name=obs_attr(s,figure,"name",false);if(ctx_none(s,name))return false;
    for(const auto wanted:names)if(text_is(s,name,wanted))return true;return false;
}
bool evf_fellowship(const Snapshot& s,const std::vector<std::string_view>& names){
    const auto fs=obs_attr(s,s.root,"fellowship");
    for(auto c:child_edges(s,obs_attr(s,fs,"companions")))if(evf_named(s,c,names))return true;return false;
}
uint64_t evf_first_region(const Snapshot& s,const std::vector<std::string_view>& names,bool units_required){
    const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));
    for(size_t i=0;i+1<armies.size();i+=2){
        const auto army=armies[i+1];
        if(units_required){
            bool friendly=false;for(auto unit:child_edges(s,obs_attr(s,army,"units")))friendly|=ctx_unit_side(s,unit)==0;
            if(!friendly)continue;
        }
        for(auto c:child_edges(s,obs_attr(s,army,"leaders")))if(evf_named(s,c,names))return armies[i];
    }return missing;
}
std::string evf_nation(const Snapshot& s,const Snapshot& t,uint64_t region){
    if(region==missing)return {};
    const auto meta=ctx_get(t,field(t,t.root,"event_region_metadata"),sanitize_name(text(s,region)));
    if(ctx_none(t,meta))return {};
    const auto nation=ctx_get(t,meta,"nation");return ctx_none(t,nation)?std::string{}:std::string(text(t,nation));
}
bool evf_tracker(const Snapshot& s,const char* tracker,const char* member){
    const auto marker=obs_attr(s,s.root,tracker,false);
    return !ctx_none(s,marker)&&truth(s,obs_attr(s,marker,member,false));
}
bool evf_witch_king(const Snapshot& s){
    const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));
    for(size_t i=1;i<armies.size();i+=2)for(auto c:child_edges(s,obs_attr(s,armies[i],"leaders"))){
        const auto name=obs_attr(s,c,"name",false);if(ctx_none(s,name))continue;
        std::string letters;for(auto ch:text(s,name)){
            if(ch>='A'&&ch<='Z')ch=ch-'A'+'a';if(ch>='a'&&ch<='z')letters+=ch;
        }if(letters.find("witchking")!=std::string::npos)return true;
    }return false;
}
bool evf_fangorn_companion(const Snapshot& s){
    const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));
    for(size_t i=0;i+1<armies.size();i+=2){
        if(sanitize_name(text(s,armies[i]))!="fangorn")continue;
        for(auto c:child_edges(s,obs_attr(s,armies[i+1],"leaders"))){
            const auto name=obs_attr(s,c,"name",false);if(ctx_none(s,name))continue;
            for(const auto known:{"Gandalf the Grey","Gandalf the White","Strider","Aragorn","Aragorn - Heir to Isildur","Boromir","Legolas","Gimli","Meriadoc","Peregrin"})
                if(text(s,name).find(known)!=std::string_view::npos)return true;
        }
    }return false;
}
bool evf_requirement(const Snapshot& s,const Snapshot& t,const std::string& req){
    auto has=[&](const char* term){return req.find(term)!=std::string::npos;};
    const std::vector<std::string_view> strider={"Strider","Aragorn","Aragorn - Heir to Isildur"};
    const std::vector<std::string_view> aragorn={"Aragorn","Aragorn - Heir to Isildur"};
    if(has("strider/aragorn is with a free peoples army in gondor or rohan")){
        const auto r=evf_first_region(s,strider,true);const auto n=evf_nation(s,t,r);
        if(r==missing||(n!="gondor"&&n!="rohan"))return false;
    }
    if(has("strider/aragorn is in any rohan region")){
        const auto r=evf_first_region(s,strider,false);if(r==missing||evf_nation(s,t,r)!="rohan")return false;
    }
    if(has("strider/aragorn is with a free peoples army")&&!has("gondor or rohan")&&evf_first_region(s,strider,true)==missing)return false;
    if(has("boromir is in a gondor region")){
        const auto r=evf_first_region(s,{"Boromir"},false);if(r==missing||evf_nation(s,t,r)!="gondor")return false;
    }
    if(has("aragorn is with a free peoples army in a region outside a free peoples nation")){
        const auto r=evf_first_region(s,aragorn,true);if(r==missing)return false;
        const auto n=evf_nation(s,t,r);
        if(n=="gondor"||n=="rohan"||n=="elves"||n=="dwarves"||n=="the north"||n=="north")return false;
    }
    // Promotion markers remain true after death; printed "in play" clauses
    // require the living map figure (a lone Companion needs no Army units).
    if(has("aragorn is in play")&&evf_first_region(s,aragorn,false)==missing)return false;
    if(has("gandalf the white is in play")&&evf_first_region(s,{"Gandalf the White"},false)==missing)return false;
    if(has("witch-king is in play")&&!evf_witch_king(s))return false;
    if(has("a companion is in fangorn")&&(evf_first_region(s,{"Gandalf the White"},false)==missing||!evf_fangorn_companion(s)))return false;
    return true;
}
