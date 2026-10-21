// Political borders apply to every walked region, including card movement.
std::set<std::string> esg_reachable(const Snapshot&,const Snapshot&,std::string_view,int);
bool esg_garrison(const Snapshot&,const Snapshot&,std::string_view);
using sj_Path=std::vector<std::string>;
std::vector<sj_Path> sj_paths(const Snapshot& s,const Snapshot& t,std::string_view source,std::string_view target,int steps){
    std::vector<sj_Path> result,queue{{}};if(source==target||esg_garrison(s,t,source))return result;
    for(size_t at=0;at<queue.size();++at){const auto route=queue[at];if(route.size()>=static_cast<size_t>(steps))continue;const std::string current=route.empty()?std::string(source):route.back();
        for(const auto& region:esg_reachable(s,t,current,1)){if(region==source||std::find(route.begin(),route.end(),region)!=route.end())continue;auto walked=route;walked.push_back(region);
            if(region==target)result.push_back(walked);else if(!contains_string(s,obs_attr(s,s.root,"besieged_strongholds",false),region))queue.push_back(walked);}}
    return result;
}
bool sj_may_follow(const Snapshot& s,const Snapshot& t,std::string_view nation,const sj_Path& route){
    if(political_view(s,nation,true).box>=3)return true;
    for(const auto& region:route){const auto owner=ctx_get(t,field(t,t.root,"region_nations"),region);if(!ctx_none(t,owner)&&ctx_nation(t,owner)!=nation)return false;}return true;
}
std::pair<std::vector<uint64_t>,sj_Path> sj_select(const Snapshot& s,const Snapshot& t,std::string_view source,std::string_view target,int steps,const std::vector<uint64_t>& units,bool exact=false){
    std::pair<std::vector<uint64_t>,sj_Path> best;
    for(const auto& path:sj_paths(s,t,source,target,steps)){std::vector<uint64_t> eligible;for(auto unit:units)if(sj_may_follow(s,t,ctx_nation(s,obs_attr(s,unit,"nation")),path))eligible.push_back(unit);
        if(exact&&eligible.size()!=units.size())continue;if(eligible.size()>best.first.size()){best.first=eligible;best.second=path;best.second.pop_back();}}
    return best;
}
