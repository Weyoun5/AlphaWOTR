// Exact card cargo, staged by category against the untouched source army.
using cag_Key=std::tuple<std::string,std::string,std::string>;
using cag_Counts=std::map<cag_Key,int64_t>;
bool cag_route_possible(const Snapshot& s,const Snapshot& t,std::string_view id,uint64_t choice,const cag_Counts& cargo,const std::vector<cag_Key>& categories,size_t remaining){
    const int steps=id=="SHADOW_Strategy_The_Shadow_Lengthens"?2:id=="SHADOW_Strategy_Shadows_Gather"?3:0;if(!steps)return true;
    std::set<std::string> nations,later;for(const auto& [key,n]:cargo)if(n&&std::get<0>(key)=="unit")nations.insert(std::get<1>(key));for(size_t i=remaining;i<categories.size();++i)if(std::get<0>(categories[i])=="unit")later.insert(std::get<1>(categories[i]));
    for(const auto& route:sj_paths(s,t,text(s,field(s,choice,"from_region")),text(s,field(s,choice,"to_region")),steps)){
        bool allowed=true;for(const auto& nation:nations)allowed&=sj_may_follow(s,t,nation,route);bool unit=!nations.empty();for(const auto& nation:later)unit|=sj_may_follow(s,t,nation,route);if(allowed&&unit)return true;}return false;
}
cag_Key cag_key(const Snapshot& s,uint64_t figure,bool unit){
    if(unit)return {"unit",std::string(ctx_nation(s,obs_attr(s,figure,"nation"))),std::string(enum_member(s,obs_attr(s,figure,"kind"),"game_state.UnitKind"))};
    if(object_type(s,figure)=="game_state.Nazgul")return {"nazgul","SAURON",""};
    const auto name=obs_attr(s,figure,"name",false);if(truth(s,name))return {"named",std::string(text(s,name)),""};
    return {"leader",std::string(ctx_nation(s,obs_attr(s,figure,"nation"))),""};
}
std::pair<std::vector<uint64_t>,std::vector<uint64_t>> cag_figures(const Snapshot& s,std::string_view source){
    const auto army=mu_army(s,source);require(army!=missing,"Cargo source has no army");std::pair<std::vector<uint64_t>,std::vector<uint64_t>> out;
    for(auto u:child_edges(s,obs_attr(s,army,"units")))if(ctx_unit_side(s,u)==1)out.first.push_back(u);
    for(auto l:child_edges(s,obs_attr(s,army,"leaders")))if(am_leader_side(s,l)==1&&!am_saruman(s,l))out.second.push_back(l);return out;
}
cag_Counts cag_inventory(const Snapshot& s,std::string_view source){cag_Counts out;const auto figures=cag_figures(s,source);for(auto u:figures.first)++out[cag_key(s,u,true)];for(auto l:figures.second)++out[cag_key(s,l,false)];return out;}
cag_Key cag_read_key(const Snapshot& s,const std::vector<uint64_t>& values){return {std::string(text(s,values[0])),std::string(text(s,values[1])),std::string(text(s,values[2]))};}
cag_Counts cag_decode(const Snapshot& s,uint64_t cargo){
    cag_Counts out;require(cargo!=missing&&(s.nodes[cargo].tag==WOTR_TUPLE||s.nodes[cargo].tag==WOTR_LIST),"Cargo requires sequence");
    for(auto row:child_edges(s,cargo)){const auto values=child_edges(s,row);require((s.nodes[row].tag==WOTR_TUPLE||s.nodes[row].tag==WOTR_LIST)&&values.size()==4,"Malformed cargo row");const auto key=cag_read_key(s,values);const auto n=typed(s,values[3],WOTR_INT,"Cargo count integer required").integer;require(n>0&&out.emplace(key,n).second,"Invalid or duplicate cargo count");}return out;
}
uint64_t cag_key_node(Snapshot& s,const cag_Key& key){return cp_container(s,WOTR_TUPLE,{rr_string(s,std::get<0>(key)),rr_string(s,std::get<1>(key)),rr_string(s,std::get<2>(key))});}
uint64_t cag_freeze(Snapshot& s,const cag_Counts& cargo){std::vector<uint64_t> rows;for(const auto& [key,n]:cargo)if(n)rows.push_back(cp_container(s,WOTR_TUPLE,{rr_string(s,std::get<0>(key)),rr_string(s,std::get<1>(key)),rr_string(s,std::get<2>(key)),rr_scalar(s,WOTR_INT,n)}));return cp_container(s,WOTR_TUPLE,rows);}
std::pair<std::vector<uint64_t>,std::vector<uint64_t>> cag_selected(const Snapshot& s,std::string_view source,uint64_t cargo){
    auto wanted=cag_decode(s,cargo);const auto available=cag_inventory(s,source);bool unit=false;for(const auto& [key,n]:wanted){const auto found=available.find(key);require(found!=available.end()&&n<=found->second,"Cargo exceeds inventory");unit|=std::get<0>(key)=="unit";}require(unit,"Cargo requires army unit");
    const auto figures=cag_figures(s,source);std::pair<std::vector<uint64_t>,std::vector<uint64_t>> out;
    for(auto u:figures.first){auto& n=wanted[cag_key(s,u,true)];if(n>0){--n;out.first.push_back(u);}}
    for(auto l:figures.second){auto& n=wanted[cag_key(s,l,false)];if(n>0){--n;out.second.push_back(l);}}return out;
}
uint64_t cag_dict_copy(Snapshot& s,uint64_t source){const auto out=rr_scalar(s,WOTR_DICT);set_edges(s,out,child_edges(s,source));return out;}
uint64_t cag_stage(Snapshot& s,const Snapshot& t,std::string_view id,uint64_t choice,std::string_view source){
    if(truth(s,field(s,choice,"_cargo_complete",false)))return choice;
    const auto available=cag_inventory(s,source);std::vector<cag_Key> categories;for(const auto& [key,n]:available)categories.push_back(key);
    std::sort(categories.begin(),categories.end(),[](const auto& a,const auto& b){return std::make_pair(std::get<0>(a)!="unit",a)<std::make_pair(std::get<0>(b)!="unit",b);});require(!categories.empty()&&std::get<0>(categories.front())=="unit","No movable army cargo");
    size_t index=0;uint64_t continuation; cag_Counts cargo;
    const auto selected_index=field(s,choice,"_cargo_index",false);
    if(selected_index!=missing){const auto i=typed(s,selected_index,WOTR_INT,"Cargo index integer").integer;require(i>=0&&static_cast<size_t>(i)<categories.size(),"Invalid cargo stage");index=static_cast<size_t>(i);cargo=cag_decode(s,field(s,choice,"cargo"));const auto keyparts=child_edges(s,field(s,choice,"_cargo_key"));require(keyparts.size()==3&&cag_read_key(s,keyparts)==categories[index],"Cargo category changed");
        const auto count=typed(s,field(s,choice,"count"),WOTR_INT,"Cargo count integer").integer;require(count>=0&&count<=available.at(categories[index])&&cargo[categories[index]]==count,"Invalid selected cargo count");for(const auto& [key,n]:cargo){const auto found=std::find(categories.begin(),categories.begin()+index+1,key);require(found!=categories.begin()+index+1&&n<=available.at(key),"Invalid accumulated cargo");}continuation=cag_dict_copy(s,field(s,choice,"_cargo_continue"));++index;
    }else{continuation=cag_dict_copy(s,choice);cp_put(s,continuation,"_from_pending",rr_scalar(s,WOTR_BOOL,1));}
    for(;index<categories.size();++index){const auto& key=categories[index];int64_t minimum=0;const bool later_unit=std::any_of(categories.begin()+index+1,categories.end(),[](const auto& k){return std::get<0>(k)=="unit";});bool have_unit=false;for(const auto& [k,n]:cargo)have_unit|=std::get<0>(k)=="unit"&&n>0;if(std::get<0>(key)=="unit"&&!later_unit&&!have_unit)minimum=1;
        const auto category_node=cag_key_node(s,key);std::vector<uint64_t> options;for(int64_t n=minimum;n<=available.at(key);++n){auto next=cargo;next[key]=n;if(!cag_route_possible(s,t,id,continuation,next,categories,index+1))continue;const auto option=cag_dict_copy(s,continuation);cp_put(s,option,"cargo",cag_freeze(s,next));cp_put(s,option,"count",rr_scalar(s,WOTR_INT,n));cp_put(s,option,"_cargo_key",category_node);cp_put(s,option,"_cargo_index",rr_scalar(s,WOTR_INT,index));cp_put(s,option,"_cargo_continue",continuation);options.push_back(option);}
        if(options.size()>1){const auto card=ev_card(t,id);rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,text(t,obs_attr(t,card,"card_name")))},{"side",rr_string(s,"sh")},{"decider",rr_string(s,"sh")},{"direct",rr_scalar(s,WOTR_BOOL,1)},{"options",cp_container(s,WOTR_LIST,options)},{"cargo_source",rr_string(s,sanitize_name(source))}}));return missing;}
        require(options.size()==1,"No legal cargo count");cargo=cag_decode(s,field(s,options.front(),"cargo"));
    }
    const auto frozen=cag_freeze(s,cargo);cag_selected(s,source,frozen);cp_put(s,continuation,"cargo",frozen);cp_put(s,continuation,"_cargo_complete",rr_scalar(s,WOTR_BOOL,1));return continuation;
}
