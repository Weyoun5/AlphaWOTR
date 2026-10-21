// Printed Nazgul/Witch-king flights and the die-seeded same-die continuation.
// The source's destination rule allows FP field armies. Card-seeded chains
// with delayed effects remain distinct and are deliberately not admitted.
bool nz_plain(const Snapshot& s,uint64_t leader){return object_type(s,leader)=="game_state.Nazgul";}
bool nz_witch(const Snapshot& s,uint64_t leader){return object_type(s,leader)=="game_logic.Minion" && ev_lower(text(s,obs_attr(s,leader,"name"))).find("witch")!=std::string::npos;}
bool nz_witch_chain(const Snapshot& s,uint64_t leader){
    if(object_type(s,leader)!="game_logic.Minion")return false;std::string value;
    for(char c:ev_lower(text(s,obs_attr(s,leader,"name"))))if(c>='a' && c<='z')value.push_back(c);
    return value.find("witchking")!=std::string::npos;
}
struct nz_Source {std::string key;int64_t plain=0;bool witch=false;};
struct nz_Layout {std::vector<nz_Source> sources;std::vector<std::string> destinations;};
nz_Layout nz_layout(const Snapshot& s,const Snapshot& t){
    nz_Layout out;std::vector<nz_Source> witch_only;std::set<std::string> unique;const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));
    for(size_t i=0;i<armies.size();i+=2){
        nz_Source row;row.key=sanitize_name(text(s,armies[i]));require(unique.insert(row.key).second,"Nazgul menu refuses duplicate normalized army keys");
        for(auto leader:child_edges(s,obs_attr(s,armies[i+1],"leaders"))){row.plain+=nz_plain(s,leader);row.witch|=nz_witch(s,leader);}
        if(row.plain)out.sources.push_back(row);else if(row.witch)witch_only.push_back(row);
    }
    // Real WotR has one Witch-king. Multiple WK-only regions would inherit
    // Python set hash order; refuse that malformed domain instead of inventing it.
    require(witch_only.size()<=1,"Nazgul menu refuses multiple Witch-king-only regions");
    out.sources.insert(out.sources.end(),witch_only.begin(),witch_only.end());
    if(out.sources.empty())return out;
    const auto sieges=obs_attr(s,s.root,"besieged_strongholds");
    const auto metadata=child_edges(t,field(t,t.root,"board_regions"));
    for(size_t i=0;i<metadata.size();i+=2){
        const auto key=std::string(text(t,metadata[i]));const auto meta=metadata[i+1];
        if(truth(t,field(t,meta,"is_stronghold",false))){
            // RB24: current control governs flights, even when FP captured
            // a Stronghold printed as belonging to a Shadow nation.
            if(at_controller(s,t,key)==0 && !contains_string(s,sieges,key))continue;
        }
        out.destinations.push_back(key);
    }
    std::sort(out.destinations.begin(),out.destinations.end());return out;
}
std::unique_ptr<Snapshot> native_nazgul_move_actions(const Snapshot& s,const Snapshot& t,int32_t side){
    require(side==0 || side==1,"Nazgul menu actor invalid");am_Builder b;std::vector<uint64_t> rows;
    for(const auto key:{"pending_combat","pending_guide_change","pending_hunt","pending_card_option","pending_card_followup","pending_discard","pending_army_move_die","pending_muster_die"})
        require(ctx_none(s,obs_attr(s,s.root,key,false)),"Nazgul ordinary menu rejects pending decision");
    for(const auto key:{"_declare_phase","_eye_allocation_phase","_reveal_place_due"})require(!truth(s,obs_attr(s,s.root,key,false)),"Nazgul ordinary menu rejects phase decision");
    ring_waiver(s,side);
    if(side==1){
        const auto layout=nz_layout(s,t);std::array<bool,7> seen{};const auto dice=child_edges(s,pa_pool(s,1));
        for(size_t i=0;i<dice.size();++i){
            const auto d=dice[i],face=obs_attr(s,d,"current");if(truth(s,obs_attr(s,d,"used")) || ctx_none(s,face))continue;
            const auto f=rr_face_index(enum_member(s,face,"game_state.ActionDieFace"));if(seen[f])continue;seen[f]=true;if(f!=0 && f!=6)continue;
            for(const auto& dst:layout.destinations)for(const auto& src:layout.sources){
                if(src.key==dst)continue;
                auto add=[&](bool wk,int64_t count){std::vector<uint64_t> p={b.str("from"),b.str(src.key),b.str("to"),b.str(dst)};
                    if(wk){p.push_back(b.str("include_wk"));p.push_back(rr_scalar(b.out,WOTR_BOOL,1));}
                    if(count>=0){p.push_back(b.str("count"));p.push_back(b.integer(count));}rows.push_back(b.row(i,"nazgul_move",p));};
                if(src.plain){add(false,-1);for(int64_t n=1;n<src.plain;++n)add(false,n);}
                if(src.witch){add(true,-1);for(int64_t n=0;n<src.plain;++n)add(true,n);}
            }
        }
    }
    b.out.root=b.container(WOTR_LIST,rows);return std::make_unique<Snapshot>(b.out);
}
using nz_Arrivals=std::vector<std::pair<std::string,int64_t>>;
int64_t nz_get(const nz_Arrivals& arr,std::string_view key){for(const auto& v:arr)if(v.first==key)return v.second;return 0;}
void nz_set(nz_Arrivals& arr,std::string_view key,int64_t value){for(auto& v:arr)if(v.first==key){v.second=value;return;}arr.emplace_back(key,value);}
uint64_t nz_options(Snapshot& s,const Snapshot& t,const nz_Arrivals& arr){
    // This builder only appends nodes. Retain each immutable string's
    // original first-match ID, avoiding a whole-graph scan for every option.
    // Option and arrival dictionaries remain distinct, as in the source.
    std::unordered_map<std::string,uint64_t> strings;strings.reserve(128);
    auto str=[&](std::string_view value){std::string key(value);const auto found=strings.find(key);if(found!=strings.end())return found->second;const auto node=rr_string(s,value);strings.emplace(std::move(key),node);return node;};
    auto dict=[&](std::initializer_list<std::pair<std::string_view,uint64_t>> fields){const auto node=rr_scalar(s,WOTR_DICT);std::vector<uint64_t> edges;edges.reserve(fields.size()*2);for(const auto& item:fields){edges.push_back(str(item.first));edges.push_back(item.second);}set_edges(s,node,edges);return node;};
    auto arrival=[&](){const auto node=rr_scalar(s,WOTR_DICT);std::vector<uint64_t> edges;edges.reserve(arr.size()*2);for(const auto& item:arr){edges.push_back(str(item.first));edges.push_back(rr_scalar(s,WOTR_INT,item.second));}set_edges(s,node,edges);return node;};
    const auto layout=nz_layout(s,t);std::vector<uint64_t> rows;const bool wk=nz_get(arr,"__wk_ok__") && !nz_get(arr,"__wk__");
    for(const auto& dst:layout.destinations)for(const auto& src:layout.sources){
        if(src.key==dst)continue;
        if(src.plain && src.plain-nz_get(arr,src.key)>0)rows.push_back(dict({{"from_region",str(src.key)},{"to_region",str(dst)},{"count",rr_scalar(s,WOTR_INT,1)},{"_naz_arr",arrival()}}));
        if(src.witch && wk)rows.push_back(dict({{"from_region",str(src.key)},{"to_region",str(dst)},{"include_wk",rr_scalar(s,WOTR_BOOL,1)},{"_naz_arr",arrival()}}));
    }
    if(nz_get(arr,"__mo_ok__") && !nz_get(arr,"__mo__")){
        const auto src=mo_region(s);if(!src.empty())for(const auto& dst:mo_reachable(s,t,src)){
            const auto names=rr_scalar(s,WOTR_LIST);set_edges(s,names,{str("Mouth of Sauron")});
            rows.push_back(dict({{"from_region",str(src)},{"to_region",str(dst)},{"include_mouth",rr_scalar(s,WOTR_BOOL,1)},{"companions",names},{"_naz_arr",arrival()}}));
        }
    }
    const auto options=rr_scalar(s,WOTR_LIST);set_edges(s,options,rows);return options;
}
void nz_open(Snapshot& s,const Snapshot& t,const nz_Arrivals& arr){
    const auto options=nz_options(s,t,arr);auto rows=child_edges(s,options);if(rows.empty())return;
    rows.push_back(ev_dict(s,{{"_naz_done",rr_scalar(s,WOTR_BOOL,1)}}));set_edges(s,options,rows);
    rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,"__nazgul_moves__")},{"card_name",rr_string(s,"__nazgul_moves__")},
        {"side",rr_string(s,"sh")},{"decider",rr_string(s,"sh")},{"direct",rr_scalar(s,WOTR_BOOL,1)},{"options",options}}));
}
void nz_transfer(Snapshot& s,const Snapshot& t,std::string_view from,std::string_view to,const std::vector<uint64_t>& figures,bool original_key){
    const auto key=am_key(s,from);const auto source=mu_army(s,from);require(source!=missing,"Nazgul source vanished");
    const auto leaders=obs_attr(s,source,"leaders");auto kept=child_edges(s,leaders);
    for(auto figure:figures){const auto found=std::find(kept.begin(),kept.end(),figure);require(found!=kept.end(),"Nazgul figure absent from source");kept.erase(found);}
    set_edges(s,leaders,kept);if(kept.empty() && !truth(s,obs_attr(s,source,"units")))am_erase(s,key);
    auto destination=mu_army(s,to);
    if(destination==missing){
        const auto region=ctx_get(s,obs_attr(s,obs_attr(s,s.root,"board"),"_regions"),to);require(region!=missing,"Nazgul destination outside Board");
        std::string dest(to);
        if(original_key){const auto meta=ctx_get(t,field(t,t.root,"board_regions"),to),orig=field(t,meta,"orig",false);if(orig!=missing)dest=std::string(text(t,orig));}
        destination=am_new_army(s,region);am_insert(s,dest,destination);
    }
    am_extend(s,destination,"leaders",figures);
}
uint64_t nz_payment(const Snapshot& s,int64_t index){
    const auto dice=child_edges(s,pa_pool(s,1));require(index>=0 && static_cast<uint64_t>(index)<dice.size(),"Nazgul payment index invalid");const auto d=dice[index];
    require(object_type(s,d)=="game_state.ActionDie" && ctx_side(s,obs_attr(s,d,"side"))==1 && !truth(s,obs_attr(s,d,"used")) && enum_is(s,obs_attr(s,d,"current"),"game_state.ActionDieFace","CHARACTER"),"Nazgul move needs named unused Shadow Character die");return d;
}
std::unique_ptr<Snapshot> apply_nazgul_move_action(const Snapshot& source,const Snapshot& t,int32_t side,int64_t index,std::string_view raw_from,std::string_view raw_to,int64_t count,bool witch){
    require(side==1,"Nazgul movement belongs to Shadow");quiet_stable_action_gate(source,t);ring_waiver(source,0);ring_waiver(source,1);nz_payment(source,index);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained) || ctx_side(source,retained)==1,"Nazgul move belongs to retained Ring actor");
    const auto from=sanitize_name(raw_from),to=sanitize_name(raw_to);const auto layout=nz_layout(source,t);
    require(from!=to && std::find(layout.destinations.begin(),layout.destinations.end(),to)!=layout.destinations.end(),"Nazgul destination illegal");
    const auto src=std::find_if(layout.sources.begin(),layout.sources.end(),[&](const nz_Source& r){return r.key==from;});require(src!=layout.sources.end(),"Nazgul source absent");
    require(count==-1 || (count>=(witch?0:1) && count<src->plain),"Nazgul count is not a generated proper subset");
    require(witch?src->witch:src->plain>0,"Requested Nazgul figure family absent");
    auto child=action_clone(source);auto& s=*child;std::vector<uint64_t> figures;
    for(auto leader:child_edges(s,obs_attr(s,mu_army(s,from),"leaders")))if(nz_plain(s,leader) && (count<0 || figures.size()<static_cast<size_t>(count)))figures.push_back(leader);
    const auto plain=figures.size();
    if(witch)for(auto leader:child_edges(s,obs_attr(s,mu_army(s,from),"leaders")))if(nz_witch(s,leader)){require(nz_witch_chain(s,leader),"Unrecognized Witch-king figure spelling");figures.push_back(leader);}
    require(!figures.empty(),"Nazgul move selected no figures");
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));rr_set_attr(s,nz_payment(s,index),"used",rr_scalar(s,WOTR_BOOL,1));
    nz_transfer(s,t,from,to,figures,true);nz_Arrivals arr={{to,static_cast<int64_t>(plain)},{"__wk_ok__",1},{"__mo_ok__",1}};if(witch)arr.emplace_back("__wk__",1);
    nz_open(s,t,arr);tc_open_hand_limit_discard(s);rr_set_attr(s,s.root,"_battle_resolved_attacker",rr_scalar(s,WOTR_NONE));return child;
}
uint64_t nz_pending(const Snapshot& s){
    const auto p=obs_attr(s,s.root,"pending_card_option",false);require(!ctx_none(s,p),"No Nazgul continuation pending");
    for(auto key:{"card_id","card_name"})require(text_is(s,field(s,p,key),"__nazgul_moves__"),"Only die-seeded Nazgul continuation is ported");
    require(text_is(s,field(s,p,"side"),"sh") && text_is(s,field(s,p,"decider"),"sh") && truth(s,field(s,p,"direct")),"Nazgul continuation owner/direct mismatch");
    const auto options=field(s,p,"options");require(typed(s,options,WOTR_LIST,"Nazgul options must be a list").count>0,"Empty Nazgul continuation");return p;
}
std::unique_ptr<Snapshot> native_nazgul_chain_actions(const Snapshot& source,int32_t side){
    require(side==1,"Nazgul continuation belongs to Shadow");nz_pending(source);auto out=std::make_unique<Snapshot>(source);auto& s=*out;const auto p=obs_attr(s,s.root,"pending_card_option");
    std::vector<uint64_t> rows;int64_t index=0;
    for(auto option:child_edges(s,field(s,p,"options"))){
        const auto params=ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",option}});
        const auto row=rr_scalar(s,WOTR_TUPLE);set_edges(s,row,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),params});rows.push_back(row);
    }
    const auto root=rr_scalar(s,WOTR_LIST);set_edges(s,root,rows);s.root=root;return ac_compact(s);
}
bool nz_current_choice(const Snapshot& state,const Snapshot& actions,uint64_t choice,const Snapshot& t,const nz_Arrivals& arr){
    const auto from=field(actions,choice,"from_region",false),to=field(actions,choice,"to_region",false);
    if(ctx_none(actions,from) || ctx_none(actions,to))return false;
    const std::string src(text(actions,from)),dst(text(actions,to));if(src==dst || sanitize_name(src)!=src || sanitize_name(dst)!=dst)return false;
    const auto mouth=field(actions,choice,"include_mouth",false),witch=field(actions,choice,"include_wk",false),count=field(actions,choice,"count",false);
    if(mouth!=missing){
        if(actions.nodes[choice].count!=10 || witch!=missing || count!=missing || actions.nodes[mouth].tag!=WOTR_BOOL || !truth(actions,mouth))return false;
        const auto names=field(actions,choice,"companions",false);
        if(names==missing || actions.nodes[names].tag!=WOTR_LIST || actions.nodes[names].count!=1 || !text_is(actions,child_edges(actions,names)[0],"Mouth of Sauron"))return false;
        return nz_get(arr,"__mo_ok__") && !nz_get(arr,"__mo__") && mo_region(state)==src && mo_reachable(state,t,src).count(dst);
    }
    if(actions.nodes[choice].count!=8)return false;
    const auto layout=nz_layout(state,t);if(std::find(layout.destinations.begin(),layout.destinations.end(),dst)==layout.destinations.end())return false;
    const auto source=std::find_if(layout.sources.begin(),layout.sources.end(),[&](const nz_Source& r){return r.key==src;});if(source==layout.sources.end())return false;
    if(witch!=missing)return count==missing && actions.nodes[witch].tag==WOTR_BOOL && truth(actions,witch) && source->witch && nz_get(arr,"__wk_ok__") && !nz_get(arr,"__wk__");
    return count!=missing && actions.nodes[count].tag==WOTR_INT && actions.nodes[count].integer==1 && source->plain-nz_get(arr,src)>0;
}
std::unique_ptr<Snapshot> apply_nazgul_chain_action(const Snapshot& source,const Snapshot& t,int32_t side,int64_t index){
    require(side==1,"Nazgul continuation belongs to Shadow");const auto p=nz_pending(source),options=field(source,p,"options");
    require(index>=0 && static_cast<uint64_t>(index)<source.nodes[options].count,"Nazgul continuation index invalid");const auto choice=source.edges[source.nodes[options].offset+index];
    for(auto key:{"_naz_post","_then_followup"})require(!truth(source,field(source,choice,key,false)),"Card-seeded Nazgul followups are not ported");
    auto child=action_clone(source);auto& s=*child;rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));quiet_stable_action_gate(s,t);ring_waiver(s,0);ring_waiver(s,1);
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    if(truth(source,field(source,choice,"_naz_done",false))){tc_open_hand_limit_discard(s);return ac_compact(s);}
    nz_Arrivals arr;const auto dict=child_edges(source,field(source,choice,"_naz_arr"));
    for(size_t i=0;i<dict.size();i+=2){const std::string key(text(source,dict[i]));const auto value=typed(source,dict[i+1],WOTR_INT,"Nazgul arrivals must be integers").integer;require(value>=0,"Nazgul arrivals cannot be negative");arr.emplace_back(key,value);}
    require(nz_get(arr,"__wk_ok__")==1 && nz_get(arr,"__mo_ok__")==1,"Nazgul arrivals must originate from a Character die");
    // Check the same current-menu predicates directly, without constructing
    // thousands of unrelated options just to validate this one chosen leg.
    require(nz_current_choice(s,source,choice,t,arr),"Nazgul continuation choice is stale or malformed");
    const auto from=sanitize_name(text(source,field(source,choice,"from_region"))),to=sanitize_name(text(source,field(source,choice,"to_region")));
    const bool mouth=truth(source,field(source,choice,"include_mouth",false)),witch=truth(source,field(source,choice,"include_wk",false));
    const auto army=mu_army(s,from);require(army!=missing,"Nazgul continuation source absent");uint64_t figure=missing;
    for(auto l:child_edges(s,obs_attr(s,army,"leaders")))if(mouth?mo_figure(s,l):witch?nz_witch_chain(s,l):nz_plain(s,l)){figure=l;break;}
    require(figure!=missing,"Nazgul continuation figure absent");nz_transfer(s,t,from,to,{figure},false);
    if(mouth)nz_set(arr,"__mo__",1);else if(witch)nz_set(arr,"__wk__",1);else nz_set(arr,to,nz_get(arr,to)+1);
    nz_open(s,t,arr);tc_open_hand_limit_discard(s);return ac_compact(s);
}
