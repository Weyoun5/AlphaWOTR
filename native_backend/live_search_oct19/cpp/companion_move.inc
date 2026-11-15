// First companion group move plus every legal same-die continuation group.
int64_t cm_allowance(const Snapshot& s,const std::vector<uint64_t>& comps){
    int64_t level=0;std::set<std::string> names;
    for(auto comp:comps){level=std::max(level,ht_integer(s,comp,"level"));names.insert(std::string(text(s,obs_attr(s,comp,"name"))));}
    if(names.erase("Gandalf the White") && (names.empty() || (names.size()==1 && (names.count("Meriadoc") || names.count("Peregrin")))))return 4;
    return level;
}
std::vector<uint64_t> cm_companions(const Snapshot& s,uint64_t army,const std::set<std::string>& exclude={}){
    std::vector<uint64_t> comps;for(auto leader:child_edges(s,obs_attr(s,army,"leaders")))
        if(object_type(s,leader)=="game_state.Companion" && !exclude.count(std::string(text(s,obs_attr(s,leader,"name")))))comps.push_back(leader);return comps;
}
std::unique_ptr<Snapshot> native_companion_move_actions(const Snapshot& s,const Snapshot& t,int64_t index){
    am_Builder b;std::vector<uint64_t> rows;const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));
    for(size_t i=0;i<armies.size();i+=2){
        const auto start=sanitize_name(text(s,armies[i]));const auto comps=cm_companions(s,armies[i+1]);if(comps.empty())continue;
        auto emit=[&](const std::vector<uint64_t>& group,bool full){const auto movement=cm_allowance(s,group);if(movement<=0)return;
            std::vector<std::string> names;for(auto c:group)names.push_back(std::string(text(s,obs_attr(s,c,"name"))));std::sort(names.begin(),names.end());
            auto destinations=cmp_reachable(s,t,start,movement);
            if(me_companion_activation_available(s,t,start,group))destinations.insert(start);
            for(const auto& destination:destinations){
                std::vector<uint64_t> params{b.str("from"),b.str(start),b.str("to"),b.str(destination)};
                if(!full){std::vector<uint64_t> encoded;for(const auto& name:names)encoded.push_back(b.str(name));params.push_back(b.str("companions"));params.push_back(b.container(WOTR_LIST,encoded));}
                rows.push_back(b.row(index,"companion_move",params));
            }
        };
        emit(comps,true);for(size_t count=1;count<comps.size();++count)cmp_combinations(comps,count,[&](const std::vector<uint64_t>& group){emit(group,false);});
    }
    b.out.root=b.container(WOTR_LIST,rows);return std::make_unique<Snapshot>(std::move(b.out));
}
uint64_t cm_chain_options(Snapshot& s,const Snapshot& t,const std::set<std::string>& moved){
    std::vector<std::pair<std::string,uint64_t>> armies;const auto source=child_edges(s,obs_attr(s,s.root,"armies"));
    for(size_t i=0;i<source.size();i+=2)armies.push_back({sanitize_name(text(s,source[i])),source[i+1]});std::sort(armies.begin(),armies.end());
    std::vector<uint64_t> options;
    for(const auto& row:armies){const auto& start=row.first;const auto comps=cm_companions(s,row.second,moved);if(comps.empty())continue;
        std::map<int64_t,std::set<std::string>> destinations;
        auto emit=[&](const std::vector<uint64_t>& group){const auto allowance=cm_allowance(s,group);if(allowance<=0)return;
            std::vector<std::string> names;for(auto comp:group)names.push_back(std::string(text(s,obs_attr(s,comp,"name"))));std::sort(names.begin(),names.end());
            auto found=destinations.find(allowance);if(found==destinations.end())found=destinations.emplace(allowance,cmp_reachable(s,t,start,allowance)).first;
            auto targets=found->second;
            if(me_companion_activation_available(s,t,start,group))targets.insert(start);
            for(const auto& destination:targets){
                std::vector<uint64_t> named,already;for(const auto& name:names)named.push_back(rr_string(s,name));for(const auto& name:moved)already.push_back(rr_string(s,name));
                const auto group_node=rr_scalar(s,WOTR_LIST),moved_node=rr_scalar(s,WOTR_LIST);set_edges(s,group_node,named);set_edges(s,moved_node,already);
                const auto option=rr_scalar(s,WOTR_DICT);set_edges(s,option,{rr_string(s,"from_region"),rr_string(s,start),rr_string(s,"to_region"),rr_string(s,destination),rr_string(s,"companions"),group_node,rr_string(s,"_comp_moved"),moved_node});options.push_back(option);
            }
        };
        for(size_t count=1;count<=comps.size();++count)cmp_combinations(comps,count,emit);
    }
    const auto result=rr_scalar(s,WOTR_LIST);set_edges(s,result,options);return result;
}
void cm_open_chain(Snapshot& s,const Snapshot& t,const std::set<std::string>& moved){
    const auto options=cm_chain_options(s,t,moved);if(!truth(s,options))return;
    const auto done=rr_scalar(s,WOTR_DICT);set_edges(s,done,{rr_string(s,"_comp_done"),rr_scalar(s,WOTR_BOOL,1)});auto entries=child_edges(s,options);entries.push_back(done);set_edges(s,options,entries);
    const auto pending=rr_scalar(s,WOTR_DICT);set_edges(s,pending,{rr_string(s,"card_id"),rr_string(s,"__companion_moves__"),rr_string(s,"card_name"),rr_string(s,"__companion_moves__"),rr_string(s,"side"),rr_string(s,"fp"),rr_string(s,"decider"),rr_string(s,"fp"),rr_string(s,"direct"),rr_scalar(s,WOTR_BOOL,1),rr_string(s,"options"),options});rr_set_attr(s,s.root,"pending_card_option",pending);
}
std::vector<uint64_t> cm_select(const Snapshot& s,std::string_view start,const std::vector<std::string>& names){
    const auto army=mu_army(s,start);require(army!=missing,"Companion move has no source army");const auto comps=cm_companions(s,army);std::set<std::string> want(names.begin(),names.end());std::vector<uint64_t> group;
    for(auto comp:comps)if(names.empty() || want.count(std::string(text(s,obs_attr(s,comp,"name")))))group.push_back(comp);
    require(!group.empty() && (names.empty() || group.size()==want.size()),"Companion group is missing from source");return group;
}
void cm_transfer(Snapshot& s,const Snapshot& t,std::string_view start,std::string_view destination,const std::vector<uint64_t>& comps,bool activate=true){
    if(start==destination){
        if(activate)movement_entry_effects(s,t,destination,0,comps);
        return;
    }
    const auto army=mu_army(s,start),list=obs_attr(s,army,"leaders");
    for(auto comp:comps){auto remaining=child_edges(s,list);const auto found=std::find_if(remaining.begin(),remaining.end(),[&](uint64_t other){return ft_companion_equal(s,other,comp);});require(found!=remaining.end(),"Moving companion disappeared");remaining.erase(found);set_edges(s,list,remaining);}
    if(!truth(s,obs_attr(s,army,"units")) && !truth(s,list)){
        // Lookups accept normalized region names, while the dictionary retains
        // original display keys created by first-leg movement and separation.
        const auto entries=child_edges(s,obs_attr(s,s.root,"armies"));
        for(size_t i=0;i<entries.size();i+=2)if(sanitize_name(text(s,entries[i]))==start){const std::string original(text(s,entries[i]));am_erase(s,original);break;}
    }
    for(auto comp:comps)hp_place(s,t,comp,destination,activate);
}
std::unique_ptr<Snapshot> apply_companion_move(const Snapshot& source,const Snapshot& t,int64_t index,std::string_view from,std::string_view to,const std::vector<std::string>& names){
    quiet_stable_action_gate(source,t);ring_waiver(source,0);ring_waiver(source,1);fm_payment_die(source,index);
    const auto start=sanitize_name(from),destination=sanitize_name(to);
    const auto comps=cm_select(source,start,names);
    require(start==destination?me_companion_activation_available(source,t,start,comps):
            cmp_reachable(source,t,start,cm_allowance(source,comps)).count(destination),"Companion move destination unreachable or stationary move has no activation");
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained) || ctx_side(source,retained)==0,"Companion move belongs to retained ring actor");
    auto child=action_clone(source);auto& s=*child;rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    rr_set_attr(s,fm_payment_die(s,index),"used",rr_scalar(s,WOTR_BOOL,1));const auto group=cm_select(s,start,names);std::set<std::string> moved;for(auto comp:group)moved.insert(std::string(text(s,obs_attr(s,comp,"name"))));
    const bool new_destination=mu_army(s,destination)==missing;
    cm_transfer(s,t,start,destination,group);
    // The first action uses board.regions[destination]['orig']; the event
    // continuation creates canonical keys. Preserve this observable difference.
    if(new_destination){
        const auto meta=ctx_get(t,field(t,t.root,"board_regions"),destination),original=ctx_get(t,meta,"orig");
        const auto armies=obs_attr(s,s.root,"armies");auto entries=child_edges(s,armies);
        for(size_t i=0;i<entries.size();i+=2)if(text_is(s,entries[i],destination)){
            entries[i]=rr_string(s,original==missing ? std::string_view(destination) : text(t,original));break;
        }
        set_edges(s,armies,entries);
    }
    cm_open_chain(s,t,moved);rr_set_attr(s,s.root,"_battle_resolved_attacker",rr_scalar(s,WOTR_NONE));return child;
}
void cm_validate_chain(const Snapshot& s){
    const auto pending=obs_attr(s,s.root,"pending_card_option");require(!ctx_none(s,pending),"Companion movement continuation missing");
    for(auto name:{"card_id","card_name"})require(text_is(s,field(s,pending,name),"__companion_moves__"),"Wrong pending card family for companion move");
    require(text_is(s,field(s,pending,"side"),"fp") && text_is(s,field(s,pending,"decider"),"fp") && truth(s,field(s,pending,"direct")),"Companion continuation owner/direct mode invalid");typed(s,field(s,pending,"options"),WOTR_LIST,"Companion continuation options must be list");
}
std::unique_ptr<Snapshot> native_companion_chain_actions(const Snapshot& source){
    cm_validate_chain(source);auto output=std::make_unique<Snapshot>(source);auto& s=*output;const auto pending=obs_attr(s,s.root,"pending_card_option"),options=field(s,pending,"options");std::vector<uint64_t> rows;int64_t index=0;
    for(auto option:child_edges(s,options)){
        const auto params=rr_scalar(s,WOTR_DICT);set_edges(s,params,{rr_string(s,"idx"),rr_scalar(s,WOTR_INT,index++),rr_string(s,"card_id"),field(s,pending,"card_id"),rr_string(s,"card_name"),field(s,pending,"card_name"),rr_string(s,"side"),field(s,pending,"side"),rr_string(s,"choices"),option});
        const auto row=rr_scalar(s,WOTR_TUPLE);set_edges(s,row,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),params});rows.push_back(row);
    }
    const auto root=rr_scalar(s,WOTR_LIST);set_edges(s,root,rows);s.root=root;return ac_compact(s);
}
std::unique_ptr<Snapshot> apply_companion_move_option(const Snapshot& source,const Snapshot& t,int64_t index){
    cm_validate_chain(source);quiet_action_gate(source,t,false,false,false,false,false,false,false,true,false,false,false,true);
    const auto pending=obs_attr(source,source.root,"pending_card_option"),options=field(source,pending,"options");
    require(index>=0 && static_cast<uint64_t>(index)<source.nodes[options].count,"Companion continuation index out of range");
    auto child=action_clone(source);auto& s=*child;const auto new_pending=obs_attr(s,s.root,"pending_card_option"),new_options=field(s,new_pending,"options"),choice=s.edges[s.nodes[new_options].offset+index];
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));
    if(!truth(s,field(s,choice,"_comp_done",false))){
        const auto start=std::string(text(s,field(s,choice,"from_region"))),destination=std::string(text(s,field(s,choice,"to_region")));std::vector<std::string> names;std::set<std::string> moved;
        for(auto name:child_edges(s,field(s,choice,"companions")))names.push_back(std::string(text(s,name)));for(auto name:child_edges(s,field(s,choice,"_comp_moved")))moved.insert(std::string(text(s,name)));
        const auto group=cm_select(s,start,names);require(start==destination || cmp_reachable(s,t,start,cm_allowance(s,group)).count(destination),"Pending companion destination no longer reachable");
        cm_transfer(s,t,start,destination,group);for(const auto& name:names)moved.insert(name);cm_open_chain(s,t,moved);
    }
    return child;
}
