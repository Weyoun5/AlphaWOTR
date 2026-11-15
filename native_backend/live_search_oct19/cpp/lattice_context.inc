// UNINCLUDED: source segmented-separation context, reusing native travel helpers.
std::unique_ptr<Snapshot> native_search_lattice_context(const Snapshot& s,const Snapshot& tables,const Snapshot& menu){
    sm_Builder b(s);bool allowed=true;
    // search_lattice.allowed examines actual vars(), not class defaults.
    for(auto [key,value]:b.pairs(attributes(s,s.root,"game_state.GameState"))){
        const auto name=text(s,key);
        if(name.size()<8||name.substr(0,8)!="pending_"||ctx_none(s,value))continue;
        const bool empty_tiles=name=="pending_hunt_tiles"&&s.nodes[value].tag==WOTR_LIST&&s.nodes[value].count==0;
        if(!empty_tiles)allowed=false;
    }
    const auto mouth=obs_attr(s,s.root,"mouth_of_sauron",false);
    if(!ctx_none(s,mouth)&&truth(s,obs_attr(s,mouth,"in_play")))allowed=false;
    std::vector<uint64_t> blocks;
    if(allowed){
        const auto fs=obs_attr(s,s.root,"fellowship");
        const auto companions=cmp_separable(s);const auto start=glb_fellowship_key(s,fs);const auto progress=ht_integer(s,fs,"progress");
        if(!truth(s,obs_attr(s,fs,"mordor_entered"))&&companions.size()>2){
            std::set<int64_t> movements;std::vector<std::string> names;
            for(auto comp:companions)names.emplace_back(text(s,obs_attr(s,comp,"name")));
            std::sort(names.begin(),names.end());
            for(size_t count=2;count<companions.size();++count)cmp_combinations(companions,count,[&](const std::vector<uint64_t>& group){
                int64_t highest=0;for(auto comp:group)highest=std::max(highest,ht_integer(s,comp,"level"));movements.insert(progress+highest);
            });
            std::vector<std::pair<int64_t,std::set<std::string>>> destination_rows;
            for(auto movement:movements)if(movement>0){
                auto destinations=cmp_reachable(s,tables,start,movement);destinations.insert(start);
                destination_rows.emplace_back(movement,std::move(destinations));
            }
            struct Group{int64_t die;std::vector<uint64_t> indices;std::set<std::string> destinations;};
            std::vector<Group> groups;
            const auto rows=ap_rows(menu);
            for(size_t index=0;index<rows.size();++index){
                const auto& row=rows[index];if(row.kind!="companion_separate")continue;
                const auto members=field(menu,row.params,"companions",false);if(ctx_none(menu,members))continue;
                typed(menu,members,WOTR_LIST,"Lattice group names must be list");
                const auto count=menu.nodes[members].count;if(count<2||count>=companions.size())continue;
                if(groups.empty()||groups.back().die!=row.die)groups.push_back({row.die,{},{}});
                groups.back().indices.push_back(b.integer(index));groups.back().destinations.emplace(text(menu,field(menu,row.params,"to_region")));
            }
            for(const auto& group:groups){
                std::vector<uint64_t> names_out,dest_sig;
                for(const auto& name:names)names_out.push_back(b.str(name));
                for(const auto& row:destination_rows){
                    std::vector<uint64_t> values;for(const auto& dest:row.second)values.push_back(b.str(dest));
                    dest_sig.push_back(b.tup({b.integer(row.first),b.tup(values)}));
                }
                const auto key=b.tup({b.integer(group.die),b.str(start),b.integer(progress),b.tup(names_out),b.tup(dest_sig)});
                std::vector<uint64_t> destinations;for(const auto& dest:group.destinations)destinations.push_back(b.str(dest));
                blocks.push_back(b.tup({b.tup(group.indices),key,b.tup(destinations)}));
            }
        }
    }
    b.out.root=b.tup({b.boolean(allowed),b.tup(blocks)});return ac_compact(b.out);
}
