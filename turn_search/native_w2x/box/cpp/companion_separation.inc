// Ordinary voluntary separation: every uncapped group and destination, followed
// by one funded action with exact input-name ordering and Guide continuation.
std::set<std::string> cmp_reachable(const Snapshot& s,const Snapshot& t,std::string_view start,int64_t steps){
    const auto sieges=obs_attr(s,s.root,"besieged_strongholds");std::set<std::string> blocked;
    const auto controls=obs_attr(s,s.root,"enemy_controlled");
    for(auto entry:child_edges(s,sieges)){
        const auto key=std::string(text(s,entry));const bool fp=contains_string(s,ctx_side_value(s,controls,0),key) ||
            (contains_string(t,field(t,t.root,"fp_region_keys"),key) && !contains_string(s,ctx_side_value(s,controls,1),key));
        if(fp)blocked.insert(key);
    }
    if(blocked.count(std::string(start)))return {};
    std::set<std::string> visited{std::string(start)},reached,frontier{std::string(start)};
    for(int64_t depth=0;depth<steps && !frontier.empty();++depth){std::set<std::string> next;
        for(const auto& key:frontier)for(const auto& adjacent:fr_neighbors(s,key)){
            const auto canonical=sanitize_name(adjacent);if(!visited.insert(canonical).second || blocked.count(canonical))continue;
            reached.insert(canonical);if(!hp_shadow_controlled_stronghold(s,t,canonical))next.insert(canonical);
        }frontier=std::move(next);
    }
    return reached;
}
template<class Callback>void cmp_combinations(const std::vector<uint64_t>& values,size_t count,Callback callback){
    std::vector<uint64_t> selected;
    auto visit=[&](auto&& self,size_t start)->void{
        if(selected.size()==count){callback(selected);return;}
        const auto needed=count-selected.size();if(values.size()-start<needed)return;
        for(size_t i=start;i+needed<=values.size();++i){selected.push_back(values[i]);self(self,i+1);selected.pop_back();}
    };visit(visit,0);
}
std::vector<uint64_t> cmp_separable(const Snapshot& s){
    std::vector<uint64_t> comps;const auto fs=obs_attr(s,s.root,"fellowship");
    for(auto comp:child_edges(s,obs_attr(s,fs,"companions")))if(!text_is(s,obs_attr(s,comp,"name"),"Gollum") && ht_integer(s,comp,"level")>0)comps.push_back(comp);
    return comps;
}
struct cmp_ProperRow{std::vector<std::string> names;std::string destination;};

// A prefix is not yet a rooted Snapshot. Never read/copy its unset root.
struct cmp_EncodedPrefix {
    std::vector<WotrValue> nodes;
    std::vector<uint64_t> edges,rows;
    std::vector<uint8_t> bytes;
    cmp_EncodedPrefix(const Snapshot& s,const std::vector<uint64_t>& r):nodes(s.nodes),edges(s.edges),rows(r),bytes(s.bytes){}
    bool matches(const Snapshot& s,const std::vector<uint64_t>& r)const{
        return rows==r && nodes.size()==s.nodes.size() && edges==s.edges && bytes==s.bytes &&
            (nodes.empty() || std::memcmp(nodes.data(),s.nodes.data(),nodes.size()*sizeof(WotrValue))==0);
    }
    size_t charge()const{return nodes.size()*sizeof(WotrValue)+(edges.size()+rows.size())*sizeof(uint64_t)+bytes.size();}
};
struct cmp_EncodedBlock {
    std::shared_ptr<const std::vector<cmp_ProperRow>> owner;
    std::unique_ptr<cmp_EncodedPrefix> prefix;
    std::shared_ptr<const Snapshot> complete;
    size_t charge=0;
};
struct cmp_LatticeMemo{
    std::mutex mutex;
    std::unordered_map<std::string,std::shared_ptr<const std::vector<cmp_ProperRow>>> blocks;
    std::unordered_map<const std::vector<cmp_ProperRow>*,cmp_EncodedBlock> encoded;
    size_t encoded_bytes=0;uint64_t encoded_hits=0,encoded_misses=0,encoded_evictions=0;
};
cmp_LatticeMemo& cmp_lattice_memo(){static cmp_LatticeMemo value;return value;}
std::unique_ptr<Snapshot> native_companion_separation_actions(const Snapshot& s,const Snapshot& t,int64_t index){
    am_Builder b;std::vector<uint64_t> rows;const auto fs=obs_attr(s,s.root,"fellowship");
    if(truth(s,obs_attr(s,fs,"mordor_entered"))){b.out.root=b.container(WOTR_LIST,rows);return std::make_unique<Snapshot>(std::move(b.out));}
    require(ctx_none(s,obs_attr(s,s.root,"pending_hunt")),"Voluntary separation menu cannot replace Hunt placement");
    const auto companions=cmp_separable(s);const auto start=glb_fellowship_key(s,fs);const auto progress=ht_integer(s,fs,"progress");
    std::unordered_map<int64_t,std::set<std::string>> destinations;
    auto emit=[&](const std::vector<uint64_t>& group){
        int64_t highest=0;std::vector<std::string> names;for(auto comp:group){highest=std::max(highest,ht_integer(s,comp,"level"));names.push_back(std::string(text(s,obs_attr(s,comp,"name"))));}
        const auto movement=progress+highest;if(movement<=0)return;auto found=destinations.find(movement);
        if(found==destinations.end()){auto reachable=cmp_reachable(s,t,start,movement);reachable.insert(start);found=destinations.emplace(movement,std::move(reachable)).first;}
        std::sort(names.begin(),names.end());
        for(const auto& dest:found->second){
            if(names.size()==1)rows.push_back(b.row(index,"companion_separate",{b.str("companion"),b.str(names[0]),b.str("to_region"),b.str(dest)}));
            else{std::vector<uint64_t> encoded;for(const auto& name:names)encoded.push_back(b.str(name));rows.push_back(b.row(index,"companion_separate",{b.str("companions"),b.container(WOTR_LIST,encoded),b.str("to_region"),b.str(dest)}));}
        }
    };
    std::shared_ptr<const std::vector<cmp_ProperRow>> encoded_owner;
    std::unique_ptr<cmp_EncodedPrefix> encoded_prefix;
    for(auto companion:companions)emit({companion});
    if(companions.size()>1){
        emit(companions);
        // Match ai._SEP_LATTICE_MEMO, including its source key omission:
        // names are sorted in the key, but a cache miss enumerates groups in
        // Fellowship order. A later reordered Fellowship reuses those rows.
        // Do not silently repair that reference behavior in the native port.
        std::set<int64_t> movements;
        for(size_t size=2;size<companions.size();++size)cmp_combinations(companions,size,[&](const std::vector<uint64_t>& group){
            int64_t highest=0;for(auto comp:group)highest=std::max(highest,ht_integer(s,comp,"level"));movements.insert(progress+highest);
        });
        std::vector<std::string> all_names;for(auto comp:companions)all_names.emplace_back(text(s,obs_attr(s,comp,"name")));std::sort(all_names.begin(),all_names.end());
        std::string key;auto part=[&](std::string_view value){key+=std::to_string(value.size());key+=':';key.append(value);};
        part(std::to_string(index));part(start);part(std::to_string(progress));part(std::to_string(all_names.size()));for(const auto& name:all_names)part(name);
        for(auto movement:movements)if(movement>0){
            auto found=destinations.find(movement);
            if(found==destinations.end()){auto reachable=cmp_reachable(s,t,start,movement);reachable.insert(start);found=destinations.emplace(movement,std::move(reachable)).first;}
            part(std::to_string(movement));part(std::to_string(found->second.size()));for(const auto& dest:found->second)part(dest);
        }
        std::shared_ptr<const std::vector<cmp_ProperRow>> block;
        auto& memo=cmp_lattice_memo();{
            std::lock_guard<std::mutex> lock(memo.mutex);const auto found=memo.blocks.find(key);
            if(found!=memo.blocks.end())block=found->second;
            else{
                auto fresh=std::make_shared<std::vector<cmp_ProperRow>>();
                for(size_t size=2;size<companions.size();++size)cmp_combinations(companions,size,[&](const std::vector<uint64_t>& group){
                    int64_t highest=0;std::vector<std::string> names;
                    for(auto comp:group){highest=std::max(highest,ht_integer(s,comp,"level"));names.emplace_back(text(s,obs_attr(s,comp,"name")));}
                    const auto movement=progress+highest;if(movement<=0)return;std::sort(names.begin(),names.end());
                    for(const auto& dest:destinations.at(movement))fresh->push_back({names,dest});
                });
                // Source clears before inserting only when the previous map
                // contains more than 256 entries, not when it reaches 256.
                if(memo.blocks.size()>256){memo.blocks.clear();memo.encoded.clear();memo.encoded_bytes=0;}block=fresh;memo.blocks.emplace(std::move(key),block);
            }
        }

        // Preserve the original raw memo lookup and its historical omissions.
        // Current singleton/full-group rows may change while this exact raw
        // proper-group block is retained; require its complete prefix to match.
        std::shared_ptr<const Snapshot> encoded_hit;
        {
            std::lock_guard<std::mutex> lock(memo.mutex);
            const auto found=memo.encoded.find(block.get());
            if(found!=memo.encoded.end() && found->second.prefix->matches(b.out,rows)){
                encoded_hit=found->second.complete;++memo.encoded_hits;
            }else ++memo.encoded_misses;
        }
        if(encoded_hit)return std::make_unique<Snapshot>(*encoded_hit);
        encoded_owner=block;encoded_prefix=std::make_unique<cmp_EncodedPrefix>(b.out,rows);
        for(const auto& row:*block){std::vector<uint64_t> names;for(const auto& name:row.names)names.push_back(b.str(name));rows.push_back(b.row(index,"companion_separate",{b.str("companions"),b.container(WOTR_LIST,names),b.str("to_region"),b.str(row.destination)}));}
    }

    b.out.root=b.container(WOTR_LIST,rows);
    auto result=std::make_unique<Snapshot>(std::move(b.out));
    if(encoded_owner){
        const size_t charge=encoded_prefix->charge()+result->nodes.size()*sizeof(WotrValue)+
            result->edges.size()*sizeof(uint64_t)+result->bytes.size();
        constexpr size_t entry_limit=8*1024*1024,total_limit=64*1024*1024;
        if(charge<=entry_limit){
            cmp_EncodedBlock cached;cached.owner=encoded_owner;cached.prefix=std::move(encoded_prefix);
            cached.complete=std::make_shared<const Snapshot>(*result);cached.charge=charge;
            auto& memo=cmp_lattice_memo();std::lock_guard<std::mutex> lock(memo.mutex);
            const auto old=memo.encoded.find(encoded_owner.get());
            if(old!=memo.encoded.end()){memo.encoded_bytes-=old->second.charge;memo.encoded.erase(old);}
            if(memo.encoded_bytes+charge>total_limit){memo.encoded.clear();memo.encoded_bytes=0;++memo.encoded_evictions;}
            memo.encoded.emplace(encoded_owner.get(),std::move(cached));memo.encoded_bytes+=charge;
        }
    }
    return result;
}
std::unique_ptr<Snapshot> apply_companion_separation(const Snapshot& source,const Snapshot& t,int64_t index,const std::vector<std::string>& names,std::string_view destination){
    quiet_stable_action_gate(source,t);ring_waiver(source,0);ring_waiver(source,1);fm_payment_die(source,index);
    const auto fellowship=obs_attr(source,source.root,"fellowship");require(!truth(source,obs_attr(source,fellowship,"mordor_entered")),"Cannot separate companions on Mordor Track");
    require(!names.empty(),"Separation needs at least one companion");const auto companions=cmp_separable(source);std::set<std::string> unique;int64_t highest=0;
    for(const auto& name:names){require(unique.insert(name).second,"Separation names must be distinct");uint64_t selected=missing;
        for(auto companion:companions)if(text_is(source,obs_attr(source,companion,"name"),name)){selected=companion;break;}
        require(selected!=missing,"Selected companion is not separable from Fellowship");highest=std::max(highest,ht_integer(source,selected,"level"));
    }
    const auto start=glb_fellowship_key(source,fellowship),key=sanitize_name(destination);const auto steps=ht_integer(source,fellowship,"progress")+highest;
    bool offered=key==start || cmp_reachable(source,t,start,steps).count(key);
    if(!offered){
        // The frozen source's memo can retain a group's old destination rows
        // after a changed level assignment. Its public executor follows that
        // row without checking range again. Accept only an exact current
        // native menu witness; unrelated forged destinations still refuse.
        auto menu=native_companion_separation_actions(source,t,index);
        for(auto row:child_edges(*menu,menu->root)){
            const auto params=child_edges(*menu,row)[2];if(!text_is(*menu,field(*menu,params,"to_region"),destination))continue;
            const auto group=field(*menu,params,"companions",false);if(ctx_none(*menu,group))continue;
            const auto members=child_edges(*menu,group);if(members.size()!=names.size())continue;
            bool match=true;for(size_t i=0;i<names.size();++i)match&=text_is(*menu,members[i],names[i]);if(match){offered=true;break;}
        }
    }
    require(offered,"Companion separation destination unreachable");
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained) || ctx_side(source,retained)==0,"Separation belongs to retained ring actor");
    auto child=action_clone(source);auto& s=*child;rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    rr_set_attr(s,fm_payment_die(s,index),"used",rr_scalar(s,WOTR_BOOL,1));const auto fs=obs_attr(s,s.root,"fellowship"),list=obs_attr(s,fs,"companions");
    for(const auto& name:names){auto remaining=child_edges(s,list);auto found=std::find_if(remaining.begin(),remaining.end(),[&](uint64_t comp){return text_is(s,obs_attr(s,comp,"name"),name);});
        require(found!=remaining.end(),"Separation companion disappeared");const auto comp=*found;remaining.erase(found);set_edges(s,list,remaining);hp_place(s,t,comp,key);
    }
    ft_guide(s);
    cbp_companion_death(s,nullptr,false); // Requirement discards; no casualty/RNG.
    rr_set_attr(s,s.root,"_battle_resolved_attacker",rr_scalar(s,WOTR_NONE));return child;
}
