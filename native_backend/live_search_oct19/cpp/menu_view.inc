// Read-only search-menu metadata. No gameplay, pruning or MCTS changes.
struct mv_Builder:sm_Builder {
    explicit mv_Builder(const Snapshot& menu):sm_Builder(menu){}
    uint64_t freeze(uint64_t n,size_t depth=0){
        require(depth<200,"Native menu key is cyclic or nested too deeply");
        if(ctx_none(s,n))return nil;
        const auto tag=s.nodes[n].tag;uint64_t result;
        if(tag==WOTR_DICT){
            std::vector<std::pair<std::string,uint64_t>> ordered;
            for(auto [k,v]:pairs(n))ordered.emplace_back(string(k),v);
            std::stable_sort(ordered.begin(),ordered.end(),[](const auto& a,const auto& b){return a.first<b.first;});
            std::vector<uint64_t> rows;std::string previous;bool has_previous=false;
            for(auto [key,value]:ordered){
                require(!has_previous||previous!=key,"Native menu key has colliding stringified dictionary keys");
                previous=key;has_previous=true;rows.push_back(tup({str(key),freeze(value,depth+1)}));
            }
            result=tup(rows);
        }else if(tag==WOTR_LIST||tag==WOTR_TUPLE||tag==WOTR_SET){
            auto values=items(n);
            if(tag==WOTR_SET){
                std::vector<std::pair<std::string,uint64_t>> ordered;
                for(auto v:values)ordered.emplace_back(ap_repr(ap_freeze(s,v)),v);
                std::stable_sort(ordered.begin(),ordered.end(),[](const auto& a,const auto& b){return a.first<b.first;});
                values.clear();for(auto& entry:ordered)values.push_back(entry.second);
            }
            std::vector<uint64_t> rows;for(auto v:values)rows.push_back(freeze(v,depth+1));result=tup(rows);
        }else{
            require(tag==WOTR_NONE||tag==WOTR_BOOL||tag==WOTR_INT||tag==WOTR_FLOAT||tag==WOTR_BIGINT||tag==WOTR_STRING||tag==WOTR_BYTES||tag==WOTR_ENUM||tag==WOTR_FROZENSET,"Native menu key has unsupported choice type");
            result=raw(n);
        }
        return result;
    }
    uint64_t key(const ap_Row& row){
        const bool choices=field(s,row.params,"choices",false)!=missing;
        // Source removes display/index fields BEFORE freezing: even a malformed
        // ignored payload must not be traversed or rejected by the key query.
        std::vector<std::pair<std::string,uint64_t>> ordered;
        for(auto [key,value]:pairs(row.params)){
            if(text_is(s,key,"card_name")||(choices&&text_is(s,key,"idx")))continue;
            ordered.emplace_back(string(key),value);
        }
        std::stable_sort(ordered.begin(),ordered.end(),[](const auto& a,const auto& b){return a.first<b.first;});
        std::vector<uint64_t> entries;std::string previous;bool has_previous=false;
        for(auto [key,value]:ordered){
            require(!has_previous||previous!=key,"Native menu key has colliding stringified dictionary keys");
            previous=key;has_previous=true;entries.push_back(tup({str(key),freeze(value)}));
        }
        return tup({str(row.kind),integer(row.die),tup(entries)});
    }
};
std::unique_ptr<Snapshot> native_menu_view(const Snapshot& menu,const Snapshot& tables){
    auto category=field(tables,tables.root,"category"),indices=field(tables,tables.root,"category_index"),free=field(tables,tables.root,"free_types");
    typed(tables,category,WOTR_DICT,"Native menu categories must be dictionary");typed(tables,indices,WOTR_DICT,"Native menu category indices must be dictionary");
    typed(tables,free,WOTR_TUPLE,"Native menu free types must be tuple");
    mv_Builder b(menu);std::vector<uint64_t> rows;std::unordered_map<uint64_t,uint64_t> keys;
    for(const auto& row:ap_rows(menu)){
        // Preserve the former c_str lookup's embedded-NUL truncation without
        // creating an owning action-type string for every descriptor.
        const auto cat_edge=string_value_edge(tables,category,row.kind.substr(0,row.kind.find('\0')));
        const auto cat=cat_edge==missing?missing:tables.edges[cat_edge];std::string name=cat==missing?"other":std::string(text(tables,cat));
        auto index=field(tables,indices,name.c_str(),false);if(index==missing)index=field(tables,indices,"other");
        typed(tables,index,WOTR_INT,"Native menu category index must be integer");
        bool is_free=false;for(auto item:child_edges(tables,free))is_free|=text_is(tables,item,row.kind);
        auto found=keys.find(row.node);auto key=found==keys.end()?b.key(row):found->second;keys[row.node]=key;
        rows.push_back(b.tup({b.integer(rows.size()),b.integer(row.node),b.integer(row.die),b.str(row.kind),b.integer(tables.nodes[index].integer),b.boolean(is_free),key}));
    }
    b.out.root=b.container(WOTR_LIST,rows);return ac_compact(b.out);
}
uint64_t mv_row_node(const Snapshot& menu,int64_t index){
    const auto& list=typed(menu,menu.root,WOTR_LIST,"Native menu input must be descriptor list");
    require(index>=0&&static_cast<uint64_t>(index)<list.count,"Native menu row index outside bounds");
    const auto node=menu.edges[list.offset+index];const auto& row=typed(menu,node,WOTR_TUPLE,"Native menu row must be descriptor tuple");
    require(row.count==3,"Native menu row must have three fields");
    typed(menu,menu.edges[row.offset],WOTR_INT,"Native menu die index must be integer");
    typed(menu,menu.edges[row.offset+1],WOTR_STRING,"Native menu action type must be string");
    typed(menu,menu.edges[row.offset+2],WOTR_DICT,"Native menu params must be dictionary");
    return node;
}
std::unique_ptr<Snapshot> native_menu_row(const Snapshot& menu,int64_t index){
    mv_Builder b(menu);b.out.root=b.raw(mv_row_node(menu,index));return ac_compact(b.out);
}
std::unique_ptr<Snapshot> native_menu_rows(const Snapshot& menu,const Snapshot& indices){
    typed(menu,menu.root,WOTR_LIST,"Native menu input must be descriptor list");
    typed(indices,indices.root,WOTR_LIST,"Native menu indices must be list");mv_Builder b(menu);std::vector<uint64_t> rows;
    for(auto i:child_edges(indices,indices.root)){
        auto value=typed(indices,i,WOTR_INT,"Native menu index must be integer").integer;
        rows.push_back(b.raw(mv_row_node(menu,value)));
    }
    b.out.root=b.container(WOTR_LIST,rows);return ac_compact(b.out);
}
std::unique_ptr<Snapshot> native_menu_alias_groups(const Snapshot& menu){
    auto rows=ap_rows(menu);std::vector<size_t> parent(rows.size());for(size_t i=0;i<parent.size();++i)parent[i]=i;
    auto root=[&](size_t i){while(parent[i]!=i)i=parent[i];return i;};
    std::unordered_map<uint64_t,size_t> owners;
    for(size_t i=0;i<rows.size();++i){
        std::set<uint64_t> seen;std::vector<uint64_t> pending={rows[i].node};
        while(!pending.empty()){
            auto n=pending.back();pending.pop_back();auto tag=menu.nodes[n].tag;
            if(tag<WOTR_LIST||tag>WOTR_OBJECT||!seen.insert(n).second)continue;
            auto found=owners.find(n);
            if(found==owners.end())owners[n]=i;
            else{auto a=root(i),b=root(found->second);parent[std::max(a,b)]=std::min(a,b);}
            for(auto c:child_edges(menu,n))pending.push_back(c);
        }
    }
    std::map<size_t,std::vector<size_t>> groups;for(size_t i=0;i<rows.size();++i)groups[root(i)].push_back(i);
    mv_Builder b(menu);std::vector<uint64_t> result;
    for(auto& group:groups){std::vector<uint64_t> members;for(auto i:group.second)members.push_back(b.integer(i));result.push_back(b.tup(members));}
    b.out.root=b.container(WOTR_LIST,result);return ac_compact(b.out);
}
