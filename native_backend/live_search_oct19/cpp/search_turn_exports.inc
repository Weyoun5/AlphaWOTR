// Owned C ABI for the native search-turn continuation.
void* wotr_search_turn_start(const void* source,const void* tables,const void* random){
    return checked<void*>([&]()->void*{return sd_start(snapshot(source),snapshot(tables),random).release();},nullptr);
}
void* wotr_search_turn_select(const void* handle,int64_t index){
    return checked<void*>([&]()->void*{require(handle,"Missing search-turn driver");return sd_select(*static_cast<const sd_Driver*>(handle),index).release();},nullptr);
}
void wotr_search_turn_destroy(void* handle){delete static_cast<sd_Driver*>(handle);}
void* wotr_search_turn_random(const void* handle){
    return checked<void*>([&]()->void*{require(handle,"Missing search-turn driver");void* result=wotr_random_clone(static_cast<const sd_Driver*>(handle)->random.get());require(result,"Cannot copy search-turn RNG");return result;},nullptr);
}
void* wotr_search_turn_snapshot(const void* handle,int32_t kind){
    return checked<void*>([&]()->void*{
        require(handle,"Missing search-turn driver");const auto& d=*static_cast<const sd_Driver*>(handle);
        if(kind==0)return new Snapshot(*d.state);
        if(kind==1){require(bool(d.raw),"No pending raw search-turn menu");return new Snapshot(*d.raw);}
        if(kind==2){require(bool(d.policy),"No pending policy search-turn menu");return new Snapshot(*d.policy);}
        if(kind==3)return sd_events(d).release();
        if(kind==5){
            require(bool(d.raw)&&bool(d.policy),"No pending combined search-turn menus");
            // Projection clones the raw graph and appends a selected root;
            // export both roots together to retain row and parameter aliases.
            auto menus=std::make_unique<Snapshot>(*d.policy);
            require(d.raw->root<menus->nodes.size(),"Policy no longer embeds its raw descriptor graph");
            menus->root=cp_container(*menus,WOTR_TUPLE,{d.raw->root,d.policy->root});return menus.release();
        }
        require(kind==4,"Unknown search-turn snapshot kind");const int next_actor=d.stage==sd_Stage::Done?sd_turn_start_actor(*d.state):-1;am_Builder b;std::vector<uint64_t> reasons;for(const auto& reason:d.reasons)reasons.push_back(b.str(reason));
        b.out.root=b.container(WOTR_DICT,{b.str("complete"),rr_scalar(b.out,WOTR_BOOL,d.stage==sd_Stage::Done),b.str("side"),b.integer(d.side),
            b.str("stage"),b.integer(static_cast<int>(d.stage)),b.str("next_actor"),next_actor>=0?b.integer(next_actor):rr_scalar(b.out,WOTR_NONE),b.str("forced_reason"),d.forced_reason.empty()?rr_scalar(b.out,WOTR_NONE):b.str(d.forced_reason),
            b.str("reasons"),b.container(WOTR_TUPLE,reasons)});
        return new Snapshot(std::move(b.out));
    },nullptr);
}
