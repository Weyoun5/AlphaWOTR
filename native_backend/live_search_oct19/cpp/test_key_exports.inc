// Differential-test boundary only; controller consumes rgk::* strings directly.
std::unique_ptr<Snapshot> rgk_test_blob(std::string value){
    auto result=std::make_unique<Snapshot>();result->nodes.push_back({WOTR_BYTES,0,0,0,0,value.size()});result->bytes.assign(value.begin(),value.end());result->root=0;return result;
}
extern "C" {
WOTR_API void* wotr_test_reply_world_key(const void* state){return checked<void*>([&]()->void*{return rgk_test_blob(rgk::world_bytes(snapshot(state))).release();},nullptr);}
WOTR_API void* wotr_test_reply_worlds_key(const void* states){return checked<void*>([&]()->void*{
    const auto& source=snapshot(states);std::vector<std::string> worlds;std::vector<double> weights;
    for(auto row:child_edges(source,source.root)){auto pair=child_edges(source,row);require(pair.size()==2,"Test world pair requires state and weight");Snapshot state(source);state.root=pair[0];worlds.push_back(rgk::world_bytes(state));weights.push_back(source.nodes[pair[1]].real);}
    return rgk_test_blob(rgk::worlds_bytes(worlds,weights)).release();},nullptr);}
WOTR_API void* wotr_test_reply_actions_key(const void* keys,int32_t raw_rows){return checked<void*>([&]()->void*{
    const auto& source=snapshot(keys);std::vector<std::string> parts;sm_Builder builder(source);
    for(auto row:child_edges(source,source.root)){std::string key;if(raw_rows)sc::direct_row_key(source,row,builder,key);else sc::key_bytes(source,row,key);parts.push_back(std::move(key));}
    return rgk_test_blob(rgk::actions_bytes(parts)).release();},nullptr);}
WOTR_API int wotr_test_reply_charge(uint64_t world,uint64_t actions,uint64_t rows,uint64_t* out){return checked<int>([&](){require(out,"Missing charge output");*out=rgk::logical_charge(world,actions,rows);return 0;},-1);}
}

// Isolated experiment boundaries: never copied into the live candidate.
extern "C" {
WOTR_API int wotr_test_actor_menu_reuse_allowed(const void* state,const void* tables,int32_t actor){
    return checked<int>([&](){auto copy=bw::deepcopy(snapshot(state));fm_prepare_menu_context_inplace(*copy,snapshot(tables),actor);
        return rp::ordinary_actor_menu_reuse(*copy,snapshot(tables),actor)?1:0;},-1);
}
WOTR_API void* wotr_test_reply_contract_create(const void* state,const void* tables,int32_t actor){
    return checked<void*>([&]()->void*{
        auto copy=bw::deepcopy(snapshot(state));auto policy=rp::policy(*copy,snapshot(tables),actor);
        auto contract=std::make_unique<rp::Contract>();contract->keys=policy.keys();contract->forced=policy.forced;
        contract->batch_actor_menu_reuse=rp::ordinary_actor_menu_reuse(*copy,snapshot(tables),actor);
        return contract.release();},nullptr);
}
WOTR_API void wotr_test_reply_contract_destroy(void* contract){delete static_cast<rp::Contract*>(contract);}
WOTR_API void* wotr_test_reply_contract_stats(const void* handle){
    return checked<void*>([&]()->void*{
        require(handle,"Missing test reply contract");const auto& c=*static_cast<const rp::Contract*>(handle);sc::Meta m;
        return m.finish(m.dict({{"eligible",m.boolean(c.batch_actor_menu_reuse)},{"cache_seeded",m.boolean(c.cache_seeded)},{"keys_count",m.integer(c.keys.size())},{"forced",m.str(c.forced)},
            {"source_reuses",m.integer(c.source_reuses)},{"source_public_reads",m.integer(c.source_public_reads)},
            {"reuse_hits",m.integer(c.actor_menu_reuse_hits)},{"key_hits",m.integer(c.key_cache_hits)},
            {"key_misses",m.integer(c.key_cache_misses)}})).release();},nullptr);
}
WOTR_API int wotr_test_reply_panel_cached(const void* contract,const void* handle,const void* tables,const void* menu,const void* random,
        int32_t actor,int32_t root_side,int64_t index,void** state_out,void** random_out,void** terminal_out,
        float* pre_obs,float* successor_obs,size_t count,uint64_t* forced_count){
    return checked<int>([&](){
        require(contract&&state_out&&random_out&&terminal_out&&forced_count,"Missing cached-panel test input/output");
        require((pre_obs||successor_obs)?count==rp::observation_size:count==0,"Reply observation size is invalid");
        auto result=rp::evaluate(snapshot(handle),snapshot(tables),snapshot(menu),random,actor,root_side,index,
            pre_obs!=nullptr,successor_obs!=nullptr,static_cast<const rp::Contract*>(contract));
        if(pre_obs)std::copy(result.pre_action_root_obs.begin(),result.pre_action_root_obs.end(),pre_obs);
        if(successor_obs)std::copy(result.successor_root_obs.begin(),result.successor_root_obs.end(),successor_obs);
        *forced_count=result.forced_count;*state_out=result.state.release();*random_out=result.random.release();
        *terminal_out=result.terminal.release();return 0;},-1);
}
}

extern "C" {
WOTR_API void* wotr_test_companion_encoded_stats(){
    return checked<void*>([&]()->void*{auto& c=cmp_lattice_memo();std::lock_guard<std::mutex> lock(c.mutex);sc::Meta m;
        return m.finish(m.dict({{"hits",m.integer(c.encoded_hits)},{"misses",m.integer(c.encoded_misses)},
            {"bytes",m.integer(c.encoded_bytes)},{"entries",m.integer(c.encoded.size())},
            {"raw_blocks",m.integer(c.blocks.size())},{"evictions",m.integer(c.encoded_evictions)}})).release();},nullptr);
}
}

extern "C" {
WOTR_API void* wotr_test_reply_contract_create_seeded(void* handle,const void* state,const void* tables,int32_t actor,const void* allowed){
    return checked<void*>([&]()->void*{
        auto& controller=sc::controller(handle);auto normalized=bw::deepcopy(snapshot(state));
        std::shared_ptr<sc::Menu> permitted;
        if(allowed)permitted=controller.make_menu(std::make_unique<Snapshot>(snapshot(allowed)));
        bool retained_all=false;auto policy=controller.own_public(*normalized,actor,permitted.get(),&retained_all);
        auto contract=std::make_unique<rp::Contract>();contract->source_menu=policy.menu;contract->keys=policy.menu->keys;contract->forced=policy.forced;
        contract->batch_actor_menu_reuse=!allowed && rp::ordinary_actor_menu_reuse(*normalized,snapshot(tables),actor);
        if(allowed){permitted->ensure_keys();contract->allowed_keys=std::unordered_set<std::string>(permitted->keys.begin(),permitted->keys.end());}
        if(retained_all && policy.forced.empty() && contract->batch_actor_menu_reuse)
            rp::seed_policy_cache(*contract,*policy.menu->snap,policy.menu->keys,policy.forced);
        return contract.release();},nullptr);
}
}

extern "C" {
WOTR_API void* wotr_test_reply_policy_lifetime(const void* first_menu,const void* second_menu){
    return checked<void*>([&]()->void*{
        rp::Policy first,second;sc::Meta m;
        auto keys=[&](const rp::Policy& policy){std::vector<uint64_t> xs;for(const auto& key:policy.keys())xs.push_back(m.str(key));return m.list(xs);};
        auto rows=[&](const rp::Policy& policy){std::vector<uint64_t> xs;for(auto row:policy.rows())xs.push_back(m.integer(row));return m.list(xs);};
        uint64_t before_keys=0,before_rows=0;bool shared_graph=false,shared_index=false,replaced=false;
        {
            rp::Contract contract;
            first.menu=std::make_shared<Snapshot>(snapshot(first_menu));first.forced="first_reason";
            rp::policy_keys(first,&contract);
            before_keys=keys(first);before_rows=rows(first);
            shared_graph=first.menu.get()==contract.key_cache->menu.get();
            shared_index=first.index.get()==contract.key_cache->index.get();
            auto held=first;
            second.menu=std::make_shared<Snapshot>(snapshot(second_menu));second.forced="second_reason";
            rp::policy_keys(second,&contract);
            // Identical graphs intentionally share only index; current forced
            // provenance must remain independent of the prior cache entry.
            replaced=!rp::same_menu_graph(*first.menu,*second.menu)?contract.key_cache->index.get()!=first.index.get():true;
            require(held.index.get()==first.index.get()&&held.menu.get()==first.menu.get(),"Held Policy owner changed");
        }
        return m.finish(m.dict({{"first_keys_before",before_keys},{"first_rows_before",before_rows},
            {"first_keys_after",keys(first)},{"first_rows_after",rows(first)},
            {"second_keys",keys(second)},{"second_rows",rows(second)},
            {"first_forced",m.str(first.forced)},{"second_forced",m.str(second.forced)},
            {"first_graph_unchanged",m.boolean(rp::same_menu_graph(*first.menu,snapshot(first_menu)))},
            {"second_graph_unchanged",m.boolean(rp::same_menu_graph(*second.menu,snapshot(second_menu)))},
            {"shared_graph",m.boolean(shared_graph)},{"shared_index",m.boolean(shared_index)},
            {"replaced",m.boolean(replaced)}})).release();},nullptr);
}
WOTR_API int wotr_test_reply_panel_owned_source(const void* contract,const void* handle,const void* tables,const void* menu,const void* random,
        int32_t actor,int32_t root_side,int64_t index,void** state_out,void** random_out,void** terminal_out,
        float* pre_obs,float* successor_obs,size_t count,uint64_t* forced_count){
    return checked<int>([&](){
        require(contract&&state_out&&random_out&&terminal_out&&forced_count,"Missing cached-panel test input/output");
        require((pre_obs||successor_obs)?count==rp::observation_size:count==0,"Reply observation size is invalid");
        const auto& admitted=*static_cast<const rp::Contract*>(contract);
        require(bool(admitted.source_menu),"Missing admitted source owner");
        require(rp::same_menu_graph(*admitted.source_menu->snap,snapshot(menu)),"Owned-source test menu graph differs");
        auto result=rp::evaluate(snapshot(handle),snapshot(tables),*admitted.source_menu->snap,random,actor,root_side,index,
            pre_obs!=nullptr,successor_obs!=nullptr,&admitted);
        if(pre_obs)std::copy(result.pre_action_root_obs.begin(),result.pre_action_root_obs.end(),pre_obs);
        if(successor_obs)std::copy(result.successor_root_obs.begin(),result.successor_root_obs.end(),successor_obs);
        *forced_count=result.forced_count;*state_out=result.state.release();*random_out=result.random.release();
        *terminal_out=result.terminal.release();return 0;},-1);
}
}
