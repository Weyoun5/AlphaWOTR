// Included after the native public reply-panel component, outside namespace sc.
namespace sc {
// These local sets borrow completed Menu::keys strings. Neither key vector is
// mutated or compacted until the sets are destroyed; no borrowed pointer escapes.
// A contiguous open-addressing table replaces one heap node per distinct key.
struct OwnBorrowedKeySet{
    std::vector<const std::string*> slots;
    explicit OwnBorrowedKeySet(size_t expected){
        if(!expected)return;
        const auto maximum=std::numeric_limits<size_t>::max();
        if(expected>maximum/2)throw std::length_error("Own key set capacity overflow");
        const auto target=expected*2;size_t capacity=2;
        while(capacity<target){if(capacity>maximum/2)throw std::length_error("Own key set capacity overflow");capacity*=2;}
        slots.resize(capacity,nullptr);
    }
    bool contains(const std::string& key)const{
        if(slots.empty())return false;
        const auto mask=slots.size()-1;size_t at=std::hash<std::string_view>{}(key)&mask;
        while(const auto* stored=slots[at]){if(*stored==key)return true;at=(at+1)&mask;}
        return false;
    }
    bool insert(const std::string& key){
        const auto mask=slots.size()-1;size_t at=std::hash<std::string_view>{}(key)&mask;
        while(const auto* stored=slots[at]){if(*stored==key)return false;at=(at+1)&mask;}
        slots[at]=&key;return true;
    }
};
bool own_retained_rows(const std::vector<std::string>& keys,const std::vector<std::string>* allowed,std::vector<size_t>& rows){
    OwnBorrowedKeySet seen(keys.size()),keep(allowed?allowed->size():0);
    if(allowed)for(const auto& key:*allowed)keep.insert(key);
    bool retained_all=true;
    for(size_t i=0;i<keys.size();++i){
        const auto& key=keys[i];
        if((!allowed||keep.contains(key))&&seen.insert(key)){
            if(!retained_all)rows.push_back(i);
        }else if(retained_all){
            retained_all=false;rows.reserve(keys.size());
            for(size_t j=0;j<i;++j)rows.push_back(j);
        }
    }
    return retained_all;
}
Policy Controller::own_public(Snapshot& state,int actor,const Menu* allowed,bool* retained_all){
    auto policy=make_policy(state,actor);auto& menu=*policy.menu;
    menu.ensure_keys();if(allowed)allowed->ensure_keys();
    std::vector<size_t> rows;
    const bool all=own_retained_rows(menu.keys,allowed?&allowed->keys:nullptr,rows);
    if(retained_all)*retained_all=all;
    if(all)return policy;
    auto compact=compact_rows_menu(menu,rows,true);
    return Policy{std::const_pointer_cast<Menu>(compact),policy.forced,policy.reasons};
}

std::optional<std::string> Controller::own_choose(const Snapshot& state,int actor,Node* node,const Menu* allowed,const Policy* admitted){
    require(own&&actor!=root_side,"Own-value reply requires the opposing actor");
    auto representative=OwnReplies::worlds(state,*T,actor,1,own->world_seed);
    own->stats["belief_key_worlds"]+=1;
    auto menu_state=bw::deepcopy(*representative[0].state);

    // Sampling and representative identity remain unchanged. An actual actor
    // policy may be reused only after its normalizer and both ordinary guards.
    bool reuse_admitted=admitted && !allowed && admitted->forced.empty() &&
        rp::ordinary_actor_menu_reuse(state,*T,actor);
    if(reuse_admitted){
        fm_prepare_menu_context_inplace(*menu_state,*T,actor);
        reuse_admitted=rp::ordinary_actor_menu_reuse(*menu_state,*T,actor);
    }
    auto policy=reuse_admitted?*admitted:own_public(*menu_state,actor,allowed);auto menu=policy.menu;
    if(reuse_admitted)own->stats["reply_admitted_menu_reuses"]+=1;
    if(!menu->size())return std::nullopt;
    if(!policy.forced.empty()){
        require(menu->size()==1,"Fixed policy requires exactly one reply");return menu->key(0);
    }
    if(node&&node->own_reply_token==0){
        require(own->next_node_token!=0,"Own-value node token exceeds uint64");
        node->own_reply_token=own->next_node_token++;
    }
    const auto token=node?node->own_reply_token:0;
    const auto world_wire=rgk::worlds_bytes({rgk::world_bytes(*representative[0].state)},{1.0});
    const auto action_wire=rgk::actions_bytes(menu->keys);
    std::string key;append_u64(key,token);append_u64(key,uint64_t(actor));
    append_u64(key,world_wire.size());key+=world_wire;key+=action_wire;
    auto entry=std::find_if(own->cache.begin(),own->cache.end(),[&](const OwnReplies::Entry& e){return e.key==key;});
    OwnReplies::SelectorPtr temporary(nullptr,wotr_reply_selector_destroy);void* selector=nullptr;
    if(entry!=own->cache.end()){
        selector=entry->selector.get();own->cache.splice(own->cache.end(),own->cache,entry);
        own->stats["reply_cache_hits"]+=1;
    }else{
        auto seed=OwnReplies::bits<4>(own->stream.get());std::vector<uint64_t> ids(menu->size());
        std::iota(ids.begin(),ids.end(),0);
        temporary.reset(wotr_reply_selector_create(ids.data(),ids.size(),seed.data(),seed.size(),1.25,2.0));
        require(bool(temporary),wotr_random_last_error());selector=temporary.get();
        const auto charge=rgk::logical_charge(world_wire.size(),action_wire.size(),menu->size());
        if(charge<=8*1024*1024){
            while(!own->cache.empty()&&(own->cache.size()>=64||own->cache_bytes+charge>8*1024*1024)){
                own->cache_bytes-=own->cache.front().charge;own->cache.pop_front();
            }
            OwnReplies::Entry fresh;fresh.key=std::move(key);fresh.charge=charge;fresh.selector=std::move(temporary);
            own->cache.push_back(std::move(fresh));own->cache_bytes+=charge;
        }
        own->stats["reply_cache_misses"]+=1;
    }
    rp::Contract contract;contract.source_menu=menu;contract.keys=menu->keys;contract.forced=policy.forced;
    contract.batch_actor_menu_reuse=!allowed && rp::ordinary_actor_menu_reuse(*menu_state,*T,actor);
    if(reuse_admitted && contract.batch_actor_menu_reuse){
        rp::seed_policy_cache(contract,*menu->snap,menu->keys,policy.forced);
        own->stats["reply_admitted_cache_seeds"]+=1;
    }
    if(allowed){allowed->ensure_keys();contract.allowed_keys=std::unordered_set<std::string>(allowed->keys.begin(),allowed->keys.end());}
    for(uint64_t p=0;p<own->panels;++p){
        uint64_t selected=0;require(wotr_reply_selector_select(selector,&selected)==0,wotr_random_last_error());
        try{
            auto sample_seed=OwnReplies::bits<4>(own->stream.get());
            auto worlds=OwnReplies::worlds(state,*T,actor,own->world_count,sample_seed);
            own->stats["belief_worlds_sampled"]+=int64_t(worlds.size());
            auto transition_seed=OwnReplies::bits<4>(own->stream.get());
            auto transitions=OwnReplies::seeded(transition_seed.data(),transition_seed.size());
            std::vector<double> values(worlds.size(),0.0),weights(worlds.size(),1.0/double(worlds.size()));
            RequestData request;request.offsets={0};
            std::vector<size_t> evaluation_rows;
            size_t world_index=0;
            for(auto& world:worlds){
                auto seed=OwnReplies::bits<4>(transitions.get());auto rng=OwnReplies::seeded(seed.data(),seed.size());
                auto result=rp::evaluate(*world.state,*T,*menu->snap,rng.get(),actor,root_side,
                                        int64_t(selected),false,true,&contract);
                if(result.forced_count)own->stats["forced_reply_transitions"]+=int64_t(result.forced_count);
                own->stats["reply_value_evaluations"]+=1;
                if(result.terminal){
                    const auto winner=field(*result.terminal,result.terminal->root,"victory");
                    const auto kind=field(*result.terminal,result.terminal->root,"victory_type");
                    require(text_is(*result.terminal,winner,"free_peoples")||text_is(*result.terminal,winner,"shadow"),"Invalid panel terminal winner");
                    values[world_index]=search_terminal(text_is(*result.terminal,winner,"free_peoples")?0:1,
                                                      victory_kind(text(*result.terminal,kind)));
                }else{
                    evaluation_rows.push_back(world_index);
                    request.leaf_ids.push_back(uint64_t(world_index));
                    request.obs.insert(request.obs.end(),result.successor_root_obs.begin(),result.successor_root_obs.end());
                    request.offsets.push_back(0);
                }
                ++world_index;
            }
            if(!evaluation_rows.empty()){
                auto reply=await_reply(std::move(request),OP_EVAL,root_side,PURPOSE_EVAL);
                require(reply.values.size()==evaluation_rows.size()&&reply.conf.size()==evaluation_rows.size(),"Own-value panel batch reply shape mismatch");
                for(size_t j=0;j<evaluation_rows.size();++j){
                    const double saved_conf=last_conf,saved_phi=last_phi;
                    try{
                        const double shaping=(root_side==0?1.0:-1.0)*profile_shaping();
                        if(conf_active)last_phi=shaping;
                        values[evaluation_rows[j]]=bound_value(conf_shrink(reply.conf[j],reply.values[j])+shaping);
                    }catch(...){last_conf=saved_conf;last_phi=saved_phi;throw;}
                    last_conf=saved_conf;last_phi=saved_phi;
                }
            }
            double ignored=0.0;
            require(wotr_reply_selector_complete(selector,selected,values.data(),values.size(),weights.data(),weights.size(),&ignored)==0,wotr_random_last_error());
        }catch(...){wotr_reply_selector_cancel(selector);throw;}
        own->stats["reply_panels"]+=1;
    }
    own->stats["reply_actor_menu_reuse_hits"]+=int64_t(contract.actor_menu_reuse_hits);
    own->stats["reply_descriptor_source_reuses"]+=int64_t(contract.source_reuses);
    own->stats["reply_descriptor_public_reads"]+=int64_t(contract.source_public_reads);
    own->stats["reply_menu_key_cache_hits"]+=int64_t(contract.key_cache_hits);
    own->stats["reply_menu_key_cache_misses"]+=int64_t(contract.key_cache_misses);
    own->stats["reply_cache_estimated_bytes"]=int64_t(own->cache_bytes);own->stats["reply_choices"]+=1;
    uint64_t best=0;require(wotr_reply_selector_best(selector,&best)==0,wotr_random_last_error());
    return menu->key(size_t(best));
}

Node* Controller::own_child(Node* node){
    require(bool(overlay),"Own-value replies require the outer hidden-hand sampler");overlay_patch(*node->state,false);
    bool retained_all=false;auto actual=own_public(*node->state,node->side,nullptr,&retained_all);auto menu=actual.menu;
    if(!menu->size()){
        auto& s=*node->state;
        if(pending_side(s,-1)>=0||!attr_none(s,"_ring_free_side")||!attr_none(s,"pending_army_move_die")||!attr_none(s,"pending_muster_die"))
            fail("Empty sampled menu during required decision");
        Node* end=nullptr;for(Node* c:node->children)if(c->action.has()&&c->action.kind()=="end_turn"){end=c;break;}
        if(end&&make_policy(s,1-node->side).menu->size()==0)return end;
        own->stats["empty_opponent_menus"]+=1;return nullptr;
    }
    const auto key=own_choose(*node->state,node->side,node,nullptr,retained_all?&actual:nullptr);
    require(bool(key),"Belief reply is absent from actual actor menu");
    size_t row=0;while(row<menu->size()&&menu->key(row)!=*key)++row;
    require(row<menu->size(),"Belief reply is absent from actual actor menu");
    auto chosen=compact_rows_menu(*menu,{row},true);size_t index=0;
    return edge_choose(node,*chosen,{1.0},true,index);
}

int64_t Controller::own_turn_action(const Snapshot& state,int actor,const Snapshot& projected,const std::vector<int64_t>& original){
    auto menu=make_menu(std::make_unique<Snapshot>(projected));menu->ensure_keys();
    auto key=own_choose(state,actor,nullptr,menu.get());own->stats["turn_reply_choices"]+=1;
    if(!key)return -1;
    // Python's turn mapping is a dict comprehension, hence last duplicate wins.
    for(size_t i=menu->size();i>0;--i)if(menu->key(i-1)==*key)return original[i-1];
    fail("Belief turn reply is absent from the projected menu");
}
} // namespace sc
