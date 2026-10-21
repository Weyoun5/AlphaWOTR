// Pure public-reply transitions. Included after the existing sc::key_bytes
// serializer; no Controller instance, model, search RNG or Python callback.
namespace rp {
constexpr size_t observation_size=4647;
using Random=std::unique_ptr<void,decltype(&wotr_random_destroy)>;
// Payloads become immutable before publication. A returned Policy owns both
// graph and index independently of the Contract's replaceable cache entry.
struct PolicyIndex {
    std::vector<int64_t> rows;
    std::vector<std::string> keys;
};
struct Policy {
    std::shared_ptr<const Snapshot> menu;
    std::shared_ptr<const PolicyIndex> index;
    std::string forced;
    const std::vector<int64_t>& rows()const{return index->rows;}
    const std::vector<std::string>& keys()const{return index->keys;}
};
struct Contract {
    std::vector<std::string> keys;
    std::string forced;
    std::optional<std::unordered_set<std::string>> allowed_keys;
    // One temporary immutable entry per reply choice. Ordinary sampled worlds
    // may share this admitted policy; all actor/allowed/forced checks remain.
    mutable std::shared_ptr<const Policy> key_cache;
    // Internal search callers retain an already admitted Menu, including its
    // stable row metadata and completed keys. Public arbitrary graphs never
    // acquire this trust merely through equal bytes or a reused raw address.
    std::shared_ptr<const sc::Menu> source_menu;
    mutable uint64_t source_reuses=0,source_public_reads=0;
    mutable uint64_t key_cache_hits=0,key_cache_misses=0;
    bool batch_actor_menu_reuse=false;
    mutable uint64_t actor_menu_reuse_hits=0;
    bool cache_seeded=false;
};

// Internal callers provide a complete, projected, first-row-unique policy
// whose descriptors and keys were already admitted on this batch's actor.
// Retain the exact descriptor representation; no synthetic key or raw ID
// is reused across an unrelated graph or across an action transition.
void seed_policy_cache(Contract& contract,const Snapshot& menu,const std::vector<std::string>& keys,const std::string& forced){
    require(contract.batch_actor_menu_reuse && !contract.allowed_keys && forced.empty(),"Invalid admitted actor policy seed");
    const auto& root=typed(menu,menu.root,WOTR_LIST,"Admitted actor menu must be a list");
    require(root.count==keys.size(),"Admitted actor policy key count mismatch");
    auto cache=std::make_shared<Policy>();cache->menu=std::make_shared<Snapshot>(menu);
    auto index=std::make_shared<PolicyIndex>();index->keys=keys;index->rows.resize(keys.size());
    std::iota(index->rows.begin(),index->rows.end(),int64_t(0));
    cache->index=std::move(index);cache->forced=forced;
    contract.key_cache=std::move(cache);contract.cache_seeded=true;
}
struct Result {
    std::unique_ptr<Snapshot> state;
    Random random{nullptr,wotr_random_destroy};
    std::unique_ptr<Snapshot> terminal;
    std::vector<float> pre_action_root_obs,successor_root_obs;
    uint64_t forced_count=0;
};
std::string action_key(const Snapshot& menu,const ap_Row& row){
    // Same validated byte serializer as Controller::Menu::ensure_keys: avoid
    // constructing a second, frozen graph for every row of every belief menu.
    sm_Builder builder(menu);std::string bytes;
    sc::direct_row_key(menu,row.node,builder,bytes);return bytes;
}
std::string action_key(const Snapshot& menu,int64_t index){
    const auto rows=ap_rows(menu);
    require(index>=0&&static_cast<size_t>(index)<rows.size(),"Reply descriptor index is outside its menu");
    return action_key(menu,rows[static_cast<size_t>(index)]);
}
bool same_menu_graph(const Snapshot& a,const Snapshot& b){
    // Byte equality is deliberately stronger than semantic action equality.
    // It includes unused nodes, alias edges, float bits, and descriptor order.
    // Any representation difference falls back to the ordinary serializer.
    return a.root==b.root && a.nodes.size()==b.nodes.size() &&
        a.edges==b.edges && a.bytes==b.bytes &&
        (a.nodes.empty() || std::memcmp(a.nodes.data(),b.nodes.data(),a.nodes.size()*sizeof(WotrValue))==0);
}
// Local deduplication table: slots hold indices into the new PolicyIndex's
// owning key vector. No serialized key is copied into a separate set node.
// The vector is reserved before insertion and the table dies before publication.
struct PolicyKeySlots{
    static constexpr size_t empty=std::numeric_limits<size_t>::max();
    std::vector<size_t> slots;
    explicit PolicyKeySlots(size_t expected){
        if(!expected)return;
        if(expected>empty/2)throw std::length_error("Reply key set capacity overflow");
        const auto target=expected*2;size_t capacity=2;
        while(capacity<target){if(capacity>empty/2)throw std::length_error("Reply key set capacity overflow");capacity*=2;}
        slots.resize(capacity,empty);
    }
    size_t find_slot(const std::vector<std::string>& keys,const std::string& key)const{
        const auto mask=slots.size()-1;size_t at=std::hash<std::string_view>{}(key)&mask;
        while(slots[at]!=empty){if(keys[slots[at]]==key)break;at=(at+1)&mask;}
        return at;
    }
};
void policy_keys(Policy& out,const Contract* contract=nullptr){
    if(contract && contract->key_cache && same_menu_graph(*out.menu,*contract->key_cache->menu)){
        out.index=contract->key_cache->index;
        ++contract->key_cache_hits;return;
    }
    auto index=std::make_shared<PolicyIndex>();
    const auto rows=ap_rows(*out.menu);
    index->keys.reserve(rows.size());index->rows.reserve(rows.size());
    PolicyKeySlots seen(rows.size());
    sm_Builder builder(*out.menu);
    for(size_t i=0;i<rows.size();++i){
        std::string key;sc::direct_row_key(*out.menu,rows[i].node,builder,key);
        // reply_evaluation.menu uses setdefault: FIRST descriptor wins.
        const auto slot=seen.find_slot(index->keys,key);
        if(seen.slots[slot]==PolicyKeySlots::empty){
            index->rows.push_back(static_cast<int64_t>(i));index->keys.push_back(std::move(key));
            // Publish an index only after its owning string is in place.
            seen.slots[slot]=index->keys.size()-1;
        }
    }
    out.index=std::move(index);
    if(contract){
        contract->key_cache=std::make_shared<Policy>(out);++contract->key_cache_misses;
    }
}
// Only the internal post-action settlement caller may request this probe.
// All pending decisions and pre-action phases retain the complete policy path.
// Ordinary turn menus cannot produce either kind of forced projection:
// combat victims require pending_combat, and fixed card choices require
// pending_card_option/card_followup. Menu normalization still runs first.
bool ordinary_forced_probe(const Snapshot& state,const Snapshot& tables,int actor){
    for(const char* field_name:{"pending_combat","pending_guide_change","pending_hunt",
            "pending_discard","pending_card_option","pending_card_followup",
            "pending_army_move_die","pending_muster_die"})
        if(!ctx_none(state,obs_attr(state,state.root,field_name,false)))return false;
    for(const char* field_name:{"_declare_phase","_eye_allocation_phase","_reveal_place_due"})
        if(truth(state,obs_attr(state,state.root,field_name,false)))return false;
    if(!truth(state,obs_attr(state,state.root,"dice_rolled",false)))return false;
    // Preserve the existing ordinary-menu support gate. This is an internal
    // successor of an admitted action, not a second public admission point.
    if(truth(state,obs_attr(state,state.root,"besieged_strongholds",false)))
        quiet_stable_action_gate(state,tables);
    else quiet_action_gate(state,tables);
    return true;
}

// Sufficient scope for reusing an initial actor menu within ONE sampling batch.
// Every world is sampled from the same source for the same actor. The sampler
// preserves that actor's hand/order, all public state, and deck/hand counts.
// Deck and hunt order affect execution only. Combat commitments are excluded.
// The only ordinary cross-hand menu dependency is whether the opponent can act
// for a pass: any unused non-Eye ensures this via the no-action skip fallback.
// All-Eye and zero-unused opponents deliberately retain complete regeneration.
bool ordinary_actor_menu_reuse(const Snapshot& state,const Snapshot& tables,int actor){
    if(!ordinary_forced_probe(state,tables,actor))return false;
    for(auto die:child_edges(state,pa_pool(state,1-actor))){
        if(truth(state,obs_attr(state,die,"used")))continue;
        const auto face=obs_attr(state,die,"current");if(ctx_none(state,face))continue;
        if(rr_face_index(enum_member(state,face,"game_state.ActionDieFace"))!=5)return true;
    }
    return false;
}
Policy policy(Snapshot& state,const Snapshot& tables,int actor,bool forced_only=false,const Contract* contract=nullptr){
    fm_prepare_menu_context_inplace(state,tables,actor);
    if(forced_only&&contract&&ordinary_forced_probe(state,tables,actor))return Policy{};
    if(!forced_only && contract && contract->batch_actor_menu_reuse && contract->key_cache &&
            ordinary_actor_menu_reuse(state,tables,actor)){
        ++contract->actor_menu_reuse_hits;return *contract->key_cache;
    }
    auto raw=native_full_actions(state,tables,actor);
    auto projected=project_native_ai_actions(state,*raw,actor);
    Policy out;out.menu=std::move(projected.actions);out.forced=std::move(projected.forced_reason);
    // Settlement consumes only forced_reason for an ordinary leaf. Preserve
    // real menu generation/normalization, but leave its unused keys unbuilt.
    // Forced menus still receive complete first-row dedup and cardinality checks.
    if(forced_only&&out.forced.empty())return out;
    policy_keys(out,forced_only?nullptr:contract);
    return out;
}
std::unique_ptr<Snapshot> terminal(const Snapshot& state,const Snapshot& tables){
    const auto marker=obs_attr(state,state.root,"_search_endturn_victory",false);
    if(!ctx_none(state,marker)){
        typed(state,marker,WOTR_DICT,"Reply victory override must be a dictionary");
        sm_Builder copy(state);copy.out.root=copy.raw(marker);return ac_compact(copy.out);
    }
    const auto result=check_victory(state,tables);
    if(!terminal_should_end(result,false))return nullptr;
    const std::string winner=result.winner==0?"free_peoples":"shadow";
    const std::string kind=result.kind==2?"ring_destroyed":"ring_corruption";
    std::string message;
    if(result.kind==2)message="The Ring is destroyed! Free Peoples win!";
    else{
        const auto fellowship=obs_attr(state,state.root,"fellowship");
        const auto corruption=obs_attr(state,fellowship,"corruption");std::set<uint64_t> active;
        message="Ring-bearers corrupted! ("+bw::json_value(state,corruption,active)+" corruption)";
    }
    am_Builder b;b.out.root=b.container(WOTR_DICT,{
        b.str("victory"),b.str(winner),b.str("victory_type"),b.str(kind),
        b.str("message"),b.str(message),b.str("free_peoples_vp"),b.integer(result.fp_vp),
        b.str("shadow_vp"),b.integer(result.sh_vp)});
    return std::make_unique<Snapshot>(std::move(b.out));
}
void apply(Result& result,const Snapshot& tables,const Policy& policy,int actor,int64_t index){
    // Only policy() outputs reach here: every row was admitted either by
    // policy_keys/ap_rows or the retained internal Menu before cache seeding.
    // The execution dispatch still validates the complete selected tuple.
    const auto& menu=*policy.menu;const auto& rows=menu.nodes[menu.root];
    require(index>=0&&static_cast<uint64_t>(index)<rows.count,"Reply action index is invalid");
    const auto& tuple=menu.nodes[menu.edges[rows.offset+static_cast<uint64_t>(index)]];
    const auto kind=text(menu,menu.edges[tuple.offset+1]);
    require(kind!="end_turn","Synthetic end_turn is not a public reply action");
    auto action=apply_native_random_descriptor(*result.state,tables,menu,result.random.get(),actor,index);
    auto resumed=sr_resume_preaction(*result.state,*action.state,tables,action.random.get(),kind);
    result.state=std::move(resumed.state);result.random=std::move(resumed.random);
}
Result evaluate(const Snapshot& source,const Snapshot& tables,const Snapshot& descriptor_menu,
                const void* initial_rng,int actor,int root_side,int64_t index,
                bool pre_obs,bool successor_obs,const Contract* contract=nullptr){
    require((actor==0||actor==1)&&(root_side==0||root_side==1)&&actor!=root_side,
            "Reply actor and root must be opposing valid sides");
    require(initial_rng,"Reply transition requires an owned Python RNG");
    std::string public_requested;std::string_view requested;
    if(contract && contract->source_menu && contract->source_menu->snap.get()==&descriptor_menu){
        ++contract->source_reuses;
        const auto& admitted=*contract->source_menu;
        require(index>=0&&static_cast<size_t>(index)<admitted.size(),"Reply descriptor index is invalid");
        require(admitted.rows[static_cast<size_t>(index)].kind!="end_turn","Synthetic end_turn is not a public reply action");
        requested=admitted.key(static_cast<size_t>(index));
    }else{
        if(contract)++contract->source_public_reads;
        // Preserve public admission of EVERY row, including unrelated malformed
        // descriptors and its error order before index/end-turn/key validation.
        const auto source_rows=ap_rows(descriptor_menu);
        require(index>=0&&static_cast<size_t>(index)<source_rows.size(),"Reply descriptor index is invalid");
        require(source_rows[static_cast<size_t>(index)].kind!="end_turn","Synthetic end_turn is not a public reply action");
        public_requested=action_key(descriptor_menu,source_rows[static_cast<size_t>(index)]);
        requested=public_requested;
    }
    Result result;result.state=bw::deepcopy(source);result.random.reset(wotr_random_clone(initial_rng));
    require(bool(result.random),"Could not clone reply transition RNG");
    auto menu=policy(*result.state,tables,actor,false,contract);
    if(contract){
        const auto* keys=&menu.keys();std::vector<std::string> filtered;
        if(contract->allowed_keys){
            for(const auto& key:menu.keys())if(contract->allowed_keys->count(key))filtered.push_back(key);
            keys=&filtered;
        }
        require(*keys==contract->keys&&menu.forced==contract->forced,"Fresh belief sample changed actor-visible legal choices");
    }
    auto selected=std::find(menu.keys().begin(),menu.keys().end(),requested);
    require(selected!=menu.keys().end(),"Reply is not legal in this actor information set");
    if(pre_obs){result.pre_action_root_obs.resize(observation_size);encode_state(*result.state,tables,root_side,result.pre_action_root_obs.data(),observation_size);}
    apply(result,tables,menu,actor,menu.rows()[static_cast<size_t>(selected-menu.keys().begin())]);
    int previous_actor=actor;bool settled=false;
    for(size_t step=0;step<128;++step){
        result.terminal=terminal(*result.state,tables);
        if(result.terminal){settled=true;break;}
        const int next=native_next_actor(*result.state,previous_actor);
        if(next<0){settled=true;break;}
        // Standalone public panels retain their complete admission behavior.
        auto next_menu=policy(*result.state,tables,next,true,contract);
        if(next_menu.forced.empty()){settled=true;break;}
        require(next_menu.rows().size()==1,"Fixed policy requires exactly one action");
        apply(result,tables,next_menu,next,next_menu.rows()[0]);
        previous_actor=next;++result.forced_count;
    }
    require(settled,"Reply fixed-policy closure exceeded 128 transitions");
    // root_value checks victory after the final public-menu normalization.
    result.terminal=terminal(*result.state,tables);
    if(successor_obs&&!result.terminal){result.successor_root_obs.resize(observation_size);encode_state(*result.state,tables,root_side,result.successor_root_obs.data(),observation_size);}
    return result;
}
} // namespace rp
