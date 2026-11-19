// extern "C" surface of the native search controller (SHARED_ABI.md section 2).
// Errors are reported through wotr_snapshot_last_error() via the engine's checked<>.
static_assert(sizeof(WotrSearchConfig)==200,"WotrSearchConfig layout drift (SHARED_ABI section 2: 192 bytes + SEARCHFIX fpu_parent at offset 192, checked by controller_binding)");
static_assert(sizeof(WotrSearchRequest)==88,"WotrSearchRequest layout drift (expects 88 bytes)");
static_assert(sizeof(WotrSearchReply)==48,"WotrSearchReply layout drift (expects 48 bytes)");
static_assert(sizeof(WotrSearchRoot)==128,"WotrSearchRoot layout drift (expects 128 bytes)");
static_assert(sizeof(WotrSearchSessionInfo)==32,"WotrSearchSessionInfo layout drift (SHARED_ABI 9.6 expects 32 bytes)");

namespace sc {
Controller& controller(void* p){require(p!=nullptr,"Search controller handle is null");return *static_cast<Controller*>(p);}
const Controller& controller(const void* p){require(p!=nullptr,"Search controller handle is null");return *static_cast<const Controller*>(p);}
const Session& session(const void* p){require(p!=nullptr,"Search session handle is null");return *static_cast<const Session*>(p);}
}

extern "C" {
WOTR_API uint32_t wotr_search_controller_abi_version(void){return 1;}
// GS04-A: census of the pre-roll discard continuation (snapshot.cpp gs04 rule; process-wide, read by the GS04 proofs)
WOTR_API int64_t wotr_search_gs04_preroll_count(void){return gs04_preroll_phase2_count;}
/* R3-V53-FOLLOWONS (additive, ABI stays 1): the widths of this controller's request buffers - floats per
   observation (batch rows) and per action row: 4647/157 with v52 tables, 4745/199 with v53 tables.
   controller_binding reads requests at these widths; a library without this export is v52-only. */
WOTR_API int wotr_search_controller_encoding_widths(const void* ctl,uint64_t* observation,uint64_t* action){
    return checked<int>([&](){require(observation&&action,"encoding widths outputs are required");const auto& c=sc::controller(ctl);
        *observation=static_cast<uint64_t>(c.obs_dim);*action=static_cast<uint64_t>(c.act_dim);return 0;},-1);
}
WOTR_API void* wotr_search_controller_create(const void* root_state,const void* observation_tables,const void* action_tables_v52,
    const void* menu_category_tables,const void* hunt_chance_tables,const void* rng,const WotrSearchConfig* config){
    return checked<void*>([&]()->void*{return new sc::Controller(root_state,observation_tables,action_tables_v52,menu_category_tables,hunt_chance_tables,rng,config);},nullptr);
}
WOTR_API uint32_t wotr_search_controller_own_value_version(){return 1;}
WOTR_API void* wotr_search_controller_create_own_value(const void* s,const void* t,const void* a,const void* c,const void* h,const void* rng,const WotrSearchConfig* cfg,const WotrOwnValueConfig* own){
    return checked<void*>([&]()->void*{require(own,"Own-value configuration is null");return new sc::Controller(s,t,a,c,h,rng,cfg,own);},nullptr);
}
WOTR_API void wotr_search_controller_destroy(void* ctl){delete static_cast<sc::Controller*>(ctl);}
WOTR_API int wotr_search_controller_advance(void* ctl,int32_t* status){
    return checked<int>([&](){require(status,"advance status output is null");*status=sc::controller(ctl).advance();return 0;},-1);
}
WOTR_API int wotr_search_controller_request(const void* ctl,WotrSearchRequest* out){
    return checked<int>([&](){const_cast<sc::Controller&>(sc::controller(ctl)).view_request(out);return 0;},-1);
}
WOTR_API int wotr_search_controller_submit(void* ctl,const WotrSearchReply* reply){
    return checked<int>([&](){sc::controller(ctl).submit(reply);return 0;},-1);
}
WOTR_API int wotr_search_controller_root(const void* ctl,WotrSearchRoot* out){
    return checked<int>([&](){const_cast<sc::Controller&>(sc::controller(ctl)).fill_root(out);return 0;},-1);
}
WOTR_API void* wotr_search_controller_root_menu(const void* ctl){
    return checked<void*>([&]()->void*{return const_cast<sc::Controller&>(sc::controller(ctl)).root_menu().release();},nullptr);
}
WOTR_API void* wotr_search_controller_rng(const void* ctl){
    return checked<void*>([&]()->void*{const auto& c=sc::controller(ctl);c.require_complete();void* out=wotr_random_clone(c.rng.get());require(out,"Cannot copy search RNG");return out;},nullptr);
}
WOTR_API void* wotr_search_controller_counters(const void* ctl){
    return checked<void*>([&]()->void*{return const_cast<sc::Controller&>(sc::controller(ctl)).export_counters().release();},nullptr);
}
/* SHARED_ABI 9.4: status 2 only.  K root children in child order; priors = stored child priors (doubles),
   is_free = action_type in _FREE_ACTION_TYPES.  Buffers owned by the controller. */
WOTR_API int wotr_search_controller_root_priors(const void* ctl,uint64_t* count,const double** priors,const uint8_t** is_free){
    return checked<int>([&](){const_cast<sc::Controller&>(sc::controller(ctl)).view_root_priors(count,priors,is_free);return 0;},-1);
}
WOTR_API int wotr_search_controller_set_root_priors(void* ctl,const double* priors,uint64_t count){
    return checked<int>([&](){sc::controller(ctl).set_root_priors(priors,count);return 0;},-1);
}
/* SHARED_ABI 9.5: COMBAT_DIST (op 4) payload, valid while the op-4 request is pending; owned metadata Snapshot
   {'context','node','action_type','root_branch_row','parent_battle'}. */
WOTR_API void* wotr_search_controller_request_payload(const void* ctl){
    return checked<void*>([&]()->void*{return const_cast<sc::Controller&>(sc::controller(ctl)).view_payload().release();},nullptr);
}
/* reply: Snapshot decoding to {'outcomes','masses','witness','plan_reprs','conf','wctk_side','ctx_repr'}; rejected atomically when stale/malformed. */
WOTR_API int wotr_search_controller_submit_payload(void* ctl,uint64_t request_id,const void* reply){
    return checked<int>([&](){sc::controller(ctl).submit_payload(request_id,reply);return 0;},-1);
}
/* SHARED_ABI 9.3: owned metadata Snapshot -> list of ('hidden_commit', [(card_id, copies)...]) and
   ('card_policy', node_id, side, root_branch_row, phase, battle, rows, probabilities) in occurrence order. */
WOTR_API void* wotr_search_controller_observer_events(const void* ctl){
    return checked<void*>([&]()->void*{return const_cast<sc::Controller&>(sc::controller(ctl)).export_observer().release();},nullptr);
}
WOTR_API void* wotr_search_controller_export(const void* ctl,int32_t with_states){
    return checked<void*>([&]()->void*{require(with_states==0||with_states==1,"with_states must be 0 or 1");return const_cast<sc::Controller&>(sc::controller(ctl)).export_tree(with_states!=0).release();},nullptr);
}
/* SHARED_ABI 9.6: tree reuse session.  detach = _stash_tree over the finished controller (NULL + last_error when the
   child has no state or is a chance node); the session owns the subtree until destroy or until a reusing create consumes it. */
WOTR_API void* wotr_search_session_detach(void* ctl,uint64_t child_index){
    return checked<void*>([&]()->void*{return sc::controller(ctl).detach(child_index).release();},nullptr);
}
WOTR_API void wotr_search_session_destroy(void* session){delete static_cast<sc::Session*>(session);}
WOTR_API int wotr_search_session_info(const void* session,WotrSearchSessionInfo* out){
    return checked<int>([&](){
        require(out&&out->struct_size==sizeof(WotrSearchSessionInfo),"WotrSearchSessionInfo struct_size mismatch");
        const sc::Session& s=sc::session(session);require(s.root!=nullptr&&s.store&&!s.store->adopted,"Search session has no root");const sc::Node& r=*s.root;
        out->side=r.side;out->is_chance=r.is_chance?1:0;out->has_state=r.state?1:0;out->children=r.children.size();out->visit_count=r.visit_count;return 0;},-1);
}
/* Owned Snapshot clone of the retained child.state (caller closes it with wotr_snapshot_destroy); the session is not mutated. */
WOTR_API void* wotr_search_session_state(const void* session){
    return checked<void*>([&]()->void*{const sc::Session& s=sc::session(session);require(s.root!=nullptr&&s.store&&!s.store->adopted&&bool(s.root->state),"Search session root has no state");return new Snapshot(*s.root->state);},nullptr);
}
/* Like create; with session != NULL the controller FIRST performs the source's entry flush of the cached root's opponent
   edge catalog (17361-17366) whether or not it adopts, then adopts when adopt != 0 and the structural conditions hold
   (has_state, !is_chance, children > 0, side == root_side, root not blinded).  *reused reports the outcome; a reused
   session is consumed (owned by the controller), otherwise it stays with the caller. */
WOTR_API void* wotr_search_controller_create_reusing(const void* root_state,const void* observation_tables,const void* action_tables_v52,
    const void* menu_category_tables,const void* hunt_chance_tables,const void* rng,const WotrSearchConfig* config,
    void* session,int32_t adopt,int32_t* reused){
    return checked<void*>([&]()->void*{
        require(reused!=nullptr,"reused output pointer is null");*reused=0;
        std::unique_ptr<sc::Controller> c(new sc::Controller(root_state,observation_tables,action_tables_v52,menu_category_tables,hunt_chance_tables,rng,config));
        if(session!=nullptr){const bool r=c->attach_session(static_cast<sc::Session*>(session),adopt!=0);*reused=r?1:0;}
        return c.release();},nullptr);
}
/* LIVE1104 S03 root plan (WOTR_ROOT_BREADTH native port; Python: wotr_bridge/live/l1104_root_breadth.py).  Additive, ABI
   stays 1; a controller that never calls request_root_plan behaves exactly as before.  request_root_plan: before the first
   advance.  Then advance returns 3 once, after root expansion and before any simulation: root_plan_view (K stored child
   priors + is_free, controller-owned buffers) and root_plan_menu (owned Snapshot, the root_menu rows) are readable, and
   set_root_plan (admitted child indices, forced-visit floor per child, probe, drop) resumes the search on the next
   advance.  root_plan_result (after completion): forced visits served and the dropped child indices in drop order. */
WOTR_API int wotr_search_controller_request_root_plan(void* ctl){
    return checked<int>([&](){sc::controller(ctl).request_root_plan();return 0;},-1);
}
WOTR_API int wotr_search_controller_root_plan_view(const void* ctl,uint64_t* count,const double** priors,const uint8_t** is_free){
    return checked<int>([&](){const_cast<sc::Controller&>(sc::controller(ctl)).view_root_plan(count,priors,is_free);return 0;},-1);
}
WOTR_API void* wotr_search_controller_root_plan_menu(const void* ctl){
    return checked<void*>([&]()->void*{return const_cast<sc::Controller&>(sc::controller(ctl)).root_plan_menu().release();},nullptr);
}
WOTR_API int wotr_search_controller_set_root_plan(void* ctl,const uint64_t* indices,uint64_t count,uint64_t floor,uint64_t probe,double drop){
    return checked<int>([&](){sc::controller(ctl).set_root_plan(indices,count,floor,probe,drop);return 0;},-1);
}
/* TOURNAMENT-DEADLINE (additive, ABI stays 1): ask a running search to stop.  Callable at any time (between
   advances: the worker is parked at its suspension boundary); the search ends at its next simulation boundary once at
   least one simulation ran, then completes as usual (root, root_menu, rng readable; simulations_run < budget). */
WOTR_API int wotr_search_controller_request_stop(void* ctl){
    return checked<int>([&](){sc::controller(ctl).stop_requested.store(true,std::memory_order_relaxed);return 0;},-1);
}
WOTR_API int wotr_search_controller_root_plan_result(const void* ctl,uint64_t* served,uint64_t* dropped_count,const uint64_t** dropped){
    return checked<int>([&](){sc::controller(ctl).root_plan_result(served,dropped_count,dropped);return 0;},-1);
}
}
