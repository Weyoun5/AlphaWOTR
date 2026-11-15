// NATIVE SEARCH WORLDS (3a, claude_bugscan_2026-10-21/native_sa, 2026-10-27): the hypothetical worlds of
// selfplay_actor_critic.search.GumbelSearcher._search_afterstate / chance.resolve on the native engine.
//
// A world is an rg_Game (the raw driver's PythonEnvironment twin) built as search.clone_environment builds one:
// a copy of the source's state (or a given successor state), its stage / active seat / committed count / first turn /
// turn limit, max_transitions = source.committed + extra_transitions, a private stream (GameRandom(seed).python, built
// by the caller), no episode end, and single-option frames exposed (PythonEnvironment.prepare returns every frame).
// Worlds are handed to Python as rg_Game handles: wotr_raw_next / encode / chance / apply / views / state / menu /
// stage / destroy work on them unchanged.
//   wotr_raw_world_expand   chance.enumerate_chance(src, src's open frame, row, determinize=hidden) -> leaf worlds
//                           (each applied through PythonEnvironment.apply's bookkeeping: committed + 1, after_action)
//   wotr_raw_world_base     chance.expand's `base`: clone(src, GameRandom(0)) + determinize (for the kernel / samples)
//   wotr_raw_world_sample   one sampled clone of `base` with the row of src's open frame applied
//   wotr_raw_world_roll     chance.roll_outcomes: clone + PythonEnvironment._roll
//   wotr_raw_world_marker   chance.battle_marker
//   wotr_raw_world_victory  PythonEnvironment._victory() (end_of_turn=False): the winner when the game ends here
//   wotr_raw_determinized   search.determinize_state on a copy of the source's state (parity tests)
rg_Game* sw_new(const rg_Game& src,const void* random,int64_t extra,std::unique_ptr<Snapshot> state){
    require(random!=nullptr,"world stream is null");require(extra>=0,"world extra transitions must be >= 0");
    auto w=std::make_unique<rg_Game>();
    w->state=state?std::move(state):std::make_unique<Snapshot>(*src.state);
    w->tables=src.tables;w->action_tables=src.action_tables;
    w->random.reset(wotr_random_clone(random));if(!w->random)throw std::invalid_argument(wotr_random_last_error());
    w->stage=src.stage;w->active=src.active;w->committed=src.committed;w->first_turn=src.first_turn;
    w->max_turns=src.max_turns;w->max_transitions=src.committed+extra;
    w->expose_forced=true;w->vocab=src.vocab;w->vocab_index=src.vocab_index;
    return w.release();
}
std::string sw_row_kind(const Snapshot& menu,int64_t row){
    const auto rows=child_edges(menu,menu.root);require(row>=0&&static_cast<uint64_t>(row)<rows.size(),"world row out of range");
    return std::string(rg_row_kind(menu,rows[static_cast<size_t>(row)]));
}
struct sw_Store {std::vector<std::pair<double,std::unique_ptr<rg_Game>>> leaves;std::string why;};
std::unordered_map<const rg_Game*,sw_Store>& sw_stores(){static thread_local std::unordered_map<const rg_Game*,sw_Store> m;return m;}

extern "C" {
// out = {status (0 exact, 1 Unsupported, 2 replay on Python), leaves kept, walks}.  determinize_side < 0: no
// determinize; else search.determinize_state(own_tag = that side, np.random.default_rng(determinize_seed)) on the base.
WOTR_API int wotr_raw_world_expand(void* handle,int64_t index,int64_t max_walks,int64_t determinize_side,uint64_t determinize_seed,
                                   const void* walk_random,int64_t* out){
    return checked<int>([&]()->int{
        auto& src=rg_game(handle);require(out!=nullptr&&walk_random!=nullptr,"world expand: null argument");
        const auto [menu,row]=cl_row(src,index);
        Snapshot base(*src.state);
        if(determinize_side>=0)sd_determinize(base,*src.tables,static_cast<int>(determinize_side),determinize_seed);
        auto& store=sw_stores()[&src];store.leaves.clear();store.why.clear();
        std::vector<std::pair<double,std::unique_ptr<Snapshot>>> leaves;int64_t walks=0;
        out[0]=cl_enumerate_from(base,*src.tables,src.frame_side,*menu,row,max_walks,walk_random,walks,leaves,store.why);
        out[2]=walks;
        if(out[0]==0){
            const std::string kind=sw_row_kind(*menu,row);
            for(auto& leaf:leaves){
                std::unique_ptr<rg_Game> w(sw_new(src,walk_random,4,std::move(leaf.second)));
                w->committed+=1;rg_after_action(*w,src.frame_phase,src.frame_side,kind);
                store.leaves.push_back({leaf.first,std::move(w)});
            }
        }
        out[1]=static_cast<int64_t>(store.leaves.size());return 0;
    },-1);
}
WOTR_API const char* wotr_raw_world_why(void* handle){
    return checked<const char*>([&]()->const char*{return sw_stores()[&rg_game(handle)].why.c_str();},nullptr);
}
// Hand leaf i of the last expand to the caller (it owns the world from now on); probability in *prob.
WOTR_API void* wotr_raw_world_leaf(void* handle,int64_t i,double* prob){
    return checked<void*>([&]()->void*{
        auto& store=sw_stores()[&rg_game(handle)];require(i>=0&&static_cast<size_t>(i)<store.leaves.size(),"world leaf index out of range");
        auto& leaf=store.leaves[static_cast<size_t>(i)];require(leaf.second!=nullptr,"world leaf already taken");
        if(prob)*prob=leaf.first;return leaf.second.release();
    },nullptr);
}
WOTR_API int wotr_raw_world_release(void* handle){
    return checked<int>([&]()->int{sw_stores().erase(&rg_game(handle));return 0;},-1);
}
// chance.expand's base: clone_environment(src, 0) (the given stream) + determinize when determinize_side >= 0.
WOTR_API void* wotr_raw_world_base(void* handle,int64_t determinize_side,uint64_t determinize_seed,const void* random){
    return checked<void*>([&]()->void*{
        auto& src=rg_game(handle);std::unique_ptr<rg_Game> w(sw_new(src,random,4,nullptr));
        if(determinize_side>=0)sd_determinize(*w->state,*src.tables,static_cast<int>(determinize_side),determinize_seed);
        return w.release();
    },nullptr);
}
// One sampled clone: clone_environment(base or src, seed_k) + apply(row of src's open frame).
WOTR_API void* wotr_raw_world_sample(void* handle,void* base,int64_t index,const void* random){
    return checked<void*>([&]()->void*{
        auto& src=rg_game(handle);const auto [menu,row]=cl_row(src,index);
        const rg_Game& from=base?rg_game(base):src;
        std::unique_ptr<rg_Game> w(sw_new(from,random,4,nullptr));
        rg_apply_row(*w,*menu,src.frame_side,src.frame_phase,row);
        return w.release();
    },nullptr);
}
// chance.roll_outcomes: clone_environment(world, seed_k) + PythonEnvironment._roll.
WOTR_API void* wotr_raw_world_roll(void* handle,const void* random){
    return checked<void*>([&]()->void*{
        auto& src=rg_game(handle);require(!src.frame_open,"world roll with an open frame");
        std::unique_ptr<rg_Game> w(sw_new(src,random,4,nullptr));rg_roll(*w);return w.release();
    },nullptr);
}
// chance.battle_marker: out = {waits for its dice (0/1), rounds fought (-1 without a battle)}.
WOTR_API int wotr_raw_world_marker(void* handle,int64_t* out){
    return checked<int>([&]()->int{
        auto& w=rg_game(handle);out[0]=0;out[1]=-1;
        if(w.ended)return 0;
        const auto& s=*w.state;const auto pc=obs_attr(s,s.root,"pending_combat",false);
        if(pc==missing||!truth(s,pc))return 0;
        require(s.nodes[pc].tag==WOTR_DICT,"pending_combat must be a dict");
        const auto phase=field(s,pc,"phase",false);
        static const char* pre[]={"attacker_siege_decision","defender_siege_choice","fp_shine","attacker_combat_card","defender_combat_card","defender_retreat"};
        bool waits=false;if(phase!=missing&&s.nodes[phase].tag==WOTR_STRING)for(const char* p:pre)if(text_is(s,phase,p))waits=true;
        const auto rounds=field(s,pc,"rounds_fought",false);int64_t r=0;
        if(rounds!=missing&&truth(s,rounds)){require(s.nodes[rounds].tag==WOTR_INT||s.nodes[rounds].tag==WOTR_BOOL,"rounds_fought must be an integer");r=s.nodes[rounds].integer;}
        out[0]=waits?1:0;out[1]=r;return 0;
    },-1);
}
// PythonEnvironment._victory(): out = {ended now (0/1), winner (0 fp, 1 sh, -1)}; ends the world like finish().
WOTR_API int wotr_raw_world_victory(void* handle,int64_t* out){
    return checked<int>([&]()->int{auto& w=rg_game(handle);const bool v=rg_victory(w);out[0]=v?1:0;out[1]=v?w.winner:-1;return 0;},-1);
}
// search.clone_environment(src, seed, extra_transitions): a world from src with the given stream.
WOTR_API void* wotr_raw_world_clone(void* handle,const void* random,int64_t extra){
    return checked<void*>([&]()->void*{return sw_new(rg_game(handle),random,extra,nullptr);},nullptr);
}
// search.determinize_state(world.state, ..., rng, ai) with the CALLER's numpy PCG64 generator: pcg = {state_hi,
// state_lo, inc_hi, inc_lo, has_uint32, uinteger} in, the advanced generator out (the afterstate searcher's sampled
// mode draws the determinization from the search sampler itself).
WOTR_API int wotr_raw_world_determinize(void* handle,int64_t side,uint64_t* pcg){
    return checked<int>([&]()->int{
        auto& w=rg_game(handle);require(pcg!=nullptr,"determinize: null generator state");
        np_Pcg64 g;g.state_hi=pcg[0];g.state_lo=pcg[1];g.inc_hi=pcg[2];g.inc_lo=pcg[3];g.has32=pcg[4]!=0;g.u32=static_cast<uint32_t>(pcg[5]);
        sd_determinize_with(*w.state,*w.tables,static_cast<int>(side),g);
        pcg[0]=g.state_hi;pcg[1]=g.state_lo;pcg[2]=g.inc_hi;pcg[3]=g.inc_lo;pcg[4]=g.has32?1:0;pcg[5]=g.u32;return 0;
    },-1);
}
// world.apply(src's open frame, index): PythonEnvironment.apply of src's frame row on the world.
WOTR_API int wotr_raw_world_apply_from(void* handle,void* source,int64_t index){
    return checked<int>([&]()->int{
        auto& w=rg_game(handle);auto& src=rg_game(source);const auto [menu,row]=cl_row(src,index);
        rg_apply_row(w,*menu,src.frame_side,src.frame_phase,row);return 0;
    },-1);
}
// An exact copy of a game / world (state, its own stream, bookkeeping, episode end); no open frame.
WOTR_API void* wotr_raw_world_copy(void* handle){
    return checked<void*>([&]()->void*{
        auto& src=rg_game(handle);std::unique_ptr<rg_Game> w(sw_new(src,src.random.get(),0,nullptr));
        w->max_transitions=src.max_transitions;w->ended=src.ended;w->status=src.status;w->winner=src.winner;w->kind=src.kind;w->reason=src.reason;
        w->forced_applied=src.forced_applied;w->learned=src.learned;w->expose_forced=src.expose_forced;
        return w.release();
    },nullptr);
}
// out = {stage, active, committed, first_turn, max_turns, max_transitions, ended, status, winner, kind}
WOTR_API int wotr_raw_world_info(void* handle,int64_t* out){
    return checked<int>([&]()->int{
        auto& w=rg_game(handle);out[0]=static_cast<int64_t>(w.stage);out[1]=w.active;out[2]=w.committed;out[3]=w.first_turn;
        out[4]=w.max_turns;out[5]=w.max_transitions;out[6]=w.ended?1:0;out[7]=w.status;out[8]=w.winner;out[9]=w.kind;return 0;
    },-1);
}
// search.determinize_state on a copy of the source's state (caller owns the snapshot).
WOTR_API void* wotr_raw_determinized(void* handle,int64_t side,uint64_t seed){
    return checked<void*>([&]()->void*{
        auto& g=rg_game(handle);auto copy=std::make_unique<Snapshot>(*g.state);sd_determinize(*copy,*g.tables,static_cast<int>(side),seed);return copy.release();
    },nullptr);
}
}
