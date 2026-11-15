// NATIVE RAW-GAME DRIVER (claude_bugscan_2026-10-21/native_raw, 2026-10-26).
//
// One raw-policy self-play game (no search) on the resident native engine, mirroring
// selfplay_actor_critic/environment.py PythonEnvironment.prepare/apply/after_action line for line:
//   opening -> (pending | declaration | tail) -> preaction (tail resume, pending) -> roll -> actions
//   (pending with the active seat as fallback, the active seat, end-of-turn victory, turn limit,
//   ai._safe_next_turn) -> opening ...
// Every engine step is an existing, parity-proven native primitive of the live box (unchanged
// sources copied into this folder): check_victory / terminal_should_end (game_logic victory),
// tr_settle (game_logic._settle_end_of_turn), st_safe_next_turn (ai._safe_next_turn),
// native_pending_actor (ai._pending_decision_side), native_next_actor (ai._next_acting_side),
// fellowship_tail (game_logic.begin/resume_fellowship_tail), roll_action_dice
// (game_logic.roll_action_dice), native_full_actions + project_native_ai_actions (ai.get_ai_policy),
// apply_native_random_descriptor (ai._execute_action), encode_state_v53 / af_encode_v53 (ai.encode_state /
// ai._encode_actions_batch_ordinary at v53).  The game RNG is ONE CPython-Random stream (wotr_random),
// the native twin of environment.GameRandom's python stream; the engine never draws from numpy.
// Single-option frames are applied here exactly as valuedata.play applies them (no encoding, no row).
// The policy forward and sampling stay in Python (torch / numpy), so they are the generator's own code.
enum class rg_Stage {Opening,Declaration,PostDeclaration,Preaction,PostAllocation,Actions};
const char* rg_stage_name(rg_Stage s){
    switch(s){case rg_Stage::Opening:return "opening";case rg_Stage::Declaration:return "declaration";
    case rg_Stage::PostDeclaration:return "post_declaration";case rg_Stage::Preaction:return "preaction";
    case rg_Stage::PostAllocation:return "post_allocation";case rg_Stage::Actions:return "actions";}
    return "?";
}
struct rg_Game {
    std::unique_ptr<Snapshot> state;
    const Snapshot* tables=nullptr;          // observation tables (Python-owned, outlive the game)
    const Snapshot* action_tables=nullptr;   // prepared v53 action tables (Python-owned)
    std::unique_ptr<void,void(*)(void*)> random{nullptr,wotr_random_destroy};
    rg_Stage stage=rg_Stage::Opening;
    int active=-1;                            // PythonEnvironment.active (-1 = None)
    int64_t committed=0,first_turn=0,max_turns=100,max_transitions=10000;
    // episode end
    bool ended=false;int status=0;            // 1 terminated, 2 truncated
    int winner=-1,kind=0;                     // TerminalResult winner / kind
    std::string reason;
    // current open frame (learned; with expose_forced also the single-option ones)
    bool frame_open=false;int frame_side=-1;rg_Stage frame_phase=rg_Stage::Opening;int64_t frame_turn=0;
    std::unique_ptr<Snapshot> menu;size_t rows=0;
    bool expose_forced=false;                 // v2: hand single-option frames to the caller too (recorders)
    bool frame_forced=false;                  // the open frame is a single-option frame
    std::string frame_reason;                 // PreparedDecision.fixed_reason ('' = None)
    std::unique_ptr<Snapshot> apply_menu;int64_t apply_index=-1;   // the full menu + row a reduced forced frame applies
    // counters
    int64_t forced_applied=0,learned=0;
    // kind vocabulary (action_type strings, in order of first appearance in this game)
    std::vector<std::string> vocab;std::unordered_map<std::string,int32_t> vocab_index;
    // v4: chance-log leaves (chance_log.inc): (probability, successor state) and the last status reason
    std::vector<std::pair<double,std::unique_ptr<Snapshot>>> chance_leaves;std::string chance_why;
};
int64_t rg_turn(const Snapshot& s){return typed(s,obs_attr(s,s.root,"turn"),WOTR_INT,"turn must be integer").integer;}
void rg_finish_victory(rg_Game& g,const TerminalResult& r){g.ended=true;g.status=1;g.winner=r.winner;g.kind=r.kind;g.reason.clear();}
void rg_finish_truncated(rg_Game& g,const char* reason){g.ended=true;g.status=2;g.winner=-1;g.kind=0;g.reason=reason;}
// PythonEnvironment._victory(end_of_turn=False)
bool rg_victory(rg_Game& g){
    const auto r=check_victory(*g.state,*g.tables);
    if(terminal_should_end(r,false)){rg_finish_victory(g,r);return true;}
    return false;
}
// PythonEnvironment._victory(end_of_turn=True): settle a CLONE; only a terminal settlement touches the real state,
// through ai._safe_next_turn (which settles it again and returns True without any reset).
bool rg_victory_end_of_turn(rg_Game& g){
    {Snapshot probe(*g.state);if(!tr_settle(probe,*g.tables))return false;}
    am_Builder events;std::vector<uint64_t> rows;
    st_safe_next_turn(*g.state,*g.tables,events,rows);
    const auto r=check_victory(*g.state,*g.tables);
    if(terminal_should_end(r,true)){rg_finish_victory(g,r);return true;}
    return false;
}
// game_logic.begin_fellowship_tail (begin) / resume_fellowship_tail
void rg_tail(rg_Game& g,bool begin){
    auto tail=fellowship_tail(*g.state,*g.tables,g.random.get(),begin);
    g.state=std::move(tail.state);g.random=std::move(tail.random);
}
// ai._turn_start_side
int rg_turn_start_side(const Snapshot& s){
    const auto p=native_pending_actor(s,0);if(p.side>=0)return p.side;
    for(int side:{0,1})if(ar_has_unused(s,side))return side;
    return -1;
}
// PythonEnvironment._roll
void rg_roll(rg_Game& g){
    rr_set_attr(*g.state,g.state->root,"_eye_allocation_phase",rr_scalar(*g.state,WOTR_BOOL,0));
    for(int side:{0,1}){
        auto rolled=roll_action_dice(*g.state,g.random.get(),side);
        g.state=std::move(rolled.state);g.random=std::move(rolled.random);
    }
    g.stage=rg_Stage::Actions;g.active=rg_turn_start_side(*g.state);
}
// not state.fellowship.mordor_entered and ai._get_declare_actions(state, allow_in_place=True)
bool rg_declare_available(const Snapshot& s){
    const auto fs=obs_attr(s,s.root,"fellowship");
    if(!truth(s,obs_attr(s,fs,"hidden"))||truth(s,obs_attr(s,fs,"mordor_entered")))return false;
    if(ht_integer(s,fs,"progress")<0)return false;
    for(const auto& region:fd_reachable(s,fs))if(!fd_interior(region))return true;
    return false;
}
std::string_view rg_row_kind(const Snapshot& menu,uint64_t row){
    const auto fields=child_edges(menu,row);require(fields.size()==3,"menu row must be (die, kind, params)");
    return text(menu,fields[1]);
}
// PythonEnvironment.after_action
void rg_after_action(rg_Game& g,rg_Stage phase,int side,std::string_view kind){
    if(phase==rg_Stage::Declaration)g.stage=rg_Stage::PostDeclaration;
    else if(phase==rg_Stage::Preaction){if(kind=="eye_allocate")g.stage=rg_Stage::PostAllocation;}
    else if(phase==rg_Stage::Actions){
        const int owner=g.active>=0?g.active:side;
        if(native_pending_actor(*g.state,owner).side<0)g.active=native_next_actor(*g.state,owner);
    }
}
// PythonEnvironment.apply (ai._execute_action, committed += 1, after_action)
void rg_apply_row(rg_Game& g,const Snapshot& menu,int side,rg_Stage phase,int64_t index){
    const auto rows=child_edges(menu,menu.root);
    require(index>=0&&static_cast<uint64_t>(index)<rows.size(),"raw game: selected row out of range");
    const std::string kind(rg_row_kind(menu,rows[static_cast<size_t>(index)]));
    auto result=apply_native_random_descriptor(*g.state,*g.tables,menu,g.random.get(),side,index);
    g.state=std::move(result.state);g.random=std::move(result.random);
    g.committed+=1;
    rg_after_action(g,phase,side,kind);
}
// PythonEnvironment._frame: returns true when a LEARNED frame (more than one option) is open; a single-option
// frame is applied immediately (valuedata.play: env.apply(frame, 0); continue).
bool rg_frame(rg_Game& g,int side){
    auto raw=native_full_actions(*g.state,*g.tables,side);
    auto projected=project_native_ai_actions_owned(*g.state,std::move(raw),side);
    auto menu=std::move(projected.actions);
    auto rows=child_edges(*menu,menu->root);
    int64_t forced=-1;
    const auto fs=obs_attr(*g.state,g.state->root,"fellowship");
    if(g.stage==rg_Stage::Declaration&&truth(*g.state,obs_attr(*g.state,fs,"hidden"))&&ht_integer(*g.state,fs,"progress")>0){
        // Preserve the sanctioned gateway-declaration convention (environment.py _frame)
        for(const char* gateway:{"morannon","minas morgul"}){
            for(size_t i=0;i<rows.size()&&forced<0;++i){
                const auto fields=child_edges(*menu,rows[i]);
                if(text(*menu,fields[1])!="fellowship_declare")continue;
                const auto region=field(*menu,fields[2],"region",false);
                if(region!=missing&&menu->nodes[region].tag==WOTR_STRING&&text(*menu,region)==gateway)forced=static_cast<int64_t>(i);
            }
            if(forced>=0)break;
        }
    }
    if(rows.empty())throw std::runtime_error(std::string("No legal action for ")+(side==0?"fp":"sh")+" in "+rg_stage_name(g.stage));
    if(forced>=0||rows.size()==1){
        g.forced_applied+=1;
        if(!g.expose_forced){rg_apply_row(g,*menu,side,g.stage,forced>=0?forced:0);return false;}
        // v2 recorders: open the single-option frame. A gateway-forced declaration is the one-row frame
        // environment.py builds ((forced,), reason 'mordor_gateway_declaration'); it applies through the full menu.
        g.frame_open=true;g.frame_forced=true;g.frame_side=side;g.frame_phase=g.stage;g.frame_turn=rg_turn(*g.state);
        g.frame_reason=forced>=0?std::string("mordor_gateway_declaration"):projected.forced_reason;
        if(forced>=0){
            auto reduced=std::make_unique<Snapshot>(*menu);const auto offset=reduced->edges.size();
            reduced->edges.push_back(rows[static_cast<size_t>(forced)]);
            reduced->root=reduced->nodes.size();reduced->nodes.push_back({WOTR_LIST,0,0,0,offset,1});
            g.apply_menu=std::move(menu);g.apply_index=forced;g.menu=std::move(reduced);
        }else{g.apply_menu.reset();g.apply_index=-1;g.menu=std::move(menu);}
        g.rows=1;return true;
    }
    g.frame_open=true;g.frame_forced=false;g.frame_reason=projected.forced_reason;g.frame_side=side;g.frame_phase=g.stage;g.frame_turn=rg_turn(*g.state);
    g.rows=rows.size();g.menu=std::move(menu);g.apply_menu.reset();g.apply_index=-1;g.learned+=1;
    return true;
}
// PythonEnvironment.prepare: advance to the next learned frame (true) or to the episode end (false).
bool rg_prepare(rg_Game& g){
    require(!g.frame_open,"raw game: a learned frame is still open (apply it first)");
    while(!g.ended){
        if(rg_victory(g))return false;
        if(g.committed>=g.max_transitions){
            if(g.stage==rg_Stage::Actions&&g.active<0&&native_pending_actor(*g.state,0).side<0&&rg_victory_end_of_turn(g))return false;
            rg_finish_truncated(g,"transition_limit");return false;
        }

        if(g.stage==rg_Stage::PostDeclaration){rg_tail(g,true);g.stage=rg_Stage::Preaction;}
        else if(g.stage==rg_Stage::PostAllocation){rg_roll(g);}
        if(g.stage==rg_Stage::Opening){
            const auto pending=native_pending_actor(*g.state,-1);
            if(pending.side>=0){if(rg_frame(g,pending.side))return true;continue;}
            if(rg_declare_available(*g.state)){
                rr_set_attr(*g.state,g.state->root,"_declare_phase",rr_scalar(*g.state,WOTR_BOOL,1));g.stage=rg_Stage::Declaration;
            }else{rg_tail(g,true);g.stage=rg_Stage::Preaction;}
        }
        if(g.stage==rg_Stage::Declaration){if(rg_frame(g,0))return true;continue;}
        if(g.stage==rg_Stage::Preaction){
            rg_tail(g,false);
            const auto pending=native_pending_actor(*g.state,-1);
            if(pending.side>=0){if(rg_frame(g,pending.side))return true;continue;}
            rg_roll(g);
        }
        if(g.stage==rg_Stage::Actions){
            const auto pending=native_pending_actor(*g.state,g.active);
            if(pending.side>=0){if(rg_frame(g,pending.side))return true;continue;}
            if(g.active>=0){if(rg_frame(g,g.active))return true;continue;}
            if(rg_victory_end_of_turn(g))return false;
            if(rg_turn(*g.state)-g.first_turn+1>=g.max_turns){rg_finish_truncated(g,"turn_limit");return false;}
            am_Builder events;std::vector<uint64_t> rows;
            st_safe_next_turn(*g.state,*g.tables,events,rows);
            g.stage=rg_Stage::Opening;
        }

    }
    return false;
}
rg_Game& rg_game(void* handle){require(handle!=nullptr,"raw game handle is null");return *static_cast<rg_Game*>(handle);}

extern "C" {
WOTR_API uint32_t wotr_raw_abi_version(){return 4;}
WOTR_API const char* wotr_raw_last_error(){return error_text.data();}
// state: a complete GameState snapshot at the game's start (PythonEnvironment.__init__ done in Python);
// random: the CPython Random stream holding environment.GameRandom's python state after that init.
// Both are copied; tables / action_tables are borrowed and must outlive the game.
WOTR_API void* wotr_raw_create(const void* state,const void* tables,const void* action_tables,const void* random,int64_t max_turns,int64_t max_transitions){
    return checked<void*>([&]()->void*{
        require(max_turns>=1&&max_transitions>=1,"Episode limits must be positive");
        auto g=std::make_unique<rg_Game>();
        g->state=std::make_unique<Snapshot>(snapshot(state));g->tables=&snapshot(tables);g->action_tables=&snapshot(action_tables);
        g->random.reset(wotr_random_clone(random));if(!g->random)throw std::invalid_argument(wotr_random_last_error());
        g->max_turns=max_turns;g->max_transitions=max_transitions;g->first_turn=rg_turn(*g->state);
        require(!truth(*g->state,obs_attr(*g->state,g->state->root,"dice_rolled")),"raw game starts before the first roll");
        g->stage=rg_Stage::Opening;
        return g.release();
    },nullptr);
}
// TURNSEARCH (2026-11-13): a game started at a MID-GAME position.  `state` is a complete GameState snapshot at a
// decision; stage / active are PythonEnvironment.stage (rg_Stage) and .active (-1 = none) of that moment.
WOTR_API void* wotr_raw_create_at(const void* state,const void* tables,const void* action_tables,const void* random,int64_t max_turns,int64_t max_transitions,int64_t stage,int64_t active){
    return checked<void*>([&]()->void*{
        require(max_turns>=1&&max_transitions>=1,"Episode limits must be positive");
        require(stage>=0&&stage<=5&&active>=-1&&active<=1,"raw game: stage / active out of range");
        auto g=std::make_unique<rg_Game>();
        g->state=std::make_unique<Snapshot>(snapshot(state));g->tables=&snapshot(tables);g->action_tables=&snapshot(action_tables);
        g->random.reset(wotr_random_clone(random));if(!g->random)throw std::invalid_argument(wotr_random_last_error());
        g->max_turns=max_turns;g->max_transitions=max_transitions;g->first_turn=rg_turn(*g->state);
        g->stage=static_cast<rg_Stage>(stage);g->active=static_cast<int>(active);
        return g.release();
    },nullptr);
}
WOTR_API void wotr_raw_destroy(void* handle){delete static_cast<rg_Game*>(handle);}
// Advance to the next learned decision (with expose_forced: the next frame). Returns 1 with info = {side, phase(0 opening
// 1 declaration 3 preaction 5 actions), turn, rows, forced} when one is open, 0 at the episode end, -1 on error.
WOTR_API int wotr_raw_next(void* handle,int64_t* info){
    return checked<int>([&]()->int{
        auto& g=rg_game(handle);require(info!=nullptr,"raw game info output is null");
        if(!rg_prepare(g))return 0;
        info[0]=g.frame_side;info[1]=static_cast<int64_t>(g.frame_phase);info[2]=g.frame_turn;info[3]=static_cast<int64_t>(g.rows);info[4]=g.frame_forced?1:0;
        return 1;
    },-1);
}
// Encode the open frame: mover observation, other-side observation (valuedata's "watch"), the menu's v53 action rows
// and primary regions, and each row's action_type as an index into this game's kind vocabulary.
WOTR_API int wotr_raw_encode(void* handle,float* observation,float* watch,size_t width,float* candidates,size_t action_width,int64_t* regions,int32_t* kinds,size_t capacity){
    return checked<int>([&]()->int{
        auto& g=rg_game(handle);require(g.frame_open,"raw game: no open frame to encode");
        require(capacity>=g.rows,"raw game: row capacity too small");
        encode_state_v53(*g.state,*g.tables,g.frame_side,observation,width);
        encode_state_v53(*g.state,*g.tables,1-g.frame_side,watch,width);
        af_encode_v53(*g.state,*g.tables,*g.menu,*g.action_tables,g.frame_side,candidates,g.rows*action_width,regions,g.rows);
        const auto rows=child_edges(*g.menu,g.menu->root);
        for(size_t i=0;i<rows.size();++i){
            const std::string kind(rg_row_kind(*g.menu,rows[i]));
            auto found=g.vocab_index.find(kind);
            if(found==g.vocab_index.end()){found=g.vocab_index.emplace(kind,static_cast<int32_t>(g.vocab.size())).first;g.vocab.push_back(kind);}
            kinds[i]=found->second;
        }
        return 0;
    },-1);
}
// TURNSEARCH (2026-11-14): the parts of wotr_raw_encode on their own - ONE side's observation of the open frame, and
// the menu's rows with their primary regions.  The same routines as there: the same numbers.
WOTR_API int wotr_raw_encode_view(void* handle,int32_t side,float* observation,size_t width){
    return checked<int>([&]()->int{
        auto& g=rg_game(handle);require(g.frame_open,"raw game: no open frame to encode");
        encode_state_v53(*g.state,*g.tables,side,observation,width);
        return 0;
    },-1);
}
WOTR_API int wotr_raw_encode_menu(void* handle,float* candidates,size_t action_width,int64_t* regions,size_t capacity){
    return checked<int>([&]()->int{
        auto& g=rg_game(handle);require(g.frame_open,"raw game: no open frame to encode");
        require(capacity>=g.rows,"raw game: row capacity too small");
        af_encode_v53(*g.state,*g.tables,*g.menu,*g.action_tables,g.frame_side,candidates,g.rows*action_width,regions,g.rows);
        return 0;
    },-1);
}
// TURNSEARCH (2026-11-14): 1 (the start value) = the army_move tails of a menu are made with the state's facts read
// once (encoding_v53.inc v53_Memo); 0 = row by row, as the certified engine makes them.  For the check that the two
// are the same numbers (checks/engine_same.py); returns the value before.
WOTR_API int wotr_raw_memo(int on){const int before=v53_memo_active;v53_memo_active=on?1:0;return before;}
WOTR_API int64_t wotr_raw_vocab_size(void* handle){return checked<int64_t>([&](){return static_cast<int64_t>(rg_game(handle).vocab.size());},-1);}
WOTR_API const char* wotr_raw_vocab(void* handle,int64_t index){
    return checked<const char*>([&]()->const char*{auto& g=rg_game(handle);require(index>=0&&static_cast<size_t>(index)<g.vocab.size(),"vocab index");return g.vocab[static_cast<size_t>(index)].c_str();},nullptr);
}
// Apply the sampled row of the open frame (PythonEnvironment.apply).
WOTR_API int wotr_raw_apply(void* handle,int64_t index){
    return checked<int>([&]()->int{
        auto& g=rg_game(handle);require(g.frame_open,"raw game: no open frame to apply");
        auto menu=std::move(g.menu);auto full=std::move(g.apply_menu);const auto mapped=g.apply_index;
        g.frame_open=false;g.rows=0;g.frame_forced=false;g.apply_index=-1;
        if(full){require(index==0,"raw game: a forced frame has one row");rg_apply_row(g,*full,g.frame_side,g.frame_phase,mapped);}
        else rg_apply_row(g,*menu,g.frame_side,g.frame_phase,index);
        return 0;
    },-1);
}
// out = {status (1 terminated, 2 truncated), winner (-1, 0 fp, 1 sh), kind (0 none, 1 military, 2 ring_destroyed,
// 3 ring_corruption), committed transitions, forced frames applied, learned frames}; reason copied into buffer.
WOTR_API int wotr_raw_end(void* handle,int64_t* out,char* reason,size_t capacity){
    return checked<int>([&]()->int{
        auto& g=rg_game(handle);require(out!=nullptr,"raw game end output is null");
        out[0]=g.ended?g.status:0;out[1]=g.winner;out[2]=g.kind;out[3]=g.committed;out[4]=g.forced_applied;out[5]=g.learned;out[6]=rg_turn(*g.state);
        if(reason&&capacity){const size_t n=std::min(capacity-1,g.reason.size());std::memcpy(reason,g.reason.data(),n);reason[n]='\0';}
        return g.ended?1:0;
    },-1);
}
// Diagnostics for the lockstep harness: copies of the current state / RNG / open menu (caller owns them).
WOTR_API void* wotr_raw_state(void* handle){return checked<void*>([&]()->void*{return new Snapshot(*rg_game(handle).state);},nullptr);}
WOTR_API void* wotr_raw_random(void* handle){
    return checked<void*>([&]()->void*{void* r=wotr_random_clone(rg_game(handle).random.get());if(!r)throw std::invalid_argument(wotr_random_last_error());return r;},nullptr);
}
WOTR_API void* wotr_raw_menu(void* handle){
    return checked<void*>([&]()->void*{auto& g=rg_game(handle);require(g.frame_open,"raw game: no open frame");return new Snapshot(*g.menu);},nullptr);
}
WOTR_API int wotr_raw_stage(void* handle,int64_t* out){
    return checked<int>([&]()->int{auto& g=rg_game(handle);out[0]=static_cast<int64_t>(g.stage);out[1]=g.active;out[2]=g.committed;return 0;},-1);
}
// v2: options (bit 0 = expose single-option frames to the caller).
WOTR_API int wotr_raw_set_options(void* handle,int64_t flags){
    return checked<int>([&]()->int{auto& g=rg_game(handle);require(!g.frame_open&&g.committed==0,"raw game options are set before the first frame");g.expose_forced=(flags&1)!=0;return 0;},-1);
}
// v2: the open frame's fixed reason (environment.PreparedDecision.fixed_reason; '' = None); valid until the next call.
WOTR_API const char* wotr_raw_fixed_reason(void* handle){
    return checked<const char*>([&]()->const char*{auto& g=rg_game(handle);require(g.frame_open,"raw game: no open frame");return g.frame_reason.c_str();},nullptr);
}
// v3: probabilities.encode_probabilities of the open frame, natively (chance_features.inc): rows x width float32.
WOTR_API int64_t wotr_raw_chance_width(){return checked<int64_t>([&](){return static_cast<int64_t>(cf_width());},-1);}
WOTR_API int wotr_raw_chance(void* handle,float* out,size_t width,size_t capacity){
    return checked<int>([&]()->int{
        auto& g=rg_game(handle);require(g.frame_open,"raw game: no open frame");require(out!=nullptr,"raw game: chance output is null");
        require(width==cf_width(),"raw game: chance width mismatch");require(capacity>=g.rows,"raw game: chance capacity too small");
        cf_encode(*g.state,*g.tables,*g.menu,g.frame_side,out,g.rows);return 0;
    },-1);
}
// v2: one row of the open frame as its own snapshot (die, kind, params): the recorders decode only what they read.
WOTR_API void* wotr_raw_row(void* handle,int64_t index){
    return checked<void*>([&]()->void*{
        auto& g=rg_game(handle);require(g.frame_open,"raw game: no open frame");
        const auto rows=child_edges(*g.menu,g.menu->root);require(index>=0&&static_cast<uint64_t>(index)<rows.size(),"raw game: row index out of range");
        Snapshot copy(*g.menu);copy.root=rows[static_cast<size_t>(index)];return ac_compact(copy).release();
    },nullptr);
}
}
