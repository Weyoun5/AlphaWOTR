// NATIVE CHANCE LOG (raw ABI 4, claude_bugscan_2026-10-21/native_cl, 2026-10-27): the alternatives of
// selfplay_actor_critic/chance_log.ChanceLogger on the native engine.
//
//   wotr_raw_chance_exact   chance.enumerate_chance(env, frame, choice, max_walks=...) for the open frame's row:
//                           the same depth-first plan stack, walk budget, fork driver (chance_driver.inc), leaf
//                           checks and probability-mass check.  Status 0 exact (leaves kept), 1 Unsupported (the
//                           caller continues with chance.expand's kernel / sampled branches), 2 replay this event on
//                           the Python enumerator (a condition Python raises as an error, or an input outside the port).
//   wotr_raw_chance_sample  one sampled clone of chance.expand: the row applied on a clone with the given stream
//                           (GameRandom(seed_k).python, built by the caller); appended to the leaves.
//   wotr_raw_chance_roll    one rolled clone of chance.roll_outcomes (PythonEnvironment._roll on a clone).
//   wotr_raw_chance_leaf    a kept leaf: probability, both observations (ai.encode_state for sh and fp) and
//                           chance_log.terminal_value.
//   wotr_raw_views          the same views of the live position (the realised outcome).
// Every walk applies the row exactly as wotr_raw_apply does (apply_native_random_descriptor) on the unchanged live
// state; the live game, its stream and its frame are never touched.
std::pair<const Snapshot*,int64_t> cl_row(const rg_Game& g,int64_t index){
    require(g.frame_open,"raw game: no open frame");
    if(g.apply_menu){require(index==0,"raw game: a forced frame has one row");return {g.apply_menu.get(),g.apply_index};}
    const auto rows=child_edges(*g.menu,g.menu->root);
    require(index>=0&&static_cast<uint64_t>(index)<rows.size(),"raw game: chance row out of range");
    return {g.menu.get(),index};
}
enum class cl_Walk {Leaf,Fork,Unsupported,Fallback};
struct cl_WalkOut {cl_Walk kind=cl_Walk::Fallback;std::unique_ptr<Snapshot> state;double prob=0.0;size_t options=0;std::string why;};
struct cl_Active {explicit cl_Active(cd_Scope* s){cd_active=s;}~cl_Active(){cd_active=nullptr;}};
// One walk of chance.enumerate_chance: apply under the driver following `plan`, on a copy of `base` (the clone's state).
cl_WalkOut cl_walk(const Snapshot& base,const Snapshot& tables,int side,const Snapshot& menu,int64_t row,const std::vector<int64_t>& plan,const void* walk_random){
    require(!cd_active&&!hf_active&&!cf_active,"nested chance driver");
    cd_Scope scope;scope.plan=plan;scope.mover=side;
    cl_WalkOut out;
    try{
        fm_Result result;
        {cl_Active on(&scope);result=apply_native_random_descriptor(base,tables,menu,walk_random,side,row);}
        if(scope.forked){out.kind=cl_Walk::Fork;out.options=scope.pending.size();return out;}
        require(result.state!=nullptr&&result.random!=nullptr,"chance walk produced no state");
        if(scope.scripts_left()){out.why="scripted dice not consumed";return out;}
        if(!cf_same_rng(walk_random,result.random.get())){out.kind=cl_Walk::Unsupported;out.why="raw RNG consumed outside the funnels";return out;}
        out.kind=cl_Walk::Leaf;out.prob=scope.prob;out.state=std::move(result.state);return out;
    }catch(const cd_Fork&){out.kind=cl_Walk::Fork;out.options=scope.pending.size();}
    catch(const cd_Unsupported& u){out.kind=cl_Walk::Unsupported;out.why=u.why;}
    catch(const cd_Fallback& f){out.kind=cl_Walk::Fallback;out.why=f.why;}
    catch(const std::exception& e){
        // chance.enumerate_chance: an engine error with a recorded fork is "fork here"; without one it propagates
        if(scope.forked){out.kind=cl_Walk::Fork;out.options=scope.pending.size();}
        else{out.kind=cl_Walk::Fallback;out.why=std::string("engine error: ")+e.what();}
    }catch(...){out.kind=cl_Walk::Fallback;out.why="unknown engine error";}
    return out;
}
// chance.enumerate_chance from `base` (the clone's state, determinized or not) for row `row` of `menu` played by
// `side`. 0 exact (leaves filled), 1 Unsupported, 2 replay on Python; `why` names the reason.
int cl_enumerate_from(const Snapshot& base,const Snapshot& tables,int side,const Snapshot& menu,int64_t row,int64_t max_walks,
                      const void* walk_random,int64_t& walks,std::vector<std::pair<double,std::unique_ptr<Snapshot>>>& leaves,std::string& why){
    leaves.clear();why.clear();walks=0;
    std::vector<std::vector<int64_t>> stack(1);
    while(!stack.empty()){
        auto plan=std::move(stack.back());stack.pop_back();
        if(++walks>max_walks){why="walk budget exceeded";leaves.clear();return 1;}
        auto w=cl_walk(base,tables,side,menu,row,plan,walk_random);
        if(w.kind==cl_Walk::Fork){
            for(size_t i=w.options;i-->0;){auto child=plan;child.push_back(static_cast<int64_t>(i));stack.push_back(std::move(child));}
            continue;
        }
        if(w.kind==cl_Walk::Unsupported){why=w.why;leaves.clear();return 1;}
        if(w.kind==cl_Walk::Fallback){why=w.why;leaves.clear();return 2;}
        leaves.push_back({w.prob,std::move(w.state)});
    }
    double total=0.0;for(const auto& leaf:leaves)total+=leaf.first;
    if(std::fabs(total-1.0)>1e-9){why="probability mass";leaves.clear();return 2;}
    return 0;
}
// chance.enumerate_chance over the open frame's row (the chance logger: no determinize).
int cl_enumerate(rg_Game& g,int64_t index,int64_t max_walks,const void* walk_random,int64_t& walks){
    const auto [menu,row]=cl_row(g,index);
    g.chance_leaves.clear();
    return cl_enumerate_from(*g.state,*g.tables,g.frame_side,*menu,row,max_walks,walk_random,walks,g.chance_leaves,g.chance_why);
}
// chance_log.terminal_value: the Shadow's result if the position decides the game now (end_of_turn=False), else NaN
double cl_terminal(const rg_Game& g,const Snapshot& s){
    try{
        const auto r=check_victory(s,*g.tables);
        if(!terminal_should_end(r,false))return std::numeric_limits<double>::quiet_NaN();
        return r.winner==1?1.0:-1.0;
    }catch(const std::exception&){return std::numeric_limits<double>::quiet_NaN();}
}
void cl_views(const rg_Game& g,const Snapshot& s,float* sh,float* fp,size_t width,double* terminal){
    require(sh!=nullptr&&fp!=nullptr&&terminal!=nullptr,"chance views: null output");
    encode_state_v53(s,*g.tables,1,sh,width);
    encode_state_v53(s,*g.tables,0,fp,width);
    *terminal=cl_terminal(g,s);
}

extern "C" {
// out = {status (0 exact, 1 Unsupported, 2 replay on Python), leaves kept, walks}; reason via wotr_raw_chance_why.
WOTR_API int wotr_raw_chance_exact(void* handle,int64_t index,int64_t max_walks,const void* walk_random,int64_t* out){
    return checked<int>([&]()->int{
        auto& g=rg_game(handle);require(out!=nullptr&&walk_random!=nullptr,"chance exact: null argument");
        int64_t walks=0;out[0]=cl_enumerate(g,index,max_walks,walk_random,walks);
        out[1]=static_cast<int64_t>(g.chance_leaves.size());out[2]=walks;return 0;
    },-1);
}
WOTR_API const char* wotr_raw_chance_why(void* handle){
    return checked<const char*>([&]()->const char*{return rg_game(handle).chance_why.c_str();},nullptr);
}
WOTR_API int wotr_raw_chance_clear(void* handle){
    return checked<int>([&]()->int{auto& g=rg_game(handle);g.chance_leaves.clear();g.chance_why.clear();return 0;},-1);
}
// One sampled clone (no driver): the open frame's row applied on a clone with `random`; appended to the leaves.
WOTR_API int wotr_raw_chance_sample(void* handle,int64_t index,const void* random){
    return checked<int>([&]()->int{
        auto& g=rg_game(handle);require(random!=nullptr,"chance sample: null stream");require(!cd_active,"chance sample under the driver");
        const auto [menu,row]=cl_row(g,index);
        auto result=apply_native_random_descriptor(*g.state,*g.tables,*menu,random,g.frame_side,row);
        g.chance_leaves.push_back({0.0,std::move(result.state)});return 0;
    },-1);
}
// One rolled clone of the current position (PythonEnvironment._roll on chance.roll_outcomes' clone).
WOTR_API int wotr_raw_chance_roll(void* handle,const void* random){
    return checked<int>([&]()->int{
        auto& g=rg_game(handle);require(random!=nullptr,"chance roll: null stream");require(!g.frame_open,"chance roll with an open frame");
        auto state=std::make_unique<Snapshot>(*g.state);
        std::unique_ptr<void,void(*)(void*)> stream{wotr_random_clone(random),wotr_random_destroy};
        if(!stream)throw std::invalid_argument(wotr_random_last_error());
        rr_set_attr(*state,state->root,"_eye_allocation_phase",rr_scalar(*state,WOTR_BOOL,0));
        for(int side:{0,1}){auto rolled=roll_action_dice(*state,stream.get(),side);state=std::move(rolled.state);stream=std::move(rolled.random);}
        g.chance_leaves.push_back({0.0,std::move(state)});return 0;
    },-1);
}
// Leaf i: out2 = {probability, terminal}; sh / fp = ai.encode_state for the Shadow / the Free Peoples.
WOTR_API int wotr_raw_chance_leaf(void* handle,int64_t i,float* sh,float* fp,size_t width,double* out2){
    return checked<int>([&]()->int{
        auto& g=rg_game(handle);require(i>=0&&static_cast<size_t>(i)<g.chance_leaves.size(),"chance leaf index out of range");
        require(out2!=nullptr,"chance leaf: null output");const auto& leaf=g.chance_leaves[static_cast<size_t>(i)];
        out2[0]=leaf.first;cl_views(g,*leaf.second,sh,fp,width,out2+1);return 0;
    },-1);
}
// The live position's views and terminal value (the realised outcome).
WOTR_API int wotr_raw_views(void* handle,float* sh,float* fp,size_t width,double* terminal){
    return checked<int>([&]()->int{auto& g=rg_game(handle);cl_views(g,*g.state,sh,fp,width,terminal);return 0;},-1);
}
}
