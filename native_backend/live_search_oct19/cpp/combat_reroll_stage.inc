// Exact combat_round_resume._rerolls, then cursor stage 9 -> 10.
// Combat-card decisions and earlier rolled faces are already fixed inputs.
// This does not enumerate/reweight outcomes or change the exact chance model.
struct crr_Result {
    std::unique_ptr<Snapshot> state;
    std::unique_ptr<void,void(*)(void*)> random{nullptr,wotr_random_destroy};
};
int64_t crr_integer(const Snapshot& s,uint64_t object,const std::string& name){
    const auto value=obs_attr(s,object,name.c_str());const auto& node=s.nodes[value];
    require(node.tag==WOTR_INT || node.tag==WOTR_BOOL,"Combat reroll requires integer fields");return node.integer;
}
uint64_t crr_cursor(const Snapshot& s){
    attributes(s,s.root,"game_state.GameState");
    const auto pending=obs_attr(s,s.root,"pending_combat");
    const auto resume=field(s,pending,"_round_resume");
    const auto cursor=field(s,resume,"cursor");
    attributes(s,cursor,"combat_round_resume.RoundCursor");
    require(text_is(s,obs_attr(s,cursor,"schema"),"combat_round_cursor_v1"),"Unknown combat round cursor schema");
    require(crr_integer(s,cursor,"stage")==9,"Native rerolls require cursor stage 9");
    require(ctx_none(s,obs_attr(s,cursor,"pending")) && ctx_none(s,obs_attr(s,cursor,"selection")),"Resolve the pending owner choice before rerolls");
    const auto combat=obs_attr(s,cursor,"combat");attributes(s,combat,"game_state.CombatState");
    require(!truth(s,obs_attr(s,combat,"battle_over")),"Cannot reroll a completed battle");return cursor;
}
bool crr_hit(int64_t face,int64_t threshold,int64_t modifier){
    // Canonical engine modifiers are small integers. Validate the additive
    // domain rather than silently overflowing a forged int64 graph field.
    require(modifier<=std::numeric_limits<int64_t>::max()-6,"Combat modifier exceeds native integer range");
    return face!=1 && (face==6 || face+modifier>=threshold);
}
uint64_t crr_list(Snapshot& s,const uint32_t* values,size_t count){
    std::vector<uint64_t> nodes;nodes.reserve(count);
    for(size_t i=0;i<count;++i)nodes.push_back(rr_scalar(s,WOTR_INT,values[i]));
    const auto result=rr_scalar(s,WOTR_LIST);set_edges(s,result,nodes);return result;
}
void crr_mark(Snapshot& s,uint64_t set,std::string_view first,std::string_view second={}){
    typed(s,set,WOTR_SET,"Observed combat rolls must be sets");
    auto entries=child_edges(s,set);
    for(auto entry:entries){
        if(second.empty()){if(text_is(s,entry,first))return;}
        else if(s.nodes[entry].tag==WOTR_TUPLE && s.nodes[entry].count==2){
            const auto offset=s.nodes[entry].offset;
            if(text_is(s,s.edges[offset],first)&&text_is(s,s.edges[offset+1],second))return;
        }
    }
    auto node=rr_string(s,first);
    if(!second.empty()){
        const auto other=rr_string(s,second),tuple=rr_scalar(s,WOTR_TUPLE);
        set_edges(s,tuple,{node,other});node=tuple;
    }
    entries.push_back(node);set_edges(s,set,entries);
}
bool crr_observed(const Snapshot& s,uint64_t cursor,std::string_view phase,std::string_view role){
    const auto observed=obs_attr(s,cursor,"observed_roll_sides");
    typed(s,observed,WOTR_SET,"Observed combat rolls must be sets");
    for(auto entry:child_edges(s,observed)){
        if(s.nodes[entry].tag!=WOTR_TUPLE || s.nodes[entry].count!=2)continue;
        const auto offset=s.nodes[entry].offset;
        if(text_is(s,s.edges[offset],phase) && text_is(s,s.edges[offset+1],role))return true;
    }
    return false;
}
void combat_reroll_side_inplace(Snapshot& changed,uint64_t cursor,uint64_t combat,void* random_source,int side){
        const std::string role=side==0?"attacker":"defender",short_role=side==0?"atk":"def";
        if(crr_observed(changed,cursor,"reroll",role))return;
        int64_t hits=0;
        // Retain old rerolls unchanged when cancelled or leadership <= 0,
        // exactly as the original cursor stage does.
        if(!truth(changed,obs_attr(changed,combat,(role+"_reroll_cancelled").c_str())) && crr_integer(changed,combat,role+"_leadership")>0){
            const auto threshold=crr_integer(changed,combat,"_resume_"+short_role+"_threshold");
            const auto leadership=crr_integer(changed,combat,role+"_leadership");
            const auto modifier=crr_integer(changed,combat,role+"_reroll_mod");
            const auto miss_modifier=crr_integer(changed,combat,role+"_dice_mod");
            const auto excluded=crr_integer(changed,combat,"_resume_"+short_role+"_auto");
            const bool confused=truth(changed,obs_attr(changed,combat,(role+"_confusion").c_str()));
            const auto rolls=obs_attr(changed,combat,(role+"_combat_rolls").c_str());
            typed(changed,rolls,WOTR_LIST,"Prior combat rolls must be a list");
            int64_t misses=0;
            for(auto index:child_edges(changed,rolls)){
                const auto face=typed(changed,index,WOTR_INT,"Prior combat faces must be integer").integer;
                require(face>=1 && face<=6,"Prior combat face must be 1..6");
                if(confused && face==1)continue;
                if(!crr_hit(face,threshold,miss_modifier))++misses;
            }
            if(excluded>0)misses=excluded>=misses?0:misses-excluded;
            const auto count=static_cast<size_t>(std::min<int64_t>({5,leadership,misses}));
            std::array<uint32_t,5> values{};
            cf_d6(random_source,values.data(),count,threshold,modifier,false);
            for(size_t i=0;i<count;++i)if(crr_hit(values[i],threshold,modifier))++hits;
            rr_set_attr(changed,combat,role+"_rerolls",crr_list(changed,values.data(),count));
            const auto previous=crr_integer(changed,combat,role+"_hits");
            require(previous<=std::numeric_limits<int64_t>::max()-hits,"Combat hit count exceeds int64 storage");
            rr_set_attr(changed,combat,role+"_hits",rr_scalar(changed,WOTR_INT,previous+hits));
        }
        rr_set_attr(changed,combat,"_resume_"+short_role+"_reroll_hits",rr_scalar(changed,WOTR_INT,hits));
    crr_mark(changed,obs_attr(changed,cursor,"observed_rolls"),"reroll");
    crr_mark(changed,obs_attr(changed,cursor,"observed_roll_sides"),"reroll",role);
}
void combat_reroll_stage_inplace(Snapshot& changed,void* random_source){
    const auto cursor=crr_cursor(changed),combat=obs_attr(changed,cursor,"combat");
    for(int side=0;side<2;++side)combat_reroll_side_inplace(changed,cursor,combat,random_source,side);
    rr_set_attr(changed,cursor,"stage",rr_scalar(changed,WOTR_INT,10));
}

crr_Result combat_reroll_stage(const Snapshot& source,const void* random_source){
    crr_Result result;result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());result.state=std::make_unique<Snapshot>(source);
    combat_reroll_stage_inplace(*result.state,result.random.get());result.state=ac_compact(*result.state);return result;
}
