// One inner round transaction through the next pending owner decision.
// Stage bodies mutate the one owned graph and RNG stream. Standalone APIs
// retain their independent copy contracts; compact once at the next decision.
bool combat_round_stage_inplace(Snapshot& source,void* random_source,int64_t selection_index,bool battle){
    const auto resume=field(source,obs_attr(source,source.root,"pending_combat"),"_round_resume"),cursor=field(source,resume,"cursor");
    attributes(source,cursor,"combat_round_resume.RoundCursor");require(text_is(source,obs_attr(source,cursor,"schema"),"combat_round_cursor_v1"),"Unknown combat round cursor schema");
    const auto stage=cp_number(source,cursor,"stage");require(stage>=0&&stage<=14,"Round stage exceeds native cursor bounds");
    require(selection_index>=-1,"Round choice index must be -1 or nonnegative");
    if(battle){require(text_is(source,field(source,resume,"schema"),"combat_battle_resume_v1"),"Native round requires battle resume schema");require(obs_attr(source,obs_attr(source,cursor,"combat"),"_live_state",false)==source.root,"Native round requires its owned live GameState");
        for(const char* key:{"downgrades","reinforcement_baseline"}){const auto map=field(source,resume,key);typed(source,map,WOTR_DICT,"Native round requires supply ledgers");const auto entries=child_edges(source,map);for(size_t i=0;i<entries.size();i+=2){validate_nation_name(enum_member(source,entries[i],"game_state.Nation"));const auto n=typed(source,entries[i+1],WOTR_INT,"Supply counts must be integer").integer;require(n>=(std::string_view(key)=="downgrades"?-1048576:0)&&n<=1048576,"Supply count exceeds native bound");}}
    }
    if(selection_index<0&&!ctx_none(source,obs_attr(source,cursor,"pending")))return false;
    if(battle)for(const char* key:{"downgrades","reinforcement_baseline"})cp_put(source,resume,key,cp_container(source,WOTR_DICT,child_edges(source,field(source,resume,key))));
    for(int steps=0;steps<5;++steps){
        auto& s=source;const auto payload=field(s,obs_attr(s,s.root,"pending_combat"),"_round_resume"),cur=field(s,payload,"cursor");const auto current=cp_number(s,cur,"stage");
        if(current<=4){combat_pre_stage_inplace(s,random_source,selection_index,battle);}
        else if(current<=8){
            std::string target;const char* selected=nullptr;if(selection_index>=0){const auto pending=obs_attr(s,cur,"pending");require(!ctx_none(s,pending),"No pending Mighty choice");const auto options=field(s,pending,"options");require(static_cast<uint64_t>(selection_index)<s.nodes[options].count,"Mighty choice index out of range");target=std::string(text(s,field(s,s.edges[s.nodes[options].offset+selection_index],"target")));selected=target.c_str();}
            combat_primary_stage_inplace(s,random_source,selected);
        }
        else if(current==9){require(selection_index<0,"Rerolls have no selectable choice");combat_reroll_stage_inplace(s,random_source);}
        else{combat_post_stage_inplace(s,selection_index);}
        selection_index=-1;
        const auto next_cursor=field(source,field(source,obs_attr(source,source.root,"pending_combat"),"_round_resume"),"cursor");
        if(!ctx_none(source,obs_attr(source,next_cursor,"pending"))||cp_number(source,next_cursor,"stage")==14)return true;
    }
    throw std::invalid_argument("Native round failed to reach a decision boundary");
}

crr_Result combat_round_stage(const Snapshot& source,const void* random_source,int64_t selection_index,bool battle){
    crr_Result result;result.state=std::make_unique<Snapshot>(source);result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());
    if(combat_round_stage_inplace(*result.state,result.random.get(),selection_index,battle))result.state=ac_compact(*result.state);return result;
}
