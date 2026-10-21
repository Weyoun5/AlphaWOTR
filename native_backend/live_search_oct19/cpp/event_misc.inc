// Small ordinary events with complete funded and learned-choice transactions.
bool em_supported(const Snapshot& t,uint64_t card){
    if(card==missing)return false;const auto name=obs_attr(t,card,"card_name");
    return text_is(t,name,"Athelas") || text_is(t,name,"Wisdom of Elrond");
}
std::vector<std::pair<std::string,std::string>> em_wisdom_options(const Snapshot& s){
    std::vector<std::pair<std::string,std::string>> result;
    for(const auto& entry:std::vector<std::pair<const char*,const char*>>{{"ROHAN","rohan"},{"GONDOR","gondor"},{"THE_NORTH","north"},{"DWARVES","dwarves"},{"ELVES","elves"}})
        if(political_view(s,entry.first,true).box<3)result.emplace_back(entry.first,entry.second);
    return result;
}
void em_wisdom(Snapshot& s,std::string_view nation){
    if(nation.empty())return;if(wt_can_activate(s,nation,"event_card"))me_activate(s,nation);
    const auto boxes=obs_attr(s,obs_attr(s,s.root,"political"),"boxes");
    const auto value=typed(s,ctx_nation_value(s,boxes,nation),WOTR_INT,"Wisdom needs integer political position").integer;
    require(value>=0 && value<=3,"Wisdom political position invalid");
    const auto next=std::min<int64_t>(wt_active(s,nation)?3:2,value+1);
    if(next>value)me_nation_assign(s,boxes,nation,rr_scalar(s,WOTR_INT,next));
}
void apply_misc_event_action_inplace(Snapshot& s,const Snapshot& t,void* rng,int side,int64_t index,std::string_view id,bool free_ents,am_Builder& ef_events,std::vector<uint64_t>& ef_annotations){
    const auto& source=s;ef_validate_free(source,t,side,index,id,free_ents);
    const int ef_marker=ctx_side(source,obs_attr(source,source.root,"_battle_resolved_attacker",false));const auto ef_unused=ef_unused_dice(source);

    // Package M: the free Ents-chain play replays the ordinary play (event_effects._effect_ents_awake_common -> ai._execute_action
    // play_event without a die); the paid play already admits stable sieges, so does the in-place variant now.
    require(side==0,"Misc event belongs to FP");if(!free_ents)quiet_stable_action_gate(source,t);else quiet_action_gate(source,t,false,false,false,false,false,false,false,false,false,false,(false||free_ents),true);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==side,"Misc event belongs to retained Ring actor");
    const auto card=ev_card(t,id);require(card!=missing && em_supported(t,card) && ctx_side(t,obs_attr(t,card,"side"))==side,"Unsupported misc event");
    require(contains_string(source,tc_hand(source,side),id),"Misc event absent from hand");require(ev_requirement(source,t,card),"Misc event requirement false");
    const auto die=free_ents?missing:ev_payment(source,t,side,index,card);const std::string face(free_ents?std::string_view{}:enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace"));
    const std::string name(text(t,obs_attr(t,card,"card_name"))),deck=ev_lower(text(t,obs_attr(t,card,"deck")));
    
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    if(!free_ents)rr_set_attr(s,ev_payment(s,t,side,index,card),"used",rr_scalar(s,WOTR_BOOL,1));
    bool deferred=false;
    if(name=="Athelas"){
        const auto fs=obs_attr(s,s.root,"fellowship"),guide=obs_attr(s,fs,"guide");
        const int threshold=!ctx_none(s,guide) && text(s,obs_attr(s,guide,"name")).find("Strider")!=std::string_view::npos?3:5;
        std::array<uint32_t,3> rolls{};hf_roll(rng,rolls.data(),rolls.size(),"card effect");
        int heals=0;std::vector<uint64_t> faces;for(auto roll:rolls){heals+=roll>=static_cast<uint32_t>(threshold);faces.push_back(ef_events.integer(roll));}
        rr_set_attr(s,fs,"corruption",rr_scalar(s,WOTR_INT,std::max<int64_t>(0,typed(s,obs_attr(s,fs,"corruption"),WOTR_INT,"Athelas corruption must be integer").integer-heals)));
        ef_annotations.push_back(ef_events.row(-1,"dice_roll",{ef_events.str("faces"),ef_events.container(WOTR_LIST,faces),ef_events.str("ctx"),ef_events.str("Athelas roll"),ef_events.str("side"),rr_scalar(ef_events.out,WOTR_NONE)}));
    }else{
        const auto options=em_wisdom_options(s);
        if(options.size()>1){
            std::vector<uint64_t> choices;for(const auto& option:options)choices.push_back(ev_dict(s,{{"nation",rr_string(s,option.second)}}));
            rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,name)},{"side",rr_string(s,"fp")},{"decider",rr_string(s,"fp")},
                {"options",cp_container(s,WOTR_LIST,choices)},{"die_face",ef_pending_face_node(s,face)},{"deck",rr_string(s,deck)}}));
            tc_open_hand_limit_discard(s);deferred=true;
        }else em_wisdom(s,options.empty()?std::string_view{}:std::string_view(options.front().first));
    }
    if(!deferred)ed_commit(s,side,id,deck,face);
    rr_set_attr(s,s.root,"_battle_resolved_attacker",rr_scalar(s,WOTR_NONE));
    

    if(free_ents)ef_finish_free(s,t,rng,ef_marker,ef_unused,ef_events,ef_annotations);
}
fm_Result apply_misc_event_action(const Snapshot& source,const Snapshot& t,const void* random_source,int side,int64_t index,std::string_view id){
    require(side==0,"Misc event belongs to FP");quiet_stable_action_gate(source,t);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==side,"Misc event belongs to retained Ring actor");
    const auto card=ev_card(t,id);require(card!=missing && em_supported(t,card) && ctx_side(t,obs_attr(t,card,"side"))==side,"Unsupported misc event");
    require(contains_string(source,tc_hand(source,side),id),"Misc event absent from hand");require(ev_requirement(source,t,card),"Misc event requirement false");
    const auto die=ev_payment(source,t,side,index,card);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace"));
    const std::string name(text(t,obs_attr(t,card,"card_name"))),deck=ev_lower(text(t,obs_attr(t,card,"deck")));
    
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());
    am_Builder ef_events;std::vector<uint64_t> ef_annotations;
    apply_misc_event_action_inplace(*result.state,t,result.random.get(),side,index,id,false,ef_events,ef_annotations);
    ef_events.out.root=ef_events.container(WOTR_LIST,ef_annotations);result.events=std::make_unique<Snapshot>(std::move(ef_events.out));return result;
}
uint64_t em_pending(const Snapshot& s){
    const auto p=obs_attr(s,s.root,"pending_card_option",false);
    require(!ctx_none(s,p) && text_is(s,field(s,p,"card_name",false),"Wisdom of Elrond") && text_is(s,field(s,p,"side",false),"fp") && text_is(s,field(s,p,"decider",false),"fp"),"No FP Wisdom choice pending");
    const auto choices=child_edges(s,field(s,p,"options"));const auto expected=em_wisdom_options(s);
    require(choices.size()>=2 && choices.size()==expected.size(),"Wisdom pending options stale");
    for(size_t i=0;i<choices.size();++i)require(typed(s,choices[i],WOTR_DICT,"Wisdom option must be dict").count==2 && text_is(s,field(s,choices[i],"nation"),expected[i].second),"Wisdom pending option order invalid");
    return p;
}
std::unique_ptr<Snapshot> native_misc_event_choice_actions(const Snapshot& source,int side){
    require(side==0,"Wisdom menu belongs to FP");em_pending(source);
    auto result=std::make_unique<Snapshot>(source);auto& s=*result;const auto p=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t index=0;
    for(auto option:child_edges(s,field(s,p,"options"))){
        const auto params=ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",option}});
        rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),params}));
    }
    s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
std::unique_ptr<Snapshot> apply_misc_event_choice(const Snapshot& source,const Snapshot& t,int side,int64_t option){
    require(side==0,"Wisdom choice belongs to FP");const auto p=em_pending(source);const auto options=child_edges(source,field(source,p,"options"));
    require(option>=0 && static_cast<uint64_t>(option)<options.size(),"Wisdom choice index out of range");
    const auto nation=em_wisdom_options(source)[option].first;
    const std::string id(text(source,field(source,p,"card_id"))),deck(text(source,field(source,p,"deck"))),face(ef_pending_face(source,p));
    const auto card=ev_card(t,id);require(card!=missing && text_is(t,obs_attr(t,card,"card_name"),"Wisdom of Elrond") && ctx_side(t,obs_attr(t,card,"side"))==0,"Wisdom pending card ID mismatch");
    require(contains_string(source,tc_hand(source,0),id),"Wisdom pending card absent from hand");
    require(deck=="strategy" && (face=="MUSTER" || face=="MUSTER_ARMY" || face=="EVENT" || face=="WILL_OF_WEST"),"Wisdom original funding metadata invalid");
    auto child=action_clone(source);auto& s=*child;rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));quiet_stable_action_gate(s,t);ring_waiver(s,0);ring_waiver(s,1);
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    em_wisdom(s,nation);ed_commit(s,side,id,deck,face);rr_set_attr(s,s.root,"_battle_resolved_attacker",rr_scalar(s,WOTR_NONE));return child;
}
