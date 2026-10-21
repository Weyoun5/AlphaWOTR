// Spirit's player target and five dice, then shared owner losses.
bool esp_supported(const Snapshot& t,uint64_t card){return card!=missing&&text_is(t,obs_attr(t,card,"card_name"),"The Spirit of Mordor");}
bool esp_pending(const Snapshot& s){const auto p=obs_attr(s,s.root,"pending_card_option",false);return !ctx_none(s,p)&&!truth(s,field(s,p,"card_casualty",false))&&text_is(s,field(s,p,"card_name",false),"The Spirit of Mordor");}
bool esp_target(const Snapshot& s,uint64_t army){
    if(army==missing)return false;std::set<std::string> nations;
    for(auto u:child_edges(s,obs_attr(s,army,"units"))){const std::string nation(enum_member(s,obs_attr(s,u,"nation"),"game_state.Nation"));if(!fp_nation(nation))nations.insert(nation);}
    return nations.size()>=2;
}
uint64_t esp_options(Snapshot& s){
    std::vector<uint64_t> options;const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));
    for(size_t i=0;i<armies.size();i+=2)if(esp_target(s,armies[i+1]))options.push_back(ev_dict(s,{{"target_region",rr_string(s,sanitize_name(text(s,armies[i])))}}));
    return cp_container(s,WOTR_LIST,options);
}
void esp_effect(Snapshot& s,const Snapshot& t,std::string_view id,std::string_view target,void* rng,am_Builder& events,std::vector<uint64_t>& rows){
    const auto region=sanitize_name(target);if(!esp_target(s,mu_army(s,region)))return;
    std::array<uint32_t,5> rolls{};hf_roll(rng,rolls.data(),rolls.size(),"card effect");
    int64_t hits=0;std::vector<uint64_t> faces;for(auto roll:rolls){hits+=roll>=5;faces.push_back(events.integer(roll));}
    rows.push_back(events.row(-1,"dice_roll",{events.str("faces"),events.container(WOTR_LIST,faces),events.str("ctx"),events.str("The Spirit of Mordor roll"),events.str("side"),rr_scalar(events.out,WOTR_NONE)}));
    if(hits>0)ecc_route(s,t,region,1,hits,id,"The Spirit of Mordor",rng);
}
fm_Result apply_spirit_event(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index,std::string_view id){
    require(side==0,"The Spirit of Mordor belongs to FP");quiet_stable_action_gate(source,t);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==0,"Spirit violates retained Ring actor");
    const auto card=ev_card(t,id);require(esp_supported(t,card)&&contains_string(source,tc_hand(source,0),id),"Spirit absent from FP hand");require(ev_requirement(source,t,card),"Spirit needs multi-nation Shadow army");
    const auto die=ev_payment(source,t,0,index,card);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace"));
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;
    rr_set_attr(s,s.root,"_casualty_operations",cp_container(s,WOTR_LIST,{}));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));rr_set_attr(s,ev_payment(s,t,0,index,card),"used",rr_scalar(s,WOTR_BOOL,1));
    const auto options=esp_options(s);const auto choices=child_edges(s,options);am_Builder events;std::vector<uint64_t> rows;
    if(choices.size()>1){rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,"The Spirit of Mordor")},{"side",rr_string(s,"fp")},{"decider",rr_string(s,"fp")},{"options",options},{"die_face",rr_string(s,face)},{"deck",rr_string(s,"strategy")}}));tc_open_hand_limit_discard(s);}
    else if(choices.size()==1){const std::string region(text(s,field(s,choices.front(),"target_region")));esp_effect(s,t,id,region,result.random.get(),events,rows);er_commit(s,t,id,"strategy",face);}
    else er_commit(s,t,id,"strategy",face); // Package W (P4): no target = the frozen zero-effect play
    cau_finish(s,t,result.random.get(),-1,events,rows);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
uint64_t esp_validate_pending(const Snapshot& s,const Snapshot& t){
    require(esp_pending(s),"No Spirit target pending");const auto p=obs_attr(s,s.root,"pending_card_option"),card=ev_card(t,text(s,field(s,p,"card_id")));
    require(esp_supported(t,card)&&text_is(s,field(s,p,"side"),"fp")&&text_is(s,field(s,p,"decider"),"fp")&&!truth(s,field(s,p,"direct",false)),"Spirit pending identity or owner mismatch");
    const auto face=text(s,field(s,p,"die_face"));require(text_is(s,field(s,p,"deck"),"strategy")&&(face=="ARMY"||face=="MUSTER_ARMY"||face=="EVENT"||face=="WILL_OF_WEST"),"Spirit pending funding invalid");
    const auto choices=child_edges(s,field(s,p,"options"));require(choices.size()>1,"Spirit modal options invalid");std::set<std::string> regions;
    for(auto choice:choices){require(typed(s,choice,WOTR_DICT,"Spirit choice must be dictionary").count==2,"Spirit stored choice shape invalid");const std::string region(text(s,field(s,choice,"target_region")));require(!region.empty()&&sanitize_name(region)==region&&regions.insert(region).second,"Spirit stored target malformed");}
    return p;
}
std::unique_ptr<Snapshot> native_spirit_choices(const Snapshot& source,const Snapshot& t,int side){
    require(side==0,"Spirit target belongs to FP");esp_validate_pending(source,t);Snapshot s(source);const auto p=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t index=0;
    for(auto choice:child_edges(s,field(s,p,"options")))rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",choice}})}));s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
fm_Result apply_spirit_choice(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index){
    require(side==0,"Spirit target belongs to FP");const auto pending=esp_validate_pending(source,t);const auto options=child_edges(source,field(source,pending,"options"));require(index>=0&&static_cast<uint64_t>(index)<options.size(),"Spirit choice index invalid");
    const std::string id(text(source,field(source,pending,"card_id"))),face(text(source,field(source,pending,"die_face"))),region(text(source,field(source,options[index],"target_region")));const int marker=ctx_side(source,obs_attr(source,source.root,"_battle_resolved_attacker",false));
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;
    rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));quiet_stable_action_gate(s,t);ring_waiver(s,0);ring_waiver(s,1);rr_set_attr(s,s.root,"_casualty_operations",cp_container(s,WOTR_LIST,{}));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    am_Builder events;std::vector<uint64_t> rows;
    // The public wrapper rechecks the global requirement; an individually
    // stale stored target instead spends the card without a roll.
    if(ev_requirement(s,t,ev_card(t,id))){esp_effect(s,t,id,region,result.random.get(),events,rows);er_commit(s,t,id,"strategy",face);}
    cau_finish(s,t,result.random.get(),marker,events,rows);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
