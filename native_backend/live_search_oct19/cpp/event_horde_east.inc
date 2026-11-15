// Ordered Horde targets; recruitment, overflow and bonus timing reuse shared code.
bool ehe_supported(const Snapshot& t,uint64_t card){return card!=missing&&text_is(t,obs_attr(t,card,"card_name"),"Horde from the East");}
bool ehe_pending(const Snapshot& s){const auto p=obs_attr(s,s.root,"pending_card_option",false);return !ctx_none(s,p)&&text_is(s,field(s,p,"card_name",false),"Horde from the East");}
std::vector<std::string> ehe_regions(const Snapshot& s,const Snapshot& t){
    std::vector<std::string> result;
    for(const char* key:{"east rhun","south rhun","khand","far harad"})
        if(ctx_get(t,field(t,t.root,"board_regions"),key)!=missing&&evt_horde_free(s,key))result.emplace_back(key);
    return result;
}
uint64_t ehe_options(Snapshot& s,const std::vector<std::string>& regions){
    std::vector<uint64_t> rows;for(const auto& key:regions)rows.push_back(ev_dict(s,{{"region",rr_string(s,key)}}));
    return cp_container(s,WOTR_LIST,rows);
}
bool ehe_effect(Snapshot& s,const Snapshot& t,std::string region){
    const auto options=ehe_regions(s,t);region=sanitize_name(region);
    if(region.empty()&&!options.empty())region=options.front();
    bool printed=false;for(const char* key:{"east rhun","south rhun","khand","far harad"})printed|=region==key;
    if(!region.empty()&&!printed)return true; // Source rejects its target without requesting a refund.
    int placed=0;if(!region.empty())for(int i=0;i<5;++i)placed+=er_place(s,t,"SOUTHRON_EASTERLINGS",region,"regular");
    if(placed)return true;
    // The public wrapper discards valid zero-effect plays. A stored target
    // which became invalid while other legal targets exist retains the card.
    return options.empty()||std::find(options.begin(),options.end(),region)!=options.end();
}
fm_Result apply_horde_event(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index,std::string_view id){
    require(side==1,"Horde belongs to Shadow");quiet_action_gate(source,t,false,false,false,false,false,false,false,false,false,false,false,true);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==1,"Horde violates retained Ring actor");
    const auto card=ev_card(t,id);require(ehe_supported(t,card)&&contains_string(source,tc_hand(source,1),id),"Horde card absent");require(ev_requirement(source,t,card),"Horde requirement false");
    const auto die=ev_payment(source,t,1,index,card);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace"));
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());
    auto& s=*result.state;rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    rr_set_attr(s,ev_payment(s,t,1,index,card),"used",rr_scalar(s,WOTR_BOOL,1));const auto regions=ehe_regions(s,t);
    if(regions.size()>1){
        rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,"Horde from the East")},
            {"side",rr_string(s,"sh")},{"decider",rr_string(s,"sh")},{"options",ehe_options(s,regions)},{"die_face",rr_string(s,face)},{"deck",rr_string(s,"strategy")}}));tc_open_hand_limit_discard(s);
    }else if(ehe_effect(s,t,regions.empty()?std::string{}:regions.front()))esr_commit(s,t,id,"strategy",face);
    am_Builder events;std::vector<uint64_t> rows;cau_finish(s,t,result.random.get(),-1,events,rows);
    events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
uint64_t ehe_validate(const Snapshot& s,const Snapshot& t){
    require(ehe_pending(s),"No Horde choice pending");const auto p=obs_attr(s,s.root,"pending_card_option");
    require(text_is(s,field(s,p,"side"),"sh")&&text_is(s,field(s,p,"decider"),"sh")&&!truth(s,field(s,p,"direct",false)),"Horde pending owner or stage invalid");
    const auto id=text(s,field(s,p,"card_id"));require(ehe_supported(t,ev_card(t,id)),"Horde pending card identity invalid");/* package R: hand membership is not a menu precondition (frozen card_option menu, ai.py 7764-7770) */
    require(text_is(s,field(s,p,"deck"),"strategy"),"Horde pending deck invalid");const auto face=text(s,field(s,p,"die_face"));
    require(face=="MUSTER"||face=="MUSTER_ARMY"||face=="EVENT","Horde pending payment invalid");
    for(auto option:child_edges(s,field(s,p,"options"))){require(typed(s,option,WOTR_DICT,"Horde option must be a dictionary").count==2,"Horde option shape invalid");text(s,field(s,option,"region"));}
    return p;
}
std::unique_ptr<Snapshot> native_horde_choices(const Snapshot& source,const Snapshot& t,int side){
    require(side==1,"Horde choices belong to Shadow");ehe_validate(source,t);auto result=std::make_unique<Snapshot>(source);auto& s=*result;
    const auto p=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t index=0;
    for(auto choice:child_edges(s,field(s,p,"options")))rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),
        ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",choice}})}));
    s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
fm_Result apply_horde_choice(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index){
    require(side==1,"Horde choice belongs to Shadow");const auto p=ehe_validate(source,t);const auto options=child_edges(source,field(source,p,"options"));
    require(index>=0&&static_cast<uint64_t>(index)<options.size(),"Horde choice index exceeds menu");
    const std::string id(text(source,field(source,p,"card_id"))),face(text(source,field(source,p,"die_face"))),region(text(source,field(source,options[index],"region")));
    const auto oldmarker=obs_attr(source,source.root,"_battle_resolved_attacker",false);const int marker=ctx_none(source,oldmarker)?-1:ctx_side(source,oldmarker);
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;
    rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));quiet_action_gate(s,t,false,false,false,false,false,false,false,false,false,false,false,true);ring_waiver(s,0);ring_waiver(s,1);
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    if(ev_requirement(s,t,ev_card(t,id))&&ehe_effect(s,t,region))esr_commit(s,t,id,"strategy",face);
    am_Builder events;std::vector<uint64_t> rows;cau_finish(s,t,result.random.get(),marker,events,rows);
    events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
