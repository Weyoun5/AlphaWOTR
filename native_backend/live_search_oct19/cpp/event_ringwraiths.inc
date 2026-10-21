// Ringwraiths event and shared Ringwraiths/Black Captain flight continuations.
// Reuses nz_options/nz_transfer and the existing free army-followup executor.
bool erw_supported(const Snapshot& t,uint64_t card){return card!=missing&&text_is(t,obs_attr(t,card,"card_name"),"The Ringwraiths Are Abroad");}
bool erw_pending(const Snapshot& s){
    const auto p=obs_attr(s,s.root,"pending_card_option",false);if(ctx_none(s,p)||!text_is(s,field(s,p,"card_name",false),"__nazgul_moves__"))return false;
    const auto options=field(s,p,"options",false);if(ctx_none(s,options)||!truth(s,options))return false;
    const auto fu=field(s,child_edges(s,options).front(),"_then_followup",false);return !ctx_none(s,fu)&&(text_is(s,field(s,fu,"kind",false),"nazgul_armies")||text_is(s,field(s,fu,"kind",false),"wk_army"));
}
void erw_grant(Snapshot& s,std::string_view kind="nazgul_armies",int64_t moves=2){rr_set_attr(s,s.root,"pending_card_followup",ev_dict(s,{{"kind",rr_string(s,kind)},{"side",rr_string(s,"sh")},{"moves_left",rr_scalar(s,WOTR_INT,moves)},{"exclude",cp_container(s,WOTR_SET,{})},{"moved_any",rr_scalar(s,WOTR_BOOL,0)}}));}
bool erw_open(Snapshot& s,const Snapshot& t,std::string_view id,const nz_Arrivals& arr,uint64_t followup){
    const auto options=nz_options(s,t,arr);auto rows=child_edges(s,options);if(rows.empty())return false;
    rows.push_back(ev_dict(s,{{"_naz_done",rr_scalar(s,WOTR_BOOL,1)}}));
    // Source shares this exact dictionary between every option and Done.
    for(auto row:rows)cp_put(s,row,"_then_followup",followup);set_edges(s,options,rows);
    rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,"__nazgul_moves__")},{"side",rr_string(s,"sh")},{"decider",rr_string(s,"sh")},{"direct",rr_scalar(s,WOTR_BOOL,1)},{"options",options}}));return true;
}
void erw_no_flight_fallback(Snapshot& s,const Snapshot& t){
    struct Figure{std::string from;uint64_t figure;};std::vector<Figure> figures;const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));
    for(size_t i=0;i<armies.size();i+=2)for(auto figure:child_edges(s,obs_attr(s,armies[i+1],"leaders")))if(af_nazgul_figure(s,figure))figures.push_back({sanitize_name(text(s,armies[i])),figure});
    if(figures.empty())return;
    // Only reached when the source shared flight builder offers no options.
    // Match source stable-first scoring and first-unit army ownership exactly.
    std::string destination;int64_t best=-1;
    auto own=[&](uint64_t army,int side){const auto units=child_edges(s,obs_attr(s,army,"units"));return !units.empty()&&ctx_unit_side(s,units.front())==side;};
    for(size_t i=0;i<armies.size();i+=2){
        if(!own(armies[i+1],1))continue;const auto from=sanitize_name(text(s,armies[i]));
        for(const auto& adjacent:fr_neighbors(s,from)){const auto defender=mu_army(s,sanitize_name(adjacent));if(defender==missing||!own(defender,0))continue;
            const int64_t score=s.nodes[obs_attr(s,defender,"units")].count+s.nodes[obs_attr(s,armies[i+1],"units")].count;if(score>best){best=score;destination=from;}}
    }
    if(destination.empty())for(size_t i=0;i<armies.size();i+=2)if(own(armies[i+1],1)){destination=sanitize_name(text(s,armies[i]));break;}
    if(destination.empty())return;
    for(const auto& figure:figures)if(figure.from!=destination)nz_transfer(s,t,figure.from,destination,{figure.figure},false);
    erw_grant(s);
}
fm_Result apply_ringwraiths_event(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index,std::string_view id){
    require(side==1,"Ringwraiths event belongs to Shadow");quiet_action_gate(source,t,false,false,false,false,false,false,false,false,true,false,false,true);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==side,"Ringwraiths event violates retained Ring actor");
    const auto card=ev_card(t,id);require(erw_supported(t,card)&&ctx_side(t,obs_attr(t,card,"side"))==side,"Unsupported Ringwraiths event");require(contains_string(source,tc_hand(source,side),id),"Ringwraiths card absent from hand");require(ev_requirement(source,t,card),"Ringwraiths requirement false");
    const auto die=ev_payment(source,t,side,index,card);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace")),deck(ev_lower(text(t,obs_attr(t,card,"deck"))));
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;
    rr_set_attr(s,s.root,"_casualty_operations",cp_container(s,WOTR_LIST,{}));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));rr_set_attr(s,ev_payment(s,t,side,index,card),"used",rr_scalar(s,WOTR_BOOL,1));
    const auto followup=ev_dict(s,{{"kind",rr_string(s,"nazgul_armies")},{"moves",rr_scalar(s,WOTR_INT,2)}});
    if(!erw_open(s,t,id,{{"__wk_ok__",1}},followup))erw_no_flight_fallback(s,t);
    ed_commit(s,side,id,deck,face);am_Builder events;std::vector<uint64_t> rows;cau_finish(s,t,result.random.get(),-1,events,rows);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
uint64_t erw_validate_pending(const Snapshot& s,const Snapshot& t,int side){
    require(side==1&&erw_pending(s),"No Shadow card-funded Nazgul chain pending");const auto p=obs_attr(s,s.root,"pending_card_option");
    const bool ringwraiths=erw_supported(t,ev_card(t,text(s,field(s,p,"card_id")))),captain=text_is(s,field(s,p,"card_id"),"__nazgul_moves__");
    require((ringwraiths||captain)&&text_is(s,field(s,p,"side"),"sh")&&text_is(s,field(s,p,"decider"),"sh")&&truth(s,field(s,p,"direct")),"Card-funded Nazgul identity or owner mismatch");
    for(auto choice:child_edges(s,field(s,p,"options"))){
        const auto fu=field(s,choice,"_then_followup",false);require(!ctx_none(s,fu)&&typed(s,fu,WOTR_DICT,"Card army grant must be dictionary").count==4,"Card army grant metadata changed");
        const auto kind=text(s,field(s,fu,"kind"));const auto moves=typed(s,field(s,fu,"moves"),WOTR_INT,"Card army move grant must be integer").integer;
        require((ringwraiths&&kind=="nazgul_armies"&&moves==2)||(captain&&kind=="wk_army"&&moves==1),"Card army grant kind or count changed");require(field(s,choice,"_naz_post",false)==missing,"Army-grant flights cannot carry a Hunt consequence");
    }return p;
}
std::unique_ptr<Snapshot> native_ringwraiths_choice_actions(const Snapshot& source,const Snapshot& t,int side){
    erw_validate_pending(source,t,side);Snapshot s(source);const auto p=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t index=0;
    for(auto option:child_edges(s,field(s,p,"options")))rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",option}})}));s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
fm_Result apply_ringwraiths_choice(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index){
    const auto pending=erw_validate_pending(source,t,side),options=field(source,pending,"options");require(index>=0&&static_cast<uint64_t>(index)<source.nodes[options].count,"Ringwraiths choice index invalid");
    const std::string id(text(source,field(source,pending,"card_id")));const int marker=ctx_side(source,obs_attr(source,source.root,"_battle_resolved_attacker",false));
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;
    const auto p=obs_attr(s,s.root,"pending_card_option"),choice=child_edges(s,field(s,p,"options"))[index],fu=field(s,choice,"_then_followup");
    rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));quiet_action_gate(s,t,false,false,false,false,false,false,false,false,true,false,false,true,true);ring_waiver(s,0);ring_waiver(s,1);rr_set_attr(s,s.root,"_casualty_operations",cp_container(s,WOTR_LIST,{}));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    if(truth(s,field(s,choice,"_naz_done",false)))erw_grant(s,text(s,field(s,fu,"kind")),s.nodes[field(s,fu,"moves")].integer);
    else{
        nz_Arrivals arr;const auto entries=child_edges(s,field(s,choice,"_naz_arr"));for(size_t i=0;i<entries.size();i+=2){const auto n=typed(s,entries[i+1],WOTR_INT,"Ringwraiths arrival count must be integer").integer;require(n>=0,"Ringwraiths arrival count negative");arr.emplace_back(std::string(text(s,entries[i])),n);}
        require(nz_get(arr,"__wk_ok__")==1&&!nz_get(arr,"__mo_ok__")&&!nz_get(arr,"__mo__"),"Card-funded flights cannot include the Mouth");cp_erase(s,choice,"_then_followup");require(nz_current_choice(s,s,choice,t,arr),"Ringwraiths flight is stale or malformed");
        const std::string from(text(s,field(s,choice,"from_region"))),to(text(s,field(s,choice,"to_region")));const bool witch=truth(s,field(s,choice,"include_wk",false));uint64_t figure=missing;
        for(auto leader:child_edges(s,obs_attr(s,mu_army(s,from),"leaders")))if(witch?nz_witch_chain(s,leader):nz_plain(s,leader)){figure=leader;break;}require(figure!=missing,"Ringwraiths flight figure absent");
        nz_transfer(s,t,from,to,{figure},false);if(witch)nz_set(arr,"__wk__",1);else nz_set(arr,to,nz_get(arr,to)+1);
        if(!erw_open(s,t,id,arr,fu))erw_grant(s,text(s,field(s,fu,"kind")),s.nodes[field(s,fu,"moves")].integer);
    }
    am_Builder events;std::vector<uint64_t> rows;ehd_finish(s,t,result.random.get(),events,rows);cau_finish(s,t,result.random.get(),marker,events,rows);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
