// Two independent joins. Select all cargo before moving any figures.
bool esl_supported(const Snapshot& t,uint64_t card){return card!=missing&&text_is(t,obs_attr(t,card,"card_name"),"The Shadow Lengthens");}
bool esl_pending(const Snapshot& s){const auto p=obs_attr(s,s.root,"pending_card_option",false);return !ctx_none(s,p)&&text_is(s,field(s,p,"card_name",false),"The Shadow Lengthens");}
uint64_t esl_moves(Snapshot& s,const std::vector<esg_Choice>& moves){
    std::vector<uint64_t> rows;for(const auto& m:moves)rows.push_back(cp_container(s,WOTR_LIST,{rr_string(s,m.from),rr_string(s,m.to),m.regular<0?rr_scalar(s,WOTR_NONE):rr_scalar(s,WOTR_INT,m.regular),m.elite<0?rr_scalar(s,WOTR_NONE):rr_scalar(s,WOTR_INT,m.elite),ctx_none(s,m.cargo)?rr_scalar(s,WOTR_NONE):m.cargo}));return cp_container(s,WOTR_LIST,rows);
}
std::vector<esg_Choice> esl_read_moves(const Snapshot& s,uint64_t value){
    std::vector<esg_Choice> result;if(ctx_none(s,value))return result;
    typed(s,value,WOTR_LIST,"Shadow Lengthens history must be a list");
    for(auto row:child_edges(s,value)){const auto parts=child_edges(s,row);require(s.nodes[row].tag==WOTR_LIST&&parts.size()==5,"Shadow Lengthens history shape invalid");
        result.push_back({std::string(text(s,parts[0])),std::string(text(s,parts[1])),ctx_none(s,parts[2])?-1:typed(s,parts[2],WOTR_INT,"Invalid regular history").integer,ctx_none(s,parts[3])?-1:typed(s,parts[3],WOTR_INT,"Invalid elite history").integer,parts[4]});}
    return result;
}
uint64_t esl_options(Snapshot& s,const Snapshot& t,const std::vector<esg_Choice>& moves){
    std::set<std::string> excluded;for(const auto& m:moves)excluded.insert(m.from);std::vector<uint64_t> rows;
    for(const auto& c:esg_options(s,t,2,excluded)){auto option=esg_option(s,c);if(!moves.empty()){auto entries=child_edges(s,option);entries.push_back(rr_string(s,"_tsl_moves"));entries.push_back(esl_moves(s,moves));set_edges(s,option,entries);}rows.push_back(option);}
    if(moves.empty())rows.push_back(ev_dict(s,{{"_tsl_done",rr_scalar(s,WOTR_BOOL,1)}}));else if(!rows.empty())rows.push_back(ev_dict(s,{{"_tsl_done",rr_scalar(s,WOTR_BOOL,1)},{"_tsl_moves",esl_moves(s,moves)}}));
    return cp_container(s,WOTR_LIST,rows);
}
uint64_t esl_witness(Snapshot& s,const esg_Choice& c,const std::vector<uint64_t>& units,const std::vector<uint64_t>& leaders){
    std::map<std::pair<std::string,std::string>,int64_t> uc;std::map<std::tuple<std::string,std::string,std::string>,int64_t> lc;
    for(auto u:units)++uc[{std::string(enum_member(s,obs_attr(s,u,"nation"),"game_state.Nation")),std::string(enum_member(s,obs_attr(s,u,"kind"),"game_state.UnitKind"))}];
    for(auto l:leaders){std::string type(object_type(s,l));type=type.substr(type.find_last_of('.')+1);const auto nation=obs_attr(s,l,"nation",false),name=obs_attr(s,l,"name",false);
        const std::string n=ctx_none(s,nation)?"":std::string(enum_member(s,nation,"game_state.Nation"));const std::string named=name==missing?"":ctx_none(s,name)?"None":std::string(text(s,name));++lc[{type,n,named}];}
    std::vector<uint64_t> ur,lr;for(const auto& [key,count]:uc)ur.push_back(cp_container(s,WOTR_TUPLE,{rr_string(s,key.first),rr_string(s,key.second),rr_scalar(s,WOTR_INT,count)}));
    for(const auto& [key,count]:lc)lr.push_back(cp_container(s,WOTR_TUPLE,{rr_string(s,std::get<0>(key)),rr_string(s,std::get<1>(key)),rr_string(s,std::get<2>(key)),rr_scalar(s,WOTR_INT,count)}));
    return cp_container(s,WOTR_TUPLE,{rr_string(s,c.from),rr_string(s,c.to),cp_container(s,WOTR_TUPLE,ur),cp_container(s,WOTR_TUPLE,lr)});
}
void esl_execute(Snapshot& s,const Snapshot& t,const std::vector<esg_Choice>& moves){
    // esg_options excludes besieged garrisons and admits source-legal besiegers.
    // Public cau_finish retains the source timing of siege reconciliation.
    rr_set_attr(s,s.root,"_card_army_move_witness",cp_container(s,WOTR_TUPLE,{rr_string(s,"SHADOW_Strategy_The_Shadow_Lengthens"),cp_container(s,WOTR_TUPLE,{})}));
    struct Plan{esg_Choice c;std::string key;uint64_t src;std::vector<uint64_t> units,leaders;std::vector<std::string> mids;};std::vector<Plan> plans;
    const auto legal=esg_options(s,t,2,{});std::set<std::string> sources;
    for(const auto& c:moves){require(sources.insert(c.from).second,"Shadow Lengthens sources must be independent");
        require(std::any_of(legal.begin(),legal.end(),[&](const esg_Choice& x){return x.from==c.from&&x.to==c.to&&x.regular==c.regular&&x.elite==c.elite;}),"Shadow Lengthens staged move is no longer legal");
        const auto src=esg_army(s,c.from);Plan p{c,am_key(s,c.from),src,{},{},esg_mids(s,c.from,c.to,2)};int64_t r=c.regular,e=c.elite;
        std::vector<uint64_t> eligible;for(auto u:child_edges(s,obs_attr(s,src,"units")))if(ctx_unit_side(s,u)==1)eligible.push_back(u);
        if(ctx_none(s,c.cargo)){const auto route=sj_select(s,t,c.from,c.to,2,eligible);eligible=route.first;p.mids=route.second;}
        for(auto u:eligible){auto& left=enum_is(s,obs_attr(s,u,"kind"),"game_state.UnitKind","ELITE")?e:r;if(left<0||left>0){p.units.push_back(u);if(left>0)--left;}}
        for(auto l:child_edges(s,obs_attr(s,src,"leaders")))if(!am_saruman(s,l)&&am_leader_side(s,l)==1)p.leaders.push_back(l);
        if(!ctx_none(s,c.cargo)){const auto exact=cag_selected(s,c.from,c.cargo);p.units=exact.first;p.leaders=exact.second;const auto route=sj_select(s,t,c.from,c.to,2,p.units,true);require(route.first.size()==p.units.size(),"Shadow Lengthens cargo cannot cross political borders");p.mids=route.second;}
        require(!p.units.empty(),"Shadow Lengthens selected no units");plans.push_back(std::move(p));
    }
    for(const auto& p:plans){std::vector<uint64_t> units,leaders;const std::set<uint64_t> moved(p.units.begin(),p.units.end()),moved_leaders(p.leaders.begin(),p.leaders.end());
        for(auto u:child_edges(s,obs_attr(s,p.src,"units")))if(!moved.count(u))units.push_back(u);for(auto l:child_edges(s,obs_attr(s,p.src,"leaders")))if(!moved_leaders.count(l))leaders.push_back(l);
        am_assign_list(s,p.src,"units",units);am_assign_list(s,p.src,"leaders",leaders);if(units.empty()&&leaders.empty())am_erase(s,p.key);
    }
    std::vector<uint64_t> witness;
    for(const auto& p:plans){auto dst=esg_army(s,p.c.to);if(dst==missing){const auto regions=obs_attr(s,obs_attr(s,s.root,"board"),"_regions"),region=ctx_get(s,regions,p.c.to);require(region!=missing,"Shadow Lengthens destination region missing");dst=am_new_army(s,region);am_insert(s,p.c.to,dst);}
        am_extend(s,dst,"units",p.units);am_extend(s,dst,"leaders",p.leaders);witness.push_back(esl_witness(s,p.c,p.units,p.leaders));
        int64_t count=0;for(auto u:child_edges(s,obs_attr(s,dst,"units")))count+=ctx_unit_side(s,u)==1;if(count>10){auto tracking=obs_attr(s,s.root,"_overstack_regions",false);if(ctx_none(s,tracking)){tracking=cp_container(s,WOTR_SET,{});rr_set_attr(s,s.root,"_overstack_regions",tracking);}me_set_member(s,tracking,p.c.to,true);}
        for(const auto& region:p.mids){movement_capture_effects(s,t,region,1);const auto nation=ctx_get(t,field(t,t.root,"movement_region_nations"),region);if(!ctx_none(t,nation)&&fp_nation(ctx_nation(t,nation)))wt_activate_if_inactive(s,ctx_nation(t,nation),"attack");}
    }
    am_orphans(s);rr_set_attr(s,s.root,"_card_army_move_witness",cp_container(s,WOTR_TUPLE,{rr_string(s,"SHADOW_Strategy_The_Shadow_Lengthens"),cp_container(s,WOTR_TUPLE,witness)}));
}
void esl_effect(Snapshot& s,const Snapshot& t,std::string_view id,uint64_t option){
    const bool done=truth(s,field(s,option,"_tsl_done",false));if(!done){option=cag_stage(s,t,id,option,text(s,field(s,option,"from_region")));if(option==missing)return;}
    auto moves=esl_read_moves(s,field(s,option,"_tsl_moves",false));
    if(!done){const auto r=field(s,option,"n_regular",false),e=field(s,option,"n_elite",false);moves.push_back({std::string(text(s,field(s,option,"from_region"))),std::string(text(s,field(s,option,"to_region"))),ctx_none(s,r)?-1:s.nodes[r].integer,ctx_none(s,e)?-1:s.nodes[e].integer,field(s,option,"cargo",false)});}
    require(moves.size()<=2,"Shadow Lengthens allows at most two sources");if(!done&&moves.size()<2){const auto options=esl_options(s,t,moves);if(truth(s,options)){
        rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,"The Shadow Lengthens")},{"side",rr_string(s,"sh")},{"decider",rr_string(s,"sh")},{"direct",rr_scalar(s,WOTR_BOOL,1)},{"options",options}}));return;}}
    esl_execute(s,t,moves);
}
fm_Result apply_shadow_lengthens_event(const Snapshot& source,const Snapshot& t,const void* random_source,int32_t side,int64_t index,std::string_view id,bool mouth_muster=false){
    require(side==1,"Shadow Lengthens belongs to Shadow");quiet_stable_action_gate(source,t);ring_waiver(source,0);ring_waiver(source,1);const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==1,"Shadow Lengthens retained Ring actor mismatch");
    const auto card=ev_card(t,id);require(esl_supported(t,card)&&contains_string(source,tc_hand(source,1),id),"Shadow Lengthens card absent");require(ev_requirement(source,t,card),"Shadow Lengthens requirement false");const auto die=ev_payment(source,t,1,index,card,mouth_muster);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace")),deck(ev_lower(text(t,obs_attr(t,card,"deck"))));
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;am_Builder events;std::vector<uint64_t> rows;
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));ev_consume_payment(s,t,1,index,card,mouth_muster);const auto options=esl_options(s,t,{});
    if(s.nodes[options].count==1){esl_execute(s,t,{});esr_commit(s,t,id,deck,face);cau_finish(s,t,result.random.get(),-1,events,rows);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;}
    if(truth(s,options)){rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,"The Shadow Lengthens")},{"side",rr_string(s,"sh")},{"decider",rr_string(s,"sh")},{"options",options},{"die_face",rr_string(s,face)},{"deck",rr_string(s,deck)}}));tc_open_hand_limit_discard(s);}
    else esr_commit(s,t,id,deck,face);cau_finish(s,t,result.random.get(),-1,events,rows);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
std::unique_ptr<Snapshot> native_shadow_lengthens_choice_actions(const Snapshot& source,int32_t side){
    require(side==1&&esl_pending(source),"No Shadow Lengthens choice for this actor");auto out=std::make_unique<Snapshot>(source);auto& s=*out;const auto p=obs_attr(s,s.root,"pending_card_option");require(text_is(s,field(s,p,"side"),"sh")&&text_is(s,field(s,p,"decider"),"sh"),"Shadow Lengthens owner mismatch");std::vector<uint64_t> rows;int64_t index=0;
    for(auto choice:child_edges(s,field(s,p,"options")))rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",choice}})}));s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
fm_Result apply_shadow_lengthens_choice(const Snapshot& source,const Snapshot& t,const void* random_source,int32_t side,int64_t index){
    require(side==1&&esl_pending(source),"No Shadow Lengthens choice for this actor");const auto p=obs_attr(source,source.root,"pending_card_option");const auto options=child_edges(source,field(source,p,"options"));require(index>=0&&static_cast<uint64_t>(index)<options.size(),"Shadow Lengthens index invalid");
    const std::string id(text(source,field(source,p,"card_id")));require(esl_supported(t,ev_card(t,id)),"Shadow Lengthens pending card mismatch");const bool direct=truth(source,field(source,p,"direct",false));std::string deck,face;if(!direct){require(contains_string(source,tc_hand(source,1),id),"Uncommitted Shadow Lengthens absent");deck=std::string(text(source,field(source,p,"deck")));face=std::string(text(source,field(source,p,"die_face")));}
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;am_Builder events;std::vector<uint64_t> rows;
    const auto pending=obs_attr(s,s.root,"pending_card_option"),current_options=field(s,pending,"options");require(text_is(s,field(s,pending,"side"),"sh")&&text_is(s,field(s,pending,"decider"),"sh"),"Shadow Lengthens owner mismatch");
    const auto history=esl_read_moves(s,field(s,child_edges(s,current_options).front(),"_tsl_moves",false));require(history.size()<=1,"Shadow Lengthens pending history invalid");if(ctx_none(s,field(s,pending,"cargo_source",false))){const auto regenerated=esl_options(s,t,history);require(tail_equal(s,current_options,regenerated),"Shadow Lengthens choices are stale");}
    const auto choice=child_edges(s,current_options)[index],marker=obs_attr(s,s.root,"_battle_resolved_attacker",false);const int marker_side=ctx_none(s,marker)?-1:ctx_side(s,marker);
    rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));quiet_action_gate(s,t,false,false,false,false,false,false,false,false,false,false,false,true,true);ring_waiver(s,0);ring_waiver(s,1);rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));esl_effect(s,t,id,choice);if(!direct)esr_commit(s,t,id,deck,face);else su_resolve_overstacks(s,t);
    cau_finish(s,t,result.random.get(),marker_side,events,rows);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
