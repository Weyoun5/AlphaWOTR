// Public quiet path, including the automatic singleton option.
// Up to four staged movements: select original cargo before moving any figures.
bool etsm_supported(const Snapshot& t,uint64_t card){return card!=missing&&text_is(t,obs_attr(t,card,"card_name"),"The Shadow is Moving");}
bool etsm_pending(const Snapshot& s){const auto p=obs_attr(s,s.root,"pending_card_option",false);return !ctx_none(s,p)&&text_is(s,field(s,p,"card_name",false),"The Shadow is Moving");}

uint64_t etsm_moves(Snapshot& s,const std::vector<esg_Choice>& moves){
    std::vector<uint64_t> rows;for(const auto& m:moves)rows.push_back(cp_container(s,WOTR_TUPLE,{rr_string(s,m.from),rr_string(s,m.to),ctx_none(s,m.cargo)?rr_scalar(s,WOTR_NONE):m.cargo}));return cp_container(s,WOTR_TUPLE,rows);
}
std::vector<esg_Choice> etsm_read_moves(const Snapshot& s,uint64_t value){
    std::vector<esg_Choice> result;if(ctx_none(s,value))return result;
    require(s.nodes[value].tag==WOTR_TUPLE||s.nodes[value].tag==WOTR_LIST,"Shadow is Moving history must be a sequence");
    for(auto row:child_edges(s,value)){const auto parts=child_edges(s,row);require((s.nodes[row].tag==WOTR_TUPLE||s.nodes[row].tag==WOTR_LIST)&&parts.size()==3,"Shadow is Moving history shape invalid");result.push_back({std::string(text(s,parts[0])),std::string(text(s,parts[1])),-1,-1,parts[2]});}
    require(result.size()<=3,"Shadow is Moving history exceeds three selections");return result;
}
std::vector<esg_Choice> etsm_legal(const Snapshot& s,const Snapshot& t,const std::set<std::string>& excluded={}){
    std::vector<esg_Choice> rows;const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));
    for(size_t i=0;i<armies.size();i+=2){const auto from=sanitize_name(text(s,armies[i]));if(excluded.count(from)||esg_garrison(s,t,from)||!esg_own(s,armies[i+1]))continue;
        for(const auto& raw:esg_neighbors(s,from)){const auto to=sanitize_name(raw);const auto dest=esg_army(s,to);const bool siege=contains_string(s,obs_attr(s,s.root,"besieged_strongholds",false),to);
            if(siege?(!esg_own(s,dest)||c9_garrison(s,t,to)==1):esg_enemy(s,dest))continue;rows.push_back({from,to});}}
    return rows;
}
uint64_t etsm_options(Snapshot& s,const Snapshot& t,const std::vector<esg_Choice>& moves){
    std::set<std::string> excluded;const auto legal=etsm_legal(s,t);for(const auto& m:moves){require(excluded.insert(m.from).second,"Shadow is Moving repeated source");require(std::any_of(legal.begin(),legal.end(),[&](const esg_Choice& c){return c.from==m.from&&c.to==m.to;}),"Shadow is Moving original cargo no longer available");}std::vector<uint64_t> rows;
    const auto history=moves.empty()?missing:etsm_moves(s,moves);
    for(const auto& c:etsm_legal(s,t,excluded)){auto option=esg_option(s,c);if(history!=missing){auto entries=child_edges(s,option);entries.push_back(rr_string(s,"_tsim_moves"));entries.push_back(history);set_edges(s,option,entries);}rows.push_back(option);}
    if(history==missing)rows.push_back(ev_dict(s,{{"_tsim_done",rr_scalar(s,WOTR_BOOL,1)}}));else if(!rows.empty())rows.push_back(ev_dict(s,{{"_tsim_done",rr_scalar(s,WOTR_BOOL,1)},{"_tsim_moves",history}}));
    return cp_container(s,WOTR_LIST,rows);
}
void etsm_execute(Snapshot& s,const Snapshot& t,const std::vector<esg_Choice>& moves){
    // etsm_legal validates all original sources and destinations before any
    // cargo moves. The existing public tail reconciles lifted sieges.
    struct Plan{esg_Choice c;std::string key;uint64_t src;std::vector<uint64_t> units,leaders;std::vector<std::string> mids;};std::vector<Plan> plans;
    const auto legal=etsm_legal(s,t);std::set<std::string> sources;
    for(const auto& c:moves){require(sources.insert(c.from).second,"Shadow is Moving sources must be independent");
        require(std::any_of(legal.begin(),legal.end(),[&](const esg_Choice& x){return x.from==c.from&&x.to==c.to&&x.regular==c.regular&&x.elite==c.elite;}),"Shadow is Moving staged move is no longer legal");
        const auto src=esg_army(s,c.from);Plan p{c,am_key(s,c.from),src,{},{},std::vector<std::string>{}};int64_t r=c.regular,e=c.elite;
        for(auto u:child_edges(s,obs_attr(s,src,"units")))if(ctx_unit_side(s,u)==1){auto& left=enum_is(s,obs_attr(s,u,"kind"),"game_state.UnitKind","ELITE")?e:r;if(c.regular<0||left>0){p.units.push_back(u);if(left>0)--left;}}
        for(auto l:child_edges(s,obs_attr(s,src,"leaders")))if(!am_saruman(s,l)&&am_leader_side(s,l)==1)p.leaders.push_back(l);if(!ctx_none(s,c.cargo)){const auto exact=cag_selected(s,c.from,c.cargo);p.units=exact.first;p.leaders=exact.second;}require(!p.units.empty(),"Shadow is Moving selected no units");plans.push_back(std::move(p));
    }
    for(const auto& p:plans){std::vector<uint64_t> units,leaders;const std::set<uint64_t> moved(p.units.begin(),p.units.end()),moved_leaders(p.leaders.begin(),p.leaders.end());
        for(auto u:child_edges(s,obs_attr(s,p.src,"units")))if(!moved.count(u))units.push_back(u);for(auto l:child_edges(s,obs_attr(s,p.src,"leaders")))if(!moved_leaders.count(l))leaders.push_back(l);
        am_assign_list(s,p.src,"units",units);am_assign_list(s,p.src,"leaders",leaders);if(units.empty()&&leaders.empty())am_erase(s,p.key);
    }
    std::vector<uint64_t> witness;
    for(const auto& p:plans){auto dst=esg_army(s,p.c.to);if(dst==missing){const auto regions=obs_attr(s,obs_attr(s,s.root,"board"),"_regions"),region=ctx_get(s,regions,p.c.to);require(region!=missing,"Shadow is Moving destination region missing");dst=am_new_army(s,region);am_insert(s,p.c.to,dst);}
        am_extend(s,dst,"units",p.units);am_extend(s,dst,"leaders",p.leaders);witness.push_back(esl_witness(s,p.c,p.units,p.leaders));
        int64_t count=0;for(auto u:child_edges(s,obs_attr(s,dst,"units")))count+=ctx_unit_side(s,u)==1;if(count>10){auto tracking=obs_attr(s,s.root,"_overstack_regions",false);if(ctx_none(s,tracking)){tracking=cp_container(s,WOTR_SET,{});rr_set_attr(s,s.root,"_overstack_regions",tracking);}me_set_member(s,tracking,p.c.to,true);}
        movement_entry_effects(s,t,p.c.to,1,p.leaders);if(!esg_enemy(s,dst))movement_capture_effects(s,t,p.c.to,1);
    }
    am_orphans(s);rr_set_attr(s,s.root,"_card_army_move_witness",cp_container(s,WOTR_TUPLE,{rr_string(s,"SHADOW_Strategy_The_Shadow_is_Moving"),cp_container(s,WOTR_TUPLE,witness)}));
}

void etsm_effect(Snapshot& s,const Snapshot& t,std::string_view id,uint64_t option){
    const bool done=truth(s,field(s,option,"_tsim_done",false));if(!done){option=cag_stage(s,t,id,option,text(s,field(s,option,"from_region")));if(option==missing)return;}
    auto moves=etsm_read_moves(s,field(s,option,"_tsim_moves",false));
    if(!done)moves.push_back({std::string(text(s,field(s,option,"from_region"))),std::string(text(s,field(s,option,"to_region"))),-1,-1,field(s,option,"cargo",false)});
    std::sort(moves.begin(),moves.end(),[](const esg_Choice& a,const esg_Choice& b){return std::tie(a.from,a.to)<std::tie(b.from,b.to);});
    require(moves.size()<=4,"Shadow is Moving allows at most four sources");if(!done&&moves.size()<4){const auto options=etsm_options(s,t,moves);if(truth(s,options)){
        rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,"The Shadow is Moving")},{"side",rr_string(s,"sh")},{"decider",rr_string(s,"sh")},{"direct",rr_scalar(s,WOTR_BOOL,1)},{"options",options}}));return;}}
    etsm_execute(s,t,moves);
}
fm_Result apply_shadow_is_moving_event(const Snapshot& source,const Snapshot& t,const void* random_source,int32_t side,int64_t index,std::string_view id,bool mouth_muster=false){
    require(side==1,"Shadow is Moving belongs to Shadow");quiet_stable_action_gate(source,t);ring_waiver(source,0);ring_waiver(source,1);const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==1,"Shadow is Moving retained Ring actor mismatch");
    const auto card=ev_card(t,id);require(etsm_supported(t,card)&&contains_string(source,tc_hand(source,1),id),"Shadow is Moving card absent");require(ev_requirement(source,t,card),"Shadow is Moving requirement false");const auto die=ev_payment(source,t,1,index,card,mouth_muster);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace")),deck(ev_lower(text(t,obs_attr(t,card,"deck"))));
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;am_Builder events;std::vector<uint64_t> rows;
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));ev_consume_payment(s,t,1,index,card,mouth_muster);const auto options=etsm_options(s,t,{});
    if(s.nodes[options].count==1){etsm_execute(s,t,{});esr_commit(s,t,id,deck,face);}
    else if(truth(s,options)){rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,"The Shadow is Moving")},{"side",rr_string(s,"sh")},{"decider",rr_string(s,"sh")},{"options",options},{"die_face",rr_string(s,face)},{"deck",rr_string(s,deck)}}));tc_open_hand_limit_discard(s);}
    else esr_commit(s,t,id,deck,face);cau_finish(s,t,result.random.get(),-1,events,rows);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
std::unique_ptr<Snapshot> native_shadow_is_moving_choice_actions(const Snapshot& source,int32_t side){
    require(side==1&&etsm_pending(source),"No Shadow is Moving choice for this actor");auto out=std::make_unique<Snapshot>(source);auto& s=*out;const auto p=obs_attr(s,s.root,"pending_card_option");require(text_is(s,field(s,p,"side"),"sh")&&text_is(s,field(s,p,"decider"),"sh"),"Shadow is Moving owner mismatch");std::vector<uint64_t> rows;int64_t index=0;
    for(auto choice:child_edges(s,field(s,p,"options")))rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",choice}})}));s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
fm_Result apply_shadow_is_moving_choice(const Snapshot& source,const Snapshot& t,const void* random_source,int32_t side,int64_t index){
    require(side==1&&etsm_pending(source),"No Shadow is Moving choice for this actor");const auto p=obs_attr(source,source.root,"pending_card_option");const auto options=child_edges(source,field(source,p,"options"));require(index>=0&&static_cast<uint64_t>(index)<options.size(),"Shadow is Moving index invalid");
    const std::string id(text(source,field(source,p,"card_id")));require(etsm_supported(t,ev_card(t,id)),"Shadow is Moving pending card mismatch");const bool direct=truth(source,field(source,p,"direct",false));std::string deck,face;if(!direct){require(contains_string(source,tc_hand(source,1),id),"Uncommitted Shadow is Moving absent");deck=std::string(text(source,field(source,p,"deck")));face=std::string(text(source,field(source,p,"die_face")));}
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;am_Builder events;std::vector<uint64_t> rows;
    const auto pending=obs_attr(s,s.root,"pending_card_option"),current_options=field(s,pending,"options");require(text_is(s,field(s,pending,"side"),"sh")&&text_is(s,field(s,pending,"decider"),"sh"),"Shadow is Moving owner mismatch");
    const auto history=etsm_read_moves(s,field(s,child_edges(s,current_options).front(),"_tsim_moves",false));require(history.size()<=3,"Shadow is Moving pending history invalid");if(ctx_none(s,field(s,pending,"cargo_source",false))){const auto regenerated=etsm_options(s,t,history);require(tail_equal(s,current_options,regenerated),"Shadow is Moving choices are stale");}
    const auto choice=child_edges(s,current_options)[index],marker=obs_attr(s,s.root,"_battle_resolved_attacker",false);const int marker_side=ctx_none(s,marker)?-1:ctx_side(s,marker);
    rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));quiet_action_gate(s,t,false,false,false,false,false,false,false,false,false,false,false,true,true);ring_waiver(s,0);ring_waiver(s,1);rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));etsm_effect(s,t,id,choice);if(!direct)esr_commit(s,t,id,deck,face);else su_resolve_overstacks(s,t);
    cau_finish(s,t,result.random.get(),marker_side,events,rows);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
