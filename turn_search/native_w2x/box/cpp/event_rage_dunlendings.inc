// Paid recruitment destination followed by one learned North/South split.
bool erg_supported(const Snapshot& t,uint64_t card){return card!=missing&&text_is(t,obs_attr(t,card,"card_name"),"Rage of the Dunlendings");}
bool erg_pending(const Snapshot& s){const auto p=obs_attr(s,s.root,"pending_card_option",false);return !ctx_none(s,p)&&text_is(s,field(s,p,"card_name",false),"Rage of the Dunlendings");}
bool erg_free(const Snapshot& s,const Snapshot& t,std::string_view key){
    const auto controls=obs_attr(s,s.root,"enemy_controlled");
    if(contains_string(s,ctx_side_value(s,controls,0),key))return contains_string(s,obs_attr(s,s.root,"besieged_strongholds"),key);
    const auto entries=child_edges(s,obs_attr(s,s.root,"armies"));
    for(size_t i=0;i<entries.size();i+=2)if(sanitize_name(text(s,entries[i]))==key)
        for(auto unit:child_edges(s,obs_attr(s,entries[i+1],"units")))if(ctx_unit_side(s,unit)==0)return false;
    const auto meta=ctx_get(t,field(t,t.root,"event_region_metadata"),key);
    if(meta!=missing&&contains_string(t,field(t,t.root,"fp_region_keys"),key)
       &&!contains_string(s,ctx_side_value(s,controls,1),key)
       &&(truth(t,ctx_get(t,meta,"is_stronghold"))||truth(t,ctx_get(t,meta,"is_city"))))return false;
    return true;
}
std::vector<std::string> erg_regions(const Snapshot& s,const Snapshot& t){
    if(mu_available(s,t,"ISENGARD","regular")<=0)return {};
    std::vector<std::string> result;
    auto add=[&](const std::string& key){if(key!="orthanc"&&std::find(result.begin(),result.end(),key)==result.end()&&erg_free(s,t,key))result.push_back(key);};
    for(const char* key:{"north dunland","south dunland","moria"})add(key);
    for(const char* key:{"north dunland","south dunland"})for(const auto& raw:esg_neighbors(s,key))add(sanitize_name(raw));
    return result;
}
std::vector<std::pair<int64_t,int64_t>> erg_splits(const Snapshot& s,const Snapshot& t,std::string_view destination){
    auto available=[&](const char* key){
        int64_t count=0;if(destination==key)return count;const auto army=hr_army(s,key);
        if(army!=missing)for(auto unit:child_edges(s,obs_attr(s,army,"units")))count+=enum_is(s,obs_attr(s,unit,"nation"),"game_state.Nation","ISENGARD");
        return count;
    };
    int64_t own=0;const auto army=hr_army(s,destination);
    if(army!=missing)for(auto unit:child_edges(s,obs_attr(s,army,"units")))own+=ctx_unit_side(s,unit)==1;
    const auto room=std::max<int64_t>(0,su_cap(s,t,destination,1)-own);
    const auto north=std::min<int64_t>(4,available("north dunland")),south=available("south dunland");
    std::vector<std::pair<int64_t,int64_t>> result;
    for(int64_t n=0;n<=north;++n)for(int64_t m=0;m<=std::min<int64_t>(south,4-n);++m)if(n+m<=room)result.emplace_back(n,m);
    return result;
}
uint64_t erg_grid(Snapshot& s,const Snapshot& t,std::string_view destination){
    std::vector<uint64_t> options;
    auto available=[&](const char* key,const char* kind){int64_t count=0;const auto army=hr_army(s,key);if(army!=missing&&destination!=key)for(auto unit:child_edges(s,obs_attr(s,army,"units")))count+=enum_is(s,obs_attr(s,unit,"nation"),"game_state.Nation","ISENGARD")&&enum_is(s,obs_attr(s,unit,"kind"),"game_state.UnitKind",kind);return count;};
    const auto nr=available("north dunland","REGULAR"),ne=available("north dunland","ELITE"),sr=available("south dunland","REGULAR"),se=available("south dunland","ELITE");
    for(const auto& [n,m]:erg_splits(s,t,destination))for(int64_t en=std::max<int64_t>(0,n-nr);en<=std::min(ne,n);++en)for(int64_t es=std::max<int64_t>(0,m-sr);es<=std::min(se,m);++es)options.push_back(ev_dict(s,{
        {"rage_split",ev_dict(s,{{"n_north",rr_scalar(s,WOTR_INT,n)},{"n_south",rr_scalar(s,WOTR_INT,m)},{"n_north_elite",rr_scalar(s,WOTR_INT,en)},{"n_south_elite",rr_scalar(s,WOTR_INT,es)}})},
        {"_rage_dest",rr_string(s,destination)}}));
    return cp_container(s,WOTR_LIST,options);
}
uint64_t erg_initial_options(Snapshot& s,const std::vector<std::string>& regions){
    std::vector<uint64_t> options;for(const auto& key:regions)options.push_back(ev_dict(s,{{"recruit_region",rr_string(s,key)}}));
    return cp_container(s,WOTR_LIST,options);
}
void erg_recruit(Snapshot& s,const Snapshot& t,std::string_view id,std::string_view region){
    if(region.empty()||mu_available(s,t,"ISENGARD","regular")<=0)return;
    const auto legal=erg_regions(s,t);if(std::find(legal.begin(),legal.end(),region)==legal.end())return;
    const bool first=er_place(s,t,"ISENGARD",region,"regular"),second=er_place(s,t,"ISENGARD",region,"regular");
    if(!first&&!second)return;
    rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,"Rage of the Dunlendings")},
        {"side",rr_string(s,"sh")},{"decider",rr_string(s,"sh")},{"direct",rr_scalar(s,WOTR_BOOL,1)},{"options",erg_grid(s,t,region)}}));
}
void erg_move(Snapshot& s,const Snapshot& t,std::string_view destination,int64_t north,int64_t south,int64_t north_elite,int64_t south_elite){
    const auto legal=erg_splits(s,t,destination);require(std::find(legal.begin(),legal.end(),std::make_pair(north,south))!=legal.end(),"Rage split exceeds movement or stacking limit");
    // Validate both sources before mutating either army.
    for(const auto& spec:std::vector<std::tuple<std::string,int64_t,int64_t>>{{"north dunland",north,north_elite},{"south dunland",south,south_elite}}){const auto& [key,wanted,elites]=spec;int64_t nr=0,ne=0;const auto army=hr_army(s,key);if(army!=missing&&key!=destination)for(auto unit:child_edges(s,obs_attr(s,army,"units")))if(enum_is(s,obs_attr(s,unit,"nation"),"game_state.Nation","ISENGARD"))(enum_is(s,obs_attr(s,unit,"kind"),"game_state.UnitKind","REGULAR")?nr:ne)++;require(elites>=0&&elites<=wanted&&elites<=ne&&wanted-elites<=nr,"Rage composition unavailable");}
    int64_t moved=0;
    for(const auto& [source,wanted]:std::vector<std::pair<std::string,int64_t>>{{"north dunland",north},{"south dunland",south}}){
        if(wanted<=0||source==destination)continue;
        const auto army=hr_army(s,source);if(army==missing)continue;
        std::vector<uint64_t> units;
        for(auto unit:child_edges(s,obs_attr(s,army,"units")))if(enum_is(s,obs_attr(s,unit,"nation"),"game_state.Nation","ISENGARD"))units.push_back(unit);
        std::stable_sort(units.begin(),units.end(),[&](auto a,auto b){
            const bool ar=enum_is(s,obs_attr(s,a,"kind"),"game_state.UnitKind","REGULAR"),br=enum_is(s,obs_attr(s,b,"kind"),"game_state.UnitKind","REGULAR");return ar&&!br;
        });
        const auto elite_wanted=source=="north dunland"?north_elite:south_elite;int64_t regular_left=wanted-elite_wanted,elite_left=elite_wanted;
        units.erase(std::remove_if(units.begin(),units.end(),[&](auto unit){auto& left=enum_is(s,obs_attr(s,unit,"kind"),"game_state.UnitKind","REGULAR")?regular_left:elite_left;if(left<=0)return true;--left;return false;}),units.end());
        for(auto unit:units){
            // Python list.remove erases the first dataclass-equal occurrence,
            // then appends the selected object. Preserve repeated aliases too.
            const auto list=obs_attr(s,army,"units");auto kept=child_edges(s,list);
            const auto kind=enum_member(s,obs_attr(s,unit,"kind"),"game_state.UnitKind");
            const auto it=std::find_if(kept.begin(),kept.end(),[&](auto other){return object_type(s,other)==object_type(s,unit)&&enum_is(s,obs_attr(s,other,"nation"),"game_state.Nation","ISENGARD")&&enum_is(s,obs_attr(s,other,"kind"),"game_state.UnitKind",kind);});
            require(it!=kept.end(),"Rage selected unit vanished");kept.erase(it);set_edges(s,list,kept);
            auto target=mu_army(s,destination);
            if(target==missing){
                const auto region=ctx_get(s,obs_attr(s,obs_attr(s,s.root,"board"),"_regions"),destination);
                require(region!=missing,"Rage destination has no Board region");target=am_new_army(s,region);am_insert(s,destination,target);
            }
            am_extend(s,target,"units",{unit});++moved;
        }
        if(!truth(s,obs_attr(s,army,"units"))&&!truth(s,obs_attr(s,army,"leaders"))){
            const auto entries=child_edges(s,obs_attr(s,s.root,"armies"));
            for(size_t i=0;i<entries.size();i+=2)if(entries[i+1]==army){const std::string key(text(s,entries[i]));am_erase(s,key);break;}
        }
    }
    am_orphans(s);require(moved>=north+south,"Rage split exceeds available units");
}
uint64_t erg_validate(const Snapshot& s,const Snapshot& t){
    require(erg_pending(s),"No Rage decision pending");const auto p=obs_attr(s,s.root,"pending_card_option");
    require(text_is(s,field(s,p,"side"),"sh")&&text_is(s,field(s,p,"decider"),"sh"),"Rage pending owner invalid");
    const auto id=field(s,p,"card_id"),card=ev_card(t,text(s,id));require(erg_supported(t,card),"Rage pending card mismatch");
    const auto options=child_edges(s,field(s,p,"options"));const bool direct=truth(s,field(s,p,"direct",false));
    if(!direct){
        /* Pending card choices retain their public identity after commitment. */
        require(text_is(s,field(s,p,"deck"),"strategy"),"Rage pending deck invalid");const auto face=text(s,field(s,p,"die_face"));
        require(face=="MUSTER"||face=="MUSTER_ARMY"||face=="EVENT","Rage original payment invalid");
        require(options.size()>=2,"Rage recruitment choices missing");
        for(auto option:options){require(typed(s,option,WOTR_DICT,"Rage target must be dict").count==2,"Rage target shape invalid");text(s,field(s,option,"recruit_region"));}
    }else{
        require(!options.empty(),"Rage split choices empty");const auto destination=text(s,field(s,options.front(),"_rage_dest"));
        // Raw menus retain stored choices; current room does not regenerate them.
        for(size_t i=0;i<options.size();++i){const auto choice=options[i],split=field(s,choice,"rage_split");
            require(typed(s,choice,WOTR_DICT,"Rage split option must be dict").count==4&&typed(s,split,WOTR_DICT,"Rage split must be dict").count==8
                &&text_is(s,field(s,choice,"_rage_dest"),destination)
                &&typed(s,field(s,split,"n_north"),WOTR_INT,"Rage North count must be integer").integer>=0
                &&typed(s,field(s,split,"n_south"),WOTR_INT,"Rage South count must be integer").integer>=0,"Rage split choices stale");
        }
    }
    return p;
}
std::unique_ptr<Snapshot> native_rage_choices(const Snapshot& source,const Snapshot& t,int side){
    require(side==1,"Rage choices belong to Shadow");erg_validate(source,t);auto result=std::make_unique<Snapshot>(source);auto& s=*result;
    const auto p=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t index=0;
    for(auto choice:child_edges(s,field(s,p,"options")))rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),
        ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",choice}})}));
    s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
void erg_gate(const Snapshot& s,const Snapshot& t,bool continuation,bool chained_overflow=false){
    quiet_action_gate(s,t,false,false,false,false,false,false,false,false,false,false,continuation,true,continuation,false,chained_overflow);
    ring_waiver(s,0);ring_waiver(s,1);
}
fm_Result apply_rage_event(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index,std::string_view id){
    require(side==1,"Rage belongs to Shadow");erg_gate(source,t,false);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==1,"Rage violates retained Ring actor");
    const auto card=ev_card(t,id);require(erg_supported(t,card)&&contains_string(source,tc_hand(source,1),id),"Rage card absent or invalid");
    require(ev_requirement(source,t,card),"Rage requirement false");const auto die=ev_payment(source,t,1,index,card);
    const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace"));const auto regions=erg_regions(source,t);
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());
    auto& s=*result.state;rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    rr_set_attr(s,ev_payment(s,t,1,index,card),"used",rr_scalar(s,WOTR_BOOL,1));
    if(regions.size()>=2){
        rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,"Rage of the Dunlendings")},
            {"side",rr_string(s,"sh")},{"decider",rr_string(s,"sh")},{"options",erg_initial_options(s,regions)},
            {"die_face",rr_string(s,face)},{"deck",rr_string(s,"strategy")}}));tc_open_hand_limit_discard(s);
    }else{erg_recruit(s,t,id,regions.empty()?std::string_view{}:std::string_view(regions.front()));esr_commit(s,t,id,"strategy",face);}
    am_Builder events;std::vector<uint64_t> rows;cau_finish(s,t,result.random.get(),-1,events,rows);
    events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
fm_Result apply_rage_choice(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index){
    require(side==1,"Rage choice belongs to Shadow");const auto p=erg_validate(source,t),options=field(source,p,"options");
    require(index>=0&&static_cast<uint64_t>(index)<source.nodes[options].count,"Rage choice index out of range");
    const bool direct=truth(source,field(source,p,"direct",false));const std::string id(text(source,field(source,p,"card_id")));
    if(direct){
        const auto tracking=obs_attr(source,source.root,"_overstack_regions",false);
        if(!ctx_none(source,tracking)){
            typed(source,tracking,WOTR_SET,"Rage overflow tracking must be a set");
            // Recruitment records the raw army key; upkeep may also record its
            // canonical spelling. Other overflow/stale keys can wait behind
            // this split. Preserve them for the ordinary resolver, which checks
            // actual excess after movement and never returns a unit twice.
            for(auto raw:child_edges(source,tracking)){
                const auto region=sanitize_name(text(source,raw));
                require(ctx_get(t,field(t,t.root,"board_regions"),region)!=missing,
                        "Rage overflow tracking references an unknown region");
            }
        }
    }
    const std::string face=direct?std::string{}:std::string(text(source,field(source,p,"die_face")));
    const auto oldmarker=obs_attr(source,source.root,"_battle_resolved_attacker",false);const int marker=ctx_none(source,oldmarker)?-1:ctx_side(source,oldmarker);
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;
    const auto choice=child_edges(s,field(s,obs_attr(s,s.root,"pending_card_option"),"options"))[index];
    rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));erg_gate(s,t,true,direct);
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    if(direct){
        const auto split=field(s,choice,"rage_split");const std::string destination(text(s,field(s,choice,"_rage_dest")));
        erg_move(s,t,destination,typed(s,field(s,split,"n_north"),WOTR_INT,"Rage total integer").integer,typed(s,field(s,split,"n_south"),WOTR_INT,"Rage total integer").integer,typed(s,field(s,split,"n_north_elite"),WOTR_INT,"Rage elite integer").integer,typed(s,field(s,split,"n_south_elite"),WOTR_INT,"Rage elite integer").integer);
        su_resolve_overstacks(s,t);
    }else if(ev_requirement(s,t,ev_card(t,id))){const std::string region(text(s,field(s,choice,"recruit_region")));erg_recruit(s,t,id,region);esr_commit(s,t,id,"strategy",face);}
    am_Builder events;std::vector<uint64_t> rows;cau_finish(s,t,result.random.get(),marker,events,rows);
    events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
