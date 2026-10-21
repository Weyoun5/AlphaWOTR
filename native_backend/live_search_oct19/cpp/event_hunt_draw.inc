// Funded Shadow Hunt-draw events. Original draws use choice/remove; Mithril
// replacements use the shared pop/redraw path and re-enter with an owned tile.
std::string ehd_kind(const Snapshot& t,uint64_t card){
    if(card==missing)return {};const auto name=obs_attr(t,card,"card_name");
    if(text_is(t,name,"Orc Patrol"))return "orc_patrol";
    if(text_is(t,name,"Isildur's Bane"))return "isildur";
    if(text_is(t,name,"Foul Thing from the Deep"))return "foul_thing";
    if(text_is(t,name,"The Breaking of the Fellowship"))return "breaking";return {};
}
bool ehd_stash_kind(std::string_view kind){return kind=="orc_patrol" || kind=="isildur" || kind=="foul_thing" || kind=="breaking";}
bool ehd_requirement(const Snapshot& s,const Snapshot& t,std::string_view kind){
    const auto fs=obs_attr(s,s.root,"fellowship");if(kind=="breaking")return !truth(s,obs_attr(s,fs,"hidden"));
    const auto meta=ctx_get(t,field(t,t.root,"event_region_metadata"),glb_fellowship_key(s,fs));
    return meta==missing || !(text_is(t,ctx_get(t,meta,"owner"),"free") && truth(t,ctx_get(t,meta,"is_settlement")));
}
std::vector<uint64_t> ehd_movable(const Snapshot& s){
    std::vector<uint64_t> result;for(auto c:child_edges(s,obs_attr(s,obs_attr(s,s.root,"fellowship"),"companions"))){const auto n=text(s,obs_attr(s,c,"name"));if(n!="Frodo" && n!="Sam" && n!="Gollum")result.push_back(c);}return result;
}
uint64_t ehd_bof_options(Snapshot& s){
    std::vector<uint64_t> opts;for(auto c:ehd_movable(s)){const auto names=rr_scalar(s,WOTR_LIST);set_edges(s,names,{obs_attr(s,c,"name")});opts.push_back(ev_dict(s,{{"companions",names}}));}
    const auto result=rr_scalar(s,WOTR_LIST);set_edges(s,result,opts);return result;
}
void ehd_foul_casualty(Snapshot& s,const Snapshot& t,void* rng,int64_t damage,am_Builder& events,std::vector<uint64_t>& rows){
    // hobbit_queue_repair_r1 mirror (production event_effects._foul_apply_after_reduction 2026-09-09; package Z 2026-09-11): "A guide may already
    // have separated while reducing this event. Even without a casualty, its placement remains a learned choice" - the frozen learned path pops
    // the hobbit placement queue (game_logic.pop_hobbit_place_queue = hd_pop_hobbit: no-op while a hunt decision is open) at BOTH no-casualty
    // returns: the damage<=0 return and the no-companions corruption return.
    if(damage<=0){hd_pop_hobbit(s);return;}const auto fs=obs_attr(s,s.root,"fellowship");std::vector<uint64_t> candidates;
    for(auto c:child_edges(s,obs_attr(s,fs,"companions")))if(!text_is(s,obs_attr(s,c,"name"),"Gollum"))candidates.push_back(c);
    if(candidates.empty()){hd_corruption(s,damage);hd_pop_hobbit(s);return;}
    const auto pick=hf_choice(rng,candidates.size(),true);const auto victim=candidates[pick];
    const auto name=std::string(text(s,obs_attr(s,victim,"name")));std::vector<uint64_t> names;for(auto c:candidates)names.push_back(events.str(text(s,obs_attr(s,c,"name"))));
    rows.push_back(events.row(-1,"random_draw",{events.str("ctx"),events.str("Foul Thing victim draw"),events.str("result"),events.str(name),events.str("pool"),events.container(WOTR_LIST,names)}));
    const auto level=ht_integer(s,victim,"level",1);const auto absorbed=std::min(damage,level?level:1);
    if((name=="Meriadoc" || name=="Peregrin") && !truth(s,obs_attr(s,fs,"mordor_entered"))){
        hc_remove(s,victim);cbp_companion_death(s,rng,true);ft_guide(s);
        rr_set_attr(s,fs,"corruption",rr_scalar(s,WOTR_INT,std::min<int64_t>(12,ht_integer(s,fs,"corruption")+std::max<int64_t>(0,absorbed-level))));hc_queue_hobbit(s,victim);
    }else hc_source_damage(s,rng,absorbed,victim,false);
    rr_set_attr(s,s.root,"_pending_hunt_damage",rr_scalar(s,WOTR_INT));hd_continue(s,t,damage-absorbed,true);
}
void ehd_foul_confront(Snapshot& s,const Snapshot& t,void* rng,uint64_t resume,int64_t damage,am_Builder& events,std::vector<uint64_t>& rows){
    const auto fs=obs_attr(s,s.root,"fellowship"),guide=obs_attr(s,fs,"guide");const auto ability=ctx_none(s,guide)?missing:obs_attr(s,guide,"guide_ability",false);
    if(damage>0 && ctx_none(s,obs_attr(s,s.root,"pending_card_option")) && (text_is(s,ability,"sacrifice_reduce_hunt") || hd_gollum_available(s,t))){
        const auto next=hpr_dict(s,ctx_none(s,resume)?std::vector<uint64_t>{}:child_edges(s,resume));fbr_put(s,next,"damage",rr_scalar(s,WOTR_INT,damage));
        const auto opts=rr_scalar(s,WOTR_LIST);set_edges(s,opts,{ev_dict(s,{{"use_foul_guide",rr_scalar(s,WOTR_INT,1)}}),ev_dict(s,{{"use_foul_guide",rr_scalar(s,WOTR_INT)}})});
        rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,"__foulguide__")},{"card_name",rr_string(s,"Foul Thing guide")},{"side",rr_string(s,"fp")},{"decider",rr_string(s,"fp")},{"direct",rr_scalar(s,WOTR_BOOL,1)},{"foul_guide",rr_scalar(s,WOTR_BOOL,1)},{"foul_resume",next},{"options",opts}}));return;
    }ehd_foul_casualty(s,t,rng,damage,events,rows);
}
void ehd_apply_tile(Snapshot& s,const Snapshot& t,void* rng,std::string_view kind,uint64_t choices,uint64_t tile,am_Builder& events,std::vector<uint64_t>& rows){
    const auto discard=obs_attr(s,s.root,"hunt_discard");auto tiles=child_edges(s,discard);tiles.push_back(tile);set_edges(s,discard,tiles);
    const bool inert=truth(s,obs_attr(s,tile,"is_eye")) || truth(s,obs_attr(s,tile,"is_fellowship_special"));int64_t damage=ht_integer(s,tile,"damage");
    const char* name=kind=="orc_patrol"?"Orc Patrol":kind=="isildur"?"Isildur's Bane":kind=="foul_thing"?"Foul Thing from the Deep":"The Breaking of the Fellowship";
    auto roll_damage=[&](){if(truth(s,obs_attr(s,tile,"is_dice_roll"))){uint32_t roll=0;hf_roll(rng,&roll,1,"shelob d6 damage");damage=roll;}};
    if(kind=="foul_thing"){if(inert)damage=0;else roll_damage();}
    const auto tile_name=obs_attr(s,tile,"special_name");rows.push_back(events.row(-1,"hunt_tile_draw",{events.str("tile_name"),events.str(truth(s,tile_name)?std::string(text(s,tile_name)):(truth(s,obs_attr(s,tile,"is_eye"))?"Eye":std::to_string(ht_integer(s,tile,"damage"))+(truth(s,obs_attr(s,tile,"reveal"))?"R":""))),events.str("ctx"),events.str(name),events.str("damage"),events.integer(damage)}));
    if(inert)return;if(kind!="foul_thing")roll_damage();const auto fs=obs_attr(s,s.root,"fellowship");
    if(kind=="breaking"){
        const auto movable=ehd_movable(s);if(damage>0 && !movable.empty())rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",field(s,choices,"_breaking_card_id")},{"card_name",rr_string(s,name)},{"side",rr_string(s,"sh")},{"decider",rr_string(s,"fp")},{"bof_separation",rr_scalar(s,WOTR_BOOL,1)},{"bof_n_left",rr_scalar(s,WOTR_INT,std::min<int64_t>(damage,movable.size()))},{"bof_region",rr_string(s,glb_fellowship_key(s,fs))},{"bof_mordor",rr_scalar(s,WOTR_BOOL,truth(s,obs_attr(s,fs,"mordor_entered")))},{"options",ehd_bof_options(s)}}));return;
    }
    ht_reveal(s,tile,truth(s,obs_attr(s,tile,"reveal")),true);
    if(kind=="isildur"){rr_set_attr(s,fs,"corruption",rr_scalar(s,WOTR_INT,std::min<int64_t>(12,ht_integer(s,fs,"corruption")+damage)));ht_settle_reveal(s);return;}
    if(kind=="orc_patrol"){rr_set_attr(s,s.root,"_pending_hunt_damage",rr_scalar(s,WOTR_INT,ht_integer(s,s.root,"_pending_hunt_damage")+damage));ht_settle_reveal(s);return;}
    if(damage<=0)return;const auto resume=ev_dict(s,{{"damage",rr_scalar(s,WOTR_INT,damage)},{"victim",rr_scalar(s,WOTR_NONE)}});
    auto reducer=hpr_card(s,"AXE_AND_BOW");if(reducer==missing)reducer=hpr_card(s,"HORN_OF_GONDOR");
    if(reducer!=missing && ctx_none(s,obs_attr(s,s.root,"pending_card_option"))){
        std::vector<uint64_t> reductions;
        for(const char* kind:{"AXE_AND_BOW","HORN_OF_GONDOR"}){const auto payer=hpr_card(s,kind);if(payer!=missing)reductions.push_back(ev_dict(s,{{"use_hunt_reduction",rr_scalar(s,WOTR_INT,1)},{"card_id",obs_attr(s,payer,"card_id")}}));}
        reductions.push_back(ev_dict(s,{{"use_hunt_reduction",rr_scalar(s,WOTR_INT)}}));
        const auto opts=rr_scalar(s,WOTR_LIST);set_edges(s,opts,reductions);
        rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",obs_attr(s,reducer,"card_id")},{"card_name",obs_attr(s,reducer,"card_name")},{"side",rr_string(s,"fp")},{"decider",rr_string(s,"fp")},{"direct",rr_scalar(s,WOTR_BOOL,1)},{"hunt_reduction",rr_scalar(s,WOTR_BOOL,1)},{"foul_resume",resume},{"options",opts}}));
    }else ehd_foul_confront(s,t,rng,ev_dict(s,{{"victim",rr_scalar(s,WOTR_NONE)}}),damage,events,rows);
}
void ehd_draw(Snapshot& s,const Snapshot& t,void* rng,std::string_view kind,uint64_t choices,am_Builder& events,std::vector<uint64_t>& rows,bool skip_wizard=false,uint64_t forced=missing){
    require(ehd_stash_kind(kind),"Unknown Hunt event continuation");
    if(forced==missing && !skip_wizard && hpr_offer_wizard(s,ev_dict(s,{{"apply_kind",rr_string(s,kind)},{"choices",choices}})))return;
    uint64_t tile=forced;if(tile==missing){
        const auto pool=obs_attr(s,s.root,"hunt_pool");if(!truth(s,pool))ht_recycle(s,rng,events,rows);if(!truth(s,pool))return;
        auto tiles=child_edges(s,pool);const auto pick=hf_pool(s,pool,rng,false);tile=tiles[pick];
        const auto first=std::find_if(tiles.begin(),tiles.end(),[&](uint64_t v){return ft_tile_equal(s,v,tile);});tiles.erase(first);set_edges(s,pool,tiles);
        if(hpr_offer_mithril(s,ev_dict(s,{{"tile",tile},{"apply_kind",rr_string(s,kind)},{"choices",choices}})))return;
    }ehd_apply_tile(s,t,rng,kind,choices,tile,events,rows);
}
void ehd_finish(Snapshot& s,const Snapshot& t,void* rng,am_Builder& events,std::vector<uint64_t>& rows){
    if(ht_integer(s,s.root,"_pending_hunt_damage")>0 && ctx_none(s,obs_attr(s,s.root,"pending_hunt")) && !hpr_offer_reduction(s))hd_settle(s,t);
    // Source common tail: hand limit, Palantir, then siege/control upkeep.
    tc_open_hand_limit_discard(s);epa_maybe(s);su_cleanup(s,t);
    ew_open(s,rng);fr_open(s,t,rng,events,rows);if(ctx_none(s,obs_attr(s,s.root,"pending_guide_change")) && !hc_guide_member(s) && truth(s,obs_attr(s,obs_attr(s,s.root,"fellowship"),"companions")))ft_guide(s);
    hpr_return_mithril(s,rng);fm_finish(s,t);epa_maybe(s);
}
fm_Result apply_hunt_event_action(const Snapshot& source,const Snapshot& t,const void* random_source,int32_t side,int64_t index,std::string_view id){
    require(side==1,"Hunt draw event belongs to Shadow");const auto definition=ev_card(t,id);const bool stable=(ehd_kind(t,definition)=="orc_patrol" || ehd_kind(t,definition)=="isildur" || ehd_kind(t,definition)=="foul_thing" || ehd_kind(t,definition)=="breaking");quiet_action_gate(source,t,false,false,false,false,false,false,false,false,true,false,false,stable);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained) || ctx_side(source,retained)==side,"Hunt event violates retained Ring actor");
    const auto card=ev_card(t,id);const auto kind=ehd_kind(t,card);require(!kind.empty() && ctx_side(t,obs_attr(t,card,"side"))==side,"Unsupported Hunt event");
    require(contains_string(source,tc_hand(source,side),id),"Hunt event absent from hand");require(ehd_requirement(source,t,kind),"Hunt event requirement false");
    const auto die=ev_payment(source,t,side,index,card);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace")),deck=ev_lower(text(t,obs_attr(t,card,"deck")));
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;am_Builder events;std::vector<uint64_t> rows;
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));rr_set_attr(s,ev_payment(s,t,side,index,card),"used",rr_scalar(s,WOTR_BOOL,1));
    const auto choices=kind=="breaking"?ev_dict(s,{{"_breaking_card_id",rr_string(s,id)}}):ev_dict(s,{});const auto fs=obs_attr(s,s.root,"fellowship"),guide=obs_attr(s,fs,"guide");
    if(kind=="breaking" && !ctx_none(s,guide) && text(s,obs_attr(s,guide,"name")).find("Gollum")!=std::string_view::npos)hd_corruption(s,1);else ehd_draw(s,t,result.random.get(),kind,choices,events,rows);
    ed_commit(s,side,id,deck,face);rr_set_attr(s,s.root,"_battle_resolved_attacker",rr_scalar(s,WOTR_NONE));ehd_finish(s,t,result.random.get(),events,rows);
    events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
bool ehd_pending(const Snapshot& s){const auto p=obs_attr(s,s.root,"pending_card_option",false);return !ctx_none(s,p) && (truth(s,field(s,p,"bof_separation",false)) || truth(s,field(s,p,"foul_guide",false)));}
std::unique_ptr<Snapshot> native_hunt_event_choice_actions(const Snapshot& source){
    require(ehd_pending(source),"No Hunt event choice pending");auto out=std::make_unique<Snapshot>(source);auto& s=*out;const auto p=obs_attr(s,s.root,"pending_card_option");
    require(text_is(s,field(s,p,"decider"),"fp"),"Hunt event choice belongs to FP");std::vector<uint64_t> rows;int64_t index=0;
    for(auto choice:child_edges(s,field(s,p,"options"))){const auto row=rr_scalar(s,WOTR_TUPLE);set_edges(s,row,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",choice}})});rows.push_back(row);}
    s.root=rr_scalar(s,WOTR_LIST);set_edges(s,s.root,rows);return ac_compact(s);
}
fm_Result apply_hunt_event_choice(const Snapshot& source,const Snapshot& t,const void* random_source,int64_t index){
    require(ehd_pending(source),"No Hunt event choice pending");hc_gate(source,false,true,false,true);const auto p=obs_attr(source,source.root,"pending_card_option");const auto options=child_edges(source,field(source,p,"options"));
    require(index>=0 && static_cast<uint64_t>(index)<options.size(),"Hunt event choice index out of range");
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;am_Builder events;std::vector<uint64_t> rows;
    const auto pending=obs_attr(s,s.root,"pending_card_option"),choice=child_edges(s,field(s,pending,"options"))[index],fs=obs_attr(s,s.root,"fellowship");
    rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    if(truth(s,field(s,pending,"bof_separation",false))){
        const auto names=child_edges(s,field(s,choice,"companions"));require(names.size()==1,"Breaking choice requires one companion");const auto name=sanitize_name(text(s,names.front()));uint64_t companion=missing;
        for(auto c:child_edges(s,obs_attr(s,fs,"companions")))if(sanitize_name(text(s,obs_attr(s,c,"name")))==name){companion=c;break;}
        require(companion!=missing,"Breaking companion absent");hc_remove(s,companion);
        if(!truth(s,field(s,pending,"bof_mordor"))){
            const auto key=std::string(text(s,field(s,pending,"bof_region")));auto army=mu_army(s,key);
            if(army==missing){const auto region=ctx_get(s,obs_attr(s,obs_attr(s,s.root,"board"),"_regions"),key);require(region!=missing,"Breaking region missing");army=mu_object(s,"game_state.Army",{{"region",region},{"units",rr_scalar(s,WOTR_LIST)},{"leaders",rr_scalar(s,WOTR_LIST)}});const auto armies=obs_attr(s,s.root,"armies");auto entries=child_edges(s,armies);entries.push_back(rr_string(s,key));entries.push_back(army);set_edges(s,armies,entries);}
            const auto leaders=obs_attr(s,army,"leaders");auto figures=child_edges(s,leaders);figures.push_back(companion);set_edges(s,leaders,figures);
        }
        cbp_companion_death(s,result.random.get(),false);const auto remaining=hpr_integer(s,pending,"bof_n_left",1)-1;
        if(remaining>0 && !ehd_movable(s).empty()){const auto next=hpr_dict(s,child_edges(s,pending));fbr_put(s,next,"bof_n_left",rr_scalar(s,WOTR_INT,remaining));fbr_put(s,next,"options",ehd_bof_options(s));rr_set_attr(s,s.root,"pending_card_option",next);}
        else ft_guide(s);
    }else{
        const auto resume=field(s,pending,"foul_resume");const auto damage=hpr_integer(s,resume,"damage");
        if(truth(s,field(s,choice,"use_foul_guide",false))){const auto guide=obs_attr(s,fs,"guide");
            if(text_is(s,obs_attr(s,guide,"guide_ability",false),"no_reveal_on_damage"))ht_commit_reveal(s);
            else{if(hc_guide_member(s))hc_remove(s,guide);if(truth(s,obs_attr(s,fs,"mordor_entered")))cbp_companion_death(s,result.random.get(),false);else hc_queue_hobbit(s,guide);ft_guide(s);}
            const auto next=hpr_dict(s,child_edges(s,resume));fbr_put(s,next,"victim",field(s,resume,"victim",false)==missing?rr_scalar(s,WOTR_NONE):field(s,resume,"victim"));ehd_foul_confront(s,t,result.random.get(),next,std::max<int64_t>(0,damage-1),events,rows);
        }else ehd_foul_casualty(s,t,result.random.get(),damage,events,rows);
        rr_set_attr(s,s.root,"_pending_hunt_damage",rr_scalar(s,WOTR_INT));
    }
    ehd_finish(s,t,result.random.get(),events,rows);cau_gandalf(s);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
