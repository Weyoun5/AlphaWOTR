// Event hits reuse the battle allocator's live reserves and physical witnesses.
// Continuations are admitted per card: Faramir, Spirit, Dreadful, Valinor, Dead Men.
void edm_effect(Snapshot&,const Snapshot&,std::string_view,uint64_t,void*,am_Builder&,std::vector<uint64_t>&);
void erv_effect(Snapshot&,const Snapshot&,std::string_view,std::string_view,void*,am_Builder&,std::vector<uint64_t>&);
void eff_casualty_resume(Snapshot& s,const Snapshot& t,std::string_view region,void* rng,am_Builder& events,std::vector<uint64_t>& rows);
void eea_effect(Snapshot&,const Snapshot&,std::string_view,bool,void*,am_Builder&,std::vector<uint64_t>&);
bool ecc_supported(const Snapshot& t,uint64_t card){return card!=missing && (eea_supported(t,card) || text_is(t,obs_attr(t,card,"card_name"),"Faramir's Rangers") || text_is(t,obs_attr(t,card,"card_name"),"The Spirit of Mordor") || text_is(t,obs_attr(t,card,"card_name"),"Dreadful Spells") || text_is(t,obs_attr(t,card,"card_name"),"Return to Valinor") || text_is(t,obs_attr(t,card,"card_name"),"Dead Men of Dunharrow"));}
bool ecc_pending(const Snapshot& s){const auto p=obs_attr(s,s.root,"pending_card_option",false);return !ctx_none(s,p) && truth(s,field(s,p,"card_casualty",false));}
uint64_t ecc_options(Snapshot& s,std::string_view region,int side,int64_t hits){
    std::vector<uint64_t> options;
    for(const auto& op:ca_operations(s,region,side,hits))options.push_back(ev_dict(s,{{"casualty_operation",rr_string(s,op.operation)},{"nation",mu_enum(s,"game_state.Nation",op.nation)},{"target_region",rr_string(s,sanitize_name(region))}}));
    return cp_container(s,WOTR_LIST,options);
}
void ecc_reconcile(Snapshot& s,std::string_view region,int side,std::string_view name,void* rng){
    const bool spare=name=="Faramir's Rangers" || name=="The Spirit of Mordor" || name=="Dreadful Spells" || name=="Return to Valinor";
    if(!spare){cbe_reconcile(s,region,side,rng);return;}
    const auto army=mu_army(s,sanitize_name(region));if(army==missing)return;const auto part=ca_partition(s,army,region,side);
    std::vector<uint64_t> dead,survivors;
    for(auto leader:part.leaders){if(object_type(s,leader)=="game_state.Leader")dead.push_back(leader);else survivors.push_back(leader);}
    crs_book_figures(s,dead,rng);survivors.insert(survivors.end(),part.rear_leaders.begin(),part.rear_leaders.end());
    rr_set_attr(s,army,"leaders",cp_container(s,WOTR_LIST,survivors));crs_drop_empty(s,region,army);
}
void ecc_aftermath(Snapshot& s,const Snapshot& t,std::string_view region,int side,std::string_view name,void* rng){
    const auto key=sanitize_name(region);auto army=mu_army(s,key);
    auto alive=[&](uint64_t a){if(a==missing)return false;for(auto u:child_edges(s,obs_attr(s,a,"units")))if(ctx_unit_side(s,u)==side)return true;return false;};
    if(army!=missing && !alive(army)){ecc_reconcile(s,key,side,name,rng);army=mu_army(s,key);}
    am_orphans(s);if(army==missing)return;
    const auto sieges=obs_attr(s,s.root,"besieged_strongholds",false);
    if(contains_string(s,sieges,key) && !alive(army) && c9_garrison(s,t,key)==side){me_set_member(s,sieges,key,false);crs_capture(s,t,key,1-side);}
}
bool ecc_route(Snapshot& s,const Snapshot& t,std::string_view region,int side,int64_t hits,std::string_view id,std::string_view name,void* rng){
    require(side==0 || side==1,"Card casualty side invalid");require(hits>=0 && hits<=1048576,"Card casualty hit budget unsupported");
    const auto definition=ev_card(t,id);require(ecc_supported(t,definition) && text_is(t,obs_attr(t,definition,"card_name"),name),"Card casualty continuation is not ported");
    const auto key=sanitize_name(region);if(hits<=0 || mu_army(s,key)==missing)return true;
    const auto owner=rr_string(s,side==0?"fp":"sh");const auto pending=ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,name)},{"card_casualty",rr_scalar(s,WOTR_BOOL,1)},{"direct",rr_scalar(s,WOTR_BOOL,1)},{"side",owner},{"decider",owner},{"cas_region",rr_string(s,key)},{"cas_side",mu_enum(s,"game_state.Side",side==0?"FREE_PEOPLES":"SHADOW")},{"cas_hits",rr_scalar(s,WOTR_INT,hits)}});
    if(name=="Return to Valinor")cp_put(s,pending,"cas_eligible_nations",cp_container(s,WOTR_TUPLE,{mu_enum(s,"game_state.Nation","ELVES")}));
    rr_set_attr(s,s.root,"pending_card_option",pending);const auto remaining=ca_forced(s,t,key,side,hits);
    if(remaining<=0){rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));ecc_aftermath(s,t,key,side,name,rng);return true;}
    cp_put(s,pending,"cas_hits",rr_scalar(s,WOTR_INT,remaining));cp_put(s,pending,"options",ecc_options(s,key,side,remaining));return false;
}
uint64_t ecc_validate_pending(const Snapshot& s,const Snapshot& t,int side){
    require(side==0 || side==1,"Card casualty actor invalid");require(ecc_pending(s),"No card casualty pending");const auto p=obs_attr(s,s.root,"pending_card_option");
    require(truth(s,field(s,p,"direct",false)) && text_is(s,field(s,p,"side"),side==0?"fp":"sh") && text_is(s,field(s,p,"decider"),side==0?"fp":"sh") && ctx_side(s,field(s,p,"cas_side"))==side,"Card casualty belongs to another actor");
    const auto card=ev_card(t,text(s,field(s,p,"card_id")));require(ecc_supported(t,card) && text_is(s,field(s,p,"card_name"),text(t,obs_attr(t,card,"card_name"))),"Card casualty continuation is not ported");
    if(text_is(s,field(s,p,"card_name"),"Return to Valinor")){
        const auto eligible=field(s,p,"cas_eligible_nations",false);require(side==0&&eligible!=missing&&typed(s,eligible,WOTR_TUPLE,"Valinor eligible nations must be tuple").count==1&&enum_member(s,child_edges(s,eligible).front(),"game_state.Nation")=="ELVES","Valinor casualties must be restricted to Elves");
        const auto region=text(s,field(s,p,"cas_region"));require(region=="rivendell"||region=="lorien"||region=="grey havens"||region=="woodland realm","Valinor casualty region is not a printed stronghold");
    }
    if(text_is(s,field(s,p,"card_name"),"Dead Men of Dunharrow")){
        const auto region=text(s,field(s,p,"cas_region"));require(side==1&&(region=="erech"||region=="lamedon"||region=="pelargir")&&ctx_none(s,field(s,p,"cas_eligible_nations",false)),"Dead Men casualty window must target a printed Shadow army");
    }
    const auto hits=typed(s,field(s,p,"cas_hits"),WOTR_INT,"Card casualty hits must be integer").integer;require(hits>0 && hits<=1048576,"Card casualty pending hit budget invalid");return p;
}
std::unique_ptr<Snapshot> native_card_casualty_actions(const Snapshot& source,const Snapshot& t,int side){
    ecc_validate_pending(source,t,side);auto out=std::make_unique<Snapshot>(source);auto& s=*out;const auto p=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t index=0;
    // Python's public menu exposes stored options; execution revalidates them.
    for(auto choice:child_edges(s,field(s,p,"options")))rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",choice}})}));
    s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
fm_Result apply_card_casualty_choice(const Snapshot& source,const Snapshot& t,const void* random_source,int side,int64_t index){
    const auto old=ecc_validate_pending(source,t,side);const auto options=child_edges(source,field(source,old,"options"));require(index>=0 && static_cast<uint64_t>(index)<options.size(),"Card casualty option index invalid");
    const std::string region(text(source,field(source,old,"cas_region"))),name(text(source,field(source,old,"card_name")));const auto hits=typed(source,field(source,old,"cas_hits"),WOTR_INT,"Card casualty hits must be integer").integer;
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;
    const auto p=obs_attr(s,s.root,"pending_card_option"),choice=child_edges(s,field(s,p,"options"))[index];const auto live=ecc_options(s,region,side,hits);bool valid=false;for(auto option:child_edges(s,live))valid|=tail_equal(s,option,choice);require(valid,"Card casualty choice is not currently available");
    const auto marker=obs_attr(s,s.root,"_battle_resolved_attacker",false);const int marker_side=ctx_none(s,marker)?-1:ctx_side(s,marker);
    const bool dreadful=name=="Dreadful Spells",valinor=name=="Return to Valinor",ents=eea_supported(t,ev_card(t,text(source,field(source,old,"card_id"))));
    const auto queued_palantir=obs_attr(s,s.root,"_palantir_draw_pending",false);
    if((dreadful||valinor)&&truth(s,queued_palantir)){
        require(side==0&&epa_live(s)&&typed(s,queued_palantir,WOTR_BOOL,"Queued Palantir grant must be Boolean").integer==1,"Card casualty queued Palantir grant lacks its live table card");
    }
    rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));quiet_action_gate(s,t,false,false,false,false,false,false,false,false,false,true,true,(name=="The Spirit of Mordor"||dreadful||valinor||name=="Dead Men of Dunharrow"||ents||name=="Faramir's Rangers"),(dreadful||valinor));ring_waiver(s,0);ring_waiver(s,1);rr_set_attr(s,s.root,"pending_card_option",p);
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));cau_dedupe(s);
    const std::string nation(enum_member(s,field(s,choice,"nation"),"game_state.Nation")),operation(text(s,field(s,choice,"casualty_operation")));const auto cost=ca_apply(s,t,region,side,hits,nation,operation),remaining=ca_forced(s,t,region,side,hits-cost);
    cp_put(s,p,"cas_hits",rr_scalar(s,WOTR_INT,remaining));am_Builder events;std::vector<uint64_t> rows;
    if(remaining>0)cp_put(s,p,"options",ecc_options(s,region,side,remaining));
    else{rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));ecc_aftermath(s,t,region,side,name,result.random.get());if(name=="Faramir's Rangers"){eff_casualty_resume(s,t,region,result.random.get(),events,rows);su_resolve_overstacks(s,t);}else if(valinor){const std::string id(text(s,field(s,p,"card_id")));erv_effect(s,t,id,region,result.random.get(),events,rows);su_resolve_overstacks(s,t);}else if(ents){const std::string id(text(s,field(s,p,"card_id")));eea_effect(s,t,id,true,result.random.get(),events,rows);su_resolve_overstacks(s,t);}else if(name=="Dead Men of Dunharrow"){const std::string id(text(s,field(s,p,"card_id")));const auto resumed=ev_dict(s,{{"target_region",rr_string(s,region)},{"_dmd_cas_resumed",rr_scalar(s,WOTR_BOOL,1)}});edm_effect(s,t,id,resumed,result.random.get(),events,rows);su_resolve_overstacks(s,t);}}
    cau_finish(s,t,result.random.get(),marker_side,events,rows);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
