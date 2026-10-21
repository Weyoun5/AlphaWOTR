// Two source stages: Shadow political target, then FP-owned physical loss.
bool esc_supported(const Snapshot& t,uint64_t card){return card!=missing&&text_is(t,obs_attr(t,card,"card_name"),"Stormcrow");}
bool esc_pending(const Snapshot& s){const auto p=obs_attr(s,s.root,"pending_card_option",false);return !ctx_none(s,p)&&text_is(s,field(s,p,"card_name",false),"Stormcrow");}
const char* esc_enum(std::string_view tag){if(tag=="elves")return "ELVES";if(tag=="dwarves")return "DWARVES";if(tag=="gondor")return "GONDOR";if(tag=="rohan")return "ROHAN";if(tag=="north"||tag=="the north")return "THE_NORTH";return nullptr;}
std::vector<std::string> esc_nations(const Snapshot& s,const Snapshot& t){
    std::set<std::string> present;
    auto add=[&](std::string_view region){if(!evt_stormcrow_region(s,t,region))return;const auto meta=ctx_get(t,field(t,t.root,"event_region_metadata"),region);auto tag=std::string(text(t,ctx_get(t,meta,"nation")));if(tag=="the north")tag="north";present.insert(tag);};
    add(glb_fellowship_key(s,obs_attr(s,s.root,"fellowship")));const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));
    for(size_t i=0;i<armies.size();i+=2)if(evt_companion_type(s,armies[i+1]))add(sanitize_name(text(s,armies[i])));
    // Python sorts Nation.value; The North follows Rohan.
    std::vector<std::string> ordered;for(const char* tag:{"dwarves","elves","gondor","rohan","north"})if(present.count(tag))ordered.emplace_back(tag);return ordered;
}
uint64_t esc_options(Snapshot& s,const Snapshot& t){std::vector<uint64_t> rows;for(const auto& nation:esc_nations(s,t))rows.push_back(ev_dict(s,{{"nation",rr_string(s,nation)}}));return cp_container(s,WOTR_LIST,rows);}
void esc_target(Snapshot& s,const Snapshot& t,std::string_view id,std::string nation){
    if(nation=="the north")nation="north";const auto names=esc_nations(s,t);if(std::find(names.begin(),names.end(),nation)==names.end())return;
    const auto en=esc_enum(nation);require(en,"Stormcrow nation unrecognized");const auto boxes=obs_attr(s,obs_attr(s,s.root,"political"),"boxes");
    const auto current=typed(s,ctx_nation_value(s,boxes,en),WOTR_INT,"Stormcrow political position must be integer").integer;me_nation_assign(s,boxes,en,rr_scalar(s,WOTR_INT,std::max<int64_t>(0,current-1)));
    std::vector<uint64_t> options;const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));
    for(size_t i=0;i<armies.size();i+=2){std::set<std::string> kinds;const auto army=armies[i+1];
        for(auto unit:child_edges(s,obs_attr(s,army,"units")))if(enum_is(s,obs_attr(s,unit,"nation"),"game_state.Nation",en))kinds.insert(enum_is(s,obs_attr(s,unit,"kind"),"game_state.UnitKind","ELITE")?"elite":"regular");
        for(auto leader:child_edges(s,obs_attr(s,army,"leaders")))if(object_type(s,leader)=="game_state.Leader"&&enum_is(s,obs_attr(s,leader,"nation"),"game_state.Nation",en))kinds.insert("leader");
        for(const auto& kind:kinds)options.push_back(ev_dict(s,{{"eliminate",ev_dict(s,{{"nation",rr_string(s,nation)},{"kind",rr_string(s,kind)},{"region",rr_string(s,sanitize_name(text(s,armies[i])))}})}}));
    }
    if(!options.empty())rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,"Stormcrow")},{"side",rr_string(s,"sh")},{"decider",rr_string(s,"fp")},{"direct",rr_scalar(s,WOTR_BOOL,1)},{"options",cp_container(s,WOTR_LIST,options)}}));
}
void esc_eliminate(Snapshot& s,uint64_t option){
    const auto pick=field(s,option,"eliminate");const std::string tag(ev_lower(text(s,field(s,pick,"nation")))),kind(text(s,field(s,pick,"kind"))),region(sanitize_name(text(s,field(s,pick,"region"))));const auto nation=esc_enum(tag);if(!nation)return;
    const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));
    for(size_t i=0;i<armies.size();i+=2)if(sanitize_name(text(s,armies[i]))==region){
        const auto list=obs_attr(s,armies[i+1],kind=="leader"?"leaders":"units");
        for(auto figure:child_edges(s,list))if(enum_is(s,obs_attr(s,figure,"nation",false),"game_state.Nation",nation)&&
          (kind=="leader"?object_type(s,figure)=="game_state.Leader":enum_is(s,obs_attr(s,figure,"kind"),"game_state.UnitKind",kind=="elite"?"ELITE":"REGULAR"))){
            // Public selection names a kind, so the first matching physical
            // figure is also the first equal list entry removed by Python.
            auto entries=child_edges(s,list);const auto at=std::find(entries.begin(),entries.end(),figure);entries.erase(at);set_edges(s,list,entries);cbp_book(s,obs_attr(s,figure,"nation"),kind=="leader"?"leader":kind=="elite"?"elite":"regular");am_orphans(s);return;
        }break;
    }
}
uint64_t esc_validate(const Snapshot& s,const Snapshot& t){
    require(esc_pending(s),"No Stormcrow pending");const auto p=obs_attr(s,s.root,"pending_card_option"),id=field(s,p,"card_id");
    require(esc_supported(t,ev_card(t,text(s,id)))&&text_is(s,field(s,p,"side"),"sh"),"Stormcrow pending identity invalid");const bool direct=truth(s,field(s,p,"direct",false));
    require(text_is(s,field(s,p,"decider"),direct?"fp":"sh"),"Stormcrow pending owner invalid");
    if(!direct){/* package R: hand membership is not a menu precondition (frozen card_option menu, ai.py 7764-7770) */require(text_is(s,field(s,p,"deck"),"strategy"),"Stormcrow paid deck invalid");const auto face=text(s,field(s,p,"die_face"));require(face=="MUSTER"||face=="MUSTER_ARMY"||face=="EVENT","Stormcrow paid face invalid");}
    const auto options=child_edges(s,field(s,p,"options"));require(options.size()>=(direct?1:2),"Stormcrow stored options empty");
    for(auto option:options){require(typed(s,option,WOTR_DICT,"Stormcrow option must be dictionary").count==2,"Stormcrow option shape invalid");
        if(direct){const auto pick=field(s,option,"eliminate");require(typed(s,pick,WOTR_DICT,"Stormcrow elimination must be dictionary").count==6,"Stormcrow elimination shape invalid");require(esc_enum(text(s,field(s,pick,"nation"))),"Stormcrow elimination nation invalid");const auto kind=text(s,field(s,pick,"kind"));require(kind=="regular"||kind=="elite"||kind=="leader","Stormcrow elimination kind invalid");text(s,field(s,pick,"region"));}
        else require(esc_enum(text(s,field(s,option,"nation"))),"Stormcrow target nation invalid");
    }return p;
}
std::unique_ptr<Snapshot> native_stormcrow_choices(const Snapshot& source,const Snapshot& t,int side){
    const auto p=esc_validate(source,t);require(side==(truth(source,field(source,p,"direct",false))?0:1),"Stormcrow choice belongs to another actor");auto result=std::make_unique<Snapshot>(source);auto& s=*result;const auto pending=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t index=0;
    for(auto choice:child_edges(s,field(s,pending,"options")))rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,pending,"card_id")},{"card_name",field(s,pending,"card_name")},{"side",field(s,pending,"side")},{"choices",choice}})}));s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
fm_Result apply_stormcrow_event(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index,std::string_view id){
    require(side==1,"Stormcrow belongs to Shadow");quiet_stable_action_gate(source,t);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==1,"Stormcrow violates retained Ring actor");
    const auto card=ev_card(t,id);require(esc_supported(t,card)&&contains_string(source,tc_hand(source,1),id),"Stormcrow card absent");require(ev_requirement(source,t,card),"Stormcrow requirement false");
    const auto die=ev_payment(source,t,1,index,card);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace"));
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));rr_set_attr(s,ev_payment(s,t,1,index,card),"used",rr_scalar(s,WOTR_BOOL,1));
    const auto options=esc_options(s,t);const auto choices=child_edges(s,options);
    if(choices.size()>1){rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,"Stormcrow")},{"side",rr_string(s,"sh")},{"decider",rr_string(s,"sh")},{"options",options},{"die_face",rr_string(s,face)},{"deck",rr_string(s,"strategy")}}));tc_open_hand_limit_discard(s);}
    else{if(!choices.empty())esc_target(s,t,id,std::string(text(s,field(s,choices.front(),"nation"))));esr_commit(s,t,id,"strategy",face);}
    am_Builder events;std::vector<uint64_t> rows;cau_finish(s,t,result.random.get(),-1,events,rows);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
fm_Result apply_stormcrow_choice(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index){
    const auto pending=esc_validate(source,t);const bool direct=truth(source,field(source,pending,"direct",false));require(side==(direct?0:1),"Stormcrow choice belongs to another actor");const auto options=child_edges(source,field(source,pending,"options"));require(index>=0&&static_cast<uint64_t>(index)<options.size(),"Stormcrow index invalid");
    const std::string id(text(source,field(source,pending,"card_id"))),face=direct?"":std::string(text(source,field(source,pending,"die_face")));const int marker=ctx_side(source,obs_attr(source,source.root,"_battle_resolved_attacker",false));
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;
    const auto choice=child_edges(s,field(s,obs_attr(s,s.root,"pending_card_option"),"options"))[index];rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));
    const bool queued=truth(s,obs_attr(s,s.root,"_palantir_draw_pending",false));
    quiet_action_gate(s,t,false,false,false,false,false,false,false,false,false,false,false,true,queued);ring_waiver(s,0);ring_waiver(s,1);
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    if(direct)esc_eliminate(s,choice);
    else if(ev_requirement(s,t,ev_card(t,id))){esc_target(s,t,id,std::string(text(s,field(s,choice,"nation"))));esr_commit(s,t,id,"strategy",face);}
    am_Builder events;std::vector<uint64_t> rows;cau_finish(s,t,result.random.get(),marker,events,rows);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
