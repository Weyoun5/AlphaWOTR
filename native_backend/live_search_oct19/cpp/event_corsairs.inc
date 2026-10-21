// Exact public Corsairs landing before its co-located battle.
bool eco_supported(const Snapshot& t,uint64_t card){return card!=missing&&text_is(t,obs_attr(t,card,"card_name"),"Corsairs of Umbar");}
bool eco_pending(const Snapshot& s){const auto p=obs_attr(s,s.root,"pending_card_option",false);return !ctx_none(s,p)&&text_is(s,field(s,p,"card_name",false),"Corsairs of Umbar");}
bool eco_coastal(const Snapshot& t,std::string_view region){
    if(region!="anfalas"&&region!="lamedon"&&region!="pelargir"&&region!="dol amroth")return false;
    const auto nation=ctx_get(t,field(t,t.root,"region_nations"),region);return !ctx_none(t,nation)&&ctx_nation(t,nation)=="GONDOR";
}
uint64_t eco_options(Snapshot& s,const Snapshot& t){
    std::vector<uint64_t> rows;const auto siege=obs_attr(s,s.root,"besieged_strongholds");
    if(!evb_units(s,hr_army(s,"umbar"),1)||(contains_string(s,siege,"umbar")&&c9_garrison(s,t,"umbar")==1))return cp_container(s,WOTR_LIST,rows);
    for(const char* region:{"anfalas","lamedon","pelargir","dol amroth"}){
        if(ctx_none(t,ctx_get(t,field(t,t.root,"board_regions"),region)))continue;
        rows.push_back(ev_dict(s,{{"target_region",rr_string(s,region)}}));
    }return cp_container(s,WOTR_LIST,rows);
}
void eco_begin(crr_Result& owned,const Snapshot& t,uint64_t pc,am_Builder& events,std::vector<uint64_t>& rows){
    auto& s=*owned.state;const std::string target(text(s,field(s,pc,"defender_region")));const bool relief=truth(s,field(s,pc,"corsairs_relief_garrison",false)),pre_siege=truth(s,field(s,pc,"is_siege"));
    if(!relief&&cag_figures(s,target).first.size()>10){
        std::set<std::pair<std::string,std::string>> types;for(auto unit:child_edges(s,obs_attr(s,hr_army(s,target),"units")))if(ctx_unit_side(s,unit)==1)types.emplace(su_nation_value(ctx_nation(s,obs_attr(s,unit,"nation"))),ev_lower(enum_member(s,obs_attr(s,unit,"kind"),"game_state.UnitKind")));
        std::vector<uint64_t> options;for(const auto& [nation,kind]:types){const auto copy=cp_container(s,WOTR_DICT,child_edges(s,pc));options.push_back(ev_dict(s,{{"overstack_remove",ev_dict(s,{{"region",field(s,pc,"defender_region")},{"nation",rr_string(s,nation)},{"kind",rr_string(s,kind)}})},{"_corsairs_landing_pc",copy}}));}
        rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,"SHADOW_Strategy_Corsairs_of_Umbar")},{"card_name",rr_string(s,"Corsairs of Umbar")},{"side",rr_string(s,"sh")},{"decider",rr_string(s,"sh")},{"direct",rr_scalar(s,WOTR_BOOL,true)},{"options",cp_container(s,WOTR_LIST,options)}}));return;
    }
    rr_set_attr(s,s.root,"pending_combat",pc);
    if(relief)cbb_run(owned,t,-1,events,rows);
    else if(pre_siege||truth(s,field(s,pc,"stronghold_region",false))){
        cbb_freeze(s,pc);
        if(pre_siege)cp_put(s,pc,"phase",rr_string(s,"attacker_siege_decision"));
        else if(at_controller(s,t,target)==0){cp_put(s,pc,"phase",rr_string(s,"defender_siege_choice"));cp_put(s,pc,"deciding_side",field(s,pc,"defender_side"));}
        else cbb_run(owned,t,-1,events,rows);
    }else cbb_run(owned,t,-1,events,rows);
}
void eco_remove(crr_Result& owned,const Snapshot& t,uint64_t choice,am_Builder& events,std::vector<uint64_t>& rows){
    auto& s=*owned.state;const auto pc=field(s,choice,"_corsairs_landing_pc"),spec=field(s,choice,"overstack_remove");const std::string target(text(s,field(s,pc,"defender_region")));const auto army=hr_army(s,target);
    require(text_is(s,field(s,spec,"region"),target)&&!truth(s,field(s,pc,"corsairs_relief_garrison",false))&&cag_figures(s,target).first.size()>10,"No Corsairs landing overstack to remove");const auto nation=eo_nation(text(s,field(s,spec,"nation")));const std::string kind(text(s,field(s,spec,"kind")));require(kind=="regular"||kind=="elite","Invalid Corsairs excess figure");
    const auto list=obs_attr(s,army,"units");auto units=child_edges(s,list);bool removed=false;for(auto it=units.begin();it!=units.end();++it)if(ctx_unit_side(s,*it)==1&&enum_is(s,obs_attr(s,*it,"nation"),"game_state.Nation",nation)&&enum_is(s,obs_attr(s,*it,"kind"),"game_state.UnitKind",kind=="elite"?"ELITE":"REGULAR")){units.erase(it);removed=true;break;}
    require(removed,"Selected Corsairs overstack figure absent");set_edges(s,list,units);eco_begin(owned,t,pc,events,rows);
}
void eco_effect(crr_Result& owned,const Snapshot& t,uint64_t choice,am_Builder& events,std::vector<uint64_t>& rows){
    auto& s=*owned.state;const auto value=field(s,choice,"target_region",false);if(ctx_none(s,value)||!truth(s,value))return;const std::string target=sanitize_name(text(s,value));
    const auto sieges=obs_attr(s,s.root,"besieged_strongholds");if(contains_string(s,sieges,"umbar")&&c9_garrison(s,t,"umbar")==1)return;
    if(!evb_units(s,hr_army(s,"umbar"),1)||!eco_coastal(t,target))return;
    const bool defender=evb_units(s,hr_army(s,target),0),pre_siege=contains_string(s,sieges,target);
    const bool relief=pre_siege&&c9_garrison(s,t,target)==1;const auto garrison=relief?cag_freeze(s,cag_inventory(s,target)):missing;
    if(!cdm_move(s,t,"umbar",target,1,false,{},-1,-1,true,field(s,choice,"cargo",false),relief))return;
    if(!defender){wt_activate_if_inactive(s,"GONDOR","attack");return;}
    const auto army=hr_army(s,target);std::set<std::string> nations;for(auto u:child_edges(s,obs_attr(s,army,"units")))if(ctx_unit_side(s,u)==0)nations.emplace(ctx_nation(s,obs_attr(s,u,"nation")));for(const auto& nation:nations)wt_defender_advance(s,nation,target);
    const auto meta=ctx_get(t,field(t,t.root,"board_regions"),target);const bool stronghold=truth(t,field(t,meta,"is_stronghold",false));
    const auto pc=ev_dict(s,{{"attacker_region",rr_string(s,target)},{"defender_region",rr_string(s,target)},{"is_siege",rr_scalar(s,WOTR_BOOL,pre_siege&&!relief)},{"stronghold_region",stronghold?rr_string(s,target):rr_scalar(s,WOTR_NONE)},{"phase",rr_string(s,"combat_setup")},{"deciding_side",mu_enum(s,"game_state.Side","SHADOW")},{"attacker_side",mu_enum(s,"game_state.Side","SHADOW")},{"defender_side",mu_enum(s,"game_state.Side","FREE_PEOPLES")},{"rounds_fought",rr_scalar(s,WOTR_INT,0)},{"attacker_no_retreat",rr_scalar(s,WOTR_BOOL,!pre_siege||relief)}});
    if(relief)cp_put(s,pc,"corsairs_relief_garrison",garrison);
    const auto tracked=obs_attr(s,s.root,"_overstack_regions",false);if(!ctx_none(s,tracked))me_set_member(s,tracked,target,false);eco_begin(owned,t,pc,events,rows);
}
fm_Result eco_finish(crr_Result owned,const Snapshot& t,int marker,am_Builder& events,std::vector<uint64_t>& rows){
    cau_finish(*owned.state,t,owned.random.get(),marker,events,rows);fm_Result result;result.state=ac_compact(*owned.state);result.random=std::move(owned.random);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
fm_Result apply_corsairs_event(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index,std::string_view id,bool mouth_muster=false){
    require(side==1,"Corsairs belongs to Shadow");quiet_stable_action_gate(source,t);ring_waiver(source,0);ring_waiver(source,1);const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==side,"Corsairs violates retained Ring actor");
    const auto card=ev_card(t,id);require(eco_supported(t,card)&&contains_string(source,tc_hand(source,side),id),"Corsairs card absent");require(ev_requirement(source,t,card),"Corsairs requirement false");const auto die=ev_payment(source,t,side,index,card,mouth_muster);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace")),deck(ev_lower(text(t,obs_attr(t,card,"deck"))));
    crr_Result owned;owned.state=action_clone(source);owned.random.reset(wotr_random_clone(rng));if(!owned.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*owned.state;am_Builder events;std::vector<uint64_t> rows;
    rr_set_attr(s,s.root,"_casualty_operations",cp_container(s,WOTR_LIST,{}));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));ev_consume_payment(s,t,side,index,card,mouth_muster);const auto options=eco_options(s,t);const auto choices=child_edges(s,options);
    if(choices.size()>1){rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,"Corsairs of Umbar")},{"side",rr_string(s,"sh")},{"decider",rr_string(s,"sh")},{"options",options},{"die_face",rr_string(s,face)},{"deck",rr_string(s,deck)}}));tc_open_hand_limit_discard(s);}
    else{if(!choices.empty()){const auto completed=cag_stage(s,t,id,choices.front(),"umbar");if(completed!=missing)eco_effect(owned,t,completed,events,rows);}esr_commit(*owned.state,t,id,deck,face);}
    return eco_finish(std::move(owned),t,-1,events,rows);
}
uint64_t eco_validate(const Snapshot& s,const Snapshot& t,int side){
    require(side==1&&eco_pending(s),"No Shadow Corsairs choice pending");const auto p=obs_attr(s,s.root,"pending_card_option");const auto id=text(s,field(s,p,"card_id"));
    require(eco_supported(t,ev_card(t,id))&&text_is(s,field(s,p,"side",false),"sh")&&text_is(s,field(s,p,"decider",false),"sh"),"Corsairs pending identity or owner invalid");
    if(truth(s,field(s,p,"direct",false)))return p;
    const auto face=text(s,field(s,p,"die_face"));require((face=="ARMY"||face=="MUSTER_ARMY"||face=="EVENT"||face=="MUSTER")&&text_is(s,field(s,p,"deck"),"strategy"),"Corsairs stored funding invalid");
    if(face=="MUSTER")require(truth(s,obs_attr(s,s.root,"sh_mouth_muster_used_this_turn",false)),"Corsairs Muster funding lacks Mouth payment marker");return p;
}
std::unique_ptr<Snapshot> native_corsairs_choices(const Snapshot& source,const Snapshot& t,int side){
    eco_validate(source,t,side);auto out=std::make_unique<Snapshot>(source);auto& s=*out;const auto p=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t index=0;
    for(auto choice:child_edges(s,field(s,p,"options")))rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",choice}})}));s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
fm_Result apply_corsairs_choice(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index){
    const auto p=eco_validate(source,t,side);const auto options=child_edges(source,field(source,p,"options"));require(index>=0&&static_cast<uint64_t>(index)<options.size(),"Corsairs choice index exceeds menu");const bool direct=truth(source,field(source,p,"direct",false));const std::string id(text(source,field(source,p,"card_id"))),face=direct?"":std::string(text(source,field(source,p,"die_face"))),deck=direct?"":std::string(text(source,field(source,p,"deck")));const int marker=ctx_side(source,obs_attr(source,source.root,"_battle_resolved_attacker",false));
    crr_Result owned;owned.state=action_clone(source);owned.random.reset(wotr_random_clone(rng));if(!owned.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*owned.state;am_Builder events;std::vector<uint64_t> rows;const auto choice=child_edges(s,field(s,obs_attr(s,s.root,"pending_card_option"),"options"))[index];rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));
    const bool queued=epa_paid_continuation_debt(s,id,direct);
    if(field(s,choice,"_corsairs_landing_pc",false)!=missing)quiet_action_gate(s,t,false,false,false,false,false,false,false,false,false,false,true,true,queued,false,true);else quiet_action_gate(s,t,false,false,false,false,false,false,false,false,false,false,false,true,queued);
    ring_waiver(s,0);ring_waiver(s,1);rr_set_attr(s,s.root,"_casualty_operations",cp_container(s,WOTR_LIST,{}));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    if(direct||ev_requirement(s,t,ev_card(t,id))){if(field(s,choice,"_corsairs_landing_pc",false)!=missing)eco_remove(owned,t,choice,events,rows);else{const auto completed=cag_stage(s,t,id,choice,"umbar");if(completed!=missing)eco_effect(owned,t,completed,events,rows);}if(!direct)esr_commit(*owned.state,t,id,deck,face);else su_resolve_overstacks(s,t);}
    return eco_finish(std::move(owned),t,marker,events,rows);
}
