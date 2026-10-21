// Eagles: FP targets, physical Nazgul hits, Shadow relocation.
bool eeg_supported(const Snapshot& t,uint64_t card){return card!=missing&&text_is(t,obs_attr(t,card,"card_name"),"The Eagles are Coming");}
bool eeg_pending(const Snapshot& s){const auto p=obs_attr(s,s.root,"pending_card_option",false);return !ctx_none(s,p)&&text_is(s,field(s,p,"card_name",false),"The Eagles are Coming");}
bool eeg_destination(const Snapshot& s){return eeg_pending(s)&&truth(s,field(s,obs_attr(s,s.root,"pending_card_option"),"direct",false));}
std::vector<uint64_t> eeg_nazgul(const Snapshot& s,uint64_t army){std::vector<uint64_t> out;if(army!=missing)for(auto l:child_edges(s,obs_attr(s,army,"leaders")))if(object_type(s,l)=="game_state.Nazgul")out.push_back(l);return out;}
bool eeg_fp_companion(const Snapshot& s,uint64_t army,bool first=false){
    if(army==missing)return false;const auto units=child_edges(s,obs_attr(s,army,"units"));if(units.empty())return false;
    bool fp=false;if(first)fp=ctx_unit_side(s,units.front())==0;else for(auto u:units)fp|=ctx_unit_side(s,u)==0;
    if(!fp)return false;for(auto l:child_edges(s,obs_attr(s,army,"leaders")))if(object_type(s,l)=="game_state.Companion")return true;return false;
}
bool eeg_near_fp(const Snapshot& s,std::string_view region,bool first=false){
    if(eeg_fp_companion(s,mu_army(s,region),first))return true;
    const auto adjacent=ctx_get(s,obs_attr(s,obs_attr(s,s.root,"board"),"_adj"),region);
    if(!ctx_none(s,adjacent))for(auto key:child_edges(s,adjacent))if(eeg_fp_companion(s,mu_army(s,sanitize_name(text(s,key))),first))return true;return false;
}
uint64_t eeg_options(Snapshot& s){
    std::vector<uint64_t> options;const auto entries=child_edges(s,obs_attr(s,s.root,"armies"));
    for(size_t i=0;i<entries.size();i+=2){const auto key=sanitize_name(text(s,entries[i]));if(!eeg_nazgul(s,entries[i+1]).empty()&&eeg_near_fp(s,key))options.push_back(ev_dict(s,{{"target_region",rr_string(s,key)}}));}
    return cp_container(s,WOTR_LIST,options);
}
std::vector<std::string> eeg_destinations(const Snapshot& s,const Snapshot& t,std::string_view origin){
    std::set<std::string> keys;const auto metadata=child_edges(t,field(t,t.root,"event_region_metadata"));const auto captured=ctx_side_value(s,obs_attr(s,s.root,"enemy_controlled"),0);
    for(size_t i=0;i<metadata.size();i+=2){const auto key=sanitize_name(text(t,metadata[i]));const auto meta=metadata[i+1];if(truth(t,field(t,meta,"is_stronghold",false))&&text_is(t,field(t,meta,"nation",false),"sauron")&&key!=sanitize_name(origin)&&!contains_string(s,captured,key))keys.insert(key);}
    return {keys.begin(),keys.end()};
}
// Python Nazgul dataclass equality considers nation, not extra attributes.
void eeg_remove_equal(Snapshot& s,uint64_t army,uint64_t victim){
    const auto list=obs_attr(s,army,"leaders");auto leaders=child_edges(s,list);const auto nation=ctx_nation(s,obs_attr(s,victim,"nation"));
    const auto it=std::find_if(leaders.begin(),leaders.end(),[&](auto leader){return object_type(s,leader)=="game_state.Nazgul"&&ctx_nation(s,obs_attr(s,leader,"nation"))==nation;});require(it!=leaders.end(),"Eagles physical Nazgul disappeared");leaders.erase(it);set_edges(s,list,leaders);
}
void eeg_relocate(Snapshot& s,std::string_view from,std::string_view destination,bool direct){
    const auto key=am_key(s,sanitize_name(from));if(key.empty())return;const auto army=mu_army(s,sanitize_name(from));const auto moving=eeg_nazgul(s,army);for(auto l:moving)eeg_remove_equal(s,army,l);
    if(direct&&!truth(s,obs_attr(s,army,"units"))&&!truth(s,obs_attr(s,army,"leaders")))am_erase(s,key);
    if(!moving.empty()){
        const std::string dest=sanitize_name(destination);auto target=direct?mu_army(s,dest):ctx_get(s,obs_attr(s,s.root,"armies"),dest);
        if(target==missing){const auto region=ctx_get(s,obs_attr(s,obs_attr(s,s.root,"board"),"_regions"),dest);require(region!=missing,"Eagles destination not on board");target=am_new_army(s,region);am_insert(s,dest,target);}
        am_extend(s,target,"leaders",moving);
    }
    if(!direct&&!truth(s,obs_attr(s,army,"units"))&&!truth(s,obs_attr(s,army,"leaders")))am_erase(s,key);
}
void eeg_effect(Snapshot& s,const Snapshot& t,std::string_view id,std::string_view requested,void* rng,am_Builder& events,std::vector<uint64_t>& rows){
    std::string key=requested.empty()?std::string{}:am_key(s,sanitize_name(requested));const auto entries=child_edges(s,obs_attr(s,s.root,"armies"));
    if(key.empty())for(size_t i=0;i<entries.size();i+=2){bool naz=false;for(auto l:child_edges(s,obs_attr(s,entries[i+1],"leaders")))naz|=hr_nazgul(s,l);const auto candidate=sanitize_name(text(s,entries[i]));if(naz&&eeg_near_fp(s,candidate,true)){key=std::string(text(s,entries[i]));break;}}
    if(key.empty())for(size_t i=0;i<entries.size();i+=2)if(!eeg_nazgul(s,entries[i+1]).empty()){key=std::string(text(s,entries[i]));break;}
    if(key.empty())return;const auto army=mu_army(s,sanitize_name(key));auto figures=eeg_nazgul(s,army);std::vector<uint32_t> rolls(std::min<size_t>(5,figures.size()));
    hf_roll(rng,rolls.data(),rolls.size(),"card effect");int hits=0;std::vector<uint64_t> faces;for(auto r:rolls){hits+=r>=5;faces.push_back(events.integer(r));}
    rows.push_back(events.row(-1,"dice_roll",{events.str("faces"),events.container(WOTR_LIST,faces),events.str("ctx"),events.str("The Eagles are Coming roll"),events.str("side"),rr_scalar(events.out,WOTR_NONE)}));
    for(int i=0;i<hits&&!figures.empty();++i){const auto victim=figures.back();figures.pop_back();eeg_remove_equal(s,army,victim);}
    if(!eeg_nazgul(s,army).empty()){
        const auto destinations=eeg_destinations(s,t,key);
        if(destinations.size()>1){std::vector<uint64_t> options;for(const auto& destination:destinations)options.push_back(ev_dict(s,{{"eagles_dest",rr_string(s,destination)},{"_eagles_from",rr_string(s,sanitize_name(key))}}));
            rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,"The Eagles are Coming")},{"side",rr_string(s,"fp")},{"decider",rr_string(s,"sh")},{"direct",rr_scalar(s,WOTR_BOOL,1)},{"options",cp_container(s,WOTR_LIST,options)}}));return;
        }
        if(destinations.size()==1)eeg_relocate(s,key,destinations.front(),false);
    }
    if(!truth(s,obs_attr(s,army,"units"))&&!truth(s,obs_attr(s,army,"leaders")))am_erase(s,key);
}
void apply_eagles_event_inplace(Snapshot& s,const Snapshot& t,void* rng,int side,int64_t index,std::string_view id,bool free_ents,am_Builder& ef_events,std::vector<uint64_t>& ef_annotations){
    const auto& source=s;ef_validate_free(source,t,side,index,id,free_ents);
    const int ef_marker=ctx_side(source,obs_attr(source,source.root,"_battle_resolved_attacker",false));const auto ef_unused=ef_unused_dice(source);

    require(side==0,"Eagles belongs to FP");quiet_action_gate(source,t,false,false,false,false,false,false,false,false,false,false,free_ents,true,false,false);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==0,"Eagles violates retained Ring actor");
    const auto card=ev_card(t,id);require(eeg_supported(t,card)&&contains_string(source,tc_hand(source,0),id),"Eagles absent from FP hand");require(ev_requirement(source,t,card),"Eagles needs a Companion army in contact with a Nazgul army");
    const auto die=free_ents?missing:ev_payment(source,t,0,index,card);const std::string face(free_ents?std::string_view{}:enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace"));
    
    rr_set_attr(s,s.root,"_casualty_operations",cp_container(s,WOTR_LIST,{}));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));if(!free_ents)rr_set_attr(s,ev_payment(s,t,0,index,card),"used",rr_scalar(s,WOTR_BOOL,1));
    const auto options=eeg_options(s);const auto choices=child_edges(s,options);
    if(choices.size()>1){rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,"The Eagles are Coming")},{"side",rr_string(s,"fp")},{"decider",rr_string(s,"fp")},{"options",options},{"die_face",ef_pending_face_node(s,face)},{"deck",rr_string(s,"character")}}));tc_open_hand_limit_discard(s);}
    else{const std::string target=choices.empty()?std::string{}:std::string(text(s,field(s,choices.front(),"target_region")));eeg_effect(s,t,id,target,rng,ef_events,ef_annotations);er_commit(s,t,id,"character",face);}
    cau_finish(s,t,rng,free_ents?ef_final_marker(s,ef_marker,ef_unused):-1,ef_events,ef_annotations);
}
fm_Result apply_eagles_event(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index,std::string_view id){
    require(side==0,"Eagles belongs to FP");quiet_stable_action_gate(source,t);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==0,"Eagles violates retained Ring actor");
    const auto card=ev_card(t,id);require(eeg_supported(t,card)&&contains_string(source,tc_hand(source,0),id),"Eagles absent from FP hand");require(ev_requirement(source,t,card),"Eagles needs a Companion army in contact with a Nazgul army");
    const auto die=ev_payment(source,t,0,index,card);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace"));
    
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());
    am_Builder ef_events;std::vector<uint64_t> ef_annotations;
    apply_eagles_event_inplace(*result.state,t,result.random.get(),side,index,id,false,ef_events,ef_annotations);
    ef_events.out.root=ef_events.container(WOTR_LIST,ef_annotations);result.events=std::make_unique<Snapshot>(std::move(ef_events.out));return result;
}
uint64_t eeg_validate_pending(const Snapshot& s,const Snapshot& t,int side){
    require(eeg_pending(s),"No Eagles choice pending");const auto p=obs_attr(s,s.root,"pending_card_option");const auto id=text(s,field(s,p,"card_id"));const bool direct=eeg_destination(s);
    require(eeg_supported(t,ev_card(t,id))&&text_is(s,field(s,p,"side"),"fp")&&text_is(s,field(s,p,"decider"),direct?"sh":"fp")&&side==(direct?1:0),"Eagles choice identity or owner mismatch");
    if(!direct){require(text_is(s,field(s,p,"deck"),"character"),"Eagles target card or deck invalid");const auto face=ef_pending_face(s,p);require(face.empty()||face=="CHARACTER"||face=="EVENT"||face=="WILL_OF_WEST","Eagles target funding invalid");}
    const auto options=child_edges(s,field(s,p,"options"));require(options.size()>1,"Eagles modal choice needs multiple options");std::set<std::string> seen;std::string from;
    for(auto choice:options){require(typed(s,choice,WOTR_DICT,"Eagles stored option must be dictionary").count==(direct?4:2),"Eagles stored option shape invalid");const std::string target(text(s,field(s,choice,direct?"eagles_dest":"target_region")));require(!target.empty()&&sanitize_name(target)==target&&seen.insert(target).second,"Eagles stored target malformed");
        if(direct){const std::string origin(text(s,field(s,choice,"_eagles_from")));require(!origin.empty()&&sanitize_name(origin)==origin&&(from.empty()||from==origin)&&target!=origin,"Eagles stored origin malformed");from=origin;
            const auto meta=ctx_get(t,field(t,t.root,"event_region_metadata"),target);require(meta!=missing&&truth(t,field(t,meta,"is_stronghold",false))&&text_is(t,field(t,meta,"nation",false),"sauron"),"Eagles destination must be Sauron stronghold");}
    }return p;
}
std::unique_ptr<Snapshot> native_eagles_choices(const Snapshot& source,const Snapshot& t,int side){
    eeg_validate_pending(source,t,side);Snapshot s(source);const auto p=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t index=0;
    for(auto choice:child_edges(s,field(s,p,"options")))rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",choice}})}));s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
fm_Result apply_eagles_choice(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index){
    const bool free_ents=ef_pending_is_free(source,obs_attr(source,source.root,"pending_card_option",false));

    const auto pending=eeg_validate_pending(source,t,side);const auto options=child_edges(source,field(source,pending,"options"));require(index>=0&&static_cast<uint64_t>(index)<options.size(),"Eagles choice index invalid");
    const std::string id(text(source,field(source,pending,"card_id")));const bool direct=eeg_destination(source);const std::string face=direct?std::string{}:std::string(ef_pending_face(source,pending));const int marker=ctx_side(source,obs_attr(source,source.root,"_battle_resolved_attacker",false));
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;const auto p=obs_attr(s,s.root,"pending_card_option"),choice=child_edges(s,field(s,p,"options"))[index];
    rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));quiet_action_gate(s,t,false,false,false,false,false,false,false,false,false,false,(direct||free_ents),true);ring_waiver(s,0);ring_waiver(s,1);
    const auto queued=obs_attr(s,s.root,"_gandalf_draw_pending",false);if(direct&&!ctx_none(s,queued))require((text_is(s,field(s,queued,"source_card_id"),id)||ef_ents_queued_grant(s,t,queued))&&text_is(s,field(s,queued,"deck"),"character"),"Eagles queued Gandalf source mismatch");
    rr_set_attr(s,s.root,"_casualty_operations",cp_container(s,WOTR_LIST,{}));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));am_Builder events;std::vector<uint64_t> rows;
    if(direct){const std::string from(text(s,field(s,choice,"_eagles_from"))),dest(text(s,field(s,choice,"eagles_dest")));eeg_relocate(s,from,dest,true);su_resolve_overstacks(s,t);}
    else if(ev_requirement(s,t,ev_card(t,id))){const std::string target(text(s,field(s,choice,"target_region")));eeg_effect(s,t,id,target,result.random.get(),events,rows);er_commit(s,t,id,"character",face);}
    cau_finish(s,t,result.random.get(),marker,events,rows);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
