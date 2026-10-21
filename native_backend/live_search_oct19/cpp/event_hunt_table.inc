// Register only persistent effects whose downstream Hunt/combat paths exist.
const char* eht_type(const Snapshot& t,uint64_t card){
    if(card==missing)return nullptr;const auto name=obs_attr(t,card,"card_name");
    if(text_is(t,name,"Denethor's Folly"))return "DENETHORS_FOLLY";
    if(text_is(t,name,"The Last Battle"))return "LAST_BATTLE";
    if(text_is(t,name,"Wizard's Staff"))return "WIZARD_STAFF";
    if(text_is(t,name,"Mithril Coat and Sting"))return "MITHRIL_COAT";
    if(text_is(t,name,"Axe and Bow"))return "AXE_AND_BOW";
    if(text_is(t,name,"Horn of Gondor"))return "HORN_OF_GONDOR";
    if(text_is(t,name,"Balrog of Moria"))return "BALROG_OF_MORIA";
    if(text_is(t,name,"Worn with Sorrow and Toil"))return "WORN_WITH_SORROW";
    if(text_is(t,name,"Flocks of Crebain"))return "FLOCKS_OF_CREBAIN";
    if(text_is(t,name,"Wormtongue"))return "WORMTONGUE";
    if(text_is(t,name,"The Palant\xc3\xadr of Orthanc"))return "PALANTIR_OF_ORTHANC";
    if(text_is(t,name,"Threats and Promises"))return "THREATS_AND_PROMISES";return nullptr;
}
// package Q: mouth_muster = the Mouth of Sauron's once-per-turn Muster-as-Army funding of a Shadow Army card (Denethor's Folly is the
// only Army card of this family; ai.py:11535-11565): payment only, the free-Ents caller and the C export keep the default.
// Registration and redraw happen only after a deferred target is resolved.
void eht_commit(Snapshot& s,const Snapshot& t,int side,std::string_view id,std::string_view deck,std::string_view face){
    const auto card=ev_card(t,id);const auto type=eht_type(t,card);require(type,"Persistent event type unavailable");
    const std::string name(text(t,obs_attr(t,card,"card_name")));
    const char* requirement=name=="Wizard's Staff"?"Gandalf the Grey":name=="Axe and Bow"?"Gimli, Legolas":name=="Horn of Gondor"?"Boromir":nullptr;
    const auto persistent=mu_object(s,"game_state.PersistentCard",{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,name)},{"side",mu_enum(s,"game_state.Side",side==0?"FREE_PEOPLES":"SHADOW")},{"card_type",mu_enum(s,"game_state.PersistentCardType",type)},{"requires_companion",requirement?rr_string(s,requirement):rr_scalar(s,WOTR_NONE)}});
    const auto list=obs_attr(s,obs_attr(s,s.root,"persistent_cards"),side==0?"fp_cards":"sh_cards");auto cards=child_edges(s,list);cards.push_back(persistent);set_edges(s,list,cards);
    const auto hand=tc_hand(s,side);auto held=child_edges(s,hand);const auto found=std::find_if(held.begin(),held.end(),[&](uint64_t c){return text_is(s,c,id);});require(found!=held.end(),"Table event card disappeared");held.erase(found);set_edges(s,hand,held);
    if(name=="The Last Battle")cau_last_battle(s,t);
    tc_open_hand_limit_discard(s);if(side==0 && !contains_string(s,hand,id))ev_gandalf_after_event(s,id,deck,face);
    epa_after_event(s,side,id,deck,face);
}
void apply_hunt_table_event_inplace(Snapshot& s,const Snapshot& t,void* rng,int side,int64_t index,std::string_view id,bool free_ents,am_Builder& ef_events,std::vector<uint64_t>& ef_annotations,bool mouth_muster=false){
    const auto& source=s;ef_validate_free(source,t,side,index,id,free_ents);
    const int ef_marker=ctx_side(source,obs_attr(source,source.root,"_battle_resolved_attacker",false));const auto ef_unused=ef_unused_dice(source);

    // Package M: the free-Ents in-place variant uses the public executor's stable branch (no `!free_ents` exclusion).
    require(side==0 || side==1,"Hunt table event actor invalid");if(text_is(t,obs_attr(t,ev_card(t,id),"card_name"),"Denethor's Folly"))quiet_action_gate(source,t,false,false,false,false,false,false,false,false,false,false,free_ents,true,false,false);else if(((text_is(t,obs_attr(t,ev_card(t,id),"card_name"),"Wizard's Staff") || text_is(t,obs_attr(t,ev_card(t,id),"card_name"),"Mithril Coat and Sting") || text_is(t,obs_attr(t,ev_card(t,id),"card_name"),"Axe and Bow")) || (((text_is(t,obs_attr(t,ev_card(t,id),"card_name"),"The Last Battle") || text_is(t,obs_attr(t,ev_card(t,id),"card_name"),"Horn of Gondor")) || (text_is(t,obs_attr(t,ev_card(t,id),"card_name"),"The Palant\xc3\xadr of Orthanc") || (side==1 && (text_is(t,obs_attr(t,ev_card(t,id),"card_name"),"Threats and Promises") || text_is(t,obs_attr(t,ev_card(t,id),"card_name"),"Worn with Sorrow and Toil") || text_is(t,obs_attr(t,ev_card(t,id),"card_name"),"Flocks of Crebain") || text_is(t,obs_attr(t,ev_card(t,id),"card_name"),"Balrog of Moria") || text_is(t,obs_attr(t,ev_card(t,id),"card_name"),"Wormtongue"))))))))quiet_action_gate(source,t,false,false,false,false,false,false,false,false,true,false,(false||free_ents),true);else quiet_action_gate(source,t,false,false,false,false,false,false,false,false,true,false,(false||free_ents));ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained) || ctx_side(source,retained)==side,"Hunt table event violates retained Ring actor");
    const auto card=ev_card(t,id);const auto type=eht_type(t,card);require(type && ctx_side(t,obs_attr(t,card,"side"))==side,"Unsupported Hunt table event");require(contains_string(source,tc_hand(source,side),id),"Hunt table card absent from hand");require(ev_requirement(source,t,card),"Hunt table companion requirement false");
    const auto die=free_ents?missing:ev_payment(source,t,side,index,card,mouth_muster);const std::string face(free_ents?std::string_view{}:enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace")),deck=ev_lower(text(t,obs_attr(t,card,"deck"))),name(text(t,obs_attr(t,card,"card_name")));
    require(name=="Denethor's Folly" || name=="The Last Battle" || ev_lower(text(t,obs_attr(t,card,"event_effect"))).find("play on the table")!=std::string::npos,"Persistent event metadata does not preserve table conservation");
    
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));if(!free_ents)ev_consume_payment(s,t,side,index,card,mouth_muster);
    if(name=="Denethor's Folly"){
        const auto nations=edf_nations(s);
        if(nations.size()>1){
            std::vector<uint64_t> options;for(const auto& nation:nations)options.push_back(ev_dict(s,{{"nation",rr_string(s,nation)}}));
            rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,name)},{"side",rr_string(s,"sh")},{"decider",rr_string(s,"sh")},
                {"options",cp_container(s,WOTR_LIST,options)},{"die_face",rr_string(s,face)},{"deck",rr_string(s,deck)}}));
            rr_set_attr(s,s.root,"_battle_resolved_attacker",rr_scalar(s,WOTR_NONE));ehd_finish(s,t,rng,ef_events,ef_annotations);return;
        }
        edf_eliminate(s);
    }
    eht_commit(s,t,side,id,deck,face);
    rr_set_attr(s,s.root,"_battle_resolved_attacker",rr_scalar(s,WOTR_NONE));ehd_finish(s,t,rng,ef_events,ef_annotations);

    if(free_ents)ef_finish_free(s,t,rng,ef_marker,ef_unused,ef_events,ef_annotations);
}
fm_Result apply_hunt_table_event(const Snapshot& source,const Snapshot& t,const void* random_source,int32_t side,int64_t index,std::string_view id,bool mouth_muster=false){
    require(side==0 || side==1,"Hunt table event actor invalid");if(text_is(t,obs_attr(t,ev_card(t,id),"card_name"),"Denethor's Folly"))quiet_stable_action_gate(source,t);else if(((text_is(t,obs_attr(t,ev_card(t,id),"card_name"),"Wizard's Staff") || text_is(t,obs_attr(t,ev_card(t,id),"card_name"),"Mithril Coat and Sting") || text_is(t,obs_attr(t,ev_card(t,id),"card_name"),"Axe and Bow")) || ((text_is(t,obs_attr(t,ev_card(t,id),"card_name"),"The Last Battle") || text_is(t,obs_attr(t,ev_card(t,id),"card_name"),"Horn of Gondor")) || (text_is(t,obs_attr(t,ev_card(t,id),"card_name"),"The Palant\xc3\xadr of Orthanc") || (side==1 && (text_is(t,obs_attr(t,ev_card(t,id),"card_name"),"Threats and Promises") || text_is(t,obs_attr(t,ev_card(t,id),"card_name"),"Worn with Sorrow and Toil") || text_is(t,obs_attr(t,ev_card(t,id),"card_name"),"Flocks of Crebain") || text_is(t,obs_attr(t,ev_card(t,id),"card_name"),"Balrog of Moria") || text_is(t,obs_attr(t,ev_card(t,id),"card_name"),"Wormtongue")))))))quiet_action_gate(source,t,false,false,false,false,false,false,false,false,true,false,false,true);else quiet_action_gate(source,t,false,false,false,false,false,false,false,false,true);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained) || ctx_side(source,retained)==side,"Hunt table event violates retained Ring actor");
    const auto card=ev_card(t,id);const auto type=eht_type(t,card);require(type && ctx_side(t,obs_attr(t,card,"side"))==side,"Unsupported Hunt table event");require(contains_string(source,tc_hand(source,side),id),"Hunt table card absent from hand");require(ev_requirement(source,t,card),"Hunt table companion requirement false");
    const auto die=ev_payment(source,t,side,index,card,mouth_muster);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace")),deck=ev_lower(text(t,obs_attr(t,card,"deck"))),name(text(t,obs_attr(t,card,"card_name")));
    require(name=="Denethor's Folly" || name=="The Last Battle" || ev_lower(text(t,obs_attr(t,card,"event_effect"))).find("play on the table")!=std::string::npos,"Persistent event metadata does not preserve table conservation");
    
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());
    am_Builder ef_events;std::vector<uint64_t> ef_annotations;
    apply_hunt_table_event_inplace(*result.state,t,result.random.get(),side,index,id,false,ef_events,ef_annotations,mouth_muster);
    ef_events.out.root=ef_events.container(WOTR_LIST,ef_annotations);result.events=std::make_unique<Snapshot>(std::move(ef_events.out));return result;
}

fm_Result apply_denethor_choice(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index){
    require(side==1,"Denethor leader choice belongs to Shadow");const auto p=edf_choice_pending(source,t);
    const auto options=child_edges(source,field(source,p,"options"));require(index>=0&&static_cast<uint64_t>(index)<options.size(),"Denethor choice index exceeds menu");
    const std::string id(text(source,field(source,p,"card_id"))),deck(text(source,field(source,p,"deck"))),face(text(source,field(source,p,"die_face"))),nation(text(source,field(source,options[index],"nation")));
    require(contains_string(source,tc_hand(source,1),id),"Denethor pending card absent");
    require(ev_requirement(source,t,ev_card(t,id)),"Denethor printed requirement no longer holds");
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());
    auto& s=*result.state;rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));
    // The paid card choice precedes its owner's hand-limit discard.
    quiet_action_gate(s,t,false,false,false,true,false,false,false,false,false,false,false,true);ring_waiver(s,0);ring_waiver(s,1);
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    edf_eliminate(s,nation);eht_commit(s,t,1,id,deck,face);
    am_Builder events;std::vector<uint64_t> rows;ehd_finish(s,t,result.random.get(),events,rows);cau_gandalf(s);
    events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
