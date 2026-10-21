// All eight card-added special tiles share this exact source transaction.
// Before Mordor the fresh tile is visible in discard; in Mordor it joins the
// pool and the entire pool is shuffled using an independently owned RNG.
uint64_t eh_template(const Snapshot& t,uint64_t card){
    if(card==missing)return missing;
    const auto templates=field(t,t.root,"special_hunt_tiles");
    return ctx_get(t,templates,text(t,obs_attr(t,card,"card_name")));
}
uint64_t eh_scalar_copy(Snapshot& s,const Snapshot& t,uint64_t node){
    const auto& value=t.nodes[node];
    if(value.tag==WOTR_STRING)return rr_string(s,text(t,node));
    require(value.tag==WOTR_NONE || value.tag==WOTR_BOOL || value.tag==WOTR_INT,"special tile template contains an unsupported scalar");
    return rr_scalar(s,value.tag,value.integer);
}
void eh_stage(Snapshot& s,const Snapshot& t,uint64_t templ,void* random){
    require(object_type(t,templ)=="game_state.HuntTile","special tile requires registered HuntTile template");
    auto copy=[&](const char* key){return eh_scalar_copy(s,t,obs_attr(t,templ,key));};
    const auto tile=mu_object(s,"game_state.HuntTile",{{"damage",copy("damage")},{"is_eye",copy("is_eye")},{"reveal",copy("reveal")},
        {"is_special",rr_scalar(s,WOTR_BOOL,1)},{"special_name",copy("special_name")},{"stops_movement",copy("stops_movement")},
        {"is_dice_roll",copy("is_dice_roll")},{"is_fellowship_special",copy("is_fellowship_special")},{"image_file",copy("image_file")}});
    const auto fs=obs_attr(s,s.root,"fellowship");const bool mordor=truth(s,obs_attr(s,fs,"mordor_entered"));
    const auto pile=obs_attr(s,s.root,mordor?"hunt_pool":"hunt_discard");typed(s,pile,WOTR_LIST,"special tile destination must be a list");
    auto entries=child_edges(s,pile);entries.push_back(tile);
    if(mordor){
        std::vector<uint64_t> permutation(entries.size()),after(entries.size());
        if(hf_shuffle(random,permutation.data(),permutation.size()))throw std::invalid_argument(wotr_random_last_error());
        for(size_t i=0;i<entries.size();++i)after[i]=entries[permutation[i]];entries=std::move(after);
    }
    set_edges(s,pile,entries);
}
void apply_special_tile_event_action_inplace(Snapshot& s,const Snapshot& t,void* rng,int side,int64_t index,std::string_view id,bool free_ents,am_Builder& ef_events,std::vector<uint64_t>& ef_annotations){
    const auto& source=s;ef_validate_free(source,t,side,index,id,free_ents);
    const int ef_marker=ctx_side(source,obs_attr(source,source.root,"_battle_resolved_attacker",false));const auto ef_unused=ef_unused_dice(source);

    // Package M: the free-Ents in-place variant uses the public executor's stable branch (no `!free_ents` exclusion).
    require(side==0 || side==1,"tile event side must be FP or SH");
    const auto entry=ev_card(t,id);
    if(entry!=missing && (((text_is(t,obs_attr(t,entry,"card_name"),"Phial of Galadriel") || text_is(t,obs_attr(t,entry,"card_name"),"Elven Rope") || text_is(t,obs_attr(t,entry,"card_name"),"\x53\x6d\xc3\xa9\x61\x67\x6f\x6c\x20\x48\x65\x6c\x70\x73\x20\x4e\x69\x63\x65\x20\x4d\x61\x73\x74\x65\x72")) || (side==1 && (text_is(t,obs_attr(t,entry,"card_name"),"Shelob's Lair") || text_is(t,obs_attr(t,entry,"card_name"),"The Ring is Mine!") || text_is(t,obs_attr(t,entry,"card_name"),"On On They Went")))) || ((text_is(t,obs_attr(t,entry,"card_name"),"Elven Cloaks") || text_is(t,obs_attr(t,entry,"card_name"),"Give it to Uss!")))))quiet_action_gate(source,t,false,false,false,false,false,false,false,false,false,false,free_ents,true,false,false);
    else quiet_action_gate(source,t,false,false,false,false,false,false,false,false,false,false,(false||free_ents));
    ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained) || ctx_side(source,retained)==side,"tile event belongs to retained Ring actor");
    const auto card=ev_card(t,id);require(card!=missing && ctx_side(t,obs_attr(t,card,"side"))==side,"tile event card unknown or wrong side");
    const auto templ=eh_template(t,card);require(templ!=missing,"card is not a special Hunt-tile event");
    require(contains_string(source,tc_hand(source,side),id),"tile event card absent from hand");require(ev_requirement(source,t,card),"tile event requirement false");
    const auto die=free_ents?missing:ev_payment(source,t,side,index,card);const std::string face(free_ents?std::string_view{}:enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace"));
    const auto deck=ev_lower(text(t,obs_attr(t,card,"deck")));
    
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    if(!free_ents)rr_set_attr(s,ev_payment(s,t,side,index,card),"used",rr_scalar(s,WOTR_BOOL,1));
    eh_stage(s,t,templ,rng);
    const auto hand=tc_hand(s,side);auto cards=child_edges(s,hand);auto pos=std::find_if(cards.begin(),cards.end(),[&](uint64_t c){return text_is(s,c,id);});
    require(pos!=cards.end(),"tile card disappeared during effect");const auto played=*pos;cards.erase(pos);set_edges(s,hand,cards);
    const std::string key=(side==0?"fp_":"sh_")+deck;const auto pile=field(s,obs_attr(s,s.root,"discard_face_up"),key.c_str());
    auto discard=child_edges(s,pile);discard.push_back(played);set_edges(s,pile,discard);tc_open_hand_limit_discard(s);
    if(side==0 && !contains_string(s,tc_hand(s,side),id))ev_gandalf_after_event(s,id,deck,face);
    epa_after_event(s,side,id,deck,face);
    rr_set_attr(s,s.root,"_battle_resolved_attacker",rr_scalar(s,WOTR_NONE));
    

    if(free_ents)ef_finish_free(s,t,rng,ef_marker,ef_unused,ef_events,ef_annotations);
}
fm_Result apply_special_tile_event_action(const Snapshot& source,const Snapshot& t,const void* random_source,int32_t side,int64_t index,std::string_view id){
    require(side==0 || side==1,"tile event side must be FP or SH");
    const auto entry=ev_card(t,id);
    if(entry!=missing && (((text_is(t,obs_attr(t,entry,"card_name"),"Phial of Galadriel") || text_is(t,obs_attr(t,entry,"card_name"),"Elven Rope") || text_is(t,obs_attr(t,entry,"card_name"),"\x53\x6d\xc3\xa9\x61\x67\x6f\x6c\x20\x48\x65\x6c\x70\x73\x20\x4e\x69\x63\x65\x20\x4d\x61\x73\x74\x65\x72")) || (side==1 && (text_is(t,obs_attr(t,entry,"card_name"),"Shelob's Lair") || text_is(t,obs_attr(t,entry,"card_name"),"The Ring is Mine!") || text_is(t,obs_attr(t,entry,"card_name"),"On On They Went")))) || (text_is(t,obs_attr(t,entry,"card_name"),"Elven Cloaks") || text_is(t,obs_attr(t,entry,"card_name"),"Give it to Uss!"))))quiet_stable_action_gate(source,t);
    else quiet_action_gate(source,t);
    ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained) || ctx_side(source,retained)==side,"tile event belongs to retained Ring actor");
    const auto card=ev_card(t,id);require(card!=missing && ctx_side(t,obs_attr(t,card,"side"))==side,"tile event card unknown or wrong side");
    const auto templ=eh_template(t,card);require(templ!=missing,"card is not a special Hunt-tile event");
    require(contains_string(source,tc_hand(source,side),id),"tile event card absent from hand");require(ev_requirement(source,t,card),"tile event requirement false");
    const auto die=ev_payment(source,t,side,index,card);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace"));
    const auto deck=ev_lower(text(t,obs_attr(t,card,"deck")));
    
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());
    am_Builder ef_events;std::vector<uint64_t> ef_annotations;
    apply_special_tile_event_action_inplace(*result.state,t,result.random.get(),side,index,id,false,ef_events,ef_annotations);
    ef_events.out.root=ef_events.container(WOTR_LIST,ef_annotations);result.events=std::make_unique<Snapshot>(std::move(ef_events.out));return result;
}
