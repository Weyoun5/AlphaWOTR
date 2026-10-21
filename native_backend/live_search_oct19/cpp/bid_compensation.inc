// Bid-compensation tokens and Dwarven Rings; physical dice remain distinct.
bool bc_has_token(const Snapshot& s,int side,bool muster){
    return side==0?truth(s,obs_attr(s,s.root,muster?"fp_political_token":"fp_card_token")):!rr_nonpositive(s,obs_attr(s,s.root,"sh_action_tokens"));
}
std::unique_ptr<Snapshot> native_token_actions(const Snapshot& s,const Snapshot& t,int32_t side){
    require(side==0||side==1,"Token actor must be FP or SH");am_Builder b;std::vector<uint64_t> rows;
    if(!truth(s,obs_attr(s,s.root,side==0?"fp_token_used_this_turn":"sh_token_used_this_turn",false))){
        if(bc_has_token(s,side,true)){
            const std::vector<std::string_view> nations=side==0?std::vector<std::string_view>{"ELVES","DWARVES","GONDOR","THE_NORTH","ROHAN"}:std::vector<std::string_view>{"ISENGARD","SAURON","SOUTHRON_EASTERLINGS"};
            for(auto nation:nations){const auto view=political_view(s,nation,false);if(can_advance(s,view,nation,"muster_die"))rows.push_back(b.row(-1,"use_token",{b.str("token_type"),b.str("muster"),b.str("nation"),b.nation(nation),b.str("side"),b.str(side==0?"fp":"sh")}));}
        }
        if(bc_has_token(s,side,false))for(const char* deck:{"character","strategy"})if(truth(s,obs_attr(s,s.root,da_deck_name(side,deck),false)))rows.push_back(b.row(-1,"use_token",{b.str("token_type"),b.str("draw"),b.str("deck"),b.str(deck),b.str("side"),b.str(side==0?"fp":"sh")}));
    }
    b.out.root=b.container(WOTR_LIST,rows);return std::make_unique<Snapshot>(std::move(b.out));
}
void bc_settle_draw(Snapshot& s,int side,std::string_view deck){
    const auto owed=obs_attr(s,s.root,"_owed_draws",false);if(ctx_none(s,owed))return;
    auto entries=child_edges(s,owed);for(size_t i=0;i<entries.size();i+=2){const auto key=entries[i];if(s.nodes[key].tag!=WOTR_TUPLE)continue;const auto pair=child_edges(s,key);
        if(pair.size()!=2||!text_is(s,pair[0],side==0?"FREE_PEOPLES":"SHADOW")||!text_is(s,pair[1],deck))continue;
        const auto count=typed(s,entries[i+1],WOTR_INT,"Draw debt must be integer").integer;
        if(count>0){entries[i+1]=rr_scalar(s,WOTR_INT,count-1);set_edges(s,owed,entries);}return;
    }
}
fm_Result apply_token_action(const Snapshot& source,const Snapshot& tables,const void* rng,int32_t side,const char* kind,const char* value){
    require((side==0||side==1)&&kind&&value,"Invalid token arguments");const bool muster=std::string_view(kind)=="muster";
    // Package P: the frozen use_token executor (ai.py:12308-12313 -> game_logic.use_action_token) reads no dice; a token row offered
    // by the unrolled turn-start menu executes under the same quiet-board proof with the roll clause waived (allow_unrolled).
    require(muster||std::string_view(kind)=="draw","Unsupported token kind");quiet_stable_action_gate(source,tables,false,muster,false,true);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==side,"Token violates retained actor");
    require(!truth(source,obs_attr(source,source.root,side==0?"fp_token_used_this_turn":"sh_token_used_this_turn",false))&&bc_has_token(source,side,muster),"Token unavailable this turn");
    if(muster){require((fp_nation(value)?0:1)==side,"Token nation belongs to another actor");const auto p=political_view(source,value,true);require(can_advance(source,p,value,"muster_die"),"Illegal token political advance");}
    else{const auto pile=obs_attr(source,source.root,da_deck_name(side,value),false);require(truth(source,pile),"Token draw deck is empty");}
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));am_Builder events;std::vector<uint64_t> rows;
    if(muster){const auto p=political_view(s,value,true);assign_box(s,p,p.box+1);}
    else{const auto card=tc_draw_event_card(s,side,value);bc_settle_draw(s,side,value);rows.push_back(events.row(-1,"card_drawn",{events.str("deck"),events.str(value),events.str("card_id"),events.str(text(s,card)),events.str("side"),events.str(side==0?"FP":"SH")}));}
    if(side==0)rr_set_attr(s,s.root,muster?"fp_political_token":"fp_card_token",rr_scalar(s,WOTR_BOOL,0));
    else rr_set_attr(s,s.root,"sh_action_tokens",rr_scalar(s,WOTR_INT,s.nodes[obs_attr(s,s.root,"sh_action_tokens")].integer-1));
    rr_set_attr(s,s.root,side==0?"fp_token_used_this_turn":"sh_token_used_this_turn",rr_scalar(s,WOTR_BOOL,1));tc_open_hand_limit_discard(s);
    events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
uint32_t bc_face_class(const Snapshot& s,uint64_t face){
    const int index=rr_face_index(enum_member(s,face,"game_state.ActionDieFace"));return index==4?(1u<<1)|(1u<<2):1u<<index;
}
bool bc_ring_available(const Snapshot& s,int side){
    if(rr_nonpositive(s,obs_attr(s,s.root,side==0?"fp_dwarven_rings":"sh_dwarven_rings",false)))return false;
    for(int owner=0;owner<2;++owner){
        if(truth(s,obs_attr(s,s.root,owner==0?"fp_token_used_this_turn":"sh_token_used_this_turn",false)))return false;
        for(auto die:child_edges(s,pa_pool(s,owner)))if(truth(s,obs_attr(s,die,"used"))&&!enum_is(s,obs_attr(s,die,"current"),"game_state.ActionDieFace","EYE"))return false;
    }return true;
}
std::vector<uint64_t> bc_ring_faces(const Snapshot& s,uint64_t die){
    std::vector<uint64_t> valid;const auto original=bc_face_class(s,obs_attr(s,die,"current"));
    for(auto face:child_edges(s,obs_attr(s,die,"faces")))if(!(bc_face_class(s,face)&original))valid.push_back(face);return valid;
}
std::unique_ptr<Snapshot> native_dwarven_ring_actions(const Snapshot& s,int32_t side){
    require(side==0||side==1,"Dwarven Ring actor must be FP or SH");am_Builder b;std::vector<uint64_t> rows;
    if(bc_ring_available(s,side)){const auto dice=child_edges(s,pa_pool(s,side));for(size_t i=0;i<dice.size();++i){const auto die=dice[i],face=obs_attr(s,die,"current");if(truth(s,obs_attr(s,die,"used"))||ctx_none(s,face)||bc_ring_faces(s,die).empty())continue;
        rows.push_back(b.row(-1,"dwarven_ring",{b.str("die_idx"),b.integer(i),b.str("old_face"),b.str(rr_face_values[rr_face_index(enum_member(s,face,"game_state.ActionDieFace"))]),b.str("side"),b.str(side==0?"fp":"sh")}));}}
    b.out.root=b.container(WOTR_LIST,rows);return std::make_unique<Snapshot>(std::move(b.out));
}
fm_Result apply_dwarven_ring_action(const Snapshot& source,const Snapshot& tables,const void* rng,int32_t side,int64_t index){
    require(side==0||side==1,"Dwarven Ring actor must be FP or SH");quiet_stable_action_gate(source,tables);ring_waiver(source,0);ring_waiver(source,1);
    require(bc_ring_available(source,side),"Dwarven Ring unavailable after turn started");const auto dice=child_edges(source,pa_pool(source,side));require(index>=0&&static_cast<uint64_t>(index)<dice.size(),"Dwarven die index out of range");
    const auto die=dice[index];require(!truth(source,obs_attr(source,die,"used"))&&!ctx_none(source,obs_attr(source,die,"current")),"Dwarven Ring needs unused rolled die");require(!bc_ring_faces(source,die).empty(),"No different face class exists");
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==side,"Dwarven Ring violates retained actor");
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;
    const auto next_die=child_edges(s,pa_pool(s,side))[index];const auto faces=bc_ring_faces(s,next_die);const auto pick=hf_choice(result.random.get(),faces.size(),false);
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));rr_set_attr(s,next_die,"current",faces[pick]);
    if(side==1&&enum_is(s,faces[pick],"game_state.ActionDieFace","EYE")){rr_set_attr(s,next_die,"used",rr_scalar(s,WOTR_BOOL,1));const auto box=obs_attr(s,s.root,"hunt_box");auto entries=child_edges(s,box);entries.push_back(next_die);set_edges(s,box,entries);rr_set_attr(s,s.root,"_battle_resolved_attacker",rr_scalar(s,WOTR_NONE));}
    const char* stock=side==0?"fp_dwarven_rings":"sh_dwarven_rings";rr_set_attr(s,s.root,stock,rr_scalar(s,WOTR_INT,s.nodes[obs_attr(s,s.root,stock)].integer-1));tc_open_hand_limit_discard(s);
    am_Builder events;events.out.root=events.container(WOTR_LIST,{});result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
