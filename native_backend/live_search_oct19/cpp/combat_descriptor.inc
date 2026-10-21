// Revalidate resident combat descriptors before their complete action handler.
fm_Result apply_native_combat_descriptor(const Snapshot& s,const Snapshot& t,const Snapshot& actions,const void* rng,int32_t side,uint64_t row){
    const auto parts=child_edges(actions,row);const std::string kind(text(actions,parts[1]));
    const auto nation=field(actions,parts[2],"nation",false);
    const bool explicit_press=kind=="siege_press"&&nation!=missing&&child_edges(actions,parts[2]).size()==2;
    auto legal=native_combat_pending_actions(s,t,side);int64_t selected=-1,index=0;
    for(auto candidate:child_edges(*legal,legal->root)){
        const auto base=child_edges(*legal,candidate);
        const bool refinement=explicit_press&&text_is(*legal,base[1],"siege_press")&&child_edges(*legal,base[2]).empty()&&ap_equal(ap_freeze(*legal,base[0]),ap_freeze(actions,parts[0]));
        if(refinement||ap_equal(ap_freeze(*legal,candidate),ap_freeze(actions,row))){selected=index;break;}++index;
    }
    require(selected>=0,"Combat descriptor is stale or belongs to another actor");
    const auto pc=obs_attr(s,s.root,"pending_combat");
    const auto phase=field(s,pc,"phase");
    // An army that disappeared before an action yields the source's no-card
    // cleanup descriptor even outside the normal card-commitment phase.
    if(kind=="combat_card_none")return apply_combat_card(s,t,rng,nullptr,false);
    if(text_is(s,phase,"attacker_combat_card")||text_is(s,phase,"defender_combat_card")){
        const auto parts=child_edges(actions,row);const auto p=parts[2],id=field(actions,p,"card_id",false);const bool from_table=truth(actions,field(actions,p,"from_table",false));
        const std::string card=ctx_none(actions,id)?std::string():std::string(text(actions,id));return apply_combat_card(s,t,rng,ctx_none(actions,id)?nullptr:card.c_str(),from_table);
    }
    if(text_is(s,phase,"fp_shine"))return apply_combat_shine(s,t,rng,selected==0?1:0);
    if(text_is(s,field(s,pc,"phase"),"casualty_allocation"))return apply_combat_casualty(s,t,rng,selected);
    if(!text_is(s,phase,"combat_effect_choice")){
        if(kind=="victor_advance")return apply_combat_advance(s,t,rng,selected);
        if(kind=="combat_retreat")return apply_combat_retreat(s,t,rng,std::string(text(actions,field(actions,parts[2],"to_region"))).c_str());
        cpl_Request request;request.kind=kind;const auto count=field(actions,parts[2],"elites_back",false);
        if(kind=="siege_press"&&!ctx_none(actions,nation)){request.nation=actions.nodes[nation].tag==WOTR_ENUM?su_nation_value(enum_member(actions,nation,"game_state.Nation")):std::string(text(actions,nation));request.has_nation=true;}
        if(!ctx_none(actions,count))request.elites_back=typed(actions,count,WOTR_INT,"Siege trim count must be integer").integer;
        return apply_combat_phase(s,t,rng,request);
    }
    const auto effect=field(s,pc,"effect_choice");
    if(text_is(s,field(s,effect,"stage"),"pre"))return apply_combat_pre_target(s,t,rng,selected);
    if(text_is(s,field(s,effect,"stage"),"round"))return apply_combat_round_choice(s,t,rng,selected);
    if(text_is(s,field(s,effect,"stage"),"onslaught_payment"))return apply_combat_onslaught_payment(s,t,rng,selected);
    return apply_combat_onslaught(s,t,rng,selected);
}
