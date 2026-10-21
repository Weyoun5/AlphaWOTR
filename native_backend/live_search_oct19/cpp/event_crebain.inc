// Learned pre-roll Crebain decision; the original Fellowship die is not paid
// until the Shadow has chosen. Strike's Hunt-only resume is dispatched below
// to its distinct executor so it cannot accidentally spend an FP die.
bool enh_crebain_pending(const Snapshot& s);
fm_Result apply_nazgul_crebain_choice(const Snapshot& source,const Snapshot& t,const void* random_source,int32_t side,int64_t index);
bool ec_pending(const Snapshot& s){
    const auto p=obs_attr(s,s.root,"pending_card_option",false);
    return !ctx_none(s,p) && text_is(s,field(s,p,"card_name",false),"Flocks of Crebain") && text_is(s,field(s,p,"side",false),"sh") && text_is(s,field(s,p,"decider",false),"sh") && truth(s,field(s,p,"direct",false));
}
bool ec_open(Snapshot& s,int64_t index){
    const auto card=hpr_card(s,"FLOCKS_OF_CREBAIN");if(card==missing || truth(s,obs_attr(s,obs_attr(s,s.root,"fellowship"),"mordor_entered")))return false;
    bool hunt=false;for(auto die:child_edges(s,obs_attr(s,s.root,"hunt_box")))hunt |= enum_is(s,obs_attr(s,die,"side"),"game_state.Side","SHADOW");if(!hunt)return false;
    const auto yes=ev_dict(s,{{"crebain_discard",rr_scalar(s,WOTR_BOOL,1)}}),no=ev_dict(s,{{"crebain_discard",rr_scalar(s,WOTR_BOOL)}}),options=rr_scalar(s,WOTR_LIST);set_edges(s,options,{yes,no});
    const auto resume=ev_dict(s,{{"die_idx",rr_scalar(s,WOTR_INT,index)},{"action_type",rr_string(s,"fellowship_move")},{"params",ev_dict(s,{{"_crebain_resolved",rr_scalar(s,WOTR_BOOL,1)}})}});
    rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",obs_attr(s,card,"card_id")},{"card_name",rr_string(s,"Flocks of Crebain")},{"side",rr_string(s,"sh")},{"decider",rr_string(s,"sh")},{"direct",rr_scalar(s,WOTR_BOOL,1)},{"options",options},{"resume_action",resume}}));return true;
}
std::unique_ptr<Snapshot> native_crebain_choice_actions(const Snapshot& source,int32_t side){
    require(side==1,"Crebain choice belongs to Shadow");require(ec_pending(source),"No Crebain choice pending");
    require(text_is(source,field(source,field(source,obs_attr(source,source.root,"pending_card_option"),"resume_action"),"action_type"),"fellowship_move") || enh_crebain_pending(source),"Unsupported Crebain continuation");
    auto out=std::make_unique<Snapshot>(source);auto& s=*out;const auto pending=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t index=0;
    for(auto choice:child_edges(s,field(s,pending,"options"))){typed(s,field(s,choice,"crebain_discard"),WOTR_BOOL,"Crebain choice must be boolean");const auto row=rr_scalar(s,WOTR_TUPLE);set_edges(s,row,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,pending,"card_id")},{"card_name",field(s,pending,"card_name")},{"side",field(s,pending,"side")},{"choices",choice}})});rows.push_back(row);}
    require(rows.size()==2,"Crebain choice needs use and keep rows");s.root=rr_scalar(s,WOTR_LIST);set_edges(s,s.root,rows);return ac_compact(s);
}
fm_Result apply_crebain_choice(const Snapshot& source,const Snapshot& t,const void* random_source,int32_t side,int64_t index){
    if(enh_crebain_pending(source))return apply_nazgul_crebain_choice(source,t,random_source,side,index);
    require(side==1,"Crebain choice belongs to Shadow");require(ec_pending(source),"No Crebain choice pending");hc_gate(source,false,true,false,true);
    const auto pending=obs_attr(source,source.root,"pending_card_option"),resume=field(source,pending,"resume_action"),params=field(source,resume,"params");
    require(text_is(source,field(source,resume,"action_type"),"fellowship_move"),"Crebain native continuation currently requires Fellowship movement");
    const auto keys=child_edges(source,params);require(keys.size()==2 && text_is(source,keys[0],"_crebain_resolved") && typed(source,keys[1],WOTR_BOOL,"Crebain resumed flag must be boolean").integer==1,"Crebain resumed move has unsupported parameters");
    const auto options=child_edges(source,field(source,pending,"options"));require(index>=0 && static_cast<uint64_t>(index)<options.size(),"Crebain index out of range");const bool use=typed(source,field(source,options[index],"crebain_discard"),WOTR_BOOL,"Crebain choice must be boolean").integer!=0;
    const auto die=typed(source,field(source,resume,"die_idx"),WOTR_INT,"Crebain original die index must be integer").integer;
    auto child=action_clone(source);auto& s=*child;rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));
    if(use){const auto card=hpr_card(s,"FLOCKS_OF_CREBAIN");if(card!=missing){cbp_discard_table(s,cbp_remove_card(s,obs_attr(s,s.root,"persistent_cards"),card));rr_set_attr(s,s.root,"_crebain_hunt_bonus",rr_scalar(s,WOTR_INT,1));}}
    auto result=apply_fellowship_move(s,t,random_source,die,true);auto& c=*result.state;
    // Package Z (engine_crebain_tail, 2026-09-11): the frozen card_option executor runs its 1.230 settle tail AFTER the resumed action
    // (ai.py: `if _pending_hunt_damage > 0 and pending_hunt is None: if not _offer_hunt_reduction(state): _settle_hunt_damage(state)`) -
    // when the resumed move's own tail has just opened the Axe and Bow / Horn of Gondor reduction window, _offer_hunt_reduction returns
    // False and _settle_hunt_damage opens the Hunt sacrifice decision as well: BOTH pendings open, the hunt decision rows decide first
    // (served batch 34005 g1 d31: the native leaf carried the window alone - a 2-row menu against the frozen 3-row hunt menu at the same
    // in-tree node, one root visit moved).  hpr_offer_reduction returns false while the window is open, exactly as the frozen offer.
    if(ht_integer(c,c.root,"_pending_hunt_damage")>0 && ctx_none(c,obs_attr(c,c.root,"pending_hunt")) && !hpr_offer_reduction(c))hd_settle(c,t);
    return result;
}
