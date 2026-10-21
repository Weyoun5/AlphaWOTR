// Card-funded Nazgul flights and their deferred Search/Strike consequences.
// Uses ordinary flight geometry but never gives the Mouth a card-funded leg.
std::string enh_kind(const Snapshot& t,uint64_t card){
    if(card==missing)return {};const auto name=obs_attr(t,card,"card_name");
    if(text_is(t,name,"Nazgûl Strike!"))return "strike";
    if(text_is(t,name,"Nazgûl Search"))return "search";return {};
}
bool enh_pending(const Snapshot& s){
    const auto p=obs_attr(s,s.root,"pending_card_option",false);if(ctx_none(s,p) || !text_is(s,field(s,p,"side",false),"sh") || !text_is(s,field(s,p,"decider",false),"sh") || !truth(s,field(s,p,"direct",false)))return false;
    const auto name=field(s,p,"card_name",false);if(text_is(s,name,"Nazgûl Strike!"))return true;
    if(!text_is(s,name,"__nazgul_moves__"))return false;const auto options=field(s,p,"options",false);if(ctx_none(s,options) || !truth(s,options))return false;
    const auto post=field(s,child_edges(s,options).front(),"_naz_post",false);return text_is(s,post,"search") || text_is(s,post,"strike");
}
bool enh_colocated(const Snapshot& s){
    const auto army=mu_army(s,glb_fellowship_key(s,obs_attr(s,s.root,"fellowship")));if(army==missing)return false;
    for(auto figure:child_edges(s,obs_attr(s,army,"leaders")))if(nz_plain(s,figure) || nz_witch_chain(s,figure))return true;return false;
}
void enh_post(Snapshot& s,const Snapshot& t,std::string_view id,std::string_view kind){
    if(!enh_colocated(s))return;
    if(kind=="search"){
        const auto fs=obs_attr(s,s.root,"fellowship");rr_set_attr(s,fs,"hidden",rr_scalar(s,WOTR_BOOL));
        if(!truth(s,obs_attr(s,fs,"mordor_entered"))){rr_set_attr(s,s.root,"_reveal_place_due",rr_scalar(s,WOTR_BOOL,1));rr_set_attr(s,s.root,"_sh_tile_drawn_regions",rr_scalar(s,WOTR_SET));}return;
    }
    require(kind=="strike","Unknown Nazgul card post effect");std::vector<uint64_t> options{ev_dict(s,{{"action",rr_string(s,"hunt")}})};
    for(auto card:child_edges(s,obs_attr(s,obs_attr(s,s.root,"persistent_cards"),"fp_cards"))){const auto definition=ev_card(t,text(s,obs_attr(s,card,"card_id")));
        if(definition!=missing && ev_lower(text(t,obs_attr(t,definition,"deck")))=="character")options.push_back(ev_dict(s,{{"action",rr_string(s,"discard_table_card")},{"target_card",obs_attr(s,card,"card_id")}}));}
    options.push_back(ev_dict(s,{{"action",rr_string(s,"none")}}));rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,"Nazgûl Strike!")},{"side",rr_string(s,"sh")},{"decider",rr_string(s,"sh")},{"direct",rr_scalar(s,WOTR_BOOL,1)},{"options",cp_container(s,WOTR_LIST,options)}}));
}
bool enh_open(Snapshot& s,const Snapshot& t,std::string_view id,std::string_view kind,const nz_Arrivals& arrivals){
    const auto options=nz_options(s,t,arrivals);auto rows=child_edges(s,options);if(rows.empty())return false;
    rows.push_back(ev_dict(s,{{"_naz_done",rr_scalar(s,WOTR_BOOL,1)}}));for(auto row:rows)fbr_put(s,row,"_naz_post",rr_string(s,kind));set_edges(s,options,rows);
    rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,"__nazgul_moves__")},{"side",rr_string(s,"sh")},{"decider",rr_string(s,"sh")},{"direct",rr_scalar(s,WOTR_BOOL,1)},{"options",options}}));return true;
}
bool enh_crebain_open(Snapshot& s,uint64_t pending,uint64_t choice){
    if(!ec_open(s,-1))return false;const auto cb=obs_attr(s,s.root,"pending_card_option"),resume=field(s,cb,"resume_action");
    const auto choices=ev_dict(s,{{"action",field(s,choice,"action")},{"_crebain_resolved",rr_scalar(s,WOTR_BOOL,1)}});
    fbr_put(s,resume,"action_type",rr_string(s,"card_option"));fbr_put(s,resume,"params",ev_dict(s,{{"choices",choices},{"_restore_pco",pending}}));return true;
}
void enh_strike(Snapshot& s,const Snapshot& t,void* rng,uint64_t pending,uint64_t choice,am_Builder& events,std::vector<uint64_t>& rows,bool resolved=false){
    const auto action=text(s,field(s,choice,"action"));require(action=="hunt" || action=="discard_table_card" || action=="none","Unknown Nazgul Strike choice");
    if(action=="none")return;
    if(action=="hunt" && !resolved && enh_crebain_open(s,pending,choice))return;
    require(enh_colocated(s),"Nazgul Strike requires a Nazgul with the Ring-bearers");
    if(action=="discard_table_card"){
        const auto id=field(s,choice,"target_card"),definition=ev_card(t,text(s,id));require(definition!=missing && ctx_side(t,obs_attr(t,definition,"side"))==0 && ev_lower(text(t,obs_attr(t,definition,"deck")))=="character","Nazgul Strike target must be FP Character table card");
        const auto persistent=obs_attr(s,s.root,"persistent_cards");uint64_t target=missing;for(auto card:child_edges(s,obs_attr(s,persistent,"fp_cards")))if(text(s,obs_attr(s,card,"card_id"))==text(s,id)){target=card;break;}
        require(target!=missing,"Nazgul Strike target no longer on table");cbp_discard_table(s,cbp_remove_card(s,persistent,target));return;
    }
    int64_t shadow=0,fp=0;for(auto die:child_edges(s,obs_attr(s,s.root,"hunt_box"))){const auto side=ctx_side(s,obs_attr(s,die,"side"));shadow+=side==1;fp+=side==0;}
    const auto hits=shadow?hr_roll(s,t,rng,std::min<int64_t>(5,shadow),6,fp,events,rows):0;if(hits)ht_resolve(s,t,rng,hits,false,false,false,events,rows);
}
fm_Result apply_nazgul_hunt_event(const Snapshot& source,const Snapshot& t,const void* random_source,int32_t side,int64_t index,std::string_view id){
    require(side==1,"Nazgul event belongs to Shadow");quiet_action_gate(source,t,false,false,false,false,false,false,false,false,true,false,false,true);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained) || ctx_side(source,retained)==1,"Nazgul event violates retained Ring actor");
    const auto card=ev_card(t,id);const auto kind=enh_kind(t,card);require(!kind.empty() && ctx_side(t,obs_attr(t,card,"side"))==1,"Unsupported Nazgul event");require(contains_string(source,tc_hand(source,1),id),"Nazgul event absent from hand");require(ev_requirement(source,t,card),"Nazgul event requirement false");
    const auto die=ev_payment(source,t,side,index,card);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace")),deck(ev_lower(text(t,obs_attr(t,card,"deck"))));
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;am_Builder events;std::vector<uint64_t> rows;
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));rr_set_attr(s,ev_payment(s,t,1,index,card),"used",rr_scalar(s,WOTR_BOOL,1));
    const bool opened=enh_open(s,t,id,kind,{{"__wk_ok__",1}});if(!opened && kind=="search")enh_post(s,t,id,kind);
    // With no flights, source Strike falls through to its empty choices, so
    // no either/or is fabricated here. Search still runs its reveal check.
    ed_commit(s,1,id,deck,face);cau_finish(s,t,result.random.get(),-1,events,rows);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
std::unique_ptr<Snapshot> native_nazgul_hunt_choice_actions(const Snapshot& source,int32_t side){
    require(side==1 && enh_pending(source),"No Shadow Nazgul event choice pending");auto out=std::make_unique<Snapshot>(source);auto& s=*out;const auto p=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t index=0;
    for(auto choice:child_edges(s,field(s,p,"options")))rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",choice}})}));s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
fm_Result apply_nazgul_hunt_choice(const Snapshot& source,const Snapshot& t,const void* random_source,int32_t side,int64_t index){
    require(side==1 && enh_pending(source),"No Shadow Nazgul event choice pending");const auto pending=obs_attr(source,source.root,"pending_card_option");const auto options=child_edges(source,field(source,pending,"options"));require(index>=0 && static_cast<uint64_t>(index)<options.size(),"Nazgul event choice index out of range");
    const std::string id(text(source,field(source,pending,"card_id")));const auto kind=enh_kind(t,ev_card(t,id));require(!kind.empty(),"Nazgul event pending card mismatch");
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;am_Builder events;std::vector<uint64_t> rows;
    const auto p=obs_attr(s,s.root,"pending_card_option"),choice=child_edges(s,field(s,p,"options"))[index],marker=obs_attr(s,s.root,"_battle_resolved_attacker",false);const auto marker_side=ctx_none(s,marker)?-1:ctx_side(s,marker);
    rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));quiet_action_gate(s,t,false,false,false,false,false,false,false,false,true,false,false,true,true);ring_waiver(s,0);ring_waiver(s,1);rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    if(text_is(s,field(s,p,"card_name"),"Nazgûl Strike!"))enh_strike(s,t,result.random.get(),p,choice,events,rows);
    else{
        require(text_is(s,field(s,choice,"_naz_post"),kind),"Nazgul post effect changed");require(!truth(s,field(s,choice,"_then_followup",false)),"Unported Nazgul army followup");
        if(truth(s,field(s,choice,"_naz_done",false)))enh_post(s,t,id,kind);
        else{
            nz_Arrivals arr;const auto dict=child_edges(s,field(s,choice,"_naz_arr"));for(size_t i=0;i<dict.size();i+=2){const auto n=typed(s,dict[i+1],WOTR_INT,"Nazgul arrivals must be integers").integer;require(n>=0,"Negative Nazgul arrivals");arr.emplace_back(std::string(text(s,dict[i])),n);}
            require(nz_get(arr,"__wk_ok__")==1 && !nz_get(arr,"__mo_ok__") && !nz_get(arr,"__mo__"),"Card-funded Nazgul flights cannot include the Mouth");
            cp_erase(s,choice,"_naz_post");require(nz_current_choice(s,s,choice,t,arr),"Nazgul event leg stale or malformed");
            const std::string from(text(s,field(s,choice,"from_region"))),to(text(s,field(s,choice,"to_region")));const bool witch=truth(s,field(s,choice,"include_wk",false));uint64_t figure=missing;
            for(auto leader:child_edges(s,obs_attr(s,mu_army(s,from),"leaders")))if(witch?nz_witch_chain(s,leader):nz_plain(s,leader)){figure=leader;break;}require(figure!=missing,"Nazgul event figure absent");
            nz_transfer(s,t,from,to,{figure},false);if(witch)nz_set(arr,"__wk__",1);else nz_set(arr,to,nz_get(arr,to)+1);
            if(!enh_open(s,t,id,kind,arr))enh_post(s,t,id,kind);
        }
    }
    ehd_finish(s,t,result.random.get(),events,rows);cau_finish(s,t,result.random.get(),marker_side,events,rows);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
bool enh_crebain_pending(const Snapshot& s){
    if(!ec_pending(s))return false;const auto p=obs_attr(s,s.root,"pending_card_option"),resume=field(s,p,"resume_action",false);return !ctx_none(s,resume) && text_is(s,field(s,resume,"action_type",false),"card_option");
}
fm_Result apply_nazgul_crebain_choice(const Snapshot& source,const Snapshot& t,const void* random_source,int32_t side,int64_t index){
    require(side==1 && enh_crebain_pending(source),"No Nazgul Strike Crebain continuation");hc_gate(source,false,true,false,true);const auto p=obs_attr(source,source.root,"pending_card_option"),resume=field(source,p,"resume_action"),params=field(source,resume,"params"),restored=field(source,params,"_restore_pco"),choices=field(source,params,"choices");
    require(source.nodes[resume].count==6 && typed(source,field(source,resume,"die_idx"),WOTR_INT,"Crebain restored index must be integer").integer==-1 && source.nodes[params].count==4 && source.nodes[choices].count==4,"Crebain restored Strike action has unsupported fields");
    require(text_is(source,field(source,restored,"card_name"),"Nazgûl Strike!") && text_is(source,field(source,restored,"side"),"sh") && text_is(source,field(source,restored,"decider"),"sh") && truth(source,field(source,restored,"direct")),"Crebain restored Strike record invalid");
    require(enh_kind(t,ev_card(t,text(source,field(source,restored,"card_id"))))=="strike" && text_is(source,field(source,choices,"action"),"hunt") && typed(source,field(source,choices,"_crebain_resolved"),WOTR_BOOL,"Crebain resumed flag must be boolean").integer==1,"Crebain resumed Strike choice invalid");
    const auto options=child_edges(source,field(source,p,"options"));require(index>=0 && static_cast<uint64_t>(index)<options.size(),"Crebain index invalid");const bool use=typed(source,field(source,options[index],"crebain_discard"),WOTR_BOOL,"Crebain choice must be boolean").integer!=0;
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;am_Builder events;std::vector<uint64_t> rows;
    // action_clone rebuilds and compacts the pending subgraph. Parent node
    // IDs above are validation-only; resume through the child's own nodes.
    const auto child_pending=obs_attr(s,s.root,"pending_card_option"),child_params=field(s,field(s,child_pending,"resume_action"),"params"),child_restored=field(s,child_params,"_restore_pco"),child_choices=field(s,child_params,"choices");
    const auto marker=obs_attr(s,s.root,"_battle_resolved_attacker",false);const auto marker_side=ctx_none(s,marker)?-1:ctx_side(s,marker);
    rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    if(use){const auto card=hpr_card(s,"FLOCKS_OF_CREBAIN");if(card!=missing){cbp_discard_table(s,cbp_remove_card(s,obs_attr(s,s.root,"persistent_cards"),card));rr_set_attr(s,s.root,"_crebain_hunt_bonus",rr_scalar(s,WOTR_INT,1));}}
    enh_strike(s,t,result.random.get(),child_restored,child_choices,events,rows,true);
    // Package Z (engine_crebain_tail, 2026-09-11): the resumed Strike card_option's OWN 1.230 tail (the Strike's Hunt roll leaves the damage
    // in _pending_hunt_damage for the executor tail: offer the reduction window, else settle) runs before the outer card_option executor's
    // tail (ehd_finish's first line: with the window open the offer returns false and the settle opens the Hunt decision too).
    if(ht_integer(s,s.root,"_pending_hunt_damage")>0 && ctx_none(s,obs_attr(s,s.root,"pending_hunt")) && !hpr_offer_reduction(s))hd_settle(s,t);
    ehd_finish(s,t,result.random.get(),events,rows);cau_finish(s,t,result.random.get(),marker_side,events,rows);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
