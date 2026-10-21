// Black Captain producer; shared Nazgul/army continuations remain
// in event_ringwraiths.inc and card_army_followup.inc.
bool ebc_supported(const Snapshot& t,uint64_t card){return card!=missing&&text_is(t,obs_attr(t,card,"card_name"),"The Black Captain Commands");}
bool ebc_pending(const Snapshot& s){const auto p=obs_attr(s,s.root,"pending_card_option",false);return !ctx_none(s,p)&&text_is(s,field(s,p,"card_name",false),"The Black Captain Commands");}
std::string ebc_wk_region(const Snapshot& s){
    const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));
    for(size_t i=0;i<armies.size();i+=2)for(auto figure:child_edges(s,obs_attr(s,armies[i+1],"leaders"))){
        const auto name=obs_attr(s,figure,"name",false);if(ctx_none(s,name))continue;std::string normalized;
        for(char c:text(s,name)){if(c>='A'&&c<='Z')c=char(c-'A'+'a');if(c>='a'&&c<='z')normalized.push_back(c);}
        if(normalized.find("witchking")!=std::string::npos)return sanitize_name(text(s,armies[i]));
    }return {};
}
uint64_t ebc_options(Snapshot& s,const Snapshot& t){
    std::vector<uint64_t> rows;const auto region=ebc_wk_region(s);
    if(!region.empty()&&mu_available(s,t,"SAURON","leader")>0)rows.push_back(ev_dict(s,{{"action",rr_string(s,"recruit")},{"witch_king_region",rr_string(s,region)}}));
    rows.push_back(ev_dict(s,{{"action",rr_string(s,"nazgul_moves")}}));return cp_container(s,WOTR_LIST,rows);
}
bool ebc_recruit(Snapshot& s,const Snapshot& t,std::string_view region){
    if(mu_available(s,t,"SAURON","leader")<=0)return false;
    int holder=-1;const auto control=child_edges(s,obs_attr(s,s.root,"enemy_controlled"));
    for(size_t i=0;i<control.size();i+=2)if(contains_string(s,control[i+1],region)){holder=ctx_side(s,control[i]);break;}
    const auto army=mu_army(s,region);bool own=false,enemy=false;
    if(army!=missing)for(auto unit:child_edges(s,obs_attr(s,army,"units")))(ctx_unit_side(s,unit)==1?own:enemy)=true;
    if(holder>=0&&holder!=1&&!own)return false;
    if(enemy&&!contains_string(s,obs_attr(s,s.root,"besieged_strongholds"),region))return false;
    // Nazgul may stand alone: er_place's plain-leader escort guard does not
    // apply. Reuse the shared figure constructor after these exact guards.
    mu_place(s,"SAURON",region,"leader");return true;
}
void ebc_effect(Snapshot& s,const Snapshot& t,uint64_t choice){
    const auto mode=text(s,field(s,choice,"action"));
    if(mode=="recruit"){
        const std::string region(text(s,field(s,choice,"witch_king_region")));for(int i=0;i<2;++i)ebc_recruit(s,t,region);
        erw_grant(s,"wk_army",1);return;
    }
    require(mode=="nazgul_moves","Unknown Black Captain mode");const auto fu=ev_dict(s,{{"kind",rr_string(s,"wk_army")},{"moves",rr_scalar(s,WOTR_INT,1)}});
    // Unlike Ringwraiths, the pinned BCC handler seeds the pseudo card ID.
    if(!erw_open(s,t,"__nazgul_moves__",{{"__wk_ok__",1}},fu))erw_grant(s,"wk_army",1);
}
fm_Result apply_black_captain_event(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index,std::string_view id){
    require(side==1,"Black Captain belongs to Shadow");quiet_stable_action_gate(source,t);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==side,"Black Captain violates retained Ring actor");
    const auto card=ev_card(t,id);require(ebc_supported(t,card)&&ctx_side(t,obs_attr(t,card,"side"))==side&&contains_string(source,tc_hand(source,side),id),"Black Captain absent from Shadow hand");require(ev_requirement(source,t,card),"Black Captain requirement false");
    const auto die=ev_payment(source,t,side,index,card);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace"));
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;
    rr_set_attr(s,s.root,"_casualty_operations",cp_container(s,WOTR_LIST,{}));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));rr_set_attr(s,ev_payment(s,t,side,index,card),"used",rr_scalar(s,WOTR_BOOL,1));
    const auto options=ebc_options(s,t);const auto rows=child_edges(s,options);
    if(rows.size()>1){
        rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,"The Black Captain Commands")},{"side",rr_string(s,"sh")},{"decider",rr_string(s,"sh")},{"options",options},{"die_face",rr_string(s,face)},{"deck",rr_string(s,"character")}}));tc_open_hand_limit_discard(s);
    }else{require(rows.size()==1,"Black Captain always offers its flight arm");ebc_effect(s,t,rows.front());ed_commit(s,side,id,"character",face);}
    am_Builder events;std::vector<uint64_t> trace;cau_finish(s,t,result.random.get(),-1,events,trace);events.out.root=events.container(WOTR_LIST,trace);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
uint64_t ebc_validate_pending(const Snapshot& s,const Snapshot& t){
    require(ebc_pending(s),"No Black Captain modal choice pending");const auto p=obs_attr(s,s.root,"pending_card_option"),card=ev_card(t,text(s,field(s,p,"card_id")));
    require(ebc_supported(t,card)&&text_is(s,field(s,p,"side"),"sh")&&text_is(s,field(s,p,"decider"),"sh")&&!truth(s,field(s,p,"direct",false)),"Black Captain pending identity or owner mismatch");
    require(text_is(s,field(s,p,"deck"),"character")&&(text_is(s,field(s,p,"die_face"),"CHARACTER")||text_is(s,field(s,p,"die_face"),"EVENT")),"Black Captain pending payment metadata invalid");
    Snapshot view(s);const auto current=ebc_options(view,t),given=field(s,p,"options");require(s.nodes[given].count==2&&ap_equal(ap_freeze(s,given),ap_freeze(view,current)),"Black Captain modal options are stale");return p;
}
std::unique_ptr<Snapshot> native_black_captain_choices(const Snapshot& source,const Snapshot& t,int side){
    require(side==1,"Black Captain choice belongs to Shadow");ebc_validate_pending(source,t);Snapshot s(source);const auto p=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t index=0;
    for(auto choice:child_edges(s,field(s,p,"options")))rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",choice}})}));s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
fm_Result apply_black_captain_choice(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index){
    require(side==1,"Black Captain choice belongs to Shadow");const auto pending=ebc_validate_pending(source,t),options=field(source,pending,"options");require(index>=0&&static_cast<uint64_t>(index)<source.nodes[options].count,"Black Captain choice index invalid");
    const std::string id(text(source,field(source,pending,"card_id"))),face(text(source,field(source,pending,"die_face")));const int marker=ctx_side(source,obs_attr(source,source.root,"_battle_resolved_attacker",false));
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;const auto p=obs_attr(s,s.root,"pending_card_option"),choice=child_edges(s,field(s,p,"options"))[index];
    rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));quiet_stable_action_gate(s,t);ring_waiver(s,0);ring_waiver(s,1);rr_set_attr(s,s.root,"_casualty_operations",cp_container(s,WOTR_LIST,{}));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));ebc_effect(s,t,choice);ed_commit(s,side,id,"character",face);
    am_Builder events;std::vector<uint64_t> rows;ehd_finish(s,t,result.random.get(),events,rows);cau_finish(s,t,result.random.get(),marker,events,rows);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
