// Worn is optional: Shadow chooses a named table card, random hand card, or decline.
bool ew_pending(const Snapshot& s){
    const auto p=obs_attr(s,s.root,"pending_card_option",false);
    return !ctx_none(s,p) && text_is(s,field(s,p,"card_id",false),"Worn with Sorrow and Toil") && text_is(s,field(s,p,"card_name",false),"__wwsat_discard__") && text_is(s,field(s,p,"side",false),"sh") && text_is(s,field(s,p,"decider",false),"sh") && truth(s,field(s,p,"direct",false));
}
void ew_debt_gate(const Snapshot& s){
    const auto debts=obs_attr(s,s.root,"_wwsat_owed",false);if(ctx_none(s,debts))return;
    typed(s,debts,WOTR_LIST,"Worn debts must be list");for(auto debt:child_edges(s,debts))require(typed(s,debt,WOTR_BOOL,"Worn debt marker must be bool").integer==1,"Worn debt marker must be true");
}
void ew_open(Snapshot& s,void* rng){
    if(ctx_none(s,obs_attr(s,s.root,"pending_combat")) && ctx_none(s,obs_attr(s,s.root,"pending_card_option",false)) && ctx_none(s,obs_attr(s,s.root,"pending_hunt")) && truth(s,obs_attr(s,s.root,"_wwsat_owed",false)))cau_wwsat(s,rng);
}
std::unique_ptr<Snapshot> native_worn_choice_actions(const Snapshot& source,int32_t side){
    require(side==1,"Worn discard choice belongs to Shadow");require(ew_pending(source),"No Worn choice pending");auto out=std::make_unique<Snapshot>(source);auto& s=*out;const auto pending=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t index=0;
    for(auto choice:child_edges(s,field(s,pending,"options"))){
        const auto family=field(s,choice,"wwsat_source",false);require(text_is(s,family,"hand") || text_is(s,family,"table") || text_is(s,family,"decline"),"Worn source must be hand, table, or decline");
        const auto row=rr_scalar(s,WOTR_TUPLE);set_edges(s,row,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,pending,"card_id")},{"card_name",field(s,pending,"card_name")},{"side",field(s,pending,"side")},{"choices",choice}})});rows.push_back(row);
    }
    s.root=rr_scalar(s,WOTR_LIST);set_edges(s,s.root,rows);return ac_compact(s);
}
fm_Result apply_worn_choice(const Snapshot& source,const Snapshot& t,const void* random_source,int32_t side,int64_t index){
    require(side==1,"Worn discard choice belongs to Shadow");require(ew_pending(source),"No Worn choice pending");hc_gate(source,false,true,true,true);ew_debt_gate(source);
    const auto options=child_edges(source,field(source,obs_attr(source,source.root,"pending_card_option"),"options"));require(index>=0 && static_cast<uint64_t>(index)<options.size(),"Worn choice index out of range");
    const auto pick=options[index];const auto family=field(source,pick,"wwsat_source",false);require(text_is(source,family,"hand") || text_is(source,family,"table") || text_is(source,family,"decline"),"Worn choice must select a source or decline");
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;am_Builder events;std::vector<uint64_t> rows;
    const auto choice=child_edges(s,field(s,obs_attr(s,s.root,"pending_card_option"),"options"))[index];
    rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    cbp_wwsat_resolve(s,result.random.get(),field(s,choice,"wwsat_source"),field(s,choice,"card_id",false));ehd_finish(s,t,result.random.get(),events,rows);cau_gandalf(s);
    events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
