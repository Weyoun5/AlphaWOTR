// Worn debt is mandatory (1.210g), but the Shadow learns the hand/table source choice (1.211f).
bool ew_pending(const Snapshot& s){
    const auto p=obs_attr(s,s.root,"pending_card_option",false);
    return !ctx_none(s,p) && text_is(s,field(s,p,"card_id",false),"Worn with Sorrow and Toil") && text_is(s,field(s,p,"card_name",false),"__wwsat_discard__") && text_is(s,field(s,p,"side",false),"sh") && text_is(s,field(s,p,"decider",false),"sh") && truth(s,field(s,p,"direct",false));
}
void ew_debt_gate(const Snapshot& s){
    const auto debts=obs_attr(s,s.root,"_wwsat_owed",false);if(ctx_none(s,debts))return;
    typed(s,debts,WOTR_LIST,"Worn debts must be list");for(auto debt:child_edges(s,debts))require(typed(s,debt,WOTR_BOOL,"Worn debt marker must be bool").integer==1,"Worn debt marker must be true");
}
void ew_open(Snapshot& s,void* rng){
    // R2-SL-00: an open hand-limit discard settles first (twin of the ai.py finally drain guard).
    if(ctx_none(s,obs_attr(s,s.root,"pending_combat")) && ctx_none(s,obs_attr(s,s.root,"pending_card_option",false)) && ctx_none(s,obs_attr(s,s.root,"pending_hunt")) && ctx_none(s,obs_attr(s,s.root,"pending_discard",false)) && truth(s,obs_attr(s,s.root,"_wwsat_owed",false)))cau_wwsat(s,rng);
}
// R2-SL-00: re-derive the picked source at answer time (twin of the ai.py executor). The options were frozen at open and a
// decision served first (an FP hand-limit discard, a table card leaving play) can remove the picked source.
bool ew_choice_live(const Snapshot& s,uint64_t choice){
    const auto source=field(s,choice,"wwsat_source"),card_id=field(s,choice,"card_id",false);
    if(text_is(s,source,"hand")){
        const auto hand=obs_attr(s,ctx_side_value(s,obs_attr(s,s.root,"hands"),0),"hand");
        for(auto id:child_edges(s,hand)){const auto name=text(s,id);if(name.find("_Character_")!=std::string_view::npos || (name.substr(0,3)=="HU_" && name.find("_CHAR_")!=std::string_view::npos))return true;}
        return false;
    }
    const auto persistent=obs_attr(s,s.root,"persistent_cards",false);size_t count=0;bool named=false;
    if(!ctx_none(s,persistent))for(auto card:child_edges(s,obs_attr(s,persistent,"fp_cards"))){const auto id=text(s,obs_attr(s,card,"card_id"));if(id.find("Character")==std::string_view::npos)continue;++count;named|=(!ctx_none(s,card_id) && id==text(s,card_id));}
    return ctx_none(s,card_id)?count==1:named;
}
std::unique_ptr<Snapshot> native_worn_choice_actions(const Snapshot& source,int32_t side){
    require(side==1,"Worn discard choice belongs to Shadow");require(ew_pending(source),"No Worn choice pending");auto out=std::make_unique<Snapshot>(source);auto& s=*out;const auto pending=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t index=0;
    for(auto choice:child_edges(s,field(s,pending,"options"))){
        const auto family=field(s,choice,"wwsat_source",false);require(text_is(s,family,"hand") || text_is(s,family,"table"),"Worn source must be hand or table; no decline");
        const auto row=rr_scalar(s,WOTR_TUPLE);set_edges(s,row,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,pending,"card_id")},{"card_name",field(s,pending,"card_name")},{"side",field(s,pending,"side")},{"choices",choice}})});rows.push_back(row);
    }
    s.root=rr_scalar(s,WOTR_LIST);set_edges(s,s.root,rows);return ac_compact(s);
}
fm_Result apply_worn_choice(const Snapshot& source,const Snapshot& t,const void* random_source,int32_t side,int64_t index){
    require(side==1,"Worn discard choice belongs to Shadow");require(ew_pending(source),"No Worn choice pending");hc_gate(source,false,true,true,true);ew_debt_gate(source);
    const auto options=child_edges(source,field(source,obs_attr(source,source.root,"pending_card_option"),"options"));require(index>=0 && static_cast<uint64_t>(index)<options.size(),"Worn choice index out of range");
    const auto pick=options[index];const auto family=field(source,pick,"wwsat_source",false);require(text_is(source,family,"hand") || text_is(source,family,"table"),"Worn choice must select hand or table; no decline");
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;am_Builder events;std::vector<uint64_t> rows;
    const auto choice=child_edges(s,field(s,obs_attr(s,s.root,"pending_card_option"),"options"))[index];
    rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    if(ew_choice_live(s,choice))cbp_wwsat_resolve(s,result.random.get(),field(s,choice,"wwsat_source"),field(s,choice,"card_id",false));
    else{ // R2-SL-00: a stale pick re-offers the CURRENT sources; none left -> cau_wwsat completes the debt as a no-op
        auto owed=obs_attr(s,s.root,"_wwsat_owed",false);if(ctx_none(s,owed)){owed=cp_container(s,WOTR_LIST,{});rr_set_attr(s,s.root,"_wwsat_owed",owed);}
        auto debts=child_edges(s,owed);debts.push_back(rr_scalar(s,WOTR_BOOL,1));set_edges(s,owed,debts);cau_wwsat(s,result.random.get());
    }
    ehd_finish(s,t,result.random.get(),events,rows);cau_gandalf(s);
    events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
