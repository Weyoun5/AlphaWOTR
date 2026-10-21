// Fellowship events that draw dice or a Companion instead of a Hunt tile.
bool el_supported(const Snapshot& t,uint64_t card){return card!=missing && (text_is(t,obs_attr(t,card,"card_name"),"Lure of the Ring") || text_is(t,obs_attr(t,card,"card_name"),"Candles of Corpses"));}
bool el_pending(const Snapshot& s){const auto p=obs_attr(s,s.root,"pending_card_option",false);return !ctx_none(s,p) && text_is(s,field(s,p,"card_name",false),"Lure of the Ring") && text_is(s,field(s,p,"side",false),"sh") && text_is(s,field(s,p,"decider",false),"fp") && truth(s,field(s,p,"direct",false));}
fm_Result apply_lure_event_action(const Snapshot& source,const Snapshot& t,const void* random_source,int32_t side,int64_t index,std::string_view id){
    require(side==1,"Lure/Candles belongs to Shadow");
    const auto card=ev_card(t,id);
    // Package J: Lure of the Ring and Candles of Corpses both preserve the validated siege armies (corruption,
    // card/payment bookkeeping, a direct FP choice); Lure's continuations need no stable-siege predicate (Package W: hc_lure_bof_siege removed).
    quiet_action_gate(source,t,false,false,false,false,false,false,false,false,true,false,false,true);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained) || ctx_side(source,retained)==side,"Lure/Candles violates retained Ring actor");
    require(el_supported(t,card) && ctx_side(t,obs_attr(t,card,"side"))==side,"Unsupported Lure/Candles card");require(contains_string(source,tc_hand(source,side),id),"Event card absent from hand");require(ev_requirement(source,t,card),"Event requirement false");
    const auto die=ev_payment(source,t,side,index,card);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace")),deck=ev_lower(text(t,obs_attr(t,card,"deck"))),name(text(t,obs_attr(t,card,"card_name")));
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;am_Builder events;std::vector<uint64_t> rows;
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));rr_set_attr(s,ev_payment(s,t,side,index,card),"used",rr_scalar(s,WOTR_BOOL,1));
    const auto fs=obs_attr(s,s.root,"fellowship"),guide=obs_attr(s,fs,"guide");const bool gollum=!ctx_none(s,guide) && text(s,obs_attr(s,guide,"name")).find("Gollum")!=std::string_view::npos;
    if(name=="Candles of Corpses"){
        uint32_t rolls[3];hf_roll(result.random.get(),rolls,3,"card effect");const int threshold=gollum?6:4;int damage=0;std::vector<uint64_t> faces;
        for(auto roll:rolls){damage+=roll>=static_cast<uint32_t>(threshold);faces.push_back(events.integer(roll));}
        rr_set_attr(s,fs,"corruption",rr_scalar(s,WOTR_INT,std::min<int64_t>(12,ht_integer(s,fs,"corruption")+damage)));
        rows.push_back(events.row(-1,"dice_roll",{events.str("faces"),events.container(WOTR_LIST,faces),events.str("ctx"),events.str("Candles of Corpses ("+std::to_string(threshold)+"+): +"+std::to_string(damage)+" corruption"),events.str("side"),events.str("SH")}));
    }else if(gollum)hd_corruption(s,1);
    else{
        const auto companions=child_edges(s,obs_attr(s,fs,"companions"));if(!companions.empty()){
            const auto pick=hf_choice(result.random.get(),companions.size(),true);const auto victim=companions[pick];std::vector<uint64_t> names;for(auto c:companions)names.push_back(events.str(text(s,obs_attr(s,c,"name"))));
            rows.push_back(events.row(-1,"random_draw",{events.str("ctx"),events.str("Lure of the Ring victim"),events.str("result"),events.str(text(s,obs_attr(s,victim,"name"))),events.str("pool"),events.container(WOTR_LIST,names)}));
            const auto options=rr_scalar(s,WOTR_LIST);std::vector<uint64_t> choices;
            for(const char* action:{"corruption","eliminate"})choices.push_back(ev_dict(s,{{"action",rr_string(s,action)},{"victim",obs_attr(s,victim,"name")},{"level",rr_scalar(s,WOTR_INT,ht_integer(s,victim,"level"))},{"_lure_drawn",rr_scalar(s,WOTR_BOOL,1)}}));set_edges(s,options,choices);
            rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,name)},{"side",rr_string(s,"sh")},{"decider",rr_string(s,"fp")},{"direct",rr_scalar(s,WOTR_BOOL,1)},{"options",options}}));
        }
    }
    ed_commit(s,side,id,deck,face);rr_set_attr(s,s.root,"_battle_resolved_attacker",rr_scalar(s,WOTR_NONE));ehd_finish(s,t,result.random.get(),events,rows);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
std::unique_ptr<Snapshot> native_lure_choice_actions(const Snapshot& source){
    require(el_pending(source),"No FP Lure choice pending");auto out=std::make_unique<Snapshot>(source);auto& s=*out;const auto p=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t index=0;
    for(auto choice:child_edges(s,field(s,p,"options"))){const auto row=rr_scalar(s,WOTR_TUPLE);set_edges(s,row,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",choice}})});rows.push_back(row);}
    s.root=rr_scalar(s,WOTR_LIST);set_edges(s,s.root,rows);return ac_compact(s);
}
fm_Result apply_lure_event_choice(const Snapshot& source,const Snapshot& t,const void* random_source,int64_t index){
    require(el_pending(source),"No FP Lure choice pending");hc_gate(source,false,true,false,true);const auto p=obs_attr(source,source.root,"pending_card_option");const auto options=child_edges(source,field(source,p,"options"));
    require(index>=0 && static_cast<uint64_t>(index)<options.size(),"Lure choice index out of range");const auto selected=options[index];require(truth(source,field(source,selected,"_lure_drawn",false)),"Lure choice must refer to completed draw");
    const std::string action(text(source,field(source,selected,"action"))),name(text(source,field(source,selected,"victim")));require(action=="corruption" || action=="eliminate","Unknown Lure choice");
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;am_Builder events;std::vector<uint64_t> rows;
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));
    const auto fs=obs_attr(s,s.root,"fellowship");uint64_t victim=missing;for(auto c:child_edges(s,obs_attr(s,fs,"companions")))if(ev_lower(text(s,obs_attr(s,c,"name")))==ev_lower(name)){victim=c;break;}require(victim!=missing,"Lure drawn companion no longer present");
    if(action=="corruption")rr_set_attr(s,fs,"corruption",rr_scalar(s,WOTR_INT,std::min<int64_t>(12,ht_integer(s,fs,"corruption")+ht_integer(s,victim,"level"))));
    else{
        hc_remove(s,victim);cbp_companion_death(s,result.random.get(),false);ft_guide(s);
        if((name=="Meriadoc" || name=="Peregrin") && !truth(s,obs_attr(s,fs,"mordor_entered")) && ctx_none(s,obs_attr(s,s.root,"pending_hunt"))){
            const auto level=ht_integer(s,victim,"level",1);rr_set_attr(s,s.root,"pending_hunt",ev_dict(s,{{"phase",rr_string(s,"hobbit_place")},{"companion",obs_attr(s,victim,"name")},{"level",rr_scalar(s,WOTR_INT,level?level:1)},{"companion_obj",victim},{"damage",rr_scalar(s,WOTR_INT)}}));
        }
    }
    ehd_finish(s,t,result.random.get(),events,rows);cau_gandalf(s);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
