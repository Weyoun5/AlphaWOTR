// Printed recruitment bypasses ordinary muster politics and can overstack.
// Shared supply/figure creation and learned overflow resolution stay native.
bool er_supported(const Snapshot& t,uint64_t card){
    if(card==missing)return false;const auto name=obs_attr(t,card,"card_name");
    if(text_is(t,name,"Kindred of Glorfindel"))return true;
    for(const char* candidate:{"The Red Arrow","Guards of the Citadel","Dain Ironfoot's Guard","Thranduil's Archers","Celeborn's Galadhrim","Grimbeorn the Old Son of Beorn","Imrahil of Dol Amroth","King Brand's Men","Swords in Eriador"})if(text_is(t,name,candidate))return true;
    return false;
}
bool er_region_open(const Snapshot& s,std::string_view region,int side,bool option_predicate=false,uint64_t target_override=missing,bool ignore_captured=false){
    const auto army=target_override==missing?mu_army(s,region):target_override;bool own=false,enemy=false;
    if(army!=missing)for(auto unit:child_edges(s,obs_attr(s,army,"units")))(ctx_unit_side(s,unit)==side?own:enemy)=true;
    const bool siege=contains_string(s,obs_attr(s,s.root,"besieged_strongholds"),region);
    if(enemy && (!siege || (option_predicate && !own)))return false;
    int holder=-1;const auto control=child_edges(s,obs_attr(s,s.root,"enemy_controlled"));
    for(size_t i=0;i<control.size();i+=2)if(contains_string(s,control[i+1],region)){holder=ctx_side(s,control[i]);if(!option_predicate)break;}
    return ignore_captured || holder<0 || holder==side || own;
}
bool er_place(Snapshot& s,const Snapshot& t,std::string_view nation,std::string_view region,std::string_view kind,uint64_t target_override=missing,bool ignore_captured=false){
    const int side=fp_nation(nation)?0:1;if(mu_available(s,t,nation,kind)<=0 || !er_region_open(s,region,side,false,target_override,ignore_captured))return false;
    const auto army=target_override==missing?mu_army(s,region):target_override;int64_t count=0;
    if(army!=missing)for(auto unit:child_edges(s,obs_attr(s,army,"units")))count+=ctx_unit_side(s,unit)==side;
    if(kind=="leader" && count==0)return false;
    mu_place(s,nation,region,kind,target_override);
    if(kind!="leader" && count>=su_cap(s,t,region,side)){
        auto pending=obs_attr(s,s.root,"_overstack_regions",false);
        if(ctx_none(s,pending)){pending=cp_container(s,WOTR_SET,{});rr_set_attr(s,s.root,"_overstack_regions",pending);}
        // Source uses the existing raw dictionary key for overflow bookkeeping.
        const auto entries=child_edges(s,obs_attr(s,s.root,"armies"));
        for(size_t i=0;i<entries.size();i+=2)if(target_override!=missing?entries[i+1]==target_override:sanitize_name(text(s,entries[i]))==region){me_set_member(s,pending,text(s,entries[i]),true);break;}
    }
    return true;
}
std::string er_default_kind(const Snapshot& s,const Snapshot& t,std::string_view nation){return mu_available(s,t,nation,"elite")>0?"elite":"regular";}
std::vector<std::string> er_guards_options(const Snapshot& s,const Snapshot& t){
    std::vector<std::string> kinds;if(!er_region_open(s,"minas tirith",0,true))return kinds;
    for(const char* kind:{"regular","elite"})if(mu_available(s,t,"GONDOR",kind)>0)kinds.emplace_back(kind);return kinds;
}
void er_effect(Snapshot& s,const Snapshot& t,std::string_view name,std::string_view selected={},am_Builder* events=nullptr,std::vector<uint64_t>* rows=nullptr){
    auto one=[&](std::string_view nation,std::string_view region,bool leader){
        const auto kind=selected.empty()?er_default_kind(s,t,nation):std::string(selected);er_place(s,t,nation,region,kind);
        if(leader)er_place(s,t,nation,region,"leader");
    };
    bool draw=false;
    if(name=="The Red Arrow"){
        const auto boxes=obs_attr(s,obs_attr(s,s.root,"political"),"boxes"),current=ctx_nation_value(s,boxes,"ROHAN");
        const auto value=typed(s,current,WOTR_INT,"Red Arrow political position must be integer").integer;
        const int64_t cap=active_nation(s,political_view(s,"ROHAN",true),"ROHAN",false)?3:2;
        if(value<cap)me_nation_assign(s,boxes,"ROHAN",rr_scalar(s,WOTR_INT,std::min<int64_t>(cap,value+1)));
        one("ROHAN","edoras",true);
    }else if(name=="Guards of the Citadel")one("GONDOR","minas tirith",true);
    else if(name=="Kindred of Glorfindel"){one("ELVES","rivendell",false);draw=true;}
    else if(name=="Dain Ironfoot's Guard")one("DWARVES","erebor",true);
    else if(name=="Thranduil's Archers"){one("ELVES","woodland realm",false);draw=true;}
    else if(name=="Celeborn's Galadhrim"){one("ELVES","lorien",false);draw=true;}
    else if(name=="Grimbeorn the Old Son of Beorn")one("THE_NORTH","carrock",true);
    else if(name=="Imrahil of Dol Amroth")one("GONDOR","dol amroth",true);
    else if(name=="King Brand's Men"){er_place(s,t,"THE_NORTH","dale","regular");er_place(s,t,"THE_NORTH","dale","regular");draw=true;}
    else if(name=="Swords in Eriador"){one("THE_NORTH","the shire",false);one("DWARVES","ered luin",false);draw=true;}
    else throw std::invalid_argument("Unsupported printed recruitment effect");
    if(draw && truth(s,obs_attr(s,s.root,"fp_strat_deck",false))){
        const auto card=tc_draw_event_card(s,0,"strategy");bc_settle_draw(s,0,"strategy"); // Package W (P9b): event_effects.draw_cards -> grant_draw settles the owed-draw ledger
        if(events && rows)rows->push_back(events->row(-1,"card_drawn",{events->str("deck"),events->str("strategy"),events->str("card_id"),events->str(text(s,card)),events->str("side"),events->str("FP")}));
    }
}
void er_commit(Snapshot& s,const Snapshot& t,std::string_view id,std::string_view deck,std::string_view face,bool allow_consumed=false){
    const auto hand=tc_hand(s,0);auto cards=child_edges(s,hand);const auto it=std::find_if(cards.begin(),cards.end(),[&](uint64_t c){return text_is(s,c,id);});
    require(it!=cards.end()||allow_consumed,"Recruitment event card absent at commit");
    if(it!=cards.end()){
        const auto played=*it;cards.erase(it);set_edges(s,hand,cards);
        const std::string key="fp_"+std::string(deck);const auto pile=field(s,obs_attr(s,s.root,"discard_face_up"),key.c_str());auto discarded=child_edges(s,pile);discarded.push_back(played);set_edges(s,pile,discarded);
    }
    su_resolve_overstacks(s,t);
    if((face=="EVENT" || face=="WILL_OF_WEST") && !contains_string(s,hand,id))
        rr_set_attr(s,s.root,"_gandalf_draw_pending",ev_dict(s,{{"source_card_id",rr_string(s,id)},{"deck",rr_string(s,deck)},{"die_face",rr_string(s,face)}}));
    tc_open_hand_limit_discard(s);su_cleanup(s,t);cau_gandalf(s);
    rr_set_attr(s,s.root,"_battle_resolved_attacker",rr_scalar(s,WOTR_NONE));
}
std::unique_ptr<Snapshot> er_apply(const Snapshot& source,const Snapshot& t,int side,int64_t index,std::string_view id,am_Builder* events,std::vector<uint64_t>* rows){
    require(side==0,"Printed recruitment event belongs to FP");const auto siege_card=ev_card(t,id);if(siege_card!=missing && (text_is(t,obs_attr(t,siege_card,"card_name"),"Kindred of Glorfindel") || text_is(t,obs_attr(t,siege_card,"card_name"),"Guards of the Citadel") || ((text_is(t,obs_attr(t,siege_card,"card_name"),"Dain Ironfoot's Guard") || (text_is(t,obs_attr(t,siege_card,"card_name"),"Imrahil of Dol Amroth") || text_is(t,obs_attr(t,siege_card,"card_name"),"Celeborn's Galadhrim") || text_is(t,obs_attr(t,siege_card,"card_name"),"Grimbeorn the Old Son of Beorn") || text_is(t,obs_attr(t,siege_card,"card_name"),"King Brand's Men") || text_is(t,obs_attr(t,siege_card,"card_name"),"Swords in Eriador"))) || text_is(t,obs_attr(t,siege_card,"card_name"),"The Red Arrow") || text_is(t,obs_attr(t,siege_card,"card_name"),"Thranduil's Archers"))))quiet_stable_action_gate(source,t);else quiet_action_gate(source,t);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==side,"Recruitment event belongs to retained Ring actor");
    const auto card=ev_card(t,id);require(card!=missing && er_supported(t,card) && ctx_side(t,obs_attr(t,card,"side"))==0,"Unsupported recruitment event");
    require(contains_string(source,tc_hand(source,side),id),"Recruitment event is absent from hand");require(ev_requirement(source,t,card),"Recruitment event requirement false");
    const auto die=ev_payment(source,t,side,index,card);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace"));
    const std::string name(text(t,obs_attr(t,card,"card_name"))),deck=ev_lower(text(t,obs_attr(t,card,"deck")));
    auto child=action_clone(source);auto& s=*child;
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));rr_set_attr(s,ev_payment(s,t,side,index,card),"used",rr_scalar(s,WOTR_BOOL,1));
    std::string selected;
    if(name=="Guards of the Citadel"){
        const auto kinds=er_guards_options(s,t);
        if(kinds.size()>1){
            std::vector<uint64_t> options;for(const auto& kind:kinds)options.push_back(ev_dict(s,{{"region",rr_string(s,"minas tirith")},{"unit_type",rr_string(s,kind)}}));
            rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,name)},{"side",rr_string(s,"fp")},{"decider",rr_string(s,"fp")},
                {"options",cp_container(s,WOTR_LIST,options)},{"die_face",rr_string(s,face)},{"deck",rr_string(s,deck)}}));
            tc_open_hand_limit_discard(s);rr_set_attr(s,s.root,"_battle_resolved_attacker",rr_scalar(s,WOTR_NONE));return child;
        }
        if(!kinds.empty())selected=kinds.front();
    }
    er_effect(s,t,name,selected,events,rows);er_commit(s,t,id,deck,face);return child;
}
std::unique_ptr<Snapshot> apply_recruit_event_action(const Snapshot& source,const Snapshot& t,int side,int64_t index,std::string_view id){
    return er_apply(source,t,side,index,id,nullptr,nullptr);
}
fm_Result apply_recruit_event_traced(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index,std::string_view id){
    fm_Result result;result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());
    am_Builder events;std::vector<uint64_t> rows;result.state=er_apply(source,t,side,index,id,&events,&rows);
    events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
uint64_t er_pending(const Snapshot& s,const Snapshot& t){
    const auto p=obs_attr(s,s.root,"pending_card_option",false);
    require(!ctx_none(s,p) && text_is(s,field(s,p,"card_name",false),"Guards of the Citadel") && text_is(s,field(s,p,"side",false),"fp") && text_is(s,field(s,p,"decider",false),"fp"),"No FP Guards recruitment is pending");
    const auto options=child_edges(s,field(s,p,"options"));const auto kinds=er_guards_options(s,t);require(options.size()==2 && kinds.size()==2,"Guards recruitment options stale");
    for(size_t i=0;i<options.size();++i)require(typed(s,options[i],WOTR_DICT,"Guards option must be dictionary").count==4 && text_is(s,field(s,options[i],"region"),"minas tirith") && text_is(s,field(s,options[i],"unit_type"),kinds[i]),"Guards recruitment option mismatch");
    return p;
}
std::unique_ptr<Snapshot> native_recruit_event_choice_actions(const Snapshot& source,const Snapshot& t,int side){
    require(side==0,"Guards recruitment menu belongs to FP");er_pending(source,t);auto out=std::make_unique<Snapshot>(source);auto& s=*out;
    const auto p=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t index=0;
    for(auto option:child_edges(s,field(s,p,"options"))){
        const auto params=ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",option}});
        rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),params}));
    }
    s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
std::unique_ptr<Snapshot> apply_recruit_event_choice(const Snapshot& source,const Snapshot& t,int side,int64_t index){
    require(side==0,"Guards recruitment choice belongs to FP");const auto p=er_pending(source,t);require(index>=0 && index<2,"Guards choice index invalid");
    const auto chosen=source.edges[source.nodes[field(source,p,"options")].offset+index];const std::string kind(text(source,field(source,chosen,"unit_type")));
    const std::string id(text(source,field(source,p,"card_id"))),deck(text(source,field(source,p,"deck"))),face(text(source,field(source,p,"die_face")));
    const auto card=ev_card(t,id);require(card!=missing && text_is(t,obs_attr(t,card,"card_name"),"Guards of the Citadel") && contains_string(source,tc_hand(source,side),id),"Guards pending card identity invalid");
    require(deck=="strategy" && (face=="EVENT" || face=="WILL_OF_WEST" || face=="MUSTER" || face=="MUSTER_ARMY"),"Guards funding metadata invalid");
    auto child=action_clone(source);auto& s=*child;rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));quiet_stable_action_gate(s,t);ring_waiver(s,0);ring_waiver(s,1);
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));er_effect(s,t,"Guards of the Citadel",kind);er_commit(s,t,id,deck,face);return child;
}
