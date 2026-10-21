// Public recruitment alternatives; AI projection independently prefers Elites.
int ert_kind(const Snapshot& t,uint64_t card){
    if(card==missing)return 0;const auto name=obs_attr(t,card,"card_name");
    if(text_is(t,name,"Cirdan's Ships"))return 1;
    if(text_is(t,name,"Riders of Th\xC3\xA9oden"))return 2;
    if(text_is(t,name,"\xC3\x89omer son of \xC3\x89omund"))return 3;return 0;
}
std::vector<std::string> ert_regions(const Snapshot& s,const Snapshot& t,int kind){
    require(kind>=1 && kind<=3,"Unknown targeted recruitment");std::vector<std::string> result;
    const auto metadata=field(t,t.root,"event_region_metadata");
    if(kind==1){
        for(const char* key:{"pelargir","dol amroth","west harondor","umbar","lamedon","anfalas","andrast","druwaith iaur","minhiriath","enedwaith","south ered luin","grey havens","forlindon","harlindon","north ered luin","evendim","arnor"}){
            if(ctx_get(t,metadata,key)==missing)continue;const auto army=mu_army(s,key);bool own=false;
            if(army!=missing)for(auto unit:child_edges(s,obs_attr(s,army,"units")))own|=ctx_unit_side(s,unit)==0;
            if(own)result.emplace_back(key);
        }
    }else if(kind==3){
        const auto regions=child_edges(t,metadata);const auto siege=obs_attr(s,s.root,"besieged_strongholds");
        for(size_t i=0;i<regions.size();i+=2){const std::string key(text(t,regions[i]));const auto meta=regions[i+1];
            const auto nation=ctx_get(t,field(t,t.root,"region_nations"),key);
            if(enum_is(t,nation,"game_state.Nation","ROHAN") && truth(t,ctx_get(t,meta,"is_settlement")) && !contains_string(s,siege,key) && er_region_open(s,key,0,true))result.push_back(key);
        }
        std::sort(result.begin(),result.end());
    }else{
        if(ctx_get(t,metadata,"edoras")!=missing && er_region_open(s,"edoras",0,true))result.emplace_back("edoras");
        const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));
        for(size_t i=0;i<armies.size();i+=2){
            bool companion=false;for(auto figure:child_edges(s,obs_attr(s,armies[i+1],"leaders")))companion|=object_type(s,figure)=="game_state.Companion";
            if(!companion)continue;const auto key=sanitize_name(text(s,armies[i]));
            const auto nation=ctx_get(t,field(t,t.root,"region_nations"),key);
            if(!enum_is(t,nation,"game_state.Nation","ROHAN") || (key!="helm s deep" && !er_region_open(s,key,0,true)))continue;
            if(std::find(result.begin(),result.end(),key)==result.end())result.push_back(key);
        }
    }
    return result;
}
std::vector<std::pair<std::string,std::string>> ert_choices(const Snapshot& s,const Snapshot& t,int kind){
    std::vector<std::string> combos;
    if(kind==1){
        const auto regular=mu_available(s,t,"ELVES","regular"),elite=mu_available(s,t,"ELVES","elite");
        const auto total=std::min<int64_t>(2,regular+elite);
        if(total==1)combos.emplace_back(elite>0?"2elite":"2regular");
        else if(total==2){
            if(regular>=2)combos.emplace_back("2regular");
            if(regular>=1&&elite>=1)combos.emplace_back("mixed");
            if(elite>=2)combos.emplace_back("2elite");
        }
    }
    std::vector<std::pair<std::string,std::string>> result;
    for(const auto& region:ert_regions(s,t,kind)){
        if(combos.empty())result.emplace_back(region,"");
        else for(const auto& combo:combos)result.emplace_back(region,combo);
    }
    return result;
}
uint64_t ert_option(Snapshot& s,const std::pair<std::string,std::string>& choice){
    auto result=ev_dict(s,{{"region",rr_string(s,choice.first)}});
    if(!choice.second.empty())cp_put(s,result,"unit_combo",rr_string(s,choice.second));
    return result;
}
void ert_effect(Snapshot& s,const Snapshot& t,int kind,std::string_view region,std::string_view selected={}){
    if(region.empty())return;
    if(kind==1){
        const auto available=mu_available(s,t,"ELVES","elite");
        const auto combo=selected.empty()?(available>=2?"2elite":available==1?"mixed":"2regular"):selected;
        require(combo=="2regular"||combo=="mixed"||combo=="2elite","Cirdan recruitment combination invalid");
        if(combo=="2elite"){er_place(s,t,"ELVES",region,"elite");er_place(s,t,"ELVES",region,"elite");}
        else if(combo=="mixed"){er_place(s,t,"ELVES",region,"regular");er_place(s,t,"ELVES",region,"elite");}
        else{er_place(s,t,"ELVES",region,"regular");er_place(s,t,"ELVES",region,"regular");}
    }else{
        er_place(s,t,"ROHAN",region,er_default_kind(s,t,"ROHAN"));er_place(s,t,"ROHAN",region,"leader");
    }
}
std::unique_ptr<Snapshot> apply_target_recruit_event(const Snapshot& source,const Snapshot& t,int side,int64_t index,std::string_view id){
    require(side==0,"Targeted recruitment belongs to FP");if(ert_kind(t,ev_card(t,id))==1||ert_kind(t,ev_card(t,id))==2||ert_kind(t,ev_card(t,id))==3)quiet_stable_action_gate(source,t);else quiet_action_gate(source,t);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==side,"Targeted event belongs to retained Ring actor");
    const auto card=ev_card(t,id);const int kind=ert_kind(t,card);require(kind && ctx_side(t,obs_attr(t,card,"side"))==0,"Unsupported targeted recruitment event");
    require(contains_string(source,tc_hand(source,side),id),"Targeted recruitment absent from hand");require(ev_requirement(source,t,card),"Targeted recruitment requirement false");
    const auto die=ev_payment(source,t,side,index,card);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace"));
    const std::string name(text(t,obs_attr(t,card,"card_name"))),deck=ev_lower(text(t,obs_attr(t,card,"deck")));
    auto child=action_clone(source);auto& s=*child;rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));rr_set_attr(s,ev_payment(s,t,side,index,card),"used",rr_scalar(s,WOTR_BOOL,1));
    const auto choices=ert_choices(s,t,kind);
    if(choices.size()>1){
        std::vector<uint64_t> options;for(const auto& choice:choices)options.push_back(ert_option(s,choice));
        rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,name)},{"side",rr_string(s,"fp")},{"decider",rr_string(s,"fp")},{"options",cp_container(s,WOTR_LIST,options)},{"die_face",rr_string(s,face)},{"deck",rr_string(s,deck)}}));
        tc_open_hand_limit_discard(s);rr_set_attr(s,s.root,"_battle_resolved_attacker",rr_scalar(s,WOTR_NONE));return child;
    }
    ert_effect(s,t,kind,choices.empty()?std::string_view{}:std::string_view(choices.front().first),choices.empty()?std::string_view{}:std::string_view(choices.front().second));er_commit(s,t,id,deck,face);return child;
}
uint64_t ert_pending(const Snapshot& s,const Snapshot& t){
    const auto p=obs_attr(s,s.root,"pending_card_option",false);require(!ctx_none(s,p) && text_is(s,field(s,p,"side",false),"fp") && text_is(s,field(s,p,"decider",false),"fp"),"No FP targeted recruitment pending");
    const auto card=ev_card(t,text(s,field(s,p,"card_id")));const int kind=ert_kind(t,card);require(kind && text_is(s,field(s,p,"card_name"),text(t,obs_attr(t,card,"card_name"))),"Targeted recruitment card mismatch");
    const auto choices=ert_choices(s,t,kind);const auto options=child_edges(s,field(s,p,"options"));require(choices.size()>1 && options.size()==choices.size(),"Targeted recruitment choices stale");
    for(size_t i=0;i<choices.size();++i){
        const auto& choice=choices[i];
        require(typed(s,options[i],WOTR_DICT,"Recruitment option must be dict").count==(choice.second.empty()?2:4) && text_is(s,field(s,options[i],"region"),choice.first),"Targeted recruitment order mismatch");
        if(!choice.second.empty())require(text_is(s,field(s,options[i],"unit_combo"),choice.second),"Cirdan recruitment supply or composition changed");
    }
    return p;
}
std::unique_ptr<Snapshot> native_target_recruit_choice_actions(const Snapshot& source,const Snapshot& t,int side){
    require(side==0,"Targeted recruitment menu belongs to FP");ert_pending(source,t);auto out=std::make_unique<Snapshot>(source);auto& s=*out;const auto p=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t index=0;
    for(auto option:child_edges(s,field(s,p,"options"))){
        const auto params=ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",option}});
        rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),params}));
    }
    s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
std::unique_ptr<Snapshot> apply_target_recruit_choice(const Snapshot& source,const Snapshot& t,int side,int64_t index){
    require(side==0,"Targeted recruitment choice belongs to FP");const auto p=ert_pending(source,t);const auto options=child_edges(source,field(source,p,"options"));require(index>=0 && static_cast<uint64_t>(index)<options.size(),"Targeted recruitment choice index invalid");
    const std::string id(text(source,field(source,p,"card_id"))),deck(text(source,field(source,p,"deck"))),face(text(source,field(source,p,"die_face"))),region(text(source,field(source,options[index],"region")));
    require(contains_string(source,tc_hand(source,0),id),"Pending targeted card absent from hand");require(deck=="strategy" && (face=="EVENT" || face=="WILL_OF_WEST" || face=="MUSTER" || face=="MUSTER_ARMY"),"Targeted recruitment original funding invalid");
    auto child=action_clone(source);auto& s=*child;rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));if(ert_kind(t,ev_card(t,id))==1||ert_kind(t,ev_card(t,id))==2||ert_kind(t,ev_card(t,id))==3)quiet_stable_action_gate(s,t);else quiet_action_gate(s,t);ring_waiver(s,0);ring_waiver(s,1);
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    const auto selected=field(source,options[index],"unit_combo",false);
    ert_effect(s,t,ert_kind(t,ev_card(t,id)),region,ctx_none(source,selected)?std::string_view{}:text(source,selected));er_commit(s,t,id,deck,face);return child;
}
