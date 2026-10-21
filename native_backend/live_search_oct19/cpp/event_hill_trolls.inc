// Complete maximum-feasible Hill-trolls upgrades, including same-army pairs.
bool ehi_supported(const Snapshot& t,uint64_t card){return card!=missing && text_is(t,obs_attr(t,card,"card_name"),"Hill-trolls");}
std::vector<std::vector<std::string>> ehi_options(const Snapshot& s,const Snapshot& t){
    const auto available=mu_available(s,t,"SAURON","elite");if(available<=0)return {};
    std::map<std::string,int64_t> counts;int64_t total=0;const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));
    for(size_t i=0;i<armies.size();i+=2){
        int64_t count=0;for(auto unit:child_edges(s,obs_attr(s,armies[i+1],"units")))if(enum_is(s,obs_attr(s,unit,"nation"),"game_state.Nation","SAURON") && enum_is(s,obs_attr(s,unit,"kind"),"game_state.UnitKind","REGULAR"))++count;
        if(count){counts[sanitize_name(text(s,armies[i]))]+=count;total+=count;}
    }
    const auto maximum=std::min<int64_t>({2,total,available});std::vector<std::vector<std::string>> result;
    if(maximum<=0)return result;
    for(auto it=counts.begin();it!=counts.end();++it){
        if(maximum==1){result.push_back({it->first});continue;}
        if(it->second>=2)result.push_back({it->first,it->first});
        for(auto other=std::next(it);other!=counts.end();++other)result.push_back({it->first,other->first});
    }
    return result;
}
void ehi_upgrade(Snapshot& s,const Snapshot& t,const std::vector<std::string>& regions){
    int upgraded=0;
    for(const auto& region:regions){
        if(upgraded>=2)break;const auto army=mu_army(s,sanitize_name(region));if(army==missing)continue;
        const auto units=obs_attr(s,army,"units");auto values=child_edges(s,units);
        for(auto& unit:values)if(enum_is(s,obs_attr(s,unit,"nation"),"game_state.Nation","SAURON") && enum_is(s,obs_attr(s,unit,"kind"),"game_state.UnitKind","REGULAR")){
            if(mu_available(s,t,"SAURON","elite")<=0)break;
            unit=mu_object(s,"game_state.Unit",{{"nation",mu_enum(s,"game_state.Nation","SAURON")},{"kind",mu_enum(s,"game_state.UnitKind","ELITE")}});
            set_edges(s,units,values);++upgraded;break;
        }
    }
}
std::unique_ptr<Snapshot> apply_hill_trolls_event(const Snapshot& source,const Snapshot& t,int side,int64_t index,std::string_view id,bool mouth_muster){
    require(side==1,"Hill-trolls belongs to Shadow");quiet_action_gate(source,t,false,false,false,false,false,false,false,false,false,false,false,true);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==side,"Hill-trolls belongs to retained Ring actor");
    const auto card=ev_card(t,id);require(ehi_supported(t,card) && ctx_side(t,obs_attr(t,card,"side"))==1,"Invalid Hill-trolls identity");
    require(contains_string(source,tc_hand(source,side),id),"Hill-trolls absent from hand");require(ev_requirement(source,t,card),"Hill-trolls requirement false");
    const auto die=ev_payment(source,t,side,index,card,mouth_muster);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace")),deck=ev_lower(text(t,obs_attr(t,card,"deck")));
    auto child=action_clone(source);auto& s=*child;rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));ev_consume_payment(s,t,side,index,card,mouth_muster);
    const auto choices=ehi_options(s,t);
    if(choices.size()>1){
        std::vector<uint64_t> options;for(const auto& choice:choices){std::vector<uint64_t> regions;for(const auto& region:choice)regions.push_back(rr_string(s,region));options.push_back(ev_dict(s,{{"regions",cp_container(s,WOTR_LIST,regions)}}));}
        rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,"Hill-trolls")},{"side",rr_string(s,"sh")},{"decider",rr_string(s,"sh")},{"options",cp_container(s,WOTR_LIST,options)},{"die_face",rr_string(s,face)},{"deck",rr_string(s,deck)}}));
        tc_open_hand_limit_discard(s);
    }else{
        if(!choices.empty())ehi_upgrade(s,t,choices.front());ed_commit(s,side,id,deck,face);
    }
    rr_set_attr(s,s.root,"_battle_resolved_attacker",rr_scalar(s,WOTR_NONE));return child;
}
uint64_t ehi_pending(const Snapshot& s,const Snapshot& t){
    const auto p=obs_attr(s,s.root,"pending_card_option",false);require(!ctx_none(s,p) && text_is(s,field(s,p,"card_name",false),"Hill-trolls") && text_is(s,field(s,p,"side",false),"sh") && text_is(s,field(s,p,"decider",false),"sh"),"No Shadow Hill-trolls choice pending");
    const auto options=child_edges(s,field(s,p,"options"));const auto expected=ehi_options(s,t);require(options.size()>1 && options.size()==expected.size(),"Hill-trolls options stale");
    for(size_t i=0;i<options.size();++i){
        require(typed(s,options[i],WOTR_DICT,"Hill-trolls option must be dictionary").count==2,"Hill-trolls option has extra fields");
        const auto regions=field(s,options[i],"regions");require(typed(s,regions,WOTR_LIST,"Hill-trolls regions must be list").count==expected[i].size(),"Hill-trolls upgrade extent mismatch");
        for(size_t j=0;j<expected[i].size();++j)require(text_is(s,s.edges[s.nodes[regions].offset+j],expected[i][j]),"Hill-trolls option order mismatch");
    }
    return p;
}
std::unique_ptr<Snapshot> native_hill_trolls_choice_actions(const Snapshot& source,const Snapshot& t,int side){
    require(side==1,"Hill-trolls menu belongs to Shadow");ehi_pending(source,t);auto out=std::make_unique<Snapshot>(source);auto& s=*out;const auto p=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t index=0;
    for(auto option:child_edges(s,field(s,p,"options"))){
        const auto params=ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",option}});
        rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),params}));
    }
    s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
std::unique_ptr<Snapshot> apply_hill_trolls_choice(const Snapshot& source,const Snapshot& t,int side,int64_t index){
    require(side==1,"Hill-trolls choice belongs to Shadow");const auto p=ehi_pending(source,t);const auto choices=ehi_options(source,t);require(index>=0 && static_cast<uint64_t>(index)<choices.size(),"Hill-trolls choice index invalid");
    const std::string id(text(source,field(source,p,"card_id"))),deck(text(source,field(source,p,"deck"))),face(text(source,field(source,p,"die_face")));
    require(ehi_supported(t,ev_card(t,id)) && contains_string(source,tc_hand(source,side),id),"Hill-trolls pending card identity invalid");
    require(deck=="strategy" && (face=="ARMY" || face=="MUSTER" || face=="MUSTER_ARMY" || face=="EVENT" || face=="WILL_OF_WEST"),"Hill-trolls original funding invalid");
    auto child=action_clone(source);auto& s=*child;rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));quiet_action_gate(s,t,false,false,false,false,false,false,false,false,false,false,false,true);ring_waiver(s,0);ring_waiver(s,1);
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));ehi_upgrade(s,t,choices[index]);ed_commit(s,side,id,deck,face);
    return child;
}
