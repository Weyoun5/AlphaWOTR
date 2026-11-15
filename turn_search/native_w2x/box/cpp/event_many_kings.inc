// Many Kings: max-extent settlement combinations and sequential live supply.
bool emk_supported(const Snapshot& t,uint64_t card){return card!=missing&&text_is(t,obs_attr(t,card,"card_name"),"Many Kings to the Service of Mordor");}
bool emk_pending(const Snapshot& s){const auto p=obs_attr(s,s.root,"pending_card_option",false);return !ctx_none(s,p)&&text_is(s,field(s,p,"card_name",false),"Many Kings to the Service of Mordor");}
std::vector<std::string> emk_regions(const Snapshot& s,const Snapshot& t){
    std::set<std::string> candidates;const auto metadata=child_edges(t,field(t,t.root,"board_regions"));const auto parents=field(t,t.root,"board_region_parent_names");
    for(size_t i=0;i<metadata.size();i+=2){
        const std::string key(text(t,metadata[i]));const auto parent=ctx_get(t,parents,key);
        if((text_is(t,parent,"southrons")||text_is(t,parent,"southron")||text_is(t,parent,"southrons and easterlings"))
           &&(truth(t,field(t,metadata[i+1],"is_town",false))||truth(t,field(t,metadata[i+1],"is_city",false))||truth(t,field(t,metadata[i+1],"is_stronghold",false))))candidates.insert(key);
    }
    std::vector<std::string> regions;const auto sieges=obs_attr(s,s.root,"besieged_strongholds"),captured=ctx_side_value(s,obs_attr(s,s.root,"enemy_controlled"),0);
    for(const auto& key:candidates){
        if(contains_string(s,sieges,key)){if(c9_garrison(s,t,key)==1||evb_units(s,hr_army(s,key),1))regions.push_back(key);continue;}
        if(contains_string(s,captured,key))continue;
        bool enemy=false;const auto entries=child_edges(s,obs_attr(s,s.root,"armies"));
        for(size_t i=0;i<entries.size();i+=2)if(sanitize_name(text(s,entries[i]))==key)enemy|=evb_units(s,entries[i+1],0);
        if(!enemy)regions.push_back(key);
    }return regions;
}
std::vector<std::vector<std::string>> emk_combinations(const Snapshot& s,const Snapshot& t){
    const auto regions=emk_regions(s,t);const size_t count=std::min<size_t>(3,regions.size());std::vector<std::vector<std::string>> result;if(!count)return result;
    std::vector<std::string> selected;
    auto visit=[&](auto&& self,size_t start)->void{if(selected.size()==count){result.push_back(selected);return;}for(size_t i=start;i<regions.size();++i){selected.push_back(regions[i]);self(self,i+1);selected.pop_back();}};
    visit(visit,0);return result;
}
uint64_t emk_options(Snapshot& s,const std::vector<std::vector<std::string>>& combinations){
    std::vector<uint64_t> rows;for(const auto& selected:combinations){std::vector<uint64_t> values;for(const auto& key:selected)values.push_back(rr_string(s,key));rows.push_back(ev_dict(s,{{"settlements",cp_container(s,WOTR_LIST,values)}}));}
    return cp_container(s,WOTR_LIST,rows);
}
bool emk_effect(Snapshot& s,const Snapshot& t,std::vector<std::string> selected){
    const auto current=emk_regions(s,t);
    if(selected.empty())selected.assign(current.begin(),current.begin()+std::min<size_t>(3,current.size()));
    else{
        selected.resize(std::min<size_t>(3,selected.size()));std::set<std::string> seen;bool valid=true;
        for(auto& key:selected){key=sanitize_name(key);valid&=seen.insert(key).second&&std::find(current.begin(),current.end(),key)!=current.end();}
        if(!valid)return current.empty(); // Source refund converts to zero-effect only when no legal options remain.
    }
    for(const auto& region:selected)for(int i=0;i<2;++i)er_place(s,t,"SOUTHRON_EASTERLINGS",region,"regular");return true;
}
fm_Result apply_many_kings_event(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index,std::string_view id){
    if(esra_has_choices(source,t,id)){
        fm_Result result;result.state=esra_begin(source,t,side,index,id);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());
        am_Builder events;std::vector<uint64_t> rows;cau_finish(*result.state,t,result.random.get(),-1,events,rows);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
    }
    require(side==1,"Many Kings belongs to Shadow");quiet_action_gate(source,t,false,false,false,false,false,false,false,false,false,false,false,true);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==1,"Many Kings violates retained Ring actor");
    const auto card=ev_card(t,id);require(emk_supported(t,card)&&contains_string(source,tc_hand(source,1),id),"Many Kings card absent");require(ev_requirement(source,t,card),"Many Kings requirement false");
    const auto die=ev_payment(source,t,1,index,card);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace"));
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());
    auto& s=*result.state;rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    rr_set_attr(s,ev_payment(s,t,1,index,card),"used",rr_scalar(s,WOTR_BOOL,1));const auto regions=emk_combinations(s,t);
    if(regions.size()>1){
        rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,"Many Kings to the Service of Mordor")},
            {"side",rr_string(s,"sh")},{"decider",rr_string(s,"sh")},{"options",emk_options(s,regions)},{"die_face",rr_string(s,face)},{"deck",rr_string(s,"strategy")}}));tc_open_hand_limit_discard(s);
    }else if(emk_effect(s,t,regions.empty()?std::vector<std::string>{}:regions.front()))esr_commit(s,t,id,"strategy",face);
    am_Builder events;std::vector<uint64_t> rows;cau_finish(s,t,result.random.get(),-1,events,rows);
    events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
uint64_t emk_validate(const Snapshot& s,const Snapshot& t){
    require(emk_pending(s),"No Many Kings choice pending");const auto p=obs_attr(s,s.root,"pending_card_option");
    require(text_is(s,field(s,p,"side"),"sh")&&text_is(s,field(s,p,"decider"),"sh")&&!truth(s,field(s,p,"direct",false)),"Many Kings pending owner or stage invalid");
    const auto id=text(s,field(s,p,"card_id"));require(emk_supported(t,ev_card(t,id)),"Many Kings pending card identity invalid");/* package R: hand membership is not a menu precondition (frozen card_option menu, ai.py 7764-7770) */
    require(text_is(s,field(s,p,"deck"),"strategy"),"Many Kings pending deck invalid");const auto face=text(s,field(s,p,"die_face"));
    require(face=="MUSTER"||face=="MUSTER_ARMY"||face=="EVENT","Many Kings pending payment invalid");
    for(auto option:child_edges(s,field(s,p,"options"))){require(typed(s,option,WOTR_DICT,"Many Kings option must be a dictionary").count==2,"Many Kings option shape invalid");for(auto value:child_edges(s,field(s,option,"settlements")))text(s,value);}
    return p;
}
std::unique_ptr<Snapshot> native_many_kings_choices(const Snapshot& source,const Snapshot& t,int side){
    if(esra_pending(source))return native_shadow_allocation_choices(source,t,side);
    require(side==1,"Many Kings choices belong to Shadow");emk_validate(source,t);auto result=std::make_unique<Snapshot>(source);auto& s=*result;
    const auto p=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t index=0;
    for(auto choice:child_edges(s,field(s,p,"options")))rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),
        ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",choice}})}));
    s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
fm_Result apply_many_kings_choice(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index){
    if(esra_pending(source)){
        const auto oldmarker=obs_attr(source,source.root,"_battle_resolved_attacker",false);const int marker=ctx_none(source,oldmarker)?-1:ctx_side(source,oldmarker);
        fm_Result result;result.state=apply_shadow_allocation_choice(source,t,side,index);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());
        am_Builder events;std::vector<uint64_t> rows;cau_finish(*result.state,t,result.random.get(),marker,events,rows);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
    }
    require(side==1,"Many Kings choice belongs to Shadow");const auto p=emk_validate(source,t);const auto options=child_edges(source,field(source,p,"options"));
    require(index>=0&&static_cast<uint64_t>(index)<options.size(),"Many Kings choice index exceeds menu");
    const std::string id(text(source,field(source,p,"card_id"))),face(text(source,field(source,p,"die_face")));std::vector<std::string> selected;for(auto key:child_edges(source,field(source,options[index],"settlements")))selected.emplace_back(text(source,key));
    const auto oldmarker=obs_attr(source,source.root,"_battle_resolved_attacker",false);const int marker=ctx_none(source,oldmarker)?-1:ctx_side(source,oldmarker);
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;
    rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));quiet_action_gate(s,t,false,false,false,false,false,false,false,false,false,false,false,true);ring_waiver(s,0);ring_waiver(s,1);
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    if(ev_requirement(s,t,ev_card(t,id))&&emk_effect(s,t,selected))esr_commit(s,t,id,"strategy",face);
    am_Builder events;std::vector<uint64_t> rows;cau_finish(s,t,result.random.get(),marker,events,rows);
    events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
