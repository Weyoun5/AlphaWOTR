// Pits of Mordor: lexicographic combinations, sequential live supplies,
// ordinary learned overflow, and one funded action across its target choice.
bool epr_supported(const Snapshot& t,uint64_t card){return card!=missing && text_is(t,obs_attr(t,card,"card_name"),"Pits of Mordor");}
std::vector<std::string> epr_regions(const Snapshot& s,const Snapshot& t){
    std::set<std::string> candidates;const auto meta=child_edges(t,field(t,t.root,"board_regions"));const auto parents=field(t,t.root,"board_region_parent_names");
    for(size_t i=0;i<meta.size();i+=2){const std::string key(text(t,meta[i]));if(truth(t,field(t,meta[i+1],"is_stronghold",false)) && text_is(t,field(t,parents,key.c_str(),false),"sauron"))candidates.insert(key);}
    std::vector<std::string> result;const auto sieges=obs_attr(s,s.root,"besieged_strongholds"),captured=ctx_side_value(s,obs_attr(s,s.root,"enemy_controlled"),0);
    for(const auto& key:candidates){const auto army=mu_army(s,key);bool own=false,enemy=false;if(army!=missing)for(auto u:child_edges(s,obs_attr(s,army,"units")))(ctx_unit_side(s,u)==1?own:enemy)=true;
        if(contains_string(s,sieges,key)){if(c9_garrison(s,t,key)==1 || own)result.push_back(key);}
        else if(!contains_string(s,captured,key) && !enemy)result.push_back(key);
    }
    return result;
}
std::vector<std::vector<std::string>> epr_combinations(const Snapshot& s,const Snapshot& t){
    const auto regions=epr_regions(s,t);std::vector<std::vector<std::string>> result;const size_t count=std::min<size_t>(3,regions.size());if(!count)return result;
    std::vector<std::string> selected;
    auto visit=[&](auto&& self,size_t start)->void{if(selected.size()==count){result.push_back(selected);return;}for(size_t i=start;i<regions.size();++i){selected.push_back(regions[i]);self(self,i+1);selected.pop_back();}};
    visit(visit,0);return result;
}
uint64_t epr_option(Snapshot& s,const std::vector<std::string>& regions){std::vector<uint64_t> values;for(const auto& key:regions)values.push_back(rr_string(s,key));return ev_dict(s,{{"strongholds",cp_container(s,WOTR_LIST,values)}});}
void epr_effect(Snapshot& s,const Snapshot& t,const std::vector<std::string>& selected){for(const auto& key:selected)for(int i=0;i<2;++i)er_place(s,t,"SAURON",key,"regular");}
std::unique_ptr<Snapshot> apply_pits_event(const Snapshot& source,const Snapshot& t,int side,int64_t index,std::string_view id){
    if(esra_has_choices(source,t,id))return esra_begin(source,t,side,index,id);
    require(side==1,"Pits belongs to Shadow");quiet_action_gate(source,t,false,false,false,false,false,false,false,false,false,false,false,true);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==1,"Pits violates retained Ring actor");
    const auto card=ev_card(t,id);require(epr_supported(t,card)&&ctx_side(t,obs_attr(t,card,"side"))==1&&contains_string(source,tc_hand(source,1),id),"Pits absent or invalid");require(ev_requirement(source,t,card),"Pits requires Sauron at war");
    const auto die=ev_payment(source,t,1,index,card);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace")),deck(ev_lower(text(t,obs_attr(t,card,"deck"))));const auto choices=epr_combinations(source,t);
    auto child=action_clone(source);auto& s=*child;rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));rr_set_attr(s,ev_payment(s,t,1,index,card),"used",rr_scalar(s,WOTR_BOOL,1));
    if(choices.size()>1){std::vector<uint64_t> options;for(const auto& choice:choices)options.push_back(epr_option(s,choice));rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,"Pits of Mordor")},{"side",rr_string(s,"sh")},{"decider",rr_string(s,"sh")},{"options",cp_container(s,WOTR_LIST,options)},{"die_face",rr_string(s,face)},{"deck",rr_string(s,deck)}}));tc_open_hand_limit_discard(s);su_cleanup(s,t);cau_gandalf(s);rr_set_attr(s,s.root,"_battle_resolved_attacker",rr_scalar(s,WOTR_NONE));}
    else{if(!choices.empty())epr_effect(s,t,choices.front());esr_commit(s,t,id,deck,face);}return child;
}
uint64_t epr_pending(const Snapshot& s,const Snapshot& t){
    const auto p=obs_attr(s,s.root,"pending_card_option",false);require(!ctx_none(s,p)&&text_is(s,field(s,p,"card_name",false),"Pits of Mordor")&&text_is(s,field(s,p,"side",false),"sh")&&text_is(s,field(s,p,"decider",false),"sh")&&!truth(s,field(s,p,"direct",false)),"No Pits target choice pending");require(epr_supported(t,ev_card(t,text(s,field(s,p,"card_id")))),"Pits pending card ID invalid");return p;
}
std::unique_ptr<Snapshot> native_pits_choice_actions(const Snapshot& source,const Snapshot& t,int side){
    if(esra_pending(source))return native_shadow_allocation_choices(source,t,side);
    require(side==1,"Pits choices belong to Shadow");epr_pending(source,t);auto out=std::make_unique<Snapshot>(source);auto& s=*out;const auto p=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t index=0;
    for(auto choice:child_edges(s,field(s,p,"options")))rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",choice}})}));
    s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
std::unique_ptr<Snapshot> apply_pits_choice(const Snapshot& source,const Snapshot& t,int side,int64_t index){
    if(esra_pending(source))return apply_shadow_allocation_choice(source,t,side,index);
    require(side==1,"Pits choice belongs to Shadow");const auto p=epr_pending(source,t);const auto options=child_edges(source,field(source,p,"options"));require(index>=0&&static_cast<uint64_t>(index)<options.size(),"Pits choice index invalid");
    const std::string id(text(source,field(source,p,"card_id"))),deck(text(source,field(source,p,"deck"))),face(text(source,field(source,p,"die_face")));require(contains_string(source,tc_hand(source,1),id),"Pits pending card absent from hand");
    std::vector<std::string> selected;for(auto key:child_edges(source,field(source,options[index],"strongholds")))selected.emplace_back(text(source,key));const auto legal=epr_combinations(source,t);require(std::find(legal.begin(),legal.end(),selected)!=legal.end(),"Pits selected strongholds no longer legal");
    auto child=action_clone(source);auto& s=*child;const auto marker=obs_attr(s,s.root,"_battle_resolved_attacker",false);rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));quiet_action_gate(s,t,false,false,false,false,false,false,false,false,false,false,false,true);ring_waiver(s,0);ring_waiver(s,1);
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));epr_effect(s,t,selected);esr_commit(s,t,id,deck,face);if(marker!=missing)rr_set_attr(s,s.root,"_battle_resolved_attacker",marker);return child;
}
