// Printed Shadow recruitment at a learned army location; kind is fixed by supply.
bool eho_supported(const Snapshot& t,uint64_t card){return card!=missing && (text_is(t,obs_attr(t,card,"card_name"),"Half-orcs and Goblin-men") || text_is(t,obs_attr(t,card,"card_name"),"Olog-hai"));}
std::string eho_nation(const Snapshot& t,uint64_t card){require(eho_supported(t,card),"Unknown shared Shadow recruitment card");return text_is(t,obs_attr(t,card,"card_name"),"Olog-hai")?"SAURON":"ISENGARD";}
std::vector<std::string> eho_regions(const Snapshot& s,const Snapshot& t,std::string_view nation="ISENGARD"){
    if(mu_available(s,t,nation,"regular")<=0 && mu_available(s,t,nation,"elite")<=0)return {};
    std::set<std::string> regions;const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));
    for(size_t i=0;i<armies.size();i+=2){bool own=false;for(auto u:child_edges(s,obs_attr(s,armies[i+1],"units")))own|=ctx_unit_side(s,u)==1;
        const auto key=sanitize_name(text(s,armies[i]));if(own && er_region_open(s,key,1,true))regions.insert(key);}
    return {regions.begin(),regions.end()};
}
// Package W (P5/P9c): the frozen play_event_card turns a handler refund (recruit_unit placed nothing) into a COMPLETED
// zero-effect play unless legal options existed and the action carried a bad or absent choice (event_effects.py:1035-1069,
// ruling 1.230); a menu row never carries a bad choice, so every offered Half-orcs / Olog-hai play completes: card discarded,
// die spent, redraw grant earned.  The zero-option play runs the handler with the frozen fallback region first.
// the frozen handler's fallback with no option: the first army (armies order) holding a Shadow unit, else nothing
std::string eho_first_shadow_army(const Snapshot& s){
    const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));
    for(size_t i=0;i<armies.size();i+=2){for(auto u:child_edges(s,obs_attr(s,armies[i+1],"units")))if(ctx_unit_side(s,u)==1)return sanitize_name(text(s,armies[i]));}
    return std::string();
}
bool eho_effect(Snapshot& s,const Snapshot& t,std::string_view region,std::string_view nation="ISENGARD"){
    return er_place(s,t,nation,region,er_default_kind(s,t,nation)); // false = recruit_unit placed nothing (the play still completes, 1.230)
}
std::unique_ptr<Snapshot> apply_half_orcs_event(const Snapshot& source,const Snapshot& t,int side,int64_t index,std::string_view id,bool mouth_muster){
    require(side==1,"Half-orcs belongs to Shadow");quiet_action_gate(source,t,false,false,false,false,false,false,false,false,false,false,false,true);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==side,"Half-orcs belongs to retained Ring actor");
    const auto card=ev_card(t,id);require(eho_supported(t,card)&&ctx_side(t,obs_attr(t,card,"side"))==1&&contains_string(source,tc_hand(source,1),id),"Half-orcs absent or invalid");
    const auto nation=eho_nation(t,card),name=std::string(text(t,obs_attr(t,card,"card_name")));const auto regions=eho_regions(source,t,nation);require(ev_requirement(source,t,card),"Half-orcs requirement false");
    const auto die=ev_payment(source,t,1,index,card,mouth_muster);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace")),deck(ev_lower(text(t,obs_attr(t,card,"deck"))));
    auto child=action_clone(source);auto& s=*child;rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));ev_consume_payment(s,t,1,index,card,mouth_muster);
    if(regions.size()>1){
        std::vector<uint64_t> options;for(const auto& region:regions)options.push_back(ev_dict(s,{{"region",rr_string(s,region)}}));
        rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,name)},{"side",rr_string(s,"sh")},{"decider",rr_string(s,"sh")},{"options",cp_container(s,WOTR_LIST,options)},{"die_face",rr_string(s,face)},{"deck",rr_string(s,deck)}}));
        tc_open_hand_limit_discard(s);rr_set_attr(s,s.root,"_battle_resolved_attacker",rr_scalar(s,WOTR_NONE));return child;
    }
    const std::string region=regions.empty()?eho_first_shadow_army(source):regions.front(); // Package W (P5): the frozen handler's fallback region
    if(!region.empty())eho_effect(s,t,region,nation);                                                        // Package W (P9c): placed nothing = a completed zero-effect play (1.230)
    esr_commit(s,t,id,deck,face);return child;
}
uint64_t eho_pending(const Snapshot& s,const Snapshot& t){
    const auto p=obs_attr(s,s.root,"pending_card_option",false);require(!ctx_none(s,p)&&text_is(s,field(s,p,"side",false),"sh")&&text_is(s,field(s,p,"decider",false),"sh")&&!truth(s,field(s,p,"direct",false)),"No Half-orcs region choice pending");
    const auto card=ev_card(t,text(s,field(s,p,"card_id")));require(eho_supported(t,card)&&text_is(s,field(s,p,"card_name",false),text(t,obs_attr(t,card,"card_name"))),"Shared recruitment pending identity invalid");
    const auto regions=eho_regions(s,t,eho_nation(t,card));const auto options=child_edges(s,field(s,p,"options"));
    require(regions.size()>1&&regions.size()==options.size(),"Half-orcs choices stale");
    for(size_t i=0;i<regions.size();++i)require(typed(s,options[i],WOTR_DICT,"Half-orcs option must be dict").count==2&&text_is(s,field(s,options[i],"region"),regions[i]),"Half-orcs choice order changed");return p;
}
std::unique_ptr<Snapshot> native_half_orcs_choice_actions(const Snapshot& source,const Snapshot& t,int side){
    require(side==1,"Half-orcs choices belong to Shadow");eho_pending(source,t);auto out=std::make_unique<Snapshot>(source);auto& s=*out;const auto p=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t index=0;
    for(auto option:child_edges(s,field(s,p,"options")))rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",option}})}));
    s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
std::unique_ptr<Snapshot> apply_half_orcs_choice(const Snapshot& source,const Snapshot& t,int side,int64_t index){
    require(side==1,"Half-orcs choice belongs to Shadow");const auto p=eho_pending(source,t);const auto options=child_edges(source,field(source,p,"options"));require(index>=0&&static_cast<uint64_t>(index)<options.size(),"Half-orcs choice index invalid");
    const std::string id(text(source,field(source,p,"card_id"))),deck(text(source,field(source,p,"deck"))),face(text(source,field(source,p,"die_face"))),region(text(source,field(source,options[index],"region")));
    require(contains_string(source,tc_hand(source,1),id)&&deck=="strategy"&&(face=="ARMY"||face=="MUSTER"||face=="MUSTER_ARMY"||face=="EVENT"||face=="WILL_OF_WEST"),"Half-orcs original funding invalid");
    auto child=action_clone(source);auto& s=*child;rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));quiet_action_gate(s,t,false,false,false,false,false,false,false,false,false,false,false,true);ring_waiver(s,0);ring_waiver(s,1);
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    const auto marker=obs_attr(s,s.root,"_battle_resolved_attacker",false);
    eho_effect(s,t,region,eho_nation(t,ev_card(t,id))); // Package W (P9c): a valid option's refund is converted to a completed zero-effect play by the frozen play_event_card (event_effects.py:1035-1069)
    esr_commit(s,t,id,deck,face);if(marker!=missing)rr_set_attr(s,s.root,"_battle_resolved_attacker",marker);return child;
}
