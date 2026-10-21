// Card-granted three-region joins: source traversal, composition, and order.
bool esg_supported(const Snapshot& t,uint64_t card){return card!=missing && text_is(t,obs_attr(t,card,"card_name"),"Shadows Gather");}
uint64_t esg_army(const Snapshot& s,std::string_view key){
    const auto armies=obs_attr(s,s.root,"armies"),exact=ctx_get(s,armies,key);return exact==missing?mu_army(s,key):exact;
}
bool esg_own(const Snapshot& s,uint64_t army){
    if(army==missing)return false;for(auto unit:child_edges(s,obs_attr(s,army,"units")))if(ctx_unit_side(s,unit)==1)return true;return false;
}
bool esg_enemy(const Snapshot& s,uint64_t army){
    if(army==missing)return false;for(auto unit:child_edges(s,obs_attr(s,army,"units")))if(ctx_unit_side(s,unit)==0)return true;return false;
}
bool esg_garrison(const Snapshot& s,const Snapshot& t,std::string_view key){return contains_string(s,obs_attr(s,s.root,"besieged_strongholds",false),key) && c9_garrison(s,t,key)==1;}
std::vector<std::string> esg_neighbors(const Snapshot& s,std::string_view key){
    const auto adjacency=obs_attr(s,obs_attr(s,s.root,"board"),"_adj");auto neighbors=ctx_get(s,adjacency,key);
    if(neighbors==missing){const std::string alt=key.substr(0,4)=="the "?std::string(key.substr(4)):"the "+std::string(key);neighbors=ctx_get(s,adjacency,alt);}
    std::vector<std::string> result;if(neighbors!=missing)for(auto value:child_edges(s,neighbors))result.emplace_back(text(s,value));std::sort(result.begin(),result.end());return result;
}
std::set<std::string> esg_reachable(const Snapshot& s,const Snapshot& t,std::string_view start,int max_regions=3){
    const auto siege=obs_attr(s,s.root,"besieged_strongholds",false);std::set<std::string> seen{std::string(start)},result;
    std::vector<std::pair<std::string,int>> queue{{std::string(start),0}};
    for(size_t at=0;at<queue.size();++at){const auto [cur,depth]=queue[at];if(depth>=max_regions)continue;
        for(const auto& raw:esg_neighbors(s,cur)){
            const auto next=sanitize_name(raw);if(!seen.insert(next).second)continue;const auto army=esg_army(s,next);const bool besieged=contains_string(s,siege,next);
            const bool join=besieged && esg_own(s,army) && c9_garrison(s,t,next)!=1;
            if((esg_enemy(s,army)||besieged)&&!join)continue;result.insert(next);if(!besieged)queue.emplace_back(next,depth+1);
        }
    }
    return result;
}
struct esg_Choice{std::string from,to;int64_t regular=-1,elite=-1;uint64_t cargo=missing;};
std::vector<esg_Choice> esg_options(const Snapshot& s,const Snapshot& t,int max_regions=3,const std::set<std::string>& excluded={}){
    const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));std::set<std::string> destinations;
    for(size_t i=0;i<armies.size();i+=2){const auto key=sanitize_name(text(s,armies[i]));if(esg_own(s,armies[i+1])&&!esg_garrison(s,t,key))destinations.insert(key);}
    std::vector<esg_Choice> result;
    for(size_t i=0;i<armies.size();i+=2){
        const auto from=sanitize_name(text(s,armies[i]));const auto army=armies[i+1];if(excluded.count(from)||esg_garrison(s,t,from)||!esg_own(s,army))continue;
        int64_t regular=0,elite=0;for(auto unit:child_edges(s,obs_attr(s,army,"units")))if(ctx_unit_side(s,unit)==1)(enum_is(s,obs_attr(s,unit,"kind"),"game_state.UnitKind","ELITE")?elite:regular)++;
        for(const auto& to:esg_reachable(s,t,from,max_regions))if(to!=from && destinations.count(to)){
            std::vector<uint64_t> units;for(auto u:child_edges(s,obs_attr(s,army,"units")))if(ctx_unit_side(s,u)==1)units.push_back(u);
            if(!sj_select(s,t,from,to,max_regions,units).first.empty())result.push_back({from,to});
        }
    }
    return result;
}
uint64_t esg_option(Snapshot& s,const esg_Choice& choice){
    std::vector<std::pair<const char*,uint64_t>> values{{"from_region",rr_string(s,choice.from)},{"to_region",rr_string(s,choice.to)}};
    if(choice.regular>=0){values.emplace_back("n_regular",rr_scalar(s,WOTR_INT,choice.regular));values.emplace_back("n_elite",rr_scalar(s,WOTR_INT,choice.elite));}return ev_dict(s,values);
}
std::vector<std::string> esg_mids(const Snapshot& s,std::string_view from,std::string_view to,int max_regions=3){
    struct Step{std::string region;std::vector<std::string> mids;int depth;};std::vector<Step> queue{{std::string(from),{},0}};std::set<std::string> seen{std::string(from)};
    const auto siege=obs_attr(s,s.root,"besieged_strongholds",false);
    for(size_t at=0;at<queue.size();++at){const auto step=queue[at];if(step.depth>=max_regions)continue;
        for(const auto& raw:esg_neighbors(s,step.region)){
            const auto next=sanitize_name(raw);if(!seen.insert(next).second)continue;if(next==to)return step.mids;
            if(contains_string(s,siege,next)||esg_enemy(s,esg_army(s,next)))continue;
            auto mids=step.mids;mids.push_back(next);queue.push_back({next,std::move(mids),step.depth+1});
        }
    }
    return {};
}
void esg_move(Snapshot& s,const Snapshot& t,const esg_Choice& choice){
    require(!esg_garrison(s,t,choice.from) && !esg_garrison(s,t,choice.to),"Shadows Gather cannot move a besieged garrison");
    if(at_shadow_blocked(s,choice.to))return;
    auto mids=esg_mids(s,choice.from,choice.to);const auto src=esg_army(s,choice.from),dst=mu_army(s,choice.to);require(src!=missing&&dst!=missing,"Shadows Gather join army disappeared");
    auto source_units=child_edges(s,obs_attr(s,src,"units"));std::vector<uint64_t> moving,kept,eligible;
    for(auto unit:source_units)(ctx_unit_side(s,unit)==1?eligible:kept).push_back(unit);
    // Choose the common legal route before applying aggregate count limits,
    // matching shadow_join_rules.select and _sg_join_move.
    if(ctx_none(s,choice.cargo)){const auto route=sj_select(s,t,choice.from,choice.to,3,eligible);eligible=route.first;mids=route.second;}
    int64_t regular=choice.regular,elite=choice.elite;
    for(auto unit:eligible){const bool e=enum_is(s,obs_attr(s,unit,"kind"),"game_state.UnitKind","ELITE");auto& left=e?elite:regular;
        if(choice.regular<0||left>0){moving.push_back(unit);if(left>0)--left;}else kept.push_back(unit);}
    require(!moving.empty(),"Shadows Gather must move units");std::vector<uint64_t> leaders,staying_leaders;
    for(auto figure:child_edges(s,obs_attr(s,src,"leaders")))(am_saruman(s,figure)||am_leader_side(s,figure)!=1?staying_leaders:leaders).push_back(figure);
    if(!ctx_none(s,choice.cargo)){
        auto selected=cag_selected(s,choice.from,choice.cargo);moving=selected.first;leaders=selected.second;
        const auto route=sj_select(s,t,choice.from,choice.to,3,moving,true);
        require(route.first.size()==moving.size(),"Shadows Gather cargo cannot cross political borders");mids=route.second;
        const std::set<uint64_t> chosen(moving.begin(),moving.end()),chosen_leaders(leaders.begin(),leaders.end());kept.clear();staying_leaders.clear();
        for(auto unit:source_units)if(!chosen.count(unit))kept.push_back(unit);
        for(auto figure:child_edges(s,obs_attr(s,src,"leaders")))if(!chosen_leaders.count(figure))staying_leaders.push_back(figure);
    }
    kept.clear();const std::set<uint64_t> moved_units(moving.begin(),moving.end());for(auto u:source_units)if(!moved_units.count(u))kept.push_back(u);
    int64_t dest_count=0;for(auto unit:child_edges(s,obs_attr(s,dst,"units")))dest_count+=ctx_unit_side(s,unit)==1;
    if(dest_count+static_cast<int64_t>(moving.size())>su_cap(s,t,choice.to,1)){
        auto tracking=obs_attr(s,s.root,"_overstack_regions",false);if(ctx_none(s,tracking)){tracking=cp_container(s,WOTR_SET,{});rr_set_attr(s,s.root,"_overstack_regions",tracking);}me_set_member(s,tracking,choice.to,true);
    }
    // Direct card movement replaces source lists, while destination lists
    // retain identity; source unit insertion order is never sorted by kind.
    am_assign_list(s,src,"units",kept);am_assign_list(s,src,"leaders",staying_leaders);
    if(kept.empty()&&staying_leaders.empty()){
        const auto armies=obs_attr(s,s.root,"armies");const std::string key=ctx_get(s,armies,choice.from)==src?choice.from:am_key(s,choice.from);am_erase(s,key);
    }
    am_extend(s,dst,"units",moving);am_extend(s,dst,"leaders",leaders);movement_entry_effects(s,t,choice.to,1,leaders);am_orphans(s);
    if(!esg_enemy(s,dst))movement_capture_effects(s,t,choice.to,1);
    for(const auto& region:mids){
        movement_capture_effects(s,t,region,1);const auto nation=ctx_get(t,field(t,t.root,"movement_region_nations"),region);
        if(!ctx_none(t,nation)&&fp_nation(ctx_nation(t,nation)))wt_activate_if_inactive(s,ctx_nation(t,nation),"attack");
    }
}
std::unique_ptr<Snapshot> apply_shadows_gather_event(const Snapshot& source,const Snapshot& t,int side,int64_t index,std::string_view id,bool mouth_muster){
    require(side==1,"Shadows Gather belongs to Shadow");quiet_stable_action_gate(source,t);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==side,"Shadows Gather belongs to retained Ring actor");
    const auto card=ev_card(t,id);require(esg_supported(t,card)&&ctx_side(t,obs_attr(t,card,"side"))==1,"Unsupported Shadows Gather card");require(contains_string(source,tc_hand(source,1),id),"Shadows Gather absent from hand");require(ev_requirement(source,t,card),"Shadows Gather requirement false");
    const auto die=ev_payment(source,t,1,index,card,mouth_muster);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace")),deck=ev_lower(text(t,obs_attr(t,card,"deck")));
    auto child=action_clone(source);auto& s=*child;rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));ev_consume_payment(s,t,1,index,card,mouth_muster);
    const auto choices=esg_options(s,t);
    if(choices.size()>1){
        std::vector<uint64_t> options;for(const auto& choice:choices)options.push_back(esg_option(s,choice));
        rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,"Shadows Gather")},{"side",rr_string(s,"sh")},{"decider",rr_string(s,"sh")},{"options",cp_container(s,WOTR_LIST,options)},{"die_face",rr_string(s,face)},{"deck",rr_string(s,deck)}}));tc_open_hand_limit_discard(s);
        su_cleanup(s,t); // Retain learned-route funding upkeep.
    }else{if(!choices.empty()){const auto choice=esg_option(s,choices.front());const auto completed=cag_stage(s,t,id,choice,choices.front().from);if(completed!=missing){auto exact=choices.front();exact.cargo=field(s,completed,"cargo");esg_move(s,t,exact);}}esr_commit(s,t,id,deck,face);}
    rr_set_attr(s,s.root,"_battle_resolved_attacker",rr_scalar(s,WOTR_NONE));return child;
}
uint64_t esg_pending(const Snapshot& s,const Snapshot& t){
    const auto p=obs_attr(s,s.root,"pending_card_option",false);require(!ctx_none(s,p)&&text_is(s,field(s,p,"card_name",false),"Shadows Gather")&&text_is(s,field(s,p,"side",false),"sh")&&text_is(s,field(s,p,"decider",false),"sh"),"No Shadow Shadows Gather choice pending");
    if(!truth(s,field(s,p,"direct",false))){
        const auto expected=esg_options(s,t);const auto options=child_edges(s,field(s,p,"options"));require(options.size()>1&&options.size()==expected.size(),"Shadows Gather choices stale");
        for(size_t i=0;i<options.size();++i)require(text_is(s,field(s,options[i],"from_region"),expected[i].from)&&text_is(s,field(s,options[i],"to_region"),expected[i].to),"Shadows Gather route stale");
    }
    return p;
}
std::unique_ptr<Snapshot> native_shadows_gather_choice_actions(const Snapshot& source,const Snapshot& t,int side){
    require(side==1,"Shadows Gather menu belongs to Shadow");esg_pending(source,t);auto result=std::make_unique<Snapshot>(source);auto& s=*result;const auto p=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t i=0;
    for(auto option:child_edges(s,field(s,p,"options"))){
        const auto params=ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,i++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",option}});
        rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),params}));
    }
    s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
std::unique_ptr<Snapshot> apply_shadows_gather_choice(const Snapshot& source,const Snapshot& t,int side,int64_t index){
    require(side==1,"Shadows Gather choice belongs to Shadow");const auto p=esg_pending(source,t);const auto options=child_edges(source,field(source,p,"options"));require(index>=0&&static_cast<uint64_t>(index)<options.size(),"Shadows Gather index invalid");
    const bool direct=truth(source,field(source,p,"direct",false));const std::string id(text(source,field(source,p,"card_id"))),deck=direct?"":std::string(text(source,field(source,p,"deck"))),face=direct?"":std::string(text(source,field(source,p,"die_face")));
    require(esg_supported(t,ev_card(t,id))&&(direct||contains_string(source,tc_hand(source,1),id)),"Shadows Gather pending card absent");
    auto child=action_clone(source);auto& s=*child;const auto choice=child_edges(s,field(s,obs_attr(s,s.root,"pending_card_option"),"options"))[index];rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));const bool queued=epa_paid_continuation_debt(s,id,direct);quiet_action_gate(s,t,false,false,false,false,false,false,false,false,false,false,false,true,queued);ring_waiver(s,0);ring_waiver(s,1);rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    const std::string from(text(s,field(s,choice,"from_region"))),to(text(s,field(s,choice,"to_region")));const auto legal=esg_options(s,t);require(std::any_of(legal.begin(),legal.end(),[&](const auto& c){return c.from==from&&c.to==to;}),"Shadows Gather route unavailable");
    const auto completed=cag_stage(s,t,id,choice,from);if(completed!=missing)esg_move(s,t,{from,to,-1,-1,field(s,completed,"cargo")});
    const auto marker=obs_attr(s,s.root,"_battle_resolved_attacker");if(!direct)esr_commit(s,t,id,deck,face);else su_resolve_overstacks(s,t);epa_maybe(s);rr_set_attr(s,s.root,"_battle_resolved_attacker",marker);return child;
}
