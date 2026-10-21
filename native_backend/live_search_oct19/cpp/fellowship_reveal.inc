// Reveal placement, optional Balrog, and per-stronghold Hunt draws. Protection
// and casualty choices remain explicit pending decisions.
bool fbr_drain_stage(Snapshot& s,uint64_t event);
bool fr_stronghold(const Snapshot& s,const Snapshot& t,std::string_view key){
    const auto meta=ctx_get(t,field(t,t.root,"board_regions"),key);
    return meta!=missing && truth(t,ctx_get(t,meta,"is_stronghold")) && contains_string(t,field(t,t.root,"shadow_region_keys"),key) &&
        !contains_string(s,ctx_side_value(s,obs_attr(s,s.root,"enemy_controlled"),0),key);
}
bool fr_pending(const Snapshot& s){
    return !ctx_none(s,obs_attr(s,s.root,"pending_hunt")) || !ctx_none(s,obs_attr(s,s.root,"pending_card_option",false)) ||
        !ctx_none(s,obs_attr(s,s.root,"pending_guide_change",false)) || truth(s,obs_attr(s,s.root,"_hunt_reveal_pending",false)) || truth(s,obs_attr(s,s.root,"_pending_hunt_damage",false));
}
std::vector<std::string> fr_neighbors(const Snapshot& s,std::string_view key){
    const auto board=obs_attr(s,s.root,"board"),adj=obs_attr(s,board,"_adj"),found=ctx_get(s,adj,key);
    require(found!=missing,"Fellowship route adjacency missing");std::vector<std::string> result;
    for(auto item:child_edges(s,found))result.push_back(std::string(text(s,item)));std::sort(result.begin(),result.end());return result;
}
std::vector<std::string> fr_path(const Snapshot& s,const Snapshot& t,std::string_view destination,bool declaration=false){
    const auto fs=obs_attr(s,s.root,"fellowship");const auto start=glb_fellowship_key(s,fs);const auto steps=ht_integer(s,fs,"progress");
    require(steps>=0 && !truth(s,obs_attr(s,fs,"mordor_entered")),"Invalid map reveal progress or Mordor context");
    struct Entry{int64_t cost,used;std::vector<std::string> route;};std::vector<Entry> frontier{{0,0,{start}}};
    std::unordered_map<std::string,int64_t> best{{start+"#0",0}};const auto seen=obs_attr(s,s.root,"_sh_tile_drawn_regions",false);
    while(!frontier.empty()){
        auto selected=std::min_element(frontier.begin(),frontier.end(),[](const Entry& a,const Entry& b){return a.cost!=b.cost?a.cost<b.cost:a.used!=b.used?a.used<b.used:a.route<b.route;});
        const auto entry=*selected;frontier.erase(selected);const auto& current=entry.route.back();
        if(best[current+"#"+std::to_string(entry.used)]!=entry.cost)continue;
        if(current==destination)return entry.route;if(entry.used==steps)continue;
        for(const auto& next:fr_neighbors(s,current)){
            const auto cost=entry.cost+(!declaration && fr_stronghold(s,t,next) && (ctx_none(s,seen) || !contains_string(s,seen,next)))+
                (next=="moria" && std::find(entry.route.begin(),entry.route.end(),"moria")==entry.route.end() && hpr_card(s,"BALROG_OF_MORIA")!=missing);
            const auto key=next+"#"+std::to_string(entry.used+1);auto old=best.find(key);
            if(old==best.end() || cost<old->second){best[key]=cost;auto route=entry.route;route.push_back(next);frontier.push_back({cost,entry.used+1,std::move(route)});}
        }
    }
    throw std::invalid_argument("Reveal destination is not reachable within progress");
}
std::vector<std::string> fr_destinations(const Snapshot& s,const Snapshot& t){
    const auto fs=obs_attr(s,s.root,"fellowship");if(truth(s,obs_attr(s,fs,"mordor_entered")))return {};
    const auto start=glb_fellowship_key(s,fs);const auto steps=ht_integer(s,fs,"progress");if(steps<=0)return {start};
    std::set<std::string> reached{start},frontier{start};
    for(int64_t step=0;step<steps && !frontier.empty();++step){std::set<std::string> next;for(const auto& key:frontier)for(const auto& neighbor:fr_neighbors(s,key)){
        const auto canonical=sanitize_name(neighbor);if(reached.insert(canonical).second)next.insert(canonical);
    }frontier=std::move(next);}
    const auto controls=ctx_side_value(s,obs_attr(s,s.root,"enemy_controlled"),1);std::vector<std::string> result;
    for(const auto& key:reached){
        const auto meta=ctx_get(t,field(t,t.root,"board_regions"),key);
        const bool settlement=meta!=missing && (truth(t,ctx_get(t,meta,"is_city")) || truth(t,ctx_get(t,meta,"is_stronghold")));
        const bool fp=key==start?!contains_string(t,field(t,t.root,"shadow_region_keys"),key):contains_string(t,field(t,t.root,"fp_region_keys"),key);
        if(!(settlement && fp && !contains_string(s,controls,key)))result.push_back(key);
    }
    if(result.empty())result.push_back(start);return result;
}
void fr_apply_stronghold(Snapshot& s,uint64_t tile){
    const auto damage=truth(s,obs_attr(s,tile,"is_eye"))?0:ht_integer(s,tile,"damage");
    ht_damage(s,damage,false);const auto discard=obs_attr(s,s.root,"hunt_discard");auto used=child_edges(s,discard);used.push_back(tile);set_edges(s,discard,used);
}
void fr_stronghold_draw(Snapshot& s,const Snapshot& t,void* random,std::string_view region,am_Builder& events,std::vector<uint64_t>& rows,bool skip_wizard=false,bool resume=false,std::string_view reason="reveal route"){
    auto seen=obs_attr(s,s.root,"_sh_tile_drawn_regions",false);
    if(!ctx_none(s,seen) && !resume){
        if(contains_string(s,seen,region))return;auto keys=child_edges(s,seen);keys.push_back(rr_string(s,region));set_edges(s,seen,keys);
    }
    if(!skip_wizard && hpr_card(s,"WIZARD_STAFF")!=missing){const auto stash=hpr_dict(s,{rr_string(s,"apply_kind"),rr_string(s,"stronghold"),rr_string(s,"region"),rr_string(s,region),rr_string(s,"reason"),rr_string(s,reason)});if(hpr_offer_wizard(s,stash))return;}
    const auto pool=obs_attr(s,s.root,"hunt_pool");if(!truth(s,pool))ht_recycle(s,random,events,rows);if(!truth(s,pool))return;
    auto tiles=child_edges(s,pool);const auto index=hf_pool(s,pool,random,false);
    const auto tile=tiles[index];const auto found=std::find_if(tiles.begin(),tiles.end(),[&](uint64_t other){return ft_tile_equal(s,other,tile);});tiles.erase(found);set_edges(s,pool,tiles);
    const auto special=obs_attr(s,tile,"special_name");const bool eye=truth(s,obs_attr(s,tile,"is_eye"));const auto damage=eye?0:ht_integer(s,tile,"damage");
    const std::string name=truth(s,special)?std::string(text(s,special)):eye?"Eye":std::to_string(damage)+(truth(s,obs_attr(s,tile,"reveal"))?"R":"");
    rows.push_back(events.row(-1,"stronghold_tile_draw",{events.str("tile_name"),events.str(name),events.str("region"),events.str(region),events.str("ctx"),events.str(reason),events.str("damage"),events.integer(damage)}));
    if(hpr_card(s,"MITHRIL_COAT")!=missing){const auto choices=hpr_dict(s,{rr_string(s,"region"),rr_string(s,region),rr_string(s,"reason"),rr_string(s,reason)});
        const auto stash=hpr_dict(s,{rr_string(s,"tile"),tile,rr_string(s,"apply_kind"),rr_string(s,"stronghold"),rr_string(s,"choices"),choices});if(hpr_offer_mithril(s,stash))return;}
    fr_apply_stronghold(s,tile);
}
void fr_drain(Snapshot& s,const Snapshot& t,void* random,am_Builder& events,std::vector<uint64_t>& rows){
    while(truth(s,obs_attr(s,s.root,"_fellowship_route_event",false)) || truth(s,obs_attr(s,s.root,"_pending_reveal_strongholds",false))){
        ht_settle_reveal(s);
        if(terminal_should_end(check_victory(s,t),false)){
            rr_set_attr(s,s.root,"_pending_reveal_strongholds",rr_scalar(s,WOTR_TUPLE));rr_set_attr(s,s.root,"_fellowship_route_event",rr_scalar(s,WOTR_NONE));rr_set_attr(s,s.root,"_reveal_place_due",rr_scalar(s,WOTR_BOOL));return;
        }
        if(fr_pending(s) || ht_integer(s,s.root,"_hunt_reveal_pending") || truth(s,obs_attr(s,s.root,"_hobbit_place_queue",false)) || truth(s,obs_attr(s,s.root,"pending_card_followup",false)) || truth(s,obs_attr(s,s.root,"pending_discard",false)))return;
        const auto event=obs_attr(s,s.root,"_fellowship_route_event",false);
        if(truth(s,event) && fbr_drain_stage(s,event))return;
        const auto pending=obs_attr(s,s.root,"_pending_reveal_strongholds",false);
        if(!truth(s,pending)){rr_set_attr(s,s.root,"_fellowship_route_event",rr_scalar(s,WOTR_NONE));return;}
        auto remaining=child_edges(s,pending);const auto region=std::string(text(s,remaining.front()));remaining.erase(remaining.begin());
        const auto next=rr_scalar(s,WOTR_TUPLE);set_edges(s,next,remaining);rr_set_attr(s,s.root,"_pending_reveal_strongholds",next);
        fr_stronghold_draw(s,t,random,region,events,rows);
        if(ht_integer(s,s.root,"_pending_hunt_damage")>0 && ctx_none(s,obs_attr(s,s.root,"pending_card_option",false)) && ctx_none(s,obs_attr(s,s.root,"pending_hunt")) && !hpr_offer_reduction(s))hd_settle(s,t);
    }
}
void fr_place(Snapshot& s,const Snapshot& t,void* random,std::string_view destination,am_Builder& events,std::vector<uint64_t>& rows){
    const auto route=fr_path(s,t,destination);const auto& key=route.back();const auto meta=ctx_get(t,field(t,t.root,"board_regions"),key);
    require(meta!=missing,"Reveal destination metadata missing");
    require(!((truth(t,ctx_get(t,meta,"is_city")) || truth(t,ctx_get(t,meta,"is_stronghold"))) && contains_string(t,field(t,t.root,"fp_region_keys"),key) &&
        !contains_string(s,ctx_side_value(s,obs_attr(s,s.root,"enemy_controlled"),1),key)),"Reveal cannot end in an unconquered FP city/stronghold");
    require(!truth(s,obs_attr(s,s.root,"_pending_reveal_strongholds",false)) && !truth(s,obs_attr(s,s.root,"_fellowship_route_event",false)),"Previous reveal route unfinished");
    auto seen=obs_attr(s,s.root,"_sh_tile_drawn_regions",false);if(ctx_none(s,seen)){seen=rr_scalar(s,WOTR_SET);rr_set_attr(s,s.root,"_sh_tile_drawn_regions",seen);}
    std::vector<uint64_t> owed,trace_route,trace_owed;std::set<std::string> unique;
    for(const auto& region:route){trace_route.push_back(events.str(region));if(fr_stronghold(s,t,region) && !contains_string(s,seen,region) && unique.insert(region).second){owed.push_back(rr_string(s,region));trace_owed.push_back(events.str(region));}}
    const auto fs=obs_attr(s,s.root,"fellowship"),board=obs_attr(s,s.root,"board"),region=ctx_get(s,obs_attr(s,board,"_regions"),key);
    require(region!=missing && text_is(s,obs_attr(s,region,"key"),key),"Reveal destination Board Region noncanonical");
    rr_set_attr(s,fs,"region",region);rr_set_attr(s,fs,"progress",rr_scalar(s,WOTR_INT));rr_set_attr(s,s.root,"_reveal_place_due",rr_scalar(s,WOTR_BOOL));
    const auto queue=rr_scalar(s,WOTR_TUPLE);set_edges(s,queue,owed);rr_set_attr(s,s.root,"_pending_reveal_strongholds",queue);
    const bool balrog=std::find(route.begin(),route.end(),"moria")!=route.end() && hpr_card(s,"BALROG_OF_MORIA")!=missing;
    const auto event=rr_scalar(s,WOTR_DICT);set_edges(s,event,{rr_string(s,"kind"),rr_string(s,"reveal"),rr_string(s,"balrog_due"),rr_scalar(s,WOTR_BOOL,balrog),rr_string(s,"conditional_strongholds"),rr_scalar(s,WOTR_TUPLE)});rr_set_attr(s,s.root,"_fellowship_route_event",event);
    rows.push_back(events.row(-1,"fellowship_reveal_route",{events.str("path"),events.container(WOTR_LIST,trace_route),events.str("strongholds"),events.container(WOTR_LIST,trace_owed)}));
    fr_drain(s,t,random,events,rows);
}
void fr_open(Snapshot& s,const Snapshot& t,void* random,am_Builder& events,std::vector<uint64_t>& rows){
    ht_settle_reveal(s);
    fr_drain(s,t,random,events,rows);if(truth(s,obs_attr(s,s.root,"_fellowship_route_event",false)) || !truth(s,obs_attr(s,s.root,"_reveal_place_due",false)))return;
    if(terminal_should_end(check_victory(s,t),false)){rr_set_attr(s,s.root,"_reveal_place_due",rr_scalar(s,WOTR_BOOL));return;}
    if(fr_pending(s) || truth(s,obs_attr(s,s.root,"_hobbit_place_queue",false)))return;
    const auto fs=obs_attr(s,s.root,"fellowship");if(truth(s,obs_attr(s,fs,"mordor_entered"))){rr_set_attr(s,s.root,"_reveal_place_due",rr_scalar(s,WOTR_BOOL));return;}
    const auto key=glb_fellowship_key(s,fs);
    if(ht_integer(s,fs,"progress")<=0 || key=="morannon" || key=="minas morgul"){fr_place(s,t,random,key,events,rows);return;}
    const auto destinations=fr_destinations(s,t);require(!destinations.empty(),"Due reveal has no legal placement");if(destinations.size()==1)fr_place(s,t,random,destinations.front(),events,rows);
}
