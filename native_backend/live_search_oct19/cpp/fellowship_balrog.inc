// RNG/trace-bearing declaration wrapper. The quiet declaration API remains
// available; only this wrapper registers card-dependent route obligations.
bool fbr_offer(Snapshot& s){
    const auto card=hpr_card(s,"BALROG_OF_MORIA");if(card==missing)return false;
    const auto yes=hpr_dict(s,{rr_string(s,"balrog_declare_draw"),rr_scalar(s,WOTR_BOOL,1)}),no=hpr_dict(s,{rr_string(s,"balrog_declare_draw"),rr_scalar(s,WOTR_BOOL)});
    const auto options=rr_scalar(s,WOTR_LIST);set_edges(s,options,{yes,no});
    const auto pending=hpr_dict(s,{rr_string(s,"card_id"),obs_attr(s,card,"card_id"),rr_string(s,"card_name"),rr_string(s,"Balrog of Moria"),rr_string(s,"side"),rr_string(s,"sh"),rr_string(s,"decider"),rr_string(s,"sh"),rr_string(s,"direct"),rr_scalar(s,WOTR_BOOL,1),rr_string(s,"options"),options});
    rr_set_attr(s,s.root,"pending_card_option",pending);return true;
}
void fbr_put(Snapshot& s,uint64_t dict,const char* key,uint64_t value){
    auto entries=child_edges(s,dict);for(size_t i=0;i<entries.size();i+=2)if(text_is(s,entries[i],key)){entries[i+1]=value;set_edges(s,dict,entries);return;}
    entries.push_back(rr_string(s,key));entries.push_back(value);set_edges(s,dict,entries);
}
bool fbr_drain_stage(Snapshot& s,uint64_t event){
    require(text_is(s,field(s,event,"kind"),"reveal") || text_is(s,field(s,event,"kind"),"declare"),"Unknown Fellowship route event kind");
    if(truth(s,field(s,event,"balrog_due"))){fbr_put(s,event,"balrog_due",rr_scalar(s,WOTR_BOOL));if(fbr_offer(s))return true;}
    if(text_is(s,field(s,event,"kind"),"declare") && !truth(s,obs_attr(s,obs_attr(s,s.root,"fellowship"),"hidden"))){
        rr_set_attr(s,s.root,"_reveal_place_due",rr_scalar(s,WOTR_BOOL));const auto owed=child_edges(s,field(s,event,"conditional_strongholds"));fbr_put(s,event,"conditional_strongholds",rr_scalar(s,WOTR_TUPLE));
        auto seen=obs_attr(s,s.root,"_sh_tile_drawn_regions",false);if(!truth(s,seen)){seen=rr_scalar(s,WOTR_SET);rr_set_attr(s,s.root,"_sh_tile_drawn_regions",seen);}
        const auto old=obs_attr(s,s.root,"_pending_reveal_strongholds",false);auto queue=truth(s,old)?child_edges(s,old):std::vector<uint64_t>{};
        for(auto region:owed)if(!contains_string(s,seen,text(s,region)))queue.push_back(region);
        const auto next=rr_scalar(s,WOTR_TUPLE);set_edges(s,next,queue);rr_set_attr(s,s.root,"_pending_reveal_strongholds",next);
    }
    return false;
}
void fbr_register_route(Snapshot& s,const Snapshot& t,void* rng,const std::vector<std::string>& route,am_Builder& events,std::vector<uint64_t>& rows){
    if(std::find(route.begin(),route.end(),"moria")==route.end() || hpr_card(s,"BALROG_OF_MORIA")==missing)return;
    require(!truth(s,obs_attr(s,s.root,"_fellowship_route_event",false)) && !truth(s,obs_attr(s,s.root,"_pending_reveal_strongholds",false)),"Previous declaration route unfinished");
    const auto fs=obs_attr(s,s.root,"fellowship");std::vector<uint64_t> conditional,trace;std::set<std::string> seen;
    for(const auto& region:route){trace.push_back(events.str(region));if(truth(s,obs_attr(s,fs,"hidden")) && fr_stronghold(s,t,region) && seen.insert(region).second)conditional.push_back(rr_string(s,region));}
    const auto owed=rr_scalar(s,WOTR_TUPLE);set_edges(s,owed,conditional);
    const auto event=hpr_dict(s,{rr_string(s,"kind"),rr_string(s,"declare"),rr_string(s,"balrog_due"),rr_scalar(s,WOTR_BOOL,1),rr_string(s,"conditional_strongholds"),owed});rr_set_attr(s,s.root,"_fellowship_route_event",event);
    rows.push_back(events.row(-1,"fellowship_declaration_route",{events.str("path"),events.container(WOTR_LIST,trace)}));fr_drain(s,t,rng,events,rows);
}
void fbr_stationary(Snapshot& s,const Snapshot& t,void* rng,am_Builder& events,std::vector<uint64_t>& rows){
    const auto fs=obs_attr(s,s.root,"fellowship");
    if(!ctx_none(s,obs_attr(s,s.root,"pending_card_option",false)) || truth(s,obs_attr(s,fs,"mordor_entered")) || glb_fellowship_key(s,fs)!="moria" || truth(s,obs_attr(s,fs,"progress")))return;
    fbr_register_route(s,t,rng,{"moria"},events,rows);
}
fm_Result apply_fellowship_declare_with_rng(const Snapshot& source,const Snapshot& t,const void* random_source,std::string_view destination,bool pass=false){
    const auto route=pass?std::vector<std::string>{}:fr_path(source,t,sanitize_name(destination),true);
    fm_Result result;result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());
    result.state=apply_fellowship_declare(source,t,destination,pass,true);auto& s=*result.state;am_Builder events;std::vector<uint64_t> rows;
    if(!pass)fbr_register_route(s,t,result.random.get(),route,events,rows);
    fr_open(s,t,result.random.get(),events,rows);hpr_return_mithril(s,result.random.get());fm_finish(s,t);
    events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
