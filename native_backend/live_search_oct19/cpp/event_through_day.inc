// Public quiet-domain movement, verified against source public journeys.
// Through a Day and a Night: direct card movement, with the pinned source's cargo,
// identity, siege and overflow behavior. Ordinary movement is not fabricated.
bool etd_supported(const Snapshot& t,uint64_t card){return card!=missing&&text_is(t,obs_attr(t,card,"card_name"),"Through a Day and a Night");}
bool etd_pending(const Snapshot& s){
    const auto p=obs_attr(s,s.root,"pending_card_option",false);
    return !ctx_none(s,p)&&text_is(s,field(s,p,"card_name",false),"Through a Day and a Night");
}

bool etd_companion(const Snapshot& s,uint64_t army){
    if(ctx_none(s,army))return false;for(auto l:child_edges(s,obs_attr(s,army,"leaders")))if(object_type(s,l)=="game_state.Companion")return true;return false;
}
bool etd_enter(const Snapshot& s,const Snapshot& t,uint64_t u,std::string_view to){
    const auto nation=ctx_get(t,field(t,t.root,"region_nations"),to);const auto own=ctx_nation(s,obs_attr(s,u,"nation"));return ctx_none(t,nation)||political_view(s,own,true).box>=3||own==ctx_nation(t,nation);
}
std::set<std::string> etd_reachable(const Snapshot& s,const Snapshot& t,const std::string& from){
    std::set<std::string> seen{from},out;std::vector<std::pair<std::string,int>> queue{{from,0}};
    for(size_t i=0;i<queue.size();++i){const auto [cur,depth]=queue[i];if(depth>=2)continue;for(const auto& raw:esg_neighbors(s,cur)){
        const auto to=sanitize_name(raw);if(!seen.insert(to).second)continue;const auto army=hr_army(s,to);const bool siege=contains_string(s,obs_attr(s,s.root,"besieged_strongholds",false),to);
        const bool join=siege&&evb_units(s,army,0)&&c9_garrison(s,t,to)!=0;if((evb_units(s,army,1)||siege)&&!join)continue;out.insert(to);if(!siege)queue.emplace_back(to,depth+1);
    }}return out;
}
struct etd_Path{std::string mid;std::vector<uint64_t> pool;};
etd_Path etd_path(const Snapshot& s,const Snapshot& t,const std::string& from,const std::string& to,const std::vector<uint64_t>& units,int64_t regular=-1,int64_t elite=-1){
    std::set<std::string> adjacent;for(const auto& raw:esg_neighbors(s,from))adjacent.insert(sanitize_name(raw));
    std::vector<std::string> candidates;
    if(adjacent.count(to))candidates.push_back("");
    else for(const auto& mid:adjacent){
        if(contains_string(s,obs_attr(s,s.root,"besieged_strongholds",false),mid)||evb_units(s,hr_army(s,mid),1))continue;
        const auto neighbors=esg_neighbors(s,mid);if(std::any_of(neighbors.begin(),neighbors.end(),[&](const std::string& n){return sanitize_name(n)==to;}))candidates.push_back(mid);
    }
    const auto army=hr_army(s,from);etd_Path best;int64_t best_leaders=-1;
    for(const auto& mid:candidates){
        std::vector<uint64_t> pool;int64_t nr=0,ne=0;
        for(auto u:units)if(etd_enter(s,t,u,to)&&(mid.empty()||etd_enter(s,t,u,mid))){pool.push_back(u);(enum_is(s,obs_attr(s,u,"kind"),"game_state.UnitKind","ELITE")?ne:nr)++;}
        const auto mr=regular<0?nr:regular,me=elite<0?ne:elite;
        if(mr>nr||me>ne||mr+me==0)continue;
        if(!ctx_none(s,army)&&mr+me==static_cast<int64_t>(units.size())&&cdm_fp_guard(s,t,army,to,mid))continue;
        int64_t leaders=0;if(!ctx_none(s,army))for(auto l:child_edges(s,obs_attr(s,army,"leaders")))leaders+=am_plain(s,l)&&am_leader_side(s,l)==0&&etd_enter(s,t,l,to)&&(mid.empty()||etd_enter(s,t,l,mid));
        if(pool.size()>best.pool.size()||(pool.size()==best.pool.size()&&leaders>best_leaders)){best={mid,std::move(pool)};best_leaders=leaders;}
    }return best;
}
uint64_t etd_options(Snapshot& s,const Snapshot& t){
    std::vector<uint64_t> rows;const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));
    for(size_t i=0;i<armies.size();i+=2){const auto from=sanitize_name(text(s,armies[i]));const auto army=armies[i+1];if(!evb_units(s,army,0)||!etd_companion(s,army))continue;
        if(contains_string(s,obs_attr(s,s.root,"besieged_strongholds",false),from)&&c9_garrison(s,t,from)==0)continue;
        std::vector<uint64_t> units;for(auto u:child_edges(s,obs_attr(s,army,"units")))if(ctx_unit_side(s,u)==0)units.push_back(u);
        for(const auto& to:etd_reachable(s,t,from)){const auto path=etd_path(s,t,from,to,units);int64_t full_r=0,full_e=0,regular=0,elite=0;for(auto u:path.pool)(enum_is(s,obs_attr(s,u,"kind"),"game_state.UnitKind","ELITE")?full_e:full_r)++;for(auto u:units)(enum_is(s,obs_attr(s,u,"kind"),"game_state.UnitKind","ELITE")?elite:regular)++;
            if(!path.pool.empty())rows.push_back(esg_option(s,{from,to}));
            for(int64_t r=0;r<=regular;++r)for(int64_t e=0;e<=elite;++e)if(r+e>0&&(path.pool.empty()||r!=full_r||e!=full_e)&&!etd_path(s,t,from,to,units,r,e).pool.empty())rows.push_back(esg_option(s,{from,to,r,e}));
        }}return cp_container(s,WOTR_LIST,rows);
}
int64_t etd_move(Snapshot& s,const Snapshot& t,const std::string& from,const std::string& to,const std::string& via,int64_t regular,int64_t elite,bool allow_overstack){
    return cdm_move(s,t,from,to,0,true,via,regular,elite,allow_overstack);
}

void etd_effect(Snapshot& s,const Snapshot& t,uint64_t choice,bool learned){
    const auto from=sanitize_name(text(s,field(s,choice,"from_region"))),to=sanitize_name(text(s,field(s,choice,"to_region")));const auto src=hr_army(s,from);
    if(!etd_companion(s,src))return;if(contains_string(s,obs_attr(s,s.root,"besieged_strongholds",false),from)&&c9_garrison(s,t,from)==0)return;
    if(!etd_reachable(s,t,from).count(to))return;std::vector<uint64_t> units;for(auto u:child_edges(s,obs_attr(s,src,"units")))if(ctx_unit_side(s,u)==0)units.push_back(u);
    const auto r=field(s,choice,"n_regular",false),e=field(s,choice,"n_elite",false);
    const auto nr=ctx_none(s,r)?-1:typed(s,r,WOTR_INT,"Through Day regular count invalid").integer,ne=ctx_none(s,e)?-1:typed(s,e,WOTR_INT,"Through Day elite count invalid").integer;
    const auto path=etd_path(s,t,from,to,units,nr,ne);if(path.pool.empty())return;
    if(etd_move(s,t,from,to,path.mid,nr,ne,learned)&&!path.mid.empty()){
        movement_capture_effects(s,t,path.mid,0);const auto nation=ctx_get(t,field(t,t.root,"movement_region_nations"),path.mid);if(!ctx_none(t,nation)&&!fp_nation(ctx_nation(t,nation)))wt_activate_if_inactive(s,ctx_nation(t,nation),"attack");}
}
fm_Result apply_through_day_event(const Snapshot& source,const Snapshot& t,const void* random_source,int32_t side,int64_t index,std::string_view id){
    require(side==0,"Through Day belongs to FP");quiet_stable_action_gate(source,t);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==0,"Through Day violates retained Ring actor");
    const auto card=ev_card(t,id);require(etd_supported(t,card)&&contains_string(source,tc_hand(source,0),id),"Through Day card absent");require(ev_requirement(source,t,card),"Through Day requirement false");
    const auto die=ev_payment(source,t,0,index,card);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace")),deck(ev_lower(text(t,obs_attr(t,card,"deck"))));
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;am_Builder events;std::vector<uint64_t> rows;
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));rr_set_attr(s,ev_payment(s,t,0,index,card),"used",rr_scalar(s,WOTR_BOOL,1));
    const auto options=etd_options(s,t);const auto choices=child_edges(s,options);
    if(choices.size()>1){
        rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,"Through a Day and a Night")},{"side",rr_string(s,"fp")},{"decider",rr_string(s,"fp")},{"options",options},{"die_face",rr_string(s,face)},{"deck",rr_string(s,deck)}}));tc_open_hand_limit_discard(s);
    }else{if(!choices.empty())etd_effect(s,t,choices.front(),false);er_commit(s,t,id,deck,face);}
    cau_finish(s,t,result.random.get(),-1,events,rows);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
std::unique_ptr<Snapshot> native_through_day_choice_actions(const Snapshot& source,int32_t side){
    require(side==0&&etd_pending(source),"No FP Through Day choice pending");auto out=std::make_unique<Snapshot>(source);auto& s=*out;const auto p=obs_attr(s,s.root,"pending_card_option");
    require(text_is(s,field(s,p,"side",false),"fp")&&text_is(s,field(s,p,"decider",false),"fp"),"Through Day pending belongs to FP");
    std::vector<uint64_t> rows;int64_t index=0;
    for(auto choice:child_edges(s,field(s,p,"options")))rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",choice}})}));
    s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
fm_Result apply_through_day_choice(const Snapshot& source,const Snapshot& t,const void* random_source,int32_t side,int64_t index){
    require(side==0&&etd_pending(source),"No FP Through Day choice pending");const auto p=obs_attr(source,source.root,"pending_card_option");const auto options=child_edges(source,field(source,p,"options"));require(index>=0&&static_cast<uint64_t>(index)<options.size(),"Through Day choice index exceeds menu");
    const std::string id(text(source,field(source,p,"card_id"))),deck(text(source,field(source,p,"deck"))),face(text(source,field(source,p,"die_face")));require(etd_supported(t,ev_card(t,id))&&contains_string(source,tc_hand(source,0),id),"Pending Through Day card absent");
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;am_Builder events;std::vector<uint64_t> rows;
    const auto choice=child_edges(s,field(s,obs_attr(s,s.root,"pending_card_option"),"options"))[index],marker=obs_attr(s,s.root,"_battle_resolved_attacker",false);const int marker_side=ctx_none(s,marker)?-1:ctx_side(s,marker);
    rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));quiet_stable_action_gate(s,t);ring_waiver(s,0);ring_waiver(s,1);
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    // play_event_card checks current requirements again after clearing the
    // pending. A newly false requirement keeps the card without a die refund.
    if(ev_requirement(s,t,ev_card(t,id))){etd_effect(s,t,choice,true);er_commit(s,t,id,deck,face);}
    cau_finish(s,t,result.random.get(),marker_side,events,rows);
    events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
