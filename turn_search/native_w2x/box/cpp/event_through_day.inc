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
// R2-E3-02 (decision D5, 2026-10-22): every legal route of one (from, to,
// composition) in sorted-intermediate order (mid "" when adjacent), each with
// its own pool, guard check and plain-leader count. Mirrors Python
// event_effects._tdaan_routes (_tdaan_path(..., _all_routes=True)).
struct etd_Route{std::string mid;std::vector<uint64_t> pool;int64_t leaders;};
std::vector<etd_Route> etd_routes(const Snapshot& s,const Snapshot& t,const std::string& from,const std::string& to,const std::vector<uint64_t>& units,int64_t regular=-1,int64_t elite=-1){
    std::set<std::string> adjacent;for(const auto& raw:esg_neighbors(s,from))adjacent.insert(sanitize_name(raw));
    std::vector<std::string> candidates;
    if(adjacent.count(to))candidates.push_back("");
    else for(const auto& mid:adjacent){
        if(contains_string(s,obs_attr(s,s.root,"besieged_strongholds",false),mid)||evb_units(s,hr_army(s,mid),1))continue;
        const auto neighbors=esg_neighbors(s,mid);if(std::any_of(neighbors.begin(),neighbors.end(),[&](const std::string& n){return sanitize_name(n)==to;}))candidates.push_back(mid);
    }
    const auto army=hr_army(s,from);std::vector<etd_Route> routes;
    for(const auto& mid:candidates){
        std::vector<uint64_t> pool;int64_t nr=0,ne=0;
        for(auto u:units)if(etd_enter(s,t,u,to)&&(mid.empty()||etd_enter(s,t,u,mid))){pool.push_back(u);(enum_is(s,obs_attr(s,u,"kind"),"game_state.UnitKind","ELITE")?ne:nr)++;}
        const auto mr=regular<0?nr:regular,me=elite<0?ne:elite;
        if(mr>nr||me>ne||mr+me==0)continue;
        if(!ctx_none(s,army)&&mr+me==static_cast<int64_t>(units.size())&&cdm_fp_guard(s,t,army,to,mid))continue;
        int64_t leaders=0;if(!ctx_none(s,army))for(auto l:child_edges(s,obs_attr(s,army,"leaders")))leaders+=am_plain(s,l)&&am_leader_side(s,l)==0&&etd_enter(s,t,l,to)&&(mid.empty()||etd_enter(s,t,l,mid));
        routes.push_back({mid,std::move(pool),leaders});
    }return routes;
}
// The deterministic DEFAULT route of a row without via_region (Python
// _tdaan_path): the largest pool, ties to more plain leaders, then the first
// intermediate in sorted order.
etd_Path etd_path(const Snapshot& s,const Snapshot& t,const std::string& from,const std::string& to,const std::vector<uint64_t>& units,int64_t regular=-1,int64_t elite=-1){
    etd_Path best;int64_t best_leaders=-1;
    for(auto& route:etd_routes(s,t,from,to,units,regular,elite))
        if(route.pool.size()>best.pool.size()||(route.pool.size()==best.pool.size()&&route.leaders>best_leaders)){best={route.mid,std::move(route.pool)};best_leaders=route.leaders;}
    return best;
}
// Python _tdaan_walk_effects: what walking THROUGH mid does for an FP army -
// (captures an empty enemy-controlled settlement: exactly when
// movement_capture_effects(..., 0) changes control; the Shadow nation the
// entry activates, else ""). ("" mid = adjacent move: no effect.)
std::pair<bool,std::string> etd_walk_effects(const Snapshot& s,const Snapshot& t,const std::string& mid){
    if(mid.empty())return {false,std::string()};
    bool captures=false;const auto meta=ctx_get(t,field(t,t.root,"board_regions"),mid);
    if(meta!=missing&&(truth(t,ctx_get(t,meta,"is_city"))||truth(t,ctx_get(t,meta,"is_stronghold"))||truth(t,ctx_get(t,meta,"is_town")))){
        const auto controls=obs_attr(s,s.root,"enemy_controlled");
        if(contains_string(t,field(t,t.root,"shadow_region_keys"),mid))captures=!contains_string(s,ctx_side_value(s,controls,0),mid);
        else if(contains_string(t,field(t,t.root,"fp_region_keys"),mid))captures=contains_string(s,ctx_side_value(s,controls,1),mid);
    }
    std::string activates;const auto nation=ctx_get(t,field(t,t.root,"movement_region_nations"),mid);
    if(!ctx_none(t,nation)&&!fp_nation(ctx_nation(t,nation))&&!wt_active(s,ctx_nation(t,nation)))activates=ctx_nation(t,nation);
    return {captures,activates};
}
int64_t etd_regulars(const Snapshot& s,const std::vector<uint64_t>& pool){
    int64_t n=0;for(auto u:pool)n+=!enum_is(s,obs_attr(s,u,"kind"),"game_state.UnitKind","ELITE");return n;
}
// R4-TDAAN-SPLIT-UNIT-IDENTITY (D5 E3-02: WHICH units take the move is the FP's choice).
// Python _tdaan_freeze_units / _tdaan_unit_sets / _tdaan_pool_holds: a split's unit identity in
// the army_card_cargo freeze form (sorted (kind,name,subtype,count) rows); the distinct sets of
// `regular` Regulars + `elite` Elites a pool can supply (am_figure_sets per kind, Regular sets
// slowest, each set sorted - the victor_advance E5-01 order).
am_FigureSet etd_freeze(const Snapshot& s,const std::vector<uint64_t>& units){
    std::map<am_FigureKey,int64_t> counts;for(auto u:units)++counts[am_figure_key(s,u,true)];return am_FigureSet(counts.begin(),counts.end());
}
std::vector<am_FigureSet> etd_unit_sets(const Snapshot& s,const std::vector<uint64_t>& pool,int64_t regular,int64_t elite){
    std::vector<uint64_t> regs,els;for(auto u:pool)(enum_is(s,obs_attr(s,u,"kind"),"game_state.UnitKind","ELITE")?els:regs).push_back(u);
    std::vector<am_FigureSet> out;const auto esets=am_figure_sets(s,els,elite,true);
    for(const auto& rs:am_figure_sets(s,regs,regular,true))for(const auto& es:esets){am_FigureSet mix=rs;mix.insert(mix.end(),es.begin(),es.end());std::sort(mix.begin(),mix.end());out.push_back(std::move(mix));}
    return out;
}
bool etd_holds(const Snapshot& s,const std::vector<uint64_t>& pool,const am_FigureSet& set){
    std::map<am_FigureKey,int64_t> have;for(auto u:pool)++have[am_figure_key(s,u,true)];
    for(const auto& entry:set){const auto found=have.find(entry.first);if(found==have.end()||found->second<entry.second)return false;}return true;
}
// Python _tdaan_row_route: the (mid, pool) a row executes - ONE body for etd_effect, the option
// surface and the v53 roster. has_via: that legal route or refusal (never re-routed). unit_set:
// only routes whose pool can carry the named units qualify; without via the best of them (most
// units, then most following plain leaders, first in sorted order - etd_path's score); none =
// refusal. Neither: the etd_path default (empty pool when nothing may move).
etd_Path etd_row_route(const Snapshot& s,const Snapshot& t,const std::string& from,const std::string& to,const std::vector<uint64_t>& units,int64_t regular,int64_t elite,bool has_via,const std::string& via,const am_FigureSet* unit_set){
    if(!unit_set&&!has_via)return etd_path(s,t,from,to,units,regular,elite);
    auto routes=etd_routes(s,t,from,to,units,regular,elite);
    if(unit_set){
        require(regular>=0&&elite>=0,"Through Day unit_set needs n_regular and n_elite");
        int64_t pr=0,pe=0;for(const auto& entry:*unit_set){require(std::get<0>(entry.first)=="unit","Through Day unit_set names a non-unit figure");(std::get<2>(entry.first)=="ELITE"?pe:pr)+=entry.second;}
        require(pr==regular&&pe==elite,"Through Day unit_set disagrees with the unit counts");
        std::vector<etd_Route> kept;for(auto& r:routes)if(etd_holds(s,r.pool,*unit_set))kept.push_back(std::move(r));routes=std::move(kept);
    }
    if(has_via){for(auto& r:routes)if(!via.empty()&&r.mid==via)return etd_Path{r.mid,std::move(r.pool)};throw std::invalid_argument("Through Day via_region is not a legal route for this group");}
    etd_Path best;int64_t best_leaders=-1;bool found=false;
    for(auto& r:routes)if(!found||r.pool.size()>best.pool.size()||(r.pool.size()==best.pool.size()&&r.leaders>best_leaders)){best=etd_Path{r.mid,std::move(r.pool)};best_leaders=r.leaders;found=true;}
    if(!found)throw std::invalid_argument("Through Day: no legal route carries the named units");
    return best;
}
// Python _tdaan_route_rows: the legal routes of ONE (from, to, composition)
// row as (via, mid, pool, unit set). One row with via "" (the etd_path default, the
// pre-D5 surface) when every route has the same walk-through effects;
// otherwise one row PER INTERMEDIATE carrying via_region. R4: a split whose counts
// several distinct unit sets fill is one row PER SET ('unit_set') - per route when the
// routes differ, else over every route's pool (routes order, set order, first appearance),
// each run by etd_row_route; a count admitting one set keeps its count-only row. covered =
// the (mid, frozen unit set) a full-army route row already represents (never repeated).
struct etd_Row{std::string via,mid;std::vector<uint64_t> pool;bool named;am_FigureSet set;};
using etd_Key=std::pair<std::string,am_FigureSet>;
std::vector<etd_Row> etd_route_rows(const Snapshot& s,const Snapshot& t,const std::string& from,const std::string& to,const std::vector<uint64_t>& units,int64_t regular,int64_t elite,const std::set<etd_Key>& covered){
    auto routes=etd_routes(s,t,from,to,units,regular,elite);std::vector<etd_Row> rows;if(routes.empty())return rows;
    const bool split=regular>=0||elite>=0;
    const auto sets_of=[&](const std::vector<uint64_t>& pool)->std::vector<am_FigureSet>{
        if(!split)return {etd_freeze(s,pool)};
        const auto nr=etd_regulars(s,pool);return etd_unit_sets(s,pool,regular<0?nr:regular,elite<0?static_cast<int64_t>(pool.size())-nr:elite);};
    std::set<std::pair<bool,std::string>> effects;for(const auto& r:routes)effects.insert(etd_walk_effects(s,t,r.mid));
    if(effects.size()>1){
        for(const auto& r:routes){const auto sets=sets_of(r.pool);const bool named=split&&sets.size()>1;
            for(const auto& set:sets)if(!covered.count(etd_Key{r.mid,set}))rows.push_back(etd_Row{r.mid,r.mid,r.pool,named,named?set:am_FigureSet()});}
        return rows;
    }
    std::vector<am_FigureSet> sets;std::set<am_FigureSet> seen;
    for(const auto& r:routes)for(const auto& set:sets_of(r.pool))if(seen.insert(set).second)sets.push_back(set);
    if(!(split&&sets.size()>1)){
        for(const auto& r:routes)for(const auto& set:sets_of(r.pool))if(covered.count(etd_Key{r.mid,set}))return rows;
        auto path=etd_path(s,t,from,to,units,regular,elite);rows.push_back(etd_Row{std::string(),path.mid,std::move(path.pool),false,am_FigureSet()});return rows;
    }
    for(const auto& set:sets){
        bool skip=false;for(const auto& r:routes)skip=skip||covered.count(etd_Key{r.mid,set})!=0;if(skip)continue;
        auto path=etd_row_route(s,t,from,to,units,regular,elite,false,std::string(),&set);rows.push_back(etd_Row{std::string(),path.mid,std::move(path.pool),true,set});
    }
    return rows;
}
// The unit_set value: a tuple of (kind,name,subtype,count) tuples in set order (Python tuple form).
uint64_t etd_set_value(Snapshot& s,const am_FigureSet& set){
    std::vector<uint64_t> rows;rows.reserve(set.size());
    for(const auto& entry:set){const auto kind=rr_string(s,std::get<0>(entry.first)),name=rr_string(s,std::get<1>(entry.first)),sub=rr_string(s,std::get<2>(entry.first)),count=rr_scalar(s,WOTR_INT,entry.second);rows.push_back(cp_container(s,WOTR_TUPLE,{kind,name,sub,count}));}
    return cp_container(s,WOTR_TUPLE,rows);
}
uint64_t etd_option(Snapshot& s,const std::string& from,const std::string& to,int64_t regular,int64_t elite,const std::string& via,const am_FigureSet* unit_set=nullptr){
    std::vector<std::pair<const char*,uint64_t>> values{{"from_region",rr_string(s,from)},{"to_region",rr_string(s,to)}};
    if(regular>=0){values.emplace_back("n_regular",rr_scalar(s,WOTR_INT,regular));values.emplace_back("n_elite",rr_scalar(s,WOTR_INT,elite));}
    if(!via.empty())values.emplace_back("via_region",rr_string(s,via));
    if(unit_set)values.emplace_back("unit_set",etd_set_value(s,*unit_set));
    return ev_dict(s,values);
}
uint64_t etd_options(Snapshot& s,const Snapshot& t){
    std::vector<uint64_t> rows;const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));
    for(size_t i=0;i<armies.size();i+=2){const auto from=sanitize_name(text(s,armies[i]));const auto army=armies[i+1];if(!evb_units(s,army,0)||!etd_companion(s,army))continue;
        if(contains_string(s,obs_attr(s,s.root,"besieged_strongholds",false),from)&&c9_garrison(s,t,from)==0)continue;
        std::vector<uint64_t> units;for(auto u:child_edges(s,obs_attr(s,army,"units")))if(ctx_unit_side(s,u)==0)units.push_back(u);
        const int64_t regular=etd_regulars(s,units),elite=static_cast<int64_t>(units.size())-regular;
        for(const auto& to:etd_reachable(s,t,from)){
            // whole movable pool of each route; covered = the (mid, frozen unit set) these rows represent
            std::set<etd_Key> covered;
            for(const auto& row:etd_route_rows(s,t,from,to,units,-1,-1,{})){covered.insert(etd_Key{row.mid,etd_freeze(s,row.pool)});rows.push_back(etd_option(s,from,to,-1,-1,row.via));}
            for(int64_t r=0;r<=regular;++r)for(int64_t e=0;e<=elite;++e)if(r+e>0)for(const auto& row:etd_route_rows(s,t,from,to,units,r,e,covered))rows.push_back(etd_option(s,from,to,r,e,row.via,row.named?&row.set:nullptr));
        }}return cp_container(s,WOTR_LIST,rows);
}
int64_t etd_move(Snapshot& s,const Snapshot& t,const std::string& from,const std::string& to,const std::string& via,int64_t regular,int64_t elite,bool allow_overstack,const am_FigureSet* unit_set=nullptr){
    return cdm_move_core(s,t,from,to,0,true,via,regular,elite,allow_overstack,missing,false,unit_set);
}

void etd_effect(Snapshot& s,const Snapshot& t,uint64_t choice,bool learned){
    const auto from=sanitize_name(text(s,field(s,choice,"from_region"))),to=sanitize_name(text(s,field(s,choice,"to_region")));const auto src=hr_army(s,from);
    if(!etd_companion(s,src))return;if(contains_string(s,obs_attr(s,s.root,"besieged_strongholds",false),from)&&c9_garrison(s,t,from)==0)return;
    if(!etd_reachable(s,t,from).count(to))return;std::vector<uint64_t> units;for(auto u:child_edges(s,obs_attr(s,src,"units")))if(ctx_unit_side(s,u)==0)units.push_back(u);
    const auto r=field(s,choice,"n_regular",false),e=field(s,choice,"n_elite",false);
    const auto nr=ctx_none(s,r)?-1:typed(s,r,WOTR_INT,"Through Day regular count invalid").integer,ne=ctx_none(s,e)?-1:typed(s,e,WOTR_INT,"Through Day elite count invalid").integer;
    // R2-E3-02: an explicit via_region (the FP player's route when routes
    // differ in consequence) must be one of the legal routes for this
    // composition - never re-routed (Python raises ValueError). R4-TDAAN-
    // SPLIT-UNIT-IDENTITY: a named unit_set moves exactly those units over a
    // route able to carry them (else refusal). Absent keys = old rows / old
    // callers: the deterministic etd_path default. ONE body: etd_row_route.
    const auto via=field(s,choice,"via_region",false),named=field(s,choice,"unit_set",false);
    am_FigureSet unit_set;const bool has_set=!ctx_none(s,named);if(has_set)unit_set=am_parse_figure_set(s,named,true);
    const bool has_via=!ctx_none(s,via);
    const auto path=etd_row_route(s,t,from,to,units,nr,ne,has_via,has_via?sanitize_name(text(s,via)):std::string(),has_set?&unit_set:nullptr);
    if(path.pool.empty())return;
    if(etd_move(s,t,from,to,path.mid,nr,ne,learned,has_set?&unit_set:nullptr)&&!path.mid.empty()){
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
    // N2-08: Python play_event auto-picks a lone option with '_from_pending', so the move may overstack (learned removal).
    }else{if(!choices.empty())etd_effect(s,t,choices.front(),true);er_commit(s,t,id,deck,face);}
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
