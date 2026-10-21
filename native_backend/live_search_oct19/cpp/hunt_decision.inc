// Learned Hunt choices and casualty chains. Reuses combat's exact companion
// departure cleanup instead of creating a second WWSAT/card implementation.
void cbp_companion_death(Snapshot& s,void* rng,bool casualty);
bool ehd_pending(const Snapshot& s);
bool el_pending(const Snapshot& s);
bool ew_pending(const Snapshot& s);
void ew_debt_gate(const Snapshot& s);
bool epa_waiting(const Snapshot& s);
uint64_t epa_pending(const Snapshot& s,int side);
void fbr_put(Snapshot& s,uint64_t dict,const char* key,uint64_t value);
#include "funded_siege_hunt.inc"
#include "nazgul_siege_hunt.inc"
#include "hunt_pair_siege.inc"
std::set<std::string> cmp_reachable(const Snapshot& s,const Snapshot& t,std::string_view start,int64_t steps);
// Package U: a QUEUED Gandalf draw grant (_gandalf_draw_pending) never refuses a Hunt-chain action - the frozen executors
// run with it queued and their common tail drains it (ai.py:9901 _maybe_open_gandalf_draw, mirrored by cau_gandalf), so the
// grant's shape is validated here and every Hunt-chain executor below ends with cau_gandalf(s) after fm_finish(s,t).
// allow_worn_grant is retained for the callers' signature and no longer gates anything.
void cau_gandalf(Snapshot& s);
// Package Z (engine_hunt_reduction_window, 2026-09-11): the canonical Free Peoples hunt-reduction window (Axe and Bow / Horn of Gondor,
// game_logic._offer_hunt_reduction / hpr_offer_reduction) may be OPEN while a Hunt decision is resolved - the frozen reveal-route draw
// opens the sacrifice decision and the window together and the frozen hunt-decision executors run with it open (the window is decided
// later; served batch 34005 g3 d50: `Hunt action has pending protection choice` where the frozen executed hunt_sacrifice_random).
bool hc_reduction_window_open(const Snapshot& s){
    const auto p=obs_attr(s,s.root,"pending_card_option",false);if(ctx_none(s,p))return false;
    if(!truth(s,field(s,p,"hunt_reduction",false)) || !truth(s,field(s,p,"direct",false)))return false;
    if(!text_is(s,field(s,p,"side"),"fp") || !text_is(s,field(s,p,"decider"),"fp"))return false;
    const bool axe=text_is(s,field(s,p,"card_id"),"FREE_PEOPLES_Character_Axe_and_Bow") && text_is(s,field(s,p,"card_name"),"Axe and Bow");
    const bool horn=text_is(s,field(s,p,"card_id"),"FREE_PEOPLES_Character_Horn_of_Gondor") && text_is(s,field(s,p,"card_name"),"Horn of Gondor");
    if(!axe && !horn)return false;
    const auto options=child_edges(s,field(s,p,"options"));return options.size()==2;
}
void hc_gate(const Snapshot& s,bool guide=false,bool protection=false,bool allow_worn_grant=false,bool allow_stable_siege=false){
    attributes(s,s.root,"game_state.GameState");
    // Package W: only a pending battle competes (frozen get_legal_actions:9 precedes the hunt branch :18); discard / die-pair / follow-up keys are no refusal (frozen: none)
    require(ctx_none(s,obs_attr(s,s.root,"pending_combat",false)),"Hunt action has an unsupported competing obligation");
    const auto grant=obs_attr(s,s.root,"_gandalf_draw_pending",false);
    if(!ctx_none(s,grant)){
        // Package U: no refusal on a queued grant (frozen: none); its shape is still pinned below.
        require(typed(s,grant,WOTR_DICT,"Queued Gandalf grant must be dictionary").count==6,"Queued Gandalf grant shape invalid");
        text(s,field(s,grant,"source_card_id"));
        require(text_is(s,field(s,grant,"deck"),"character")||text_is(s,field(s,grant,"deck"),"strategy"),"Queued Gandalf deck invalid");
        require(text_is(s,field(s,grant,"die_face"),"EVENT")||text_is(s,field(s,grant,"die_face"),"WILL_OF_WEST"),"Queued Gandalf funding invalid");
    }
    if(!guide)require(ctx_none(s,obs_attr(s,s.root,"pending_guide_change")),"Guide choice precedes Hunt choice");
    if(!protection){
        // An EVENT-funded Hunt can store its Palantir bonus while damage or
        // Hobbit placement still has priority. Keep actual protection checks.
        if(epa_waiting(s))epa_pending(s,1);
        else if(hc_reduction_window_open(s)){}   // Package Z: the frozen hunt-decision executors run with the reduction window open (decided later)
        else require(ctx_none(s,obs_attr(s,s.root,"pending_card_option",false)),"Hunt action has pending protection choice");
        require(!truth(s,obs_attr(s,s.root,"_wizardstaff_stash",false)),"Hunt action has unresolved Wizard Staff draw");
        const auto stash=obs_attr(s,s.root,"_mithril_stash",false);require(!truth(s,stash) || truth(s,field(s,stash,"return_pending",false)),"Hunt action has unresolved Mithril tile choice");
    }
    ew_debt_gate(s);
    hpr_card_gate(s);
    (void)allow_stable_siege; // Package W: the stable-siege clause and the hand-limit clause are lifted (frozen: none)
    // Source deduplication preserves spelling and is a no-op for normalized
    // unique entries. Companion moves legitimately create original-name keys.
    const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));std::set<std::string> keys;
    for(size_t i=0;i<armies.size();i+=2){const auto key=sanitize_name(text(s,armies[i]));require(keys.insert(key).second,"Hunt action requires unique normalized army keys");}
}
void hc_remove(Snapshot& s,uint64_t companion){
    const auto fs=obs_attr(s,s.root,"fellowship"),list=obs_attr(s,fs,"companions");auto companions=child_edges(s,list);
    const auto found=std::find_if(companions.begin(),companions.end(),[&](uint64_t other){return ft_companion_equal(s,other,companion);});
    require(found!=companions.end(),"Hunt casualty companion absent");companions.erase(found);set_edges(s,list,companions);
}
bool hc_guide_member(const Snapshot& s){
    const auto fs=obs_attr(s,s.root,"fellowship"),guide=obs_attr(s,fs,"guide");
    for(auto companion:child_edges(s,obs_attr(s,fs,"companions")))if(ft_companion_equal(s,guide,companion))return true;return false;
}
void hc_queue_hobbit(Snapshot& s,uint64_t hob){
    auto queue=obs_attr(s,s.root,"_hobbit_place_queue",false);if(ctx_none(s,queue)){queue=rr_scalar(s,WOTR_LIST);rr_set_attr(s,s.root,"_hobbit_place_queue",queue);}
    auto entries=child_edges(s,queue);entries.push_back(hob);set_edges(s,queue,entries);
}
void hc_source_damage(Snapshot& s,void* rng,int64_t damage,uint64_t victim,bool guide){
    if(damage<=0)return;
    const auto fs=obs_attr(s,s.root,"fellowship");const auto name=text(s,obs_attr(s,victim,"name"));
    const bool hob=name=="Meriadoc" || name=="Peregrin";
    // A guide with the Hobbit separation ability is exposed as its separate
    // action, not this casualty row. Random Hobbit draws queue real placement.
    hc_remove(s,victim);cbp_companion_death(s,rng,!(guide && hob));
    require(!hob || truth(s,obs_attr(s,fs,"mordor_entered")),"Hobbit guide absorption requires placement action");
    ft_guide(s);
    hd_corruption(s,std::max<int64_t>(0,damage-ht_integer(s,victim,"level")));
}
void hc_inner(Snapshot& s,const Snapshot& t,void* rng,std::string_view kind,am_Builder& events,std::vector<uint64_t>& rows){
    const auto ph=obs_attr(s,s.root,"pending_hunt");require(truth(s,ph),"Hunt choice requires pending damage");
    const auto damage_node=field(s,ph,"damage");const auto damage=typed(s,damage_node,WOTR_INT,"Pending Hunt damage must be integer").integer;
    const auto fs=obs_attr(s,s.root,"fellowship");const bool casualty=truth(s,field(s,ph,"casualty_taken",false));
    if(kind=="hunt_take_corruption"){hd_corruption(s,damage);hd_finish(s,t);return;}
    if(kind=="hunt_gollum_reveal" || kind=="hunt_gollum_no_reveal"){
        const bool reveal=kind=="hunt_gollum_reveal";
        if(reveal)ht_commit_reveal(s);
        const auto remaining=reveal?std::max<int64_t>(0,damage-1):damage;
        if(remaining>0 && hd_sacrificable(s))hd_pending(s,remaining,"hunt_sacrifice",casualty);
        else{hd_corruption(s,remaining);hd_finish(s,t);}return;
    }
    if(kind=="hunt_sacrifice_guide"){
        const auto guide=obs_attr(s,fs,"guide");const auto absorbed=std::min(damage,ht_integer(s,guide,"level"));
        hc_source_damage(s,rng,absorbed,guide,true);hd_continue(s,t,damage-absorbed,true);return;
    }
    if(kind=="hunt_guide_separate"){
        const auto hob=obs_attr(s,fs,"guide");
        if(!ctx_none(s,hob)){
            if(hc_guide_member(s))hc_remove(s,hob);
            if(truth(s,obs_attr(s,fs,"mordor_entered"))){cbp_companion_death(s,rng,false);rows.push_back(events.row(-1,"random_draw",{events.str("ctx"),events.str("separation on Mordor track"),events.str("result"),events.str(std::string(text(s,obs_attr(s,hob,"name")))+" eliminated")}));}
            else hc_queue_hobbit(s,hob);
        }
        ft_guide(s);hd_continue(s,t,std::max<int64_t>(0,damage-1),casualty);return;
    }
    if(kind=="hunt_sacrifice_random"){
        std::vector<uint64_t> candidates;
        for(auto comp:child_edges(s,obs_attr(s,fs,"companions")))if(!text_is(s,obs_attr(s,comp,"name"),"Gollum") && ht_integer(s,comp,"level")>0)candidates.push_back(comp);
        if(candidates.empty()){hd_corruption(s,damage);hd_finish(s,t);return;}
        const auto chosen=hf_choice(rng,candidates.size(),true);const auto victim=candidates[chosen];
        const auto name=std::string(text(s,obs_attr(s,victim,"name")));std::vector<uint64_t> names;for(auto comp:candidates)names.push_back(events.str(text(s,obs_attr(s,comp,"name"))));
        rows.push_back(events.row(-1,"random_draw",{events.str("ctx"),events.str("hunt casualty draw"),events.str("result"),events.str(name),events.str("pool"),events.container(WOTR_LIST,names)}));
        const auto absorbed=std::min(damage,ht_integer(s,victim,"level"));
        if((name=="Meriadoc" || name=="Peregrin") && !truth(s,obs_attr(s,fs,"mordor_entered"))){
            hc_remove(s,victim);cbp_companion_death(s,rng,true);ft_guide(s);
            // Source remove_hobbit_for_placement always clamps even zero net.
            const auto corruption=ht_integer(s,fs,"corruption");rr_set_attr(s,fs,"corruption",rr_scalar(s,WOTR_INT,std::min<int64_t>(12,corruption+std::max<int64_t>(0,absorbed-ht_integer(s,victim,"level")))));
            hc_queue_hobbit(s,victim);
        }else hc_source_damage(s,rng,absorbed,victim,false);
        hd_continue(s,t,damage-absorbed,true);return;
    }
    throw std::invalid_argument("Hunt action kind not implemented");
}
fm_Result apply_hunt_choice(const Snapshot& source,const Snapshot& t,const void* random_source,std::string_view kind){
    const bool staff_kind=kind=="hunt_take_corruption" || kind=="hunt_sacrifice_guide" || kind=="hunt_sacrifice_random";
    hc_gate(source,false,false,false,true);auto menu=native_hunt_damage_actions(source);bool legal=false;
    for(auto row:child_edges(*menu,menu->root)){const auto values=child_edges(*menu,row);legal|=text_is(*menu,values[1],kind);}require(legal,"Hunt action is not among current legal choices");
    fm_Result result;result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());result.state=action_clone(source);auto& s=*result.state;am_Builder b;std::vector<uint64_t> rows;
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    if(truth(s,obs_attr(s,s.root,"_declare_phase",false)))rr_set_attr(s,s.root,"_declare_phase",rr_scalar(s,WOTR_BOOL));
    hc_inner(s,t,result.random.get(),kind,b,rows);epa_maybe(s);ew_open(s,result.random.get());fr_open(s,t,result.random.get(),b,rows);
    if(!ctx_none(s,obs_attr(s,s.root,"pending_guide_change")) || hc_guide_member(s) || !truth(s,obs_attr(s,obs_attr(s,s.root,"fellowship"),"companions"))){}else ft_guide(s);
    hpr_return_mithril(s,result.random.get());fm_finish(s,t);epa_maybe(s);cau_gandalf(s);b.out.root=b.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(b.out));return result;
}
fm_Result apply_hunt_guide_choice(const Snapshot& source,const Snapshot& t,const void* random_source,std::string_view name){
    hc_gate(source,true,ehd_pending(source) || ew_pending(source) || epa_waiting(source),false,true);const auto pending=obs_attr(source,source.root,"pending_guide_change");
    require(!ctx_none(source,pending) && contains_string(source,field(source,pending,"candidates"),name),"Hunt Guide choice not among pending candidates");
    fm_Result result;result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());result.state=action_clone(source);auto& s=*result.state;am_Builder b;std::vector<uint64_t> rows;
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    if(truth(s,obs_attr(s,s.root,"_declare_phase",false)))rr_set_attr(s,s.root,"_declare_phase",rr_scalar(s,WOTR_BOOL));
    const auto fs=obs_attr(s,s.root,"fellowship");const auto comps=child_edges(s,obs_attr(s,fs,"companions"));
    if(comps.empty())ft_guide(s);
    else{
        int64_t highest=std::numeric_limits<int64_t>::min();for(auto comp:comps)highest=std::max(highest,ht_integer(s,comp,"level"));
        uint64_t selected=missing;for(auto comp:comps)if(ht_integer(s,comp,"level")==highest && text_is(s,obs_attr(s,comp,"name"),name)){selected=comp;break;}
        require(selected!=missing,"Chosen Hunt Guide is not highest level");rr_set_attr(s,fs,"guide",selected);
    }
    rr_set_attr(s,s.root,"pending_guide_change",rr_scalar(s,WOTR_NONE));epa_maybe(s);ew_open(s,result.random.get());fr_open(s,t,result.random.get(),b,rows);hpr_return_mithril(s,result.random.get());fm_finish(s,t);epa_maybe(s);cau_gandalf(s);
    b.out.root=b.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(b.out));return result;
}
bool hp_shadow_controlled_stronghold(const Snapshot& s,const Snapshot& t,std::string_view key){
    const auto meta=ctx_get(t,field(t,t.root,"board_regions"),key);if(meta==missing || !truth(t,ctx_get(t,meta,"is_stronghold")))return false;
    const auto control=obs_attr(s,s.root,"enemy_controlled");
    if(contains_string(s,ctx_side_value(s,control,1),key))return true;
    return contains_string(t,field(t,t.root,"shadow_region_keys"),key) && !contains_string(s,ctx_side_value(s,control,0),key);
}
std::vector<std::string> hp_destinations(const Snapshot& s,const Snapshot& t){
    const auto ph=obs_attr(s,s.root,"pending_hunt");require(text_is(s,field(s,ph,"phase"),"hobbit_place"),"Hobbit placement pending required");
    const auto bound=field(s,ph,"group_destination",false);
    if(!ctx_none(s,bound))return {std::string(text(s,bound))};
    const auto fs=obs_attr(s,s.root,"fellowship");const auto start=glb_fellowship_key(s,fs);const auto level_node=field(s,ph,"level",false);
    const auto level=ctx_none(s,level_node)?1:typed(s,level_node,WOTR_INT,"Hobbit placement level must be integer").integer;
    const auto steps=ht_integer(s,fs,"progress")+(level?level:1);
    // Frozen ai._get_hunt_decision_actions routes hobbit_place through ai._companion_reachable
    // (movement = progress + level): a besieged Free Peoples stronghold can be neither entered nor
    // left, a Shadow-controlled stronghold stops movement, and placing in place is always legal.
    // Reuse the proved companion reachability instead of a siege-blind walk (package G).
    std::set<std::string> reached=steps>0?cmp_reachable(s,t,start,steps):std::set<std::string>{};reached.insert(start);
    return {reached.begin(),reached.end()};
}
std::unique_ptr<Snapshot> native_hobbit_place_actions(const Snapshot& s,const Snapshot& t){
    const auto pending=obs_attr(s,s.root,"pending_hunt");const auto name=text(s,field(s,pending,"companion"));am_Builder b;std::vector<uint64_t> rows;
    for(const auto& key:hp_destinations(s,t))rows.push_back(b.row(-1,"companion_separate",{b.str("companion"),b.str(name),b.str("to_region"),b.str(key)}));
    b.out.root=b.container(WOTR_LIST,rows);return std::make_unique<Snapshot>(std::move(b.out));
}
void hp_place(Snapshot& s,const Snapshot& t,uint64_t companion,std::string_view key,bool activate=true){
    auto army=mu_army(s,key);
    if(army==missing){
        const auto board=obs_attr(s,s.root,"board"),region=ctx_get(s,obs_attr(s,board,"_regions"),key);require(region!=missing && text_is(s,obs_attr(s,region,"key"),key),"Hobbit placement requires canonical Board Region");
        const auto units=rr_scalar(s,WOTR_LIST),leaders=rr_scalar(s,WOTR_LIST);army=mu_object(s,"game_state.Army",{{"region",region},{"units",units},{"leaders",leaders}});
        const auto armies=obs_attr(s,s.root,"armies");auto entries=child_edges(s,armies);entries.push_back(rr_string(s,key));entries.push_back(army);set_edges(s,armies,entries);
    }
    const auto leaders=obs_attr(s,army,"leaders");auto figures=child_edges(s,leaders);figures.push_back(companion);set_edges(s,leaders,figures);
    if(!activate)return;
    const auto meta=ctx_get(t,field(t,t.root,"board_regions"),key),nation=ctx_get(t,field(t,t.root,"movement_region_nations"),key);
    if(meta==missing || ctx_none(t,nation) || !fp_nation(ctx_nation(t,nation)) ||
       !(truth(t,ctx_get(t,meta,"is_city")) || truth(t,ctx_get(t,meta,"is_stronghold"))) || contains_string(s,ctx_side_value(s,obs_attr(s,s.root,"enemy_controlled"),1),key))return;
    // Arrival belongs to this figure, not to pre-existing destination figures.
    for(auto figure:{companion})if(object_type(s,figure)=="game_state.Companion"){
        const auto allowed=ctx_get(t,field(t,t.root,"companion_nations"),text(s,obs_attr(s,figure,"name")));bool activates=allowed!=missing && t.nodes[allowed].tag==WOTR_NONE;
        if(allowed!=missing && t.nodes[allowed].tag!=WOTR_NONE)for(auto item:child_edges(t,allowed))activates|=ctx_nation(t,item)==ctx_nation(t,nation);
        if(activates){wt_activate_if_inactive(s,ctx_nation(t,nation),"companion_move",key);return;}
    }
}
fm_Result apply_hobbit_placement(const Snapshot& source,const Snapshot& t,const void* random_source,std::string_view destination){
    hc_gate(source,false,false,false,true);const auto key=sanitize_name(destination);const auto legal=hp_destinations(source,t);
    require(std::find(legal.begin(),legal.end(),key)!=legal.end(),"Hobbit placement destination unreachable");
    fm_Result result;result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());result.state=action_clone(source);auto& s=*result.state;am_Builder b;std::vector<uint64_t> rows;
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    if(truth(s,obs_attr(s,s.root,"_declare_phase",false)))rr_set_attr(s,s.root,"_declare_phase",rr_scalar(s,WOTR_BOOL));
    const auto pending=obs_attr(s,s.root,"pending_hunt"),companion=field(s,pending,"companion_obj",false);
    // Reveal and route-draw drivers drain this queue before the next tile.
    // Retain individual public descriptors while binding this tile's group.
    const bool same_tile=truth(s,obs_attr(s,s.root,"_hobbit_place_queue",false));
    require(!ctx_none(s,companion),"Hobbit placement requires physical pending companion");hp_place(s,t,companion,key);hd_finish(s,t);
    const auto next=obs_attr(s,s.root,"pending_hunt",false);
    if(same_tile&&!ctx_none(s,next)&&text_is(s,field(s,next,"phase",false),"hobbit_place"))fbr_put(s,next,"group_destination",rr_string(s,key));
    epa_maybe(s);ew_open(s,result.random.get());fr_open(s,t,result.random.get(),b,rows);hpr_return_mithril(s,result.random.get());fm_finish(s,t);epa_maybe(s);cau_gandalf(s);b.out.root=b.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(b.out));return result;
}
std::unique_ptr<Snapshot> native_reveal_actions(const Snapshot& s,const Snapshot& t){
    am_Builder b;std::vector<uint64_t> rows;for(const auto& key:fr_destinations(s,t))rows.push_back(b.row(-1,"fellowship_reveal_place",{b.str("region"),b.str(key)}));
    b.out.root=b.container(WOTR_LIST,rows);return std::make_unique<Snapshot>(std::move(b.out));
}
fm_Result apply_fellowship_reveal(const Snapshot& source,const Snapshot& t,const void* random_source,std::string_view destination){
    hc_gate(source,false,false,false,true);require(!fr_pending(source) && truth(source,obs_attr(source,source.root,"_reveal_place_due",false)),"Fellowship reveal placement is not due or damage is pending");
    const auto key=sanitize_name(destination);const auto legal=fr_destinations(source,t);require(std::find(legal.begin(),legal.end(),key)!=legal.end(),"Reveal destination not in legal menu");
    fm_Result result;result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());result.state=action_clone(source);auto& s=*result.state;am_Builder b;std::vector<uint64_t> rows;
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    if(truth(s,obs_attr(s,s.root,"_declare_phase",false)))rr_set_attr(s,s.root,"_declare_phase",rr_scalar(s,WOTR_BOOL));
    fr_place(s,t,result.random.get(),key,b,rows);
    // A route draw can already have opened protection. Await that choice.
    if(ht_integer(s,s.root,"_pending_hunt_damage")>0
       && ctx_none(s,obs_attr(s,s.root,"pending_hunt",false))
       && ctx_none(s,obs_attr(s,s.root,"pending_card_option",false))
       && !hpr_offer_reduction(s))hd_settle(s,t);
    epa_maybe(s);ew_open(s,result.random.get());fr_open(s,t,result.random.get(),b,rows);hpr_return_mithril(s,result.random.get());fm_finish(s,t);epa_maybe(s);cau_gandalf(s);
    b.out.root=b.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(b.out));return result;
}
