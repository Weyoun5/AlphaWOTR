uint64_t cau_persistent(const Snapshot& s,std::string_view kind);
// Complete ordinary Fellowship movement, including the real pending damage
// and reveal surfaces. Native graph/RNG children are committed together.
uint64_t fm_payment_die(const Snapshot& s,int64_t index){
    const auto pool=pa_pool(s,0);const auto dice=child_edges(s,pool);
    require(index>=0 && static_cast<uint64_t>(index)<dice.size(),"Fellowship payment die index out of range");const auto die=dice[index];
    require(object_type(s,die)=="game_state.ActionDie" && enum_is(s,obs_attr(s,die,"side"),"game_state.Side","FREE_PEOPLES"),"Fellowship payment requires FP die");
    require(!truth(s,obs_attr(s,die,"used")),"Fellowship payment die already used");const auto face=obs_attr(s,die,"current");
    require(enum_is(s,face,"game_state.ActionDieFace","CHARACTER") || enum_is(s,face,"game_state.ActionDieFace","WILL_OF_WEST"),"Fellowship movement needs Character or Will die");
    return die;
}
bool fm_finish(Snapshot& s,const Snapshot& tables){
    if(ctx_none(s,obs_attr(s,s.root,"pending_combat",false)) && cau_persistent(s,"LAST_BATTLE")!=missing)cau_last_battle(s,tables);
    const auto pending=obs_attr(s,s.root,"_fellowship_move_die_pending",false);if(ctx_none(s,pending))return false;
    const auto index=typed(s,pending,WOTR_INT,"Pending Fellowship die index must be integer").integer;const auto dice=child_edges(s,pa_pool(s,0));
    require(index>=0 && static_cast<uint64_t>(index)<dice.size(),"Pending Fellowship movement die index invalid");const auto die=dice[index];
    require(enum_is(s,obs_attr(s,die,"side"),"game_state.Side","FREE_PEOPLES") && truth(s,obs_attr(s,die,"used")) && !ctx_none(s,obs_attr(s,die,"current")),"Pending Fellowship die must be consumed");
    const auto box=obs_attr(s,s.root,"hunt_box");auto entries=child_edges(s,box);require(std::find(entries.begin(),entries.end(),die)==entries.end(),"Pending Fellowship die already in Hunt Box");
    for(auto key:{"pending_combat","pending_hunt","pending_guide_change","pending_card_option","pending_card_followup","pending_discard","pending_army_move_die","pending_muster_die"})
        if(!ctx_none(s,obs_attr(s,s.root,key,false)))return false;
    for(auto key:{"_pending_hunt_damage","_hunt_reveal_pending","_reveal_place_due","_pending_reveal_strongholds","_fellowship_route_event","_hobbit_place_queue","_overstack_regions","_wwsat_owed","_mithril_stash","_wizardstaff_stash"})
        if(truth(s,obs_attr(s,s.root,key,false)))return false;
    const auto pc=obs_attr(s,s.root,"persistent_cards");
    bool add=true;for(auto name:{"fp_cards","sh_cards"})for(auto card:child_edges(s,obs_attr(s,pc,name)))
        add &= !enum_is(s,obs_attr(s,card,"card_type"),"game_state.PersistentCardType","LAST_BATTLE");
    if(add){entries.push_back(die);set_edges(s,box,entries);}rr_set_attr(s,s.root,"_fellowship_move_die_pending",rr_scalar(s,WOTR_NONE));return true;
}
struct fm_Result {
    std::unique_ptr<Snapshot> state;
    std::unique_ptr<void,void(*)(void*)> random{nullptr,wotr_random_destroy};
    std::unique_ptr<Snapshot> events;
};
fm_Result apply_fellowship_move(const Snapshot& source,const Snapshot& t,const void* random_source,int64_t index,bool crebain_resolved=false){
    // Hunt lookup reads both armies at a siege. At the public action boundary,
    // reconcile the admitted siege/control and overflow context like Python.
    quiet_action_gate(source,t,false,false,false,false,false,false,false,false,true,false,false,true);ring_waiver(source,0);ring_waiver(source,1);fm_payment_die(source,index);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained) || ctx_side(source,retained)==0,"Fellowship action belongs to retained ring actor");
    const auto fellowship=obs_attr(source,source.root,"fellowship");require(truth(source,obs_attr(source,fellowship,"hidden")),"Revealed Fellowship must hide before moving");
    const bool mordor=truth(source,obs_attr(source,fellowship,"mordor_entered"));
    require(mordor?ht_integer(source,fellowship,"mordor_step")<5:ht_integer(source,fellowship,"progress")<12,"Fellowship already at movement limit");
    fm_Result result;result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());
    result.state=action_clone(source);auto& s=*result.state;am_Builder b;std::vector<uint64_t> rows;
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    if(!crebain_resolved && ec_open(s,index)){
        // The original move has not paid a die yet. Preserve its inherited
        // battle-routing marker through the Shadow's pre-roll reaction.
        pa_common_upkeep(s,t);
        b.out.root=b.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(b.out));return result;
    }
    const auto die=fm_payment_die(s,index);rr_set_attr(s,die,"used",rr_scalar(s,WOTR_BOOL,1));
    rr_set_attr(s,s.root,"_fellowship_move_die_pending",rr_scalar(s,WOTR_INT,index));
    // Validate the physical obligation before drawing. There are no pending
    // consequences yet, so check identity explicitly without closing it.
    const auto box=child_edges(s,obs_attr(s,s.root,"hunt_box"));require(std::find(box.begin(),box.end(),die)==box.end(),"Fellowship payment die already in Hunt Box");
    rr_set_attr(s,s.root,"fellowship_acted_in_mordor",rr_scalar(s,WOTR_BOOL,1));const auto fs=obs_attr(s,s.root,"fellowship");
    if(mordor){
        const auto step=ht_integer(s,fs,"mordor_step");rr_set_attr(s,fs,"mordor_step",rr_scalar(s,WOTR_INT,step+1));
        const auto tile=ht_resolve(s,t,result.random.get(),0,true,true,false,b,rows);
        if(tile.stop)rr_set_attr(s,fs,"mordor_step",rr_scalar(s,WOTR_INT,step));
    }else{
        rr_set_attr(s,fs,"progress",rr_scalar(s,WOTR_INT,ht_integer(s,fs,"progress")+1));
        const auto params=hr_parameters(s,t);const auto hits=params.dice?hr_roll(s,t,result.random.get(),params.dice,6,params.fp,b,rows):0;
        if(hits)ht_resolve(s,t,result.random.get(),hits,false,false,false,b,rows);
    }
    cau_last_battle(s,t); // Source rechecks immediately after movement, before damage choices.
    if(ht_integer(s,s.root,"_pending_hunt_damage")>0 && !hpr_offer_reduction(s))hd_settle(s,t);
    pa_common_upkeep(s,t);
    ew_open(s,result.random.get());fr_open(s,t,result.random.get(),b,rows);hpr_return_mithril(s,result.random.get());fm_finish(s,t);
    rr_set_attr(s,s.root,"_battle_resolved_attacker",rr_scalar(s,WOTR_NONE));
    b.out.root=b.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(b.out));return result;
}
