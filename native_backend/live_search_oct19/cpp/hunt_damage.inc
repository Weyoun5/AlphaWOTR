// Damage decision opening and chain bookkeeping shared by movement and choices.
bool hd_gollum_available(const Snapshot& s,const Snapshot& t){
    const auto fs=obs_attr(s,s.root,"fellowship"),guide=obs_attr(s,fs,"guide");
    if(ctx_none(s,guide) || !text_is(s,obs_attr(s,guide,"guide_ability",false),"no_reveal_on_damage") || !truth(s,obs_attr(s,fs,"hidden")))return false;
    if(ht_integer(s,s.root,"_hunt_reveal_pending")==2)return false;
    if(ht_integer(s,fs,"progress")>0 || truth(s,obs_attr(s,fs,"mordor_entered")))return true;
    const auto key=glb_fellowship_key(s,fs);const auto meta=ctx_get(t,field(t,t.root,"board_regions"),key);
    const bool settlement=meta!=missing && (truth(t,ctx_get(t,meta,"is_city")) || truth(t,ctx_get(t,meta,"is_stronghold")));
    return !(settlement && contains_string(t,field(t,t.root,"fp_region_keys"),key) &&
        !contains_string(s,ctx_side_value(s,obs_attr(s,s.root,"enemy_controlled"),1),key));
}
bool hd_sacrificable(const Snapshot& s){
    const auto fs=obs_attr(s,s.root,"fellowship");
    for(auto companion:child_edges(s,obs_attr(s,fs,"companions")))
        if(!text_is(s,obs_attr(s,companion,"name"),"Gollum") && ht_integer(s,companion,"level")>0)return true;
    return false;
}
void hd_corruption(Snapshot& s,int64_t damage){
    if(damage<=0)return;const auto fs=obs_attr(s,s.root,"fellowship");const auto corruption=ht_integer(s,fs,"corruption");
    require(corruption>=0,"Hunt corruption must be nonnegative");
    rr_set_attr(s,fs,"corruption",rr_scalar(s,WOTR_INT,corruption>=12 || damage>=12-corruption?12:corruption+damage));
}
void hd_pending(Snapshot& s,int64_t damage,const char* phase,int casualty=-1){
    std::vector<uint64_t> values{rr_string(s,"damage"),rr_scalar(s,WOTR_INT,damage),rr_string(s,"phase"),rr_string(s,phase)};
    if(casualty>=0){values.push_back(rr_string(s,"casualty_taken"));values.push_back(rr_scalar(s,WOTR_BOOL,casualty));}
    const auto pending=rr_scalar(s,WOTR_DICT);set_edges(s,pending,values);rr_set_attr(s,s.root,"pending_hunt",pending);
}
void hd_settle(Snapshot& s,const Snapshot& t){
    const auto damage=ht_integer(s,s.root,"_pending_hunt_damage");
    if(damage<=0 || !ctx_none(s,obs_attr(s,s.root,"pending_hunt")))return;
    const auto protection=obs_attr(s,s.root,"pending_card_option",false);
    if(!ctx_none(s,protection)){
        if(truth(s,field(s,protection,"hunt_reduction",false)))return;
        const auto options=field(s,protection,"options",false);
        if(!ctx_none(s,options))for(auto option:child_edges(s,options))
            for(const char* key:{"use_hunt_reduction","wizardstaff_prevent","mithril_redraw"})
                if(field(s,option,key,false)!=missing)return;
    }
    if(hd_gollum_available(s,t))hd_pending(s,damage,"hunt_gollum");
    else if(hd_sacrificable(s))hd_pending(s,damage,"hunt_sacrifice");
    else hd_corruption(s,damage);
    rr_set_attr(s,s.root,"_pending_hunt_damage",rr_scalar(s,WOTR_INT));
}
void hd_pop_hobbit(Snapshot& s){
    if(!ctx_none(s,obs_attr(s,s.root,"pending_hunt",false)))return;
    const auto queue=obs_attr(s,s.root,"_hobbit_place_queue",false);if(!truth(s,queue))return;
    typed(s,queue,WOTR_LIST,"Hobbit placement queue must be a list");auto entries=child_edges(s,queue);const auto hob=entries.front();entries.erase(entries.begin());set_edges(s,queue,entries);
    attributes(s,hob,"game_state.Companion");const auto level=ht_integer(s,hob,"level",1);
    const auto pending=rr_scalar(s,WOTR_DICT);set_edges(s,pending,{rr_string(s,"phase"),rr_string(s,"hobbit_place"),rr_string(s,"companion"),obs_attr(s,hob,"name"),rr_string(s,"level"),rr_scalar(s,WOTR_INT,level?level:1),rr_string(s,"companion_obj"),hob,rr_string(s,"damage"),rr_scalar(s,WOTR_INT)});
    rr_set_attr(s,s.root,"pending_hunt",pending);
}
void hd_finish(Snapshot& s,const Snapshot& t){
    rr_set_attr(s,s.root,"pending_hunt",rr_scalar(s,WOTR_NONE));
    if(ht_integer(s,s.root,"_pending_hunt_damage")>0)hd_settle(s,t);else hd_pop_hobbit(s);
}
void hd_continue(Snapshot& s,const Snapshot& t,int64_t damage,bool casualty){
    if(damage<=0){hd_finish(s,t);return;}
    const auto guide=obs_attr(s,obs_attr(s,s.root,"fellowship"),"guide");
    const bool hobbit=!ctx_none(s,guide) && text_is(s,obs_attr(s,guide,"guide_ability",false),"sacrifice_reduce_hunt");
    if(hd_gollum_available(s,t))hd_pending(s,damage,"hunt_gollum",casualty);
    else if((!casualty && hd_sacrificable(s)) || hobbit)hd_pending(s,damage,"hunt_sacrifice",casualty);
    else{hd_corruption(s,damage);hd_finish(s,t);}
}
std::unique_ptr<Snapshot> native_settle_hunt_damage(const Snapshot& source,const Snapshot& t){
    auto result=std::make_unique<Snapshot>(source);hd_settle(*result,t);return result;
}
std::unique_ptr<Snapshot> native_hunt_damage_actions(const Snapshot& s){
    const auto ph=obs_attr(s,s.root,"pending_hunt");require(!ctx_none(s,ph),"Hunt menu requires pending decision");
    const auto phase=field(s,ph,"phase");am_Builder b;std::vector<uint64_t> rows;
    auto row=[&](const char* kind){rows.push_back(b.row(-1,kind,{}));};
    if(text_is(s,phase,"hunt_gollum")){row("hunt_gollum_reveal");row("hunt_gollum_no_reveal");}
    else if(text_is(s,phase,"hunt_sacrifice")){
        row("hunt_take_corruption");const auto guide=obs_attr(s,obs_attr(s,s.root,"fellowship"),"guide");
        const bool hobbit=!ctx_none(s,guide) && text_is(s,obs_attr(s,guide,"guide_ability",false),"sacrifice_reduce_hunt");
        if(!truth(s,field(s,ph,"casualty_taken",false)) && hd_sacrificable(s)){if(!hobbit)row("hunt_sacrifice_guide");row("hunt_sacrifice_random");}
        if(hobbit)rows.push_back(b.row(-1,"hunt_guide_separate",{b.str("companion"),b.str(text(s,obs_attr(s,guide,"name")))}));
    }else throw std::invalid_argument("Hunt placement phase needs its own destination menu");
    b.out.root=b.container(WOTR_LIST,rows);return std::make_unique<Snapshot>(std::move(b.out));
}
