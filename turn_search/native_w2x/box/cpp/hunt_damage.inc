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
// R4-HOBBIT-QUEUE-CLOSURE (= game_logic.hobbit_place_tiles): the originating Hunt tile of each _hobbit_place_queue
// entry, stored in _hobbit_place_tiles. 0 = the Hunt damage chain being resolved; a positive value = a group of its own
// (a Lure hobbit, or a chain closed by hd_finish before the next tile's damage). Aligned on the queue's tail: an entry
// without a stored marker reads 0, and a marker whose entry was removed elsewhere (a queue reset) is dropped.
std::vector<int64_t> hc_place_tiles(const Snapshot& s){
    const auto queue=obs_attr(s,s.root,"_hobbit_place_queue",false);const size_t n=truth(s,queue)?child_edges(s,queue).size():0;
    std::vector<int64_t> tiles;const auto stored=obs_attr(s,s.root,"_hobbit_place_tiles",false);
    if(!ctx_none(s,stored))for(auto item:child_edges(s,stored))tiles.push_back(typed(s,item,WOTR_INT,"Hobbit placement tile marker must be an integer").integer);
    if(tiles.size()>n)tiles.erase(tiles.begin(),tiles.begin()+static_cast<ptrdiff_t>(tiles.size()-n));
    else if(tiles.size()<n)tiles.insert(tiles.begin(),n-tiles.size(),0);
    return tiles;
}
bool hc_place_tiles_stored(const Snapshot& s){return !ctx_none(s,obs_attr(s,s.root,"_hobbit_place_tiles",false));}
void hc_store_place_tiles(Snapshot& s,const std::vector<int64_t>& tiles){
    std::vector<uint64_t> items;for(auto tile:tiles)items.push_back(rr_scalar(s,WOTR_INT,tile));
    const auto list=rr_scalar(s,WOTR_LIST);set_edges(s,list,items);rr_set_attr(s,s.root,"_hobbit_place_tiles",list);
}
// = game_logic._close_hobbit_place_group: the chain that just closed keeps its own group (its still-queued hobbits get one
// fresh marker), so a hobbit of the next tile's chain never joins them.
void hc_close_place_tiles(Snapshot& s){
    auto tiles=hc_place_tiles(s);if(std::find(tiles.begin(),tiles.end(),0)==tiles.end())return;
    const auto fresh=*std::max_element(tiles.begin(),tiles.end())+1;
    for(auto& tile:tiles)if(tile==0)tile=fresh;
    hc_store_place_tiles(s,tiles);
}
void hd_pop_hobbit(Snapshot& s){
    if(!ctx_none(s,obs_attr(s,s.root,"pending_hunt",false)))return;
    const auto queue=obs_attr(s,s.root,"_hobbit_place_queue",false);if(!truth(s,queue))return;
    typed(s,queue,WOTR_LIST,"Hobbit placement queue must be a list");
    // R4-HOBBIT-QUEUE-CLOSURE (= game_logic.pop_hobbit_place_queue): group_lead marks a placement whose next queued hobbit
    // separated on the SAME Hunt tile (equal marker); only then does apply_hobbit_placement bind it to this destination.
    auto tiles=hc_place_tiles(s);const bool stored=hc_place_tiles_stored(s);
    auto entries=child_edges(s,queue);const auto hob=entries.front();entries.erase(entries.begin());set_edges(s,queue,entries);
    const auto tile=tiles.front();tiles.erase(tiles.begin());if(stored)hc_store_place_tiles(s,tiles);
    attributes(s,hob,"game_state.Companion");const auto level=ht_integer(s,hob,"level",1);
    const auto pending=rr_scalar(s,WOTR_DICT);
    std::vector<uint64_t> values{rr_string(s,"phase"),rr_string(s,"hobbit_place"),rr_string(s,"companion"),obs_attr(s,hob,"name"),rr_string(s,"level"),rr_scalar(s,WOTR_INT,level?level:1),rr_string(s,"companion_obj"),hob,rr_string(s,"damage"),rr_scalar(s,WOTR_INT)};
    if(!entries.empty() && tiles.front()==tile){values.push_back(rr_string(s,"group_lead"));values.push_back(rr_scalar(s,WOTR_BOOL,1));}
    set_edges(s,pending,values);
    rr_set_attr(s,s.root,"pending_hunt",pending);
}
void hd_settle(Snapshot& s,const Snapshot& t){
    const auto damage=ht_integer(s,s.root,"_pending_hunt_damage");
    if(!ctx_none(s,obs_attr(s,s.root,"pending_hunt")))return;
    const auto protection=obs_attr(s,s.root,"pending_card_option",false);
    if(!ctx_none(s,protection)){
        if(truth(s,field(s,protection,"hunt_reduction",false)))return;
        const auto options=field(s,protection,"options",false);
        if(!ctx_none(s,options))for(auto option:child_edges(s,options))
            for(const char* key:{"use_hunt_reduction","wizardstaff_prevent","mithril_redraw"})
                if(field(s,option,key,false)!=missing)return;
    }
    // R4-HOBBIT-QUEUE-CLOSURE (= game_logic._settle_hunt_damage): a settle that closes the damage WITHOUT a decision -
    // nothing left to confront (an Axe/Horn reduction took the last point) or direct corruption - drains the queue itself.
    if(damage<=0){hd_pop_hobbit(s);return;}
    if(hd_gollum_available(s,t))hd_pending(s,damage,"hunt_gollum");
    else if(hd_sacrificable(s))hd_pending(s,damage,"hunt_sacrifice");
    else{hd_corruption(s,damage);rr_set_attr(s,s.root,"_pending_hunt_damage",rr_scalar(s,WOTR_INT));hd_pop_hobbit(s);return;}
    rr_set_attr(s,s.root,"_pending_hunt_damage",rr_scalar(s,WOTR_INT));
}
void hd_finish(Snapshot& s,const Snapshot& t){
    rr_set_attr(s,s.root,"pending_hunt",rr_scalar(s,WOTR_NONE));
    // R4-HOBBIT-QUEUE-CLOSURE (= game_logic._finish_hunt_decision): the next tile's damage is confronted while this chain's
    // hobbits may still be queued - they keep their own group (hc_close_place_tiles).
    if(ht_integer(s,s.root,"_pending_hunt_damage")>0){hc_close_place_tiles(s);hd_settle(s,t);}
    // R3-HOBBIT-PLACE-STRAND (= game_logic._finish_hunt_decision): drain also after a DIRECT settle (no decision opened);
    // hd_pop_hobbit is a no-op when the settle opened one, and a protection choice that still owes the damage keeps the queue.
    if(ht_integer(s,s.root,"_pending_hunt_damage")<=0)hd_pop_hobbit(s);
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
