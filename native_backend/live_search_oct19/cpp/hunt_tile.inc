// Ordinary unmodified Hunt tile draw/apply foundations. These leave accumulated
// damage and reveal obligations to the learned decision/placement continuation.
int64_t ht_integer(const Snapshot& s,uint64_t object,const char* name,int64_t fallback=0){
    const auto value=obs_attr(s,object,name,false);return value==missing?fallback:typed(s,value,WOTR_INT,"Hunt scalar must be an integer").integer;
}
void ht_card_gate(const Snapshot& s){
    const auto pc=obs_attr(s,s.root,"persistent_cards");
    for(auto name:{"fp_cards","sh_cards"})for(auto card:child_edges(s,obs_attr(s,pc,name))){
        const auto kind=obs_attr(s,card,"card_type");
        require(!enum_is(s,kind,"game_state.PersistentCardType","WIZARD_STAFF") &&
                !enum_is(s,kind,"game_state.PersistentCardType","MITHRIL_COAT"),
                "Hunt tile requires Wizard Staff/Mithril decision continuation");
    }
}
int64_t ht_recycle(Snapshot& s,void* random,am_Builder& events,std::vector<uint64_t>& rows){
    const auto pool=obs_attr(s,s.root,"hunt_pool");typed(s,pool,WOTR_LIST,"Hunt Pool must be list");if(truth(s,pool))return 0;
    const auto discard=obs_attr(s,s.root,"hunt_discard");
    typed(s,discard,WOTR_LIST,"Hunt discard must be list");std::vector<uint64_t> back;
    for(auto tile:child_edges(s,discard)){
        attributes(s,tile,"game_state.HuntTile");
        // Rulebook p40: every empty-pool refill returns standard tiles only.
        // Mordor entry's separate Eye/special restoration is not a refill.
        if(truth(s,obs_attr(s,tile,"is_special",false)))continue;
        back.push_back(tile);auto live=child_edges(s,discard);
        const auto found=std::find_if(live.begin(),live.end(),[&](uint64_t other){return ft_tile_equal(s,other,tile);});
        require(found!=live.end(),"Hunt recycle tile missing from discard");live.erase(found);set_edges(s,discard,live);
    }
    if(!back.empty()){
        std::vector<uint64_t> permutation(back.size()),shuffled(back.size());
        if(hf_shuffle(random,permutation.data(),permutation.size()))throw std::invalid_argument(wotr_random_last_error());
        for(size_t i=0;i<back.size();++i)shuffled[i]=back[permutation[i]];
        auto live=child_edges(s,pool);live.insert(live.end(),shuffled.begin(),shuffled.end());set_edges(s,pool,live);
        rows.push_back(events.row(-1,"hunt_pool_recycled",{events.str("count"),events.integer(static_cast<int64_t>(back.size()))}));
    }
    return static_cast<int64_t>(back.size());
}
void ht_damage(Snapshot& s,int64_t damage,bool add_zero){
    const auto fs=obs_attr(s,s.root,"fellowship");
    if(damage<0){
        const auto corruption=ht_integer(s,fs,"corruption");
        require(corruption>=0,"Hunt corruption must be nonnegative");
        rr_set_attr(s,fs,"corruption",rr_scalar(s,WOTR_INT,damage < -corruption?0:corruption+damage));
    }else if(damage>0 || add_zero){
        const auto pending=ht_integer(s,s.root,"_pending_hunt_damage");
        require(pending<=std::numeric_limits<int64_t>::max()-damage,"Hunt damage accumulator overflow");
        rr_set_attr(s,s.root,"_pending_hunt_damage",rr_scalar(s,WOTR_INT,pending+damage));
    }
}
void ht_commit_reveal(Snapshot& s){
    rr_set_attr(s,s.root,"_hunt_reveal_pending",rr_scalar(s,WOTR_INT));
    const auto fs=obs_attr(s,s.root,"fellowship");const bool was_hidden=truth(s,obs_attr(s,fs,"hidden"));
    rr_set_attr(s,fs,"hidden",rr_scalar(s,WOTR_BOOL));
    if(was_hidden && !truth(s,obs_attr(s,fs,"mordor_entered"))){
        rr_set_attr(s,s.root,"_reveal_place_due",rr_scalar(s,WOTR_BOOL,1));
        rr_set_attr(s,s.root,"_sh_tile_drawn_regions",rr_scalar(s,WOTR_SET));
    }
}
bool ht_settle_reveal(Snapshot& s){
    const auto pending=ht_integer(s,s.root,"_hunt_reveal_pending");if(!pending)return false;
    if(ht_integer(s,s.root,"_pending_hunt_damage") || truth(s,obs_attr(s,s.root,"_hobbit_place_queue",false)))return false;
    for(const char* key:{"pending_hunt","pending_guide_change","pending_card_option"})
        if(!ctx_none(s,obs_attr(s,s.root,key,false)))return false;
    rr_set_attr(s,s.root,"_hunt_reveal_pending",rr_scalar(s,WOTR_INT));
    const auto guide=obs_attr(s,obs_attr(s,s.root,"fellowship"),"guide");
    if(pending==1 && !ctx_none(s,guide) && text_is(s,obs_attr(s,guide,"guide_ability",false),"no_reveal_on_damage"))return true;
    ht_commit_reveal(s);return true;
}
void ht_reveal(Snapshot& s,uint64_t tile,bool reveal,bool ordinary){
    (void)ordinary;
    if(!reveal)return;const auto fs=obs_attr(s,s.root,"fellowship");
    if(!truth(s,obs_attr(s,fs,"hidden")))return;
    const auto guide=obs_attr(s,fs,"guide");
    const bool unavoidable=truth(s,obs_attr(s,tile,"is_eye")) || truth(s,obs_attr(s,tile,"is_special",false));
    if(!ctx_none(s,guide) && text_is(s,obs_attr(s,guide,"guide_ability",false),"no_reveal_on_damage") &&
       !unavoidable)return;
    rr_set_attr(s,s.root,"_hunt_reveal_pending",rr_scalar(s,WOTR_INT,unavoidable?2:1));
}
struct ht_Tile {uint64_t tile=missing;int64_t damage=0;bool reveal=false,stop=false;};
ht_Tile ht_apply(Snapshot& s,const Snapshot& t,void* random,uint64_t tile,int64_t successes,bool mordor,bool track,bool eye_no_effect,am_Builder& events,std::vector<uint64_t>& rows){
    ht_Tile result;attributes(s,tile,"game_state.HuntTile");result.tile=tile;
    const bool eye=truth(s,obs_attr(s,tile,"is_eye"));const auto special=obs_attr(s,tile,"special_name");
    const bool printed_reveal=truth(s,obs_attr(s,tile,"reveal"));
    const auto damage=ht_integer(s,tile,"damage");
    const auto discard=obs_attr(s,s.root,"hunt_discard");typed(s,discard,WOTR_LIST,"Hunt discard must be list");
    auto append=[&](){auto used=child_edges(s,discard);used.push_back(tile);set_edges(s,discard,used);};
    if(!track && eye_no_effect && eye){append();return result;}
    if(track){
        result.damage=eye?static_cast<int64_t>(s.nodes[obs_attr(s,s.root,"hunt_box")].count):damage;
        result.reveal=eye || printed_reveal;result.stop=truth(s,obs_attr(s,tile,"stops_movement",false));
        if(!eye && truth(s,obs_attr(s,tile,"is_dice_roll",false))){
            uint32_t roll=0;hf_roll(random,&roll,1,"shelob d6 damage");result.damage=roll;
            const auto context=(special==missing?std::string("Shelob"):s.nodes[special].tag==WOTR_NONE?std::string("None"):std::string(text(s,special)))+" d6 damage";
            rows.push_back(events.row(-1,"dice_roll",{events.str("faces"),events.container(WOTR_LIST,{events.integer(roll)}),events.str("ctx"),events.str(context),events.str("side"),events.str("SH")}));
        }
        ht_damage(s,result.damage,false);ht_reveal(s,tile,result.reveal,false);ht_settle_reveal(s);append();return result;
    }
    result.damage=eye?(mordor?static_cast<int64_t>(s.nodes[obs_attr(s,s.root,"hunt_box")].count):successes):damage;
    ht_damage(s,result.damage,true);append();bool ignore=false;
    if(eye_no_effect){
        const auto fs=obs_attr(s,s.root,"fellowship");const auto destination=glb_fellowship_key(s,fs);const auto meta=ctx_get(t,field(t,t.root,"board_regions"),destination);
        ignore=!truth(s,obs_attr(s,fs,"hidden")) || (contains_string(t,field(t,t.root,"fp_region_keys"),destination) && meta!=missing &&
            (truth(t,ctx_get(t,meta,"is_city")) || truth(t,ctx_get(t,meta,"is_stronghold"))));
    }
    result.reveal=printed_reveal && !ignore;ht_reveal(s,tile,result.reveal,true);ht_settle_reveal(s);return result;
}
void ht_trace_draw(const Snapshot& s,uint64_t tile,std::string_view context,am_Builder& events,std::vector<uint64_t>& rows){
    const auto special=obs_attr(s,tile,"special_name");const bool eye=truth(s,obs_attr(s,tile,"is_eye"));
    const auto name=truth(s,special)?std::string(text(s,special)):eye?std::string("Eye"):std::to_string(ht_integer(s,tile,"damage"))+(truth(s,obs_attr(s,tile,"reveal"))?"R":"");
    rows.push_back(events.row(-1,"hunt_tile_draw",{events.str("tile_name"),events.str(name),events.str("ctx"),events.str(context)}));
}
ht_Tile ht_resolve(Snapshot& s,const Snapshot& t,void* random,int64_t successes,bool mordor,bool track,bool eye_no_effect,am_Builder& events,std::vector<uint64_t>& rows,bool skip_wizard=false){
    if(!skip_wizard && hpr_card(s,"WIZARD_STAFF")!=missing){
        const auto stash=track?hpr_dict(s,{rr_string(s,"mordor_track"),rr_scalar(s,WOTR_BOOL,1),rr_string(s,"skip_auto_sacrifice"),rr_scalar(s,WOTR_BOOL,1)}):
            hpr_dict(s,{rr_string(s,"successes"),rr_scalar(s,WOTR_INT,successes),rr_string(s,"mordor"),rr_scalar(s,WOTR_BOOL,mordor),rr_string(s,"eye_no_effect"),rr_scalar(s,WOTR_BOOL,eye_no_effect)});
        if(hpr_offer_wizard(s,stash))return {};
    }
    const auto pool=obs_attr(s,s.root,"hunt_pool");typed(s,pool,WOTR_LIST,"Hunt Pool must be list");if(!truth(s,pool))ht_recycle(s,random,events,rows);if(!truth(s,pool))return {};
    auto tiles=child_edges(s,pool);const auto index=hf_pool(s,pool,random,true);const auto tile=tiles[index];tiles.erase(tiles.begin()+index);set_edges(s,pool,tiles);
    ht_trace_draw(s,tile,track?"mordor hunt":eye_no_effect?"balrog extra draw":mordor?"mordor hunt":"hunt",events,rows);
    if(hpr_card(s,"MITHRIL_COAT")!=missing){const auto stash=track?hpr_dict(s,{rr_string(s,"tile"),tile,rr_string(s,"successes"),rr_scalar(s,WOTR_INT),rr_string(s,"mordor"),rr_scalar(s,WOTR_BOOL,1),rr_string(s,"mordor_track"),rr_scalar(s,WOTR_BOOL,1),rr_string(s,"skip_auto_sacrifice"),rr_scalar(s,WOTR_BOOL,1)}):
        hpr_dict(s,{rr_string(s,"tile"),tile,rr_string(s,"successes"),rr_scalar(s,WOTR_INT,successes),rr_string(s,"mordor"),rr_scalar(s,WOTR_BOOL,mordor),rr_string(s,"eye_no_effect"),rr_scalar(s,WOTR_BOOL,eye_no_effect)});
    if(hpr_offer_mithril(s,stash))return {};}
    return ht_apply(s,t,random,tile,successes,mordor,track,eye_no_effect,events,rows);
}
struct ht_Result {
    std::unique_ptr<Snapshot> state;
    std::unique_ptr<void,void(*)(void*)> random{nullptr,wotr_random_destroy};
    std::unique_ptr<Snapshot> events,value;
};
ht_Result native_hunt_tile(const Snapshot& source,const Snapshot& tables,const void* random_source,int64_t successes,bool mordor,bool track,bool eye_no_effect){
    // Legacy primitive contract rejects card windows; full funded actions
    // expose those choices through the protection action family.
    ht_card_gate(source);
    ht_Result result;result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());
    result.state=std::make_unique<Snapshot>(source);am_Builder b;std::vector<uint64_t> rows;
    const auto tile=ht_resolve(*result.state,tables,result.random.get(),successes,mordor,track,eye_no_effect,b,rows);
    b.out.root=b.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(b.out));
    if(track){am_Builder value;value.out.root=value.container(WOTR_DICT,{value.str("damage"),value.integer(tile.damage),value.str("reveal"),rr_scalar(value.out,WOTR_BOOL,tile.reveal),value.str("stops_movement"),rr_scalar(value.out,WOTR_BOOL,tile.stop)});result.value=std::make_unique<Snapshot>(std::move(value.out));}
    else{auto& s=*result.state;const auto game=s.root;s.root=tile.tile==missing?rr_scalar(s,WOTR_NONE):tile.tile;result.value=ac_compact(s);s.root=game;}
    return result;
}
