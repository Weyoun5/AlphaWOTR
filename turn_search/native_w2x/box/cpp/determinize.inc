// selfplay_actor_critic.search.determinize_state(state, gl, own_tag, sides, np.random.default_rng(seed), ai=ai) on a
// native state (claude_bugscan_2026-10-21/native_cl, 3a).  Same draws in the same order from the same generator
// (np_generator.inc): resample_committed_card's weighted choice, the opponent's per-deck hidden-card permutation, the
// own decks' permutations, the hunt pool's permutation.  Canonical orders as Python sorts them: card ids by str
// (code points), Hunt tiles by repr (HuntTile.__repr__), stably.
bool sd_movable(const Snapshot& s,uint64_t c,int deck){
    // search._movable: isinstance(card, str) and not card.startswith("_unknown_") and gl.card_deck_of(card) == deck_name
    if(s.nodes[c].tag!=WOTR_STRING)return false;
    if(text(s,c).rfind("_unknown_",0)==0)return false;
    return glb_card_deck(s,c)==deck;
}
std::string sd_str(const Snapshot& s,uint64_t c){
    if(s.nodes[c].tag!=WOTR_STRING)throw std::invalid_argument("determinize: a deck entry is not a string");
    return std::string(text(s,c));
}
// HuntTile.__repr__
std::string sd_tile_repr(const Snapshot& s,uint64_t tile){
    const auto special=obs_attr(s,tile,"special_name",false);
    if(special!=missing&&truth(s,special))return sd_str(s,special);
    if(truth(s,obs_attr(s,tile,"is_eye")))return "Eye";
    const auto& d=s.nodes[obs_attr(s,tile,"damage")];
    if(d.tag!=WOTR_INT)throw std::invalid_argument("determinize: Hunt tile damage is not an integer");
    return std::to_string(d.integer)+(truth(s,obs_attr(s,tile,"reveal"))?"R":"");
}
uint64_t sd_root_list(const Snapshot& s,const char* name){
    // getattr(state, name, None): missing or None -> missing
    const auto n=obs_attr(s,s.root,name,false);
    if(n==missing||ctx_none(s,n))return missing;
    if(s.nodes[n].tag!=WOTR_LIST)throw std::invalid_argument(std::string("determinize: ")+name+" is not a list");
    return n;
}
// search.resample_committed_card
void sd_resample_committed(Snapshot& s,const Snapshot& t,int own,np_Pcg64& g){
    const auto attrs=attributes(s,s.root,"game_state.GameState");
    const auto truth_card=glb_hidden_commit(s,attrs,own);
    if(truth_card==missing)return;
    const std::string true_id=sd_str(s,truth_card);
    const auto pending=obs_attr(s,s.root,"pending_combat");
    if(truth(s,field(s,pending,"atk_hidden_cc",false)))return;
    const auto mixture=so_commit_mixture(s,t);
    if(mixture.empty())return;
    std::vector<double> weights;for(const auto& m:mixture)weights.push_back(static_cast<double>(m.second));
    const std::string card=mixture[np_choice_weighted(g,weights)].first;
    if(card==true_id)return;
    const int opp=1-own;const char* tag=opp==0?"fp":"sh";
    std::vector<uint64_t> zones;
    const auto info=ctx_side_value(s,obs_attr(s,s.root,"hands"),opp);
    if(info!=missing&&!ctx_none(s,info)){
        zones.push_back(obs_attr(s,info,"hand"));
        const auto fd=obs_attr(s,info,"discarded_face_down",false);zones.push_back(fd==missing?missing:fd);
    }
    zones.push_back(obs_attr(s,s.root,(std::string(tag)+"_char_deck").c_str(),false));
    zones.push_back(obs_attr(s,s.root,(std::string(tag)+"_strat_deck").c_str(),false));
    uint64_t source=missing;size_t source_at=0;
    for(auto z:zones){
        if(z==missing||ctx_none(s,z))continue;
        const auto items=child_edges(s,z);bool hit=false;
        for(size_t i=0;i<items.size();++i)if(text_is(s,items[i],card)){source=z;source_at=i;hit=true;break;}
        if(hit)break;
    }
    uint64_t pile=missing;size_t pile_at=0;
    const auto discards=obs_attr(s,s.root,"discard_face_up",false);
    if(discards!=missing&&truth(s,discards)){
        const auto entries=child_edges(s,discards);std::vector<std::pair<std::string,uint64_t>> named;
        for(size_t i=0;i+1<entries.size();i+=2)named.push_back({sd_str(s,entries[i]),entries[i+1]});
        std::sort(named.begin(),named.end(),[](const auto& a,const auto& b){return a.first<b.first;});
        for(const auto& [name,list]:named){
            const auto items=child_edges(s,list);bool hit=false;
            for(size_t i=items.size();i-->0;)if(text_is(s,items[i],true_id)){pile=list;pile_at=i;hit=true;break;}
            if(hit)break;
        }
    }
    if(source==missing||pile==missing)throw std::runtime_error("Committed combat card cannot be exchanged consistently");
    {auto items=child_edges(s,source);items[source_at]=rr_string(s,true_id);set_edges(s,source,items);}
    {auto items=child_edges(s,pile);items[pile_at]=rr_string(s,card);set_edges(s,pile,items);}
    cp_put(s,pending,"atk_combat_card",rr_string(s,card));
}
void sd_permute(Snapshot& s,uint64_t list,const std::vector<uint64_t>& canonical,np_Pcg64& g){
    const auto order=np_permutation(g,canonical.size());std::vector<uint64_t> out(canonical.size());
    for(size_t i=0;i<order.size();++i)out[i]=canonical[order[i]];
    set_edges(s,list,out);
}
// search.determinize_state with a given generator (advanced in place)
void sd_determinize_with(Snapshot& s,const Snapshot& t,int own,np_Pcg64& g){
    sd_resample_committed(s,t,own,g);
    const int opp=1-own;const std::string otag=opp==0?"fp":"sh",mtag=own==0?"fp":"sh";
    const auto info=ctx_side_value(s,obs_attr(s,s.root,"hands"),opp);
    uint64_t hand=missing,face_down=missing;
    if(info!=missing&&!ctx_none(s,info)){
        hand=obs_attr(s,info,"hand");
        const auto fd=obs_attr(s,info,"discarded_face_down",false);
        if(fd!=missing&&s.nodes[fd].tag==WOTR_LIST)face_down=fd;     // isinstance(face_down, list) else []
    }
    const char* suffixes[2]={"_char_deck","_strat_deck"};
    for(int deck=0;deck<2;++deck){
        const auto pile=sd_root_list(s,(otag+suffixes[deck]).c_str());
        if(pile==missing)continue;
        struct Zone{uint64_t list;std::vector<size_t> slots;};std::vector<Zone> zones;
        std::vector<std::string> pool;
        for(auto z:{hand,pile,face_down}){
            Zone zone{z,{}};
            if(z!=missing){const auto items=child_edges(s,z);for(size_t i=0;i<items.size();++i)if(sd_movable(s,items[i],deck)){zone.slots.push_back(i);pool.push_back(std::string(text(s,items[i])));}}
            zones.push_back(std::move(zone));
        }
        std::sort(pool.begin(),pool.end());
        if(pool.size()<2)continue;
        const auto order=np_permutation(g,pool.size());size_t next=0;
        for(auto& zone:zones){
            if(zone.list==missing||zone.slots.empty())continue;
            auto items=child_edges(s,zone.list);
            for(auto slot:zone.slots)items[slot]=rr_string(s,pool[order[next++]]);
            set_edges(s,zone.list,items);
        }
    }
    for(int deck=0;deck<2;++deck){
        const auto pile=sd_root_list(s,(mtag+suffixes[deck]).c_str());
        if(pile==missing)continue;
        auto items=child_edges(s,pile);if(items.size()<=1)continue;
        std::vector<std::pair<std::string,uint64_t>> keyed;for(auto c:items)keyed.push_back({sd_str(s,c),c});
        std::stable_sort(keyed.begin(),keyed.end(),[](const auto& a,const auto& b){return a.first<b.first;});
        std::vector<uint64_t> canonical;for(auto& k:keyed)canonical.push_back(k.second);
        sd_permute(s,pile,canonical,g);
    }
    const auto pool=sd_root_list(s,"hunt_pool");
    if(pool!=missing){
        auto items=child_edges(s,pool);
        if(items.size()>1){
            std::vector<std::pair<std::string,uint64_t>> keyed;for(auto tile:items)keyed.push_back({sd_tile_repr(s,tile),tile});
            std::stable_sort(keyed.begin(),keyed.end(),[](const auto& a,const auto& b){return a.first<b.first;});
            std::vector<uint64_t> canonical;for(auto& k:keyed)canonical.push_back(k.second);
            sd_permute(s,pool,canonical,g);
        }
    }
}
// search.determinize_state(..., np.random.default_rng(seed), ...)
void sd_determinize(Snapshot& s,const Snapshot& t,int own,uint64_t seed){np_Pcg64 g=np_default_rng(seed);sd_determinize_with(s,t,own,g);}
