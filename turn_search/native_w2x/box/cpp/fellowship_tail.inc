// Frozen game_logic begin/resume_fellowship_tail and positional Mordor entry.
// Direct driver primitive: resident graph clone, explicit cloned RNG. Guide
// ties remain pending learned decisions. Earlier effects remain pending.
struct ft_Result {
    std::unique_ptr<Snapshot> state;
    std::unique_ptr<void,void(*)(void*)> random{nullptr,wotr_random_destroy};
    std::unique_ptr<Snapshot> events;
    bool finished=false;
};
void fbr_stationary(Snapshot& s,const Snapshot& t,void* rng,am_Builder& events,std::vector<uint64_t>& rows);
bool ft_pending(const Snapshot& s){
    for(auto key:{"pending_combat","pending_hunt","pending_guide_change","pending_card_option","pending_card_followup","pending_discard"})
        if(!ctx_none(s,obs_attr(s,s.root,key,false)))return true;
    for(auto key:{"_pending_hunt_damage","_hunt_reveal_pending","_reveal_place_due","_mithril_stash","_hobbit_place_queue","_pending_reveal_strongholds","_fellowship_route_event"})
        if(truth(s,obs_attr(s,s.root,key,false)))return true;
    return false;
}
bool ft_equal(const Snapshot& s,uint64_t a,uint64_t b,unsigned depth=0){
    if(a==b)return true;
    require(depth<32,"Fellowship value equality nesting exceeds support");
    const auto& x=s.nodes[a];const auto& y=s.nodes[b];
    const bool xn=x.tag==WOTR_INT || x.tag==WOTR_BOOL || x.tag==WOTR_FLOAT;
    const bool yn=y.tag==WOTR_INT || y.tag==WOTR_BOOL || y.tag==WOTR_FLOAT;
    if(xn && yn){
        if(x.tag!=WOTR_FLOAT && y.tag!=WOTR_FLOAT)return x.integer==y.integer;
        const long double xv=x.tag==WOTR_FLOAT?static_cast<long double>(x.real):static_cast<long double>(x.integer);
        const long double yv=y.tag==WOTR_FLOAT?static_cast<long double>(y.real):static_cast<long double>(y.integer);
        return xv==yv;
    }
    if(x.tag!=y.tag)return false;
    if(x.tag==WOTR_NONE)return true;
    if(x.tag==WOTR_STRING || x.tag==WOTR_BYTES || x.tag==WOTR_BIGINT)return text(s,a)==text(s,b);
    require(x.tag==WOTR_LIST || x.tag==WOTR_TUPLE || x.tag==WOTR_ENUM,
            "Unsupported value in Fellowship dataclass comparison");
    if(x.count!=y.count)return false;
    for(uint64_t i=0;i<x.count;++i)if(!ft_equal(s,s.edges[x.offset+i],s.edges[y.offset+i],depth+1))return false;
    return true;
}
bool ft_companion_equal(const Snapshot& s,uint64_t a,uint64_t b){
    if(a==b)return true;
    if(ctx_none(s,a) || object_type(s,a)!="game_state.Companion")return false;
    for(auto key:{"name","level","leadership","captain_of_the_west","guide_ability","activates_nations","activation_regions","can_muster_nation","muster_regions","bonus_movement","solo_hobbit_movement"})
        if(!ft_equal(s,obs_attr(s,a,key),obs_attr(s,b,key)))return false;
    return true;
}
bool ft_tile_equal(const Snapshot& s,uint64_t a,uint64_t b){
    if(a==b)return true;
    if(object_type(s,a)!="game_state.HuntTile" || object_type(s,b)!="game_state.HuntTile")return false;
    for(auto key:{"damage","is_eye","reveal","is_special","special_name","stops_movement","is_dice_roll","is_fellowship_special","image_file"})
        if(!ft_equal(s,obs_attr(s,a,key),obs_attr(s,b,key)))return false;
    return true;
}
void ft_guide(Snapshot& s){
    const auto fs=obs_attr(s,s.root,"fellowship"),list=obs_attr(s,fs,"companions");
    typed(s,list,WOTR_LIST,"Guide selection requires a companion list");const auto companions=child_edges(s,list);
    if(companions.empty()){
        const auto name=rr_string(s,"Gollum"),zero=rr_scalar(s,WOTR_INT),flag=rr_scalar(s,WOTR_BOOL),none=rr_scalar(s,WOTR_NONE);
        const auto nations=rr_scalar(s,WOTR_LIST),regions=rr_scalar(s,WOTR_LIST),muster=rr_scalar(s,WOTR_LIST),ability=rr_string(s,"no_reveal_on_damage");
        const auto guide=mu_object(s,"game_state.Companion",{{"name",name},{"level",zero},{"leadership",zero},{"captain_of_the_west",flag},{"guide_ability",ability},{"activates_nations",nations},{"activation_regions",regions},{"can_muster_nation",none},{"muster_regions",muster},{"bonus_movement",zero},{"solo_hobbit_movement",flag}});
        rr_set_attr(s,fs,"guide",guide);return;
    }
    int64_t highest=std::numeric_limits<int64_t>::min();std::vector<uint64_t> best;
    for(auto companion:companions){
        attributes(s,companion,"game_state.Companion");
        const auto level=typed(s,obs_attr(s,companion,"level"),WOTR_INT,"Companion level must be integer").integer;
        if(level>highest){highest=level;best.clear();}if(level==highest)best.push_back(companion);
    }
    if(best.size()==1){rr_set_attr(s,fs,"guide",best.front());return;}
    std::vector<uint64_t> names;for(auto companion:best)names.push_back(obs_attr(s,companion,"name"));
    const auto candidates=rr_scalar(s,WOTR_LIST);set_edges(s,candidates,names);
    const auto pending=rr_scalar(s,WOTR_DICT);set_edges(s,pending,{rr_string(s,"candidates"),candidates});
    rr_set_attr(s,s.root,"pending_guide_change",pending);
    const auto guide=obs_attr(s,fs,"guide");bool present=false;
    for(auto companion:best)present|=ft_companion_equal(s,guide,companion);
    if(!present)rr_set_attr(s,fs,"guide",best.front());
}
void ft_refill(Snapshot& s,void* random){
    const auto pool=obs_attr(s,s.root,"hunt_pool"),discard=obs_attr(s,s.root,"hunt_discard");
    typed(s,pool,WOTR_LIST,"Mordor Hunt Pool must be a list");typed(s,discard,WOTR_LIST,"Hunt discard must be a list");
    for(auto tile:child_edges(s,discard)){
        attributes(s,tile,"game_state.HuntTile");
        if(!truth(s,obs_attr(s,tile,"is_eye")) && !truth(s,obs_attr(s,tile,"is_special",false)))continue;
        auto live_pool=child_edges(s,pool);live_pool.push_back(tile);set_edges(s,pool,live_pool);
        auto live_discard=child_edges(s,discard);
        const auto found=std::find_if(live_discard.begin(),live_discard.end(),[&](uint64_t other){return ft_tile_equal(s,other,tile);});
        require(found!=live_discard.end(),"Returned Hunt tile is missing from discard");live_discard.erase(found);set_edges(s,discard,live_discard);
    }
    const auto before=child_edges(s,pool);std::vector<uint64_t> permutation(before.size()),after(before.size());
    if(hf_shuffle(random,permutation.data(),permutation.size()))throw std::invalid_argument(wotr_random_last_error());
    for(size_t i=0;i<before.size();++i)after[i]=before[permutation[i]];set_edges(s,pool,after);
}
ft_Result fellowship_tail(const Snapshot& source,const Snapshot& tables,const void* random_source,bool begin){
    attributes(source,source.root,"game_state.GameState");ft_Result result;
    result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());
    result.state=std::make_unique<Snapshot>(source);auto& s=*result.state;am_Builder events;std::vector<uint64_t> rows;
    auto finish=[&]()->ft_Result{events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return std::move(result);};
    if(begin){
        require(!truth(s,obs_attr(s,s.root,"dice_rolled")),"Fellowship tail must precede action dice rolls");
        rr_set_attr(s,s.root,"_declare_phase",rr_scalar(s,WOTR_BOOL));
        if(truth(s,obs_attr(s,s.root,"_eye_allocation_phase",false))){result.finished=true;return finish();}
        if(!truth(s,obs_attr(s,s.root,"_fellowship_tail_phase",false))){fbr_stationary(s,tables,result.random.get(),events,rows);rr_set_attr(s,s.root,"_fellowship_tail_phase",rr_scalar(s,WOTR_BOOL,1));}
    }
    if(!truth(s,obs_attr(s,s.root,"_fellowship_tail_phase",false))){result.finished=true;return finish();}
    require(!truth(s,obs_attr(s,s.root,"dice_rolled")),"Fellowship tail must precede action dice rolls");
    if(ft_pending(s))return finish();
    if(terminal_should_end(check_victory(s,tables),false)){
        rr_set_attr(s,s.root,"_fellowship_tail_phase",rr_scalar(s,WOTR_BOOL));result.finished=true;return finish();
    }
    const auto fs=obs_attr(s,s.root,"fellowship");
    if(!truth(s,obs_attr(s,fs,"mordor_entered"))){
        const auto key=glb_fellowship_key(s,fs);
        if(key=="morannon" || key=="minas morgul"){
            rr_set_attr(s,fs,"mordor_entered",rr_scalar(s,WOTR_BOOL,1));
            rr_set_attr(s,fs,"mordor_step",rr_scalar(s,WOTR_INT));rr_set_attr(s,fs,"progress",rr_scalar(s,WOTR_INT));
            ft_refill(s,result.random.get());
        }
    }
    rr_set_attr(s,s.root,"_fellowship_tail_phase",rr_scalar(s,WOTR_BOOL));
    if(!truth(s,obs_attr(s,s.root,"_eye_allocation_phase",false))){ft_guide(s);rr_set_attr(s,s.root,"_eye_allocation_phase",rr_scalar(s,WOTR_BOOL,1));}
    result.finished=true;return finish();
}
