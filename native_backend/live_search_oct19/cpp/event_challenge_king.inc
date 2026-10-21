// Challenge uses random.choice + value removal, unlike ordinary Hunt pop.
bool eck_supported(const Snapshot& t,uint64_t card){return card!=missing&&text_is(t,obs_attr(t,card,"card_name"),"Challenge of the King");}
void eck_effect(Snapshot& s,const Snapshot& t,void* rng,am_Builder& events,std::vector<uint64_t>& annotations){
    const auto pool=obs_attr(s,s.root,"hunt_pool");std::vector<uint64_t> drawn;
    for(int i=0;i<3;++i){
        if(!truth(s,pool))ht_recycle(s,rng,events,annotations);
        auto live=child_edges(s,pool);if(live.empty())continue;
        const auto chosen=hf_pool(s,pool,rng,false);
        const auto tile=live[chosen];const auto found=std::find_if(live.begin(),live.end(),[&](uint64_t other){return ft_tile_equal(s,other,tile);});
        require(found!=live.end(),"Challenge selected tile missing from pool");live.erase(found);set_edges(s,pool,live);drawn.push_back(tile);
    }
    const auto eyes=std::count_if(drawn.begin(),drawn.end(),[&](uint64_t tile){return truth(s,obs_attr(s,tile,"is_eye",false));});
    if(eyes==3){
        const auto fs=obs_attr(s,s.root,"fellowship"),members=obs_attr(s,fs,"companions");uint64_t dead=missing;
        auto named=[&](uint64_t figure){const auto name=text(s,obs_attr(s,figure,"name"));return name.find("Strider")!=std::string_view::npos||name.find("Aragorn")!=std::string_view::npos;};
        for(auto comp:child_edges(s,members))if(named(comp)){auto live=child_edges(s,members);const auto found=std::find_if(live.begin(),live.end(),[&](uint64_t other){return ft_companion_equal(s,other,comp);});require(found!=live.end(),"Challenge companion missing");live.erase(found);set_edges(s,members,live);dead=comp;ft_guide(s);break;}
        if(dead==missing){
            const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));
            for(size_t i=0;i<armies.size()&&dead==missing;i+=2){const auto army=armies[i+1],leaders=obs_attr(s,army,"leaders");
                for(auto leader:child_edges(s,leaders))if(object_type(s,leader)=="game_state.Companion"&&named(leader)){
                    auto live=child_edges(s,leaders);const auto found=std::find_if(live.begin(),live.end(),[&](uint64_t other){return ft_companion_equal(s,other,leader);});require(found!=live.end(),"Challenge board companion missing");live.erase(found);set_edges(s,leaders,live);dead=leader;
                    if(!truth(s,obs_attr(s,army,"units"))&&live.empty())am_erase(s,text(s,armies[i]));break;
                }
            }am_orphans(s);
        }
        if(dead!=missing&&object_type(s,dead)=="game_state.Companion")cbp_companion_death(s,rng,false);
        auto live=child_edges(s,pool);live.insert(live.end(),drawn.begin(),drawn.end());set_edges(s,pool,live);
    }else{
        const auto removed=obs_attr(s,s.root,"hunt_removed");auto pool_items=child_edges(s,pool),removed_items=child_edges(s,removed);
        for(auto tile:drawn)(truth(s,obs_attr(s,tile,"is_eye",false))?removed_items:pool_items).push_back(tile);
        set_edges(s,pool,pool_items);set_edges(s,removed,removed_items);
    }
    const auto live=child_edges(s,pool);std::vector<uint64_t> permutation(live.size()),shuffled(live.size());
    if(hf_shuffle(rng,permutation.data(),permutation.size()))throw std::invalid_argument(wotr_random_last_error());
    for(size_t i=0;i<live.size();++i)shuffled[i]=live[permutation[i]];set_edges(s,pool,shuffled);
}
void apply_challenge_event_inplace(Snapshot& s,const Snapshot& t,void* rng,int side,int64_t index,std::string_view id,bool free_ents,am_Builder& ef_events,std::vector<uint64_t>& ef_annotations){
    const auto& source=s;ef_validate_free(source,t,side,index,id,free_ents);
    const int ef_marker=ctx_side(source,obs_attr(source,source.root,"_battle_resolved_attacker",false));const auto ef_unused=ef_unused_dice(source);

    require(side==0,"Challenge belongs to FP");quiet_action_gate(source,t,false,false,false,false,false,false,false,false,true,false,(false||free_ents),true);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==0,"Challenge violates retained Ring actor");
    const auto card=ev_card(t,id);require(eck_supported(t,card)&&contains_string(source,tc_hand(source,0),id),"Challenge card absent");require(ev_requirement(source,t,card),"Challenge requirement false");
    const auto die=free_ents?missing:ev_payment(source,t,0,index,card);const std::string face(free_ents?std::string_view{}:enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace"));
    
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));if(!free_ents)rr_set_attr(s,ev_payment(s,t,0,index,card),"used",rr_scalar(s,WOTR_BOOL,1));
    eck_effect(s,t,rng,ef_events,ef_annotations);
    er_commit(s,t,id,"character",face,true); // Shared optional already-consumed flag; source WWSAT can remove this card first.
    cau_finish(s,t,rng,free_ents?ef_final_marker(s,ef_marker,ef_unused):-1,ef_events,ef_annotations);
}
fm_Result apply_challenge_event(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index,std::string_view id){
    require(side==0,"Challenge belongs to FP");quiet_action_gate(source,t,false,false,false,false,false,false,false,false,true,false,false,true);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==0,"Challenge violates retained Ring actor");
    const auto card=ev_card(t,id);require(eck_supported(t,card)&&contains_string(source,tc_hand(source,0),id),"Challenge card absent");require(ev_requirement(source,t,card),"Challenge requirement false");
    const auto die=ev_payment(source,t,0,index,card);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace"));
    
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());
    am_Builder ef_events;std::vector<uint64_t> ef_annotations;
    apply_challenge_event_inplace(*result.state,t,result.random.get(),side,index,id,false,ef_events,ef_annotations);
    ef_events.out.root=ef_events.container(WOTR_LIST,ef_annotations);result.events=std::make_unique<Snapshot>(std::move(ef_events.out));return result;
}
