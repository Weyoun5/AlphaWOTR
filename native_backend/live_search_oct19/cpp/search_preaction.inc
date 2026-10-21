// Search preaction continuation, preserving actual declaration/guide/eye timing.
crr_Result sr_roll_current(const Snapshot& source,const void* rng){
    require(!truth(source,obs_attr(source,source.root,"dice_rolled")),"Current turn dice were already rolled");
    require(ctx_none(source,obs_attr(source,source.root,"pending_guide_change"))&&!truth(source,obs_attr(source,source.root,"_fellowship_tail_phase",false)),"Fellowship and Guide must finish before action rolls");
    auto prepared=std::make_unique<Snapshot>(source);rr_set_attr(*prepared,prepared->root,"_eye_allocation_phase",rr_scalar(*prepared,WOTR_BOOL,0));
    const auto initial=prepared->nodes[obs_attr(*prepared,prepared->root,"hunt_box")].count;
    auto fp=roll_action_dice(*prepared,rng,0);auto sh=roll_action_dice(*fp.state,fp.random.get(),1);
    crr_Result result;result.state=std::move(sh.state);result.random=std::move(sh.random);auto& s=*result.state;
    std::vector<uint64_t> sides;
    for(int side=0;side<2;++side){
        std::vector<std::string> faces;for(auto die:child_edges(s,pa_pool(s,side))){const auto face=obs_attr(s,die,"current");if(!ctx_none(s,face))faces.emplace_back(enum_member(s,face,"game_state.ActionDieFace"));}
        std::sort(faces.begin(),faces.end());std::vector<uint64_t> values;for(const auto& face:faces)values.push_back(rr_string(s,face));sides.push_back(cp_container(s,WOTR_TUPLE,values));
    }
    const auto final=s.nodes[obs_attr(s,s.root,"hunt_box")].count;sides.push_back(rr_scalar(s,WOTR_INT,final>initial?final-initial:0));
    rr_set_attr(s,s.root,"_search_roll_ms",cp_container(s,WOTR_TUPLE,sides));return result;
}
crr_Result sr_resume_preaction(const Snapshot& before,const Snapshot& after,const Snapshot& t,const void* rng,std::string_view kind,am_Builder* events=nullptr,std::vector<uint64_t>* rows=nullptr){
    crr_Result result;result.state=std::make_unique<Snapshot>(after);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());
    bool begin=truth(before,obs_attr(before,before.root,"_declare_phase",false))&&(kind=="fellowship_declare"||kind=="pass"||kind=="declare_decline");
    if(begin||truth(before,obs_attr(before,before.root,"_fellowship_tail_phase",false))){
        auto tail=fellowship_tail(*result.state,t,result.random.get(),begin);result.state=std::move(tail.state);result.random=std::move(tail.random);
        if(events&&rows&&tail.events){auto added=fm_import(events->out,*tail.events);rows->insert(rows->end(),added.begin(),added.end());}
    }else if(truth(before,obs_attr(before,before.root,"_eye_allocation_phase",false))&&kind=="eye_allocate")result=sr_roll_current(*result.state,result.random.get());
    const auto& s=*result.state;
    if(truth(s,obs_attr(s,s.root,"_eye_allocation_phase",false))&&ctx_none(s,obs_attr(s,s.root,"pending_guide_change"))){
        bool available=false;for(auto die:child_edges(s,pa_pool(s,1)))available|=!truth(s,obs_attr(s,die,"used"));
        if(!available)result=sr_roll_current(s,result.random.get());
    }
    return result;
}
