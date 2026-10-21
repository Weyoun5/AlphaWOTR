// game_logic.roll_action_dice, frozen lines 202--214. This helper returns an
// unpublished state and advancing RNG clone; neither input is ever mutated.
// Requires wotr_random.h at translation-unit scope and existing graph helpers.
struct dr_Result {
    std::unique_ptr<Snapshot> state;
    std::unique_ptr<void,void(*)(void*)> random{nullptr,wotr_random_destroy};
    std::unique_ptr<Snapshot> faces;
};
dr_Result roll_action_dice(const Snapshot& source,const void* random_source,int32_t side){
    require(side==0 || side==1,"Action-dice pool side must be FP or SH");
    attributes(source,source.root,"game_state.GameState");
    const auto dice=obs_attr(source,source.root,"dice");
    const auto edge=enum_value_edge(source,dice,"game_state.Side",side==0?"FREE_PEOPLES":"SHADOW");
    require(edge!=missing,"Action-dice pool is missing");const auto pool=source.edges[edge];
    typed(source,pool,WOTR_LIST,"Action-dice pool must be a list");
    dr_Result result;
    result.random.reset(wotr_random_clone(random_source));
    if(!result.random)throw std::invalid_argument(wotr_random_last_error());
    result.state=std::make_unique<Snapshot>(source);auto& changed=*result.state;
    std::vector<uint64_t> rolled;
    rolled.reserve(source.nodes[pool].count);
    // Read the live pool on every iteration: aliases may put that same list
    // in the Hunt Box. An appended Eye die is then visited again and skipped
    // as used, just as Python's list iterator behaves.
    for(uint64_t index=0;index<changed.nodes[pool].count;++index){
        const auto die=changed.edges[changed.nodes[pool].offset+index];
        require(object_type(changed,die)=="game_state.ActionDie","Pool entry must be an ActionDie");
        if(truth(changed,obs_attr(changed,die,"used")))continue;
        const auto face_list=obs_attr(changed,die,"faces");
        require(changed.nodes[face_list].tag==WOTR_LIST || changed.nodes[face_list].tag==WOTR_TUPLE,"Die faces must be a list or tuple");
        const auto choices=child_edges(changed,face_list);
        require(!choices.empty(),"Cannot choose from an empty die-face sequence");
        for(auto face:choices){const auto name=enum_member(changed,face,"game_state.ActionDieFace");rr_face_index(name);}
        uint64_t selected=0;
        selected=hf_choice(result.random.get(),choices.size(),false);
        const auto face=choices[static_cast<size_t>(selected)];
        rr_set_attr(changed,die,"current",face);rolled.push_back(face);
        if(side==1 && enum_is(changed,face,"game_state.ActionDieFace","EYE")){
            rr_set_attr(changed,die,"used",rr_scalar(changed,WOTR_BOOL,1));
            const auto box=obs_attr(changed,changed.root,"hunt_box");
            typed(changed,box,WOTR_LIST,"Hunt Box must be a list");
            auto entries=child_edges(changed,box);entries.push_back(die);set_edges(changed,box,entries);
        }
    }
    rr_set_attr(changed,changed.root,"dice_rolled",rr_scalar(changed,WOTR_BOOL,1));
    const auto face_root=rr_scalar(changed,WOTR_LIST);set_edges(changed,face_root,rolled);
    const auto game_root=changed.root;changed.root=face_root;
    result.faces=ac_compact(changed);changed.root=game_root;
    result.state=ac_compact(changed);return result;
}
