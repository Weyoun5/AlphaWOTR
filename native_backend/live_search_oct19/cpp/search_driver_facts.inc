// Only ordered physical dice, hand contents, public discard and root commit slot.
// No whole-state view and no write-through from the returned owned projection.
std::unique_ptr<Snapshot> native_search_driver_facts(const Snapshot& s){
    sm_Builder b(s);std::vector<uint64_t> result;
    auto fields=[&](uint64_t object,std::initializer_list<const char*> names){
        std::vector<uint64_t> values;
        for(auto name:names){auto value=obs_attr(s,object,name,false);if(value!=missing){values.push_back(b.str(name));values.push_back(b.raw(value));}}
        return b.container(WOTR_DICT,values);
    };
    for(auto key:{"dice","hands","discard_face_up","_commit_slot"}){
        auto node=obs_attr(s,s.root,key,false);if(node==missing)continue;
        uint64_t projected;
        if((std::string_view(key)=="dice"||std::string_view(key)=="hands")&&s.nodes[node].tag==WOTR_DICT){
            std::vector<uint64_t> mapping;
            for(auto [side,value]:b.pairs(node)){
                mapping.push_back(b.raw(side));
                if(ctx_none(s,value)){mapping.push_back(b.raw(value));continue;}
                if(std::string_view(key)=="hands"){mapping.push_back(fields(value,{"hand"}));continue;}
                require(s.nodes[value].tag==WOTR_LIST||s.nodes[value].tag==WOTR_TUPLE,"Search dice rows require list or tuple");
                std::vector<uint64_t> rows;for(auto die:b.items(value))rows.push_back(fields(die,{"used","current"}));
                mapping.push_back(b.container(s.nodes[value].tag,rows));
            }
            projected=b.container(WOTR_DICT,mapping);
        }else projected=b.raw(node);
        result.push_back(b.str(key));result.push_back(projected);
    }
    std::vector<uint64_t> hidden;auto pc=b.attr("pending_combat");
    for(int side=0;side<2;++side){
        uint64_t cid=missing;
        if(truth(s,pc)){
            typed(s,pc,WOTR_DICT,"Hidden commitment requires a pending combat dictionary");
            auto candidate=b.get(pc,"atk_combat_card"),owner=b.get(pc,"attacker_side");
            if(truth(s,candidate)&&!enum_is(s,owner,"game_state.Side",side?"SHADOW":"FREE_PEOPLES")&&
                (truth(s,b.get(pc,"atk_hidden_cc"))||text_is(s,b.get(pc,"phase"),"defender_combat_card")))cid=candidate;
        }
        hidden.push_back(b.raw(cid));
    }
    result.push_back(b.str("hidden_commit"));result.push_back(b.tup(hidden));
    b.out.root=b.container(WOTR_DICT,result);return ac_compact(b.out);
}
