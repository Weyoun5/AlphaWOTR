// Staged compact search facts; existing metadata/actor APIs retain their roles.
std::string svf_special_normal(std::string_view input){
    std::string out;
    for(size_t i=0;i<input.size();++i){unsigned char c=input[i];
        if(c<128){out.push_back(c>='A'&&c<='Z'?char(c-'A'+'a'):char(c));continue;}
        // Canonical Smeagol spelling plus its capital/decomposed variants.
        if(c==0xc3&&i+1<input.size()&&(static_cast<unsigned char>(input[i+1])==0xa9||static_cast<unsigned char>(input[i+1])==0x89)){out+='e';++i;continue;}
        if(c==0xcc&&i+1<input.size()&&static_cast<unsigned char>(input[i+1])==0x81){++i;continue;}
        throw std::domain_error("Search Hunt diagnostic requires unported Unicode normalization");
    }return out;
}
std::string svf_tile_key(const Snapshot& s,uint64_t tile){
    auto name=obs_attr(s,tile,"special_name",false);
    if(truth(s,name)){
        typed(s,name,WOTR_STRING,"Hunt special name must be a string");auto normal=svf_special_normal(text(s,name));
        const std::pair<const char*,const char*> names[]={{"on on they went","oonw"},{"shelob's lair","shelob"},{"the ring is mine!","ringmine"},{"give it to uss!","giveuss"},{"smeagol helps nice master","smeagol"},{"elven cloaks","ecloak"},{"elven rope","erope"},{"phial of galadriel","phial"}};
        for(auto row:names)if(normal==row.first)return row.second;
        std::replace(normal.begin(),normal.end(),' ','_');return "special?"+normal;
    }
    if(truth(s,obs_attr(s,tile,"is_eye")))return "eye";
    return std::to_string(kc_integer(s,obs_attr(s,tile,"damage"),0))+(truth(s,obs_attr(s,tile,"reveal"))?"r":"");
}
std::unique_ptr<Snapshot> native_search_value_facts(const Snapshot& s){
    sm_Builder b(s);std::vector<uint64_t> presence;
    for(auto k:{"_ring_free_side","pending_army_move_die","pending_muster_die","pending_combat"})presence.push_back(b.boolean(!ctx_none(s,b.attr(k))));
    bool gate=false;auto pco=b.attr("pending_card_option");
    if(!ctx_none(s,pco)&&s.nodes[pco].tag==WOTR_DICT){
        gate=truth(s,ctx_get(s,pco,"hunt_reduction"));
        if(!gate){auto options=ctx_get(s,pco,"options");if(truth(s,options))for(auto o:child_edges(s,options))if(s.nodes[o].tag==WOTR_DICT){
            for(auto k:{"mithril_redraw","wizardstaff_prevent","balrog_declare_draw"})gate=gate||ctx_get(s,o,k)!=missing;
        }}
    }
    int64_t fp=0,sh=0;auto box=b.items(b.attr("hunt_box"));
    for(auto d:box){auto side=obs_attr(s,d,"side",false);if(!ctx_none(s,side)&&s.nodes[side].tag==WOTR_ENUM){auto es=child_edges(s,side);if(text_is(s,es[0],"game_state.Side")){fp+=text_is(s,es[1],"FREE_PEOPLES");sh+=text_is(s,es[1],"SHADOW");}}}
    auto bonus=kc_integer(s,b.attr("_crebain_hunt_bonus"),0);uint64_t pool=b.nil;
    try{
        auto tiles=b.attr("hunt_pool");require(tiles==missing||s.nodes[tiles].tag!=WOTR_NONE,"Hunt diagnostic pool is not iterable");
        std::vector<std::string> keys;for(auto tile:b.items(tiles))keys.push_back(svf_tile_key(s,tile));std::sort(keys.begin(),keys.end());
        std::vector<uint64_t> values;for(const auto& k:keys)values.push_back(b.str(k));pool=b.tup(values);
    }catch(const std::domain_error&){throw;}catch(const std::exception&){pool=b.nil;}
    b.out.root=b.container(WOTR_DICT,{b.str("pending_presence"),b.tup(presence),b.str("hunt_card_option_gate"),b.boolean(gate),b.str("hunt_box_counts"),b.tup({b.integer(fp),b.integer(sh)}),b.str("hunt_box_total"),b.integer(box.size()),b.str("crebain_initial"),b.integer(bonus),b.str("hunt_pool_keys"),pool});
    return ac_compact(b.out);
}
