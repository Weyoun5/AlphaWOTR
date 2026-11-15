// Frozen ai.encode_state global block, offsets [3182,3401). Included inside
// snapshot.cpp's anonymous namespace after observation_support.inc.
double glb_num(const Snapshot& s,uint64_t id,double fallback=0.0){
    if(id==missing)return fallback;
    const auto& n=s.nodes[id];
    if(n.tag==WOTR_INT || n.tag==WOTR_BOOL)return static_cast<double>(n.integer);
    if(n.tag==WOTR_FLOAT)return n.real;
    throw std::invalid_argument("global observation requires a numeric value");
}
uint64_t glb_get(const Snapshot& s,uint64_t dict,std::string_view key,bool required=false){
    const auto edge=string_value_edge(s,dict,key);
    require(!required || edge!=missing,"missing global observation dictionary key");
    return edge==missing ? missing : s.edges[edge];
}
uint64_t glb_table(const Snapshot& tables,std::string_view key){return glb_get(tables,tables.root,key,true);}
std::vector<uint64_t> glb_items(const Snapshot& s,uint64_t id){
    if(id==missing)return {};
    const auto& n=s.nodes[id];
    require(n.tag==WOTR_LIST || n.tag==WOTR_TUPLE || n.tag==WOTR_SET || n.tag==WOTR_FROZENSET,
            "global observation requires a sequence");
    return child_edges(s,id);
}
size_t glb_len(const Snapshot& s,uint64_t id){
    if(id==missing)return 0;
    const auto& n=s.nodes[id];
    if(n.tag==WOTR_DICT)return static_cast<size_t>(n.count/2);
    if(n.tag==WOTR_LIST || n.tag==WOTR_TUPLE || n.tag==WOTR_SET || n.tag==WOTR_FROZENSET || n.tag==WOTR_BYTES)
        return static_cast<size_t>(n.count);
    if(n.tag==WOTR_STRING){
        const auto value=text(s,id);size_t count=0;
        for(unsigned char c:value)if((c&0xc0)!=0x80)++count;
        return count;
    }
    throw std::invalid_argument("global observation value has no length");
}
uint64_t glb_side_value(const Snapshot& s,uint64_t dict,int32_t side){
    const auto edge=enum_value_edge(s,dict,"game_state.Side",side==0 ? "FREE_PEOPLES" : "SHADOW");
    return edge==missing ? missing : s.edges[edge];
}
int glb_name_index(std::string_view name,const std::vector<std::string_view>& names){
    for(size_t i=0;i<names.size();++i)if(names[i]==name)return static_cast<int>(i);
    return -1;
}
std::vector<std::string_view> glb_names(const Snapshot& tables,std::string_view key){
    std::vector<std::string_view> result;
    for(auto id:glb_items(tables,glb_table(tables,key)))result.push_back(text(tables,id));
    return result;
}
std::string glb_fellowship_key(const Snapshot& s,uint64_t fellowship){
    const auto region=obs_attr(s,fellowship,"region");
    if(s.nodes[region].tag==WOTR_STRING)return sanitize_name(text(s,region));
    if(s.nodes[region].tag==WOTR_NONE)return "none";
    if(s.nodes[region].tag==WOTR_BOOL)return s.nodes[region].integer ? "true" : "false";
    if(s.nodes[region].tag==WOTR_INT)return sanitize_name(std::to_string(s.nodes[region].integer));
    if(s.nodes[region].tag==WOTR_OBJECT){
        for(const char* key:{"key","name","orig_name"}){
            const auto value=obs_attr(s,region,key,false);
            if(value!=missing)return sanitize_name(text(s,value));
        }
    }
    throw std::invalid_argument("unsupported strict fellowship region representation");
}
std::array<double,2> glb_live_vp(const Snapshot& s,const Snapshot& tables){
    const auto state=attributes(s,s.root,"game_state.GameState");
    const auto controlled=field(s,state,"enemy_controlled");
    const auto fp=glb_side_value(s,controlled,0),sh=glb_side_value(s,controlled,1);
    const auto lists=glb_items(tables,glb_table(tables,"vp_settlements"));
    require(lists.size()==2,"VP settlement table requires two factions");
    std::array<double,2> result{};
    for(size_t owner=0;owner<2;++owner){
        for(auto entry:glb_items(tables,lists[owner])){
            const auto pair=glb_items(tables,entry);require(pair.size()==2,"invalid VP settlement pair");
            if(contains_string(s,owner==0 ? sh : fp,text(tables,pair[0])))
                result[1-owner]+=glb_num(tables,pair[1]);
        }
    }
    return result;
}
int glb_card_index(const Snapshot& s,uint64_t cid,const Snapshot& tables,uint64_t indices){
    if(cid==missing)return -1;
    const auto& n=s.nodes[cid];
    require(n.tag!=WOTR_LIST && n.tag!=WOTR_DICT && n.tag!=WOTR_SET,"unhashable card identity");
    if(n.tag!=WOTR_STRING)return -1;
    const auto value=glb_get(tables,indices,text(s,cid));
    if(value==missing)return -1;
    const auto& index=typed(tables,value,WOTR_INT,"card index must be INT");
    require(index.integer>=0 && index.integer<96,"card index outside original embedding width");
    return static_cast<int>(index.integer);
}
int glb_card_deck(const Snapshot& s,uint64_t cid){
    if(s.nodes[cid].tag!=WOTR_STRING)return -1;
    const auto value=text(s,cid);
    if(value.find("_Character_")!=std::string_view::npos)return 0;
    if(value.find("_Strategy_")!=std::string_view::npos)return 1;
    if(value.size()>=3 && value.substr(0,3)=="HU_"){
        if(value.find("_CHAR_")!=std::string_view::npos)return 0;
        if(value.find("_STRAT_")!=std::string_view::npos)return 1;
    }
    return -1;
}
uint64_t glb_hidden_commit(const Snapshot& s,uint64_t state,int32_t side){
    const auto pc=field(s,state,"pending_combat",false);
    if(!truth(s,pc))return missing;
    const auto cid=glb_get(s,pc,"atk_combat_card");
    const auto attacker=glb_get(s,pc,"attacker_side");
    if(!truth(s,cid) || (attacker!=missing && enum_is(s,attacker,"game_state.Side",side==0 ? "FREE_PEOPLES" : "SHADOW")))
        return missing;
    const auto phase=glb_get(s,pc,"phase");
    if(truth(s,glb_get(s,pc,"atk_hidden_cc")) || (phase!=missing && text_is(s,phase,"defender_combat_card")))
        return cid;
    return missing;
}
bool glb_same_card(const Snapshot& s,uint64_t a,uint64_t b){
    if(a==missing || b==missing)return false;
    if(a==b)return true;
    return s.nodes[a].tag==WOTR_STRING && s.nodes[b].tag==WOTR_STRING && text(s,a)==text(s,b);
}
double glb_ring(const Snapshot& s,uint64_t count,uint64_t used){
    const auto& n=s.nodes[count];
    int64_t integer=0;
    if(n.tag==WOTR_INT || n.tag==WOTR_BOOL)integer=n.integer;
    else if(n.tag==WOTR_FLOAT){
        require(n.real==n.real && n.real>=static_cast<double>(std::numeric_limits<int64_t>::min()) &&
                n.real<9223372036854775808.0,"ring integer conversion outside range");
        integer=static_cast<int64_t>(n.real);
    }else throw std::invalid_argument("unsupported ring integer conversion");
    const double value=static_cast<double>(integer);
    return truth(s,used) ? -(value+1.0)/3.0 : value/3.0;
}

void append_globals(const Snapshot& s,const Snapshot& tables,int32_t side,std::vector<float>& out){
    require(side==0 || side==1,"invalid globals observation side");
    const size_t first=out.size();
    const auto state=attributes(s,s.root,"game_state.GameState");
    const auto get=[&](const char* name,bool required=true){return field(s,state,name,required);};
    const auto add=[&](double value){out.push_back(static_cast<float>(value));};
    const auto number=[&](const char* name,bool required=true){return glb_num(s,get(name,required));};
    const auto fellowship=get("fellowship");
    const auto fattr=[&](const char* name,bool required=true){return obs_attr(s,fellowship,name,required);};
    const auto company=glb_items(s,fattr("companions"));
    const auto guide=fattr("guide");
    const bool mordor=truth(s,fattr("mordor_entered"));
    double distance=0;
    if(!mordor){
        const auto maximum=glb_num(tables,glb_table(tables,"max_mordor_distance"));
        require(maximum!=0,"zero Mordor distance divisor");
        const auto location=glb_fellowship_key(s,fellowship);
        const auto value=glb_get(tables,glb_table(tables,"distance_to_mordor"),location);
        distance=glb_num(tables,value,maximum)/maximum;
    }
    add(glb_num(s,fattr("progress"))/12.0);add(glb_num(s,fattr("corruption"))/12.0);
    add(truth(s,fattr("hidden")) ? 1 : 0);add(mordor ? 1 : 0);
    add(glb_num(s,fattr("mordor_step"))/5.0);add(static_cast<double>(company.size())/8.0);
    add((truth(s,guide) ? glb_num(s,obs_attr(s,guide,"level")) : 0)/3.0);add(1.0-distance);
    const auto vp=glb_live_vp(s,tables);add(vp[side]/10.0);add(vp[1-side]/10.0);

    const auto pool=glb_items(s,get("hunt_pool"));
    double eyes=0,special_sh=0,special_fp=0,stops=0,damage=0,damage_count=0;
    for(auto tile:pool){
        const bool eye=truth(s,obs_attr(s,tile,"is_eye"));
        if(eye)eyes+=1;
        if(truth(s,obs_attr(s,tile,"is_special",false)) && !truth(s,obs_attr(s,tile,"is_fellowship_special",false)))special_sh+=1;
        if(truth(s,obs_attr(s,tile,"is_fellowship_special",false)))special_fp+=1;
        if(truth(s,obs_attr(s,tile,"stops_movement",false)))stops+=1;
        if(!eye){const auto d=glb_num(s,obs_attr(s,tile,"damage"));if(d>0){damage+=d;damage_count+=1;}}
    }
    double sh_in_box=0,fp_in_box=0;
    for(auto die:glb_items(s,get("hunt_box"))){
        const auto owner=obs_attr(s,die,"side");
        if(enum_is(s,owner,"game_state.Side","SHADOW"))sh_in_box+=1;
        if(enum_is(s,owner,"game_state.Side","FREE_PEOPLES"))fp_in_box+=1;
    }
    const double allocated=number("eyes_allocated",false);
    add(allocated/5.0);add(std::max(0.0,sh_in_box-allocated)/4.0);add(fp_in_box/5.0);
    add(static_cast<double>(pool.size())/16.0);add(eyes/4.0);add(special_sh/4.0);add(special_fp/4.0);add(stops/4.0);
    add((damage_count ? damage/damage_count : 0)/3.0);add(mordor ? 1 : 0);

    const auto faces=glb_names(tables,"action_die_face_names");require(faces.size()==7,"expected seven die faces");
    std::array<std::array<double,7>,2> hist{};std::array<double,2> unused{};
    const auto dice=get("dice");
    for(int owner=0;owner<2;++owner){
        for(auto die:glb_items(s,glb_side_value(s,dice,owner))){
            if(truth(s,obs_attr(s,die,"used")))continue;
            const auto face=obs_attr(s,die,"current");
            if(s.nodes[face].tag==WOTR_NONE)continue;
            unused[owner]+=1;
            const auto index=glb_name_index(enum_member(s,face,"game_state.ActionDieFace"),faces);
            require(index>=0,"unrecognized visible action-die face");hist[owner][index]+=1;
        }
    }
    add(unused[side]/10.0);add(unused[1-side]/10.0);
    for(int owner:{side,1-side})for(double count:hist[owner])add(count/4.0);

    const auto hands=get("hands"),my_info=glb_side_value(s,hands,side),enemy_info=glb_side_value(s,hands,1-side);
    const auto my_hand=glb_items(s,my_info==missing ? missing : obs_attr(s,my_info,"hand"));
    const auto enemy_hand=glb_items(s,enemy_info==missing ? missing : obs_attr(s,enemy_info,"hand"));
    std::array<double,2> enemy_decks{};
    for(auto cid:enemy_hand){const auto deck=glb_card_deck(s,cid);if(deck>=0)enemy_decks[deck]+=1;}
    double my_count=static_cast<double>(my_hand.size()),enemy_count=static_cast<double>(enemy_hand.size());
    const auto pending_hand=get("pending_combat",false);
    if(pending_hand!=missing&&s.nodes[pending_hand].tag==WOTR_DICT)for(int role=0;role<2;++role){
        const auto hidden_hand=field(s,pending_hand,role==0?"atk_hidden_cc":"def_hidden_cc",false);if(!truth(s,hidden_hand))continue;
        const auto owner=field(s,pending_hand,role==0?"attacker_side":"defender_side",false);if(owner==missing||s.nodes[owner].tag!=WOTR_ENUM)continue;
        if(enum_is(s,owner,"game_state.Side",side==0?"FREE_PEOPLES":"SHADOW"))my_count=std::max(0.0,my_count-1.0);
        else if(enum_is(s,owner,"game_state.Side",side==0?"SHADOW":"FREE_PEOPLES")){enemy_count=std::max(0.0,enemy_count-1.0);const auto deck=field(s,hidden_hand,"deck",false);if(text_is(s,deck,"character"))enemy_decks[0]=std::max(0.0,enemy_decks[0]-1.0);else if(text_is(s,deck,"strategy"))enemy_decks[1]=std::max(0.0,enemy_decks[1]-1.0);}
    }
    add(my_count/6.0);add(enemy_count/6.0);
    add(enemy_decks[0]/6.0);add(enemy_decks[1]/6.0);
    for(const char* name:{"fp_char_deck","fp_strat_deck","sh_char_deck","sh_strat_deck"}){
        const auto deck=get(name,false);add(truth(s,deck) ? static_cast<double>(glb_len(s,deck))/24.0 : 0);
    }
    const auto indices=glb_table(tables,"card_indices");
    for(size_t i=0;i<6;++i)add(i<my_hand.size() ? glb_card_index(s,my_hand[i],tables,indices) : -1);
    std::array<double,96> status{};
    auto hidden=glb_hidden_commit(s,state,side);
    const auto discards=get("discard_face_up",false);
    if(discards!=missing){
        const auto& dict=typed(s,discards,WOTR_DICT,"face-up discards must be dictionary");
        for(uint64_t i=1;i<dict.count;i+=2)for(auto cid:glb_items(s,s.edges[dict.offset+i])){
            if(hidden!=missing && glb_same_card(s,cid,hidden)){hidden=missing;continue;}
            const auto index=glb_card_index(s,cid,tables,indices);if(index>=0)status[index]=0.5;
        }
    }
    for(auto cid:glb_items(s,my_info==missing ? missing : obs_attr(s,my_info,"discarded_face_down"))){
        const auto index=glb_card_index(s,cid,tables,indices);if(index>=0)status[index]=0.5;
    }
    const auto persistent=get("persistent_cards");
    if(s.nodes[persistent].tag!=WOTR_NONE){
        for(const char* name:{"fp_cards","sh_cards"})for(auto card:glb_items(s,obs_attr(s,persistent,name,false))){
            const auto index=glb_card_index(s,obs_attr(s,card,"card_id",false),tables,indices);
            if(index>=0)status[index]=1;
        }
    }
    for(double value:status)add(value);

    const auto flag=[&](const char* character,const char* attribute){
        const auto object=get(character);return truth(s,object) && truth(s,obs_attr(s,object,attribute)) ? 1.0 : 0.0;
    };
    const std::array<double,5> characters={flag("gandalf_the_white","in_play"),flag("aragorn_crowned","crowned"),
        flag("witch_king","in_play"),flag("saruman","in_play"),flag("mouth_of_sauron","in_play")};
    if(side==0)for(double value:characters)add(value);
    else for(size_t i:{2,3,4,0,1})add(characters[i]);
    const std::array<double,3> fp_resources={glb_ring(s,get("fp_elven_rings"),get("fp_ring_used_this_turn")),
        truth(s,get("fp_dwarven_rings")) ? number("fp_dwarven_rings")/1.0 : 0,truth(s,get("fp_political_token")) ? 1.0 : 0.0};
    const std::array<double,3> sh_resources={glb_ring(s,get("sh_elven_rings"),get("sh_ring_used_this_turn")),
        truth(s,get("sh_dwarven_rings")) ? number("sh_dwarven_rings")/1.0 : 0,number("sh_action_tokens")/3.0};
    for(int owner:{side,1-side})for(double value:owner==0 ? fp_resources : sh_resources)add(value);
    add(number("turn")/15.0);
    const auto political=get("political"),boxes=obs_attr(s,political,"boxes");
    double fp_war=0,sh_war=0;
    const auto nations=glb_names(tables,"nation_names");require(nations.size()==8,"expected eight nation slots");
    for(auto nation:nations){
        const auto edge=enum_value_edge(s,boxes,"game_state.Nation",nation);
        require(edge!=missing,"missing political at-war nation");
        if(glb_num(s,s.edges[edge])>=3){if(fp_nation(nation))fp_war+=1;else sh_war+=1;}
    }
    add(fp_war/5.0);add(sh_war/3.0);

    const std::vector<std::string_view> companion_names={"Gandalf the Grey","Strider","Legolas","Gimli","Boromir","Meriadoc","Peregrin","Gollum","Aragorn","Gandalf the White"};
    std::array<bool,10> alive{},in_company{},first_level_seen{},separated{};
    std::array<double,10> company_levels{},separated_levels{};
    std::array<int,10> separated_regions;separated_regions.fill(-1);
    for(auto companion:company){
        const auto i=glb_name_index(text(s,obs_attr(s,companion,"name")),companion_names);if(i<0)continue;
        alive[i]=in_company[i]=true;
        if(!first_level_seen[i]){company_levels[i]=glb_num(s,obs_attr(s,companion,"level"))/3.0;first_level_seen[i]=true;}
    }
    const auto guide_name=obs_attr(s,guide,"name",false);
    if(guide_name!=missing && text_is(s,guide_name,"Gollum"))alive[7]=in_company[7]=true;
    const auto regions=glb_names(tables,"region_keys");
    const auto armies=get("armies");const auto& army_dict=typed(s,armies,WOTR_DICT,"army map must be dictionary");
    std::array<double,3> minions{};
    const std::vector<std::string_view> minion_names={"Witch King","Saruman","Mouth of Sauron"};
    std::array<std::array<double,2>,8> deployed{};
    for(uint64_t i=0;i<army_dict.count;i+=2){
        const auto raw_key=s.edges[army_dict.offset+i],army=s.edges[army_dict.offset+i+1];
        int region_index=-2;
        const auto locate=[&](){if(region_index==-2)region_index=glb_name_index(sanitize_name(text(s,raw_key)),regions);return region_index;};
        for(auto leader:glb_items(s,obs_attr(s,army,"leaders"))){
            const auto name=obs_attr(s,leader,"name",false);
            if(s.nodes[leader].tag==WOTR_OBJECT && object_type(s,leader)=="game_state.Companion"){
                const auto index=glb_name_index(text(s,obs_attr(s,leader,"name")),companion_names);
                if(index>=0){alive[index]=separated[index]=true;separated_levels[index]=glb_num(s,obs_attr(s,leader,"level"));separated_regions[index]=locate();}
            }
            if(name!=missing && s.nodes[name].tag==WOTR_STRING){
                const auto index=glb_name_index(text(s,name),minion_names);
                if(index>=0){const auto location=locate();if(location>=0)minions[index]=(location+1)/static_cast<double>(regions.size());}
            }
        }
        for(auto unit:glb_items(s,obs_attr(s,army,"units"))){
            const auto nation=enum_member(s,obs_attr(s,unit,"nation"),"game_state.Nation");
            const auto index=glb_name_index(nation,nations);require(index>=0,"deployed unit nation missing from table");
            const auto kind=obs_attr(s,unit,"kind");
            ++deployed[index][enum_is(s,kind,"game_state.UnitKind","REGULAR") ? 0 : 1];
        }
    }
    for(size_t i=0;i<10;++i){
        double level=company_levels[i];
        if(alive[i] && level==0 && separated[i])level=separated_levels[i]/3.0;
        const auto region=separated_regions[i];
        double location=region>=0 && !in_company[i] ? (region+1)/static_cast<double>(regions.size()) : 0;
        // Gollum cannot occupy the map; the existing zero location slot
        // carries the public reveal icon pending damage/Guide resolution.
        if(i==7 && truth(s,field(s,state,"_hunt_reveal_pending",false)))
            location=glb_num(s,field(s,state,"_hunt_reveal_pending",false))/2.0;
        add(alive[i] ? 1 : 0);add(in_company[i] ? 1 : 0);add(level);add(location);
    }
    for(double value:minions)add(value);
    const auto limits=glb_table(tables,"nation_max_figures");
    for(size_t i=0;i<nations.size();++i){
        const auto edge=enum_value_edge(tables,limits,"game_state.Nation",nations[i]);
        const auto maximum=edge==missing ? missing : tables.edges[edge];
        const auto regular=maximum==missing ? missing : glb_get(tables,maximum,"regular");
        const auto elite=maximum==missing ? missing : glb_get(tables,maximum,"elite");
        add(deployed[i][0]/std::max(1.0,glb_num(tables,regular,15)));
        add(deployed[i][1]/std::max(1.0,glb_num(tables,elite,5)));
    }
    require(out.size()-first==219,"global observation block width mismatch");
}
