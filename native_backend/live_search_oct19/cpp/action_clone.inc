// Ordinary ai.clone_state semantics, frozen static_r3 ai.py:747--1039.
// Included after shared graph/obs_attr helpers. No Python callbacks.
// Includes the paused _round_resume deepcopy branch with Board-only memo reuse.
struct ac_Builder {
    const Snapshot& source;
    Snapshot work;
    std::unordered_map<std::string,uint64_t> strings;
    uint64_t none, zero, no, yes;
    using Fields=std::vector<std::pair<std::string,uint64_t>>;
    explicit ac_Builder(const Snapshot& s):source(s){
        // Reserve before copying, avoiding copy-then-reallocate of each input
        // vector when fresh clone objects are appended. Only newly emitted
        // field/type names are interned: scanning and allocating an index of
        // every board/card string costs more than retaining a few equal-valued
        // STRING nodes. String identity is not part of the graph scalar API.
        work.nodes.reserve(s.nodes.size()*2+128);work.nodes.assign(s.nodes.begin(),s.nodes.end());
        work.edges.reserve(s.edges.size()*2+256);work.edges.assign(s.edges.begin(),s.edges.end());
        work.bytes.reserve(s.bytes.size()+4096);work.bytes.assign(s.bytes.begin(),s.bytes.end());
        work.root=s.root;strings.reserve(128);
        none=scalar(WOTR_NONE);zero=scalar(WOTR_INT);no=scalar(WOTR_BOOL);yes=scalar(WOTR_BOOL,1);
    }
    uint64_t scalar(uint32_t tag,int64_t value=0){auto n=work.nodes.size();work.nodes.push_back({tag,0,value,0,0,0});return n;}
    uint64_t str(std::string_view value){
        std::string own(value);auto found=strings.find(own);if(found!=strings.end())return found->second;
        auto offset=work.bytes.size();work.bytes.insert(work.bytes.end(),own.begin(),own.end());
        auto n=work.nodes.size();work.nodes.push_back({WOTR_STRING,0,0,0,offset,own.size()});strings.emplace(std::move(own),n);return n;
    }
    uint64_t container(uint32_t tag,const std::vector<uint64_t>& values={}){
        auto offset=work.edges.size();work.edges.insert(work.edges.end(),values.begin(),values.end());
        auto n=work.nodes.size();work.nodes.push_back({tag,0,0,0,offset,values.size()});return n;
    }
    template<class Iterator>
    uint64_t object_fields(std::string_view type,Iterator first,Iterator last,size_t count,bool slots){
        // Unit/Army/ActionDie construction is frequent. Emit into the already
        // reserved arena instead of allocating several tiny vectors per piece.
        const auto type_node=str(type),dict_name=slots?missing:str("__dict__");
        const auto offset=work.edges.size();
        if(slots)work.edges.push_back(type_node);
        for(auto it=first;it!=last;++it){const auto key=str(it->first);work.edges.push_back(key);work.edges.push_back(it->second);}
        auto result=work.nodes.size();
        work.nodes.push_back({slots?WOTR_OBJECT:WOTR_DICT,0,0,0,offset,count*2+(slots?1:0)});
        if(slots)return result;
        const auto object_offset=work.edges.size();
        work.edges.push_back(type_node);work.edges.push_back(dict_name);work.edges.push_back(result);
        result=work.nodes.size();work.nodes.push_back({WOTR_OBJECT,0,0,0,object_offset,3});return result;
    }
    uint64_t object(std::string_view type,const Fields& fields,bool slots=false){
        return object_fields(type,fields.begin(),fields.end(),fields.size(),slots);
    }
    uint64_t object(std::string_view type,std::initializer_list<std::pair<std::string_view,uint64_t>> fields,bool slots=false){
        return object_fields(type,fields.begin(),fields.end(),fields.size(),slots);
    }
    uint64_t attr(uint64_t object,const char* name,uint64_t absent=missing)const{
        auto n=obs_attr(source,object,name,false);
        if(n==missing){require(absent!=missing,"action clone missing required source attribute");return absent;}
        return n;
    }
    bool present(uint64_t n)const{return n!=missing && source.nodes[n].tag!=WOTR_NONE;}
    bool nonempty(uint64_t n)const{return n!=missing && truth(source,n);}
    uint64_t get(uint64_t dict,const char* name)const{
        if(dict==missing || source.nodes[dict].tag==WOTR_NONE)return missing;
        return field(source,dict,name,false);
    }
    static void put(Fields& fields,const std::string& key,uint64_t value){
        for(auto& p:fields)if(p.first==key){p.second=value;return;}
        fields.emplace_back(key,value);
    }
    std::vector<uint64_t> iterable(uint64_t n)const{
        require(n!=missing,"action clone missing iterable");auto tag=source.nodes[n].tag;
        require(tag==WOTR_LIST||tag==WOTR_TUPLE||tag==WOTR_SET||tag==WOTR_FROZENSET||tag==WOTR_DICT,
                "action clone requires a supported builtin iterable");
        auto values=child_edges(source,n);
        if(tag==WOTR_DICT){std::vector<uint64_t> keys;for(size_t i=0;i<values.size();i+=2)keys.push_back(values[i]);return keys;}
        return values;
    }
    bool same_key(uint64_t a,uint64_t b)const{
        if(a==b)return true;const auto& x=source.nodes[a];const auto& y=source.nodes[b];
        if((x.tag==WOTR_INT||x.tag==WOTR_BOOL)&&(y.tag==WOTR_INT||y.tag==WOTR_BOOL))return x.integer==y.integer;
        if(x.tag!=y.tag)return false;
        if(x.tag==WOTR_STRING||x.tag==WOTR_BYTES||x.tag==WOTR_BIGINT)
            return x.count==y.count && std::equal(source.bytes.begin()+x.offset,source.bytes.begin()+x.offset+x.count,source.bytes.begin()+y.offset);
        if(x.tag==WOTR_NONE)return true;
        if(x.tag==WOTR_ENUM||x.tag==WOTR_TUPLE){
            if(x.count!=y.count)return false;for(uint64_t i=0;i<x.count;++i)if(!same_key(source.edges[x.offset+i],source.edges[y.offset+i]))return false;return true;
        }
        if(x.tag==WOTR_FLOAT)return x.real==y.real;
        return false;
    }
    uint64_t copy_iter(uint64_t n,uint32_t tag){
        if(tag==WOTR_TUPLE && source.nodes[n].tag==WOTR_TUPLE)return n;
        auto values=iterable(n);
        if(tag==WOTR_SET){
            std::vector<uint64_t> unique;
            for(auto value:values){bool seen=false;for(auto old:unique)if(same_key(old,value)){seen=true;break;}if(!seen)unique.push_back(value);}
            values=std::move(unique);
        }
        // list()/set() produce private containers even from externally bound lists.
        return container(tag,values);
    }
    uint64_t copy_dict(uint64_t n){typed(source,n,WOTR_DICT,"action clone requires dictionary");return container(WOTR_DICT,child_edges(source,n));}
    void dict_put(uint64_t dict,const char* key,uint64_t value){
        auto edges=child_edges(work,dict);bool replaced=false;
        for(size_t i=0;i<edges.size();i+=2)if(text_is(work,edges[i],key)){edges[i+1]=value;replaced=true;}
        if(!replaced){edges.push_back(str(key));edges.push_back(value);}set_edges(work,dict,edges);
    }
    uint64_t or_empty_copy(uint64_t n,uint32_t tag){return nonempty(n)?(tag==WOTR_DICT?copy_dict(n):copy_iter(n,tag)):container(tag);}
    uint64_t optional_dict(uint64_t n,bool by_truth){return (by_truth?nonempty(n):present(n))?copy_dict(n):none;}
    uint64_t replace_character(uint64_t n){
        if(!nonempty(n))return none;
        const auto type=object_type(source,n);Fields fields;
        auto add=[&](const char* key,uint64_t fallback=missing){fields.emplace_back(key,attr(n,key,fallback));};
        if(type=="game_state.ShadowMinion"){
            for(auto key:{"name","level","leadership"})add(key);
            for(auto key:{"in_play","was_ever_in_play","immobile","infinite_movement","combat_card_draw","isengard_elite_leadership"})add(key,no);
        }else if(type=="game_state.GandalfTheWhite"){
            add("level",scalar(WOTR_INT,3));add("leadership",scalar(WOTR_INT,1));add("captain_of_the_west",no);add("in_play",no);
            add("base_movement",scalar(WOTR_INT,3));add("solo_hobbit_movement",scalar(WOTR_INT,4));add("can_cancel_nazgul_leadership",yes);
        }else if(type=="game_state.Aragorn"){
            add("level",scalar(WOTR_INT,3));add("leadership",scalar(WOTR_INT,2));add("captain_of_the_west",yes);add("crowned",no);add("crown_regions");
        }else if(type=="game_state.Companion"){
            add("name");add("level");add("leadership",scalar(WOTR_INT,1));add("captain_of_the_west",no);add("guide_ability",none);
            add("activates_nations");add("activation_regions");add("can_muster_nation",none);add("muster_regions");
            add("bonus_movement",zero);add("solo_hobbit_movement",no);
        }else throw std::invalid_argument("action clone unsupported dataclasses.replace character class");
        return object(type,fields);
    }
    uint64_t route_event(uint64_t event){
        if(!present(event))return none;
        const auto& d=typed(source,event,WOTR_DICT,"Malformed Fellowship route event");
        require(d.count==6,"Malformed Fellowship route event");
        auto kind=get(event,"kind"),due=get(event,"balrog_due"),queue=get(event,"conditional_strongholds");
        require(text_is(source,kind,"reveal")||text_is(source,kind,"declare"),"Malformed Fellowship route event");
        typed(source,due,WOTR_BOOL,"Malformed Fellowship route event");
        require(queue!=missing && (source.nodes[queue].tag==WOTR_LIST||source.nodes[queue].tag==WOTR_TUPLE),"Malformed conditional stronghold order");
        auto values=child_edges(source,queue);require(values.size()<=8,"Malformed conditional stronghold order");
        std::vector<std::string_view> names;
        for(auto value:values){auto name=text(source,value);bool allowed=false;
            for(auto valid:{"barad dur","dol guldur","minas morgul","morannon","moria","mount gundabad","orthanc","umbar"})if(name==valid)allowed=true;
            require(allowed && std::find(names.begin(),names.end(),name)==names.end(),"Malformed conditional stronghold order");names.push_back(name);
        }
        return container(WOTR_DICT,{str("kind"),kind,str("balrog_due"),due,str("conditional_strongholds"),copy_iter(queue,WOTR_TUPLE)});
    }
};

std::unique_ptr<Snapshot> ac_compact(const Snapshot& s){
    std::vector<uint64_t> mapping(s.nodes.size(),missing),order;order.reserve(s.nodes.size());order.push_back(s.root);mapping[s.root]=0;
    size_t edge_count=0,byte_count=0;
    auto discover=[&](uint64_t n){require(n<s.nodes.size(),"action clone bad retained reference");if(mapping[n]==missing){mapping[n]=order.size();order.push_back(n);}};
    for(size_t cursor=0;cursor<order.size();++cursor){
        const auto& n=s.nodes[order[cursor]];
        if(n.flags==1)discover(static_cast<uint64_t>(n.integer));
        if(n.tag>=WOTR_LIST && n.tag<=WOTR_ENUM){edge_count+=n.count;for(uint64_t i=0;i<n.count;++i)discover(s.edges[n.offset+i]);}
        else if(n.tag==WOTR_STRING||n.tag==WOTR_BYTES||n.tag==WOTR_BIGINT)byte_count+=n.count;
    }
    auto out=std::make_unique<Snapshot>();out->root=0;out->nodes.reserve(order.size());
    out->edges.reserve(edge_count);out->bytes.reserve(byte_count);
    for(auto index:order){auto n=s.nodes[index];
        if(n.flags==1)n.integer=static_cast<int64_t>(mapping[static_cast<uint64_t>(n.integer)]);
        if(n.tag==WOTR_STRING||n.tag==WOTR_BYTES||n.tag==WOTR_BIGINT){
            auto offset=out->bytes.size();out->bytes.insert(out->bytes.end(),s.bytes.begin()+n.offset,s.bytes.begin()+n.offset+n.count);n.offset=offset;
        }else if(n.tag>=WOTR_LIST && n.tag<=WOTR_ENUM){
            auto offset=out->edges.size();for(uint64_t i=0;i<n.count;++i)out->edges.push_back(mapping[s.edges[n.offset+i]]);n.offset=offset;
        }
        out->nodes.push_back(n);
    }
    // This is an internal constructor, never a raw external-buffer import.
    // Source nodes are validated, builder nodes have canonical tags/shapes,
    // every retained reference was checked/remapped above, and flags1 nodes
    // can only be retained once (new copied containers always have flags0).
    // create() would redundantly validate and copy all three owned vectors.
    return out;
}

struct ac_Deepcopy {
    const Snapshot& source;
    Snapshot work;
    std::vector<uint64_t> memo;
    ac_Deepcopy(const Snapshot& s,uint64_t board):source(s),memo(s.nodes.size(),missing){
        work.nodes.reserve(s.nodes.size()*2+16);work.nodes.assign(s.nodes.begin(),s.nodes.end());
        work.edges.reserve(s.edges.size()*2+32);work.edges.assign(s.edges.begin(),s.edges.end());
        // deepcopy creates no new text/byte payload. Atomic values and object
        // field/type names retain their original graph nodes.
        work.bytes=s.bytes;work.root=s.root;
        require(board<s.nodes.size(),"Paused clone missing Board reference");
        memo[board]=board;
    }
    uint64_t copy(uint64_t original){
        require(original<source.nodes.size(),"Paused clone invalid graph reference");
        if(memo[original]!=missing)return memo[original];
        const auto source_value=source.nodes[original];
        if(source_value.tag<=WOTR_BYTES || source_value.tag==WOTR_ENUM || source_value.tag==WOTR_BIGINT){
            memo[original]=original;return original;
        }
        require(source_value.tag>=WOTR_LIST && source_value.tag<=WOTR_OBJECT,"Paused clone unsupported value type");
        // Register before descending so mutable containers, object __dict__
        // aliases, CombatState._live_state and tuple/list cycles reconnect.
        auto fresh=source_value;
        fresh.offset=fresh.count=0;
        if(fresh.flags==1){fresh.flags=0;fresh.integer=0;}
        const auto result=work.nodes.size();work.nodes.push_back(fresh);memo[original]=result;
        std::vector<uint64_t> children;children.reserve(source_value.count);
        bool unchanged=true;
        for(uint64_t i=0;i<source_value.count;++i){
            auto old_child=source.edges[source_value.offset+i],new_child=copy(old_child);
            children.push_back(new_child);unchanged=unchanged && new_child==old_child;
        }
        if(source_value.tag==WOTR_OBJECT && children.size()==3 && text_is(source,source.edges[source_value.offset+1],"__dict__")){
            // Python _reconstruct creates the new object, deepcopies its state,
            // then updates a fresh __dict__. The memoized state dictionary is
            // therefore distinct from this final object dictionary: explicit
            // aliases to vars(original) retain the intermediate copied dict.
            const auto copied_attributes=work.nodes[children[2]];
            require(copied_attributes.tag==WOTR_DICT,"Paused object state must be a dictionary");
            children[2]=work.nodes.size();work.nodes.push_back(copied_attributes);
        }
        // copy.deepcopy(tuple) returns the original tuple if every contained
        // reference stayed identical. In a representable tuple/list cycle a
        // copied mutable child ensures this is false, so provisional tuple
        // registration also preserves the recursive alias correctly.
        if(source_value.tag==WOTR_TUPLE && unchanged){memo[original]=original;return original;}
        const auto offset=work.edges.size();work.edges.insert(work.edges.end(),children.begin(),children.end());
        work.nodes[result].offset=offset;work.nodes[result].count=children.size();return result;
    }
};

std::unique_ptr<Snapshot> ac_paused_clone(const Snapshot& source){
    const auto board=obs_attr(source,source.root,"board");
    // Supported engine GameStates have a Board, rather than a forged self
    // reference whose deepcopy memo would return and mutate the input root.
    require(object_type(source,board)=="board.Board","Paused clone requires an engine Board");
    ac_Deepcopy copier(source,board);copier.work.root=copier.copy(source.root);
    const auto attrs=attributes(copier.work,copier.work.root,"game_state.GameState");
    auto original=child_edges(copier.work,attrs);std::vector<uint64_t> retained;retained.reserve(original.size());
    for(size_t i=0;i<original.size();i+=2){
        const auto key=original[i];
        if(text_is(copier.work,key,"_last_round_call")||text_is(copier.work,key,"_last_round_phase_call")||text_is(copier.work,key,"_last_battle_result"))continue;
        retained.push_back(key);retained.push_back(original[i+1]);
    }
    set_edges(copier.work,attrs,retained);
    return ac_compact(copier.work);
}

std::unique_ptr<Snapshot> action_clone_uncached(const Snapshot& source){
    attributes(source,source.root,"game_state.GameState");
    auto pending=obs_attr(source,source.root,"pending_combat",false);
    if(pending!=missing && truth(source,pending)){
        auto resume=field(source,pending,"_round_resume",false);
        if(resume!=missing && truth(source,resume))return ac_paused_clone(source);
    }
    ac_Builder b(source);auto root=source.root;
    auto optional=[&](const char* key){return obs_attr(source,root,key,false);};
    auto attr=[&](const char* key,uint64_t fallback=missing){return b.attr(root,key,fallback);};
    auto pc=optional("pending_combat");
    ac_Builder::Fields fields;
    auto put=[&](const char* key,uint64_t value){ac_Builder::put(fields,key,value);};
    auto copy_mapped=[&](uint64_t dictionary,auto transform){
        typed(source,dictionary,WOTR_DICT,"action clone requires mapping");auto entries=child_edges(source,dictionary);
        for(size_t i=1;i<entries.size();i+=2)entries[i]=transform(entries[i]);return b.container(WOTR_DICT,entries);
    };
    auto armies=copy_mapped(attr("armies"),[&](uint64_t army){
        attributes(source,army,"game_state.Army");std::vector<uint64_t> units;
        for(auto unit:b.iterable(b.attr(army,"units")))units.push_back(b.object("game_state.Unit",{{"nation",b.attr(unit,"nation")},{"kind",b.attr(unit,"kind")}}));
        return b.object("game_state.Army",{{"region",b.attr(army,"region")},{"units",b.container(WOTR_LIST,units)},{"leaders",b.copy_iter(b.attr(army,"leaders"),WOTR_LIST)}});
    });
    const auto old_dice=attr("dice");typed(source,old_dice,WOTR_DICT,"action clone dice requires mapping");
    auto original_pools=child_edges(source,old_dice);std::vector<uint64_t> new_pools=original_pools;
    for(size_t i=1;i<new_pools.size();i+=2){std::vector<uint64_t> dice;
        for(auto die:b.iterable(original_pools[i])){
            require(object_type(source,die)=="game_state.ActionDie","action clone invalid die class");
            dice.push_back(b.object("game_state.ActionDie",{{"faces",b.attr(die,"faces")},{"current",b.attr(die,"current")},{"side",b.attr(die,"side")},{"used",b.attr(die,"used")}},true));
        }
        new_pools[i]=b.container(WOTR_LIST,dice);
    }
    auto dice=b.container(WOTR_DICT,new_pools);std::vector<uint64_t> hunt_box;
    for(auto old_hunt:b.iterable(attr("hunt_box")))for(size_t side=1;side<original_pools.size();side+=2){
        auto old_list=b.iterable(original_pools[side]);auto new_list=child_edges(b.work,new_pools[side]);
        for(size_t i=0;i<old_list.size();++i)if(old_list[i]==old_hunt){hunt_box.push_back(new_list[i]);break;}
    }
    auto fellowship=attr("fellowship");
    auto new_fellowship=b.object("game_state.Fellowship",{{"companions",b.copy_iter(b.attr(fellowship,"companions"),WOTR_LIST)},
        {"region",b.attr(fellowship,"region")},{"guide",b.attr(fellowship,"guide")},{"progress",b.attr(fellowship,"progress",b.zero)},
        {"corruption",b.attr(fellowship,"corruption",b.zero)},{"hidden",b.attr(fellowship,"hidden",b.yes)},
        {"mordor_entered",b.attr(fellowship,"mordor_entered",b.no)},{"mordor_step",b.attr(fellowship,"mordor_step",b.zero)}});
    auto hands=copy_mapped(attr("hands"),[&](uint64_t pi){return b.object("game_state.PlayerInfo",{
        {"hand",b.copy_iter(b.attr(pi,"hand"),WOTR_LIST)},{"discarded_face_down",b.copy_iter(b.attr(pi,"discarded_face_down"),WOTR_LIST)},{"needs_discard",b.no}});});
    auto politics=attr("political"),vp=attr("vp"),decks=attr("decks"),cards=attr("persistent_cards");
    put("board",attr("board"));put("armies",armies);
    put("enemy_controlled",copy_mapped(attr("enemy_controlled"),[&](uint64_t n){return b.copy_iter(n,WOTR_SET);}));
    put("fellowship",new_fellowship);put("political",b.object("game_state.PoliticalTrack",{{"boxes",b.copy_dict(b.attr(politics,"boxes"))},{"active",b.copy_dict(b.attr(politics,"active"))}}));
    put("vp",b.object("game_state.VictoryPoints",{{"free_peoples",b.attr(vp,"free_peoples",b.zero)},{"shadow",b.attr(vp,"shadow",b.zero)}}));
    put("dice",dice);put("turn",attr("turn"));
    for(auto key:{"hunt_pool","hunt_discard","hunt_removed"})put(key,b.copy_iter(attr(key),WOTR_LIST));
    put("hunt_box",b.container(WOTR_LIST,hunt_box));
    put("decks",b.object("game_state.DeckStatus",{{"fp_character",b.attr(decks,"fp_character")},{"fp_strategy",b.attr(decks,"fp_strategy")},{"sh_character",b.attr(decks,"sh_character")},{"sh_strategy",b.attr(decks,"sh_strategy")}}));
    put("hands",hands);put("discard_face_up",copy_mapped(attr("discard_face_up"),[&](uint64_t n){return b.copy_iter(n,WOTR_LIST);}));
    put("fp_elven_rings",attr("fp_elven_rings",b.scalar(WOTR_INT,3)));put("fp_dwarven_rings",attr("fp_dwarven_rings",b.zero));
    put("fp_political_token",attr("fp_political_token",b.no));put("fp_card_token",attr("fp_card_token",b.no));
    for(auto key:{"sh_elven_rings","sh_dwarven_rings","sh_action_tokens"})put(key,attr(key,b.zero));
    put("eyes_allocated",attr("eyes_allocated",b.zero));
    for(auto key:{"fp_ring_used_this_turn","sh_ring_used_this_turn","fp_ring_waiver","sh_ring_waiver","sh_mouth_muster_used_this_turn","fp_token_used_this_turn","sh_token_used_this_turn","dice_rolled"})put(key,attr(key,b.no));
    for(auto key:{"witch_king","saruman","mouth_of_sauron","gandalf_the_white","aragorn_crowned"})put(key,b.replace_character(optional(key)));
    put("gollum",attr("gollum",b.none));for(auto key:{"fp_bonus_dice","sh_bonus_dice"})put(key,attr(key,b.zero));
    put("persistent_cards",b.object("game_state.PersistentCardState",{{"fp_cards",b.copy_iter(b.attr(cards,"fp_cards"),WOTR_LIST)},{"sh_cards",b.copy_iter(b.attr(cards,"sh_cards"),WOTR_LIST)}}));
    put("pending_hunt_tiles",b.copy_iter(attr("pending_hunt_tiles"),WOTR_LIST));
    for(auto key:{"pending_army_move_die","pending_army_move_from","pending_army_move_side"})put(key,attr(key,b.none));
    put("pending_army_move_preexist",b.optional_dict(optional("pending_army_move_preexist"),true));
    for(auto key:{"pending_muster_die","pending_muster_first_type","pending_muster_region","pending_muster_side"})put(key,attr(key,b.none));
    auto casualties=optional("fp_casualties");put("fp_casualties",b.nonempty(casualties)?copy_mapped(casualties,[&](uint64_t n){return b.copy_dict(n);}):b.container(WOTR_DICT));
    auto combat=b.present(pc)?b.copy_dict(pc):b.none;put("pending_combat",combat);
    put("_wormtongue_attack_pending",attr("_wormtongue_attack_pending",b.none));
    for(auto key:{"pending_hunt","pending_guide_change"})put(key,b.optional_dict(optional(key),false));
    put("_gandalf_draw_pending",b.optional_dict(optional("_gandalf_draw_pending"),false));
    auto siege=optional("besieged_strongholds");put("besieged_strongholds",siege==missing?b.container(WOTR_SET):b.copy_iter(siege,WOTR_SET));
    put("fellowship_acted_in_mordor",attr("fellowship_acted_in_mordor",b.no));
    // Dynamic insertion order matches the statements following GameState(...).
    for(auto key:{"fp_char_deck","fp_strat_deck","sh_char_deck","sh_strat_deck"}){auto n=optional(key);if(b.present(n))put(key,b.copy_iter(n,WOTR_LIST));}
    auto option=optional("pending_card_option"),option_copy=b.optional_dict(option,true);
    if(b.nonempty(option)){
        b.dict_put(option_copy,"options",b.copy_iter(b.get(option,"options"),WOTR_LIST));
        auto resume=b.get(option,"resume_action");if(b.nonempty(resume)){
            auto resume_copy=b.copy_dict(resume);b.dict_put(resume_copy,"params",b.or_empty_copy(b.get(resume,"params"),WOTR_DICT));b.dict_put(option_copy,"resume_action",resume_copy);
        }
    }
    put("pending_card_option",option_copy);put("_crebain_hunt_bonus",attr("_crebain_hunt_bonus",b.zero));
    put("_reveal_place_due",attr("_reveal_place_due",b.no));put("_hunt_reveal_pending",attr("_hunt_reveal_pending",b.zero));put("_ents_chain_n",attr("_ents_chain_n",b.zero));
    put("_mithril_stash",b.optional_dict(optional("_mithril_stash"),true));put("_fellowship_move_die_pending",attr("_fellowship_move_die_pending",b.none));
    put("_wizardstaff_stash",b.optional_dict(optional("_wizardstaff_stash"),true));auto over=optional("_overstack_regions");if(b.nonempty(over))put("_overstack_regions",b.copy_iter(over,WOTR_SET));
    auto follow=optional("pending_card_followup"),follow_copy=b.optional_dict(follow,true);
    if(b.nonempty(follow))b.dict_put(follow_copy,"exclude",b.or_empty_copy(b.get(follow,"exclude"),WOTR_SET));
    put("pending_card_followup",follow_copy);put("_owed_draws",b.or_empty_copy(optional("_owed_draws"),WOTR_DICT));
    for(auto key:{"_ring_free_side","fp_ring_waiver_face","sh_ring_waiver_face"})put(key,attr(key,b.none));
    if(b.present(pc)){
        for(auto key:{"_onslaught_payment","_observed_round_completed","atk_hidden_cc","def_hidden_cc"}){
            auto n=b.get(pc,key);if(n!=missing && source.nodes[n].tag==WOTR_DICT)b.dict_put(combat,key,b.copy_dict(n));
        }
        auto effect=b.get(pc,"effect_choice");if(effect!=missing && source.nodes[effect].tag==WOTR_DICT){
            auto changed=b.copy_dict(effect),counts=b.get(effect,"counts");
            if(counts!=missing && source.nodes[counts].tag==WOTR_LIST)b.dict_put(changed,"counts",b.copy_iter(counts,WOTR_LIST));b.dict_put(combat,"effect_choice",changed);
        }
    }
    put("_pending_hunt_damage",attr("_pending_hunt_damage",b.zero));put("pending_discard",b.optional_dict(optional("pending_discard"),false));
    for(auto key:{"_eye_allocation_phase","_declare_phase","_fellowship_tail_phase"})put(key,attr(key,b.no));
    auto provenance=optional("_rx_hunt_observation");if(provenance!=missing)put("_rx_hunt_observation",b.copy_iter(provenance,WOTR_TUPLE));
    put("_search_turn_crossings",attr("_search_turn_crossings",b.zero));
    const auto terminal_marker=optional("_search_endturn_victory");if(terminal_marker!=missing)put("_search_endturn_victory",b.present(terminal_marker)?b.copy_dict(terminal_marker):b.none);
    put("_hobbit_place_queue",b.or_empty_copy(optional("_hobbit_place_queue"),WOTR_LIST));
    auto drawn=optional("_sh_tile_drawn_regions");put("_sh_tile_drawn_regions",b.present(drawn)?b.copy_iter(drawn,WOTR_SET):b.none);
    put("_pending_reveal_strongholds",b.or_empty_copy(optional("_pending_reveal_strongholds"),WOTR_TUPLE));
    put("_fellowship_route_event",b.route_event(optional("_fellowship_route_event")));
    for(auto key:{"_wwsat_owed","_dg_src_log","_casualty_operations"})put(key,b.or_empty_copy(optional(key),WOTR_LIST));
    put("_fp_used_hunt_box_last_turn",attr("_fp_used_hunt_box_last_turn",b.no));put("_palantir_draw_pending",attr("_palantir_draw_pending",b.no));
    put("_battle_resolved_attacker",attr("_battle_resolved_attacker",b.none));
    b.work.root=b.object("game_state.GameState",fields);return ac_compact(b.work);
}

std::unique_ptr<Snapshot> action_clone(const Snapshot& source){
    std::lock_guard<std::mutex> lock(source.action_clone_mutex);
    if(source.action_clone_template){
        auto child=std::make_unique<Snapshot>(*source.action_clone_template);
        ++source.action_clone_hits;return child;
    }
    // Both cache and result own complete native arrays. A mutable result never
    // borrows the template's storage, and neither inherits another cache.
    std::shared_ptr<const Snapshot> prepared=action_clone_uncached(source);
    auto child=std::make_unique<Snapshot>(*prepared);
    source.action_clone_template=std::move(prepared);++source.action_clone_misses;
    return child;
}
