// Exact source logical search projections, not a graph hash.
// ai.py _state_sig/_chance_sig, timing/route/round helpers are the authority.
uint64_t kc_copy(Snapshot& out,const Snapshot& in,uint64_t n,std::unordered_map<uint64_t,uint64_t>& memo);
struct sm_Builder:am_Builder {
    const Snapshot& s;uint64_t nil;std::unordered_map<uint64_t,uint64_t> copied;
    explicit sm_Builder(const Snapshot& source):s(source){nil=rr_scalar(out,WOTR_NONE);}
    uint64_t tup(const std::vector<uint64_t>& xs){return container(WOTR_TUPLE,xs);}
    uint64_t tup(std::initializer_list<uint64_t> xs){return container(WOTR_TUPLE,xs);}
    uint64_t boolean(bool x){return rr_scalar(out,WOTR_BOOL,x);}
    uint64_t attr(const char* k){return obs_attr(s,s.root,k,false);}
    uint64_t get(uint64_t n,const char* k){return truth(s,n)?ctx_get(s,n,k):missing;}
    std::vector<uint64_t> items(uint64_t n){return ctx_none(s,n)?std::vector<uint64_t>{}:child_edges(s,n);}
    uint64_t raw(uint64_t n){
        if(n==missing)return nil;auto found=copied.find(n);if(found!=copied.end())return found->second;
        auto v=s.nodes[n];const auto result=out.nodes.size();copied[n]=result;out.nodes.push_back(v);
        if(v.tag==WOTR_STRING||v.tag==WOTR_BYTES||v.tag==WOTR_BIGINT){const auto offset=out.bytes.size();out.bytes.insert(out.bytes.end(),s.bytes.begin()+v.offset,s.bytes.begin()+v.offset+v.count);out.nodes[result].offset=offset;}
        else if(v.tag>=WOTR_LIST&&v.tag<=WOTR_ENUM){std::vector<uint64_t> children;for(auto c:items(n))children.push_back(raw(c));set_edges(out,result,children);if(v.flags==1)out.nodes[result].integer=raw(v.integer);}
        return result;
    }
    std::string type_name(uint64_t n){
        if(ctx_none(s,n))return "NoneType";const auto tag=s.nodes[n].tag;
        if(tag==WOTR_OBJECT||tag==WOTR_ENUM){std::string full(tag==WOTR_OBJECT?object_type(s,n):text(s,s.edges[s.nodes[n].offset]));return full.substr(full.rfind('.')+1);}
        switch(tag){case WOTR_BOOL:return "bool";case WOTR_INT:case WOTR_BIGINT:return "int";case WOTR_FLOAT:return "float";case WOTR_STRING:return "str";case WOTR_BYTES:return "bytes";case WOTR_DICT:return "dict";case WOTR_LIST:return "list";case WOTR_TUPLE:return "tuple";case WOTR_SET:return "set";case WOTR_FROZENSET:return "frozenset";}throw std::invalid_argument("Unknown search projection type");
    }
    // round_signature sorts observed (phase, role) tuple keys by Python str.
    // Tuple str uses repr of each element; do not use the policy freezer,
    // which intentionally changes list/dict/set types into tuples.
    std::string tuple_repr(uint64_t n,size_t depth=0){
        require(depth<200,"Combat signature tuple nested too deeply");
        if(ctx_none(s,n))return "None";
        const auto& v=s.nodes[n];
        if(v.tag==WOTR_TUPLE){
            auto es=items(n);std::string result="(";
            for(size_t i=0;i<es.size();++i){if(i)result+=", ";result+=tuple_repr(es[i],depth+1);}
            if(es.size()==1)result+=",";
            return result+")";
        }
        if(v.tag==WOTR_STRING){ap_Frozen f;f.tag=WOTR_STRING;f.value=std::string(text(s,n));return ap_repr(f);}
        if(v.tag==WOTR_BOOL)return v.integer?"True":"False";
        if(v.tag==WOTR_INT)return std::to_string(v.integer);
        if(v.tag==WOTR_BIGINT)return std::string(reinterpret_cast<const char*>(s.bytes.data()+v.offset),v.count);
        throw std::invalid_argument("Unsupported combat signature tuple repr element");
    }
    std::string string(uint64_t n){
        if(ctx_none(s,n))return "None";const auto& v=s.nodes[n];
        if(v.tag==WOTR_STRING)return std::string(text(s,n));
        if(v.tag==WOTR_BIGINT)return std::string(reinterpret_cast<const char*>(s.bytes.data()+v.offset),v.count);
        if(v.tag==WOTR_TUPLE)return tuple_repr(n);
        if(v.tag==WOTR_BOOL)return v.integer?"True":"False";
        if(v.tag==WOTR_INT)return std::to_string(v.integer);
        if(v.tag==WOTR_ENUM)return type_name(n)+"."+std::string(text(s,s.edges[v.offset+1]));
        throw std::invalid_argument("Search string projection requires an explicit supported scalar");
    }
    uint64_t name(uint64_t n){if(ctx_none(s,n))return missing;if(s.nodes[n].tag==WOTR_ENUM)return s.edges[s.nodes[n].offset+1];if(s.nodes[n].tag==WOTR_OBJECT)return obs_attr(s,n,"name",false);return missing;}
    std::vector<std::pair<uint64_t,uint64_t>> pairs(uint64_t n){std::vector<std::pair<uint64_t,uint64_t>> r;if(ctx_none(s,n))return r;typed(s,n,WOTR_DICT,"Search projection requires dictionary");auto es=items(n);for(size_t i=0;i<es.size();i+=2)r.emplace_back(es[i],es[i+1]);return r;}
    std::vector<std::pair<uint64_t,uint64_t>> sorted_pairs(uint64_t n){auto r=pairs(n);std::stable_sort(r.begin(),r.end(),[&](auto a,auto b){return string(a.first)<string(b.first);});return r;}
    uint64_t sorted_strings(uint64_t n){std::vector<std::string> xs;for(auto c:items(n))xs.push_back(string(c));std::sort(xs.begin(),xs.end());std::vector<uint64_t> r;for(auto& x:xs)r.push_back(str(x));return tup(r);}
    uint64_t fields(uint64_t n,std::initializer_list<const char*> keys,bool object=false){std::vector<uint64_t> r;for(auto k:keys)r.push_back(raw(object?obs_attr(s,n,k,false):get(n,k)));return tup(r);}
    uint64_t jsonfreeze(uint64_t n){if(ctx_none(s,n))return nil;const auto tag=s.nodes[n].tag;if(tag==WOTR_DICT){std::vector<uint64_t> r;for(auto [k,v]:sorted_pairs(n))r.push_back(tup({str(string(k)),jsonfreeze(v)}));return tup(r);}if(tag==WOTR_LIST||tag==WOTR_TUPLE){std::vector<uint64_t> r;for(auto c:items(n))r.push_back(jsonfreeze(c));return tup(r);}return raw(n);}
    uint64_t roundfreeze(uint64_t n,std::unordered_map<uint64_t,size_t>& seen){
        if(ctx_none(s,n))return nil;const auto tag=s.nodes[n].tag;
        if(tag==WOTR_ENUM){const auto fullname=std::string(text(s,s.edges[s.nodes[n].offset]));auto dot=fullname.rfind('.');return tup({str(fullname.substr(0,dot)),str(fullname.substr(dot+1)),raw(name(n))});}
        if(tag<=WOTR_STRING||tag==WOTR_BIGINT)return raw(n);
        auto old=seen.find(n);if(old!=seen.end())return tup({str("ref"),integer(old->second)});seen[n]=seen.size();
        if(tag==WOTR_DICT){std::vector<uint64_t> r;for(auto [k,v]:sorted_pairs(n))if(!text_is(s,k,"_live_state"))r.push_back(tup({roundfreeze(k,seen),roundfreeze(v,seen)}));return tup({str("dict"),tup(r)});}
        if(tag==WOTR_LIST||tag==WOTR_TUPLE||tag==WOTR_SET||tag==WOTR_FROZENSET){auto es=items(n);if(tag==WOTR_SET||tag==WOTR_FROZENSET)std::stable_sort(es.begin(),es.end(),[&](auto a,auto b){return string(a)<string(b);});std::vector<uint64_t> r;for(auto c:es)r.push_back(roundfreeze(c,seen));return tup({str(tag==WOTR_SET||tag==WOTR_FROZENSET?"set":type_name(n)),tup(r)});}
        if(tag==WOTR_OBJECT){const auto fullname=std::string(object_type(s,n));auto dot=fullname.rfind('.');const auto attrs=obs_attr(s,n,"__dict__",false);require(attrs!=missing,"Unsupported slotted combat signature object");return tup({str(fullname.substr(0,dot)),str(fullname.substr(dot+1)),roundfreeze(attrs,seen)});}
        throw std::invalid_argument("Unsupported combat signature value");
    }
    uint64_t round(uint64_t n){std::unordered_map<uint64_t,size_t> seen;return roundfreeze(n,seen);}
    uint64_t public_context(){
        // Explicit public rules state. Never include unseen deck identities
        // or the shuffled order of a Hunt bag.
        std::vector<uint64_t> fields,tiles,box,decks,hands,discards;
        for(auto key:{"enemy_controlled","besieged_strongholds","persistent_cards","witch_king","saruman","mouth_of_sauron","gandalf_the_white","aragorn_crowned","fp_dwarven_rings","sh_dwarven_rings","fp_political_token","fp_card_token","sh_action_tokens","fp_token_used_this_turn","sh_token_used_this_turn","sh_mouth_muster_used_this_turn","dice_rolled","eyes_allocated","fellowship_acted_in_mordor","_fp_used_hunt_box_last_turn","_hunt_reveal_pending","_wormtongue_attack_pending","game_over","winner","_search_endturn_victory"})
            fields.push_back((std::string_view(key)=="_hunt_reveal_pending"||std::string_view(key)=="_wormtongue_attack_pending")
                ? integer(static_cast<int64_t>(ctx_int(s,attr(key)))) : round(attr(key)));
        using Tile=std::tuple<int64_t,bool,bool,bool,std::string,bool,bool,bool>;
        for(auto key:{"hunt_pool","hunt_discard","hunt_removed"}){
            std::vector<Tile> values;
            for(auto tile:items(attr(key))){
                auto value=[&](const char* k){return obs_attr(s,tile,k);};
                auto special=value("special_name");
                values.emplace_back(typed(s,value("damage"),WOTR_INT,"Hunt damage must be integer").integer,
                    truth(s,value("is_eye")),truth(s,value("reveal")),truth(s,value("is_special")),
                    truth(s,special)?string(special):"",truth(s,value("stops_movement")),
                    truth(s,value("is_dice_roll")),truth(s,value("is_fellowship_special")));
            }
            std::sort(values.begin(),values.end());std::vector<uint64_t> rows;
            for(auto& [damage,eye,reveal,special,nm,stop,roll,fp]:values)
                rows.push_back(tup({integer(damage),boolean(eye),boolean(reveal),boolean(special),str(nm),boolean(stop),boolean(roll),boolean(fp)}));
            tiles.push_back(tup(rows));
        }
        std::vector<std::tuple<std::string,std::string,bool>> dice;
        for(auto die:items(attr("hunt_box"))){auto face=obs_attr(s,die,"current");dice.emplace_back(string(name(obs_attr(s,die,"side"))),truth(s,face)?string(name(face)):"-",truth(s,obs_attr(s,die,"used")));}
        std::sort(dice.begin(),dice.end());for(auto& [side,face,used]:dice)box.push_back(tup({str(side),str(face),boolean(used)}));
        for(auto key:{"fp_char_deck","fp_strat_deck","sh_char_deck","sh_strat_deck"})decks.push_back(integer(items(attr(key)).size()));
        for(int side=0;side<2;++side){
            auto entry=enum_value_edge(s,attr("hands"),"game_state.Side",side?"SHADOW":"FREE_PEOPLES");require(entry!=missing,"Search hand missing");
            auto player=s.edges[entry];int chars=0;for(auto card:items(obs_attr(s,player,"hand")))chars+=glb_card_deck(s,card)==0;
            hands.push_back(tup({integer(chars),integer(items(obs_attr(s,player,"discarded_face_down")).size())}));
        }
        for(auto [key,value]:sorted_pairs(truth(s,attr("discard_face_up"))?attr("discard_face_up"):missing))discards.push_back(tup({str(string(key)),sorted_strings(value)}));
        return tup({tup(fields),round(obs_attr(s,attr("fellowship"),"guide")),tup(tiles),tup(box),tup(decks),tup(hands),tup(discards)});
    }
    uint64_t decision_context(bool sort_dice){
        // Exact ai._search_decision_context. Public paid-action cursors also
        // belong in chance keys; full menus stay in retained-tree keys only.
        std::vector<uint64_t> ordinary,retained;
        for(auto key:{"pending_army_move_die","pending_army_move_from","pending_army_move_side","pending_army_move_preexist","pending_muster_die","pending_muster_first_type","pending_muster_region","pending_muster_side"})
            ordinary.push_back(round(attr(key)));
        if(!sort_dice)for(auto key:{"pending_combat","pending_hunt","pending_guide_change","pending_card_option","pending_discard","_pending_hunt_damage","_hobbit_place_queue","_wwsat_owed","_overstack_regions","_palantir_draw_pending","_owed_draws","_battle_resolved_attacker","_crebain_hunt_bonus","_ents_chain_n","_wizardstaff_stash","_mithril_stash"})
            retained.push_back(round(attr(key)));
        return tup({tup(ordinary),sort_dice?nil:tup(retained)});
    }
    uint64_t followup(){
        const auto pending=attr("pending_card_followup");if(ctx_none(s,pending))return nil;
        typed(s,pending,WOTR_DICT,"Card followup search context must be dictionary");
        // Match round_signature(dict(pending, exclude=tuple(sorted(...)))).
        // Copy only this small context; the outer dictionary must be a fresh
        // shallow copy so nested aliases retain Python's round-signature IDs.
        sm_Builder normalized(s);const auto original=normalized.raw(pending);
        const auto context=cp_container(normalized.out,WOTR_DICT,child_edges(normalized.out,original));
        cp_put(normalized.out,context,"exclude",normalized.sorted_strings(get(pending,"exclude")));
        sm_Builder frozen(normalized.out);const auto key=frozen.round(context);
        std::unordered_map<uint64_t,uint64_t> memo;return kc_copy(out,frozen.out,key,memo);
    }
    uint64_t queue(uint64_t n){float ignored[64]{};tail_stronghold_queue(s,n,ignored,"Malformed pending reveal stronghold draws");std::vector<uint64_t> r;for(auto c:items(n))r.push_back(raw(c));return tup(r);}
    uint64_t timing(){
        auto stash=attr("_mithril_stash");uint64_t context=nil,wizard=nil;
        if(truth(s,stash)){auto tile=get(stash,"tile");auto props=fields(tile,{"damage","is_eye","reveal","is_special","special_name","stops_movement","is_dice_roll","is_fellowship_special"},true);auto continuation=fields(stash,{"successes","mordor","mordor_track","skip_auto_sacrifice","eye_no_effect","apply_kind","return_pending"});auto choices=get(stash,"choices");context=tup({props,continuation,truth(s,choices)?jsonfreeze(choices):tup({})});}
        auto ws=attr("_wizardstaff_stash");if(truth(s,ws))wizard=fields(ws,{"apply_kind","successes","mordor","mordor_track","eye_no_effect","region","reason"});
        return tup({raw(attr("_fellowship_move_die_pending")),context,wizard,truth(s,attr("_reveal_place_due"))?sorted_strings(attr("_sh_tile_drawn_regions")):tup({}),queue(attr("_pending_reveal_strongholds"))});
    }
    uint64_t route(){auto e=attr("_fellowship_route_event");if(ctx_none(s,e))return nil;tail_keys(s,e,{"kind","balrog_due","conditional_strongholds"},true,"Malformed Fellowship route event");auto kind=get(e,"kind"),due=get(e,"balrog_due"),q=get(e,"conditional_strongholds");require(text_is(s,kind,"declare")||text_is(s,kind,"reveal"),"Malformed route kind");typed(s,due,WOTR_BOOL,"Malformed route due");require(tail_seq(s,q),"Malformed route queue");return tup({raw(kind),raw(due),queue(q)});}
    uint64_t breaking(){auto p=attr("pending_card_option");if(ctx_none(s,p))return integer(0);typed(s,p,WOTR_DICT,"Malformed pending card context");auto sep=get(p,"bof_separation"),count=get(p,"bof_n_left");if(!truth(s,sep)){require(count==missing,"Breaking count without separation");return integer(0);}require(s.nodes[sep].tag==WOTR_BOOL&&s.nodes[sep].integer==1,"Malformed Breaking flag");auto n=tail_integer(s,count,"Malformed Breaking count");require(n>=1&&n<=8,"Malformed Breaking count");int eligible=0;for(auto c:items(obs_attr(s,attr("fellowship"),"companions"))){auto nm=name(c);eligible+=!text_is(s,nm,"Frodo")&&!text_is(s,nm,"Sam")&&!text_is(s,nm,"Gollum");}require(n<=eligible,"Breaking count exceeds companions");return raw(count);}
    std::string leader(uint64_t n){const auto t=type_name(n);if(t=="Nazgul")return "N";if(t=="Companion"||t=="Minion"){auto nm=name(n);return nm==missing?"":string(nm);}auto nation=obs_attr(s,n,"nation",false),nm=name(nation);return "L:"+(nm==missing?std::string():string(nm));}
    std::string nation_value(uint64_t n){const auto nm=name(n);static const std::map<std::string,std::string> names={{"ELVES","Elves"},{"DWARVES","Dwarves"},{"GONDOR","Gondor"},{"THE_NORTH","The North"},{"ROHAN","Rohan"},{"ISENGARD","Isengard"},{"SAURON","Sauron"},{"SOUTHRON_EASTERLINGS","Southrons & Easterlings"}};if(nm==missing)return string(n);auto found=names.find(string(nm));require(found!=names.end(),"Unknown search nation");return found->second;}
    bool hashable(uint64_t n,std::set<uint64_t>& seen){auto tag=out.nodes[n].tag;if(tag==WOTR_LIST||tag==WOTR_DICT||tag==WOTR_SET||tag==WOTR_OBJECT)return false;if(tag!=WOTR_TUPLE&&tag!=WOTR_FROZENSET)return true;if(!seen.insert(n).second)return false;for(auto c:child_edges(out,n))if(!hashable(c,seen))return false;seen.erase(n);return true;}
    uint64_t state_key(bool sort_dice){try{
        const auto private_combat=attr("pending_combat");
        if(!sort_dice && (truth(s,get(private_combat,"atk_hidden_cc")) || truth(s,get(private_combat,"def_hidden_cc")) ||
            (text_is(s,get(private_combat,"phase"),"defender_combat_card") && truth(s,get(private_combat,"atk_combat_card")))))return nil;
        std::vector<uint64_t> armyrows;for(auto [k,a]:sorted_pairs(attr("armies"))){auto units=items(obs_attr(s,a,"units")),leaders=items(obs_attr(s,a,"leaders"));int elites=0;std::map<std::pair<std::string,std::string>,int> census;std::vector<std::string> ids;
            for(auto u:units){auto kind=obs_attr(s,u,"kind"),nation=obs_attr(s,u,"nation");elites+=enum_is(s,kind,"game_state.UnitKind","ELITE");++census[{nation_value(nation),string(name(kind))}];}for(auto l:leaders)ids.push_back(leader(l));std::sort(ids.begin(),ids.end());std::vector<uint64_t> counts,ls;for(auto& [key,n]:census)counts.push_back(tup({tup({str(key.first),str(key.second)}),integer(n)}));for(auto& id:ids)ls.push_back(str(id));armyrows.push_back(tup({raw(k),integer(units.size()),integer(elites),integer(leaders.size()),tup(counts),tup(ls)}));}
        std::vector<uint64_t> dice;for(int side=0;side<2;++side){std::vector<std::tuple<std::string,bool,uint64_t>> rows;for(auto d:items(pa_pool(s,side))){auto current=obs_attr(s,d,"current"),used=obs_attr(s,d,"used");rows.emplace_back(truth(s,current)?string(name(current)):"-",truth(s,used),used);}if(sort_dice)std::stable_sort(rows.begin(),rows.end(),[](auto& a,auto& b){return std::tie(std::get<0>(a),std::get<1>(a))<std::tie(std::get<0>(b),std::get<1>(b));});for(auto& [face,used,original]:rows)dice.push_back(tup({str(face),raw(original)}));}
        auto f=attr("fellowship"),political=attr("political"),boxes=obs_attr(s,political,"boxes"),active=obs_attr(s,political,"active");std::vector<uint64_t> pol;for(auto nation:{"ELVES","DWARVES","GONDOR","THE_NORTH","ROHAN","ISENGARD","SAURON","SOUTHRON_EASTERLINGS"}){auto be=enum_value_edge(s,boxes,"game_state.Nation",nation),ae=enum_value_edge(s,active,"game_state.Nation",nation);static const std::map<std::string,std::string> nv={{"ELVES","Elves"},{"DWARVES","Dwarves"},{"GONDOR","Gondor"},{"THE_NORTH","The North"},{"ROHAN","Rohan"},{"ISENGARD","Isengard"},{"SAURON","Sauron"},{"SOUTHRON_EASTERLINGS","Southrons & Easterlings"}};pol.push_back(tup({str(nv.at(nation)),be==missing?integer(0):raw(s.edges[be]),boolean(ae!=missing&&truth(s,s.edges[ae]))}));}
        std::vector<std::string> companions;for(auto c:items(obs_attr(s,f,"companions"))){auto nm=name(c);companions.push_back(nm==missing?"":string(nm));}std::sort(companions.begin(),companions.end());std::vector<uint64_t> cs;for(auto& c:companions)cs.push_back(str(c));
        auto pc=attr("pending_combat"),pco=attr("pending_card_option");std::vector<uint64_t> casualties;auto casualty_field=attr("fp_casualties");auto cps=pairs(truth(s,casualty_field)?casualty_field:missing);std::sort(cps.begin(),cps.end(),[&](auto a,auto b){return string(name(a.first))<string(name(b.first));});for(auto [n,c]:cps){std::vector<uint64_t> counts;for(auto [k,v]:sorted_pairs(c))counts.push_back(tup({raw(k),raw(v)}));casualties.push_back(tup({raw(name(n)),tup(counts)}));}
        auto hand=[&](int side){auto e=enum_value_edge(s,attr("hands"),"game_state.Side",side?"SHADOW":"FREE_PEOPLES");require(e!=missing,"Search hand missing");return obs_attr(s,s.edges[e],"hand");};
        std::vector<uint64_t> r={raw(attr("turn")),tup(armyrows),tup(dice),timing(),route(),breaking(),str(glb_fellowship_key(s,f))};for(auto k:{"progress","corruption","hidden","mordor_entered","mordor_step"})r.push_back(raw(obs_attr(s,f,k)));r.push_back(integer(cs.size()));r.push_back(tup(cs));r.push_back(integer(items(hand(0)).size()));r.push_back(integer(items(hand(1)).size()));r.push_back(integer(items(attr("hunt_box")).size()));r.push_back(tup(pol));r.push_back(boolean(!ctx_none(s,pc)));r.push_back(round(get(pc,"_round_resume")));r.push_back(round(get(pc,"_onslaught_payment")));r.push_back(fields(pco,{"card_casualty","cas_region","cas_side","cas_hits"}));r.push_back(fields(pc,{"phase","alloc_stage","deciding_side","atk_hits","def_hits"}));r.push_back(tup(casualties));r.push_back(raw(get(pc,"attacker_rearguard")));
        for(auto k:{"pending_hunt","pending_card_option","pending_discard","pending_guide_change"})r.push_back(boolean(!ctx_none(s,attr(k))));r.push_back(followup());for(auto k:{"_declare_phase","_eye_allocation_phase","_fellowship_tail_phase","_reveal_place_due"})r.push_back(boolean(truth(s,attr(k))));for(auto k:{"fp_elven_rings","sh_elven_rings"})r.push_back(integer(static_cast<int64_t>(ctx_int(s,attr(k)))));for(auto k:{"fp_ring_used_this_turn","sh_ring_used_this_turn"})r.push_back(boolean(truth(s,attr(k))));r.push_back(raw(name(attr("_ring_free_side"))));for(auto k:{"fp_ring_waiver","sh_ring_waiver"})r.push_back(boolean(truth(s,attr(k))));for(auto k:{"fp_ring_waiver_face","sh_ring_waiver_face"})r.push_back(str(string(attr(k))));
        auto queued=attr("_gandalf_draw_pending");bool open=!ctx_none(s,pco)&&truth(s,get(pco,"gandalf_draw"));r.push_back(open?str("open"):ctx_none(s,queued)?nil:str("queued"));std::vector<uint64_t> qs;for(auto [k,v]:sorted_pairs(truth(s,queued)?queued:missing))qs.push_back(tup({raw(k),raw(v)}));r.push_back(tup(qs));r.push_back(open?raw(get(pco,"source_card_id")):nil);r.push_back(decision_context(sort_dice));r.push_back(public_context());auto result=tup(r);std::set<uint64_t> seen;return hashable(result,seen)?result:nil;
    }catch(const std::exception&){return nil;}}
    uint64_t prj(uint64_t n){if(ctx_none(s,n)||s.nodes[n].tag!=WOTR_DICT)return boolean(!ctx_none(s,n));std::vector<uint64_t> r;for(auto [k,v]:sorted_pairs(n)){uint64_t projected;auto tag=s.nodes[v].tag;if(tag<=WOTR_STRING||tag==WOTR_BIGINT)projected=raw(v);else if(tag==WOTR_LIST||tag==WOTR_TUPLE){std::vector<uint64_t> xs;for(auto c:items(v)){auto ct=s.nodes[c].tag;xs.push_back((ct==WOTR_BOOL||ct==WOTR_INT||ct==WOTR_FLOAT||ct==WOTR_STRING||ct==WOTR_BIGINT)?raw(c):str(name(c)==missing?type_name(c):string(name(c))));}projected=tup(xs);}else projected=str(name(v)==missing?type_name(v):string(name(v)));r.push_back(tup({str(string(k)),projected}));}return tup(r);}
    uint64_t hand(int side){auto e=enum_value_edge(s,attr("hands"),"game_state.Side",side?"SHADOW":"FREE_PEOPLES");require(e!=missing,"Search hand missing");return obs_attr(s,s.edges[e],"hand");}
    uint64_t chance_key(int root,uint64_t sorted_state){uint64_t hands=nil;try{if(root<0)hands=tup({sorted_strings(hand(0)),sorted_strings(hand(1))});else{auto other=items(hand(1-root));int chars=0;for(auto c:other)chars+=s.nodes[c].tag==WOTR_STRING&&text(s,c).find("_Character_")!=std::string_view::npos;hands=tup({sorted_strings(hand(root)),integer(chars),integer(other.size()-chars)});}}catch(const std::exception&){}std::vector<uint64_t> pub;for(auto [k,v]:sorted_pairs(truth(s,attr("discard_face_up"))?attr("discard_face_up"):missing))pub.push_back(tup({str(string(k)),sorted_strings(v)}));auto winner=attr("winner");return tup({sorted_state,tup(pub),prj(attr("pending_combat")),prj(attr("pending_hunt")),prj(attr("pending_discard")),hands,raw(attr("_search_roll_ms")),boolean(truth(s,attr("game_over"))),str(truth(s,winner)?string(winner):""),prj(attr("pending_card_option")),integer(static_cast<int64_t>(ctx_int(s,attr("_pending_hunt_damage")))),prj(attr("pending_guide_change")),integer(items(attr("_hobbit_place_queue")).size()),boolean(truth(s,attr("_reveal_place_due")))});}
    uint64_t metadata(int root){std::vector<uint64_t> dice;for(int side=0;side<2;++side){auto ds=items(pa_pool(s,side));int unused=0,face=0;for(auto d:ds){bool free=!truth(s,obs_attr(s,d,"used"));unused+=free;face+=free&&!ctx_none(s,obs_attr(s,d,"current"));}dice.push_back(tup({integer(ds.size()),integer(unused),integer(face)}));}std::vector<uint64_t> flags,zones;for(auto k:{"_declare_phase","_eye_allocation_phase","_fellowship_tail_phase","_reveal_place_due"})flags.push_back(boolean(truth(s,attr(k))));for(auto k:{"fp_char_deck","fp_strat_deck","sh_char_deck","sh_strat_deck","hunt_pool"})zones.push_back(integer(items(attr(k)).size()));auto crossing=attr("_search_turn_crossings"),pc=attr("pending_combat");bool isdict=!ctx_none(s,pc)&&s.nodes[pc].tag==WOTR_DICT;auto rounds=isdict?get(pc,"rounds_fought"):missing;auto commit=tup({boolean(isdict),isdict?raw(get(pc,"atk_combat_card")):nil,isdict?raw(get(pc,"_commit_nonce")):nil,rounds==missing?integer(0):raw(rounds)});return container(WOTR_DICT,{str("dice_counts"),tup(dice),str("phase_flags"),tup(flags),str("dice_rolled"),raw(attr("dice_rolled")),str("turn_crossings"),crossing==missing?integer(0):raw(crossing),str("endturn_victory"),raw(attr("_search_endturn_victory")),str("roll_multiset"),raw(attr("_search_roll_ms")),str("own_hand"),root<0?nil:sorted_strings(hand(root)),str("hidden_zone_counts"),tup(zones),str("commit_context"),commit});}
};
std::unique_ptr<Snapshot> native_search_metadata(const Snapshot& s,int root){require(root>=-1&&root<=1,"Search root side must be None/FP/SH");sm_Builder b(s);auto key=b.state_key(false),sorted=b.state_key(true);b.out.root=b.tup({key,sorted,b.chance_key(root,sorted),b.metadata(root)});return ac_compact(b.out);}
