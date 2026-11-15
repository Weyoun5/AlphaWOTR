// Encoding generation v53 (R2-D5-V53-NATIVE; USER D5 2026-10-22 "implement all d5, including
// retraining" - retraining itself waits for D6). Native mirror of src/encoding_v53.py, whose
// docstring is the normative layout (claude_bugscan_2026-10-21/round2/V53_SPEC.md):
//   observation v53 (4745) = encode_state (v52, 4647, UNCHANGED) ++ v53_state_tail  (98)
//   action      v53  (199) = af_encode    (v52,  157, UNCHANGED) ++ v53_action_tail (42)
// The v52 entry points are not modified: the v53 entry points call them and append the tails,
// so a v52 vector is the bit-identical prefix of its v53 vector. Included inside snapshot.cpp's
// anonymous namespace after action_features.inc (reuses af_* / am_* / ca_partition / glb_* /
// ctx_* helpers); identical in native_backend/cpp and native_backend/live_search_oct19/cpp.
// R4-AF-CHECK-V53: the WOTR_NATIVE_AF_CHECK reference tail + af_check_encode_v53 (before af_encode_v53)
// compile only where WOTR_NATIVE_AF_CHECK_AVAILABLE is defined (the live box's action_features.inc).
// R5-NATIVE-OBS-CHECK-V53: the WOTR_NATIVE_OBS_CHECK reference state tail + obs_check_encode_state_v53 (before
// encode_state_v53) compile only where WOTR_NATIVE_OBS_CHECK_AVAILABLE is defined (the live box's snapshot.cpp).
// Read-only graph queries, no Python callback, failures loud (std::invalid_argument). Every
// float lane is computed in double and rounded to float32 once (the NumPy assignment boundary).
constexpr size_t V53_OBS_PREFIX=4647,V53_STATE_TAIL=98,V53_OBS_DIM=V53_OBS_PREFIX+V53_STATE_TAIL;
constexpr size_t V53_ACT_PREFIX=157,V53_ACTION_TAIL=42,V53_ACT_DIM=V53_ACT_PREFIX+V53_ACTION_TAIL;
constexpr int V53_HAND=0,V53_THREAT=96;                                             // state tail offsets
constexpr int V53_ABILITY=0,V53_FIGURE=1,V53_LEADER=14,V53_REGULAR=22,V53_ELITE=30,V53_VIA=38;  // action tail offsets
constexpr double V53_LEADER_NORM=4.0,V53_UNIT_NORM=10.0;
constexpr const char* V53_OBSERVATION_CONTRACT="v53_hand_threat_context_v1";
constexpr const char* V53_ACTION_CONTRACT="v53_identity_route_choices_v1";
constexpr const char* V53_ACTION_SCHEMA="wotr_action_features_v53_199";
constexpr const char* V53_TDAAN_CARD="FREE_PEOPLES_Strategy_Through_a_Day_and_a_Night";
constexpr const char* V53_WOSES_CARD="FREE_PEOPLES_Strategy_Paths_of_the_Woses";   // R4B-WOSES-SPLIT-ROWS (encoding_v53.WOSES_CARD)
const std::array<const char*,8> V53_NATIONS{{"GONDOR","ROHAN","THE_NORTH","DWARVES","ELVES","ISENGARD","SAURON","SOUTHRON_EASTERLINGS"}};
const std::array<const char*,13> V53_CHARACTERS{{"Gandalf the Grey","Strider","Legolas","Gimli","Boromir","Meriadoc","Peregrin",
    "Aragorn","Gandalf the White","Gollum","Witch King","Saruman","Mouth of Sauron"}};   // == ai._ACTION_CHARACTERS

int v53_nation_index(std::string_view nation){
    for(int i=0;i<8;++i)if(nation==V53_NATIONS[i])return i;
    throw std::invalid_argument("v53 encoding: unknown nation");
}
bool v53_is_nazgul(const Snapshot& s,uint64_t leader){return object_type(s,leader)=="game_state.Nazgul";}
// game_logic.leader_must_follow: a plain Leader (never Nazgul, Companions or the named Minions).
bool v53_must_follow(const Snapshot& s,uint64_t leader){
    const auto type=object_type(s,leader);
    if(type=="game_state.Nazgul"||type=="game_state.Companion")return false;
    const auto name=obs_attr(s,leader,"name",false);
    if(text_is(s,name,"Saruman")||text_is(s,name,"Witch King")||text_is(s,name,"Mouth of Sauron"))return false;
    return type=="game_state.Leader"||type=="game_logic.Minion";
}
// PoliticalTrack.is_at_war: boxes[nation] >= 3 (a missing nation is a KeyError in Python).
bool v53_at_war(const Snapshot& s,std::string_view nation){
    const auto boxes=obs_attr(s,obs_attr(s,s.root,"political"),"boxes");
    const auto value=ctx_nation_value(s,boxes,nation);require(value!=missing,"v53 encoding: political box missing nation");
    return ctx_num(s,value)>=3;
}
// encoding_v53._figure_index: ai._ACTION_CHAR_IDX on the sanitized name (aliases included).
int64_t v53_figure(const Snapshot& a,std::string_view name){
    const auto index=af_map_int(a,"characters",sanitize_name(name),-1);
    if(index<0||index>=13)throw std::invalid_argument("v53 encoding: unknown named figure");
    return index;
}

// ---------------------------------------------------------------------------------------------
// state tail (encoding_v53.state_tail): own hand multi-hot (A1-03) + exact threat counts (A1-02)
// ---------------------------------------------------------------------------------------------
void v53_state_tail(const Snapshot& s,const Snapshot& t,int side,float* out){
    std::fill(out,out+V53_STATE_TAIL,0.0f);
    // [HAND +96]: every card of the ENCODING side's own hand (ruling 1.261); ids that are not
    // real cards (live HU_<side>_... placeholders) light nothing, exactly like the six slots.
    const auto own=ctx_side_value(s,obs_attr(s,s.root,"hands"),side);
    if(!ctx_none(s,own)){
        const auto indices=glb_table(t,"card_indices");
        for(auto cid:glb_items(s,obs_attr(s,own,"hand"))){const auto index=glb_card_index(s,cid,t,indices);if(index>=0)out[V53_HAND+index]=1.0f;}
    }
    // [THREAT +2]: VP settlements with an opposing army UNIT in or adjacent to them (pairwise,
    // per target; leaders alone do not count), / number of such settlements (12 FP, 10 Shadow).
    std::array<std::set<std::string>,2> occupied;
    const auto& armies=typed(s,obs_attr(s,s.root,"armies"),WOTR_DICT,"v53 encoding: armies dictionary required");
    for(uint64_t i=0;i<armies.count;i+=2){
        const auto units=ctx_items(s,obs_attr(s,s.edges[armies.offset+i+1],"units"));if(units.empty())continue;
        const auto key=sanitize_name(text(s,s.edges[armies.offset+i]));
        for(auto u:units)occupied[ctx_unit_side(s,u)].insert(key);
    }
    auto targets=[&](const char* table){
        std::vector<std::string> names;
        for(auto entry:ctx_items(t,glb_table(t,table))){const auto pair=ctx_items(t,entry);require(pair.size()==2,"v53 encoding: invalid VP target tuple");names.emplace_back(text(t,pair[0]));}
        require(!names.empty(),"v53 encoding: empty VP target table");return names;
    };
    const auto fp=targets("fp_vp_targets"),sh=targets("sh_vp_targets");
    const auto adj=glb_table(t,"board_adj");
    auto threatened=[&](const std::vector<std::string>& list,const std::set<std::string>& by){
        int64_t count=0;
        for(const auto& target:list){
            bool hit=by.count(target)!=0;
            if(!hit){const auto neighbors=ctx_get(t,adj,target);if(!ctx_none(t,neighbors))for(auto nb:ctx_items(t,neighbors))if(by.count(std::string(text(t,nb)))){hit=true;break;}}
            count+=hit;
        }
        return count;
    };
    const auto& mine=side==0?fp:sh;const auto& theirs=side==0?sh:fp;
    out[V53_THREAT]=static_cast<float>(static_cast<double>(threatened(mine,occupied[1-side]))/static_cast<double>(mine.size()));
    out[V53_THREAT+1]=static_cast<float>(static_cast<double>(threatened(theirs,occupied[side]))/static_cast<double>(theirs.size()));
}
// The v53 observation tables are the complete v52 tables (observation_contract stays the v52
// prefix contract the untouched encode_state checks) plus the appended v53 keys.
void v53_check_observation_tables(const Snapshot& t){
    require(text_is(t,field(t,t.root,"observation_contract_v53",false),V53_OBSERVATION_CONTRACT),"v53 observation requires v53 observation tables");
    const auto width=field(t,t.root,"feature_count_v53",false),tail=field(t,t.root,"v53_state_tail_dim",false);
    require(width!=missing&&t.nodes[width].tag==WOTR_INT&&t.nodes[width].integer==static_cast<int64_t>(V53_OBS_DIM)&&
            tail!=missing&&t.nodes[tail].tag==WOTR_INT&&t.nodes[tail].integer==static_cast<int64_t>(V53_STATE_TAIL),"v53 observation table layout mismatch");
}
#ifdef WOTR_NATIVE_OBS_CHECK_AVAILABLE
// ---------------------------------------------------------------------------------------------
// >>> R5-NATIVE-OBS-CHECK-V53 reference v53 state tail (WOTR_NATIVE_OBS_CHECK)
// Compiled only in a box whose snapshot.cpp defines WOTR_NATIVE_OBS_CHECK_AVAILABLE (the live box,
// whose encode_state_ref is ROOT's v52 observation copied verbatim; the main tree has no reference
// encoders and compiles this block out). Before R5-NATIVE-OBS-CHECK-V53 the check compared only the
// 4647-wide v52 prefix inside encode_state, so a v53 observation whose 98 state-tail lanes (or whose
// 4745-wide assembly) differed from the reference passed. v53_state_tail_ref is ROOT's production
// v53_state_tail (native_backend/cpp/encoding_v53.inc) VERBATIM apart from `_ref` renames: its own
// name and every helper the live box's v52 reference already copies (obs_attr, glb_*, ctx_*), so it
// shares only the snapshot core helpers with the production tail and the live box's optimized
// observation helpers. src/test_round5_R5_NATIVE_OBS_CHECK_V53.py pins the copy: a change to the
// production state tail is copied here.
void v53_state_tail_ref(const Snapshot& s,const Snapshot& t,int side,float* out){
    std::fill(out,out+V53_STATE_TAIL,0.0f);
    // [HAND +96]: every card of the ENCODING side's own hand (ruling 1.261); ids that are not
    // real cards (live HU_<side>_... placeholders) light nothing, exactly like the six slots.
    const auto own=ctx_side_value_ref(s,obs_attr_ref(s,s.root,"hands"),side);
    if(!ctx_none_ref(s,own)){
        const auto indices=glb_table_ref(t,"card_indices");
        for(auto cid:glb_items_ref(s,obs_attr_ref(s,own,"hand"))){const auto index=glb_card_index_ref(s,cid,t,indices);if(index>=0)out[V53_HAND+index]=1.0f;}
    }
    // [THREAT +2]: VP settlements with an opposing army UNIT in or adjacent to them (pairwise,
    // per target; leaders alone do not count), / number of such settlements (12 FP, 10 Shadow).
    std::array<std::set<std::string>,2> occupied;
    const auto& armies=typed(s,obs_attr_ref(s,s.root,"armies"),WOTR_DICT,"v53 encoding: armies dictionary required");
    for(uint64_t i=0;i<armies.count;i+=2){
        const auto units=ctx_items_ref(s,obs_attr_ref(s,s.edges[armies.offset+i+1],"units"));if(units.empty())continue;
        const auto key=sanitize_name(text(s,s.edges[armies.offset+i]));
        for(auto u:units)occupied[ctx_unit_side_ref(s,u)].insert(key);
    }
    auto targets=[&](const char* table){
        std::vector<std::string> names;
        for(auto entry:ctx_items_ref(t,glb_table_ref(t,table))){const auto pair=ctx_items_ref(t,entry);require(pair.size()==2,"v53 encoding: invalid VP target tuple");names.emplace_back(text(t,pair[0]));}
        require(!names.empty(),"v53 encoding: empty VP target table");return names;
    };
    const auto fp=targets("fp_vp_targets"),sh=targets("sh_vp_targets");
    const auto adj=glb_table_ref(t,"board_adj");
    auto threatened=[&](const std::vector<std::string>& list,const std::set<std::string>& by){
        int64_t count=0;
        for(const auto& target:list){
            bool hit=by.count(target)!=0;
            if(!hit){const auto neighbors=ctx_get_ref(t,adj,target);if(!ctx_none_ref(t,neighbors))for(auto nb:ctx_items_ref(t,neighbors))if(by.count(std::string(text(t,nb)))){hit=true;break;}}
            count+=hit;
        }
        return count;
    };
    const auto& mine=side==0?fp:sh;const auto& theirs=side==0?sh:fp;
    out[V53_THREAT]=static_cast<float>(static_cast<double>(threatened(mine,occupied[1-side]))/static_cast<double>(mine.size()));
    out[V53_THREAT+1]=static_cast<float>(static_cast<double>(threatened(theirs,occupied[side]))/static_cast<double>(theirs.size()));
}
// <<< R5-NATIVE-OBS-CHECK-V53 reference v53 state tail
uint64_t obs_check_v53_observations_checked=0;   // v53 observations whose 4745 lanes matched the reference (reported by the controller at destroy)
void obs_check_encode_state_v53(const Snapshot& s,const Snapshot& tables,int32_t side,const std::vector<float>& out,const char* fast_error){
    // Reference observation = encode_state_ref (ROOT v52 observation) ++ v53_state_tail_ref, built in the production
    // order (the v52 prefix first: encode_state runs before the tail) into a fresh buffer and compared bit for bit with
    // the ASSEMBLED buffer encode_state_v53 is about to publish, over all V53_OBS_DIM lanes. An exception on one side
    // must be matched by the same text on the other (the caller rethrows the optimized exception).
    require(out.size()==V53_OBS_DIM,"WOTR_NATIVE_OBS_CHECK: v53 observation buffer width mismatch");
    std::vector<float> ref(V53_OBS_DIM,0.0f);std::string ref_error;
    try{
        encode_state_ref(s,tables,side,ref.data(),V53_OBS_PREFIX);
        v53_state_tail_ref(s,tables,side,ref.data()+V53_OBS_PREFIX);
    }catch(const std::exception& e){ref_error=e.what();if(ref_error.empty())ref_error="<empty>";}
    const std::string fast(fast_error?fast_error:"");
    if(fast!=ref_error)throw std::logic_error("WOTR_NATIVE_OBS_CHECK: v53 exception mismatch: optimized='"+fast+"' reference='"+ref_error+"'");
    if(!ref_error.empty())return;
    for(size_t i=0;i<V53_OBS_DIM;++i){
        uint32_t u,v;std::memcpy(&u,out.data()+i,4);std::memcpy(&v,ref.data()+i,4);
        if(u!=v)throw std::logic_error("WOTR_NATIVE_OBS_CHECK: v53 observation feature "+std::to_string(i)+(i<V53_OBS_PREFIX?std::string(" (v52 prefix)"):" (v53 state tail lane "+std::to_string(i-V53_OBS_PREFIX)+")")+" differs: optimized "+obs_check_hex(u)+" reference "+obs_check_hex(v));
    }
    obs_check_v53_observations_checked++;
}
#endif
void encode_state_v53(const Snapshot& s,const Snapshot& tables,int32_t side,float* output,size_t count){
    require(side==0||side==1,"invalid observation side");
    require(count==V53_OBS_DIM,"v53 observation output count must be 4745");
    check_array(output,count);
    v53_check_observation_tables(tables);
    std::vector<float> out(V53_OBS_DIM,0.0f);
    encode_state(s,tables,side,out.data(),V53_OBS_PREFIX);   // the v52 observation, unchanged (checked there under WOTR_NATIVE_OBS_CHECK)
    const auto tail=[&](){v53_state_tail(s,tables,side,out.data()+V53_OBS_PREFIX);};
#ifdef WOTR_NATIVE_OBS_CHECK_AVAILABLE
    if(obs_check_active){   // R5-NATIVE-OBS-CHECK-V53 (WOTR_NATIVE_OBS_CHECK=1): the assembled 4745 lanes against the reference; any bit or exception-text difference throws
        std::string fast_error;bool failed=false;std::exception_ptr pending;
        try{tail();}
        catch(const std::exception& e){fast_error=e.what();if(fast_error.empty())fast_error="<empty>";failed=true;pending=std::current_exception();}
        obs_check_encode_state_v53(s,tables,side,out,failed?fast_error.c_str():nullptr);
        if(failed)std::rethrow_exception(pending);
    }else tail();
#else
    tail();
#endif
    std::memcpy(output,out.data(),count*sizeof(float));      // refusal leaves the output untouched
}

// ---------------------------------------------------------------------------------------------
// action tail (encoding_v53.action_tail)
// ---------------------------------------------------------------------------------------------
struct v53_Roster{bool present=false;std::vector<uint64_t> units,leaders;};
// encoding_v53._write_roster: named figures by identity, plain Leaders / Regulars / Elites per
// nation; Nazgul stay on the v52 composition column (fungible, ruling 1.128).
void v53_write_roster(const Snapshot& s,const Snapshot& a,const v53_Roster& roster,float* out){
    std::array<int64_t,8> regular{},elite{},leader{};
    for(auto u:roster.units){const auto i=v53_nation_index(ctx_nation(s,obs_attr(s,u,"nation")));(enum_is(s,obs_attr(s,u,"kind"),"game_state.UnitKind","REGULAR")?regular:elite)[i]++;}
    for(auto l:roster.leaders){
        const auto key=am_figure_key(s,l,false);const auto& kind=std::get<0>(key);
        if(kind=="nazgul")continue;
        if(kind=="named")out[V53_FIGURE+v53_figure(a,std::get<1>(key))]=1.0f;
        else if(kind=="leader")++leader[v53_nation_index(std::get<1>(key))];
        else throw std::invalid_argument("v53 encoding: unexpected leader figure");
    }
    for(int i=0;i<8;++i){
        out[V53_LEADER+i]=static_cast<float>(static_cast<double>(leader[i])/V53_LEADER_NORM);
        out[V53_REGULAR+i]=static_cast<float>(static_cast<double>(regular[i])/V53_UNIT_NORM);
        out[V53_ELITE+i]=static_cast<float>(static_cast<double>(elite[i])/V53_UNIT_NORM);
    }
}
std::vector<uint64_t> v53_first(const std::vector<uint64_t>& seq,int64_t count){
    const auto n=static_cast<size_t>(std::max<int64_t>(0,std::min<int64_t>(count,static_cast<int64_t>(seq.size()))));
    return std::vector<uint64_t>(seq.begin(),seq.begin()+static_cast<ptrdiff_t>(n));
}
// ai._border_pred9: on a voluntary move into another Nation's borders a non-belligerent figure
// may not cross (W4-038; 1.213 a figure of the destination's own Nation may always enter).
struct v53_Border{bool active=false;std::string dest;};
v53_Border v53_border_pred9(const Snapshot& s,const Snapshot& t,const std::vector<uint64_t>& own_units,const std::vector<uint64_t>& leaders,int side,const std::string& to){
    std::set<std::string> nations;
    for(auto u:own_units)nations.insert(ctx_nation(s,obs_attr(s,u,"nation")));
    for(auto l:leaders)if(v53_must_follow(s,l)&&am_leader_side(s,l)==side)nations.insert(ctx_nation(s,obs_attr(s,l,"nation")));
    bool all=true;for(const auto& n:nations)if(!v53_at_war(s,n)){all=false;break;}
    if(all)return {};
    const auto dest=ctx_get(t,glb_table(t,"region_nations"),to);   // ai._region_nation
    if(ctx_none(t,dest))return {};
    return {true,ctx_nation(t,dest)};
}
bool v53_nation_ok(const Snapshot& s,const v53_Border& border,uint64_t nation){
    if(!border.active)return true;const auto name=ctx_nation(s,nation);
    return v53_at_war(s,name)||name==border.dest;
}
// army_move_participants.selected: the nation-filtered, counted figures ordinary movement moves.
v53_Roster v53_selected(const Snapshot& s,const Snapshot& m,uint64_t p,const std::vector<uint64_t>& army_units,const std::vector<uint64_t>& army_leaders,int side,const v53_Border& border){
    v53_Roster out;out.present=true;
    std::vector<uint64_t> units;
    for(auto u:army_units)if(ctx_unit_side(s,u)==side&&v53_nation_ok(s,border,obs_attr(s,u,"nation")))units.push_back(u);
    const std::array<std::pair<const char*,const char*>,2> kinds{{{"ELITE","n_elite"},{"REGULAR","n_regular"}}};
    for(const auto& kind:kinds){
        std::vector<uint64_t> pool;for(auto u:units)if(enum_is(s,obs_attr(s,u,"kind"),"game_state.UnitKind",kind.first))pool.push_back(u);
        const auto value=af_get(m,p,kind.second);
        const int64_t count=ctx_none(m,value)?static_cast<int64_t>(pool.size()):af_integer(m,value);
        require(count>=0&&count<=static_cast<int64_t>(pool.size()),"Movement exceeds eligible army units");
        out.units.insert(out.units.end(),pool.begin(),pool.begin()+static_cast<ptrdiff_t>(count));
    }
    require(!out.units.empty(),"Army movement requires an eligible unit");
    std::vector<uint64_t> leaders;
    for(auto l:army_leaders)if(am_leader_side(s,l)==side&&!am_saruman(s,l)&&(!border.active||!v53_must_follow(s,l)||v53_nation_ok(s,border,obs_attr(s,l,"nation"))))leaders.push_back(l);
    const auto nazgul_value=af_get(m,p,"n_nazgul"),leader_value=af_get(m,p,"n_leaders"),take=af_get(m,p,"take_leaders");
    const bool has_nazgul=!ctx_none(m,nazgul_value);const int64_t nazgul_count=has_nazgul?af_integer(m,nazgul_value):0;
    const int64_t count=!ctx_none(m,leader_value)?af_integer(m,leader_value):((take==missing||truth(m,take))?static_cast<int64_t>(leaders.size()):0);
    require(count>=0&&(!has_nazgul||nazgul_count>=0),"Negative movement leader count");
    std::vector<uint64_t> ordered,nazgul_pool,others;
    for(auto l:leaders)if(v53_must_follow(s,l))ordered.push_back(l);
    for(auto l:leaders)if(!v53_must_follow(s,l))ordered.push_back(l);
    for(auto l:leaders)if(v53_is_nazgul(s,l))nazgul_pool.push_back(l);
    std::set<uint64_t> chosen;std::vector<uint64_t> nazgul;
    const auto leader_set=af_get(m,p,"leader_set");
    if(!ctx_none(m,leader_set)){
        // E5-01: the row NAMES its non-Nazgul figures (first matches in list order); Nazgul stay on n_nazgul.
        std::vector<uint64_t> pool;for(auto l:leaders)if(!v53_is_nazgul(s,l))pool.push_back(l);
        const auto named=am_pick(s,pool,am_parse_figure_set(m,leader_set,false),false);
        require(!ctx_none(m,leader_value)&&static_cast<int64_t>(named.size())==count,"leader_set disagrees with n_leaders");
        chosen.insert(named.begin(),named.end());
        if(has_nazgul)nazgul=v53_first(nazgul_pool,nazgul_count);
    }else if(!has_nazgul){
        const auto first=v53_first(ordered,count);chosen.insert(first.begin(),first.end());
    }else{
        nazgul=v53_first(nazgul_pool,nazgul_count);
        for(auto l:ordered)if(!v53_is_nazgul(s,l))others.push_back(l);
        const auto first=v53_first(others,count);chosen.insert(first.begin(),first.end());
    }
    out.leaders=nazgul;for(auto l:leaders)if(chosen.count(l))out.leaders.push_back(l);
    return out;
}
int v53_acting_side(const Snapshot& m,uint64_t p,int side){
    const auto tag=af_get(m,p,"side");
    if(text_is(m,tag,"fp"))return 0;
    if(text_is(m,tag,"sh"))return 1;
    return side;
}
// encoding_v53.army_move_roster: the executor's own selection (border predicate over the FULL
// source army, the preexisting view on the second Army-die leg, then selected).
v53_Roster v53_army_move_roster(const Snapshot& s,const Snapshot& t,const Snapshot& m,uint64_t p,int side){
    const auto from=af_get(m,p,"from");const auto src=af_army(s,m,from);   // ai._armies_lookup
    if(src==missing)return {};
    const int act=v53_acting_side(m,p,side);
    const auto frm=sanitize_name(af_string(m,from)),to=sanitize_name(af_string(m,af_get(m,p,"to")));
    const auto units=ctx_items(s,obs_attr(s,src,"units")),leaders=ctx_items(s,obs_attr(s,src,"leaders"));
    v53_Border border;
    if(frm!=to){std::vector<uint64_t> own;for(auto u:units)if(ctx_unit_side(s,u)==act)own.push_back(u);border=v53_border_pred9(s,t,own,leaders,act,to);}
    const auto pending=obs_attr(s,s.root,"pending_army_move_die",false),pending_from=obs_attr(s,s.root,"pending_army_move_from",false);
    if(!ctx_none(s,pending)&&frm==sanitize_name(truth(s,pending_from)?af_string(s,pending_from):std::string())){
        const auto view=am_preexisting(s,src,obs_attr(s,s.root,"pending_army_move_preexist",false));   // army_move_participants.preexisting
        return v53_selected(s,m,p,view.first,view.second,act,border);
    }
    return v53_selected(s,m,p,units,leaders,act,border);
}
// encoding_v53.victor_advance_roster: the battle participants (army_participants.
// partition_battle_army == ca_partition) filled exactly as game_logic.move_army fills the counts.
v53_Roster v53_victor_advance_roster(const Snapshot& s,const Snapshot& m,uint64_t p){
    const auto pc=obs_attr(s,s.root,"pending_combat",false);
    const auto from=af_get(m,p,"from");const auto src=af_army(s,m,from);
    if(src==missing||!truth(s,pc))return {};
    const int vside=text_is(m,af_get(m,p,"side"),"fp")?0:1;
    const auto frm=sanitize_name(af_string(m,from));
    const auto parts=ca_partition(s,src,frm,vside,pc);
    const std::set<uint64_t> unit_ids(parts.units.begin(),parts.units.end()),leader_ids(parts.leaders.begin(),parts.leaders.end());
    std::set<uint64_t> chosen_units,chosen_leaders;bool has_unit_set=false,has_leader_set=false;
    const auto unit_set=af_get(m,p,"unit_set");
    if(!ctx_none(m,unit_set)){
        has_unit_set=true;std::vector<uint64_t> pool;for(auto u:parts.units)if(ctx_unit_side(s,u)==vside)pool.push_back(u);
        const auto picked=am_pick(s,pool,am_parse_figure_set(m,unit_set,true),true);chosen_units.insert(picked.begin(),picked.end());
    }
    const auto leader_set=af_get(m,p,"leader_set");
    if(!ctx_none(m,leader_set)){
        has_leader_set=true;std::vector<uint64_t> pool;for(auto l:parts.leaders)if(!v53_is_nazgul(s,l)&&!am_saruman(s,l)&&am_leader_side(s,l)==vside)pool.push_back(l);
        const auto picked=am_pick(s,pool,am_parse_figure_set(m,leader_set,false),false);chosen_leaders.insert(picked.begin(),picked.end());
    }
    auto take=[](const std::vector<uint64_t>& seq,int64_t qty,auto pred){
        std::vector<uint64_t> moved;for(auto x:seq)if(qty&&pred(x)&&static_cast<int64_t>(moved.size())<qty)moved.push_back(x);return moved;
    };
    auto requested=[&](const char* key){const auto v=af_get(m,p,key);return truth(m,v)?af_integer(m,v):int64_t(0);};   // int(params.get(key) or 0)
    std::vector<uint64_t> own_elite,own_regular;
    const auto units=ctx_items(s,obs_attr(s,src,"units"));
    for(auto u:units)if(ctx_unit_side(s,u)==vside&&unit_ids.count(u)&&enum_is(s,obs_attr(s,u,"kind"),"game_state.UnitKind","ELITE"))own_elite.push_back(u);
    for(auto u:units)if(ctx_unit_side(s,u)==vside&&unit_ids.count(u)&&enum_is(s,obs_attr(s,u,"kind"),"game_state.UnitKind","REGULAR"))own_regular.push_back(u);
    auto unit_ok=[&](uint64_t u){return !has_unit_set||chosen_units.count(u)!=0;};
    v53_Roster out;out.present=true;
    out.units=take(own_elite,requested("n_elite"),unit_ok);
    const auto regulars=take(own_regular,requested("n_regular"),unit_ok);out.units.insert(out.units.end(),regulars.begin(),regulars.end());
    const auto leaders=ctx_items(s,obs_attr(s,src,"leaders"));
    auto movable=[&](uint64_t l){return !am_saruman(s,l)&&leader_ids.count(l)!=0&&am_leader_side(s,l)==vside&&(!has_leader_set||v53_is_nazgul(s,l)||chosen_leaders.count(l)!=0);};
    const int64_t take_all=truth(m,af_get(m,p,"take_leaders"))?99:0;
    const auto leader_value=af_get(m,p,"n_leaders"),nazgul_value=af_get(m,p,"n_nazgul");
    int64_t move_leaders=!ctx_none(m,leader_value)?af_integer(m,leader_value):take_all;
    const int64_t move_nazgul=!ctx_none(m,nazgul_value)?af_integer(m,nazgul_value):take_all;
    if(move_leaders){int64_t available=0;for(auto l:leaders)available+=(movable(l)&&!v53_is_nazgul(s,l))?1:0;move_leaders=std::min(move_leaders,available);}
    std::vector<uint64_t> pool;for(auto l:leaders)if(movable(l)&&v53_is_nazgul(s,l))pool.push_back(l);
    out.leaders=v53_first(pool,std::max<int64_t>(0,std::min<int64_t>(move_nazgul,static_cast<int64_t>(pool.size()))));
    const std::set<uint64_t> nazgul(out.leaders.begin(),out.leaders.end());
    std::vector<uint64_t> rest,ordered;for(auto l:leaders)if(!nazgul.count(l))rest.push_back(l);
    for(auto l:rest)if(v53_must_follow(s,l))ordered.push_back(l);
    for(auto l:rest)if(!v53_must_follow(s,l))ordered.push_back(l);
    const auto others=take(ordered,move_leaders,[&](uint64_t l){return movable(l)&&!v53_is_nazgul(s,l);});
    const std::set<uint64_t> other_ids(others.begin(),others.end());
    for(auto l:rest)if(other_ids.count(l))out.leaders.push_back(l);
    return out;
}
// encoding_v53._write_via (E3-02): a Through a Day and a Night row naming its walked-through
// region; [2]/[3] = event_effects._tdaan_walk_effects (the handler's two consequences).
void v53_write_via(const Snapshot& s,const Snapshot& t,const Snapshot& m,const Snapshot& a,uint64_t p,float* out){
    const auto choices=af_get(m,p,"choices");
    if(!truth(m,choices)||m.nodes[choices].tag!=WOTR_DICT)return;
    const auto via=af_get(m,choices,"via_region");
    if(ctx_none(m,via))return;
    require(text_is(m,af_get(m,p,"card_id"),V53_TDAAN_CARD),"v53 encoding: via_region on an unsupported card");
    const auto mid=sanitize_name(af_string(m,via));
    const auto index=af_map_int(a,"region_index",mid,-1);
    require(index>=0,"v53 encoding: unknown via_region");
    const double regions=af_number(a,field(a,a.root,"regions"));
    bool captures=false;
    const auto meta=ctx_get(t,glb_table(t,"board_regions"),mid);
    if(!ctx_none(t,meta)&&(truth(t,ctx_get(t,meta,"is_city"))||truth(t,ctx_get(t,meta,"is_stronghold"))||truth(t,ctx_get(t,meta,"is_town")))){
        const auto controlled=obs_attr(s,s.root,"enemy_controlled");
        if(contains_string(t,glb_table(t,"shadow_region_keys"),mid))captures=!contains_string(s,ctx_side_value(s,controlled,0),mid);
        else if(contains_string(t,glb_table(t,"fp_region_keys"),mid))captures=contains_string(s,ctx_side_value(s,controlled,1),mid);
    }
    bool activates=false;
    const auto nation=ctx_get(t,glb_table(t,"region_nations"),mid);   // ai._region_nation
    if(!ctx_none(t,nation)){
        const auto name=ctx_nation(t,nation);
        if(!fp_nation(name))activates=!truth(s,ctx_nation_value(s,obs_attr(s,obs_attr(s,s.root,"political"),"active"),name));
    }
    out[V53_VIA]=1.0f;
    out[V53_VIA+1]=static_cast<float>(static_cast<double>(index+1)/regions);
    out[V53_VIA+2]=captures?1.0f:0.0f;
    out[V53_VIA+3]=activates?1.0f:0.0f;
}
// R4-TDAAN-SPLIT-UNIT-IDENTITY (D5 E3-02): encoding_v53.tdaan_roster = event_effects.
// _tdaan_move_roster - the figures a Through a Day and a Night card_option row moves: the
// handler's own route (event_through_day.inc etd_row_route, unit_set rows name their units) and
// selection (event_direct_army_move.inc cdm_pick, border rule at the destination and the walked-
// through region), before any overstack resolution. Not present for any other card_option row,
// a row naming no from/to region, a source without an army, or a row that moves nothing.
v53_Roster v53_tdaan_roster(const Snapshot& s,const Snapshot& t,const Snapshot& m,uint64_t p){
    const auto choices=af_get(m,p,"choices");
    if(!text_is(m,af_get(m,p,"card_id"),V53_TDAAN_CARD)||!truth(m,choices)||m.nodes[choices].tag!=WOTR_DICT)return {};
    const auto from_value=af_get(m,choices,"from_region"),to_value=af_get(m,choices,"to_region");
    if(ctx_none(m,from_value)||ctx_none(m,to_value))return {};
    const auto from=sanitize_name(af_string(m,from_value)),to=sanitize_name(af_string(m,to_value));
    const auto src=hr_army(s,from);if(ctx_none(s,src))return {};
    std::vector<uint64_t> units;for(auto u:child_edges(s,obs_attr(s,src,"units")))if(ctx_unit_side(s,u)==0)units.push_back(u);
    const auto r=af_get(m,choices,"n_regular"),e=af_get(m,choices,"n_elite");
    const int64_t regular=ctx_none(m,r)?-1:af_integer(m,r),elite=ctx_none(m,e)?-1:af_integer(m,e);
    const auto via=af_get(m,choices,"via_region"),named=af_get(m,choices,"unit_set");
    am_FigureSet unit_set;const bool has_set=!ctx_none(m,named);if(has_set)unit_set=am_parse_figure_set(m,named,true);
    const bool has_via=!ctx_none(m,via);
    const auto path=etd_row_route(s,t,from,to,units,regular,elite,has_via,has_via?sanitize_name(af_string(m,via)):std::string(),has_set?&unit_set:nullptr);
    if(path.pool.empty())return {};
    const auto picked=cdm_pick(s,t,src,to,0,true,path.mid,regular,elite,has_set?&unit_set:nullptr);
    if(picked.moving.empty())return {};
    v53_Roster out;out.present=true;out.units=picked.moving;out.leaders=picked.leaders;return out;
}
// R4B-WOSES-SPLIT-ROWS: encoding_v53.woses_roster = event_effects._woses_move_roster - the figures a
// Paths of the Woses card_option row moves: the handler's own selection (event_direct_army_move.inc
// cdm_pick, border rule at the destination, the row's counts and unit_set), before any overstack
// resolution. Not present for any other card_option row, a row naming no from/to region, a source
// without an army, or a row that moves nothing.
v53_Roster v53_woses_roster(const Snapshot& s,const Snapshot& t,const Snapshot& m,uint64_t p){
    const auto choices=af_get(m,p,"choices");
    if(!text_is(m,af_get(m,p,"card_id"),V53_WOSES_CARD)||!truth(m,choices)||m.nodes[choices].tag!=WOTR_DICT)return {};
    const auto from_value=af_get(m,choices,"from_region"),to_value=af_get(m,choices,"to_region");
    if(ctx_none(m,from_value)||ctx_none(m,to_value))return {};
    const auto from=sanitize_name(af_string(m,from_value)),to=sanitize_name(af_string(m,to_value));
    const auto src=hr_army(s,from);if(ctx_none(s,src))return {};
    const auto r=af_get(m,choices,"n_regular"),e=af_get(m,choices,"n_elite"),named=af_get(m,choices,"unit_set");
    const int64_t regular=ctx_none(m,r)?-1:af_integer(m,r),elite=ctx_none(m,e)?-1:af_integer(m,e);
    am_FigureSet unit_set;const bool has_set=!ctx_none(m,named);if(has_set)unit_set=am_parse_figure_set(m,named,true);
    const auto picked=cdm_pick(s,t,src,to,0,true,std::string(),regular,elite,has_set?&unit_set:nullptr);
    if(picked.moving.empty())return {};
    v53_Roster out;out.present=true;out.units=picked.moving;out.leaders=picked.leaders;return out;
}
void v53_action_tail(const Snapshot& s,const Snapshot& t,const Snapshot& m,const Snapshot& a,uint64_t row,int side,float* out){
    std::fill(out,out+V53_ACTION_TAIL,0.0f);
    const auto triplet=child_edges(m,row);require(triplet.size()==3,"Action feature row needs three fields");
    const auto kind=text(m,triplet[1]);const auto p=triplet[2];typed(m,p,WOTR_DICT,"Action feature params must be mapping");
    if(kind=="advance_political"){
        // A1-01: the Legolas / Gimli / Boromir ability advance (rulings 1.106 / 1.107 / 1.125)
        const auto name=af_get(m,p,"companion_ability");
        if(truth(m,name)){out[V53_ABILITY]=1.0f;out[V53_FIGURE+v53_figure(a,af_string(m,name))]=1.0f;}
    }else if(kind=="army_move"||kind=="victor_advance"){
        const auto roster=kind=="army_move"?v53_army_move_roster(s,t,m,p,side):v53_victor_advance_roster(s,m,p);
        if(roster.present)v53_write_roster(s,a,roster,out);
    }else if(kind=="card_option"){
        // R4-TDAAN-SPLIT-UNIT-IDENTITY: a Through a Day and a Night row's moved figures share the
        // E5-01 lanes (encoding_v53.action_tail), then its route lanes (E3-02); R4B-WOSES-SPLIT-ROWS:
        // a Paths of the Woses row's moved figures share the same lanes
        auto roster=v53_tdaan_roster(s,t,m,p);
        if(!roster.present)roster=v53_woses_roster(s,t,m,p);
        if(roster.present)v53_write_roster(s,a,roster,out);
        v53_write_via(s,t,m,a,p,out);
    }else if(kind=="siege_trim"){
        // R17-ENG-SIEGETRIM (ruling 1.181; ai._v53_siege_trim_lanes): a trim row naming its returned units ('unit_set')
        // lights the per-nation Regular / Elite lanes with them; a row without a set keeps the all-zero tail
        const auto unit_set=af_get(m,p,"unit_set");
        if(!ctx_none(m,unit_set))for(const auto& entry:am_parse_figure_set(m,unit_set,true)){
            const auto& sub=std::get<2>(entry.first);require(sub=="REGULAR"||sub=="ELITE","v53 encoding: unexpected siege trim unit kind");
            out[(sub=="REGULAR"?V53_REGULAR:V53_ELITE)+v53_nation_index(std::get<1>(entry.first))]=static_cast<float>(static_cast<double>(entry.second)/V53_UNIT_NORM);
        }
    }
    // every other action type keeps an all-zero tail
}
// The v53 action tables are the complete v52 row tables (schema / contract / width of the
// untouched af_encode) plus the pinned v53 layout, checked lane by lane against this mirror.
void v53_check_action_tables(const Snapshot& a){
    require(text_is(a,field(a,a.root,"schema"),"wotr_action_features_v52_157")&&text_is(a,field(a,a.root,"action_encoding_contract"),"observed_combat_choices_v8")&&
            af_integer(a,field(a,a.root,"width"))==static_cast<int64_t>(V53_ACT_PREFIX),"v53 action tables need the v52 row tables");
    require(text_is(a,field(a,a.root,"v53_schema",false),V53_ACTION_SCHEMA)&&text_is(a,field(a,a.root,"v53_action_contract",false),V53_ACTION_CONTRACT),
            "v53 action encoding requires v53 action tables");
    auto integer=[&](uint64_t dict,const char* key){const auto v=field(a,dict,key,false);require(v!=missing&&a.nodes[v].tag==WOTR_INT,"v53 action table integer missing");return a.nodes[v].integer;};
    auto real=[&](const char* key){const auto v=field(a,a.root,key,false);require(v!=missing&&a.nodes[v].tag==WOTR_FLOAT,"v53 action table norm missing");return a.nodes[v].real;};
    require(integer(a.root,"v53_width")==static_cast<int64_t>(V53_ACT_DIM)&&integer(a.root,"v53_prefix_width")==static_cast<int64_t>(V53_ACT_PREFIX),"v53 action width mismatch");
    const auto tail=field(a,a.root,"v53_tail",false);require(tail!=missing&&a.nodes[tail].tag==WOTR_DICT,"v53 action tail layout missing");
    require(integer(tail,"ABILITY")==V53_ABILITY&&integer(tail,"FIGURE")==V53_FIGURE&&integer(tail,"LEADER")==V53_LEADER&&integer(tail,"REGULAR")==V53_REGULAR&&
            integer(tail,"ELITE")==V53_ELITE&&integer(tail,"VIA")==V53_VIA&&integer(tail,"VIA_DIM")==4&&integer(tail,"TAIL_DIM")==static_cast<int64_t>(V53_ACTION_TAIL),
            "v53 action tail layout differs from the native mirror");
    const auto nations=ctx_items(a,field(a,a.root,"v53_nations"));require(nations.size()==8,"v53 nation order mismatch");
    for(size_t i=0;i<8;++i)require(text_is(a,nations[i],V53_NATIONS[i]),"v53 nation order mismatch");
    const auto characters=ctx_items(a,field(a,a.root,"v53_characters"));require(characters.size()==13,"v53 figure order mismatch");
    for(size_t i=0;i<13;++i)require(text_is(a,characters[i],V53_CHARACTERS[i])&&af_map_int(a,"characters",sanitize_name(V53_CHARACTERS[i]),-1)==static_cast<int64_t>(i),"v53 figure order mismatch");
    require(real("v53_leader_norm")==V53_LEADER_NORM&&real("v53_unit_norm")==V53_UNIT_NORM,"v53 norms mismatch");
    require(text_is(a,field(a,a.root,"v53_tdaan_card",false),V53_TDAAN_CARD),"v53 Through a Day and a Night card mismatch");
}
#ifdef WOTR_NATIVE_AF_CHECK_AVAILABLE
// ---------------------------------------------------------------------------------------------
// >>> R4-AF-CHECK-V53 reference v53 action tail (WOTR_NATIVE_AF_CHECK)
// Compiled only in a box whose action_features.inc defines WOTR_NATIVE_AF_CHECK_AVAILABLE (the
// live box; the main tree has no reference encoders and compiles this block out). Before
// R4-AF-CHECK-V53 the check compared only the 157 v52 columns inside af_encode, so a v53 row whose
// tail lanes (or re-strided prefix) differed from the reference passed. The `*_ref` code below is
// ROOT's production tail (native_backend/cpp/encoding_v53.inc, struct v53_Roster .. v53_action_tail)
// VERBATIM apart from the `_ref` renames and calls no production tail function, so it keeps the
// certified code if a box's tail is ever optimized (as af_ref_encode_row keeps ROOT's v52 row).
// src/test_round4_R4_AF_CHECK_V53.py pins the copy: a change to the production tail is copied here.
struct v53_Roster_ref{bool present=false;std::vector<uint64_t> units,leaders;};
// encoding_v53._write_roster: named figures by identity, plain Leaders / Regulars / Elites per
// nation; Nazgul stay on the v52 composition column (fungible, ruling 1.128).
void v53_write_roster_ref(const Snapshot& s,const Snapshot& a,const v53_Roster_ref& roster,float* out){
    std::array<int64_t,8> regular{},elite{},leader{};
    for(auto u:roster.units){const auto i=v53_nation_index(ctx_nation(s,obs_attr(s,u,"nation")));(enum_is(s,obs_attr(s,u,"kind"),"game_state.UnitKind","REGULAR")?regular:elite)[i]++;}
    for(auto l:roster.leaders){
        const auto key=am_figure_key(s,l,false);const auto& kind=std::get<0>(key);
        if(kind=="nazgul")continue;
        if(kind=="named")out[V53_FIGURE+v53_figure(a,std::get<1>(key))]=1.0f;
        else if(kind=="leader")++leader[v53_nation_index(std::get<1>(key))];
        else throw std::invalid_argument("v53 encoding: unexpected leader figure");
    }
    for(int i=0;i<8;++i){
        out[V53_LEADER+i]=static_cast<float>(static_cast<double>(leader[i])/V53_LEADER_NORM);
        out[V53_REGULAR+i]=static_cast<float>(static_cast<double>(regular[i])/V53_UNIT_NORM);
        out[V53_ELITE+i]=static_cast<float>(static_cast<double>(elite[i])/V53_UNIT_NORM);
    }
}
std::vector<uint64_t> v53_first_ref(const std::vector<uint64_t>& seq,int64_t count){
    const auto n=static_cast<size_t>(std::max<int64_t>(0,std::min<int64_t>(count,static_cast<int64_t>(seq.size()))));
    return std::vector<uint64_t>(seq.begin(),seq.begin()+static_cast<ptrdiff_t>(n));
}
// ai._border_pred9: on a voluntary move into another Nation's borders a non-belligerent figure
// may not cross (W4-038; 1.213 a figure of the destination's own Nation may always enter).
struct v53_Border_ref{bool active=false;std::string dest;};
v53_Border_ref v53_border_pred9_ref(const Snapshot& s,const Snapshot& t,const std::vector<uint64_t>& own_units,const std::vector<uint64_t>& leaders,int side,const std::string& to){
    std::set<std::string> nations;
    for(auto u:own_units)nations.insert(ctx_nation(s,obs_attr(s,u,"nation")));
    for(auto l:leaders)if(v53_must_follow(s,l)&&am_leader_side(s,l)==side)nations.insert(ctx_nation(s,obs_attr(s,l,"nation")));
    bool all=true;for(const auto& n:nations)if(!v53_at_war(s,n)){all=false;break;}
    if(all)return {};
    const auto dest=ctx_get(t,glb_table(t,"region_nations"),to);   // ai._region_nation
    if(ctx_none(t,dest))return {};
    return {true,ctx_nation(t,dest)};
}
bool v53_nation_ok_ref(const Snapshot& s,const v53_Border_ref& border,uint64_t nation){
    if(!border.active)return true;const auto name=ctx_nation(s,nation);
    return v53_at_war(s,name)||name==border.dest;
}
// army_move_participants.selected: the nation-filtered, counted figures ordinary movement moves.
v53_Roster_ref v53_selected_ref(const Snapshot& s,const Snapshot& m,uint64_t p,const std::vector<uint64_t>& army_units,const std::vector<uint64_t>& army_leaders,int side,const v53_Border_ref& border){
    v53_Roster_ref out;out.present=true;
    std::vector<uint64_t> units;
    for(auto u:army_units)if(ctx_unit_side(s,u)==side&&v53_nation_ok_ref(s,border,obs_attr(s,u,"nation")))units.push_back(u);
    const std::array<std::pair<const char*,const char*>,2> kinds{{{"ELITE","n_elite"},{"REGULAR","n_regular"}}};
    for(const auto& kind:kinds){
        std::vector<uint64_t> pool;for(auto u:units)if(enum_is(s,obs_attr(s,u,"kind"),"game_state.UnitKind",kind.first))pool.push_back(u);
        const auto value=af_get(m,p,kind.second);
        const int64_t count=ctx_none(m,value)?static_cast<int64_t>(pool.size()):af_integer(m,value);
        require(count>=0&&count<=static_cast<int64_t>(pool.size()),"Movement exceeds eligible army units");
        out.units.insert(out.units.end(),pool.begin(),pool.begin()+static_cast<ptrdiff_t>(count));
    }
    require(!out.units.empty(),"Army movement requires an eligible unit");
    std::vector<uint64_t> leaders;
    for(auto l:army_leaders)if(am_leader_side(s,l)==side&&!am_saruman(s,l)&&(!border.active||!v53_must_follow(s,l)||v53_nation_ok_ref(s,border,obs_attr(s,l,"nation"))))leaders.push_back(l);
    const auto nazgul_value=af_get(m,p,"n_nazgul"),leader_value=af_get(m,p,"n_leaders"),take=af_get(m,p,"take_leaders");
    const bool has_nazgul=!ctx_none(m,nazgul_value);const int64_t nazgul_count=has_nazgul?af_integer(m,nazgul_value):0;
    const int64_t count=!ctx_none(m,leader_value)?af_integer(m,leader_value):((take==missing||truth(m,take))?static_cast<int64_t>(leaders.size()):0);
    require(count>=0&&(!has_nazgul||nazgul_count>=0),"Negative movement leader count");
    std::vector<uint64_t> ordered,nazgul_pool,others;
    for(auto l:leaders)if(v53_must_follow(s,l))ordered.push_back(l);
    for(auto l:leaders)if(!v53_must_follow(s,l))ordered.push_back(l);
    for(auto l:leaders)if(v53_is_nazgul(s,l))nazgul_pool.push_back(l);
    std::set<uint64_t> chosen;std::vector<uint64_t> nazgul;
    const auto leader_set=af_get(m,p,"leader_set");
    if(!ctx_none(m,leader_set)){
        // E5-01: the row NAMES its non-Nazgul figures (first matches in list order); Nazgul stay on n_nazgul.
        std::vector<uint64_t> pool;for(auto l:leaders)if(!v53_is_nazgul(s,l))pool.push_back(l);
        const auto named=am_pick(s,pool,am_parse_figure_set(m,leader_set,false),false);
        require(!ctx_none(m,leader_value)&&static_cast<int64_t>(named.size())==count,"leader_set disagrees with n_leaders");
        chosen.insert(named.begin(),named.end());
        if(has_nazgul)nazgul=v53_first_ref(nazgul_pool,nazgul_count);
    }else if(!has_nazgul){
        const auto first=v53_first_ref(ordered,count);chosen.insert(first.begin(),first.end());
    }else{
        nazgul=v53_first_ref(nazgul_pool,nazgul_count);
        for(auto l:ordered)if(!v53_is_nazgul(s,l))others.push_back(l);
        const auto first=v53_first_ref(others,count);chosen.insert(first.begin(),first.end());
    }
    out.leaders=nazgul;for(auto l:leaders)if(chosen.count(l))out.leaders.push_back(l);
    return out;
}
int v53_acting_side_ref(const Snapshot& m,uint64_t p,int side){
    const auto tag=af_get(m,p,"side");
    if(text_is(m,tag,"fp"))return 0;
    if(text_is(m,tag,"sh"))return 1;
    return side;
}
// encoding_v53.army_move_roster: the executor's own selection (border predicate over the FULL
// source army, the preexisting view on the second Army-die leg, then selected).
v53_Roster_ref v53_army_move_roster_ref(const Snapshot& s,const Snapshot& t,const Snapshot& m,uint64_t p,int side){
    const auto from=af_get(m,p,"from");const auto src=af_army(s,m,from);   // ai._armies_lookup
    if(src==missing)return {};
    const int act=v53_acting_side_ref(m,p,side);
    const auto frm=sanitize_name(af_string(m,from)),to=sanitize_name(af_string(m,af_get(m,p,"to")));
    const auto units=ctx_items(s,obs_attr(s,src,"units")),leaders=ctx_items(s,obs_attr(s,src,"leaders"));
    v53_Border_ref border;
    if(frm!=to){std::vector<uint64_t> own;for(auto u:units)if(ctx_unit_side(s,u)==act)own.push_back(u);border=v53_border_pred9_ref(s,t,own,leaders,act,to);}
    const auto pending=obs_attr(s,s.root,"pending_army_move_die",false),pending_from=obs_attr(s,s.root,"pending_army_move_from",false);
    if(!ctx_none(s,pending)&&frm==sanitize_name(truth(s,pending_from)?af_string(s,pending_from):std::string())){
        const auto view=am_preexisting(s,src,obs_attr(s,s.root,"pending_army_move_preexist",false));   // army_move_participants.preexisting
        return v53_selected_ref(s,m,p,view.first,view.second,act,border);
    }
    return v53_selected_ref(s,m,p,units,leaders,act,border);
}
// encoding_v53.victor_advance_roster: the battle participants (army_participants.
// partition_battle_army == ca_partition) filled exactly as game_logic.move_army fills the counts.
v53_Roster_ref v53_victor_advance_roster_ref(const Snapshot& s,const Snapshot& m,uint64_t p){
    const auto pc=obs_attr(s,s.root,"pending_combat",false);
    const auto from=af_get(m,p,"from");const auto src=af_army(s,m,from);
    if(src==missing||!truth(s,pc))return {};
    const int vside=text_is(m,af_get(m,p,"side"),"fp")?0:1;
    const auto frm=sanitize_name(af_string(m,from));
    const auto parts=ca_partition(s,src,frm,vside,pc);
    const std::set<uint64_t> unit_ids(parts.units.begin(),parts.units.end()),leader_ids(parts.leaders.begin(),parts.leaders.end());
    std::set<uint64_t> chosen_units,chosen_leaders;bool has_unit_set=false,has_leader_set=false;
    const auto unit_set=af_get(m,p,"unit_set");
    if(!ctx_none(m,unit_set)){
        has_unit_set=true;std::vector<uint64_t> pool;for(auto u:parts.units)if(ctx_unit_side(s,u)==vside)pool.push_back(u);
        const auto picked=am_pick(s,pool,am_parse_figure_set(m,unit_set,true),true);chosen_units.insert(picked.begin(),picked.end());
    }
    const auto leader_set=af_get(m,p,"leader_set");
    if(!ctx_none(m,leader_set)){
        has_leader_set=true;std::vector<uint64_t> pool;for(auto l:parts.leaders)if(!v53_is_nazgul(s,l)&&!am_saruman(s,l)&&am_leader_side(s,l)==vside)pool.push_back(l);
        const auto picked=am_pick(s,pool,am_parse_figure_set(m,leader_set,false),false);chosen_leaders.insert(picked.begin(),picked.end());
    }
    auto take=[](const std::vector<uint64_t>& seq,int64_t qty,auto pred){
        std::vector<uint64_t> moved;for(auto x:seq)if(qty&&pred(x)&&static_cast<int64_t>(moved.size())<qty)moved.push_back(x);return moved;
    };
    auto requested=[&](const char* key){const auto v=af_get(m,p,key);return truth(m,v)?af_integer(m,v):int64_t(0);};   // int(params.get(key) or 0)
    std::vector<uint64_t> own_elite,own_regular;
    const auto units=ctx_items(s,obs_attr(s,src,"units"));
    for(auto u:units)if(ctx_unit_side(s,u)==vside&&unit_ids.count(u)&&enum_is(s,obs_attr(s,u,"kind"),"game_state.UnitKind","ELITE"))own_elite.push_back(u);
    for(auto u:units)if(ctx_unit_side(s,u)==vside&&unit_ids.count(u)&&enum_is(s,obs_attr(s,u,"kind"),"game_state.UnitKind","REGULAR"))own_regular.push_back(u);
    auto unit_ok=[&](uint64_t u){return !has_unit_set||chosen_units.count(u)!=0;};
    v53_Roster_ref out;out.present=true;
    out.units=take(own_elite,requested("n_elite"),unit_ok);
    const auto regulars=take(own_regular,requested("n_regular"),unit_ok);out.units.insert(out.units.end(),regulars.begin(),regulars.end());
    const auto leaders=ctx_items(s,obs_attr(s,src,"leaders"));
    auto movable=[&](uint64_t l){return !am_saruman(s,l)&&leader_ids.count(l)!=0&&am_leader_side(s,l)==vside&&(!has_leader_set||v53_is_nazgul(s,l)||chosen_leaders.count(l)!=0);};
    const int64_t take_all=truth(m,af_get(m,p,"take_leaders"))?99:0;
    const auto leader_value=af_get(m,p,"n_leaders"),nazgul_value=af_get(m,p,"n_nazgul");
    int64_t move_leaders=!ctx_none(m,leader_value)?af_integer(m,leader_value):take_all;
    const int64_t move_nazgul=!ctx_none(m,nazgul_value)?af_integer(m,nazgul_value):take_all;
    if(move_leaders){int64_t available=0;for(auto l:leaders)available+=(movable(l)&&!v53_is_nazgul(s,l))?1:0;move_leaders=std::min(move_leaders,available);}
    std::vector<uint64_t> pool;for(auto l:leaders)if(movable(l)&&v53_is_nazgul(s,l))pool.push_back(l);
    out.leaders=v53_first_ref(pool,std::max<int64_t>(0,std::min<int64_t>(move_nazgul,static_cast<int64_t>(pool.size()))));
    const std::set<uint64_t> nazgul(out.leaders.begin(),out.leaders.end());
    std::vector<uint64_t> rest,ordered;for(auto l:leaders)if(!nazgul.count(l))rest.push_back(l);
    for(auto l:rest)if(v53_must_follow(s,l))ordered.push_back(l);
    for(auto l:rest)if(!v53_must_follow(s,l))ordered.push_back(l);
    const auto others=take(ordered,move_leaders,[&](uint64_t l){return movable(l)&&!v53_is_nazgul(s,l);});
    const std::set<uint64_t> other_ids(others.begin(),others.end());
    for(auto l:rest)if(other_ids.count(l))out.leaders.push_back(l);
    return out;
}
// encoding_v53._write_via (E3-02): a Through a Day and a Night row naming its walked-through
// region; [2]/[3] = event_effects._tdaan_walk_effects (the handler's two consequences).
void v53_write_via_ref(const Snapshot& s,const Snapshot& t,const Snapshot& m,const Snapshot& a,uint64_t p,float* out){
    const auto choices=af_get(m,p,"choices");
    if(!truth(m,choices)||m.nodes[choices].tag!=WOTR_DICT)return;
    const auto via=af_get(m,choices,"via_region");
    if(ctx_none(m,via))return;
    require(text_is(m,af_get(m,p,"card_id"),V53_TDAAN_CARD),"v53 encoding: via_region on an unsupported card");
    const auto mid=sanitize_name(af_string(m,via));
    const auto index=af_map_int(a,"region_index",mid,-1);
    require(index>=0,"v53 encoding: unknown via_region");
    const double regions=af_number(a,field(a,a.root,"regions"));
    bool captures=false;
    const auto meta=ctx_get(t,glb_table(t,"board_regions"),mid);
    if(!ctx_none(t,meta)&&(truth(t,ctx_get(t,meta,"is_city"))||truth(t,ctx_get(t,meta,"is_stronghold"))||truth(t,ctx_get(t,meta,"is_town")))){
        const auto controlled=obs_attr(s,s.root,"enemy_controlled");
        if(contains_string(t,glb_table(t,"shadow_region_keys"),mid))captures=!contains_string(s,ctx_side_value(s,controlled,0),mid);
        else if(contains_string(t,glb_table(t,"fp_region_keys"),mid))captures=contains_string(s,ctx_side_value(s,controlled,1),mid);
    }
    bool activates=false;
    const auto nation=ctx_get(t,glb_table(t,"region_nations"),mid);   // ai._region_nation
    if(!ctx_none(t,nation)){
        const auto name=ctx_nation(t,nation);
        if(!fp_nation(name))activates=!truth(s,ctx_nation_value(s,obs_attr(s,obs_attr(s,s.root,"political"),"active"),name));
    }
    out[V53_VIA]=1.0f;
    out[V53_VIA+1]=static_cast<float>(static_cast<double>(index+1)/regions);
    out[V53_VIA+2]=captures?1.0f:0.0f;
    out[V53_VIA+3]=activates?1.0f:0.0f;
}
// R4-TDAAN-SPLIT-UNIT-IDENTITY (D5 E3-02): encoding_v53.tdaan_roster = event_effects.
// _tdaan_move_roster - the figures a Through a Day and a Night card_option row moves: the
// handler's own route (event_through_day.inc etd_row_route, unit_set rows name their units) and
// selection (event_direct_army_move.inc cdm_pick, border rule at the destination and the walked-
// through region), before any overstack resolution. Not present for any other card_option row,
// a row naming no from/to region, a source without an army, or a row that moves nothing.
v53_Roster_ref v53_tdaan_roster_ref(const Snapshot& s,const Snapshot& t,const Snapshot& m,uint64_t p){
    const auto choices=af_get(m,p,"choices");
    if(!text_is(m,af_get(m,p,"card_id"),V53_TDAAN_CARD)||!truth(m,choices)||m.nodes[choices].tag!=WOTR_DICT)return {};
    const auto from_value=af_get(m,choices,"from_region"),to_value=af_get(m,choices,"to_region");
    if(ctx_none(m,from_value)||ctx_none(m,to_value))return {};
    const auto from=sanitize_name(af_string(m,from_value)),to=sanitize_name(af_string(m,to_value));
    const auto src=hr_army(s,from);if(ctx_none(s,src))return {};
    std::vector<uint64_t> units;for(auto u:child_edges(s,obs_attr(s,src,"units")))if(ctx_unit_side(s,u)==0)units.push_back(u);
    const auto r=af_get(m,choices,"n_regular"),e=af_get(m,choices,"n_elite");
    const int64_t regular=ctx_none(m,r)?-1:af_integer(m,r),elite=ctx_none(m,e)?-1:af_integer(m,e);
    const auto via=af_get(m,choices,"via_region"),named=af_get(m,choices,"unit_set");
    am_FigureSet unit_set;const bool has_set=!ctx_none(m,named);if(has_set)unit_set=am_parse_figure_set(m,named,true);
    const bool has_via=!ctx_none(m,via);
    const auto path=etd_row_route(s,t,from,to,units,regular,elite,has_via,has_via?sanitize_name(af_string(m,via)):std::string(),has_set?&unit_set:nullptr);
    if(path.pool.empty())return {};
    const auto picked=cdm_pick(s,t,src,to,0,true,path.mid,regular,elite,has_set?&unit_set:nullptr);
    if(picked.moving.empty())return {};
    v53_Roster_ref out;out.present=true;out.units=picked.moving;out.leaders=picked.leaders;return out;
}
// R4B-WOSES-SPLIT-ROWS: encoding_v53.woses_roster = event_effects._woses_move_roster - the figures a
// Paths of the Woses card_option row moves: the handler's own selection (event_direct_army_move.inc
// cdm_pick, border rule at the destination, the row's counts and unit_set), before any overstack
// resolution. Not present for any other card_option row, a row naming no from/to region, a source
// without an army, or a row that moves nothing.
v53_Roster_ref v53_woses_roster_ref(const Snapshot& s,const Snapshot& t,const Snapshot& m,uint64_t p){
    const auto choices=af_get(m,p,"choices");
    if(!text_is(m,af_get(m,p,"card_id"),V53_WOSES_CARD)||!truth(m,choices)||m.nodes[choices].tag!=WOTR_DICT)return {};
    const auto from_value=af_get(m,choices,"from_region"),to_value=af_get(m,choices,"to_region");
    if(ctx_none(m,from_value)||ctx_none(m,to_value))return {};
    const auto from=sanitize_name(af_string(m,from_value)),to=sanitize_name(af_string(m,to_value));
    const auto src=hr_army(s,from);if(ctx_none(s,src))return {};
    const auto r=af_get(m,choices,"n_regular"),e=af_get(m,choices,"n_elite"),named=af_get(m,choices,"unit_set");
    const int64_t regular=ctx_none(m,r)?-1:af_integer(m,r),elite=ctx_none(m,e)?-1:af_integer(m,e);
    am_FigureSet unit_set;const bool has_set=!ctx_none(m,named);if(has_set)unit_set=am_parse_figure_set(m,named,true);
    const auto picked=cdm_pick(s,t,src,to,0,true,std::string(),regular,elite,has_set?&unit_set:nullptr);
    if(picked.moving.empty())return {};
    v53_Roster_ref out;out.present=true;out.units=picked.moving;out.leaders=picked.leaders;return out;
}
void v53_action_tail_ref(const Snapshot& s,const Snapshot& t,const Snapshot& m,const Snapshot& a,uint64_t row,int side,float* out){
    std::fill(out,out+V53_ACTION_TAIL,0.0f);
    const auto triplet=child_edges(m,row);require(triplet.size()==3,"Action feature row needs three fields");
    const auto kind=text(m,triplet[1]);const auto p=triplet[2];typed(m,p,WOTR_DICT,"Action feature params must be mapping");
    if(kind=="advance_political"){
        // A1-01: the Legolas / Gimli / Boromir ability advance (rulings 1.106 / 1.107 / 1.125)
        const auto name=af_get(m,p,"companion_ability");
        if(truth(m,name)){out[V53_ABILITY]=1.0f;out[V53_FIGURE+v53_figure(a,af_string(m,name))]=1.0f;}
    }else if(kind=="army_move"||kind=="victor_advance"){
        const auto roster=kind=="army_move"?v53_army_move_roster_ref(s,t,m,p,side):v53_victor_advance_roster_ref(s,m,p);
        if(roster.present)v53_write_roster_ref(s,a,roster,out);
    }else if(kind=="card_option"){
        // R4-TDAAN-SPLIT-UNIT-IDENTITY: a Through a Day and a Night row's moved figures share the
        // E5-01 lanes (encoding_v53.action_tail), then its route lanes (E3-02); R4B-WOSES-SPLIT-ROWS:
        // a Paths of the Woses row's moved figures share the same lanes
        auto roster=v53_tdaan_roster_ref(s,t,m,p);
        if(!roster.present)roster=v53_woses_roster_ref(s,t,m,p);
        if(roster.present)v53_write_roster_ref(s,a,roster,out);
        v53_write_via_ref(s,t,m,a,p,out);
    }else if(kind=="siege_trim"){
        // R17-ENG-SIEGETRIM (ruling 1.181; ai._v53_siege_trim_lanes): a trim row naming its returned units ('unit_set')
        // lights the per-nation Regular / Elite lanes with them; a row without a set keeps the all-zero tail
        const auto unit_set=af_get(m,p,"unit_set");
        if(!ctx_none(m,unit_set))for(const auto& entry:am_parse_figure_set(m,unit_set,true)){
            const auto& sub=std::get<2>(entry.first);require(sub=="REGULAR"||sub=="ELITE","v53 encoding: unexpected siege trim unit kind");
            out[(sub=="REGULAR"?V53_REGULAR:V53_ELITE)+v53_nation_index(std::get<1>(entry.first))]=static_cast<float>(static_cast<double>(entry.second)/V53_UNIT_NORM);
        }
    }
    // every other action type keeps an all-zero tail
}
// <<< R4-AF-CHECK-V53 reference v53 action tail
uint64_t af_check_v53_rows_checked=0;   // v53 rows whose 199 columns and primary region matched the reference (reported by the controller at destroy)
void af_check_encode_v53(const Snapshot& s,const Snapshot& t,const Snapshot& menu,const Snapshot& tables,int side,const std::vector<uint64_t>& actions,size_t rows,size_t count,const std::vector<float>& features,const std::vector<int64_t>& primary,const char* fast_error){
    // Reference row = ROOT v52 row (af_ref_encode_row) ++ reference tail (v53_action_tail_ref), built in the production
    // order (every v52 row first: af_encode runs before the tails) and compared with the ASSEMBLED rows af_encode_v53
    // returns over all 199 columns plus the primary region. An exception on one side must be matched by the same text
    // on the other (the caller rethrows the optimized exception).
    af_ref_Batch ref(s,tables);std::vector<float> rf(count,0.0f);std::vector<int64_t> rp(rows,-1);std::string ref_error;
    try{
        for(size_t i=0;i<rows;++i)af_ref_encode_row(s,t,menu,tables,ref,actions[i],side,rf.data()+i*V53_ACT_DIM,rp[i]);
        for(size_t i=0;i<rows;++i)v53_action_tail_ref(s,t,menu,tables,actions[i],side,rf.data()+i*V53_ACT_DIM+V53_ACT_PREFIX);
    }catch(const std::exception& e){ref_error=e.what();if(ref_error.empty())ref_error="<empty>";}
    const std::string fast(fast_error?fast_error:"");
    if(fast!=ref_error)throw std::logic_error("WOTR_NATIVE_AF_CHECK: v53 exception mismatch: optimized='"+fast+"' reference='"+ref_error+"'");
    if(!ref_error.empty())return;
    for(size_t i=0;i<rows;++i){
        const float* a=features.data()+i*V53_ACT_DIM;const float* b=rf.data()+i*V53_ACT_DIM;
        if(std::memcmp(a,b,V53_ACT_DIM*sizeof(float))!=0||primary[i]!=rp[i]){
            size_t col=0;while(col<V53_ACT_DIM&&std::memcmp(a+col,b+col,sizeof(float))==0)++col;
            const std::string where=col>=V53_ACT_DIM?std::string("the primary region"):col<V53_ACT_PREFIX?"feature column "+std::to_string(col)+" (v52 prefix)":"feature column "+std::to_string(col)+" (v53 tail lane "+std::to_string(col-V53_ACT_PREFIX)+")";
            throw std::logic_error("WOTR_NATIVE_AF_CHECK: v53 row "+std::to_string(i)+" ("+std::string(text(menu,menu.edges[menu.nodes[actions[i]].offset+1]))+") differs from the reference encoder at "+where+" (optimized primary "+std::to_string(primary[i])+" vs reference "+std::to_string(rp[i])+")");
        }
    }
    af_check_v53_rows_checked+=rows;
}
#endif
void af_encode_v53(const Snapshot& s,const Snapshot& t,const Snapshot& menu,const Snapshot& tables,int side,float* output,size_t count,int64_t* regions,size_t rows){
    require(side==0||side==1,"Action feature actor must be FP or SH");
    v53_check_action_tables(tables);
    const auto actions=child_edges(menu,menu.root);
    require(actions.size()==rows&&rows<=std::numeric_limits<size_t>::max()/V53_ACT_DIM&&count==rows*V53_ACT_DIM,"v53 action feature output size mismatch");
    check_array(output,count);check_array(regions,rows);
    std::vector<float> prefix(rows*V53_ACT_PREFIX,0.0f);std::vector<int64_t> primary(rows,-1);
    af_encode(s,t,menu,tables,side,prefix.data(),prefix.size(),primary.data(),rows);   // the v52 rows, unchanged
    std::vector<float> features(count,0.0f);
    const auto assemble=[&](){
        for(size_t i=0;i<rows;++i){
            std::copy(prefix.begin()+static_cast<ptrdiff_t>(i*V53_ACT_PREFIX),prefix.begin()+static_cast<ptrdiff_t>((i+1)*V53_ACT_PREFIX),features.begin()+static_cast<ptrdiff_t>(i*V53_ACT_DIM));
            v53_action_tail(s,t,menu,tables,actions[i],side,features.data()+i*V53_ACT_DIM+V53_ACT_PREFIX);
        }
    };
#ifdef WOTR_NATIVE_AF_CHECK_AVAILABLE
    if(af_check_active){   // R4-AF-CHECK-V53 (WOTR_NATIVE_AF_CHECK=1): the assembled 199-wide rows against the reference; any byte or exception-text difference throws
        std::string fast_error;bool failed=false;std::exception_ptr pending;
        try{assemble();}
        catch(const std::exception& e){fast_error=e.what();if(fast_error.empty())fast_error="<empty>";failed=true;pending=std::current_exception();}
        af_check_encode_v53(s,t,menu,tables,side,actions,rows,count,features,primary,failed?fast_error.c_str():nullptr);
        if(failed)std::rethrow_exception(pending);
    }else assemble();
#else
    assemble();
#endif
    if(count)std::memcpy(output,features.data(),count*sizeof(float));
    if(rows)std::memcpy(regions,primary.data(),rows*sizeof(int64_t));
}
