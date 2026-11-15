// Unincluded v52 direct action encoder. No graph mutation or Python callback.
// All arithmetic is evaluated in double, then assigned to float32 exactly at
// the original NumPy assignment boundary. Existing column meanings stay fixed.
std::string af_string(const Snapshot& s,uint64_t node){
    if(ctx_none(s,node))return "None";const auto& n=s.nodes[node];
    if(n.tag==WOTR_STRING)return std::string(text(s,node));
    if(n.tag==WOTR_INT)return std::to_string(n.integer);
    if(n.tag==WOTR_BOOL)return n.integer?"True":"False";
    if(n.tag==WOTR_ENUM){const auto children=child_edges(s,node);std::string type(text(s,children[0]));return type.substr(type.rfind('.')+1)+"."+std::string(text(s,children[1]));}
    throw std::invalid_argument("Action feature string conversion type is not ported");
}
double af_number(const Snapshot& s,uint64_t node,double fallback=0){
    if(node==missing)return fallback;const auto& n=s.nodes[node];
    if(n.tag==WOTR_INT||n.tag==WOTR_BOOL)return static_cast<double>(n.integer);
    if(n.tag==WOTR_FLOAT)return n.real;
    throw std::invalid_argument("Action feature requires a numeric value");
}
int64_t af_integer(const Snapshot& s,uint64_t node,int64_t fallback=0){
    if(node==missing)return fallback;const auto v=af_number(s,node);require(std::isfinite(v)&&v>=-9223372036854775808.0&&v<9223372036854775808.0,"Action feature integer out of range");return static_cast<int64_t>(v);
}
uint64_t af_get(const Snapshot& s,uint64_t dict,const char* key){return ap_get(s,dict,key);}
uint64_t af_or(const Snapshot& s,uint64_t dict,std::initializer_list<const char*> keys){for(const auto key:keys){const auto v=af_get(s,dict,key);if(truth(s,v))return v;}return missing;}
std::string af_text_or(const Snapshot& s,uint64_t dict,const char* key,std::string fallback={}){const auto v=af_get(s,dict,key);return v==missing?fallback:af_string(s,v);}
int64_t af_map_int(const Snapshot& tables,const char* map,std::string_view key,int64_t fallback=-1){const auto v=ctx_get(tables,ctx_get(tables,tables.root,map),key);return ctx_none(tables,v)?fallback:typed(tables,v,WOTR_INT,"Action table integer required").integer;}
std::string af_map_text(const Snapshot& tables,const char* map,std::string_view key,std::string fallback={}){const auto v=ctx_get(tables,ctx_get(tables,tables.root,map),key);return ctx_none(tables,v)?fallback:std::string(text(tables,v));}
int af_offset(const Snapshot& tables,const char* name){return static_cast<int>(af_map_int(tables,"offsets",name));}
uint64_t af_army(const Snapshot& s,const Snapshot& m,uint64_t region){
    if(ctx_none(m,region))return missing;const auto armies=obs_attr(s,s.root,"armies");
    const auto key=af_string(m,region);if(m.nodes[region].tag==WOTR_STRING){const auto exact=ctx_get(s,armies,key);if(!ctx_none(s,exact))return exact;}
    return mu_army(s,sanitize_name(key));
}
struct af_Composition{int64_t regular=0,elite=0,leaders=0;};
af_Composition af_composition(const Snapshot& s,uint64_t army,int side){
    af_Composition out;if(ctx_none(s,army))return out;
    for(auto u:child_edges(s,obs_attr(s,army,"units")))if(ctx_unit_side(s,u)==side)(enum_is(s,obs_attr(s,u,"kind"),"game_state.UnitKind","REGULAR")?out.regular:out.elite)++;
    for(auto l:child_edges(s,obs_attr(s,army,"leaders")))out.leaders+=cp_leader_side(s,l)==side;return out;
}
// One immutable batch can share roster scans and source-name lookup. This is
// discarded after every call, so no mutable-state or process-history key is
// introduced. Non-string diagnostic inputs keep the uncached source lookup.
// Speed pass (controller/engine_patches/action_features.inc.patch): every memo below is a pure
// function of (state, action tables, observation tables, the row's own text) that the row code
// already read; keys are string_views into the menu bytes (alive for the whole call) or into
// `owned`.  Lazily resolved entries raise exactly where the per-row code raised.
bool af_cursor_active(const Snapshot&);
struct af_Batch {
    const Snapshot& state;
    const Snapshot& metadata;
    std::unordered_map<std::string_view,uint64_t> armies;
    std::unordered_map<std::string_view,int64_t> character_indices;
    struct Target {int64_t region;float stronghold,city,distance;int side;};
    std::unordered_map<std::string_view,Target> targets;
    std::unordered_map<int64_t,float> faces;
    bool cursor_ready=false,cursor_active=false,hobbit_ready=false,hobbit=false;
    std::array<std::unordered_map<uint64_t,af_Composition>,2> compositions;
    std::array<std::unordered_map<uint64_t,at_Formation>,2> formations;
    int comp,chars,src,tgt,size,optkind,optval,term,leave,cardtail;
    // memo storage (speed pass)
    static constexpr int64_t NONE_INDEX=std::numeric_limits<int64_t>::min();
    std::vector<std::unique_ptr<std::string>> owned;                        // stable storage for keys that do not view menu bytes (<deque> is not among snapshot.cpp's headers)
    std::unordered_map<std::string_view,int64_t> region_memo;                // raw name -> region_index value (NONE_INDEX: table has none)
    struct KindMeta{int64_t catindex;int64_t subtype;bool subtype_ready;};
    std::unordered_map<std::string_view,KindMeta> kind_memo;                 // action type -> (category index, subtypes index resolved on first use)
    enum Off{OFF_TABLEKILL=0,OFF_TGT_MYSIDE,OFF_TGT_ENEMYSIDE,OFF_DIEFACE,OFF_CARDOPT,OFF_SKIP,OFF_EMPTIES,OFF_MORDOR,OFF_COUNT};
    std::array<int,OFF_COUNT> lazy_offsets;                                  // -2 = unresolved
    std::unordered_map<uint64_t,std::pair<int64_t,int64_t>> move_leaders;    // army -> (own leaders, nazgul) as army_move take_leaders counts them
    std::unordered_map<uint64_t,int64_t> nazgul_counts;                      // army -> Nazgul leader count
    bool regions_ready=false;double regions_value=0;int64_t hobbit_subtype=-2;
    af_Batch(const Snapshot& s,const Snapshot& tables):state(s),metadata(tables){
        auto at=[&](const char* name,int width=1){const auto value=af_offset(tables,name);require(value>=0&&width<=133-value,"Action feature offset exceeds row");return value;};
        comp=at("_ACT_COMP_OFF",4);chars=at("_ACT_CHAR_OFF",13);src=at("_ACT_SRC_OFF",3);tgt=at("_ACT_TGT_OFF",3);size=at("_ACT_OPTSIZE_OFF");optkind=at("_ACT_OPTKIND_OFF");optval=at("_ACT_OPTVAL_OFF");term=at("_ACT_OPTTERM_OFF");leave=at("_ACT_LEAVE_OFF");cardtail=at("_ACT_CARD_CHOICE_TAIL_OFF",12);
        lazy_offsets.fill(-2);
    }
    std::string_view keep(std::string_view value){owned.push_back(std::make_unique<std::string>(value));return std::string_view(*owned.back());}
    // view of a menu value as af_string would render it: string nodes view the menu bytes, other kinds are rendered once into `owned`
    std::string_view view(const Snapshot& m,uint64_t node){if(!ctx_none(m,node)&&m.nodes[node].tag==WOTR_STRING)return text(m,node);return keep(af_string(m,node));}
    int64_t character(std::string_view name){
        const auto found=character_indices.find(name);if(found!=character_indices.end())return found->second;
        const auto index=af_map_int(metadata,"characters",sanitize_name(name));character_indices.emplace(keep(name),index);return index;
    }
    bool combat_active(){if(!cursor_ready){cursor_active=af_cursor_active(state);cursor_ready=true;}return cursor_active;}
    bool hobbit_place(){if(!hobbit_ready){hobbit=text_is(state,af_get(state,obs_attr(state,state.root,"pending_hunt",false),"phase"),"hobbit_place");hobbit_ready=true;}return hobbit;}
    float face(int64_t index,int side){
        if(index<0)return 0;const auto found=faces.find(index);if(found!=faces.end())return found->second;
        float result=0;const auto pool=child_edges(state,pa_pool(state,side));if(static_cast<uint64_t>(index)<pool.size()){const auto current=obs_attr(state,pool[index],"current");if(!ctx_none(state,current))result=static_cast<float>(af_map_int(metadata,"faces",enum_member(state,current,"game_state.ActionDieFace"),0)/7.0);}
        faces.emplace(index,result);return result;
    }
    const Target& target(const Snapshot& tables,std::string_view name,int side){
        const auto found=targets.find(name);if(found!=targets.end())return found->second;
        const auto key=sanitize_name(name);Target result;result.region=af_map_int(metadata,"region_index",key);
        const auto meta=ctx_get(tables,ctx_get(tables,tables.root,"board_regions"),key);
        result.stronghold=truth(tables,af_get(tables,meta,"is_stronghold"));result.city=truth(tables,af_get(tables,meta,"is_city"));
        const auto parent=af_get(tables,meta,"parent_id");result.side=-1;if(!ctx_none(tables,parent)){const auto id=af_integer(tables,parent);result.side=id>=6&&id<=9?1:id>=1&&id<=5?0:-1;}
        const auto distances=ctx_get(tables,tables.root,side==1?"dist_to_fp_target":"dist_to_sh_target");result.distance=static_cast<float>(af_number(tables,ctx_get(tables,distances,key),10)/10.0);
        return targets.emplace(keep(name),result).first->second;
    }
    uint64_t army(const Snapshot& menu,uint64_t region){
        if(ctx_none(menu,region))return missing;
        if(menu.nodes[region].tag!=WOTR_STRING)return af_army(state,menu,region);
        const std::string_view key=text(menu,region);const auto found=armies.find(key);
        if(found!=armies.end())return found->second;
        const auto result=af_army(state,menu,region);armies.emplace(key,result);return result;
    }
    af_Composition composition(uint64_t army,int side){
        if(army==missing)return {};auto& cache=compositions[side];const auto found=cache.find(army);
        if(found!=cache.end())return found->second;
        const auto value=af_composition(state,army,side);cache.emplace(army,value);return value;
    }
    const at_Formation& formation(uint64_t army,int side){
        auto& cache=formations[side];const auto found=cache.find(army);
        if(found!=cache.end())return found->second;
        return cache.emplace(army,at_formation(state,army,side)).first->second;
    }
    // af_number(a, field(a, a.root, "regions")) resolved on first use
    double regions(){if(!regions_ready){regions_value=af_number(metadata,field(metadata,metadata.root,"regions"));regions_ready=true;}return regions_value;}
    // the row's offset(name) lambda: af_offset + require, resolved on first use per name
    int offset(int which){
        static const char* names[OFF_COUNT]={"_ACT_TABLEKILL_OFF","_ACT_TGT_MYSIDE_OFF","_ACT_TGT_ENEMYSIDE_OFF","_ACT_DIEFACE_OFF","_ACT_CARDOPT_OFF","_ACT_SKIP_OFF","_ACT_EMPTIES_OFF","_ACT_MORDOR_OFF"};
        int& slot=lazy_offsets[which];if(slot==-2){const auto i=af_offset(metadata,names[which]);require(i>=0&&i<133,"Invalid action feature offset");slot=i;}return slot;
    }
    // category -> category_index (with the row's validation) per action type; the subtypes index is resolved on first use
    KindMeta& kind_meta(std::string_view kind){
        const auto found=kind_memo.find(kind);if(found!=kind_memo.end())return found->second;
        const auto category=af_map_text(metadata,"category",kind,"other");const auto catindex=af_map_int(metadata,"category_index",category,12);require(catindex>=0&&catindex<14,"Invalid action category");
        return kind_memo.emplace(kind,KindMeta{catindex,-1,false}).first->second;
    }
    int64_t subtype(std::string_view kind,KindMeta& meta){if(!meta.subtype_ready){meta.subtype=af_map_int(metadata,"subtypes",kind);meta.subtype_ready=true;}return meta.subtype;}
    int64_t hobbit_place_subtype(){if(hobbit_subtype==-2)hobbit_subtype=af_map_int(metadata,"subtypes","hobbit_place");return hobbit_subtype;}
    // af_region(a, name, fallback) = af_map_int(a, "region_index", sanitize_name(name), fallback), memoized on the raw name
    int64_t region_name(std::string_view name,int64_t fallback){
        const auto found=region_memo.find(name);if(found!=region_memo.end())return found->second==NONE_INDEX?fallback:found->second;
        const int64_t value=af_map_int(metadata,"region_index",sanitize_name(name),NONE_INDEX);region_memo.emplace(keep(name),value);return value==NONE_INDEX?fallback:value;
    }
    int64_t region(const Snapshot& m,uint64_t node,int64_t fallback){return region_name(view(m,node),fallback);}
    // army_move take_leaders count: (own-side non-Nazgul leaders, Nazgul when side is SH) over the army's leaders
    const std::pair<int64_t,int64_t>& army_move_leaders(uint64_t army,int side){
        const auto found=move_leaders.find(army);if(found!=move_leaders.end())return found->second;
        int64_t nl=0,nz=0;for(auto l:child_edges(state,obs_attr(state,army,"leaders"))){if((object_type(state,l)=="game_state.Nazgul")){if(side==1)++nz;}else if(cp_leader_side(state,l)==side)++nl;}
        return move_leaders.emplace(army,std::make_pair(nl,nz)).first->second;
    }
    int64_t nazgul_count(uint64_t army){
        const auto found=nazgul_counts.find(army);if(found!=nazgul_counts.end())return found->second;
        int64_t n=0;for(auto l:child_edges(state,obs_attr(state,army,"leaders")))n+=(object_type(state,l)=="game_state.Nazgul");
        nazgul_counts.emplace(army,n);return n;
    }
};
std::string af_option_region(const Snapshot& m,uint64_t choices){
    const auto simple=af_or(m,choices,{"region","target_region","to_region","destination","recruit_region","stronghold","witch_king_region","nazgul_destination","boromir_region","strider_region","attack_target","from_region","attack_from","eagles_dest"});
    if(simple!=missing)return af_string(m,simple);
    const auto list=af_or(m,choices,{"strongholds","regions","settlements"});
    if(list!=missing&&(m.nodes[list].tag==WOTR_LIST||m.nodes[list].tag==WOTR_TUPLE)){const auto values=child_edges(m,list);if(!values.empty())return af_string(m,values.front());}return {};
}
std::string af_underscore(std::string s){std::replace(s.begin(),s.end(),' ','_');return s;}
std::string af_nation_alias(std::string s){if(s=="north")return "the_north";if(s=="southrons_and_easterlings")return "southrons_easterlings";return s;}
std::string af_normal(std::string s){s=ev_lower(s);s.erase(std::remove_if(s.begin(),s.end(),[](unsigned char c){return !(c>='a'&&c<='z')&&!(c>='0'&&c<='9');}),s.end());return s;}
int64_t af_region(const Snapshot& a,std::string_view name,int64_t fallback=0){return af_map_int(a,"region_index",sanitize_name(name),fallback);}
int64_t af_region_strict(const Snapshot& m,const Snapshot& a,uint64_t value){require(value!=missing&&m.nodes[value].tag==WOTR_STRING,"Unknown card-choice region");const auto result=af_map_int(a,"region_index",text(m,value));require(result>=0,"Unknown card-choice region");return result;}
int64_t af_count(const Snapshot& m,uint64_t value,int64_t maximum){require(value!=missing&&m.nodes[value].tag==WOTR_INT,"Invalid card-choice troop count");const auto n=m.nodes[value].integer;require(n>=0&&n<=maximum,"Invalid card-choice troop count");return n;}
void af_card_tail(const Snapshot& m,const Snapshot& a,uint64_t choices,float* out){
    const auto nr=static_cast<double>(typed(a,field(a,a.root,"regions"),WOTR_INT,"Action region count missing").integer);
    std::vector<uint64_t> lists;for(const char* key:{"strongholds","settlements","regions"}){const auto v=af_get(m,choices,key);if(v!=missing)lists.push_back(v);}require(lists.size()<=1,"Ambiguous multi-region card choice");
    if(!lists.empty()){const auto list=lists[0];require(m.nodes[list].tag==WOTR_LIST||m.nodes[list].tag==WOTR_TUPLE,"Card-choice region list required");const auto values=child_edges(m,list);require(values.size()>=1&&values.size()<=3,"Card-choice region count invalid");for(size_t i=0;i<values.size();++i){const auto region=(af_region_strict(m,a,values[i])+1)/nr;if(i)out[i-1]=static_cast<float>(region);}}
    const auto recruit=af_get(m,choices,"recruit_counts");if(recruit!=missing){
        require(lists.size()==1&&(m.nodes[recruit].tag==WOTR_LIST||m.nodes[recruit].tag==WOTR_TUPLE),"Invalid regional recruitment allocation");const auto counts=child_edges(m,recruit);require(counts.size()==child_edges(m,lists[0]).size()&&counts.size()<=3&&af_get(m,choices,"rage_split")==missing&&af_get(m,choices,"dice_faces")==missing,"Invalid regional recruitment allocation");int64_t total=0;
        for(size_t i=0;i<counts.size();++i){const auto n=af_count(m,counts[i],3);require(n>0,"Invalid regional recruitment count");total+=n;out[2+i]=static_cast<float>(n/6.0);}require(total<=6,"Invalid regional recruitment total");
    }
    const auto split=af_get(m,choices,"rage_split");if(split!=missing){typed(m,split,WOTR_DICT,"Invalid Dunland split");const auto north=af_count(m,af_get(m,split,"n_north"),4),south=af_count(m,af_get(m,split,"n_south"),4);require(north+south<=4,"Invalid Dunland split total");out[2]=static_cast<float>(north/4.0);out[3]=static_cast<float>(south/4.0);out[4]=static_cast<float>((af_region_strict(m,a,af_get(m,choices,"_rage_dest"))+1)/nr);}
    if(split!=missing&&(af_get(m,split,"n_north_elite")!=missing||af_get(m,split,"n_south_elite")!=missing)){
        require(af_get(m,choices,"n_regular")==missing&&af_get(m,choices,"n_elite")==missing,"Mixed Rage and generic troop compositions");
        out[5]=1;out[6]=static_cast<float>(af_count(m,af_get(m,split,"n_north_elite"),af_count(m,af_get(m,split,"n_north"),4))/4.0);out[7]=static_cast<float>(af_count(m,af_get(m,split,"n_south_elite"),af_count(m,af_get(m,split,"n_south"),4))/4.0);
    }
    const auto dice=af_get(m,choices,"dice_faces");if(dice!=missing){
        for(const char* key:{"rage_split","n_regular","n_elite","strongholds","settlements","regions"})require(af_get(m,choices,key)==missing,"Mixed Lidless Eye and troop/region choices");
        require(m.nodes[dice].tag==WOTR_LIST||m.nodes[dice].tag==WOTR_TUPLE,"Invalid Lidless Eye die composition");const auto selected=child_edges(m,dice);
        require(selected.size()<=3&&af_count(m,af_get(m,choices,"dice_count"),3)==static_cast<int64_t>(selected.size()),"Invalid Lidless Eye die composition");
        const std::array<const char*,5> faces{"ARMY","CHARACTER","EVENT","MUSTER","MUSTER_ARMY"};std::array<int,5> counts{};
        for(auto face:selected){int index=-1;for(int i=0;i<5;++i)if(text_is(m,face,faces[i]))index=i;require(index>=0,"Invalid Lidless Eye face");++counts[index];}for(int i=0;i<5;++i)out[i]=static_cast<float>(counts[i]/3.0);
    }
    const auto regular=af_get(m,choices,"n_regular"),elite=af_get(m,choices,"n_elite");if(!ctx_none(m,regular)||!ctx_none(m,elite)){out[5]=1;out[6]=static_cast<float>((ctx_none(m,regular)?0:af_count(m,regular,10))/10.0);out[7]=static_cast<float>((ctx_none(m,elite)?0:af_count(m,elite,10))/10.0);}
    const auto fp=af_get(m,choices,"target_fp_army");if(fp!=missing)out[8]=static_cast<float>((af_region_strict(m,a,fp)+1)/nr);
    for(const auto& pair:std::vector<std::pair<const char*,int>>{{"use_hunt_reduction",9},{"wizardstaff_prevent",11}}){const auto value=af_get(m,choices,pair.first);if(value==missing)continue;require((m.nodes[value].tag==WOTR_INT||m.nodes[value].tag==WOTR_BOOL)&&(m.nodes[value].integer==0||m.nodes[value].integer==1),"Invalid binary card choice");out[pair.second]=truth(m,value)?1.0f:-1.0f;}
    const auto deck=af_get(m,choices,"palantir_deck");if(deck!=missing){require(text_is(m,deck,"character")||text_is(m,deck,"strategy"),"Invalid Palantir deck choice");out[10]=text_is(m,deck,"character")?-1.0f:1.0f;}
}
bool af_cursor_active(const Snapshot& s){
    const auto pc=obs_attr(s,s.root,"pending_combat",false),payload=af_get(s,pc,"_round_resume"),cursor=af_get(s,payload,"cursor");
    if(!ctx_none(s,cursor))require(text_is(s,obs_attr(s,cursor,"schema",false),"combat_round_cursor_v1"),"Unknown combat continuation schema");
    const auto effect=af_get(s,pc,"effect_choice");const bool onslaught=text_is(s,af_get(s,effect,"stage"),"onslaught_payment")||(text_is(s,af_get(s,effect,"stage"),"post")&&text_is(s,af_get(s,effect,"effect"),"onslaught"));
    return !ctx_none(s,cursor)||onslaught;
}
void af_cargo_tail(const Snapshot& m,uint64_t choices,float* out){
    const std::array<std::array<const char*,3>,12> categories{{
        {{"unit","ISENGARD","REGULAR"}},{{"unit","ISENGARD","ELITE"}},{{"unit","SAURON","REGULAR"}},{{"unit","SAURON","ELITE"}},{{"unit","SOUTHRON_EASTERLINGS","REGULAR"}},{{"unit","SOUTHRON_EASTERLINGS","ELITE"}},
        {{"nazgul","SAURON",""}},{{"named","Witch King",""}},{{"named","Mouth of Sauron",""}},{{"leader","ISENGARD",""}},{{"leader","SAURON",""}},{{"leader","SOUTHRON_EASTERLINGS",""}}}};
    auto identify=[&](uint64_t row,size_t size){require(m.nodes[row].tag==WOTR_LIST||m.nodes[row].tag==WOTR_TUPLE,"Malformed cargo tuple");const auto values=child_edges(m,row);require(values.size()==size,"Malformed cargo tuple");for(int i=0;i<12;++i)if(text_is(m,values[0],categories[i][0])&&text_is(m,values[1],categories[i][1])&&text_is(m,values[2],categories[i][2]))return i;throw std::invalid_argument("Unknown cargo category");};
    const auto key=af_get(m,choices,"_cargo_key");if(!ctx_none(m,key))out[identify(key,3)]=1;
    const auto cargo=af_get(m,choices,"cargo");if(cargo!=missing){std::set<int> seen;for(auto row:child_edges(m,cargo)){const auto category=identify(row,4);require(seen.insert(category).second,"Duplicate cargo category");const auto count=af_count(m,child_edges(m,row)[3],10);require(count>0,"Zero cargo row invalid");out[12+category]=static_cast<float>(count/10.0);}}
}
void af_combat_tail(const Snapshot& m,const Snapshot& a,std::string_view kind,uint64_t params,bool active,float* out){
    if(!active||kind!="combat_effect_choice")return;
    const auto target=af_get(m,params,"target"),count=af_get(m,params,"count");out[0]=1;out[1]=count!=missing&&!ctx_none(m,count)&&af_number(m,count)==0&&!truth(m,target);out[2]=static_cast<float>(af_number(m,count)/10.0);
    if(truth(m,target)){const auto index=af_map_int(a,"targets",af_normal(af_string(m,target)));require(index>=0&&index<15,"Unknown combat target");out[3+index]=1;}
    const auto nation=af_get(m,params,"nation");if(nation!=missing){auto name=m.nodes[nation].tag==WOTR_ENUM?std::string(enum_member(m,nation,"game_state.Nation")):af_string(m,nation);if(name=="NORTH")name="THE_NORTH";const auto values=child_edges(a,field(a,a.root,"payment_nations"));int code=0;for(size_t i=0;i<values.size();++i)if(text_is(a,values[i],name))code=static_cast<int>(i+1);require(code>0,"Unknown combat payment nation");for(int bit=0;bit<4;++bit)out[18+bit]=(code>>bit)&1;}
    const auto unit=af_get(m,params,"kind");if(unit!=missing){require(text_is(m,unit,"regular")||text_is(m,unit,"elite"),"Unknown combat payment kind");out[24+(text_is(m,unit,"elite")?1:0)]=1;}
    const auto operation=af_get(m,params,"operation");if(operation!=missing){require(text_is(m,operation,"remove")||text_is(m,operation,"downgrade"),"Unknown combat payment operation");out[26+(text_is(m,operation,"downgrade")?1:0)]=1;}
    for(int i=0;i<28;++i)require(std::isfinite(out[i]),"Nonfinite combat action");
}
void af_encode_row(const Snapshot& s,const Snapshot& t,const Snapshot& m,const Snapshot& a,af_Batch& batch,uint64_t row,int side,float* f,int64_t& primary){
    const auto& rownode=m.nodes[row];require(rownode.count==3,"Action feature row needs three fields");const auto die=m.edges[rownode.offset],p=m.edges[rownode.offset+2];const std::string_view kind=text(m,m.edges[rownode.offset+1]);typed(m,p,WOTR_DICT,"Action feature params must be mapping");
    const auto offset=[&](int which){return batch.offset(which);};
    const auto comp=batch.comp,chars=batch.chars,src=batch.src,tgt=batch.tgt,size=batch.size,optkind=batch.optkind,optval=batch.optval,term=batch.term,leave=batch.leave;
    const double regions=batch.regions();auto& kind_meta=batch.kind_meta(kind);f[kind_meta.catindex]=1;
    if(kind=="discard_table_card")f[offset(af_Batch::OFF_TABLEKILL)]=1;
    const auto source=af_get(m,p,"from"),target=af_or(m,p,{"to","region","to_region"});const auto army=source==missing?missing:batch.army(m,source);const auto own=truth(m,source)?batch.composition(army,side):af_Composition{};
    if(source!=missing)f[14]=static_cast<float>(batch.region(m,source,0)/regions);
    int64_t target_primary=-1;
    if(target!=missing){const auto& description=batch.target(t,batch.view(m,target),side);target_primary=description.region;f[15]=static_cast<float>(std::max<int64_t>(0,description.region)/regions);f[18]=description.stronghold;f[19]=description.city;f[20]=description.distance;
        if(description.side!=-1)f[offset(description.side==side?af_Batch::OFF_TGT_MYSIDE:af_Batch::OFF_TGT_ENEMYSIDE)]=1;}
    auto mark=[&](std::string_view name){if(name.empty())return;const auto index=batch.character(name);if(index>=0){require(index<13,"Character feature index invalid");f[chars+index]=1;}};
    auto mark_values=[&](uint64_t values){if(!truth(m,values))return;const auto& vn=m.nodes[values];for(uint64_t i=0;i<vn.count;++i)mark(batch.view(m,m.edges[vn.offset+i]));};
    // Separation lattices dominate opening FP menus. These canonical rows
    // read only the target, moving companions, actual die face and hobbit
    // placement flag. Unknown parameters retain the complete encoder below.
    if(kind=="companion_separate"){
        bool canonical=true;const auto& pn=m.nodes[p];
        for(uint64_t i=0;i<pn.count;i+=2){const auto key=text(m,m.edges[pn.offset+i]);canonical&=key=="companion"||key=="companions"||key=="to_region"||key=="side";}
        if(canonical){
            f[33]=-1;const auto name=af_get(m,p,"companion");if(truth(m,name))mark(batch.view(m,name));mark_values(af_get(m,p,"companions"));
            if(!ctx_none(m,die))f[offset(af_Batch::OFF_DIEFACE)]=batch.face(af_integer(m,die),side);
            const auto sub=batch.subtype(kind,kind_meta);if(sub>=0){require(sub<comp-34,"Action subtype index invalid");f[34+sub]=1;}
            else if(batch.hobbit_place())f[34+batch.hobbit_place_subtype()]=1;
            const auto enemy=target==missing?af_Composition{}:batch.composition(batch.army(m,target),1-side);f[tgt]=static_cast<float>(enemy.regular/10.0);f[tgt+1]=static_cast<float>(enemy.elite/10.0);f[tgt+2]=static_cast<float>(enemy.leaders/8.0);
            af_combat_tail(m,a,kind,p,batch.combat_active(),f+105);primary=target_primary;return;
        }
    }
    const auto nation=af_get(m,p,"nation");if(nation!=missing){const auto name=m.nodes[nation].tag==WOTR_ENUM?ev_lower(enum_member(m,nation,"game_state.Nation")):af_underscore(ev_lower(af_string(m,nation)));const std::array<const char*,8> names{"gondor","rohan","the_north","dwarves","elves","isengard","sauron","southron_easterlings"};for(int i=0;i<8;++i)if(name==names[i]){f[17]=static_cast<float>((i+1)/8.0);f[25+i]=1;}}
    f[21]=kind=="army_attack"||kind=="siege_assault"||kind=="siege_press";
    const std::array<const char*,7> faces{"Character","Army","Muster","Event","Muster/Army","Eye of Sauron","Will of the West"};
    auto face_value=[&](uint64_t value){for(size_t i=0;i<faces.size();++i)if(text_is(m,value,faces[i]))return (i+1)/7.0;return 0.0;};
    auto num=[&](const char* key,double fallback=0){return af_number(m,af_get(m,p,key),fallback);};auto intval=[&](const char* key){const auto v=af_get(m,p,key);return truth(m,v)?af_integer(m,v):0;};auto flag=[&](const char* key,bool fallback=false){const auto v=af_get(m,p,key);return v==missing?fallback:truth(m,v);};
    auto casualties=[&](uint64_t op,const char* message){bool valid=false;for(auto v:child_edges(a,field(a,a.root,"casualty_operations")))valid|=text_is(m,op,text(a,v));require(valid,message);f[23]=text_is(m,op,"reduce_elite");f[24]=text_is(m,op,"remove_elite");};
    if(kind=="elven_ring"){f[23]=static_cast<float>(face_value(af_get(m,p,"new_face")));f[24]=static_cast<float>(face_value(af_get(m,p,"old_face")));}
    else if(kind=="eye_allocate")f[23]=static_cast<float>(num("count")/3.0);
    else if(kind=="casualty_allocate")casualties(af_get(m,p,"operation"),"Unversioned or invalid casualty operation");
    else if(kind=="nazgul_move"){const auto requested=af_get(m,p,"count");int64_t n=0;if(ctx_none(m,requested)){if(army!=missing)n=batch.nazgul_count(army);}else n=af_integer(m,requested);f[23]=static_cast<float>(std::min<int64_t>(n,8)/8.0);f[24]=flag("include_wk");}
    else if(kind=="use_token"){f[23]=text_is(m,af_get(m,p,"token_type"),"muster")?0.4f:0.8f;if(flag("deck"))f[24]=text_is(m,af_get(m,p,"deck"),"character")?0.5f:1.0f;}
    else if(kind=="dwarven_ring")f[23]=static_cast<float>(face_value(af_get(m,p,"old_face")));
    else if(kind=="choose_guide"){const auto names=child_edges(a,field(a,a.root,"companion_names"));for(size_t i=0;i<names.size();++i)if(text_is(m,af_get(m,p,"companion"),text(a,names[i])))f[23]=static_cast<float>((i+1)/static_cast<double>(names.size()));}
    if((kind=="army_move"||kind=="victor_advance")&&!ctx_none(m,af_get(m,p,"n_regular"))){f[23]=static_cast<float>((intval("n_regular")+intval("n_elite"))/10.0+intval("n_elite")/100.0);f[24]=flag("take_leaders",true);}
    if(kind=="draw_card")f[24]=text_is(m,af_get(m,p,"deck"),"character")?0.5f:1.0f;
    if(kind=="muster_recruit")f[24]=text_is(m,af_get(m,p,"type"),"leader")?0.5f:0.25f;
    else if(kind=="muster_elite")f[24]=0.75f;
    else if(kind=="muster_minion"){f[24]=1;auto name=af_text_or(m,p,"minion");std::replace(name.begin(),name.end(),'_',' ');name=ev_lower(name);const auto start=name.find_first_not_of(" \t\r\n"),end=name.find_last_not_of(" \t\r\n");name=start==std::string::npos?std::string{}:name.substr(start,end-start+1);const int which=name=="witch king"||name=="witchking"?1:name=="saruman"?2:name=="mouth of sauron"||name=="mouthofsauron"?3:0;f[23]=static_cast<float>(which/3.0);}
    auto card_index=[&](uint64_t node){return ctx_none(m,node)?int64_t(-1):af_map_int(t,"card_indices",af_string(m,node));};f[33]=-1;
    if(kind=="play_event"||kind=="discard_card"||kind=="combat_card_play"||kind=="combat_effect_choice"||kind=="card_option"||kind=="discard_table_card")f[33]=static_cast<float>(card_index(af_get(m,p,"card_id")));
    const auto choices=af_get(m,p,"choices");
    if(kind=="card_option"){
        if(truth(m,choices))typed(m,choices,WOTR_DICT,"Card choices must be a mapping");f[offset(af_Batch::OFF_CARDOPT)]=1;
        if(af_integer(m,af_get(m,choices,"use_hunt_reduction"))==1&&truth(m,af_get(m,choices,"card_id")))f[33]=static_cast<float>(card_index(af_get(m,choices,"card_id")));
        const auto worn=af_get(m,choices,"wwsat_source");if(worn!=missing){require(text_is(m,worn,"table")||text_is(m,worn,"hand")||text_is(m,worn,"decline"),"Invalid Worn source");f[optval]=text_is(m,worn,"table")?1.0f:text_is(m,worn,"hand")?-1.0f:0.5f;if(text_is(m,worn,"table"))f[33]=static_cast<float>(card_index(af_get(m,choices,"card_id")));}
        const auto unit=af_get(m,choices,"unit_type");if(text_is(m,unit,"regular"))f[optkind]=0.34f;else if(text_is(m,unit,"elite"))f[optkind]=0.67f;
        const auto region=af_option_region(m,choices);if(!region.empty())f[15]=static_cast<float>(batch.region_name(region,0)/regions);const auto from=af_or(m,choices,{"from_region","attack_from"});if(truth(m,from))f[14]=static_cast<float>(batch.region(m,from,0)/regions);
        auto nation_choice=[&](uint64_t node,bool value){if(!truth(m,node))return;std::string name=value&&m.nodes[node].tag==WOTR_ENUM?af_map_text(a,"nation_values",enum_member(m,node,"game_state.Nation")):af_string(m,node);name=af_nation_alias(af_underscore(sanitize_name(name)));const std::array<const char*,8> names{"gondor","rohan","the_north","dwarves","elves","isengard","sauron","southrons_easterlings"};for(int i=0;i<8;++i)if(name==names[i])f[25+i]=1;};
        nation_choice(af_get(m,choices,"nation"),true);mark_values(af_get(m,choices,"companions"));const auto victim=af_get(m,choices,"victim");if(truth(m,victim))mark(batch.view(m,victim));
        auto companion=af_text_or(m,choices,"companion_action");if(companion.rfind("sep_",0)==0||companion.rfind("move_",0)==0){companion=companion.substr(companion.find('_')+1);std::replace(companion.begin(),companion.end(),'_',' ');mark(companion);}
        require(af_get(m,choices,"cas_downgrades")==missing,"Legacy card casualty labels require re-extraction");const auto operation=af_get(m,choices,"casualty_operation");if(operation!=missing)casualties(operation,"Invalid card casualty operation");
        for(const auto& field:std::vector<std::pair<const char*,double>>{{"dice_count",3.0},{"count",8.0},{"recruit_count",3.0}}){const auto value=af_get(m,choices,field.first);if(!ctx_none(m,value))f[23]=static_cast<float>(af_number(m,value)/field.second);}
        if(truth(m,af_get(m,choices,"include_wk")))f[24]=1;
        uint64_t nested=missing;for(const char* key:{"overstack_remove","move_group","rage_move","eliminate"}){const auto value=af_get(m,choices,key);if(value!=missing&&m.nodes[value].tag==WOTR_DICT){nested=value;break;}}
        if(nested!=missing){const auto r=af_or(m,nested,{"region","to"});if(r!=missing&&!f[15])f[15]=static_cast<float>(batch.region(m,r,0)/regions);nation_choice(af_get(m,nested,"nation"),false);const auto k=af_get(m,nested,"kind");f[optkind]=text_is(m,k,"regular")?0.34f:text_is(m,k,"elite")?0.67f:text_is(m,k,"leader")?1.0f:0.0f;const auto group=af_get(m,nested,"companions");mark_values(group);if(truth(m,group))f[size]=static_cast<float>(std::min<size_t>(child_edges(m,group).size(),9)/9.0);}
        for(const char* key:{"crebain_discard","balrog_declare_draw","move_to_fellowship","mithril_redraw"}){const auto value=af_get(m,choices,key);if(value!=missing){f[optval]=truth(m,value)?1.0f:-1.0f;break;}}
        const auto action=af_text_or(m,choices,"action");if(action=="eliminate"||action=="hide"||action=="hunt")f[optval]=1;else if(action=="corruption"||action=="move"||action=="discard_table_card")f[optval]=-1;else if(action=="heal_only")f[optval]=0.5f;else if(action=="none")f[optval]=0.25f;
        for(const char* key:{"target_card","army_card"}){const auto value=af_get(m,choices,key);if(truth(m,value)){const auto index=card_index(value);if(index>=0)f[33]=static_cast<float>(index);}}
        const auto character=af_get(m,choices,"character_card");if(truth(m,character)){const auto index=card_index(character);if(index>=0)f[23]=static_cast<float>((index+1)/100.0);}
        for(const char* key:{"_tsim_done","_ffm_done","_rage_done","_ents_chain_done","_naz_done","_comp_done","_tsl_done"})if(truth(m,af_get(m,choices,key)))f[term]=1;if(text_is(m,af_get(m,choices,"destination"),"__remove_from_play__"))f[term]=1;
        const auto next=af_get(m,choices,"ents_chain_play");if(truth(m,next))f[33]=static_cast<float>(card_index(next));
        const auto nr=af_get(m,choices,"n_regular"),ne=af_get(m,choices,"n_elite");if(!ctx_none(m,nr)||!ctx_none(m,ne)){const double r=truth(m,nr)?af_number(m,nr):0,e=truth(m,ne)?af_number(m,ne):0;f[comp]=static_cast<float>(r/10.0);f[comp+1]=static_cast<float>(e/10.0);f[size]=static_cast<float>(std::min(r+e,10.0)/10.0);}
        if(text_is(m,af_get(m,p,"card_id"),"__gandalf_draw__")){const auto source_card=af_get(m,p,"source_card_id");require(source_card!=missing&&m.nodes[source_card].tag==WOTR_STRING,"Gandalf requires source card");const auto index=card_index(source_card);const auto expected=text(m,source_card).find("_Character_")!=std::string_view::npos?"character":"strategy";require(index>=0&&text_is(m,af_get(m,p,"deck"),expected),"Gandalf requires real source and deck");const auto take=af_get(m,choices,"gandalf_draw");typed(m,take,WOTR_BOOL,"Gandalf action must encode draw or decline");f[33]=static_cast<float>(index);f[23]=static_cast<float>(-(index+1)/100.0);f[24]=text_is(m,af_get(m,p,"deck"),"character")?0.5f:1.0f;f[optval]=truth(m,take)?1.0f:-1.0f;}
    }
    const auto skip=offset(af_Batch::OFF_SKIP),dieface=offset(af_Batch::OFF_DIEFACE);if(flag("include_wk"))f[skip]=1;
    if(kind=="skip"){f[skip]=1;const auto face=af_or(m,p,{"face"});f[dieface]=static_cast<float>(af_map_int(a,"faces",face==missing?std::string{}:af_string(m,face),0)/7.0);}
    else if(kind!="pass"&&!ctx_none(m,die))f[dieface]=batch.face(af_integer(m,die),side);
    const auto subtype=batch.subtype(kind,kind_meta);if(subtype>=0){require(subtype<comp-34,"Action subtype index invalid");f[34+subtype]=1;if(kind=="rearguard_split")f[23]=static_cast<float>(num("leave_behind",1)/10.0);else if(kind=="combat_effect_choice"){f[23]=static_cast<float>(num("count")/10.0);if(text_is(m,af_get(m,p,"target"),"leader"))f[optkind]=1;}else if(kind=="shine"||kind=="shine_none")f[23]=kind=="shine";}
    else if(kind=="companion_separate"&&batch.hobbit_place())f[34+batch.hobbit_place_subtype()]=1;
    int64_t nr=0,ne=0,nl=0,nz=0;if(army!=missing){const auto total=own.regular+own.elite;f[16]=static_cast<float>(total/10.0);if((kind=="army_move"||kind=="victor_advance")&&!ctx_none(m,af_get(m,p,"n_regular"))){const auto left=std::max<int64_t>(0,total-intval("n_regular")-intval("n_elite"));f[leave]=static_cast<float>(left/10.0);f[offset(af_Batch::OFF_EMPTIES)]=left==0;}}
    if(kind=="army_move"){nr=intval("n_regular");ne=intval("n_elite");if(flag("take_leaders",true)&&army!=missing){const auto& counts=batch.army_move_leaders(army,side);nl=counts.first;nz=counts.second;}if(!ctx_none(m,af_get(m,p,"n_nazgul")))nz=intval("n_nazgul");if(!ctx_none(m,af_get(m,p,"n_leaders")))nl=intval("n_leaders");}
    else if(kind=="victor_advance"){nr=intval("n_regular");ne=intval("n_elite");nz=intval("n_nazgul");if(!ctx_none(m,af_get(m,p,"n_leaders")))nl=intval("n_leaders");}
    else if(kind=="army_attack"&&army!=missing){const auto& formation=batch.formation(army,side);for(auto u:formation.units)(enum_is(s,obs_attr(s,u,"kind"),"game_state.UnitKind","REGULAR")?nr:ne)++;nl=formation.leaders.size();f[leave]=static_cast<float>((own.regular+own.elite-nr-ne)/10.0);}
    else if(kind=="combat_retreat"){const auto region=af_or(m,p,{"from","to"});const auto retreat=batch.army(m,region);if(retreat!=missing){const auto partition=ca_partition(s,retreat,af_string(m,region),side);for(auto u:partition.units)(enum_is(s,obs_attr(s,u,"kind"),"game_state.UnitKind","REGULAR")?nr:ne)++;nl=partition.leaders.size();}}
    else if(kind=="muster_recruit"){const auto type=af_text_or(m,p,"type","regular");nr=type=="regular";ne=type=="elite";nl=type=="leader";nz=type=="nazgul";if(text_is(m,af_get(m,p,"saruman"),"settlements"))nr=3;}
    else if(kind=="muster_elite")ne=text_is(m,af_get(m,p,"saruman"),"swap2")?2:1;
    else if(kind=="muster_minion"||kind=="mouth_move")nl=1;
    else if(kind=="nazgul_move"){if(army!=missing)nz=batch.nazgul_count(army);if(!ctx_none(m,af_get(m,p,"count")))nz=std::min(nz,intval("count"));}
    else if(kind=="siege_trim"){ne=intval("elites_back");nr=std::max<int64_t>(0,intval("excess")-ne);}
    f[comp]=static_cast<float>(nr/10.0);f[comp+1]=static_cast<float>(ne/10.0);f[comp+2]=static_cast<float>(nl/7.0);f[comp+3]=static_cast<float>(nz/8.0);
    if(kind=="companion_separate"||kind=="choose_guide"){const auto name=af_get(m,p,"companion");if(truth(m,name))mark(batch.view(m,name));mark_values(af_get(m,p,"companions"));}
    else if(kind=="companion_move"){const auto group=af_get(m,p,"companions"),single=af_get(m,p,"companion");mark_values(group);if(truth(m,single))mark(batch.view(m,single));if(!truth(m,group)&&!truth(m,single)&&army!=missing)for(auto l:child_edges(s,obs_attr(s,army,"leaders")))if(object_type(s,l)=="game_state.Companion")mark(text(s,obs_attr(s,l,"name")));}
    else if(kind=="hunt_sacrifice_guide"){const auto name=af_or(m,p,{"forced_companion","companion"});if(name!=missing)mark(batch.view(m,name));else{const auto guide=obs_attr(s,obs_attr(s,s.root,"fellowship"),"guide");if(!ctx_none(s,guide))mark(text(s,obs_attr(s,guide,"name")));}}
    else if(kind=="combat_effect_choice"){const auto name=af_get(m,p,"target");if(truth(m,name)&&!text_is(m,name,"leader"))mark(batch.view(m,name));}
    else if(kind=="muster_minion"){const auto name=af_get(m,p,"minion");if(truth(m,name))mark(batch.view(m,name));}
    else if(kind=="crown_aragorn")mark("Aragorn");else if(kind=="special_character"){const auto name=af_or(m,p,{"character","minion"});if(name!=missing)mark(batch.view(m,name));}
    f[src]=static_cast<float>(own.regular/10.0);f[src+1]=static_cast<float>(own.elite/10.0);f[src+2]=static_cast<float>(own.leaders/8.0);const auto enemy=target==missing?af_Composition{}:batch.composition(batch.army(m,target),1-side);f[tgt]=static_cast<float>(enemy.regular/10.0);f[tgt+1]=static_cast<float>(enemy.elite/10.0);f[tgt+2]=static_cast<float>(enemy.leaders/8.0);
    if(kind=="fellowship_declare"){const auto r=af_or(m,p,{"region","to"});const auto key=r==missing?std::string{}:sanitize_name(af_string(m,r));if(key=="morannon"||key=="minas morgul"||flag("mordor_entry"))f[offset(af_Batch::OFF_MORDOR)]=1;}
    if(kind=="card_option"){af_card_tail(m,a,choices,f+batch.cardtail);af_cargo_tail(m,choices,f+133);}af_combat_tail(m,a,kind,p,batch.combat_active(),f+105);
    primary=-1;if(kind=="card_option"){const auto r=af_option_region(m,choices);if(!r.empty())primary=batch.region_name(r,-1);}if(primary<0)for(const char* key:{"to","to_region","region","from"}){const auto value=af_get(m,p,key);if(truth(m,value)){const auto r=batch.region(m,value,-1);if(r>=0){primary=r;break;}}}
}
// ---------------------------------------------------------------------------------------------------------------
// WOTR_NATIVE_AF_CHECK reference encoder.  `af_ref_Batch` / `af_ref_encode_row` are ROOT cpp/action_features.inc
// lines 42-95 and 133-229 VERBATIM apart from the two renames (assembled by controller/af_draft_tool.py).  When
// `af_check_active` is set (the controller reads WOTR_NATIVE_AF_CHECK=1 at create; never in timed runs) `af_encode`
// runs both encoders on every row and throws on the first differing feature byte, primary region or exception text.
#define WOTR_NATIVE_AF_CHECK_AVAILABLE 1
int af_check_active=0;                 // set by search_controller.inc from WOTR_NATIVE_AF_CHECK=1
uint64_t af_check_rows_checked=0;      // rows encoded twice and found byte-identical (reported by the controller at destroy)
struct af_ref_Batch {
    const Snapshot& state;
    const Snapshot& metadata;
    std::unordered_map<std::string,uint64_t> armies;
    std::unordered_map<std::string,int64_t> character_indices;
    struct Target {int64_t region;float stronghold,city,distance;int side;};
    std::unordered_map<std::string,Target> targets;
    std::unordered_map<int64_t,float> faces;
    bool cursor_ready=false,cursor_active=false,hobbit_ready=false,hobbit=false;
    std::array<std::unordered_map<uint64_t,af_Composition>,2> compositions;
    std::array<std::unordered_map<uint64_t,at_Formation>,2> formations;
    int comp,chars,src,tgt,size,optkind,optval,term,leave,cardtail;
    af_ref_Batch(const Snapshot& s,const Snapshot& tables):state(s),metadata(tables){
        auto at=[&](const char* name,int width=1){const auto value=af_offset(tables,name);require(value>=0&&width<=133-value,"Action feature offset exceeds row");return value;};
        comp=at("_ACT_COMP_OFF",4);chars=at("_ACT_CHAR_OFF",13);src=at("_ACT_SRC_OFF",3);tgt=at("_ACT_TGT_OFF",3);size=at("_ACT_OPTSIZE_OFF");optkind=at("_ACT_OPTKIND_OFF");optval=at("_ACT_OPTVAL_OFF");term=at("_ACT_OPTTERM_OFF");leave=at("_ACT_LEAVE_OFF");cardtail=at("_ACT_CARD_CHOICE_TAIL_OFF",12);
    }
    int64_t character(const std::string& name){
        const auto found=character_indices.find(name);if(found!=character_indices.end())return found->second;
        const auto index=af_map_int(metadata,"characters",sanitize_name(name));character_indices.emplace(name,index);return index;
    }
    bool combat_active(){if(!cursor_ready){cursor_active=af_cursor_active(state);cursor_ready=true;}return cursor_active;}
    bool hobbit_place(){if(!hobbit_ready){hobbit=text_is(state,af_get(state,obs_attr(state,state.root,"pending_hunt",false),"phase"),"hobbit_place");hobbit_ready=true;}return hobbit;}
    float face(int64_t index,int side){
        if(index<0)return 0;const auto found=faces.find(index);if(found!=faces.end())return found->second;
        float result=0;const auto pool=child_edges(state,pa_pool(state,side));if(static_cast<uint64_t>(index)<pool.size()){const auto current=obs_attr(state,pool[index],"current");if(!ctx_none(state,current))result=static_cast<float>(af_map_int(metadata,"faces",enum_member(state,current,"game_state.ActionDieFace"),0)/7.0);}
        faces.emplace(index,result);return result;
    }
    const Target& target(const Snapshot& tables,const std::string& name,int side){
        const auto found=targets.find(name);if(found!=targets.end())return found->second;
        const auto key=sanitize_name(name);Target result;result.region=af_map_int(metadata,"region_index",key);
        const auto meta=ctx_get(tables,ctx_get(tables,tables.root,"board_regions"),key);
        result.stronghold=truth(tables,af_get(tables,meta,"is_stronghold"));result.city=truth(tables,af_get(tables,meta,"is_city"));
        const auto parent=af_get(tables,meta,"parent_id");result.side=-1;if(!ctx_none(tables,parent)){const auto id=af_integer(tables,parent);result.side=id>=6&&id<=9?1:id>=1&&id<=5?0:-1;}
        const auto distances=ctx_get(tables,tables.root,side==1?"dist_to_fp_target":"dist_to_sh_target");result.distance=static_cast<float>(af_number(tables,ctx_get(tables,distances,key),10)/10.0);
        return targets.emplace(name,result).first->second;
    }
    uint64_t army(const Snapshot& menu,uint64_t region){
        if(ctx_none(menu,region))return missing;
        if(menu.nodes[region].tag!=WOTR_STRING)return af_army(state,menu,region);
        const std::string key(text(menu,region));const auto found=armies.find(key);
        if(found!=armies.end())return found->second;
        const auto result=af_army(state,menu,region);armies.emplace(key,result);return result;
    }
    af_Composition composition(uint64_t army,int side){
        if(army==missing)return {};auto& cache=compositions[side];const auto found=cache.find(army);
        if(found!=cache.end())return found->second;
        const auto value=af_composition(state,army,side);cache.emplace(army,value);return value;
    }
    const at_Formation& formation(uint64_t army,int side){
        auto& cache=formations[side];const auto found=cache.find(army);
        if(found!=cache.end())return found->second;
        return cache.emplace(army,at_formation(state,army,side)).first->second;
    }
};
void af_ref_encode_row(const Snapshot& s,const Snapshot& t,const Snapshot& m,const Snapshot& a,af_ref_Batch& batch,uint64_t row,int side,float* f,int64_t& primary){
    const auto triplet=child_edges(m,row);require(triplet.size()==3,"Action feature row needs three fields");const auto die=triplet[0],p=triplet[2];const std::string kind(text(m,triplet[1]));typed(m,p,WOTR_DICT,"Action feature params must be mapping");
    const auto offset=[&](const char* name){const auto i=af_offset(a,name);require(i>=0&&i<133,"Invalid action feature offset");return i;};
    const auto comp=batch.comp,chars=batch.chars,src=batch.src,tgt=batch.tgt,size=batch.size,optkind=batch.optkind,optval=batch.optval,term=batch.term,leave=batch.leave;
    const double regions=af_number(a,field(a,a.root,"regions"));const auto category=af_map_text(a,"category",kind,"other");const auto catindex=af_map_int(a,"category_index",category,12);require(catindex>=0&&catindex<14,"Invalid action category");f[catindex]=1;
    if(kind=="discard_table_card")f[offset("_ACT_TABLEKILL_OFF")]=1;
    const auto source=af_get(m,p,"from"),target=af_or(m,p,{"to","region","to_region"});const auto army=source==missing?missing:batch.army(m,source);const auto own=truth(m,source)?batch.composition(army,side):af_Composition{};
    if(source!=missing)f[14]=static_cast<float>(af_region(a,af_string(m,source))/regions);
    int64_t target_primary=-1;
    if(target!=missing){const auto& description=batch.target(t,af_string(m,target),side);target_primary=description.region;f[15]=static_cast<float>(std::max<int64_t>(0,description.region)/regions);f[18]=description.stronghold;f[19]=description.city;f[20]=description.distance;
        if(description.side!=-1)f[offset(description.side==side?"_ACT_TGT_MYSIDE_OFF":"_ACT_TGT_ENEMYSIDE_OFF")]=1;}
    auto mark=[&](std::string name){if(name.empty())return;const auto index=batch.character(name);if(index>=0){require(index<13,"Character feature index invalid");f[chars+index]=1;}};
    auto mark_values=[&](uint64_t values){if(!truth(m,values))return;for(auto value:child_edges(m,values))mark(af_string(m,value));};
    // Separation lattices dominate opening FP menus. These canonical rows
    // read only the target, moving companions, actual die face and hobbit
    // placement flag. Unknown parameters retain the complete encoder below.
    if(kind=="companion_separate"){
        bool canonical=true;const auto fields=child_edges(m,p);
        for(size_t i=0;i<fields.size();i+=2){const auto key=text(m,fields[i]);canonical&=key=="companion"||key=="companions"||key=="to_region"||key=="side";}
        if(canonical){
            f[33]=-1;const auto name=af_get(m,p,"companion");if(truth(m,name))mark(af_string(m,name));mark_values(af_get(m,p,"companions"));
            if(!ctx_none(m,die))f[offset("_ACT_DIEFACE_OFF")]=batch.face(af_integer(m,die),side);
            const auto sub=af_map_int(a,"subtypes",kind);if(sub>=0){require(sub<comp-34,"Action subtype index invalid");f[34+sub]=1;}
            else if(batch.hobbit_place())f[34+af_map_int(a,"subtypes","hobbit_place")]=1;
            const auto enemy=target==missing?af_Composition{}:batch.composition(batch.army(m,target),1-side);f[tgt]=static_cast<float>(enemy.regular/10.0);f[tgt+1]=static_cast<float>(enemy.elite/10.0);f[tgt+2]=static_cast<float>(enemy.leaders/8.0);
            af_combat_tail(m,a,kind,p,batch.combat_active(),f+105);primary=target_primary;return;
        }
    }
    const auto nation=af_get(m,p,"nation");if(nation!=missing){const auto name=m.nodes[nation].tag==WOTR_ENUM?ev_lower(enum_member(m,nation,"game_state.Nation")):af_underscore(ev_lower(af_string(m,nation)));const std::array<const char*,8> names{"gondor","rohan","the_north","dwarves","elves","isengard","sauron","southron_easterlings"};for(int i=0;i<8;++i)if(name==names[i]){f[17]=static_cast<float>((i+1)/8.0);f[25+i]=1;}}
    f[21]=kind=="army_attack"||kind=="siege_assault"||kind=="siege_press";
    const std::array<const char*,7> faces{"Character","Army","Muster","Event","Muster/Army","Eye of Sauron","Will of the West"};
    auto face_value=[&](uint64_t value){for(size_t i=0;i<faces.size();++i)if(text_is(m,value,faces[i]))return (i+1)/7.0;return 0.0;};
    auto num=[&](const char* key,double fallback=0){return af_number(m,af_get(m,p,key),fallback);};auto intval=[&](const char* key){const auto v=af_get(m,p,key);return truth(m,v)?af_integer(m,v):0;};auto flag=[&](const char* key,bool fallback=false){const auto v=af_get(m,p,key);return v==missing?fallback:truth(m,v);};
    auto casualties=[&](uint64_t op,const char* message){bool valid=false;for(auto v:child_edges(a,field(a,a.root,"casualty_operations")))valid|=text_is(m,op,text(a,v));require(valid,message);f[23]=text_is(m,op,"reduce_elite");f[24]=text_is(m,op,"remove_elite");};
    if(kind=="elven_ring"){f[23]=static_cast<float>(face_value(af_get(m,p,"new_face")));f[24]=static_cast<float>(face_value(af_get(m,p,"old_face")));}
    else if(kind=="eye_allocate")f[23]=static_cast<float>(num("count")/3.0);
    else if(kind=="casualty_allocate")casualties(af_get(m,p,"operation"),"Unversioned or invalid casualty operation");
    else if(kind=="nazgul_move"){const auto requested=af_get(m,p,"count");int64_t n=0;if(ctx_none(m,requested)){if(army!=missing)for(auto l:child_edges(s,obs_attr(s,army,"leaders")))n+=(object_type(s,l)=="game_state.Nazgul");}else n=af_integer(m,requested);f[23]=static_cast<float>(std::min<int64_t>(n,8)/8.0);f[24]=flag("include_wk");}
    else if(kind=="use_token"){f[23]=text_is(m,af_get(m,p,"token_type"),"muster")?0.4f:0.8f;if(flag("deck"))f[24]=text_is(m,af_get(m,p,"deck"),"character")?0.5f:1.0f;}
    else if(kind=="dwarven_ring")f[23]=static_cast<float>(face_value(af_get(m,p,"old_face")));
    else if(kind=="choose_guide"){const auto names=child_edges(a,field(a,a.root,"companion_names"));for(size_t i=0;i<names.size();++i)if(text_is(m,af_get(m,p,"companion"),text(a,names[i])))f[23]=static_cast<float>((i+1)/static_cast<double>(names.size()));}
    if((kind=="army_move"||kind=="victor_advance")&&!ctx_none(m,af_get(m,p,"n_regular"))){f[23]=static_cast<float>((intval("n_regular")+intval("n_elite"))/10.0+intval("n_elite")/100.0);f[24]=flag("take_leaders",true);}
    if(kind=="draw_card")f[24]=text_is(m,af_get(m,p,"deck"),"character")?0.5f:1.0f;
    if(kind=="muster_recruit")f[24]=text_is(m,af_get(m,p,"type"),"leader")?0.5f:0.25f;
    else if(kind=="muster_elite")f[24]=0.75f;
    else if(kind=="muster_minion"){f[24]=1;auto name=af_text_or(m,p,"minion");std::replace(name.begin(),name.end(),'_',' ');name=ev_lower(name);const auto start=name.find_first_not_of(" \t\r\n"),end=name.find_last_not_of(" \t\r\n");name=start==std::string::npos?std::string{}:name.substr(start,end-start+1);const int which=name=="witch king"||name=="witchking"?1:name=="saruman"?2:name=="mouth of sauron"||name=="mouthofsauron"?3:0;f[23]=static_cast<float>(which/3.0);}
    auto card_index=[&](uint64_t node){return ctx_none(m,node)?int64_t(-1):af_map_int(t,"card_indices",af_string(m,node));};f[33]=-1;
    if(kind=="play_event"||kind=="discard_card"||kind=="combat_card_play"||kind=="combat_effect_choice"||kind=="card_option"||kind=="discard_table_card")f[33]=static_cast<float>(card_index(af_get(m,p,"card_id")));
    const auto choices=af_get(m,p,"choices");
    if(kind=="card_option"){
        if(truth(m,choices))typed(m,choices,WOTR_DICT,"Card choices must be a mapping");f[offset("_ACT_CARDOPT_OFF")]=1;
        if(af_integer(m,af_get(m,choices,"use_hunt_reduction"))==1&&truth(m,af_get(m,choices,"card_id")))f[33]=static_cast<float>(card_index(af_get(m,choices,"card_id")));
        const auto worn=af_get(m,choices,"wwsat_source");if(worn!=missing){require(text_is(m,worn,"table")||text_is(m,worn,"hand")||text_is(m,worn,"decline"),"Invalid Worn source");f[optval]=text_is(m,worn,"table")?1.0f:text_is(m,worn,"hand")?-1.0f:0.5f;if(text_is(m,worn,"table"))f[33]=static_cast<float>(card_index(af_get(m,choices,"card_id")));}
        const auto unit=af_get(m,choices,"unit_type");if(text_is(m,unit,"regular"))f[optkind]=0.34f;else if(text_is(m,unit,"elite"))f[optkind]=0.67f;
        const auto region=af_option_region(m,choices);if(!region.empty())f[15]=static_cast<float>(af_region(a,region)/regions);const auto from=af_or(m,choices,{"from_region","attack_from"});if(truth(m,from))f[14]=static_cast<float>(af_region(a,af_string(m,from))/regions);
        auto nation_choice=[&](uint64_t node,bool value){if(!truth(m,node))return;std::string name=value&&m.nodes[node].tag==WOTR_ENUM?af_map_text(a,"nation_values",enum_member(m,node,"game_state.Nation")):af_string(m,node);name=af_nation_alias(af_underscore(sanitize_name(name)));const std::array<const char*,8> names{"gondor","rohan","the_north","dwarves","elves","isengard","sauron","southrons_easterlings"};for(int i=0;i<8;++i)if(name==names[i])f[25+i]=1;};
        nation_choice(af_get(m,choices,"nation"),true);mark_values(af_get(m,choices,"companions"));const auto victim=af_get(m,choices,"victim");if(truth(m,victim))mark(af_string(m,victim));
        auto companion=af_text_or(m,choices,"companion_action");if(companion.rfind("sep_",0)==0||companion.rfind("move_",0)==0){companion=companion.substr(companion.find('_')+1);std::replace(companion.begin(),companion.end(),'_',' ');mark(companion);}
        require(af_get(m,choices,"cas_downgrades")==missing,"Legacy card casualty labels require re-extraction");const auto operation=af_get(m,choices,"casualty_operation");if(operation!=missing)casualties(operation,"Invalid card casualty operation");
        for(const auto& field:std::vector<std::pair<const char*,double>>{{"dice_count",3.0},{"count",8.0},{"recruit_count",3.0}}){const auto value=af_get(m,choices,field.first);if(!ctx_none(m,value))f[23]=static_cast<float>(af_number(m,value)/field.second);}
        if(truth(m,af_get(m,choices,"include_wk")))f[24]=1;
        uint64_t nested=missing;for(const char* key:{"overstack_remove","move_group","rage_move","eliminate"}){const auto value=af_get(m,choices,key);if(value!=missing&&m.nodes[value].tag==WOTR_DICT){nested=value;break;}}
        if(nested!=missing){const auto r=af_or(m,nested,{"region","to"});if(r!=missing&&!f[15])f[15]=static_cast<float>(af_region(a,af_string(m,r))/regions);nation_choice(af_get(m,nested,"nation"),false);const auto k=af_get(m,nested,"kind");f[optkind]=text_is(m,k,"regular")?0.34f:text_is(m,k,"elite")?0.67f:text_is(m,k,"leader")?1.0f:0.0f;const auto group=af_get(m,nested,"companions");mark_values(group);if(truth(m,group))f[size]=static_cast<float>(std::min<size_t>(child_edges(m,group).size(),9)/9.0);}
        for(const char* key:{"crebain_discard","balrog_declare_draw","move_to_fellowship","mithril_redraw"}){const auto value=af_get(m,choices,key);if(value!=missing){f[optval]=truth(m,value)?1.0f:-1.0f;break;}}
        const auto action=af_text_or(m,choices,"action");if(action=="eliminate"||action=="hide"||action=="hunt")f[optval]=1;else if(action=="corruption"||action=="move"||action=="discard_table_card")f[optval]=-1;else if(action=="heal_only")f[optval]=0.5f;else if(action=="none")f[optval]=0.25f;
        for(const char* key:{"target_card","army_card"}){const auto value=af_get(m,choices,key);if(truth(m,value)){const auto index=card_index(value);if(index>=0)f[33]=static_cast<float>(index);}}
        const auto character=af_get(m,choices,"character_card");if(truth(m,character)){const auto index=card_index(character);if(index>=0)f[23]=static_cast<float>((index+1)/100.0);}
        for(const char* key:{"_tsim_done","_ffm_done","_rage_done","_ents_chain_done","_naz_done","_comp_done","_tsl_done"})if(truth(m,af_get(m,choices,key)))f[term]=1;if(text_is(m,af_get(m,choices,"destination"),"__remove_from_play__"))f[term]=1;
        const auto next=af_get(m,choices,"ents_chain_play");if(truth(m,next))f[33]=static_cast<float>(card_index(next));
        const auto nr=af_get(m,choices,"n_regular"),ne=af_get(m,choices,"n_elite");if(!ctx_none(m,nr)||!ctx_none(m,ne)){const double r=truth(m,nr)?af_number(m,nr):0,e=truth(m,ne)?af_number(m,ne):0;f[comp]=static_cast<float>(r/10.0);f[comp+1]=static_cast<float>(e/10.0);f[size]=static_cast<float>(std::min(r+e,10.0)/10.0);}
        if(text_is(m,af_get(m,p,"card_id"),"__gandalf_draw__")){const auto source_card=af_get(m,p,"source_card_id");require(source_card!=missing&&m.nodes[source_card].tag==WOTR_STRING,"Gandalf requires source card");const auto index=card_index(source_card);const auto expected=text(m,source_card).find("_Character_")!=std::string_view::npos?"character":"strategy";require(index>=0&&text_is(m,af_get(m,p,"deck"),expected),"Gandalf requires real source and deck");const auto take=af_get(m,choices,"gandalf_draw");typed(m,take,WOTR_BOOL,"Gandalf action must encode draw or decline");f[33]=static_cast<float>(index);f[23]=static_cast<float>(-(index+1)/100.0);f[24]=text_is(m,af_get(m,p,"deck"),"character")?0.5f:1.0f;f[optval]=truth(m,take)?1.0f:-1.0f;}
    }
    const auto skip=offset("_ACT_SKIP_OFF"),dieface=offset("_ACT_DIEFACE_OFF");if(flag("include_wk"))f[skip]=1;
    if(kind=="skip"){f[skip]=1;const auto face=af_or(m,p,{"face"});f[dieface]=static_cast<float>(af_map_int(a,"faces",face==missing?std::string{}:af_string(m,face),0)/7.0);}
    else if(kind!="pass"&&!ctx_none(m,die))f[dieface]=batch.face(af_integer(m,die),side);
    const auto subtype=af_map_int(a,"subtypes",kind);if(subtype>=0){require(subtype<comp-34,"Action subtype index invalid");f[34+subtype]=1;if(kind=="rearguard_split")f[23]=static_cast<float>(num("leave_behind",1)/10.0);else if(kind=="combat_effect_choice"){f[23]=static_cast<float>(num("count")/10.0);if(text_is(m,af_get(m,p,"target"),"leader"))f[optkind]=1;}else if(kind=="shine"||kind=="shine_none")f[23]=kind=="shine";}
    else if(kind=="companion_separate"&&batch.hobbit_place())f[34+af_map_int(a,"subtypes","hobbit_place")]=1;
    int64_t nr=0,ne=0,nl=0,nz=0;if(army!=missing){const auto total=own.regular+own.elite;f[16]=static_cast<float>(total/10.0);if((kind=="army_move"||kind=="victor_advance")&&!ctx_none(m,af_get(m,p,"n_regular"))){const auto left=std::max<int64_t>(0,total-intval("n_regular")-intval("n_elite"));f[leave]=static_cast<float>(left/10.0);f[offset("_ACT_EMPTIES_OFF")]=left==0;}}
    if(kind=="army_move"){nr=intval("n_regular");ne=intval("n_elite");if(flag("take_leaders",true)&&army!=missing)for(auto l:child_edges(s,obs_attr(s,army,"leaders"))){if((object_type(s,l)=="game_state.Nazgul")){if(side==1)++nz;}else if(cp_leader_side(s,l)==side)++nl;}if(!ctx_none(m,af_get(m,p,"n_nazgul")))nz=intval("n_nazgul");if(!ctx_none(m,af_get(m,p,"n_leaders")))nl=intval("n_leaders");}
    else if(kind=="victor_advance"){nr=intval("n_regular");ne=intval("n_elite");nz=intval("n_nazgul");if(!ctx_none(m,af_get(m,p,"n_leaders")))nl=intval("n_leaders");}
    else if(kind=="army_attack"&&army!=missing){const auto& formation=batch.formation(army,side);for(auto u:formation.units)(enum_is(s,obs_attr(s,u,"kind"),"game_state.UnitKind","REGULAR")?nr:ne)++;nl=formation.leaders.size();f[leave]=static_cast<float>((own.regular+own.elite-nr-ne)/10.0);}
    else if(kind=="combat_retreat"){const auto region=af_or(m,p,{"from","to"});const auto retreat=batch.army(m,region);if(retreat!=missing){const auto partition=ca_partition(s,retreat,af_string(m,region),side);for(auto u:partition.units)(enum_is(s,obs_attr(s,u,"kind"),"game_state.UnitKind","REGULAR")?nr:ne)++;nl=partition.leaders.size();}}
    else if(kind=="muster_recruit"){const auto type=af_text_or(m,p,"type","regular");nr=type=="regular";ne=type=="elite";nl=type=="leader";nz=type=="nazgul";if(text_is(m,af_get(m,p,"saruman"),"settlements"))nr=3;}
    else if(kind=="muster_elite")ne=text_is(m,af_get(m,p,"saruman"),"swap2")?2:1;
    else if(kind=="muster_minion"||kind=="mouth_move")nl=1;
    else if(kind=="nazgul_move"){if(army!=missing)for(auto l:child_edges(s,obs_attr(s,army,"leaders")))nz+=(object_type(s,l)=="game_state.Nazgul");if(!ctx_none(m,af_get(m,p,"count")))nz=std::min(nz,intval("count"));}
    else if(kind=="siege_trim"){ne=intval("elites_back");nr=std::max<int64_t>(0,intval("excess")-ne);}
    f[comp]=static_cast<float>(nr/10.0);f[comp+1]=static_cast<float>(ne/10.0);f[comp+2]=static_cast<float>(nl/7.0);f[comp+3]=static_cast<float>(nz/8.0);
    if(kind=="companion_separate"||kind=="choose_guide"){const auto name=af_get(m,p,"companion");if(truth(m,name))mark(af_string(m,name));mark_values(af_get(m,p,"companions"));}
    else if(kind=="companion_move"){const auto group=af_get(m,p,"companions"),single=af_get(m,p,"companion");mark_values(group);if(truth(m,single))mark(af_string(m,single));if(!truth(m,group)&&!truth(m,single)&&army!=missing)for(auto l:child_edges(s,obs_attr(s,army,"leaders")))if(object_type(s,l)=="game_state.Companion")mark(std::string(text(s,obs_attr(s,l,"name"))));}
    else if(kind=="hunt_sacrifice_guide"){const auto name=af_or(m,p,{"forced_companion","companion"});if(name!=missing)mark(af_string(m,name));else{const auto guide=obs_attr(s,obs_attr(s,s.root,"fellowship"),"guide");if(!ctx_none(s,guide))mark(std::string(text(s,obs_attr(s,guide,"name"))));}}
    else if(kind=="combat_effect_choice"){const auto name=af_get(m,p,"target");if(truth(m,name)&&!text_is(m,name,"leader"))mark(af_string(m,name));}
    else if(kind=="muster_minion"){const auto name=af_get(m,p,"minion");if(truth(m,name))mark(af_string(m,name));}
    else if(kind=="crown_aragorn")mark("Aragorn");else if(kind=="special_character"){const auto name=af_or(m,p,{"character","minion"});if(name!=missing)mark(af_string(m,name));}
    f[src]=static_cast<float>(own.regular/10.0);f[src+1]=static_cast<float>(own.elite/10.0);f[src+2]=static_cast<float>(own.leaders/8.0);const auto enemy=target==missing?af_Composition{}:batch.composition(batch.army(m,target),1-side);f[tgt]=static_cast<float>(enemy.regular/10.0);f[tgt+1]=static_cast<float>(enemy.elite/10.0);f[tgt+2]=static_cast<float>(enemy.leaders/8.0);
    if(kind=="fellowship_declare"){const auto r=af_or(m,p,{"region","to"});const auto key=r==missing?std::string{}:sanitize_name(af_string(m,r));if(key=="morannon"||key=="minas morgul"||flag("mordor_entry"))f[offset("_ACT_MORDOR_OFF")]=1;}
    if(kind=="card_option"){af_card_tail(m,a,choices,f+batch.cardtail);af_cargo_tail(m,choices,f+133);}af_combat_tail(m,a,kind,p,batch.combat_active(),f+105);
    primary=-1;if(kind=="card_option"){const auto r=af_option_region(m,choices);if(!r.empty())primary=af_region(a,r,-1);}if(primary<0)for(const char* key:{"to","to_region","region","from"}){const auto value=af_get(m,p,key);if(truth(m,value)){const auto r=af_region(a,af_string(m,value),-1);if(r>=0){primary=r;break;}}}
}
void af_check_encode(const Snapshot& s,const Snapshot& t,const Snapshot& menu,const Snapshot& tables,int side,const std::vector<uint64_t>& actions,size_t rows,size_t count,const std::vector<float>& features,const std::vector<int64_t>& primary,const char* fast_error){
    // Reference pass (ROOT per-row code) against the optimized pass: float bytes and primary region per row; an exception
    // on one side must be matched by the same exception text on the other (the caller rethrows the optimized exception).
    af_ref_Batch ref(s,tables);std::vector<float> rf(count,0.0f);std::vector<int64_t> rp(rows,-1);std::string ref_error;
    try{for(size_t i=0;i<rows;++i)af_ref_encode_row(s,t,menu,tables,ref,actions[i],side,rf.data()+i*157,rp[i]);}
    catch(const std::exception& e){ref_error=e.what();if(ref_error.empty())ref_error="<empty>";}
    const std::string fast(fast_error?fast_error:"");
    if(fast!=ref_error)throw std::logic_error("WOTR_NATIVE_AF_CHECK: exception mismatch: optimized='"+fast+"' reference='"+ref_error+"'");
    if(!ref_error.empty())return;
    for(size_t i=0;i<rows;++i){
        const float* a=features.data()+i*157;const float* b=rf.data()+i*157;
        if(std::memcmp(a,b,157*sizeof(float))!=0||primary[i]!=rp[i]){
            size_t col=0;while(col<157&&std::memcmp(a+col,b+col,sizeof(float))==0)++col;
            throw std::logic_error("WOTR_NATIVE_AF_CHECK: row "+std::to_string(i)+" ("+std::string(text(menu,menu.edges[menu.nodes[actions[i]].offset+1]))+") differs from the reference encoder at "+(col<157?"feature column "+std::to_string(col):std::string("the primary region"))+" (optimized primary "+std::to_string(primary[i])+" vs reference "+std::to_string(rp[i])+")");
        }
    }
    af_check_rows_checked+=rows;
}
void af_encode(const Snapshot& s,const Snapshot& t,const Snapshot& menu,const Snapshot& tables,int side,float* output,size_t count,int64_t* regions,size_t rows){
    require(side==0||side==1,"Action feature actor must be FP or SH");require(text_is(tables,field(tables,tables.root,"schema"),"wotr_action_features_v52_157")&&text_is(tables,field(tables,tables.root,"action_encoding_contract"),"observed_combat_choices_v8")&&af_integer(tables,field(tables,tables.root,"width"))==157,"Action feature schema mismatch");
    const auto actions=child_edges(menu,menu.root);require(actions.size()==rows&&rows<=std::numeric_limits<size_t>::max()/157&&count==rows*157,"Action feature output size mismatch");check_array(output,count);check_array(regions,rows);
    af_Batch batch(s,tables);std::vector<float> features(count,0.0f);std::vector<int64_t> primary(rows,-1);
    if(!af_check_active){for(size_t i=0;i<rows;++i)af_encode_row(s,t,menu,tables,batch,actions[i],side,features.data()+i*157,primary[i]);}
    else{   // WOTR_NATIVE_AF_CHECK=1: optimized pass, then the ROOT reference pass; any byte or exception-text difference throws
        std::string fast_error;bool failed=false;std::exception_ptr pending;
        try{for(size_t i=0;i<rows;++i)af_encode_row(s,t,menu,tables,batch,actions[i],side,features.data()+i*157,primary[i]);}
        catch(const std::exception& e){fast_error=e.what();if(fast_error.empty())fast_error="<empty>";failed=true;pending=std::current_exception();}
        af_check_encode(s,t,menu,tables,side,actions,rows,count,features,primary,failed?fast_error.c_str():nullptr);
        if(failed)std::rethrow_exception(pending);
    }
    if(count)std::memcpy(output,features.data(),count*sizeof(float));if(rows)std::memcpy(regions,primary.data(),rows*sizeof(int64_t));
}

void af_prepare_tables(Snapshot& s){
    require(text_is(s,field(s,s.root,"schema"),"wotr_action_features_v52_157")&&text_is(s,field(s,s.root,"action_encoding_contract"),"observed_combat_choices_v8"),"Action table preparation requires v8 metadata");
    // Same owned dictionary-index representation used by observation tables.
    // Strings refer to the prepared copy, never a movable source byte vector.
    auto prepared=std::make_shared<ContextPrepared>();prepared->bytes=s.bytes;prepared->dictionaries.resize(s.nodes.size());
    for(uint64_t d=0;d<s.nodes.size();++d){
        const auto& node=s.nodes[d];if(node.tag!=WOTR_DICT||node.count==0)continue;
        auto index=std::make_unique<ContextPrepared::Index>();index->reserve(node.count/2);
        for(uint64_t i=0;i<node.count;i+=2){const auto k=s.edges[node.offset+i];const auto& key=s.nodes[k];if(key.tag==WOTR_STRING){const auto name=key.count?std::string_view(reinterpret_cast<const char*>(prepared->bytes.data()+key.offset),key.count):std::string_view();(*index)[name]=s.edges[node.offset+i+1];}}
        prepared->dictionaries[d]=std::move(index);
    }
    s.context_prepared=std::move(prepared);
}
