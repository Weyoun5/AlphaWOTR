// Frozen ai.encode_state offsets [3401,3746). Included inside snapshot.cpp's
// anonymous namespace. Read-only graph queries; no Python or fallback vectors.
bool ctx_none(const Snapshot& s,uint64_t n){return n==missing || s.nodes[n].tag==WOTR_NONE;}
uint64_t ctx_get(const Snapshot& s,uint64_t d,std::string_view key){
    if(ctx_none(s,d))return missing;
    if(s.context_prepared){
        typed(s,d,WOTR_DICT,"prepared context requires dictionary");
        require(d<s.context_prepared->dictionaries.size(),"stale prepared context index");
        const auto& index=s.context_prepared->dictionaries[d];
        if(index){const auto found=index->find(key);return found==index->end() ? missing : found->second;}
    }
    const auto edge=string_value_edge(s,d,key);return edge==missing ? missing : s.edges[edge];
}
uint64_t ctx_attr(const Snapshot& s,uint64_t obj,std::string_view name){
    if(ctx_none(s,obj))return missing;
    const auto& n=typed(s,obj,WOTR_OBJECT,"context attribute requires object");
    for(uint64_t i=1;i<n.count;i+=2){
        const auto key=s.edges[n.offset+i],value=s.edges[n.offset+i+1];
        if(text_is(s,key,name))return value;
        if(text_is(s,key,"__dict__")){
            const auto result=ctx_get(s,value,name);if(result!=missing)return result;
        }
    }
    return missing;
}
double ctx_num(const Snapshot& s,uint64_t n,double absent=0){
    if(n==missing)return absent;
    const auto& v=s.nodes[n];
    if(v.tag==WOTR_INT || v.tag==WOTR_BOOL)return static_cast<double>(v.integer);
    if(v.tag==WOTR_FLOAT)return v.real;
    throw std::invalid_argument("context requires numeric scalar");
}
double ctx_getnum(const Snapshot& s,uint64_t d,std::string_view key,double absent=0){return ctx_num(s,ctx_get(s,d,key),absent);}
double ctx_ornum(const Snapshot& s,uint64_t n){return truth(s,n) ? ctx_num(s,n) : 0.;}
double ctx_int(const Snapshot& s,uint64_t n){
    const double value=ctx_ornum(s,n);
    require(value>=-9007199254740991. && value<=9007199254740991.,"context integer outside exact supported range");
    return static_cast<double>(static_cast<int64_t>(value));
}
std::string ctx_string(const Snapshot& s,uint64_t n,std::string_view absent=""){
    if(n==missing)return std::string(absent);
    if(s.nodes[n].tag==WOTR_STRING)return std::string(text(s,n));
    if(s.nodes[n].tag==WOTR_NONE)return "None";
    if(s.nodes[n].tag==WOTR_BOOL)return truth(s,n) ? "True" : "False";
    if(s.nodes[n].tag==WOTR_INT)return std::to_string(s.nodes[n].integer);
    throw std::invalid_argument("unsupported non-string context identifier");
}
std::vector<uint64_t> ctx_items(const Snapshot& s,uint64_t n){
    if(n==missing)return {};
    const auto tag=s.nodes[n].tag;
    require(tag==WOTR_LIST || tag==WOTR_TUPLE || tag==WOTR_SET || tag==WOTR_FROZENSET,"context requires iterable sequence");
    return child_edges(s,n);
}
size_t ctx_len_or(const Snapshot& s,uint64_t n){
    if(!truth(s,n))return 0;
    const auto& v=s.nodes[n];
    if(v.tag==WOTR_DICT)return static_cast<size_t>(v.count/2);
    require(v.tag==WOTR_LIST || v.tag==WOTR_TUPLE || v.tag==WOTR_SET || v.tag==WOTR_FROZENSET,"context length requires collection");
    return static_cast<size_t>(v.count);
}
int ctx_side(const Snapshot& s,uint64_t n){
    if(ctx_none(s,n))return -1;
    const auto name=enum_member(s,n,"game_state.Side");
    require(name=="FREE_PEOPLES" || name=="SHADOW","unknown context side");return name=="FREE_PEOPLES" ? 0 : 1;
}
const std::string& ctx_nation(const Snapshot& s,uint64_t n){
    // Nation names form a closed vocabulary. References to these immutable
    // strings remain valid across Snapshot mutation and builder reallocation.
    // Callers that need ownership still copy; read-only predicates do not.
    const auto name=enum_member(s,n,"game_state.Nation");
    static const std::array<std::string,8> names={"ELVES","DWARVES","GONDOR","THE_NORTH","ROHAN","ISENGARD","SAURON","SOUTHRON_EASTERLINGS"};
    for(const auto& candidate:names)if(candidate==name)return candidate;
    throw std::invalid_argument("unknown Nation enum member name");
}
int ctx_unit_side(const Snapshot& s,uint64_t unit){return fp_nation(validate_nation_name(enum_member(s,ctx_attr(s,unit,"nation"),"game_state.Nation"))) ? 0 : 1;}
bool ctx_companion(const Snapshot& s,uint64_t n){return !ctx_none(s,n) && object_type(s,n)=="game_state.Companion";}
bool ctx_plain_leader_family(const Snapshot& s,uint64_t n){
    const auto t=object_type(s,n);return t=="game_state.Leader" || t=="game_state.Nazgul" || t=="game_logic.Minion";
}
bool ctx_minion_name(const Snapshot& s,uint64_t n){
    const auto name=ctx_attr(s,n,"name");return text_is(s,name,"Witch King") || text_is(s,name,"Saruman") || text_is(s,name,"Mouth of Sauron");
}
int ctx_participant_leader_side(const Snapshot& s,uint64_t leader){
    if(ctx_companion(s,leader))return 0;
    if(object_type(s,leader)=="game_state.Nazgul" || (ctx_plain_leader_family(s,leader)&&ctx_minion_name(s,leader)))return 1;
    if(ctx_plain_leader_family(s,leader)&&!truth(s,ctx_attr(s,leader,"name")))return ctx_unit_side(s,leader);
    throw std::invalid_argument("unsupported board leader in battle participant view");
}
uint64_t ctx_side_value(const Snapshot& s,uint64_t dict,int side){
    if(ctx_none(s,dict))return missing;
    const auto edge=enum_value_edge(s,dict,"game_state.Side",side==0 ? "FREE_PEOPLES" : "SHADOW");
    return edge==missing ? missing : s.edges[edge];
}
uint64_t ctx_nation_value(const Snapshot& s,uint64_t dict,std::string_view nation){
    if(ctx_none(s,dict))return missing;
    const auto edge=enum_value_edge(s,dict,"game_state.Nation",nation);return edge==missing ? missing : s.edges[edge];
}
struct ctx_State{
    const Snapshot& s;const Snapshot& t;
    uint64_t attrs,armies,fellowship,political,boxes,controlled,meta;
    std::vector<std::string> keys,nations;
    std::unordered_map<std::string,size_t> indices;
    std::unordered_map<std::string,uint64_t> normalized;
    std::vector<std::pair<std::string,uint64_t>> raw;
    // One encoding traverses the same neighboring armies many times. Cache
    // their side counts only for this read-only ctx_State lifetime; no derived
    // data survives into another position, clone, mutation or observation.
    mutable std::unordered_map<uint64_t,std::array<int,2>> unit_counts;
    ctx_State(const Snapshot& state,const Snapshot& tables):s(state),t(tables){
        attrs=attributes(s,s.root,"game_state.GameState");meta=t.root;
        // Metadata failures must never be mistaken for a Python availability
        // predicate exception and turned into a zero feature.
        for(const char* key:{"fp_vp_targets","sh_vp_targets","dist_to_fp_target","dist_to_sh_target",
            "fp_settle_neighbors_1","fp_settle_neighbors_2","sh_settle_neighbors_1","sh_settle_neighbors_2",
            "distance_to_mordor","nation_max_figures","live_vp_fp_settlements","live_vp_sh_settlements",
            "fp_vp_victory","shadow_vp_victory","board_regions","board_adj","region_nations",
            "shadow_strongholds","fp_region_keys","card_indices"})
            require(ctx_get(t,meta,key)!=missing,"missing required context metadata table");
        armies=field(s,attrs,"armies");typed(s,armies,WOTR_DICT,"context armies dictionary");
        fellowship=field(s,attrs,"fellowship");political=field(s,attrs,"political");
        boxes=ctx_attr(s,political,"boxes");controlled=field(s,attrs,"enemy_controlled");
        for(auto n:ctx_items(t,field(t,meta,"region_keys"))){keys.emplace_back(text(t,n));indices[keys.back()]=keys.size()-1;}
        for(auto n:ctx_items(t,field(t,meta,"nation_names")))nations.emplace_back(text(t,n));
        require(keys.size()==105 && nations.size()==8,"context requires pinned 105-region/eight-nation layout");
        const auto& a=s.nodes[armies];
        for(uint64_t i=0;i<a.count;i+=2){
            std::string key(text(s,s.edges[a.offset+i]));const auto value=s.edges[a.offset+i+1];
            raw.emplace_back(key,value);normalized[sanitize_name(key)]=value;
        }
    }
    uint64_t get(std::string_view name)const{return ctx_get(s,attrs,name);}
    uint64_t table(std::string_view name)const{const auto n=ctx_get(t,meta,name);require(n!=missing,"missing context metadata table");return n;}
    uint64_t army(const std::string& key)const{auto found=normalized.find(key);return found==normalized.end() ? missing : found->second;}
    uint64_t lookup(const std::string& key)const{
        auto direct=ctx_get(s,armies,key);if(!ctx_none(s,direct))return direct;
        const auto norm=sanitize_name(key);for(const auto& pair:raw)if(sanitize_name(pair.first)==norm)return pair.second;
        return missing;
    }
    bool control(int side,const std::string& key)const{return contains_string(s,ctx_side_value(s,controlled,side),key);}
    bool atwar(std::string_view nation)const{auto n=ctx_nation_value(s,boxes,nation);require(n!=missing,"political box missing nation");return ctx_num(s,n)>=3;}
    size_t index(const std::string& key)const{auto p=indices.find(key);return p==indices.end() ? keys.size() : p->second;}
    std::vector<uint64_t> figures(uint64_t army,const char* kind)const{
        if(ctx_none(s,army))return {};return ctx_items(s,ctx_attr(s,army,kind));
    }
    int units(uint64_t army,int side)const{
        if(ctx_none(s,army))return 0;
        const auto known=unit_counts.find(army);if(known!=unit_counts.end())return known->second[side];
        std::array<int,2> counts{};for(auto u:figures(army,"units"))++counts[ctx_unit_side(s,u)];
        unit_counts.emplace(army,counts);return counts[side];
    }
    bool flag(const char* object,const char* attr)const{return truth(s,ctx_attr(s,get(object),attr));}
    bool in_fellowship(std::string_view name)const{
        for(auto c:ctx_items(s,ctx_attr(s,fellowship,"companions")))if(text_is(s,ctx_attr(s,c,"name"),name))return true;return false;
    }
};
std::vector<std::pair<std::string,double>> ctx_targets(const ctx_State& c,const char* table){
    std::vector<std::pair<std::string,double>> out;
    for(auto n:ctx_items(c.t,c.table(table))){auto parts=ctx_items(c.t,n);require(parts.size()==2,"invalid settlement tuple");out.emplace_back(text(c.t,parts[0]),ctx_num(c.t,parts[1]));}
    return out;
}
double ctx_live_vp(const ctx_State& c,int side){
    double vp=0;for(const auto& p:ctx_targets(c,side==0 ? "live_vp_sh_settlements" : "live_vp_fp_settlements"))if(c.control(side,p.first))vp+=p.second;return vp;
}
int ctx_unused(const ctx_State& c,int side){
    const auto dice=ctx_side_value(c.s,c.get("dice"),side);int count=0;
    for(auto die:ctx_items(c.s,dice))if(!truth(c.s,ctx_attr(c.s,die,"used"))&&!ctx_none(c.s,ctx_attr(c.s,die,"current")))++count;return count;
}
std::vector<std::string> ctx_neighbors(const ctx_State& c,const char* table,const std::string& key){
    const auto value=ctx_get(c.t,c.table(table),key);std::vector<std::string> result;
    if(value!=missing)for(auto n:ctx_items(c.t,value))result.emplace_back(text(c.t,n));return result;
}
void ctx_military_strategy(const ctx_State& c,int side,std::vector<float>& out){
    auto push=[&](double x){out.push_back(static_cast<float>(x));};
    std::array<std::unordered_map<std::string,int>,2> forces;
    for(const auto& key:c.keys){
        const auto army=c.army(key);if(!truth(c.s,army))continue;
        std::array<int,2> units{},leaders{};
        for(auto u:c.figures(army,"units"))++units[ctx_unit_side(c.s,u)];
        for(auto ld:c.figures(army,"leaders")){
            int owner=-1;
            if(ctx_minion_name(c.s,ld))owner=1;
            else if(ctx_attr(c.s,ld,"nation")!=missing)owner=ctx_unit_side(c.s,ld);
            else if(ctx_companion(c.s,ld))owner=0;
            if(owner>=0)++leaders[owner];
        }
        for(int sd=0;sd<2;++sd)if(units[sd]+leaders[sd]>0)forces[sd][key]=units[sd];
    }
    const auto fp_targets=ctx_targets(c,"fp_vp_targets"),sh_targets=ctx_targets(c,"sh_vp_targets");
    auto& mine=side==0 ? fp_targets : sh_targets;auto& enemy=side==0 ? sh_targets : fp_targets;
    const char* distance[2]={"dist_to_fp_target","dist_to_sh_target"};
    for(int owner:{side,1-side}){
        std::array<double,3> sums{};const auto dist=c.table(distance[1-owner]);
        // Ordered source region traversal preserves the Python summation order.
        for(const auto& key:c.keys){auto p=forces[owner].find(key);if(p==forces[owner].end())continue;
            const double d=ctx_num(c.t,ctx_get(c.t,dist,key),99);for(int i=0;i<3;++i)if(d<=i+1)sums[i]+=p->second;}
        push(sums[0]/20.);push(sums[1]/30.);push(sums[2]/40.);
    }
    for(int target_owner:{side,1-side}){
        const auto& targets=target_owner==0 ? fp_targets : sh_targets;int threatened=0;
        const auto dist=c.table(distance[target_owner]);
        for(const auto& target:targets){
            for(const auto& key:c.keys){if(forces[1-target_owner].find(key)==forces[1-target_owner].end())continue;
                const double d=key==target.first ? 0 : ctx_num(c.t,ctx_get(c.t,dist,key),99);
                if(d<=1){++threatened;break;}}
        }
        push(threatened/10.);
    }
    const char* n1[2]={"fp_settle_neighbors_1","sh_settle_neighbors_1"};
    const char* n2[2]={"fp_settle_neighbors_2","sh_settle_neighbors_2"};
    double en_vp1=0,en_vp2=0,my_vp1=0,my_vp2=0;int en_max=0,my_min=5,my_undef=0,en_undef=0;
    for(const auto& target:mine){
        const int defence=std::min(5,c.units(c.army(target.first),side));if(!defence)++my_undef;my_min=std::min(my_min,defence);
        bool one=false,two=false;
        for(const auto& nb:ctx_neighbors(c,n1[side],target.first)){int strength=c.units(c.army(nb),1-side);if(strength){one=true;en_max=std::max(en_max,std::min(5,strength));}}
        if(one)en_vp1+=target.second;two=one;
        if(!two)for(const auto& nb:ctx_neighbors(c,n2[side],target.first))if(c.units(c.army(nb),1-side)){two=true;break;}
        if(two)en_vp2+=target.second;
    }
    for(const auto& target:enemy){
        if(!c.units(c.army(target.first),1-side))++en_undef;
        bool one=false,two=false;
        for(const auto& nb:ctx_neighbors(c,n1[1-side],target.first))if(c.units(c.army(nb),side)){one=true;break;}
        if(one)my_vp1+=target.second;two=one;
        if(!two)for(const auto& nb:ctx_neighbors(c,n2[1-side],target.first))if(c.units(c.army(nb),side)){two=true;break;}
        if(two)my_vp2+=target.second;
    }
    const auto& s=c.s;
    const double steps=truth(s,ctx_attr(s,c.fellowship,"mordor_entered")) ? 5-ctx_num(s,ctx_attr(s,c.fellowship,"mordor_step")) : ctx_num(c.t,ctx_get(c.t,c.table("distance_to_mordor"),fellowship_key(s,c.attrs)),15)+5;
    const double vp_limit=ctx_num(c.t,c.table(side==0 ? "shadow_vp_victory" : "fp_vp_victory"));
    for(double x:{en_vp1/10,en_vp2/10,my_vp1/10,my_vp2/10,en_max/5.,my_min/5.,my_undef/12.,en_undef/10.,steps/20.,(12-ctx_num(s,ctx_attr(s,c.fellowship,"corruption")))/12.,ctx_num(s,c.get("turn"))/25.,std::max(0.,vp_limit-ctx_live_vp(c,1-side))/10.})push(x);
}

void ctx_pools(const ctx_State& c,std::vector<float>& out){
    std::unordered_map<std::string,std::array<double,3>> used;
    for(const auto& entry:c.raw){
        for(auto u:c.figures(entry.second,"units")){auto nation=ctx_nation(c.s,ctx_attr(c.s,u,"nation"));++used[nation][enum_is(c.s,ctx_attr(c.s,u,"kind"),"game_state.UnitKind","REGULAR") ? 0 : 1];}
        for(auto l:c.figures(entry.second,"leaders")){
            auto type=object_type(c.s,l);if(type=="game_state.Leader" || type=="game_state.Nazgul")++used[ctx_nation(c.s,ctx_attr(c.s,l,"nation"))][2];
        }
    }
    const char* names[]={"regular","elite","leader"};const auto caps=c.table("nation_max_figures");
    for(const auto& nation:c.nations){const auto maxes=ctx_nation_value(c.t,caps,nation);
        auto casualties=ctx_nation_value(c.s,c.get("fp_casualties"),nation);
        if(!truth(c.s,casualties))casualties=missing;
        for(size_t i=0;i<3;++i){double cap=ctx_getnum(c.t,maxes,names[i]);double amount=used[nation][i];if(fp_nation(nation))amount+=ctx_getnum(c.s,casualties,names[i]);
            out.push_back(static_cast<float>((cap>0 ? std::max(0.,cap-amount) : 0)/std::max(1.,cap)));}
    }
}

bool ctx_availability(const ctx_State& c,int which){
    const auto& s=c.s;
    if(which<3){
        const char* names[]={"witch_king","saruman","mouth_of_sauron"};
        if(c.flag(names[which],"in_play") || c.flag(names[which],"was_ever_in_play"))return false;
        if(which==0){
            bool any_fp=false;for(const auto& nation:c.nations)if(fp_nation(nation)&&c.atwar(nation))any_fp=true;
            if(!c.atwar("SAURON")||!any_fp)return false;
            for(const auto& entry:c.raw)for(auto u:c.figures(entry.second,"units"))if(enum_is(s,ctx_attr(s,u,"nation"),"game_state.Nation","SAURON"))return true;
            return false;
        }
        if(which==1)return c.atwar("ISENGARD")&&!c.control(0,"orthanc");
        bool all_fp=true;for(const auto& nation:c.nations)if(fp_nation(nation)&&!c.atwar(nation))all_fp=false;
        if(!all_fp&&!truth(s,ctx_attr(s,c.fellowship,"mordor_entered")))return false;
        for(auto key:ctx_items(c.t,c.table("shadow_strongholds")))if(!c.control(0,std::string(text(c.t,key))))return true;return false;
    }
    if(which==3){
        if(c.flag("gandalf_the_white","in_play")||c.in_fellowship("Gandalf the Grey"))return false;
        if(!c.flag("witch_king","was_ever_in_play")&&!c.flag("saruman","was_ever_in_play")&&!c.flag("mouth_of_sauron","was_ever_in_play"))return false;
        for(const auto& entry:c.raw)for(auto ld:c.figures(entry.second,"leaders"))if(ctx_companion(s,ld)&&text_is(s,ctx_attr(s,ld,"name"),"Gandalf the Grey"))return true;
        for(const char* key:{"fangorn","grey havens","rivendell","lorien","woodland realm"})if(!c.control(1,key))return true;return false;
    }
    if(c.flag("aragorn_crowned","crowned")||c.in_fellowship("Strider"))return false;
    std::string region;
    for(const auto& entry:c.raw)for(auto ld:c.figures(entry.second,"leaders"))if(ctx_companion(s,ld)&&text_is(s,ctx_attr(s,ld,"name"),"Strider")){region=entry.first;break;}
    if(region.empty())return false;region=sanitize_name(region);
    return (region=="minas tirith"||region=="pelargir"||region=="dol amroth")&&!c.control(1,region);
}
void ctx_immediate(const ctx_State& c,int side,std::vector<float>& out){
    for(int owner:{side,1-side}){
        double value=0;int targets=0,dice=ctx_unused(c,owner);const auto metadata=c.table("board_regions");
        for(const auto& key:c.keys){const auto info=ctx_get(c.t,metadata,key);
            if(!truth(c.t,ctx_get(c.t,info,"is_city"))&&!truth(c.t,ctx_get(c.t,info,"is_stronghold")))continue;
            const auto nation=ctx_get(c.t,c.table("region_nations"),key);
            if(ctx_none(c.t,nation) || (fp_nation(ctx_nation(c.t,nation)) ? 0 : 1)==owner || c.control(owner,key))continue;
            int best=0;for(const auto& nb:ctx_neighbors(c,"board_adj",key))best=std::max(best,c.units(c.army(nb),owner));
            if(best>c.units(c.army(key),1-owner)&&targets<dice){value+=truth(c.t,ctx_get(c.t,info,"is_stronghold")) ? 2 : 1;++targets;}
        }
        out.push_back(static_cast<float>(value/10.));
    }
    for(int which=0;which<5;++which){bool allowed=false;
        try{allowed=ctx_availability(c,which);}catch(const std::exception&){
            if(which==4)for(const auto& entry:c.raw)for(auto ld:c.figures(entry.second,"leaders"))if(ctx_companion(c.s,ld)&&text_is(c.s,ctx_attr(c.s,ld,"name"),"Strider"))allowed=true;
        }
        out.push_back(allowed ? 1.f : 0.f);
    }
}

std::array<double,3> ctx_battle_roster(const ctx_State& c,const std::string& region,int side,uint64_t pc){
    const auto army=c.lookup(region);if(ctx_none(c.s,army))return {};
    std::unordered_map<std::string,double> units_due,leaders_due;
    const auto payload=ctx_get(c.s,pc,"attacker_rearguard");
    if(!ctx_none(c.s,payload)){
        const auto parts=ctx_items(c.s,payload);require(parts.size()==3,"malformed frozen rearguard payload");
        if(text_is(c.s,parts[0],sanitize_name(region))){
            require(side>=0,"rearguard owner requires Side");
            if(text_is(c.s,parts[1],side==0 ? "FREE_PEOPLES" : "SHADOW")){
                const auto signature=ctx_items(c.s,parts[2]);require(signature.size()==3,"malformed rearguard signature");
                require(text_is(c.s,signature[0],"forced_rearguard_v1"),"unknown attack formation version");
                for(auto node:ctx_items(c.s,signature[1])){auto row=ctx_items(c.s,node);require(row.size()==3,"malformed rearguard unit count");units_due[std::string(text(c.s,row[0]))+"|"+std::string(text(c.s,row[1]))]=ctx_num(c.s,row[2]);}
                for(auto node:ctx_items(c.s,signature[2])){auto row=ctx_items(c.s,node);require(row.size()==2,"malformed rearguard leader count");leaders_due[std::string(text(c.s,row[0]))]=ctx_num(c.s,row[1]);}
            }
        }
    }
    std::unordered_map<std::string,double> relief_due;
    const auto relief=ctx_get(c.s,pc,"corsairs_relief_garrison");
    if(truth(c.s,relief)&&side==ctx_side(c.s,ctx_get(c.s,pc,"attacker_side"))&&sanitize_name(region)==sanitize_name(ctx_string(c.s,ctx_get(c.s,pc,"defender_region"))))
        for(auto node:ctx_items(c.s,relief)){const auto row=ctx_items(c.s,node);require(row.size()==4,"Malformed relief roster");relief_due[ctx_string(c.s,row[0])+"|"+ctx_string(c.s,row[1])+"|"+ctx_string(c.s,row[2])]=ctx_num(c.s,row[3]);}
    std::array<double,3> result{};
    for(auto unit:c.figures(army,"units")){
        const auto nation=ctx_nation(c.s,ctx_attr(c.s,unit,"nation"));
        const auto kind=enum_member(c.s,ctx_attr(c.s,unit,"kind"),"game_state.UnitKind");
        if(ctx_unit_side(c.s,unit)!=side)continue;
        auto& reserve=relief_due["unit|"+nation+"|"+std::string(kind)];if(reserve>0){--reserve;continue;}
        auto& due=units_due[nation+"|"+std::string(kind)];if(due>0){--due;continue;}
        if(kind=="REGULAR")++result[0];else if(kind=="ELITE")++result[1];
    }
    for(auto leader:c.figures(army,"leaders")){
        const auto owner=ctx_participant_leader_side(c.s,leader);if(owner!=side)continue;
        if(!relief_due.empty()&&!ctx_companion(c.s,leader)){const auto key=ctx_minion_name(c.s,leader)?"named|"+ctx_string(c.s,ctx_attr(c.s,leader,"name"))+"|":std::string(object_type(c.s,leader)=="game_state.Nazgul"?"nazgul|":"leader|")+ctx_nation(c.s,ctx_attr(c.s,leader,"nation"))+"|";
        auto& reserve=relief_due[key];if(reserve>0){--reserve;continue;}}
        if(ctx_plain_leader_family(c.s,leader)&&!ctx_minion_name(c.s,leader)){
            auto& due=leaders_due[ctx_nation(c.s,ctx_attr(c.s,leader,"nation"))];if(due>0){--due;continue;}
        }
        // The encoder independently side-filters the resulting roster. These
        // classes already have the same owner in this supported participant set.
        ++result[2];
    }
    for(const auto& item:units_due)require(item.second<=0,"protected rearguard unit missing");
    for(const auto& item:leaders_due)require(item.second<=0,"protected rearguard leader missing");
    for(const auto& item:relief_due)require(item.second<=0,"protected relief figure missing");
    return result;
}
int ctx_card_index(const ctx_State& c,uint64_t card){
    if(card==missing || c.s.nodes[card].tag!=WOTR_STRING)return -1;
    const auto n=ctx_get(c.t,c.table("card_indices"),text(c.s,card));return n==missing ? -1 : static_cast<int>(ctx_num(c.t,n));
}
int ctx_stack_cap(const ctx_State& c,const std::string& raw,int side){
    const auto key=sanitize_name(raw);if(!contains_string(c.s,c.get("besieged_strongholds"),key))return 10;
    int owner=-1;const auto& controlled=c.s.nodes[c.controlled];
    for(uint64_t i=0;i<controlled.count;i+=2){
        if(contains_string(c.s,c.s.edges[controlled.offset+i+1],key)){owner=ctx_side(c.s,c.s.edges[controlled.offset+i]);break;}
    }
    if(owner<0)owner=contains_string(c.t,c.table("fp_region_keys"),key) ? 0 : 1;
    return side==owner ? 5 : 10;
}

void ctx_pending(const ctx_State& c,int side,std::vector<float>& out){
    const auto& s=c.s;auto push=[&](double value){out.push_back(static_cast<float>(value));};
    const char* companions[]={"Gandalf the Grey","Strider","Legolas","Gimli","Boromir","Meriadoc","Peregrin","Gollum","Aragorn","Gandalf the White"};
    const auto guide=ctx_attr(s,c.fellowship,"guide");
    const auto guide_name=ctx_none(s,guide) ? std::string() : sanitize_name(ctx_string(s,ctx_attr(s,guide,"name")));
    bool matched=false;
    for(const char* name:companions){bool match=!ctx_none(s,guide)&&!matched&&sanitize_name(name)==guide_name;push(match);matched=matched||match;}

    const bool pam=!ctx_none(s,c.get("pending_army_move_die")),pmu=!ctx_none(s,c.get("pending_muster_die"));
    const auto pco=c.get("pending_card_option"),pc=c.get("pending_combat");
    push(pam);push(pmu);push(!ctx_none(s,pco));
    std::vector<double> pending_from(c.keys.size(),0),battle_region(c.keys.size(),0);
    const auto from=pam ? c.get("pending_army_move_from") : pmu ? c.get("pending_muster_region") : missing;
    if(truth(s,from)){auto i=c.index(sanitize_name(ctx_string(s,from)));if(i<c.keys.size())pending_from[i]=1;}
    for(double x:pending_from)push(x);
    std::array<double,27> combat_ctx{};
    if(!ctx_none(s,pc)){
        const auto def=ctx_string(s,ctx_get(s,pc,"defender_region")),atk=ctx_string(s,ctx_get(s,pc,"attacker_region"));
        const auto bi=c.index(sanitize_name(def)),ai=c.index(sanitize_name(atk));
        if(bi<c.keys.size())battle_region[bi]=1;
        if(ai<c.keys.size()&&ai!=bi)battle_region[ai]=-1;
        const auto as=ctx_side(s,ctx_get(s,pc,"attacker_side")),ds=ctx_side(s,ctx_get(s,pc,"defender_side"));
        combat_ctx[0]=1;combat_ctx[1]=truth(s,ctx_get(s,pc,"is_siege"));combat_ctx[2]=as==side;
        const auto attacker=ctx_battle_roster(c,atk,as,pc),defender=ctx_battle_roster(c,def,ds,pc);
        if(as>=0){combat_ctx[3]=attacker[0]/10.;combat_ctx[4]=attacker[1]/10.;combat_ctx[5]=attacker[2]/7.;}
        if(ds>=0){combat_ctx[6]=defender[0]/10.;combat_ctx[7]=defender[1]/10.;combat_ctx[8]=defender[2]/7.;}
        const double rounds=ctx_getnum(s,pc,"rounds_fought");combat_ctx[9]=std::min(rounds,5.)/5.;
        const char* phases[]={"defender_siege_choice","attacker_combat_card","defender_combat_card","combat_effect_choice","combat_round_result","defender_retreat","attacker_siege_decision","attacker_casualty_policy","defender_casualty_policy","victor_advance","casualty_allocation","fp_shine"};
        for(size_t i=0;i<12;++i)if(text_is(s,ctx_get(s,pc,"phase"),phases[i]))combat_ctx[10+i]=1;
        combat_ctx[22]=std::max(0.,ctx_getnum(s,pc,"mandatory_rounds")-rounds)/3.;
        combat_ctx[23]=truth(s,ctx_get(s,pc,"no_def_card_r1_unless_companion"));
        combat_ctx[24]=std::min(ctx_getnum(s,pc,"atk_dice_reduction"),5.)/5.;
        combat_ctx[25]=std::min(ctx_getnum(s,pc,"def_dice_reduction"),5.)/5.;
        combat_ctx[26]=truth(s,ctx_get(s,pc,"attacker_no_retreat"));
    }
    const bool card_allocating=truth(s,ctx_get(s,pco,"card_casualty"));
    if(card_allocating){auto i=c.index(sanitize_name(ctx_string(s,ctx_get(s,pco,"cas_region"))));if(i<c.keys.size())battle_region[i]=1;}
    for(double x:battle_region)push(x);for(double x:combat_ctx)push(x);
    const auto ph=c.get("pending_hunt");std::array<double,6> hunt{};
    if(!ctx_none(s,ph)){
        hunt[0]=1;hunt[1]=std::min(ctx_getnum(s,ph,"damage"),6.)/6.;
        const char* phases[]={"hunt_sacrifice","hunt_gollum","hobbit_place"};
        const auto phase=ctx_string(s,ctx_get(s,ph,"phase"));for(size_t i=0;i<3;++i)if(phase==phases[i])hunt[2+i]=1;
        hunt[5]=std::min(ctx_int(s,ctx_get(s,ph,"level")),4.)/4.;
    }
    for(double x:hunt)push(x);
    std::array<double,6> card{};
    if(!ctx_none(s,pco)){
        card[0]=(ctx_card_index(c,ctx_get(s,pco,"card_id"))+1)/97.;
        card[1]=std::min<size_t>(ctx_len_or(s,ctx_get(s,pco,"options")),40)/40.;
        auto decider=ctx_get(s,pco,"decider");if(decider==missing)decider=ctx_get(s,pco,"side");
        card[2]=(text_is(s,decider,"fp"))==(side==0);
        card[3]=truth(s,ctx_get(s,pco,"direct"));card[4]=truth(s,ctx_get(s,pco,"resume_action"));
        card[5]=(text_is(s,ctx_get(s,pco,"side"),"fp"))==(side==0);
    }
    const bool gandalf_open=!ctx_none(s,pco)&&truth(s,ctx_get(s,pco,"gandalf_draw"));
    const auto gandalf=gandalf_open ? pco : c.get("_gandalf_draw_pending");
    if(!ctx_none(s,gandalf)){
        const auto source=ctx_get(s,gandalf,"source_card_id");
        require(source!=missing && s.nodes[source].tag==WOTR_STRING,"Gandalf draw requires source card string");
        const int index=ctx_card_index(c,source);const auto cid=text(s,source);
        const char* expected=cid.find("_Character_")!=std::string_view::npos ? "character" : "strategy";
        require(index>=0&&text_is(s,ctx_get(s,gandalf,"deck"),expected),"Gandalf draw source/deck mismatch");
        const int code=index+1+97*text_is(s,ctx_get(s,gandalf,"deck"),"strategy")+194*gandalf_open+388*(card[4]!=0);
        card[4]=-static_cast<double>(code)/776.;if(gandalf_open)card[0]=(index+1)/97.;
    }
    for(double x:card)push(x);

    const auto pd=c.get("pending_discard"),follow=c.get("pending_card_followup");
    push(!ctx_none(s,pd));push(!ctx_none(s,pd)&&ctx_side(s,ctx_get(s,pd,"side"))==side);
    push(!ctx_none(s,c.get("pending_guide_change")));push(truth(s,c.get("_eye_allocation_phase")));push(truth(s,c.get("_declare_phase")));
    push(!ctx_none(s,follow));push(truth(s,follow)&&text_is(s,ctx_get(s,follow,"kind"),"wk_army"));
    push(truth(s,follow) ? ctx_getnum(s,follow,"moves_left")/2. : 0.);
    push(std::min<size_t>(ctx_len_or(s,c.get("pending_hunt_tiles")),3)/3.);
    push(truth(s,c.get("_crebain_hunt_bonus")));
    auto over=c.get("_overstack_regions");if(!truth(s,over))over=missing;
    push(std::min<size_t>(ctx_len_or(s,over),3)/3.);int excess=0;
    for(auto raw:ctx_items(s,over)){
        const auto key=ctx_string(s,raw);const auto army=c.lookup(key);if(ctx_none(s,army))continue;
        for(int owner=0;owner<2;++owner)excess+=std::max(0,c.units(army,owner)-ctx_stack_cap(c,key,owner));
    }
    push(std::min(excess,6)/6.);push(std::min(ctx_int(s,c.get("_ents_chain_n")),4.)/4.);
    const auto stash=c.get("_mithril_stash");
    if(truth(s,stash)){
        const auto tile=ctx_get(s,stash,"tile");push(1);
        push(std::min(std::max(ctx_int(s,ctx_attr(s,tile,"damage")),-2.),3.)/3.);
        push(truth(s,ctx_attr(s,tile,"is_eye")));push(truth(s,ctx_attr(s,tile,"reveal")));
        push(truth(s,ctx_attr(s,tile,"stops_movement")));push(truth(s,ctx_get(s,stash,"mordor")));
    }else for(int i=0;i<6;++i)push(0);

    push(truth(s,c.get("fellowship_acted_in_mordor")));
    const auto pre=c.get("pending_army_move_preexist");
    push(std::min(ctx_int(s,ctx_get(s,pre,"regular")),10.)/10.);
    push(std::min(ctx_int(s,ctx_get(s,pre,"elite")),10.)/10.);
    push(std::min(ctx_int(s,ctx_get(s,pre,"leaders")),7.)/7.);
    const auto first=c.get("pending_muster_first_type");for(const char* type:{"regular","elite","leader"})push(text_is(s,first,type));
    const auto owner=pam ? c.get("pending_army_move_side") : pmu ? c.get("pending_muster_side") : missing;
    push(ctx_side(s,owner)==side);push(!ctx_none(s,owner)&&ctx_side(s,owner)!=side);
    const bool allocating=!ctx_none(s,pc)&&text_is(s,ctx_get(s,pc,"phase"),"casualty_allocation");
    push(card_allocating ? ctx_ornum(s,ctx_get(s,pco,"cas_hits"))/10. : allocating ? ctx_ornum(s,ctx_get(s,pc,"atk_hits"))/10. : 0.);
    push(allocating ? ctx_ornum(s,ctx_get(s,pc,"def_hits"))/10. : 0.);
    push(card_allocating ? 2 : allocating ? (text_is(s,ctx_get(s,pc,"alloc_stage"),"attacker") ? 1 : -1) : 0);
    push(truth(s,c.get("_fellowship_tail_phase")));
}

void append_context(const Snapshot& s,const Snapshot& tables,int32_t side,std::vector<float>& out){
    require(side==0 || side==1,"context side must be FP/SH");
    require(out.size()==3401,"context must begin at observation offset 3401");
    ctx_State c(s,tables);
    ctx_military_strategy(c,side,out);require(out.size()==3421,"military/strategy context width drift");
    ctx_pools(c,out);require(out.size()==3445,"reinforcement context width drift");
    ctx_immediate(c,side,out);require(out.size()==3452,"availability context width drift");
    ctx_pending(c,side,out);require(out.size()==3746,"pending context width drift");
}
