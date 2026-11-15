// NATIVE COMBAT KERNEL (2b, claude_bugscan_2026-10-21, 2026-10-27): src/combat_dist.py exact_distribution (stage0,
// stages A-B, _finish) on a kernel context, byte-exact, plus the 9.5 reply fields orchestrator.combat_distribution_reply
// builds around it (plan_conf_case, wctk_side, repr(ctx), repr(plan)).
//
// combat_dist builds the round on a scratch GameState with the REAL engine (build_round -> combat.setup_combat) and
// runs the deterministic appliers (apply_cancel_effects, apply_words_of_power, apply_dice_modifiers,
// apply_dice_count_modifiers, apply_dread_and_despair, apply_confusion_effect, apply_reroll_cancellation,
// apply_they_are_terrible).  The scratch round is fully determined by the context: fresh armies built from the
// ArmySpecs (units: regulars, elites, elites2 in that order; leaders as Leader / Nazgul / Companion / Minion), the
// fixture regions of _pick_region (only their settlement kind matters), the initial game's empty persistent cards,
// and pending_combat = {'gtw_shine': shine} when shine is given.  This file mirrors exactly those code paths (with
// figure identity kept as ids, Python's first-max / first-min / stable-sort choices, dict insertion orders and the
// same float operation order, CPython float_pow via cd_pow).
#include <optional>
#include <map>
#include <set>
#include <unordered_map>
#include <cstdio>
struct ck_Error {std::string why;};
static const char* const ck_NATIONS[8]={"ELVES","DWARVES","GONDOR","THE_NORTH","ROHAN","ISENGARD","SAURON","SOUTHRON_EASTERLINGS"};
static const char* const ck_NATION_VALUES[8]={"Elves","Dwarves","Gondor","The North","Rohan","Isengard","Sauron","Southrons & Easterlings"};
enum {CK_LEADER=0,CK_NAZGUL=1,CK_COMPANION=2,CK_MINION=3};
inline int ck_side_of_nation(int n){return n>=5?1:0;}           // combat.side_of_nation: 1 = SHADOW
struct ck_Leader {int kind=CK_LEADER;std::string name;int64_t level=1,leadership=1;int nation=-1;int id=0;};
struct ck_Unit {int nation=2;bool elite=false;};
struct ck_Army {std::vector<ck_Unit> units;std::vector<ck_Leader> leaders;};
struct ck_LeaderSpec {std::string kind,name;int64_t level=1,leadership=1;int nation=-1;};
struct ck_ArmySpec {int64_t regulars=0,elites=0;int nation=2,elite_nation=-1;std::vector<ck_LeaderSpec> leaders;int64_t elites2=0;int elite2_nation=-1;};
struct ck_Value {int kind=0;int64_t i=0;std::string s;};          // 0 None, 1 bool, 2 int, 3 str
struct ck_Ctx {
    ck_ArmySpec atk,dfn;bool has_atk_card=false,has_def_card=false;std::string atk_card,def_card;bool is_siege=false,city=false;
    int shine=-1;std::vector<std::pair<std::string,ck_Value>> choices;bool defer=false;bool has_atk_region=false,has_def_region=false;
    std::string atk_region,def_region;bool saruman=false;
};
struct ck_Card {bool present=false;std::string name,effect;int64_t value=99;};   // name / effect lowercased

// ---- context payload (orchestrator.exchange_context_payload layout: 2 army rows + 10 scalars)
int ck_nation(const Snapshot& s,uint64_t n){
    if(ctx_none(s,n))return -1;
    const std::string_view m=enum_member(s,n,"game_state.Nation");
    for(int i=0;i<8;++i)if(m==ck_NATIONS[i])return i;
    throw ck_Error{"unknown Nation in kernel context"};
}
int64_t ck_int(const Snapshot& s,uint64_t n){const auto& v=s.nodes[n];if(v.tag!=WOTR_INT&&v.tag!=WOTR_BOOL)throw ck_Error{"kernel context integer expected"};return v.integer;}
ck_Value ck_value(const Snapshot& s,uint64_t n){
    ck_Value v;if(ctx_none(s,n))return v;const auto& x=s.nodes[n];
    if(x.tag==WOTR_BOOL){v.kind=1;v.i=x.integer;}else if(x.tag==WOTR_INT){v.kind=2;v.i=x.integer;}
    else if(x.tag==WOTR_STRING){v.kind=3;v.s=std::string(text(s,n));}else throw ck_Error{"unsupported kernel choice value"};
    return v;
}
ck_ArmySpec ck_army_spec(const Snapshot& s,uint64_t row){
    const auto f=child_edges(s,row);if(f.size()!=7)throw ck_Error{"ArmySpec row must have 7 entries"};
    ck_ArmySpec a;a.regulars=ck_int(s,f[0]);a.elites=ck_int(s,f[1]);a.nation=ck_nation(s,f[2]);a.elite_nation=ck_nation(s,f[3]);
    if(a.nation<0)throw ck_Error{"ArmySpec nation is None"};
    for(auto l:child_edges(s,f[4])){
        const auto x=child_edges(s,l);if(x.size()!=5)throw ck_Error{"LeaderSpec must have 5 entries"};
        ck_LeaderSpec ls;ls.kind=std::string(text(s,x[0]));ls.name=std::string(text(s,x[1]));ls.level=ck_int(s,x[2]);ls.leadership=ck_int(s,x[3]);ls.nation=ck_nation(s,x[4]);
        a.leaders.push_back(std::move(ls));
    }
    a.elites2=ck_int(s,f[5]);a.elite2_nation=ck_nation(s,f[6]);return a;
}
ck_Ctx ck_parse(const Snapshot& s){
    const auto f=child_edges(s,s.root);if(f.size()!=12)throw ck_Error{"ExchangeContext payload must have 12 entries"};
    ck_Ctx c;c.atk=ck_army_spec(s,f[0]);c.dfn=ck_army_spec(s,f[1]);
    if(!ctx_none(s,f[2])){c.has_atk_card=true;c.atk_card=std::string(text(s,f[2]));}
    if(!ctx_none(s,f[3])){c.has_def_card=true;c.def_card=std::string(text(s,f[3]));}
    c.is_siege=truth(s,f[4]);c.city=truth(s,f[5]);c.shine=ctx_none(s,f[6])?-1:(truth(s,f[6])?1:0);
    for(auto item:child_edges(s,f[7])){const auto kv=child_edges(s,item);if(kv.size()!=2)throw ck_Error{"choice item must be a pair"};c.choices.push_back({std::string(text(s,kv[0])),ck_value(s,kv[1])});}
    c.defer=truth(s,f[8]);
    if(!ctx_none(s,f[9])){c.has_atk_region=true;c.atk_region=std::string(text(s,f[9]));}
    if(!ctx_none(s,f[10])){c.has_def_region=true;c.def_region=std::string(text(s,f[10]));}
    c.saruman=truth(s,f[11]);return c;
}
// dict(ctx.choices) (+ defer_casualties for round_choices): later keys win, first position kept
struct ck_Choices {
    std::vector<std::pair<std::string,ck_Value>> items;
    const ck_Value* get(const std::string& k)const{for(const auto& it:items)if(it.first==k)return &it.second;return nullptr;}
    void put(const std::string& k,ck_Value v){for(auto& it:items)if(it.first==k){it.second=std::move(v);return;}items.push_back({k,std::move(v)});}
    bool has_value(const std::string& k)const{auto v=get(k);return v&&v->kind!=0;}           // ch.get(k) is not None
    int64_t integer(const std::string& k)const{auto v=get(k);if(!v||v->kind==0)throw ck_Error{"choice is None"};if(v->kind==3)throw ck_Error{"choice is a string"};return v->i;}
    bool truthy(const std::string& k)const{auto v=get(k);if(!v||v->kind==0)return false;if(v->kind==3)return !v->s.empty();return v->i!=0;}
    // ch.get(k, fallback) for a casualty policy: a None value is None (it then selects _apply_hits' default branch, "")
    std::string str_or(const std::string& k,const std::string& fallback)const{auto v=get(k);if(!v)return fallback;if(v->kind==0)return std::string();if(v->kind!=3)throw ck_Error{"choice must be a string"};return v->s;}
    // ch.get(k) for a target: None (missing / None) or a string
    std::optional<std::string> target(const std::string& k)const{auto v=get(k);if(!v||v->kind==0)return std::nullopt;if(v->kind!=3)throw ck_Error{"target must be a string"};return v->s;}
    // ch.get(k) != 0 (None != 0 is True)
    bool not_zero(const std::string& k)const{auto v=get(k);if(!v||v->kind==0)return true;if(v->kind==3)return true;return v->i!=0;}
};

// ---- the card table (combat_dist.combat_card: first card whose _norm(combat_name) matches)
std::string ck_norm(std::string_view value){
    std::string out;size_t i=0;
    while(i<value.size()){
        const auto first=static_cast<unsigned char>(value[i]);
        if(first<0x80){out.push_back(first>='A'&&first<='Z'?char(first-'A'+'a'):char(first));++i;continue;}
        size_t len=first>=0xf0?4:first>=0xe0?3:2;if(i+len>value.size())throw ck_Error{"truncated UTF-8 card name"};
        uint32_t point=first&(len==2?0x1f:len==3?0x0f:0x07);for(size_t j=1;j<len;++j)point=(point<<6)|(static_cast<unsigned char>(value[i+j])&0x3f);
        const auto* begin=kWotrAsciiMappings;const auto* end=begin+kWotrAsciiMappingCount;
        const auto* e=std::lower_bound(begin,end,point,[](const WotrAsciiMapping& m,uint32_t w){return m.codepoint<w;});
        if(e!=end&&e->codepoint==point){for(size_t j=0;j<e->length;++j){char c=kWotrAsciiBytes[e->offset+j];out.push_back(c>='A'&&c<='Z'?char(c-'A'+'a'):c);}}
        else out.append(value.substr(i,len));
        i+=len;
    }
    return out;
}
std::string ck_lower(std::string_view v){std::string o(v);for(auto& c:o)if(c>='A'&&c<='Z')c=char(c-'A'+'a');return o;}
ck_Card ck_card(const Snapshot& t,bool present,const std::string& name){
    ck_Card c;if(!present||name.empty())return c;
    const std::string key=ck_norm(name);const auto cards=child_edges(t,field(t,t.root,"cards"));
    for(size_t i=1;i<cards.size();i+=2){
        const auto card=cards[i];const auto nm=obs_attr(t,card,"combat_name",false);
        if(nm==missing||ctx_none(t,nm)||!truth(t,nm))continue;
        if(ck_norm(text(t,nm))!=key)continue;
        c.present=true;c.name=ck_lower(text(t,nm));c.effect=ck_lower(text(t,obs_attr(t,card,"combat_effect")));
        c.value=ck_int(t,obs_attr(t,card,"combat_value"));return c;
    }
    throw ck_Error{"combat card not found: "+name};
}
inline bool ck_has(const std::string& s,const char* part){return s.find(part)!=std::string::npos;}

// ---- the scratch round (build_round + setup_combat on the context's fresh armies)
struct ck_Combat {
    ck_Army atk,dfn;int atk_side=0,def_side=1;bool is_siege=false;int64_t siege_round=1;bool city_defense=false;
    ck_Card atk_card,def_card;bool atk_cc=false,def_cc=false;
    int64_t atk_lead=0,def_lead=0,atk_dm=0,def_dm=0,atk_rm=0,def_rm=0,atk_red=0,def_red=0;
    bool has_atk_cap=false,has_def_cap=false;int64_t atk_cap=0,def_cap=0;bool atk_rrc=false,def_rrc=false;
    std::vector<ck_Leader> companions;int64_t naz=0;bool atk_conf=false,def_conf=false;bool battle_over=false;
    int shined=-1,shine=-1,wop=-1;bool wop_is_atk=false;bool fs_atk=false,fs_def=false;int64_t shine_cancelled=0;bool saruman=false;
    std::vector<std::string> forfeited;
};
int ck_leader_side(const ck_Leader& l){            // combat._combat_leader_side
    if(l.kind==CK_NAZGUL||l.kind==CK_MINION)return 1;if(l.kind==CK_COMPANION)return 0;return l.nation<0?0:ck_side_of_nation(l.nation);
}
int ck_army_side(const ck_Army& a){return a.units.empty()?0:ck_side_of_nation(a.units[0].nation);}
const char* const ck_CAPTAINS[6]={"Gandalf the Grey","Strider","Legolas","Gimli","Boromir","Aragorn"};
bool ck_captain(const std::string& n){for(auto c:ck_CAPTAINS)if(n==c)return true;return false;}
int64_t ck_strength(const ck_Army& a,int cancelled){        // combat.calc_combat_strength(army, cancelled_captain)
    int64_t s=static_cast<int64_t>(a.units.size());const int own=a.units.empty()?-1:ck_army_side(a);
    for(const auto& l:a.leaders){if(own>=0&&ck_leader_side(l)!=own)continue;if(l.kind==CK_COMPANION&&ck_captain(l.name)){if(cancelled>=0&&l.id==cancelled)continue;++s;}}
    return std::min<int64_t>(5,s);
}
int64_t ck_calc_leadership(const ck_Army& a,bool cancel_nazgul){
    int64_t lead=0;const int own=a.units.empty()?-1:ck_army_side(a);
    for(const auto& l:a.leaders){
        if(own>=0&&ck_leader_side(l)!=own)continue;
        if(l.kind==CK_COMPANION)lead+=l.leadership;
        else if(l.kind==CK_NAZGUL)lead+=cancel_nazgul?0:1;
        else if(l.kind==CK_MINION){const auto nm=ck_lower(l.name);if(ck_has(nm,"witch"))lead+=cancel_nazgul?0:2;else if(ck_has(nm,"mouth"))lead+=2;else lead+=1;}
        else lead+=1;
    }
    return lead;
}
void ck_recompute_leadership(ck_Combat& c){         // combat._recompute_round_leadership
    const bool shine=c.shine>=0;
    auto pool=[&](const ck_Army& a,int side){int64_t n=0;for(const auto& l:a.leaders){if(ck_leader_side(l)!=side)continue;if(l.kind==CK_NAZGUL)++n;}
        for(const auto& l:a.leaders){if(ck_leader_side(l)!=side)continue;if(l.kind==CK_MINION&&ck_has(ck_lower(l.name),"witch"))n+=2;}return n;};
    auto side_lead=[&](const ck_Army& a,int side){
        int64_t lead=ck_calc_leadership(a,shine&&side==1);
        if(c.saruman&&side==1)for(const auto& u:a.units)if(u.nation==5&&u.elite)++lead;
        if(c.shined>=0)for(const auto& l:a.leaders)if(l.id==c.shined){lead=std::max<int64_t>(0,lead-l.leadership);break;}
        return lead;
    };
    c.atk_lead=side_lead(c.atk,c.atk_side);c.def_lead=side_lead(c.dfn,c.def_side);
    c.naz=pool(c.atk,c.atk_side)+pool(c.dfn,c.def_side);
}
bool ck_board_flag(const Snapshot& t,const std::string& region,const char* flag){
    const auto meta=field(t,field(t,t.root,"board_regions"),region.c_str(),false);
    return meta!=missing&&!ctx_none(t,meta)&&truth(t,field(t,meta,flag,false));
}
bool ck_board_has(const Snapshot& t,const std::string& region){return field(t,field(t,t.root,"board_regions"),region.c_str(),false)!=missing;}
bool ck_fp_region(const Snapshot& t,const std::string& region){return contains_string(t,field(t,t.root,"fp_region_keys"),region);}
std::string ck_pick_region(const Snapshot& t,int side,const char* kind){   // combat_dist._pick_region
    const std::string k(kind);
    if(k=="field"&&side==0&&ck_board_has(t,"dagorlad"))return "dagorlad";
    if(k=="field"&&side==1&&ck_board_has(t,"gorgoroth"))return "gorgoroth";
    if(k=="stronghold"&&side==0&&ck_board_has(t,"minas tirith"))return "minas tirith";
    if(k=="stronghold"&&side==1&&ck_board_has(t,"morannon"))return "morannon";
    const auto regions=child_edges(t,field(t,t.root,"board_regions"));std::vector<std::string> names;
    for(size_t i=0;i<regions.size();i+=2)names.push_back(std::string(text(t,regions[i])));
    std::sort(names.begin(),names.end());
    for(const auto& name:names){
        const bool sh=ck_board_flag(t,name,"is_stronghold"),ci=ck_board_flag(t,name,"is_city"),fo=ck_board_flag(t,name,"is_fort"),fp=ck_fp_region(t,name);
        if(side==0&&!fp)continue;if(side==1&&fp)continue;
        if(k=="field"&&!(sh||ci||fo))return name;if(k=="stronghold"&&sh)return name;if(k=="city"&&ci)return name;
    }
    throw ck_Error{"no fixture region"};
}
ck_Leader ck_mk_leader(const ck_LeaderSpec& sp,int default_nation,int& next_id){
    ck_Leader l;l.id=next_id++;const int nat=sp.nation>=0?sp.nation:default_nation;
    if(sp.kind=="nazgul"){l.kind=CK_NAZGUL;l.nation=6;}
    else if(sp.kind=="companion"){l.kind=CK_COMPANION;l.name=sp.name;l.level=sp.level;l.leadership=sp.leadership;}
    else if(sp.kind=="minion"){l.kind=CK_MINION;l.nation=nat;l.name=sp.name;}
    else{l.kind=CK_LEADER;l.nation=nat;}
    return l;
}
std::vector<ck_Unit> ck_mk_units(const ck_ArmySpec& sp){
    std::vector<ck_Unit> u;const int en=sp.elite_nation>=0?sp.elite_nation:sp.nation;
    for(int64_t i=0;i<sp.regulars;++i)u.push_back({sp.nation,false});
    for(int64_t i=0;i<sp.elites;++i)u.push_back({en,true});
    if(sp.elites2){const int en2=sp.elite2_nation>=0?sp.elite2_nation:sp.nation;for(int64_t i=0;i<sp.elites2;++i)u.push_back({en2,true});}
    return u;
}
// army_participants._leader_side (raises for an unsupported figure, as the engine does)
int ck_participant_leader_side(const ck_Leader& l){
    if(l.kind==CK_COMPANION)return 0;
    if(l.kind==CK_NAZGUL)return 1;
    if(l.kind==CK_MINION&&(l.name=="Saruman"||l.name=="Witch King"||l.name=="Mouth of Sauron"))return 1;
    if((l.kind==CK_LEADER)||(l.kind==CK_MINION&&l.name.empty()))return l.nation<0?0:ck_side_of_nation(l.nation);
    throw ck_Error{"Unsupported board leader/character"};
}
// build_round + setup_combat
ck_Combat ck_build_round(const Snapshot& t,const ck_Ctx& ctx,const ck_Choices& ch){
    ck_Combat c;int next_id=1;
    const int atk_side0=ck_side_of_nation(ctx.atk.nation),def_side0=ck_side_of_nation(ctx.dfn.nation);
    const char* def_kind=ctx.is_siege?"stronghold":(ctx.city?"city":"field");
    const std::string atk_region=ctx.has_atk_region?ctx.atk_region:ck_pick_region(t,atk_side0,"field");
    const std::string def_region=ctx.has_def_region?ctx.def_region:ck_pick_region(t,def_side0,def_kind);
    ck_Army a,d;a.units=ck_mk_units(ctx.atk);for(const auto& ls:ctx.atk.leaders)a.leaders.push_back(ck_mk_leader(ls,ctx.atk.nation,next_id));
    d.units=ck_mk_units(ctx.dfn);for(const auto& ls:ctx.dfn.leaders)d.leaders.push_back(ck_mk_leader(ls,ctx.dfn.nation,next_id));
    const std::string atk_key=sanitize_name(atk_region),def_key=sanitize_name(def_region);
    if(atk_key==def_key)throw ck_Error{"co-located kernel round is outside the port"};
    // _saruman_in_play: any Minion named *saruman* in ANY army (the planted one when ctx.saruman_in_play)
    bool saruman=ctx.saruman;
    for(const auto* army:{&a,&d})for(const auto& l:army->leaders)if(l.kind==CK_MINION&&ck_has(ck_lower(l.name),"saruman"))saruman=true;
    auto mixed=[](const ck_Army& x){bool fp=false,sh=false;for(const auto& u:x.units)(ck_side_of_nation(u.nation)?sh:fp)=true;return fp&&sh;};
    if(mixed(a)||mixed(d))throw ck_Error{"mixed-side kernel army is outside the port"};
    // partition_battle_army(attacker): figures of the other side stay behind
    {const int fside=ck_army_side(a);ck_Army part;bool rest=false;
     for(const auto& u:a.units){if(ck_side_of_nation(u.nation)!=fside)rest=true;else part.units.push_back(u);}
     for(const auto& l:a.leaders){if(ck_participant_leader_side(l)!=fside)rest=true;else part.leaders.push_back(l);}
     if(rest)a=std::move(part);}
    c.atk_side=ck_army_side(a);c.def_side=ck_army_side(d);
    if(c.atk_side==c.def_side)throw ck_Error{"Cannot attack friendly army"};
    const bool is_city=ck_board_flag(t,def_key,"is_city"),fort=ck_board_flag(t,def_key,"is_fort");
    c.is_siege=ctx.is_siege;c.city_defense=(is_city||fort)&&!ctx.is_siege;
    std::vector<ck_Leader> atk_comp,def_comp;
    for(const auto& l:a.leaders)if(ck_leader_side(l)==c.atk_side&&l.kind==CK_COMPANION)atk_comp.push_back(l);
    for(const auto& l:d.leaders)if(ck_leader_side(l)==c.def_side&&l.kind==CK_COMPANION)def_comp.push_back(l);
    auto nazgul_ldr=[&](const ck_Army& x,int side){int64_t n=0,wk=0;for(const auto& l:x.leaders){if(ck_leader_side(l)!=side)continue;if(l.kind==CK_NAZGUL)++n;if(l.kind==CK_MINION&&ck_has(ck_lower(l.name),"witch"))wk+=2;}return n+wk;};
    const int64_t atk_naz_ldr=nazgul_ldr(a,c.atk_side),def_naz_ldr=nazgul_ldr(d,c.def_side);
    int gtw=-1;for(const auto* list:{&atk_comp,&def_comp})for(const auto& l:*list)if(gtw<0&&l.name=="Gandalf the White")gtw=l.id;
    const bool gtw_shines=gtw>=0&&ctx.shine!=0;
    c.atk=std::move(a);c.dfn=std::move(d);
    c.companions=def_comp;c.companions.insert(c.companions.end(),atk_comp.begin(),atk_comp.end());
    c.shined=(ctx.shine==1&&gtw>=0)?gtw:-1;c.saruman=saruman;
    c.shine=gtw_shines?gtw:-1;c.shine_cancelled=gtw_shines?(c.atk_side==1?atk_naz_ldr:def_naz_ldr):0;
    ck_recompute_leadership(c);
    c.atk_card=ck_card(t,ctx.has_atk_card,ctx.atk_card);c.def_card=ck_card(t,ctx.has_def_card,ctx.def_card);
    c.atk_red=ch.get("attacker_dice_reduction")&&ch.truthy("attacker_dice_reduction")?ch.integer("attacker_dice_reduction"):0;
    c.def_red=ch.get("defender_dice_reduction")&&ch.truthy("defender_dice_reduction")?ch.integer("defender_dice_reduction"):0;
    if(ch.truthy("disable_city_defense"))c.city_defense=false;
    return c;
}
// ---- the deterministic appliers
bool ck_defender_first(const ck_Combat& c){
    const int64_t ai=c.atk_card.present&&!c.atk_cc?c.atk_card.value:99,di=c.def_card.present&&!c.def_cc?c.def_card.value:99;
    return di<=ai;
}
bool ck_forfeited(const ck_Combat& c,const std::string& name){return std::find(c.forfeited.begin(),c.forfeited.end(),name)!=c.forfeited.end();}
void ck_apply_cancel(ck_Combat& c){
    c.forfeited.clear();if(c.shined>=0)for(const auto& x:c.companions)if(x.id==c.shined){c.forfeited.push_back(x.name);break;}
    // _sh0.name: the shined Companion object is one of the battle rosters (always among companions_in_battle)
    auto resolve=[&](bool is_atk){
        const ck_Card& card=is_atk?c.atk_card:c.def_card;const bool cancelled=is_atk?c.atk_cc:c.def_cc;
        if(!card.present||cancelled)return;
        const auto& e=card.effect;if(!ck_has(e,"cancel the effects of the combat card"))return;
        const bool forfeit=ck_has(e,"forfeit")&&ck_has(e,"leadership");
        if(forfeit){
            const ck_Card& opp=is_atk?c.def_card:c.atk_card;if(!opp.present)return;
            for(const auto& x:c.companions)if(ck_forfeited(c,x.name))return;
        }
        if(is_atk){if(c.def_card.present)c.def_cc=true;}else{if(c.atk_card.present)c.atk_cc=true;}
        if(forfeit){
            int64_t lead=0;
            for(const auto& x:c.companions){if(x.id==c.shined||x.id==c.wop)continue;lead+=x.leadership;if(!ck_forfeited(c,x.name))c.forfeited.push_back(x.name);}
            if(is_atk)c.atk_lead=std::max<int64_t>(0,c.atk_lead-lead);else c.def_lead=std::max<int64_t>(0,c.def_lead-lead);
        }
    };
    if(ck_defender_first(c)){resolve(false);resolve(true);}else{resolve(true);resolve(false);}
}
const ck_Leader* ck_pick_named(const std::vector<ck_Leader>& list,const std::optional<std::string>& name){
    if(!name||name->empty()||*name=="leader")return nullptr;
    for(const auto& x:list)if(x.name==*name)return &x;return nullptr;
}
void ck_apply_words_of_power(ck_Combat& c,const ck_Choices& ch){
    for(int pass=0;pass<2;++pass){
        const bool is_atk=pass==0;const ck_Card& card=is_atk?c.atk_card:c.def_card;const bool cancelled=is_atk?c.atk_cc:c.def_cc;
        if(!card.present||cancelled)continue;const auto& e=card.effect;
        if(!(ck_has(e,"companion")&&ck_has(e,"leadership")&&ck_has(e,"cancelled")))continue;
        if(c.naz<1)continue;
        if(c.companions.empty())continue;
        const ck_Leader* target=ck_pick_named(c.companions,ch.target(is_atk?"atk_effect_target":"def_effect_target"));
        if(!target){const ck_Leader* best=&c.companions[0];for(const auto& x:c.companions)if(x.leadership>best->leadership)best=&x;target=best;}
        const bool fp_is_attacker=!is_atk;
        if(c.shine>=0&&target->id==c.shine){if(c.atk_side==1)c.atk_lead+=c.shine_cancelled;else c.def_lead+=c.shine_cancelled;}
        if(target->id!=c.shined){const int64_t lead=target->leadership;if(fp_is_attacker)c.atk_lead=std::max<int64_t>(0,c.atk_lead-lead);else c.def_lead=std::max<int64_t>(0,c.def_lead-lead);}
        c.wop=target->id;c.wop_is_atk=fp_is_attacker;
    }
}
// re.search(r"add ([12]) to all dice on (?:your|their) ([^.]*)", effect): (amount, target) of the first match
bool ck_add_to_all(const std::string& e,int& amount,std::string& target){
    for(size_t pos=e.find("add ");pos!=std::string::npos;pos=e.find("add ",pos+1)){
        size_t i=pos+4;if(i>=e.size()||(e[i]!='1'&&e[i]!='2'))continue;const int amt=e[i]-'0';++i;
        const std::string mid=" to all dice on ";if(e.compare(i,mid.size(),mid)!=0)continue;i+=mid.size();
        if(e.compare(i,5,"your ")==0)i+=5;else if(e.compare(i,6,"their ")==0)i+=6;else continue;
        size_t j=i;while(j<e.size()&&e[j]!='.')++j;amount=amt;target=e.substr(i,j-i);return true;
    }
    return false;
}
void ck_apply_dice_modifiers(ck_Combat& c){
    auto apply=[&](const std::string& e,bool is_atk){
        int64_t& dm=is_atk?c.atk_dm:c.def_dm;int64_t& rm=is_atk?c.atk_rm:c.def_rm;int64_t& lead=is_atk?c.atk_lead:c.def_lead;int64_t& enemy_dm=is_atk?c.def_dm:c.atk_dm;
        const ck_Card& opp=is_atk?c.def_card:c.atk_card;
        if(ck_has(e,"both armies add 2")){c.atk_dm+=2;c.def_dm+=2;c.atk_rm+=2;c.def_rm+=2;}
        else if(ck_has(e,"both armies add 1")){c.atk_dm+=1;c.def_dm+=1;c.atk_rm+=1;c.def_rm+=1;}
        else if(ck_has(e,"subtract 1")&&ck_has(e,"nazg")&&ck_has(e,"leader re-roll")){}
        else if(ck_has(e,"subtract 2")&&ck_has(e,"nazg")&&ck_has(e,"add 1")){if(c.shine<0){lead=std::max<int64_t>(0,lead-2);dm+=1;}}
        else if(ck_has(e,"inflict")&&ck_has(e,"up to two hits")){}
        else{
            int amt=0;std::string target;
            if(ck_add_to_all(e,amt,target)){
                if(ck_has(target,"combat"))dm+=amt;
                if(ck_has(target,"re-roll")){if(ck_has(e,"did not play a card")){if(!opp.present)rm+=amt;}else rm+=amt;}
            }
        }
        if(ck_has(e,"subtract 1 from all dice")&&ck_has(e,"shadow"))enemy_dm+=-1;
    };
    if(c.atk_card.present&&!c.atk_cc)apply(c.atk_card.effect,true);
    if(c.def_card.present&&!c.def_cc)apply(c.def_card.effect,false);
}
void ck_apply_dice_count(ck_Combat& c){
    if(c.atk_card.present&&!c.atk_cc){const auto& e=c.atk_card.effect;
        if(ck_has(e,"maximum of two dice")){c.has_def_cap=true;c.def_cap=2;}else if(ck_has(e,"maximum of three dice")){c.has_def_cap=true;c.def_cap=3;}
        if(ck_has(e,"rolls one die less")&&ck_has(e,"companion"))c.def_red+=static_cast<int64_t>(c.companions.size());}
    if(c.def_card.present&&!c.def_cc){const auto& e=c.def_card.effect;
        if(ck_has(e,"maximum of two dice")){c.has_atk_cap=true;c.atk_cap=2;}else if(ck_has(e,"maximum of three dice")){c.has_atk_cap=true;c.atk_cap=3;}
        if(ck_has(e,"rolls one die less")&&ck_has(e,"companion"))c.atk_red+=static_cast<int64_t>(c.companions.size());}
}
void ck_apply_dread(ck_Combat& c,const ck_Choices& ch){
    for(int pass=0;pass<2;++pass){
        const bool is_atk=pass==0;const ck_Card& card=is_atk?c.atk_card:c.def_card;const bool cancelled=is_atk?c.atk_cc:c.def_cc;
        if(!card.present||cancelled)continue;const auto& e=card.effect;
        if(!(ck_has(e,"forfeit")&&ck_has(e,"nazg")&&ck_has(e,"one die less")))continue;
        if(c.shine>=0)continue;
        const int64_t available=c.naz;if(available<=0)continue;
        const ck_Army& fp=is_atk?c.dfn:c.atk;const int64_t fp_strength=ck_strength(fp,-1);
        const std::string key=is_atk?"atk_effect_count":"def_effect_count";int64_t forfeit;
        if(ch.has_value(key)){forfeit=ch.integer(key);if(!(1<=forfeit&&forfeit<=available))throw ck_Error{"Dread and Despair must forfeit between 1 and the available Nazgul leadership"};}
        else{const int64_t desired=std::max<int64_t>(0,fp_strength-1);forfeit=std::min(std::max<int64_t>(1,desired),std::max<int64_t>(1,available-1));}
        if(forfeit>0){
            if(is_atk){c.atk_lead-=forfeit;c.has_def_cap=true;c.def_cap=std::max<int64_t>(1,fp_strength-forfeit);}
            else{c.def_lead-=forfeit;c.has_atk_cap=true;c.atk_cap=std::max<int64_t>(1,fp_strength-forfeit);}
            c.naz-=forfeit;
        }
    }
}
void ck_apply_confusion(ck_Combat& c){
    if(c.def_card.present&&!c.def_cc){const auto& e=c.def_card.effect;if(ck_has(e,"result of")&&ck_has(e,"1")&&ck_has(e,"shadow army"))c.atk_conf=true;}
    if(c.atk_card.present&&!c.atk_cc){const auto& e=c.atk_card.effect;if(ck_has(e,"result of")&&ck_has(e,"1")&&ck_has(e,"shadow army"))c.def_conf=true;}
}
void ck_apply_reroll_cancellation(ck_Combat& c){
    const int64_t naz=c.shine>=0?0:c.naz;
    if(c.atk_card.present&&!c.atk_cc&&ck_has(c.atk_card.effect,"leader re-roll is cancelled")&&naz>=c.def_lead){c.def_rrc=true;c.fs_def=true;}
    if(c.def_card.present&&!c.def_cc&&ck_has(c.def_card.effect,"leader re-roll is cancelled")&&naz>=c.atk_lead){c.atk_rrc=true;c.fs_atk=true;}
}
void ck_apply_they_are_terrible(ck_Combat& c){
    for(int pass=0;pass<2;++pass){
        const bool is_atk=pass==0;const ck_Card& card=is_atk?c.atk_card:c.def_card;const bool cancelled=is_atk?c.atk_cc:c.def_cc;
        if(!card.present||cancelled)continue;const auto& e=card.effect;
        if(!(ck_has(e,"subtract 1")&&ck_has(e,"nazg")&&ck_has(e,"leader re-roll")))continue;
        if(c.shine>=0)continue;
        if(c.naz>=1){c.naz-=1;if(is_atk){c.atk_lead-=1;c.atk_rm+=1;}else{c.def_lead-=1;c.def_rm+=1;}}
    }
}
// ---- stage 0 parameters
struct ck_Side {
    int64_t t=5,m=0,rr_m=0;bool has_cap=false;int64_t cap=0,red=0,L3=0,L5=0;bool rr_cancelled=false,confused=false,track_ones=false;
    std::string eff,name;bool has_card=false;int ma_kind=0;/*0 none,1 anduril,2 ma*/int64_t ma_pay=0,ma_max=0;bool ma_blocked=false;
    int64_t captains=0;std::string policy="preserve_elites";int pre_kind=0;/*0 none,1 ss,2 charge,3 db*/std::vector<int> sar_lead;
};
struct ck_Params {
    ck_Side atk,dfn;bool def_first=false;int64_t naz=0;int64_t ra=0,ea=0,rd=0,ed=0;
    std::vector<ck_Leader> atk_leaders,def_leaders,companions;bool is_siege=false,city_r1=false,battle_over=false,destA=false,destD=false;
    bool defer=false;
};
void ck_counts(const ck_Army& a,int64_t& r,int64_t& e){r=0;e=0;for(const auto& u:a.units)(u.elite?e:r)+=1;}
ck_Side ck_side_params(const ck_Combat& c,bool is_atk,const ck_Choices& ch,bool conf_present){
    ck_Side s;const ck_Card& card=is_atk?c.atk_card:c.def_card;const bool cancelled=is_atk?c.atk_cc:c.def_cc;
    s.has_card=card.present&&!cancelled;s.eff=s.has_card?card.effect:"";s.name=s.has_card?card.name:"";
    if(is_atk){s.t=(c.is_siege||(c.city_defense&&c.siege_round==1))?6:5;s.m=c.atk_dm;s.rr_m=c.atk_rm;s.has_cap=c.has_atk_cap;s.cap=c.atk_cap;s.red=c.atk_red;s.rr_cancelled=c.atk_rrc;s.confused=c.atk_conf;s.policy=ch.str_or("atk_casualty_policy","preserve_elites");}
    else{s.t=5;s.m=c.def_dm;s.rr_m=c.def_rm;s.has_cap=c.has_def_cap;s.cap=c.def_cap;s.red=c.def_red;s.rr_cancelled=c.def_rrc;s.confused=c.def_conf;s.policy=ch.str_or("def_casualty_policy","preserve_elites");}
    s.track_ones=s.confused||conf_present||(1+s.m>=s.t);
    if(s.has_card&&ck_has(s.eff,"before the combat roll")&&ck_has(s.eff,"additional attack")){
        if(ck_has(s.eff,"leadership"))s.pre_kind=1;else if(ck_has(s.eff,"elite units"))s.pre_kind=2;else if(ck_has(s.eff,"three combat dice"))s.pre_kind=3;
    }
    if(s.has_card&&ck_has(s.eff,"forfeit")&&ck_has(s.eff,"leadership")&&ck_has(s.eff,"change")&&ck_has(s.eff,"missed")){
        s.ma_blocked=is_atk?c.fs_atk:c.fs_def;
        std::vector<ck_Leader> avail;for(const auto& x:c.companions)if(x.id!=c.shined&&x.id!=c.wop)avail.push_back(x);
        if(ck_has(s.eff,"aragorn")||ck_has(s.eff,"strider")){
            const ck_Leader* hero=nullptr;for(const auto& x:avail){const auto nm=ck_lower(x.name);if(ck_has(nm,"aragorn")||ck_has(nm,"strider")){hero=&x;break;}}
            if(hero){const bool crowned=hero->leadership>=2;s.ma_kind=1;s.ma_max=crowned?2:1;s.ma_pay=hero->leadership;}
        }else if(!avail.empty()){
            const auto named=ch.target(is_atk?"atk_effect_target":"def_effect_target");const ck_Leader* target=nullptr;
            if(ch.get(is_atk?"atk_effect_target":"def_effect_target")&&ch.get(is_atk?"atk_effect_target":"def_effect_target")->kind!=0)target=ck_pick_named(avail,named);
            else{target=&avail[0];for(const auto& x:avail)if(x.leadership<target->leadership)target=&x;}
            if(!target)throw ck_Error{"Mighty Attack payer is not currently eligible"};
            s.ma_kind=2;s.ma_max=1;s.ma_pay=target->leadership;
        }
    }
    return s;
}
ck_Params ck_stage0(const Snapshot& t,const ck_Ctx& ctx,ck_Combat* out_combat=nullptr){
    ck_Choices ch;for(const auto& it:ctx.choices)ch.put(it.first,it.second);
    if(ctx.defer){ck_Value v;v.kind=1;v.i=1;ch.put("defer_casualties",v);}
    ck_Combat c=ck_build_round(t,ctx,ch);
    ck_apply_cancel(c);ck_apply_words_of_power(c,ch);ck_apply_dice_modifiers(c);ck_apply_dice_count(c);ck_apply_dread(c,ch);
    ck_Params p;p.defer=ch.truthy("defer_casualties");p.is_siege=c.is_siege;p.city_r1=c.city_defense&&c.siege_round==1;
    p.battle_over=c.battle_over;ck_counts(c.atk,p.ra,p.ea);ck_counts(c.dfn,p.rd,p.ed);
    p.atk_leaders=c.atk.leaders;p.def_leaders=c.dfn.leaders;p.companions=c.companions;p.def_first=ck_defender_first(c);
    const int64_t L3a=c.atk_lead,L3d=c.def_lead;
    if(!c.battle_over){ck_apply_confusion(c);ck_apply_reroll_cancellation(c);ck_apply_they_are_terrible(c);}
    const bool conf_present=c.atk_conf||c.def_conf;
    p.atk=ck_side_params(c,true,ch,conf_present);p.dfn=ck_side_params(c,false,ch,conf_present);
    p.atk.L3=L3a;p.dfn.L3=L3d;p.atk.L5=c.atk_lead;p.dfn.L5=c.def_lead;p.naz=c.naz;
    if(c.saruman){
        for(auto* pr:{&p.atk,&p.dfn}){const ck_Army& army=pr==&p.atk?c.atk:c.dfn;for(const auto& u:army.units)if(u.elite)pr->sar_lead.push_back(u.nation==5?1:0);}
    }
    auto capt=[&](const std::vector<ck_Leader>& leaders,bool side_is_atk){
        const int cancelled=(c.wop>=0&&c.wop_is_atk==side_is_atk)?c.wop:-1;int64_t n=0;
        for(const auto& l:leaders)if(l.kind==CK_COMPANION&&ck_captain(l.name)&&l.id!=cancelled)++n;return n;
    };
    p.atk.captains=capt(p.atk_leaders,true);p.dfn.captains=capt(p.def_leaders,false);
    if(out_combat)*out_combat=std::move(c);
    return p;
}

// ---- probability helpers (combat_dist)
int64_t ck_hit_faces(int64_t t,int64_t m){int64_t n=0;for(int64_t r=2;r<=6;++r)if(r==6||r+m>=t)++n;return n;}
std::vector<std::pair<int64_t,double>> ck_binom_terms(int64_t n,double p){
    std::vector<std::pair<int64_t,double>> out;
    for(int64_t k=0;k<=n;++k){const double pr=(static_cast<double>(cd_comb(n,k))*cd_pow(p,static_cast<double>(k)))*cd_pow(1.0-p,static_cast<double>(n-k));if(pr>0.0)out.push_back({k,pr});}
    return out;
}
int64_t ck_fact(int64_t n){int64_t f=1;for(int64_t i=2;i<=n;++i)f*=i;return f;}
std::vector<std::pair<std::pair<int64_t,int64_t>,double>> ck_trinom_terms(int64_t n,double ph,double po){
    const double poth=(1.0-ph)-po;std::vector<std::pair<std::pair<int64_t,int64_t>,double>> out;
    for(int64_t h=0;h<=n;++h)for(int64_t o=0;o<=n-h;++o){const int64_t r=n-h-o;
        const double pr=(((static_cast<double>(ck_fact(n))/static_cast<double>(ck_fact(h)*ck_fact(o)*ck_fact(r)))*cd_pow(ph,static_cast<double>(h)))*cd_pow(po,static_cast<double>(o)))*cd_pow(poth,static_cast<double>(r));
        if(pr>0.0)out.push_back({{h,o},pr});}
    return out;
}
std::pair<int64_t,int64_t> ck_apply_hits(int64_t r,int64_t e,int64_t h,const std::string& policy){
    if(policy=="preserve_units"){while(h>0&&(r+e)>0){if(e>0){e-=1;r+=1;}else r-=1;h-=1;}return {r,e};}
    if(policy=="spend_elites"){while(h>0&&(r+e)>0){if(e>0){e-=1;if(h>=2)h-=2;else{r+=1;h-=1;}}else{r-=1;h-=1;}}return {r,e};}
    const int64_t k=std::min(r,h);r-=k;h-=k;
    while(h>0&&e>0){e-=1;if(h>=2)h-=2;else{r+=1;h-=1;}}
    return {r,e};
}
std::string ck_fx(const ck_Leader& l){
    if(l.kind==CK_COMPANION)return "C:"+l.name;if(l.kind==CK_NAZGUL)return "N";if(l.kind==CK_MINION)return "M:"+l.name;return "L";
}
// ---- the distribution (ExchangeDist: insertion-ordered outcomes, first witness plan per outcome)
struct ck_PlanItem {int64_t h=0,o=0;bool tuple=false;};
using ck_Plan=std::vector<ck_PlanItem>;
struct ck_Obs {int64_t v[12];bool destA,destD;std::vector<std::string> fxa,fxd;bool deferred;int64_t dhd,dha;};
struct ck_Dist {
    std::vector<ck_Obs> obs;std::vector<double> mass;std::vector<ck_Plan> plan;std::unordered_map<std::string,size_t> index;bool witness=false;
    static std::string key(const ck_Obs& o){
        std::string k;for(int i=0;i<12;++i){k+=std::to_string(o.v[i]);k+=',';}
        k+=o.destA?'1':'0';k+=o.destD?'1':'0';k+='|';for(const auto& s:o.fxa){k+=s;k+='\x1f';}k+='|';for(const auto& s:o.fxd){k+=s;k+='\x1f';}
        k+='|';k+=o.deferred?'1':'0';k+=',';k+=std::to_string(o.dhd);k+=',';k+=std::to_string(o.dha);return k;
    }
    void add(ck_Obs o,double pr,const ck_Plan& cur){
        std::sort(o.fxa.begin(),o.fxa.end());std::sort(o.fxd.begin(),o.fxd.end());
        const auto k=key(o);auto it=index.find(k);
        if(it==index.end()){index.emplace(k,obs.size());obs.push_back(std::move(o));mass.push_back(0.0+pr);plan.push_back(cur);}
        else mass[it->second]=mass[it->second]+pr;
    }
};
struct ck_Emit {
    ck_Dist& dist;ck_Plan cur;
    void operator()(double prob,int64_t ah,int64_t dh,int64_t pa,int64_t pd,int64_t ba,int64_t bd,int64_t ca,int64_t cd_,int64_t ra,int64_t ea,int64_t rd,int64_t ed,
                    bool destA,bool destD,const std::vector<std::string>& fxa,const std::vector<std::string>& fxd,bool deferred=false,int64_t dhd=0,int64_t dha=0){
        ck_Obs o;const int64_t v[12]={ah,dh,pa,pd,ba,bd,ca,cd_,ra,ea,rd,ed};std::copy(v,v+12,o.v);o.destA=destA;o.destD=destD;o.fxa=fxa;o.fxd=fxd;o.deferred=deferred;o.dhd=dhd;o.dha=dha;
        dist.add(std::move(o),prob,cur);
    }
};
// combat_dist._finish
void ck_finish(const ck_Params& p,const ck_Choices& ch,ck_Emit& emit,double prob,int64_t ra,int64_t ea,int64_t rd,int64_t ed,int64_t preA,int64_t preD,
               int64_t atkH,int64_t defH,int64_t rrA,int64_t rrD){
    std::vector<ck_Leader> rostA=p.atk_leaders,rostD=p.def_leaders,comps=p.companions;
    std::vector<std::string> fxa,fxd;int64_t bonusA=0,bonusD=0,cancA=0,cancD=0;
    auto remove_id=[](std::vector<ck_Leader>& v,int id){for(size_t i=0;i<v.size();++i)if(v[i].id==id){v.erase(v.begin()+static_cast<ptrdiff_t>(i));return;}};
    auto fs=[&](const ck_Side& sp,int64_t rh,std::vector<ck_Leader>& enemy,std::vector<std::string>& enemy_fx,const std::optional<std::string>& tgt){
        if(!sp.has_card)return;
        if(!ck_has(sp.eff,"re-roll")||!ck_has(sp.eff,"nazg")||!ck_has(sp.name,"fateful strike"))return;
        if(rh>=1){
            if(tgt){
                std::vector<ck_Leader> eligible;for(const auto& l:enemy)if((*tgt=="nazgul"&&l.kind==CK_NAZGUL)||(*tgt!="nazgul"&&rh>=2&&l.kind==CK_MINION&&l.name==*tgt))eligible.push_back(l);
                if(eligible.empty())throw ck_Error{"Fateful Strike target is not currently eligible"};
                const auto victim=eligible[0];remove_id(enemy,victim.id);enemy_fx.push_back(ck_fx(victim));return;
            }
            if(rh>=2){for(const auto& l:std::vector<ck_Leader>(enemy))if(l.kind==CK_MINION){remove_id(enemy,l.id);enemy_fx.push_back(ck_fx(l));return;}}
            for(const auto& l:std::vector<ck_Leader>(enemy))if(l.kind==CK_NAZGUL){remove_id(enemy,l.id);enemy_fx.push_back(ck_fx(l));return;}
        }
    };
    auto bow=[&](const ck_Side& sp,bool sp_is_atk,int64_t rh,std::vector<ck_Leader>& enemy,std::vector<std::string>& enemy_fx,const std::optional<std::string>& tgt){
        if(!sp.has_card)return;
        if(!ck_has(sp.eff,"eliminate")||!ck_has(sp.eff,"minion")||!ck_has(sp.name,"westernesse"))return;
        if(rh>=1){
            std::vector<ck_Leader> minions,named;for(const auto& l:enemy)if(l.kind==CK_MINION)minions.push_back(l);
            for(const auto& l:minions)if(tgt&&!tgt->empty()&&l.name==*tgt)named.push_back(l);
            if(tgt&&!minions.empty()&&named.empty())throw ck_Error{"Blade target is not a participating Minion"};
            const auto& list=named.empty()?minions:named;
            for(const auto& l:list){if(sp_is_atk)atkH=std::max<int64_t>(0,atkH-1);else defH=std::max<int64_t>(0,defH-1);remove_id(enemy,l.id);enemy_fx.push_back(ck_fx(l));return;}
        }
    };
    auto bonus=[&](const ck_Side& sp,bool is_att){
        if(!sp.has_card)return;
        const int64_t at=atkH+preA+bonusA,dt=defH+preD+bonusD;const int64_t my=is_att?at:dt,en=is_att?dt:at;int64_t add=0;
        const auto& e=sp.eff;const bool scored=ck_has(e,"score at least one hit")||ck_has(e,"scores at least one hit");
        if(scored&&ck_has(e,"score one additional hit")&&my>=1)add+=1;
        if(scored&&ck_has(e,"score two additional hits")&&my>=1)add+=2;
        if(ck_has(e,"more total hits")&&ck_has(e,"score one additional hit")&&my>en)add+=1;
        if(is_att)bonusA+=add;else bonusD+=add;
    };
    bonus(p.dfn,false);bonus(p.atk,true);
    if(p.atk.has_card&&ck_has(p.atk.eff,"two or more hits")&&ck_has(p.atk.eff,"cancel one hit")&&defH>=2)cancA+=1;
    if(p.dfn.has_card&&ck_has(p.dfn.eff,"two or more hits")&&ck_has(p.dfn.eff,"cancel one hit")&&atkH>=2)cancD+=1;
    auto bb=[&](const ck_Side& sp,int64_t rh,std::vector<ck_Leader>& enemy,std::vector<std::string>& enemy_fx,const std::optional<std::string>& tgt){
        if(!sp.has_card)return;if(p.naz<1)return;
        if(!ck_has(sp.eff,"re-roll")||!ck_has(sp.eff,"eliminate")||!ck_has(sp.eff,"leader"))return;
        if(!ck_has(sp.name,"black breath"))return;if(rh<1)return;
        std::vector<ck_Leader> viable;for(const auto& x:comps)if(x.level<=rh)viable.push_back(x);
        if(tgt&&*tgt=="leader")viable.clear();
        if(!viable.empty()){
            const ck_Leader* target=ck_pick_named(viable,tgt);ck_Leader chosen;
            if(target)chosen=*target;else{const ck_Leader* best=&viable[0];for(const auto& x:viable)if(x.level>best->level)best=&x;chosen=*best;}
            remove_id(comps,chosen.id);
            for(const auto& l:std::vector<ck_Leader>(enemy))if(l.kind==CK_COMPANION&&l.name==chosen.name){remove_id(enemy,l.id);break;}
            enemy_fx.push_back("C:"+chosen.name);return;
        }
        for(const auto& l:std::vector<ck_Leader>(enemy))if(l.kind!=CK_COMPANION&&l.kind!=CK_NAZGUL){remove_id(enemy,l.id);enemy_fx.push_back(ck_fx(l));return;}
    };
    auto hd_sac=[&](bool is_atk,std::vector<ck_Leader>& rost,std::vector<std::string>& fx,int64_t fc,int64_t total_enemy,const std::optional<std::string>& tgt){
        if(fc<=0||total_enemy<=0)return;
        if((fc==1&&!tgt)||(tgt&&*tgt=="leader")){
            for(const auto& l:std::vector<ck_Leader>(rost))if(l.kind!=CK_COMPANION&&l.kind!=CK_NAZGUL){remove_id(rost,l.id);fx.push_back(ck_fx(l));if(is_atk)cancA+=1;else cancD+=1;return;}
        }
        const ck_Leader* named=ck_pick_named(comps,tgt);
        std::vector<ck_Leader> viable;for(const auto& x:comps)if(x.level>=std::max<int64_t>(1,fc))viable.push_back(x);
        std::stable_sort(viable.begin(),viable.end(),[](const ck_Leader& a,const ck_Leader& b){return a.level<b.level;});
        if(viable.empty()){viable=comps;std::stable_sort(viable.begin(),viable.end(),[](const ck_Leader& a,const ck_Leader& b){return -a.level< -b.level;});}
        if(!named&&viable.empty())return;
        const ck_Leader target=named?*named:viable[0];
        remove_id(comps,target.id);
        for(const auto& l:std::vector<ck_Leader>(rost))if(l.kind==CK_COMPANION&&l.name==target.name){remove_id(rost,l.id);break;}
        fx.push_back("C:"+target.name);
        const int64_t amt=std::min(std::min(std::max<int64_t>(fc,1),target.level),total_enemy);
        if(is_atk)cancA+=amt;else cancD+=amt;
    };
    auto hd=[&](const ck_Side& sp,bool is_atk,std::vector<ck_Leader>& rost,std::vector<std::string>& fx){
        if(!sp.has_card||!ck_has(sp.name,"heroic death"))return;
        int64_t total_enemy=is_atk?defH+bonusD-cancA:atkH+bonusA-cancD;total_enemy=std::max<int64_t>(0,total_enemy);
        const int64_t army_size=is_atk?(ra+ea):(rd+ed);
        const std::string ckey=is_atk?"atk_effect_count":"def_effect_count";
        if(ch.has_value(ckey)){hd_sac(is_atk,rost,fx,ch.integer(ckey),total_enemy,ch.target(is_atk?"atk_effect_target":"def_effect_target"));return;}
        if(total_enemy<=army_size/2)return;   // non-negative operands: // == /
        const ck_Leader* best=nullptr;int64_t best_value=0;
        for(const auto& x:comps){const int64_t cancel=std::min(x.level,total_enemy);if(cancel>best_value){best=&x;best_value=cancel;}}
        if(best&&best_value>=2){
            const ck_Leader chosen=*best;remove_id(comps,chosen.id);
            for(const auto& l:std::vector<ck_Leader>(rost))if(l.kind==CK_COMPANION&&l.name==chosen.name){remove_id(rost,l.id);break;}
            fx.push_back("C:"+chosen.name);if(is_atk)cancA+=best_value;else cancD+=best_value;return;
        }
        for(const auto& l:std::vector<ck_Leader>(rost))if(l.kind!=CK_COMPANION&&l.kind!=CK_NAZGUL){remove_id(rost,l.id);fx.push_back(ck_fx(l));if(is_atk)cancA+=1;else cancD+=1;return;}
    };
    auto bb_hd_side=[&](bool is_att){
        if(is_att){fs(p.atk,rrA,rostD,fxd,ch.target("atk_effect_target"));bow(p.atk,true,rrA,rostD,fxd,ch.target("atk_effect_target"));
            if(ch.not_zero("atk_effect_count"))bb(p.atk,rrA,rostD,fxd,ch.target("atk_effect_target"));hd(p.atk,true,rostA,fxa);}
        else{fs(p.dfn,rrD,rostA,fxa,ch.target("def_effect_target"));bow(p.dfn,false,rrD,rostA,fxa,ch.target("def_effect_target"));
            if(ch.not_zero("def_effect_count"))bb(p.dfn,rrD,rostA,fxa,ch.target("def_effect_target"));hd(p.dfn,false,rostD,fxd);}
    };
    if(p.def_first){bb_hd_side(false);bb_hd_side(true);}else{bb_hd_side(true);bb_hd_side(false);}
    const int64_t finalA=std::max<int64_t>(0,atkH+bonusA-cancD),finalD=std::max<int64_t>(0,defH+bonusD-cancA);
    if(p.defer){emit(prob,atkH,defH,preA,preD,bonusA,bonusD,cancA,cancD,ra,ea,rd,ed,false,false,fxa,fxd,true,finalA,finalD);return;}
    const auto [rd2,ed2]=ck_apply_hits(rd,ed,finalA,p.dfn.policy);const auto [ra2,ea2]=ck_apply_hits(ra,ea,finalD,p.atk.policy);
    const bool destA=(ra2+ea2==0),destD=(rd2+ed2==0),over=destA||destD;
    if(over){emit(prob,atkH,defH,preA,preD,bonusA,bonusD,cancA,cancD,ra2,ea2,rd2,ed2,destA,destD,fxa,fxd);return;}
    struct S9 {int64_t ra,ea,rd,ed,bA,bD;bool dA,dD,over;int64_t xd,xa;double pf;};
    auto g_side=[&](const S9& st,bool is_att)->std::vector<S9>{
        const ck_Side& sp=is_att?p.atk:p.dfn;if(st.over)return {st};if(!sp.has_card)return {st};
        const auto& eff=sp.eff;const int64_t my_r=is_att?st.ra:st.rd,my_e=is_att?st.ea:st.ed,en_r=is_att?st.rd:st.ra,en_e=is_att?st.ed:st.ea;
        const std::string& en_pol=is_att?p.dfn.policy:p.atk.policy;
        std::string stripped=eff;for(size_t pos;(pos=stripped.find("result of "))!=std::string::npos;)stripped.erase(pos,10);
        const bool onslaught=(ck_has(eff,"up to four additional hits")&&ck_has(stripped,"4+"))||(ck_has(eff,"four additional hits")&&ck_has(eff,"score one hit"));
        if(onslaught){
            const int64_t cap=2*my_e+my_r,k=std::min<int64_t>(4,cap);
            if(k>0&&ch.truthy("defer_onslaught"))return {st};
            if(en_r+en_e==0||k<=0)return {st};
            const auto [m_r,m_e]=ck_apply_hits(my_r,my_e,k,sp.policy);std::vector<S9> out;
            for(const auto& term:ck_binom_terms(k,0.5)){
                int64_t e_r=en_r,e_e=en_e;for(int64_t i=0;i<term.first;++i){auto x=ck_apply_hits(e_r,e_e,1,en_pol);e_r=x.first;e_e=x.second;}
                const bool ndA=is_att?(m_r+m_e==0):(e_r+e_e==0),ndD=is_att?(e_r+e_e==0):(m_r+m_e==0),nover=ndA||ndD;
                if(is_att)out.push_back({m_r,m_e,e_r,e_e,st.bA,st.bD,st.dA||ndA,st.dD||ndD,nover,st.xd+term.first,st.xa+k,st.pf*term.second});
                else out.push_back({e_r,e_e,m_r,m_e,st.bA,st.bD,st.dA||ndA,st.dD||ndD,nover,st.xd+k,st.xa+term.first,st.pf*term.second});
            }
            return out;
        }
        if(ck_has(eff,"elite")&&ck_has(eff,"additional")&&ck_has(eff,"after removing casualties")){
            if(en_r+en_e==0)return {st};const int64_t ec=std::min<int64_t>(5,my_e);if(ec<=0)return {st};std::vector<S9> out;
            for(const auto& term:ck_binom_terms(ec,static_cast<double>(ck_hit_faces(5,0))/6.0)){
                const int64_t h=term.first;int64_t e_r=en_r,e_e=en_e;if(h!=0){auto x=ck_apply_hits(en_r,en_e,h,en_pol);e_r=x.first;e_e=x.second;}
                const int64_t nb_A=st.bA+(is_att?h:0),nb_D=st.bD+(is_att?0:h);const bool dead=e_r+e_e==0;
                if(is_att)out.push_back({st.ra,st.ea,e_r,e_e,nb_A,nb_D,st.dA,st.dD||dead,st.over||dead,st.xd+h,st.xa,st.pf*term.second});
                else out.push_back({e_r,e_e,st.rd,st.ed,nb_A,nb_D,st.dA||dead,st.dD,st.over||dead,st.xd,st.xa+h,st.pf*term.second});
            }
            return out;
        }
        if(ck_has(eff,"twice as many")&&ck_has(eff,"automatic hit")){
            if(en_r+en_e==0)return {st};
            if((my_r+my_e)>=2*(en_r+en_e)){
                auto x=ck_apply_hits(en_r,en_e,1,en_pol);const bool dead=x.first+x.second==0;
                if(is_att)return {S9{st.ra,st.ea,x.first,x.second,st.bA+1,st.bD,st.dA,st.dD||dead,st.over||dead,st.xd+1,st.xa,st.pf}};
                return {S9{x.first,x.second,st.rd,st.ed,st.bA,st.bD+1,st.dA||dead,st.dD,st.over||dead,st.xd,st.xa+1,st.pf}};
            }
        }
        return {st};
    };
    std::vector<S9> states{{ra2,ea2,rd2,ed2,bonusA,bonusD,destA,destD,over,0,0,1.0}};
    for(bool is_att:p.def_first?std::vector<bool>{false,true}:std::vector<bool>{true,false}){std::vector<S9> nxt;for(const auto& s9:states){auto more=g_side(s9,is_att);nxt.insert(nxt.end(),more.begin(),more.end());}states=std::move(nxt);}
    for(const auto& s9:states)emit(prob*s9.pf,atkH,defH,preA,preD,s9.bA,s9.bD,cancA,cancD,s9.ra,s9.ea,s9.rd,s9.ed,s9.dA,s9.dD,fxa,fxd);
}
// combat_dist.exact_distribution
ck_Dist ck_exact_distribution(const Snapshot& t,const ck_Ctx& ctx,bool want_witness){
    if(want_witness&&!ctx.defer)throw ck_Error{"witness plans are defer-mode only (in-tree rounds); full-round mode branches inside phase 9"};
    ck_Params base=ck_stage0(t,ctx);
    ck_Choices ch;for(const auto& it:ctx.choices)ch.put(it.first,it.second);
    if(ctx.defer){ck_Value v;v.kind=1;v.i=1;ch.put("defer_casualties",v);}
    ck_Dist dist;dist.witness=want_witness;ck_Emit emit{dist,{}};
    if(base.battle_over){emit(1.0,0,0,0,0,0,0,0,0,base.ra,base.ea,base.rd,base.ed,base.destA,base.destD,{},{});return dist;}
    // ---- Stage A: pre-combat additional attacks
    struct AState {int64_t ra,ea,rd,ed,preA,preD;bool over,dA,dD;};
    auto akey=[](const AState& s){return std::to_string(s.ra)+","+std::to_string(s.ea)+","+std::to_string(s.rd)+","+std::to_string(s.ed)+","+std::to_string(s.preA)+","+std::to_string(s.preD)+","+(s.over?"1":"0")+(s.dA?"1":"0")+(s.dD?"1":"0");};
    struct AEntry {AState st;double pr;ck_Plan plan;};
    std::vector<AEntry> a_states{{{base.ra,base.ea,base.rd,base.ed,0,0,false,false,false},1.0,{}}};
    for(bool is_att:base.def_first?std::vector<bool>{false,true}:std::vector<bool>{true,false}){
        const ck_Side& sp=is_att?base.atk:base.dfn;if(sp.pre_kind==0)continue;
        std::vector<AEntry> nxt;std::unordered_map<std::string,size_t> idx;
        auto put=[&](const AState& k,double add,const ck_Plan& plan){const auto kk=akey(k);auto it=idx.find(kk);
            if(it==idx.end()){idx.emplace(kk,nxt.size());nxt.push_back({k,0.0+add,plan});}else nxt[it->second].pr=nxt[it->second].pr+add;};
        for(const auto& entry:a_states){
            const AState& st=entry.st;const double pr=entry.pr;
            if(st.over){put(st,pr,entry.plan);continue;}
            int64_t dice,tt;
            if(sp.pre_kind==1){dice=std::min<int64_t>(5,sp.L3);tt=(is_att&&(base.is_siege||base.city_r1))?6:5;}
            else if(sp.pre_kind==2){const int64_t my_e=is_att?st.ea:st.ed;dice=std::min<int64_t>(5,my_e);tt=(is_att&&(base.is_siege||base.city_r1))?6:5;}
            else{dice=3;tt=4;}
            if(dice<=0){put(st,pr,entry.plan);continue;}
            const double ph=static_cast<double>(ck_hit_faces(tt,0))/6.0;
            for(const auto& term:ck_binom_terms(dice,ph)){
                const int64_t h=term.first;AState s2=st;
                if(is_att){s2.preA+=h;if(h>0){auto x=ck_apply_hits(st.rd,st.ed,h,base.dfn.policy);s2.rd=x.first;s2.ed=x.second;if(s2.rd+s2.ed==0){s2.dD=true;s2.over=true;}}}
                else{s2.preD+=h;if(h>0){auto x=ck_apply_hits(st.ra,st.ea,h,base.atk.policy);s2.ra=x.first;s2.ea=x.second;if(s2.ra+s2.ea==0){s2.dA=true;s2.over=true;}}}
                ck_Plan plan=entry.plan;ck_PlanItem it;it.h=h;plan.push_back(it);put(s2,pr*term.second,plan);
            }
        }
        a_states=std::move(nxt);
    }
    std::map<std::pair<int64_t,int64_t>,std::vector<std::pair<int64_t,double>>> reroll_memo;
    auto reroll_terms=[&](int64_t k,int64_t h6)->const std::vector<std::pair<int64_t,double>>&{
        static const std::vector<std::pair<int64_t,double>> none{{0,1.0}};if(k<=0)return none;
        auto key=std::make_pair(k,h6);auto it=reroll_memo.find(key);if(it==reroll_memo.end())it=reroll_memo.emplace(key,ck_binom_terms(k,static_cast<double>(h6)/6.0)).first;return it->second;
    };
    for(const auto& entry:a_states){
        ck_Params p=base;
        int64_t ra0=entry.st.ra,ea0=entry.st.ea,rd0=entry.st.rd,ed0=entry.st.ed;const int64_t preA=entry.st.preA,preD=entry.st.preD;const double pr0=entry.pr;
        if(entry.st.over){emit.cur=entry.plan;emit(pr0,0,0,preA,preD,0,0,0,0,ra0,ea0,rd0,ed0,entry.st.dA,entry.st.dD,{},{});continue;}
        if(p.atk.name=="relentless assault"||p.dfn.name=="relentless assault"){
            for(int pass=0;pass<2;++pass){
                const bool is_att=pass==0;ck_Side& sp=is_att?p.atk:p.dfn;if(sp.name!="relentless assault"||!sp.has_card)continue;
                int64_t r=is_att?ra0:rd0,e=is_att?ea0:ed0;const std::string key=is_att?"atk_effect_count":"def_effect_count";
                const int64_t k=ch.has_value(key)?std::max<int64_t>(0,std::min({ch.integer(key),int64_t(2),r+2*e})):std::min<int64_t>(2,std::max<int64_t>(0,r+e-2));
                auto x=ck_apply_hits(r,e,k,"preserve_elites");r=x.first;e=x.second;
                sp.m+=k;sp.track_ones=sp.confused||p.atk.confused||p.dfn.confused||(1+sp.m>=sp.t);
                if(is_att){ra0=r;ea0=e;}else{rd0=r;ed0=e;}
            }
            if(ra0+ea0==0||rd0+ed0==0){emit.cur=entry.plan;emit(pr0,0,0,preA,preD,0,0,0,0,ra0,ea0,rd0,ed0,ra0+ea0==0,rd0+ed0==0,{},{});continue;}
        }
        if(!base.atk.sar_lead.empty()||!base.dfn.sar_lead.empty()){
            auto drop=[](const ck_Side& sp,int64_t e_now){int64_t n=0;for(size_t i=static_cast<size_t>(std::max<int64_t>(0,e_now));i<sp.sar_lead.size();++i)n+=sp.sar_lead[i];return n;};
            const int64_t d0=drop(p.atk,ea0),d1=drop(p.dfn,ed0);
            if(d0||d1){p.atk.L5=std::max<int64_t>(0,p.atk.L5-d0);p.dfn.L5=std::max<int64_t>(0,p.dfn.L5-d1);}
        }
        auto n_dice=[](const ck_Side& sp,int64_t units,int64_t capt){int64_t s=std::min<int64_t>(5,units+capt);int64_t eff=s-sp.red;if(sp.has_cap)eff=std::min(eff,sp.cap);return std::max<int64_t>(1,eff);};
        const int64_t nA=n_dice(p.atk,ra0+ea0,p.atk.captains),nD=n_dice(p.dfn,rd0+ed0,p.dfn.captains);
        const int64_t h6A=ck_hit_faces(p.atk.t,p.atk.m),h6D=ck_hit_faces(p.dfn.t,p.dfn.m),h6Ar=ck_hit_faces(p.atk.t,p.atk.rr_m),h6Dr=ck_hit_faces(p.dfn.t,p.dfn.rr_m);
        auto branches=[&](const ck_Side& sp,int64_t n,int64_t h6){
            std::vector<std::pair<std::pair<int64_t,int64_t>,double>> out;
            if(sp.track_ones)return ck_trinom_terms(n,static_cast<double>(h6)/6.0,1.0/6.0);
            for(const auto& term:ck_binom_terms(n,static_cast<double>(h6)/6.0))out.push_back({{term.first,0},term.second});
            return out;
        };
        const auto brA=branches(p.atk,nA,h6A),brD=branches(p.dfn,nD,h6D);
        auto post_roll=[](const ck_Side& sp,int64_t n,int64_t hits,int64_t ones,int64_t t,int64_t m,int64_t& conv,int64_t& k){
            int64_t L=sp.L5;conv=0;
            if(sp.ma_kind&&!sp.ma_blocked){const int64_t ma_miss=n-hits-((1+m>=t)?ones:0);if(ma_miss>0){conv=sp.ma_kind==1?std::min(sp.ma_max,ma_miss):1;L=std::max<int64_t>(0,L-sp.ma_pay);}}
            const int64_t oth=n-hits-ones;const int64_t pool0=sp.confused?oth:oth+ones;const int64_t pool=std::max<int64_t>(0,pool0-conv);
            k=(sp.rr_cancelled||L<=0)?0:std::min<int64_t>(std::min<int64_t>(5,L),pool);
        };
        for(const auto& a:brA){
            const int64_t hA=a.first.first,oA=a.first.second;const double prA=a.second;int64_t convA,kA;post_roll(p.atk,nA,hA,oA,p.atk.t,p.atk.m,convA,kA);
            ck_PlanItem statA;statA.h=hA;statA.o=oA;statA.tuple=p.atk.track_ones;
            for(const auto& d:brD){
                const int64_t hD=d.first.first,oD=d.first.second;const double prD=d.second;int64_t convD,kD;post_roll(p.dfn,nD,hD,oD,p.dfn.t,p.dfn.m,convD,kD);
                ck_PlanItem statD;statD.h=hD;statD.o=oD;statD.tuple=p.dfn.track_ones;
                const int64_t base_atk=hA+convA+(p.dfn.confused?oD:0),base_def=hD+convD+(p.atk.confused?oA:0);
                for(const auto& ra_t:reroll_terms(kA,h6Ar))for(const auto& rd_t:reroll_terms(kD,h6Dr)){
                    const double prob=(((pr0*prA)*prD)*ra_t.second)*rd_t.second;
                    ck_Plan plan=entry.plan;plan.push_back(statA);if(kA>0){ck_PlanItem x;x.h=ra_t.first;plan.push_back(x);}plan.push_back(statD);if(kD>0){ck_PlanItem x;x.h=rd_t.first;plan.push_back(x);}
                    emit.cur=std::move(plan);
                    ck_finish(p,ch,emit,prob,ra0,ea0,rd0,ed0,preA,preD,base_atk+ra_t.first,base_def+rd_t.first,ra_t.first,rd_t.first);
                }
            }
        }
    }
    return dist;
}
// ---- the 9.5 reply around it
std::string ck_nation_repr(int n){return n<0?std::string("None"):std::string("<Nation.")+ck_NATIONS[n]+": '"+ck_NATION_VALUES[n]+"'>";}
// CPython str.__repr__ on UTF-8 text: single quotes unless the text holds ' and no ", backslash escapes for the quote,
// backslash, \t \n \r, \xhh for other C0 / DEL / C1 controls and U+00A0 / U+00AD (not str.isprintable); every other
// code point (the Latin letters of card and figure names) verbatim.
std::string ck_str_repr(const std::string& s){
    const bool single=s.find('\'')==std::string::npos||s.find('"')!=std::string::npos;const char q=single?'\'':'"';
    std::string o(1,q);size_t i=0;
    auto hex=[&](uint32_t c){char buf[8];std::snprintf(buf,sizeof buf,"%cx%02x",0x5c,static_cast<unsigned>(c));o+=buf;};
    while(i<s.size()){
        const auto b=static_cast<unsigned char>(s[i]);
        if(b<0x80){
            if(b==static_cast<unsigned char>(q)||b==0x5c){o+=static_cast<char>(0x5c);o+=static_cast<char>(b);}
            else if(b=='\t'){o+=static_cast<char>(0x5c);o+='t';}else if(b=='\n'){o+=static_cast<char>(0x5c);o+='n';}else if(b=='\r'){o+=static_cast<char>(0x5c);o+='r';}
            else if(b<0x20||b==0x7f)hex(b);else o+=static_cast<char>(b);
            ++i;continue;
        }
        const size_t len=b>=0xf0?4:b>=0xe0?3:2;if(i+len>s.size())throw ck_Error{"truncated UTF-8 in repr"};
        uint32_t c=b&(len==2?0x1f:len==3?0x0f:0x07);for(size_t j=1;j<len;++j)c=(c<<6)|(static_cast<unsigned char>(s[i+j])&0x3f);
        if(c<0xa0||c==0xa0||c==0xad)hex(c);else o.append(s,i,len);
        i+=len;
    }
    o+=q;return o;
}
std::string ck_value_repr(const ck_Value& v){if(v.kind==0)return "None";if(v.kind==1)return v.i?"True":"False";if(v.kind==2)return std::to_string(v.i);return ck_str_repr(v.s);}
std::string ck_army_repr(const ck_ArmySpec& a){
    std::string leaders="(";for(size_t i=0;i<a.leaders.size();++i){const auto& l=a.leaders[i];if(i)leaders+=", ";
        leaders+="LeaderSpec(kind="+ck_str_repr(l.kind)+", name="+ck_str_repr(l.name)+", level="+std::to_string(l.level)+", leadership="+std::to_string(l.leadership)+", nation="+ck_nation_repr(l.nation)+")";}
    leaders+=a.leaders.size()==1?",)":")";
    return "ArmySpec(regulars="+std::to_string(a.regulars)+", elites="+std::to_string(a.elites)+", nation="+ck_nation_repr(a.nation)+", elite_nation="+ck_nation_repr(a.elite_nation)+
           ", leaders="+leaders+", elites2="+std::to_string(a.elites2)+", elite2_nation="+ck_nation_repr(a.elite2_nation)+")";
}
std::string ck_ctx_repr(const ck_Ctx& c){
    std::string ch="(";for(size_t i=0;i<c.choices.size();++i){if(i)ch+=", ";ch+="("+ck_str_repr(c.choices[i].first)+", "+ck_value_repr(c.choices[i].second)+")";}
    ch+=c.choices.size()==1?",)":")";
    auto opt=[&](bool has,const std::string& s){return has?ck_str_repr(s):std::string("None");};
    return "ExchangeContext(attacker="+ck_army_repr(c.atk)+", defender="+ck_army_repr(c.dfn)+", attacker_card="+opt(c.has_atk_card,c.atk_card)+", defender_card="+opt(c.has_def_card,c.def_card)+
           ", is_siege="+(c.is_siege?"True":"False")+", city="+(c.city?"True":"False")+", shine="+(c.shine<0?"None":c.shine?"True":"False")+", choices="+ch+
           ", defer_casualties="+(c.defer?"True":"False")+", attacker_region="+opt(c.has_atk_region,c.atk_region)+", defender_region="+opt(c.has_def_region,c.def_region)+
           ", saruman_in_play="+(c.saruman?"True":"False")+")";
}
std::string ck_plan_repr(const ck_Plan& plan){
    std::string s="(";for(size_t i=0;i<plan.size();++i){if(i)s+=", ";s+=plan[i].tuple?"("+std::to_string(plan[i].h)+", "+std::to_string(plan[i].o)+")":std::to_string(plan[i].h);}
    s+=plan.size()==1?",)":")";return s;
}
bool ck_conf_case(const ck_Ctx& c){for(int i=0;i<2;++i){const bool has=i==0?c.has_atk_card:c.has_def_card;const auto& nm=i==0?c.atk_card:c.def_card;if(has&&!nm.empty()&&ck_has(ck_lower(nm),"confusion"))return true;}return false;}
std::optional<std::string> ck_wctk_side(const Snapshot& t,const ck_Ctx& c){
    std::optional<std::string> side;
    if(c.has_atk_card&&!c.atk_card.empty()&&ck_has(ck_lower(c.atk_card),"we come to kill"))side="attacker";
    if(c.has_def_card&&!c.def_card.empty()&&ck_has(ck_lower(c.def_card),"we come to kill"))side="defender";
    if(side){const auto p0=ck_stage0(t,c);const ck_Side& sp0=*side=="attacker"?p0.atk:p0.dfn;if(!ck_has(sp0.name,"we come to kill"))side.reset();}
    return side;
}

// ---- test / service exports: the 9.5 reply as JSON (masses as IEEE-754 bit patterns)
thread_local std::string ck_last_error,ck_last_json;
std::string ck_json_str(const std::string& s){
    std::string o="\"";for(unsigned char ch:s){if(ch=='"'||ch==0x5c){o+=static_cast<char>(0x5c);o+=static_cast<char>(ch);}else if(ch<0x20){char buf[8];std::snprintf(buf,sizeof buf,"%cu%04x",0x5c,ch);o+=buf;}else o+=static_cast<char>(ch);}
    return o+"\"";
}
std::string ck_reply_json(const Snapshot& t,const Snapshot& ctx_snap,bool want_witness){
    const ck_Ctx ctx=ck_parse(ctx_snap);const ck_Dist dist=ck_exact_distribution(t,ctx,want_witness);
    std::string j="{\"outcomes\":[";
    for(size_t i=0;i<dist.obs.size();++i){
        const auto& o=dist.obs[i];if(i)j+=",";j+="[";for(int k=0;k<12;++k){j+=std::to_string(o.v[k]);j+=",";}
        j+=o.destA?"true,":"false,";j+=o.destD?"true,":"false,";
        j+="[";for(size_t k=0;k<o.fxa.size();++k){if(k)j+=",";j+=ck_json_str(o.fxa[k]);}j+="],[";for(size_t k=0;k<o.fxd.size();++k){if(k)j+=",";j+=ck_json_str(o.fxd[k]);}j+="],";
        j+=o.deferred?"true,":"false,";j+=std::to_string(o.dhd)+","+std::to_string(o.dha)+"]";
    }
    j+="],\"masses\":[";
    for(size_t i=0;i<dist.mass.size();++i){uint64_t bits;std::memcpy(&bits,&dist.mass[i],8);if(i)j+=",";j+="\""+std::to_string(bits)+"\"";}
    j+="],\"witness\":[";
    for(size_t i=0;i<dist.plan.size();++i){if(i)j+=",";j+="[";const auto& pl=dist.plan[i];for(size_t k=0;k<pl.size();++k){if(k)j+=",";j+=pl[k].tuple?"["+std::to_string(pl[k].h)+","+std::to_string(pl[k].o)+"]":std::to_string(pl[k].h);}j+="]";}
    j+="],\"plan_reprs\":[";
    for(size_t i=0;i<dist.plan.size();++i){if(i)j+=",";j+=ck_json_str(ck_plan_repr(dist.plan[i]));}
    j+="],\"conf\":";j+=ck_conf_case(ctx)?"true":"false";
    const auto w=ck_wctk_side(t,ctx);j+=",\"wctk_side\":";j+=w?ck_json_str(*w):"null";
    j+=",\"ctx_repr\":"+ck_json_str(ck_ctx_repr(ctx))+"}";
    return j;
}
extern "C" {
WOTR_API const char* wotr_ck_reply_json(const void* tables,const void* ctx,int64_t want_witness){
    try{ck_last_error.clear();ck_last_json=ck_reply_json(snapshot(tables),snapshot(ctx),want_witness!=0);return ck_last_json.c_str();}
    catch(const ck_Error& e){ck_last_error="ck: "+e.why;}catch(const std::exception& e){ck_last_error=e.what();}catch(...){ck_last_error="unknown combat kernel error";}
    return nullptr;
}
WOTR_API const char* wotr_ck_last_error(){return ck_last_error.c_str();}
}
