// NATIVE CHANCE FEATURES (claude_bugscan_2026-10-21/native_chance, 2026-10-27): selfplay_actor_critic/probabilities.py
// encode_probabilities, line for line, on the native GameState graph of the raw-game driver.
//
// Every function below names the Python function it mirrors.  Arithmetic follows the source's operation order in
// binary64 (Python floats) and the NumPy 2.2.5 float32 rules where the source writes into its float32 row:
//   * a Python float or bool assigned into the row is rounded once to float32;
//   * `row[k] += x` (float32 element, Python float x) is a float32 addition of x rounded to float32 (NEP 50);
//   * the hunt damage distribution is a float64 array (elementwise d += p * pmf), rounded to float32 when stored.
// x ** n follows CPython float_pow (special cases first, then the C runtime pow, which CPython and this DLL share:
// both link the UCRT; the box builds with /fp:strict, so no contraction or reassociation).
// Combat state: probabilities.py works on a detached shallow copy of a CombatState and only rebinds its scalar fields;
// here the copy is a value struct (cf_Combat) read from the graph, and the graph is never written except for the
// two engine calls the source makes on its view: army_participants.freeze_attack (cbb_freeze, into the local attack
// context) and combat.setup_combat (cbs_setup), both on a per-frame scratch copy of the state.
// A source exception (unknown card, an ineligible Mighty Attack payer, an out-of-range Dread and Despair count, a
// pmf index past 5) is a std::exception here: the caller refuses the frame exactly where the Python raises.

struct cf_Schema{std::vector<std::string> names;std::unordered_map<std::string,size_t> index;};
const cf_Schema& cf_schema(){
    static const cf_Schema schema=[]{
        cf_Schema s;
        auto add=[&](const std::string& n){s.index.emplace(n,s.names.size());s.names.push_back(n);};
        for(const char* n:{"combat_available","opponent_card_unknown","own_card_present","enemy_card_present",
                           "own_card_cancelled","enemy_card_cancelled","unresolved_choices","current_roster_condition"})add(n);
        std::vector<std::string> side={"dice_over_5","leadership_over_10","hit_probability","reroll_hit_probability",
            "roll_observed","confusion_self_hits_over_5","bonus_per_success_over_2","shield_expected_cancel",
            "conversion_limit_over_2","conversion_cost_over_5","initiative_over_10","later_effect"};
        for(const char* p:{"roll_hits_","pre_hits_","additional_hits_"})for(int i=0;i<6;++i)side.push_back(p+std::to_string(i));
        for(const char* prefix:{"own","enemy"})for(const auto& n:side)add(std::string(prefix)+"_"+n);
        for(const char* n:{"hunt_available","hunt_mordor","raw_before_mitigation","tile_draw_probability",
                           "tile_reveal_icon_probability","tile_stop_icon_probability","tile_available"})add(n);
        for(int i=0;i<6;++i)add("hunt_hits_"+std::to_string(i));
        for(int i=-2;i<16;++i)add("raw_damage_"+std::to_string(i));
        add("raw_damage_16_plus");
        return s;
    }();
    return schema;
}
size_t cf_width(){return cf_schema().names.size();}
size_t cf_I(const std::string& name){const auto& m=cf_schema().index;auto it=m.find(name);require(it!=m.end(),("unknown chance feature "+name).c_str());return it->second;}
// numpy float32 row helpers
inline void cf_set(float* row,size_t k,double v){row[k]=static_cast<float>(v);}
inline void cf_add(float* row,size_t k,double v){const float x=static_cast<float>(v);row[k]=row[k]+x;}
template<size_t N> void cf_put(float* row,const std::string& name,const std::array<double,N>& values){const size_t start=cf_I(name);for(size_t i=0;i<N;++i)row[start+i]=static_cast<float>(values[i]);}

// ---------------------------------------------------------------- Python arithmetic
bool cf_odd(double x){return std::fmod(std::fabs(x),2.0)==1.0;}
double cf_pow(double iv,double iw){               // CPython float_pow (finite arguments)
    if(iw==0.0)return 1.0;
    if(std::isnan(iv))return iv;
    require(std::isfinite(iv)&&std::isfinite(iw),"chance features: non-finite power");
    if(iv==0.0){require(iw>=0.0,"chance features: 0.0 to a negative power");return cf_odd(iw)?iv:0.0;}
    bool negate=false;
    if(iv<0.0){require(iw==std::floor(iw),"chance features: negative base to a fractional power");iv=-iv;negate=cf_odd(iw);}
    if(iv==1.0)return negate?-1.0:1.0;
    const double ix=std::pow(iv,iw);
    require(std::isfinite(ix),"chance features: power overflow");
    return negate?-ix:ix;
}
int64_t cf_comb(int64_t n,int64_t k){int64_t c=1;for(int64_t i=1;i<=k;++i)c=c*(n-k+i)/i;return c;}
int64_t cf_fact(int64_t n){int64_t f=1;for(int64_t i=2;i<=n;++i)f*=i;return f;}
// combat_dist._binom_terms
std::vector<std::pair<int64_t,double>> cf_binom(int64_t n,double p){
    std::vector<std::pair<int64_t,double>> out;
    for(int64_t k=0;k<=n;++k){
        double pr=static_cast<double>(cf_comb(n,k))*cf_pow(p,static_cast<double>(k));
        pr=pr*cf_pow(1.0-p,static_cast<double>(n-k));
        if(pr>0.0)out.push_back({k,pr});
    }
    return out;
}
// combat_dist._trinom_terms
std::vector<std::pair<std::pair<int64_t,int64_t>,double>> cf_trinom(int64_t n,double ph,double po){
    require(n>=0&&n<=20,"chance features: trinomial size out of range");
    const double poth=(1.0-ph)-po;
    std::vector<std::pair<std::pair<int64_t,int64_t>,double>> out;
    for(int64_t h=0;h<=n;++h)for(int64_t o=0;o<=n-h;++o){
        const int64_t r=n-h-o;
        double pr=static_cast<double>(cf_fact(n)/(cf_fact(h)*cf_fact(o)*cf_fact(r)));   // exact multinomial (int / int)
        pr=pr*cf_pow(ph,static_cast<double>(h));pr=pr*cf_pow(po,static_cast<double>(o));pr=pr*cf_pow(poth,static_cast<double>(r));
        if(pr>0.0)out.push_back({{h,o},pr});
    }
    return out;
}
// combat_dist._hit_faces
int64_t cf_hit_faces(int64_t threshold,int64_t mod){int64_t n=0;for(int64_t r=2;r<=6;++r)if(r==6||r+mod>=threshold)++n;return n;}
// probabilities.dice_pmf
std::array<double,6> cf_dice_pmf(int64_t dice,int64_t hit_faces,int64_t reroll_faces,int64_t leadership,bool confused=false,
                                 const std::pair<int64_t,int64_t>* observed=nullptr,const int64_t* rerolled=nullptr,int64_t converted=0){
    std::array<double,6> pmf{};
    std::vector<std::pair<std::pair<int64_t,int64_t>,double>> outcomes;
    if(observed==nullptr)outcomes=cf_trinom(dice,static_cast<double>(hit_faces)/6.,1/6.);
    else outcomes.push_back({*observed,1.});
    for(const auto& item:outcomes){
        const int64_t hits=item.first.first,ones=item.first.second;const double probability=item.second;
        const int64_t misses=std::max<int64_t>(0,dice-hits-(confused?ones:0)-converted);
        const int64_t count=std::min<int64_t>({5,std::max<int64_t>(0,leadership),misses});
        std::vector<std::pair<int64_t,double>> rolls;
        if(rerolled==nullptr)rolls=cf_binom(count,static_cast<double>(reroll_faces)/6.);else rolls.push_back({*rerolled,1.});
        for(const auto& r:rolls){
            int64_t idx=hits+converted+r.first;if(idx<0)idx+=6;
            require(idx>=0&&idx<6,"chance features: dice_pmf index out of range (IndexError)");
            pmf[static_cast<size_t>(idx)]+=probability*r.second;
        }
    }
    return pmf;
}
// probabilities.plain_pmf
std::array<double,6> cf_plain_pmf(int64_t dice,int64_t faces){
    std::array<double,6> out{};
    for(const auto& t:cf_binom(dice,static_cast<double>(faces)/6.)){require(t.first<6,"chance features: plain_pmf index out of range (IndexError)");out[static_cast<size_t>(t.first)]=t.second;}
    return out;
}

// ---------------------------------------------------------------- graph reads
int64_t cf_int(const Snapshot& s,uint64_t n,const char* what){
    require(n!=missing,what);const auto& v=s.nodes[n];
    require(v.tag==WOTR_INT||v.tag==WOTR_BOOL,what);return v.integer;
}
int64_t cf_attr_int(const Snapshot& s,uint64_t obj,const char* name,bool optional=false,int64_t fallback=0){
    const auto n=obs_attr(s,obj,name,!optional);if(n==missing)return fallback;return cf_int(s,n,name);
}
bool cf_attr_truth(const Snapshot& s,uint64_t obj,const char* name,bool optional=false){
    const auto n=obs_attr(s,obj,name,!optional);return n!=missing&&truth(s,n);
}
uint64_t cf_attr_object(const Snapshot& s,uint64_t obj,const char* name,bool optional=true){
    const auto n=obs_attr(s,obj,name,!optional);return (n==missing||ctx_none(s,n))?missing:n;
}
std::vector<int64_t> cf_int_list(const Snapshot& s,uint64_t list){
    std::vector<int64_t> out;require(list!=missing,"chance features: missing roll list");
    for(auto x:child_edges(s,list))out.push_back(cf_int(s,x,"roll must be integer"));
    return out;
}
std::string cf_name(const Snapshot& s,uint64_t figure){const auto n=obs_attr(s,figure,"name",false);return (n==missing||s.nodes[n].tag!=WOTR_STRING)?std::string():std::string(text(s,n));}
// getattr(figure, 'leadership', fallback) [or 0 when or_zero]
int64_t cf_leadership(const Snapshot& s,uint64_t figure,int64_t fallback,bool or_zero=false){
    const auto n=obs_attr(s,figure,"leadership",false);if(n==missing)return fallback;
    if(ctx_none(s,n)){require(or_zero,"chance features: Companion leadership is None");return 0;}
    return cf_int(s,n,"Companion leadership must be integer");
}

struct cf_Card{bool present=false;std::string name,effect;int64_t value=0;};
cf_Card cf_card_node(const Snapshot& s,uint64_t node){
    cf_Card c;if(node==missing||ctx_none(s,node))return c;
    c.present=true;c.name=cp_lower(text(s,obs_attr(s,node,"combat_name")));c.effect=cp_lower(text(s,obs_attr(s,node,"combat_effect")));
    const auto v=obs_attr(s,node,"combat_value",false);c.value=(v==missing||ctx_none(s,v))?std::numeric_limits<int64_t>::min():cf_int(s,v,"combat_value must be integer");
    return c;
}
// cards.get_card(cid).to_combat_card() from the observation tables' card data (combat_setup.inc cbs_card's source)
cf_Card cf_card_id(const Snapshot& tables,std::string_view cid){
    const auto card=field(tables,field(tables,tables.root,"cards"),std::string(cid).c_str(),false);
    require(card!=missing&&!ctx_none(tables,card),"chance features: unknown card id (get_card raises)");
    cf_Card c;c.present=true;c.name=cp_lower(text(tables,obs_attr(tables,card,"combat_name")));c.effect=cp_lower(text(tables,obs_attr(tables,card,"combat_effect")));
    const auto v=obs_attr(tables,card,"combat_value",false);c.value=(v==missing||ctx_none(tables,v))?std::numeric_limits<int64_t>::min():cf_int(tables,v,"combat_value must be integer");
    return c;
}
int64_t cf_card_value(const cf_Card& c){require(c.value!=std::numeric_limits<int64_t>::min(),"chance features: combat card without a combat value");return c.value;}

// A choice value as probabilities.py reads it (the choices dict: count / target / casualty policy)
struct cf_Choice{bool none=true;bool is_int=false;int64_t i=0;bool is_str=false;std::string str;};
cf_Choice cf_choice(const Snapshot& s,uint64_t n){
    cf_Choice c;if(n==missing||ctx_none(s,n))return c;c.none=false;const auto& v=s.nodes[n];
    if(v.tag==WOTR_INT||v.tag==WOTR_BOOL){c.is_int=true;c.i=v.integer;}
    else if(v.tag==WOTR_STRING){c.is_str=true;c.str=std::string(text(s,n));}
    return c;
}
using cf_Choices=std::map<std::string,cf_Choice>;
cf_Choice cf_get(const cf_Choices& ch,const std::string& key){auto it=ch.find(key);return it==ch.end()?cf_Choice{}:it->second;}
// combat._pick_named_companion
uint64_t cf_pick_named(const Snapshot& s,const std::vector<uint64_t>& companions,const cf_Choice& name){
    if(name.none)return missing;
    if(name.is_int&&name.i==0)return missing;                      // not name
    if(name.is_str&&(name.str.empty()||name.str=="leader"))return missing;
    if(!name.is_str)return missing;                               // getattr(c, 'name', '') == <non-string>
    for(auto c:companions)if(cf_name(s,c)==name.str)return c;
    return missing;
}

// ---------------------------------------------------------------- the detached CombatState
struct cf_Combat{
    const Snapshot* g=nullptr;
    int attacker_side=0;
    cf_Card card[2];bool cancelled[2]={false,false};
    int64_t leadership[2]={0,0},dice_mod[2]={0,0},reroll_mod[2]={0,0},dice_reduction[2]={0,0};
    bool has_cap[2]={false,false};int64_t cap[2]={0,0};
    bool reroll_cancelled[2]={false,false},confusion[2]={false,false},foul_stench[2]={false,false};
    int64_t nazgul=0,shine_cancelled_leadership=0,siege_round=0;
    std::vector<uint64_t> companions;
    uint64_t shine=missing,shined=missing,wop=missing;bool wop_is_attacker=false;
    bool is_siege=false,city_defense=false;
    uint64_t army[2]={missing,missing};
    uint64_t combat_rolls[2]={missing,missing},rerolls[2]={missing,missing},pre_rolls[2]={missing,missing};
    int64_t resume_auto[2]={0,0};
    bool active(int r)const{return card[r].present&&!cancelled[r];}
};
cf_Combat cf_read_combat(const Snapshot& s,uint64_t combat){
    cf_Combat c;c.g=&s;
    c.attacker_side=ctx_side(s,obs_attr(s,combat,"attacker_side"));require(c.attacker_side==0||c.attacker_side==1,"combat attacker side");
    for(int r=0;r<2;++r){
        const std::string role=cp_role(r),sh=cp_short(r);
        c.card[r]=cf_card_node(s,obs_attr(s,combat,(role+"_card").c_str()));
        c.cancelled[r]=cf_attr_truth(s,combat,(role+"_card_cancelled").c_str());
        c.leadership[r]=cf_attr_int(s,combat,(role+"_leadership").c_str());
        c.dice_mod[r]=cf_attr_int(s,combat,(role+"_dice_mod").c_str());
        c.reroll_mod[r]=cf_attr_int(s,combat,(role+"_reroll_mod").c_str());
        c.dice_reduction[r]=cf_attr_int(s,combat,(role+"_dice_reduction").c_str(),true,0);
        const auto cap=obs_attr(s,combat,(role+"_dice_cap").c_str());c.has_cap[r]=!ctx_none(s,cap);if(c.has_cap[r])c.cap[r]=cf_int(s,cap,"dice cap must be integer");
        c.reroll_cancelled[r]=cf_attr_truth(s,combat,(role+"_reroll_cancelled").c_str());
        c.confusion[r]=cf_attr_truth(s,combat,(role+"_confusion").c_str());
        c.foul_stench[r]=cf_attr_truth(s,combat,("_foul_stench_cancel_"+sh).c_str(),true);
        c.army[r]=obs_attr(s,combat,(role+"_army").c_str());
        c.combat_rolls[r]=obs_attr(s,combat,(role+"_combat_rolls").c_str());
        c.rerolls[r]=obs_attr(s,combat,(role+"_rerolls").c_str());
        c.pre_rolls[r]=obs_attr(s,combat,(role+"_pre_combat_rolls").c_str());
        c.resume_auto[r]=cf_attr_int(s,combat,("_resume_"+sh+"_auto").c_str(),true,0);
    }
    c.nazgul=cf_attr_int(s,combat,"nazgul_in_battle");
    c.companions=child_edges(s,obs_attr(s,combat,"companions_in_battle"));
    c.shine=cf_attr_object(s,combat,"_shine_companion");c.shined=cf_attr_object(s,combat,"_shined_companion");
    c.wop=cf_attr_object(s,combat,"_wop_cancelled");c.wop_is_attacker=cf_attr_truth(s,combat,"_wop_cancelled_is_attacker",true);
    c.shine_cancelled_leadership=cf_attr_int(s,combat,"_shine_cancelled_leadership",true,0);
    c.is_siege=cf_attr_truth(s,combat,"is_siege");c.city_defense=cf_attr_truth(s,combat,"city_defense");
    c.siege_round=cf_attr_int(s,combat,"siege_round");
    return c;
}
// combat.calc_combat_strength / count_elite_units over the combat's own graph
int64_t cf_strength(const cf_Combat& c,int r,uint64_t cancelled_captain){return cp_strength(*c.g,c.army[r],cancelled_captain);}
int64_t cf_elites(const cf_Combat& c,int r){int64_t n=0;for(auto u:child_edges(*c.g,obs_attr(*c.g,c.army[r],"units")))n+=enum_is(*c.g,obs_attr(*c.g,u,"kind"),"game_state.UnitKind","ELITE");return n;}

// combat.apply_cancel_effects
void cf_apply_cancel_effects(cf_Combat& c){
    const Snapshot& s=*c.g;std::set<std::string> forfeited;if(c.shined!=missing)forfeited.insert(cf_name(s,c.shined));
    auto resolve=[&](int r){
        if(!c.active(r))return;const std::string& effect=c.card[r].effect;
        if(!cp_has(effect,"cancel the effects of the combat card"))return;
        const bool is_forfeit=cp_has(effect,"forfeit")&&cp_has(effect,"leadership");
        if(is_forfeit){
            if(!c.card[1-r].present)return;
            for(auto comp:c.companions)if(forfeited.count(cf_name(s,comp)))return;
        }
        if(c.card[1-r].present)c.cancelled[1-r]=true;
        if(is_forfeit){
            int64_t sum=0;
            for(auto comp:c.companions){if(comp==c.shined||comp==c.wop)continue;sum+=cf_leadership(s,comp,1);forfeited.insert(cf_name(s,comp));}
            c.leadership[r]=std::max<int64_t>(0,c.leadership[r]-sum);
        }
    };
    auto init=[&](int r){return c.active(r)?cf_card_value(c.card[r]):int64_t(99);};
    if(init(1)<=init(0)){resolve(1);resolve(0);}else{resolve(0);resolve(1);}
}
// combat.apply_words_of_power
void cf_apply_words_of_power(cf_Combat& c,const cf_Choices& ch){
    const Snapshot& s=*c.g;
    for(int r=0;r<2;++r){
        if(!c.active(r))continue;const std::string& effect=c.card[r].effect;
        if(!(cp_has(effect,"companion")&&cp_has(effect,"leadership")&&cp_has(effect,"cancelled")))continue;
        if(c.nazgul<1)continue;
        if(c.companions.empty())continue;
        uint64_t target=cf_pick_named(s,c.companions,cf_get(ch,cp_short(r)+"_effect_target"));
        if(target==missing){int64_t best=0;bool first=true;for(auto comp:c.companions){const auto l=cf_leadership(s,comp,0,true);if(first||l>best){best=l;target=comp;first=false;}}}
        const int fp=1-r;        // fp_is_attacker = not is_atk  ->  FP role index
        if(target==c.shine){const int shadow=c.attacker_side==1?0:1;c.leadership[shadow]+=c.shine_cancelled_leadership;}
        if(target!=c.shined){const auto l=cf_leadership(s,target,0,true);c.leadership[fp]=std::max<int64_t>(0,c.leadership[fp]-l);}
        c.wop=target;c.wop_is_attacker=(fp==0);
    }
}
// combat.apply_dice_modifiers
void cf_apply_dice_modifiers(cf_Combat& c){
    auto apply=[&](const std::string& effect,int r){
        const int e=1-r;
        if(cp_has(effect,"both armies add 2")){for(int k=0;k<2;++k){c.dice_mod[k]+=2;c.reroll_mod[k]+=2;}}
        else if(cp_has(effect,"both armies add 1")){for(int k=0;k<2;++k){c.dice_mod[k]+=1;c.reroll_mod[k]+=1;}}
        else if(cp_has(effect,"subtract 1")&&cp_has(effect,"nazg")&&cp_has(effect,"leader re-roll")){}
        else if(cp_has(effect,"subtract 2")&&cp_has(effect,"nazg")&&cp_has(effect,"add 1")){
            if(c.shine==missing){c.leadership[r]=std::max<int64_t>(0,c.leadership[r]-2);c.dice_mod[r]+=1;}
        }
        else if(cp_has(effect,"inflict")&&cp_has(effect,"up to two hits")){}
        else{
            // re.search(r"add ([12]) to all dice on (?:your|their) ([^.]*)", effect): leftmost match
            size_t best=std::string::npos,start=0;int amount=0;
            for(int n=1;n<=2;++n)for(const char* owner:{"your","their"}){
                const std::string prefix=std::string("add ")+std::to_string(n)+" to all dice on "+owner+" ";
                const auto p=effect.find(prefix);if(p!=std::string::npos&&(best==std::string::npos||p<best)){best=p;start=p+prefix.size();amount=n;}
            }
            if(amount){
                const auto stop=effect.find('.',start);const std::string target=effect.substr(start,stop==std::string::npos?std::string::npos:stop-start);
                if(cp_has(target,"combat"))c.dice_mod[r]+=amount;
                if(cp_has(target,"re-roll")){
                    if(cp_has(effect,"did not play a card")){if(!c.card[e].present)c.reroll_mod[r]+=amount;}
                    else c.reroll_mod[r]+=amount;
                }
            }
        }
        if(cp_has(effect,"subtract 1 from all dice")&&cp_has(effect,"shadow"))c.dice_mod[e]+=-1;
    };
    if(c.active(0))apply(c.card[0].effect,0);
    if(c.active(1))apply(c.card[1].effect,1);
}
// combat.apply_dice_count_modifiers
void cf_apply_dice_count_modifiers(cf_Combat& c){
    for(int r=0;r<2;++r){
        if(!c.active(r))continue;const std::string& effect=c.card[r].effect;const int e=1-r;
        if(cp_has(effect,"maximum of two dice")){c.has_cap[e]=true;c.cap[e]=2;}
        else if(cp_has(effect,"maximum of three dice")){c.has_cap[e]=true;c.cap[e]=3;}
        if(cp_has(effect,"rolls one die less")&&cp_has(effect,"companion"))c.dice_reduction[e]+=static_cast<int64_t>(c.companions.size());
    }
}
// combat.apply_dread_and_despair
void cf_apply_dread_and_despair(cf_Combat& c,const cf_Choices& ch){
    for(int r=0;r<2;++r){
        if(!c.active(r))continue;const std::string& effect=c.card[r].effect;
        if(!(cp_has(effect,"forfeit")&&cp_has(effect,"nazg")&&cp_has(effect,"one die less")))continue;
        if(c.shine!=missing)continue;
        const int64_t available=c.nazgul;if(available<=0)continue;
        const int fp=1-r;const int64_t fp_strength=cf_strength(c,fp,missing);
        const auto fc=cf_get(ch,cp_short(r)+"_effect_count");int64_t forfeit;
        if(!fc.none){require(fc.is_int,"chance features: Dread and Despair count must be integer");forfeit=fc.i;
            require(1<=forfeit&&forfeit<=available,"Dread and Despair must forfeit between 1 and the available Nazgul leadership");}
        else{const int64_t desired=std::max<int64_t>(0,fp_strength-1);forfeit=std::min<int64_t>(std::max<int64_t>(1,desired),std::max<int64_t>(1,available-1));}
        if(forfeit>0){c.leadership[r]-=forfeit;c.has_cap[fp]=true;c.cap[fp]=std::max<int64_t>(1,fp_strength-forfeit);c.nazgul-=forfeit;}
    }
}
// combat.apply_confusion_effect
void cf_apply_confusion(cf_Combat& c){
    auto conf=[&](int r){const std::string& e=c.card[r].effect;return c.active(r)&&cp_has(e,"result of")&&cp_has(e,"1")&&cp_has(e,"shadow army");};
    if(conf(1))c.confusion[0]=true;
    if(conf(0))c.confusion[1]=true;
}
// combat.apply_reroll_cancellation
void cf_apply_reroll_cancellation(cf_Combat& c){
    const int64_t nazgul=c.shine!=missing?0:c.nazgul;
    for(int r=0;r<2;++r){
        if(!c.active(r)||!cp_has(c.card[r].effect,"leader re-roll is cancelled"))continue;
        const int e=1-r;
        if(nazgul>=c.leadership[e]){c.reroll_cancelled[e]=true;c.foul_stench[e]=true;}
    }
}
// combat.apply_they_are_terrible
void cf_apply_they_are_terrible(cf_Combat& c){
    for(int r=0;r<2;++r){
        if(!c.active(r))continue;const std::string& effect=c.card[r].effect;
        if(!(cp_has(effect,"subtract 1")&&cp_has(effect,"nazg")&&cp_has(effect,"leader re-roll")))continue;
        if(c.shine!=missing)continue;
        if(c.nazgul>=1){c.nazgul-=1;c.leadership[r]-=1;c.reroll_mod[r]+=1;}
    }
}
// combat_dist._mk_side_params (the fields the chance features read)
struct cf_SideParams{bool ma=false;bool ma_blocked=false;int64_t ma_pay=0,ma_max=0;int pre_kind=0;};   // pre 1 ss, 2 charge, 3 db
cf_SideParams cf_side_params(const cf_Combat& c,int r,const cf_Choices& ch){
    const Snapshot& s=*c.g;cf_SideParams p;
    const bool has_card=c.active(r);const std::string eff=has_card?c.card[r].effect:std::string();
    if(has_card&&cp_has(eff,"before the combat roll")&&cp_has(eff,"additional attack")){
        if(cp_has(eff,"leadership"))p.pre_kind=1;else if(cp_has(eff,"elite units"))p.pre_kind=2;else if(cp_has(eff,"three combat dice"))p.pre_kind=3;
    }
    if(has_card&&cp_has(eff,"forfeit")&&cp_has(eff,"leadership")&&cp_has(eff,"change")&&cp_has(eff,"missed")){
        p.ma_blocked=c.foul_stench[r];
        std::vector<uint64_t> avail;for(auto x:c.companions)if(x!=c.shined&&x!=c.wop)avail.push_back(x);
        if(cp_has(eff,"aragorn")||cp_has(eff,"strider")){
            uint64_t hero=missing;for(auto x:avail){const auto n=cp_lower(cf_name(s,x));if(cp_has(n,"aragorn")||cp_has(n,"strider")){hero=x;break;}}
            if(hero!=missing){const bool crowned=cf_attr_truth(s,hero,"crowned",true)||cf_leadership(s,hero,1)>=2;p.ma=true;p.ma_max=crowned?2:1;p.ma_pay=cf_leadership(s,hero,1);}
        }else if(!avail.empty()){
            const auto named=cf_get(ch,cp_short(r)+"_effect_target");uint64_t target=missing;
            if(!named.none)target=cf_pick_named(s,avail,named);
            else{int64_t best=0;bool first=true;for(auto x:avail){const auto l=cf_leadership(s,x,1);if(first||l<best){best=l;target=x;first=false;}}}
            require(target!=missing,"Mighty Attack payer is not currently eligible");
            p.ma=true;p.ma_max=1;p.ma_pay=cf_leadership(s,target,1);
        }
    }
    return p;
}

// ---------------------------------------------------------------- cursor view (combat_round_resume.RoundCursor)
struct cf_Cursor{bool present=false;int64_t stage=-1;cf_Choices choices;bool pre_remaining=false;bool has_ra=false;int64_t ra=0;
    std::vector<std::pair<std::string,std::string>> visible;uint64_t combat=missing;
    bool sees(const char* kind,const std::string& role)const{for(const auto& v:visible)if(v.first==kind&&v.second==role)return true;return false;}};
cf_Cursor cf_read_cursor(const Snapshot& s,uint64_t pc){
    cf_Cursor c;if(pc==missing)return c;
    const auto resume=field(s,pc,"_round_resume",false);if(resume==missing||ctx_none(s,resume)||!truth(s,resume))return c;
    const auto cursor=field(s,resume,"cursor",false);if(cursor==missing||ctx_none(s,cursor))return c;
    c.present=true;c.stage=cf_attr_int(s,cursor,"stage");
    const auto choices=obs_attr(s,cursor,"choices");const auto entries=child_edges(s,choices);
    for(size_t i=0;i+1<entries.size();i+=2)c.choices[std::string(text(s,entries[i]))]=cf_choice(s,entries[i+1]);
    const auto pre=obs_attr(s,cursor,"pre_remaining");c.pre_remaining=!ctx_none(s,pre)&&truth(s,pre);
    const auto ra=obs_attr(s,cursor,"ra_remaining");c.has_ra=!ctx_none(s,ra);if(c.has_ra)c.ra=cf_int(s,ra,"ra_remaining must be integer");
    for(auto e:child_edges(s,obs_attr(s,cursor,"observed_roll_sides"))){
        const auto parts=child_edges(s,e);if(parts.size()!=2)continue;
        c.visible.push_back({std::string(text(s,parts[0])),std::string(text(s,parts[1]))});
    }
    c.combat=obs_attr(s,cursor,"combat");return c;
}

// ---------------------------------------------------------------- probabilities.py combat rows
struct cf_Action{std::string kind;const Snapshot* g=nullptr;uint64_t params=missing;};
uint64_t cf_param(const cf_Action& a,const char* key){return field(*a.g,a.params,key,false);}
struct cf_Parts{cf_Combat c;int own=0,enemy=1;bool unknown=false,unresolved=false;cf_Choices choices;};
// probabilities._cards_and_parameters (cursor may be absent: present=false)
cf_Parts cf_cards_and_parameters(const cf_Combat& base,const Snapshot& state,const Snapshot& tables,uint64_t pc,const cf_Action& action,int side,const cf_Cursor& cursor){
    cf_Parts out;out.c=base;cf_Combat& c=out.c;
    out.own=c.attacker_side==side?0:1;out.enemy=1-out.own;const std::string sh=cp_short(out.own);
    const bool choosing=action.kind=="combat_card_play"||action.kind=="combat_card_none";
    out.unknown=choosing;
    if(cursor.present)out.choices=cursor.choices;
    else for(const char* k:{"atk_effect_count","def_effect_count","atk_effect_target","def_effect_target"}){
        const auto v=field(state,pc,k,false);if(v!=missing&&!ctx_none(state,v))out.choices[k]=cf_choice(state,v);
    }
    if(choosing){
        c.card[0]=c.card[1]=cf_Card{};
        if(action.kind=="combat_card_play")c.card[out.own]=cf_card_id(tables,text(*action.g,cf_param(action,"card_id")));
        out.choices.clear();
    }else if(!cursor.present){
        for(int r=0;r<2;++r){const auto cid=field(state,pc,(cp_short(r)+"_combat_card").c_str(),false);
            c.card[r]=(cid!=missing&&!ctx_none(state,cid)&&truth(state,cid))?cf_card_id(tables,text(state,cid)):cf_Card{};}
    }
    if(action.kind=="combat_effect_choice")for(const char* f:{"count","target"}){const auto v=cf_param(action,f);if(v!=missing)out.choices[sh+"_effect_"+f]=cf_choice(*action.g,v);}
    const int64_t stage=cursor.present?cursor.stage:-1;
    out.unresolved=cursor.present&&cursor.pre_remaining;
    auto active_name=[&](int r){return c.active(r)?c.card[r].name:std::string();};
    if(stage<0){
        cf_apply_cancel_effects(c);
        bool all=true;
        for(int r=0;r<2;++r)if(active_name(r)=="words of power"&&!c.companions.empty())all&=!cf_get(out.choices,cp_short(r)+"_effect_target").none;
        if(all)cf_apply_words_of_power(c,out.choices);else out.unresolved=true;
    }
    if(stage<=2){cf_apply_dice_modifiers(c);cf_apply_dice_count_modifiers(c);}
    for(int r=0;r<2;++r){
        const std::string name=active_name(r),tag=cp_short(r);
        if(name=="dread and despair"&&stage<=3){
            if(cf_get(out.choices,tag+"_effect_count").none)out.unresolved=true;else cf_apply_dread_and_despair(c,out.choices);
        }
        if(name=="relentless assault"&&stage<=4){
            auto count=cf_get(out.choices,tag+"_effect_count");
            if(cursor.present&&cursor.has_ra){count=cf_Choice{};count.none=false;count.is_int=true;count.i=cursor.ra;}
            if(!count.none){require(count.is_int,"chance features: Relentless Assault count must be integer");c.dice_mod[r]+=count.i;}
            out.unresolved=true;
        }
    }
    if(stage<=5)cf_apply_confusion(c);
    if(stage<=6){cf_apply_reroll_cancellation(c);cf_apply_they_are_terrible(c);}
    return out;
}
// probabilities._side_features: returns (effect, pmf, pre-roll pending)
struct cf_SideResult{std::string effect;std::array<double,6> pmf{};bool pre_pending=false;};
cf_SideResult cf_side_features(float* row,const std::string& prefix,const cf_Combat& c,int r,const cf_Cursor& cursor,const cf_Choices& choices){
    const Snapshot& s=*c.g;const bool attacker=r==0;const std::string role=cp_role(r),sh=cp_short(r);
    const bool active=c.active(r);const std::string effect=active?c.card[r].effect:std::string();
    const uint64_t cancelled=(c.wop_is_attacker==attacker)?c.wop:missing;
    int64_t dice=cf_strength(c,r,cancelled);
    dice-=c.dice_reduction[r];
    dice=std::max<int64_t>(1,c.has_cap[r]?std::min(dice,c.cap[r]):dice);
    const int64_t threshold=(attacker&&(c.is_siege||(c.city_defense&&c.siege_round==1)))?6:5;
    const int64_t modifier=c.dice_mod[r];
    const int64_t faces=cf_hit_faces(threshold,modifier),rr_faces=cf_hit_faces(threshold,c.reroll_mod[r]);
    const int64_t leadership=c.reroll_cancelled[r]?0:c.leadership[r];
    bool has_observed=false,has_rerolled=false;std::pair<int64_t,int64_t> observed{0,0};int64_t rerolled=0,converted=0;
    double ones=static_cast<double>(dice)/6.;bool ones_int=false;int64_t ones_count=0;
    if(cursor.present&&cursor.sees("combat",role)){
        const auto rolls=cf_int_list(s,c.combat_rolls[r]);dice=static_cast<int64_t>(rolls.size());
        ones_count=std::count(rolls.begin(),rolls.end(),int64_t(1));ones_int=true;
        int64_t hits=0;for(auto x:rolls)hits+=(x!=1&&(x==6||x+modifier>=threshold));
        observed={hits,ones_count};has_observed=true;
        const bool conversion_done=cursor.stage>(attacker?7:8)||cursor.sees("reroll",role);
        if(conversion_done)converted=c.resume_auto[r];
    }
    if(cursor.present&&cursor.sees("reroll",role)){
        has_rerolled=true;for(auto x:cf_int_list(s,c.rerolls[r]))rerolled+=(x!=1&&(x==6||x+c.reroll_mod[r]>=threshold));
    }
    const bool confused=c.confusion[r];
    const auto pmf=cf_dice_pmf(dice,faces,rr_faces,leadership,confused,has_observed?&observed:nullptr,has_rerolled?&rerolled:nullptr,converted);
    // values dict: dice/5, leadership/10, faces/6, rr_faces/6, float(observed), ones/5 if confused else 0, value/10 if card
    cf_set(row,cf_I(prefix+"_dice_over_5"),static_cast<double>(dice)/5);
    cf_set(row,cf_I(prefix+"_leadership_over_10"),static_cast<double>(leadership)/10);
    cf_set(row,cf_I(prefix+"_hit_probability"),static_cast<double>(faces)/6);
    cf_set(row,cf_I(prefix+"_reroll_hit_probability"),static_cast<double>(rr_faces)/6);
    cf_set(row,cf_I(prefix+"_roll_observed"),has_observed?1.0:0.0);
    cf_set(row,cf_I(prefix+"_confusion_self_hits_over_5"),confused?(ones_int?static_cast<double>(ones_count)/5:ones/5):0.0);
    cf_set(row,cf_I(prefix+"_initiative_over_10"),active?static_cast<double>(cf_card_value(c.card[r]))/10:0.0);
    const auto params=cf_side_params(c,r,choices);
    cf_set(row,cf_I(prefix+"_conversion_limit_over_2"),!params.ma_blocked?static_cast<double>(params.ma_max)/2:0.0);
    cf_set(row,cf_I(prefix+"_conversion_cost_over_5"),!params.ma_blocked?static_cast<double>(params.ma_pay)/5:0.0);
    if(cp_has(effect,"score one additional hit")&&!cp_has(effect,"more total hits"))cf_set(row,cf_I(prefix+"_bonus_per_success_over_2"),.5);
    if(cp_has(effect,"score two additional hits"))cf_set(row,cf_I(prefix+"_bonus_per_success_over_2"),1.);
    const bool later=cp_has(effect,"after")||params.ma||cp_has(effect,"retreat");
    cf_set(row,cf_I(prefix+"_later_effect"),later?1.0:0.0);
    int64_t pre_dice=0,pre_threshold=threshold;
    if(params.pre_kind==1)pre_dice=std::min<int64_t>(5,c.leadership[r]);
    else if(params.pre_kind==2)pre_dice=std::min<int64_t>(5,cf_elites(c,r));
    else if(params.pre_kind==3){pre_dice=3;pre_threshold=4;}
    auto pre=cf_plain_pmf(pre_dice,cf_hit_faces(pre_threshold,0));
    const bool pre_seen=cursor.present&&cursor.sees("pre_combat",role);
    if(pre_seen){
        int64_t hits=0;for(auto x:cf_int_list(s,c.pre_rolls[r]))hits+=(x!=1&&(x==6||x>=pre_threshold));
        for(int i=0;i<6;++i)pre[static_cast<size_t>(i)]=(i==hits)?1.0:0.0;
    }
    cf_put(row,prefix+"_roll_hits_0",pmf);
    cf_put(row,prefix+"_pre_hits_0",pre);
    return cf_SideResult{effect,pmf,pre_dice>0&&!pre_seen};
}
// probabilities.combat_features
void cf_combat_features(float* row,const cf_Combat& base,const Snapshot& state,const Snapshot& tables,uint64_t pc,const cf_Action& action,int side,const cf_Cursor& cursor){
    std::fill(row,row+cf_width(),0.0f);
    cf_set(row,cf_I("combat_available"),1);
    auto parts=cf_cards_and_parameters(base,state,tables,pc,action,side,cursor);const cf_Combat& c=parts.c;
    cf_set(row,cf_I("opponent_card_unknown"),parts.unknown?1:0);
    cf_set(row,cf_I("unresolved_choices"),parts.unresolved?1:0);
    cf_SideResult results[2];const char* prefixes[2]={"own","enemy"};const int roles[2]={parts.own,parts.enemy};
    for(int i=0;i<2;++i){
        const int r=roles[i];
        cf_set(row,cf_I(std::string(prefixes[i])+"_card_present"),c.card[r].present?1:0);
        cf_set(row,cf_I(std::string(prefixes[i])+"_card_cancelled"),c.cancelled[r]?1:0);
        results[i]=cf_side_features(row,prefixes[i],c,r,cursor,parts.choices);
    }
    cf_set(row,cf_I("current_roster_condition"),(parts.unresolved||results[0].pre_pending||results[1].pre_pending)?1.0:0.0);
    for(int i=0;i<2;++i){
        if(cp_has(results[i].effect,"two or more hits")&&cp_has(results[i].effect,"cancel one hit")){
            double sum=0;for(size_t k=2;k<6;++k)sum=sum+results[1-i].pmf[k];
            cf_set(row,cf_I(std::string(prefixes[i])+"_shield_expected_cancel"),sum);
        }
    }
}

// ---------------------------------------------------------------- probabilities.hunt_features
void cf_hunt_features(float* row,const Snapshot& s,const Snapshot& tables){
    std::fill(row,row+cf_width(),0.0f);
    const auto fs=obs_attr(s,s.root,"fellowship");const bool mordor=truth(s,obs_attr(s,fs,"mordor_entered"));
    cf_set(row,cf_I("hunt_available"),1);cf_set(row,cf_I("raw_before_mitigation"),1);cf_set(row,cf_I("hunt_mordor"),mordor?1:0);
    std::array<double,6> hits{};
    if(mordor)hits={1.,0.,0.,0.,0.,0.};
    else{
        const auto p=hr_parameters(s,tables);
        int64_t modifier=p.fp;
        const auto bonus=obs_attr(s,s.root,"_crebain_hunt_bonus",false);
        if(bonus!=missing)modifier+=cf_int(s,bonus,"Crebain Hunt modifier must be integer");
        int64_t faces=0;for(int64_t r=1;r<=6;++r)faces+=(r>1&&r+modifier>=6);
        hits=cf_dice_pmf(p.dice,faces,faces,static_cast<int64_t>(p.nazgul)+static_cast<int64_t>(p.army)+static_cast<int64_t>(p.stronghold));
    }
    cf_put(row,"hunt_hits_0",hits);
    // pool = state.hunt_pool or [t for t in state.hunt_discard if not t.is_special]
    std::vector<uint64_t> pool=child_edges(s,obs_attr(s,s.root,"hunt_pool"));
    if(pool.empty())for(auto t:child_edges(s,obs_attr(s,s.root,"hunt_discard")))if(!truth(s,obs_attr(s,t,"is_special")))pool.push_back(t);
    std::map<std::array<int64_t,5>,int64_t> counter;
    for(auto t:pool){
        require(!ctx_none(s,obs_attr(s,t,"damage")),"chance features: hunt tile without damage in the bag");
        std::array<int64_t,5> key={cf_int(s,obs_attr(s,t,"damage"),"tile damage"),cf_int(s,obs_attr(s,t,"is_eye"),"tile eye"),
            cf_int(s,obs_attr(s,t,"reveal"),"tile reveal"),cf_int(s,obs_attr(s,t,"stops_movement"),"tile stop"),cf_int(s,obs_attr(s,t,"is_dice_roll"),"tile d6")};
        counter[key]+=1;
    }
    cf_set(row,cf_I("tile_available"),pool.empty()?0:1);
    const int64_t eyes=static_cast<int64_t>(child_edges(s,obs_attr(s,s.root,"hunt_box")).size());
    std::array<double,19> distribution{};
    if(!mordor)distribution[2]=hits[0];
    const size_t reveal_k=cf_I("tile_reveal_icon_probability"),stop_k=cf_I("tile_stop_icon_probability");
    int64_t total=0;for(const auto& kv:counter)total+=kv.second;
    auto components=[&](int64_t successes,std::array<double,19>& damage,double& reveal,double& stop){   // probabilities._tile_components
        damage={};reveal=stop=0.;
        if(!total){damage[2]=1.;return;}
        for(const auto& kv:counter){
            const auto& k=kv.first;const double weight=static_cast<double>(kv.second)/static_cast<double>(total);
            std::vector<std::pair<int64_t,double>> outcomes;
            if(k[1])outcomes.push_back({mordor?eyes:successes,1.});
            else if(k[4]&&mordor)for(int64_t face=1;face<=6;++face)outcomes.push_back({face,1/6.});
            else outcomes.push_back({k[0],1.});
            for(const auto& o:outcomes){int64_t idx=std::min<int64_t>(18,o.first+2);if(idx<0)idx+=19;require(idx>=0&&idx<19,"chance features: hunt damage index out of range");damage[static_cast<size_t>(idx)]+=weight*o.second;}
            reveal+=weight*static_cast<double>(k[2]);stop+=weight*static_cast<double>(k[3]);
        }
    };
    std::array<double,19> pmf{};double reveal=0,stop=0;
    if(mordor){components(0,pmf,reveal,stop);for(size_t j=0;j<19;++j)distribution[j]+=1.*pmf[j];cf_add(row,reveal_k,1.*reveal);cf_add(row,stop_k,1.*stop);}
    else for(int64_t sx=1;sx<=5;++sx){
        const double probability=hits[static_cast<size_t>(sx)];
        components(sx,pmf,reveal,stop);
        for(size_t j=0;j<19;++j)distribution[j]+=probability*pmf[j];
        cf_add(row,reveal_k,probability*reveal);cf_add(row,stop_k,probability*stop);
    }
    cf_set(row,cf_I("tile_draw_probability"),pool.empty()?0.0:(mordor?1.:1-hits[0]));
    cf_put(row,"raw_damage_-2",distribution);
}

// ---------------------------------------------------------------- probabilities.encode_probabilities
// state: the frame's GameState graph; menu: the frame's (die, action_type, params) rows; out: rows x cf_width() float32.
void cf_encode(const Snapshot& state,const Snapshot& tables,const Snapshot& menu,int side,float* out,size_t rows_expected){
    const size_t W=cf_width();
    const auto rows=child_edges(menu,menu.root);require(rows.size()==rows_expected,"chance features: row count mismatch");
    std::fill(out,out+rows.size()*W,0.0f);
    const auto pc_raw=obs_attr(state,state.root,"pending_combat",false);
    const bool has_pc=pc_raw!=missing&&!ctx_none(state,pc_raw)&&truth(state,pc_raw);   // pc = ... or {}
    const uint64_t pc=has_pc?pc_raw:missing;
    const cf_Cursor cursor=has_pc?cf_read_cursor(state,pc):cf_Cursor{};
    std::vector<float> hunt;bool hunt_ready=false;
    std::unique_ptr<Snapshot> work;                         // scratch copy for setup_combat / freeze_attack
    auto scratch=[&]()->Snapshot&{if(!work)work=std::make_unique<Snapshot>(state);return *work;};
    bool base_ready=false;cf_Combat base;
    std::vector<std::pair<std::tuple<std::string,std::string,bool>,std::vector<float>>> attacks;
    for(size_t i=0;i<rows.size();++i){
        float* row=out+i*W;const auto f=child_edges(menu,rows[i]);require(f.size()==3,"menu row must be (die, kind, params)");
        cf_Action a;a.kind=std::string(text(menu,f[1]));a.g=&menu;a.params=f[2];
        try{
        if(a.kind=="fellowship_move"||a.kind=="mordor_move"){
            if(!hunt_ready){hunt.assign(W,0.0f);cf_hunt_features(hunt.data(),state,tables);hunt_ready=true;}
            std::copy(hunt.begin(),hunt.end(),row);
        }else if(a.kind=="army_attack"||a.kind=="siege_assault"){
            std::string source,target;bool siege;
            Snapshot& w=scratch();uint64_t context;
            if(a.kind=="army_attack"){
                source=std::string(text(menu,cf_param(a,"from")));target=std::string(text(menu,cf_param(a,"to")));
                const auto sp=cf_param(a,"siege");
                siege=sp!=missing&&truth(menu,sp)&&at_controller(state,tables,sanitize_name(target))!=side;
                context=cp_container(w,WOTR_DICT,{rr_string(w,"attacker_region"),rr_string(w,source),rr_string(w,"defender_region"),rr_string(w,target),
                    rr_string(w,"attacker_side"),mu_enum(w,"game_state.Side",side==0?"FREE_PEOPLES":"SHADOW"),
                    rr_string(w,"defender_side"),mu_enum(w,"game_state.Side",side==0?"SHADOW":"FREE_PEOPLES"),
                    rr_string(w,"is_siege"),rr_scalar(w,WOTR_BOOL,siege?1:0)});
            }else{
                require(has_pc,"chance features: siege_assault without a pending combat (KeyError)");
                std::vector<uint64_t> kv;const auto entries=child_edges(state,pc);
                for(size_t k=0;k+1<entries.size();k+=2){
                    const auto key=entries[k];
                    if(state.nodes[key].tag==WOTR_STRING){const auto t=text(state,key);
                        if(t=="atk_combat_card"||t=="def_combat_card"||t=="atk_effect_count"||t=="def_effect_count"||t=="atk_effect_target"||t=="def_effect_target")continue;}
                    kv.push_back(key);kv.push_back(entries[k+1]);
                }
                context=cp_container(w,WOTR_DICT,kv);cp_put(w,context,"is_siege",rr_scalar(w,WOTR_BOOL,1));
                source=std::string(text(state,field(state,pc,"attacker_region")));target=std::string(text(state,field(state,pc,"defender_region")));siege=true;
            }
            const auto key=std::make_tuple(source,target,siege);
            auto found=std::find_if(attacks.begin(),attacks.end(),[&](const auto& e){return e.first==key;});
            if(found==attacks.end()){
                if(a.kind=="army_attack")cbb_freeze(w,context);
                // probabilities._base_combat on _CombatContext(state, context)
                const auto saved=obs_attr(w,w.root,"pending_combat",false);
                rr_set_attr(w,w.root,"pending_combat",context);
                const auto atk_side_node=field(w,context,"attacker_side");
                const int atk_side=ctx_side(w,atk_side_node);
                const auto is_siege=field(w,context,"is_siege",false);
                const auto combat=cbs_setup(w,tables,text(w,field(w,context,"attacker_region")),text(w,field(w,context,"defender_region")),
                    rr_scalar(w,WOTR_NONE),rr_scalar(w,WOTR_NONE),is_siege!=missing&&truth(w,is_siege),atk_side);
                rr_set_attr(w,w.root,"pending_combat",saved==missing?rr_scalar(w,WOTR_NONE):saved);
                cf_Combat cb=cf_read_combat(w,combat);
                const auto rounds=field(w,context,"rounds_fought",false);if(rounds!=missing&&truth(w,rounds))cb.city_defense=false;
                for(int r=0;r<2;++r){const auto red=field(w,context,(cp_short(r)+"_dice_reduction").c_str(),false);cb.dice_reduction[r]=red==missing?0:cf_int(w,red,"dice reduction must be integer");}
                std::vector<float> rv(W,0.0f);
                cf_combat_features(rv.data(),cb,w,tables,context,a,side,cf_Cursor{});
                cf_set(rv.data(),cf_I("opponent_card_unknown"),1);cf_set(rv.data(),cf_I("unresolved_choices"),1);
                attacks.push_back({key,std::move(rv)});found=attacks.end()-1;
            }
            std::copy(found->second.begin(),found->second.end(),row);
        }else if((a.kind=="combat_card_play"||a.kind=="combat_card_none"||a.kind=="combat_effect_choice")&&has_pc){
            const auto ec=field(state,pc,"effect_choice",false);
            if(ec!=missing&&!ctx_none(state,ec)&&truth(state,ec)){
                const auto st=field(state,ec,"stage",false);
                if(st!=missing&&state.nodes[st].tag==WOTR_STRING&&(text(state,st)=="post"||text(state,st)=="onslaught_payment")){
                    const auto eff=field(state,ec,"effect",false);
                    if(eff!=missing&&state.nodes[eff].tag==WOTR_STRING&&text(state,eff)=="onslaught"){
                        const auto count=cf_param(a,"count");require(count!=missing,"chance features: onslaught row without count (KeyError)");
                        cf_set(row,cf_I("combat_available"),1);
                        cf_put(row,"own_additional_hits_0",cf_plain_pmf(cf_int(menu,count,"onslaught count must be integer"),3));
                    }
                    continue;
                }
            }
            if(!base_ready){
                if(cursor.present)base=cf_read_combat(state,cursor.combat);
                else{
                    // probabilities._base_combat(state, pc): the view is the state itself (its own pending_combat)
                    Snapshot& w=scratch();
                    const auto wpc=obs_attr(w,w.root,"pending_combat");
                    const auto combat=cbs_setup(w,tables,text(w,field(w,wpc,"attacker_region")),text(w,field(w,wpc,"defender_region")),
                        rr_scalar(w,WOTR_NONE),rr_scalar(w,WOTR_NONE),[&]{const auto v=field(w,wpc,"is_siege",false);return v!=missing&&truth(w,v);}(),
                        ctx_side(w,field(w,wpc,"attacker_side")));
                    base=cf_read_combat(w,combat);
                    const auto rounds=field(w,wpc,"rounds_fought",false);if(rounds!=missing&&truth(w,rounds))base.city_defense=false;
                    for(int r=0;r<2;++r){const auto red=field(w,wpc,(cp_short(r)+"_dice_reduction").c_str(),false);base.dice_reduction[r]=red==missing?0:cf_int(w,red,"dice reduction must be integer");}
                }
                base_ready=true;
            }
            cf_combat_features(row,base,state,tables,pc,a,side,cursor);
        }
        }catch(const std::exception& e){throw std::runtime_error(std::string(e.what())+" [chance features row "+std::to_string(i)+" "+a.kind+"]");}
    }
}
