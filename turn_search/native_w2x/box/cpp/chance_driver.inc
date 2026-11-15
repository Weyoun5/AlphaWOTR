// NATIVE CHANCE DRIVER (claude_bugscan_2026-10-21/native_cl, 2026-10-27): selfplay_actor_critic/chance.py's ChanceDriver
// on the native engine's randomness funnels, for the chance logger's exact enumeration (chance.enumerate_chance).
//
// chance.driven() patches the Python funnels (game_logic._hunt_roll / roll_d6_batch / pop_hunt_tile / pick_hunt_tile /
// pick_random_companion / pick_random_card / draw_event_card, combat.roll_d6_batch, combat_round_resume._primary_side,
// random.shuffle) with a driver that follows a plan of option indices and, at the first unplanned draw, records the
// options with their probabilities and aborts the walk (Fork).  The native twins of those funnels consult cd_active
// first: hf_roll (game_logic.roll_d6_batch), hf_choice funnel 1 / 2 (pick_random_companion / pick_random_card),
// hf_pool (pop_hunt_tile / pick_hunt_tile), hf_shuffle (random.shuffle), cf_d6 (combat.roll_d6_batch),
// tc_draw_event_card (draw_event_card), hr_roll (_hunt_roll) and cp_primary_side (_primary_side).  Every other draw
// (hf_choice funnel 0: the raw random.choice sites) stays on the walk's own stream, the twin of the clone's
// GameRandom(0) python stream, exactly as the unpatched Python draws do; the walk compares that stream afterwards
// (chance.enumerate_chance's "raw RNG consumed outside the funnels").  Option lists, their order and the probability
// arithmetic are chance.py's (CPython float_pow, the same left-to-right products and sums).
//
// Exceptions: cd_Fork (Python Fork, a BaseException) and cd_Unsupported (chance.Unsupported) are NOT std::exception,
// so the engine's few catch(const std::exception&) blocks never swallow them; cd_Fallback marks a condition the Python
// driver raises as RuntimeError (a script mismatch, an index out of range) or an input outside the port: the caller
// then replays that whole event on the Python enumerator, which decides it exactly as before.
struct cd_Fork {};
struct cd_Unsupported {std::string why;};
struct cd_Fallback {std::string why;};
struct cd_Option {double p=0.0;int64_t value=0;std::vector<uint32_t> faces;};
struct cd_Script {std::string ctx;std::vector<uint32_t> faces;};
struct cd_Scope {
    std::vector<int64_t> plan;size_t ptr=0;double prob=1.0;int mover=-1;
    bool forked=false;std::string pending_kind;std::vector<cd_Option> pending;cd_Option chosen;
    std::vector<cd_Script> gl_script,cb_script;size_t gl_head=0,cb_head=0;
    const cd_Option& choose(const char* kind,std::vector<cd_Option> options){
        if(forked)throw cd_Fork{};                         // engine upkeep running while a Fork unwinds
        if(ptr<plan.size()){
            const int64_t i=plan[ptr++];
            if(i<0||static_cast<size_t>(i)>=options.size())throw cd_Fallback{"chance plan index out of range"};
            chosen=options[static_cast<size_t>(i)];prob*=chosen.p;return chosen;
        }
        forked=true;pending_kind=kind;pending=std::move(options);throw cd_Fork{};
    }
    bool scripts_left()const{return gl_head<gl_script.size()||cb_head<cb_script.size();}
};
thread_local cd_Scope* cd_active=nullptr;

// CPython float_pow on the finite arguments chance.py raises to a power (_binom: p ** k, (1.0 - p) ** (n - k))
double cd_pow(double iv,double iw){
    if(iw==0.0)return 1.0;
    if(std::isnan(iv))return iv;
    if(iv==0.0)return std::fmod(std::fabs(iw),2.0)==1.0?iv:0.0;
    bool negate=false;
    if(iv<0.0){if(iw!=std::floor(iw))throw cd_Fallback{"negative base to a fractional power"};iv=-iv;negate=std::fmod(std::fabs(iw),2.0)==1.0;}
    if(iv==1.0)return negate?-1.0:1.0;
    const double ix=std::pow(iv,iw);return negate?-ix:ix;
}
int64_t cd_comb(int64_t n,int64_t k){if(k<0||k>n)return 0;int64_t c=1;for(int64_t i=1;i<=k;++i)c=c*(n-k+i)/i;return c;}
// chance._binom: math.comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))
double cd_binom(int64_t n,int64_t k,double p){return (static_cast<double>(cd_comb(n,k))*cd_pow(p,static_cast<double>(k)))*cd_pow(1.0-p,static_cast<double>(n-k));}
// chance.total_hits_pmf
std::vector<double> cd_total_hits_pmf(int64_t n,double p_roll,int64_t rerolls,double p_reroll){
    if(n<0||n>64||rerolls<0)throw cd_Fallback{"hits pmf outside the port's domain"};
    std::vector<double> out(static_cast<size_t>(n+1),0.0);
    for(int64_t h=0;h<=n;++h){const int64_t k=std::min(rerolls,n-h);for(int64_t h2=0;h2<=k;++h2)out[static_cast<size_t>(h+h2)]+=cd_binom(n,h,p_roll)*cd_binom(k,h2,p_reroll);}
    return out;
}
// [(k, k, p) for k, p in enumerate(pmf) if p > 0.0]
std::vector<cd_Option> cd_pmf_options(const std::vector<double>& pmf){
    std::vector<cd_Option> out;for(size_t k=0;k<pmf.size();++k)if(pmf[k]>0.0){cd_Option o;o.p=pmf[k];o.value=static_cast<int64_t>(k);out.push_back(o);}
    return out;
}
// [6] * h + [1] * (n - h)
std::vector<uint32_t> cd_faces(int64_t hits,int64_t n){std::vector<uint32_t> f(static_cast<size_t>(n),1u);for(int64_t i=0;i<hits;++i)f[static_cast<size_t>(i)]=6u;return f;}
// chance.card_dice_options(n): itertools.combinations_with_replacement(range(1, 7), n), multinomial ways / float(6 ** n)
std::vector<cd_Option> cd_card_dice_options(size_t n){
    std::vector<cd_Option> options;std::vector<uint32_t> cur(n,1u);
    auto fact=[](int64_t k){int64_t f=1;for(int64_t i=2;i<=k;++i)f*=i;return f;};
    int64_t denom=1;for(size_t i=0;i<n;++i)denom*=6;
    while(true){
        int64_t ways=fact(static_cast<int64_t>(n));
        for(uint32_t v=1;v<=6;++v){int64_t c=0;for(auto x:cur)c+=x==v;ways/=fact(c);}
        cd_Option o;o.p=static_cast<double>(ways)/static_cast<double>(denom);o.faces=cur;options.push_back(std::move(o));
        size_t i=n;while(i>0&&cur[i-1]==6u)--i;           // next non-decreasing tuple (lexicographic)
        if(i==0)break;
        const uint32_t v=cur[i-1]+1u;for(size_t j=i-1;j<n;++j)cur[j]=v;
    }
    return options;
}
// chance._p_gl_roll (game_logic.roll_d6_batch under the driver)
void cd_gl_roll(uint32_t* faces,size_t count,std::string_view ctx){
    auto& d=*cd_active;
    if(d.gl_head<d.gl_script.size()){
        const auto& head=d.gl_script[d.gl_head];
        if(head.ctx!=ctx||head.faces.size()!=count)throw cd_Fallback{"hunt script mismatch at "+std::string(ctx)};
        std::copy(head.faces.begin(),head.faces.end(),faces);++d.gl_head;return;
    }
    if(ctx=="shelob d6 damage"&&count==1){
        std::vector<cd_Option> options;for(int v=1;v<=6;++v){cd_Option o;o.p=1.0/6.0;o.value=v;options.push_back(o);}
        faces[0]=static_cast<uint32_t>(d.choose("shelob_d6",std::move(options)).value);return;
    }
    if(ctx=="card effect"&&count>=1&&count<=3){
        const auto& pick=d.choose("card_dice",cd_card_dice_options(count));std::copy(pick.faces.begin(),pick.faces.end(),faces);return;
    }
    throw cd_Unsupported{"game_logic.roll_d6_batch ctx="+std::string(ctx)+" n="+std::to_string(count)};
}
// chance._p_cb_roll (combat.roll_d6_batch under the driver): scripted batches only
void cd_cb_roll(uint32_t* faces,size_t count,std::string_view ctx){
    auto& d=*cd_active;
    if(d.cb_head<d.cb_script.size()){
        const auto& head=d.cb_script[d.cb_head];
        if(head.ctx!=ctx||head.faces.size()!=count)throw cd_Fallback{"combat script mismatch at "+std::string(ctx)};
        std::copy(head.faces.begin(),head.faces.end(),faces);++d.cb_head;return;
    }
    throw cd_Unsupported{"combat.roll_d6_batch ctx="+std::string(ctx)+" n="+std::to_string(count)};
}
// chance._p_companion / _p_card: one option per candidate, 1.0 / n (no single-candidate shortcut, as chance.py)
int64_t cd_uniform(size_t count,const char* kind){
    std::vector<cd_Option> options;for(size_t i=0;i<count;++i){cd_Option o;o.p=1.0/static_cast<double>(count);o.value=static_cast<int64_t>(i);options.push_back(o);}
    return cd_active->choose(kind,std::move(options)).value;
}
// hunt_dist.tile_key on a native HuntTile (special names: game_state.SPECIAL_HUNT_TILES, matched as hunt_dist._norm does)
std::string cd_tile_key(const Snapshot& s,uint64_t tile){
    const auto special=obs_attr(s,tile,"special_name",false);
    if(special!=missing&&truth(s,special)){
        static const std::pair<const char*,const char*> keys[]={{"On On They Went","oonw"},{"Shelob's Lair","shelob"},{"The Ring is Mine!","ringmine"},
            {"Give it to Uss!","giveuss"},{"Sm\xc3\xa9" "agol Helps Nice Master","smeagol"},{"Elven Cloaks","ecloak"},{"Elven Rope","erope"},{"Phial of Galadriel","phial"}};
        if(s.nodes[special].tag!=WOTR_STRING)throw cd_Fallback{"special Hunt tile name is not a string"};
        const auto name=text(s,special);for(const auto& k:keys)if(name==k.first)return k.second;
        throw cd_Fallback{"special Hunt tile outside the known names"};
    }
    if(truth(s,obs_attr(s,tile,"is_eye")))return "eye";
    const auto dmg=obs_attr(s,tile,"damage");const auto& n=s.nodes[dmg];
    if(n.tag!=WOTR_INT)throw cd_Fallback{"Hunt tile damage is not an integer"};
    return std::to_string(n.integer)+(truth(s,obs_attr(s,tile,"reveal"))?"r":"");
}
// chance.tile_options + _p_pop / _p_pick: one option per tile kind (sorted keys), its first index, count / float(size)
size_t cd_pool(const Snapshot& s,uint64_t pool){
    const auto tiles=child_edges(s,pool);
    std::map<std::string,std::pair<size_t,size_t>> kinds;
    for(size_t i=0;i<tiles.size();++i){auto key=cd_tile_key(s,tiles[i]);auto f=kinds.find(key);if(f==kinds.end())kinds[key]={i,1};else ++f->second.second;}
    std::vector<cd_Option> options;
    for(const auto& e:kinds){cd_Option o;o.p=static_cast<double>(e.second.second)/static_cast<double>(tiles.size());o.value=static_cast<int64_t>(e.second.first);options.push_back(o);}
    return static_cast<size_t>(cd_active->choose("tile",std::move(options)).value);
}
// chance._p_draw: a draw by the MOVING side from its own deck when every card is known: one option per distinct card
// (sorted), count / float(size); the chosen card's last copy moves to the end of the deck, which the draw then pops
void cd_draw(Snapshot& s,int side,std::string_view deck){
    if(!cd_active||side!=cd_active->mover)return;
    const std::string attr=std::string(side==0?"fp":"sh")+(deck=="character"?"_char_deck":"_strat_deck");
    const auto list=obs_attr(s,s.root,attr.c_str(),false);if(list==missing||!truth(s,list))return;
    if(s.nodes[list].tag!=WOTR_LIST)throw cd_Fallback{"event deck is not a list"};
    auto cards=child_edges(s,list);if(cards.size()<=1)return;
    std::map<std::string,int64_t> count;
    for(auto c:cards){if(s.nodes[c].tag!=WOTR_STRING)return;const std::string id(text(s,c));if(id.rfind("_unknown_",0)==0)return;count[id]+=1;}
    std::vector<std::string> ids;std::vector<cd_Option> options;
    for(const auto& e:count){cd_Option o;o.p=static_cast<double>(e.second)/static_cast<double>(cards.size());o.value=static_cast<int64_t>(ids.size());ids.push_back(e.first);options.push_back(o);}
    const std::string want=ids[static_cast<size_t>(cd_active->choose("draw",std::move(options)).value)];
    size_t last=cards.size();for(size_t i=cards.size();i-->0;)if(text_is(s,cards[i],want)){last=i;break;}
    const auto node=cards[last];cards.erase(cards.begin()+static_cast<ptrdiff_t>(last));cards.push_back(node);set_edges(s,list,cards);
}
