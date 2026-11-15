// Exact search-only determinization. No gameplay or search policy changes.
// Source ai._SimOverlay owns mutable zones independently of the search state.
using so_Cards=std::vector<std::string>;
struct so_Atom {uint32_t tag=WOTR_NONE;int64_t number=0;std::string value;
    bool operator==(const so_Atom& b)const {const bool numeric=(tag==WOTR_INT||tag==WOTR_BOOL)&&(b.tag==WOTR_INT||b.tag==WOTR_BOOL);return (tag==b.tag||numeric)&&number==b.number&&value==b.value;}};
using so_Tile=std::array<so_Atom,8>;
struct so_Overlay {
    int root=0;std::array<so_Cards,2> hand,deck,fdown,own;std::vector<so_Tile> pool;
    bool cid_set=false,true_set=false,round_set=false,slot_set=false,hidden=false,table_set=false;
    std::string cid,true_cid,slot_deck;int64_t round=0,slot_index=0;
    std::string table_cid,table_name;   // R3-SL25-RESIDUALS: ai._SimOverlay.commit_table (card id, card name)
};
so_Overlay& so_overlay(void* p){require(p!=nullptr,"Search overlay handle is null");return *static_cast<so_Overlay*>(p);}
bool so_ok(const Snapshot& s,uint64_t n){return n!=missing&&s.nodes[n].tag==WOTR_STRING&&text(s,n).substr(0,9)!="_unknown_";}
int so_deck(std::string_view c){if(c.find("_Character_")!=c.npos)return 0;if(c.find("_Strategy_")!=c.npos)return 1;if(c.substr(0,3)=="HU_"&&c.find("_CHAR_")!=c.npos)return 0;return 1;}
std::string so_attr(int side,int d){return std::string(side==0?"fp":"sh")+(d==0?"_char_deck":"_strat_deck");}
so_Cards so_cards(const Snapshot& s,uint64_t list,int deck=-1){so_Cards r;if(ctx_none(s,list))return r;for(auto n:child_edges(s,list))if(so_ok(s,n)&&(deck<0||so_deck(text(s,n))==deck))r.emplace_back(text(s,n));return r;}
so_Atom so_atom(const Snapshot& s,uint64_t n){so_Atom a;if(ctx_none(s,n))return a;a.tag=s.nodes[n].tag;if(a.tag==WOTR_STRING)a.value=text(s,n);else if(a.tag==WOTR_INT||a.tag==WOTR_BOOL)a.number=s.nodes[n].integer;else throw std::invalid_argument("Search overlay key requires None, string, int or bool");return a;}
so_Tile so_tile(const Snapshot& s,uint64_t n){so_Tile t;int i=0;for(auto k:{"damage","is_eye","reveal","is_special","special_name","stops_movement","is_dice_roll","is_fellowship_special"}){auto a=obs_attr(s,n,k,false);t[i]=so_atom(s,a);if(i==3||i>=5)t[i]={WOTR_BOOL,truth(s,a)?1:0,{}};++i;}return t;}
uint64_t so_atom_node(Snapshot& s,const so_Atom& a){return a.tag==WOTR_STRING?rr_string(s,a.value):rr_scalar(s,a.tag,a.number);}
size_t so_pick(void* rng,size_t size){uint64_t n=0;if(wotr_random_randbelow(rng,size,&n))throw std::invalid_argument(wotr_random_last_error());return size_t(n);}
template<class T>void so_shuffle(std::vector<T>& xs,void* rng){std::vector<uint64_t> ix(xs.size());for(size_t i=0;i<ix.size();++i)ix[i]=i;if(wotr_random_shuffle(rng,ix.data(),ix.size()))throw std::invalid_argument(wotr_random_last_error());auto old=xs;for(size_t i=0;i<ix.size();++i)xs[i]=old[ix[i]];}
template<class T>bool so_remove(std::vector<T>& xs,const T& x){auto i=std::find(xs.begin(),xs.end(),x);if(i==xs.end())return false;xs.erase(i);return true;}
template<class T>std::vector<T> so_diff(const std::vector<T>& a,const std::vector<T>& b){
    // Counter subtraction preserves first encounter order, not lexical order.
    std::vector<T> seen,result;for(const auto& x:a)if(std::find(seen.begin(),seen.end(),x)==seen.end()){seen.push_back(x);auto n=std::count(a.begin(),a.end(),x)-std::count(b.begin(),b.end(),x);while(n-->0)result.push_back(x);}return result;
}
// Build both ordered Counter differences from one compact count vector.
// Common patches conserve card inventory; that path allocates no card strings.
std::array<so_Cards,2> so_card_differences(const so_Cards& a,const so_Cards& b){
    std::array<so_Cards,2> result;if(a==b||(a.size()==b.size()&&std::equal(a.begin(),a.end(),b.rbegin())))return result;
    std::vector<std::pair<std::string_view,int64_t>> counts;counts.reserve(a.size()+b.size());
    for(const auto& c:a)counts.emplace_back(c,1);for(const auto& c:b)counts.emplace_back(c,-1);
    std::sort(counts.begin(),counts.end(),[](const auto& x,const auto& y){return x.first<y.first;});
    size_t used=0;for(size_t i=0;i<counts.size();){size_t next=i+1;int64_t n=counts[i].second;while(next<counts.size()&&counts[next].first==counts[i].first)n+=counts[next++].second;if(n)counts[used++]={counts[i].first,n};i=next;}counts.resize(used);
    if(counts.empty())return result;
    for(int side=0;side<2;++side)for(const auto& c:(side?b:a)){
        auto found=std::lower_bound(counts.begin(),counts.end(),std::string_view(c),[](const auto& x,std::string_view y){return x.first<y;});
        if(found==counts.end()||found->first!=c)continue;auto n=side?-found->second:found->second;
        if(n>0){while(n-->0)result[side].push_back(c);found->second=0;}
    }return result;
}
template<class T>std::vector<T> so_join(const std::vector<T>& a,const std::vector<T>& b,const std::vector<T>& c={}){auto r=a;r.insert(r.end(),b.begin(),b.end());r.insert(r.end(),c.begin(),c.end());return r;}
bool so_require(so_Overlay& o,void* rng,const std::string& cid){auto d=so_deck(cid);auto& h=o.hand[d];if(std::find(h.begin(),h.end(),cid)!=h.end())return true;if(h.empty())return false;auto victim=h[so_pick(rng,h.size())];bool found=false;for(auto* z:{&o.deck[d],&o.fdown[d]}){auto at=std::find(z->begin(),z->end(),cid);if(at!=z->end()){*at=victim;found=true;break;}}if(!found)return false;so_remove(h,victim);h.push_back(cid);return true;}
// R10-PY-SEARCH-ATHELAS (ai._pending_held_cards): the hand cards the open pending_card_option decided by `side` names as HELD by its decider -
// the Ents Awake free chain's {'ents_chain_play': card} rows, which eea_chain builds from the FP hand when the chain opens (the chain's nested
// free play refuses a card the hand does not hold). Options order, distinct; empty when no such pending is open. A cached in-tree chain state
// keeps the options of the sample that built it; so_patch conditions every later sample on them.
so_Cards so_pending_held(const Snapshot& s,int side){
    so_Cards r;const auto p=obs_attr(s,s.root,"pending_card_option",false);if(p==missing||s.nodes[p].tag!=WOTR_DICT)return r;
    auto decider=field(s,p,"decider",false);if(decider==missing)decider=field(s,p,"side",false);
    if(decider==missing||s.nodes[decider].tag!=WOTR_STRING||text(s,decider)!=(side==0?"fp":"sh"))return r;
    const auto options=field(s,p,"options",false);if(ctx_none(s,options))return r;
    for(auto option:child_edges(s,options)){if(s.nodes[option].tag!=WOTR_DICT)continue;const auto cid=field(s,option,"ents_chain_play",false);if(!so_ok(s,cid))continue;
        const std::string c(text(s,cid));if(std::find(r.begin(),r.end(),c)==r.end())r.push_back(c);}
    return r;
}
// R10-PY-SEARCH-ATHELAS (ai._SimOverlay._condition_held): every named card of deck d must be IN the sampled hand. A missing one is exchanged
// with a uniformly drawn hand card the pending does not name, which takes its deck / face-down slot (so_require's exchange: per-deck counts
// and the union multiset unchanged). No RNG and no change when the sample already holds them all; an inconsistent cached state is loud.
void so_condition_held(so_Overlay& o,int d,const so_Cards& held,void* rng){
    auto& h=o.hand[d];
    for(const auto& cid:held){
        if(so_deck(cid)!=d||std::find(h.begin(),h.end(),cid)!=h.end())continue;
        so_Cards free;for(const auto& c:h)if(std::find(held.begin(),held.end(),c)==held.end())free.push_back(c);
        so_Cards* zone=nullptr;for(auto* z:{&o.deck[d],&o.fdown[d]})if(std::find(z->begin(),z->end(),cid)!=z->end()){zone=z;break;}
        if(!zone||free.empty())throw std::invalid_argument("Hidden sample cannot hold the pending card "+cid+" ("+(zone?"no free hand slot":"not in the unseen pool")+"): it is outside the unseen pool or no other hand card can make room");
        const auto victim=free[so_pick(rng,free.size())];*std::find(zone->begin(),zone->end(),cid)=victim;so_remove(h,victim);h.push_back(cid);
    }
}
void so_recon(so_Overlay& o,int d,const so_Cards& sh,const so_Cards& sd,const so_Cards& sf,void* rng){
    auto& h=o.hand[d];auto& deck=o.deck[d];auto& fd=o.fdown[d];const auto sz=so_join(sh,sd,sf),oz=so_join(h,deck,fd);
    auto differences=so_card_differences(oz,sz);
    for(auto& c:differences[0])if(!so_remove(h,c)&&!so_remove(deck,c))so_remove(fd,c);
    for(auto& c:differences[1])deck.insert(deck.begin()+so_pick(rng,deck.size()+1),c);
    while(h.size()<sh.size()&&!deck.empty()){h.push_back(deck.front());deck.erase(deck.begin());}
    while(h.size()<sh.size()&&!fd.empty()){auto n=so_pick(rng,fd.size());h.push_back(fd[n]);fd.erase(fd.begin()+n);}
    while(h.size()>sh.size()){auto n=so_pick(rng,h.size());fd.push_back(h[n]);h.erase(h.begin()+n);}
    // Python evaluates the insert position before its pop() argument.
    while(deck.size()<sd.size()&&!fd.empty()){auto insert=so_pick(rng,deck.size()+1),remove=so_pick(rng,fd.size());auto c=fd[remove];fd.erase(fd.begin()+remove);deck.insert(deck.begin()+insert,c);}
    while(deck.size()>sd.size()){auto n=so_pick(rng,deck.size());fd.push_back(deck[n]);deck.erase(deck.begin()+n);}
}
std::vector<std::pair<std::string,int>> so_mixture(const Snapshot& source,const Snapshot& tables,int deck,const so_Cards& extra={}){
    const auto pc=obs_attr(source,source.root,"pending_combat");require(!ctx_none(source,pc),"Commitment mixture requires combat");const int actor=cbb_side(source,pc,0);const auto info=ctx_side_value(source,obs_attr(source,source.root,"hands"),actor);
    auto candidates=so_join(so_cards(source,ctx_none(source,info)?missing:obs_attr(source,info,"hand")),so_cards(source,ctx_none(source,info)?missing:obs_attr(source,info,"discarded_face_down",false)));
    for(int d=0;d<2;++d){auto xs=so_cards(source,obs_attr(source,source.root,so_attr(actor,d).c_str(),false));candidates.insert(candidates.end(),xs.begin(),xs.end());}candidates.insert(candidates.end(),extra.begin(),extra.end());
    Snapshot s(source);auto combat=cbs_setup(s,tables,text(source,field(source,pc,"attacker_region")),text(source,field(source,pc,"defender_region")),rr_scalar(s,WOTR_NONE),rr_scalar(s,WOTR_NONE),truth(source,field(source,pc,"is_siege",false)),actor);
    std::map<std::string,int> mix;std::set<std::string> rejected;if(actor==0&&sanitize_name(text(source,field(source,pc,"defender_region")))=="minas tirith"&&cau_persistent(source,"DENETHORS_FOLLY")!=missing)return {};
    for(auto& c:candidates){if(deck>=0&&so_deck(c)!=deck)continue;if(mix.count(c)){++mix[c];continue;}if(rejected.count(c))continue;auto card=field(tables,field(tables,tables.root,"cards"),c.c_str(),false);if(ccm_legal(s,tables,combat,card,0))mix[c]=1;else rejected.insert(c);}return {mix.begin(),mix.end()};
}
// R2-SL-25 (ai._hidden_commit_table_card): the on-table Balrog of Moria's id when it could be a Shadow attacker's hidden Character commit (1.149b).
std::string so_table_commit(const Snapshot& s,uint64_t pc,int deck){if(deck!=0||cbb_side(s,pc,0)!=1)return {};const auto card=cau_persistent(s,"BALROG_OF_MORIA");return card==missing?std::string():std::string(text(s,obs_attr(s,card,"card_id")));}
// R2-SL-25 / R3-SL25-RESIDUALS (ai._hidden_commit_mixture): a hidden commit's identity must be playable and payable. An unseen-pool card (plus
// `extra`) needs a same-deck hand card (held>0, the public count); the on-table card `table` (Durin's Bane, 1.149b) pays for itself and is a
// candidate beside them (mixed case, copy weight 1). Empty = searched as 'no card' by so_sample/so_patch, never a raise.
std::vector<std::pair<std::string,int>> so_hidden_mixture(const Snapshot& s,const Snapshot& t,int d,const std::string& table,size_t held,so_Cards extra){
    if(!table.empty())extra.push_back(table);auto mix=so_mixture(s,t,d,extra);if(held>0)return mix;std::vector<std::pair<std::string,int>> only;for(auto& x:mix)if(!table.empty()&&x.first==table)only.push_back(x);return only;}
// ai._atk_commit_mixture: hidden commit -> the hidden belief; a search root whose hidden commit so_patch resolved in a belief listing the table
// card (search-scoped _commit_from_table marker) -> the same hidden belief over the PRE-commit world (a hand payment returns to the hand, a table
// payment to the table); a PUBLIC table commit (pc atk_cc_from_table, cbc_table_record) -> certain; otherwise the public same-type pool.
std::vector<std::pair<std::string,int>> so_commit_mixture(const Snapshot& s,const Snapshot& t){auto pc=obs_attr(s,s.root,"pending_combat",false);if(!truth(s,pc)||!truth(s,field(s,pc,"atk_combat_card",false)))return {};auto hidden=field(s,pc,"atk_hidden_cc",false);
    const auto info=ctx_side_value(s,obs_attr(s,s.root,"hands"),cbb_side(s,pc,0));const auto hand=ctx_none(s,info)?missing:obs_attr(s,info,"hand");
    if(truth(s,hidden)){const int d=text_is(s,field(s,hidden,"deck",false),"character")?0:1;return so_hidden_mixture(s,t,d,so_table_commit(s,pc,d),so_cards(s,hand,d).size(),{});}
    auto id=std::string(text(s,field(s,pc,"atk_combat_card")));const int d=so_deck(id);const auto from_table=obs_attr(s,s.root,"_commit_from_table",false);
    if(truth(s,from_table)){const std::string table(text(s,from_table));const size_t held=so_cards(s,hand,d).size();if(id==table)return so_hidden_mixture(s,t,d,table,held,{});return so_hidden_mixture(s,t,d,table,held+1,{id});}
    const auto record=field(s,pc,"atk_cc_from_table",false);if(record!=missing&&!ctx_none(s,record)&&text_is(s,record,id))return {{id,1}};return so_mixture(s,t,d,{id});}
std::unique_ptr<Snapshot> so_mixture_graph(const Snapshot& s,const Snapshot& t){am_Builder b;std::vector<uint64_t> rows;for(auto& [id,count]:so_commit_mixture(s,t))rows.push_back(b.container(WOTR_TUPLE,{b.str(id),b.integer(count)}));b.out.root=b.container(WOTR_LIST,rows);return std::make_unique<Snapshot>(b.out);}
std::unique_ptr<so_Overlay> so_sample(const Snapshot& s,const Snapshot& tables,int root,void* rng){
    require(root==0||root==1,"Search overlay root must be FP or SH");auto ptr=std::make_unique<so_Overlay>();auto& o=*ptr;o.root=root;const int opp=1-root;const auto pc=obs_attr(s,s.root,"pending_combat",false),slot=obs_attr(s,s.root,"_commit_slot",false);
    if(!ctx_none(s,slot)){auto xs=child_edges(s,slot);require(xs.size()==2,"Malformed search commit slot");o.slot_set=true;o.slot_deck=text(s,xs[0]);o.slot_index=typed(s,xs[1],WOTR_INT,"Malformed search commit index").integer;}
    if(truth(s,pc)&&truth(s,field(s,pc,"atk_combat_card",false))&&root!=cbb_side(s,pc,0)&&(truth(s,field(s,pc,"atk_hidden_cc",false))||text_is(s,field(s,pc,"phase",false),"defender_combat_card"))){
        // R2-SL-25 (ai._SimOverlay.__init__): the mixture comes first (RNG-free). An identity-free HIDDEN (K1) commit is searched as 'no card':
        // no identity, no reserved slot, round still pinned. A public commit folds itself back into its mixture, so empty stays an invariant failure.
        const auto hidden=field(s,pc,"atk_hidden_cc",false),committed=field(s,pc,"atk_combat_card");o.hidden=truth(s,hidden);
        auto mix=so_commit_mixture(s,tables);require(o.hidden||!mix.empty(),"Hidden combat-card commitment has no representable candidate");
        // R3-SL25-RESIDUALS (ai._SimOverlay.__init__ commit_table): the on-table card the hidden belief lists, as (card id, card name from the
        // card table); a resolved hidden root keeps it as the search-scoped _commit_from_table marker. RNG-free.
        {std::string table;if(o.hidden)table=so_table_commit(s,pc,text_is(s,field(s,hidden,"deck",false),"character")?0:1);else{const auto marker=obs_attr(s,s.root,"_commit_from_table",false);if(truth(s,marker))table=std::string(text(s,marker));}
            if(!table.empty()&&std::any_of(mix.begin(),mix.end(),[&](const auto& x){return x.first==table;})){const auto card=field(tables,field(tables,tables.root,"cards"),table.c_str(),false);if(ctx_none(tables,card))throw std::invalid_argument("Table commit identity has no card record: "+table);o.table_set=true;o.table_cid=table;o.table_name=std::string(text(tables,obs_attr(tables,card,"card_name")));}}
        if(o.hidden&&!mix.empty()){o.slot_set=true;o.slot_deck=std::string(opp==0?"fp_":"sh_")+std::string(text(s,field(s,hidden,"deck")));o.slot_index=static_cast<int64_t>(child_edges(s,field(s,obs_attr(s,s.root,"discard_face_up"),o.slot_deck.c_str())).size());}auto original=std::string(text(s,committed));
        if(!mix.empty()){std::vector<double> weights;double total=0;for(auto& x:mix){total+=x.second;weights.push_back(total);}uint64_t pick=0;if(wotr_random_choices(rng,mix.size(),weights.data(),weights.size(),1,&pick,1))throw std::invalid_argument(wotr_random_last_error());o.cid_set=true;o.cid=mix[pick].first;}
        o.round_set=true;auto round=field(s,pc,"rounds_fought",false);o.round=round==missing?0:typed(s,round,WOTR_INT,"Overlay round must be integer").integer;if(!o.hidden){o.true_set=true;o.true_cid=original;}
    }
    auto info=ctx_side_value(s,obs_attr(s,s.root,"hands"),opp);auto hand=so_cards(s,ctx_none(s,info)?missing:obs_attr(s,info,"hand")),fd=so_cards(s,ctx_none(s,info)?missing:obs_attr(s,info,"discarded_face_down",false));
    for(int d=0;d<2;++d){auto dl=so_cards(s,obs_attr(s,s.root,so_attr(opp,d).c_str(),false));so_Cards hd,df;for(auto& c:hand)if(so_deck(c)==d)hd.push_back(c);for(auto& c:fd)if(so_deck(c)==d)df.push_back(c);auto pool=so_join(dl,hd,df);
        // R2-SL-25: only an identity drawn from the unseen pool costs a hand card; the on-table Balrog pays with the table card.
        // R3-SL25-RESIDUALS: a table payment (table_cid) was never in the unseen pool and cost no hand card; the hand count moves by one when this
        // world's payment and the represented one differ in kind.
        bool spent=false;int64_t shift=0;if(o.true_set&&so_deck(o.true_cid)==d&&o.cid!=o.true_cid){const bool true_hand=!(o.table_set&&o.true_cid==o.table_cid),cid_hand=!(o.table_set&&o.cid==o.table_cid);if(true_hand)pool.push_back(o.true_cid);so_remove(pool,o.cid);shift=int64_t(true_hand)-int64_t(cid_hand);}else if(o.hidden&&o.cid_set&&so_deck(o.cid)==d)spent=so_remove(pool,o.cid);
        std::sort(pool.begin(),pool.end());so_shuffle(pool,rng);const int64_t hand_count=static_cast<int64_t>(hd.size())-int(spent)+shift;require(hand_count>=0,"Hidden combat-card commitment has no hand card to spend");auto h=std::min(static_cast<size_t>(hand_count),pool.size()),m=std::min(h+dl.size(),pool.size());o.hand[d]={pool.begin(),pool.begin()+h};o.deck[d]={pool.begin()+h,pool.begin()+m};o.fdown[d]={pool.begin()+m,pool.end()};
    }
    for(int d=0;d<2;++d){o.own[d]=so_cards(s,obs_attr(s,s.root,so_attr(root,d).c_str(),false));so_shuffle(o.own[d],rng);}auto pool=obs_attr(s,s.root,"hunt_pool",false);if(!ctx_none(s,pool))for(auto tile:child_edges(s,pool))o.pool.push_back(so_tile(s,tile));so_shuffle(o.pool,rng);return ptr;
}
std::unique_ptr<Snapshot> so_export(const so_Overlay& o){
    am_Builder b;auto nil=rr_scalar(b.out,WOTR_NONE);std::vector<uint64_t> fields;auto set=[&](const char* k,uint64_t v){fields.push_back(b.str(k));fields.push_back(v);};auto side=[&](int n){return b.container(WOTR_ENUM,{b.str("game_state.Side"),b.str(n==0?"FREE_PEOPLES":"SHADOW")});};
    set("root_side",side(o.root));set("opp_side",side(1-o.root));set("opp_tag",b.str(o.root==0?"sh":"fp"));set("own_tag",b.str(o.root==0?"fp":"sh"));
    for(auto named:{std::make_pair("hand",&o.hand),std::make_pair("deck",&o.deck),std::make_pair("fdown",&o.fdown),std::make_pair("own_deck",&o.own)}){std::vector<uint64_t> values;for(int d=0;d<2;++d){std::vector<uint64_t> xs;for(auto& c:(*named.second)[d])xs.push_back(b.str(c));values.push_back(b.str(d==0?"char":"strat"));values.push_back(b.container(WOTR_LIST,xs));}set(named.first,b.container(WOTR_DICT,values));}
    std::vector<uint64_t> keys;for(auto& tile:o.pool){std::vector<uint64_t> xs;for(auto& atom:tile)xs.push_back(so_atom_node(b.out,atom));keys.push_back(b.container(WOTR_TUPLE,xs));}set("pool",b.container(WOTR_LIST,keys));set("commit_cid",o.cid_set?b.str(o.cid):nil);set("commit_true",o.true_set?b.str(o.true_cid):nil);set("commit_bkey",o.round_set?b.integer(o.round):nil);set("commit_slot",o.slot_set?b.container(WOTR_TUPLE,{b.str(o.slot_deck),b.integer(o.slot_index)}):nil);set("commit_hidden",rr_scalar(b.out,WOTR_BOOL,o.hidden));set("commit_table",o.table_set?b.container(WOTR_TUPLE,{b.str(o.table_cid),b.str(o.table_name)}):nil);b.out.root=b.container(WOTR_DICT,fields);return std::make_unique<Snapshot>(b.out);
}
std::unique_ptr<so_Overlay> so_import(const Snapshot& s){
    auto p=std::make_unique<so_Overlay>();auto& o=*p;o.root=ctx_side(s,field(s,s.root,"root_side"));for(auto named:{std::make_pair("hand",&o.hand),std::make_pair("deck",&o.deck),std::make_pair("fdown",&o.fdown),std::make_pair("own_deck",&o.own)})for(int d=0;d<2;++d)(*named.second)[d]=so_cards(s,field(s,field(s,s.root,named.first),d==0?"char":"strat"));
    for(auto k:child_edges(s,field(s,s.root,"pool"))){auto xs=child_edges(s,k);require(xs.size()==8,"Overlay tile key requires eight fields");so_Tile tile;for(size_t i=0;i<8;++i)tile[i]=so_atom(s,xs[i]);o.pool.push_back(tile);}
    auto cid=field(s,s.root,"commit_cid"),original=field(s,s.root,"commit_true"),round=field(s,s.root,"commit_bkey"),slot=field(s,s.root,"commit_slot");o.cid_set=!ctx_none(s,cid);if(o.cid_set)o.cid=text(s,cid);o.true_set=!ctx_none(s,original);if(o.true_set)o.true_cid=text(s,original);o.round_set=!ctx_none(s,round);if(o.round_set)o.round=typed(s,round,WOTR_INT,"Overlay round requires integer").integer;o.slot_set=!ctx_none(s,slot);if(o.slot_set){auto xs=child_edges(s,slot);require(xs.size()==2,"Overlay slot requires pair");o.slot_deck=text(s,xs[0]);o.slot_index=typed(s,xs[1],WOTR_INT,"Overlay slot requires integer").integer;}o.hidden=truth(s,field(s,s.root,"commit_hidden"));
    auto table=field(s,s.root,"commit_table");o.table_set=!ctx_none(s,table);if(o.table_set){auto xs=child_edges(s,table);require(xs.size()==2,"Overlay table commit requires pair");o.table_cid=text(s,xs[0]);o.table_name=text(s,xs[1]);}return p;
}
// Search patch only: preserve distinct containers even if an imported graph
// shares all/part of their edge storage. Never assume raw ABI spans are unique.
void so_assign(Snapshot& s,uint64_t node,const std::vector<uint64_t>& values){
    const auto old=s.nodes[node];if(child_edges(s,node)==values)return;
    bool exclusive=old.count==values.size();
    if(exclusive&&old.count)for(size_t i=0;i<s.nodes.size();++i){const auto& other=s.nodes[i];if(i==node||other.tag<WOTR_LIST||other.tag>WOTR_ENUM||!other.count)continue;if(old.offset<other.offset+other.count&&other.offset<old.offset+old.count){exclusive=false;break;}}
    if(!exclusive){set_edges(s,node,values);return;}
    invalidate_action_clone(s);s.context_prepared.reset();std::copy(values.begin(),values.end(),s.edges.begin()+old.offset);
}
void so_put(Snapshot& s,uint64_t dictionary,std::string_view key,uint64_t value){auto xs=child_edges(s,dictionary);for(size_t i=0;i<xs.size();i+=2)if(text_is(s,xs[i],key)){xs[i+1]=value;so_assign(s,dictionary,xs);return;}xs.push_back(rr_string(s,key));xs.push_back(value);so_assign(s,dictionary,xs);}
struct so_Writer {
    Snapshot& s;using Entry=std::pair<size_t,uint64_t>;std::vector<Entry> strings;
    static size_t key(std::string_view value){return std::hash<std::string_view>{}(value);}
    void index(){strings.clear();strings.reserve(128);for(size_t i=0;i<s.nodes.size();++i)if(s.nodes[i].tag==WOTR_STRING){auto v=text(s,i);if(v.find("_Character_")!=v.npos||v.find("_Strategy_")!=v.npos||v.substr(0,3)=="HU_")strings.emplace_back(key(v),i);}std::sort(strings.begin(),strings.end());}
    explicit so_Writer(Snapshot& state):s(state){index();}
    uint64_t str(const std::string& value){auto h=key(value);auto at=std::lower_bound(strings.begin(),strings.end(),Entry{h,0});for(;at!=strings.end()&&at->first==h;++at)if(text_is(s,at->second,value))return at->second;auto n=rr_string(s,value);auto item=Entry{h,n};strings.insert(std::lower_bound(strings.begin(),strings.end(),item),item);return n;}
    void write(uint64_t node,const so_Cards& values,bool reverse=false,const std::vector<uint64_t>& pinned={}){auto xs=pinned;if(reverse)for(auto i=values.rbegin();i!=values.rend();++i)xs.push_back(str(*i));else for(auto& c:values)xs.push_back(str(c));so_assign(s,node,xs);}
};
void so_patch(so_Overlay& o,Snapshot& s,void* rng){
    const auto pending=obs_attr(s,s.root,"pending_combat",false);
    if(o.hidden&&!ctx_none(s,pending)&&truth(s,field(s,pending,"atk_hidden_cc",false))){
        const auto rounds=field(s,pending,"rounds_fought",false);const int64_t number=ctx_none(s,rounds)?0:s.nodes[rounds].integer;
        if(o.round_set&&number==o.round&&!o.cid_set){
            // R2-SL-25 (ai._SimOverlay.patch): no playable, payable identity - this world's attacker plays no card (the live reveal refuses the real card).
            cbc_table_record(s,pending,0,rr_scalar(s,WOTR_NONE),false);cbc_complete_commit(s,pending,0,rr_scalar(s,WOTR_NONE));cp_erase(s,pending,"atk_hidden_cc");
        }else if(o.round_set&&number==o.round){
            require(o.cid_set&&o.slot_set,"Hidden commitment requires sampled identity and reserved slot");
            const int opponent=1-o.root,d=so_deck(o.cid);const auto info=ctx_side_value(s,obs_attr(s,s.root,"hands"),opponent),hand=obs_attr(s,info,"hand");auto cards=child_edges(s,hand);uint64_t board=missing;
            if(std::none_of(cards.begin(),cards.end(),[&](auto n){return text_is(s,n,o.cid);})){auto victim=std::find_if(cards.begin(),cards.end(),[&](auto n){return so_ok(s,n)&&so_deck(text(s,n))==d;});uint64_t source=missing;size_t source_index=0;
                for(auto zone:{obs_attr(s,info,"discarded_face_down"),obs_attr(s,s.root,so_attr(opponent,d).c_str())}){auto xs=child_edges(s,zone);auto at=std::find_if(xs.begin(),xs.end(),[&](auto n){return text_is(s,n,o.cid);});if(at!=xs.end()){source=zone;source_index=static_cast<size_t>(at-xs.begin());break;}}
                // R2-SL-25: the on-table Balrog (1.149b) pays with the table card, never a hand card.
                if(source==missing&&so_table_commit(s,pending,d)==o.cid)board=cau_persistent(s,"BALROG_OF_MORIA");
                if(board==missing){require(victim!=cards.end()&&source!=missing,"Hidden combat-card commitment cannot conserve its hand payment");auto xs=child_edges(s,source);xs[source_index]=*victim;so_assign(s,source,xs);*victim=rr_string(s,o.cid);so_assign(s,hand,cards);}
            }
            const auto id=rr_string(s,o.cid);if(board!=missing)cbp_discard_table(s,cbp_remove_card(s,obs_attr(s,s.root,"persistent_cards"),board));else cbc_spend_hand(s,opponent,id,o.slot_deck);cbc_table_record(s,pending,0,id,board!=missing);cbc_complete_commit(s,pending,0,id);cp_erase(s,pending,"atk_hidden_cc");rr_set_attr(s,s.root,"_commit_slot",cp_container(s,WOTR_TUPLE,{rr_string(s,o.slot_deck),rr_scalar(s,WOTR_INT,o.slot_index)}));
            // search-scoped like _commit_slot (ai._SimOverlay.patch); R3-SL25-RESIDUALS: names the table card the hidden belief lists, whichever identity resolved
            if(o.table_set)rr_set_attr(s,s.root,"_commit_from_table",rr_string(s,o.table_cid));
        }
    }
    bool represented_set=o.true_set;auto represented=o.true_cid;auto public_dict=obs_attr(s,s.root,"discard_face_up",false);
    auto public_slot=[&]()->std::pair<uint64_t,int64_t>{if(!o.slot_set||ctx_none(s,public_dict))return {missing,0};auto list=field(s,public_dict,o.slot_deck.c_str(),false);if(ctx_none(s,list))return {missing,0};auto size=int64_t(s.nodes[list].count),ix=o.slot_index;if(ix>=size)return {missing,0};if(ix<0)ix+=size;require(ix>=0,"Overlay public slot index out of range");return {list,ix};};
    auto slot=public_slot();if(slot.first!=missing){auto record=child_edges(s,slot.first)[slot.second];if(!text_is(s,record,"_committed_face_down_")){represented_set=!ctx_none(s,record);if(represented_set)represented=text(s,record);}}
    if(o.slot_set&&o.cid_set&&(!represented_set||represented!=o.cid)&&!ctx_none(s,public_dict)){auto es=child_edges(s,public_dict);for(size_t j=0;j<es.size();j+=2){auto xs=child_edges(s,es[j+1]);for(size_t k=0;k<xs.size();++k)if(!(text_is(s,es[j],o.slot_deck)&&int64_t(k)==o.slot_index)&&text_is(s,xs[k],o.cid))throw std::invalid_argument("Sampled commitment conflicts with public card history");}}
    // R3-SL25-RESIDUALS (ai._SimOverlay.patch _xtab9): an exchange between a TABLE world and a HAND world also moves the table card and one hand
    // card: 1 = to_table (the represented hand payment returns to the hand, the table card takes the slot), 2 = to_hand (the table card returns,
    // a hand card pays). Only inside this card window (later states are world-keyed); outside it the exchange is refused.
    int xtab=0;
    if(o.table_set&&o.cid_set&&represented_set&&o.cid!=represented&&(o.table_cid==o.cid||o.table_cid==represented)){
        const auto window=obs_attr(s,s.root,"pending_combat",false);bool inside=false;
        if(truth(s,window)&&truth(s,field(s,window,"atk_combat_card",false))&&truth(s,field(s,window,"_commit_nonce",false))){auto round=field(s,window,"rounds_fought",false);inside=o.round_set&&(round==missing?0:s.nodes[round].integer)==o.round;}
        require(inside,"Sampled table/hand commitment exchange outside its card window");xtab=o.cid==o.table_cid?1:2;
    }
    // The explicit conflict refusal above precedes any cache/state/overlay/RNG edit.
    so_Writer writer(s);auto info=ctx_side_value(s,obs_attr(s,s.root,"hands"),1-o.root);
    if(!ctx_none(s,info)){
        auto hand=obs_attr(s,info,"hand"),fd=obs_attr(s,info,"discarded_face_down");
        // R10-PY-SEARCH-ATHELAS: hand cards this state's open pending of the sampled side names as held; each reconcile is conditioned on them.
        const auto held=so_pending_held(s,1-o.root);
        for(int d=0;d<2;++d){auto dl=obs_attr(s,s.root,so_attr(1-o.root,d).c_str(),false);if(ctx_none(s,dl))continue;auto sh=so_cards(s,hand,d),sf=so_cards(s,fd,d),sd=so_cards(s,dl);
            if(o.cid_set&&represented_set&&o.cid!=represented&&so_deck(represented)==d){
                if(xtab==1)sh.push_back(represented);
                else if(xtab==2){auto at=std::find(sh.begin(),sh.end(),o.cid);if(at!=sh.end())sh.erase(at);else{require(!sh.empty(),"Sampled hand commitment has no hand card to spend");const auto victim=sh.front();sh.erase(sh.begin());
                    auto in_fd=std::find(sf.begin(),sf.end(),o.cid);if(in_fd!=sf.end())*in_fd=victim;else{auto in_deck=std::find(sd.begin(),sd.end(),o.cid);require(in_deck!=sd.end(),"Sampled hand commitment is outside the unseen pool");*in_deck=victim;}}}
                else for(auto* xs:{&sh,&sf,&sd}){auto at=std::find(xs->begin(),xs->end(),o.cid);if(at!=xs->end()){*at=represented;break;}}
            }
            so_recon(o,d,sh,sd,sf,rng);so_condition_held(o,d,held,rng);writer.write(dl,o.deck[d],true);
        }
        for(auto named:{std::make_pair(hand,&o.hand),std::make_pair(fd,&o.fdown)}){std::vector<uint64_t> pinned;for(auto n:child_edges(s,named.first))if(!so_ok(s,n))pinned.push_back(n);writer.write(named.first,so_join((*named.second)[0],(*named.second)[1]),false,pinned);}
    }
    if(xtab){
        // R3-SL25-RESIDUALS: the table side of the exchange, with the public from-table record (cbc_table_record) kept in step.
        const auto persistent=obs_attr(s,s.root,"persistent_cards"),window=obs_attr(s,s.root,"pending_combat");uint64_t found=missing;
        for(const char* key:{"fp_cards","sh_cards"}){for(auto card:child_edges(s,obs_attr(s,persistent,key)))if(text_is(s,obs_attr(s,card,"card_id"),o.table_cid)){found=card;break;}if(found!=missing)break;}
        if(xtab==1){require(found!=missing,"Sampled table commitment: the table card is not on the table");cbp_remove_card(s,persistent,found);cp_put(s,window,"atk_cc_from_table",rr_string(s,o.table_cid));}
        else{require(found==missing,"Sampled hand commitment: the table card is already on the table");
            const auto card=mu_object(s,"game_state.PersistentCard",{{"card_id",rr_string(s,o.table_cid)},{"card_name",rr_string(s,o.table_name)},{"side",mu_enum(s,"game_state.Side",o.root==0?"SHADOW":"FREE_PEOPLES")},{"card_type",mu_enum(s,"game_state.PersistentCardType","BALROG_OF_MORIA")},{"requires_companion",rr_scalar(s,WOTR_NONE)}});
            const auto list=obs_attr(s,persistent,o.root==0?"sh_cards":"fp_cards");auto cards=child_edges(s,list);cards.push_back(card);set_edges(s,list,cards);cbc_table_record(s,window,0,rr_scalar(s,WOTR_NONE),false);}
    }
    for(int d=0;d<2;++d){auto dl=obs_attr(s,s.root,so_attr(o.root,d).c_str(),false);if(ctx_none(s,dl))continue;auto sc=so_cards(s,dl);auto differences=so_card_differences(o.own[d],sc);for(auto& c:differences[0])so_remove(o.own[d],c);for(auto& c:differences[1])o.own[d].insert(o.own[d].begin()+so_pick(rng,o.own[d].size()+1),c);writer.write(dl,o.own[d],true);}
    auto pool=obs_attr(s,s.root,"hunt_pool",false);if(!ctx_none(s,pool)){std::vector<so_Tile> keys;auto objects=child_edges(s,pool);for(auto n:objects)keys.push_back(so_tile(s,n));auto old=o.pool;if(old!=keys&&!(old.size()==keys.size()&&std::equal(old.begin(),old.end(),keys.rbegin()))){for(auto& k:so_diff(old,keys))so_remove(o.pool,k);for(auto& k:so_diff(keys,old))o.pool.insert(o.pool.begin()+so_pick(rng,o.pool.size()+1),k);}
        std::vector<uint64_t> result;std::vector<bool> used(objects.size());for(auto& key:o.pool){bool found=false;for(size_t j=objects.size();j>0;--j)if(!used[j-1]&&keys[j-1]==key){used[j-1]=true;result.push_back(objects[j-1]);found=true;break;}require(found,"Overlay hunt bucket exhausted");}std::reverse(result.begin(),result.end());so_assign(s,pool,result);
    }
    if(o.cid_set||o.hidden){auto pc=obs_attr(s,s.root,"pending_combat",false);if(truth(s,pc)&&truth(s,field(s,pc,"atk_combat_card",false))&&truth(s,field(s,pc,"_commit_nonce",false))){auto round=field(s,pc,"rounds_fought",false);auto number=round==missing?0:s.nodes[round].integer;if(o.round_set&&number==o.round){if(truth(s,field(s,pc,"atk_hidden_cc",false))){cp_erase(s,pc,"atk_hidden_cc");so_put(s,pc,"atk_combat_card",o.cid_set?rr_string(s,o.cid):rr_scalar(s,WOTR_NONE));}else if(o.cid_set)so_put(s,pc,"atk_combat_card",rr_string(s,o.cid));}}
        slot=public_slot();if(slot.first!=missing){auto xs=child_edges(s,slot.first);xs[slot.second]=o.cid_set?rr_string(s,o.cid):rr_scalar(s,WOTR_NONE);so_assign(s,slot.first,xs);}
    }
}
std::unique_ptr<Snapshot> so_root_clone(const Snapshot& source,int root){
    require(root==0||root==1,"Search overlay root must be FP or SH");auto result=action_clone(source);auto& s=*result;auto original_pc=obs_attr(source,source.root,"pending_combat",false);
    bool blind=truth(source,original_pc)&&truth(source,field(source,original_pc,"atk_combat_card",false))&&root!=cbb_side(source,original_pc,0)&&(truth(source,field(source,original_pc,"atk_hidden_cc",false))||text_is(source,field(source,original_pc,"phase",false),"defender_combat_card"));
    if(blind){try{auto cid=std::string(text(source,field(source,original_pc,"atk_combat_card")));auto public_dict=obs_attr(s,s.root,"discard_face_up",false);if(!ctx_none(s,public_dict)){auto es=child_edges(s,public_dict);for(size_t j=0;j<es.size();j+=2){auto xs=child_edges(s,es[j+1]);for(size_t k=xs.size();k>0;--k)if(text_is(s,xs[k-1],cid)){xs[k-1]=rr_string(s,"_committed_face_down_");set_edges(s,es[j+1],xs);rr_set_attr(s,s.root,"_commit_slot",cp_container(s,WOTR_TUPLE,{es[j],rr_scalar(s,WOTR_INT,k-1)}));break;}if(!ctx_none(s,obs_attr(s,s.root,"_commit_slot",false)))break;}}
        auto pc=obs_attr(s,s.root,"pending_combat",false);if(!ctx_none(s,pc))cp_put(s,pc,"_commit_nonce",rr_scalar(s,WOTR_BOOL,1));}catch(const std::exception&){/* Source root blinding catches errors after preserving any partial edits. */}}
    return result;
}
// Preparation is not an action executor. Missing hand cards remain strict in
// learned-policy mode; legacy mode conditions only the selected card edge.
int so_prepare(so_Overlay& o,Snapshot& s,void* rng,int actor,const Snapshot* action,bool learned){
    so_patch(o,s,rng);if(!action||actor!=1-o.root)return 0;auto row=child_edges(*action,action->root);require(row.size()==3,"Overlay action descriptor requires triple");auto kind=row[1],params=row[2];const auto& a=*action;
    bool consumes=text_is(a,kind,"play_event")||text_is(a,kind,"discard_card")||(text_is(a,kind,"combat_card_play")&&!truth(a,field(a,params,"from_table",false)));
    if(consumes){auto id=field(a,params,"card_id",false);if(truth(a,id)&&so_ok(a,id)){auto cid=std::string(text(a,id));auto& h=o.hand[so_deck(cid)];if(std::find(h.begin(),h.end(),cid)==h.end()){
        if(learned)throw std::invalid_argument("Selected card is absent from consistent sampled hand");if(!so_require(o,rng,cid))throw std::invalid_argument("Hidden sample cannot supply selected card: "+cid);so_patch(o,s,rng);return 1;
    }}}return 0;
}
