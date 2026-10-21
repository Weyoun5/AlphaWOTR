// Native MCTS controller: a faithful C++ port of the frozen ai.py search controller
// (SHARED_ABI.md, milestone 1).  Appended to the END of snapshot.cpp by
// controller/prepare.py, i.e. outside the engine's anonymous namespace; every
// engine internal (action_clone, native_full_actions, project_native_ai_actions,
// native_menu_view, af_encode, encode_state, apply_native_random_descriptor,
// sr_resume_preaction, sd_* turn driver, so_* overlay, hf_Scope hunt forcing,
// check_victory, native_pending_actor, native_next_actor, ...) is reached through
// the unnamed namespace's implicit using-directive.
//
// Source anchors (static_r3 ai.py) are quoted next to each port.  Unsupported
// branches (SHARED_ABI.md section 0) throw sc::Refusal("unported: <branch>").
#include <thread>
#include <array>
#include <list>
#include <numeric>
#include "wotr_own_value.h"
#include "wotr_reply_selector.h"
#include <condition_variable>
#include <optional>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cctype>
#include <unordered_set>

#ifdef WOTR_NATIVE_FM_PROFILE_AVAILABLE
// Clock for the native_full_actions callee breakdown of profiling probe boxes (controller/make_profile_probe.py): the
// probe fragment forward-declares fm_now_us inside the engine's unnamed namespace, which this reopening completes.
namespace { uint64_t fm_now_us(){return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());} }
#endif

extern "C" {
struct WotrSearchConfig {
    uint32_t struct_size;
    int32_t  root_side;
    int64_t  num_simulations;
    int32_t  sims_menu_scale; int64_t sims_per_child; int64_t sims_floor;
    double   c_puct;
    int64_t  leaf_batch;
    int64_t  turn_horizon;
    int64_t  max_children;
    int64_t  chance_k_max;
    int64_t  kernel_k_max;
    int64_t  chance_draws;
    int32_t  free_cands; int64_t free_cands_n; int64_t free_force;
    int32_t  opponent_policy_mode;
    double   opponent_policy_epsilon;
    int32_t  conf_leaf; int32_t conf_track; int32_t conf_ref;
    int32_t  determinize;
    int32_t  add_noise;
    int32_t  overlay_assert;
    int32_t  reward_contract;
    int32_t  fp_profile;
    int32_t  sh_profile;
    double   phi_rings_rho;
    int32_t  trace;
};
struct WotrSearchRequest {
    uint32_t struct_size;
    uint64_t request_id;
    int32_t  operation;
    int32_t  model_side;
    int32_t  purpose;
    uint64_t batch;
    uint64_t rows;
    const uint64_t* leaf_ids;
    const float*    observations;
    const float*    actions;
    const int64_t*  regions;
    const uint64_t* offsets;
};
struct WotrSearchReply {
    uint32_t struct_size;
    uint64_t request_id;
    const float*  priors;
    const double* values;
    const double* conf;
    const double* mixed;
};
struct WotrSearchRoot {
    uint32_t struct_size;
    uint64_t children;
    const uint64_t* visits; const uint64_t* term_visits; const uint64_t* forced_visits;
    const double* value_sum; const double* value_sq; const double* conf_sum; const double* phi_sum;
    const float*  priors_clean;
    const int8_t* proven;
    const double* proven_value;
    const int32_t* side;
    int32_t root_side;
    uint64_t simulations_run;
    uint64_t budget;
};
/* SHARED_ABI 9.6: structural facts of a detached subtree (32 bytes, natural alignment). */
struct WotrSearchSessionInfo {
    uint32_t struct_size; int32_t side; int32_t is_chance; int32_t has_state;
    uint64_t children; uint64_t visit_count;
};
}

namespace sc {
constexpr size_t OBS_DIM=4647,ACT_DIM=157;
constexpr int OP_PV=1,OP_PVM=2,OP_EVAL=3,OP_DIST=4;
constexpr int PURPOSE_ROOT=1,PURPOSE_LEAF=2,PURPOSE_OPPONENT=3,PURPOSE_FLUSH_POLICY=4,PURPOSE_FLUSH_ROOT=5,PURPOSE_EVAL=6,PURPOSE_TURN=7,PURPOSE_DIST=8;
constexpr int STATUS_REQUEST=0,STATUS_COMPLETE=1,STATUS_ROOT_PRIORS=2;
constexpr uint64_t NO_ID=std::numeric_limits<uint64_t>::max();

struct Refusal:public std::runtime_error{explicit Refusal(const std::string& branch):std::runtime_error("unported: "+branch){}};
struct Cancelled{};
[[noreturn]] void unported(const std::string& branch){throw Refusal(branch);}
[[noreturn]] void fail(const std::string& what){throw std::runtime_error(what);}

using RngPtr=std::unique_ptr<void,void(*)(void*)>;
RngPtr rng_clone(const void* p){RngPtr r(wotr_random_clone(p),wotr_random_destroy);if(!r)throw std::invalid_argument(wotr_random_last_error());return r;}

// ---------------------------------------------------------------- graph helpers
uint64_t graph_copy(Snapshot& out,const Snapshot& src,uint64_t node,std::unordered_map<uint64_t,uint64_t>& memo){
    if(node==missing)return rr_scalar(out,WOTR_NONE);
    auto found=memo.find(node);if(found!=memo.end())return found->second;
    auto v=src.nodes[node];const auto result=out.nodes.size();memo[node]=result;out.nodes.push_back(v);
    out.nodes[result].offset=0;out.nodes[result].count=0;
    if(v.tag==WOTR_STRING||v.tag==WOTR_BYTES||v.tag==WOTR_BIGINT){
        const auto offset=out.bytes.size();out.bytes.insert(out.bytes.end(),src.bytes.begin()+static_cast<ptrdiff_t>(v.offset),src.bytes.begin()+static_cast<ptrdiff_t>(v.offset+v.count));
        out.nodes[result].offset=offset;out.nodes[result].count=v.count;
    }else if(v.tag>=WOTR_LIST&&v.tag<=WOTR_ENUM){
        std::vector<uint64_t> children;for(auto c:child_edges(src,node))children.push_back(graph_copy(out,src,c,memo));
        set_edges(out,result,children);
        if(v.flags==1)out.nodes[result].integer=static_cast<int64_t>(graph_copy(out,src,static_cast<uint64_t>(v.integer),memo));
    }
    return result;
}

// Canonical bytes for a native policy key (menu_view row metadata index 6) such
// that byte equality == Python tuple equality of ai_policies.action_policy_key
// (numeric cross-type equality 1 == 1.0 == True folded to one integer form).
void append_u64(std::string& out,uint64_t x){char b[8];std::memcpy(b,&x,8);out.append(b,8);}
void key_bytes(const Snapshot& v,uint64_t n,std::string& out){
    const auto& node=v.nodes[n];
    switch(node.tag){
    case WOTR_NONE:out+='N';return;
    case WOTR_BOOL:case WOTR_INT:out+='I';append_u64(out,static_cast<uint64_t>(node.integer));return;
    case WOTR_FLOAT:{
        const double d=node.real;
        if(std::isfinite(d)&&std::trunc(d)==d&&d>=-9223372036854775808.0&&d<9223372036854775808.0){out+='I';append_u64(out,static_cast<uint64_t>(static_cast<int64_t>(d)));return;}
        uint64_t bits;std::memcpy(&bits,&d,8);out+='F';append_u64(out,bits);return;
    }
    case WOTR_STRING:case WOTR_BYTES:case WOTR_BIGINT:{
        out+=node.tag==WOTR_STRING?'S':node.tag==WOTR_BYTES?'B':'G';append_u64(out,node.count);
        out.append(reinterpret_cast<const char*>(v.bytes.data()+node.offset),static_cast<size_t>(node.count));return;
    }
    case WOTR_ENUM:{auto xs=child_edges(v,n);out+='E';key_bytes(v,xs[0],out);key_bytes(v,xs[1],out);return;}
    case WOTR_TUPLE:{auto xs=child_edges(v,n);out+='T';append_u64(out,xs.size());for(auto c:xs)key_bytes(v,c,out);return;}
    case WOTR_FROZENSET:{
        auto xs=child_edges(v,n);std::vector<std::string> parts;for(auto c:xs){std::string p;key_bytes(v,c,p);parts.push_back(std::move(p));}
        std::sort(parts.begin(),parts.end());out+='Z';append_u64(out,parts.size());for(auto& p:parts)out+=p;return;
    }
    default:throw std::invalid_argument("Native policy key has unsupported value type");
    }
}

// Metadata Snapshot builder for the ABI exports (decoded by snapshot.unpack_metadata).
struct Meta {
    am_Builder b;
    uint64_t none(){return rr_scalar(b.out,WOTR_NONE);}
    uint64_t boolean(bool x){return rr_scalar(b.out,WOTR_BOOL,x?1:0);}
    uint64_t integer(int64_t x){return b.integer(x);}
    uint64_t real(double x){return rr_scalar(b.out,WOTR_FLOAT,0,x);}
    uint64_t str(std::string_view s){return b.str(s);}
    uint64_t bytes(const std::vector<uint8_t>& data){
        const auto offset=b.out.bytes.size();b.out.bytes.insert(b.out.bytes.end(),data.begin(),data.end());
        const auto node=b.out.nodes.size();b.out.nodes.push_back({WOTR_BYTES,0,0,0,offset,data.size()});return node;
    }
    uint64_t list(const std::vector<uint64_t>& xs){return b.container(WOTR_LIST,xs);}
    uint64_t tuple(const std::vector<uint64_t>& xs){return b.container(WOTR_TUPLE,xs);}
    uint64_t dict(const std::vector<std::pair<std::string,uint64_t>>& kv){
        std::vector<uint64_t> xs;for(const auto& item:kv){xs.push_back(str(item.first));xs.push_back(item.second);}return b.container(WOTR_DICT,xs);
    }
    uint64_t dict_nodes(const std::vector<std::pair<uint64_t,uint64_t>>& kv){std::vector<uint64_t> xs;for(const auto& item:kv){xs.push_back(item.first);xs.push_back(item.second);}return b.container(WOTR_DICT,xs);}
    uint64_t enum_member(const char* type,const std::string& member){return b.container(WOTR_ENUM,{str(type),str(member)});}
    uint64_t copy(const Snapshot& src,uint64_t node){std::unordered_map<uint64_t,uint64_t> memo;return graph_copy(b.out,src,node,memo);}
    std::unique_ptr<Snapshot> finish(uint64_t root){b.out.root=root;return std::make_unique<Snapshot>(std::move(b.out));}
};

// ---------------------------------------------------------------- direct policy-key serializer
// Produces bytes IDENTICAL to key_bytes(<row key of native_menu_view(menu,tables)>), i.e. the composition
// mv_Builder::key -> mv_Builder::freeze -> key_bytes, without building the intermediate frozen graph.
// (sm_Builder::raw copies a scalar/enum/frozenset subtree unchanged, so key_bytes over the copy equals
// key_bytes over the original node.)  WOTR_NATIVE_KEY_CHECK=1 cross-checks every row against the old path.
// Dictionary entries in mv_Builder order: stringified key (sm_Builder::string), stable by position.  STRING keys
// (the only kind real descriptors carry) are viewed in place; other key kinds keep the stringified storage.
struct KeyEntry{uint64_t k=0,v=0;uint32_t pos=0;bool is_text=true;std::string storage;};
// Real descriptors use string keys. Keep their borrowed node IDs in trivial
// inline storage; mixed keys still use the original owning-string path.
struct TextKeyEntry{uint64_t k,v;uint32_t pos;};
struct KeyEntries{
    static constexpr size_t inline_capacity=16;
    std::array<TextKeyEntry,inline_capacity> inline_text;
    std::vector<TextKeyEntry> spill_text;
    std::vector<KeyEntry> generic;
    size_t text_count=0;bool text_only=true,spilled=false;
    size_t size()const{return text_only?text_count:generic.size();}
    TextKeyEntry* text_data(){return spilled?spill_text.data():inline_text.data();}
    const TextKeyEntry* text_data()const{return spilled?spill_text.data():inline_text.data();}
    std::string_view key(const Snapshot& s,size_t i)const{
        if(text_only)return text(s,text_data()[i].k);
        const auto& e=generic[i];return e.is_text?text(s,e.k):std::string_view(e.storage);
    }
    uint64_t value(size_t i)const{return text_only?text_data()[i].v:generic[i].v;}
};
void dict_entries(const Snapshot& s,uint64_t dict,sm_Builder& b,bool skip_card_name,bool skip_idx,KeyEntries& entries){
    const auto& d=typed(s,dict,WOTR_DICT,"Search projection requires dictionary");
    entries.text_count=0;entries.text_only=true;entries.spilled=false;
    entries.spill_text.clear();entries.generic.clear();
    // Inspect tags only. A mixed dictionary then follows the old conversion
    // and error order, including keys that stringify to the same text.
    for(uint64_t i=0;i<d.count;i+=2)if(s.nodes[s.edges[d.offset+i]].tag!=WOTR_STRING){entries.text_only=false;break;}
    if(entries.text_only){
        entries.spilled=d.count/2>KeyEntries::inline_capacity;
        if(entries.spilled)entries.spill_text.reserve(static_cast<size_t>(d.count/2));
        for(uint64_t i=0;i<d.count;i+=2){
            const auto k=s.edges[d.offset+i];
            if(skip_card_name&&text_is(s,k,"card_name"))continue;if(skip_idx&&text_is(s,k,"idx"))continue;
            const TextKeyEntry e{k,s.edges[d.offset+i+1],static_cast<uint32_t>(i)};
            if(entries.spilled)entries.spill_text.push_back(e);else entries.inline_text[entries.text_count]=e;
            ++entries.text_count;
        }
        auto* first=entries.text_data();
        if(entries.text_count>1)std::sort(first,first+entries.text_count,[&](const TextKeyEntry& x,const TextKeyEntry& y){
            const auto vx=text(s,x.k),vy=text(s,y.k);if(vx!=vy)return vx<vy;return x.pos<y.pos;
        });
        return;
    }
    auto& generic=entries.generic;generic.reserve(d.count/2);
    for(uint64_t i=0;i<d.count;i+=2){
        const auto k=s.edges[d.offset+i];
        if(skip_card_name&&text_is(s,k,"card_name"))continue;if(skip_idx&&text_is(s,k,"idx"))continue;
        KeyEntry e;e.k=k;e.v=s.edges[d.offset+i+1];e.pos=static_cast<uint32_t>(i);e.is_text=s.nodes[k].tag==WOTR_STRING;if(!e.is_text)e.storage=b.string(k);
        generic.push_back(std::move(e));
    }
    auto view=[&](const KeyEntry& e){return e.is_text?text(s,e.k):std::string_view(e.storage);};
    std::sort(generic.begin(),generic.end(),[&](const KeyEntry& x,const KeyEntry& y){const auto vx=view(x),vy=view(y);if(vx!=vy)return vx<vy;return x.pos<y.pos;});
}
void freeze_bytes(const Snapshot& s,uint64_t n,sm_Builder& b,std::string& out,size_t depth=0){
    require(depth<200,"Native menu key is cyclic or nested too deeply");
    if(ctx_none(s,n)){out+='N';return;}
    const auto tag=s.nodes[n].tag;
    if(tag==WOTR_DICT){
        KeyEntries entries;dict_entries(s,n,b,false,false,entries);
        out+='T';append_u64(out,entries.size());std::string_view previous;bool has_previous=false;
        for(size_t i=0;i<entries.size();++i){
            const std::string_view key=entries.key(s,i);
            require(!has_previous||previous!=key,"Native menu key has colliding stringified dictionary keys");previous=key;has_previous=true;
            out+='T';append_u64(out,2);out+='S';append_u64(out,key.size());out.append(key.data(),key.size());freeze_bytes(s,entries.value(i),b,out,depth+1);
        }
        return;
    }
    if(tag==WOTR_LIST||tag==WOTR_TUPLE){
        const auto& nd=s.nodes[n];out+='T';append_u64(out,nd.count);
        for(uint64_t i=0;i<nd.count;++i)freeze_bytes(s,s.edges[nd.offset+i],b,out,depth+1);return;
    }
    if(tag==WOTR_SET){
        auto values=b.items(n);
        std::vector<std::pair<std::string,uint64_t>> ordered;for(auto v:values)ordered.emplace_back(ap_repr(ap_freeze(s,v)),v);
        std::stable_sort(ordered.begin(),ordered.end(),[](const auto& x,const auto& y){return x.first<y.first;});
        out+='T';append_u64(out,ordered.size());for(auto& entry:ordered)freeze_bytes(s,entry.second,b,out,depth+1);return;
    }
    require(tag==WOTR_NONE||tag==WOTR_BOOL||tag==WOTR_INT||tag==WOTR_FLOAT||tag==WOTR_BIGINT||tag==WOTR_STRING||tag==WOTR_BYTES||tag==WOTR_ENUM||tag==WOTR_FROZENSET,"Native menu key has unsupported choice type");
    key_bytes(s,n,out);
}
void direct_row_key(const Snapshot& s,uint64_t rownode,sm_Builder& b,std::string& out){
    const auto& t=s.nodes[rownode];require(t.tag==WOTR_TUPLE&&t.count==3,"Native menu row must be descriptor tuple");
    const auto die=s.edges[t.offset],kind=s.edges[t.offset+1],params=s.edges[t.offset+2];
    typed(s,die,WOTR_INT,"Native menu die index must be integer");typed(s,kind,WOTR_STRING,"Native menu action type must be string");typed(s,params,WOTR_DICT,"Native menu params must be dictionary");
    const bool choices=field(s,params,"choices",false)!=missing;
    KeyEntries entries;dict_entries(s,params,b,true,choices,entries);
    out.reserve(out.size()+192);
    out+='T';append_u64(out,3);
    {const auto kt=text(s,kind);out+='S';append_u64(out,kt.size());out.append(kt.data(),kt.size());}
    out+='I';append_u64(out,static_cast<uint64_t>(s.nodes[die].integer));
    out+='T';append_u64(out,entries.size());std::string_view previous;bool has_previous=false;
    for(size_t i=0;i<entries.size();++i){
        const std::string_view key=entries.key(s,i);
        require(!has_previous||previous!=key,"Native menu key has colliding stringified dictionary keys");previous=key;has_previous=true;
        out+='T';append_u64(out,2);out+='S';append_u64(out,key.size());out.append(key.data(),key.size());freeze_bytes(s,entries.value(i),b,out);
    }
}

// ---------------------------------------------------------------- tree types
struct Controller;
struct RowMeta{std::string kind;int64_t die=0;int64_t category=0;bool is_free=false;};
struct ActionRef;
// A policy menu: the projected descriptor Snapshot plus per-row metadata.  Policy keys are computed
// lazily (first opponent-side consumer) for the whole menu at once; children/edges reference the menu
// instead of copying key strings.  Menus are immutable after construction apart from the lazy keys.
struct Menu:std::enable_shared_from_this<Menu>{
    // ctl is only a profiling/key-check hook (ensure_keys); a menu retained in a 9.6 session outlives its controller,
    // so detach nulls it and the adopting controller re-points it (hence mutable through shared_ptr<const Menu>).
    std::shared_ptr<const Snapshot> snap;std::vector<RowMeta> rows;mutable Controller* ctl=nullptr;
    mutable std::vector<std::string> keys;mutable bool keys_ready=false;
    size_t size()const{return rows.size();}
    uint64_t node(size_t i)const{const auto& list=snap->nodes[snap->root];return snap->edges[list.offset+i];}
    void ensure_keys()const;
    const std::string& key(size_t i)const{if(!keys_ready)ensure_keys();return keys[i];}
    ActionRef ref(size_t i)const;
};
struct ActionRef {
    std::shared_ptr<const Menu> owner;int64_t row=-1;
    bool has()const{return owner!=nullptr;}
    const Snapshot& snap()const{return *owner->snap;}
    uint64_t node()const{return owner->node(static_cast<size_t>(row));}
    const std::string& kind()const{return owner->rows[static_cast<size_t>(row)].kind;}
    uint64_t params()const{const auto& t=owner->snap->nodes[node()];return owner->snap->edges[t.offset+2];}
    const std::string& key()const{return owner->key(static_cast<size_t>(row));}
};
inline ActionRef Menu::ref(size_t i)const{return ActionRef{shared_from_this(),static_cast<int64_t>(i)};}
struct Policy{std::shared_ptr<Menu> menu;std::string forced;std::vector<std::string> reasons;};

struct Node;
// MC4 kernel world bank (ai.py 14385-14434): per sampled committed-card identity, the chance node's
// (outcomes, kernel_cover, drops).  Python aliases the live outcomes list; here the active bank is re-saved
// before every switch, which is observably identical.
struct World{std::vector<Node*> outcomes;bool has_cover=false;double cover=0.0;uint64_t drops=0;};
struct Edge{double prior=0.0;Node* child=nullptr;ActionRef payload;const std::string& key()const{return payload.key();}};
struct SvHash{size_t operator()(std::string_view v)const noexcept{return std::hash<std::string_view>{}(v);}};
// index keys are views into Menu::keys storage, which is filled once and kept alive by the edge payloads' owners.
struct Catalog{std::vector<Edge> records;std::vector<Node*> physical;std::unordered_map<std::string_view,size_t,SvHash> index;bool disabled=false;};

struct Node {
    uint64_t own_reply_token=0;
    uint64_t id=0;Node* parent=nullptr;int side=0;ActionRef action;double prior=0.0;
    // Work package LB2 (leaf_batch, 2026-09-13): 1-based creation index of a NO_ID node (a _sample_local scratch or an over-cap
    // round-sampling transient; 0 for tree nodes).  Its request correlation id is NO_ID - scratch_seq (leaf_corr): unique per
    // controller, never a tree node id, never NO_ID - so two such leaves deferred into ONE PVM/EVAL request stay distinct rows
    // (the bridge refuses duplicate leaf ids; Python keys its pending records by the distinct MCTSNode objects).
    uint64_t scratch_seq=0;
    std::vector<Node*> children;
    uint64_t visit_count=0;double value_sum=0.0,value_sq=0.0;uint64_t term_visits=0;
    bool is_terminal=false;double terminal_value=0.0;
    bool is_chance=false;bool has_outcomes=false;std::vector<Node*> outcomes;
    bool has_sig=false;std::vector<uint8_t> out_sig;
    uint64_t drops=0;
    int proven=0;double proven_value=0.0;          // 0 none, 1 'W', -1 'L'
    bool has_cover=false;double kernel_cover=0.0;
    bool round_sampling=false;
    // kernel_sets None / kernel_key None when the flags are false.  The bank list is INSERTION-ORDERED like the
    // Python dict (first registration order); the None key is stored as the impossible "\0" string (bank_key).
    bool has_kernel_sets=false;std::vector<std::pair<std::string,World>> kernel_sets;bool has_kernel_key=false;std::string kernel_key;
    uint64_t forced_visits=0;double conf_sum=0.0,phi_sum=0.0;
    std::string forced_policy_reason;
    std::unique_ptr<Snapshot> state;
    std::unique_ptr<Catalog> catalog;
    std::shared_ptr<const Snapshot> action_row;     // cached (die, type, params) tuple graph
    double q_value()const{return visit_count>0?value_sum/static_cast<double>(visit_count):0.0;}
};
inline World* find_world(Node* n,const std::string& key){for(auto& bank:n->kernel_sets)if(bank.first==key)return &bank.second;return nullptr;}
inline World& world_slot(Node* n,const std::string& key){if(World* w=find_world(n,key))return *w;n->kernel_sets.push_back({key,World{}});return n->kernel_sets.back().second;}   // dict __setitem__: existing key keeps its position
// SHARED_ABI 9.6: a detached subtree (ai.py 17843-17859 _stash_tree).  `store->nodes` owns every node of the subtree in
// the old arena's creation order (root first); states, children, chance banks/kernel_sets, statistics, priors and edge
// catalogs travel with the nodes.  Menus are shared_ptr-owned by the ActionRefs/edges, so nothing else is copied.
// The store is shared with the controller the subtree came from: its root view / tree export / observer events keep
// reading the retained nodes after the detach (the source's post-stash tree: chosen child with parent None, still listed
// in root.children) whatever the order of session destroy and controller reads.  `adopted` is set when a reusing
// controller moved the nodes into its own arena; from then on the old controller's views are closed.
struct SessionStore{std::vector<std::unique_ptr<Node>> nodes;bool adopted=false;};
struct Session{std::shared_ptr<SessionStore> store;Node* root=nullptr;};

struct PendingRec {
    Node* node=nullptr;std::shared_ptr<Menu> menu;int net_side=0;
    std::vector<float> vec,ae;std::vector<int64_t> ri;
    bool opp=false;std::vector<float> root_vec;
    bool has_shp_node=false;double shp_node=0.0,shp_root=0.0;
    int64_t max_children=-1;
    std::vector<std::vector<Node*>> paths;
    std::vector<float> priors;double v=0.0,conf_v=1.0;std::vector<double> mixed;
    double v_root=0.0,conf_root=1.0;
};
struct PendingEvals{std::vector<std::unique_ptr<PendingRec>> recs;std::unordered_map<Node*,PendingRec*> by_node;PendingRec* last=nullptr;};

struct RequestData {
    WotrSearchRequest view{};
    std::vector<uint64_t> leaf_ids;std::vector<float> obs,actions;std::vector<int64_t> regions;std::vector<uint64_t> offsets;
    void finalize(int op,int side,int purpose){
        view.struct_size=sizeof(WotrSearchRequest);view.operation=op;view.model_side=side;view.purpose=purpose;
        view.batch=leaf_ids.size();view.rows=regions.size();
        view.leaf_ids=leaf_ids.data();view.observations=obs.data();view.actions=actions.data();view.regions=regions.data();view.offsets=offsets.data();
    }
};
struct ReplyData{std::vector<float> priors;std::vector<double> values,conf,mixed;};

struct Event{std::string kind;std::vector<int64_t> i;std::vector<double> d;std::vector<uint64_t> ids;bool leaf_none=false;};
// One element of a kernel outcome tuple (combat_dist obs), with Python equality (bool/int/float cross-equal).
struct Atom{int kind=0;int64_t i=0;double d=0.0;std::string s;std::vector<Atom> children;   // 0 None, 1 int/bool, 2 float, 3 str, 4 tuple/list, 5 frozenset/set, 6 bytes
    bool operator==(const Atom& o)const{
        const bool an=kind==1||kind==2,bn=o.kind==1||o.kind==2;
        if(an&&bn){if(kind==1&&o.kind==1)return i==o.i;return (kind==1?static_cast<double>(i):d)==(o.kind==1?static_cast<double>(o.i):o.d);}
        if(kind!=o.kind)return false;
        if(kind==0)return true;if(kind==3||kind==6)return s==o.s;
        if(children.size()!=o.children.size())return false;
        if(kind==4){for(size_t k=0;k<children.size();++k)if(children[k]!=o.children[k])return false;return true;}
        for(const auto& c:children){bool found=false;for(const auto& oc:o.children)if(c==oc){found=true;break;}if(!found)return false;}   // set equality (elements unique)
        return true;
    }
    bool operator!=(const Atom& o)const{return !(*this==o);}
    bool truthy()const{return kind==1?i!=0:kind==2?d!=0.0:(kind==3||kind==6)?!s.empty():(kind==4||kind==5)?!children.empty():false;}
    int64_t as_int_or0()const{return kind==1?i:kind==2?static_cast<int64_t>(d):0;}   // int(x or 0) on the integer roster/hit fields
};
using Obs=std::vector<Atom>;
struct PlanItem{int64_t hits=0;int64_t ones=-1;bool is_tuple=false;};
// Decoded SHARED_ABI 9.5 reply.
struct DistReply{std::vector<Obs> outcomes;std::vector<double> masses;std::vector<std::optional<std::vector<PlanItem>>> witness;std::vector<std::string> plan_reprs;bool conf=false;std::optional<std::string> wctk_side;std::string ctx_repr;};
// CombatForecastObserver call points (SHARED_ABI section 9.3): kind 1 = hidden_commit(mixture), kind 2 = card_policy(node, menu, probabilities).
// Work package RA (native_runaway, 2026-09-11): a card_policy event keeps the export's per-row data (the action kind, interned per
// controller; the card_id node copied for the rows that carry one) instead of the sampled Menu, so the draw menu dies with the draw
// (the frozen observer keeps no action list either).
struct ObsEvent{int kind=0;std::vector<std::pair<std::string,int>> mixture;Node* node=nullptr;std::vector<uint32_t> kinds;std::vector<std::pair<uint32_t,std::shared_ptr<const Snapshot>>> cids;std::vector<double> probabilities;};
struct HighDrop{std::string cls;double dropped=0.0;int64_t support=0,kept=0;ActionRef action;int64_t eyes=0;bool has_pool=false;std::vector<std::string> pool;bool kernel=false;int64_t support_raw=0;std::string ctx_repr;};

struct Counters {
    std::map<std::string,int64_t> overlay,kernel,search,edge,lattice,kernel_fallback;
    std::vector<double> dropped_mass;std::vector<int64_t> support;std::map<std::string,std::vector<double>> by_class;std::vector<HighDrop> high_drop;
    std::map<int64_t,std::vector<double>> dropped_at;                      // KERNEL_STATS['dropped_at'] {16,24,32}
    Meta detail;std::vector<std::pair<std::string,uint64_t>> fallback_detail;   // KERNEL_STATS['fallback_detail']: (reason, detail graph node in `detail`)
    Counters(){
        for(const char* k:{"samples","sample_error","patches","patch_error","cond_swaps","require_fail","mask_empty","mask_error","lazy_event","lazy_discard","lazy_combat","lazy_no_template","lazy_rejected","det_draw_chance","assert_checks","commit_blind","learned_policy_draws","no_action_samples"})overlay[k]=0;
        for(const char* k:{"combat_chance_nodes","kernel_nodes","opp_card_nodes","opp_card_fallback","hunt_class_applies","hunt_nodes","hunt_det","hunt_walks"})kernel[k]=0;
        for(const char* k:{"apply_fail","leaf_batches","leaf_batched","leaf_batch_fallback"})search[k]=0;
        for(const char* k:{"snapshots_created","selections","materialized","catalog_fallbacks"})edge[k]=0;
        for(const char* k:{"numeric_list_pruning_rows","borrowed_pruning_rows","catalog_block_skipped_rows","flat_projection_menus","flat_projection_rows","full_iteration"})lattice[k]=0;
    }
};

struct Victory{bool ended=false;int winner=-1;int kind=0;};
struct RollMs{std::vector<std::string> fp,sh;int64_t eyes=0;};
struct PV{std::vector<double> priors;double value=0.0;std::vector<double> mixed;};
struct HuntStep{bool fork=false;size_t options=0;double prob=0.0;std::unique_ptr<Snapshot> child;};

bool is_hunt_parent(std::string_view k){
    for(const char* x:{"fellowship_move","fellowship_reveal_place","hobbit_place","hunt_gollum_reveal","hunt_gollum_no_reveal","hunt_take_corruption","hunt_sacrifice_guide","hunt_sacrifice_random","hunt_guide_separate","choose_guide","card_option"})if(k==x)return true;
    return false;
}
bool is_kernel_parent(std::string_view k){return k=="combat_card_play"||k=="combat_card_none"||k=="combat_effect_choice";}
bool is_free_action(std::string_view k){return k=="elven_ring"||k=="use_token"||k=="dwarven_ring";}
int victory_kind(std::string_view k){if(k=="military")return 1;if(k=="ring_destroyed")return 2;if(k=="ring_corruption")return 3;return 0;}

// ---------------------------------------------------------------- controller
#include "own_reply_types.inc"
struct Controller {
    std::unique_ptr<OwnReplies> own;
    Policy own_public(Snapshot&,int,const Menu*,bool* retained_all=nullptr);
    std::optional<std::string> own_choose(const Snapshot&,int,Node*,const Menu*,const Policy* admitted=nullptr);
    Node* own_child(Node*);
    int64_t own_turn_action(const Snapshot&,int,const Snapshot&,const std::vector<int64_t>&);
    WotrSearchConfig cfg{};int root_side=0;bool conf_active=false,profiles_active=false;bool trace=false;
    const Snapshot* input=nullptr;const Snapshot *T=nullptr,*A=nullptr,*C=nullptr,*H=nullptr;
    RngPtr rng{nullptr,wotr_random_destroy};
    std::vector<std::unique_ptr<Node>> arena;Node* root=nullptr;
    double last_conf=1.0,last_phi=0.0;
    std::unique_ptr<so_Overlay> overlay;int overlay_opp=-1;
    PendingEvals* pending_box=nullptr;
    std::unordered_map<std::string,std::vector<double>> policy_cache;
    Counters counters;std::vector<Event> events;
    uint64_t simulations_run=0,budget=0;std::vector<float> priors_clean;std::vector<size_t> force_idx;
    std::shared_ptr<const Menu> end_turn_menu;bool key_check=false;bool root_blinded=false;
    bool reused_root=false;   // 9.6: root adopted from a Session (ai.py 17423-17426); search() skips clone/expansion/noise
    bool detached=false;      // 9.6: a root child's subtree was moved into a Session (once per controller)
    std::shared_ptr<SessionStore> detached_store;Node* detached_child=nullptr;   // shared with the Session: keeps the retained nodes readable here
    std::vector<std::unique_ptr<Node>> scratch_arena;   // _sample_local scratch nodes (+ their children): never linked into the tree, never exported
    uint64_t scratch_count=0;                            // LB2: NO_ID nodes created so far (Node::scratch_seq source)
    std::vector<ObsEvent> observer_events;
    std::vector<std::string> obs_kind_table;std::unordered_map<std::string,uint32_t> obs_kind_index;   // RA: interned action kinds of the card_policy events
    std::unordered_map<std::string,std::pair<int64_t,bool>> kind_meta;   // kind -> (category index, is_free): pure function of the category tables
    std::map<std::string,uint64_t> prof_rows;
    // root view storage
    std::vector<uint64_t> rv_visits,rv_term,rv_forced;std::vector<double> rv_sum,rv_sq,rv_conf,rv_phi,rv_pv;std::vector<int8_t> rv_proven;std::vector<int32_t> rv_side;
    // threading
    std::thread worker;std::mutex mu;std::condition_variable cv;
    enum class Phase{Idle,Running,RequestPending,RootPriorsPending,ReplyReady,Complete,Failed,Cancelled};Phase phase=Phase::Idle;
    std::string error;uint64_t request_counter=0;RequestData request;ReplyData reply;bool cancel=false;
    std::unique_ptr<Snapshot> pending_payload;DistReply dist_reply;          // op-4 (COMBAT_DIST) payload / decoded reply
    std::vector<double> rp_priors;std::vector<uint8_t> rp_free;             // 9.4 root-priors exchange buffers
    // ------------------------------------------------ optional phase profile (WOTR_NATIVE_SEARCH_PROFILE=1, read once at create).
    // Diagnostic only: microseconds + call counts per phase; exported under counters['profile'] when enabled.
    // Composite phases (turn_driver, hunt_walks, select_outcome, flush_total) CONTAIN leaf phases; await_reply is time blocked on Python.
    struct ProfAcc{uint64_t us=0,calls=0;};
    bool profiling=false;std::map<std::string,ProfAcc> prof;uint64_t prof_total_us=0;
    std::string apply_fault;   // WOTR_NATIVE_APPLY_FAULT=<action kind>: test-only fault injection for the apply diagnostics probe (controller/probe_failure_export.py); empty = inert
    struct Scoped{Controller* c;const char* k;std::chrono::steady_clock::time_point t0;
        Scoped(Controller* ctl,const char* key):c(ctl->profiling?ctl:nullptr),k(key){if(c)t0=std::chrono::steady_clock::now();}
        Scoped(const Scoped&)=delete;Scoped& operator=(const Scoped&)=delete;
        ~Scoped(){if(c){auto& a=c->prof[k];a.us+=static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-t0).count());a.calls+=1;}}};

    Controller(const void* root_state,const void* obs_tables,const void* act_tables,const void* cat_tables,const void* hunt_tables,const void* rng_in,const WotrSearchConfig* config,const WotrOwnValueConfig* own_config=nullptr){
        require(config,"Search config is null");require(config->struct_size==sizeof(WotrSearchConfig),"WotrSearchConfig struct_size mismatch");cfg=*config;
        input=&snapshot(root_state);T=&snapshot(obs_tables);A=&snapshot(act_tables);C=&snapshot(cat_tables);H=&snapshot(hunt_tables);
        require(rng_in,"Search RNG handle is null");rng=rng_clone(rng_in);
        require(cfg.root_side==0||cfg.root_side==1,"root_side must be 0 or 1");root_side=cfg.root_side;
        if(own_config){require(cfg.opponent_policy_mode==2,"Own-value entry requires mode 2");require(cfg.leaf_batch==1,"Own-value v0 requires sequential leaves");own=std::make_unique<OwnReplies>(*own_config);}
        else if(cfg.opponent_policy_mode!=1)unported("legacy_puct opponent policy mode");
        // 9.4: add_noise is accepted (root priors exchanged through status 2 / root_priors / set_root_priors).
        if(cfg.conf_ref!=0)unported("_CONF_REF other than 'abs'");
        if(cfg.reward_contract!=2)unported("reward contract other than predecision_profile_v1");
        if(cfg.determinize!=1)unported("non-determinized search (learned opponent policy requires joint hidden-hand sampling)");
        if(cfg.phi_rings_rho!=0.0)unported("_PHI_RINGS_RHO ring shaping");
        if(own){
            // Own-value replies evaluate only the searching player's model.
            // A live bridge may therefore omit the opponent's profile.
            const int root_profile=root_side==0?cfg.fp_profile:cfg.sh_profile;
            const int opponent_profile=root_side==0?cfg.sh_profile:cfg.fp_profile;
            require(root_profile>=1&&root_profile<=3,"own-value search requires a root profile (1 ring/corruption, 2 military, 3 mixed)");
            require(opponent_profile>=0&&opponent_profile<=3,"own-value opponent profile must be absent or valid");
        }else require(cfg.fp_profile>=1&&cfg.fp_profile<=3&&cfg.sh_profile>=1&&cfg.sh_profile<=3,"predecision_profile_v1 requires fp/sh profiles (1 ring/corruption, 2 military, 3 mixed)");
        require(cfg.leaf_batch>=1,"leaf_batch must be >= 1");require(cfg.num_simulations>=0,"num_simulations must be >= 0");
        require(std::isfinite(cfg.c_puct),"c_puct must be finite");require(cfg.chance_k_max>=0&&cfg.kernel_k_max>=0&&cfg.chance_draws>=0&&cfg.max_children>=0,"search width settings must be >= 0");
        require(cfg.free_cands_n>=0&&cfg.free_force>=0&&cfg.sims_per_child>=1&&cfg.sims_floor>=1,"free/sims settings out of domain");
        require(std::isfinite(cfg.opponent_policy_epsilon)&&cfg.opponent_policy_epsilon>0.0&&cfg.opponent_policy_epsilon<=1.0,"Opponent policy epsilon must be finite and in (0, 1]");
        conf_active=cfg.conf_leaf||cfg.conf_track;profiles_active=cfg.fp_profile!=0||cfg.sh_profile!=0;trace=cfg.trace!=0;
        {const char* env=std::getenv("WOTR_NATIVE_SEARCH_PROFILE");profiling=env&&std::string(env)=="1";}
        {const char* env=std::getenv("WOTR_NATIVE_KEY_CHECK");key_check=env&&std::string(env)=="1";}
        {const char* env=std::getenv("WOTR_NATIVE_APPLY_FAULT");if(env&&*env)apply_fault=env;}   // test-only: the first apply of this action kind throws (probe_failure_export.py); never set in parity or timed runs
#ifdef WOTR_NATIVE_AF_CHECK_AVAILABLE
        {const char* env=std::getenv("WOTR_NATIVE_AF_CHECK");if(env&&std::string(env)=="1")af_check_active=1;}   // engine action-feature cross-check (controller/engine_patches/action_features.inc.patch); process-sticky flag, never set in timed runs
#endif
#ifdef WOTR_NATIVE_FULLMENU_CHECK_AVAILABLE
        {const char* env=std::getenv("WOTR_NATIVE_FULLMENU_CHECK");if(env&&std::string(env)=="1")fm_check_active=1;}   // complete-menu cross-check (controller/engine_patches/full_action_menu.inc.patch); process-sticky, never in timed runs
#endif
#ifdef WOTR_NATIVE_OBS_CHECK_AVAILABLE
        {const char* env=std::getenv("WOTR_NATIVE_OBS_CHECK");if(env&&std::string(env)=="1")obs_check_active=1;}   // observation-encoder cross-check (controller/engine_patches/snapshot.cpp.patch); process-sticky, never in timed runs
#endif
#ifdef WOTR_NATIVE_AMBUILDER_CHECK_AVAILABLE
        {const char* env=std::getenv("WOTR_NATIVE_FULLMENU_CHECK");if(env&&std::string(env)=="1")am_check_active=1;}   // am_Builder::str intern self-check (controller/engine_patches/action_menu.inc.patch), same switch as the menu check
#endif
#ifdef WOTR_NATIVE_FM_PROFILE_AVAILABLE
        if(profiling)fm_prof_active=true;   // native_full_actions callee breakdown (profiling probe boxes only, controller/make_profile_probe.py)
#endif
        require(text_is(*T,field(*T,T->root,"schema"),"wotr_native_observation_tables_v1"),"observation tables schema mismatch");
        require(text_is(*A,field(*A,A->root,"schema"),"wotr_action_features_v52_157"),"action tables schema mismatch");
        typed(*C,field(*C,C->root,"category"),WOTR_DICT,"menu category tables mismatch");
        require(text_is(*H,field(*H,H->root,"schema"),"source_hunt_options_v1"),"hunt chance tables schema mismatch");
        attributes(*input,input->root,"game_state.GameState");
        // 9.2: blinded roots are accepted (so_root_clone at search start); 9.3: observer call points are exported, so
        // CombatForecastObserver root windows are accepted too.
        root_blinded=hidden_commit(*input,root_side);
        am_Builder b;const auto row=b.row(-1,"end_turn",{});b.out.root=b.container(WOTR_LIST,{row});
        end_turn_menu=make_menu(std::make_unique<Snapshot>(b.out));
    }
    ~Controller(){
        {std::lock_guard<std::mutex> lk(mu);cancel=true;}
        cv.notify_all();if(worker.joinable())worker.join();
        if(profiling){   // stderr fallback for drivers without a session; the structured form is counters['profile']
            std::fprintf(stderr,"[native-search-profile] search_total_us=%llu\n",static_cast<unsigned long long>(prof_total_us));
            for(const auto& item:prof)std::fprintf(stderr,"[native-search-profile] %-18s us=%12llu calls=%8llu\n",item.first.c_str(),static_cast<unsigned long long>(item.second.us),static_cast<unsigned long long>(item.second.calls));
        }
#ifdef WOTR_NATIVE_AF_CHECK_AVAILABLE
        if(af_check_active)std::fprintf(stderr,"[native-af-check] rows_checked=%llu mismatches=0 (a mismatch throws inside af_encode)\n",static_cast<unsigned long long>(af_check_rows_checked));
#endif
#ifdef WOTR_NATIVE_FULLMENU_CHECK_AVAILABLE
        if(fm_check_active)std::fprintf(stderr,"[native-fullmenu-check] menus_checked=%llu rows_checked=%llu mismatches=0 (a mismatch throws inside native_full_actions)\n",static_cast<unsigned long long>(fm_check_menus_checked),static_cast<unsigned long long>(fm_check_rows_checked));
#endif
#ifdef WOTR_NATIVE_OBS_CHECK_AVAILABLE
        if(obs_check_active)std::fprintf(stderr,"[native-obs-check] observations_checked=%llu mismatches=0 (a mismatch throws inside encode_state)\n",static_cast<unsigned long long>(obs_check_observations_checked));
#endif
#ifdef WOTR_NATIVE_AMBUILDER_CHECK_AVAILABLE
        if(am_check_active)std::fprintf(stderr,"[native-fullmenu-check] am_builder_interns_checked=%llu mismatches=0 (a mismatch throws inside am_Builder::str)\n",static_cast<unsigned long long>(am_check_interns));
#endif
#ifdef WOTR_NATIVE_FM_MEMO_AVAILABLE
        if(fm_check_active)std::fprintf(stderr,"[native-fullmenu-check] memo event_reuse=%llu attack_reuse=%llu attack_rows_rewritten=%llu (per-call memos of controller/engine_patches/full_action_menu.inc.patch; every reused menu passed the graph-identity pass)\n",static_cast<unsigned long long>(fm_memo_event_reuse),static_cast<unsigned long long>(fm_memo_attack_reuse),static_cast<unsigned long long>(fm_memo_attack_rows_rewritten));
#endif
#ifdef WOTR_NATIVE_QG_MEMO_AVAILABLE
        if(fm_check_active)std::fprintf(stderr,"[native-fullmenu-check] quiet_gate memo hits=%llu runs=%llu (per-call memo of controller/engine_patches/quiet_action_gate.inc.patch)\n",static_cast<unsigned long long>(qg_memo_hits),static_cast<unsigned long long>(qg_memo_runs));
#endif
#ifdef WOTR_NATIVE_FM_PROFILE_AVAILABLE
        if(fm_prof_active)for(const auto& item:fm_prof)std::fprintf(stderr,"[native-fm-profile] %-26s us=%12llu calls=%8llu rows=%10llu\n",item.first.c_str(),static_cast<unsigned long long>(item.second.us),static_cast<unsigned long long>(item.second.calls),static_cast<unsigned long long>(item.second.rows));
#endif
    }

    // ------------------------------------------------ suspension boundary
    ReplyData await_reply(RequestData&& r,int op,int side,int purpose){
        if(own&&(op==OP_PV||op==OP_PVM||op==OP_EVAL))require(side==root_side,"Own-value search attempted non-root numeric inference");
        std::unique_lock<std::mutex> lk(mu);
        r.finalize(op,side,purpose);r.view.request_id=++request_counter;request=std::move(r);
        if(trace){Event e;e.kind="INFERENCE";e.i={static_cast<int64_t>(request.view.request_id),op,side,purpose};e.ids=request.leaf_ids;events.push_back(std::move(e));}
        phase=Phase::RequestPending;cv.notify_all();
        {Scoped _sp(this,"await_reply");cv.wait(lk,[&]{return phase==Phase::ReplyReady||cancel;});}
        if(cancel)throw Cancelled{};
        phase=Phase::Running;ReplyData out=std::move(reply);reply=ReplyData{};return out;
    }
    void run(){
        try{
            const auto t0=std::chrono::steady_clock::now();search();
            if(profiling)prof_total_us=static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-t0).count());
            std::lock_guard<std::mutex> lk(mu);phase=Phase::Complete;}
        catch(const Cancelled&){std::lock_guard<std::mutex> lk(mu);phase=Phase::Cancelled;}
        catch(const std::exception& e){std::lock_guard<std::mutex> lk(mu);error=e.what();phase=Phase::Failed;}
        catch(...){std::lock_guard<std::mutex> lk(mu);error="unknown controller exception";phase=Phase::Failed;}
        cv.notify_all();
    }
    int advance(){
        std::unique_lock<std::mutex> lk(mu);
        if(phase==Phase::Complete)return STATUS_COMPLETE;
        if(phase==Phase::Failed)fail(error);
        if(phase==Phase::RequestPending)fail("An inference request is pending; submit its reply before advancing");
        if(phase==Phase::RootPriorsPending)fail("Root priors are pending; call set_root_priors before advancing");
        if(phase==Phase::Idle){phase=Phase::Running;worker=std::thread([this]{run();});}
        else if(phase==Phase::ReplyReady){cv.notify_all();}   // the worker flips ReplyReady -> Running itself (its wait predicate)
        else fail("Controller is not resumable in its current phase");
        cv.wait(lk,[&]{return phase==Phase::RequestPending||phase==Phase::RootPriorsPending||phase==Phase::Complete||phase==Phase::Failed||phase==Phase::Cancelled;});
        if(phase==Phase::RequestPending)return STATUS_REQUEST;
        if(phase==Phase::RootPriorsPending)return STATUS_ROOT_PRIORS;
        if(phase==Phase::Complete)return STATUS_COMPLETE;
        fail(error.empty()?"controller cancelled":error);
    }
    // 9.4 (worker side): hand the K stored child priors to Python, block until set_root_priors stored the noised ones.
    void root_noise_exchange(){
        std::unique_lock<std::mutex> lk(mu);
        rp_priors.clear();rp_free.clear();
        for(Node* c:root->children){rp_priors.push_back(c->prior);rp_free.push_back(c->action.has()&&c->action.owner->rows[static_cast<size_t>(c->action.row)].is_free?1:0);}
        phase=Phase::RootPriorsPending;cv.notify_all();
        cv.wait(lk,[&]{return phase==Phase::ReplyReady||cancel;});
        if(cancel)throw Cancelled{};
        phase=Phase::Running;
    }
    void view_root_priors(uint64_t* count,const double** priors,const uint8_t** is_free){
        require(count&&priors&&is_free,"root_priors outputs are required");
        std::lock_guard<std::mutex> lk(mu);require(phase==Phase::RootPriorsPending,"Root priors are not pending (status 2 only)");
        *count=rp_priors.size();*priors=rp_priors.data();*is_free=rp_free.data();
    }
    void set_root_priors(const double* priors,uint64_t count){
        std::lock_guard<std::mutex> lk(mu);require(phase==Phase::RootPriorsPending,"Root priors are not pending (status 2 only)");
        require(priors&&count==root->children.size(),"set_root_priors count must equal the number of root children");
        for(uint64_t i=0;i<count;++i)require(std::isfinite(priors[i]),"Root priors must be finite");
        for(uint64_t i=0;i<count;++i)root->children[i]->prior=priors[i];   // _apply_root_noise stores c.prior in place
        phase=Phase::ReplyReady;
    }
    void view_request(WotrSearchRequest* out){
        require(out&&out->struct_size==sizeof(WotrSearchRequest),"WotrSearchRequest struct_size mismatch");
        std::lock_guard<std::mutex> lk(mu);require(phase==Phase::RequestPending,"No inference request is pending");*out=request.view;
    }
    // 9.5: the COMBAT_DIST payload (valid while the op-4 request is pending) and its reply.
    std::unique_ptr<Snapshot> view_payload(){
        std::lock_guard<std::mutex> lk(mu);require(phase==Phase::RequestPending&&request.view.operation==OP_DIST&&pending_payload,"No COMBAT_DIST request is pending");
        return std::make_unique<Snapshot>(*pending_payload);
    }
    static Atom decode_atom(const Snapshot& r,uint64_t n,size_t depth=0){
        require(depth<64,"COMBAT_DIST reply: outcome element nested too deeply");
        Atom a;const auto& v=r.nodes[n];
        if(v.tag==WOTR_NONE){a.kind=0;return a;}
        if(v.tag==WOTR_BOOL||v.tag==WOTR_INT){a.kind=1;a.i=v.integer;return a;}
        if(v.tag==WOTR_FLOAT){a.kind=2;a.d=v.real;return a;}
        if(v.tag==WOTR_STRING){a.kind=3;a.s=std::string(text(r,n));return a;}
        if(v.tag==WOTR_BYTES){a.kind=6;a.s.assign(reinterpret_cast<const char*>(r.bytes.data()+v.offset),static_cast<size_t>(v.count));return a;}
        if(v.tag==WOTR_TUPLE||v.tag==WOTR_LIST||v.tag==WOTR_FROZENSET||v.tag==WOTR_SET){
            a.kind=(v.tag==WOTR_TUPLE||v.tag==WOTR_LIST)?4:5;for(auto c:child_edges(r,n))a.children.push_back(decode_atom(r,c,depth+1));return a;
        }
        throw std::invalid_argument("COMBAT_DIST reply: outcome element must be None/bool/int/float/str/bytes/tuple/frozenset");
    }
    uint64_t detail_atom(const Atom& a){
        if(a.kind==0)return counters.detail.none();if(a.kind==1)return counters.detail.integer(a.i);if(a.kind==2)return counters.detail.real(a.d);if(a.kind==3)return counters.detail.str(a.s);
        if(a.kind==6){std::vector<uint8_t> bytes(a.s.begin(),a.s.end());return counters.detail.bytes(bytes);}
        std::vector<uint64_t> xs;for(const auto& c:a.children)xs.push_back(detail_atom(c));return counters.detail.b.container(a.kind==4?WOTR_TUPLE:WOTR_FROZENSET,xs);
    }
    static DistReply decode_dist_reply(const Snapshot& r){
        DistReply out;typed(r,r.root,WOTR_DICT,"COMBAT_DIST reply must decode to a dict");
        auto get=[&](const char* key){const auto n=field(r,r.root,key,false);require(n!=missing,std::string("COMBAT_DIST reply lacks '"+std::string(key)+"'").c_str());return n;};
        auto seq=[&](uint64_t n,const char* what){const auto& v=r.nodes[n];require(v.tag==WOTR_LIST||v.tag==WOTR_TUPLE,what);return child_edges(r,n);};
        const auto outcomes=seq(get("outcomes"),"COMBAT_DIST reply outcomes must be a sequence"),masses=seq(get("masses"),"COMBAT_DIST reply masses must be a sequence");
        const auto witness=seq(get("witness"),"COMBAT_DIST reply witness must be a sequence"),reprs=seq(get("plan_reprs"),"COMBAT_DIST reply plan_reprs must be a sequence");
        require(outcomes.size()==masses.size()&&outcomes.size()==witness.size()&&outcomes.size()==reprs.size(),"COMBAT_DIST reply: outcomes/masses/witness/plan_reprs lengths differ");
        for(auto o:outcomes){const auto xs=seq(o,"COMBAT_DIST reply outcome must be a tuple");require(xs.size()==19,"COMBAT_DIST reply outcome must have 19 elements");Obs obs;for(auto x:xs)obs.push_back(decode_atom(r,x));out.outcomes.push_back(std::move(obs));}
        for(auto m:masses){const auto& v=r.nodes[m];double value;if(v.tag==WOTR_FLOAT)value=v.real;else if(v.tag==WOTR_INT||v.tag==WOTR_BOOL)value=static_cast<double>(v.integer);else throw std::invalid_argument("COMBAT_DIST reply mass must be numeric");require(std::isfinite(value),"COMBAT_DIST reply mass must be finite");out.masses.push_back(value);}
        for(auto w:witness){
            if(ctx_none(r,w)){out.witness.emplace_back(std::nullopt);continue;}
            std::vector<PlanItem> plan;
            for(auto item:seq(w,"COMBAT_DIST reply witness plan must be a sequence")){
                const auto& v=r.nodes[item];PlanItem p;
                if(v.tag==WOTR_INT||v.tag==WOTR_BOOL){p.hits=v.integer;p.ones=-1;p.is_tuple=false;}
                else{const auto pair=seq(item,"COMBAT_DIST reply plan item must be int or (int, int)");require(pair.size()==2,"COMBAT_DIST reply plan pair must have two elements");
                    for(auto q:pair){const auto& t=r.nodes[q];require(t.tag==WOTR_INT||t.tag==WOTR_BOOL,"COMBAT_DIST reply plan pair elements must be int");}
                    p.hits=r.nodes[pair[0]].integer;p.ones=r.nodes[pair[1]].integer;p.is_tuple=true;}
                plan.push_back(p);
            }
            out.witness.emplace_back(std::move(plan));
        }
        for(auto s:reprs)out.plan_reprs.emplace_back(text(r,s));
        const auto conf=get("conf");require(r.nodes[conf].tag==WOTR_BOOL,"COMBAT_DIST reply conf must be bool");out.conf=r.nodes[conf].integer!=0;
        const auto side=get("wctk_side");
        if(!ctx_none(r,side)){const auto name=std::string(text(r,side));require(name=="attacker"||name=="defender","COMBAT_DIST reply wctk_side must be None/attacker/defender");out.wctk_side=name;}
        out.ctx_repr=std::string(text(r,get("ctx_repr")));
        return out;
    }
    void submit_payload(uint64_t request_id,const void* reply_handle){
        require(reply_handle,"COMBAT_DIST reply handle is null");const Snapshot& r=snapshot(reply_handle);
        std::lock_guard<std::mutex> lk(mu);require(phase==Phase::RequestPending&&request.view.operation==OP_DIST,"No COMBAT_DIST request is pending");
        require(request_id==request.view.request_id,"Reply request_id does not match the pending request");
        DistReply decoded=decode_dist_reply(r);   // everything validated before any mutation
        dist_reply=std::move(decoded);reply=ReplyData{};phase=Phase::ReplyReady;
    }
    void submit(const WotrSearchReply* in){
        require(in&&in->struct_size==sizeof(WotrSearchReply),"WotrSearchReply struct_size mismatch");
        std::lock_guard<std::mutex> lk(mu);require(phase==Phase::RequestPending,"No inference request is pending");
        require(request.view.operation!=OP_DIST,"The pending request is COMBAT_DIST; use wotr_search_controller_submit_payload");
        require(in->request_id==request.view.request_id,"Reply request_id does not match the pending request");
        const auto B=static_cast<size_t>(request.view.batch),rows=static_cast<size_t>(request.view.rows);
        require(in->values&&in->conf,"Reply values/conf are required");
        ReplyData data;data.values.assign(in->values,in->values+B);data.conf.assign(in->conf,in->conf+B);
        for(auto v:data.values)require(std::isfinite(v),"Non-finite search value in a predecision reward contract");
        if(request.view.operation!=OP_EVAL){
            require(in->priors&&in->mixed,"Reply priors/mixed are required for policy requests");
            data.priors.assign(in->priors,in->priors+rows);data.mixed.assign(in->mixed,in->mixed+rows);
            for(auto p:data.priors)require(std::isfinite(p)&&p>=0.0f,"Invalid search policy weights in reply");
            for(auto m:data.mixed)require(std::isfinite(m)&&m>=0.0,"Invalid mixed opponent policy in reply");
        }
        reply=std::move(data);phase=Phase::ReplyReady;
    }
    void require_complete()const{require(phase==Phase::Complete,"Search is not complete");}

    // ------------------------------------------------ tracing
    void ev(const char* kind,std::initializer_list<int64_t> ints){if(!trace)return;Event e;e.kind=kind;e.i=ints;events.push_back(std::move(e));}
    // Trace-only RNG checkpoint (Mersenne index + word 0) after every RNG-consuming primitive; never in timed runs.
    void rng_mark(const char* what){if(!trace)return;std::vector<uint32_t> w(624);uint32_t index=0;int32_t has=0;uint64_t bits=0;if(wotr_random_export(rng.get(),w.data(),w.size(),&index,&has,&bits))throw std::invalid_argument(wotr_random_last_error());Event e;e.kind=std::string("RNG:")+what;e.i={static_cast<int64_t>(index),static_cast<int64_t>(w[0])};events.push_back(std::move(e));}

    // ------------------------------------------------ arena
    Node* new_node(std::unique_ptr<Snapshot> state,int side,Node* parent,ActionRef action,double prior){
        auto n=std::make_unique<Node>();n->state=std::move(state);n->side=side;n->parent=parent;n->action=std::move(action);n->prior=prior;
        Node* raw=n.get();
        if(parent&&parent->id==NO_ID){n->id=NO_ID;n->scratch_seq=++scratch_count;scratch_arena.push_back(std::move(n));return raw;}   // children of a _sample_local scratch node are discarded with it
        n->id=arena.size();arena.push_back(std::move(n));return raw;
    }
    // ai.py 17484 / 13613: MCTSNode(copy.deepcopy(node.state), node.side) - a parentless scratch outside the tree.
    // Also ai.py 14305-14310: the over-cap round-sampling outcome (select_outcome, keep=false) - a transient MCTSNode(parent=chance).
    Node* scratch_node(std::unique_ptr<Snapshot> state,int side){
        auto n=std::make_unique<Node>();n->id=NO_ID;n->scratch_seq=++scratch_count;n->state=std::move(state);n->side=side;Node* raw=n.get();scratch_arena.push_back(std::move(n));return raw;
    }
    static int64_t node_id_for_trace(const Node* n){return n->id==NO_ID?-1:static_cast<int64_t>(n->id);}
    // LB2: the request correlation id of a leaf (SHARED_ABI WotrSearchRequest.leaf_ids).  Tree nodes: the node id.  NO_ID nodes
    // (never exported; Python evaluates each as its own pending record / its own pv call): NO_ID - scratch_seq, unique per
    // controller.  Before this, every NO_ID leaf was sent as NO_ID itself, and a leaf batch holding two of them (e.g. two
    // over-cap transients of one combat round-sampling chance node) was refused by the numeric bridge
    // ('Leaf correlation IDs must be unique within a request') - the served K=8/16 game failures of 2026-09-12.
    static uint64_t leaf_corr(const Node* n){return n->id==NO_ID?NO_ID-n->scratch_seq:n->id;}
    ActionRef end_turn_action()const{return ActionRef{end_turn_menu,0};}
    const Snapshot& action_row(Node* node){
        if(!node->action_row)node->action_row=std::shared_ptr<const Snapshot>(native_menu_row(node->action.snap(),node->action.row));
        return *node->action_row;
    }
    // native_menu_view's per-row category/free classification, memoized per action type (pure function of the tables).
    const std::pair<int64_t,bool>& meta_for(const std::string& kind){
        auto found=kind_meta.find(kind);if(found!=kind_meta.end())return found->second;
        const auto category=field(*C,C->root,"category"),indices=field(*C,C->root,"category_index"),free=field(*C,C->root,"free_types");
        typed(*C,category,WOTR_DICT,"Native menu categories must be dictionary");typed(*C,indices,WOTR_DICT,"Native menu category indices must be dictionary");typed(*C,free,WOTR_TUPLE,"Native menu free types must be tuple");
        auto cat=field(*C,category,kind.c_str(),false);std::string name=cat==missing?"other":std::string(text(*C,cat));
        auto index=field(*C,indices,name.c_str(),false);if(index==missing)index=field(*C,indices,"other");
        typed(*C,index,WOTR_INT,"Native menu category index must be integer");
        bool is_free=false;for(auto item:child_edges(*C,free))is_free|=text_is(*C,item,kind);
        return kind_meta.emplace(kind,std::make_pair(C->nodes[index].integer,is_free)).first->second;
    }
    // Wrap a descriptor list Snapshot (the projected policy menu) as a Menu with per-row metadata; keys stay lazy.
    std::shared_ptr<Menu> make_menu(std::unique_ptr<Snapshot> snap){
        auto menu=std::make_shared<Menu>();menu->ctl=this;const Snapshot& s=*snap;
        const auto& list=typed(s,s.root,WOTR_LIST,"Native menu input must be descriptor list");menu->rows.reserve(list.count);
        for(uint64_t i=0;i<list.count;++i){
            const auto rownode=s.edges[list.offset+i];const auto& t=typed(s,rownode,WOTR_TUPLE,"Native menu row must be descriptor tuple");require(t.count==3,"Native menu row must have three fields");
            const auto die=s.edges[t.offset],kind=s.edges[t.offset+1],params=s.edges[t.offset+2];
            typed(s,die,WOTR_INT,"Native menu die index must be integer");typed(s,kind,WOTR_STRING,"Native menu action type must be string");typed(s,params,WOTR_DICT,"Native menu params must be dictionary");
            RowMeta m;m.die=s.nodes[die].integer;m.kind=std::string(text(s,kind));const auto& meta=meta_for(m.kind);m.category=meta.first;m.is_free=meta.second;
            menu->rows.push_back(std::move(m));
        }
        menu->snap=std::shared_ptr<const Snapshot>(std::move(snap));return menu;
    }

    // ------------------------------------------------ state facts
    static bool hidden_commit(const Snapshot& s,int side){
        const auto pc=obs_attr(s,s.root,"pending_combat",false);if(!truth(s,pc))return false;
        typed(s,pc,WOTR_DICT,"Hidden commitment requires a pending combat dictionary");
        const auto cid=field(s,pc,"atk_combat_card",false);if(!truth(s,cid))return false;
        const auto owner=field(s,pc,"attacker_side",false);
        if(enum_is(s,owner,"game_state.Side",side?"SHADOW":"FREE_PEOPLES"))return false;
        return truth(s,field(s,pc,"atk_hidden_cc",false))||text_is(s,field(s,pc,"phase",false),"defender_combat_card");
    }
    static bool attr_none(const Snapshot& s,const char* k){return ctx_none(s,obs_attr(s,s.root,k,false));}
    static bool attr_true(const Snapshot& s,const char* k){return truth(s,obs_attr(s,s.root,k,false));}
    static int pending_side(const Snapshot& s,int fallback){return native_pending_actor(s,fallback).side;}
    static int64_t crossings(const Snapshot& s){const auto n=obs_attr(s,s.root,"_search_turn_crossings",false);return ctx_none(s,n)?0:typed(s,n,WOTR_INT,"Search turn crossings must be integer").integer;}
    std::array<int64_t,5> hidden_zone_lens(const Snapshot& s)const{
        std::array<int64_t,5> out{};int i=0;
        for(const char* k:{"fp_char_deck","fp_strat_deck","sh_char_deck","sh_strat_deck","hunt_pool"}){const auto n=obs_attr(s,s.root,k,false);out[i++]=ctx_none(s,n)?0:static_cast<int64_t>(s.nodes[n].count);}
        return out;
    }
    std::vector<uint8_t> chance_sig(const Snapshot& s){Scoped _sp(this,"chance_sig");auto w=native_chance_signature_wire(s,root_side);return w->bytes;}
    static std::optional<RollMs> roll_ms(const Snapshot& s){
        const auto n=obs_attr(s,s.root,"_search_roll_ms",false);if(ctx_none(s,n))return std::nullopt;
        const auto parts=child_edges(s,n);require(parts.size()==3,"Malformed _search_roll_ms");RollMs ms;
        for(auto f:child_edges(s,parts[0]))ms.fp.emplace_back(text(s,f));for(auto f:child_edges(s,parts[1]))ms.sh.emplace_back(text(s,f));
        ms.eyes=typed(s,parts[2],WOTR_INT,"Malformed _search_roll_ms eyes").integer;return ms;
    }
    // ai.py 14620-14645 _roll_ms_prob: exact multinomial probability of a roll multiset.
    static double mprob(const std::vector<std::string>& faces,int64_t extra){
        const int64_t n=static_cast<int64_t>(faces.size())+extra;if(n<=0)return 1.0;require(n<=20,"roll multiset exceeds factorial domain");
        std::map<std::string,int64_t> cnt;for(const auto& f:faces)cnt[f]+=1;if(extra)cnt["EYE"]+=extra;
        auto fact=[](int64_t k){int64_t r=1;for(int64_t i=2;i<=k;++i)r*=i;return r;};
        const int64_t num=fact(n);int64_t den=1;for(const auto& c:cnt)den*=fact(c.second);
        const double quotient=static_cast<double>(num)/static_cast<double>(den);
        const double base=1.0/6.0;return quotient*std::pow(base,static_cast<double>(n));
    }
    static double roll_ms_prob(const RollMs& ms){return mprob(ms.fp,0)*mprob(ms.sh,ms.eyes);}

    // ------------------------------------------------ policy / encodings
    Policy make_policy(Snapshot& s,int side){
        fm_prepare_menu_context_inplace(s,*T,side); // Python menu cleanup is committed on the owned search state.
        std::unique_ptr<Snapshot> raw;{Scoped _sp(this,"full_actions");raw=native_full_actions(s,*T,side);}
        NativePolicyProjection proj;{Scoped _sp(this,"project_policy");proj=project_native_ai_actions_owned(s,std::move(raw),side);}
        std::shared_ptr<Menu> menu;{Scoped _sp(this,"menu_meta");menu=make_menu(std::move(proj.actions));}
        if(profiling)prof_rows["full_actions_rows"]+=menu->size();
        return Policy{menu,proj.forced_reason,proj.reasons};
    }
    std::vector<float> observe(const Snapshot& s,int side){Scoped _sp(this,"encode_state");std::vector<float> out(OBS_DIM);encode_state(s,*T,side,out.data(),OBS_DIM);return out;}
    void action_matrix(const Snapshot& s,const Snapshot& menu,int side,size_t rows,std::vector<float>& ae,std::vector<int64_t>& ri){
        Scoped _sp(this,"af_encode");if(profiling)prof_rows["af_encode_rows"]+=rows;ae.assign(rows*ACT_DIM,0.0f);ri.assign(rows,0);af_encode(s,*T,menu,*A,side,ae.data(),rows*ACT_DIM,ri.data(),rows);
    }

    // ------------------------------------------------ terminal / profile (reward_contract.profile_terminal, ai.py 16790-16844)
    static double profile_terminal(int side,int profile,int kind,bool won){
        const int winner=won?side:1-side;
        const bool route_ok=kind==1||(winner==0&&kind==2)||(winner==1&&kind==3);
        if(!route_ok)throw std::invalid_argument("Invalid victory route for winner");
        if(!won)return -1.0;
        const int target=profile==2?1:(side==0?2:3);
        if(profile==3||kind==target)return 1.0;
        return side==0?0.0:0.2;
    }
    static double profile_shaping(){return 0.0;}   // ai.py 13017: uses_predecision() -> 0.0
    double search_terminal(int winner,int kind)const{
        const int profile=root_side==0?cfg.fp_profile:cfg.sh_profile;
        double v=profile_terminal(root_side,profile,kind,root_side==winner);
        const double sgn=root_side==0?1.0:-1.0;v+=sgn*profile_shaping();return v;
    }
    int proof_mark(int winner,int kind)const{
        const int mark=winner==root_side?1:-1;const int profile=root_side==0?cfg.fp_profile:cfg.sh_profile;
        const double value=profile_terminal(root_side,profile,kind,winner==root_side);const double bound=mark==1?1.0:-1.0;
        return std::isfinite(value)&&std::fabs(value-bound)<=1e-9?mark:0;
    }
    Victory victory(const Snapshot& s){Scoped _sp(this,"settle_victory");const auto r=check_victory(s,*T);return Victory{terminal_should_end(r,false),r.winner,r.kind};}
    static double bound_value(double v){const double m=v<1.0?v:1.0;return m>-1.0?m:-1.0;}

    // ai.py 12663-12728 _conf_shrink ('abs' reference).  NaN conf = provider
    // exception/NaN -> identity with NO _LAST_LEAF_CONF write (source semantics).
    double conf_shrink(double c,double v_raw){
        if(!conf_active)return v_raw;
        if(!(c==c))return v_raw;
        c=c<0.0?0.0:(c>1.0?1.0:c);
        if(!cfg.conf_leaf){last_conf=c;return v_raw;}
        last_conf=c;return c*v_raw;
    }

    // ------------------------------------------------ inference wrappers
    // ai.py 12881-12982 _get_priors_and_value (remote branch; features_vec supplied by the caller)
    PV get_priors_and_value(const Snapshot& s,int side,const Menu& menu,const std::vector<float>& vec,int purpose,uint64_t leaf_id){
        RequestData r;r.leaf_ids={leaf_id};r.obs=vec;action_matrix(s,*menu.snap,side,menu.size(),r.actions,r.regions);r.offsets={0,menu.size()};
        const auto rows=menu.size();ReplyData rep=await_reply(std::move(r),OP_PV,side,purpose);
        require(rep.priors.size()==rows&&rep.mixed.size()==rows&&rep.values.size()==1&&rep.conf.size()==1,"PV reply shape mismatch");
        PV out;out.priors.assign(rep.priors.begin(),rep.priors.end());
        double v=conf_shrink(rep.conf[0],rep.values[0]);
        if(profiles_active){double shp=profile_shaping();if(side!=0)shp=-shp;if(conf_active)last_phi=shp;v=v+shp;}
        out.value=bound_value(v);out.mixed=std::move(rep.mixed);return out;
    }
    // ai.py 13062-13108 _mcts_eval_state (remote branch)
    double mcts_eval_state(const Snapshot& s,int side,uint64_t leaf_id){
        const double shp=profile_shaping();const double sgn=side==0?1.0:-1.0;
        if(conf_active)last_phi=sgn*shp;
        RequestData r;r.leaf_ids={leaf_id};r.obs=observe(s,side);r.offsets={0,0};/* ABI: EVAL offsets are [B+1] zeros */
        ReplyData rep=await_reply(std::move(r),OP_EVAL,side,PURPOSE_EVAL);
        require(rep.values.size()==1&&rep.conf.size()==1,"EVAL reply shape mismatch");
        return bound_value(conf_shrink(rep.conf[0],rep.values[0])+sgn*shp);
    }

    // 9.5 request: op 4, batch 1, rows 0, leaf_ids [node], model_side = node side, purpose 8; payload graph beside it.
    static uint64_t battle_tuple(Meta& m,const Snapshot& s){
        const auto pc=obs_attr(s,s.root,"pending_combat",false);if(ctx_none(s,pc)||s.nodes[pc].tag!=WOTR_DICT)return m.none();
        const auto ar=field(s,pc,"attacker_region",false),dr=field(s,pc,"defender_region",false),as=field(s,pc,"attacker_side",false),ds=field(s,pc,"defender_side",false);
        if(ar==missing||dr==missing||as==missing||ds==missing)return m.none();
        const auto turn=obs_attr(s,s.root,"turn",false);const auto rounds=field(s,pc,"rounds_fought",false);
        return m.tuple({turn==missing?m.none():m.copy(s,turn),m.copy(s,ar),m.copy(s,dr),m.copy(s,as),m.copy(s,ds),rounds==missing?m.integer(0):m.copy(s,rounds)});
    }
    DistReply request_dist(Node* node,const Snapshot& ctx){
        Meta m;const auto rb=root_branch_row(node);
        std::unique_ptr<Snapshot> payload=m.finish(m.dict({{"context",m.copy(ctx,ctx.root)},{"node",m.integer(node_id_for_trace(node))},{"action_type",m.str(node->action.kind())},
            {"root_branch_row",rb?m.integer(static_cast<int64_t>(*rb)):m.none()},{"parent_battle",battle_tuple(m,*node->parent->state)}}));
        RequestData r;r.leaf_ids={node->id};r.obs.assign(OBS_DIM,0.0f);r.offsets={0,0};
        {std::lock_guard<std::mutex> lk(mu);pending_payload=std::move(payload);}
        await_reply(std::move(r),OP_DIST,node->side,PURPOSE_DIST);
        std::lock_guard<std::mutex> lk(mu);pending_payload.reset();DistReply out=std::move(dist_reply);dist_reply=DistReply{};return out;
    }

    // ------------------------------------------------ opponent policy cache (ai.py 16980-17003)
    std::string opponent_policy_key(int side,const std::vector<float>& vec,const Menu& menu){
        Scoped _sp(this,"opponent_key");menu.ensure_keys();size_t total=1+OBS_DIM*4+8;for(size_t i=0;i<menu.size();++i)total+=menu.key(i).size();
        std::string key;key.reserve(total);key+=static_cast<char>(side);
        key.append(reinterpret_cast<const char*>(vec.data()),vec.size()*sizeof(float));append_u64(key,menu.size());
        for(size_t i=0;i<menu.size();++i)key+=menu.key(i);return key;
    }
    const std::vector<double>& remember_opponent_policy(int side,const std::vector<float>& vec,const Menu& menu,const std::vector<double>& mixed,std::string key){
        require(mixed.size()==menu.size(),"Mixed opponent policy length mismatch");
        if(key.empty())key=opponent_policy_key(side,vec,menu);
        if(policy_cache.size()>=256)policy_cache.clear();
        auto& slot=policy_cache[std::move(key)];slot=mixed;return slot;
    }

    // ------------------------------------------------ overlay (ai.py 15105-15165, driver Overlay wrapper)
    void sample_overlay(){
        Scoped _sp(this,"so_sample");overlay.reset();rng_mark("pre_sample");
        try{
            if(hidden_commit(*root->state,root_side)){
                // _SimOverlay.__init__ 14830-14845: mixture first (raises when empty), then observer.hidden_commit, then the draw.
                auto mixture=so_commit_mixture(*root->state,*T);
                if(mixture.empty())fail("Hidden combat-card commitment has no representable candidate");
                ObsEvent e;e.kind=1;e.mixture=std::move(mixture);observer_events.push_back(std::move(e));
            }
            overlay=so_sample(*root->state,*T,root_side,rng.get());rng_mark("sample");overlay_opp=1-root_side;counters.overlay["samples"]+=1;if(overlay->cid_set)counters.overlay["commit_blind"]+=1;
        }
        catch(...){overlay.reset();counters.overlay["sample_error"]+=1;throw;}
    }
    void overlay_patch(Snapshot& s,bool count_error){
        Scoped _sp(this,"so_patch");counters.overlay["patches"]+=1;
        try{so_patch(*overlay,s,rng.get());rng_mark("patch");}
        catch(...){if(count_error)counters.overlay["patch_error"]+=1;throw;}
    }
    void overlay_prepare(Snapshot& parent,int actor,Node* node){
        if(!overlay)return;Scoped _sp(this,"so_prepare");counters.overlay["patches"]+=1;
        try{so_prepare(*overlay,parent,rng.get(),actor,node&&node->action.has()?&action_row(node):nullptr,true);rng_mark("prepare");}
        catch(...){counters.overlay["patch_error"]+=1;throw;}
    }
    void overlay_assert_state(const Snapshot& s){
        if(!overlay||!cfg.overlay_assert)return;Scoped _sp(this,"overlay_assert");counters.overlay["assert_checks"]+=1;
        auto got_graph=native_overlay_assert_hand(s,overlay_opp);std::vector<std::string> got;for(auto n:child_edges(*got_graph,got_graph->root))got.emplace_back(text(*got_graph,n));
        std::vector<std::string> want=overlay->hand[0];want.insert(want.end(),overlay->hand[1].begin(),overlay->hand[1].end());std::sort(want.begin(),want.end());
        if(got!=want){std::string msg="overlay fingerprint mismatch: state=[";for(auto& c:got)msg+=c+",";msg+="] overlay=[";for(auto& c:want)msg+=c+",";msg+="]";throw std::runtime_error(msg);}
    }
    // ai.py 14364-14384 _blinded_commit_key: the sampled committed-card identity under the source conditions.
    std::optional<std::string> blinded_commit_key(const Snapshot& parent){
        if(!overlay)return std::nullopt;
        if(!overlay->cid_set){if(overlay->hidden)fail("Hidden combat-card commitment has no sampled identity");return std::nullopt;}
        const auto pc=obs_attr(parent,parent.root,"pending_combat",false);
        if(ctx_none(parent,pc)||parent.nodes[pc].tag!=WOTR_DICT)return std::nullopt;
        if(!truth(parent,field(parent,pc,"atk_combat_card",false))||!truth(parent,field(parent,pc,"_commit_nonce",false)))return std::nullopt;
        const auto rounds=field(parent,pc,"rounds_fought",false);const int64_t n=ctx_none(parent,rounds)?0:parent.nodes[rounds].integer;
        if(!(overlay->round_set&&n==overlay->round))return std::nullopt;
        return overlay->cid;
    }
    // ai.py 14385-14397 _kernel_register_world (None key keeps the legacy single-set node).
    void kernel_register_world(Node* node){
        const auto key=blinded_commit_key(*node->parent->state);
        node->has_kernel_key=key.has_value();node->kernel_key=key.value_or(std::string());
        if(!key)return;
        node->has_kernel_sets=true;World& w=world_slot(node,*key);w.outcomes=node->outcomes;w.has_cover=node->has_cover;w.cover=node->kernel_cover;w.drops=node->drops;
    }
    static std::string bank_key(const Node* node){return node->has_kernel_key?node->kernel_key:std::string(1,'\0');}   // Python dict key None
    // ---------------------------------------------- combat kernel (ai.py 15598-15987, SHARED_ABI 9.5)
    static std::string py_repr_str(const std::string& v){ap_Frozen f;f.tag=WOTR_STRING;f.value=v;return ap_repr(f);}
    static int64_t pc_int(const Snapshot& s,uint64_t pc,const char* key){   // int(pc.get(key, 0) or 0)
        const auto n=field(s,pc,key,false);if(ctx_none(s,n))return 0;const auto& v=s.nodes[n];
        if(v.tag==WOTR_INT||v.tag==WOTR_BOOL)return v.integer;if(v.tag==WOTR_FLOAT)return static_cast<int64_t>(v.real);throw std::invalid_argument("pending_combat hit count must be numeric");
    }
    // combat_dist 369-401 / 403-440 in binary64 with the source's operation order.
    static int hit_faces(int threshold,int mod){int n=0;for(int r=2;r<=6;++r)if(r==6||r+mod>=threshold)++n;return n;}
    static std::vector<std::pair<int64_t,double>> binom_terms(int64_t n,double p){
        std::vector<std::pair<int64_t,double>> out;
        for(int64_t k=0;k<=n;++k){
            double comb=1.0;{int64_t c=1;for(int64_t i=1;i<=k;++i)c=c*(n-k+i)/i;comb=static_cast<double>(c);}   // math.comb exact, then int*float
            const double pr=comb*std::pow(p,static_cast<double>(k))*std::pow(1.0-p,static_cast<double>(n-k));
            if(pr>0.0)out.push_back({k,pr});
        }
        return out;
    }
    static std::pair<int64_t,int64_t> apply_hits_preserve_elites(int64_t r,int64_t e,int64_t h){
        const int64_t k=std::min(r,h);r-=k;h-=k;
        while(h>0&&e>0){e-=1;if(h>=2)h-=2;else{r+=1;h-=1;}}
        return {r,e};
    }
    // combat_round_search.requires_phase_sampling 48-64 over native facts.
    bool requires_phase_sampling(const Snapshot& parent,const Snapshot& probe){
        for(const Snapshot* s:{&parent,&probe}){const auto pc=obs_attr(*s,s->root,"pending_combat",false);if(!ctx_none(*s,pc)&&s->nodes[pc].tag==WOTR_DICT&&truth(*s,field(*s,pc,"_round_resume",false)))return true;}
        const auto info=obs_attr(probe,probe.root,"_last_round_call",false);if(ctx_none(probe,info)||probe.nodes[info].tag!=WOTR_DICT)return false;
        const auto choices=field(probe,info,"choices",false);if(ctx_none(probe,choices)||probe.nodes[choices].tag!=WOTR_DICT||!truth(probe,field(probe,choices,"defer_round_choices",false)))return false;
        static const char* adaptive[]={"dread and despair","relentless assault","mighty attack","fateful strike","blade of westernesse","heroic death","black breath","charge","sudden strike","durin's bane"};
        const auto cards=field(*T,T->root,"cards");
        for(const char* key:{"atk_card","def_card"}){
            const auto cid=field(probe,info,key,false);if(!truth(probe,cid)||probe.nodes[cid].tag!=WOTR_STRING)continue;
            const auto card=field(*T,cards,std::string(text(probe,cid)).c_str(),false);if(ctx_none(*T,card))continue;
            const auto name=obs_attr(*T,card,"combat_name",false);if(ctx_none(*T,name)||T->nodes[name].tag!=WOTR_STRING)continue;
            std::string lower(text(*T,name));const auto b=lower.find_first_not_of(" \t\r\n"),e=lower.find_last_not_of(" \t\r\n");lower=b==std::string::npos?std::string():lower.substr(b,e-b+1);
            for(auto& c:lower)c=static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            for(const char* a:adaptive)if(lower==a)return true;
        }
        return false;
    }
    // native_combat_search_units arguments: {region: str, side: Side, kind: UnitKind|None, pc: {'attacker_rearguard': ...}}
    std::unique_ptr<Snapshot> units_args(const std::string& region,int side,int kind,const Snapshot* rear_src,uint64_t rear_node){
        Meta m;const uint64_t rear=rear_src?m.copy(*rear_src,rear_node):m.none();
        return m.finish(m.dict({{"region",m.str(region)},{"side",m.enum_member("game_state.Side",side==0?"FREE_PEOPLES":"SHADOW")},
            {"kind",kind<0?m.none():m.enum_member("game_state.UnitKind",kind==0?"REGULAR":"ELITE")},{"pc",m.dict({{"attacker_rearguard",rear}})}}));
    }
    struct UnitCounts{int64_t regular=0,elite=0;std::set<std::string> nations;};
    UnitCounts alloc_units(const Snapshot& state,const std::string& region,int side,int kind,const Snapshot* rear_src,uint64_t rear_node){
        auto args=units_args(region,side,kind,rear_src,rear_node);auto units=native_combat_search_units(state,*args);UnitCounts out;
        for(auto u:child_edges(*units,units->root)){
            const auto k=enum_member(*units,obs_attr(*units,u,"kind"),"game_state.UnitKind");if(k=="REGULAR")++out.regular;else ++out.elite;
            out.nations.emplace(enum_member(*units,obs_attr(*units,u,"nation"),"game_state.Nation"));
        }
        return out;
    }
    struct RoundInfo{std::string atk_region,def_region;int atk_side=0,def_side=0;std::optional<std::string> atk_card,def_card;bool is_siege=false;uint64_t rearguard=missing;const Snapshot* src=nullptr;uint64_t node=missing;};
    RoundInfo round_info(const Snapshot& probe,uint64_t info){
        RoundInfo r;r.src=&probe;r.node=info;
        r.atk_region=std::string(text(probe,field(probe,info,"atk_region")));r.def_region=std::string(text(probe,field(probe,info,"def_region")));
        r.atk_side=ctx_side(probe,field(probe,info,"atk_side"));r.def_side=ctx_side(probe,field(probe,info,"def_side"));
        for(auto item:{std::make_pair("atk_card",&r.atk_card),std::make_pair("def_card",&r.def_card)}){const auto n=field(probe,info,item.first,false);if(truth(probe,n)&&probe.nodes[n].tag==WOTR_STRING)*item.second=std::string(text(probe,n));}
        r.is_siege=truth(probe,field(probe,info,"is_siege",false));r.rearguard=field(probe,info,"attacker_rearguard",false);
        return r;
    }
    // ai.py 15959-15987 _predict_alloc_pause.
    bool predict_alloc_pause(const Snapshot& state,const RoundInfo& info,const Obs& obs){
        struct Row{std::string region;int side;int64_t r,e,hits;};
        for(const Row& row:{Row{info.atk_region,info.atk_side,obs[8].as_int_or0(),obs[9].as_int_or0(),obs[18].as_int_or0()},Row{info.def_region,info.def_side,obs[10].as_int_or0(),obs[11].as_int_or0(),obs[17].as_int_or0()}}){
            const int64_t hits=row.hits;if(hits<=0||row.r+row.e<=0)continue;
            if(hits>=2*row.e+row.r&&(row.side==1||row.e==0))continue;
            if(row.e&&(row.r||hits>=2))return true;
            const auto units=alloc_units(state,row.region,row.side,row.e?1:0,info.src,info.rearguard);
            if(units.nations.size()>1)return true;
        }
        return false;
    }
    // ai.py 15931-15958 _gh_alloc_expected.
    static std::pair<int64_t,int64_t> gh_alloc_expected(const std::optional<std::string>& atk_card,const std::optional<std::string>& def_card,const Obs& obs){
        const auto a=apply_hits_preserve_elites(obs[8].as_int_or0(),obs[9].as_int_or0(),obs[18].as_int_or0()),d=apply_hits_preserve_elites(obs[10].as_int_or0(),obs[11].as_int_or0(),obs[17].as_int_or0());
        const int64_t atk_n=a.first+a.second,def_n=d.first+d.second;int64_t hd=0,ha=0;
        auto lower=[](const std::optional<std::string>& s){std::string v=s.value_or("");for(auto& c:v)c=static_cast<char>(std::tolower(static_cast<unsigned char>(c)));return v;};
        if(atk_n>0&&def_n>0){
            if(lower(atk_card).find("great host")!=std::string::npos&&atk_n>=2*def_n)hd=1;
            if(lower(def_card).find("great host")!=std::string::npos&&def_n>=2*atk_n)ha=1;
        }
        return {hd,ha};
    }
    struct Forced{std::unique_ptr<Snapshot> state;bool mismatch=false;std::string message;};
    struct PlanMismatch{std::string message;};   // combat_dist.PlanMismatch, thrown out of the kept-outcome loop
    static std::string plan_item_repr(const PlanItem& p){return p.is_tuple?"("+std::to_string(p.hits)+", "+std::to_string(p.ones)+")":std::to_string(p.hits);}
    // combat_dist.run_with_plan via the native compatible forced-combat scope (combat_forcing_errors_exports.inc);
    // PlanMismatch texts as experiments/combat_forcing_errors/native_compatible.py mismatch().
    Forced forced_apply(const Snapshot& parent,const ActionRef& a,int side,const std::vector<PlanItem>& plan,bool confusion){
        Forced out;std::vector<int64_t> flat;std::vector<uint8_t> tags;
        for(const auto& p:plan){flat.push_back(p.hits);flat.push_back(p.is_tuple?p.ones:-1);tags.push_back(p.is_tuple?1:0);}
        require(plan.size()<=256,"Combat plan exceeds supported batch count");
        try{
            cfc_Scope compatible(tags.data(),tags.size());cf_Scope scope(flat.data(),plan.size(),confusion);
            auto result=apply_native_random_descriptor(parent,*T,a.snap(),rng.get(),side,a.row);
            if(!cf_same_rng(rng.get(),result.random.get()))throw cfc_Error(4,0,"",0,0,0);
            if(scope.plan.next!=scope.plan.size)throw cfc_Error(5,0,"",0,0,0);
            out.state=std::move(result.state);
        }catch(const cfc_Error& e){
            out.mismatch=true;std::string message;
            if(e.kind==1)message="plan exhausted (extra batch)";
            else if(e.kind==2)message="stat "+(e.index<plan.size()?plan_item_repr(plan[e.index]):std::string("?"))+" not in batch support";
            else if(e.kind==3){out.message="raw roll_d6_batch("+py_repr_str(e.context)+")";return out;}
            else if(e.kind==4){out.message="non-batch RNG consumed during forced apply";return out;}
            else if(e.kind==5){out.message="plan not fully consumed";return out;}
            else throw std::runtime_error("Unknown native combat forcing failure classification");
            out.message=message+" [batch ctx="+py_repr_str(e.context)+" n="+std::to_string(e.count)+" t="+std::to_string(e.threshold)+" m="+std::to_string(e.modifier)+"]";
        }
        return out;
    }
    // ai.py _army_dump (diagnostic repr) over the native army projection.
    std::string army_dump(const Snapshot& state,const std::string& region){
        Meta m;auto args=m.finish(m.str(region));auto army=native_combat_search_army(state,*args);
        if(ctx_none(*army,army->root))return "None";
        static const std::map<std::string,std::string> values={{"ELVES","Elves"},{"DWARVES","Dwarves"},{"GONDOR","Gondor"},{"THE_NORTH","The North"},{"ROHAN","Rohan"},{"ISENGARD","Isengard"},{"SAURON","Sauron"},{"SOUTHRON_EASTERLINGS","Southrons & Easterlings"}};
        std::string units="[",leaders="[";bool first=true;
        for(auto u:child_edges(*army,obs_attr(*army,army->root,"units"))){const auto nation=std::string(enum_member(*army,obs_attr(*army,u,"nation"),"game_state.Nation"));const auto found=values.find(nation);
            if(!first)units+=", ";first=false;units+="("+py_repr_str(found==values.end()?nation:found->second)+", "+py_repr_str(std::string(enum_member(*army,obs_attr(*army,u,"kind"),"game_state.UnitKind")))+")";}
        units+="]";first=true;
        for(auto l:child_edges(*army,obs_attr(*army,army->root,"leaders"))){
            std::string type(object_type(*army,l));type=type.substr(type.rfind('.')+1);
            const auto name=obs_attr(*army,l,"name",false),nation=obs_attr(*army,l,"nation",false);sm_Builder reader(*army);
            const std::string nm=ctx_none(*army,name)?std::string():reader.string(name);
            const std::string nt=ctx_none(*army,nation)?std::string():(army->nodes[nation].tag==WOTR_ENUM?"Nation."+std::string(enum_member(*army,nation,"game_state.Nation")):reader.string(nation));
            if(!first)leaders+=", ";first=false;leaders+="("+py_repr_str(type)+", "+py_repr_str(nm)+", "+py_repr_str(nt)+")";}
        leaders+="]";return "("+units+", "+leaders+")";
    }
    uint64_t detail_obs_slice(const Obs& obs){std::vector<uint64_t> xs;for(size_t i=8;i<19;++i)xs.push_back(detail_atom(obs[i]));return counters.detail.tuple(xs);}
    uint64_t detail_cards(const std::optional<std::string>& a,const std::optional<std::string>& d){return counters.detail.tuple({detail_opt_str(a),detail_opt_str(d)});}
    // Install merged kernel branches (insertion-ordered merge by chance signature, stable sort by -mass); shared by both kernels.
    void install_kernel_branches(Node* node,std::vector<std::pair<std::unique_ptr<Snapshot>,double>>& branches,double cover,const std::string& cls,double dropped_entry,int64_t support_entry /* <0: merged count */){
        struct Merged{std::vector<uint8_t> sig;std::unique_ptr<Snapshot> state;double mass;};std::vector<Merged> merged;
        for(auto& b:branches){auto sg=chance_sig(*b.first);auto it=std::find_if(merged.begin(),merged.end(),[&](const Merged& m){return m.sig==sg;});if(it!=merged.end())it->mass+=b.second;else merged.push_back({std::move(sg),std::move(b.first),b.second});}
        std::vector<size_t> order(merged.size());for(size_t i=0;i<order.size();++i)order[i]=i;
        std::stable_sort(order.begin(),order.end(),[&](size_t a,size_t b){return -merged[a].mass< -merged[b].mass;});
        node->is_chance=true;node->has_outcomes=true;node->outcomes.clear();
        for(size_t i:order){auto& m=merged[i];Node* br=new_node(std::move(m.state),node->side,node,ActionRef{},m.mass);br->out_sig=m.sig;br->has_sig=true;node->outcomes.push_back(br);}
        node->drops=static_cast<uint64_t>(cfg.chance_draws);node->has_cover=true;node->kernel_cover=cover;
        kernel_register_world(node);
        counters.kernel["kernel_nodes"]+=1;
        const int64_t support=support_entry<0?static_cast<int64_t>(node->outcomes.size()):support_entry;
        if(counters.dropped_mass.size()<500000){counters.dropped_mass.push_back(dropped_entry);counters.support.push_back(support);counters.by_class[cls].push_back(dropped_entry);}
    }
    // ai.py 15868-15930 _kernel_expand_onslaught_post.
    bool kernel_expand_onslaught_post(Node* node){
        if(cfg.overlay_assert)overlay_assert_state(*node->parent->state);
        const Snapshot& parent=*node->parent->state;const auto pc=obs_attr(parent,parent.root,"pending_combat",false);
        const auto payment=(!ctx_none(parent,pc)&&parent.nodes[pc].tag==WOTR_DICT)?field(parent,pc,"_onslaught_payment",false):missing;
        const Snapshot& menu=node->action.snap();const auto params=node->action.params();int64_t k;
        if(!ctx_none(parent,payment)&&truth(parent,payment)){
            const bool elite_remove=text_is(menu,field(menu,params,"kind",false),"elite")&&text_is(menu,field(menu,params,"operation",false),"remove");
            k=pc_int(parent,payment,"paid")+(elite_remove?2:1);
        }else{const auto count=field(menu,params,"count",false);k=ctx_none(menu,count)?0:menu.nodes[count].integer;}
        const std::string kind=node->action.kind();
        if(k<=0){kstat_fallback("onslaught_post_k0",detail_dict({{"a",detail_str(kind)}}));return false;}
        std::vector<std::pair<std::unique_ptr<Snapshot>,double>> branches;
        for(const auto& term:binom_terms(k,hit_faces(4,0)/6.0)){
            PlanItem item;item.hits=term.first;auto forced=forced_apply(*node->parent->state,node->action,node->side,{item},false);
            if(forced.mismatch){kstat_fallback("plan_mismatch",detail_dict({{"a",detail_str(kind)},{"err",detail_str(forced.message.substr(0,200))},{"cards",counters.detail.tuple({counters.detail.str("onslaught_post"),counters.detail.integer(k)})}}));return false;}
            branches.push_back({std::move(forced.state),term.second});
        }
        install_kernel_branches(node,branches,1.0,"onslaught_post",0.0,-1);   // source: support = len(merged)
        return true;
    }
    // ai.py 15598-15867 _kernel_expand_combat over the COMBAT_DIST request.
    bool kernel_expand_combat(Node* node,const Snapshot& probe){
        const Snapshot& parent=*node->parent->state;const std::string kind=node->action.kind();
        if(cfg.overlay_assert)overlay_assert_state(parent);
        counters.kernel["combat_chance_nodes"]+=1;
        if(requires_phase_sampling(parent,probe)){node->round_sampling=true;kstat_fallback("observed_round_phase");return false;}
        const auto info_node=obs_attr(probe,probe.root,"_last_round_call",false);
        if(ctx_none(probe,info_node)||probe.nodes[info_node].tag!=WOTR_DICT){
            const auto pcd=obs_attr(parent,parent.root,"pending_combat",false);const bool pc_dict=!ctx_none(parent,pcd)&&parent.nodes[pcd].tag==WOTR_DICT;
            const auto ecd=pc_dict?field(parent,pcd,"effect_choice",false):missing;const bool ec_dict=!ctx_none(parent,ecd)&&parent.nodes[ecd].tag==WOTR_DICT;
            const auto stage=ec_dict?field(parent,ecd,"stage",false):missing;
            if(kind=="combat_effect_choice"&&(text_is(parent,stage,"post")||text_is(parent,stage,"onslaught_payment")))return kernel_expand_onslaught_post(node);
            auto opt=[&](uint64_t n,const Snapshot& s){return (n==missing||ctx_none(s,n))?counters.detail.none():counters.detail.copy(s,n);};
            kstat_fallback("no_round_call",detail_dict({{"a",detail_str(kind)},{"params",counters.detail.copy(node->action.snap(),node->action.params())},
                {"pc_phase",opt(pc_dict?field(parent,pcd,"phase",false):missing,parent)},{"ec_stage",opt(stage,parent)},{"ec_effect",opt(ec_dict?field(parent,ecd,"effect",false):missing,parent)}}));
            return false;
        }
        const RoundInfo info=round_info(probe,info_node);
        std::optional<std::string> ctx_atk,ctx_def;std::string ctx_repr;Obs cur_obs;bool have_cur=false;size_t cur_oi=0;std::vector<std::string> plan_reprs;
        auto error_detail=[&](const std::string& err){
            std::vector<std::pair<std::string,uint64_t>> kv;
            for(const char* key:{"atk_region","def_region","atk_side","def_side","atk_card","def_card","is_siege"}){const auto n=field(probe,info_node,key,false);sm_Builder reader(probe);kv.push_back({key,detail_str(n==missing?"None":reader.string(n))});}
            return detail_dict({{"a",detail_str(kind)},{"err",detail_str(err)},{"info",detail_dict(kv)}});
        };
        try{
            // _kernel_ctx over the native projection
            Snapshot info_snap;{std::unordered_map<uint64_t,uint64_t> memo;info_snap.root=graph_copy(info_snap,probe,info_node,memo);}
            auto ctx=native_kernel_context(parent,*T,info_snap,info.atk_card?info.atk_card->c_str():nullptr,info.def_card?info.def_card->c_str():nullptr,true);
            {const auto parts=child_edges(*ctx,ctx->root);require(parts.size()==12,"Native kernel context must have 12 fields");
             if(!ctx_none(*ctx,parts[2]))ctx_atk=std::string(text(*ctx,parts[2]));if(!ctx_none(*ctx,parts[3]))ctx_def=std::string(text(*ctx,parts[3]));}
            const DistReply dist=request_dist(node,*ctx);ctx_repr=dist.ctx_repr;plan_reprs=dist.plan_reprs;
            // OUTCOME-CLASS COARSENING by obs[8:19] (insertion order; max-mass member witnesses the class)
            struct Class{Obs key;double mass;size_t witness_index;double max_mass;};std::vector<Class> classes;
            for(size_t i=0;i<dist.outcomes.size();++i){
                const Obs& ob=dist.outcomes[i];Obs key(ob.begin()+8,ob.begin()+19);const double m9=dist.masses[i];
                auto it=std::find_if(classes.begin(),classes.end(),[&](const Class& c){return c.key==key;});
                if(it==classes.end())classes.push_back({std::move(key),m9,i,m9});
                else{it->mass+=m9;if(m9>it->max_mass){it->witness_index=i;it->max_mass=m9;}}
            }
            std::vector<size_t> order(classes.size());for(size_t i=0;i<order.size();++i)order[i]=i;
            std::stable_sort(order.begin(),order.end(),[&](size_t a,size_t b){return -classes[a].mass< -classes[b].mass;});
            std::vector<std::pair<size_t,double>> outs;for(size_t i:order)outs.push_back({classes[i].witness_index,classes[i].mass});
            double total=0.0;for(const auto& o:outs)total+=o.second;
            if(outs.empty()||std::fabs(total-1.0)>1e-6){kstat_fallback("mass_error");return false;}
            size_t k_eff=std::min<size_t>(static_cast<size_t>(cfg.kernel_k_max),outs.size());
            double cover=0.0;for(size_t i=0;i<k_eff;++i)cover+=outs[i].second;
            const size_t k_hard=cfg.chance_k_max?static_cast<size_t>(96):outs.size();   // _KERNEL_K_HARD when CHANCE_K_MAX truthy
            while(k_eff<std::min(outs.size(),k_hard)&&cover<0.90){cover+=outs[k_eff].second;++k_eff;}
            const size_t kept_n=k_eff;
            if(1.0-cover>0.25){
                HighDrop h;h.kernel=true;h.cls=kind;h.dropped=std::nearbyint((1.0-cover)*1e6)/1e6;h.support=static_cast<int64_t>(outs.size());h.support_raw=static_cast<int64_t>(dist.outcomes.size());h.kept=static_cast<int64_t>(kept_n);h.ctx_repr=ctx_repr.substr(0,900);
                counters.high_drop.push_back(std::move(h));
                if(counters.high_drop.size()>20){std::stable_sort(counters.high_drop.begin(),counters.high_drop.end(),[](const HighDrop& a,const HighDrop& b){return -a.dropped< -b.dropped;});counters.high_drop.resize(20);}
            }
            const bool conf=dist.conf;const std::optional<std::string> wctk_side=dist.wctk_side;
            std::vector<std::pair<std::unique_ptr<Snapshot>,double>> branches;
            for(size_t idx=0;idx<kept_n;++idx){
                const size_t oi=outs[idx].first;const double mass=outs[idx].second;const Obs& obs=dist.outcomes[oi];cur_obs=obs;have_cur=true;cur_oi=oi;
                const auto& plan=dist.witness[oi];
                if(!plan){kstat_fallback("no_witness");return false;}
                if(wctk_side&&obs[16].truthy()&&!predict_alloc_pause(parent,info,obs)){
                    const auto a2=apply_hits_preserve_elites(obs[8].as_int_or0(),obs[9].as_int_or0(),obs[18].as_int_or0()),d2=apply_hits_preserve_elites(obs[10].as_int_or0(),obs[11].as_int_or0(),obs[17].as_int_or0());
                    const bool atk=*wctk_side=="attacker";const auto own_post=atk?a2:d2,en_post=atk?d2:a2;const int64_t ec9=std::min<int64_t>(5,own_post.second);
                    if(a2.first+a2.second>0&&d2.first+d2.second>0&&ec9>0){
                        const std::string ro=atk?info.atk_region:info.def_region,re9=atk?info.def_region:info.atk_region;const int so=atk?info.atk_side:info.def_side,se=atk?info.def_side:info.atk_side;
                        const char* enr=atk?"def":"atk";
                        for(const auto& term:binom_terms(ec9,hit_faces(5,0)/6.0)){
                            const int64_t h9=term.first;std::vector<PlanItem> extended=*plan;PlanItem extra;extra.hits=h9;extended.push_back(extra);
                            auto forced=forced_apply(parent,node->action,node->side,extended,conf);
                            if(forced.mismatch)throw PlanMismatch{forced.message};
                            const Snapshot& st=*forced.state;const auto pcb=obs_attr(st,st.root,"pending_combat",false);const bool pcb_dict=!ctx_none(st,pcb)&&st.nodes[pcb].tag==WOTR_DICT;
                            const bool paused=pcb_dict&&text_is(st,field(st,pcb,"phase",false),"casualty_allocation")&&text_is(st,field(st,pcb,"alloc_stage",false),atk?"defender":"attacker")&&pc_int(st,pcb,(std::string(enr)+"_hits").c_str())==h9;
                            const auto en_pred=paused?en_post:apply_hits_preserve_elites(en_post.first,en_post.second,h9);
                            const auto enemy=alloc_units(st,re9,se,-1,info.src,info.rearguard),own=alloc_units(st,ro,so,-1,info.src,info.rearguard);
                            if(std::make_pair(enemy.regular,enemy.elite)!=en_pred||std::make_pair(own.regular,own.elite)!=own_post){
                                kstat_fallback("branch_verify",detail_dict({{"a",detail_str(kind)},{"cards",detail_cards(ctx_atk,ctx_def)},{"obs",detail_obs_slice(obs)},{"wctk_h",counters.detail.integer(h9)}}));return false;
                            }
                            branches.push_back({std::move(forced.state),mass*term.second});
                        }
                        continue;
                    }
                }
                auto forced=forced_apply(parent,node->action,node->side,*plan,conf);
                if(forced.mismatch)throw PlanMismatch{forced.message};
                const Snapshot& st=*forced.state;const auto pcb=obs_attr(st,st.root,"pending_combat",false);
                if(!ctx_none(st,pcb)&&st.nodes[pcb].tag==WOTR_DICT&&text_is(st,field(st,pcb,"phase",false),"casualty_allocation")){
                    bool vok;const bool p9=truth(st,field(st,pcb,"_p9_done",false));
                    if(p9){const auto gh=gh_alloc_expected(ctx_atk,ctx_def,obs);vok=pc_int(st,pcb,"def_hits")==gh.first&&pc_int(st,pcb,"atk_hits")==gh.second;}
                    else{Atom def_hits;def_hits.kind=1;def_hits.i=pc_int(st,pcb,"def_hits");Atom atk_hits;atk_hits.kind=1;atk_hits.i=pc_int(st,pcb,"atk_hits");
                        Atom zero;zero.kind=1;zero.i=0;vok=def_hits==obs[17]&&(text_is(st,field(st,pcb,"alloc_stage",false),"attacker")?atk_hits==obs[18]:atk_hits==zero);}
                    if(!vok){kstat_fallback("branch_verify",detail_dict({{"a",detail_str(kind)},{"cards",detail_cards(ctx_atk,ctx_def)},{"obs",detail_obs_slice(obs)},{"p9",counters.detail.boolean(p9)}}));return false;}
                }
                branches.push_back({std::move(forced.state),mass});
            }
            install_kernel_branches(node,branches,cover,kind,std::max(0.0,1.0-cover),static_cast<int64_t>(outs.size()));
            // counterfactual coverage at larger caps (cap-tuning data)
            double run=0.0;
            for(size_t i=0;i<outs.size();++i){run+=outs[i].second;if(i+1==16||i+1==24||i+1==32)counters.dropped_at[static_cast<int64_t>(i+1)].push_back(std::max(0.0,1.0-run));}
            for(int64_t k9:{16,24,32})if(static_cast<int64_t>(outs.size())<k9)counters.dropped_at[k9].push_back(0.0);
            return true;
        }catch(const PlanMismatch& pm){
            // ai.py 15795-15815: diagnostic detail then the counted sampled-path fallback
            const auto cur=have_cur?detail_obs_slice(cur_obs):counters.detail.none();
            const uint64_t plan_repr=have_cur&&cur_oi<plan_reprs.size()?detail_str(plan_reprs[cur_oi].substr(0,120)):counters.detail.none();
            kstat_fallback("plan_mismatch",detail_dict({{"a",detail_str(kind)},{"err",detail_str(pm.message.substr(0,200))},{"cards",detail_cards(ctx_atk,ctx_def)},{"obs",cur},{"ctx",detail_str(ctx_repr.substr(0,700))},{"plan",plan_repr},
                {"atk_army",detail_str(army_dump(parent,info.atk_region).substr(0,400))},{"def_army",detail_str(army_dump(parent,info.def_region).substr(0,400))},{"regions",counters.detail.tuple({detail_str(info.atk_region),detail_str(info.def_region)})}}));
            return false;
        }catch(const Cancelled&){throw;}
        catch(const std::exception& ex){kstat_fallback("error",error_detail(std::string("RuntimeError: ")+std::string(ex.what()).substr(0,200)));throw;}
    }
    // ai.py 14398-14434 _kernel_serve_world.
    Node* kernel_serve_world(Node* chance){
        const auto key=blinded_commit_key(*chance->parent->state);
        if(!key||(chance->has_kernel_key&&chance->kernel_key==*key))return nullptr;
        {World& active=world_slot(chance,bank_key(chance));active.outcomes=chance->outcomes;active.has_cover=chance->has_cover;active.cover=chance->kernel_cover;active.drops=chance->drops;}
        if(const World* ent=find_world(chance,*key)){
            chance->outcomes=ent->outcomes;chance->has_cover=ent->has_cover;chance->kernel_cover=ent->cover;chance->drops=ent->drops;
            chance->has_kernel_key=true;chance->kernel_key=*key;return nullptr;
        }
        overlay_prepare(*chance->parent->state,chance->side,chance);
        std::unique_ptr<Snapshot> probe;
        try{auto applied=apply_detect(*chance->parent->state,chance->action,chance->side,chance);probe=std::move(applied.first);}
        catch(const std::exception& error){kstat_fallback("blind_world_apply",detail_dict({{"key",detail_str(*key)},{"err",detail_str(std::string("RuntimeError: ")+std::string(error.what()).substr(0,200))}}));throw;}
        if(kernel_expand_combat(chance,*probe))return nullptr;
        kstat_fallback("blind_world_expand",detail_dict({{"key",detail_str(*key)}}));
        Node* branch=new_node(std::move(probe),chance->side,chance,ActionRef{},1.0);branch->out_sig=chance_sig(*branch->state);branch->has_sig=true;
        chance->outcomes={branch};chance->has_cover=false;chance->kernel_cover=0.0;chance->drops=0;
        kernel_register_world(chance);
        return branch;
    }

    // ------------------------------------------------ apply (ai.py 14204-14215 + driver.apply/resume)
    // Executor-refusal diagnostics (2026-09-10, no behaviour change): an engine exception thrown inside an apply is re-raised with
    // its text suffixed by ` [apply <kind>/<card_id> depth=<d> besieged={...} pending_combat=<phase|none>]` - the action kind, the
    // row's card_id (when its params carry one), the depth of the applied node (root children = 1) and the PARENT state's siege
    // facts - so a served-game executor refusal names its row without a trace replay.  Text only: every consumer reads last_error
    // (advance -> fail(error)); Refusal (`unported: ...`) and the non-std control-flow exceptions (Cancelled, hf_Fork, PlanMismatch)
    // pass through untouched; the apply_fail / blind_world_apply sites count and rethrow by type, never by message.  The suffix is
    // built under its own catch-all (never a second exception) and the tree stays exportable after the failure (export_tree needs
    // only the attached arena, export_observer accepts Phase::Failed).
    static int node_depth(const Node* node){int d=0;for(const Node* p=node;p&&p->parent;p=p->parent)++d;return d;}
    static std::string apply_context(const Snapshot& parent,const ActionRef& a,const Node* node){
        try{
            std::string out=" [apply "+(a.has()?a.kind():std::string("?"));
            if(a.has()){const Snapshot& snap=a.snap();const auto cid=field(snap,a.params(),"card_id",false);if(cid!=missing&&snap.nodes[cid].tag==WOTR_STRING)out+="/"+std::string(text(snap,cid));}
            out+=" depth="+std::to_string(node_depth(node));
            std::vector<std::string> sieges;const auto bs=obs_attr(parent,parent.root,"besieged_strongholds",false);
            if(!ctx_none(parent,bs))for(auto k:child_edges(parent,bs))if(parent.nodes[k].tag==WOTR_STRING)sieges.emplace_back(text(parent,k));
            std::sort(sieges.begin(),sieges.end());out+=" besieged={";for(size_t i=0;i<sieges.size();++i)out+=(i?", ":"")+sieges[i];out+="}";
            const auto pc=obs_attr(parent,parent.root,"pending_combat",false);std::string phase="none";
            if(!ctx_none(parent,pc)&&parent.nodes[pc].tag==WOTR_DICT){const auto ph=field(parent,pc,"phase",false);phase=(ph!=missing&&parent.nodes[ph].tag==WOTR_STRING)?std::string(text(parent,ph)):std::string("?");}
            return out+" pending_combat="+phase+"]";
        }catch(...){return " [apply context unavailable]";}
    }
    std::pair<std::unique_ptr<Snapshot>,bool> apply_detect(const Snapshot& parent,const ActionRef& a,int actor,const Node* node){
        try{
            if(!apply_fault.empty()&&a.has()&&a.kind()==apply_fault)fail("injected apply fault (WOTR_NATIVE_APPLY_FAULT)");   // test-only, see apply_fault
            return apply_detect_engine(parent,a,actor,node->id);
        }
        catch(const Refusal&){throw;}
        catch(const std::exception& e){fail(std::string(e.what())+apply_context(parent,a,node));}
    }
    std::pair<std::unique_ptr<Snapshot>,bool> apply_detect_engine(const Snapshot& parent,const ActionRef& a,int actor,uint64_t node_id){
        if(cfg.overlay_assert)overlay_assert_state(parent);
        auto pre=rng_clone(rng.get());const std::string kind=a.kind();std::unique_ptr<Snapshot> st;
        if(kind=="end_turn")st=turn_apply(parent,node_id);
        else{Scoped _sp(this,"apply_descriptor");auto fm=apply_native_random_descriptor(parent,*T,a.snap(),rng.get(),actor,a.row);st=std::move(fm.state);rng=std::move(fm.random);rng_mark("apply");}
        {Scoped _sp(this,"resume_preaction");auto res=sr_resume_preaction(parent,*st,*T,rng.get(),kind);st=std::move(res.state);rng=std::move(res.random);rng_mark("resume");}
        Scoped _sp(this,"rng_compare");return {std::move(st),!cf_same_rng(pre.get(),rng.get())};
    }
    // search_turn_adapter/native_turn.py apply(): native continuation + greedy phase picks.
    std::unique_ptr<Snapshot> turn_apply(const Snapshot& parent,uint64_t node_id){
        Scoped _sp(this,"turn_driver");auto driver=sd_start(parent,*T,rng.get());
        while(true){
            rng=rng_clone(driver->random.get());rng_mark("turn");
            if(driver->stage==sd_Stage::Done)return std::make_unique<Snapshot>(*driver->state);
            const int side=driver->side;require(side==0||side==1,"Search turn selector has no side");require(bool(driver->policy),"Search turn selector has no policy menu");
            const int64_t index=greedy_prior_action(*driver->state,side,*driver->policy,node_id);
            auto input=sd_clone(*driver);input->random=rng_clone(rng.get());driver=sd_select(*input,index);
        }
    }
    // ai.py 14437-14464 _greedy_prior_action through native_turn.greedy()/policy_again().
    int64_t greedy_prior_action(const Snapshot& s,int side,const Snapshot& policy,uint64_t node_id){
        NativePolicyProjection again;{Scoped _sp(this,"turn_project");again=project_native_ai_actions(s,policy,side);}
        const auto policy_rows=child_edges(policy,policy.root),selected=child_edges(*again.actions,again.actions->root);
        std::vector<int64_t> projected;
        for(auto r:selected){int64_t found=-1;for(size_t i=0;i<policy_rows.size();++i)if(policy_rows[i]==r){found=static_cast<int64_t>(i);break;}require(found>=0,"Second policy projection invented an action");projected.push_back(found);}
        if(!again.forced_reason.empty()){require(!projected.empty(),"Forced policy without an action");return projected[0];}
        if(projected.empty())return -1;
        if(own&&side!=root_side)return own_turn_action(s,side,*again.actions,projected);
        RequestData r;r.leaf_ids={node_id};r.obs=observe(s,side);action_matrix(s,*again.actions,side,projected.size(),r.actions,r.regions);r.offsets={0,projected.size()};
        ReplyData rep=await_reply(std::move(r),OP_PV,side,PURPOSE_TURN);
        require(rep.priors.size()==projected.size()&&rep.values.size()==1&&rep.conf.size()==1,"Turn PV reply shape mismatch");
        double v=conf_shrink(rep.conf[0],rep.values[0]);
        if(profiles_active){double shp=profile_shaping();if(side!=0)shp=-shp;if(conf_active)last_phi=shp;v=v+shp;}
        (void)bound_value(v);
        size_t best=0;for(size_t j=1;j<rep.priors.size();++j)if(rep.priors[j]>rep.priors[best])best=j;
        return projected[best];
    }

    // ------------------------------------------------ hunt kernel (ai.py 16033-16050, 16119-16248; hunt_search_adapter/native_expansion.py)
    bool hunt_gate(Node* node){
        if(node->action.kind()!="card_option")return true;
        Scoped _sp(this,"value_facts");auto facts=native_search_value_facts(*node->parent->state);return truth(*facts,field(*facts,facts->root,"hunt_card_option_gate"));
    }
    HuntStep hunt_plan_apply(const Snapshot& parent,const ActionRef& a,int side,const std::vector<int64_t>& plan){
        HuntStep out;hf_Scope scope(parent,*H,plan.data(),plan.size());
        try{
            auto result=apply_native_random_descriptor(parent,*T,a.snap(),rng.get(),side,a.row);
            auto settled=sr_resume_preaction(parent,*result.state,*T,result.random.get(),a.kind());
            result.state=std::move(settled.state);result.random=std::move(settled.random);
            require(scope.next==plan.size(),"Hunt plan not fully consumed");require(cf_same_rng(rng.get(),result.random.get()),"raw RNG state mutation in driven Hunt walk");
            out.prob=scope.probability;out.child=std::move(result.state);return out;
        }catch(const hf_Fork&){out.fork=true;out.options=scope.pending.size();out.prob=scope.probability;return out;}
    }
    // ai.py 15405-15420 _kstat_fallback (detail entries bounded at 20000; 'error'/'hunt_error' loud paths re-raise at the call sites).
    // `detail` is a node in counters.detail (Meta) or NO_ID for None.
    void kstat_fallback(const char* reason,uint64_t detail=NO_ID){
        counters.kernel_fallback[reason]+=1;
        if(detail!=NO_ID&&counters.fallback_detail.size()<20000)counters.fallback_detail.push_back({reason,detail});
    }
    uint64_t detail_dict(std::vector<std::pair<std::string,uint64_t>> kv){return counters.detail.dict(kv);}
    uint64_t detail_str(const std::string& s){return counters.detail.str(s);}
    uint64_t detail_opt_str(const std::optional<std::string>& s){return s?counters.detail.str(*s):counters.detail.none();}
    std::pair<int,std::unique_ptr<Snapshot>> hunt_expand_try(Node* node){
        Scoped _sp(this,"hunt_walks");counters.kernel["hunt_class_applies"]+=1;
        Snapshot& parent=*node->parent->state;if(cfg.overlay_assert)overlay_assert_state(parent);
        struct Leaf{std::unique_ptr<Snapshot> state;double prob;};std::vector<Leaf> leaves;std::vector<std::vector<int64_t>> stack{{}};uint64_t walks=0;bool forked=false;
        while(!stack.empty()){
            auto plan=std::move(stack.back());stack.pop_back();++walks;if(walks>4096)fail("hunt walk budget exceeded");
            auto step=hunt_plan_apply(parent,node->action,node->side,plan);rng_mark("hunt_step");
            if(step.fork){forked=true;for(size_t i=0;i<step.options;++i){auto next=plan;next.push_back(static_cast<int64_t>(i));stack.push_back(std::move(next));}}
            else leaves.push_back({std::move(step.child),step.prob});
        }
        counters.kernel["hunt_walks"]+=static_cast<int64_t>(walks);
        if(!forked){
            if(leaves.size()==1&&std::fabs(leaves[0].prob-1.0)<1e-9){counters.kernel["hunt_det"]+=1;return {2,std::move(leaves[0].state)};}
            kstat_fallback("hunt_shape");return {0,nullptr};
        }
        double total=0.0;for(const auto& l:leaves)total+=l.prob;
        if(leaves.empty()||std::fabs(total-1.0)>1e-6){kstat_fallback("hunt_mass");return {0,nullptr};}
        struct Merged{std::vector<uint8_t> sig;std::unique_ptr<Snapshot> state;double mass;};std::vector<Merged> merged;
        for(auto& l:leaves){auto sg=chance_sig(*l.state);auto it=std::find_if(merged.begin(),merged.end(),[&](const Merged& m){return m.sig==sg;});if(it!=merged.end())it->mass+=l.prob;else merged.push_back({std::move(sg),std::move(l.state),l.prob});}
        std::vector<size_t> order(merged.size());for(size_t i=0;i<order.size();++i)order[i]=i;
        std::stable_sort(order.begin(),order.end(),[&](size_t a,size_t b){return -merged[a].mass< -merged[b].mass;});
        const size_t keep=std::min<size_t>(order.size(),static_cast<size_t>(cfg.kernel_k_max));
        double cover=0.0;for(size_t i=0;i<keep;++i)cover+=merged[order[i]].mass;
        const std::string kind=node->action.kind();
        if(1.0-cover>0.25){
            HighDrop h;h.cls="hunt:"+kind;h.dropped=std::nearbyint((1.0-cover)*1e6)/1e6;h.support=static_cast<int64_t>(order.size());h.kept=static_cast<int64_t>(keep);h.action=node->action;
            try{auto facts=native_search_value_facts(parent);h.eyes=typed(*facts,field(*facts,facts->root,"hunt_box_total"),WOTR_INT,"hunt_box_total").integer;const auto pool=field(*facts,facts->root,"hunt_pool_keys");if(!ctx_none(*facts,pool)){h.has_pool=true;for(auto k:child_edges(*facts,pool))h.pool.emplace_back(text(*facts,k));}}catch(const std::exception&){h.has_pool=false;}
            counters.high_drop.push_back(std::move(h));
            if(counters.high_drop.size()>20){std::stable_sort(counters.high_drop.begin(),counters.high_drop.end(),[](const HighDrop& a,const HighDrop& b){return -a.dropped< -b.dropped;});counters.high_drop.resize(20);}
        }
        node->is_chance=true;node->has_outcomes=true;node->outcomes.clear();
        for(size_t i=0;i<keep;++i){auto& m=merged[order[i]];Node* br=new_node(std::move(m.state),node->side,node,ActionRef{},m.mass);br->out_sig=m.sig;br->has_sig=true;node->outcomes.push_back(br);}
        node->drops=static_cast<uint64_t>(cfg.chance_draws);node->has_cover=true;node->kernel_cover=cover;
        counters.kernel["hunt_nodes"]+=1;
        if(counters.dropped_mass.size()<500000){const double dropped=1.0-cover;const double clipped=dropped>0.0?dropped:0.0;counters.dropped_mass.push_back(clipped);counters.support.push_back(static_cast<int64_t>(order.size()));counters.by_class["hunt:"+kind].push_back(clipped);}
        return {1,nullptr};
    }

    // ------------------------------------------------ pruning (search_lattice.prune numeric route, topk 305-316)
    std::vector<size_t> prune_candidates(const Menu& menu,const std::vector<double>& priors,int64_t max_children){
        Scoped _sp(this,"prune");const size_t n=menu.size();counters.lattice["numeric_list_pruning_rows"]+=static_cast<int64_t>(n);
        const bool free_guarantee=cfg.free_cands!=0;
        auto topk=[&](std::vector<size_t> indices,int64_t count){
            std::vector<size_t> chosen;if(count<=0)return chosen;
            if(static_cast<int64_t>(indices.size())<=count)return indices;
            std::stable_sort(indices.begin(),indices.end(),[&](size_t a,size_t b){if(priors[a]!=priors[b])return priors[a]>priors[b];return a>b;});
            chosen.assign(indices.begin(),indices.begin()+static_cast<ptrdiff_t>(count));return chosen;
        };
        std::set<size_t> selected;std::map<int64_t,std::vector<size_t>> groups;std::vector<size_t> free_rows;
        for(size_t i=0;i<n;++i){groups[menu.rows[i].category].push_back(i);if(free_guarantee&&menu.rows[i].is_free)free_rows.push_back(i);}
        for(auto& g:groups)for(auto i:topk(g.second,3))selected.insert(i);
        if(free_guarantee&&!free_rows.empty())for(auto i:topk(free_rows,cfg.free_cands_n))selected.insert(i);
        if(static_cast<int64_t>(selected.size())<max_children){
            std::vector<size_t> remaining;for(size_t i=0;i<n;++i)if(!selected.count(i))remaining.push_back(i);
            for(auto i:topk(remaining,max_children-static_cast<int64_t>(selected.size())))selected.insert(i);
        }
        return std::vector<size_t>(selected.begin(),selected.end());
    }
    void make_children(Node* node,const Menu& menu,const std::vector<double>& priors,int64_t max_children){
        Scoped _sp(this,"make_children");if(max_children<0)max_children=cfg.max_children?cfg.max_children:(int64_t(1)<<30);
        require(priors.size()==menu.size(),"Prior count does not match the menu");
        if(static_cast<int64_t>(menu.size())<=max_children){
            for(size_t i=0;i<menu.size();++i)node->children.push_back(new_node(nullptr,node->side,node,menu.ref(i),priors[i]));
            return;
        }
        const auto selected=prune_candidates(menu,priors,max_children);
        double total=0.0;for(auto i:selected)total+=priors[i];
        // RA: the children of a pruned expansion reference a sub-menu of exactly the selected rows (keys lazy), never the full menu
        auto sub=compact_rows_menu(menu,selected,false);
        for(size_t j=0;j<selected.size();++j){const size_t i=selected[j];const double p=total>0.0?priors[i]/total:1.0/static_cast<double>(selected.size());node->children.push_back(new_node(nullptr,node->side,node,ActionRef{sub,static_cast<int64_t>(j)},p));}
    }

    // ------------------------------------------------ expansion (ai.py 13599-13668 _expand_node, 13885-13921 _defer_expand)
    double expand_node(Node* node,int64_t max_children,bool sample_local){
        int net_side=node->side;
        if(overlay&&node->side==overlay_opp&&node->state)overlay_patch(*node->state,true);
        Policy policy=make_policy(*node->state,node->side);std::shared_ptr<Menu> menu=policy.menu;
        if(menu->size()==0&&!sample_local&&overlay&&node->side==overlay_opp){
            // ai.py 13612-13614: evaluate this sampled world's no-action routing on a scratch copy; the real node keeps no children.
            Node* scratch=scratch_node(std::make_unique<Snapshot>(*node->state),node->side);
            return expand_node(scratch,max_children,true);
        }
        if(menu->size()==0){
            const int oth=1-node->side;Snapshot& s=*node->state;
            if(pending_side(s,-1)<0&&ar_has_unused(s,oth)){
                if(overlay&&oth==overlay_opp&&node->state)overlay_patch(*node->state,true);
                Policy other=make_policy(s,oth);
                if(other.menu->size()>0){node->side=oth;net_side=oth;policy=other;menu=other.menu;}
            }
        }
        if(menu->size()==0)return mcts_eval_state(*node->state,root_side,leaf_corr(node));
        if(!policy.forced.empty()){
            node->forced_policy_reason=policy.forced;node->children.push_back(new_node(nullptr,node->side,node,menu->ref(0),1.0));
            ev("EXPAND",{node_id_for_trace(node),static_cast<int64_t>(menu->size()),0});return 0.0;
        }
        if(own&&node->side!=root_side)return mcts_eval_state(*node->state,root_side,leaf_corr(node));
        if(pending_box)return defer_expand(node,policy,net_side,max_children);
        const auto vec=observe(*node->state,node->side);
        PV pv=get_priors_and_value(*node->state,node->side,*menu,vec,node==root?PURPOSE_ROOT:PURPOSE_LEAF,leaf_corr(node));
        if(node->side!=root_side)remember_opponent_policy(node->side,vec,*menu,pv.mixed,std::string());
        double value=pv.value;
        if(node->side!=root_side)value=mcts_eval_state(*node->state,root_side,leaf_corr(node));
        make_children(node,*menu,pv.priors,max_children);
        ev("EXPAND",{node_id_for_trace(node),static_cast<int64_t>(menu->size()),0});
        return value;
    }
    double defer_expand(Node* node,const Policy& policy,int net_side,int64_t max_children){
        auto rec=std::make_unique<PendingRec>();rec->node=node;rec->menu=policy.menu;rec->net_side=net_side;
        rec->vec=observe(*node->state,node->side);action_matrix(*node->state,*policy.menu->snap,node->side,policy.menu->size(),rec->ae,rec->ri);
        if(profiles_active){rec->has_shp_node=true;double shp=profile_shaping();rec->shp_node=node->side==0?shp:-shp;}
        rec->opp=node->side!=root_side;
        if(rec->opp){rec->root_vec=observe(*node->state,root_side);rec->shp_root=(root_side==0?1.0:-1.0)*profile_shaping();}
        rec->max_children=max_children;PendingRec* raw=rec.get();
        pending_box->recs.push_back(std::move(rec));pending_box->by_node[node]=raw;pending_box->last=raw;
        ev("EXPAND",{node_id_for_trace(node),static_cast<int64_t>(policy.menu->size()),1});
        return std::numeric_limits<double>::quiet_NaN();
    }
    // ai.py 13989-14094 _flush_pending
    void flush_pending(PendingEvals& coll){
        auto& recs=coll.recs;
        // Source tracer semantics: FLUSH_BEGIN/FLUSH_END are recorded for EVERY _flush_pending call, including zero-record flushes.
        ev("FLUSH_BEGIN",{static_cast<int64_t>(recs.size())});
        if(recs.empty()){ev("FLUSH_END",{0});return;}
        Scoped _sp(this,"flush_total");
        counters.search["leaf_batches"]+=1;counters.search["leaf_batched"]+=static_cast<int64_t>(recs.size());
        std::vector<std::pair<int,std::vector<PendingRec*>>> groups;
        for(auto& r:recs){auto it=std::find_if(groups.begin(),groups.end(),[&](const auto& g){return g.first==r->net_side;});if(it==groups.end()){groups.push_back({r->net_side,{}});it=groups.end()-1;}it->second.push_back(r.get());}
        for(auto& g:groups){
            RequestData r;uint64_t off=0;r.offsets.push_back(0);
            for(auto* rec:g.second){r.leaf_ids.push_back(leaf_corr(rec->node));r.obs.insert(r.obs.end(),rec->vec.begin(),rec->vec.end());r.actions.insert(r.actions.end(),rec->ae.begin(),rec->ae.end());r.regions.insert(r.regions.end(),rec->ri.begin(),rec->ri.end());off+=rec->ri.size();r.offsets.push_back(off);}
            ReplyData rep=await_reply(std::move(r),OP_PVM,g.first,PURPOSE_FLUSH_POLICY);
            require(rep.values.size()==g.second.size()&&rep.conf.size()==g.second.size()&&rep.priors.size()==off&&rep.mixed.size()==off,"Deferred policy/value reply shape mismatch");
            size_t start=0;
            for(size_t j=0;j<g.second.size();++j){auto* rec=g.second[j];const size_t rows=rec->ri.size();
                rec->priors.assign(rep.priors.begin()+static_cast<ptrdiff_t>(start),rep.priors.begin()+static_cast<ptrdiff_t>(start+rows));
                rec->mixed.assign(rep.mixed.begin()+static_cast<ptrdiff_t>(start),rep.mixed.begin()+static_cast<ptrdiff_t>(start+rows));
                rec->v=rep.values[j];rec->conf_v=rep.conf[j];start+=rows;}
        }
        std::vector<PendingRec*> opp;for(auto& r:recs)if(r->opp)opp.push_back(r.get());
        if(!opp.empty()){
            RequestData r;r.offsets.assign(opp.size()+1,0);/* ABI: EVAL offsets are [B+1] zeros */
            for(auto* rec:opp){r.leaf_ids.push_back(leaf_corr(rec->node));r.obs.insert(r.obs.end(),rec->root_vec.begin(),rec->root_vec.end());}
            ReplyData rep=await_reply(std::move(r),OP_EVAL,root_side,PURPOSE_FLUSH_ROOT);
            require(rep.values.size()==opp.size()&&rep.conf.size()==opp.size(),"Deferred root value reply shape mismatch");
            for(size_t j=0;j<opp.size();++j){opp[j]->v_root=rep.values[j];opp[j]->conf_root=rep.conf[j];}
        }
        Scoped _spb(this,"flush_bookkeeping");
        for(auto& rp:recs){
            PendingRec& r=*rp;Node* node=r.node;const Menu& menu=*r.menu;
            std::vector<double> priors(r.priors.begin(),r.priors.end());
            last_conf=1.0;last_phi=0.0;
            double v=conf_shrink(r.conf_v,r.v);
            if(r.has_shp_node){if(conf_active)last_phi=r.shp_node;v=v+r.shp_node;}
            double value=v;
            if(r.opp)remember_opponent_policy(node->side,r.vec,menu,r.mixed,std::string());
            if(r.opp){last_conf=1.0;if(conf_active)last_phi=r.shp_root;value=conf_shrink(r.conf_root,r.v_root)+r.shp_root;}
            value=bound_value(value);
            const double conf=last_conf,phi=last_phi;
            make_children(node,menu,priors,r.max_children);
            for(auto& path:r.paths)for(Node* n:path){
                const double s_leaf=n->side==root_side?value:-value;const double s_virt=n->side==root_side?-1.0:1.0;
                n->value_sum+=s_leaf-s_virt;n->value_sq+=value*value;
                if(conf_active){n->conf_sum+=conf;n->phi_sum+=phi;}
            }
        }
        ev("FLUSH_END",{static_cast<int64_t>(recs.size())});
    }

    // ------------------------------------------------ learned opponent draw (ai.py 17019-17063 + search_edge_catalog.choose)
    size_t choices(const std::vector<double>& weights){
        require(!weights.empty(),"Empty choices population");std::vector<double> cumulative(weights.size());
        cumulative[0]=weights[0];for(size_t i=1;i<weights.size();++i)cumulative[i]=cumulative[i-1]+weights[i];
        uint64_t out=0;if(wotr_random_choices(rng.get(),weights.size(),cumulative.data(),cumulative.size(),1,&out,1))throw std::invalid_argument(wotr_random_last_error());rng_mark("choices");
        return static_cast<size_t>(out);
    }
    // Work package RA (native_runaway, 2026-09-11): the frozen search_edge_catalog keeps a COMPACT payload per record (the pickled
    // action fields) and never the sampled action list; the records one draw creates therefore share a sub-menu copied from exactly
    // those rows (native_menu_rows' copy; RowMeta and the computed keys carried over), so the draw Menu - its descriptor snapshot and
    // all its row keys - is released when the draw returns instead of living for the whole search.
    // `with_keys`: the records of a draw carry the keys already computed on the draw menu; the children of a pruned expansion (F5)
    // leave them lazy - a later ensure_keys() on the sub-menu derives the identical strings from the identical row content.
    std::shared_ptr<const Menu> compact_rows_menu(const Menu& menu,const std::vector<size_t>& idx,bool with_keys){
        auto m=std::make_shared<Menu>();m->ctl=this;m->rows.reserve(idx.size());if(with_keys)m->keys.reserve(idx.size());
        mv_Builder b(*menu.snap);std::vector<uint64_t> rows;rows.reserve(idx.size());
        for(size_t i:idx){rows.push_back(b.raw(mv_row_node(*menu.snap,static_cast<int64_t>(i))));m->rows.push_back(menu.rows[i]);if(with_keys)m->keys.push_back(menu.key(i));}
        b.out.root=b.container(WOTR_LIST,rows);m->snap=std::shared_ptr<const Snapshot>(ac_compact(b.out));m->keys_ready=with_keys;return m;
    }
    Node* materialize(Edge& edge,Node* node,const ActionRef& action){
        if(edge.child==nullptr){edge.child=new_node(nullptr,node->side,node,action,edge.prior);counters.edge["materialized"]+=1;}
        else edge.child->action=action;
        return edge.child;
    }
    void edge_flush(Node* node){
        Catalog* cat=node->catalog.get();if(cat==nullptr||cat->disabled)return;
        std::vector<Node*> children;for(auto& e:cat->records)children.push_back(materialize(e,node,e.payload));
        node->children=children;cat->physical=children;cat->disabled=true;counters.edge["catalog_fallbacks"]+=1;
    }
    Node* edge_choose(Node* node,const Menu& menu,const std::vector<double>& probabilities,bool forced,size_t& chosen_row){
        Scoped _sp(this,"edge_choose");if(!node->catalog)node->catalog=std::make_unique<Catalog>();Catalog& cat=*node->catalog;menu.ensure_keys();
        if(cat.disabled){chosen_row=forced?0:choices(probabilities);return full_choice_after(node,menu,probabilities,forced,chosen_row);}
        if(node->children.size()<cat.physical.size())fail("Snapshot edge catalog requires append-only child list");
        for(size_t i=0;i<cat.physical.size();++i)if(node->children[i]!=cat.physical[i])fail("Snapshot edge catalog requires append-only child list");
        std::vector<Node*> new_children(node->children.begin()+static_cast<ptrdiff_t>(cat.physical.size()),node->children.end());
        for(Node* c:new_children)if(!c->action.has()){edge_flush(node);node->children.insert(node->children.end(),new_children.begin(),new_children.end());chosen_row=forced?0:choices(probabilities);return full_choice_after(node,menu,probabilities,forced,chosen_row);}
        cat.records.reserve(cat.records.size()+new_children.size()+menu.size());cat.index.reserve(cat.records.size()+new_children.size()+menu.size());
        for(Node* c:new_children){Edge e;e.prior=c->prior;e.child=c;e.payload=c->action;cat.records.push_back(std::move(e));cat.index[cat.records.back().key()]=cat.records.size()-1;}   // dict overwrite semantics: a later edge with the same key wins
        cat.physical=node->children;
        // RA: the rows that get a record = the rows whose key is not in the catalog, first occurrence per key (the single-pass
        // semantics of the loop this replaces); their records share one compact sub-menu and the index views ITS key storage.
        std::vector<size_t> fresh_rows;{std::unordered_set<std::string_view,SvHash> seen;
            for(size_t i=0;i<menu.size();++i){const std::string& key=menu.key(i);if(cat.index.find(key)==cat.index.end()&&seen.insert(std::string_view(key)).second)fresh_rows.push_back(i);}}
        const int64_t created=static_cast<int64_t>(fresh_rows.size());
        if(!fresh_rows.empty()){auto sub=compact_rows_menu(menu,fresh_rows,true);
            for(size_t j=0;j<fresh_rows.size();++j){Edge e;e.prior=probabilities[fresh_rows[j]];e.child=nullptr;e.payload=ActionRef{sub,static_cast<int64_t>(j)};cat.records.push_back(std::move(e));cat.index.emplace(std::string_view(cat.records.back().key()),cat.records.size()-1);}}
        counters.edge["snapshots_created"]+=created;
        chosen_row=forced?0:choices(probabilities);
        Edge& edge=cat.records[cat.index.at(menu.key(chosen_row))];const bool fresh=edge.child==nullptr;
        Node* child=materialize(edge,node,edge.payload);   // RA: the frozen _ordinary_materialize rebuilds the child's Action from edge.payload, never from the sampled list
        if(fresh){node->children.push_back(child);cat.physical.push_back(child);}
        counters.edge["selections"]+=1;return child;
    }
    Node* full_choice_after(Node* node,const Menu& menu,const std::vector<double>& probabilities,bool forced,size_t chosen){
        // search_edge_catalog.full_choice; the caller pre-drew the index so the chosen row can be reported.
        std::unordered_map<std::string_view,Node*,SvHash> existing;for(Node* c:node->children)if(c->action.has())existing[c->action.key()]=c;
        std::vector<Node*> eligible;eligible.reserve(menu.size());
        for(size_t i=0;i<menu.size();++i){const std::string& key=menu.key(i);auto it=existing.find(key);Node* child;
            if(it==existing.end()){child=new_node(nullptr,node->side,node,menu.ref(i),probabilities[i]);node->children.push_back(child);existing[key]=child;}else child=it->second;
            eligible.push_back(child);}
        return eligible[forced?0:chosen];
    }
    Node* learned_opponent_child(Node* node){
        overlay_patch(*node->state,false);
        Policy policy=make_policy(*node->state,node->side);const Menu& menu=*policy.menu;
        if(menu.size()==0){
            Snapshot& s=*node->state;
            if(pending_side(s,-1)>=0||!attr_none(s,"_ring_free_side")||!attr_none(s,"pending_army_move_die")||!attr_none(s,"pending_muster_die"))fail("Empty sampled menu during required decision");
            const int other=1-node->side;Node* end_turn=nullptr;
            for(Node* c:node->children)if(c->action.has()&&c->action.kind()=="end_turn"){end_turn=c;break;}
            if(end_turn&&make_policy(s,other).menu->size()==0)return end_turn;
            counters.overlay["no_action_samples"]+=1;return nullptr;
        }
        std::vector<double> probabilities;bool cache_hit=false;const bool forced=!policy.forced.empty();
        if(forced){if(menu.size()!=1)fail("Fixed AI policy must expose exactly one transition");probabilities={1.0};}
        else{
            const auto vec=observe(*node->state,node->side);const std::string key=opponent_policy_key(node->side,vec,menu);
            auto it=policy_cache.find(key);
            if(it!=policy_cache.end()){probabilities=it->second;cache_hit=true;}
            else{
                const double saved_conf=last_conf,saved_phi=last_phi;PV pv;
                try{pv=get_priors_and_value(*node->state,node->side,menu,vec,PURPOSE_OPPONENT,leaf_corr(node));}
                catch(...){last_conf=saved_conf;last_phi=saved_phi;throw;}
                last_conf=saved_conf;last_phi=saved_phi;
                probabilities=remember_opponent_policy(node->side,vec,menu,pv.mixed,key);
            }
            counters.overlay["learned_policy_draws"]+=1;
        }
        observe_card_policy(node,policy.menu,probabilities);
        size_t chosen_row=0;Node* child=edge_choose(node,menu,probabilities,forced,chosen_row);
        ev("OPPONENT_DRAW",{static_cast<int64_t>(node->id),static_cast<int64_t>(chosen_row),cache_hit?1:0});
        return child;
    }

    // ai.py 17057 _segment_observe (EntryAdapter.install observe): a source-segmented (lattice) menu asserts no pending
    // combat and skips; every flat menu reaches observer.card_policy -> recorded for the Python replay (SHARED_ABI 9.3).
    // Segmentation (PartitionQuery.wrap) needs the synchronous route and a lattice context with blocks; blocks exist
    // only for companion_separate rows, so the native context is consulted only when the menu has such rows.
    void observe_card_policy(Node* node,const std::shared_ptr<Menu>& menu,const std::vector<double>& probabilities){
        bool segmented=false;
        if(pending_box==nullptr){
            bool separation=false;for(const auto& row:menu->rows)if(row.kind=="companion_separate"){separation=true;break;}
            if(separation){auto ctx=native_search_lattice_context(*node->state,*T,*menu->snap);const auto parts=child_edges(*ctx,ctx->root);segmented=truth(*ctx,parts[0])&&ctx->nodes[parts[1]].count>0;}
        }
        if(segmented){require(ctx_none(*node->state,obs_attr(*node->state,node->state->root,"pending_combat",false)),"Segmented opponent menu with a pending combat");return;}
        // RA F6: the frozen CombatForecastObserver.card_policy returns before reading the rows unless the node's pending combat is in a
        // combat-card phase (the orchestrator's replay rebuilds exactly that phase from the exported phase node), so an event of any other
        // phase was exported, decoded (one tuple per menu row: ~3.2M rows at a 1000-simulation mid-game root) and discarded; record only
        // the events the observer can use - the observer's state after the replay is identical.
        {const Snapshot& s=*node->state;const auto pc=obs_attr(s,s.root,"pending_combat",false);if(ctx_none(s,pc)||s.nodes[pc].tag!=WOTR_DICT)return;
         const auto ph=field(s,pc,"phase",false);if(ph==missing||s.nodes[ph].tag!=WOTR_STRING)return;
         if(!(text_is(s,ph,"attacker_combat_card")||text_is(s,ph,"defender_combat_card")))return;}
        ObsEvent e;e.kind=2;e.node=node;e.probabilities=probabilities;e.kinds.reserve(menu->size());
        for(size_t i=0;i<menu->size();++i){const std::string& kind=menu->rows[i].kind;auto it=obs_kind_index.find(kind);
            if(it==obs_kind_index.end()){it=obs_kind_index.emplace(kind,static_cast<uint32_t>(obs_kind_table.size())).first;obs_kind_table.push_back(kind);}
            e.kinds.push_back(it->second);const auto& t=menu->snap->nodes[menu->node(i)];const auto cid=field(*menu->snap,menu->snap->edges[t.offset+2],"card_id",false);
            if(cid!=missing){Meta mc;const uint64_t n=mc.copy(*menu->snap,cid);e.cids.emplace_back(static_cast<uint32_t>(i),std::shared_ptr<const Snapshot>(mc.finish(n)));}}
        observer_events.push_back(std::move(e));
    }
    // CombatForecastObserver.root_branch: the root child on the node's path (None for the root itself).  A detached child
    // is parentless but still one of root.children (the source's post-stash tree), so the walk stops at any parentless node
    // or at a direct root child and then resolves it through root.children.
    std::optional<size_t> root_branch_row(const Node* node)const{
        if(node==root)return std::nullopt;
        const Node* cur=node;while(cur->parent!=nullptr&&cur->parent!=root)cur=cur->parent;
        for(size_t i=0;i<root->children.size();++i)if(root->children[i]==cur)return i;
        return std::nullopt;
    }

    // ------------------------------------------------ selection (ai.py 17066-17142 _select_child, 14282-14361 _select_outcome)
    Node* select_child(Node* node){
        const bool masked=overlay&&node->side==overlay_opp&&node->state;
        if(!masked)edge_flush(node);
        if(masked)return own?own_child(node):learned_opponent_child(node);
        Scoped _sp(this,"select_child");const double pscale=1.0;
        std::vector<Node*> cand;for(Node* c:node->children)if(c->proven==0)cand.push_back(c);
        if(cand.empty())cand=node->children;
        const double sqrt_parent=node->visit_count>0?std::sqrt(static_cast<double>(node->visit_count)):1.0;
        double best_score=-std::numeric_limits<double>::infinity();Node* best=nullptr;
        for(Node* child:cand){
            const double q=child->side==node->side?child->q_value():-child->q_value();
            const double u=cfg.c_puct*child->prior*pscale*sqrt_parent/(1.0+static_cast<double>(child->visit_count));
            const double score=q+u;if(score>best_score){best_score=score;best=child;}
        }
        if(best)ev("SELECT",{static_cast<int64_t>(node->id),static_cast<int64_t>(best->id),static_cast<int64_t>(node->visit_count)});
        return best;
    }
    Node* select_outcome(Node* chance){
        Scoped _sp(this,"select_outcome");
        if(chance->round_sampling){
            // ai.py 14296-14312: keep sampling the true phase transition on every visit; over-cap outcomes are transient.
            overlay_prepare(*chance->parent->state,chance->side,chance);
            auto applied=apply_detect(*chance->parent->state,chance->action,chance->side,chance);
            auto sig=chance_sig(*applied.first);
            for(Node* b:chance->outcomes)if(b->has_sig&&b->out_sig==sig){b->prior+=1.0;return b;}
            const bool keep=chance->outcomes.size()<static_cast<size_t>(cfg.chance_k_max);
            Node* branch=keep?new_node(std::move(applied.first),chance->side,chance,ActionRef{},1.0):scratch_node(std::move(applied.first),chance->side);
            if(!keep){branch->parent=chance;}   // transient: parent link for settle, never in the outcome list / export
            branch->out_sig=std::move(sig);branch->has_sig=true;
            if(keep)chance->outcomes.push_back(branch);
            return branch;
        }
        if(chance->has_kernel_sets){Node* first=kernel_serve_world(chance);if(first)return first;}
        auto& outs=chance->outcomes;
        double drawn=0.0;for(Node* b:outs)drawn+=b->prior;drawn+=static_cast<double>(chance->drops);
        if(chance->drops>=static_cast<uint64_t>(cfg.chance_draws))drawn=static_cast<double>(cfg.chance_draws);
        if(drawn<static_cast<double>(cfg.chance_draws)){
            overlay_prepare(*chance->parent->state,chance->side,chance);
            std::unique_ptr<Snapshot> st;bool used=false;
            try{auto applied=apply_detect(*chance->parent->state,chance->action,chance->side,chance);st=std::move(applied.first);used=applied.second;}
            catch(...){counters.search["apply_fail"]+=1;throw;}
            (void)used;
            auto sg=chance_sig(*st);
            const auto ms=chance->action.kind()=="end_turn"?roll_ms(*st):std::nullopt;
            Node* hit=nullptr;for(Node* b:outs)if(b->has_sig&&b->out_sig==sg){hit=b;break;}
            if(hit){if(!ms)hit->prior+=1.0;else chance->drops+=1;}
            else if(outs.size()<static_cast<size_t>(cfg.chance_k_max)){
                Node* branch=new_node(std::move(st),chance->side,chance,ActionRef{},ms?roll_ms_prob(*ms):1.0);
                branch->out_sig=std::move(sg);branch->has_sig=true;outs.push_back(branch);if(ms)chance->drops+=1;return branch;
            }else chance->drops+=1;
            if(outs.empty())return nullptr;
        }
        double tot=0.0;for(Node* b:outs)tot+=b->prior;
        // MC4: a keyed node's visit_count spans every world; the ACTIVE set's own visit total keeps the descent share exact per world.
        uint64_t active_visits=0;if(chance->has_kernel_sets)for(Node* b:outs)active_visits+=b->visit_count;
        const double n_vis=static_cast<double>(chance->has_kernel_sets?active_visits:chance->visit_count);
        Node* best=outs[0];double best_key=static_cast<double>(best->visit_count)-n_vis*(best->prior/tot);
        for(size_t i=1;i<outs.size();++i){const double key=static_cast<double>(outs[i]->visit_count)-n_vis*(outs[i]->prior/tot);if(key<best_key){best_key=key;best=outs[i];}}
        return best;
    }

    // ------------------------------------------------ settle (ai.py 16921-16977)
    void mark_terminal(Node* node,int winner,int kind,int pre_side){
        node->is_terminal=true;node->terminal_value=search_terminal(winner,kind);node->proven=proof_mark(winner,kind);node->proven_value=node->terminal_value;node->side=pre_side;
        ev("SETTLE",{static_cast<int64_t>(node->id),node->side,1});
    }
    double settle_new_node(Node* node,int pre_side){
        const Snapshot& s=*node->state;
        const auto etv=obs_attr(s,s.root,"_search_endturn_victory",false);
        if(!ctx_none(s,etv)){
            typed(s,etv,WOTR_DICT,"search victory marker must be dictionary");
            const auto winner_node=field(s,etv,"victory"),kind_node=field(s,etv,"victory_type");
            const int winner=text_is(s,winner_node,"free_peoples")?0:text_is(s,winner_node,"shadow")?1:-1;
            const int kind=victory_kind(text(s,kind_node));if(winner<0||kind==0)fail("Invalid victory route in end-turn marker");
            mark_terminal(node,winner,kind,pre_side);return node->terminal_value;
        }
        const Victory v=victory(s);
        if(v.ended){mark_terminal(node,v.winner,v.kind,pre_side);return node->terminal_value;}
        const ActionRef* ea=node->action.has()?&node->action:(node->parent&&node->parent->action.has()?&node->parent->action:nullptr);
        int next_side;{Scoped _sp(this,"settle_actor");next_side=ea&&ea->kind()=="end_turn"?sd_turn_start_actor(s):native_next_actor(s,pre_side);}
        if(next_side<0){
            if(!attr_true(s,"dice_rolled")&&(attr_true(s,"_declare_phase")||attr_true(s,"_eye_allocation_phase")||attr_true(s,"_fellowship_tail_phase")))fail("Unfinished preaction phase cannot become end_turn");
            node->side=pre_side;
            const double value=mcts_eval_state(s,root_side,leaf_corr(node));
            if(cfg.turn_horizon>0&&crossings(s)<cfg.turn_horizon)node->children.push_back(new_node(nullptr,pre_side,node,end_turn_action(),1.0));
            ev("SETTLE",{static_cast<int64_t>(node->id),node->side,0});return value;
        }
        node->side=next_side;ev("SETTLE",{static_cast<int64_t>(node->id),node->side,0});
        return expand_node(node,-1,false);
    }
    void propagate_proofs(const std::vector<Node*>&){/* ai.py 16845: learned_policy returns immediately */}

    // ------------------------------------------------ one simulation (ai.py 17445-17663 _simulate_from)
    void simulate_from(Node* node,std::vector<Node*> path){
        last_conf=1.0;last_phi=0.0;
        sample_overlay();
        bool leaf_term=false;double leaf_value=0.0;
        while(true){
            if(node->is_terminal){leaf_value=node->terminal_value;leaf_term=true;break;}
            if(node->proven!=0){leaf_value=node->proven_value;leaf_term=true;break;}
            if(node->is_chance){
                Node* branch=select_outcome(node);if(!branch)fail("Chance search produced no valid outcome");
                ev("CHANCE_SELECT",{static_cast<int64_t>(node->id),static_cast<int64_t>(branch->id)});
                path.push_back(branch);node=branch;
                if(!node->is_terminal&&node->children.empty()&&node->visit_count==0){
                    leaf_value=settle_new_node(node,node->parent->side);leaf_term=node->is_terminal;
                    if(!node->forced_policy_reason.empty()&&!leaf_term)continue;
                    break;
                }
                continue;
            }
            if(!node->children.empty()||(overlay&&node->state&&node->side==overlay_opp)){
                if(node==root&&!force_idx.empty()){
                    size_t fp9=NO_ID;
                    for(size_t i9:force_idx){Node* c9=root->children[i9];
                        if(c9->proven==0&&c9->visit_count<static_cast<uint64_t>(cfg.free_force)&&(fp9==NO_ID||c9->visit_count<root->children[fp9]->visit_count))fp9=i9;}
                    if(fp9!=NO_ID){root->children[fp9]->forced_visits+=1;node=root->children[fp9];path.push_back(node);continue;}
                }
                Node* selected=select_child(node);
                if(selected==nullptr){
                    // ai.py 17480-17492: fresh-node no-action routing in this sampled world (scratch node, discarded).
                    Node* scratch=scratch_node(std::make_unique<Snapshot>(*node->state),node->side);
                    leaf_value=expand_node(scratch,-1,true);
                    if(!scratch->forced_policy_reason.empty())fail("Unexpected pending policy in empty ordinary menu");
                    break;
                }
                node=selected;path.push_back(node);continue;
            }
            if(!node->state){
                overlay_prepare(*node->parent->state,node->side,node);
                const std::string kind=node->action.kind();
                if(is_hunt_parent(kind)&&hunt_gate(node)){
                    auto hunted=hunt_expand_try(node);
                    if(hunted.first==1){
                        Node* branch=node->outcomes[0];path.push_back(branch);
                        leaf_value=settle_new_node(branch,node->side);leaf_term=branch->is_terminal;
                        if(!branch->forced_policy_reason.empty()&&!leaf_term){node=branch;continue;}
                        break;
                    }
                    if(hunted.first==2){
                        node->state=std::move(hunted.second);
                        leaf_value=settle_new_node(node,node->parent->side);leaf_term=node->is_terminal;
                        if(!node->forced_policy_reason.empty()&&!leaf_term)continue;
                        break;
                    }
                }
                const auto pl35=overlay?std::optional<std::array<int64_t,5>>(hidden_zone_lens(*node->parent->state)):std::nullopt;
                std::unique_ptr<Snapshot> new_state;bool used_rng=false;
                try{auto applied=apply_detect(*node->parent->state,node->action,node->side,node);new_state=std::move(applied.first);used_rng=applied.second;}
                catch(...){counters.search["apply_fail"]+=1;throw;}
                if(!used_rng&&pl35&&hidden_zone_lens(*new_state)!=*pl35){used_rng=true;counters.overlay["det_draw_chance"]+=1;}
                if(used_rng||blinded_commit_key(*node->parent->state).has_value()){
                    node->is_chance=true;node->has_outcomes=true;node->outcomes.clear();
                    if(is_kernel_parent(kind)&&kernel_expand_combat(node,*new_state)){   // 9.5 hook (refuses today)
                        Node* branch=node->outcomes[0];path.push_back(branch);
                        leaf_value=settle_new_node(branch,node->side);leaf_term=branch->is_terminal;
                        if(!branch->forced_policy_reason.empty()&&!leaf_term){node=branch;continue;}
                        break;
                    }
                    const auto ms0=kind=="end_turn"?roll_ms(*new_state):std::nullopt;
                    Node* branch=new_node(std::move(new_state),node->side,node,ActionRef{},ms0?roll_ms_prob(*ms0):1.0);
                    if(ms0)node->drops+=1;
                    branch->out_sig=chance_sig(*branch->state);branch->has_sig=true;node->outcomes.push_back(branch);
                    kernel_register_world(node);
                    path.push_back(branch);
                    leaf_value=settle_new_node(branch,node->side);leaf_term=branch->is_terminal;
                    if(!branch->forced_policy_reason.empty()&&!leaf_term){node=branch;continue;}
                    break;
                }
                node->state=std::move(new_state);
                leaf_value=settle_new_node(node,node->parent->side);leaf_term=node->is_terminal;
                if(!node->forced_policy_reason.empty()&&!leaf_term)continue;
                break;
            }
            if(pending_box){auto it=pending_box->by_node.find(node);if(it!=pending_box->by_node.end()){pending_box->last=it->second;leaf_value=std::numeric_limits<double>::quiet_NaN();break;}}
            leaf_value=mcts_eval_state(*node->state,root_side,leaf_corr(node));break;
        }
        if(leaf_value!=leaf_value){
            require(pending_box&&pending_box->last,"Deferred leaf without a pending collector");
            pending_box->last->paths.push_back(path);
            for(Node* n:path){n->visit_count+=1;n->value_sum+=(n->side==root_side?-1.0:1.0);}
            if(trace){Event e;e.kind="SIM_RETURN";e.i={static_cast<int64_t>(simulations_run),0,1};e.d={0.0,last_conf,last_phi};for(Node* n:path)e.ids.push_back(n->id);e.leaf_none=true;events.push_back(std::move(e));}
            ++simulations_run;return;
        }
        for(auto it=path.rbegin();it!=path.rend();++it){Node* n=*it;
            n->visit_count+=1;
            if(n->side==root_side)n->value_sum+=leaf_value;else n->value_sum-=leaf_value;
            n->value_sq+=leaf_value*leaf_value;
            if(leaf_term)n->term_visits+=1;
            if(conf_active){n->conf_sum+=leaf_term?1.0:last_conf;n->phi_sum+=last_phi;}
        }
        if(leaf_term)propagate_proofs(path);
        if(trace){Event e;e.kind="SIM_RETURN";e.i={static_cast<int64_t>(simulations_run),leaf_term?1:0,0};e.d={leaf_value,last_conf,last_phi};for(Node* n:path)e.ids.push_back(n->id);events.push_back(std::move(e));}
        ++simulations_run;
    }

    // ------------------------------------------------ budget + main loop (ai.py 13860-13873, 17331-17884)
    int64_t sims_budget(int64_t num,int64_t k)const{
        if(cfg.sims_menu_scale)return std::min(num,std::max(cfg.sims_floor,cfg.sims_per_child*k));
        if(k<=6)return std::min(num,std::max<int64_t>(120,60*k));
        return num;
    }
    void search(){
        // ai.py 17404-17435: clone_state, plus ROOT BLINDING (commit slot + nonce) when the root is the blinded defender = so_root_clone.
        rng_mark("create");
        if(!reused_root){
            {Scoped _sp(this,"action_clone");root=new_node(root_blinded?so_root_clone(*input,root_side):action_clone(*input),root_side,nullptr,ActionRef{},0.0);}rng_mark("after_root_clone");
            expand_node(root,-1,false);root->visit_count=1;
        }
        // 9.6 / ai.py 17423-17426: a reused root keeps its children, priors, visit_count, forced_visits and subtree;
        // no clone, no root expansion request, no visit_count reset (attach_session already set root->parent = None).
        if(root->children.empty()){budget=0;return;}
        for(Node* c:root->children)priors_clean.push_back(static_cast<float>(c->prior));   // 17440: CLEAN priors before root noise (reused roots: the retained child priors)
        if(cfg.add_noise&&!reused_root)root_noise_exchange();   // 9.4 / 17493 `add_noise and not _reused`: Python draws the Dirichlet noise on its numpy stream and applies _apply_root_noise
        if(cfg.free_force>0){
            std::vector<std::pair<float,size_t>> fr;
            for(size_t i=0;i<root->children.size();++i){Node* c=root->children[i];if(c->action.has()&&is_free_action(c->action.kind()))fr.push_back({priors_clean[i],i});}
            std::stable_sort(fr.begin(),fr.end(),[](const auto& a,const auto& b){if(a.first!=b.first)return a.first>b.first;return a.second>b.second;});
            for(size_t i=0;i<fr.size()&&static_cast<int64_t>(i)<cfg.free_cands_n;++i)force_idx.push_back(fr[i].second);
        }
        budget=static_cast<uint64_t>(sims_budget(cfg.num_simulations,static_cast<int64_t>(root->children.size())));
        const int64_t num_simulations=static_cast<int64_t>(budget);
        if(cfg.leaf_batch>1){
            int64_t done=0;
            while(done<num_simulations&&root->proven==0){
                const int64_t k=std::min(cfg.leaf_batch,num_simulations-done);PendingEvals coll;pending_box=&coll;
                try{for(int64_t i=0;i<k;++i){if(root->proven!=0)break;simulate_from(root,{root});++done;}}
                catch(...){pending_box=nullptr;throw;}
                pending_box=nullptr;flush_pending(coll);
            }
        }else{
            for(int64_t sim=0;sim<num_simulations;++sim){if(root->proven!=0)break;simulate_from(root,{root});}
        }
        overlay.reset();
    }

    // ------------------------------------------------ tree reuse session (SHARED_ABI 9.6; ai.py 17361-17366, 17399-17431, 17843-17859)
    // Views stay readable after a detach (the retained nodes are shared with the Session); they close only once another
    // controller adopted the subtree (its nodes were moved and renumbered there).
    void require_attached()const{require(!(detached_store&&detached_store->adopted),"The detached subtree was adopted by another controller; root/tree/observer views of this controller are closed");}
    // Export order = node id (creation order).  After a detach the retained nodes live in the shared store; nodes with
    // NO_ID (children materialized by a later NON-adopted entry flush) did not exist during this search and are skipped.
    std::vector<const Node*> exportable_nodes()const{
        std::vector<const Node*> ordered;ordered.reserve(arena.size()+(detached_store?detached_store->nodes.size():0));
        auto place=[&](const Node* n){if(!n||n->id==NO_ID)return;if(n->id>=ordered.size())ordered.resize(n->id+1,nullptr);require(ordered[n->id]==nullptr,"Duplicate node id in export");ordered[n->id]=n;};
        for(const auto& np:arena)place(np.get());
        if(detached_store)for(const auto& np:detached_store->nodes)place(np.get());
        for(const Node* n:ordered)require(n!=nullptr,"Node id gap in export");
        return ordered;
    }
    // Every node reachable from `start` through children, the active outcome bank and every saved kernel_sets bank,
    // plus every Menu referenced by their actions and edge-catalog records.
    static void collect_subtree(Node* start,std::vector<Node*>& nodes,std::vector<std::shared_ptr<const Menu>>& menus){
        std::unordered_set<Node*> seen;std::vector<Node*> stack{start};
        while(!stack.empty()){
            Node* n=stack.back();stack.pop_back();if(!seen.insert(n).second)continue;nodes.push_back(n);
            if(n->action.owner)menus.push_back(n->action.owner);
            if(n->catalog)for(const Edge& e:n->catalog->records)if(e.payload.owner)menus.push_back(e.payload.owner);
            for(Node* c:n->children)stack.push_back(c);
            for(Node* c:n->outcomes)stack.push_back(c);
            for(auto& bank:n->kernel_sets)for(Node* c:bank.second.outcomes)stack.push_back(c);
        }
    }
    // _stash_tree 17843-17859 over the finished controller: move root.children[idx]'s subtree out of the arena.
    // NULL + last_error when the child has no state or is a chance node (the source stashes nothing then).
    std::unique_ptr<Session> detach(uint64_t idx){
        {std::lock_guard<std::mutex> lk(mu);require(phase==Phase::Complete,"Search is not complete");}
        require(!detached,"A subtree was already detached from this controller");
        require(root!=nullptr&&idx<root->children.size(),"child_index is out of range");
        Node* child=root->children[static_cast<size_t>(idx)];
        require(bool(child->state),"Root child has no state (child.state is None): not stashable");
        require(!child->is_chance,"Root child is a chance node: not stashable");
        std::vector<Node*> nodes;std::vector<std::shared_ptr<const Menu>> menus;collect_subtree(child,nodes,menus);
        std::unordered_set<Node*> keep(nodes.begin(),nodes.end());
        size_t owned=0;for(const auto& np:arena)if(keep.count(np.get()))++owned;
        require(owned==keep.size(),"Detached subtree references nodes outside the controller arena");
        auto session=std::make_unique<Session>();session->store=std::make_shared<SessionStore>();
        std::vector<std::unique_ptr<Node>> rest;rest.reserve(arena.size()-owned);session->store->nodes.reserve(owned);
        for(auto& np:arena){if(keep.count(np.get()))session->store->nodes.push_back(std::move(np));else rest.push_back(std::move(np));}   // arena order = creation order; ids unchanged
        arena=std::move(rest);
        child->parent=nullptr;session->root=child;           // 17852 child.parent = None: the upward link is cut; root.children keeps the child
        for(auto& m:menus)m->ctl=nullptr;                      // the old controller may die before the session
        detached=true;detached_store=session->store;detached_child=child;
        return session;
    }
    // create_reusing: entry flush (17361-17366) of the cached root's opponent edge catalog, whether or not adopting, then
    // the structural half of the reuse decision (17417-17420 + root_side currency, blinded roots never reuse).  Returns
    // `reused`; a reused Session is consumed here (its nodes now belong to this controller's arena).
    bool attach_session(Session* s,bool adopt){
        require(s!=nullptr&&s->root!=nullptr&&s->store!=nullptr,"Search session has no root");
        require(!s->store->adopted,"Search session was already adopted by a controller");
        Node* cand=s->root;
        const bool structural=bool(cand->state)&&!cand->is_chance&&!cand->children.empty()&&cand->side==root_side&&!root_blinded;
        if(adopt&&structural){
            std::vector<Node*> nodes;std::vector<std::shared_ptr<const Menu>> menus;collect_subtree(cand,nodes,menus);
            // Node ids = creation order in the NEW controller: the retained nodes first (their old relative order, root = 0),
            // then whatever this search creates (the entry flush's materialized children included).  Children a previous
            // NON-adopted flush materialized (NO_ID) are ordinary retained children here and get ids like every other node.
            for(auto& np:s->store->nodes){np->id=arena.size();arena.push_back(std::move(np));}
            s->store->nodes.clear();s->store->adopted=true;   // the old controller's views close
            for(auto& m:menus)m->ctl=this;
            root=cand;root->parent=nullptr;reused_root=true;
            edge_flush(root);                                  // search_edge_catalog.flush on the cached root (counters on this search)
            delete s;                                          // consumed: Python released its pointer
            return true;
        }
        // Not adopted: the flush still happens (STATS catalog_fallbacks/materialized on this search); the children it
        // materializes are owned by the Session's store (freed with it) and carry NO_ID: they did not exist during the
        // search that produced the session, so that controller's export skips them.
        auto& nodes=s->store->nodes;const size_t before=nodes.size();
        std::swap(arena,nodes);
        try{edge_flush(cand);}catch(...){std::swap(arena,nodes);throw;}
        std::swap(arena,nodes);
        for(size_t i=before;i<nodes.size();++i)nodes[i]->id=NO_ID;
        return false;
    }

    // ------------------------------------------------ exports
    void fill_root(WotrSearchRoot* out){
        require(out&&out->struct_size==sizeof(WotrSearchRoot),"WotrSearchRoot struct_size mismatch");require_complete();require_attached();
        const auto& ch=root->children;const size_t K=ch.size();
        rv_visits.resize(K);rv_term.resize(K);rv_forced.resize(K);rv_sum.resize(K);rv_sq.resize(K);rv_conf.resize(K);rv_phi.resize(K);rv_pv.resize(K);rv_proven.resize(K);rv_side.resize(K);
        for(size_t i=0;i<K;++i){Node* c=ch[i];rv_visits[i]=c->visit_count;rv_term[i]=c->term_visits;rv_forced[i]=c->forced_visits;rv_sum[i]=c->value_sum;rv_sq[i]=c->value_sq;rv_conf[i]=c->conf_sum;rv_phi[i]=c->phi_sum;rv_pv[i]=c->proven_value;rv_proven[i]=static_cast<int8_t>(c->proven);rv_side[i]=c->side;}
        out->children=K;out->visits=rv_visits.data();out->term_visits=rv_term.data();out->forced_visits=rv_forced.data();out->value_sum=rv_sum.data();out->value_sq=rv_sq.data();out->conf_sum=rv_conf.data();out->phi_sum=rv_phi.data();
        out->priors_clean=priors_clean.data();out->proven=rv_proven.data();out->proven_value=rv_pv.data();out->side=rv_side.data();out->root_side=root_side;out->simulations_run=simulations_run;out->budget=budget;
    }
    std::unique_ptr<Snapshot> root_menu(){
        require_complete();require_attached();Meta m;std::vector<uint64_t> rows;
        for(Node* c:root->children){require(c->action.has(),"Root child without action");rows.push_back(m.copy(c->action.snap(),c->action.node()));}
        return m.finish(m.list(rows));
    }
    std::unique_ptr<Snapshot> export_counters(){
        Meta m;auto ints=[&](const std::map<std::string,int64_t>& src){std::vector<std::pair<std::string,uint64_t>> kv;for(const auto& item:src)kv.push_back({item.first,m.integer(item.second)});return kv;};
        auto kernel=ints(counters.kernel);
        kernel.push_back({"fallback",m.dict(ints(counters.kernel_fallback))});
        {std::vector<uint64_t> xs;for(double d:counters.dropped_mass)xs.push_back(m.real(d));kernel.push_back({"dropped_mass",m.list(xs)});}
        {std::vector<uint64_t> xs;for(auto v:counters.support)xs.push_back(m.integer(v));kernel.push_back({"support",m.list(xs)});}
        {std::vector<std::pair<std::string,uint64_t>> kv;for(const auto& item:counters.by_class){std::vector<uint64_t> xs;for(double d:item.second)xs.push_back(m.real(d));kv.push_back({item.first,m.list(xs)});}kernel.push_back({"by_class",m.dict(kv)});}
        {std::vector<uint64_t> xs;for(const auto& fd:counters.fallback_detail)xs.push_back(m.tuple({m.str(fd.first),m.copy(counters.detail.b.out,fd.second)}));kernel.push_back({"fallback_detail",m.list(xs)});}
        {std::vector<uint64_t> xs;for(const auto& h:counters.high_drop){
            if(h.kernel){xs.push_back(m.dict({{"class",m.str(h.cls)},{"dropped",m.real(h.dropped)},{"support",m.integer(h.support)},{"support_raw",m.integer(h.support_raw)},{"kept",m.integer(h.kept)},{"ctx",m.str(h.ctx_repr)}}));continue;}
            std::vector<uint64_t> pool;for(const auto& k:h.pool)pool.push_back(m.str(k));
            const uint64_t params=h.action.has()?m.copy(h.action.snap(),h.action.params()):m.dict({});
            xs.push_back(m.dict({{"class",m.str(h.cls)},{"dropped",m.real(h.dropped)},{"support",m.integer(h.support)},{"kept",m.integer(h.kept)},{"desc",m.dict({{"params",params},{"eyes",m.integer(h.eyes)},{"pool",h.has_pool?m.list(pool):m.none()}})}}));}
            kernel.push_back({"high_drop",m.list(xs)});}
        {std::vector<std::pair<uint64_t,uint64_t>> kv;for(const auto& item:counters.dropped_at){std::vector<uint64_t> xs;for(double d:item.second)xs.push_back(m.real(d));kv.push_back({m.integer(item.first),m.list(xs)});}kernel.push_back({"dropped_at",m.dict_nodes(kv)});}
        std::vector<std::pair<std::string,uint64_t>> top={{"overlay",m.dict(ints(counters.overlay))},{"kernel",m.dict(kernel)},{"search",m.dict(ints(counters.search))},{"edge",m.dict(ints(counters.edge))},{"lattice",m.dict(ints(counters.lattice))}};
        if(own)top.push_back({"own_value_search",m.dict(ints(own->stats))});
        if(profiling){   // diagnostic breakdown, present ONLY under WOTR_NATIVE_SEARCH_PROFILE=1
            std::vector<std::pair<std::string,uint64_t>> phases;
            for(const auto& item:prof)phases.push_back({item.first,m.dict({{"us",m.integer(static_cast<int64_t>(item.second.us))},{"calls",m.integer(static_cast<int64_t>(item.second.calls))}})});
            std::vector<std::pair<std::string,uint64_t>> rows;for(const auto& item:prof_rows)rows.push_back({item.first,m.integer(static_cast<int64_t>(item.second))});
            top.push_back({"profile",m.dict({{"search_total_us",m.integer(static_cast<int64_t>(prof_total_us))},{"phases",m.dict(phases)},{"rows",m.dict(rows)}})});
        }
        return m.finish(m.dict(top));
    }
    // SHARED_ABI 9.3: observer call points in occurrence order (readable once the worker has finished or failed).
    std::unique_ptr<Snapshot> export_observer(){
        {std::lock_guard<std::mutex> lk(mu);require(phase==Phase::Complete||phase==Phase::Failed,"Search is still running");}
        require_attached();   // card_policy events reference nodes that may now live in the session
        Meta m;std::vector<uint64_t> evs;
        for(const auto& e:observer_events){
            if(e.kind==1){std::vector<uint64_t> mix;for(const auto& item:e.mixture)mix.push_back(m.tuple({m.str(item.first),m.integer(item.second)}));evs.push_back(m.tuple({m.str("hidden_commit"),m.list(mix)}));continue;}
            const Snapshot& s=*e.node->state;const auto pc=obs_attr(s,s.root,"pending_combat",false);const bool has_pc=!ctx_none(s,pc)&&s.nodes[pc].tag==WOTR_DICT;
            uint64_t phase_node=m.none(),battle=m.none();
            if(has_pc){
                const auto ph=field(s,pc,"phase",false);if(ph!=missing)phase_node=m.copy(s,ph);
                const auto ar=field(s,pc,"attacker_region",false),dr=field(s,pc,"defender_region",false),as=field(s,pc,"attacker_side",false),ds=field(s,pc,"defender_side",false);
                if(ar!=missing&&dr!=missing&&as!=missing&&ds!=missing){
                    const auto turn=obs_attr(s,s.root,"turn",false);const auto rounds=field(s,pc,"rounds_fought",false);
                    battle=m.tuple({turn==missing?m.none():m.copy(s,turn),m.copy(s,ar),m.copy(s,dr),m.copy(s,as),m.copy(s,ds),rounds==missing?m.integer(0):m.copy(s,rounds)});
                }
            }
            std::vector<uint64_t> rows;rows.reserve(e.kinds.size());size_t ci=0;
            for(size_t i=0;i<e.kinds.size();++i){uint64_t cid=m.none();if(ci<e.cids.size()&&e.cids[ci].first==i){cid=m.copy(*e.cids[ci].second,e.cids[ci].second->root);++ci;}
                rows.push_back(m.tuple({m.str(obs_kind_table[e.kinds[i]]),cid}));}
            std::vector<uint64_t> probs;for(double p:e.probabilities)probs.push_back(m.real(p));
            const auto rb=root_branch_row(e.node);
            evs.push_back(m.tuple({m.str("card_policy"),m.integer(node_id_for_trace(e.node)),m.integer(e.node->side),rb?m.integer(static_cast<int64_t>(*rb)):m.none(),phase_node,battle,m.list(rows),m.list(probs)}));
        }
        return m.finish(m.list(evs));
    }
    std::unique_ptr<Snapshot> export_tree(bool with_states){
        require_attached();
        Meta m;std::vector<uint64_t> nodes;
        for(const Node* np:exportable_nodes()){const Node& n=*np;std::vector<uint64_t> ch,oc;for(Node* c:n.children)if(c->id!=NO_ID)ch.push_back(m.integer(static_cast<int64_t>(c->id)));for(Node* c:n.outcomes)oc.push_back(m.integer(static_cast<int64_t>(c->id)));
            // kernel_sets: None for single-world nodes; else [(world_key|None, [outcome ids], kernel_cover|None, drops), ...] in
            // insertion order, the ACTIVE key's entry showing the live outcomes/cover/drops (Python aliases the live list),
            // the other banks their saved values.  Inactive-bank outcomes are arena nodes and therefore appear in 'nodes'.
            uint64_t ks=m.none();
            if(n.has_kernel_sets){
                const std::string active=bank_key(&n);std::vector<uint64_t> banks;
                for(const auto& bank:n.kernel_sets){
                    const bool live=bank.first==active;const std::vector<Node*>& outs=live?n.outcomes:bank.second.outcomes;
                    const bool has_cover=live?n.has_cover:bank.second.has_cover;const double cover=live?n.kernel_cover:bank.second.cover;const uint64_t drops=live?n.drops:bank.second.drops;
                    std::vector<uint64_t> ids;for(Node* c:outs)ids.push_back(m.integer(static_cast<int64_t>(c->id)));
                    banks.push_back(m.tuple({bank.first==std::string(1,'\0')?m.none():m.str(bank.first),m.list(ids),has_cover?m.real(cover):m.none(),m.integer(static_cast<int64_t>(drops))}));
                }
                ks=m.list(banks);
            }
            std::vector<std::pair<std::string,uint64_t>> kv={
                {"id",m.integer(static_cast<int64_t>(n.id))},{"parent",n.parent?m.integer(static_cast<int64_t>(n.parent->id)):m.none()},{"side",m.integer(n.side)},
                {"action",n.action.has()?m.copy(n.action.snap(),n.action.node()):m.none()},{"prior",m.real(n.prior)},
                {"visit_count",m.integer(static_cast<int64_t>(n.visit_count))},{"value_sum",m.real(n.value_sum)},{"value_sq",m.real(n.value_sq)},{"conf_sum",m.real(n.conf_sum)},{"phi_sum",m.real(n.phi_sum)},
                {"term_visits",m.integer(static_cast<int64_t>(n.term_visits))},{"forced_visits",m.integer(static_cast<int64_t>(n.forced_visits))},
                {"is_chance",m.boolean(n.is_chance)},{"is_terminal",m.boolean(n.is_terminal)},{"terminal_value",m.real(n.terminal_value)},
                {"proven",n.proven==0?m.none():m.str(n.proven==1?"W":"L")},{"proven_value",m.real(n.proven_value)},
                {"kernel_cover",n.has_cover?m.real(n.kernel_cover):m.none()},{"drops",m.integer(static_cast<int64_t>(n.drops))},
                {"out_sig",n.has_sig?m.bytes(n.out_sig):m.none()},{"children",m.list(ch)},{"outcomes",n.has_outcomes?m.list(oc):m.none()},
                {"kernel_key",n.has_kernel_key?m.str(n.kernel_key):m.none()},{"kernel_sets",ks},{"round_sampling",m.boolean(n.round_sampling)},
                {"has_state",m.boolean(bool(n.state))},{"state_graph",m.none()},
                {"state_wire",with_states&&n.state?m.bytes(chance_sig(*n.state)):m.none()}};
            nodes.push_back(m.dict(kv));}
        std::vector<uint64_t> evs;
        for(const Event& e:events){std::vector<uint64_t> xs{m.str(e.kind)};
            if(e.kind=="SIM_RETURN"){std::vector<uint64_t> path;for(auto id:e.ids)path.push_back(m.integer(static_cast<int64_t>(id)));
                xs.push_back(m.integer(e.i[0]));xs.push_back(m.list(path));xs.push_back(e.leaf_none?m.none():m.real(e.d[0]));xs.push_back(m.real(e.d[1]));xs.push_back(m.real(e.d[2]));xs.push_back(m.boolean(e.i[1]!=0));xs.push_back(m.boolean(e.i[2]!=0));}
            else if(e.kind=="INFERENCE"){for(auto v:e.i)xs.push_back(m.integer(v));std::vector<uint64_t> ids;for(auto id:e.ids)ids.push_back(m.integer(static_cast<int64_t>(id)));xs.push_back(m.list(ids));}
            else if(e.kind=="EXPAND"||e.kind=="SETTLE"||e.kind=="OPPONENT_DRAW"){xs.push_back(m.integer(e.i[0]));xs.push_back(m.integer(e.i[1]));xs.push_back(m.boolean(e.i[2]!=0));}
            else for(auto v:e.i)xs.push_back(m.integer(v));
            evs.push_back(m.tuple(xs));}
        return m.finish(m.dict({{"nodes",m.list(nodes)},{"events",m.list(evs)}}));
    }
};

// Lazy policy keys for a whole menu (first opponent-side consumer).  Under WOTR_NATIVE_KEY_CHECK=1 every row is
// cross-checked against the former route (native_menu_view row key -> key_bytes); a mismatch throws.
void Menu::ensure_keys()const{
    if(keys_ready)return;
    std::unique_ptr<Controller::Scoped> timer;if(ctl&&ctl->profiling){timer=std::make_unique<Controller::Scoped>(ctl,"policy_keys");ctl->prof_rows["policy_key_rows"]+=rows.size();}
    sm_Builder b(*snap);std::vector<std::string> out(rows.size());
    for(size_t i=0;i<rows.size();++i)direct_row_key(*snap,node(i),b,out[i]);
    if(ctl&&ctl->key_check){
        auto view=native_menu_view(*snap,*ctl->C);const auto view_rows=child_edges(*view,view->root);require(view_rows.size()==rows.size(),"Key check: menu view row count differs");
        for(size_t i=0;i<rows.size();++i){const auto f=child_edges(*view,view_rows[i]);std::string reference;key_bytes(*view,f[6],reference);require(reference==out[i],"Key check: direct policy key differs from native_menu_view key");}
    }
    keys=std::move(out);keys_ready=true;
}
} // namespace sc
