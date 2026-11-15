// Native descriptor graph for a strictly labelled subset of raw legal actions.
// Frozen ai.py:6144, 7491 and 7662--7833 are the ordering/legality authority.
// This is NOT a complete menu or the projected/deduplicated AI policy.
// Speed pass (controller/engine_patches/action_menu.inc.patch): interning looks keys up as string_views (no per-call
// std::string temporary); map keys view `owned`, a stable store of the interned texts.  Same writes to `out`, same
// first-occurrence node ids.  Under WOTR_NATIVE_FULLMENU_CHECK=1 (am_check_active, set by search_controller.inc) the former
// std::string-keyed map runs in parallel and every call must agree on hit/miss and node id.
#define WOTR_NATIVE_AMBUILDER_CHECK_AVAILABLE 1
int am_check_active=0;uint64_t am_check_interns=0;
// Speed pass (growth policy, controller/engine_patches/action_menu.inc.patch): capacity only.  am_grow makes room for `more`
// further elements; when the capacity is short it grows to max(needed, capacity*3/2) (the library's own factor), so an exact
// per-call estimate costs one reallocation and a short estimate degrades to the ordinary geometric growth.  Reserving never
// changes what is written or in which order.
template<class V> void am_grow(V& v,size_t more){const size_t need=v.size()+more;if(v.capacity()<need)v.reserve(std::max(need,v.capacity()+v.capacity()/2));}
struct am_Builder {
    Snapshot out;
    std::unordered_map<std::string_view,uint64_t> strings;
    std::vector<std::shared_ptr<std::string>> owned;        // shared, never mutated: a copied builder keeps valid keys and interns on independently, as the std::string map did
    std::unordered_map<std::string,uint64_t> strings_ref;   // reference interning (certified code path), maintained only under the check
    am_Builder(){out.nodes.reserve(512);out.edges.reserve(1024);out.bytes.reserve(2048);strings.reserve(64);owned.reserve(64);}
    void grow(size_t nodes,size_t edges,size_t bytes){am_grow(out.nodes,nodes);am_grow(out.edges,edges);am_grow(out.bytes,bytes);}   // speed pass (growth policy): capacity only
    uint64_t str(std::string_view value){
        const auto found=strings.find(value);const bool hit=found!=strings.end();uint64_t node;
        if(hit)node=found->second;
        else{
            const auto offset=out.bytes.size();out.bytes.insert(out.bytes.end(),value.begin(),value.end());
            node=out.nodes.size();out.nodes.push_back({WOTR_STRING,0,0,0,offset,value.size()});
            owned.push_back(std::make_shared<std::string>(value));strings.emplace(std::string_view(*owned.back()),node);
        }
        if(am_check_active){
            std::string key(value);const auto ref=strings_ref.find(key);const bool ref_hit=ref!=strings_ref.end();
            if(ref_hit!=hit||(ref_hit&&ref->second!=node))throw std::logic_error("WOTR_NATIVE_FULLMENU_CHECK: am_Builder::str interning differs from the reference for '"+key+"'");
            if(!ref_hit)strings_ref.emplace(std::move(key),node);++am_check_interns;
        }
        return node;
    }
    uint64_t integer(int64_t value){return rr_scalar(out,WOTR_INT,value);}
    uint64_t container(uint32_t tag,const std::vector<uint64_t>& children){
        const auto offset=out.edges.size();out.edges.insert(out.edges.end(),children.begin(),children.end());
        const auto node=out.nodes.size();out.nodes.push_back({tag,0,0,0,offset,children.size()});return node;
    }
    uint64_t container(uint32_t tag,std::initializer_list<uint64_t> children){
        const auto offset=out.edges.size();out.edges.insert(out.edges.end(),children.begin(),children.end());
        const auto node=out.nodes.size();out.nodes.push_back({tag,0,0,0,offset,children.size()});return node;
    }
    uint64_t nation(std::string_view name){return container(WOTR_ENUM,{str("game_state.Nation"),str(name)});}
    uint64_t row(int64_t die,std::string_view type,const std::vector<uint64_t>& parameters){
        const auto index=integer(die),kind=str(type),params=container(WOTR_DICT,parameters);
        // speed pass: the three tuple edges are appended directly (identical to container(WOTR_TUPLE,{index,kind,params}) without its temporary vector)
        const auto offset=out.edges.size();out.edges.push_back(index);out.edges.push_back(kind);out.edges.push_back(params);
        const auto node=out.nodes.size();out.nodes.push_back({WOTR_TUPLE,0,0,0,offset,3});return node;
    }
    uint64_t row(int64_t die,std::string_view type,std::initializer_list<uint64_t> parameters){
        const auto index=integer(die),kind=str(type),params=container(WOTR_DICT,parameters);
        // speed pass: the three tuple edges are appended directly (identical to container(WOTR_TUPLE,{index,kind,params}) without its temporary vector)
        const auto offset=out.edges.size();out.edges.push_back(index);out.edges.push_back(kind);out.edges.push_back(params);
        const auto node=out.nodes.size();out.nodes.push_back({WOTR_TUPLE,0,0,0,offset,3});return node;
    }
};

std::unique_ptr<Snapshot> political_ring_actions(const Snapshot& s,const Snapshot& tables,int32_t side){
    require(side==0 || side==1,"political/ring menu side must be FP or SH");
    quiet_stable_action_gate(s,tables,false,true);
    ring_waiver(s,side); // Raw get_legal_actions checks only the requested side.
    const auto pool=pa_pool(s,side);const auto& dice=s.nodes[pool];
    std::vector<std::pair<uint64_t,int>> unused;unused.reserve(dice.count);
    for(uint64_t i=0;i<dice.count;++i){
        const auto die=s.edges[dice.offset+i];
        require(object_type(s,die)=="game_state.ActionDie","political/ring menu requires ActionDie");
        if(truth(s,obs_attr(s,die,"used")))continue;
        const auto current=obs_attr(s,die,"current");
        if(s.nodes[current].tag==WOTR_NONE)continue;
        const auto face=enum_member(s,current,"game_state.ActionDieFace");
        unused.emplace_back(i,rr_face_index(face));
    }
    am_Builder b;std::vector<uint64_t> rows;
    const std::vector<std::string_view> nations=side==0
        ? std::vector<std::string_view>{"ELVES","DWARVES","GONDOR","THE_NORTH","ROHAN"}
        : std::vector<std::string_view>{"ISENGARD","SAURON","SOUTHRON_EASTERLINGS"};
    std::array<bool,7> seen{};
    for(const auto& entry:unused){
        const auto index=entry.first;const int face=entry.second;
        if(seen[face])continue;seen[face]=true;
        if(face!=2 && face!=4 && face!=6)continue; // MUSTER, MUSTER_ARMY, WILL_OF_WEST
        for(const auto nation:nations){
            const auto p=political_view(s,nation,false);
            if(!can_advance(s,p,nation,"muster_die"))continue;
            rows.push_back(b.row(static_cast<int64_t>(index),"advance_political",{b.str("nation"),b.nation(nation)}));
        }
    }
    const auto stock=obs_attr(s,s.root,side==0?"fp_elven_rings":"sh_elven_rings");
    const auto used=obs_attr(s,s.root,side==0?"fp_ring_used_this_turn":"sh_ring_used_this_turn");
    if(!rr_nonpositive(s,stock) && !truth(s,used)){
        // Source destination order intentionally differs from enum order.
        const std::vector<int> destinations=side==0?std::vector<int>{0,2,3,4}:std::vector<int>{0,2,3,4,1,5};
        for(const auto& entry:unused){
            const auto index=entry.first;const int current=entry.second;
            if(side==1 && current==5)continue;
            for(const auto next:destinations){
                if(next==current)continue;
                rows.push_back(b.row(-1,"elven_ring",{
                    b.str("target_die_idx"),b.integer(static_cast<int64_t>(index)),
                    b.str("new_face"),b.str(rr_face_values[next]),
                    b.str("old_face"),b.str(rr_face_values[current]),
                    b.str("side"),b.str(side==0?"fp":"sh")}));
            }
        }
    }
    b.out.root=b.container(WOTR_LIST,rows);
    return std::make_unique<Snapshot>(std::move(b.out));
}
