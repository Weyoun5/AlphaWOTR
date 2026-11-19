// The controller's COMBAT_DIST reply (SHARED_ABI 9.5) computed in the module, in place of the Python service
// (orchestrator.service_combat_request -> combat_distribution_reply: ai._kernel_dist(ctx, want_witness=True),
// combat_dist.plan_conf_case, wctk_side, repr(ctx)). The distribution is their own C++ port of it, the turn search's
// combat kernel (turn_search/native_w2x/box/cpp/combat_kernel.inc, "byte-exact" with combat_dist.exact_distribution),
// compiled into this translation unit after the live box's snapshot.cpp (driver/box_unity.cpp). This file only
// unpacks the request payload's 'context' and packs the reply dict as snapshot.encode_graph packs Python's.
namespace awreply {
struct Builder {
    std::vector<WotrValue> v; std::vector<uint64_t> e; std::vector<uint8_t> b;
    std::unordered_map<std::string,uint64_t> strings;     // encode_graph interns equal str objects into one node
    uint64_t node(uint32_t tag){WotrValue x{};x.tag=tag;v.push_back(x);return v.size()-1;}
    uint64_t none(){return node(WOTR_NONE);}
    uint64_t boolean(bool x){auto n=node(WOTR_BOOL);v[n].integer=x?1:0;return n;}
    uint64_t integer(int64_t x){auto n=node(WOTR_INT);v[n].integer=x;return n;}
    uint64_t real(double x){auto n=node(WOTR_FLOAT);v[n].real=x;return n;}
    uint64_t str(const std::string& s){
        auto it=strings.find(s);if(it!=strings.end())return it->second;
        auto n=node(WOTR_STRING);v[n].offset=b.size();v[n].count=s.size();b.insert(b.end(),s.begin(),s.end());strings[s]=n;return n;
    }
    // a container: its node first, then its children (encode_graph visits the children after creating the node)
    template<class F> uint64_t container(uint32_t tag,size_t count,F child){
        auto n=node(tag);std::vector<uint64_t> kids;kids.reserve(count);
        for(size_t i=0;i<count;++i)kids.push_back(child(i));
        v[n].offset=e.size();v[n].count=kids.size();e.insert(e.end(),kids.begin(),kids.end());return n;
    }
};
std::unique_ptr<Snapshot> reply(const Snapshot& tables,const Snapshot& payload){
    const auto ctx_node=field(payload,payload.root,"context",false);
    require(ctx_node!=missing,"COMBAT_DIST payload has no context");
    Snapshot ctx_snap(payload);ctx_snap.root=ctx_node;
    const ck_Ctx ctx=ck_parse(ctx_snap);
    require(ctx.defer,"witness plans are defer-mode only (in-tree rounds)");
    const ck_Dist dist=ck_exact_distribution(tables,ctx,true);
    const auto wctk=ck_wctk_side(tables,ctx);
    Builder g;
    const std::vector<std::string> keys={"outcomes","masses","witness","plan_reprs","conf","wctk_side","ctx_repr"};
    const auto root=g.container(WOTR_DICT,2*keys.size(),[&](size_t i)->uint64_t{
        const std::string& k=keys[i/2];
        if(i%2==0)return g.str(k);
        if(k=="outcomes")return g.container(WOTR_LIST,dist.obs.size(),[&](size_t j){
            const ck_Obs& o=dist.obs[j];
            return g.container(WOTR_TUPLE,19,[&](size_t m)->uint64_t{
                if(m<12)return g.integer(o.v[m]);
                if(m==12)return g.boolean(o.destA);
                if(m==13)return g.boolean(o.destD);
                if(m==14)return g.container(WOTR_TUPLE,o.fxa.size(),[&](size_t q){return g.str(o.fxa[q]);});
                if(m==15)return g.container(WOTR_TUPLE,o.fxd.size(),[&](size_t q){return g.str(o.fxd[q]);});
                if(m==16)return g.boolean(o.deferred);
                return g.integer(m==17?o.dhd:o.dha);});});
        if(k=="masses")return g.container(WOTR_LIST,dist.mass.size(),[&](size_t j){return g.real(dist.mass[j]);});
        if(k=="witness")return g.container(WOTR_LIST,dist.plan.size(),[&](size_t j){
            const ck_Plan& p=dist.plan[j];
            return g.container(WOTR_LIST,p.size(),[&](size_t q)->uint64_t{
                if(!p[q].tuple)return g.integer(p[q].h);
                return g.container(WOTR_TUPLE,2,[&](size_t r){return g.integer(r==0?p[q].h:p[q].o);});});});
        if(k=="plan_reprs")return g.container(WOTR_LIST,dist.plan.size(),[&](size_t j){return g.str(ck_plan_repr(dist.plan[j]));});
        if(k=="conf")return g.boolean(ck_conf_case(ctx));
        if(k=="wctk_side")return wctk?g.str(*wctk):g.none();
        return g.str(ck_ctx_repr(ctx));});
    return create(g.v.data(),g.v.size(),g.e.data(),g.e.size(),g.b.data(),g.b.size(),root);
}
}  // namespace awreply
extern "C" {
// The 9.5 reply snapshot (wotr_snapshot_destroy) for a COMBAT_DIST request payload
// (wotr_search_controller_request_payload); tables: the observation tables. Null on error (wotr_snapshot_last_error).
WOTR_API void* awasm_combat_reply(const void* tables,const void* payload){
    return checked<void*>([&]()->void*{
        try{return awreply::reply(snapshot(tables),snapshot(payload)).release();}
        catch(const ck_Error& e){throw std::invalid_argument("combat kernel: "+e.why);}
    },nullptr);
}
}
