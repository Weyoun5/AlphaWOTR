// NATIVE KERNEL OUTCOMES (2b, claude_bugscan_2026-10-21, 2026-10-27): selfplay_actor_critic/chance.kernel_outcomes
// with ai._kernel_expand_combat (search limits chance_k=0: the whole support; no forecast observer; search root None)
// on the native engine, the distribution from combat_kernel.inc.  Line for line with the native search controller's
// kernel_expand_combat / kernel_expand_onslaught_post (themselves ports of ai.py 15598-15930), as free functions over a
// state, the tables, the action row and a stream.
//   status 0: leaves (mass / total, successor state), in the merged-by-chance-signature, -mass stable order;
//   status 1: chance.kernel_outcomes returns None (not a kernel parent, no RNG, a fallback reason, an exception);
//   status 2: a condition outside the port (the caller replays the event on Python and logs a native refusal).
struct kx_Leaf {double mass=0.0;std::unique_ptr<Snapshot> state;};
thread_local std::string kx_reason;   // why the last kernel_outcomes returned None (diagnostics)
struct kx_Mismatch {std::string message;};
int64_t kx_pc_int(const Snapshot& s,uint64_t pc,const char* key){
    const auto n=field(s,pc,key,false);if(ctx_none(s,n))return 0;const auto& v=s.nodes[n];
    if(v.tag==WOTR_INT||v.tag==WOTR_BOOL)return v.integer;if(v.tag==WOTR_FLOAT)return static_cast<int64_t>(v.real);throw std::invalid_argument("pending_combat hit count must be numeric");
}
// combat_round_search.requires_phase_sampling
bool kx_requires_phase_sampling(const Snapshot& parent,const Snapshot& probe,const Snapshot& T){
    for(const Snapshot* s:{&parent,&probe}){const auto pc=obs_attr(*s,s->root,"pending_combat",false);if(!ctx_none(*s,pc)&&s->nodes[pc].tag==WOTR_DICT&&truth(*s,field(*s,pc,"_round_resume",false)))return true;}
    const auto info=obs_attr(probe,probe.root,"_last_round_call",false);if(ctx_none(probe,info)||probe.nodes[info].tag!=WOTR_DICT)return false;
    const auto choices=field(probe,info,"choices",false);if(ctx_none(probe,choices)||probe.nodes[choices].tag!=WOTR_DICT||!truth(probe,field(probe,choices,"defer_round_choices",false)))return false;
    static const char* adaptive[]={"dread and despair","relentless assault","mighty attack","fateful strike","blade of westernesse","heroic death","black breath","charge","sudden strike","durin's bane"};
    const auto cards=field(T,T.root,"cards");
    for(const char* key:{"atk_card","def_card"}){
        const auto cid=field(probe,info,key,false);if(!truth(probe,cid)||probe.nodes[cid].tag!=WOTR_STRING)continue;
        const auto card=field(T,cards,std::string(text(probe,cid)).c_str(),false);if(ctx_none(T,card))continue;
        const auto name=obs_attr(T,card,"combat_name",false);if(ctx_none(T,name)||T.nodes[name].tag!=WOTR_STRING)continue;
        std::string lower(text(T,name));const auto b=lower.find_first_not_of(" \t\r\n"),e=lower.find_last_not_of(" \t\r\n");lower=b==std::string::npos?std::string():lower.substr(b,e-b+1);
        for(auto& c:lower)c=static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        for(const char* a:adaptive)if(lower==a)return true;
    }
    return false;
}
struct kx_RoundInfo {std::string atk_region,def_region;int atk_side=0,def_side=0;std::optional<std::string> atk_card,def_card;std::shared_ptr<Snapshot> allocation_context;};
kx_RoundInfo kx_round_info(const Snapshot& parent,const Snapshot& probe,uint64_t info){
    kx_RoundInfo r;
    r.atk_region=std::string(text(probe,field(probe,info,"atk_region")));r.def_region=std::string(text(probe,field(probe,info,"def_region")));
    r.atk_side=ctx_side(probe,field(probe,info,"atk_side"));r.def_side=ctx_side(probe,field(probe,info,"def_side"));
    for(auto item:{std::make_pair("atk_card",&r.atk_card),std::make_pair("def_card",&r.def_card)}){const auto n=field(probe,info,item.first,false);if(truth(probe,n)&&probe.nodes[n].tag==WOTR_STRING)*item.second=std::string(text(probe,n));}
    const auto pc=obs_attr(parent,parent.root,"pending_combat",false);
    auto pending=[&](const char* key){return ctx_none(parent,pc)?missing:field(parent,pc,key,false);};
    const auto rear=field(probe,info,"attacker_rearguard",false),reserve=field(probe,info,"corsairs_relief_garrison",false);
    sc::Meta m;
    const auto local_rear=ctx_none(probe,rear)?m.copy(parent,pending("attacker_rearguard")):m.copy(probe,rear);
    const auto local_reserve=reserve==missing?m.copy(parent,pending("corsairs_relief_garrison")):m.copy(probe,reserve);
    r.allocation_context=m.finish(m.dict({{"attacker_rearguard",local_rear},{"corsairs_relief_garrison",local_reserve},
        {"attacker_side",m.copy(probe,field(probe,info,"atk_side"))},{"defender_region",m.copy(probe,field(probe,info,"def_region"))}}));
    return r;
}
struct kx_UnitCounts {int64_t regular=0,elite=0;std::set<std::string> nations;};
kx_UnitCounts kx_alloc_units(const Snapshot& state,const std::string& region,int side,int kind,const Snapshot& context){
    sc::Meta m;
    auto args=m.finish(m.dict({{"region",m.str(region)},{"side",m.enum_member("game_state.Side",side==0?"FREE_PEOPLES":"SHADOW")},
        {"kind",kind<0?m.none():m.enum_member("game_state.UnitKind",kind==0?"REGULAR":"ELITE")},{"pc",m.copy(context,context.root)}}));
    auto units=native_combat_search_units(state,*args);kx_UnitCounts out;
    for(auto u:child_edges(*units,units->root)){
        const auto k=enum_member(*units,obs_attr(*units,u,"kind"),"game_state.UnitKind");if(k=="REGULAR")++out.regular;else ++out.elite;
        out.nations.emplace(enum_member(*units,obs_attr(*units,u,"nation"),"game_state.Nation"));
    }
    return out;
}
std::pair<int64_t,int64_t> kx_hits_pe(int64_t r,int64_t e,int64_t h){const int64_t k=std::min(r,h);r-=k;h-=k;while(h>0&&e>0){e-=1;if(h>=2)h-=2;else{r+=1;h-=1;}}return {r,e};}
// ai._predict_alloc_pause
bool kx_predict_alloc_pause(const Snapshot& state,const kx_RoundInfo& info,const ck_Obs& obs){
    struct Row{std::string region;int side;int64_t r,e,hits;};
    for(const Row& row:{Row{info.atk_region,info.atk_side,obs.v[8],obs.v[9],obs.dha},Row{info.def_region,info.def_side,obs.v[10],obs.v[11],obs.dhd}}){
        const int64_t hits=row.hits;if(hits<=0||row.r+row.e<=0)continue;
        if(hits>=2*row.e+row.r&&(row.side==1||row.e==0))continue;
        if(row.e&&(row.r||hits>=2))return true;
        const auto units=kx_alloc_units(state,row.region,row.side,row.e?1:0,*info.allocation_context);
        if(units.nations.size()>1)return true;
    }
    return false;
}
// ai._gh_alloc_expected
std::pair<int64_t,int64_t> kx_gh_alloc_expected(const std::optional<std::string>& atk_card,const std::optional<std::string>& def_card,const ck_Obs& obs){
    const auto a=kx_hits_pe(obs.v[8],obs.v[9],obs.dha),d=kx_hits_pe(obs.v[10],obs.v[11],obs.dhd);
    const int64_t atk_n=a.first+a.second,def_n=d.first+d.second;int64_t hd=0,ha=0;
    auto lower=[](const std::optional<std::string>& s){std::string v=s.value_or("");for(auto& c:v)c=static_cast<char>(std::tolower(static_cast<unsigned char>(c)));return v;};
    if(atk_n>0&&def_n>0){
        if(lower(atk_card).find("great host")!=std::string::npos&&atk_n>=2*def_n)hd=1;
        if(lower(def_card).find("great host")!=std::string::npos&&def_n>=2*atk_n)ha=1;
    }
    return {hd,ha};
}
// combat_dist.run_with_plan through the native forced-combat scope (the controller's forced_apply)
std::unique_ptr<Snapshot> kx_forced_apply(const Snapshot& parent,const Snapshot& T,const Snapshot& menu,int64_t row,int side,const ck_Plan& plan,bool confusion,const void* stream){
    std::vector<int64_t> flat;std::vector<uint8_t> tags;
    for(const auto& p:plan){flat.push_back(p.h);flat.push_back(p.tuple?p.o:-1);tags.push_back(p.tuple?1:0);}
    require(plan.size()<=256,"Combat plan exceeds supported batch count");
    try{
        cfc_Scope compatible(tags.data(),tags.size());cf_Scope scope(flat.data(),plan.size(),confusion);
        auto result=apply_native_random_descriptor(parent,T,menu,stream,side,row);
        if(!cf_same_rng(stream,result.random.get()))throw cfc_Error(4,0,"",0,0,0);
        if(scope.plan.next!=scope.plan.size)throw cfc_Error(5,0,"",0,0,0);
        return std::move(result.state);
    }catch(const cfc_Error& e){throw kx_Mismatch{"plan mismatch "+std::to_string(e.kind)};}
}
// merge by _chance_sig (search root None), insertion order, then stable sort by -mass (install_kernel_branches)
void kx_merge(std::vector<std::pair<std::unique_ptr<Snapshot>,double>>& branches,std::vector<kx_Leaf>& out){
    struct M{std::vector<uint8_t> sig;std::unique_ptr<Snapshot> state;double mass;};std::vector<M> merged;
    for(auto& b:branches){auto w=native_chance_signature_wire(*b.first,-1);auto it=std::find_if(merged.begin(),merged.end(),[&](const M& m){return m.sig==w->bytes;});
        if(it!=merged.end())it->mass+=b.second;else merged.push_back({w->bytes,std::move(b.first),b.second});}
    std::vector<size_t> order(merged.size());for(size_t i=0;i<order.size();++i)order[i]=i;
    std::stable_sort(order.begin(),order.end(),[&](size_t a,size_t b){return -merged[a].mass< -merged[b].mass;});
    out.clear();for(size_t i:order)out.push_back({merged[i].mass,std::move(merged[i].state)});
}
// ai._kernel_expand_onslaught_post
bool kx_onslaught_post(const Snapshot& parent,const Snapshot& T,const Snapshot& menu,int64_t row,int side,const void* stream,std::vector<kx_Leaf>& out){
    const auto pc=obs_attr(parent,parent.root,"pending_combat",false);
    const auto payment=(!ctx_none(parent,pc)&&parent.nodes[pc].tag==WOTR_DICT)?field(parent,pc,"_onslaught_payment",false):missing;
    const auto rows=child_edges(menu,menu.root);const auto f=child_edges(menu,rows[static_cast<size_t>(row)]);const auto params=f[2];int64_t k;
    if(!ctx_none(parent,payment)&&truth(parent,payment)){
        const bool elite_remove=text_is(menu,field(menu,params,"kind",false),"elite")&&text_is(menu,field(menu,params,"operation",false),"remove");
        k=kx_pc_int(parent,payment,"paid")+(elite_remove?2:1);
    }else{const auto count=field(menu,params,"count",false);k=ctx_none(menu,count)?0:menu.nodes[count].integer;}
    if(k<=0){kx_reason="onslaught_post_k0";return false;}
    std::vector<std::pair<std::unique_ptr<Snapshot>,double>> branches;
    try{
        for(const auto& term:ck_binom_terms(k,static_cast<double>(ck_hit_faces(4,0))/6.0)){
            ck_PlanItem item;item.h=term.first;branches.push_back({kx_forced_apply(parent,T,menu,row,side,{item},false,stream),term.second});
        }
    }catch(const kx_Mismatch& m){kx_reason="onslaught plan_mismatch: "+m.message;return false;}
    kx_merge(branches,out);return true;
}
// ai._kernel_expand_combat (chance_k=0: every class kept)
bool kx_expand_combat(const Snapshot& parent,const Snapshot& probe,const Snapshot& T,const Snapshot& menu,int64_t row,int side,const std::string& kind,const void* stream,std::vector<kx_Leaf>& out){
    if(kx_requires_phase_sampling(parent,probe,T)){kx_reason="observed_round_phase";return false;}
    const auto info_node=obs_attr(probe,probe.root,"_last_round_call",false);
    if(ctx_none(probe,info_node)||probe.nodes[info_node].tag!=WOTR_DICT){
        const auto pcd=obs_attr(parent,parent.root,"pending_combat",false);const bool pc_dict=!ctx_none(parent,pcd)&&parent.nodes[pcd].tag==WOTR_DICT;
        const auto ecd=pc_dict?field(parent,pcd,"effect_choice",false):missing;const bool ec_dict=!ctx_none(parent,ecd)&&parent.nodes[ecd].tag==WOTR_DICT;
        const auto stage=ec_dict?field(parent,ecd,"stage",false):missing;
        if(kind=="combat_effect_choice"&&(text_is(parent,stage,"post")||text_is(parent,stage,"onslaught_payment")))return kx_onslaught_post(parent,T,menu,row,side,stream,out);
        kx_reason="no_round_call";return false;
    }
    const kx_RoundInfo info=kx_round_info(parent,probe,info_node);
    Snapshot info_snap;{std::unordered_map<uint64_t,uint64_t> memo;info_snap.root=sc::graph_copy(info_snap,probe,info_node,memo);}
    auto ctx_snap=native_kernel_context(parent,T,info_snap,info.atk_card?info.atk_card->c_str():nullptr,info.def_card?info.def_card->c_str():nullptr,true);
    const ck_Ctx ctx=ck_parse(*ctx_snap);
    const ck_Dist dist=ck_exact_distribution(T,ctx,true);
    // OUTCOME-CLASS COARSENING by obs[8:19] (insertion order; the max-mass member witnesses the class)
    struct Class{std::string key;double mass;size_t witness;double max_mass;};std::vector<Class> classes;std::unordered_map<std::string,size_t> idx;
    auto class_key=[](const ck_Obs& o){std::string k;for(int i=8;i<12;++i){k+=std::to_string(o.v[i]);k+=',';}k+=o.destA?'1':'0';k+=o.destD?'1':'0';k+='|';
        for(const auto& s:o.fxa){k+=s;k+='\x1f';}k+='|';for(const auto& s:o.fxd){k+=s;k+='\x1f';}k+='|';k+=o.deferred?'1':'0';k+=','+std::to_string(o.dhd)+','+std::to_string(o.dha);return k;};
    for(size_t i=0;i<dist.obs.size();++i){
        const auto key=class_key(dist.obs[i]);const double m9=dist.mass[i];auto it=idx.find(key);
        if(it==idx.end()){idx.emplace(key,classes.size());classes.push_back({key,m9,i,m9});}
        else{auto& c=classes[it->second];c.mass+=m9;if(m9>c.max_mass){c.witness=i;c.max_mass=m9;}}
    }
    std::vector<size_t> order(classes.size());for(size_t i=0;i<order.size();++i)order[i]=i;
    std::stable_sort(order.begin(),order.end(),[&](size_t a,size_t b){return -classes[a].mass< -classes[b].mass;});
    double total=0.0;for(size_t i:order)total+=classes[i].mass;
    if(order.empty()||std::fabs(total-1.0)>1e-6){kx_reason="mass_error";return false;}
    const bool conf=ck_conf_case(ctx);const std::optional<std::string> wctk=ck_wctk_side(T,ctx);
    std::vector<std::pair<std::unique_ptr<Snapshot>,double>> branches;
    try{
        for(size_t idx_out:order){
            const size_t oi=classes[idx_out].witness;const double mass=classes[idx_out].mass;const ck_Obs& obs=dist.obs[oi];const ck_Plan& plan=dist.plan[oi];
            if(wctk&&obs.deferred&&!kx_predict_alloc_pause(parent,info,obs)){
                const auto a2=kx_hits_pe(obs.v[8],obs.v[9],obs.dha),d2=kx_hits_pe(obs.v[10],obs.v[11],obs.dhd);
                const bool atk=*wctk=="attacker";const auto own_post=atk?a2:d2,en_post=atk?d2:a2;const int64_t ec9=std::min<int64_t>(5,own_post.second);
                if(a2.first+a2.second>0&&d2.first+d2.second>0&&ec9>0){
                    const std::string ro=atk?info.atk_region:info.def_region,re9=atk?info.def_region:info.atk_region;const int so=atk?info.atk_side:info.def_side,se=atk?info.def_side:info.atk_side;
                    const char* enr=atk?"def":"atk";
                    for(const auto& term:ck_binom_terms(ec9,static_cast<double>(ck_hit_faces(5,0))/6.0)){
                        const int64_t h9=term.first;ck_Plan extended=plan;ck_PlanItem extra;extra.h=h9;extended.push_back(extra);
                        auto st=kx_forced_apply(parent,T,menu,row,side,extended,conf,stream);
                        const auto pcb=obs_attr(*st,st->root,"pending_combat",false);const bool pcb_dict=!ctx_none(*st,pcb)&&st->nodes[pcb].tag==WOTR_DICT;
                        const bool paused=pcb_dict&&text_is(*st,field(*st,pcb,"phase",false),"casualty_allocation")&&text_is(*st,field(*st,pcb,"alloc_stage",false),atk?"defender":"attacker")&&kx_pc_int(*st,pcb,(std::string(enr)+"_hits").c_str())==h9;
                        const auto en_pred=paused?en_post:kx_hits_pe(en_post.first,en_post.second,h9);
                        const auto enemy=kx_alloc_units(*st,re9,se,-1,*info.allocation_context),own=kx_alloc_units(*st,ro,so,-1,*info.allocation_context);
                        if(std::make_pair(enemy.regular,enemy.elite)!=en_pred||std::make_pair(own.regular,own.elite)!=own_post){kx_reason="branch_verify wctk";return false;}
                        branches.push_back({std::move(st),mass*term.second});
                    }
                    continue;
                }
            }
            auto st=kx_forced_apply(parent,T,menu,row,side,plan,conf,stream);
            const auto pcb=obs_attr(*st,st->root,"pending_combat",false);
            if(!ctx_none(*st,pcb)&&st->nodes[pcb].tag==WOTR_DICT&&text_is(*st,field(*st,pcb,"phase",false),"casualty_allocation")){
                bool vok;const bool p9=truth(*st,field(*st,pcb,"_p9_done",false));
                if(p9){const auto gh=kx_gh_alloc_expected(ctx.has_atk_card?std::optional<std::string>(ctx.atk_card):std::nullopt,
                                                         ctx.has_def_card?std::optional<std::string>(ctx.def_card):std::nullopt,obs);vok=kx_pc_int(*st,pcb,"def_hits")==gh.first&&kx_pc_int(*st,pcb,"atk_hits")==gh.second;}
                else vok=kx_pc_int(*st,pcb,"def_hits")==obs.dhd&&(text_is(*st,field(*st,pcb,"alloc_stage",false),"attacker")?kx_pc_int(*st,pcb,"atk_hits")==obs.dha:kx_pc_int(*st,pcb,"atk_hits")==0);
                if(!vok){kx_reason="branch_verify obs("+std::to_string(obs.dhd)+","+std::to_string(obs.dha)+") pcb def_hits="+std::to_string(kx_pc_int(*st,pcb,"def_hits"))+" atk_hits="+std::to_string(kx_pc_int(*st,pcb,"atk_hits"))+
                    " alloc_stage="+(text_is(*st,field(*st,pcb,"alloc_stage",false),"attacker")?"attacker":"other")+" p9="+(p9?"1":"0")+" plan="+ck_plan_repr(plan)+" class="+std::to_string(idx_out)+" ctx="+ck_ctx_repr(ctx);return false;}
            }
            branches.push_back({std::move(st),mass});
        }
    }catch(const kx_Mismatch& m){kx_reason="plan_mismatch: "+m.message;return false;}
    kx_merge(branches,out);return true;
}
// chance.kernel_outcomes on `state` for row `row` of `menu`
int kx_kernel_outcomes(const Snapshot& state,const Snapshot& T,const Snapshot& menu,int64_t row,int side,const void* stream,std::vector<kx_Leaf>& leaves,std::string& why){
    leaves.clear();
    const auto rows=child_edges(menu,menu.root);require(row>=0&&static_cast<uint64_t>(row)<rows.size(),"kernel row out of range");
    const std::string kind(rg_row_kind(menu,rows[static_cast<size_t>(row)]));
    kx_reason.clear();
    if(kind!="combat_card_play"&&kind!="combat_card_none"&&kind!="combat_effect_choice"){kx_reason="not a kernel parent";return 1;}
    for(const char* flag:{"_declare_phase","_fellowship_tail_phase","_eye_allocation_phase"})
        if(truth(state,obs_attr(state,state.root,flag,false))){why=std::string("kernel parent under ")+flag+" (ai._resume_preaction_for_search)";return 2;}
    try{
        auto probe=apply_native_random_descriptor(state,T,menu,stream,side,row);
        if(cf_same_rng(stream,probe.random.get())){kx_reason="no RNG";return 1;}          // used == False
        std::vector<kx_Leaf> outcomes;
        if(!kx_expand_combat(state,*probe.state,T,menu,row,side,kind,stream,outcomes))return 1;
        double total=0.0;for(const auto& o:outcomes)total+=o.mass;
        if(outcomes.empty()||std::fabs(total-1.0)>1e-6){kx_reason="outcome mass";return 1;}
        for(auto& o:outcomes){if(!o.state)return 1;leaves.push_back({o.mass/total,std::move(o.state)});}
        return 0;
    }catch(const ck_Error& e){leaves.clear();kx_reason="ck: "+e.why;return 1;}           // chance.kernel_outcomes: except Exception -> None
    catch(const std::exception& e){leaves.clear();kx_reason=std::string("exception: ")+e.what();return 1;}
}
extern "C" {
// chance.kernel_outcomes(world, frame = src's open frame, choice = index) on `base` (or src): out = {status, leaves};
// the leaves become worlds (clone_environment(world, 0, state=state), committed + 1, after_action), taken with
// wotr_raw_world_leaf like an exact expansion's.
WOTR_API int wotr_raw_world_kernel(void* handle,void* base,int64_t index,int64_t* out){
    return checked<int>([&]()->int{
        auto& src=rg_game(handle);const rg_Game& from=base?rg_game(base):src;const auto [menu,row]=cl_row(src,index);
        auto& store=sw_stores()[&src];store.leaves.clear();store.why.clear();
        std::vector<kx_Leaf> leaves;
        std::unique_ptr<void,void(*)(void*)> stream{wotr_random_clone(from.random.get()),wotr_random_destroy};if(!stream)throw std::invalid_argument(wotr_random_last_error());
        out[0]=kx_kernel_outcomes(*from.state,*src.tables,*menu,row,src.frame_side,stream.get(),leaves,store.why);
        if(out[0]==1)store.why=kx_reason;
        if(out[0]==0){
            const std::string kind=sw_row_kind(*menu,row);
            for(auto& leaf:leaves){
                std::unique_ptr<rg_Game> w(sw_new(from,stream.get(),4,std::move(leaf.state)));
                w->committed+=1;rg_after_action(*w,src.frame_phase,src.frame_side,kind);
                store.leaves.push_back({leaf.mass,std::move(w)});
            }
        }
        out[1]=static_cast<int64_t>(store.leaves.size());return 0;
    },-1);
}
}
