// Explicit combat_dist.run_with_plan scope. Ordinary RNG calls keep their path.
// No Python callbacks, process-global monkeypatches, or fabricated child state.
struct cf_Plan {
    const int64_t* pairs; size_t size, next=0; bool confusion;
};
thread_local cf_Plan* cf_active=nullptr;
struct cf_Scope {
    cf_Plan plan;
    cf_Scope(const int64_t* pairs,size_t count,bool confusion):plan{pairs,count,0,confusion}{
        require(!cf_active,"Nested native combat forcing is not supported");
        check_array(pairs,count*2);cf_active=&plan;
    }
    ~cf_Scope(){cf_active=nullptr;}
    void done(){require(plan.next==plan.size,"Combat plan not fully consumed");}
};
void cf_d6(void* rng,uint32_t* faces,size_t count,int64_t threshold,int64_t modifier,bool combat_roll,const char* context=nullptr){
    if(!cf_active){if(wotr_random_d6(rng,faces,count))throw std::invalid_argument(wotr_random_last_error());return;}
    // Source leader_reroll returns before roll_d6_batch on zero misses.
    if(!count)return;
    require(count<=5,"Forced combat batch exceeds exact-kernel domain");
    require(threshold>=-1048576&&threshold<=1048576&&modifier>=-1048576&&modifier<=1048576,"Forced combat threshold/modifier exceeds supported domain");
    auto& plan=*cf_active;if(!context)context=combat_roll?"combat roll":"leader reroll";cfc_require(plan.next<plan.size,"Combat plan exhausted (extra batch)",1,plan.next,context,count,threshold,modifier);
    const auto hits=plan.pairs[plan.next*2],stored_ones=plan.pairs[plan.next*2+1];++plan.next;
    const bool track=combat_roll&&(plan.confusion||modifier>=threshold-1);
    const bool tuple=cfc_active?cfc_active->tuples[plan.next-1]!=0:stored_ones>=0;
    const int64_t ones=cfc_active?(tuple?stored_ones:0):(stored_ones<0?0:stored_ones);
    cfc_require(hits>=0&&ones>=0&&hits<=static_cast<int64_t>(count)&&ones<=static_cast<int64_t>(count)&&hits+ones<=static_cast<int64_t>(count)&&stored_ones>=-1,"Combat statistic outside batch support",2,plan.next-1,context,count,threshold,modifier);
    cfc_require(track||(cfc_active?!tuple:stored_ones==-1),"Tuple combat statistic in scalar batch",2,plan.next-1,context,count,threshold,modifier);
    int hit_faces=0;for(int face=2;face<=6;++face)hit_faces+=face==6||modifier>=threshold-face;
    const double ph=hit_faces/6.0,po=1.0/6.0,pm=track?1.0-ph-po:1.0-ph;
    // Preserve source floating support, including tiny nonzero multinomial terms.
    const double probability=std::pow(ph,hits)*std::pow(track?po:1.0,ones)*std::pow(pm,count-hits-ones);
    cfc_require(probability>0.0,"Combat statistic outside batch support",2,plan.next-1,context,count,threshold,modifier);
    uint32_t miss=1;for(int face=5;face>=2;--face)if(modifier<threshold-face){miss=face;break;}
    size_t at=0;for(int64_t i=0;i<hits;++i)faces[at++]=6;for(int64_t i=0;i<ones;++i)faces[at++]=1;
    while(at<count)faces[at++]=miss;
}
bool cf_same_rng(const void* before,const void* after){
    std::array<uint32_t,624> a{},b{};uint32_t ai=0,bi=0;int32_t ag=0,bg=0;uint64_t av=0,bv=0;
    if(wotr_random_export(before,a.data(),a.size(),&ai,&ag,&av)||wotr_random_export(after,b.data(),b.size(),&bi,&bg,&bv))throw std::invalid_argument(wotr_random_last_error());
    return a==b&&ai==bi&&ag==bg&&av==bv;
}
