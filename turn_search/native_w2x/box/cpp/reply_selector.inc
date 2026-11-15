// Reply allocation reference: full_search/r3/runtime/src/reply_selection.py.
// Accurate finite summation follows CPython 3.11.12 mathmodule.c math_fsum;
// license retained in CPYTHON_LICENSE.txt. No Python callbacks or data objects.
namespace rs {
double fsum(const std::vector<double>& values){
    std::vector<double> partials;partials.reserve(32);
    for(double x:values){
        size_t kept=0;
        for(double y:partials){
            if(std::fabs(x)<std::fabs(y))std::swap(x,y);
            volatile double hi=x+y;volatile double yr=hi-x;volatile double lo=y-yr;
            if(lo!=0.0)partials[kept++]=lo;x=hi;
        }
        partials.resize(kept);
        if(x!=0.0){check(std::isfinite(x),"intermediate overflow in fsum");partials.push_back(x);}
    }
    volatile double hi=0.0,lo=0.0,yr=0.0;size_t n=partials.size();
    if(n){
        hi=partials[--n];
        while(n){double x=hi,y=partials[--n];hi=x+y;yr=hi-x;lo=y-yr;if(lo!=0.0)break;}
        if(n&&((lo<0.0&&partials[n-1]<0.0)||(lo>0.0&&partials[n-1]>0.0))){
            double y=lo*2.0,x=hi+y;yr=x-hi;if(y==yr)hi=x;
        }
    }return hi;
}
double expected(const double* values,size_t count,const double* probabilities,size_t pcount){
    check(count&&count==pcount,"One probability is required for each utility");
    check(count<=allocation_limit&&values&&probabilities,"Invalid reply outcome arrays");
    for(size_t i=0;i<count;++i)check(std::isfinite(values[i]),"Reply utilities must be finite");
    std::vector<double> weights;weights.reserve(count);
    for(size_t i=0;i<count;++i){check(std::isfinite(probabilities[i])&&probabilities[i]>=0.0,"Outcome probabilities must be finite and nonnegative");weights.push_back(probabilities[i]);}
    const double total=fsum(weights);check(std::isfinite(total)&&total>0.0,"Outcome probability mass must be positive and finite");
    for(size_t i=0;i<count;++i){volatile double weight=weights[i]/total;weights[i]=weight*values[i];}
    return fsum(weights);
}
struct Arm {uint64_t panels=0;double utility_sum=0.0;double mean()const{return utility_sum/double(panels);}};
struct Selector {
    std::vector<uint64_t> keys,order;std::vector<Arm> stats;
    double exploration=1.25,widening=2.0;uint64_t completed=0,pending=UINT64_MAX;
    std::unique_ptr<void,decltype(&wotr_random_destroy)> rng{nullptr,wotr_random_destroy};
    Selector(const uint64_t* ids,size_t count,const uint32_t* seed,size_t seed_count,double exp,double widen):exploration(exp),widening(widen){
        check(count&&count<=allocation_limit&&ids,"A nonempty unique legal action menu is required");
        keys.assign(ids,ids+count);std::unordered_set<uint64_t> unique(keys.begin(),keys.end());check(unique.size()==count,"A nonempty unique legal action menu is required");
        check(std::isfinite(exp)&&exp>0.0,"Exploration must be positive and finite");check(std::isfinite(widen)&&widen>0.0,"Widening must be positive and finite");
        rng.reset(wotr_random_create_seeded(seed,seed_count));check(rng!=nullptr,wotr_random_last_error());
        order.resize(count);stats.resize(count);
        check(wotr_random_shuffle(rng.get(),order.data(),count)==0,wotr_random_last_error());
    }
    uint64_t select(){
        check(pending==UINT64_MAX,"Complete or cancel the pending reply before selecting again");
        check(completed<UINT64_MAX,"Reply completion count exceeds native integer range");
        const double span=widening*std::sqrt(double(completed+1));check(std::isfinite(span),"cannot convert float infinity to integer");
        size_t width=span>=double(keys.size())?keys.size():std::max<size_t>(1,size_t(std::ceil(span)));
        uint64_t chosen=UINT64_MAX;
        for(size_t i=0;i<width;++i)if(stats[order[i]].panels==0){chosen=order[i];break;}
        if(chosen==UINT64_MAX){
            double best=0.0;
            for(size_t i=0;i<width;++i){const auto row=order[i];const auto& arm=stats[row];
                const double score=arm.mean()-exploration*std::sqrt(std::log(double(completed+1))/double(arm.panels));
                if(i==0||score<best){best=score;chosen=row;}
            }
        }
        pending=chosen;return keys[chosen];
    }
    double complete(uint64_t key,const double* values,size_t count,const double* probabilities,size_t pcount){
        check(pending!=UINT64_MAX&&keys[pending]==key,"Completion does not identify the pending reply");
        const double value=expected(values,count,probabilities,pcount);auto& arm=stats[pending];
        const double updated=arm.utility_sum+value;check(std::isfinite(updated),"Accumulated reply utility is not finite");
        check(arm.panels<UINT64_MAX&&completed<UINT64_MAX,"Reply completion count exceeds native integer range");
        arm.utility_sum=updated;++arm.panels;++completed;pending=UINT64_MAX;return value;
    }
    uint64_t best()const{
        uint64_t chosen=UINT64_MAX;double value=0;
        for(size_t i=0;i<stats.size();++i)if(stats[i].panels&&(chosen==UINT64_MAX||stats[i].mean()<value)){chosen=i;value=stats[i].mean();}
        check(chosen!=UINT64_MAX,"No completed reply evaluations");return keys[chosen];
    }
};
Selector& selector(void* p){check(p,"Null reply selector");return *static_cast<Selector*>(p);}
}
