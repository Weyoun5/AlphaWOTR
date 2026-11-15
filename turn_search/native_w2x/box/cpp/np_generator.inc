// numpy 2.2.5 np.random.default_rng(seed) on the native side (claude_bugscan_2026-10-21/native_cl, 2026-10-27): the
// generator selfplay_actor_critic.search.determinize_state draws from (np.random.default_rng(decision_seed)).
//   SeedSequence(seed): entropy = the seed's little-endian uint32 words, pool of 4 words mixed with hashmix / mix;
//   generate_state(4, uint64); PCG64 (XSL-RR 128/64) seeded with pcg64_set_seed(state = v[0]:v[1], inc = v[2]:v[3]);
//   next_uint32 halves a 64-bit draw (low first, high buffered); next_double = (next64 >> 11) * 2^-53;
//   Generator.permutation(n) = arange + _shuffle_raw (Fisher-Yates from the top, random_interval masked rejection);
//   Generator.choice(n, p) = cumsum / last, one next_double, searchsorted(side='right'); numpy's pairwise float sum.
// Verified against numpy itself (tools/np_generator_check.py).
#include <intrin.h>
struct np_Pcg64 {
    unsigned __int64 state_hi=0,state_lo=0,inc_hi=0,inc_lo=0;bool has32=false;uint32_t u32=0;
};
// 128-bit helpers (MSVC: _umul128)
inline void np_mul128(uint64_t ahi,uint64_t alo,uint64_t bhi,uint64_t blo,uint64_t& rhi,uint64_t& rlo){
    uint64_t hi;const uint64_t lo=_umul128(alo,blo,&hi);hi+=ahi*blo+alo*bhi;rhi=hi;rlo=lo;
}
inline void np_add128(uint64_t& hi,uint64_t& lo,uint64_t bhi,uint64_t blo){const uint64_t l=lo+blo;hi+=bhi+(l<lo?1:0);lo=l;}
inline void np_step(np_Pcg64& g){
    uint64_t hi,lo;np_mul128(g.state_hi,g.state_lo,0x2360ED051FC65DA4ULL,0x4385DF649FCCF645ULL,hi,lo);
    np_add128(hi,lo,g.inc_hi,g.inc_lo);g.state_hi=hi;g.state_lo=lo;
}
inline uint64_t np_next64(np_Pcg64& g){
    np_step(g);const uint64_t x=g.state_hi^g.state_lo;const unsigned r=static_cast<unsigned>(g.state_hi>>58);
    return (x>>r)|(x<<((64u-r)&63u));
}
inline uint32_t np_next32(np_Pcg64& g){
    if(g.has32){g.has32=false;return g.u32;}
    const uint64_t n=np_next64(g);g.has32=true;g.u32=static_cast<uint32_t>(n>>32);return static_cast<uint32_t>(n&0xffffffffULL);
}
inline double np_next_double(np_Pcg64& g){return static_cast<double>(np_next64(g)>>11)*(1.0/9007199254740992.0);}
np_Pcg64 np_default_rng(uint64_t seed){
    constexpr uint32_t INIT_A=0x43b0d7e5u,MULT_A=0x931e8875u,INIT_B=0x8b51f9ddu,MULT_B=0x58f38dedu,MIX_L=0xca01f9ddu,MIX_R=0x4973f715u;
    std::vector<uint32_t> ent;if(seed==0)ent.push_back(0);while(seed){ent.push_back(static_cast<uint32_t>(seed&0xffffffffULL));seed>>=32;}
    uint32_t h=INIT_A;
    auto hashmix=[&](uint32_t v){v^=h;h*=MULT_A;v*=h;v^=v>>16;return v;};
    auto mix=[&](uint32_t x,uint32_t y){uint32_t r=MIX_L*x-MIX_R*y;r^=r>>16;return r;};
    uint32_t pool[4];
    for(size_t i=0;i<4;++i)pool[i]=hashmix(i<ent.size()?ent[i]:0u);
    for(size_t s=0;s<4;++s)for(size_t d=0;d<4;++d)if(s!=d)pool[d]=mix(pool[d],hashmix(pool[s]));
    for(size_t s=4;s<ent.size();++s)for(size_t d=0;d<4;++d)pool[d]=mix(pool[d],hashmix(ent[s]));
    uint32_t hb=INIT_B,w[8];
    for(size_t i=0;i<8;++i){uint32_t v=pool[i%4];v^=hb;hb*=MULT_B;v*=hb;v^=v>>16;w[i]=v;}
    uint64_t v64[4];for(size_t i=0;i<4;++i)v64[i]=static_cast<uint64_t>(w[2*i])|(static_cast<uint64_t>(w[2*i+1])<<32);
    np_Pcg64 g;
    // pcg_setseq_128_srandom_r(initstate = v[0]:v[1], initseq = v[2]:v[3])
    g.inc_hi=(v64[2]<<1)|(v64[3]>>63);g.inc_lo=(v64[3]<<1)|1u;
    g.state_hi=0;g.state_lo=0;np_step(g);np_add128(g.state_hi,g.state_lo,v64[0],v64[1]);np_step(g);
    return g;
}
// distributions.c random_interval
inline uint64_t np_interval(np_Pcg64& g,uint64_t max){
    if(max==0)return 0;
    uint64_t mask=max;mask|=mask>>1;mask|=mask>>2;mask|=mask>>4;mask|=mask>>8;mask|=mask>>16;mask|=mask>>32;
    uint64_t value;
    if(max<=0xffffffffULL){while((value=(np_next32(g)&mask))>max){}}
    else{while((value=(np_next64(g)&mask))>max){}}
    return value;
}
// Generator.permutation(n)
std::vector<size_t> np_permutation(np_Pcg64& g,size_t n){
    std::vector<size_t> a(n);for(size_t i=0;i<n;++i)a[i]=i;
    for(size_t i=n;i-->1;){const size_t j=static_cast<size_t>(np_interval(g,i));std::swap(a[i],a[j]);}
    return a;
}
// numpy pairwise_sum_DOUBLE (np.sum of a contiguous float64 vector)
double np_pairwise_sum(const double* a,size_t n){
    if(n<8){double r=0.0;for(size_t i=0;i<n;++i)r+=a[i];return r;}
    if(n<=128){
        double r[8];for(size_t j=0;j<8;++j)r[j]=a[j];
        size_t i=8;for(;i<n-(n%8);i+=8)for(size_t j=0;j<8;++j)r[j]+=a[i+j];
        double res=((r[0]+r[1])+(r[2]+r[3]))+((r[4]+r[5])+(r[6]+r[7]));
        for(;i<n;++i)res+=a[i];return res;
    }
    size_t n2=n/2;n2-=n2%8;return np_pairwise_sum(a,n2)+np_pairwise_sum(a+n2,n-n2);
}
// rng.choice(len(w), p=w / w.sum()) for non-negative weights summing to a positive number
size_t np_choice_weighted(np_Pcg64& g,const std::vector<double>& w){
    require(!w.empty(),"numpy choice from an empty population");
    const double total=np_pairwise_sum(w.data(),w.size());require(total>0.0,"numpy choice weights must sum to a positive value");
    std::vector<double> cdf(w.size());double acc=0.0;
    for(size_t i=0;i<w.size();++i){acc+=w[i]/total;cdf[i]=acc;}
    const double last=cdf.back();for(auto& c:cdf)c/=last;
    const double u=np_next_double(g);
    return static_cast<size_t>(std::upper_bound(cdf.begin(),cdf.end(),u)-cdf.begin());
}
