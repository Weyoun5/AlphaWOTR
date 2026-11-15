// CPython 3.11.12 Random compatibility, imported and integer-seeded. MT19937's
// transition and tempering constants follow the published standard. Sequence
// draw ordering is verified against installed Lib/random.py and CPython's
// Modules/_randommodule.c (v3.11.12). No Python headers or callbacks.
#include "wotr_random.h"
#include "wotr_reply_selector.h"
#include <array>
#include <vector>
#include <unordered_set>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace {
thread_local std::string random_error;
thread_local bool forbidden_draw_scope=false;
void check(bool condition,const char* why){if(!condition)throw std::invalid_argument(why);}
constexpr size_t allocation_limit=1048576;
struct State {
    std::array<uint32_t,624> words{};
    uint32_t index=624;
    int32_t has_gauss=0;
    uint64_t gauss_bits=0;
    uint32_t next(){
        check(!forbidden_draw_scope,"raw RNG in driven Hunt walk");
        if(index==624){
            for(size_t i=0;i<624;++i){
                const auto combined=(words[i]&0x80000000U)|(words[(i+1)%624]&0x7fffffffU);
                words[i]=words[(i+397)%624]^(combined>>1)^((combined&1U)?0x9908b0dfU:0U);
            }
            index=0;
        }
        auto value=words[index++];
        value^=value>>11;value^=(value<<7)&0x9d2c5680U;
        value^=(value<<15)&0xefc60000U;value^=value>>18;return value;
    }
    double real(){const auto high=next()>>5,low=next()>>6;return (high*67108864.0+low)*(1.0/9007199254740992.0);}
    uint64_t bits(unsigned count){
        if(!count)return 0;
        if(count<=32)return next()>>(32-count);
        uint64_t value=next();return value|(uint64_t(next()>>(64-count))<<32);
    }
    uint64_t below(uint64_t bound){
        unsigned count=0;for(auto x=bound;x;x>>=1)++count;
        uint64_t value;do{value=bits(count);}while(value>=bound);return value;
    }
};
#include "seed_integer.inc"
struct Stream {State state;mutable std::mutex mutex;};
Stream& handle(void* p){check(p!=nullptr,"Null random stream");return *static_cast<Stream*>(p);}
const Stream& handle(const void* p){check(p!=nullptr,"Null random stream");return *static_cast<const Stream*>(p);}
State imported(const uint32_t* words,size_t count,uint32_t index,int32_t has,uint64_t gauss){
    check(words && count==624,"Random state requires exactly 624 uint32 words");
    check(index<=624,"Random state index must be 0..624");check(has==0||has==1,"Gaussian presence must be boolean");
    check(has || !gauss,"Absent Gaussian payload must be zero");
    State result;std::copy(words,words+624,result.words.begin());result.index=index;result.has_gauss=has;result.gauss_bits=gauss;return result;
}
template<class F>int call(F&& f){try{random_error.clear();f();return 0;}catch(const std::exception& e){random_error=e.what();return -1;}catch(...){random_error="Unknown random-stream error";return -1;}}
template<class T>void output(T* p,size_t count){check(!count||p,"Missing random output buffer");check(count<=allocation_limit,"Random batch exceeds supported allocation bound");}
}
#include "reply_selector.inc"
extern "C" {
#include "reply_selector_exports.inc"
int wotr_random_begin_forbidden_scope(){return call([&]{check(!forbidden_draw_scope,"Nested forbidden RNG scope");forbidden_draw_scope=true;});}
void wotr_random_end_forbidden_scope(){forbidden_draw_scope=false;}
const char* wotr_random_last_error(){return random_error.c_str();}
void* wotr_random_create(const uint32_t* words,size_t n,uint32_t index,int32_t has,uint64_t gauss){
    try{random_error.clear();auto state=imported(words,n,index,has,gauss);auto result=std::make_unique<Stream>();result->state=state;return result.release();}
    catch(const std::exception& e){random_error=e.what();return nullptr;}catch(...){random_error="Unknown random-stream error";return nullptr;}
}
void* wotr_random_create_seeded(const uint32_t* limbs,size_t count){
    try{random_error.clear();auto state=seeded_integer(limbs,count);auto result=std::make_unique<Stream>();result->state=state;return result.release();}
    catch(const std::exception& e){random_error=e.what();return nullptr;}catch(...){random_error="Unknown integer-seed error";return nullptr;}
}
void* wotr_random_clone(const void* p){
    try{random_error.clear();const auto& original=handle(p);std::lock_guard<std::mutex> guard(original.mutex);auto result=std::make_unique<Stream>();result->state=original.state;return result.release();}
    catch(const std::exception& e){random_error=e.what();return nullptr;}catch(...){random_error="Unknown random-stream error";return nullptr;}
}
void wotr_random_destroy(void* p){delete static_cast<Stream*>(p);}
int wotr_random_export(const void* p,uint32_t* words,size_t n,uint32_t* index,int32_t* has,uint64_t* gauss){return call([&]{
    const auto& stream=handle(p);check(words&&n==624&&index&&has&&gauss,"Invalid random-state export buffers");
    std::lock_guard<std::mutex> guard(stream.mutex);std::copy(stream.state.words.begin(),stream.state.words.end(),words);
    *index=stream.state.index;*has=stream.state.has_gauss;*gauss=stream.state.gauss_bits;
});}
int wotr_random_setstate(void* p,const uint32_t* words,size_t n,uint32_t index,int32_t has,uint64_t gauss){return call([&]{
    auto& stream=handle(p);auto state=imported(words,n,index,has,gauss);std::lock_guard<std::mutex> guard(stream.mutex);stream.state=state;
});}
int wotr_random_double(void* p,double* out){return call([&]{auto& stream=handle(p);check(out,"Missing double output");std::lock_guard<std::mutex> guard(stream.mutex);*out=stream.state.real();});}
int wotr_random_getrandbits(void* p,uint32_t bits,uint32_t* out,size_t count){return call([&]{
    auto& stream=handle(p);check(bits<=allocation_limit,"getrandbits exceeds supported bit bound");
    check(count==(uint64_t(bits)+31)/32,"Incorrect getrandbits limb count");output(out,count);
    std::lock_guard<std::mutex> guard(stream.mutex);for(size_t i=0;i<count;++i){unsigned take=std::min<uint32_t>(32,bits-uint32_t(i*32));out[i]=stream.state.next()>>(32-take);}
});}
int wotr_random_randbelow(void* p,uint64_t bound,uint64_t* out){return call([&]{
    auto& stream=handle(p);check(bound&&out,"randbelow requires positive bound and output");std::lock_guard<std::mutex> guard(stream.mutex);*out=stream.state.below(bound);
});}
int wotr_random_randint(void* p,int64_t low,int64_t high,int64_t* out){return call([&]{
    auto& stream=handle(p);check(low<=high&&out,"Invalid randint range/output");const auto width=uint64_t(high)-uint64_t(low)+1;
    check(width,"Full 2**64 randint range is unsupported");std::lock_guard<std::mutex> guard(stream.mutex);
    auto value=uint64_t(low)+stream.state.below(width);static_assert(sizeof(value)==sizeof(*out),"int64 layout");std::memcpy(out,&value,sizeof(value));
});}
int wotr_random_d6(void* p,uint32_t* out,size_t n){return call([&]{auto& stream=handle(p);output(out,n);std::lock_guard<std::mutex> guard(stream.mutex);for(size_t i=0;i<n;++i)out[i]=uint32_t(stream.state.below(6))+1;});}
int wotr_random_shuffle(void* p,uint64_t* out,size_t n){return call([&]{
    auto& stream=handle(p);output(out,n);std::lock_guard<std::mutex> guard(stream.mutex);
    for(size_t i=0;i<n;++i)out[i]=i;
    for(size_t i=n;i>1;--i)std::swap(out[i-1],out[stream.state.below(i)]);
});}
int wotr_random_sample(void* p,uint64_t n,size_t k,uint64_t* out,size_t count){return call([&]{
    auto& stream=handle(p);check(count==k&&k<=n,"Invalid sample size");output(out,count);
    size_t threshold=21;if(k>5){size_t power=1;while(power<k*3)power*=4;threshold+=power;}
    std::vector<uint64_t> pool,result(k);std::unordered_set<uint64_t> selected;
    if(n<=threshold){pool.resize(size_t(n));for(size_t i=0;i<pool.size();++i)pool[i]=i;}else selected.reserve(k);
    std::lock_guard<std::mutex> guard(stream.mutex);auto changed=stream.state;
    for(size_t i=0;i<k;++i){
        auto value=changed.below(n-(pool.empty()?0:i));
        if(!pool.empty()){result[i]=pool[size_t(value)];pool[size_t(value)]=pool[size_t(n-i-1)];}
        else {while(selected.find(value)!=selected.end())value=changed.below(n);selected.insert(value);result[i]=value;}
    }
    std::copy(result.begin(),result.end(),out);stream.state=changed;
});}
int wotr_random_choices(void* p,uint64_t n,const double* cumulative,size_t weights,size_t k,uint64_t* out,size_t count){return call([&]{
    auto& stream=handle(p);check(n>0&&n<=9007199254740992ULL,"choices population outside supported 1..2**53 range");check(count==k,"Incorrect choices output count");output(out,count);
    double total=double(n);if(cumulative){check(weights==n,"Incorrect cumulative weight count");total=cumulative[n-1];check(std::isfinite(total)&&total>0,"Invalid total weight");
        for(size_t i=0;i<weights;++i)check(std::isfinite(cumulative[i]),"Nonfinite cumulative weight unsupported");
    }else check(weights==0,"Weights pointer missing");
    std::lock_guard<std::mutex> guard(stream.mutex);
    for(size_t i=0;i<k;++i){const double target=stream.state.real()*total;
        if(!cumulative)out[i]=uint64_t(std::floor(target));
        else{size_t low=0,high=weights-1;while(low<high){auto mid=low+(high-low)/2;if(target<cumulative[mid])high=mid;else low=mid+1;}out[i]=low;}
    }
});}
}
