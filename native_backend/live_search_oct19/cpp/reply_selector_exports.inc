void* wotr_reply_selector_create(const uint64_t* keys,size_t count,const uint32_t* seed,size_t seed_count,double exploration,double widening){
    try{random_error.clear();return new rs::Selector(keys,count,seed,seed_count,exploration,widening);}
    catch(const std::exception& e){random_error=e.what();return nullptr;}catch(...){random_error="Unknown selector creation error";return nullptr;}
}
void wotr_reply_selector_destroy(void* p){delete static_cast<rs::Selector*>(p);}
int wotr_reply_selector_select(void* p,uint64_t* key){return call([&]{check(key,"Missing reply key output");*key=rs::selector(p).select();});}
int wotr_reply_selector_complete(void* p,uint64_t key,const double* values,size_t count,const double* weights,size_t wcount,double* value){return call([&]{check(value,"Missing reply utility output");*value=rs::selector(p).complete(key,values,count,weights,wcount);});}
int wotr_reply_selector_cancel(void* p){return call([&]{rs::selector(p).pending=UINT64_MAX;});}
int wotr_reply_selector_best(void* p,uint64_t* key){return call([&]{check(key,"Missing reply key output");*key=rs::selector(p).best();});}
int wotr_reply_selector_expected(const double* values,size_t count,const double* weights,size_t wcount,double* value){return call([&]{check(value,"Missing reply utility output");*value=rs::expected(values,count,weights,wcount);});}
// Read-only numeric diagnostics; no graph or Python object serialization.
int wotr_reply_selector_stats(void* p,uint64_t* panels,double* sums,uint64_t* order,size_t count,uint64_t* completed,uint64_t* pending){return call([&]{
    auto& s=rs::selector(p);check(count==s.keys.size()&&panels&&sums&&order&&completed&&pending,"Invalid selector stats buffers");
    for(size_t i=0;i<count;++i){panels[i]=s.stats[i].panels;sums[i]=s.stats[i].utility_sum;order[i]=s.order[i];}
    *completed=s.completed;*pending=s.pending;
});}
void* wotr_reply_selector_rng_clone(void* p){
    try{random_error.clear();return wotr_random_clone(rs::selector(p).rng.get());}
    catch(const std::exception& e){random_error=e.what();return nullptr;}catch(...){random_error="Unknown selector RNG error";return nullptr;}
}
