// CPython integer-seed normalization and MT19937 init_by_array.
// Algorithm reference: CPython v3.11.12 Modules/_randommodule.c, lines 164-209,
// 267-303. MT authors' copyright/terms are in MT19937_LICENSE.txt.
// Unsigned host uint32_t limbs encode a little-endian integer, not a byte span.
State seeded_integer(const uint32_t* limbs,size_t count){
    check(count<=allocation_limit,"Integer seed exceeds supported limb bound");
    check(count==0||limbs,"Missing integer seed limbs");
    // CPython sizes the array from the integer's bit length. Zero has one limb;
    // redundant high zero limbs must not change the integer's initialization.
    while(count>0&&limbs[count-1]==0)--count;
    const uint32_t zero=0;if(count==0){limbs=&zero;count=1;}
    State result;auto& w=result.words;w[0]=19650218U;
    for(size_t i=1;i<w.size();++i)w[i]=1812433253U*(w[i-1]^(w[i-1]>>30))+uint32_t(i);
    size_t i=1,j=0;
    for(size_t remaining=std::max(w.size(),count);remaining>0;--remaining){
        w[i]=(w[i]^((w[i-1]^(w[i-1]>>30))*1664525U))+limbs[j]+uint32_t(j);
        if(++i==w.size()){w[0]=w.back();i=1;}
        if(++j==count)j=0;
    }
    for(size_t remaining=w.size()-1;remaining>0;--remaining){
        w[i]=(w[i]^((w[i-1]^(w[i-1]>>30))*1566083941U))-uint32_t(i);
        if(++i==w.size()){w[0]=w.back();i=1;}
    }
    w[0]=0x80000000U;
    // Default State supplies index 624 and clears the Python Gaussian cache.
    return result;
}
