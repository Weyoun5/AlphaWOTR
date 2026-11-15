// Explicit compatibility context; ordinary and legacy forcing leave it null.
struct cfc_Error:std::exception {
    int kind;size_t index;std::string context;int64_t count,threshold,modifier;
    cfc_Error(int k,size_t i,const char* c,int64_t n,int64_t t,int64_t m):kind(k),index(i),context(c?c:""),count(n),threshold(t),modifier(m){}
    const char* what()const noexcept override{return "Source-compatible combat plan mismatch";}
};
struct cfc_Config {const uint8_t* tuples;size_t count;};
thread_local cfc_Config* cfc_active=nullptr;
struct cfc_Scope {
    cfc_Config config;
    cfc_Scope(const uint8_t* tuples,size_t count):config{tuples,count}{require(!cfc_active,"Nested compatible combat forcing is not supported");check_array(tuples,count);for(size_t i=0;i<count;++i)require(tuples[i]<=1,"Combat statistic kind must be scalar or tuple");cfc_active=&config;}
    ~cfc_Scope(){cfc_active=nullptr;}
};
void cfc_require(bool condition,const char* legacy_message,int kind,size_t index,const char* context,size_t count,int64_t threshold,int64_t modifier){
    if(condition)return;if(cfc_active)throw cfc_Error(kind,index,context,static_cast<int64_t>(count),threshold,modifier);throw std::invalid_argument(legacy_message);
}
