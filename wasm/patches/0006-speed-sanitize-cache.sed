# file: cpp/snapshot.cpp
# targets: wasm32 wasm64 native
# Speed (phase 5 profile: sanitize_name was 11% of a 300-simulation decision in wasm, mostly re-sanitizing the same
# few hundred region and army keys): the result of sanitize_name memoized per input in a bounded table (cleared at
# 8,192 entries). Only successful results are kept; an invalid name still throws, so the function is unchanged.
# string_value_edge (6.7%) returned the last matching key of a dictionary after scanning all of it; it now scans from
# the end and stops at the first match from the end, the same edge.
s/^std::string sanitize_name(std::string_view value){/std::string sanitize_name_uncached(std::string_view value);\
std::string sanitize_name(std::string_view value){\
    static thread_local std::unordered_map<std::string,std::string> cache;\
    const std::string key(value);auto it=cache.find(key);if(it!=cache.end())return it->second;\
    std::string result=sanitize_name_uncached(value);if(cache.size()>=8192)cache.clear();cache.emplace(key,result);return result;\
}\
std::string sanitize_name_uncached(std::string_view value){/
/^uint64_t string_value_edge(const Snapshot& s,uint64_t dict,std::string_view key){/,/^}/{
s/    for(uint64_t i=0;i<d.count;i+=2)/    for(uint64_t i=d.count;i>=2;i-=2)/
s/        if(text_is(s,s.edges\[d.offset+i\],key))result=d.offset+i+1;/        if(text_is(s,s.edges[d.offset+i-2],key)){result=d.offset+i-1;break;}/
}
