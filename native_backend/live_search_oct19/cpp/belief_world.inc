// Actor-conditioned sampler matching full_search/r3/root_hand.py.
// All graph copying, canonicalization and sampling stays in native storage.
namespace bw {
std::string quoted(std::string_view value){
    std::string out="\"";constexpr char hex[]="0123456789abcdef";
    auto escape=[&](uint32_t c){out+="\\u";for(int n=12;n>=0;n-=4)out+=hex[(c>>n)&15];};
    for(size_t i=0;i<value.size();){
        uint32_t c=static_cast<unsigned char>(value[i++]);
        if(c>=128){int extra=c<224?1:c<240?2:3;c&=(1u<<(6-extra))-1;while(extra--){require(i<value.size(),"Invalid tile UTF-8");c=(c<<6)|(static_cast<unsigned char>(value[i++])&63);}}
        if(c=='"'||c=='\\'){out+='\\';out+=char(c);}
        else if(c=='\b')out+="\\b";else if(c=='\f')out+="\\f";
        else if(c=='\n')out+="\\n";else if(c=='\r')out+="\\r";else if(c=='\t')out+="\\t";
        else if(c<32||c>=127){if(c<=65535)escape(c);else{c-=65536;escape(0xd800+(c>>10));escape(0xdc00+(c&1023));}}
        else out+=char(c);
    }return out+'"';
}
std::string real_json(double value){
    require(std::isfinite(value),"Out of range float values are not JSON compliant");
    char buffer[128];auto encoded=std::to_chars(buffer,buffer+sizeof(buffer),value,std::chars_format::scientific);
    require(encoded.ec==std::errc(),"Cannot encode tile floating property");
    std::string raw(buffer,encoded.ptr);auto e=raw.find('e');int exponent=std::stoi(raw.substr(e+1));
    if(exponent < -4 || exponent >=16)return raw;
    bool neg=raw[0]=='-';std::string digits=raw.substr(neg?1:0,e-(neg?1:0));
    auto dot=digits.find('.');if(dot!=digits.npos)digits.erase(dot,1);
    std::string result=neg?"-":"";int split=exponent+1;
    if(split<=0)result+="0."+std::string(-split,'0')+digits;
    else if(size_t(split)>=digits.size())result+=digits+std::string(split-digits.size(),'0')+".0";
    else result+=digits.substr(0,split)+"."+digits.substr(split);
    return result;
}
std::string json_value(const Snapshot& s,uint64_t n,std::set<uint64_t>& active){
    const auto& v=s.nodes[n];
    switch(v.tag){
    case WOTR_NONE:return "null";case WOTR_BOOL:return v.integer?"true":"false";
    case WOTR_INT:return std::to_string(v.integer);case WOTR_FLOAT:return real_json(v.real);
    case WOTR_STRING:return quoted(text(s,n));
    case WOTR_BIGINT:return std::string(reinterpret_cast<const char*>(s.bytes.data()+v.offset),v.count);
    }
    require(active.insert(n).second,"Circular reference in tile property");
    std::string out;
    if(v.tag==WOTR_DICT){
        std::vector<std::pair<std::string,uint64_t>> rows;auto es=child_edges(s,n);
        for(size_t i=0;i<es.size();i+=2){require(s.nodes[es[i]].tag==WOTR_STRING,"Tile JSON attribute keys must be strings");rows.emplace_back(text(s,es[i]),es[i+1]);}
        std::stable_sort(rows.begin(),rows.end(),[](const auto& a,const auto& b){return a.first<b.first;});out="{";
        for(size_t i=0;i<rows.size();++i){if(i)out+=',';out+=quoted(rows[i].first)+":"+json_value(s,rows[i].second,active);}out+='}';
    }else{
        require(v.tag==WOTR_LIST||v.tag==WOTR_TUPLE,"Tile property is not JSON serializable");out="[";auto es=child_edges(s,n);
        for(size_t i=0;i<es.size();++i){if(i)out+=',';out+=json_value(s,es[i],active);}out+=']';
    }active.erase(n);return out;
}
void canonicalize(Snapshot& s,int actor){
    for(int d=0;d<2;++d){auto n=obs_attr(s,s.root,so_attr(actor,d).c_str(),false);if(ctx_none(s,n))continue;
        auto xs=child_edges(s,n);std::stable_sort(xs.begin(),xs.end(),[&](auto a,auto b){return text(s,a)<text(s,b);});so_assign(s,n,xs);}
    auto pool=obs_attr(s,s.root,"hunt_pool",false);if(ctx_none(s,pool))return;
    std::vector<std::pair<std::string,uint64_t>> rows;
    for(auto tile:child_edges(s,pool)){std::set<uint64_t> active;rows.emplace_back(json_value(s,generic_attributes(s,tile),active),tile);}
    std::stable_sort(rows.begin(),rows.end(),[](const auto& a,const auto& b){return a.first<b.first;});
    std::vector<uint64_t> ordered;for(const auto& row:rows)ordered.push_back(row.second);so_assign(s,pool,ordered);
}
void blind(Snapshot& s,int actor){
    auto pc=obs_attr(s,s.root,"pending_combat",false);
    if(!truth(s,pc)||!truth(s,field(s,pc,"atk_combat_card",false))||actor==cbb_side(s,pc,0))return;
    if(!truth(s,field(s,pc,"atk_hidden_cc",false))&&!text_is(s,field(s,pc,"phase",false),"defender_combat_card"))return;
    const auto cid=std::string(text(s,field(s,pc,"atk_combat_card")));auto public_dict=obs_attr(s,s.root,"discard_face_up",false);
    bool found=false;
    if(!ctx_none(s,public_dict)){auto es=child_edges(s,public_dict);
        for(size_t j=0;j<es.size()&&!found;j+=2){auto xs=child_edges(s,es[j+1]);
            for(size_t k=xs.size();k>0;--k)if(text_is(s,xs[k-1],cid)){
                xs[k-1]=rr_string(s,"_committed_face_down_");so_assign(s,es[j+1],xs);
                rr_set_attr(s,s.root,"_commit_slot",cp_container(s,WOTR_TUPLE,{es[j],rr_scalar(s,WOTR_INT,k-1)}));found=true;break;
            }
        }
    }
    cp_put(s,pc,"_commit_nonce",rr_scalar(s,WOTR_BOOL,1));
}
struct World {std::unique_ptr<Snapshot> state;std::unique_ptr<void,decltype(&wotr_random_destroy)> rng{nullptr,wotr_random_destroy};};
// Python copy.deepcopy detaches registered global dice-face lists too.
// Retain internal aliases/values, but do not let a later bonus die join an old
// copied list through the import/export codec's external-global marker.
std::unique_ptr<Snapshot> deepcopy(const Snapshot& source){
    auto result=std::make_unique<Snapshot>(source);
    for(auto& node:result->nodes)if(node.tag==WOTR_LIST&&node.flags==1){node.flags=0;node.integer=0;}
    return result;
}
World sample(const Snapshot& source,const Snapshot& tables,int actor,const void* initial_rng){
    require(actor==0||actor==1,"Belief actor must be FP or SH");require(initial_rng,"Belief RNG is null");
    World world;world.state=deepcopy(source);world.rng.reset(wotr_random_clone(initial_rng));
    require(world.rng!=nullptr,"Could not clone belief RNG");canonicalize(*world.state,actor);blind(*world.state,actor);
    auto overlay=so_sample(*world.state,tables,actor,world.rng.get());so_patch(*overlay,*world.state,world.rng.get());
    return world;
}
}
