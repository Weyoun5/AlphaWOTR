// reply_graph_key_v1: independent native encoding of reachable semantic graph.
namespace rgk {
void u64(std::string& out,uint64_t n){for(unsigned i=0;i<8;++i)out+=char((n>>(8*i))&255);}
void blob(std::string& out,std::string_view value){u64(out,value.size());out.append(value.data(),value.size());}
void real(std::string& out,double value){uint64_t bits;std::memcpy(&bits,&value,8);u64(out,bits);}
bool scalar(const Snapshot& s,uint64_t n,std::string& out){
    const auto& v=s.nodes[n];
    switch(v.tag){
    case WOTR_NONE:out+='N';return true;
    case WOTR_BOOL:out+='B';out+=char(v.integer!=0);return true;
    case WOTR_INT:out+='I';blob(out,std::to_string(v.integer));return true;
    case WOTR_BIGINT:out+='I';blob(out,std::string_view(reinterpret_cast<const char*>(s.bytes.data()+v.offset),v.count));return true;
    case WOTR_FLOAT:out+='F';real(out,v.real);return true;
    case WOTR_STRING:case WOTR_BYTES:out+=v.tag==WOTR_STRING?'U':'Y';blob(out,std::string_view(reinterpret_cast<const char*>(s.bytes.data()+v.offset),v.count));return true;
    case WOTR_ENUM:{auto xs=child_edges(s,n);require(xs.size()==2,"Malformed identity enum");out+='E';blob(out,text(s,xs[0]));blob(out,text(s,xs[1]));return true;}
    default:return false;
    }
}
std::string atom(const Snapshot&,uint64_t);
std::vector<uint64_t> sorted_members(const Snapshot& s,uint64_t n){
    std::vector<std::pair<std::string,uint64_t>> values;for(auto x:child_edges(s,n))values.emplace_back(atom(s,x),x);
    std::sort(values.begin(),values.end(),[](const auto& a,const auto& b){return a.first<b.first;});
    std::vector<uint64_t> out;for(size_t i=0;i<values.size();++i){require(i==0||values[i-1].first!=values[i].first,"Ambiguous typed set member order");out.push_back(values[i].second);}return out;
}
std::string atom(const Snapshot& s,uint64_t n){
    std::string out;if(scalar(s,n,out))return out;
    const auto tag=s.nodes[n].tag;require(tag==WOTR_TUPLE||tag==WOTR_FROZENSET,"Unsupported typed set member");
    auto xs=tag==WOTR_TUPLE?child_edges(s,n):sorted_members(s,n);out+=tag==WOTR_TUPLE?'T':'Z';u64(out,xs.size());for(auto x:xs)out+=atom(s,x);return out;
}
struct Encoder {
    const Snapshot& s;std::string out="RGK1G";std::unordered_map<uint64_t,uint64_t> memo;
    explicit Encoder(const Snapshot& source):s(source){}
    void visit(uint64_t n){
        if(scalar(s,n,out))return;
        const auto old=memo.find(n);if(old!=memo.end()){out+='R';u64(out,old->second);return;}
        const auto id=memo.size();memo[n]=id;const auto tag=s.nodes[n].tag;
        auto xs=child_edges(s,n);
        if(tag==WOTR_OBJECT){
            require(!xs.empty()&&xs.size()%2==1,"Malformed identity object");out+='O';u64(out,id);blob(out,text(s,xs[0]));u64(out,(xs.size()-1)/2);
            for(size_t i=1;i<xs.size();i+=2){blob(out,text(s,xs[i]));visit(xs[i+1]);}return;
        }
        require(tag==WOTR_LIST||tag==WOTR_TUPLE||tag==WOTR_DICT||tag==WOTR_SET||tag==WOTR_FROZENSET,"Unsupported identity graph type");
        out+=tag==WOTR_LIST?'L':tag==WOTR_TUPLE?'T':tag==WOTR_DICT?'D':tag==WOTR_SET?'S':'Z';u64(out,id);u64(out,tag==WOTR_DICT?xs.size()/2:xs.size());
        if(tag==WOTR_SET||tag==WOTR_FROZENSET)xs=sorted_members(s,n);
        for(auto x:xs)visit(x);
    }
};
std::string world_bytes(const Snapshot& source){Encoder encoder(source);encoder.visit(source.root);return std::move(encoder.out);}
std::string worlds_bytes(const std::vector<std::string>& worlds,const std::vector<double>& weights){
    require(worlds.size()==weights.size(),"One weight is required per identity world");std::string out="RGK1W";u64(out,worlds.size());
    for(size_t i=0;i<worlds.size();++i){real(out,weights[i]);blob(out,worlds[i]);}return out;
}
std::string actions_bytes(const std::vector<std::string>& keys){std::string out="RGK1A";u64(out,keys.size());for(const auto& key:keys)blob(out,key);return out;}
uint64_t logical_charge(size_t world_bytes,size_t action_bytes,uint64_t rows){
    const auto max=std::numeric_limits<uint64_t>::max();require(rows<=(max-4096)/512,"Identity charge exceeds uint64");uint64_t result=4096+512*rows;
    require(world_bytes<=max-result,"Identity charge exceeds uint64");result+=world_bytes;
    require(action_bytes<=max-result,"Identity charge exceeds uint64");return result+action_bytes;
}
}
