// Trusted native projection -> data-only protocol-4 stream. No GLOBAL/REDUCE,
// object construction, gameplay callbacks, or alternative hash implementation.
struct ckw_Writer {
    const Snapshot& s;std::vector<uint8_t> bytes;std::unordered_map<uint64_t,uint32_t> memo;
    explicit ckw_Writer(const Snapshot& input):s(input){bytes={0x80,4};}
    void put(uint8_t x){bytes.push_back(x);}
    void u32(uint32_t x){for(int i=0;i<4;++i)put(uint8_t(x>>(i*8)));}
    void raw(std::string_view x){bytes.insert(bytes.end(),x.begin(),x.end());}
    void string(std::string_view x){require(x.size()<=UINT32_MAX,"Chance wire string exceeds protocol bound");put('X');u32(uint32_t(x.size()));raw(x);}
    void get(uint32_t id){put('j');u32(id);}
    void remember(uint64_t id){require(memo.size()<UINT32_MAX,"Chance wire memo exceeds protocol bound");auto next=uint32_t(memo.size());memo[id]=next;put('r');u32(next);}
    void value(uint64_t id,int depth=0){
        require(depth<=400,"Chance wire immutable cycle exceeds depth bound");
        auto old=memo.find(id);if(old!=memo.end()){get(old->second);return;}
        const auto& v=s.nodes[id];require(!v.flags,"Chance wire cannot encode external list bindings");
        switch(v.tag){
        case WOTR_NONE:put('N');break;
        case WOTR_BOOL:put(v.integer?0x88:0x89);break;
        case WOTR_INT:
            // LONG1 accepts a signed little-endian two's-complement integer.
            put(0x8a);put(8);for(int i=0;i<8;++i)put(uint8_t(uint64_t(v.integer)>>(i*8)));break;
        case WOTR_FLOAT:{put('G');uint64_t bits;std::memcpy(&bits,&v.real,8);for(int i=7;i>=0;--i)put(uint8_t(bits>>(i*8)));break;}
        case WOTR_STRING:string(text(s,id));break;
        case WOTR_BYTES:require(v.count<=UINT32_MAX,"Chance wire bytes exceed protocol bound");put('B');u32(uint32_t(v.count));bytes.insert(bytes.end(),s.bytes.begin()+v.offset,s.bytes.begin()+v.offset+v.count);break;
        case WOTR_BIGINT:{
            std::string_view number(reinterpret_cast<const char*>(s.bytes.data()+v.offset),size_t(v.count));require(!number.empty(),"Chance wire bigint is empty");size_t start=(number.front()=='-'||number.front()=='+')?1:0;require(start<number.size(),"Chance wire bigint has no digits");
            for(size_t i=start;i<number.size();++i)require(number[i]>='0'&&number[i]<='9',"Chance wire bigint is not decimal");put('L');raw(number);put('L');put('\n');break;
        }
        case WOTR_ENUM:{
            auto entries=child_edges(s,id);require(entries.size()==2,"Chance wire enum requires type/member");
            put('(');string("enum");string(text(s,entries[0]));string(text(s,entries[1]));put('t');put('Q');break;
        }
        case WOTR_LIST:case WOTR_DICT:case WOTR_SET:{
            put(v.tag==WOTR_LIST?']':v.tag==WOTR_DICT?'}':0x8f);remember(id);
            auto entries=child_edges(s,id);if(!entries.empty()){put('(');for(auto child:entries)value(child,depth+1);put(v.tag==WOTR_LIST?'e':v.tag==WOTR_DICT?'u':0x90);}return;
        }
        case WOTR_TUPLE:case WOTR_FROZENSET:{
            put('(');for(auto child:child_edges(s,id))value(child,depth+1);
            // Mutable descendants can complete this tuple recursively, exactly
            // as in Python's pickle/deepcopy tuple/list cycle handling.
            auto completed=memo.find(id);
            if(completed!=memo.end()){put('1');get(completed->second);return;}
            put(v.tag==WOTR_TUPLE?'t':0x91);break;
        }
        default:throw std::invalid_argument("Chance wire cannot encode object metadata");
        }
        remember(id);
    }
    std::vector<uint8_t> finish(){value(s.root);put('.');return std::move(bytes);}
};
std::unique_ptr<Snapshot> native_chance_signature_wire(const Snapshot& s,int root){
    require(root>=-1&&root<=1,"Search root side must be None/FP/SH");
    sm_Builder b(s);auto ordered=b.state_key(true);b.out.root=b.chance_key(root,ordered);
    auto data=ckw_Writer(b.out).finish();auto result=std::make_unique<Snapshot>();
    WotrValue value{};value.tag=WOTR_BYTES;value.count=data.size();result->nodes.push_back(value);result->bytes=std::move(data);result->root=0;return result;
}
// Component-only entry: verifies the data codec independently of projection.
std::unique_ptr<Snapshot> native_metadata_wire(const Snapshot& s){
    auto data=ckw_Writer(s).finish();auto result=std::make_unique<Snapshot>();WotrValue value{};value.tag=WOTR_BYTES;value.count=data.size();result->nodes.push_back(value);result->bytes=std::move(data);result->root=0;return result;
}
