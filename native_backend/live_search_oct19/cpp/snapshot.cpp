#include "wotr_snapshot.h"
#include "wotr_random.h"
#include "unicode_ascii_table.h"
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <set>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <tuple>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace {
static_assert(sizeof(WotrValue)==40, "WotrValue ABI size");
static_assert(offsetof(WotrValue, integer)==8 && offsetof(WotrValue, real)==16 &&
              offsetof(WotrValue, offset)==24 && offsetof(WotrValue, count)==32,
              "WotrValue ABI offsets");
thread_local std::array<char,512> error_text{};

void error(const char* message) noexcept {
    if(!message) message="unknown snapshot error";
    const size_t length=std::min(std::strlen(message),error_text.size()-1);
    std::memcpy(error_text.data(),message,length);error_text[length]='\0';
}
template<class Result,class Function>
Result checked(Function function,Result failure) noexcept {
    error_text[0]='\0';
    try{return function();}
    catch(const std::exception& e){error(e.what());}
    catch(...){error("unknown snapshot exception");}
    return failure;
}
void require(bool condition,const char* message){if(!condition)throw std::invalid_argument(message);}
template<class T> void check_array(const T* pointer,size_t count){
    require(count==0 || pointer!=nullptr,"nonempty snapshot buffer has null pointer");
    require(count<=std::vector<T>().max_size(),"snapshot count exceeds vector capacity");
    require(count<=static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max())/sizeof(T),
            "snapshot count exceeds addressable byte range");
}
template<class T> std::vector<T> copy_array(const T* pointer,size_t count){
    return count ? std::vector<T>(pointer,pointer+count) : std::vector<T>();
}
bool positive_zero(double number) noexcept {
    uint64_t bits;std::memcpy(&bits,&number,sizeof(bits));return bits==0;
}
void range(uint64_t offset,uint64_t count,size_t size){
    require(offset<=size && count<=size-static_cast<size_t>(offset),"snapshot range out of bounds");
}
void decimal(const uint8_t* bytes,size_t count){
    require(count!=0,"empty bigint decimal");
    size_t first=bytes[0]=='-' ? 1 : 0;
    require(first<count,"bigint minus without digits");
    require(bytes[first]>='0' && bytes[first]<='9',"invalid bigint decimal digit");
    require(bytes[first]!='0' || (first==0 && count==1),"noncanonical bigint zero or leading zero");
    for(size_t i=first;i<count;++i)
        require(bytes[i]>='0' && bytes[i]<='9',"invalid bigint decimal digit");
}

struct ContextPrepared {
    using Index=std::unordered_map<std::string_view,uint64_t>;
    std::vector<uint8_t> bytes;
    std::vector<std::unique_ptr<Index>> dictionaries;
};
struct Snapshot {
    std::shared_ptr<const ContextPrepared> context_prepared;
    std::vector<WotrValue> nodes;
    std::vector<uint64_t> edges;
    std::vector<uint8_t> bytes;
    uint64_t root;
    size_t string_index_scanned=0;
    std::vector<std::pair<size_t,uint64_t>> string_index;
    // Derived data only: excluded from graph export/checkpoints and never
    // inherited by copies. Raw callers must serialize mutation/destruction
    // against readers; the Python owner already does so with its state lock.
    mutable std::mutex action_clone_mutex;
    mutable std::shared_ptr<const Snapshot> action_clone_template;
    mutable uint64_t action_clone_hits=0,action_clone_misses=0;
    Snapshot()=default;
    Snapshot(const Snapshot& s):context_prepared(s.context_prepared),nodes(s.nodes),
        edges(s.edges),bytes(s.bytes),root(s.root){}
};
void invalidate_action_clone(Snapshot& s){
    std::lock_guard<std::mutex> lock(s.action_clone_mutex);
    s.action_clone_template.reset();
}
const Snapshot& snapshot(const void* handle){
    require(handle!=nullptr,"snapshot handle is null");return *static_cast<const Snapshot*>(handle);
}
void validate_external_bindings(const Snapshot&);
std::unique_ptr<Snapshot> create(const WotrValue* nodes,size_t node_count,
                                 const uint64_t* edges,size_t edge_count,
                                 const uint8_t* bytes,size_t byte_count,uint64_t root){
    check_array(nodes,node_count);check_array(edges,edge_count);check_array(bytes,byte_count);
    require(node_count!=0,"snapshot must contain a node");
    require(root<node_count,"snapshot root out of bounds");
    for(size_t i=0;i<edge_count;++i)require(edges[i]<node_count,"snapshot edge reference out of bounds");
    for(size_t i=0;i<node_count;++i){
        const auto& n=nodes[i];
        require(n.tag<=WOTR_BIGINT,"unknown snapshot value tag");
        require(n.flags==0 || (n.flags==1 && n.tag==WOTR_LIST),"unsupported snapshot flags");
        if(n.tag!=WOTR_FLOAT)require(positive_zero(n.real),"unused real field must be positive zero");
        if(n.flags==1){
            require(n.integer>=0 && static_cast<uint64_t>(n.integer)<node_count,"external binding name reference out of bounds");
            require(nodes[n.integer].tag==WOTR_STRING,"external binding name must be string");
        }else if(n.tag!=WOTR_INT && n.tag!=WOTR_BOOL)require(n.integer==0,"unused integer field must be zero");
        if(n.tag<=WOTR_FLOAT){
            require(n.offset==0 && n.count==0,"scalar range fields must be zero");
            if(n.tag==WOTR_BOOL)require(n.integer==0 || n.integer==1,"invalid snapshot boolean");
            continue;
        }
        if(n.tag==WOTR_STRING || n.tag==WOTR_BYTES || n.tag==WOTR_BIGINT){
            range(n.offset,n.count,byte_count);
            if(n.tag==WOTR_BIGINT){
                require(n.count!=0,"empty bigint decimal");
                decimal(bytes+static_cast<size_t>(n.offset),static_cast<size_t>(n.count));
            }
            continue;
        }
        range(n.offset,n.count,edge_count);
        if(n.tag==WOTR_DICT)require(n.count%2==0,"dictionary requires key/value edge pairs");
        if(n.tag==WOTR_OBJECT){
            require(n.count>=1 && n.count%2==1,"object requires type and field/value edge pairs");
            require(nodes[edges[n.offset]].tag==WOTR_STRING,"object type must reference a string");
            for(uint64_t j=1;j<n.count;j+=2)
                require(nodes[edges[n.offset+j]].tag==WOTR_STRING,"object field name must reference a string");
        }
        if(n.tag==WOTR_ENUM){
            require(n.count==2,"enum requires exactly two edges");
            require(nodes[edges[n.offset]].tag==WOTR_STRING && nodes[edges[n.offset+1]].tag==WOTR_STRING,
                    "enum type and member must reference strings");
        }
    }
    auto result=std::make_unique<Snapshot>();
    result->nodes=copy_array(nodes,node_count);result->edges=copy_array(edges,edge_count);
    result->bytes=copy_array(bytes,byte_count);result->root=root;
    validate_external_bindings(*result);
    return result;
}
// These accessors deliberately operate on graph references, never reconstructed
// Python objects. Containers keep their node identity when their edges change.
constexpr uint64_t missing = std::numeric_limits<uint64_t>::max();
const WotrValue& typed(const Snapshot& s,uint64_t index,uint32_t tag,const char* message){
    require(index<s.nodes.size() && s.nodes[index].tag==tag,message);return s.nodes[index];
}
std::string_view text(const Snapshot& s,uint64_t index){
    const auto& n=typed(s,index,WOTR_STRING,"expected string graph node");
    if(!n.count)return {};
    return std::string_view(reinterpret_cast<const char*>(s.bytes.data()+n.offset),static_cast<size_t>(n.count));
}
bool text_is(const Snapshot& s,uint64_t index,std::string_view value){
    if(index>=s.nodes.size() || s.nodes[index].tag!=WOTR_STRING)return false;
    const auto& n=s.nodes[index];
    return n.count==value.size() && (!n.count || std::memcmp(s.bytes.data()+n.offset,value.data(),value.size())==0);
}
bool enum_is(const Snapshot& s,uint64_t index,std::string_view type,std::string_view member){
    if(index>=s.nodes.size() || s.nodes[index].tag!=WOTR_ENUM)return false;
    const auto& n=s.nodes[index];
    return text_is(s,s.edges[n.offset],type) && text_is(s,s.edges[n.offset+1],member);
}
void validate_external_bindings(const Snapshot& s){
    const std::array<std::string_view,6> fp={"CHARACTER","CHARACTER","MUSTER","EVENT","MUSTER_ARMY","WILL_OF_WEST"};
    const std::array<std::string_view,6> sh={"CHARACTER","ARMY","MUSTER","EVENT","MUSTER_ARMY","EYE"};
    std::array<bool,2> seen{};
    for(const auto& node:s.nodes){
        if(node.flags!=1)continue;
        const bool free=text_is(s,static_cast<uint64_t>(node.integer),"game_state.FP_FACES");
        require(free || text_is(s,static_cast<uint64_t>(node.integer),"game_state.SH_FACES"),"unregistered external list binding");
        const size_t binding=free ? 0 : 1;
        require(!seen[binding],"duplicate external list binding");seen[binding]=true;
        require(node.count==6,"external die faces must have six entries");
        const auto& expected=free ? fp : sh;
        for(size_t i=0;i<expected.size();++i)
            require(enum_is(s,s.edges[node.offset+i],"game_state.ActionDieFace",expected[i]),"external die face binding differs from frozen rules");
    }
}
uint64_t string_value_edge(const Snapshot& s,uint64_t dict,std::string_view key){
    const auto& d=typed(s,dict,WOTR_DICT,"expected attribute dictionary");
    uint64_t result=missing;
    for(uint64_t i=0;i<d.count;i+=2)
        if(text_is(s,s.edges[d.offset+i],key))result=d.offset+i+1;
    return result;
}
uint64_t enum_value_edge(const Snapshot& s,uint64_t dict,std::string_view type,std::string_view member){
    const auto& d=typed(s,dict,WOTR_DICT,"expected enum-keyed dictionary");
    uint64_t result=missing;
    for(uint64_t i=0;i<d.count;i+=2)
        if(enum_is(s,s.edges[d.offset+i],type,member))result=d.offset+i+1;
    return result;
}
uint64_t attributes(const Snapshot& s,uint64_t object,std::string_view class_name){
    const auto& o=typed(s,object,WOTR_OBJECT,"expected engine object graph node");
    require(text_is(s,s.edges[o.offset],class_name),"unexpected engine object class");
    uint64_t result=missing;
    for(uint64_t i=1;i<o.count;i+=2){
        if(text_is(s,s.edges[o.offset+i],"__dict__")){
            require(result==missing,"duplicate engine object __dict__");
            result=s.edges[o.offset+i+1];
        }
    }
    require(result!=missing,"engine object has no __dict__");
    typed(s,result,WOTR_DICT,"engine __dict__ must be a dictionary");return result;
}
uint64_t field(const Snapshot& s,uint64_t attrs,const char* name,bool required=true){
    const auto edge=string_value_edge(s,attrs,name);
    require(!required || edge!=missing,"required engine attribute is missing");
    return edge==missing ? missing : s.edges[edge];
}
std::string_view validate_nation_name(std::string_view result){
    const char* allowed[]={"ELVES","DWARVES","GONDOR","THE_NORTH","ROHAN","ISENGARD","SAURON","SOUTHRON_EASTERLINGS"};
    bool found=false;for(const auto* item:allowed)if(result==item)found=true;
    require(found,"unknown Nation enum member name");return result;
}
std::string_view nation_argument(const char* name){
    require(name!=nullptr,"nation name is null");
    return validate_nation_name(name);
}
std::string_view method_argument(const char* name){
    require(name!=nullptr,"political method is null");const std::string_view result(name);
    require(result=="muster_die" || result=="attack" || result=="companion_ability" || result=="event_card",
            "unsupported political advance method");return result;
}
bool fp_nation(std::string_view nation){
    return nation=="ELVES" || nation=="DWARVES" || nation=="GONDOR" || nation=="THE_NORTH" || nation=="ROHAN";
}
struct PoliticalView{
    uint64_t state_attrs,political_attrs,boxes,box_edge;
    int box;
};
PoliticalView political_view(const Snapshot& s,std::string_view nation,bool require_box){
    PoliticalView result{};
    result.state_attrs=attributes(s,s.root,"game_state.GameState");
    result.political_attrs=attributes(s,field(s,result.state_attrs,"political"),"game_state.PoliticalTrack");
    result.boxes=field(s,result.political_attrs,"boxes");
    result.box_edge=enum_value_edge(s,result.boxes,"game_state.Nation",nation);
    require(!require_box || result.box_edge!=missing,"political boxes dictionary is missing nation");
    int64_t box=result.box_edge==missing ? 0 : typed(s,s.edges[result.box_edge],WOTR_INT,"political box must be INT").integer;
    require(box>=0 && box<=3,"political rule requires canonical box zero through three");
    result.box=static_cast<int>(box);return result;
}
bool active_nation(const Snapshot& s,const PoliticalView& p,std::string_view nation,bool required){
    const auto active=field(s,p.political_attrs,"active");
    const auto edge=enum_value_edge(s,active,"game_state.Nation",nation);
    require(!required || edge!=missing,"political active dictionary is missing nation");
    if(edge==missing)return false;
    return typed(s,s.edges[edge],WOTR_BOOL,"political active value must be BOOL").integer!=0;
}
std::array<uint64_t,2> card_lists(const Snapshot& s,const PoliticalView& p){
    const auto attrs=attributes(s,field(s,p.state_attrs,"persistent_cards"),"game_state.PersistentCardState");
    std::array<uint64_t,2> result={field(s,attrs,"fp_cards"),field(s,attrs,"sh_cards")};
    for(auto list:result)typed(s,list,WOTR_LIST,"persistent cards must be lists");
    return result;
}
uint64_t threats_card(const Snapshot& s,const PoliticalView& p){
    for(auto list:card_lists(s,p)){
        const auto& l=s.nodes[list];
        for(uint64_t i=0;i<l.count;++i){
            const auto card=s.edges[l.offset+i];
            const auto attrs=attributes(s,card,"game_state.PersistentCard");
            const auto kind=field(s,attrs,"card_type",false);
            if(kind!=missing && enum_is(s,kind,"game_state.PersistentCardType","THREATS_AND_PROMISES"))
                return card;
        }
    }
    return missing;
}
bool can_advance(const Snapshot& s,const PoliticalView& p,std::string_view nation,std::string_view method){
    if(p.box>=3)return false;
    if(p.box==2 && !active_nation(s,p,nation,false))return false;
    if(threats_card(s,p)!=missing){
        // This order preserves the current Python hook: it accesses active[nat]
        // even when the method is not muster, and applies to either faction.
        if(!active_nation(s,p,nation,true) && method=="muster_die")return false;
    }
    return true;
}
std::vector<uint64_t> child_edges(const Snapshot& s,uint64_t node){
    const auto& n=s.nodes[node];
    return std::vector<uint64_t>(s.edges.begin()+static_cast<ptrdiff_t>(n.offset),
                                 s.edges.begin()+static_cast<ptrdiff_t>(n.offset+n.count));
}
void set_edges(Snapshot& s,uint64_t node,const std::vector<uint64_t>& edges){
    if(s.nodes[node].tag==WOTR_STRING){s.string_index.clear();s.string_index_scanned=0;}
    invalidate_action_clone(s);
    s.context_prepared.reset();
    const auto offset=s.edges.size();s.edges.insert(s.edges.end(),edges.begin(),edges.end());
    s.nodes[node].offset=offset;s.nodes[node].count=edges.size();
}
void assign_box(Snapshot& s,const PoliticalView& p,int next){
    WotrValue value{};value.tag=WOTR_INT;value.integer=next;
    const auto fresh=s.nodes.size();s.nodes.push_back(value);
    auto edges=child_edges(s,p.boxes);
    edges[static_cast<size_t>(p.box_edge-s.nodes[p.boxes].offset)]=fresh;
    set_edges(s,p.boxes,edges);
}
std::string_view card_id(const Snapshot& s,uint64_t card){
    return text(s,field(s,attributes(s,card,"game_state.PersistentCard"),"card_id"));
}
uint64_t remove_card(Snapshot& s,const PoliticalView& p,std::string_view id){
    for(auto list:card_lists(s,p)){
        const auto& l=s.nodes[list];
        for(uint64_t i=0;i<l.count;++i){
            const auto card=s.edges[l.offset+i];
            if(card_id(s,card)==id){
                auto edges=child_edges(s,list);edges.erase(edges.begin()+static_cast<ptrdiff_t>(i));
                set_edges(s,list,edges);return card;
            }
        }
    }
    return missing;
}
void discard_card(Snapshot& s,const PoliticalView& p,uint64_t card){
    const auto attrs=attributes(s,card,"game_state.PersistentCard");
    const auto cid_node=field(s,attrs,"card_id");
    const auto cid=text(s,cid_node);
    const auto side=field(s,attrs,"side",false);
    const std::string prefix=side!=missing && enum_is(s,side,"game_state.Side","FREE_PEOPLES") ? "fp_" : "sh_";
    const std::string deck=prefix+(cid.find("Character")!=std::string::npos ? "character" : "strategy");
    const auto discards=field(s,p.state_attrs,"discard_face_up",false);
    if(discards==missing || s.nodes[discards].tag!=WOTR_DICT)return;
    const auto edge=string_value_edge(s,discards,deck);
    if(edge==missing)return;
    const auto list=s.edges[edge];typed(s,list,WOTR_LIST,"discard destination must be a list");
    auto children=child_edges(s,list);children.push_back(cid_node);set_edges(s,list,children);
}
int advance(Snapshot& original,std::string_view nation,std::string_view method){
    const auto view=political_view(original,nation,true);
    const bool allowed=can_advance(original,view,nation,method);
    if(!allowed && view.box<3)throw std::invalid_argument("political advance is not legal");
    if(view.box>=3)return view.box;
    // Stage the complete transaction: no graph reachable through the handle
    // changes if validation, lookup, allocation or discard handling fails.
    Snapshot changed=original;
    assign_box(changed,view,view.box+1);
    if(fp_nation(nation) && (method=="attack" || method=="companion_ability")){
        const auto card=threats_card(changed,view);
        if(card!=missing){
            const auto removed=remove_card(changed,view,card_id(changed,card));
            discard_card(changed,view,removed==missing ? card : removed);
        }
    }
    auto validated=create(changed.nodes.data(),changed.nodes.size(),changed.edges.data(),changed.edges.size(),
                          changed.bytes.data(),changed.bytes.size(),changed.root);
    original.nodes.swap(validated->nodes);original.edges.swap(validated->edges);original.bytes.swap(validated->bytes);
    original.string_index.clear();original.string_index_scanned=0;
    original.context_prepared.swap(validated->context_prepared);
    std::swap(original.root,validated->root);
    invalidate_action_clone(original);
    return view.box+1;
}
bool truth(const Snapshot& s,uint64_t node){
    if(node==missing)return false;
    const auto& n=s.nodes[node];
    switch(n.tag){
        case WOTR_NONE:return false;
        case WOTR_BOOL:case WOTR_INT:return n.integer!=0;
        case WOTR_FLOAT:return n.real!=0.0;
        case WOTR_STRING:case WOTR_BYTES:case WOTR_LIST:case WOTR_TUPLE:
        case WOTR_DICT:case WOTR_SET:case WOTR_FROZENSET:return n.count!=0;
        case WOTR_BIGINT:return !(n.count==1 && s.bytes[n.offset]=='0');
        // Registered engine objects have no __bool__/__len__ override.
        case WOTR_OBJECT:case WOTR_ENUM:return true;
        default:throw std::invalid_argument("unsupported graph truthiness");
    }
}
std::string sanitize_name(std::string_view value){
    std::string result;bool separated=false;
    auto append=[&](char c){
        if(separated && !result.empty())result.push_back(' ');
        result.push_back(c);separated=false;
    };
    auto ascii=[&](unsigned char c){
        if(c=='&'){
            separated=true;append('a');append('n');append('d');separated=true;
        }else if(c>='A' && c<='Z'){
            append(static_cast<char>(c-'A'+'a'));
        }else if((c>='a' && c<='z') || (c>='0' && c<='9')){
            append(static_cast<char>(c));
        }else separated=true;
    };
    size_t i=0;
    while(i<value.size()){
        const auto first=static_cast<unsigned char>(value[i++]);
        uint32_t point=first;size_t trailing=0;
        if(first>=0x80){
            if(first>=0xc2 && first<=0xdf){point=first&0x1f;trailing=1;}
            else if(first>=0xe0 && first<=0xef){point=first&0x0f;trailing=2;}
            else if(first>=0xf0 && first<=0xf4){point=first&0x07;trailing=3;}
            else throw std::invalid_argument("invalid UTF-8 leading byte");
            require(trailing<=value.size()-i,"truncated UTF-8 name");
            for(size_t j=0;j<trailing;++j){
                const auto next=static_cast<unsigned char>(value[i++]);
                require((next&0xc0)==0x80,"invalid UTF-8 continuation byte");
                point=(point<<6)|(next&0x3f);
            }
            require((trailing==1 && point>=0x80) || (trailing==2 && point>=0x800) ||
                    (trailing==3 && point>=0x10000),"overlong UTF-8 name");
            require(point<=0x10ffff,"UTF-8 codepoint outside Unicode range");
            // Surrogates are accepted to match Python's UTF-8 surrogatepass;
            // NFKD leaves them unchanged and ASCII-ignore drops them.
        }
        if(point<128){ascii(static_cast<unsigned char>(point));continue;}
        const auto* begin=kWotrAsciiMappings;
        const auto* end=begin+kWotrAsciiMappingCount;
        const auto* entry=std::lower_bound(begin,end,point,
            [](const WotrAsciiMapping& mapping,uint32_t wanted){return mapping.codepoint<wanted;});
        if(entry!=end && entry->codepoint==point){
            for(size_t j=0;j<entry->length;++j)ascii(kWotrAsciiBytes[entry->offset+j]);
        }
    }
    return result;
}
std::string_view object_type(const Snapshot& s,uint64_t node){
    const auto& o=typed(s,node,WOTR_OBJECT,"expected prefix engine object");
    const auto name=text(s,s.edges[o.offset]);
    require(name.find('\0')==std::string_view::npos,"embedded NUL in engine object type");
    return name;
}
uint64_t generic_attributes(const Snapshot& s,uint64_t node){
    const auto type=object_type(s,node);return attributes(s,node,type);
}
std::string_view enum_member(const Snapshot& s,uint64_t node,const char* expected){
    const auto& e=typed(s,node,WOTR_ENUM,"expected prefix enum value");
    require(text_is(s,s.edges[e.offset],expected),"unexpected prefix enum class");
    return text(s,s.edges[e.offset+1]);
}
bool contains_string(const Snapshot& s,uint64_t container,std::string_view key){
    if(container==missing)return false;
    const auto& n=s.nodes[container];
    if(n.tag==WOTR_STRING)return text(s,container).find(key)!=std::string::npos;
    require(n.tag==WOTR_LIST || n.tag==WOTR_TUPLE || n.tag==WOTR_SET || n.tag==WOTR_FROZENSET || n.tag==WOTR_DICT,
            "unsupported exact-membership container");
    const uint64_t step=n.tag==WOTR_DICT ? 2 : 1;
    for(uint64_t i=0;i<n.count;i+=step)if(text_is(s,s.edges[n.offset+i],key))return true;
    return false;
}
std::string fellowship_key(const Snapshot& s,uint64_t state_attrs){
    const auto fship=field(s,state_attrs,"fellowship",false);
    if(fship==missing || s.nodes[fship].tag==WOTR_NONE)return {};
    const auto attrs=attributes(s,fship,"game_state.Fellowship");
    const auto region=field(s,attrs,"region",false);
    if(region==missing)return {}; // Python helper raises; encode_state catches it.
    if(s.nodes[region].tag==WOTR_STRING)return sanitize_name(text(s,region));
    if(s.nodes[region].tag==WOTR_NONE)return "none"; // str(None) in the original helper.
    require(s.nodes[region].tag==WOTR_OBJECT,"unsupported fellowship region representation");
    const auto region_attrs=generic_attributes(s,region);
    for(const char* name:{"key","name","orig_name"}){
        const auto value=field(s,region_attrs,name,false);
        if(value!=missing){
            // A present non-string attribute causes sanitize to raise, and the
            // Python prefix catches it rather than trying the next attribute.
            if(s.nodes[value].tag!=WOTR_STRING)return {};
            return sanitize_name(text(s,value));
        }
    }
    throw std::invalid_argument("fellowship object has no supported region attribute");
}
void encode_prefix(const Snapshot& s,int32_t side,const char* const* region_keys,size_t nr,
                   const char* const* nation_names,size_t nn,float* output,size_t count){
    require(side==0 || side==1,"invalid prefix observation side");
    check_array(region_keys,nr);check_array(nation_names,nn);check_array(output,count);
    const size_t max=std::numeric_limits<size_t>::max();
    require(nn<=(max-14)/2 && nn<=max/4,"prefix feature width overflows");
    const auto width=14+2*nn;
    require(nr<=(max-4*nn)/width && count==nr*width+4*nn,"prefix output count must be exact");
    require(nr<=static_cast<size_t>(std::numeric_limits<int32_t>::max()) &&
            nn<=static_cast<size_t>(std::numeric_limits<int32_t>::max()),"prefix metadata index range too large");
    std::vector<std::string> keys;
    std::vector<std::string_view> names;
    keys.reserve(nr);names.reserve(nn);
    for(size_t i=0;i<nr;++i){
        require(region_keys[i]!=nullptr,"null prefix region key");
        const std::string key(region_keys[i]);sanitize_name(key);keys.push_back(key);
    }
    std::unordered_map<std::string_view,int32_t> nation_index;
    for(size_t i=0;i<nn;++i){
        auto name=nation_argument(nation_names[i]);
        require(nation_index.emplace(name,static_cast<int32_t>(i)).second,"duplicate prefix nation name");
        names.push_back(std::move(name));
    }
    const auto state_attrs=attributes(s,s.root,"game_state.GameState");
    const auto army_dict=field(s,state_attrs,"armies");
    const auto& army_entries=typed(s,army_dict,WOTR_DICT,"armies must be dictionary");
    std::unordered_map<std::string,uint64_t> armies;
    for(uint64_t i=0;i<army_entries.count;i+=2){
        auto key=sanitize_name(text(s,s.edges[army_entries.offset+i]));
        armies[std::move(key)]=s.edges[army_entries.offset+i+1];
    }
    const auto controlled=field(s,state_attrs,"enemy_controlled");
    const auto fp_edge=enum_value_edge(s,controlled,"game_state.Side","FREE_PEOPLES");
    const auto sh_edge=enum_value_edge(s,controlled,"game_state.Side","SHADOW");
    const auto fp=fp_edge==missing ? missing : s.edges[fp_edge];
    const auto sh=sh_edge==missing ? missing : s.edges[sh_edge];
    const auto siege=field(s,state_attrs,"besieged_strongholds",false);
    const auto fellowship=fellowship_key(s,state_attrs);
    const auto political=attributes(s,field(s,state_attrs,"political"),"game_state.PoliticalTrack");
    const auto boxes=field(s,political,"boxes"),active=field(s,political,"active");
    std::vector<WotrNation> nations; nations.reserve(nn);
    for(const auto& name:names){
        const auto box_edge=enum_value_edge(s,boxes,"game_state.Nation",name);
        const auto active_edge=enum_value_edge(s,active,"game_state.Nation",name);
        const auto box=box_edge==missing ? 0 : typed(s,s.edges[box_edge],WOTR_INT,"prefix political box must be INT").integer;
        require(box>=0 && box<=3,"prefix political box must be zero through three");
        nations.push_back({fp_nation(name) ? 0 : 1,static_cast<int32_t>(box),
                           active_edge!=missing && truth(s,s.edges[active_edge]) ? 1 : 0});
    }
    std::vector<WotrRegion> regions;regions.reserve(nr);
    std::vector<WotrUnit> units;std::vector<WotrLeader> leaders;
    for(size_t i=0;i<nr;++i){
        const auto& key=keys[i];const auto found=armies.find(key);
        const auto army=found==armies.end() ? missing : found->second;
        const bool occupied=truth(s,army);
        regions.push_back({occupied ? 1 : 0,contains_string(s,siege,key) ? 1 : 0,
            contains_string(s,fp,key) ? 1 : 0,contains_string(s,sh,key) ? 1 : 0,key==fellowship ? 1 : 0});
        if(!occupied)continue;
        const auto attrs=attributes(s,army,"game_state.Army");
        const auto& army_units=typed(s,field(s,attrs,"units"),WOTR_LIST,"army units must be list");
        for(uint64_t j=0;j<army_units.count;++j){
            const auto unit_attrs=attributes(s,s.edges[army_units.offset+j],"game_state.Unit");
            const auto name=enum_member(s,field(s,unit_attrs,"nation"),"game_state.Nation");
            const auto nation=nation_index.find(name);require(nation!=nation_index.end(),"unit nation absent from prefix metadata");
            const auto kind=enum_member(s,field(s,unit_attrs,"kind"),"game_state.UnitKind");
            require(kind=="REGULAR" || kind=="ELITE","unsupported prefix unit kind");
            units.push_back({static_cast<int32_t>(i),nation->second,kind=="REGULAR" ? 0 : 1});
        }
        const auto& army_leaders=typed(s,field(s,attrs,"leaders"),WOTR_LIST,"army leaders must be list");
        for(uint64_t j=0;j<army_leaders.count;++j){
            const auto leader=s.edges[army_leaders.offset+j];
            if(s.nodes[leader].tag!=WOTR_OBJECT)continue; // no recognized name/class/nation attributes
            const auto type=object_type(s,leader);
            const auto leader_attrs=generic_attributes(s,leader);
            const auto name_node=field(s,leader_attrs,"name",false);
            int32_t leader_side=0,named=0,nazgul=0;
            if(name_node!=missing && (text_is(s,name_node,"Witch King") || text_is(s,name_node,"Saruman") || text_is(s,name_node,"Mouth of Sauron"))){
                leader_side=1;named=1;
            }else if(type=="game_state.Companion"){named=1;}
            else if(type=="game_state.Nazgul"){leader_side=1;nazgul=1;}
            else if(type=="game_logic.Minion"){leader_side=1;named=1;}
            else{
                const auto nation_node=field(s,leader_attrs,"nation",false);if(nation_node==missing)continue;
                const auto nation=enum_member(s,nation_node,"game_state.Nation");
                validate_nation_name(nation);leader_side=fp_nation(nation) ? 0 : 1;
            }
            leaders.push_back({static_cast<int32_t>(i),leader_side,named,nazgul});
        }
    }
    using Owner=std::unique_ptr<void,void(*)(void*)>;
    Owner projected(wotr_position_create(regions.data(),regions.size(),nations.data(),nations.size(),
                units.data(),units.size(),leaders.data(),leaders.size()),wotr_position_destroy);
    if(!projected)throw std::runtime_error(wotr_last_error());
    if(wotr_encode_regions_politics(projected.get(),side,output,count)!=0)
        throw std::runtime_error(wotr_last_error());
    const auto debt=field(s,state_attrs,"_wormtongue_attack_pending",false);
    if(debt!=missing&&s.nodes[debt].tag!=WOTR_NONE)
        for(size_t i=0;i<names.size();++i)if(names[i]=="ROHAN")
            output[nr*(14+2*nn)+4*i+1]-=.25f*static_cast<float>(typed(s,debt,WOTR_INT,"Invalid Wormtongue attack reaction").integer);
}
#include "observation_support.inc"
#include "ring_rules.inc"
#include "observation_globals.inc"
#include "observation_context.inc"
#include "observation_tails.inc"
#include "action_clone.inc"
#include "political_table_gate.inc"
#include "quiet_action_gate.inc"
#include "political_action.inc"
#include "ring_action.inc"
#include "terminal.inc"
#include "action_menu.inc"
#include "companion_political.inc"
#include "muster_action.inc"
#include "muster_menu.inc"
#include "saruman_muster.inc"
#include "movement_effects.inc"
void af_scope(const Snapshot&,const Snapshot&);
#include "army_move.inc"
#include "army_move_menu.inc"
uint64_t hf_choice(void*,size_t,bool);
#include "dice_roll.inc"
#include "turn_cards.inc"
#include "draw_action.inc"
#include "mouth_move.inc"
#include "minion_muster.inc"
void epa_after_event(Snapshot&,int,std::string_view,std::string_view,std::string_view);
void epa_maybe(Snapshot&);
#include "event_actions.inc"
#include "event_dice.inc"
#include "discard_action.inc"
#include "turn_reset.inc"
#include "eye_allocation.inc"
#include "combat_forcing_errors.inc"
#include "combat_forcing.inc"
#include "combat_reroll_stage.inc"
#include "combat_primary_stage.inc"
#include "combat_post_stage.inc"
#include "combat_pre_stage.inc"
#include "combat_round_stage.inc"
#include "hunt_forcing.inc"
#include "combat_publish.inc"
#include "combat_allocation.inc"
#include "army_attack_menu.inc"
#include "combat_pending_menu.inc"
#include "combat_deferred_post.inc"
#include "fellowship_declare.inc"
#include "fellowship_tail.inc"
#include "guide_choice.inc"
#include "ai_policy_projection.inc"
#include "actor_routing.inc"
#include "hunt_roll.inc"
#include "hunt_protection_open.inc"
#include "hunt_tile.inc"
#include "hunt_damage.inc"
#include "fellowship_reveal.inc"
#include "fellowship_move.inc"
#include "hunt_decision.inc"
#include "hunt_protection.inc"
#include "event_crebain.inc"
#include "fellowship_balrog.inc"
#include "event_hunt_draw.inc"
#include "event_lure.inc"
std::vector<std::string> edf_nations(const Snapshot&);
uint64_t edf_choice_pending(const Snapshot&,const Snapshot&);
void edf_eliminate(Snapshot&,std::string_view={});
#include "event_hunt_table.inc"
#include "event_misc.inc"
#include "fellowship_hide.inc"
#include "companion_separation.inc"
#include "companion_move.inc"
#include "nazgul_move.inc"
#include "will_special.inc"
#include "event_hunt_tiles.inc"
#include "bid_compensation.inc"
std::unique_ptr<Snapshot> native_palantir_draw_actions(const Snapshot&,int);
bool er_supported(const Snapshot&,uint64_t);
bool epr_supported(const Snapshot&,uint64_t);
std::unique_ptr<Snapshot> apply_pits_event(const Snapshot&,const Snapshot&,int,int64_t,std::string_view);
std::unique_ptr<Snapshot> apply_pits_choice(const Snapshot&,const Snapshot&,int,int64_t);
std::unique_ptr<Snapshot> native_pits_choice_actions(const Snapshot&,const Snapshot&,int);
bool esm_supported(const Snapshot&,uint64_t);
bool esm_pending(const Snapshot&);
std::unique_ptr<Snapshot> native_misty_choices(const Snapshot&,const Snapshot&,int);
bool emk_supported(const Snapshot&,uint64_t);
bool emk_pending(const Snapshot&);
std::unique_ptr<Snapshot> native_many_kings_choices(const Snapshot&,const Snapshot&,int);
bool eaw_supported(const Snapshot&,uint64_t);
bool eaw_pending(const Snapshot&);
std::unique_ptr<Snapshot> native_another_way_choices(const Snapshot&,const Snapshot&,int);
bool esw_supported(const Snapshot&,uint64_t);
bool esw_pending(const Snapshot&);
std::unique_ptr<Snapshot> native_swifter_choices(const Snapshot&,const Snapshot&,int);
bool eck_supported(const Snapshot&,uint64_t);
bool eea_supported(const Snapshot&,uint64_t);
bool eea_pending(const Snapshot&);
std::unique_ptr<Snapshot> native_ents_choices(const Snapshot&,const Snapshot&,int);
bool esc_supported(const Snapshot&,uint64_t);
bool esc_pending(const Snapshot&);
std::unique_ptr<Snapshot> native_stormcrow_choices(const Snapshot&,const Snapshot&,int);
bool eal_supported(const Snapshot&,uint64_t);
bool eal_pending(const Snapshot&);
std::unique_ptr<Snapshot> native_alone_choices(const Snapshot&,const Snapshot&,int);
bool etb_supported(const Snapshot&,uint64_t);
bool etb_pending(const Snapshot&);
std::unique_ptr<Snapshot> native_back_again_choices(const Snapshot&,const Snapshot&,int);
bool ehe_supported(const Snapshot&,uint64_t);
bool ehe_pending(const Snapshot&);
std::unique_ptr<Snapshot> native_horde_choices(const Snapshot&,const Snapshot&,int);
bool erg_supported(const Snapshot&,uint64_t);
bool erg_pending(const Snapshot&);
std::unique_ptr<Snapshot> native_rage_choices(const Snapshot&,const Snapshot&,int);
bool eco_supported(const Snapshot&,uint64_t);
bool eco_pending(const Snapshot&);
std::unique_ptr<Snapshot> native_corsairs_choices(const Snapshot&,const Snapshot&,int);
bool ehuf_supported(const Snapshot&,uint64_t);
bool ehuf_pending(const Snapshot&);
std::unique_ptr<Snapshot> native_help_unlooked_choice_actions(const Snapshot&,const Snapshot&,int);
bool egr_supported(const Snapshot&,uint64_t);
bool egr_pending(const Snapshot&);
std::unique_ptr<Snapshot> native_grond_choice_actions(const Snapshot&,const Snapshot&,int);
bool euh_supported(const Snapshot&,uint64_t);
bool euh_pending(const Snapshot&);
std::unique_ptr<Snapshot> native_uruk_choice_actions(const Snapshot&,const Snapshot&,int);
bool etd_supported(const Snapshot&,uint64_t);
bool etd_pending(const Snapshot&);
std::unique_ptr<Snapshot> native_through_day_choice_actions(const Snapshot&,int32_t);
bool epw_supported(const Snapshot&,uint64_t);
bool epw_pending(const Snapshot&);
std::unique_ptr<Snapshot> native_woses_choice_actions(const Snapshot&,int32_t);
bool egw_supported(const Snapshot&,uint64_t);
bool egw_pending(const Snapshot&);
std::unique_ptr<Snapshot> native_gwaihir_choice_actions(const Snapshot&,int32_t);
bool eff_supported(const Snapshot&,uint64_t);
bool eff_pending(const Snapshot&);
bool ecc_pending(const Snapshot&);
std::unique_ptr<Snapshot> native_faramir_choice_actions(const Snapshot&,const Snapshot&,int);
std::unique_ptr<Snapshot> native_card_casualty_actions(const Snapshot&,const Snapshot&,int);
std::unique_ptr<Snapshot> native_card_army_followup_actions(const Snapshot&,const Snapshot&,int);
bool ept_supported(const Snapshot&,uint64_t);
uint64_t ept_first(const Snapshot&);
std::unique_ptr<Snapshot> native_power_kill_for_die(const Snapshot&,const Snapshot&,int,int64_t);
std::unique_ptr<Snapshot> native_power_payment_actions(const Snapshot&,const Snapshot&,int);
std::unique_ptr<Snapshot> apply_power_event(const Snapshot&,const Snapshot&,int,int64_t,std::string_view);
std::unique_ptr<Snapshot> apply_power_kill(const Snapshot&,const Snapshot&,int,int64_t,std::string_view);
std::unique_ptr<Snapshot> apply_power_payment(const Snapshot&,const Snapshot&,int,int64_t);
std::string enh_kind(const Snapshot&,uint64_t);
bool enh_pending(const Snapshot&);
std::unique_ptr<Snapshot> native_nazgul_hunt_choice_actions(const Snapshot&,int32_t);
bool etsm_supported(const Snapshot&,uint64_t);
bool etsm_pending(const Snapshot&);
std::unique_ptr<Snapshot> native_shadow_is_moving_choice_actions(const Snapshot&,int32_t);
bool esl_supported(const Snapshot&,uint64_t);
bool eho_supported(const Snapshot&,uint64_t);
std::unique_ptr<Snapshot> apply_half_orcs_event(const Snapshot&,const Snapshot&,int,int64_t,std::string_view,bool=false);
std::unique_ptr<Snapshot> native_half_orcs_choice_actions(const Snapshot&,const Snapshot&,int);
std::unique_ptr<Snapshot> apply_half_orcs_choice(const Snapshot&,const Snapshot&,int,int64_t);
bool esl_pending(const Snapshot&);
std::unique_ptr<Snapshot> native_shadow_lengthens_choice_actions(const Snapshot&,int32_t);
int ert_kind(const Snapshot&,uint64_t);
int esr_kind(const Snapshot&,uint64_t);
bool esg_supported(const Snapshot&,uint64_t);
bool efm_supported(const Snapshot&,uint64_t);
bool efm_pending(const Snapshot&);
bool edm_supported(const Snapshot&,uint64_t);
bool edm_pending(const Snapshot&);
bool edm_cross_side(const Snapshot&);
std::unique_ptr<Snapshot> native_dead_men_choices(const Snapshot&,const Snapshot&,int);
bool erv_supported(const Snapshot&,uint64_t);
bool eeg_supported(const Snapshot&,uint64_t);
bool eeg_pending(const Snapshot&);
bool eeg_destination(const Snapshot&);
std::unique_ptr<Snapshot> native_eagles_choices(const Snapshot&,const Snapshot&,int);
bool eds_supported(const Snapshot&,uint64_t);
bool eds_pending(const Snapshot&);
std::unique_ptr<Snapshot> native_dreadful_choices(const Snapshot&,const Snapshot&,int);
bool esp_supported(const Snapshot&,uint64_t);
bool esp_pending(const Snapshot&);
std::unique_ptr<Snapshot> native_spirit_choices(const Snapshot&,const Snapshot&,int);
bool egc_supported(const Snapshot&,uint64_t);
bool egc_pending(const Snapshot&);
std::unique_ptr<Snapshot> native_grey_company_choices(const Snapshot&,const Snapshot&,int);
bool ehs_supported(const Snapshot&,uint64_t);
bool ehs_pending(const Snapshot&);
std::unique_ptr<Snapshot> native_house_stewards_choices(const Snapshot&,const Snapshot&,int);
bool ebc_supported(const Snapshot&,uint64_t);
bool ebc_pending(const Snapshot&);
std::unique_ptr<Snapshot> native_black_captain_choices(const Snapshot&,const Snapshot&,int);
bool erw_supported(const Snapshot&,uint64_t);
bool erw_pending(const Snapshot&);
std::unique_ptr<Snapshot> native_ringwraiths_choice_actions(const Snapshot&,const Snapshot&,int);
bool ecw_supported(const Snapshot&,uint64_t);
bool ecw_pending(const Snapshot&);
std::unique_ptr<Snapshot> native_cruel_weather_choices(const Snapshot&,const Snapshot&,int);
std::unique_ptr<Snapshot> apply_target_recruit_event(const Snapshot&,const Snapshot&,int,int64_t,std::string_view);
std::unique_ptr<Snapshot> native_target_recruit_choice_actions(const Snapshot&,const Snapshot&,int);
std::unique_ptr<Snapshot> apply_target_recruit_choice(const Snapshot&,const Snapshot&,int,int64_t);
std::unique_ptr<Snapshot> apply_shadow_recruit_event(const Snapshot&,const Snapshot&,int,int64_t,std::string_view);
std::unique_ptr<Snapshot> apply_shadows_gather_event(const Snapshot&,const Snapshot&,int,int64_t,std::string_view,bool=false);
std::unique_ptr<Snapshot> native_shadows_gather_choice_actions(const Snapshot&,const Snapshot&,int);
std::unique_ptr<Snapshot> apply_shadows_gather_choice(const Snapshot&,const Snapshot&,int,int64_t);
std::unique_ptr<Snapshot> native_companion_event_choice_actions(const Snapshot&,int32_t);
bool ehi_supported(const Snapshot&,uint64_t);
std::unique_ptr<Snapshot> apply_hill_trolls_event(const Snapshot&,const Snapshot&,int,int64_t,std::string_view,bool=false);
std::unique_ptr<Snapshot> native_hill_trolls_choice_actions(const Snapshot&,const Snapshot&,int);
std::unique_ptr<Snapshot> apply_hill_trolls_choice(const Snapshot&,const Snapshot&,int,int64_t);
std::unique_ptr<Snapshot> native_worn_choice_actions(const Snapshot&,int32_t);
std::unique_ptr<Snapshot> apply_recruit_event_action(const Snapshot&,const Snapshot&,int,int64_t,std::string_view);
std::unique_ptr<Snapshot> native_recruit_event_choice_actions(const Snapshot&,const Snapshot&,int);
std::unique_ptr<Snapshot> apply_recruit_event_choice(const Snapshot&,const Snapshot&,int,int64_t);
std::unique_ptr<Snapshot> native_overstack_choice_actions(const Snapshot&,int);
std::unique_ptr<Snapshot> apply_overstack_choice(const Snapshot&,const Snapshot&,int,int64_t);
std::unique_ptr<Snapshot> native_denethor_kill_for_die(const Snapshot&,int,int64_t);
std::unique_ptr<Snapshot> apply_denethor_kill(const Snapshot&,const Snapshot&,int,int64_t,std::string_view,std::string_view);
std::unique_ptr<Snapshot> native_palantir_kill_for_die(const Snapshot&,int,int64_t);
std::unique_ptr<Snapshot> apply_palantir_kill(const Snapshot&,const Snapshot&,int,int64_t,std::string_view,std::string_view);
bool ev_mouth_supported(const Snapshot& t,uint64_t card){
    return card!=missing && ctx_side(t,obs_attr(t,card,"side"))==1 && text_is(t,obs_attr(t,card,"card_type"),"Army") &&
        (eco_supported(t,card) || ev_simple_supported(t,card) || eho_supported(t,card) || ehi_supported(t,card) || esg_supported(t,card) || etsm_supported(t,card) || esl_supported(t,card) || euh_supported(t,card) ||
         // package Q (engine_ports layer 1): Return to Valinor and Denethor's Folly (the only Shadow Army card of the persistent-table
         // family) complete the r4 table's 11 Shadow Army cards; the frozen menu stamps every Army card alike (ai.py:7818-7825)
         erv_supported(t,card) || eht_type(t,card)!=nullptr);
}
bool esra_has_choices(const Snapshot&,const Snapshot&,std::string_view);
bool esra_pending(const Snapshot&);
std::unique_ptr<Snapshot> esra_begin(const Snapshot&,const Snapshot&,int,int64_t,std::string_view);
std::unique_ptr<Snapshot> native_shadow_allocation_choices(const Snapshot&,const Snapshot&,int);
std::unique_ptr<Snapshot> apply_shadow_allocation_choice(const Snapshot&,const Snapshot&,int,int64_t);
#include "action_dispatch.inc"
#include "full_action_menu.inc"
#include "next_turn.inc"
#include "siege_upkeep.inc"
#include "combat_envelope.inc"
#include "combat_trace.inc"
#include "combat_resume_envelope.inc"
#include "combat_action_upkeep.inc"
#include "event_palantir.inc"
#include "event_palantir_removal.inc"
#include "event_denethor.inc"
#include "event_worn.inc"
#include "event_power.inc"
#include "event_nazgul_hunt.inc"
#include "event_recruit.inc"
#include "event_house_stewards.inc"
#include "event_grey_company.inc"
#include "event_card_casualty.inc"
#include "event_faramir.inc"
#include "event_spirit_mordor.inc"
#include "event_recruit_targets.inc"
#include "event_shadow_recruit.inc"
#include "event_dreadful_spells.inc"
#include "event_eagles.inc"
#include "event_return_valinor.inc"
#include "event_dead_men.inc"
#include "event_horde_east.inc"
#include "event_i_will_go_alone.inc"
#include "event_there_back_again.inc"
#include "event_we_prove_swifter.inc"
#include "event_challenge_king.inc"
#include "event_stormcrow.inc"
#include "event_ents_awake.inc"
#include "event_another_way.inc"
#include "event_misty_mountains.inc"
#include "event_many_kings.inc"
#include "event_pits_mordor.inc"
#include "event_recruit_allocation.inc"
#include "event_half_orcs.inc"
#include "event_companion_moves.inc"
#include "event_gwaihir.inc"
#include "event_paths_woses.inc"
#include "event_cruel_weather.inc"
#include "event_hill_trolls.inc"
#include "event_overstack.inc"
#include "combat_casualty_action.inc"
#include "combat_onslaught_action.inc"
#include "combat_setup.inc"
#include "combat_bootstrap.inc"
#include "combat_entry_action.inc"
#include "combat_entry_menu.inc"
#include "combat_relocate.inc"
#include "event_shadow_join_rules.inc"
#include "card_army_cargo.inc"
#include "event_shadows_gather.inc"
#include "event_rage_dunlendings.inc"
#include "event_shadow_lengthens.inc"
#include "event_direct_army_move.inc"
#include "event_through_day.inc"
#include "event_shadow_is_moving.inc"
#include "combat_phase_action.inc"
#include "combat_phase_menu.inc"
#include "combat_advance_action.inc"
#include "combat_retreat_action.inc"
#include "army_attack_action.inc"
#include "card_army_followup.inc"
#include "event_ringwraiths.inc"
#include "event_black_captain.inc"
#include "event_fighting_uruk_hai.inc"
#include "event_grond.inc"
#include "event_help_unlooked.inc"
#include "event_corsairs.inc"
#include "combat_descriptor.inc"
#include "event_free_dispatch.inc"
#include "random_action_dispatch.inc"
#include "search_turn.inc"
#include "search_metadata.inc"
#include "chance_signature_wire.inc"
#include "overlay_assert.inc"
#include "kernel_context.inc"
#include "combat_search_facts.inc"
#include "combat_error_diagnostics.inc"
#include "search_value_facts.inc"
#include "search_driver_facts.inc"
#include "search_overlay.inc"
#include "belief_world.inc"
#include "search_preaction.inc"
#include "search_turn_driver.inc"
#include "card_cargo_observation.inc"

// ---------------------------------------------------------------------------------------------------------------
// WOTR_NATIVE_OBS_CHECK reference encoder.  From here to encode_state_ref: ROOT encode_prefix (snapshot.cpp 466-561),
// observation_support.inc, observation_globals.inc, observation_context.inc, observation_tails.inc and ROOT encode_state
// (865-892) VERBATIM with every column-0 definition renamed `_ref` (80 names; controller/ref_copy_tool.py, assembled by
// controller/obs_draft_tool.py), so the reference shares only the snapshot core helpers with the live encoder.
// `obs_check_active` is set by search_controller.inc from WOTR_NATIVE_OBS_CHECK=1 (never in timed runs); the `encode_state`
// wrapper after encode_state_impl then encodes every observation twice and throws on the first differing float bits
// or exception text.
#define WOTR_NATIVE_OBS_CHECK_AVAILABLE 1
int obs_check_active=0;uint64_t obs_check_observations_checked=0;
std::string obs_check_hex(uint32_t v){static const char digits[]="0123456789abcdef";std::string s="0x";for(int i=7;i>=0;--i)s+=digits[(v>>(i*4))&15];return s;}
void encode_prefix_ref(const Snapshot& s,int32_t side,const char* const* region_keys,size_t nr,
                   const char* const* nation_names,size_t nn,float* output,size_t count){
    require(side==0 || side==1,"invalid prefix observation side");
    check_array(region_keys,nr);check_array(nation_names,nn);check_array(output,count);
    const size_t max=std::numeric_limits<size_t>::max();
    require(nn<=(max-14)/2 && nn<=max/4,"prefix feature width overflows");
    const auto width=14+2*nn;
    require(nr<=(max-4*nn)/width && count==nr*width+4*nn,"prefix output count must be exact");
    require(nr<=static_cast<size_t>(std::numeric_limits<int32_t>::max()) &&
            nn<=static_cast<size_t>(std::numeric_limits<int32_t>::max()),"prefix metadata index range too large");
    std::vector<std::string> keys;
    std::vector<std::string_view> names;
    keys.reserve(nr);names.reserve(nn);
    for(size_t i=0;i<nr;++i){
        require(region_keys[i]!=nullptr,"null prefix region key");
        const std::string key(region_keys[i]);sanitize_name(key);keys.push_back(key);
    }
    std::unordered_map<std::string_view,int32_t> nation_index;
    for(size_t i=0;i<nn;++i){
        auto name=nation_argument(nation_names[i]);
        require(nation_index.emplace(name,static_cast<int32_t>(i)).second,"duplicate prefix nation name");
        names.push_back(std::move(name));
    }
    const auto state_attrs=attributes(s,s.root,"game_state.GameState");
    const auto army_dict=field(s,state_attrs,"armies");
    const auto& army_entries=typed(s,army_dict,WOTR_DICT,"armies must be dictionary");
    std::unordered_map<std::string,uint64_t> armies;
    for(uint64_t i=0;i<army_entries.count;i+=2){
        auto key=sanitize_name(text(s,s.edges[army_entries.offset+i]));
        armies[std::move(key)]=s.edges[army_entries.offset+i+1];
    }
    const auto controlled=field(s,state_attrs,"enemy_controlled");
    const auto fp_edge=enum_value_edge(s,controlled,"game_state.Side","FREE_PEOPLES");
    const auto sh_edge=enum_value_edge(s,controlled,"game_state.Side","SHADOW");
    const auto fp=fp_edge==missing ? missing : s.edges[fp_edge];
    const auto sh=sh_edge==missing ? missing : s.edges[sh_edge];
    const auto siege=field(s,state_attrs,"besieged_strongholds",false);
    const auto fellowship=fellowship_key(s,state_attrs);
    const auto political=attributes(s,field(s,state_attrs,"political"),"game_state.PoliticalTrack");
    const auto boxes=field(s,political,"boxes"),active=field(s,political,"active");
    std::vector<WotrNation> nations; nations.reserve(nn);
    for(const auto& name:names){
        const auto box_edge=enum_value_edge(s,boxes,"game_state.Nation",name);
        const auto active_edge=enum_value_edge(s,active,"game_state.Nation",name);
        const auto box=box_edge==missing ? 0 : typed(s,s.edges[box_edge],WOTR_INT,"prefix political box must be INT").integer;
        require(box>=0 && box<=3,"prefix political box must be zero through three");
        nations.push_back({fp_nation(name) ? 0 : 1,static_cast<int32_t>(box),
                           active_edge!=missing && truth(s,s.edges[active_edge]) ? 1 : 0});
    }
    std::vector<WotrRegion> regions;regions.reserve(nr);
    std::vector<WotrUnit> units;std::vector<WotrLeader> leaders;
    for(size_t i=0;i<nr;++i){
        const auto& key=keys[i];const auto found=armies.find(key);
        const auto army=found==armies.end() ? missing : found->second;
        const bool occupied=truth(s,army);
        regions.push_back({occupied ? 1 : 0,contains_string(s,siege,key) ? 1 : 0,
            contains_string(s,fp,key) ? 1 : 0,contains_string(s,sh,key) ? 1 : 0,key==fellowship ? 1 : 0});
        if(!occupied)continue;
        const auto attrs=attributes(s,army,"game_state.Army");
        const auto& army_units=typed(s,field(s,attrs,"units"),WOTR_LIST,"army units must be list");
        for(uint64_t j=0;j<army_units.count;++j){
            const auto unit_attrs=attributes(s,s.edges[army_units.offset+j],"game_state.Unit");
            const auto name=enum_member(s,field(s,unit_attrs,"nation"),"game_state.Nation");
            const auto nation=nation_index.find(name);require(nation!=nation_index.end(),"unit nation absent from prefix metadata");
            const auto kind=enum_member(s,field(s,unit_attrs,"kind"),"game_state.UnitKind");
            require(kind=="REGULAR" || kind=="ELITE","unsupported prefix unit kind");
            units.push_back({static_cast<int32_t>(i),nation->second,kind=="REGULAR" ? 0 : 1});
        }
        const auto& army_leaders=typed(s,field(s,attrs,"leaders"),WOTR_LIST,"army leaders must be list");
        for(uint64_t j=0;j<army_leaders.count;++j){
            const auto leader=s.edges[army_leaders.offset+j];
            if(s.nodes[leader].tag!=WOTR_OBJECT)continue; // no recognized name/class/nation attributes
            const auto type=object_type(s,leader);
            const auto leader_attrs=generic_attributes(s,leader);
            const auto name_node=field(s,leader_attrs,"name",false);
            int32_t leader_side=0,named=0,nazgul=0;
            if(name_node!=missing && (text_is(s,name_node,"Witch King") || text_is(s,name_node,"Saruman") || text_is(s,name_node,"Mouth of Sauron"))){
                leader_side=1;named=1;
            }else if(type=="game_state.Companion"){named=1;}
            else if(type=="game_state.Nazgul"){leader_side=1;nazgul=1;}
            else if(type=="game_logic.Minion"){leader_side=1;named=1;}
            else{
                const auto nation_node=field(s,leader_attrs,"nation",false);if(nation_node==missing)continue;
                const auto nation=enum_member(s,nation_node,"game_state.Nation");
                validate_nation_name(nation);leader_side=fp_nation(nation) ? 0 : 1;
            }
            leaders.push_back({static_cast<int32_t>(i),leader_side,named,nazgul});
        }
    }
    using Owner=std::unique_ptr<void,void(*)(void*)>;
    Owner projected(wotr_position_create(regions.data(),regions.size(),nations.data(),nations.size(),
                units.data(),units.size(),leaders.data(),leaders.size()),wotr_position_destroy);
    if(!projected)throw std::runtime_error(wotr_last_error());
    if(wotr_encode_regions_politics(projected.get(),side,output,count)!=0)
        throw std::runtime_error(wotr_last_error());
    const auto debt=field(s,state_attrs,"_wormtongue_attack_pending",false);
    if(debt!=missing&&s.nodes[debt].tag!=WOTR_NONE)
        for(size_t i=0;i<names.size();++i)if(names[i]=="ROHAN")
            output[nr*(14+2*nn)+4*i+1]-=.25f*static_cast<float>(typed(s,debt,WOTR_INT,"Invalid Wormtongue attack reaction").integer);
}
// Included inside snapshot.cpp's private namespace, after graph helpers.
// No Python objects or callbacks; object fields may live in slots or __dict__.
uint64_t obs_attr_ref(const Snapshot& s,uint64_t object,const char* name,bool required=true){
    if(object==missing || object>=s.nodes.size() || s.nodes[object].tag!=WOTR_OBJECT){
        require(!required,"observation expected an object attribute");return missing;
    }
    const auto& node=s.nodes[object];
    uint64_t dictionary=missing;
    for(uint64_t i=1;i<node.count;i+=2){
        const auto key=s.edges[node.offset+i];
        if(text_is(s,key,name))return s.edges[node.offset+i+1];
        if(text_is(s,key,"__dict__"))dictionary=s.edges[node.offset+i+1];
    }
    if(dictionary!=missing){
        const auto edge=string_value_edge(s,dictionary,name);
        if(edge!=missing)return s.edges[edge];
    }
    require(!required,"required observation attribute is missing");return missing;
}
// Frozen ai.encode_state_ref global block, offsets [3182,3401). Included inside
// snapshot.cpp's anonymous namespace after observation_support.inc.
double glb_num_ref(const Snapshot& s,uint64_t id,double fallback=0.0){
    if(id==missing)return fallback;
    const auto& n=s.nodes[id];
    if(n.tag==WOTR_INT || n.tag==WOTR_BOOL)return static_cast<double>(n.integer);
    if(n.tag==WOTR_FLOAT)return n.real;
    throw std::invalid_argument("global observation requires a numeric value");
}
uint64_t glb_get_ref(const Snapshot& s,uint64_t dict,std::string_view key,bool required=false){
    const auto edge=string_value_edge(s,dict,key);
    require(!required || edge!=missing,"missing global observation dictionary key");
    return edge==missing ? missing : s.edges[edge];
}
uint64_t glb_table_ref(const Snapshot& tables,std::string_view key){return glb_get_ref(tables,tables.root,key,true);}
std::vector<uint64_t> glb_items_ref(const Snapshot& s,uint64_t id){
    if(id==missing)return {};
    const auto& n=s.nodes[id];
    require(n.tag==WOTR_LIST || n.tag==WOTR_TUPLE || n.tag==WOTR_SET || n.tag==WOTR_FROZENSET,
            "global observation requires a sequence");
    return child_edges(s,id);
}
size_t glb_len_ref(const Snapshot& s,uint64_t id){
    if(id==missing)return 0;
    const auto& n=s.nodes[id];
    if(n.tag==WOTR_DICT)return static_cast<size_t>(n.count/2);
    if(n.tag==WOTR_LIST || n.tag==WOTR_TUPLE || n.tag==WOTR_SET || n.tag==WOTR_FROZENSET || n.tag==WOTR_BYTES)
        return static_cast<size_t>(n.count);
    if(n.tag==WOTR_STRING){
        const auto value=text(s,id);size_t count=0;
        for(unsigned char c:value)if((c&0xc0)!=0x80)++count;
        return count;
    }
    throw std::invalid_argument("global observation value has no length");
}
uint64_t glb_side_value_ref(const Snapshot& s,uint64_t dict,int32_t side){
    const auto edge=enum_value_edge(s,dict,"game_state.Side",side==0 ? "FREE_PEOPLES" : "SHADOW");
    return edge==missing ? missing : s.edges[edge];
}
int glb_name_index_ref(std::string_view name,const std::vector<std::string_view>& names){
    for(size_t i=0;i<names.size();++i)if(names[i]==name)return static_cast<int>(i);
    return -1;
}
std::vector<std::string_view> glb_names_ref(const Snapshot& tables,std::string_view key){
    std::vector<std::string_view> result;
    for(auto id:glb_items_ref(tables,glb_table_ref(tables,key)))result.push_back(text(tables,id));
    return result;
}
std::string glb_fellowship_key_ref(const Snapshot& s,uint64_t fellowship){
    const auto region=obs_attr_ref(s,fellowship,"region");
    if(s.nodes[region].tag==WOTR_STRING)return sanitize_name(text(s,region));
    if(s.nodes[region].tag==WOTR_NONE)return "none";
    if(s.nodes[region].tag==WOTR_BOOL)return s.nodes[region].integer ? "true" : "false";
    if(s.nodes[region].tag==WOTR_INT)return sanitize_name(std::to_string(s.nodes[region].integer));
    if(s.nodes[region].tag==WOTR_OBJECT){
        for(const char* key:{"key","name","orig_name"}){
            const auto value=obs_attr_ref(s,region,key,false);
            if(value!=missing)return sanitize_name(text(s,value));
        }
    }
    throw std::invalid_argument("unsupported strict fellowship region representation");
}
std::array<double,2> glb_live_vp_ref(const Snapshot& s,const Snapshot& tables){
    const auto state=attributes(s,s.root,"game_state.GameState");
    const auto controlled=field(s,state,"enemy_controlled");
    const auto fp=glb_side_value_ref(s,controlled,0),sh=glb_side_value_ref(s,controlled,1);
    const auto lists=glb_items_ref(tables,glb_table_ref(tables,"vp_settlements"));
    require(lists.size()==2,"VP settlement table requires two factions");
    std::array<double,2> result{};
    for(size_t owner=0;owner<2;++owner){
        for(auto entry:glb_items_ref(tables,lists[owner])){
            const auto pair=glb_items_ref(tables,entry);require(pair.size()==2,"invalid VP settlement pair");
            if(contains_string(s,owner==0 ? sh : fp,text(tables,pair[0])))
                result[1-owner]+=glb_num_ref(tables,pair[1]);
        }
    }
    return result;
}
int glb_card_index_ref(const Snapshot& s,uint64_t cid,const Snapshot& tables,uint64_t indices){
    if(cid==missing)return -1;
    const auto& n=s.nodes[cid];
    require(n.tag!=WOTR_LIST && n.tag!=WOTR_DICT && n.tag!=WOTR_SET,"unhashable card identity");
    if(n.tag!=WOTR_STRING)return -1;
    const auto value=glb_get_ref(tables,indices,text(s,cid));
    if(value==missing)return -1;
    const auto& index=typed(tables,value,WOTR_INT,"card index must be INT");
    require(index.integer>=0 && index.integer<96,"card index outside original embedding width");
    return static_cast<int>(index.integer);
}
int glb_card_deck_ref(const Snapshot& s,uint64_t cid){
    if(s.nodes[cid].tag!=WOTR_STRING)return -1;
    const auto value=text(s,cid);
    if(value.find("_Character_")!=std::string_view::npos)return 0;
    if(value.find("_Strategy_")!=std::string_view::npos)return 1;
    if(value.size()>=3 && value.substr(0,3)=="HU_"){
        if(value.find("_CHAR_")!=std::string_view::npos)return 0;
        if(value.find("_STRAT_")!=std::string_view::npos)return 1;
    }
    return -1;
}
uint64_t glb_hidden_commit_ref(const Snapshot& s,uint64_t state,int32_t side){
    const auto pc=field(s,state,"pending_combat",false);
    if(!truth(s,pc))return missing;
    const auto cid=glb_get_ref(s,pc,"atk_combat_card");
    const auto attacker=glb_get_ref(s,pc,"attacker_side");
    if(!truth(s,cid) || (attacker!=missing && enum_is(s,attacker,"game_state.Side",side==0 ? "FREE_PEOPLES" : "SHADOW")))
        return missing;
    const auto phase=glb_get_ref(s,pc,"phase");
    if(truth(s,glb_get_ref(s,pc,"atk_hidden_cc")) || (phase!=missing && text_is(s,phase,"defender_combat_card")))
        return cid;
    return missing;
}
bool glb_same_card_ref(const Snapshot& s,uint64_t a,uint64_t b){
    if(a==missing || b==missing)return false;
    if(a==b)return true;
    return s.nodes[a].tag==WOTR_STRING && s.nodes[b].tag==WOTR_STRING && text(s,a)==text(s,b);
}
double glb_ring_ref(const Snapshot& s,uint64_t count,uint64_t used){
    const auto& n=s.nodes[count];
    int64_t integer=0;
    if(n.tag==WOTR_INT || n.tag==WOTR_BOOL)integer=n.integer;
    else if(n.tag==WOTR_FLOAT){
        require(n.real==n.real && n.real>=static_cast<double>(std::numeric_limits<int64_t>::min()) &&
                n.real<9223372036854775808.0,"ring integer conversion outside range");
        integer=static_cast<int64_t>(n.real);
    }else throw std::invalid_argument("unsupported ring integer conversion");
    const double value=static_cast<double>(integer);
    return truth(s,used) ? -(value+1.0)/3.0 : value/3.0;
}

void append_globals_ref(const Snapshot& s,const Snapshot& tables,int32_t side,std::vector<float>& out){
    require(side==0 || side==1,"invalid globals observation side");
    const size_t first=out.size();
    const auto state=attributes(s,s.root,"game_state.GameState");
    const auto get=[&](const char* name,bool required=true){return field(s,state,name,required);};
    const auto add=[&](double value){out.push_back(static_cast<float>(value));};
    const auto number=[&](const char* name,bool required=true){return glb_num_ref(s,get(name,required));};
    const auto fellowship=get("fellowship");
    const auto fattr=[&](const char* name,bool required=true){return obs_attr_ref(s,fellowship,name,required);};
    const auto company=glb_items_ref(s,fattr("companions"));
    const auto guide=fattr("guide");
    const bool mordor=truth(s,fattr("mordor_entered"));
    double distance=0;
    if(!mordor){
        const auto maximum=glb_num_ref(tables,glb_table_ref(tables,"max_mordor_distance"));
        require(maximum!=0,"zero Mordor distance divisor");
        const auto location=glb_fellowship_key_ref(s,fellowship);
        const auto value=glb_get_ref(tables,glb_table_ref(tables,"distance_to_mordor"),location);
        distance=glb_num_ref(tables,value,maximum)/maximum;
    }
    add(glb_num_ref(s,fattr("progress"))/12.0);add(glb_num_ref(s,fattr("corruption"))/12.0);
    add(truth(s,fattr("hidden")) ? 1 : 0);add(mordor ? 1 : 0);
    add(glb_num_ref(s,fattr("mordor_step"))/5.0);add(static_cast<double>(company.size())/8.0);
    add((truth(s,guide) ? glb_num_ref(s,obs_attr_ref(s,guide,"level")) : 0)/3.0);add(1.0-distance);
    const auto vp=glb_live_vp_ref(s,tables);add(vp[side]/10.0);add(vp[1-side]/10.0);

    const auto pool=glb_items_ref(s,get("hunt_pool"));
    double eyes=0,special_sh=0,special_fp=0,stops=0,damage=0,damage_count=0;
    for(auto tile:pool){
        const bool eye=truth(s,obs_attr_ref(s,tile,"is_eye"));
        if(eye)eyes+=1;
        if(truth(s,obs_attr_ref(s,tile,"is_special",false)) && !truth(s,obs_attr_ref(s,tile,"is_fellowship_special",false)))special_sh+=1;
        if(truth(s,obs_attr_ref(s,tile,"is_fellowship_special",false)))special_fp+=1;
        if(truth(s,obs_attr_ref(s,tile,"stops_movement",false)))stops+=1;
        if(!eye){const auto d=glb_num_ref(s,obs_attr_ref(s,tile,"damage"));if(d>0){damage+=d;damage_count+=1;}}
    }
    double sh_in_box=0,fp_in_box=0;
    for(auto die:glb_items_ref(s,get("hunt_box"))){
        const auto owner=obs_attr_ref(s,die,"side");
        if(enum_is(s,owner,"game_state.Side","SHADOW"))sh_in_box+=1;
        if(enum_is(s,owner,"game_state.Side","FREE_PEOPLES"))fp_in_box+=1;
    }
    const double allocated=number("eyes_allocated",false);
    add(allocated/5.0);add(std::max(0.0,sh_in_box-allocated)/4.0);add(fp_in_box/5.0);
    add(static_cast<double>(pool.size())/16.0);add(eyes/4.0);add(special_sh/4.0);add(special_fp/4.0);add(stops/4.0);
    add((damage_count ? damage/damage_count : 0)/3.0);add(mordor ? 1 : 0);

    const auto faces=glb_names_ref(tables,"action_die_face_names");require(faces.size()==7,"expected seven die faces");
    std::array<std::array<double,7>,2> hist{};std::array<double,2> unused{};
    const auto dice=get("dice");
    for(int owner=0;owner<2;++owner){
        for(auto die:glb_items_ref(s,glb_side_value_ref(s,dice,owner))){
            if(truth(s,obs_attr_ref(s,die,"used")))continue;
            const auto face=obs_attr_ref(s,die,"current");
            if(s.nodes[face].tag==WOTR_NONE)continue;
            unused[owner]+=1;
            const auto index=glb_name_index_ref(enum_member(s,face,"game_state.ActionDieFace"),faces);
            require(index>=0,"unrecognized visible action-die face");hist[owner][index]+=1;
        }
    }
    add(unused[side]/10.0);add(unused[1-side]/10.0);
    for(int owner:{side,1-side})for(double count:hist[owner])add(count/4.0);

    const auto hands=get("hands"),my_info=glb_side_value_ref(s,hands,side),enemy_info=glb_side_value_ref(s,hands,1-side);
    const auto my_hand=glb_items_ref(s,my_info==missing ? missing : obs_attr_ref(s,my_info,"hand"));
    const auto enemy_hand=glb_items_ref(s,enemy_info==missing ? missing : obs_attr_ref(s,enemy_info,"hand"));
    std::array<double,2> enemy_decks{};
    for(auto cid:enemy_hand){const auto deck=glb_card_deck_ref(s,cid);if(deck>=0)enemy_decks[deck]+=1;}
    double my_count=static_cast<double>(my_hand.size()),enemy_count=static_cast<double>(enemy_hand.size());
    const auto pending_hand=get("pending_combat",false);
    if(pending_hand!=missing&&s.nodes[pending_hand].tag==WOTR_DICT)for(int role=0;role<2;++role){
        const auto hidden_hand=field(s,pending_hand,role==0?"atk_hidden_cc":"def_hidden_cc",false);if(!truth(s,hidden_hand))continue;
        const auto owner=field(s,pending_hand,role==0?"attacker_side":"defender_side",false);if(owner==missing||s.nodes[owner].tag!=WOTR_ENUM)continue;
        if(enum_is(s,owner,"game_state.Side",side==0?"FREE_PEOPLES":"SHADOW"))my_count=std::max(0.0,my_count-1.0);
        else if(enum_is(s,owner,"game_state.Side",side==0?"SHADOW":"FREE_PEOPLES")){enemy_count=std::max(0.0,enemy_count-1.0);const auto deck=field(s,hidden_hand,"deck",false);if(text_is(s,deck,"character"))enemy_decks[0]=std::max(0.0,enemy_decks[0]-1.0);else if(text_is(s,deck,"strategy"))enemy_decks[1]=std::max(0.0,enemy_decks[1]-1.0);}
    }
    add(my_count/6.0);add(enemy_count/6.0);
    add(enemy_decks[0]/6.0);add(enemy_decks[1]/6.0);
    for(const char* name:{"fp_char_deck","fp_strat_deck","sh_char_deck","sh_strat_deck"}){
        const auto deck=get(name,false);add(truth(s,deck) ? static_cast<double>(glb_len_ref(s,deck))/24.0 : 0);
    }
    const auto indices=glb_table_ref(tables,"card_indices");
    for(size_t i=0;i<6;++i)add(i<my_hand.size() ? glb_card_index_ref(s,my_hand[i],tables,indices) : -1);
    std::array<double,96> status{};
    auto hidden=glb_hidden_commit_ref(s,state,side);
    const auto discards=get("discard_face_up",false);
    if(discards!=missing){
        const auto& dict=typed(s,discards,WOTR_DICT,"face-up discards must be dictionary");
        for(uint64_t i=1;i<dict.count;i+=2)for(auto cid:glb_items_ref(s,s.edges[dict.offset+i])){
            if(hidden!=missing && glb_same_card_ref(s,cid,hidden)){hidden=missing;continue;}
            const auto index=glb_card_index_ref(s,cid,tables,indices);if(index>=0)status[index]=0.5;
        }
    }
    for(auto cid:glb_items_ref(s,my_info==missing ? missing : obs_attr_ref(s,my_info,"discarded_face_down"))){
        const auto index=glb_card_index_ref(s,cid,tables,indices);if(index>=0)status[index]=0.5;
    }
    const auto persistent=get("persistent_cards");
    if(s.nodes[persistent].tag!=WOTR_NONE){
        for(const char* name:{"fp_cards","sh_cards"})for(auto card:glb_items_ref(s,obs_attr_ref(s,persistent,name,false))){
            const auto index=glb_card_index_ref(s,obs_attr_ref(s,card,"card_id",false),tables,indices);
            if(index>=0)status[index]=1;
        }
    }
    for(double value:status)add(value);

    const auto flag=[&](const char* character,const char* attribute){
        const auto object=get(character);return truth(s,object) && truth(s,obs_attr_ref(s,object,attribute)) ? 1.0 : 0.0;
    };
    const std::array<double,5> characters={flag("gandalf_the_white","in_play"),flag("aragorn_crowned","crowned"),
        flag("witch_king","in_play"),flag("saruman","in_play"),flag("mouth_of_sauron","in_play")};
    if(side==0)for(double value:characters)add(value);
    else for(size_t i:{2,3,4,0,1})add(characters[i]);
    const std::array<double,3> fp_resources={glb_ring_ref(s,get("fp_elven_rings"),get("fp_ring_used_this_turn")),
        truth(s,get("fp_dwarven_rings")) ? number("fp_dwarven_rings")/1.0 : 0,truth(s,get("fp_political_token")) ? 1.0 : 0.0};
    const std::array<double,3> sh_resources={glb_ring_ref(s,get("sh_elven_rings"),get("sh_ring_used_this_turn")),
        truth(s,get("sh_dwarven_rings")) ? number("sh_dwarven_rings")/1.0 : 0,number("sh_action_tokens")/3.0};
    for(int owner:{side,1-side})for(double value:owner==0 ? fp_resources : sh_resources)add(value);
    add(number("turn")/15.0);
    const auto political=get("political"),boxes=obs_attr_ref(s,political,"boxes");
    double fp_war=0,sh_war=0;
    const auto nations=glb_names_ref(tables,"nation_names");require(nations.size()==8,"expected eight nation slots");
    for(auto nation:nations){
        const auto edge=enum_value_edge(s,boxes,"game_state.Nation",nation);
        require(edge!=missing,"missing political at-war nation");
        if(glb_num_ref(s,s.edges[edge])>=3){if(fp_nation(nation))fp_war+=1;else sh_war+=1;}
    }
    add(fp_war/5.0);add(sh_war/3.0);

    const std::vector<std::string_view> companion_names={"Gandalf the Grey","Strider","Legolas","Gimli","Boromir","Meriadoc","Peregrin","Gollum","Aragorn","Gandalf the White"};
    std::array<bool,10> alive{},in_company{},first_level_seen{},separated{};
    std::array<double,10> company_levels{},separated_levels{};
    std::array<int,10> separated_regions;separated_regions.fill(-1);
    for(auto companion:company){
        const auto i=glb_name_index_ref(text(s,obs_attr_ref(s,companion,"name")),companion_names);if(i<0)continue;
        alive[i]=in_company[i]=true;
        if(!first_level_seen[i]){company_levels[i]=glb_num_ref(s,obs_attr_ref(s,companion,"level"))/3.0;first_level_seen[i]=true;}
    }
    const auto guide_name=obs_attr_ref(s,guide,"name",false);
    if(guide_name!=missing && text_is(s,guide_name,"Gollum"))alive[7]=in_company[7]=true;
    const auto regions=glb_names_ref(tables,"region_keys");
    const auto armies=get("armies");const auto& army_dict=typed(s,armies,WOTR_DICT,"army map must be dictionary");
    std::array<double,3> minions{};
    const std::vector<std::string_view> minion_names={"Witch King","Saruman","Mouth of Sauron"};
    std::array<std::array<double,2>,8> deployed{};
    for(uint64_t i=0;i<army_dict.count;i+=2){
        const auto raw_key=s.edges[army_dict.offset+i],army=s.edges[army_dict.offset+i+1];
        int region_index=-2;
        const auto locate=[&](){if(region_index==-2)region_index=glb_name_index_ref(sanitize_name(text(s,raw_key)),regions);return region_index;};
        for(auto leader:glb_items_ref(s,obs_attr_ref(s,army,"leaders"))){
            const auto name=obs_attr_ref(s,leader,"name",false);
            if(s.nodes[leader].tag==WOTR_OBJECT && object_type(s,leader)=="game_state.Companion"){
                const auto index=glb_name_index_ref(text(s,obs_attr_ref(s,leader,"name")),companion_names);
                if(index>=0){alive[index]=separated[index]=true;separated_levels[index]=glb_num_ref(s,obs_attr_ref(s,leader,"level"));separated_regions[index]=locate();}
            }
            if(name!=missing && s.nodes[name].tag==WOTR_STRING){
                const auto index=glb_name_index_ref(text(s,name),minion_names);
                if(index>=0){const auto location=locate();if(location>=0)minions[index]=(location+1)/static_cast<double>(regions.size());}
            }
        }
        for(auto unit:glb_items_ref(s,obs_attr_ref(s,army,"units"))){
            const auto nation=enum_member(s,obs_attr_ref(s,unit,"nation"),"game_state.Nation");
            const auto index=glb_name_index_ref(nation,nations);require(index>=0,"deployed unit nation missing from table");
            const auto kind=obs_attr_ref(s,unit,"kind");
            ++deployed[index][enum_is(s,kind,"game_state.UnitKind","REGULAR") ? 0 : 1];
        }
    }
    for(size_t i=0;i<10;++i){
        double level=company_levels[i];
        if(alive[i] && level==0 && separated[i])level=separated_levels[i]/3.0;
        const auto region=separated_regions[i];
        const double location=region>=0 && !in_company[i] ? (region+1)/static_cast<double>(regions.size()) : 0;
        add(alive[i] ? 1 : 0);add(in_company[i] ? 1 : 0);add(level);add(location);
    }
    for(double value:minions)add(value);
    const auto limits=glb_table_ref(tables,"nation_max_figures");
    for(size_t i=0;i<nations.size();++i){
        const auto edge=enum_value_edge(tables,limits,"game_state.Nation",nations[i]);
        const auto maximum=edge==missing ? missing : tables.edges[edge];
        const auto regular=maximum==missing ? missing : glb_get_ref(tables,maximum,"regular");
        const auto elite=maximum==missing ? missing : glb_get_ref(tables,maximum,"elite");
        add(deployed[i][0]/std::max(1.0,glb_num_ref(tables,regular,15)));
        add(deployed[i][1]/std::max(1.0,glb_num_ref(tables,elite,5)));
    }
    require(out.size()-first==219,"global observation block width mismatch");
}
// Frozen ai.encode_state_ref offsets [3401,3746). Included inside snapshot.cpp's
// anonymous namespace. Read-only graph queries; no Python or fallback vectors.
bool ctx_none_ref(const Snapshot& s,uint64_t n){return n==missing || s.nodes[n].tag==WOTR_NONE;}
uint64_t ctx_get_ref(const Snapshot& s,uint64_t d,std::string_view key){
    if(ctx_none_ref(s,d))return missing;
    if(s.context_prepared){
        typed(s,d,WOTR_DICT,"prepared context requires dictionary");
        require(d<s.context_prepared->dictionaries.size(),"stale prepared context index");
        const auto& index=s.context_prepared->dictionaries[d];
        if(index){const auto found=index->find(key);return found==index->end() ? missing : found->second;}
    }
    const auto edge=string_value_edge(s,d,key);return edge==missing ? missing : s.edges[edge];
}
uint64_t ctx_attr_ref(const Snapshot& s,uint64_t obj,std::string_view name){
    if(ctx_none_ref(s,obj))return missing;
    const auto& n=typed(s,obj,WOTR_OBJECT,"context attribute requires object");
    for(uint64_t i=1;i<n.count;i+=2){
        const auto key=s.edges[n.offset+i],value=s.edges[n.offset+i+1];
        if(text_is(s,key,name))return value;
        if(text_is(s,key,"__dict__")){
            const auto result=ctx_get_ref(s,value,name);if(result!=missing)return result;
        }
    }
    return missing;
}
double ctx_num_ref(const Snapshot& s,uint64_t n,double absent=0){
    if(n==missing)return absent;
    const auto& v=s.nodes[n];
    if(v.tag==WOTR_INT || v.tag==WOTR_BOOL)return static_cast<double>(v.integer);
    if(v.tag==WOTR_FLOAT)return v.real;
    throw std::invalid_argument("context requires numeric scalar");
}
double ctx_getnum_ref(const Snapshot& s,uint64_t d,std::string_view key,double absent=0){return ctx_num_ref(s,ctx_get_ref(s,d,key),absent);}
double ctx_ornum_ref(const Snapshot& s,uint64_t n){return truth(s,n) ? ctx_num_ref(s,n) : 0.;}
double ctx_int_ref(const Snapshot& s,uint64_t n){
    const double value=ctx_ornum_ref(s,n);
    require(value>=-9007199254740991. && value<=9007199254740991.,"context integer outside exact supported range");
    return static_cast<double>(static_cast<int64_t>(value));
}
std::string ctx_string_ref(const Snapshot& s,uint64_t n,std::string_view absent=""){
    if(n==missing)return std::string(absent);
    if(s.nodes[n].tag==WOTR_STRING)return std::string(text(s,n));
    if(s.nodes[n].tag==WOTR_NONE)return "None";
    if(s.nodes[n].tag==WOTR_BOOL)return truth(s,n) ? "True" : "False";
    if(s.nodes[n].tag==WOTR_INT)return std::to_string(s.nodes[n].integer);
    throw std::invalid_argument("unsupported non-string context identifier");
}
std::vector<uint64_t> ctx_items_ref(const Snapshot& s,uint64_t n){
    if(n==missing)return {};
    const auto tag=s.nodes[n].tag;
    require(tag==WOTR_LIST || tag==WOTR_TUPLE || tag==WOTR_SET || tag==WOTR_FROZENSET,"context requires iterable sequence");
    return child_edges(s,n);
}
size_t ctx_len_or_ref(const Snapshot& s,uint64_t n){
    if(!truth(s,n))return 0;
    const auto& v=s.nodes[n];
    if(v.tag==WOTR_DICT)return static_cast<size_t>(v.count/2);
    require(v.tag==WOTR_LIST || v.tag==WOTR_TUPLE || v.tag==WOTR_SET || v.tag==WOTR_FROZENSET,"context length requires collection");
    return static_cast<size_t>(v.count);
}
int ctx_side_ref(const Snapshot& s,uint64_t n){
    if(ctx_none_ref(s,n))return -1;
    const auto name=enum_member(s,n,"game_state.Side");
    require(name=="FREE_PEOPLES" || name=="SHADOW","unknown context side");return name=="FREE_PEOPLES" ? 0 : 1;
}
std::string ctx_nation_ref(const Snapshot& s,uint64_t n){return std::string(validate_nation_name(enum_member(s,n,"game_state.Nation")));}
int ctx_unit_side_ref(const Snapshot& s,uint64_t unit){return fp_nation(ctx_nation_ref(s,ctx_attr_ref(s,unit,"nation"))) ? 0 : 1;}
bool ctx_companion_ref(const Snapshot& s,uint64_t n){return !ctx_none_ref(s,n) && object_type(s,n)=="game_state.Companion";}
bool ctx_plain_leader_family_ref(const Snapshot& s,uint64_t n){
    const auto t=object_type(s,n);return t=="game_state.Leader" || t=="game_state.Nazgul" || t=="game_logic.Minion";
}
bool ctx_minion_name_ref(const Snapshot& s,uint64_t n){
    const auto name=ctx_attr_ref(s,n,"name");return text_is(s,name,"Witch King") || text_is(s,name,"Saruman") || text_is(s,name,"Mouth of Sauron");
}
int ctx_participant_leader_side_ref(const Snapshot& s,uint64_t leader){
    if(ctx_companion_ref(s,leader))return 0;
    if(object_type(s,leader)=="game_state.Nazgul" || (ctx_plain_leader_family_ref(s,leader)&&ctx_minion_name_ref(s,leader)))return 1;
    if(ctx_plain_leader_family_ref(s,leader)&&!truth(s,ctx_attr_ref(s,leader,"name")))return ctx_unit_side_ref(s,leader);
    throw std::invalid_argument("unsupported board leader in battle participant view");
}
uint64_t ctx_side_value_ref(const Snapshot& s,uint64_t dict,int side){
    if(ctx_none_ref(s,dict))return missing;
    const auto edge=enum_value_edge(s,dict,"game_state.Side",side==0 ? "FREE_PEOPLES" : "SHADOW");
    return edge==missing ? missing : s.edges[edge];
}
uint64_t ctx_nation_value_ref(const Snapshot& s,uint64_t dict,std::string_view nation){
    if(ctx_none_ref(s,dict))return missing;
    const auto edge=enum_value_edge(s,dict,"game_state.Nation",nation);return edge==missing ? missing : s.edges[edge];
}
struct ctx_State_ref{
    const Snapshot& s;const Snapshot& t;
    uint64_t attrs,armies,fellowship,political,boxes,controlled,meta;
    std::vector<std::string> keys,nations;
    std::unordered_map<std::string,size_t> indices;
    std::unordered_map<std::string,uint64_t> normalized;
    std::vector<std::pair<std::string,uint64_t>> raw;
    // One encoding traverses the same neighboring armies many times. Cache
    // their side counts only for this read-only ctx_State_ref lifetime; no derived
    // data survives into another position, clone, mutation or observation.
    mutable std::unordered_map<uint64_t,std::array<int,2>> unit_counts;
    ctx_State_ref(const Snapshot& state,const Snapshot& tables):s(state),t(tables){
        attrs=attributes(s,s.root,"game_state.GameState");meta=t.root;
        // Metadata failures must never be mistaken for a Python availability
        // predicate exception and turned into a zero feature.
        for(const char* key:{"fp_vp_targets","sh_vp_targets","dist_to_fp_target","dist_to_sh_target",
            "fp_settle_neighbors_1","fp_settle_neighbors_2","sh_settle_neighbors_1","sh_settle_neighbors_2",
            "distance_to_mordor","nation_max_figures","live_vp_fp_settlements","live_vp_sh_settlements",
            "fp_vp_victory","shadow_vp_victory","board_regions","board_adj","region_nations",
            "shadow_strongholds","fp_region_keys","card_indices"})
            require(ctx_get_ref(t,meta,key)!=missing,"missing required context metadata table");
        armies=field(s,attrs,"armies");typed(s,armies,WOTR_DICT,"context armies dictionary");
        fellowship=field(s,attrs,"fellowship");political=field(s,attrs,"political");
        boxes=ctx_attr_ref(s,political,"boxes");controlled=field(s,attrs,"enemy_controlled");
        for(auto n:ctx_items_ref(t,field(t,meta,"region_keys"))){keys.emplace_back(text(t,n));indices[keys.back()]=keys.size()-1;}
        for(auto n:ctx_items_ref(t,field(t,meta,"nation_names")))nations.emplace_back(text(t,n));
        require(keys.size()==105 && nations.size()==8,"context requires pinned 105-region/eight-nation layout");
        const auto& a=s.nodes[armies];
        for(uint64_t i=0;i<a.count;i+=2){
            std::string key(text(s,s.edges[a.offset+i]));const auto value=s.edges[a.offset+i+1];
            raw.emplace_back(key,value);normalized[sanitize_name(key)]=value;
        }
    }
    uint64_t get(std::string_view name)const{return ctx_get_ref(s,attrs,name);}
    uint64_t table(std::string_view name)const{const auto n=ctx_get_ref(t,meta,name);require(n!=missing,"missing context metadata table");return n;}
    uint64_t army(const std::string& key)const{auto found=normalized.find(key);return found==normalized.end() ? missing : found->second;}
    uint64_t lookup(const std::string& key)const{
        auto direct=ctx_get_ref(s,armies,key);if(!ctx_none_ref(s,direct))return direct;
        const auto norm=sanitize_name(key);for(const auto& pair:raw)if(sanitize_name(pair.first)==norm)return pair.second;
        return missing;
    }
    bool control(int side,const std::string& key)const{return contains_string(s,ctx_side_value_ref(s,controlled,side),key);}
    bool atwar(std::string_view nation)const{auto n=ctx_nation_value_ref(s,boxes,nation);require(n!=missing,"political box missing nation");return ctx_num_ref(s,n)>=3;}
    size_t index(const std::string& key)const{auto p=indices.find(key);return p==indices.end() ? keys.size() : p->second;}
    std::vector<uint64_t> figures(uint64_t army,const char* kind)const{
        if(ctx_none_ref(s,army))return {};return ctx_items_ref(s,ctx_attr_ref(s,army,kind));
    }
    int units(uint64_t army,int side)const{
        if(ctx_none_ref(s,army))return 0;
        const auto known=unit_counts.find(army);if(known!=unit_counts.end())return known->second[side];
        std::array<int,2> counts{};for(auto u:figures(army,"units"))++counts[ctx_unit_side_ref(s,u)];
        unit_counts.emplace(army,counts);return counts[side];
    }
    bool flag(const char* object,const char* attr)const{return truth(s,ctx_attr_ref(s,get(object),attr));}
    bool in_fellowship(std::string_view name)const{
        for(auto c:ctx_items_ref(s,ctx_attr_ref(s,fellowship,"companions")))if(text_is(s,ctx_attr_ref(s,c,"name"),name))return true;return false;
    }
};
std::vector<std::pair<std::string,double>> ctx_targets_ref(const ctx_State_ref& c,const char* table){
    std::vector<std::pair<std::string,double>> out;
    for(auto n:ctx_items_ref(c.t,c.table(table))){auto parts=ctx_items_ref(c.t,n);require(parts.size()==2,"invalid settlement tuple");out.emplace_back(text(c.t,parts[0]),ctx_num_ref(c.t,parts[1]));}
    return out;
}
double ctx_live_vp_ref(const ctx_State_ref& c,int side){
    double vp=0;for(const auto& p:ctx_targets_ref(c,side==0 ? "live_vp_sh_settlements" : "live_vp_fp_settlements"))if(c.control(side,p.first))vp+=p.second;return vp;
}
int ctx_unused_ref(const ctx_State_ref& c,int side){
    const auto dice=ctx_side_value_ref(c.s,c.get("dice"),side);int count=0;
    for(auto die:ctx_items_ref(c.s,dice))if(!truth(c.s,ctx_attr_ref(c.s,die,"used"))&&!ctx_none_ref(c.s,ctx_attr_ref(c.s,die,"current")))++count;return count;
}
std::vector<std::string> ctx_neighbors_ref(const ctx_State_ref& c,const char* table,const std::string& key){
    const auto value=ctx_get_ref(c.t,c.table(table),key);std::vector<std::string> result;
    if(value!=missing)for(auto n:ctx_items_ref(c.t,value))result.emplace_back(text(c.t,n));return result;
}
void ctx_military_strategy_ref(const ctx_State_ref& c,int side,std::vector<float>& out){
    auto push=[&](double x){out.push_back(static_cast<float>(x));};
    std::array<std::unordered_map<std::string,int>,2> forces;
    for(const auto& key:c.keys){
        const auto army=c.army(key);if(!truth(c.s,army))continue;
        std::array<int,2> units{},leaders{};
        for(auto u:c.figures(army,"units"))++units[ctx_unit_side_ref(c.s,u)];
        for(auto ld:c.figures(army,"leaders")){
            int owner=-1;
            if(ctx_minion_name_ref(c.s,ld))owner=1;
            else if(ctx_attr_ref(c.s,ld,"nation")!=missing)owner=ctx_unit_side_ref(c.s,ld);
            else if(ctx_companion_ref(c.s,ld))owner=0;
            if(owner>=0)++leaders[owner];
        }
        for(int sd=0;sd<2;++sd)if(units[sd]+leaders[sd]>0)forces[sd][key]=units[sd];
    }
    const auto fp_targets=ctx_targets_ref(c,"fp_vp_targets"),sh_targets=ctx_targets_ref(c,"sh_vp_targets");
    auto& mine=side==0 ? fp_targets : sh_targets;auto& enemy=side==0 ? sh_targets : fp_targets;
    const char* distance[2]={"dist_to_fp_target","dist_to_sh_target"};
    for(int owner:{side,1-side}){
        std::array<double,3> sums{};const auto dist=c.table(distance[1-owner]);
        // Ordered source region traversal preserves the Python summation order.
        for(const auto& key:c.keys){auto p=forces[owner].find(key);if(p==forces[owner].end())continue;
            const double d=ctx_num_ref(c.t,ctx_get_ref(c.t,dist,key),99);for(int i=0;i<3;++i)if(d<=i+1)sums[i]+=p->second;}
        push(sums[0]/20.);push(sums[1]/30.);push(sums[2]/40.);
    }
    for(int target_owner:{side,1-side}){
        const auto& targets=target_owner==0 ? fp_targets : sh_targets;int threatened=0;
        const auto dist=c.table(distance[target_owner]);
        for(const auto& target:targets){
            for(const auto& key:c.keys){if(forces[1-target_owner].find(key)==forces[1-target_owner].end())continue;
                const double d=key==target.first ? 0 : ctx_num_ref(c.t,ctx_get_ref(c.t,dist,key),99);
                if(d<=1){++threatened;break;}}
        }
        push(threatened/10.);
    }
    const char* n1[2]={"fp_settle_neighbors_1","sh_settle_neighbors_1"};
    const char* n2[2]={"fp_settle_neighbors_2","sh_settle_neighbors_2"};
    double en_vp1=0,en_vp2=0,my_vp1=0,my_vp2=0;int en_max=0,my_min=5,my_undef=0,en_undef=0;
    for(const auto& target:mine){
        const int defence=std::min(5,c.units(c.army(target.first),side));if(!defence)++my_undef;my_min=std::min(my_min,defence);
        bool one=false,two=false;
        for(const auto& nb:ctx_neighbors_ref(c,n1[side],target.first)){int strength=c.units(c.army(nb),1-side);if(strength){one=true;en_max=std::max(en_max,std::min(5,strength));}}
        if(one)en_vp1+=target.second;two=one;
        if(!two)for(const auto& nb:ctx_neighbors_ref(c,n2[side],target.first))if(c.units(c.army(nb),1-side)){two=true;break;}
        if(two)en_vp2+=target.second;
    }
    for(const auto& target:enemy){
        if(!c.units(c.army(target.first),1-side))++en_undef;
        bool one=false,two=false;
        for(const auto& nb:ctx_neighbors_ref(c,n1[1-side],target.first))if(c.units(c.army(nb),side)){one=true;break;}
        if(one)my_vp1+=target.second;two=one;
        if(!two)for(const auto& nb:ctx_neighbors_ref(c,n2[1-side],target.first))if(c.units(c.army(nb),side)){two=true;break;}
        if(two)my_vp2+=target.second;
    }
    const auto& s=c.s;
    const double steps=truth(s,ctx_attr_ref(s,c.fellowship,"mordor_entered")) ? 5-ctx_num_ref(s,ctx_attr_ref(s,c.fellowship,"mordor_step")) : ctx_num_ref(c.t,ctx_get_ref(c.t,c.table("distance_to_mordor"),fellowship_key(s,c.attrs)),15)+5;
    const double vp_limit=ctx_num_ref(c.t,c.table(side==0 ? "shadow_vp_victory" : "fp_vp_victory"));
    for(double x:{en_vp1/10,en_vp2/10,my_vp1/10,my_vp2/10,en_max/5.,my_min/5.,my_undef/12.,en_undef/10.,steps/20.,(12-ctx_num_ref(s,ctx_attr_ref(s,c.fellowship,"corruption")))/12.,ctx_num_ref(s,c.get("turn"))/25.,std::max(0.,vp_limit-ctx_live_vp_ref(c,1-side))/10.})push(x);
}

void ctx_pools_ref(const ctx_State_ref& c,std::vector<float>& out){
    std::unordered_map<std::string,std::array<double,3>> used;
    for(const auto& entry:c.raw){
        for(auto u:c.figures(entry.second,"units")){auto nation=ctx_nation_ref(c.s,ctx_attr_ref(c.s,u,"nation"));++used[nation][enum_is(c.s,ctx_attr_ref(c.s,u,"kind"),"game_state.UnitKind","REGULAR") ? 0 : 1];}
        for(auto l:c.figures(entry.second,"leaders")){
            auto type=object_type(c.s,l);if(type=="game_state.Leader" || type=="game_state.Nazgul")++used[ctx_nation_ref(c.s,ctx_attr_ref(c.s,l,"nation"))][2];
        }
    }
    const char* names[]={"regular","elite","leader"};const auto caps=c.table("nation_max_figures");
    for(const auto& nation:c.nations){const auto maxes=ctx_nation_value_ref(c.t,caps,nation);
        auto casualties=ctx_nation_value_ref(c.s,c.get("fp_casualties"),nation);
        if(!truth(c.s,casualties))casualties=missing;
        for(size_t i=0;i<3;++i){double cap=ctx_getnum_ref(c.t,maxes,names[i]);double amount=used[nation][i];if(fp_nation(nation))amount+=ctx_getnum_ref(c.s,casualties,names[i]);
            out.push_back(static_cast<float>((cap>0 ? std::max(0.,cap-amount) : 0)/std::max(1.,cap)));}
    }
}

bool ctx_availability_ref(const ctx_State_ref& c,int which){
    const auto& s=c.s;
    if(which<3){
        const char* names[]={"witch_king","saruman","mouth_of_sauron"};
        if(c.flag(names[which],"in_play") || c.flag(names[which],"was_ever_in_play"))return false;
        if(which==0){
            bool any_fp=false;for(const auto& nation:c.nations)if(fp_nation(nation)&&c.atwar(nation))any_fp=true;
            if(!c.atwar("SAURON")||!any_fp)return false;
            for(const auto& entry:c.raw)for(auto u:c.figures(entry.second,"units"))if(enum_is(s,ctx_attr_ref(s,u,"nation"),"game_state.Nation","SAURON"))return true;
            return false;
        }
        if(which==1)return c.atwar("ISENGARD")&&!c.control(0,"orthanc");
        bool all_fp=true;for(const auto& nation:c.nations)if(fp_nation(nation)&&!c.atwar(nation))all_fp=false;
        if(!all_fp&&!truth(s,ctx_attr_ref(s,c.fellowship,"mordor_entered")))return false;
        for(auto key:ctx_items_ref(c.t,c.table("shadow_strongholds")))if(!c.control(0,std::string(text(c.t,key))))return true;return false;
    }
    if(which==3){
        if(c.flag("gandalf_the_white","in_play")||c.in_fellowship("Gandalf the Grey"))return false;
        if(!c.flag("witch_king","was_ever_in_play")&&!c.flag("saruman","was_ever_in_play")&&!c.flag("mouth_of_sauron","was_ever_in_play"))return false;
        for(const auto& entry:c.raw)for(auto ld:c.figures(entry.second,"leaders"))if(ctx_companion_ref(s,ld)&&text_is(s,ctx_attr_ref(s,ld,"name"),"Gandalf the Grey"))return true;
        for(const char* key:{"fangorn","grey havens","rivendell","lorien","woodland realm"})if(!c.control(1,key))return true;return false;
    }
    if(c.flag("aragorn_crowned","crowned")||c.in_fellowship("Strider"))return false;
    std::string region;
    for(const auto& entry:c.raw)for(auto ld:c.figures(entry.second,"leaders"))if(ctx_companion_ref(s,ld)&&text_is(s,ctx_attr_ref(s,ld,"name"),"Strider")){region=entry.first;break;}
    if(region.empty())return false;region=sanitize_name(region);
    return (region=="minas tirith"||region=="pelargir"||region=="dol amroth")&&!c.control(1,region);
}
void ctx_immediate_ref(const ctx_State_ref& c,int side,std::vector<float>& out){
    for(int owner:{side,1-side}){
        double value=0;int targets=0,dice=ctx_unused_ref(c,owner);const auto metadata=c.table("board_regions");
        for(const auto& key:c.keys){const auto info=ctx_get_ref(c.t,metadata,key);
            if(!truth(c.t,ctx_get_ref(c.t,info,"is_city"))&&!truth(c.t,ctx_get_ref(c.t,info,"is_stronghold")))continue;
            const auto nation=ctx_get_ref(c.t,c.table("region_nations"),key);
            if(ctx_none_ref(c.t,nation) || (fp_nation(ctx_nation_ref(c.t,nation)) ? 0 : 1)==owner || c.control(owner,key))continue;
            int best=0;for(const auto& nb:ctx_neighbors_ref(c,"board_adj",key))best=std::max(best,c.units(c.army(nb),owner));
            if(best>c.units(c.army(key),1-owner)&&targets<dice){value+=truth(c.t,ctx_get_ref(c.t,info,"is_stronghold")) ? 2 : 1;++targets;}
        }
        out.push_back(static_cast<float>(value/10.));
    }
    for(int which=0;which<5;++which){bool allowed=false;
        try{allowed=ctx_availability_ref(c,which);}catch(const std::exception&){
            if(which==4)for(const auto& entry:c.raw)for(auto ld:c.figures(entry.second,"leaders"))if(ctx_companion_ref(c.s,ld)&&text_is(c.s,ctx_attr_ref(c.s,ld,"name"),"Strider"))allowed=true;
        }
        out.push_back(allowed ? 1.f : 0.f);
    }
}

std::array<double,3> ctx_battle_roster_ref(const ctx_State_ref& c,const std::string& region,int side,uint64_t pc){
    const auto army=c.lookup(region);if(ctx_none_ref(c.s,army))return {};
    std::unordered_map<std::string,double> units_due,leaders_due;
    const auto payload=ctx_get_ref(c.s,pc,"attacker_rearguard");
    if(!ctx_none_ref(c.s,payload)){
        const auto parts=ctx_items_ref(c.s,payload);require(parts.size()==3,"malformed frozen rearguard payload");
        if(text_is(c.s,parts[0],sanitize_name(region))){
            require(side>=0,"rearguard owner requires Side");
            if(text_is(c.s,parts[1],side==0 ? "FREE_PEOPLES" : "SHADOW")){
                const auto signature=ctx_items_ref(c.s,parts[2]);require(signature.size()==3,"malformed rearguard signature");
                require(text_is(c.s,signature[0],"forced_rearguard_v1"),"unknown attack formation version");
                for(auto node:ctx_items_ref(c.s,signature[1])){auto row=ctx_items_ref(c.s,node);require(row.size()==3,"malformed rearguard unit count");units_due[std::string(text(c.s,row[0]))+"|"+std::string(text(c.s,row[1]))]=ctx_num_ref(c.s,row[2]);}
                for(auto node:ctx_items_ref(c.s,signature[2])){auto row=ctx_items_ref(c.s,node);require(row.size()==2,"malformed rearguard leader count");leaders_due[std::string(text(c.s,row[0]))]=ctx_num_ref(c.s,row[1]);}
            }
        }
    }
    std::unordered_map<std::string,double> relief_due;
    const auto relief=ctx_get_ref(c.s,pc,"corsairs_relief_garrison");
    if(truth(c.s,relief)&&side==ctx_side_ref(c.s,ctx_get_ref(c.s,pc,"attacker_side"))&&sanitize_name(region)==sanitize_name(ctx_string_ref(c.s,ctx_get_ref(c.s,pc,"defender_region"))))
        for(auto node:ctx_items_ref(c.s,relief)){const auto row=ctx_items_ref(c.s,node);require(row.size()==4,"Malformed relief roster");relief_due[ctx_string_ref(c.s,row[0])+"|"+ctx_string_ref(c.s,row[1])+"|"+ctx_string_ref(c.s,row[2])]=ctx_num_ref(c.s,row[3]);}
    std::array<double,3> result{};
    for(auto unit:c.figures(army,"units")){
        const auto nation=ctx_nation_ref(c.s,ctx_attr_ref(c.s,unit,"nation"));
        const auto kind=enum_member(c.s,ctx_attr_ref(c.s,unit,"kind"),"game_state.UnitKind");
        if(ctx_unit_side_ref(c.s,unit)!=side)continue;
        auto& reserve=relief_due["unit|"+nation+"|"+std::string(kind)];if(reserve>0){--reserve;continue;}
        auto& due=units_due[nation+"|"+std::string(kind)];if(due>0){--due;continue;}
        if(kind=="REGULAR")++result[0];else if(kind=="ELITE")++result[1];
    }
    for(auto leader:c.figures(army,"leaders")){
        const auto owner=ctx_participant_leader_side_ref(c.s,leader);if(owner!=side)continue;
        if(!relief_due.empty()&&!ctx_companion_ref(c.s,leader)){const auto key=ctx_minion_name_ref(c.s,leader)?"named|"+ctx_string_ref(c.s,ctx_attr_ref(c.s,leader,"name"))+"|":std::string(object_type(c.s,leader)=="game_state.Nazgul"?"nazgul|":"leader|")+ctx_nation_ref(c.s,ctx_attr_ref(c.s,leader,"nation"))+"|";
        auto& reserve=relief_due[key];if(reserve>0){--reserve;continue;}}
        if(ctx_plain_leader_family_ref(c.s,leader)&&!ctx_minion_name_ref(c.s,leader)){
            auto& due=leaders_due[ctx_nation_ref(c.s,ctx_attr_ref(c.s,leader,"nation"))];if(due>0){--due;continue;}
        }
        // The encoder independently side-filters the resulting roster. These
        // classes already have the same owner in this supported participant set.
        ++result[2];
    }
    for(const auto& item:units_due)require(item.second<=0,"protected rearguard unit missing");
    for(const auto& item:leaders_due)require(item.second<=0,"protected rearguard leader missing");
    for(const auto& item:relief_due)require(item.second<=0,"protected relief figure missing");
    return result;
}
int ctx_card_index_ref(const ctx_State_ref& c,uint64_t card){
    if(card==missing || c.s.nodes[card].tag!=WOTR_STRING)return -1;
    const auto n=ctx_get_ref(c.t,c.table("card_indices"),text(c.s,card));return n==missing ? -1 : static_cast<int>(ctx_num_ref(c.t,n));
}
int ctx_stack_cap_ref(const ctx_State_ref& c,const std::string& raw,int side){
    const auto key=sanitize_name(raw);if(!contains_string(c.s,c.get("besieged_strongholds"),key))return 10;
    int owner=-1;const auto& controlled=c.s.nodes[c.controlled];
    for(uint64_t i=0;i<controlled.count;i+=2){
        if(contains_string(c.s,c.s.edges[controlled.offset+i+1],key)){owner=ctx_side_ref(c.s,c.s.edges[controlled.offset+i]);break;}
    }
    if(owner<0)owner=contains_string(c.t,c.table("fp_region_keys"),key) ? 0 : 1;
    return side==owner ? 5 : 10;
}

void ctx_pending_ref(const ctx_State_ref& c,int side,std::vector<float>& out){
    const auto& s=c.s;auto push=[&](double value){out.push_back(static_cast<float>(value));};
    const char* companions[]={"Gandalf the Grey","Strider","Legolas","Gimli","Boromir","Meriadoc","Peregrin","Gollum","Aragorn","Gandalf the White"};
    const auto guide=ctx_attr_ref(s,c.fellowship,"guide");
    const auto guide_name=ctx_none_ref(s,guide) ? std::string() : sanitize_name(ctx_string_ref(s,ctx_attr_ref(s,guide,"name")));
    bool matched=false;
    for(const char* name:companions){bool match=!ctx_none_ref(s,guide)&&!matched&&sanitize_name(name)==guide_name;push(match);matched=matched||match;}

    const bool pam=!ctx_none_ref(s,c.get("pending_army_move_die")),pmu=!ctx_none_ref(s,c.get("pending_muster_die"));
    const auto pco=c.get("pending_card_option"),pc=c.get("pending_combat");
    push(pam);push(pmu);push(!ctx_none_ref(s,pco));
    std::vector<double> pending_from(c.keys.size(),0),battle_region(c.keys.size(),0);
    const auto from=pam ? c.get("pending_army_move_from") : pmu ? c.get("pending_muster_region") : missing;
    if(truth(s,from)){auto i=c.index(sanitize_name(ctx_string_ref(s,from)));if(i<c.keys.size())pending_from[i]=1;}
    for(double x:pending_from)push(x);
    std::array<double,27> combat_ctx{};
    if(!ctx_none_ref(s,pc)){
        const auto def=ctx_string_ref(s,ctx_get_ref(s,pc,"defender_region")),atk=ctx_string_ref(s,ctx_get_ref(s,pc,"attacker_region"));
        const auto bi=c.index(sanitize_name(def)),ai=c.index(sanitize_name(atk));
        if(bi<c.keys.size())battle_region[bi]=1;
        if(ai<c.keys.size()&&ai!=bi)battle_region[ai]=-1;
        const auto as=ctx_side_ref(s,ctx_get_ref(s,pc,"attacker_side")),ds=ctx_side_ref(s,ctx_get_ref(s,pc,"defender_side"));
        combat_ctx[0]=1;combat_ctx[1]=truth(s,ctx_get_ref(s,pc,"is_siege"));combat_ctx[2]=as==side;
        const auto attacker=ctx_battle_roster_ref(c,atk,as,pc),defender=ctx_battle_roster_ref(c,def,ds,pc);
        if(as>=0){combat_ctx[3]=attacker[0]/10.;combat_ctx[4]=attacker[1]/10.;combat_ctx[5]=attacker[2]/7.;}
        if(ds>=0){combat_ctx[6]=defender[0]/10.;combat_ctx[7]=defender[1]/10.;combat_ctx[8]=defender[2]/7.;}
        const double rounds=ctx_getnum_ref(s,pc,"rounds_fought");combat_ctx[9]=std::min(rounds,5.)/5.;
        const char* phases[]={"defender_siege_choice","attacker_combat_card","defender_combat_card","combat_effect_choice","combat_round_result","defender_retreat","attacker_siege_decision","attacker_casualty_policy","defender_casualty_policy","victor_advance","casualty_allocation","fp_shine"};
        for(size_t i=0;i<12;++i)if(text_is(s,ctx_get_ref(s,pc,"phase"),phases[i]))combat_ctx[10+i]=1;
        combat_ctx[22]=std::max(0.,ctx_getnum_ref(s,pc,"mandatory_rounds")-rounds)/3.;
        combat_ctx[23]=truth(s,ctx_get_ref(s,pc,"no_def_card_r1_unless_companion"));
        combat_ctx[24]=std::min(ctx_getnum_ref(s,pc,"atk_dice_reduction"),5.)/5.;
        combat_ctx[25]=std::min(ctx_getnum_ref(s,pc,"def_dice_reduction"),5.)/5.;
        combat_ctx[26]=truth(s,ctx_get_ref(s,pc,"attacker_no_retreat"));
    }
    const bool card_allocating=truth(s,ctx_get_ref(s,pco,"card_casualty"));
    if(card_allocating){auto i=c.index(sanitize_name(ctx_string_ref(s,ctx_get_ref(s,pco,"cas_region"))));if(i<c.keys.size())battle_region[i]=1;}
    for(double x:battle_region)push(x);for(double x:combat_ctx)push(x);
    const auto ph=c.get("pending_hunt");std::array<double,6> hunt{};
    if(!ctx_none_ref(s,ph)){
        hunt[0]=1;hunt[1]=std::min(ctx_getnum_ref(s,ph,"damage"),6.)/6.;
        const char* phases[]={"hunt_sacrifice","hunt_gollum","hobbit_place"};
        const auto phase=ctx_string_ref(s,ctx_get_ref(s,ph,"phase"));for(size_t i=0;i<3;++i)if(phase==phases[i])hunt[2+i]=1;
        hunt[5]=std::min(ctx_int_ref(s,ctx_get_ref(s,ph,"level")),4.)/4.;
    }
    for(double x:hunt)push(x);
    std::array<double,6> card{};
    if(!ctx_none_ref(s,pco)){
        card[0]=(ctx_card_index_ref(c,ctx_get_ref(s,pco,"card_id"))+1)/97.;
        card[1]=std::min<size_t>(ctx_len_or_ref(s,ctx_get_ref(s,pco,"options")),40)/40.;
        auto decider=ctx_get_ref(s,pco,"decider");if(decider==missing)decider=ctx_get_ref(s,pco,"side");
        card[2]=(text_is(s,decider,"fp"))==(side==0);
        card[3]=truth(s,ctx_get_ref(s,pco,"direct"));card[4]=truth(s,ctx_get_ref(s,pco,"resume_action"));
        card[5]=(text_is(s,ctx_get_ref(s,pco,"side"),"fp"))==(side==0);
    }
    const bool gandalf_open=!ctx_none_ref(s,pco)&&truth(s,ctx_get_ref(s,pco,"gandalf_draw"));
    const auto gandalf=gandalf_open ? pco : c.get("_gandalf_draw_pending");
    if(!ctx_none_ref(s,gandalf)){
        const auto source=ctx_get_ref(s,gandalf,"source_card_id");
        require(source!=missing && s.nodes[source].tag==WOTR_STRING,"Gandalf draw requires source card string");
        const int index=ctx_card_index_ref(c,source);const auto cid=text(s,source);
        const char* expected=cid.find("_Character_")!=std::string_view::npos ? "character" : "strategy";
        require(index>=0&&text_is(s,ctx_get_ref(s,gandalf,"deck"),expected),"Gandalf draw source/deck mismatch");
        const int code=index+1+97*text_is(s,ctx_get_ref(s,gandalf,"deck"),"strategy")+194*gandalf_open+388*(card[4]!=0);
        card[4]=-static_cast<double>(code)/776.;if(gandalf_open)card[0]=(index+1)/97.;
    }
    for(double x:card)push(x);

    const auto pd=c.get("pending_discard"),follow=c.get("pending_card_followup");
    push(!ctx_none_ref(s,pd));push(!ctx_none_ref(s,pd)&&ctx_side_ref(s,ctx_get_ref(s,pd,"side"))==side);
    push(!ctx_none_ref(s,c.get("pending_guide_change")));push(truth(s,c.get("_eye_allocation_phase")));push(truth(s,c.get("_declare_phase")));
    push(!ctx_none_ref(s,follow));push(truth(s,follow)&&text_is(s,ctx_get_ref(s,follow,"kind"),"wk_army"));
    push(truth(s,follow) ? ctx_getnum_ref(s,follow,"moves_left")/2. : 0.);
    push(std::min<size_t>(ctx_len_or_ref(s,c.get("pending_hunt_tiles")),3)/3.);
    push(truth(s,c.get("_crebain_hunt_bonus")));
    auto over=c.get("_overstack_regions");if(!truth(s,over))over=missing;
    push(std::min<size_t>(ctx_len_or_ref(s,over),3)/3.);int excess=0;
    for(auto raw:ctx_items_ref(s,over)){
        const auto key=ctx_string_ref(s,raw);const auto army=c.lookup(key);if(ctx_none_ref(s,army))continue;
        for(int owner=0;owner<2;++owner)excess+=std::max(0,c.units(army,owner)-ctx_stack_cap_ref(c,key,owner));
    }
    push(std::min(excess,6)/6.);push(std::min(ctx_int_ref(s,c.get("_ents_chain_n")),4.)/4.);
    const auto stash=c.get("_mithril_stash");
    if(truth(s,stash)){
        const auto tile=ctx_get_ref(s,stash,"tile");push(1);
        push(std::min(std::max(ctx_int_ref(s,ctx_attr_ref(s,tile,"damage")),-2.),3.)/3.);
        push(truth(s,ctx_attr_ref(s,tile,"is_eye")));push(truth(s,ctx_attr_ref(s,tile,"reveal")));
        push(truth(s,ctx_attr_ref(s,tile,"stops_movement")));push(truth(s,ctx_get_ref(s,stash,"mordor")));
    }else for(int i=0;i<6;++i)push(0);

    push(truth(s,c.get("fellowship_acted_in_mordor")));
    const auto pre=c.get("pending_army_move_preexist");
    push(std::min(ctx_int_ref(s,ctx_get_ref(s,pre,"regular")),10.)/10.);
    push(std::min(ctx_int_ref(s,ctx_get_ref(s,pre,"elite")),10.)/10.);
    push(std::min(ctx_int_ref(s,ctx_get_ref(s,pre,"leaders")),7.)/7.);
    const auto first=c.get("pending_muster_first_type");for(const char* type:{"regular","elite","leader"})push(text_is(s,first,type));
    const auto owner=pam ? c.get("pending_army_move_side") : pmu ? c.get("pending_muster_side") : missing;
    push(ctx_side_ref(s,owner)==side);push(!ctx_none_ref(s,owner)&&ctx_side_ref(s,owner)!=side);
    const bool allocating=!ctx_none_ref(s,pc)&&text_is(s,ctx_get_ref(s,pc,"phase"),"casualty_allocation");
    push(card_allocating ? ctx_ornum_ref(s,ctx_get_ref(s,pco,"cas_hits"))/10. : allocating ? ctx_ornum_ref(s,ctx_get_ref(s,pc,"atk_hits"))/10. : 0.);
    push(allocating ? ctx_ornum_ref(s,ctx_get_ref(s,pc,"def_hits"))/10. : 0.);
    push(card_allocating ? 2 : allocating ? (text_is(s,ctx_get_ref(s,pc,"alloc_stage"),"attacker") ? 1 : -1) : 0);
    push(truth(s,c.get("_fellowship_tail_phase")));
}

void append_context_ref(const Snapshot& s,const Snapshot& tables,int32_t side,std::vector<float>& out){
    require(side==0 || side==1,"context side must be FP/SH");
    require(out.size()==3401,"context must begin at observation offset 3401");
    ctx_State_ref c(s,tables);
    ctx_military_strategy_ref(c,side,out);require(out.size()==3421,"military/strategy context width drift");
    ctx_pools_ref(c,out);require(out.size()==3445,"reinforcement context width drift");
    ctx_immediate_ref(c,side,out);require(out.size()==3452,"availability context width drift");
    ctx_pending_ref(c,side,out);require(out.size()==3746,"pending context width drift");
}
// Exact v45--v52 append-only observation contract. Included in snapshot.cpp.
// Source oracle: the nine frozen static_r3 context modules; never executes Python.
bool tail_none_ref(const Snapshot& s,uint64_t n){return n==missing || s.nodes[n].tag==WOTR_NONE;}
uint64_t tail_get_ref(const Snapshot& s,uint64_t n,const char* key){
    if(tail_none_ref(s,n))return missing;
    return field(s,n,key,false);
}
uint64_t tail_attr_ref(const Snapshot& s,uint64_t n,const char* key){
    if(tail_none_ref(s,n))return missing;
    const auto& o=typed(s,n,WOTR_OBJECT,"expected observation object");
    for(uint64_t i=1;i<o.count;i+=2)if(text_is(s,s.edges[o.offset+i],key))return s.edges[o.offset+i+1];
    for(uint64_t i=1;i<o.count;i+=2)if(text_is(s,s.edges[o.offset+i],"__dict__"))return tail_get_ref(s,s.edges[o.offset+i+1],key);
    return missing;
}
bool tail_seq_ref(const Snapshot& s,uint64_t n){return n!=missing&&(s.nodes[n].tag==WOTR_LIST||s.nodes[n].tag==WOTR_TUPLE);}
std::vector<uint64_t> tail_items_ref(const Snapshot& s,uint64_t n){
    if(tail_none_ref(s,n))return {};
    const auto tag=s.nodes[n].tag;
    require(tag==WOTR_LIST||tag==WOTR_TUPLE||tag==WOTR_SET||tag==WOTR_FROZENSET,"expected observation collection");
    return child_edges(s,n);
}
double tail_number_ref(const Snapshot& s,uint64_t n,double absent=0){
    if(n==missing)return absent;
    const auto& v=s.nodes[n];
    if(v.tag==WOTR_INT||v.tag==WOTR_BOOL)return static_cast<double>(v.integer);
    if(v.tag==WOTR_FLOAT)return v.real;
    if(v.tag==WOTR_STRING||v.tag==WOTR_BIGINT){
        const std::string value(reinterpret_cast<const char*>(s.bytes.data()+v.offset),v.count);
        size_t used=0;const double number=std::stod(value,&used);
        while(used<value.size() && (value[used]==' '||value[used]=='\t'||value[used]=='\n'||value[used]=='\r'))++used;
        require(used==value.size(),"invalid numeric observation value");return number;
    }
    throw std::invalid_argument("invalid numeric observation value");
}
int64_t tail_integer_ref(const Snapshot& s,uint64_t n,const char* message){return typed(s,n,WOTR_INT,message).integer;}
bool tail_equal_ref(const Snapshot& s,uint64_t a,uint64_t b,size_t depth=0){
    if(a==b)return true;
    if(a==missing||b==missing)return false;
    require(depth<100,"cyclic observation history equality");
    const auto& x=s.nodes[a];const auto& y=s.nodes[b];
    const bool xn=x.tag==WOTR_INT||x.tag==WOTR_BOOL||x.tag==WOTR_FLOAT;
    const bool yn=y.tag==WOTR_INT||y.tag==WOTR_BOOL||y.tag==WOTR_FLOAT;
    if(xn&&yn){
        if(x.tag!=WOTR_FLOAT&&y.tag!=WOTR_FLOAT)return x.integer==y.integer;
        return static_cast<long double>(x.tag==WOTR_FLOAT?x.real:static_cast<long double>(x.integer))==
               static_cast<long double>(y.tag==WOTR_FLOAT?y.real:static_cast<long double>(y.integer));
    }
    if(x.tag!=y.tag)return false;
    if(x.tag==WOTR_NONE)return true;
    if(x.tag==WOTR_STRING||x.tag==WOTR_BYTES||x.tag==WOTR_BIGINT)
        return x.count==y.count&&(!x.count||std::memcmp(s.bytes.data()+x.offset,s.bytes.data()+y.offset,x.count)==0);
    if(x.count!=y.count)return false;
    if(x.tag==WOTR_DICT){
        for(uint64_t i=0;i<x.count;i+=2){bool found=false;
            for(uint64_t j=0;j<y.count;j+=2)if(tail_equal_ref(s,s.edges[x.offset+i],s.edges[y.offset+j],depth+1)){
                found=tail_equal_ref(s,s.edges[x.offset+i+1],s.edges[y.offset+j+1],depth+1);break;}
            if(!found)return false;
        }return true;
    }
    for(uint64_t i=0;i<x.count;++i)if(!tail_equal_ref(s,s.edges[x.offset+i],s.edges[y.offset+i],depth+1))return false;
    return true;
}
template<size_t N> int tail_index_ref(const Snapshot& s,uint64_t n,const char* const (&names)[N]){
    for(size_t i=0;i<N;++i)if(text_is(s,n,names[i]))return static_cast<int>(i);return -1;
}
const char* const tail_companions_ref[]={"Gandalf the Grey","Strider","Legolas","Gimli","Boromir","Meriadoc","Peregrin","Aragorn","Gandalf the White","Gollum"};
const char* const tail_strongholds_ref[]={"barad dur","dol guldur","minas morgul","morannon","moria","mount gundabad","orthanc","umbar"};
const char* const tail_sources_ref[]={"hunt","mordor","balrog","orc_patrol","foul_thing","isildur","stronghold","breaking"};
void tail_keys_ref(const Snapshot& s,uint64_t n,std::initializer_list<const char*> allowed,bool exact,const char* message){
    const auto& d=typed(s,n,WOTR_DICT,message);
    require(!exact||d.count==allowed.size()*2,message);
    for(uint64_t i=0;i<d.count;i+=2){bool found=false;
        for(auto key:allowed)if(text_is(s,s.edges[d.offset+i],key))found=true;
        require(found,message);
    }
}
int tail_region_ref(const Snapshot& t,uint64_t keys,std::string_view key){
    const auto& list=t.nodes[keys];for(uint64_t i=0;i<list.count;++i)if(text_is(t,t.edges[list.offset+i],key))return static_cast<int>(i);return -1;
}
uint64_t tail_army_ref(const Snapshot& s,uint64_t attrs,std::string_view key){
    const auto armies=field(s,attrs,"armies");const auto exact=tail_get_ref(s,armies,std::string(key).c_str());
    if(!tail_none_ref(s,exact))return exact;
    const auto& d=s.nodes[armies];
    for(uint64_t i=0;i<d.count;i+=2)if(sanitize_name(text(s,s.edges[d.offset+i]))==key)return s.edges[d.offset+i+1];
    return missing;
}
void tail_commitments_ref(const Snapshot& s,const Snapshot& t,uint64_t keys,uint64_t attrs,uint64_t pending,float* out){
    const bool tsl=text_is(s,tail_get_ref(s,pending,"card_id"),"SHADOW_Strategy_The_Shadow_Lengthens");
    const bool tsim=text_is(s,tail_get_ref(s,pending,"card_id"),"SHADOW_Strategy_The_Shadow_is_Moving");
    if(!tsl&&!tsim)return;
    const auto options=tail_items_ref(s,tail_get_ref(s,pending,"options"));const char* hkey=tsl?"_tsl_moves":"_tsim_moves";
    uint64_t history=missing;bool any=false,all=true;
    for(auto option:options){
        if(tsl)tail_keys_ref(s,option,{"from_region","to_region","n_regular","n_elite","_tsl_moves","_tsl_done","_from_pending","cargo","count","_cargo_key","_cargo_index","_cargo_continue","_cargo_complete"},false,"TSL choice includes a group distinction outside the movement contract");
        else tail_keys_ref(s,option,{"from_region","to_region","_tsim_moves","_tsim_done","_from_pending","cargo","count","_cargo_key","_cargo_index","_cargo_continue","_cargo_complete"},false,"TSiM pending includes unsupported group or legacy immediate history");
        const auto h=tail_get_ref(s,option,hkey);all=all&&h!=missing;
        if(h!=missing){if(any)require(tail_equal_ref(s,history,h),"Conflicting movement commitment histories");else history=h;any=true;}
    }
    if(!any)return;
    if(tail_seq_ref(s,history)&&tail_items_ref(s,history).empty())return;
    require(all&&truth(s,tail_get_ref(s,pending,"direct")),"Incomplete movement commitment provenance");
    require(tail_seq_ref(s,history),"Malformed movement commitment history");
    auto moves=tail_items_ref(s,history);require(tsl?moves.size()==1:moves.size()>=1&&moves.size()<=3,"Invalid committed movement count");
    struct Move{std::string source,target;int source_index,target_index,regular,elite;};std::vector<Move> result;
    for(auto move:moves){
        require(tail_seq_ref(s,move),"Malformed movement commitment");auto parts=tail_items_ref(s,move);
        require(tsl?(parts.size()==2||parts.size()==4||parts.size()==5):(parts.size()==2||parts.size()==3),"Malformed movement commitment");
        const auto source=sanitize_name(text(s,parts[0])),target=sanitize_name(text(s,parts[1]));
        int si=tail_region_ref(t,keys,source),ti=tail_region_ref(t,keys,target);
        require(si>=0&&ti>=0&&si!=ti,"Invalid movement commitment regions");
        const auto army=tail_army_ref(s,attrs,source);require(!tail_none_ref(s,army),"Staged source army missing before movement");
        int counts[2]={0,0};for(auto unit:tail_items_ref(s,tail_attr_ref(s,army,"units"))){
            auto nat=tail_attr_ref(s,unit,"nation");bool shadow=enum_is(s,nat,"game_state.Nation","SAURON")||enum_is(s,nat,"game_state.Nation","ISENGARD")||enum_is(s,nat,"game_state.Nation","SOUTHRON_EASTERLINGS");
            if(shadow){const auto kind=tail_attr_ref(s,unit,"kind");counts[0]+=enum_is(s,kind,"game_state.UnitKind","REGULAR");counts[1]+=enum_is(s,kind,"game_state.UnitKind","ELITE");}
        }
        if(tsl&&parts.size()>=4)for(int i=0;i<2;++i)if(!tail_none_ref(s,parts[i+2])){
            auto n=tail_integer_ref(s,parts[i+2],"Malformed committed count");require(n>=0&&n<=counts[i],"Committed count exceeds source");counts[i]=static_cast<int>(n);}
        const auto cargo=(tsl&&parts.size()==5)?parts[4]:(!tsl&&parts.size()==3)?parts[2]:missing;
        if(!tail_none_ref(s,cargo)){counts[0]=counts[1]=0;for(auto row:tail_items_ref(s,cargo)){const auto values=tail_items_ref(s,row);require(values.size()==4,"Malformed committed cargo");if(text_is(s,values[0],"unit")){const auto n=tail_integer_ref(s,values[3],"Cargo count integer");require(n>0&&n<=10,"Invalid committed cargo count");const int kind=text_is(s,values[2],"REGULAR")?0:text_is(s,values[2],"ELITE")?1:-1;require(kind>=0,"Unknown committed cargo kind");counts[kind]+=static_cast<int>(n);}}}
        require(counts[0]+counts[1]>0,"Cannot commit empty Shadow army");
        result.push_back({source,target,si,ti,counts[0],counts[1]});
    }
    if(tsim){std::sort(result.begin(),result.end(),[](const Move&a,const Move&b){return a.source<b.source;});
        for(size_t i=1;i<result.size();++i)require(result[i-1].source!=result[i].source,"Duplicate staged TSiM source");}
    for(size_t i=0;i<result.size();++i){const auto&m=result[i];auto p=out+108*i;p[0]=1;p[1+m.source_index]=1;p[1+m.target_index]=-1;p[106]=static_cast<float>(m.regular/10.0);p[107]=static_cast<float>(m.elite/10.0);}
}
void tail_nazgul_ref(const Snapshot& s,const Snapshot& t,uint64_t keys,uint64_t pending,float* out){
    if(!text_is(s,tail_get_ref(s,pending,"card_id"),"__nazgul_moves__")&&!text_is(s,tail_get_ref(s,pending,"card_name"),"__nazgul_moves__"))return;
    require(truth(s,tail_get_ref(s,pending,"direct")),"Nazgul continuation lacks direct movement context");
    const auto opts=tail_get_ref(s,pending,"options");require(tail_seq_ref(s,opts)&&truth(s,opts),"Nazgul continuation has no public options");
    uint64_t history=missing;
    for(auto option:tail_items_ref(s,opts)){
        typed(s,option,WOTR_DICT,"Invalid Nazgul continuation option");if(truth(s,tail_get_ref(s,option,"_naz_done")))continue;
        const auto h=tail_get_ref(s,option,"_naz_arr");typed(s,h,WOTR_DICT,"Nazgul flight lacks public arrival provenance");
        if(history!=missing)require(tail_equal_ref(s,history,h),"Conflicting Nazgul arrival histories");else history=h;
    }
    require(history!=missing,"Nazgul continuation has no remaining flight");out[0]=1;
    const char* const flags[]={"__wk_ok__","__wk__","__mo_ok__","__mo__"};const auto& d=s.nodes[history];
    for(uint64_t i=0;i<d.count;i+=2){auto k=s.edges[d.offset+i],v=s.edges[d.offset+i+1];auto n=tail_integer_ref(s,v,"Invalid Nazgul history count");
        const int flag=tail_index_ref(s,k,flags);if(flag>=0){require(n==0||n==1,"Invalid Nazgul Minion eligibility flag");out[106+flag]=static_cast<float>(n);}
        else{const auto region=tail_region_ref(t,keys,text(s,k));require(region>=0&&n>=0&&n<=9,"Invalid Nazgul arrival region or count");out[1+region]=static_cast<float>(n/9.0);}
    }
    require(!truth(s,tail_get_ref(s,history,"__wk__"))||truth(s,tail_get_ref(s,history,"__wk_ok__")),"Moved Witch-king was not eligible for this chain");
    require(!truth(s,tail_get_ref(s,history,"__mo__"))||truth(s,tail_get_ref(s,history,"__mo_ok__")),"Moved Mouth was not eligible for this chain");
}
void tail_companion_moves_ref(const Snapshot& s,uint64_t pending,float* out){
    if(!text_is(s,tail_get_ref(s,pending,"card_id"),"__companion_moves__"))return;
    require(truth(s,tail_get_ref(s,pending,"direct")),"Companion continuation lacks direct movement provenance");
    const auto opts=tail_get_ref(s,pending,"options");require(tail_seq_ref(s,opts)&&truth(s,opts),"Companion continuation has no public options");
    unsigned first=0;bool found=false;
    for(auto option:tail_items_ref(s,opts)){
        typed(s,option,WOTR_DICT,"Malformed Companion movement option");if(truth(s,tail_get_ref(s,option,"_comp_done")))continue;
        const auto h=tail_get_ref(s,option,"_comp_moved");require(tail_seq_ref(s,h)&&truth(s,h),"Missing already-moved Companion names");unsigned mask=0;
        for(auto name:tail_items_ref(s,h)){int i=tail_index_ref(s,name,tail_companions_ref);require(i>=0&&i<9&&!(mask&(1u<<i)),"Malformed already-moved Companion names");mask|=1u<<i;}
        require(!found||mask==first,"Conflicting Companion movement histories");first=mask;found=true;
    }
    require(found,"Companion continuation has missing histories");out[0]=1;for(int i=0;i<9;++i)out[1+i]=(first>>i)&1;
}
void tail_die_ref(const Snapshot& s,uint64_t attrs,float* out){
    const auto index=tail_get_ref(s,attrs,"_fellowship_move_die_pending");if(tail_none_ref(s,index))return;
    auto idx=tail_integer_ref(s,index,"Invalid pending Fellowship movement die index");
    const auto dice=field(s,attrs,"dice");const auto edge=enum_value_edge(s,dice,"game_state.Side","FREE_PEOPLES");require(edge!=missing,"Missing FP dice");auto items=tail_items_ref(s,s.edges[edge]);
    require(idx>=0&&static_cast<uint64_t>(idx)<items.size(),"Invalid pending Fellowship movement die index");auto die=items[static_cast<size_t>(idx)];
    require(enum_is(s,tail_attr_ref(s,die,"side"),"game_state.Side","FREE_PEOPLES")&&truth(s,tail_attr_ref(s,die,"used"))&&!tail_none_ref(s,tail_attr_ref(s,die,"current")),"Pending Fellowship movement die must be consumed");
    for(auto other:tail_items_ref(s,field(s,attrs,"hunt_box")))require(other!=die,"Pending Fellowship movement die is already in the Hunt Box");out[0]=1;
}
int tail_source_ref(const Snapshot& s,uint64_t stash){
    const auto kind=tail_get_ref(s,stash,"apply_kind");if(!tail_none_ref(s,kind)){int k=tail_index_ref(s,kind,tail_sources_ref);require(k>=3,"Unknown Hunt protection draw source");return k;}
    if(truth(s,tail_get_ref(s,stash,"mordor_track")))return 1;return truth(s,tail_get_ref(s,stash,"eye_no_effect"))?2:0;
}
void tail_stash_ref(const Snapshot& s,uint64_t stash,float* out){
    if(tail_none_ref(s,stash))return;typed(s,stash,WOTR_DICT,"Malformed Hunt protection continuation");require(truth(s,stash),"Malformed Hunt protection continuation");
    int source=tail_source_ref(s,stash);auto count=tail_get_ref(s,stash,"successes");auto n=count==missing?0:tail_integer_ref(s,count,"Hunt successes must be an integer from zero to five");
    require(n>=0&&n<=5,"Hunt successes must be an integer from zero to five");if(source<7)out[source]=1;out[7]=static_cast<float>(n/5.0);
}
void tail_stronghold_queue_ref(const Snapshot& s,uint64_t queue,float* out,const char* message){
    if(tail_none_ref(s,queue))return;require(tail_seq_ref(s,queue)&&s.nodes[queue].count<=8,message);unsigned seen=0;size_t position=0;
    for(auto item:tail_items_ref(s,queue)){int k=tail_index_ref(s,item,tail_strongholds_ref);require(k>=0&&!(seen&(1u<<k)),message);seen|=1u<<k;out[position++*8+k]=1;}
}
void tail_hunt_ref(const Snapshot& s,uint64_t attrs,uint64_t pending,float* protection,float* route,float* event,float* breaking){
    const auto mithril=tail_get_ref(s,attrs,"_mithril_stash"),wizard=tail_get_ref(s,attrs,"_wizardstaff_stash");tail_stash_ref(s,mithril,protection);tail_stash_ref(s,wizard,protection+9);
    if(!tail_none_ref(s,mithril)){auto returning=tail_get_ref(s,mithril,"return_pending");if(returning!=missing)protection[8]=static_cast<float>(typed(s,returning,WOTR_BOOL,"Malformed Mithril return status").integer);}
    if(truth(s,tail_get_ref(s,attrs,"_reveal_place_due"))){const auto seen=tail_get_ref(s,attrs,"_sh_tile_drawn_regions");if(truth(s,seen))for(auto item:tail_items_ref(s,seen)){int i=tail_index_ref(s,item,tail_strongholds_ref);require(i>=0,"Unknown stronghold in reveal-event history");protection[17+i]=1;}}
    tail_stronghold_queue_ref(s,tail_get_ref(s,attrs,"_pending_reveal_strongholds"),route,"Malformed pending reveal stronghold draws");
    const auto e=tail_get_ref(s,attrs,"_fellowship_route_event");if(!tail_none_ref(s,e)){
        tail_keys_ref(s,e,{"kind","balrog_due","conditional_strongholds"},true,"Malformed Fellowship route event");const char* const kinds[]={"reveal","declare"};int k=tail_index_ref(s,tail_get_ref(s,e,"kind"),kinds);require(k>=0,"Malformed Fellowship route event");event[k]=1;
        event[2]=static_cast<float>(typed(s,tail_get_ref(s,e,"balrog_due"),WOTR_BOOL,"Malformed Fellowship route event").integer);
        const auto q=tail_get_ref(s,e,"conditional_strongholds");require(tail_seq_ref(s,q),"Malformed conditional stronghold order");tail_stronghold_queue_ref(s,q,event+3,"Malformed conditional stronghold order");
    }
    breaking[0]=truth(s,mithril)&&tail_source_ref(s,mithril)==7;breaking[1]=truth(s,wizard)&&tail_source_ref(s,wizard)==7;
    if(tail_none_ref(s,pending))return;typed(s,pending,WOTR_DICT,"Malformed pending card context");const auto sep=tail_get_ref(s,pending,"bof_separation"),count=tail_get_ref(s,pending,"bof_n_left");
    if(!truth(s,sep)){require(count==missing,"Breaking count without a separation context");return;}
    require(s.nodes[sep].tag==WOTR_BOOL&&s.nodes[sep].integer==1,"Malformed Breaking separation context");auto n=tail_integer_ref(s,count,"Breaking remaining choices must be an integer from one to eight");require(n>=1&&n<=8,"Breaking remaining choices must be an integer from one to eight");
    int eligible=0;auto fship=field(s,attrs,"fellowship");for(auto c:tail_items_ref(s,tail_attr_ref(s,fship,"companions"))){auto name=tail_attr_ref(s,c,"name");eligible+=!text_is(s,name,"Frodo")&&!text_is(s,name,"Sam")&&!text_is(s,name,"Gollum");}
    require(n<=eligible,"Breaking remaining choices exceed eligible Companions");breaking[2]=static_cast<float>(n/8.0);
}
std::string tail_normal_ref(const Snapshot& s,uint64_t node){
    if(node==missing)return {};const auto& n=s.nodes[node];std::string input;
    if(n.tag==WOTR_STRING)input=std::string(text(s,node));else if(n.tag==WOTR_INT)input=std::to_string(n.integer);else return {};
    std::string result;for(size_t i=0;i<input.size();++i){unsigned char c=input[i];
        if(c>='A'&&c<='Z')c=c-'A'+'a';if((c>='a'&&c<='z')||(c>='0'&&c<='9'))result.push_back(static_cast<char>(c));
        // Python lower() expands dotted capital I to ASCII i + combining dot,
        // and maps Kelvin sign to k; subsequent ASCII regex keeps those bytes.
        else if(c==0xc4&&i+1<input.size()&&static_cast<unsigned char>(input[i+1])==0xb0){result+='i';++i;}
        else if(c==0xe2&&i+2<input.size()&&static_cast<unsigned char>(input[i+1])==0x84&&static_cast<unsigned char>(input[i+2])==0xaa){result+='k';i+=2;}
    }return result;
}
int tail_target_ref(const Snapshot& s,uint64_t node){
    const auto normal=tail_normal_ref(s,node);const char* names[]={"gandalfthegrey","strider","legolas","gimli","boromir","meriadoc","peregrin","aragorn","gandalfthewhite","gollum","witchking","saruman","mouthofsauron","leader","nazgul"};
    for(int i=0;i<15;++i)if(normal==names[i])return i;
    if(normal=="merry"||normal=="meriadocbrandybuck")return 5;if(normal=="pippin"||normal=="peregrintook")return 6;if(normal=="witchkingofangmar")return 10;return -1;
}
int tail_owner_ref(const Snapshot& s,uint64_t owner){if(text_is(s,owner,"atk"))return 0;require(text_is(s,owner,"def"),"Unknown combat effect owner");return 1;}
void tail_combat_ref(const Snapshot& s,uint64_t attrs,float* out){
    auto pc=tail_get_ref(s,attrs,"pending_combat");if(!truth(s,pc))pc=missing;
    auto payload=tail_get_ref(s,pc,"_round_resume");if(!truth(s,payload))payload=missing;
    const auto cursor=tail_get_ref(s,payload,"cursor");
    if(tail_none_ref(s,cursor)){
        auto effect=tail_get_ref(s,pc,"effect_choice");if(!truth(s,effect))effect=missing;
        if(text_is(s,tail_get_ref(s,effect,"effect"),"onslaught")&&text_is(s,tail_get_ref(s,effect,"stage"),"post"))out[125]=out[127]=1;
        else if(text_is(s,tail_get_ref(s,effect,"stage"),"onslaught_payment")){
            out[126]=out[127]=1;auto payment=tail_get_ref(s,pc,"_onslaught_payment");const char* names[]={"remaining","paid","count"};
            for(int i=0;i<3;++i)out[128+i]=static_cast<float>(tail_number_ref(s,field(s,payment,names[i]))/10.0);
        }
        if(out[127])out[24+tail_owner_ref(s,tail_get_ref(s,effect,"owner"))]=1;
        return; // Python intentionally does not finite-check this legacy branch.
    }
    require(text_is(s,tail_attr_ref(s,cursor,"schema"),"combat_round_cursor_v1"),"Unknown combat continuation schema");
    const auto combat=tail_attr_ref(s,cursor,"combat");const auto stage_node=tail_attr_ref(s,cursor,"stage");auto stage=tail_integer_ref(s,stage_node,"Invalid combat continuation stage");
    if(stage==4&&truth(s,tail_attr_ref(s,cursor,"ra_remaining")))out[14]=1;
    else if(stage==1&&truth(s,tail_attr_ref(s,cursor,"pre_remaining")))out[120]=1;
    else if(stage>=14)out[15]=1;
    else{if(stage<0)stage+=14;require(stage>=0&&stage<14,"Unknown combat continuation stage");out[stage]=1;}
    auto pending=tail_attr_ref(s,cursor,"pending");if(truth(s,pending)){
        const auto effect=tail_get_ref(s,pending,"effect");if(text_is(s,effect,"pre_combat_casualty")){out[121]=1;out[122]=static_cast<float>(tail_number_ref(s,tail_attr_ref(s,cursor,"pre_remaining"))/10);out[truth(s,tail_attr_ref(s,cursor,"pre_victim_attacker"))?123:124]=1;}
        else{const char* const effects[]={"dread_and_despair","relentless_assault","mighty_attack","fateful_strike","blade_of_westernesse","black_breath","heroic_death","relentless_assault_payment"};int i=tail_index_ref(s,effect,effects);require(i>=0,"Unknown combat continuation effect");out[16+i]=1;}
        out[24+tail_owner_ref(s,tail_get_ref(s,pending,"owner"))]=1;
    }
    const char* const phases[]={"pre_combat","combat","reroll"};bool observed[3]={false,false,false},owners[2][3]={{false,false,false},{false,false,false}};
    for(auto p:tail_items_ref(s,tail_attr_ref(s,cursor,"observed_rolls"))){int i=tail_index_ref(s,p,phases);require(i>=0,"Unknown observed combat roll phase");observed[i]=true;}
    require(!observed[2]||observed[1],"Noncausal observed combat roll phases");
    for(auto p:tail_items_ref(s,tail_attr_ref(s,cursor,"observed_roll_sides"))){require(s.nodes[p].tag==WOTR_TUPLE&&s.nodes[p].count==2,"Unknown observed combat roll owner");auto a=child_edges(s,p);int phase=tail_index_ref(s,a[0],phases);int role=text_is(s,a[1],"attacker")?0:text_is(s,a[1],"defender")?1:-1;require(phase>=0&&role>=0,"Unknown observed combat roll owner");owners[role][phase]=true;}
    const char* roles[]={"attacker","defender"};const char* rolls[]={"pre_combat_rolls","combat_rolls","rerolls"};
    const char* nums[]={"combat_strength","leadership","dice_mod","reroll_mod","hits","pre_combat_hits","bonus_hits","hits_cancelled","final_hits"};
    for(int r=0;r<2;++r){
        for(int p=0;p<3;++p)if(owners[r][p]){int slot=r*3+p;out[26+slot]=1;const auto key=std::string(roles[r])+"_"+rolls[p];
            for(auto die:tail_items_ref(s,tail_attr_ref(s,combat,key.c_str()))){auto d=tail_integer_ref(s,die,"Invalid observed d6");require(d>=1&&d<=6,"Invalid observed d6");auto& bin=out[32+6*slot+d-1];bin+=0.2f;}}
        for(int j=0;j<9;++j){if((j==5&&!observed[0])||((j==4||j==6||j==7||j==8)&&!observed[1]))continue;const auto key=std::string(roles[r])+"_"+nums[j];out[68+r*10+j]=static_cast<float>(tail_number_ref(s,tail_attr_ref(s,combat,key.c_str()))/10);}
        if(observed[2])out[68+r*10+9]=static_cast<float>(tail_number_ref(s,tail_attr_ref(s,combat,r==0?"_resume_atk_reroll_hits":"_resume_def_reroll_hits"))/5);
    }
    std::vector<uint64_t> leaders;for(const char* key:{"attacker_army","defender_army"}){auto part=tail_items_ref(s,tail_attr_ref(s,tail_attr_ref(s,combat,key),"leaders"));leaders.insert(leaders.end(),part.begin(),part.end());}
    auto companions=tail_items_ref(s,tail_attr_ref(s,combat,"companions_in_battle"));leaders.insert(leaders.end(),companions.begin(),companions.end());
    unsigned paid=0;for(auto target:tail_items_ref(s,tail_attr_ref(s,cursor,"paid_leadership_targets"))){int i=tail_target_ref(s,target);if(i>=0)paid|=1u<<i;}
    const auto suppressed=tail_attr_ref(s,combat,"_wop_cancelled"),shined=tail_attr_ref(s,combat,"_shined_companion"),forfeited=tail_attr_ref(s,combat,"_leadership_forfeited");
    for(auto leader:leaders){auto name=tail_attr_ref(s,leader,"name");int i=tail_target_ref(s,name);if(i<0||i>=10)continue;
        out[88+i]=1;out[98+i]=leader==suppressed;bool forfeit=false;for(auto f:tail_items_ref(s,forfeited))if(tail_equal_ref(s,name,f))forfeit=true;
        out[108+i]=leader==shined||forfeit||(paid&(1u<<i));
    }
    auto remaining=tail_attr_ref(s,cursor,"ra_remaining");out[118]=truth(s,remaining)?static_cast<float>(tail_number_ref(s,remaining)/10):0;out[119]=static_cast<float>(tail_number_ref(s,tail_attr_ref(s,cursor,"ra_paid"))/10);
    for(size_t i=0;i<131;++i)require(std::isfinite(out[i]),"Nonfinite public combat observation");
}
void append_tails_ref(const Snapshot& s,const Snapshot& tables,int32_t side,std::vector<float>& out){
    (void)side;const auto attrs=attributes(s,s.root,"game_state.GameState");const auto keys=field(tables,tables.root,"region_keys");require(tables.nodes[keys].count==105,"Tail contract requires 105 ordered regions");
    const size_t start=out.size();out.resize(start+735,0.0f);float* tail=out.data()+start;
    const auto raw_pending=tail_get_ref(s,attrs,"pending_card_option");const auto pending=truth(s,raw_pending)?raw_pending:missing;
    tail_commitments_ref(s,tables,keys,attrs,pending,tail);
    tail_nazgul_ref(s,tables,keys,pending,tail+324);
    tail_companion_moves_ref(s,pending,tail+434);
    tail_die_ref(s,attrs,tail+444);
    tail_hunt_ref(s,attrs,raw_pending,tail+445,tail+470,tail+534,tail+601);
    tail_combat_ref(s,attrs,tail+604);
}
void encode_state_ref(const Snapshot& s,const Snapshot& tables,int32_t side,float* output,size_t count){
    require(side==0 || side==1,"invalid observation side");
    require(count==4647,"full observation output count must be 4647");
    check_array(output,count);
    require(text_is(tables,field(tables,tables.root,"schema"),"wotr_native_observation_tables_v1"),"unsupported observation table schema");
    require(text_is(tables,field(tables,tables.root,"observation_contract"),"v52_observed_combat_round_v3"),"unsupported observation contract");
    require(typed(tables,field(tables,tables.root,"feature_count"),WOTR_INT,"invalid feature count").integer==4647,"unsupported observation feature count");
    typed(tables,field(tables,tables.root,"cards"),WOTR_DICT,"invalid observation cards table");
    std::vector<std::string> regions,nations;
    auto read_names=[&](const char* key,std::vector<std::string>& names){
        for(auto node:child_edges(tables,field(tables,tables.root,key))){
            auto value=text(tables,node);
            require(!value.empty() && value.find('\0')==std::string_view::npos,"invalid observation table name");
            names.emplace_back(value);
        }
    };
    read_names("region_keys",regions);read_names("nation_names",nations);
    require(regions.size()==105 && nations.size()==8,"unsupported observation table dimensions");
    std::vector<const char*> region_keys,nation_names;
    for(const auto& key:regions)region_keys.push_back(key.c_str());
    for(const auto& name:nations)nation_names.push_back(name.c_str());
    std::vector<float> out(3182);out.reserve(count);
    encode_prefix_ref(s,side,region_keys.data(),regions.size(),nation_names.data(),nations.size(),out.data(),out.size());
    append_globals_ref(s,tables,side,out);require(out.size()==3401,"globals feature count mismatch");
    append_context_ref(s,tables,side,out);require(out.size()==3746,"context feature count mismatch");
    append_tails_ref(s,tables,side,out);append_cargo_observation(s,tables,out);require(out.size()==count,"tail feature count mismatch");
    std::memcpy(output,out.data(),count*sizeof(float));
}
void encode_state_impl(const Snapshot& s,const Snapshot& tables,int32_t side,float* output,size_t count){
    require(side==0 || side==1,"invalid observation side");
    require(count==4647,"full observation output count must be 4647");
    check_array(output,count);
    require(text_is(tables,field(tables,tables.root,"schema"),"wotr_native_observation_tables_v1"),"unsupported observation table schema");
    require(text_is(tables,field(tables,tables.root,"observation_contract"),"v52_observed_combat_round_v3"),"unsupported observation contract");
    require(typed(tables,field(tables,tables.root,"feature_count"),WOTR_INT,"invalid feature count").integer==4647,"unsupported observation feature count");
    typed(tables,field(tables,tables.root,"cards"),WOTR_DICT,"invalid observation cards table");
    std::vector<std::string> regions,nations;
    auto read_names=[&](const char* key,std::vector<std::string>& names){
        for(auto node:child_edges(tables,field(tables,tables.root,key))){
            auto value=text(tables,node);
            require(!value.empty() && value.find('\0')==std::string_view::npos,"invalid observation table name");
            names.emplace_back(value);
        }
    };
    read_names("region_keys",regions);read_names("nation_names",nations);
    require(regions.size()==105 && nations.size()==8,"unsupported observation table dimensions");
    std::vector<const char*> region_keys,nation_names;
    for(const auto& key:regions)region_keys.push_back(key.c_str());
    for(const auto& name:nations)nation_names.push_back(name.c_str());
    std::vector<float> out(3182);out.reserve(count);
    encode_prefix(s,side,region_keys.data(),regions.size(),nation_names.data(),nations.size(),out.data(),out.size());
    append_globals(s,tables,side,out);require(out.size()==3401,"globals feature count mismatch");
    append_context(s,tables,side,out);require(out.size()==3746,"context feature count mismatch");
    append_tails(s,tables,side,out);append_cargo_observation(s,tables,out);require(out.size()==count,"tail feature count mismatch");
    std::memcpy(output,out.data(),count*sizeof(float));
}
void encode_state(const Snapshot& s,const Snapshot& tables,int32_t side,float* output,size_t count){
    if(!obs_check_active){encode_state_impl(s,tables,side,output,count);return;}
    std::string fast_error;std::exception_ptr pending;
    try{encode_state_impl(s,tables,side,output,count);}catch(const std::exception& e){fast_error=e.what();if(fast_error.empty())fast_error="<empty>";pending=std::current_exception();}
    std::vector<float> ref(count==4647?count:0,0.0f);std::string ref_error;
    try{encode_state_ref(s,tables,side,ref.data(),count);}catch(const std::exception& e){ref_error=e.what();if(ref_error.empty())ref_error="<empty>";}
    if(fast_error!=ref_error)throw std::logic_error("WOTR_NATIVE_OBS_CHECK: exception mismatch: optimized='"+fast_error+"' reference='"+ref_error+"'");
    if(pending)std::rethrow_exception(pending);
    for(size_t i=0;i<count;++i){uint32_t u,v;std::memcpy(&u,output+i,4);std::memcpy(&v,ref.data()+i,4);if(u!=v)throw std::logic_error("WOTR_NATIVE_OBS_CHECK: observation feature "+std::to_string(i)+" differs: optimized "+obs_check_hex(u)+" reference "+obs_check_hex(v));}
    obs_check_observations_checked++;
}
#include "action_features.inc"
#include "menu_view.inc"
#include "lattice_context.inc"
} // namespace

extern "C" {
#include "search_turn_exports.inc"
#include "search_turn_select_random.inc"
void* wotr_snapshot_search_driver_facts(const void* state){return checked<void*>([&]()->void*{return native_search_driver_facts(snapshot(state)).release();},nullptr);}
void* wotr_snapshot_combat_search_army(const void* s,const void* a){return checked<void*>([&]()->void*{return native_combat_search_army(snapshot(s),snapshot(a)).release();},nullptr);}
void* wotr_snapshot_chance_signature_wire(const void* s,int32_t root){return checked<void*>([&]()->void*{return native_chance_signature_wire(snapshot(s),root).release();},nullptr);}
void* wotr_snapshot_metadata_wire(const void* s){return checked<void*>([&]()->void*{return native_metadata_wire(snapshot(s)).release();},nullptr);}
void* wotr_snapshot_combat_search_facts(const void* s){return checked<void*>([&]()->void*{return native_combat_search_facts(snapshot(s)).release();},nullptr);}
void* wotr_snapshot_combat_search_units(const void* s,const void* args){return checked<void*>([&]()->void*{return native_combat_search_units(snapshot(s),snapshot(args)).release();},nullptr);}
void* wotr_snapshot_overlay_assert_hand(const void* s,int32_t side){return checked<void*>([&]()->void*{return native_overlay_assert_hand(snapshot(s),side).release();},nullptr);}
void* wotr_snapshot_search_value_facts(const void* state){return checked<void*>([&]()->void*{return native_search_value_facts(snapshot(state)).release();},nullptr);}
void* wotr_snapshot_kernel_context(const void* state,const void* tables,const void* info,const char* atk,const char* def,int32_t defer){return checked<void*>([&]()->void*{require(defer==0||defer==1,"Kernel defer must be Boolean");return native_kernel_context(snapshot(state),snapshot(tables),snapshot(info),atk,def,defer!=0).release();},nullptr);}
#include "search_overlay_exports.inc"
#include "belief_world_exports.inc"
#include "combat_forcing_errors_exports.inc"
void* wotr_snapshot_search_metadata(const void* state,int32_t root){return checked<void*>([&]()->void*{return native_search_metadata(snapshot(state),root).release();},nullptr);}
uint32_t wotr_snapshot_abi_version(){return 1;}
int wotr_snapshot_prepare_observation_tables(void* handle){
    return checked<int>([&](){
        snapshot(handle);auto& s=*static_cast<Snapshot*>(handle);
        require(text_is(s,field(s,s.root,"schema"),"wotr_native_observation_tables_v1"),"preparation requires observation tables");
        require(text_is(s,field(s,s.root,"observation_contract"),"v52_observed_combat_round_v3"),"preparation requires current observation contract");
        auto prepared=std::make_shared<ContextPrepared>();
        prepared->bytes=s.bytes;prepared->dictionaries.resize(s.nodes.size());
        for(uint64_t d=0;d<s.nodes.size();++d){
            const auto& node=s.nodes[d];if(node.tag!=WOTR_DICT || node.count<16)continue;
            auto index=std::make_unique<ContextPrepared::Index>();index->reserve(node.count/2);
            for(uint64_t i=0;i<node.count;i+=2){
                const auto k=s.edges[node.offset+i];const auto& key=s.nodes[k];
                if(key.tag==WOTR_STRING){
                    const auto value=key.count ? std::string_view(reinterpret_cast<const char*>(prepared->bytes.data()+key.offset),key.count) : std::string_view();
                    (*index)[value]=s.edges[node.offset+i+1];
                }
            }
            prepared->dictionaries[d]=std::move(index);
        }
        s.context_prepared=std::move(prepared);return 0;
    },-1);
}
int wotr_snapshot_ring_waiver_active(const void* handle,int32_t side){
    return checked<int>([&](){return ring_waiver(snapshot(handle),side);},-1);
}
int wotr_snapshot_use_elven_ring(void* handle,int32_t side,int64_t die_index,const char* face,
                                 char* message,size_t capacity){
    return checked<int>([&](){
        require(face!=nullptr,"Elven Ring face is null");
        require(message!=nullptr && capacity>0,"Elven Ring message buffer is empty");
        auto changed=snapshot(handle);
        const auto result=use_ring(changed,side,die_index,face);
        require(result.size()<capacity,"Elven Ring message buffer is too small");
        auto& original=*static_cast<Snapshot*>(handle);
        original.nodes.swap(changed.nodes);original.edges.swap(changed.edges);original.bytes.swap(changed.bytes);
    original.string_index.clear();original.string_index_scanned=0;
        original.context_prepared.swap(changed.context_prepared);
        std::swap(original.root,changed.root);
        invalidate_action_clone(original);
        std::memcpy(message,result.c_str(),result.size()+1);
        return 0;
    },-1);
}
int wotr_snapshot_encode_state(const void* handle,const void* tables,int32_t side,float* output,size_t count){
    return checked<int>([&](){encode_state(snapshot(handle),snapshot(tables),side,output,count);return 0;},-1);
}
const char* wotr_snapshot_last_error(){return error_text.data();}
void* wotr_snapshot_create(const WotrValue* n,size_t nn,const uint64_t* e,size_t ne,
                          const uint8_t* b,size_t nb,uint64_t root){
    return checked<void*>([&]()->void*{return create(n,nn,e,ne,b,nb,root).release();},nullptr);
}
void* wotr_snapshot_clone(const void* handle){
    return checked<void*>([&]()->void*{return new Snapshot(snapshot(handle));},nullptr);
}
void* wotr_snapshot_action_clone(const void* handle){
    return checked<void*>([&]()->void*{return action_clone(snapshot(handle)).release();},nullptr);
}
int wotr_snapshot_action_clone_cache_info(const void* handle,uint64_t* hits,uint64_t* misses,uint64_t* bytes){
    return checked<int>([&](){
        require(hits && misses && bytes,"clone cache output pointers must be nonnull");
        const auto& s=snapshot(handle);std::lock_guard<std::mutex> lock(s.action_clone_mutex);
        const auto& cached=s.action_clone_template;
        const auto retained=cached?cached->nodes.size()*sizeof(WotrValue)+cached->edges.size()*sizeof(uint64_t)+cached->bytes.size():0;
        *hits=s.action_clone_hits;*misses=s.action_clone_misses;*bytes=retained;return 0;
    },-1);
}
int wotr_snapshot_victory(const void* handle,const void* tables,int32_t end_of_turn,int32_t search,int64_t* output,size_t count){
    return checked<int>([&](){
        require(count==5 && output,"victory output requires five int64 values");
        require((end_of_turn==0 || end_of_turn==1) && (search==0 || search==1),"victory mode flags must be boolean");
        const auto result=search?search_victory(snapshot(handle),snapshot(tables),end_of_turn!=0):check_victory(snapshot(handle),snapshot(tables));
        const int64_t values[]={result.winner,result.kind,result.fp_vp,result.sh_vp,search?result.winner>=0:terminal_should_end(result,end_of_turn!=0)};
        std::copy(values,values+5,output);return 0;
    },-1);
}
void* wotr_snapshot_political_ring_actions(const void* handle,const void* tables,int32_t side){
    return checked<void*>([&]()->void*{return political_ring_actions(snapshot(handle),snapshot(tables),side).release();},nullptr);
}
void* wotr_snapshot_muster_actions(const void* handle,const void* tables,int32_t side){
    return checked<void*>([&]()->void*{return native_muster_actions(snapshot(handle),snapshot(tables),side).release();},nullptr);
}
void* wotr_snapshot_army_move_actions(const void* handle,const void* tables,int32_t side){
    return checked<void*>([&]()->void*{return native_army_move_actions(snapshot(handle),snapshot(tables),side).release();},nullptr);
}
void* wotr_snapshot_companion_political_actions(const void* handle,const void* tables,int32_t side){
    return checked<void*>([&]()->void*{return native_companion_political_actions(snapshot(handle),snapshot(tables),side).release();},nullptr);
}
void* wotr_snapshot_apply_companion_political_action(const void* handle,const void* tables,int32_t side,int64_t index,const char* nation,const char* name){
    return checked<void*>([&]()->void*{
        require(nation && name,"companion political action needs nation and name");
        return apply_companion_political_action(snapshot(handle),snapshot(tables),side,index,nation,name).release();
    },nullptr);
}
void* wotr_snapshot_apply_descriptor(const void* handle,const void* tables,const void* menu,int32_t side,int64_t index){
    return checked<void*>([&]()->void*{return apply_native_descriptor(snapshot(handle),snapshot(tables),snapshot(menu),side,index).release();},nullptr);
}
void* wotr_snapshot_project_policy(const void* handle,const void* menu,int32_t side,void** metadata){
    return checked<void*>([&]()->void*{
        require(metadata,"policy projection needs metadata output");
        auto result=project_native_ai_actions(snapshot(handle),snapshot(menu),side);
        am_Builder b;std::vector<uint64_t> reasons;for(const auto& reason:result.reasons)reasons.push_back(b.str(reason));
        const auto forced=result.forced_reason.empty()?rr_scalar(b.out,WOTR_NONE):b.str(result.forced_reason);
        const auto reason_tuple=b.container(WOTR_TUPLE,reasons);b.out.root=b.container(WOTR_TUPLE,{forced,reason_tuple});
        auto meta=std::make_unique<Snapshot>(b.out);*metadata=meta.release();return result.actions.release();
    },nullptr);
}
int wotr_snapshot_next_actor(const void* handle,int32_t side){
    return checked<int>([&](){return native_next_actor(snapshot(handle),side);},-2);
}
void* wotr_snapshot_full_actions(const void* handle,const void* tables,int32_t side){
    return checked<void*>([&]()->void*{return native_full_actions(snapshot(handle),snapshot(tables),side).release();},nullptr);
}
void* wotr_snapshot_hunt_params(const void* handle,const void* tables){
    return checked<void*>([&]()->void*{return native_hunt_params(snapshot(handle),snapshot(tables)).release();},nullptr);
}
void* wotr_snapshot_hunt_roll(const void* handle,const void* tables,const void* random,int64_t count,int64_t threshold,int64_t modifier,
    void** random_out,int64_t* successes,void** events){
    return checked<void*>([&]()->void*{
        require(random_out && successes && events && random_out!=events,"Hunt needs distinct outputs");
        auto result=native_hunt_roll(snapshot(handle),snapshot(tables),random,count,threshold,modifier);
        *random_out=result.random.release();*successes=result.successes;*events=result.events.release();return result.state.release();
    },nullptr);
}
void* wotr_snapshot_combat_pre_stage(const void* handle,const void* random,int64_t selection,int32_t battle,void** random_out){
    return checked<void*>([&]()->void*{
        require(random_out && (battle==0 || battle==1),"pre-combat needs RNG output and Boolean battle context");
        auto result=combat_pre_stage(snapshot(handle),random,selection,battle!=0);
        *random_out=result.random.release();return result.state.release();
    },nullptr);
}
void* wotr_snapshot_combat_round_stage(const void* handle,const void* random,int64_t selection,int32_t battle,void** random_out){
    return checked<void*>([&]()->void*{
        require(random_out && (battle==0 || battle==1),"round needs RNG output and Boolean battle context");
        auto result=combat_round_stage(snapshot(handle),random,selection,battle!=0);
        *random_out=result.random.release();return result.state.release();
    },nullptr);
}
void* wotr_snapshot_combat_publish_paused(const void* handle,const void* random,void** random_out){
    return checked<void*>([&]()->void*{
        require(random_out,"publication needs RNG output");auto result=combat_publish_paused(snapshot(handle),random);
        *random_out=result.random.release();return result.state.release();
    },nullptr);
}
void* wotr_snapshot_apply_fellowship_move(const void* handle,const void* tables,const void* random,int64_t index,void** random_out,void** events){
    return checked<void*>([&]()->void*{
        require(random_out && events && random_out!=events,"Fellowship move needs distinct outputs");
        auto result=apply_fellowship_move(snapshot(handle),snapshot(tables),random,index);
        *random_out=result.random.release();*events=result.events.release();return result.state.release();
    },nullptr);
}
void* wotr_snapshot_mouth_move_actions(const void* handle,const void* tables,int32_t side){
    return checked<void*>([&]()->void*{return native_mouth_move_actions(snapshot(handle),snapshot(tables),side).release();},nullptr);
}
void* wotr_snapshot_apply_mouth_move(const void* handle,const void* tables,int32_t side,int64_t index,const char* from,const char* to){
    return checked<void*>([&]()->void*{
        require(from&&to,"Mouth move needs regions");return apply_mouth_move_action(snapshot(handle),snapshot(tables),side,index,from,to).release();
    },nullptr);
}
void* wotr_snapshot_combat_casualty_menu(const void* handle,int32_t side,const char* region,int64_t hits){
    return checked<void*>([&]()->void*{return combat_casualty_menu(snapshot(handle),side,region,hits).release();},nullptr);
}
void* wotr_snapshot_combat_casualty_operation(const void* handle,const void* tables,int32_t side,const char* region,int64_t hits,const char* nation,const char* operation){
    return checked<void*>([&]()->void*{return combat_casualty_operation(snapshot(handle),snapshot(tables),side,region,hits,nation,operation).release();},nullptr);
}
void* wotr_snapshot_fellowship_actions(const void* handle,int64_t index){
    return checked<void*>([&]()->void*{return native_fellowship_actions(snapshot(handle),index).release();},nullptr);
}
void* wotr_snapshot_apply_fellowship_hide(const void* handle,const void* tables,int64_t index,int32_t strider){
    return checked<void*>([&]()->void*{
        require(strider==0||strider==1,"Strider flag must be Boolean");return apply_fellowship_hide(snapshot(handle),snapshot(tables),index,strider!=0).release();
    },nullptr);
}
void* wotr_snapshot_minion_muster_actions(const void* handle,const void* tables,int32_t side){
    return checked<void*>([&]()->void*{return native_minion_muster_actions(snapshot(handle),snapshot(tables),side).release();},nullptr);
}
void* wotr_snapshot_apply_minion_muster(const void* handle,const void* tables,int32_t side,int64_t index,const char* name,const char* region){
    return checked<void*>([&]()->void*{require(name&&region,"Minion name and region required");return apply_minion_muster_action(snapshot(handle),snapshot(tables),side,index,name,region).release();},nullptr);
}
void* wotr_snapshot_companion_actions(const void* handle,const void* tables,int64_t index,int32_t mode){
    return checked<void*>([&]()->void*{
        if(mode==0)return native_companion_separation_actions(snapshot(handle),snapshot(tables),index).release();
        if(mode==1)return native_companion_move_actions(snapshot(handle),snapshot(tables),index).release();
        require(mode==2,"Unknown companion menu");return native_companion_chain_actions(snapshot(handle)).release();
    },nullptr);
}
void* wotr_snapshot_apply_companion(const void* handle,const void* tables,int64_t index,int32_t mode,const char* from,const char* to,const char* const* names,size_t count){
    return checked<void*>([&]()->void*{
        require(from&&to,"Companion regions required");check_array(names,count);require(count<=1048576,"Companion selection exceeds native bound");
        std::vector<std::string> selected;for(size_t i=0;i<count;++i){require(names[i],"Companion name required");selected.emplace_back(names[i]);}
        if(mode==0)return apply_companion_separation(snapshot(handle),snapshot(tables),index,selected,to).release();
        if(mode==1)return apply_companion_move(snapshot(handle),snapshot(tables),index,from,to,selected).release();
        require(mode==2,"Unknown companion action");return apply_companion_move_option(snapshot(handle),snapshot(tables),index).release();
    },nullptr);
}
void* wotr_snapshot_combat_deferred_post(const void* handle,const void* tables,const void* random,void** random_out,void** info){
    return checked<void*>([&]()->void*{
        require(random_out&&info&&random_out!=info,"Deferred combat needs distinct outputs");
        auto result=combat_deferred_post(snapshot(handle),snapshot(tables),random);am_Builder b;
        b.out.root=b.container(WOTR_DICT,{b.str("attacker_destroyed"),rr_scalar(b.out,WOTR_BOOL,result.attacker_destroyed),b.str("defender_destroyed"),rr_scalar(b.out,WOTR_BOOL,result.defender_destroyed),b.str("deferred_effect"),result.role<0?rr_scalar(b.out,WOTR_NONE):b.str(result.role==0?"attacker":"defender"),b.str("post_hits_on_attacker"),b.integer(result.hits_on_attacker),b.str("post_hits_on_defender"),b.integer(result.hits_on_defender)});
        auto meta=std::make_unique<Snapshot>(b.out);*info=meta.release();*random_out=result.owned.random.release();return result.owned.state.release();
    },nullptr);
}
void* wotr_snapshot_combat_allocation_settle(const void* handle,const void* tables,const void* random,int32_t enter,void** random_out){
    return checked<void*>([&]()->void*{
        require(random_out&&(enter==0||enter==1),"Allocation settle requires RNG output and Boolean mode");
        auto result=combat_allocation_settle(snapshot(handle),snapshot(tables),random,enter!=0);
        *random_out=result.random.release();return result.state.release();
    },nullptr);
}
void* wotr_snapshot_cleanup_sieges(const void* handle,const void* tables){
    return checked<void*>([&]()->void*{auto child=std::make_unique<Snapshot>(snapshot(handle));su_cleanup(*child,snapshot(tables));return child.release();},nullptr);
}
void* wotr_snapshot_safe_next_turn(const void* handle,const void* tables,int32_t* won,void** annotations){
    return checked<void*>([&]()->void*{
        require(won&&annotations,"Search turn needs outputs");auto child=std::make_unique<Snapshot>(snapshot(handle));am_Builder b;std::vector<uint64_t> rows;
        const bool victory=st_safe_next_turn(*child,snapshot(tables),b,rows);b.out.root=b.container(WOTR_LIST,rows);auto events=std::make_unique<Snapshot>(b.out);
        *won=victory;*annotations=events.release();return child.release();
    },nullptr);
}
int wotr_snapshot_apply_descriptor_hunt_plan(const void* handle,const void* tables,const void* menu,const void* random,const void* chance,const int64_t* plan,size_t count,int32_t side,int64_t index,int32_t resume,void** child_out,void** random_out,void** events_out,void** request_out,double* probability){
    return checked<int>([&]()->int{
        require(child_out&&random_out&&events_out&&request_out&&probability,"Hunt plan outputs are required");
        require(child_out!=random_out&&child_out!=events_out&&child_out!=request_out&&random_out!=events_out&&random_out!=request_out&&events_out!=request_out,"Hunt plan outputs must be distinct");
        require(resume==0||resume==1,"Hunt preaction flag must be Boolean");
        hf_Scope scope(snapshot(handle),snapshot(chance),plan,count);
        try{
            auto result=apply_native_random_descriptor(snapshot(handle),snapshot(tables),snapshot(menu),random,side,index);
            if(resume){
                const auto& actions=snapshot(menu);const auto list=child_edges(actions,actions.root);const auto row=list.at(static_cast<size_t>(index));const auto kind=text(actions,actions.edges[actions.nodes[row].offset+1]);
                am_Builder events;auto rows=fm_import(events.out,*result.events);
                auto settled=sr_resume_preaction(snapshot(handle),*result.state,snapshot(tables),result.random.get(),kind,&events,&rows);
                result.state=std::move(settled.state);result.random=std::move(settled.random);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));
            }
            require(scope.next==count,"Hunt plan not fully consumed");require(cf_same_rng(random,result.random.get()),"raw RNG state mutation in driven Hunt walk");
            *probability=scope.probability;*child_out=result.state.release();*random_out=result.random.release();*events_out=result.events.release();*request_out=nullptr;return 0;
        }catch(const hf_Fork&){auto packet=scope.packet();*probability=scope.probability;*request_out=packet.release();*child_out=nullptr;*random_out=nullptr;*events_out=nullptr;return 1;}
    },-1);
}
void* wotr_snapshot_apply_descriptor_forced_combat(const void* handle,const void* tables,const void* menu,const void* random,int32_t side,int64_t index,const int64_t* plan,size_t count,int32_t confusion,void** random_out,void** events){
    return checked<void*>([&]()->void*{
        require(random_out&&events&&random_out!=events,"Forced action needs distinct outputs");
        require(count<=256&&(confusion==0||confusion==1),"Invalid forced combat plan domain");
        cf_Scope scope(plan,count,confusion!=0);
        auto result=apply_native_random_descriptor(snapshot(handle),snapshot(tables),snapshot(menu),random,side,index);
        scope.done();require(cf_same_rng(random,result.random.get()),"Non-batch RNG consumed during forced combat action");
        *random_out=result.random.release();*events=result.events.release();return result.state.release();
    },nullptr);
}
void* wotr_snapshot_apply_descriptor_random(const void* handle,const void* tables,const void* menu,const void* random,int32_t side,int64_t index,void** random_out,void** events){
    return checked<void*>([&]()->void*{
        require(random_out&&events&&random_out!=events,"Random action needs distinct outputs");
        auto result=apply_native_random_descriptor(snapshot(handle),snapshot(tables),snapshot(menu),random,side,index);
        *random_out=result.random.release();*events=result.events.release();return result.state.release();
    },nullptr);
}
void* wotr_snapshot_combat_resume_settle(const void* handle,const void* tables,const void* random,int64_t selection,void** random_out){
    return checked<void*>([&]()->void*{
        require(random_out,"Combat resume needs RNG output");auto result=combat_resume_settle(snapshot(handle),snapshot(tables),random,selection);
        *random_out=result.random.release();return result.state.release();
    },nullptr);
}
void* wotr_snapshot_resume_preaction(const void* before,const void* handle,const void* tables,const void* random,const char* kind,void** random_out){
    return checked<void*>([&]()->void*{
        require(kind&&random_out,"Preaction resume needs action kind and RNG output");auto result=sr_resume_preaction(snapshot(before),snapshot(handle),snapshot(tables),random,kind);
        *random_out=result.random.release();return result.state.release();
    },nullptr);
}
void* wotr_snapshot_resume_preaction_with_events(const void* before,const void* handle,const void* tables,const void* random,const char* kind,void** random_out,void** events_out){
    return checked<void*>([&]()->void*{
        require(kind&&random_out&&events_out&&random_out!=events_out,"Preaction resume needs distinct outputs");am_Builder events;std::vector<uint64_t> rows;
        auto result=sr_resume_preaction(snapshot(before),snapshot(handle),snapshot(tables),random,kind,&events,&rows);events.out.root=events.container(WOTR_LIST,rows);
        auto payload=std::make_unique<Snapshot>(std::move(events.out));*random_out=result.random.release();*events_out=payload.release();return result.state.release();
    },nullptr);
}
void* wotr_snapshot_hunt_protection_actions(const void* handle){
    return checked<void*>([&]()->void*{return native_hunt_protection_actions(snapshot(handle)).release();},nullptr);
}
void* wotr_snapshot_lure_choice_actions(const void* handle){
    return checked<void*>([&]()->void*{return native_lure_choice_actions(snapshot(handle)).release();},nullptr);
}
void* wotr_snapshot_apply_lure_event(const void* handle,const void* tables,const void* random,int32_t side,int64_t index,const char* id,void** random_out,void** events){
    return checked<void*>([&]()->void*{require(id&&random_out&&events&&random_out!=events,"Lure event requires distinct outputs");auto result=apply_lure_event_action(snapshot(handle),snapshot(tables),random,side,index,id);*random_out=result.random.release();*events=result.events.release();return result.state.release();},nullptr);
}
void* wotr_snapshot_apply_lure_event_choice(const void* handle,const void* tables,const void* random,int64_t index,void** random_out,void** events){
    return checked<void*>([&]()->void*{require(random_out&&events&&random_out!=events,"Lure choice requires distinct outputs");auto result=apply_lure_event_choice(snapshot(handle),snapshot(tables),random,index);*random_out=result.random.release();*events=result.events.release();return result.state.release();},nullptr);
}
void* wotr_snapshot_apply_hunt_protection(const void* handle,const void* tables,const void* random,int64_t index,void** random_out,void** events){
    return checked<void*>([&]()->void*{
        require(random_out&&events&&random_out!=events,"Hunt protection needs distinct outputs");auto result=apply_hunt_protection(snapshot(handle),snapshot(tables),random,index);
        *random_out=result.random.release();*events=result.events.release();return result.state.release();
    },nullptr);
}
void* wotr_snapshot_play_event_menu(const void* handle,const void* tables,int32_t side,int64_t index,const char* category){
    return checked<void*>([&]()->void*{require(category,"Event category required");return native_play_event_for_die(snapshot(handle),snapshot(tables),side,index,category).release();},nullptr);
}
void* wotr_snapshot_apply_simple_event(const void* handle,const void* tables,int32_t side,int64_t index,const char* id){
    return checked<void*>([&]()->void*{require(id,"Event card ID required");return apply_simple_event_action(snapshot(handle),snapshot(tables),side,index,id).release();},nullptr);
}
void* wotr_snapshot_gandalf_draw_actions(const void* handle,int32_t side){
    return checked<void*>([&]()->void*{return native_gandalf_draw_actions(snapshot(handle),side).release();},nullptr);
}
void* wotr_snapshot_apply_gandalf_draw(const void* handle,const void* tables,int32_t side,int32_t draw){
    return checked<void*>([&]()->void*{require(draw==0||draw==1,"Gandalf draw flag must be Boolean");return apply_gandalf_draw_action(snapshot(handle),snapshot(tables),side,draw!=0).release();},nullptr);
}
void* wotr_snapshot_apply_combat_round_choice(const void* handle,const void* tables,const void* random,int64_t index,void** random_out,void** events){
    return checked<void*>([&]()->void*{
        require(random_out&&events&&random_out!=events,"Combat action needs distinct outputs");auto result=apply_combat_round_choice(snapshot(handle),snapshot(tables),random,index);
        *random_out=result.random.release();*events=result.events.release();return result.state.release();
    },nullptr);
}
void* wotr_snapshot_apply_combat_casualty(const void* handle,const void* tables,const void* random,int64_t index,void** random_out,void** events){
    return checked<void*>([&]()->void*{
        require(random_out&&events&&random_out!=events,"Combat action needs distinct outputs");auto result=apply_combat_casualty(snapshot(handle),snapshot(tables),random,index);
        *random_out=result.random.release();*events=result.events.release();return result.state.release();
    },nullptr);
}
void* wotr_snapshot_apply_combat_onslaught(const void* handle,const void* tables,const void* random,int64_t count,void** random_out,void** events){
    return checked<void*>([&]()->void*{
        require(random_out&&events&&random_out!=events,"Combat action needs distinct outputs");auto result=apply_combat_onslaught(snapshot(handle),snapshot(tables),random,count);
        *random_out=result.random.release();*events=result.events.release();return result.state.release();
    },nullptr);
}
void* wotr_snapshot_combat_setup(const void* handle,const void* tables,const char* attacker,const char* defender,const char* attacker_card,const char* defender_card,int32_t siege,int32_t side){
    return checked<void*>([&]()->void*{
        require(attacker&&defender&&attacker_card&&defender_card&&(siege==0||siege==1)&&side>=-1&&side<=1,"Invalid combat setup arguments");
        return combat_setup(snapshot(handle),snapshot(tables),attacker,defender,attacker_card,defender_card,siege!=0,side).release();
    },nullptr);
}
void* wotr_snapshot_combat_round_begin(const void* handle,const void* tables,const void* random,int64_t mode,void** random_out,void** events){
    return checked<void*>([&]()->void*{require(random_out&&events&&random_out!=events&&mode>=0&&mode<=2,"Invalid combat beginning arguments");auto result=combat_round_begin(snapshot(handle),snapshot(tables),random,static_cast<int>(mode)-1);*random_out=result.random.release();*events=result.events.release();return result.state.release();},nullptr);
}
void* wotr_snapshot_army_attack_menu(const void* handle,const void* tables,int32_t side,int64_t index,int32_t leader){
    return checked<void*>([&]()->void*{require(leader==0||leader==1,"Leader requirement must be Boolean");return native_army_attack_for_die(snapshot(handle),snapshot(tables),side,index,leader!=0).release();},nullptr);
}
void* wotr_snapshot_apply_army_attack(const void* handle,const void* tables,const void* random,int32_t side,int64_t index,const char* from,const char* to,void** random_out,void** events){
    return checked<void*>([&]()->void*{require(from&&to&&random_out&&events&&random_out!=events,"Invalid army attack arguments");auto result=apply_army_attack(snapshot(handle),snapshot(tables),random,side,index,from,to);*random_out=result.random.release();*events=result.events.release();return result.state.release();},nullptr);
}
void* wotr_snapshot_nazgul_move_actions(const void* handle,const void* tables,int32_t side){
    return checked<void*>([&]()->void*{return native_nazgul_move_actions(snapshot(handle),snapshot(tables),side).release();},nullptr);
}
void* wotr_snapshot_nazgul_chain_actions(const void* handle,int32_t side){
    return checked<void*>([&]()->void*{return native_nazgul_chain_actions(snapshot(handle),side).release();},nullptr);
}
void* wotr_snapshot_apply_nazgul_move(const void* handle,const void* tables,int32_t side,int64_t index,const char* from,const char* to,int64_t count,int32_t witch){
    return checked<void*>([&]()->void*{require(from&&to&&(witch==0||witch==1),"Invalid Nazgul move arguments");return apply_nazgul_move_action(snapshot(handle),snapshot(tables),side,index,from,to,count,witch!=0).release();},nullptr);
}
void* wotr_snapshot_apply_nazgul_chain(const void* handle,const void* tables,int32_t side,int64_t index){
    return checked<void*>([&]()->void*{return apply_nazgul_chain_action(snapshot(handle),snapshot(tables),side,index).release();},nullptr);
}
void* wotr_snapshot_will_special_actions(const void* handle,const void* tables,int32_t side){
    return checked<void*>([&]()->void*{return native_will_special_actions(snapshot(handle),snapshot(tables),side).release();},nullptr);
}
void* wotr_snapshot_apply_will_special(const void* handle,const void* tables,int32_t side,int64_t index,int32_t gandalf,const char* region){
    return checked<void*>([&]()->void*{require(region&&(gandalf==0||gandalf==1),"Invalid promotion arguments");return apply_will_special_action(snapshot(handle),snapshot(tables),side,index,gandalf!=0,region).release();},nullptr);
}
void* wotr_snapshot_apply_combat_card(const void* handle,const void* tables,const void* random,const char* id,int32_t from_table,void** random_out,void** events){
    return checked<void*>([&]()->void*{require(random_out&&events&&random_out!=events&&(from_table==0||from_table==1),"Invalid combat card arguments");auto result=apply_combat_card(snapshot(handle),snapshot(tables),random,id,from_table!=0);*random_out=result.random.release();*events=result.events.release();return result.state.release();},nullptr);
}
void* wotr_snapshot_apply_combat_shine(const void* handle,const void* tables,const void* random,int64_t value,void** random_out,void** events){
    return checked<void*>([&]()->void*{require(random_out&&events&&random_out!=events,"Combat Shine needs distinct outputs");auto result=apply_combat_shine(snapshot(handle),snapshot(tables),random,value);*random_out=result.random.release();*events=result.events.release();return result.state.release();},nullptr);
}
void* wotr_snapshot_apply_combat_pre_target(const void* handle,const void* tables,const void* random,int64_t index,void** random_out,void** events){
    return checked<void*>([&]()->void*{require(random_out&&events&&random_out!=events,"Combat target needs distinct outputs");auto result=apply_combat_pre_target(snapshot(handle),snapshot(tables),random,index);*random_out=result.random.release();*events=result.events.release();return result.state.release();},nullptr);
}
void* wotr_snapshot_combat_entry_actions(const void* handle,const void* tables,int32_t side){
    return checked<void*>([&]()->void*{return native_combat_entry_actions(snapshot(handle),snapshot(tables),side).release();},nullptr);
}
void* wotr_snapshot_apply_combat_retreat(const void* handle,const void* tables,const void* random,const char* region,void** random_out,void** events){
    return checked<void*>([&]()->void*{require(region&&random_out&&events&&random_out!=events,"Native action requires distinct outputs");auto result=apply_combat_retreat(snapshot(handle),snapshot(tables),random,region);*random_out=result.random.release();*events=result.events.release();return result.state.release();},nullptr);
}
void* wotr_snapshot_apply_hunt_table_event(const void* handle,const void* tables,const void* random,int32_t side,int64_t index,const char* id,void** random_out,void** events){
    return checked<void*>([&]()->void*{require(id&&random_out&&events&&random_out!=events,"Native action requires distinct outputs");auto result=apply_hunt_table_event(snapshot(handle),snapshot(tables),random,side,index,id);*random_out=result.random.release();*events=result.events.release();return result.state.release();},nullptr);
}
void* wotr_snapshot_apply_token(const void* handle,const void* tables,const void* random,int32_t side,const char* kind,const char* value,void** random_out,void** events){
    return checked<void*>([&]()->void*{require(kind&&value&&random_out&&events&&random_out!=events,"Native action requires distinct outputs");auto result=apply_token_action(snapshot(handle),snapshot(tables),random,side,kind,value);*random_out=result.random.release();*events=result.events.release();return result.state.release();},nullptr);
}
void* wotr_snapshot_apply_dwarven_ring(const void* handle,const void* tables,const void* random,int32_t side,int64_t index,void** random_out,void** events){
    return checked<void*>([&]()->void*{require(random_out&&events&&random_out!=events,"Native action requires distinct outputs");auto result=apply_dwarven_ring_action(snapshot(handle),snapshot(tables),random,side,index);*random_out=result.random.release();*events=result.events.release();return result.state.release();},nullptr);
}
void* wotr_snapshot_token_actions(const void* handle,const void* tables,int32_t side){
    return checked<void*>([&]()->void*{return native_token_actions(snapshot(handle),snapshot(tables),side).release();},nullptr);
}
void* wotr_snapshot_dwarven_ring_actions(const void* handle,int32_t side){
    return checked<void*>([&]()->void*{return native_dwarven_ring_actions(snapshot(handle),side).release();},nullptr);
}
void* wotr_snapshot_apply_recruit_event(const void* handle,const void* tables,int32_t side,int64_t index,const char* id){
    return checked<void*>([&]()->void*{require(id,"Card ID required");return apply_recruit_event_action(snapshot(handle),snapshot(tables),side,index,id).release();},nullptr);
}
void* wotr_snapshot_apply_recruit_event_traced(const void* handle,const void* tables,const void* random,int32_t side,int64_t index,const char* id,void** random_out,void** events){
    return checked<void*>([&]()->void*{require(id&&random_out&&events&&random_out!=events,"Recruit event requires distinct outputs");auto result=apply_recruit_event_traced(snapshot(handle),snapshot(tables),random,side,index,id);*random_out=result.random.release();*events=result.events.release();return result.state.release();},nullptr);
}
void* wotr_snapshot_recruit_event_choice_actions(const void* handle,const void* tables,int32_t side){
    return checked<void*>([&]()->void*{return native_recruit_event_choice_actions(snapshot(handle),snapshot(tables),side).release();},nullptr);
}
void* wotr_snapshot_apply_recruit_event_choice(const void* handle,const void* tables,int32_t side,int64_t index){
    return checked<void*>([&]()->void*{return apply_recruit_event_choice(snapshot(handle),snapshot(tables),side,index).release();},nullptr);
}
void* wotr_snapshot_apply_hill_trolls_event(const void* handle,const void* tables,int32_t side,int64_t index,const char* id){
    return checked<void*>([&]()->void*{require(id,"Card ID required");return apply_hill_trolls_event(snapshot(handle),snapshot(tables),side,index,id).release();},nullptr);
}
void* wotr_snapshot_hill_trolls_choice_actions(const void* handle,const void* tables,int32_t side){
    return checked<void*>([&]()->void*{return native_hill_trolls_choice_actions(snapshot(handle),snapshot(tables),side).release();},nullptr);
}
void* wotr_snapshot_apply_hill_trolls_choice(const void* handle,const void* tables,int32_t side,int64_t index){
    return checked<void*>([&]()->void*{return apply_hill_trolls_choice(snapshot(handle),snapshot(tables),side,index).release();},nullptr);
}
void* wotr_snapshot_apply_target_recruit_event(const void* handle,const void* tables,int32_t side,int64_t index,const char* id){
    return checked<void*>([&]()->void*{require(id,"Card ID required");return apply_target_recruit_event(snapshot(handle),snapshot(tables),side,index,id).release();},nullptr);
}
void* wotr_snapshot_apply_shadow_recruit_event(const void* handle,const void* tables,int32_t side,int64_t index,const char* id){
    return checked<void*>([&]()->void*{require(id,"Card ID required");return apply_shadow_recruit_event(snapshot(handle),snapshot(tables),side,index,id).release();},nullptr);
}
void* wotr_snapshot_apply_shadows_gather_event(const void* handle,const void* tables,int32_t side,int64_t index,const char* id){
    return checked<void*>([&]()->void*{require(id,"Card ID required");return apply_shadows_gather_event(snapshot(handle),snapshot(tables),side,index,id).release();},nullptr);
}
void* wotr_snapshot_apply_target_recruit_choice(const void* handle,const void* tables,int32_t side,int64_t index){
    return checked<void*>([&]()->void*{return apply_target_recruit_choice(snapshot(handle),snapshot(tables),side,index).release();},nullptr);
}
void* wotr_snapshot_apply_shadows_gather_choice(const void* handle,const void* tables,int32_t side,int64_t index){
    return checked<void*>([&]()->void*{return apply_shadows_gather_choice(snapshot(handle),snapshot(tables),side,index).release();},nullptr);
}
void* wotr_snapshot_apply_crebain_choice(const void* handle,const void* tables,const void* random,int32_t side,int64_t index,void** random_out,void** events){
    return checked<void*>([&]()->void*{require(random_out&&events&&random_out!=events,"Event choice requires distinct outputs");auto result=apply_crebain_choice(snapshot(handle),snapshot(tables),random,side,index);*random_out=result.random.release();*events=result.events.release();return result.state.release();},nullptr);
}
void* wotr_snapshot_apply_companion_event_choice(const void* handle,const void* tables,const void* random,int32_t side,int64_t index,void** random_out,void** events){
    return checked<void*>([&]()->void*{require(random_out&&events&&random_out!=events,"Event choice requires distinct outputs");auto result=apply_companion_event_choice(snapshot(handle),snapshot(tables),random,side,index);*random_out=result.random.release();*events=result.events.release();return result.state.release();},nullptr);
}
void* wotr_snapshot_target_recruit_choice_actions(const void* handle,const void* tables,int32_t side){
    return checked<void*>([&]()->void*{return native_target_recruit_choice_actions(snapshot(handle),snapshot(tables),side).release();},nullptr);
}
void* wotr_snapshot_shadows_gather_choice_actions(const void* handle,const void* tables,int32_t side){
    return checked<void*>([&]()->void*{return native_shadows_gather_choice_actions(snapshot(handle),snapshot(tables),side).release();},nullptr);
}
void* wotr_snapshot_crebain_choice_actions(const void* handle,int32_t side){
    return checked<void*>([&]()->void*{return native_crebain_choice_actions(snapshot(handle),side).release();},nullptr);
}
void* wotr_snapshot_companion_event_choice_actions(const void* handle,int32_t side){
    return checked<void*>([&]()->void*{return native_companion_event_choice_actions(snapshot(handle),side).release();},nullptr);
}
void* wotr_snapshot_apply_companion_event(const void* handle,const void* tables,const void* random,int32_t side,int64_t index,const char* id,void** random_out,void** events){
    return checked<void*>([&]()->void*{require(id&&random_out&&events&&random_out!=events,"Companion event requires distinct outputs");auto result=apply_companion_event(snapshot(handle),snapshot(tables),random,side,index,id);*random_out=result.random.release();*events=result.events.release();return result.state.release();},nullptr);
}
void* wotr_snapshot_apply_shadow_lengthens_event(const void* handle,const void* tables,const void* random,int32_t side,int64_t index,const char* id,void** random_out,void** events){
    return checked<void*>([&]()->void*{require(id&&random_out&&events&&random_out!=events,"Event requires distinct outputs");auto result=apply_shadow_lengthens_event(snapshot(handle),snapshot(tables),random,side,index,id);*random_out=result.random.release();*events=result.events.release();return result.state.release();},nullptr);
}
void* wotr_snapshot_apply_nazgul_hunt_event(const void* handle,const void* tables,const void* random,int32_t side,int64_t index,const char* id,void** random_out,void** events){
    return checked<void*>([&]()->void*{require(id&&random_out&&events&&random_out!=events,"Event requires distinct outputs");auto result=apply_nazgul_hunt_event(snapshot(handle),snapshot(tables),random,side,index,id);*random_out=result.random.release();*events=result.events.release();return result.state.release();},nullptr);
}
void* wotr_snapshot_apply_shadow_lengthens_choice(const void* handle,const void* tables,const void* random,int32_t side,int64_t index,void** random_out,void** events){
    return checked<void*>([&]()->void*{require(random_out&&events&&random_out!=events,"Choice requires distinct outputs");auto result=apply_shadow_lengthens_choice(snapshot(handle),snapshot(tables),random,side,index);*random_out=result.random.release();*events=result.events.release();return result.state.release();},nullptr);
}
void* wotr_snapshot_apply_nazgul_hunt_choice(const void* handle,const void* tables,const void* random,int32_t side,int64_t index,void** random_out,void** events){
    return checked<void*>([&]()->void*{require(random_out&&events&&random_out!=events,"Choice requires distinct outputs");auto result=apply_nazgul_hunt_choice(snapshot(handle),snapshot(tables),random,side,index);*random_out=result.random.release();*events=result.events.release();return result.state.release();},nullptr);
}
void* wotr_snapshot_shadow_lengthens_choice_actions(const void* handle,int32_t side){return checked<void*>([&]()->void*{return native_shadow_lengthens_choice_actions(snapshot(handle),side).release();},nullptr);}
void* wotr_snapshot_palantir_draw_actions(const void* handle,int32_t side){return checked<void*>([&]()->void*{return native_palantir_draw_actions(snapshot(handle),side).release();},nullptr);}
void* wotr_snapshot_apply_palantir_draw(const void* handle,const void* tables,const void* random,int32_t side,int64_t index,void** random_out,void** events){
    return checked<void*>([&]()->void*{require(random_out&&events&&random_out!=events,"Palantir choice requires distinct outputs");auto result=apply_palantir_draw(snapshot(handle),snapshot(tables),random,side,index);*random_out=result.random.release();*events=result.events.release();return result.state.release();},nullptr);
}
void* wotr_snapshot_nazgul_hunt_choice_actions(const void* handle,int32_t side){return checked<void*>([&]()->void*{return native_nazgul_hunt_choice_actions(snapshot(handle),side).release();},nullptr);}
void* wotr_snapshot_apply_faramir_event(const void* handle,const void* tables,const void* random,int32_t side,int64_t index,const char* id,void** random_out,void** events){
    return checked<void*>([&]()->void*{require(id&&random_out&&events&&random_out!=events,"Faramir requires distinct outputs");auto result=apply_faramir_event(snapshot(handle),snapshot(tables),random,side,index,id);*random_out=result.random.release();*events=result.events.release();return result.state.release();},nullptr);
}
void* wotr_snapshot_apply_faramir_choice(const void* handle,const void* tables,const void* random,int32_t side,int64_t index,void** random_out,void** events){
    return checked<void*>([&]()->void*{require(random_out&&events&&random_out!=events,"Faramir choice requires distinct outputs");auto result=apply_faramir_choice(snapshot(handle),snapshot(tables),random,side,index);*random_out=result.random.release();*events=result.events.release();return result.state.release();},nullptr);
}
void* wotr_snapshot_apply_card_casualty_choice(const void* handle,const void* tables,const void* random,int32_t side,int64_t index,void** random_out,void** events){
    return checked<void*>([&]()->void*{require(random_out&&events&&random_out!=events,"Card casualty requires distinct outputs");auto result=apply_card_casualty_choice(snapshot(handle),snapshot(tables),random,side,index);*random_out=result.random.release();*events=result.events.release();return result.state.release();},nullptr);
}
void* wotr_snapshot_faramir_choice_actions(const void* handle,const void* tables,int32_t side){return checked<void*>([&]()->void*{return native_faramir_choice_actions(snapshot(handle),snapshot(tables),side).release();},nullptr);}
void* wotr_snapshot_card_casualty_actions(const void* handle,const void* tables,int32_t side){return checked<void*>([&]()->void*{return native_card_casualty_actions(snapshot(handle),snapshot(tables),side).release();},nullptr);}
void* wotr_snapshot_apply_pits_event(const void* handle,const void* tables,int32_t side,int64_t index,const char* id){return checked<void*>([&]()->void*{require(id,"Pits card ID required");return apply_pits_event(snapshot(handle),snapshot(tables),side,index,id).release();},nullptr);}
void* wotr_snapshot_apply_pits_choice(const void* handle,const void* tables,int32_t side,int64_t index){return checked<void*>([&]()->void*{return apply_pits_choice(snapshot(handle),snapshot(tables),side,index).release();},nullptr);}
void* wotr_snapshot_pits_choice_actions(const void* handle,const void* tables,int32_t side){return checked<void*>([&]()->void*{return native_pits_choice_actions(snapshot(handle),snapshot(tables),side).release();},nullptr);}
void* wotr_snapshot_apply_gwaihir_event(const void* handle,const void* tables,const void* random,int32_t side,int64_t index,const char* id,void** random_out,void** events){
    return checked<void*>([&]()->void*{require(id&&random_out&&events&&random_out!=events,"Gwaihir requires distinct outputs");auto result=apply_gwaihir_event(snapshot(handle),snapshot(tables),random,side,index,id);*random_out=result.random.release();*events=result.events.release();return result.state.release();},nullptr);
}
void* wotr_snapshot_apply_gwaihir_choice(const void* handle,const void* tables,const void* random,int32_t side,int64_t index,void** random_out,void** events){
    return checked<void*>([&]()->void*{require(random_out&&events&&random_out!=events,"Gwaihir choice requires distinct outputs");auto result=apply_gwaihir_choice(snapshot(handle),snapshot(tables),random,side,index);*random_out=result.random.release();*events=result.events.release();return result.state.release();},nullptr);
}
void* wotr_snapshot_gwaihir_choice_actions(const void* handle,int32_t side){return checked<void*>([&]()->void*{return native_gwaihir_choice_actions(snapshot(handle),side).release();},nullptr);}
void* wotr_snapshot_apply_power_event(const void* handle,const void* tables,int32_t side,int64_t index,const char* id){return checked<void*>([&]()->void*{require(id,"Card ID required");return apply_power_event(snapshot(handle),snapshot(tables),side,index,id).release();},nullptr);}
void* wotr_snapshot_apply_half_orcs_event(const void* handle,const void* tables,int32_t side,int64_t index,const char* id){return checked<void*>([&]()->void*{require(id,"Card ID required");return apply_half_orcs_event(snapshot(handle),snapshot(tables),side,index,id).release();},nullptr);}
void* wotr_snapshot_apply_half_orcs_choice(const void* handle,const void* tables,int32_t side,int64_t index){return checked<void*>([&]()->void*{return apply_half_orcs_choice(snapshot(handle),snapshot(tables),side,index).release();},nullptr);}
void* wotr_snapshot_half_orcs_choice_actions(const void* handle,const void* tables,int32_t side){return checked<void*>([&]()->void*{return native_half_orcs_choice_actions(snapshot(handle),snapshot(tables),side).release();},nullptr);}
void* wotr_snapshot_apply_power_kill(const void* handle,const void* tables,int32_t side,int64_t index,const char* id){return checked<void*>([&]()->void*{require(id,"Card ID required");return apply_power_kill(snapshot(handle),snapshot(tables),side,index,id).release();},nullptr);}
void* wotr_snapshot_apply_power_payment(const void* handle,const void* tables,int32_t side,int64_t index){return checked<void*>([&]()->void*{return apply_power_payment(snapshot(handle),snapshot(tables),side,index).release();},nullptr);}
void* wotr_snapshot_power_kill_actions(const void* handle,const void* tables,int32_t side){return checked<void*>([&]()->void*{return native_power_kill_actions(snapshot(handle),snapshot(tables),side).release();},nullptr);}
void* wotr_snapshot_power_payment_actions(const void* handle,const void* tables,int32_t side){return checked<void*>([&]()->void*{return native_power_payment_actions(snapshot(handle),snapshot(tables),side).release();},nullptr);}
void* wotr_snapshot_worn_choice_actions(const void* handle,int32_t side){
    return checked<void*>([&]()->void*{return native_worn_choice_actions(snapshot(handle),side).release();},nullptr);
}
void* wotr_snapshot_apply_worn_choice(const void* handle,const void* tables,const void* random,int32_t side,int64_t index,void** random_out,void** events){
    return checked<void*>([&]()->void*{require(random_out&&events&&random_out!=events,"Worn choice requires distinct outputs");auto result=apply_worn_choice(snapshot(handle),snapshot(tables),random,side,index);*random_out=result.random.release();*events=result.events.release();return result.state.release();},nullptr);
}
void* wotr_snapshot_overstack_choice_actions(const void* handle,int32_t side){
    return checked<void*>([&]()->void*{return native_overstack_choice_actions(snapshot(handle),side).release();},nullptr);
}
void* wotr_snapshot_apply_overstack_choice(const void* handle,const void* tables,int32_t side,int64_t index){
    return checked<void*>([&]()->void*{return apply_overstack_choice(snapshot(handle),snapshot(tables),side,index).release();},nullptr);
}
void* wotr_snapshot_combat_relocate(const void* handle,const void* tables,const char* from,const char* to,int32_t side,int64_t regular,int64_t elite,int64_t leaders,int64_t nazgul,int32_t battle,int32_t defer){
    return checked<void*>([&]()->void*{require(from&&to&&(battle==0||battle==1)&&(defer==0||defer==1),"Invalid combat relocation arguments");return combat_relocate(snapshot(handle),snapshot(tables),crl_Request{from,to,side,regular,elite,leaders,nazgul,battle!=0,defer!=0}).release();},nullptr);
}
void* wotr_snapshot_combat_phase_actions(const void* handle,const void* tables,int32_t side){
    return checked<void*>([&]()->void*{return native_combat_phase_actions(snapshot(handle),snapshot(tables),side).release();},nullptr);
}
void* wotr_snapshot_apply_combat_phase(const void* handle,const void* tables,const void* random,const char* kind,int64_t elites_back,void** random_out,void** events){
    return checked<void*>([&]()->void*{require(kind&&random_out&&events&&random_out!=events,"Combat phase requires distinct outputs");auto result=apply_combat_phase(snapshot(handle),snapshot(tables),random,cpl_Request{kind,elites_back});*random_out=result.random.release();*events=result.events.release();return result.state.release();},nullptr);
}
void* wotr_snapshot_apply_combat_advance(const void* handle,const void* tables,const void* random,int64_t index,void** random_out,void** events){
    return checked<void*>([&]()->void*{require(random_out&&events&&random_out!=events,"Combat advance requires distinct outputs");auto result=apply_combat_advance(snapshot(handle),snapshot(tables),random,index);*random_out=result.random.release();*events=result.events.release();return result.state.release();},nullptr);
}
void* wotr_snapshot_apply_hunt_event(const void* handle,const void* tables,const void* random,int32_t side,int64_t index,const char* id,void** random_out,void** events){
    return checked<void*>([&]()->void*{require(id&&random_out&&events&&random_out!=events,"Hunt event needs card and distinct outputs");auto result=apply_hunt_event_action(snapshot(handle),snapshot(tables),random,side,index,id);*random_out=result.random.release();*events=result.events.release();return result.state.release();},nullptr);
}
void* wotr_snapshot_hunt_event_choice_actions(const void* handle){
    return checked<void*>([&]()->void*{return native_hunt_event_choice_actions(snapshot(handle)).release();},nullptr);
}
void* wotr_snapshot_apply_hunt_event_choice(const void* handle,const void* tables,const void* random,int64_t index,void** random_out,void** events){
    return checked<void*>([&]()->void*{require(random_out&&events&&random_out!=events,"Hunt event choice needs distinct outputs");auto result=apply_hunt_event_choice(snapshot(handle),snapshot(tables),random,index);*random_out=result.random.release();*events=result.events.release();return result.state.release();},nullptr);
}
void* wotr_snapshot_apply_misc_event(const void* handle,const void* tables,const void* random,int32_t side,int64_t index,const char* id,void** random_out,void** events){
    return checked<void*>([&]()->void*{require(id&&random_out&&events&&random_out!=events,"Event needs card and distinct outputs");auto result=apply_misc_event_action(snapshot(handle),snapshot(tables),random,side,index,id);*random_out=result.random.release();*events=result.events.release();return result.state.release();},nullptr);
}
void* wotr_snapshot_misc_event_choice_actions(const void* handle,int32_t side){
    return checked<void*>([&]()->void*{return native_misc_event_choice_actions(snapshot(handle),side).release();},nullptr);
}
void* wotr_snapshot_apply_misc_event_choice(const void* handle,const void* tables,int32_t side,int64_t index){
    return checked<void*>([&]()->void*{return apply_misc_event_choice(snapshot(handle),snapshot(tables),side,index).release();},nullptr);
}
void* wotr_snapshot_apply_combat_onslaught_payment(const void* handle,const void* tables,const void* random,int64_t index,void** random_out,void** events){
    return checked<void*>([&]()->void*{
        require(random_out&&events&&random_out!=events,"Combat action needs distinct outputs");auto result=apply_combat_onslaught_payment(snapshot(handle),snapshot(tables),random,index);
        *random_out=result.random.release();*events=result.events.release();return result.state.release();
    },nullptr);
}
void* wotr_snapshot_apply_dice_event(const void* handle,const void* tables,int32_t side,int64_t index,const char* id){
    return checked<void*>([&]()->void*{require(id,"Event card ID required");return apply_dice_event_action(snapshot(handle),snapshot(tables),side,index,id).release();},nullptr);
}
void* wotr_snapshot_dice_event_choice_actions(const void* handle,int32_t side){
    return checked<void*>([&]()->void*{return native_dice_event_choice_actions(snapshot(handle),side).release();},nullptr);
}
void* wotr_snapshot_apply_dice_event_choice(const void* handle,const void* tables,int32_t side,int64_t index){
    return checked<void*>([&]()->void*{return apply_dice_event_choice(snapshot(handle),snapshot(tables),side,index).release();},nullptr);
}
void* wotr_snapshot_fellowship_declare_with_rng(const void* handle,const void* tables,const void* random,const char* destination,int32_t pass,void** random_out,void** events){
    return checked<void*>([&]()->void*{
        require(destination&&random_out&&events&&random_out!=events&&(pass==0||pass==1),"Invalid traced declaration arguments");auto result=apply_fellowship_declare_with_rng(snapshot(handle),snapshot(tables),random,destination,pass!=0);
        *random_out=result.random.release();*events=result.events.release();return result.state.release();
    },nullptr);
}
void* wotr_snapshot_fellowship_tail_with_events(const void* handle,const void* tables,const void* random,int32_t begin,void** random_out,int32_t* finished,void** events){
    return checked<void*>([&]()->void*{
        require(random_out&&finished&&events&&random_out!=events&&(begin==0||begin==1),"Invalid traced Fellowship tail arguments");auto result=fellowship_tail(snapshot(handle),snapshot(tables),random,begin!=0);
        *random_out=result.random.release();*finished=result.finished;*events=result.events.release();return result.state.release();
    },nullptr);
}
void* wotr_snapshot_apply_special_tile_event(const void* handle,const void* tables,const void* random,int32_t side,int64_t index,const char* id,void** random_out,void** events){
    return checked<void*>([&]()->void*{
        require(id&&random_out&&events&&random_out!=events,"Tile event needs card ID and distinct outputs");auto result=apply_special_tile_event_action(snapshot(handle),snapshot(tables),random,side,index,id);
        *random_out=result.random.release();*events=result.events.release();return result.state.release();
    },nullptr);
}
void* wotr_snapshot_hunt_menu(const void* handle,const void* tables,int32_t mode){
    return checked<void*>([&]()->void*{
        if(mode==0)return native_hunt_damage_actions(snapshot(handle)).release();
        if(mode==1)return native_hobbit_place_actions(snapshot(handle),snapshot(tables)).release();
        require(mode==2,"Unknown Hunt menu family");return native_reveal_actions(snapshot(handle),snapshot(tables)).release();
    },nullptr);
}
void* wotr_snapshot_hunt_decision(const void* handle,const void* tables,const void* random,int32_t mode,const char* selection,void** random_out,void** events){
    return checked<void*>([&]()->void*{
        require(selection && random_out && events && random_out!=events,"Hunt decision needs selection and distinct outputs");
        require(mode>=0&&mode<=3,"Unknown Hunt decision family");
        const auto& s=snapshot(handle);const auto& t=snapshot(tables);
        auto result=mode==0?apply_hunt_choice(s,t,random,selection):mode==1?apply_hunt_guide_choice(s,t,random,selection):mode==2?apply_hobbit_placement(s,t,random,selection):apply_fellowship_reveal(s,t,random,selection);
        *random_out=result.random.release();*events=result.events.release();return result.state.release();
    },nullptr);
}
void* wotr_snapshot_draw_card_actions(const void* handle,const void* tables,int32_t side){
    return checked<void*>([&]()->void*{return native_draw_card_actions(snapshot(handle),snapshot(tables),side).release();},nullptr);
}
void* wotr_snapshot_apply_draw_card(const void* handle,const void* tables,int32_t side,int64_t index,const char* deck){
    return checked<void*>([&]()->void*{
        require(deck,"draw deck required");return apply_draw_card_action(snapshot(handle),snapshot(tables),side,index,deck).release();
    },nullptr);
}
void* wotr_snapshot_hunt_tile(const void* handle,const void* tables,const void* random,int64_t successes,int32_t mordor,int32_t track,int32_t eye_no_effect,void** random_out,void** value,void** events){
    return checked<void*>([&]()->void*{
        require(random_out && value && events && random_out!=value && random_out!=events && value!=events,"Hunt tile needs distinct outputs");
        require((mordor==0 || mordor==1) && (track==0 || track==1) && (eye_no_effect==0 || eye_no_effect==1),"Hunt tile flags must be Boolean");
        auto result=native_hunt_tile(snapshot(handle),snapshot(tables),random,successes,mordor!=0,track!=0,eye_no_effect!=0);
        *random_out=result.random.release();*value=result.value.release();*events=result.events.release();return result.state.release();
    },nullptr);
}
void* wotr_snapshot_pending_actor(const void* handle,int32_t fallback){
    return checked<void*>([&]()->void*{
        const auto result=native_pending_actor(snapshot(handle),fallback);am_Builder b;
        const auto kind=result.kind.empty()?rr_scalar(b.out,WOTR_NONE):b.str(result.kind);
        const auto side=result.side<0?rr_scalar(b.out,WOTR_NONE):b.integer(result.side);
        b.out.root=b.container(WOTR_TUPLE,{kind,side});return std::make_unique<Snapshot>(b.out).release();
    },nullptr);
}
void* wotr_snapshot_combat_post_stage(const void* handle,int64_t selection){
    return checked<void*>([&]()->void*{return combat_post_stage(snapshot(handle),selection).release();},nullptr);
}
void* wotr_snapshot_fellowship_tail(const void* handle,const void* tables,const void* random_source,int32_t begin,void** random_out,int32_t* finished){
    return checked<void*>([&]()->void*{
        require(random_out && finished && (begin==0 || begin==1),"invalid Fellowship tail output or mode");
        auto result=fellowship_tail(snapshot(handle),snapshot(tables),random_source,begin!=0);
        *random_out=result.random.release();*finished=result.finished;return result.state.release();
    },nullptr);
}
void* wotr_snapshot_fellowship_declare_actions(const void* handle,const void* tables,int32_t side){
    return checked<void*>([&]()->void*{return native_fellowship_declare_actions(snapshot(handle),snapshot(tables),side).release();},nullptr);
}
void* wotr_snapshot_guide_actions(const void* handle,int32_t side){
    return checked<void*>([&]()->void*{return native_guide_actions(snapshot(handle),side).release();},nullptr);
}
void* wotr_snapshot_apply_guide_choice(const void* handle,const void* tables,const char* name){
    return checked<void*>([&]()->void*{
        require(name,"Guide name required");return apply_guide_choice(snapshot(handle),snapshot(tables),name).release();
    },nullptr);
}
void* wotr_snapshot_apply_fellowship_declare(const void* handle,const void* tables,const char* destination,int32_t pass){
    return checked<void*>([&]()->void*{
        require(destination && (pass==0 || pass==1),"invalid declaration parameters");
        return apply_fellowship_declare(snapshot(handle),snapshot(tables),destination,pass!=0).release();
    },nullptr);
}
void* wotr_snapshot_discard_card_actions(const void* handle,const void* tables,int32_t side){
    return checked<void*>([&]()->void*{return discard_card_actions(snapshot(handle),snapshot(tables),side).release();},nullptr);
}
void* wotr_snapshot_apply_discard_card(const void* handle,const void* tables,int32_t side,const char* card){
    return checked<void*>([&]()->void*{
        require(card,"discard card ID required");
        return apply_discard_card(snapshot(handle),snapshot(tables),side,card).release();
    },nullptr);
}
void* wotr_snapshot_saruman_muster_actions(const void* handle,const void* tables,int32_t side){
    return checked<void*>([&]()->void*{return saruman_muster_actions(snapshot(handle),snapshot(tables),side).release();},nullptr);
}
void* wotr_snapshot_apply_saruman_muster_action(const void* handle,const void* tables,int32_t side,int64_t index,const char* mode){
    return checked<void*>([&]()->void*{
        require(mode,"Saruman mode string required");
        return apply_saruman_muster_action(snapshot(handle),snapshot(tables),side,index,mode).release();
    },nullptr);
}
void* wotr_snapshot_eye_allocation_actions(const void* handle,int32_t side){
    return checked<void*>([&]()->void*{return eye_allocation_actions(snapshot(handle),side).release();},nullptr);
}
void* wotr_snapshot_apply_eye_allocation(const void* handle,const void* tables,int64_t count){
    return checked<void*>([&]()->void*{return apply_eye_allocation(snapshot(handle),snapshot(tables),count).release();},nullptr);
}
void* wotr_snapshot_roll_action_dice(const void* handle,const void* random_source,int32_t side,void** random_out,void** faces_out){
    return checked<void*>([&]()->void*{
        require(random_out && faces_out && random_out!=faces_out,"dice roll needs distinct output handles");
        auto result=roll_action_dice(snapshot(handle),random_source,side);
        *random_out=result.random.release();*faces_out=result.faces.release();
        return result.state.release();
    },nullptr);
}
void* wotr_snapshot_combat_reroll_stage(const void* handle,const void* random_source,void** random_out){
    return checked<void*>([&]()->void*{
        require(random_out,"combat rerolls need RNG output");
        auto result=combat_reroll_stage(snapshot(handle),random_source);
        *random_out=result.random.release();return result.state.release();
    },nullptr);
}
void* wotr_snapshot_combat_primary_stage(const void* handle,const void* random_source,const char* target,void** random_out){
    return checked<void*>([&]()->void*{
        require(random_out,"combat primary stage needs RNG output");
        auto result=combat_primary_stage(snapshot(handle),random_source,target);
        *random_out=result.random.release();return result.state.release();
    },nullptr);
}
void* wotr_snapshot_draw_event_card(const void* handle,int32_t side,const char* deck,void** draw_out){
    return checked<void*>([&]()->void*{
        require(deck && draw_out,"card draw requires deck and output");
        auto child=std::make_unique<Snapshot>(snapshot(handle));
        const auto card=tc_draw_event_card(*child,side,deck);
        am_Builder b;const auto id=b.str(text(*child,card));
        const auto annotation=b.row(-1,"card_drawn",{b.str("deck"),b.str(deck),b.str("card_id"),id,
            b.str("side"),b.str(side==0?"FP":"SH")});
        b.out.root=b.container(WOTR_TUPLE,{id,annotation});
        auto result=std::make_unique<Snapshot>(b.out);
        *draw_out=result.release();return child.release();
    },nullptr);
}
void* wotr_snapshot_open_hand_limit_discard(const void* handle,int32_t* opened){
    return checked<void*>([&]()->void*{
        require(opened,"hand limit needs output flag");
        auto child=std::make_unique<Snapshot>(snapshot(handle));
        const auto result=tc_open_hand_limit_discard(*child);*opened=result;return child.release();
    },nullptr);
}
void* wotr_snapshot_next_turn(const void* handle,const void* tables,int32_t* won,void** annotations){
    return checked<void*>([&]()->void*{
        require(won && annotations,"next turn needs outputs");
        auto child=std::make_unique<Snapshot>(snapshot(handle));nt_enforce_control(*child,snapshot(tables));
        const auto turn=typed(*child,obs_attr(*child,child->root,"turn"),WOTR_INT,"turn must be integer").integer;
        require(turn<std::numeric_limits<int64_t>::max(),"next turn exceeds int64");
        const bool victory=tr_settled_reset(*child,snapshot(tables),turn+1,true);am_Builder b;std::vector<uint64_t> rows;
        if(!victory)nt_deal(*child,b,rows);b.out.root=b.container(WOTR_LIST,rows);auto events=std::make_unique<Snapshot>(b.out);
        *won=victory;*annotations=events.release();return child.release();
    },nullptr);
}
void* wotr_snapshot_enforce_control(const void* handle,const void* tables){
    return checked<void*>([&]()->void*{
        auto child=std::make_unique<Snapshot>(snapshot(handle));nt_enforce_control(*child,snapshot(tables));return child.release();
    },nullptr);
}
void* wotr_snapshot_begin_turn_reset(const void* handle,int64_t turn,int32_t deal,int32_t bonus,int32_t* won,void** annotations){
    return checked<void*>([&]()->void*{
        require(won && annotations && (deal==0 || deal==1) && (bonus==0 || bonus==1),"invalid turn reset outputs or flags");
        auto child=std::make_unique<Snapshot>(snapshot(handle));
        const bool victory=tr_reset(*child,turn,bonus);am_Builder b;std::vector<uint64_t> rows;
        if(deal){
            for(int side=0;side<2;++side)for(const auto deck:{"character","strategy"}){
                uint64_t card;
                try{card=tc_draw_event_card(*child,side,deck);}
                catch(const std::runtime_error& error){
                    if(std::string_view(error.what()).find("deck empty")!=std::string_view::npos)continue;
                    throw;
                }
                rows.push_back(b.row(-1,"card_drawn",{b.str("deck"),b.str(deck),b.str("card_id"),b.str(text(*child,card)),
                    b.str("side"),b.str(side==0?"FP":"SH")}));
            }
            tc_open_hand_limit_discard(*child);
        }
        b.out.root=b.container(WOTR_LIST,rows);auto events=std::make_unique<Snapshot>(b.out);
        *won=victory;*annotations=events.release();return child.release();
    },nullptr);
}
void* wotr_snapshot_begin_turn_reset_settled(const void* handle,const void* tables,int64_t turn,int32_t deal,int32_t bonus,int32_t* won,void** annotations){
    return checked<void*>([&]()->void*{
        require(won && annotations && (deal==0 || deal==1) && (bonus==0 || bonus==1),"invalid turn reset outputs or flags");
        auto child=std::make_unique<Snapshot>(snapshot(handle));
        const bool victory=tr_settled_reset(*child,snapshot(tables),turn,bonus);am_Builder b;std::vector<uint64_t> rows;
        if(deal&&!victory)nt_deal(*child,b,rows);
        b.out.root=b.container(WOTR_LIST,rows);auto events=std::make_unique<Snapshot>(b.out);
        *won=victory;*annotations=events.release();return child.release();
    },nullptr);
}
void* wotr_snapshot_apply_army_move(const void* handle,const void* tables,int32_t side,int64_t index,
    const char* from,const char* to,int64_t regular,int64_t elite,int64_t leaders,int64_t nazgul,int32_t take_leaders,int32_t pass){
    return checked<void*>([&]()->void*{
        require(from && to,"army movement needs region strings");
        require((take_leaders==0 || take_leaders==1) && (pass==0 || pass==1),"army movement flags must be Boolean");
        ArmyMoveRequest req;req.side=side;req.die_index=index;req.from=from;req.to=to;
        req.regular=regular;req.elite=elite;req.leaders=leaders;req.nazgul=nazgul;
        req.take_leaders=take_leaders;req.pass=pass;
        return apply_army_move(snapshot(handle),snapshot(tables),req).release();
    },nullptr);
}
void* wotr_snapshot_apply_muster_action(const void* handle,const void* tables,int32_t side,int64_t index,
                                       const char* nation,const char* region,const char* kind){
    return checked<void*>([&]()->void*{
        require(nation && region && kind,"muster parameters must be nonnull strings");
        return apply_muster_action(snapshot(handle),snapshot(tables),side,index,nation,region,kind).release();
    },nullptr);
}
void* wotr_snapshot_apply_political_action(const void* handle,const void* tables,
                                          int32_t side,int64_t index,const char* nation){
    return checked<void*>([&]()->void*{
        return apply_political_action(snapshot(handle),snapshot(tables),side,index,nation_argument(nation)).release();
    },nullptr);
}
int wotr_snapshot_next_quiet_actor(const void* handle,const void* tables,int32_t side){
    return checked<int>([&](){return next_quiet_actor(snapshot(handle),snapshot(tables),side);},-1);
}
void* wotr_snapshot_apply_elven_ring_action(const void* handle,const void* tables,
                                           int32_t side,int64_t index,const char* face){
    return checked<void*>([&]()->void*{
        require(face!=nullptr,"Elven Ring face name is null");
        return apply_elven_ring_action(snapshot(handle),snapshot(tables),side,index,face).release();
    },nullptr);
}
void wotr_snapshot_destroy(void* handle){
    checked<int>([&](){delete static_cast<Snapshot*>(handle);return 0;},-1);
}
size_t wotr_snapshot_node_count(const void* handle){return checked<size_t>([&](){return snapshot(handle).nodes.size();},0);}
size_t wotr_snapshot_edge_count(const void* handle){return checked<size_t>([&](){return snapshot(handle).edges.size();},0);}
size_t wotr_snapshot_byte_count(const void* handle){return checked<size_t>([&](){return snapshot(handle).bytes.size();},0);}
uint64_t wotr_snapshot_root(const void* handle){return checked<uint64_t>([&](){return snapshot(handle).root;},0);}
int wotr_snapshot_export(const void* handle,WotrValue* n,size_t nn,uint64_t* e,size_t ne,uint8_t* b,size_t nb){
    return checked<int>([&](){
        const auto& s=snapshot(handle);
        require(nn==s.nodes.size() && ne==s.edges.size() && nb==s.bytes.size(),"snapshot output counts must be exact");
        check_array(n,nn);check_array(e,ne);check_array(b,nb);
        // All failure-producing checks precede every output write.
        if(nn)std::memcpy(n,s.nodes.data(),nn*sizeof(WotrValue));
        if(ne)std::memcpy(e,s.edges.data(),ne*sizeof(uint64_t));
        if(nb)std::memcpy(b,s.bytes.data(),nb*sizeof(uint8_t));
        return 0;
    },-1);
}
int wotr_snapshot_replace_integer(void* handle,uint64_t node,int64_t expected,int64_t replacement){
    return checked<int>([&](){
        const auto& s=snapshot(handle);
        require(node<s.nodes.size(),"integer replacement node out of bounds");
        require(s.nodes[node].tag==WOTR_INT,"integer replacement requires INT node");
        require(s.nodes[node].integer==expected,"integer replacement expected value mismatch");
        invalidate_action_clone(*static_cast<Snapshot*>(handle));
        static_cast<Snapshot*>(handle)->nodes[node].integer=replacement;
        return 0;
    },-1);
}
int wotr_snapshot_can_advance_political(const void* handle,const char* nation_name,const char* method_name){
    return checked<int>([&](){
        const auto nation=nation_argument(nation_name),method=method_argument(method_name);
        const auto& s=snapshot(handle);const auto view=political_view(s,nation,false);
        return can_advance(s,view,nation,method) ? 1 : 0;
    },-1);
}
int wotr_snapshot_advance_political(void* handle,const char* nation_name,const char* method_name){
    return checked<int>([&](){
        const auto nation=nation_argument(nation_name),method=method_argument(method_name);
        snapshot(handle);
        return advance(*static_cast<Snapshot*>(handle),nation,method);
    },-1);
}
WOTR_API int wotr_snapshot_encode_actions_v52(const void* state,const void* tables,const void* menu,const void* action_tables,int32_t side,float* output,size_t count,int64_t* regions,size_t rows){
    return checked<int>([&](){af_encode(snapshot(state),snapshot(tables),snapshot(menu),snapshot(action_tables),side,output,count,regions,rows);return 0;},-1);
}
WOTR_API int wotr_snapshot_prepare_action_tables_v52(void* handle){
    return checked<int>([&](){snapshot(handle);af_prepare_tables(*static_cast<Snapshot*>(handle));return 0;},-1);
}
WOTR_API void* wotr_snapshot_search_menu_view(const void* menu,const void* tables){
    return checked<void*>([&](){return native_menu_view(snapshot(menu),snapshot(tables)).release();},nullptr);
}
WOTR_API void* wotr_snapshot_search_menu_row(const void* menu,int64_t index){
    return checked<void*>([&](){return native_menu_row(snapshot(menu),index).release();},nullptr);
}
WOTR_API void* wotr_snapshot_search_menu_rows(const void* menu,const void* indices){
    return checked<void*>([&](){return native_menu_rows(snapshot(menu),snapshot(indices)).release();},nullptr);
}
WOTR_API void* wotr_snapshot_search_menu_alias_groups(const void* menu){
    return checked<void*>([&](){return native_menu_alias_groups(snapshot(menu)).release();},nullptr);
}
WOTR_API void* wotr_snapshot_search_lattice_context(const void* state,const void* tables,const void* menu){return checked<void*>([&](){return native_search_lattice_context(snapshot(state),snapshot(tables),snapshot(menu)).release();},nullptr);}
int wotr_snapshot_encode_prefix(const void* handle,int32_t side,const char* const* region_keys,size_t nr,
                               const char* const* nation_names,size_t nn,float* output,size_t count){
    return checked<int>([&](){encode_prefix(snapshot(handle),side,region_keys,nr,nation_names,nn,output,count);return 0;},-1);
}
int64_t wotr_snapshot_normalize_name(const char* input,size_t length,char* output,size_t capacity){
    return checked<int64_t>([&](){
        check_array(input,length);check_array(output,capacity);
        const auto normalized=sanitize_name(length ? std::string_view(input,length) : std::string_view());
        require(normalized.size()<=static_cast<size_t>(std::numeric_limits<int64_t>::max()),"normalized length overflows");
        if(output==nullptr && capacity==0)return static_cast<int64_t>(normalized.size());
        require(capacity>=normalized.size(),"normalization output capacity is too small");
        if(!normalized.empty())std::memcpy(output,normalized.data(),normalized.size());
        return static_cast<int64_t>(normalized.size());
    },-1);
}
}
#include "reply_graph_key.inc"
#include "search_controller.inc"
#include "wotr_reply_panel.h"
#include "reply_panel.inc"
#include "own_reply_methods.inc"
#include "search_controller_exports.inc"
#include "reply_panel_exports.inc"

#include "test_key_exports.inc"
