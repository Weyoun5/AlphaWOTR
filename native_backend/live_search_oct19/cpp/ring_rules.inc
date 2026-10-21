// Exact game_logic.use_elven_ring / ring_waiver_active helpers.
// Included in snapshot.cpp's anonymous namespace after obs_attr. No die payment
// or full action dispatch is performed. Mutations commit only after validation.
const std::array<std::string_view,7> rr_face_names={
    "CHARACTER","ARMY","MUSTER","EVENT","MUSTER_ARMY","EYE","WILL_OF_WEST"};
const std::array<std::string_view,7> rr_face_values={
    "Character","Army","Muster","Event","Muster/Army","Eye of Sauron","Will of the West"};
int rr_face_index(std::string_view name){
    for(size_t i=0;i<rr_face_names.size();++i)if(rr_face_names[i]==name)return static_cast<int>(i);
    throw std::invalid_argument("Invalid Elven Ring result");
}
uint64_t rr_scalar(Snapshot& s,uint32_t tag,int64_t integer=0,double real=0){
    const auto result=s.nodes.size();s.nodes.push_back({tag,0,integer,real,0,0});return result;
}
uint64_t rr_string(Snapshot& s,std::string_view value){
    using Entry=std::pair<size_t,uint64_t>;
    auto hash=[](std::string_view x){return std::hash<std::string_view>{}(x);};
    if(s.string_index_scanned>s.nodes.size()){s.string_index.clear();s.string_index_scanned=0;}
    const bool fresh=s.string_index_scanned==0;
    if(fresh)s.string_index.reserve(256);
    for(size_t i=s.string_index_scanned;i<s.nodes.size();++i)if(s.nodes[i].tag==WOTR_STRING){
        Entry item{hash(text(s,i)),i};
        if(fresh)s.string_index.push_back(item);
        else s.string_index.insert(std::lower_bound(s.string_index.begin(),s.string_index.end(),item),item);
    }
    if(fresh)std::sort(s.string_index.begin(),s.string_index.end());
    s.string_index_scanned=s.nodes.size();
    const auto wanted=hash(value);
    auto found=std::lower_bound(s.string_index.begin(),s.string_index.end(),Entry{wanted,0});
    for(;found!=s.string_index.end()&&found->first==wanted;++found)
        if(text_is(s,found->second,value))return found->second;
    // Callers supply literals or ABI-owned strings, never a slice that grows
    // out of this same byte vector.
    const auto offset=s.bytes.size();s.bytes.insert(s.bytes.end(),value.begin(),value.end());
    const auto result=s.nodes.size();s.nodes.push_back({WOTR_STRING,0,0,0,offset,value.size()});
    Entry item{wanted,result};s.string_index.insert(std::lower_bound(s.string_index.begin(),s.string_index.end(),item),item);
    s.string_index_scanned=s.nodes.size();return result;
}
uint64_t rr_enum(Snapshot& s,std::string_view name){
    for(size_t i=0;i<s.nodes.size();++i)if(enum_is(s,i,"game_state.ActionDieFace",name))return i;
    const auto type=rr_string(s,"game_state.ActionDieFace"),member=rr_string(s,name);
    const auto offset=s.edges.size();s.edges.push_back(type);s.edges.push_back(member);
    const auto result=s.nodes.size();s.nodes.push_back({WOTR_ENUM,0,0,0,offset,2});return result;
}
void rr_set_attr(Snapshot& s,uint64_t object,std::string_view name,uint64_t value){
    const auto& original=typed(s,object,WOTR_OBJECT,"Elven Ring target must be an object");
    uint64_t dictionary=missing,slot=missing;
    for(uint64_t i=1;i<original.count;i+=2){
        if(text_is(s,s.edges[original.offset+i],name))slot=i+1;
        if(text_is(s,s.edges[original.offset+i],"__dict__"))dictionary=s.edges[original.offset+i+1];
    }
    if(slot!=missing){
        auto edges=child_edges(s,object);edges[slot]=value;set_edges(s,object,edges);return;
    }
    require(dictionary!=missing,"Elven Ring attribute has no mutable storage");
    typed(s,dictionary,WOTR_DICT,"Elven Ring attributes must be dictionary");
    const auto edge=string_value_edge(s,dictionary,name);
    auto children=child_edges(s,dictionary);
    if(edge!=missing)children[edge-s.nodes[dictionary].offset]=value;
    else{const auto key=rr_string(s,name);children.push_back(key);children.push_back(value);}
    set_edges(s,dictionary,children);
}
bool rr_nonpositive(const Snapshot& s,uint64_t value){
    const auto& n=s.nodes[value];
    if(n.tag==WOTR_INT || n.tag==WOTR_BOOL)return n.integer<=0;
    if(n.tag==WOTR_FLOAT)return n.real<=0;
    throw std::invalid_argument("Elven Ring counter must be numeric");
}
uint64_t rr_adjust_counter(Snapshot& s,uint64_t value,int change){
    const auto n=s.nodes[value];
    if(n.tag==WOTR_FLOAT)return rr_scalar(s,WOTR_FLOAT,0,n.real+static_cast<double>(change));
    require(n.tag==WOTR_INT || n.tag==WOTR_BOOL,"Elven Ring counter must be numeric");
    require((change<0 && n.integer>std::numeric_limits<int64_t>::min()) ||
            (change>0 && n.integer<std::numeric_limits<int64_t>::max()),"Elven Ring counter exceeds int64 storage");
    return rr_scalar(s,WOTR_INT,n.integer+change);
}
int ring_waiver(const Snapshot& s,int32_t side){
    require(side==0 || side==1,"Invalid Elven Ring side");
    const auto waiver=obs_attr(s,s.root,side==0 ? "fp_ring_waiver" : "sh_ring_waiver",false);
    require(!truth(s,waiver),"Legacy Elven Ring waiver has no reliable physical target; resync or replay this game");
    return 0;
}
std::string use_ring(Snapshot& original,int32_t side,int64_t die_index,std::string_view new_face){
    require(side==0 || side==1,"Invalid Elven Ring side");
    const int destination=rr_face_index(new_face);
    ring_waiver(original,side);
    const char* ring_name=side==0 ? "fp_elven_rings" : "sh_elven_rings";
    const char* used_name=side==0 ? "fp_ring_used_this_turn" : "sh_ring_used_this_turn";
    const auto ring=obs_attr(original,original.root,ring_name);
    require(!rr_nonpositive(original,ring),"No Elven Rings available");
    require(!truth(original,obs_attr(original,original.root,used_name)),"Already used an Elven Ring this turn");
    require(new_face!="WILL_OF_WEST","Cannot change to Will of the West with ring");
    if(side==0){
        require(new_face!="EYE","Eye is not a Free Peoples face");
        require(new_face!="ARMY","Free Peoples Army result is Muster/Army");
    }
    const auto dice=obs_attr(original,original.root,"dice");
    const auto pool_edge=enum_value_edge(original,dice,"game_state.Side",side==0 ? "FREE_PEOPLES" : "SHADOW");
    require(pool_edge!=missing,"Elven Ring side has no die pool");
    const auto pool=original.edges[pool_edge];
    const auto& dice_pool=typed(original,pool,WOTR_LIST,"Elven Ring dice pool must be a list");
    if(die_index<0 || static_cast<uint64_t>(die_index)>=dice_pool.count)
        throw std::invalid_argument("Invalid die index "+std::to_string(die_index));
    const auto die=original.edges[dice_pool.offset+static_cast<uint64_t>(die_index)];
    require(!truth(original,obs_attr(original,die,"used")),"Cannot change a used die");
    const auto current=obs_attr(original,die,"current");
    require(original.nodes[current].tag!=WOTR_NONE,"Cannot change an unrolled die");
    const auto old_face=enum_member(original,current,"game_state.ActionDieFace");
    const int source=rr_face_index(old_face);
    require(side!=1 || old_face!="EYE","Shadow cannot change an Eye die with ring");
    require(old_face!=new_face,"Elven Ring must change the die result");
    // Construct the return value before any graph commit (views still borrow
    // the original graph). Source deliberately retains pool membership on Eye.
    const bool to_eye=side==1 && new_face=="EYE";
    const std::string message="Changed "+std::string(rr_face_values[source])+
        (to_eye ? " to Eye (moved to Hunt Box)" :
         " to "+std::string(rr_face_values[destination])+" with Elven Ring");
    Snapshot changed=original;
    rr_set_attr(changed,changed.root,ring_name,rr_adjust_counter(changed,ring,-1));
    rr_set_attr(changed,changed.root,used_name,rr_scalar(changed,WOTR_BOOL,1));
    if(side==0){
        const auto shadow=obs_attr(changed,changed.root,"sh_elven_rings");
        rr_set_attr(changed,changed.root,"sh_elven_rings",rr_adjust_counter(changed,shadow,1));
    }
    rr_set_attr(changed,die,"current",rr_enum(changed,new_face));
    rr_set_attr(changed,changed.root,side==0 ? "fp_ring_waiver" : "sh_ring_waiver",rr_scalar(changed,WOTR_BOOL,0));
    rr_set_attr(changed,changed.root,side==0 ? "fp_ring_waiver_face" : "sh_ring_waiver_face",rr_scalar(changed,WOTR_NONE));
    if(to_eye){
        rr_set_attr(changed,die,"used",rr_scalar(changed,WOTR_BOOL,1));
        const auto box=obs_attr(changed,changed.root,"hunt_box");
        typed(changed,box,WOTR_LIST,"Elven Ring Hunt Box must be a list");
        auto children=child_edges(changed,box);children.push_back(die);set_edges(changed,box,children);
    }
    auto validated=create(changed.nodes.data(),changed.nodes.size(),changed.edges.data(),changed.edges.size(),
                          changed.bytes.data(),changed.bytes.size(),changed.root);
    original.nodes.swap(validated->nodes);original.edges.swap(validated->edges);original.bytes.swap(validated->bytes);
    original.string_index.clear();original.string_index_scanned=0;
    original.context_prepared.swap(validated->context_prepared);
    std::swap(original.root,validated->root);
    invalidate_action_clone(original);
    return message;
}
