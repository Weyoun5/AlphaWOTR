// Included inside snapshot.cpp's private namespace, after graph helpers.
// No Python objects or callbacks; object fields may live in slots or __dict__.
uint64_t obs_attr(const Snapshot& s,uint64_t object,const char* name,bool required=true){
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
