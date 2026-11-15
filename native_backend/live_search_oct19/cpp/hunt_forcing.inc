// Exact HuntDriver option-index execution. Pure probability tables come from
// the pinned Python helper; state-facing work stays on the resident graph.
struct hf_Fork {};
struct hf_Option {bool rolls=false;std::vector<uint32_t> faces;uint64_t index=0;double probability=0;};
struct hf_Scope;
thread_local hf_Scope* hf_active=nullptr;
struct hf_Scope {
    const int64_t* plan;size_t count,next=0;const Snapshot& chance;
    double probability=1;int hit_faces=0;std::vector<hf_Option> pending;
    std::string request;size_t request_size=0;
    hf_Scope(const Snapshot& state,const Snapshot& tables,const int64_t* values,size_t size):plan(values),count(size),chance(tables){
        require(!hf_active&&!cf_active,"Nested Hunt/combat forcing is not supported");
        check_array(values,size);require(size<=4096,"Hunt plan exceeds walk domain");
        require(text_is(chance,field(chance,chance.root,"schema"),"source_hunt_options_v1"),"Invalid Hunt option table schema");
        int64_t modifier=cp_number(state,state.root,"_crebain_hunt_bonus",0,true);
        require(modifier>=-1048576&&modifier<=1048576,"Hunt modifier exceeds forcing domain");
        for(auto die:child_edges(state,obs_attr(state,state.root,"hunt_box")))modifier+=enum_is(state,obs_attr(state,die,"side"),"game_state.Side","FREE_PEOPLES");
        for(int face=2;face<=6;++face)hit_faces+=face+modifier>=6;
        if(wotr_random_begin_forbidden_scope())throw std::invalid_argument(wotr_random_last_error());
        hf_active=this;
    }
    ~hf_Scope(){hf_active=nullptr;wotr_random_end_forbidden_scope();}
    hf_Option choose(std::vector<hf_Option> options,std::string_view kind,size_t size){
        if(next==count){pending=std::move(options);request=std::string(kind);request_size=size;throw hf_Fork{};}
        int64_t index=plan[next++];const auto length=static_cast<int64_t>(options.size());
        require(index>=-length&&index<length,"Hunt plan option index out of range");if(index<0)index+=length;
        auto result=options[static_cast<size_t>(index)];probability*=result.probability;return result;
    }
    std::vector<hf_Option> rows(uint64_t rows)const{
        std::vector<hf_Option> result;
        for(auto row:child_edges(chance,rows)){
            typed(chance,row,WOTR_TUPLE,"Hunt option must be tuple");auto pair=child_edges(chance,row);require(pair.size()==2,"Hunt option needs value and probability");
            hf_Option option;option.rolls=true;option.probability=typed(chance,pair[1],WOTR_FLOAT,"Hunt probability must be float").real;
            require(option.probability>=0&&option.probability<=1,"Hunt probability out of range");
            for(auto face:child_edges(chance,pair[0])){auto value=typed(chance,face,WOTR_INT,"Hunt face must be int").integer;require(value>=1&&value<=6,"Hunt face out of range");option.faces.push_back(static_cast<uint32_t>(value));}
            result.push_back(std::move(option));
        }
        require(!result.empty(),"Hunt options cannot be empty");return result;
    }
    uint64_t at(uint64_t list,size_t index)const{typed(chance,list,WOTR_LIST,"Hunt probability grid must be list");require(index<chance.nodes[list].count,"Hunt probability grid index out of range");return chance.edges[chance.nodes[list].offset+index];}
    std::unique_ptr<Snapshot> packet()const{
        am_Builder b;std::vector<uint64_t> rows;
        for(const auto& option:pending){
            uint64_t value;
            if(option.rolls){std::vector<uint64_t> faces;for(auto face:option.faces)faces.push_back(b.integer(face));value=b.container(WOTR_LIST,faces);}
            else value=b.integer(option.index);
            auto mass=rr_scalar(b.out,WOTR_FLOAT);b.out.nodes[mass].real=option.probability;
            rows.push_back(b.container(WOTR_TUPLE,{value,mass}));
        }
        b.out.root=b.container(WOTR_DICT,{b.str("kind"),b.str(request),b.str("size"),b.integer(request_size),b.str("options"),b.container(WOTR_LIST,rows)});
        return std::make_unique<Snapshot>(std::move(b.out));
    }
};
void hf_roll(void* rng,uint32_t* faces,size_t count,std::string_view kind){
    if(!hf_active){if(wotr_random_d6(rng,faces,count))throw std::invalid_argument(wotr_random_last_error());return;}
    auto& scope=*hf_active;uint64_t rows=missing;
    if(kind=="hunt roll"){require(count<=5,"Hunt primary forcing exceeds five dice");rows=scope.at(scope.at(field(scope.chance,scope.chance.root,"primary"),scope.hit_faces),count);}
    else if(kind=="hunt reroll"){require(count==1,"Hunt reroll must have one die");rows=scope.at(field(scope.chance,scope.chance.root,"reroll"),scope.hit_faces);}
    else if(kind=="shelob d6 damage"){require(count==1,"Shelob must roll one die");rows=field(scope.chance,scope.chance.root,"shelob");}
    else throw std::invalid_argument("unexpected roll ctx: '"+std::string(kind)+"'");
    auto option=scope.choose(scope.rows(rows),kind,count);require(option.faces.size()==count,"Hunt table face count mismatch");std::copy(option.faces.begin(),option.faces.end(),faces);
}
uint64_t hf_choice(void* rng,size_t count,bool companion){
    require(count>0,"Hunt choice has no candidates");
    if(!hf_active){uint64_t index=0;if(wotr_random_randbelow(rng,count,&index))throw std::invalid_argument(wotr_random_last_error());return index;}
    if(count==1&&!companion)return 0;
    std::vector<hf_Option> options;for(size_t i=0;i<count;++i)options.push_back({false,{},i,1.0/static_cast<double>(count)});
    return hf_active->choose(std::move(options),companion?"companion":"choice",count).index;
}
std::string hf_tile_key(const Snapshot& s,uint64_t tile){
    const auto special=obs_attr(s,tile,"special_name",false);
    if(truth(s,special)){
        const auto& t=hf_active->chance;const auto key=ctx_get(t,field(t,t.root,"specials"),text(s,special));
        require(key!=missing,"Unproved special Hunt tile key in forcing");return std::string(text(t,key));
    }
    if(truth(s,obs_attr(s,tile,"is_eye")))return "eye";
    return std::to_string(cp_number(s,tile,"damage"))+(truth(s,obs_attr(s,tile,"reveal"))?"r":"");
}
uint64_t hf_pool(const Snapshot& s,uint64_t pool,void* rng,bool back){
    const auto tiles=child_edges(s,pool);require(!tiles.empty(),"Cannot choose an empty Hunt Pool");
    if(!hf_active)return back?tiles.size()-1:hf_choice(rng,tiles.size(),false);
    std::map<std::string,std::pair<size_t,size_t>> kinds;
    for(size_t i=0;i<tiles.size();++i){auto key=hf_tile_key(s,tiles[i]);auto found=kinds.find(key);if(found==kinds.end())kinds[key]={i,1};else ++found->second.second;}
    std::vector<hf_Option> options;for(const auto& entry:kinds)options.push_back({false,{},entry.second.first,static_cast<double>(entry.second.second)/static_cast<double>(tiles.size())});
    return hf_active->choose(std::move(options),"pool",tiles.size()).index;
}
int hf_shuffle(void* rng,uint64_t* indices,size_t count){
    if(!hf_active)return wotr_random_shuffle(rng,indices,count);
    for(size_t i=0;i<count;++i)indices[i]=i;return 0;
}
