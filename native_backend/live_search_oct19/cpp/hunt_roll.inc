// Hunt chance primitives; deterministic board lookup plus explicit RNG stream.
struct hr_Params {int64_t dice=0,level=0,fp=0,eyes=0;bool nazgul=false,army=false,stronghold=false;};
uint64_t hr_army(const Snapshot& s,std::string_view key){
    const auto armies=obs_attr(s,s.root,"armies"),exact=ctx_get(s,armies,key);
    if(!ctx_none(s,exact))return exact;
    const auto pairs=child_edges(s,armies);
    for(size_t i=0;i<pairs.size();i+=2)if(sanitize_name(text(s,pairs[i]))==key)return pairs[i+1];
    return missing;
}
bool hr_nazgul(const Snapshot& s,uint64_t leader){
    if(object_type(s,leader)=="game_state.Nazgul")return true;
    const auto name=obs_attr(s,leader,"name",false);if(ctx_none(s,name))return false;
    std::string letters;
    for(unsigned char ch:text(s,name))if(ch>='A' && ch<='Z')letters.push_back(char(ch+32));else if(ch>='a' && ch<='z')letters.push_back(char(ch));
    return letters.find("witchking")!=std::string::npos;
}
hr_Params hr_sources(const Snapshot& s,const Snapshot& t){
    hr_Params result;const auto fs=obs_attr(s,s.root,"fellowship");const auto key=glb_fellowship_key(s,fs);const auto army=hr_army(s,key);
    if(!ctx_none(s,army)){
        for(auto leader:child_edges(s,obs_attr(s,army,"leaders")))result.nazgul|=hr_nazgul(s,leader);
        for(auto unit:child_edges(s,obs_attr(s,army,"units")))result.army|=ctx_unit_side(s,unit)==1;
    }
    const auto meta=ctx_get(t,field(t,t.root,"board_regions"),key);
    if(meta!=missing && truth(t,ctx_get(t,meta,"is_stronghold"))){
        const auto control=obs_attr(s,s.root,"enemy_controlled");
        const bool fp=contains_string(t,field(t,t.root,"fp_region_keys"),key);
        const bool sh=contains_string(t,field(t,t.root,"shadow_region_keys"),key);
        result.stronghold=sh?!contains_string(s,ctx_side_value(s,control,0),key):
            fp && contains_string(s,ctx_side_value(s,control,1),key);
    }
    return result;
}
hr_Params hr_parameters(const Snapshot& s,const Snapshot& t){
    auto result=hr_sources(s,t);
    const auto box=obs_attr(s,s.root,"hunt_box");typed(s,box,WOTR_LIST,"Hunt Box must be list");
    for(auto die:child_edges(s,box)){
        require(object_type(s,die)=="game_state.ActionDie","Hunt Box requires ActionDie");const auto side=obs_attr(s,die,"side");
        result.fp+=enum_is(s,side,"game_state.Side","FREE_PEOPLES");result.level+=enum_is(s,side,"game_state.Side","SHADOW");
        result.eyes+=enum_is(s,obs_attr(s,die,"current"),"game_state.ActionDieFace","EYE");
    }
    result.dice=std::min<int64_t>(5,result.level);return result;
}
std::unique_ptr<Snapshot> native_hunt_params(const Snapshot& s,const Snapshot& t){
    const auto p=hr_parameters(s,t);am_Builder b;std::vector<uint64_t> names;
    if(p.nazgul)names.push_back(b.str("Nazgul present"));if(p.army)names.push_back(b.str("Shadow army present"));if(p.stronghold)names.push_back(b.str("Shadow stronghold"));
    const auto available=rr_scalar(b.out,WOTR_BOOL,!names.empty());
    b.out.root=b.container(WOTR_DICT,{b.str("threshold"),b.integer(6),b.str("num_dice"),b.integer(p.dice),b.str("hunt_level"),b.integer(p.level),b.str("fp_dice_in_box"),b.integer(p.fp),b.str("fp_modifier"),b.integer(p.fp),b.str("eyes"),b.integer(p.eyes),b.str("reroll_sources"),b.container(WOTR_LIST,names),b.str("reroll_available"),available});
    return std::make_unique<Snapshot>(std::move(b.out));
}
struct hr_Result {
    std::unique_ptr<Snapshot> state;
    std::unique_ptr<void,void(*)(void*)> random{nullptr,wotr_random_destroy};
    std::unique_ptr<Snapshot> events;
    int64_t successes=0;
};
int64_t hr_roll(Snapshot& s,const Snapshot& t,void* random,int64_t count,int64_t threshold,int64_t modifier,am_Builder& events,std::vector<uint64_t>& rows){
    const auto bonus=obs_attr(s,s.root,"_crebain_hunt_bonus",false);
    if(truth(s,bonus)){
        const auto value=typed(s,bonus,WOTR_INT,"Crebain Hunt modifier must be integer").integer;
        require((value>=0 && modifier<=std::numeric_limits<int64_t>::max()-value) || (value<0 && modifier>=std::numeric_limits<int64_t>::min()-value),"Hunt modifier overflow");
        modifier+=value;rr_set_attr(s,s.root,"_crebain_hunt_bonus",rr_scalar(s,WOTR_INT));
    }
    if(count<=0)return 0;
    require(count<=1048576,"Hunt roll batch exceeds native allocation limit");
    require(modifier<=std::numeric_limits<int64_t>::max()-6,"Hunt modifier overflow");
    std::vector<uint32_t> rolls(static_cast<size_t>(count));
    hf_roll(random,rolls.data(),rolls.size(),"hunt roll");
    int64_t successes=0;std::vector<uint64_t> faces;
    for(auto roll:rolls){successes+=roll>1 && int64_t(roll)+modifier>=threshold;faces.push_back(events.integer(roll));}
    rows.push_back(events.row(-1,"dice_roll",{events.str("faces"),events.container(WOTR_LIST,faces),events.str("ctx"),events.str("hunt roll"),events.str("side"),events.str("SH")}));
    const auto p=hr_sources(s,t);const auto attempts=std::min<int64_t>(p.nazgul+p.army+p.stronghold,count-successes);faces.clear();
    for(int64_t i=0;i<attempts;++i){uint32_t roll=0;hf_roll(random,&roll,1,"hunt reroll");successes+=roll>1 && int64_t(roll)+modifier>=threshold;faces.push_back(events.integer(roll));}
    if(!faces.empty())rows.push_back(events.row(-1,"dice_roll",{events.str("faces"),events.container(WOTR_LIST,faces),events.str("ctx"),events.str("hunt re-roll"),events.str("side"),events.str("SH")}));
    return successes;
}
hr_Result native_hunt_roll(const Snapshot& source,const Snapshot& tables,const void* random_source,int64_t count,int64_t threshold,int64_t modifier){
    hr_Result result;result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());
    result.state=std::make_unique<Snapshot>(source);am_Builder b;std::vector<uint64_t> rows;
    result.successes=hr_roll(*result.state,tables,result.random.get(),count,threshold,modifier,b,rows);
    b.out.root=b.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(b.out));return result;
}
