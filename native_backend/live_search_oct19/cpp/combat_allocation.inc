// Deferred casualties use live per-operation reserves, unlike pre-attack
// payment's frozen battle baseline. Every protected rearguard stays excluded.
struct ca_Partition{std::vector<uint64_t> units,leaders,rear_units,rear_leaders;};
using ca_CargoKey=std::tuple<std::string,std::string,std::string>;
std::map<ca_CargoKey,int64_t> cag_decode(const Snapshot&,uint64_t);
ca_CargoKey cag_key(const Snapshot&,uint64_t,bool);
ca_Partition ca_partition(const Snapshot& s,uint64_t army,std::string_view region,int side,uint64_t context=missing){
    ca_Partition result;if(ctx_none(s,army))return result;
    if(context==missing)context=obs_attr(s,s.root,"pending_combat",false);
    const auto payload=ctx_none(s,context)?missing:field(s,context,"attacker_rearguard",false);
    std::unordered_map<std::string,int64_t> units_due,leaders_due;
    std::map<ca_CargoKey,int64_t> reserve;
    const auto protected_cargo=ctx_none(s,context)?missing:field(s,context,"corsairs_relief_garrison",false);
    if(truth(s,protected_cargo)&&ctx_side(s,field(s,context,"attacker_side"))==side&&sanitize_name(text(s,field(s,context,"defender_region")))==sanitize_name(region))reserve=cag_decode(s,protected_cargo);
    if(!ctx_none(s,payload)){const auto p=child_edges(s,payload);require(p.size()==3,"Frozen rearguard payload requires three fields");
        if(text_is(s,p[0],sanitize_name(region))&&text_is(s,p[1],side==0?"FREE_PEOPLES":"SHADOW")){
            const auto signature=child_edges(s,p[2]);require(signature.size()==3&&text_is(s,signature[0],"forced_rearguard_v1"),"Unknown attack formation version");
            for(auto tuple:child_edges(s,signature[1])){const auto entry=child_edges(s,tuple);require(entry.size()==3,"Frozen unit guard requires nation,kind,count");const auto count=typed(s,entry[2],WOTR_INT,"Frozen guard count must be integer").integer;require(count>=0&&count<=1048576,"Invalid frozen unit guard count");units_due[std::string(text(s,entry[0]))+":"+std::string(text(s,entry[1]))]=count;}
            for(auto tuple:child_edges(s,signature[2])){const auto entry=child_edges(s,tuple);require(entry.size()==2,"Frozen leader guard requires nation,count");const auto count=typed(s,entry[1],WOTR_INT,"Frozen guard count must be integer").integer;require(count>=0&&count<=1048576,"Invalid frozen leader guard count");leaders_due[std::string(text(s,entry[0]))]=count;}
        }
    }
    for(auto unit:child_edges(s,obs_attr(s,army,"units"))){const std::string nation(enum_member(s,obs_attr(s,unit,"nation"),"game_state.Nation")),kind(enum_member(s,obs_attr(s,unit,"kind"),"game_state.UnitKind"));
        if(ctx_unit_side(s,unit)!=side)result.rear_units.push_back(unit);else if(!reserve.empty()&&reserve[cag_key(s,unit,true)]>0){--reserve[cag_key(s,unit,true)];result.rear_units.push_back(unit);}else if(units_due[nation+":"+kind]>0){--units_due[nation+":"+kind];result.rear_units.push_back(unit);}else result.units.push_back(unit);
    }
    for(auto leader:child_edges(s,obs_attr(s,army,"leaders"))){
        const auto type=object_type(s,leader);const bool plain=type=="game_state.Leader"||type=="game_state.Nazgul";const auto nation=plain?std::string(enum_member(s,obs_attr(s,leader,"nation"),"game_state.Nation")):std::string();
        if(cp_leader_side(s,leader)!=side)result.rear_leaders.push_back(leader);else if(!reserve.empty()&&reserve[cag_key(s,leader,false)]>0){--reserve[cag_key(s,leader,false)];result.rear_leaders.push_back(leader);}else if(plain&&leaders_due[nation]>0){--leaders_due[nation];result.rear_leaders.push_back(leader);}else result.leaders.push_back(leader);
    }
    for(const auto& entry:units_due)require(entry.second<=0,"A protected rearguard unit is missing");for(const auto& entry:leaders_due)require(entry.second<=0,"A protected rearguard leader is missing");for(const auto& entry:reserve)require(entry.second<=0,"A protected Corsairs garrison figure is missing");return result;
}
std::vector<uint64_t> ca_units(const Snapshot& s,std::string_view region,int side){
    auto units=ca_partition(s,mu_army(s,sanitize_name(region)),region,side).units;
    const auto card=obs_attr(s,s.root,"pending_card_option",false);if(!ctx_none(s,card)&&truth(s,field(s,card,"card_casualty",false))&&enum_is(s,field(s,card,"cas_side",false),"game_state.Side",side==0?"FREE_PEOPLES":"SHADOW")){
        const auto place=field(s,card,"cas_region",false);if(place!=missing&&sanitize_name(text(s,place))==sanitize_name(region)){const auto eligible=field(s,card,"cas_eligible_nations",false);if(!ctx_none(s,eligible)){std::vector<uint64_t> keep;for(auto unit:units){const auto name=enum_member(s,obs_attr(s,unit,"nation"),"game_state.Nation");bool found=false;for(auto nation:child_edges(s,eligible))found|=enum_is(s,nation,"game_state.Nation",name);if(found)keep.push_back(unit);}units=std::move(keep);}}
    }return units;
}
struct ca_Operation{std::string nation,operation;int64_t cost;};
std::vector<ca_Operation> ca_operations(const Snapshot& s,std::string_view region,int side,int64_t hits){
    require(side==0||side==1,"Casualty side must be FP or Shadow");std::vector<ca_Operation> result;if(hits<=0)return result;const auto units=ca_units(s,region,side);
    for(const char* nation:{"ELVES","DWARVES","GONDOR","THE_NORTH","ROHAN","ISENGARD","SAURON","SOUTHRON_EASTERLINGS"}){
        bool regular=false,elite=false;for(auto unit:units)if(enum_is(s,obs_attr(s,unit,"nation"),"game_state.Nation",nation)){regular|=enum_is(s,obs_attr(s,unit,"kind"),"game_state.UnitKind","REGULAR");elite|=enum_is(s,obs_attr(s,unit,"kind"),"game_state.UnitKind","ELITE");}
        if(regular)result.push_back({nation,"remove_regular",1});if(elite){result.push_back({nation,"reduce_elite",1});if(hits>=2)result.push_back({nation,"remove_elite",2});}
    }return result;
}
int64_t ca_apply(Snapshot& s,const Snapshot& tables,std::string_view region,int side,int64_t hits,std::string_view nation,std::string_view operation){
    const auto options=ca_operations(s,region,side,hits);int64_t cost=0;for(const auto& option:options)if(option.nation==nation&&option.operation==operation){cost=option.cost;break;}require(cost>0,"Casualty operation exceeds hit budget or participating roster");
    const bool regular=operation=="remove_regular";uint64_t unit=missing;for(auto candidate:ca_units(s,region,side))if(enum_is(s,obs_attr(s,candidate,"nation"),"game_state.Nation",nation)&&(enum_is(s,obs_attr(s,candidate,"kind"),"game_state.UnitKind","REGULAR")==regular)){unit=candidate;break;}require(unit!=missing,"Participating casualty figure is missing");
    const auto army=mu_army(s,sanitize_name(region)),units=obs_attr(s,army,"units"),nation_node=obs_attr(s,unit,"nation");uint64_t source=rr_scalar(s,WOTR_NONE);
    if(operation=="reduce_elite"){
        const auto box=obs_attr(s,s.root,"fp_casualties",false),counts=ctx_nation_value(s,box,nation),number=ctx_none(s,counts)?missing:field(s,counts,"regular",false);
        const bool from_box=side==0&&number!=missing&&typed(s,number,WOTR_INT,"FP casualty count must be integer").integer>0;
        if(from_box){source=rr_string(s,"cas");cp_put(s,counts,"regular",rr_scalar(s,WOTR_INT,s.nodes[number].integer-1));}
        else if(mu_available(s,tables,nation,"regular")>0)source=rr_string(s,"pool");
        auto log=obs_attr(s,s.root,"_dg_src_log",false);if(ctx_none(s,log)){log=cp_container(s,WOTR_LIST,{});rr_set_attr(s,s.root,"_dg_src_log",log);}auto entries=child_edges(s,log);entries.push_back(cp_container(s,WOTR_TUPLE,{nation_node,source}));set_edges(s,log,entries);
        // list.remove uses dataclass equality: the first identical figure may
        // belong to the frozen count-based rear, which is then re-reserved on
        // the next partition. Match the source's physical ordering exactly.
        auto deployed=child_edges(s,units);auto it=std::find_if(deployed.begin(),deployed.end(),[&](uint64_t other){return enum_is(s,obs_attr(s,other,"nation"),"game_state.Nation",nation)&&enum_is(s,obs_attr(s,other,"kind"),"game_state.UnitKind","ELITE");});require(it!=deployed.end(),"Casualty figure disappeared");deployed.erase(it);
        if(!ctx_none(s,source))deployed.push_back(mu_object(s,"game_state.Unit",{{"nation",nation_node},{"kind",mu_enum(s,"game_state.UnitKind","REGULAR")}}));set_edges(s,units,deployed);
    }else{
        auto deployed=child_edges(s,units);auto it=std::find_if(deployed.begin(),deployed.end(),[&](uint64_t other){return enum_is(s,obs_attr(s,other,"nation"),"game_state.Nation",nation)&&(enum_is(s,obs_attr(s,other,"kind"),"game_state.UnitKind","REGULAR")==regular);});require(it!=deployed.end(),"Casualty figure disappeared");deployed.erase(it);set_edges(s,units,deployed);
    }
    if(side==0)cbp_book(s,nation_node,regular?"regular":"elite");
    const auto card=obs_attr(s,s.root,"pending_card_option",false),pc=obs_attr(s,s.root,"pending_combat",false);const bool card_loss=!ctx_none(s,card)&&truth(s,field(s,card,"card_casualty",false));
    const auto stage=card_loss?rr_string(s,"card"):ctx_none(s,pc)?rr_scalar(s,WOTR_NONE):field(s,pc,"alloc_stage",false);
    const auto witness=cp_container(s,WOTR_DICT,{rr_string(s,"stage"),stage==missing?rr_scalar(s,WOTR_NONE):stage,rr_string(s,"region"),rr_string(s,region),rr_string(s,"side"),mu_enum(s,"game_state.Side",side==0?"FREE_PEOPLES":"SHADOW"),rr_string(s,"nation"),nation_node,rr_string(s,"operation"),rr_string(s,operation),rr_string(s,"cost"),rr_scalar(s,WOTR_INT,cost),rr_string(s,"remaining_before"),rr_scalar(s,WOTR_INT,hits),rr_string(s,"replacement_source"),source});
    const auto old=obs_attr(s,s.root,"_casualty_operations",false);auto entries=ctx_none(s,old)?std::vector<uint64_t>{}:child_edges(s,old);entries.push_back(witness);rr_set_attr(s,s.root,"_casualty_operations",cp_container(s,WOTR_LIST,entries));return cost;
}
std::unique_ptr<Snapshot> combat_casualty_operation(const Snapshot& source,const Snapshot& tables,int side,const char* region,int64_t hits,const char* nation,const char* operation){
    require(region&&nation&&operation,"Casualty operation requires explicit region/nation/operation");require(hits>=0&&hits<=1048576,"Casualty hit budget exceeds native range");auto result=std::make_unique<Snapshot>(source);ca_apply(*result,tables,region,side,hits,nation,operation);return ac_compact(*result);
}
std::unique_ptr<Snapshot> combat_casualty_menu(const Snapshot& source,int side,const char* region,int64_t hits){
    require(region,"Casualty menu requires explicit region");am_Builder build;std::vector<uint64_t> rows;
    for(const auto& option:ca_operations(source,region,side,hits))rows.push_back(build.row(-1,"casualty_allocate",{build.str("nation"),build.nation(option.nation),build.str("operation"),build.str(option.operation),build.str("casualties"),build.integer(hits)}));build.out.root=build.container(WOTR_LIST,rows);return std::make_unique<Snapshot>(build.out);
}
int64_t ca_forced(Snapshot& s,const Snapshot& tables,std::string_view region,int side,int64_t hits){
    const auto units=ca_units(s,region,side);int64_t capacity=0;bool all_regular=true;
    for(auto unit:units){const bool elite=enum_is(s,obs_attr(s,unit,"kind"),"game_state.UnitKind","ELITE");capacity+=elite?2:1;all_regular&=!elite;}
    if(hits>=capacity&&(side==1||all_regular)){
        for(auto unit:units){const std::string nation(enum_member(s,obs_attr(s,unit,"nation"),"game_state.Nation"));const char* operation=enum_is(s,obs_attr(s,unit,"kind"),"game_state.UnitKind","ELITE")?"remove_elite":"remove_regular";hits-=ca_apply(s,tables,region,side,hits,nation,operation);}return 0;
    }
    while(hits>0){const auto options=ca_operations(s,region,side,hits);if(options.empty())return 0;if(options.size()!=1)break;const auto choice=options.front();hits-=ca_apply(s,tables,region,side,hits,choice.nation,choice.operation);}return hits;
}
bool ca_enter(Snapshot& s,const Snapshot& tables,uint64_t pc){
    while(true){const auto stage=field(s,pc,"alloc_stage",false);const bool attacker=text_is(s,stage,"attacker"),defender=text_is(s,stage,"defender");if(!attacker&&!defender)return false;
        const auto role=attacker?"attacker":"defender",key=attacker?"atk_hits":"def_hits";const auto region_node=field(s,pc,(std::string(role)+"_region").c_str());const std::string region(text(s,region_node));const auto side_node=field(s,pc,(std::string(role)+"_side").c_str());const int side=enum_is(s,side_node,"game_state.Side","FREE_PEOPLES")?0:1;
        const auto old=field(s,pc,key,false);const auto hits=ctx_none(s,old)?0:typed(s,old,WOTR_INT,"Deferred hits must be integer").integer;const auto remaining=ca_forced(s,tables,region,side,hits);cp_put(s,pc,key,rr_scalar(s,WOTR_INT,remaining));
        if(remaining>0){cp_put(s,pc,"phase",rr_string(s,"casualty_allocation"));cp_put(s,pc,"deciding_side",side_node);return true;}
        cp_put(s,pc,"alloc_stage",attacker?rr_string(s,"defender"):rr_scalar(s,WOTR_NONE));
    }
}
