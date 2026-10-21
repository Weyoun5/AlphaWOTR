// Ordinary muster family, including pending second recruit and its pass.
// Card abilities (including Saruman's alternatives) remain separate actions.
uint64_t mu_enum(Snapshot& s,std::string_view type,std::string_view member){
    for(uint64_t i=0;i<s.nodes.size();++i)if(enum_is(s,i,type,member))return i;
    const auto cls=rr_string(s,type),name=rr_string(s,member),offset=s.edges.size();
    s.edges.push_back(cls);s.edges.push_back(name);const auto n=s.nodes.size();
    s.nodes.push_back({WOTR_ENUM,0,0,0,offset,2});return n;
}
uint64_t mu_object(Snapshot& s,std::string_view type,
                   std::initializer_list<std::pair<std::string_view,uint64_t>> fields){
    std::vector<uint64_t> values;values.reserve(fields.size()*2);
    for(auto& p:fields){values.push_back(rr_string(s,p.first));values.push_back(p.second);}
    const auto dict=rr_scalar(s,WOTR_DICT);set_edges(s,dict,values);
    const auto cls=rr_string(s,type),name=rr_string(s,"__dict__"),node=rr_scalar(s,WOTR_OBJECT);
    set_edges(s,node,{cls,name,dict});return node;
}
uint64_t mu_army(const Snapshot& s,std::string_view region){
    const auto armies=obs_attr(s,s.root,"armies");
    const auto& d=typed(s,armies,WOTR_DICT,"muster requires armies dictionary");
    for(uint64_t i=0;i<d.count;i+=2)
        if(sanitize_name(text(s,s.edges[d.offset+i]))==region)return s.edges[d.offset+i+1];
    return missing;
}
int64_t mu_available(const Snapshot& s,const Snapshot& tables,std::string_view nation,std::string_view kind){
    const auto maxes=ctx_nation_value(tables,field(tables,tables.root,"nation_max_figures"),nation);
    const auto cap_node=ctx_get(tables,maxes,kind);
    if(cap_node==missing)return 0;
    const auto cap=typed(tables,cap_node,WOTR_INT,"muster figure cap must be integer").integer;
    if(cap<=0)return 0;
    int64_t used=0;const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));
    for(size_t i=1;i<armies.size();i+=2){
        const auto army=armies[i];
        if(kind=="leader"){
            for(auto leader:child_edges(s,obs_attr(s,army,"leaders"))){
                const auto type=object_type(s,leader);
                if((type=="game_state.Leader" || type=="game_state.Nazgul") &&
                    enum_is(s,obs_attr(s,leader,"nation"),"game_state.Nation",nation))++used;
            }
        }else for(auto unit:child_edges(s,obs_attr(s,army,"units"))){
            if(enum_is(s,obs_attr(s,unit,"nation"),"game_state.Nation",nation)){
                const bool regular=enum_is(s,obs_attr(s,unit,"kind"),"game_state.UnitKind","REGULAR");
                if((kind=="regular")==regular)++used;
            }
        }
    }
    if(fp_nation(nation)){
        const auto casualties=obs_attr(s,s.root,"fp_casualties",false);
        if(truth(s,casualties)){
            const auto counts=ctx_nation_value(s,casualties,nation);
            if(truth(s,counts)){
                const auto count=ctx_get(s,counts,kind);
                if(count!=missing){
                    const auto n=typed(s,count,WOTR_INT,"muster casualties must be integer").integer;
                    require(n>=0 && n<=std::numeric_limits<int64_t>::max()-used,"invalid muster casualties");used+=n;
                }
            }
        }
    }
    return used>=cap?0:cap-used;
}
void mu_validate(const Snapshot& s,const Snapshot& tables,int32_t side,std::string_view nation,
                 std::string_view region,std::string_view kind){
    require(kind=="elite" || kind=="regular" || kind=="leader","unknown muster figure type");
    require((fp_nation(nation)?0:1)==side,"muster nation has wrong owner");
    require(political_view(s,nation,true).box==3,"muster nation must be At War");
    const auto meta=ctx_get(tables,field(tables,tables.root,"board_regions"),region);
    require(meta!=missing,"muster region is unknown");
    require(truth(tables,ctx_get(tables,meta,"is_city")) || truth(tables,ctx_get(tables,meta,"is_stronghold")) ||
            truth(tables,ctx_get(tables,meta,"is_town")),"muster needs a settlement");
    require(enum_is(tables,ctx_get(tables,field(tables,tables.root,"region_nations"),region),"game_state.Nation",nation),"muster needs the recruiting nation's settlement");
    require(!contains_string(s,obs_attr(s,s.root,"besieged_strongholds"),region),"muster cannot recruit into besieged stronghold");
    const auto enemy=ctx_side_value(s,obs_attr(s,s.root,"enemy_controlled"),1-side);
    require(!contains_string(s,enemy,region),"muster settlement is enemy controlled");
    const auto army=mu_army(s,region);
    if(kind!="leader"){
        require(army==missing || typed(s,obs_attr(s,army,"units"),WOTR_LIST,"muster units must be a list").count<10,"muster stacking limit reached");
    }else if(side==0){
        bool has_unit=false;
        if(army!=missing)for(auto unit:child_edges(s,obs_attr(s,army,"units")))has_unit|=ctx_unit_side(s,unit)==side;
        require(has_unit,"a lone Free Peoples leader cannot be recruited");
    }else require(truth(tables,ctx_get(tables,meta,"is_stronghold")),"Shadow leaders recruit at strongholds");
    require(mu_available(s,tables,nation,kind)>0,"muster reinforcement pool is exhausted");
}
void mu_place(Snapshot& s,std::string_view nation,std::string_view region,std::string_view kind,uint64_t target_override=missing){
    auto army=target_override==missing?mu_army(s,region):target_override;
    if(army==missing){
        const auto board=obs_attr(s,s.root,"board"),regions=obs_attr(s,board,"_regions");
        const auto region_node=ctx_get(s,regions,region);require(region_node!=missing,"muster Board.region is missing");
        require(object_type(s,region_node)=="board.Region" && text_is(s,obs_attr(s,region_node,"key"),region),
                "muster requires a canonical Board.region entry");
        const auto units=rr_scalar(s,WOTR_LIST),leaders=rr_scalar(s,WOTR_LIST);
        army=mu_object(s,"game_state.Army",{{"region",region_node},{"units",units},{"leaders",leaders}});
        const auto armies=obs_attr(s,s.root,"armies");auto entries=child_edges(s,armies);
        entries.push_back(rr_string(s,region));entries.push_back(army);set_edges(s,armies,entries);
    }
    const auto nat=mu_enum(s,"game_state.Nation",nation);
    uint64_t figure;
    if(kind=="leader")figure=mu_object(s,nation=="SAURON"?"game_state.Nazgul":"game_state.Leader",{{"nation",nat}});
    else{
        const auto unit_kind=mu_enum(s,"game_state.UnitKind",kind=="elite"?"ELITE":"REGULAR");
        figure=mu_object(s,"game_state.Unit",{{"nation",nat},{"kind",unit_kind}});
    }
    const auto list=obs_attr(s,army,kind=="leader"?"leaders":"units");
    auto figures=child_edges(s,list);figures.push_back(figure);set_edges(s,list,figures);
}
std::unique_ptr<Snapshot> apply_muster_action(const Snapshot& s,const Snapshot& tables,int32_t side,
        int64_t die_index,std::string_view nation,std::string_view region,std::string_view kind){
    require(side==0 || side==1,"muster side must be FP or SH");
    const auto region_key=sanitize_name(region);
    quiet_stable_action_gate(s,tables,true);ring_waiver(s,0);ring_waiver(s,1);
    const auto retained=ctx_side(s,obs_attr(s,s.root,"_ring_free_side",false));
    require(retained<0 || retained==side,"muster belongs to the retained ring actor");
    const auto pending=obs_attr(s,s.root,"pending_muster_die");
    const bool second=!ctx_none(s,pending),pass=kind=="pass";
    int64_t payment=die_index;
    if(second){
        payment=typed(s,pending,WOTR_INT,"pending muster index must be integer").integer;
        require(ctx_side(s,obs_attr(s,s.root,"pending_muster_side"))==side,"pending muster has another owner");
        require(pass || die_index==payment,"second muster must use the stored die");
        require(kind!="elite","second muster cannot recruit an elite");
        if(!pass)require(sanitize_name(text(s,obs_attr(s,s.root,"pending_muster_region")))!=region_key,"second muster needs a different settlement");
    }else require(!pass,"muster pass requires a pending recruitment");
    pa_payment_die(s,side,payment);
    if(!pass)mu_validate(s,tables,side,validate_nation_name(nation),region_key,kind);
    auto child=action_clone(s);auto& out=*child;
    rr_set_attr(out,out.root,"_casualty_operations",rr_scalar(out,WOTR_LIST));
    rr_set_attr(out,out.root,"_ring_free_side",rr_scalar(out,WOTR_NONE));
    if(!pass)mu_place(out,nation,region_key,kind);
    if(kind=="elite" || second){
        const auto die=pa_payment_die(out,side,payment);rr_set_attr(out,die,"used",rr_scalar(out,WOTR_BOOL,1));
        if(second)for(auto key:{"pending_muster_die","pending_muster_first_type","pending_muster_region","pending_muster_side"})
            rr_set_attr(out,out.root,key,rr_scalar(out,WOTR_NONE));
        rr_set_attr(out,out.root,"_battle_resolved_attacker",rr_scalar(out,WOTR_NONE));
    }else{
        rr_set_attr(out,out.root,"pending_muster_die",rr_scalar(out,WOTR_INT,die_index));
        rr_set_attr(out,out.root,"pending_muster_first_type",rr_string(out,kind));
        rr_set_attr(out,out.root,"pending_muster_region",rr_string(out,region));
        rr_set_attr(out,out.root,"pending_muster_side",mu_enum(out,"game_state.Side",side==0?"FREE_PEOPLES":"SHADOW"));
    }
    // Placement satisfies the same no-cleanup invariants (own settlement,
    // supplied figure, stack <=10, FP leader accompanied). No deferred tail opens.
    quiet_stable_action_gate(out,tables,true);
    return child;
}
