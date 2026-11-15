// One learned return to reinforcements, including legitimate stale pendings.
std::string eo_nation(std::string_view value){
    for(const char* nation:{"ELVES","DWARVES","GONDOR","THE_NORTH","ROHAN","ISENGARD","SAURON","SOUTHRON_EASTERLINGS"})if(su_nation_value(nation)==value)return nation;
    throw std::invalid_argument("Overstack choice nation invalid");
}
uint64_t eo_pending(const Snapshot& s,int side){
    require(side==0 || side==1,"Overstack actor invalid");const auto p=obs_attr(s,s.root,"pending_card_option",false);
    require(!ctx_none(s,p) && text_is(s,field(s,p,"card_id",false),"__overstack__") && text_is(s,field(s,p,"card_name",false),"__overstack__") && truth(s,field(s,p,"direct",false)),"No direct overstack choice pending");
    require(text_is(s,field(s,p,"side",false),side==0?"fp":"sh") && text_is(s,field(s,p,"decider",false),side==0?"fp":"sh"),"Overstack belongs to another actor");
    const auto options=child_edges(s,field(s,p,"options"));require(options.size()>=2,"Overstack needs multiple original alternatives");
    for(auto option:options){
        require(typed(s,option,WOTR_DICT,"Overstack option must be dict").count==2,"Overstack option has extra fields");
        const auto spec=field(s,option,"overstack_remove");require(typed(s,spec,WOTR_DICT,"Overstack spec must be dict").count==6,"Overstack spec shape invalid");
        const auto nation=eo_nation(text(s,field(s,spec,"nation")));require((fp_nation(nation)?0:1)==side,"Overstack option nation belongs to another actor");
        require(text_is(s,field(s,spec,"kind"),"regular") || text_is(s,field(s,spec,"kind"),"elite"),"Overstack figure kind invalid");
        text(s,field(s,spec,"region"));
    }
    return p;
}
std::unique_ptr<Snapshot> native_overstack_choice_actions(const Snapshot& source,int side){
    eo_pending(source,side);auto out=std::make_unique<Snapshot>(source);auto& s=*out;const auto p=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t index=0;
    for(auto option:child_edges(s,field(s,p,"options"))){
        const auto params=ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",option}});
        rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),params}));
    }
    s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
std::unique_ptr<Snapshot> apply_overstack_choice(const Snapshot& source,const Snapshot& t,int side,int64_t index){
    const auto p=eo_pending(source,side);const auto options=child_edges(source,field(source,p,"options"));require(index>=0 && static_cast<uint64_t>(index)<options.size(),"Overstack choice index invalid");
    // This narrow guard is supplied by the shared integration: ordinary quiet
    // invariants still apply, with only the validated overflow/queued bonus/
    // coexisting mandatory discard allowed. A validated stable siege may
    // temporarily exceed its garrison cap only at a tracked overflow region.
    quiet_action_gate(source,t,false,false,false,true,false,false,false,false,false,false,true,true,true);
    const auto spec=field(source,options[index],"overstack_remove");const std::string raw(text(source,field(source,spec,"region"))),region=sanitize_name(raw);
    const auto nation=eo_nation(text(source,field(source,spec,"nation")));const std::string kind(text(source,field(source,spec,"kind")));
    auto child=action_clone(source);auto& s=*child;rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    const auto army=mu_army(s,region);
    if(army!=missing){
        const auto units=obs_attr(s,army,"units");auto entries=child_edges(s,units);size_t own=0;for(auto unit:entries)own+=ctx_unit_side(s,unit)==side;
        if(own<=static_cast<size_t>(su_cap(s,t,region,side))){
            const auto tracked=obs_attr(s,s.root,"_overstack_regions",false);if(truth(s,tracked)){me_set_member(s,tracked,region,false);me_set_member(s,tracked,raw,false);}
        }else{
            for(auto it=entries.begin();it!=entries.end();++it)if(ctx_unit_side(s,*it)==side && enum_is(s,obs_attr(s,*it,"nation"),"game_state.Nation",nation) && enum_is(s,obs_attr(s,*it,"kind"),"game_state.UnitKind",kind=="elite"?"ELITE":"REGULAR")){
                entries.erase(it);set_edges(s,units,entries);break;
            }
        }
    }
    su_resolve_overstacks(s,t);tc_open_hand_limit_discard(s);epa_maybe(s);su_cleanup(s,t);cau_gandalf(s);
    // A die-less reaction retains the battle-routing marker. Returned pieces
    // are reinforcements: no casualty counter or casualty operation is added.
    return child;
}
