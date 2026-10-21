// Almanac p35: a short shared Regular supply remains a regional choice.
// Existing complete-supply paths and their payloads remain unchanged.
struct esra_Allocation {
    std::vector<std::string> regions;
    std::vector<int64_t> counts;
};
struct esra_Spec {
    std::string key="regions",nation="SAURON",name;
    std::vector<std::pair<std::string,int64_t>> limits;
    int64_t total=0,capacity=0;
};
esra_Spec esra_spec(const Snapshot& s,const Snapshot& t,std::string_view id){
    esra_Spec spec;const auto card=ev_card(t,id);if(card==missing)return spec;
    spec.name=std::string(text(t,obs_attr(t,card,"card_name")));
    if(spec.name=="Orcs Multiplying Again")spec.limits={{"dol guldur",3},{"mount gundabad",3}};
    else if(spec.name=="Monsters Roused")spec.limits={{"angmar",1},{"ettenmoors",1},{"weather hills",1}};
    else if(spec.name=="A New Power is Rising"){
        spec.nation="ISENGARD";spec.limits={{"north dunland",2},{"orthanc",2-std::min<int64_t>(2,mu_available(s,t,"ISENGARD","elite"))},{"south dunland",2}};
    }else if(spec.name=="Many Kings to the Service of Mordor"){
        spec.key="settlements";spec.nation="SOUTHRON_EASTERLINGS";for(const auto& r:emk_regions(s,t))spec.limits.emplace_back(r,2);
    }else if(spec.name=="Pits of Mordor"){
        spec.key="strongholds";for(const auto& r:epr_regions(s,t))spec.limits.emplace_back(r,2);
    }else return spec;
    spec.limits.erase(std::remove_if(spec.limits.begin(),spec.limits.end(),[&](const auto& p){return p.second<=0||!er_region_open(s,p.first,1,true);}),spec.limits.end());
    std::sort(spec.limits.begin(),spec.limits.end());
    std::vector<int64_t> capacities;for(const auto& p:spec.limits)capacities.push_back(p.second);
    std::sort(capacities.rbegin(),capacities.rend());for(size_t i=0;i<std::min<size_t>(3,capacities.size());++i)spec.capacity+=capacities[i];
    spec.total=std::min<int64_t>(spec.capacity,mu_available(s,t,spec.nation,"regular"));return spec;
}
std::vector<esra_Allocation> esra_options(const esra_Spec& spec){
    std::vector<esra_Allocation> out;if(spec.total<=0||spec.total>=spec.capacity)return out;
    esra_Allocation selected;
    auto visit=[&](auto&& self,size_t at,int64_t remaining)->void{
        if(at==spec.limits.size()){if(!remaining)out.push_back(selected);return;}
        self(self,at+1,remaining);
        if(selected.regions.size()==3)return;
        for(int64_t n=1;n<=std::min<int64_t>(remaining,spec.limits[at].second);++n){
            selected.regions.push_back(spec.limits[at].first);selected.counts.push_back(n);
            self(self,at+1,remaining-n);selected.regions.pop_back();selected.counts.pop_back();
        }
    };visit(visit,0,spec.total);return out;
}
bool esra_has_choices(const Snapshot& s,const Snapshot& t,std::string_view id){const auto spec=esra_spec(s,t,id);return spec.total>0&&spec.total<spec.capacity;}
uint64_t esra_option(Snapshot& s,const esra_Spec& spec,const esra_Allocation& choice){
    std::vector<uint64_t> regions,counts;for(const auto& r:choice.regions)regions.push_back(rr_string(s,r));for(auto n:choice.counts)counts.push_back(rr_scalar(s,WOTR_INT,n));
    return ev_dict(s,{{spec.key.c_str(),cp_container(s,WOTR_LIST,regions)},{"recruit_counts",cp_container(s,WOTR_LIST,counts)}});
}
void esra_effect(Snapshot& s,const Snapshot& t,const esra_Spec& spec,const esra_Allocation& choice){
    for(size_t i=0;i<choice.regions.size();++i)for(int64_t n=0;n<choice.counts[i];++n)
        require(er_place(s,t,spec.nation,choice.regions[i],"regular"),"Validated recruitment allocation could not place its unit");
    if(spec.name=="Monsters Roused")er_place(s,t,"SAURON","trollshaws","elite");
    if(spec.name=="A New Power is Rising"){
        const auto elites=std::min<int64_t>(2,mu_available(s,t,"ISENGARD","elite"));
        for(int64_t n=0;n<elites;++n)er_place(s,t,"ISENGARD","orthanc","elite");
    }
}
std::unique_ptr<Snapshot> esra_begin(const Snapshot& source,const Snapshot& t,int side,int64_t index,std::string_view id){
    require(side==1,"Shadow allocation belongs to Shadow");quiet_action_gate(source,t,false,false,false,false,false,false,false,false,false,false,false,true);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==side,"Allocation violates retained Ring actor");
    const auto card=ev_card(t,id);require(card!=missing&&contains_string(source,tc_hand(source,1),id)&&ev_requirement(source,t,card),"Allocation card absent or requirement false");
    const auto spec=esra_spec(source,t,id);const auto choices=esra_options(spec);require(!choices.empty(),"No shortage allocation available");
    const auto die=ev_payment(source,t,side,index,card);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace"));
    auto child=action_clone(source);auto& s=*child;rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));rr_set_attr(s,ev_payment(s,t,side,index,card),"used",rr_scalar(s,WOTR_BOOL,1));
    if(choices.size()>1){
        std::vector<uint64_t> options;for(const auto& choice:choices)options.push_back(esra_option(s,spec,choice));
        rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,spec.name)},{"side",rr_string(s,"sh")},{"decider",rr_string(s,"sh")},{"options",cp_container(s,WOTR_LIST,options)},{"die_face",rr_string(s,face)},{"deck",rr_string(s,"strategy")}}));
        tc_open_hand_limit_discard(s);rr_set_attr(s,s.root,"_battle_resolved_attacker",rr_scalar(s,WOTR_NONE));
    }else{esra_effect(s,t,spec,choices.front());esr_commit(s,t,id,"strategy",face);}
    return child;
}
bool esra_pending(const Snapshot& s){
    const auto p=obs_attr(s,s.root,"pending_card_option",false);if(ctx_none(s,p))return false;
    const auto options=field(s,p,"options",false);if(ctx_none(s,options))return false;
    const auto rows=child_edges(s,options);return !rows.empty()&&field(s,rows[0],"recruit_counts",false)!=missing;
}
std::unique_ptr<Snapshot> native_shadow_allocation_choices(const Snapshot& source,const Snapshot& t,int side){
    require(side==1&&esra_pending(source),"No Shadow allocation pending");auto result=std::make_unique<Snapshot>(source);auto& s=*result;
    const auto p=obs_attr(s,s.root,"pending_card_option");require(text_is(s,field(s,p,"side"),"sh")&&text_is(s,field(s,p,"decider"),"sh"),"Allocation owner invalid");
    const auto spec=esra_spec(s,t,text(s,field(s,p,"card_id")));const auto expected=esra_options(spec);const auto actual=child_edges(s,field(s,p,"options"));require(expected.size()==actual.size(),"Allocation options stale");
    std::vector<uint64_t> rows;int64_t index=0;
    for(size_t i=0;i<actual.size();++i){
        const auto wanted=esra_option(s,spec,expected[i]);require(ap_equal(ap_freeze(s,wanted),ap_freeze(s,actual[i])),"Allocation option changed");
        rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",actual[i]}})}));
    }
    s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
std::unique_ptr<Snapshot> apply_shadow_allocation_choice(const Snapshot& source,const Snapshot& t,int side,int64_t index){
    require(side==1&&esra_pending(source),"No Shadow allocation pending");native_shadow_allocation_choices(source,t,side);
    const auto p=obs_attr(source,source.root,"pending_card_option");const std::string id(text(source,field(source,p,"card_id"))),face(text(source,field(source,p,"die_face")));
    const auto spec=esra_spec(source,t,id);const auto choices=esra_options(spec);require(index>=0&&static_cast<uint64_t>(index)<choices.size(),"Allocation choice out of range");
    require(contains_string(source,tc_hand(source,1),id),"Allocation card absent");
    auto child=action_clone(source);auto& s=*child;const auto marker=obs_attr(s,s.root,"_battle_resolved_attacker",false);
    rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));quiet_action_gate(s,t,false,false,false,false,false,false,false,false,false,false,false,true);ring_waiver(s,0);ring_waiver(s,1);
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    esra_effect(s,t,spec,choices[index]);esr_commit(s,t,id,"strategy",face);if(marker!=missing)rr_set_attr(s,s.root,"_battle_resolved_attacker",marker);return child;
}
