// Dead Men core; shared recruitment's explicit captured exception
// and the casualty continuation callback are prepared in the integration plan.
int cpm_settlement(const Snapshot&,const Snapshot&,std::string_view);
uint64_t crp_army(Snapshot&,std::string_view);
bool edm_supported(const Snapshot& t,uint64_t card){return card!=missing&&text_is(t,obs_attr(t,card,"card_name"),"Dead Men of Dunharrow");}
bool edm_pending(const Snapshot& s){const auto p=obs_attr(s,s.root,"pending_card_option",false);return !ctx_none(s,p)&&!truth(s,field(s,p,"card_casualty",false))&&text_is(s,field(s,p,"card_name",false),"Dead Men of Dunharrow");}
bool edm_required(std::string_view name){return name=="Strider"||name=="Aragorn"||name=="Aragorn - Heir to Isildur";}
uint64_t edm_options(Snapshot& s){
    const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));
    for(size_t i=0;i<armies.size();i+=2){std::vector<std::string> required,optional;std::set<std::string> seen;bool duplicate=false;
        for(auto comp:cm_companions(s,armies[i+1])){const std::string name(text(s,obs_attr(s,comp,"name")));duplicate|=!seen.insert(name).second;(edm_required(name)?required:optional).push_back(name);}
        if(required.empty())continue;require(required.size()==1&&!duplicate,"Dead Men requires one identifiable Strider/Aragorn group");std::sort(optional.begin(),optional.end());
        std::vector<uint64_t> groups;std::vector<std::string> selected{required.front()};
        auto choose=[&](auto&& recurse,size_t start,size_t remaining)->void{if(!remaining){std::vector<uint64_t> names;for(const auto& name:selected)names.push_back(rr_string(s,name));groups.push_back(cp_container(s,WOTR_LIST,names));return;}for(size_t n=start;n+remaining<=optional.size();++n){selected.push_back(optional[n]);recurse(recurse,n+1,remaining-1);selected.pop_back();}};
        for(size_t count=0;count<=optional.size();++count)choose(choose,0,count);
        std::vector<uint64_t> options;for(const char* region:{"erech","lamedon","pelargir"})for(auto group:groups)options.push_back(ev_dict(s,{{"target_region",rr_string(s,region)},{"companions",group}}));return cp_container(s,WOTR_LIST,options);
    }return cp_container(s,WOTR_LIST,{});
}
void edm_arrival(Snapshot& s,const Snapshot& t,std::string_view target,uint64_t choice){
    require(target=="erech"||target=="lamedon"||target=="pelargir","Dead Men destination invalid");const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));
    for(size_t i=0;i<armies.size();i+=2){const std::string origin=sanitize_name(text(s,armies[i]));if(origin==target)continue;const auto comps=cm_companions(s,armies[i+1]);bool found=false;for(auto c:comps)found|=edm_required(text(s,obs_attr(s,c,"name")));if(!found)continue;
        const auto requested=field(s,choice,"companions",false);std::vector<uint64_t> group;
        if(ctx_none(s,requested))group=comps;
        else{const auto names=child_edges(s,requested);std::set<std::string> want;bool leader=false;for(auto n:names){const std::string name(text(s,n));require(want.insert(name).second,"Dead Men companion group has duplicates");leader|=edm_required(name);}require(leader,"Dead Men group must include Strider/Aragorn");
            for(auto c:comps)if(want.count(std::string(text(s,obs_attr(s,c,"name")))))group.push_back(c);require(group.size()==want.size(),"Dead Men companion group unavailable");}
        cm_transfer(s,t,origin,target,group);break;
    }
    const auto army=mu_army(s,target);if(army==missing)return;std::set<std::string> nations;for(auto u:child_edges(s,obs_attr(s,army,"units")))if(ctx_unit_side(s,u)==1)nations.emplace(ctx_nation(s,obs_attr(s,u,"nation")));for(const auto& nation:nations)wt_defender_advance(s,nation,target);
}
void edm_recruit(Snapshot& s,const Snapshot& t,std::string_view target,uint64_t choice){
    const auto available=std::min<int64_t>(3,mu_available(s,t,"GONDOR","regular"));const auto selected=field(s,choice,"recruit_count",false);
    if(ctx_none(s,selected)&&available>0){std::vector<uint64_t> options;for(int64_t count=0;count<=available;++count)options.push_back(ev_dict(s,{{"target_region",rr_string(s,target)},{"_dmd_recruit_only",rr_scalar(s,WOTR_BOOL,1)},{"recruit_count",rr_scalar(s,WOTR_INT,count)}}));
        rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,"FREE_PEOPLES_Character_Dead_Men_of_Dunharrow")},{"card_name",rr_string(s,"Dead Men of Dunharrow")},{"side",rr_string(s,"fp")},{"decider",rr_string(s,"fp")},{"direct",rr_scalar(s,WOTR_BOOL,1)},{"options",cp_container(s,WOTR_LIST,options)}}));return;
    }
    const auto count=ctx_none(s,selected)?3:typed(s,selected,WOTR_INT,"Dead Men recruit count must be integer").integer;
    if(truth(s,field(s,choice,"_dmd_recruit_only",false)))require(count>=0&&count<=available,"Dead Men recruitment count unavailable");
    for(int64_t n=0;n<std::clamp<int64_t>(count,0,3);++n)er_place(s,t,"GONDOR",target,"regular",missing,true);
    const auto captured=ctx_side_value(s,obs_attr(s,s.root,"enemy_controlled"),1),army=mu_army(s,target);if(army==missing||!contains_string(s,captured,target))return;bool fp=false,shadow=false;for(auto u:child_edges(s,obs_attr(s,army,"units")))(ctx_unit_side(s,u)==0?fp:shadow)=true;if(fp&&!shadow)me_set_member(s,captured,target,false);
}
std::vector<std::string> edm_retreat_options(const Snapshot& s,const Snapshot& t,std::string_view target){
    std::vector<std::string> destinations;for(const auto& adjacent:fr_neighbors(s,target)){const auto key=sanitize_name(adjacent);if(cpm_settlement(s,t,key)==0)continue;const auto army=mu_army(s,key);bool fp=false;if(army!=missing)for(auto u:child_edges(s,obs_attr(s,army,"units")))fp|=ctx_unit_side(s,u)==0;if(!fp)destinations.push_back(key);}return destinations;
}
void edm_after_losses(Snapshot& s,const Snapshot& t,std::string_view target,uint64_t choice,void* rng){
    auto army=mu_army(s,target);std::vector<uint64_t> units,leaders;if(army!=missing){for(auto u:child_edges(s,obs_attr(s,army,"units")))if(ctx_unit_side(s,u)==1)units.push_back(u);for(auto l:child_edges(s,obs_attr(s,army,"leaders")))if(am_leader_side(s,l)==1)leaders.push_back(l);}
    if(!units.empty()){
        const auto destinations=edm_retreat_options(s,t,target);const auto requested=field(s,choice,"destination",false);std::string destination;
        if(!ctx_none(s,requested)){const std::string want(text(s,requested));if(std::find(destinations.begin(),destinations.end(),want)!=destinations.end())destination=want;else if(!destinations.empty())destination=destinations.front();}
        else if(destinations.size()>1){std::vector<uint64_t> options;for(const auto& dest:destinations)options.push_back(ev_dict(s,{{"destination",rr_string(s,dest)},{"_dmd_target",rr_string(s,target)}}));rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,"FREE_PEOPLES_Character_Dead_Men_of_Dunharrow")},{"card_name",rr_string(s,"Dead Men of Dunharrow")},{"side",rr_string(s,"fp")},{"decider",rr_string(s,"sh")},{"direct",rr_scalar(s,WOTR_BOOL,1)},{"options",cp_container(s,WOTR_LIST,options)}}));return;}
        else if(!destinations.empty())destination=destinations.front();
        if(!destination.empty()){
            const auto dst=crp_army(s,destination);const auto unit_list=obs_attr(s,army,"units"),leader_list=obs_attr(s,army,"leaders");
            for(auto u:units){auto values=child_edges(s,unit_list);const auto it=std::find_if(values.begin(),values.end(),[&](auto v){return tail_equal(s,v,u);});require(it!=values.end(),"Dead Men retreat unit disappeared");values.erase(it);set_edges(s,unit_list,values);am_extend(s,dst,"units",{u});}
            for(auto l:leaders){auto values=child_edges(s,leader_list);const auto it=std::find_if(values.begin(),values.end(),[&](auto v){return tail_equal(s,v,l);});require(it!=values.end(),"Dead Men retreat leader disappeared");values.erase(it);set_edges(s,leader_list,values);am_extend(s,dst,"leaders",{l});}
            int64_t count=0;for(auto u:child_edges(s,obs_attr(s,dst,"units")))count+=ctx_unit_side(s,u)==1;if(count>su_cap(s,t,destination,1)){auto tracked=obs_attr(s,s.root,"_overstack_regions",false);if(ctx_none(s,tracked)){tracked=cp_container(s,WOTR_SET,{});rr_set_attr(s,s.root,"_overstack_regions",tracked);}me_set_member(s,tracked,destination,true);}crs_drop_empty(s,target,army);
        }else{const auto list=obs_attr(s,army,"units");auto values=child_edges(s,list);values.erase(std::remove_if(values.begin(),values.end(),[&](auto u){return ctx_unit_side(s,u)==1;}),values.end());set_edges(s,list,values);cbe_reconcile(s,target,1,rng);}
    }
    edm_recruit(s,t,target,choice);
}
void edm_effect(Snapshot& s,const Snapshot& t,std::string_view id,uint64_t choice,void* rng,am_Builder& events,std::vector<uint64_t>& rows){
    const auto specified=field(s,choice,"target_region",false),retreat=field(s,choice,"destination",false);const std::string target=sanitize_name(ctx_none(s,retreat)?(ctx_none(s,specified)?"erech":std::string(text(s,specified))):std::string(text(s,field(s,choice,"_dmd_target"))));
    require(target=="erech"||target=="lamedon"||target=="pelargir","Dead Men destination invalid");
    if(truth(s,field(s,choice,"_dmd_recruit_only",false))){edm_recruit(s,t,target,choice);return;}
    if(ctx_none(s,retreat)&&!truth(s,field(s,choice,"_dmd_cas_resumed",false))){edm_arrival(s,t,target,choice);const auto army=mu_army(s,target);bool shadow=false;if(army!=missing)for(auto u:child_edges(s,obs_attr(s,army,"units")))shadow|=ctx_unit_side(s,u)==1;
        if(shadow){uint32_t roll;hf_roll(rng,&roll,1,"card effect");rows.push_back(events.row(-1,"dice_roll",{events.str("faces"),events.container(WOTR_LIST,{events.integer(roll)}),events.str("ctx"),events.str("card effect roll"),events.str("side"),rr_scalar(events.out,WOTR_NONE)}));if(!ecc_route(s,t,target,1,roll,id,"Dead Men of Dunharrow",rng))return;}
    }
    edm_after_losses(s,t,target,choice,rng);
}
bool edm_cross_side(const Snapshot& s){if(!edm_pending(s))return false;const auto p=obs_attr(s,s.root,"pending_card_option");return truth(s,field(s,p,"direct",false))&&text_is(s,field(s,p,"decider",false),"sh");}
void apply_dead_men_event_inplace(Snapshot& s,const Snapshot& t,void* rng,int side,int64_t index,std::string_view id,bool free_ents,am_Builder& ef_events,std::vector<uint64_t>& ef_annotations){
    const auto& source=s;ef_validate_free(source,t,side,index,id,free_ents);
    const int ef_marker=ctx_side(source,obs_attr(source,source.root,"_battle_resolved_attacker",false));const auto ef_unused=ef_unused_dice(source);

    require(side==0,"Dead Men belongs to FP");quiet_action_gate(source,t,false,false,false,false,false,false,false,false,false,false,free_ents,true,false,false);ring_waiver(source,0);ring_waiver(source,1);const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==0,"Dead Men violates retained Ring actor");
    const auto card=ev_card(t,id);require(edm_supported(t,card)&&contains_string(source,tc_hand(source,0),id),"Dead Men absent from FP hand");require(ev_requirement(source,t,card),"Dead Men needs Strider/Aragorn in Rohan");const auto die=free_ents?missing:ev_payment(source,t,0,index,card);const std::string face(free_ents?std::string_view{}:enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace"));
    
    rr_set_attr(s,s.root,"_casualty_operations",cp_container(s,WOTR_LIST,{}));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));if(!free_ents)rr_set_attr(s,ev_payment(s,t,0,index,card),"used",rr_scalar(s,WOTR_BOOL,1));const auto options=edm_options(s);
    require(child_edges(s,options).size()>=3,"Dead Men canonical companion group unavailable");rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,"Dead Men of Dunharrow")},{"side",rr_string(s,"fp")},{"decider",rr_string(s,"fp")},{"options",options},{"die_face",ef_pending_face_node(s,face)},{"deck",rr_string(s,"character")}}));
    cau_finish(s,t,rng,free_ents?ef_final_marker(s,ef_marker,ef_unused):-1,ef_events,ef_annotations);
}
fm_Result apply_dead_men_event(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index,std::string_view id){
    require(side==0,"Dead Men belongs to FP");quiet_stable_action_gate(source,t);ring_waiver(source,0);ring_waiver(source,1);const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==0,"Dead Men violates retained Ring actor");
    const auto card=ev_card(t,id);require(edm_supported(t,card)&&contains_string(source,tc_hand(source,0),id),"Dead Men absent from FP hand");require(ev_requirement(source,t,card),"Dead Men needs Strider/Aragorn in Rohan");const auto die=ev_payment(source,t,0,index,card);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace"));
    
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());
    am_Builder ef_events;std::vector<uint64_t> ef_annotations;
    apply_dead_men_event_inplace(*result.state,t,result.random.get(),side,index,id,false,ef_events,ef_annotations);
    ef_events.out.root=ef_events.container(WOTR_LIST,ef_annotations);result.events=std::make_unique<Snapshot>(std::move(ef_events.out));return result;
}
uint64_t edm_validate_pending(const Snapshot& s,const Snapshot& t,int side){
    require(edm_pending(s),"No Dead Men choice pending");const auto p=obs_attr(s,s.root,"pending_card_option");const auto id=text(s,field(s,p,"card_id"));const bool direct=truth(s,field(s,p,"direct",false)),shadow=edm_cross_side(s);
    require(edm_supported(t,ev_card(t,id))&&text_is(s,field(s,p,"side"),"fp")&&text_is(s,field(s,p,"decider"),shadow?"sh":"fp")&&side==(shadow?1:0),"Dead Men identity or choice owner mismatch");
    if(!direct){require(text_is(s,field(s,p,"deck"),"character"),"Dead Men target card or deck invalid");const auto face=ef_pending_face(s,p);require(face.empty()||face=="CHARACTER"||face=="EVENT"||face=="WILL_OF_WEST","Dead Men funding invalid");}
    const auto options=child_edges(s,field(s,p,"options"));require(options.size()>1,"Dead Men choice needs multiple options");std::set<std::string> regions;std::set<int64_t> counts;
    for(auto option:options){const auto count=typed(s,option,WOTR_DICT,"Dead Men option must be dictionary").count;
        if(shadow){require(count==4,"Dead Men retreat option malformed");const std::string target(text(s,field(s,option,"_dmd_target"))),destination(text(s,field(s,option,"destination")));require((target=="erech"||target=="lamedon"||target=="pelargir")&&!destination.empty()&&sanitize_name(destination)==destination&&regions.insert(destination).second,"Dead Men retreat option invalid");}
        else{const std::string target(text(s,field(s,option,"target_region")));require(target=="erech"||target=="lamedon"||target=="pelargir","Dead Men target invalid");
            if(direct){require(count==6&&truth(s,field(s,option,"_dmd_recruit_only",false)),"Dead Men recruitment option malformed");const auto n=typed(s,field(s,option,"recruit_count"),WOTR_INT,"Dead Men count must be integer").integer;require(n>=0&&n<=3&&counts.insert(n).second,"Dead Men recruit option invalid");}
            else{require(count==4,"Dead Men initial option malformed");const auto names=child_edges(s,field(s,option,"companions"));std::set<std::string> selected;bool required=false;for(auto name:names){const std::string value(text(s,name));require(selected.insert(value).second,"Dead Men group duplicate");required|=edm_required(value);}require(required,"Dead Men group omits Strider/Aragorn");}
        }
    }return p;
}
std::unique_ptr<Snapshot> native_dead_men_choices(const Snapshot& source,const Snapshot& t,int side){
    edm_validate_pending(source,t,side);Snapshot s(source);const auto p=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t index=0;for(auto choice:child_edges(s,field(s,p,"options")))rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",choice}})}));s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
fm_Result apply_dead_men_choice(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index){
    const bool free_ents=ef_pending_is_free(source,obs_attr(source,source.root,"pending_card_option",false));

    const auto old=edm_validate_pending(source,t,side);const auto options=child_edges(source,field(source,old,"options"));require(index>=0&&static_cast<uint64_t>(index)<options.size(),"Dead Men index invalid");const bool direct=truth(source,field(source,old,"direct",false));const std::string id(text(source,field(source,old,"card_id"))),face=direct?std::string{}:std::string(ef_pending_face(source,old));const int marker=ctx_side(source,obs_attr(source,source.root,"_battle_resolved_attacker",false));
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;const auto p=obs_attr(s,s.root,"pending_card_option"),choice=child_edges(s,field(s,p,"options"))[index];
    rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));quiet_action_gate(s,t,false,false,false,false,false,false,false,false,false,false,(direct||free_ents),true,false,false,direct);ring_waiver(s,0);ring_waiver(s,1);rr_set_attr(s,s.root,"_casualty_operations",cp_container(s,WOTR_LIST,{}));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    const auto queued=obs_attr(s,s.root,"_gandalf_draw_pending",false);if(direct&&!ctx_none(s,queued))require((text_is(s,field(s,queued,"source_card_id"),id)||ef_ents_queued_grant(s,t,queued))&&text_is(s,field(s,queued,"deck"),"character"),"Dead Men queued Gandalf grant identity invalid");
    am_Builder events;std::vector<uint64_t> rows;if(direct){edm_effect(s,t,id,choice,result.random.get(),events,rows);su_resolve_overstacks(s,t);}else if(ev_requirement(s,t,ev_card(t,id))){edm_effect(s,t,id,choice,result.random.get(),events,rows);er_commit(s,t,id,"character",face);}
    cau_finish(s,t,result.random.get(),marker,events,rows);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
