// exact paid Grey Company upgrade and compulsory draws.
bool egc_supported(const Snapshot& t,uint64_t card){return card!=missing&&text_is(t,obs_attr(t,card,"card_name"),"The Grey Company");}
bool egc_pending(const Snapshot& s){const auto p=obs_attr(s,s.root,"pending_card_option",false);return !ctx_none(s,p)&&text_is(s,field(s,p,"card_name",false),"The Grey Company");}
std::string egc_nation_value(std::string_view name){return name=="THE_NORTH"?"the north":ev_lower(name);}
uint64_t egc_options(Snapshot& s,const Snapshot& t){
    std::vector<uint64_t> options;const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));
    for(size_t i=0;i<armies.size();i+=2){
        bool strider=false;for(auto f:child_edges(s,obs_attr(s,armies[i+1],"leaders")))if(object_type(s,f)=="game_state.Companion"){
            const auto name=obs_attr(s,f,"name");strider|=text_is(s,name,"Strider")||text_is(s,name,"Aragorn")||text_is(s,name,"Aragorn - Heir to Isildur");
        }
        if(!strider)continue;std::set<std::string> nations;
        for(auto u:child_edges(s,obs_attr(s,armies[i+1],"units"))){const std::string nation(enum_member(s,obs_attr(s,u,"nation"),"game_state.Nation"));if(fp_nation(nation)&&enum_is(s,obs_attr(s,u,"kind"),"game_state.UnitKind","REGULAR"))nations.insert(nation);}
        // Enum names and source Nation.value have the same order for FP nations.
        for(const auto& nation:nations)if(mu_available(s,t,nation,"elite")>0)options.push_back(ev_dict(s,{{"strider_region",rr_string(s,sanitize_name(text(s,armies[i])))},{"nation",rr_string(s,egc_nation_value(nation))}}));
    }return cp_container(s,WOTR_LIST,options);
}
void egc_effect(Snapshot& s,const Snapshot& t,uint64_t choice,am_Builder& events,std::vector<uint64_t>& rows){
    if(!ctx_none(s,choice)){
        const auto army=mu_army(s,sanitize_name(text(s,field(s,choice,"strider_region"))));
        if(army!=missing){const auto units=obs_attr(s,army,"units");auto members=child_edges(s,units);const std::string selected=ev_lower(text(s,field(s,choice,"nation")));
            for(size_t i=0;i<members.size();++i){const auto u=members[i],nation=obs_attr(s,u,"nation");const std::string name(enum_member(s,nation,"game_state.Nation"));
                if(fp_nation(name)&&enum_is(s,obs_attr(s,u,"kind"),"game_state.UnitKind","REGULAR")&&egc_nation_value(name)==selected){
                    if(mu_available(s,t,name,"elite")>0){
                        // Permanent FP casualty only: allocator removal would add
                        // an operation annotation and reorder the source list.
                        cbp_book(s,nation,"regular");members[i]=mu_object(s,"game_state.Unit",{{"nation",nation},{"kind",mu_enum(s,"game_state.UnitKind","ELITE")}});set_edges(s,units,members);
                    }break;
                }
            }
        }
    }
    for(int i=0;i<2&&truth(s,obs_attr(s,s.root,"fp_strat_deck",false));++i){const auto card=tc_draw_event_card(s,0,"strategy");bc_settle_draw(s,0,"strategy");rows.push_back(events.row(-1,"card_drawn",{events.str("deck"),events.str("strategy"),events.str("card_id"),events.str(text(s,card)),events.str("side"),events.str("FP")}));}
}
void apply_grey_company_event_inplace(Snapshot& s,const Snapshot& t,void* rng,int side,int64_t index,std::string_view id,bool free_ents,am_Builder& ef_events,std::vector<uint64_t>& ef_annotations){
    const auto& source=s;ef_validate_free(source,t,side,index,id,free_ents);
    const int ef_marker=ctx_side(source,obs_attr(source,source.root,"_battle_resolved_attacker",false));const auto ef_unused=ef_unused_dice(source);

    require(side==0,"The Grey Company belongs to FP");quiet_action_gate(source,t,false,false,false,false,false,false,false,false,false,false,(false||free_ents),true);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==side,"Grey Company violates retained Ring actor");
    const auto card=ev_card(t,id);require(egc_supported(t,card)&&ctx_side(t,obs_attr(t,card,"side"))==0&&contains_string(source,tc_hand(source,0),id),"Grey Company absent from FP hand");require(ev_requirement(source,t,card),"Grey Company requirement false");
    const auto die=free_ents?missing:ev_payment(source,t,0,index,card);const std::string face(free_ents?std::string_view{}:enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace"));
    
    rr_set_attr(s,s.root,"_casualty_operations",cp_container(s,WOTR_LIST,{}));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));if(!free_ents)rr_set_attr(s,ev_payment(s,t,0,index,card),"used",rr_scalar(s,WOTR_BOOL,1));
    const auto options=egc_options(s,t);const auto choices=child_edges(s,options);
    if(choices.size()>1){rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,"The Grey Company")},{"side",rr_string(s,"fp")},{"decider",rr_string(s,"fp")},{"options",options},{"die_face",ef_pending_face_node(s,face)},{"deck",rr_string(s,"character")}}));tc_open_hand_limit_discard(s);}
    else{egc_effect(s,t,choices.empty()?missing:choices.front(),ef_events,ef_annotations);er_commit(s,t,id,"character",face);}
    cau_finish(s,t,rng,free_ents?ef_final_marker(s,ef_marker,ef_unused):-1,ef_events,ef_annotations);
}
fm_Result apply_grey_company_event(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index,std::string_view id){
    require(side==0,"The Grey Company belongs to FP");quiet_stable_action_gate(source,t);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==side,"Grey Company violates retained Ring actor");
    const auto card=ev_card(t,id);require(egc_supported(t,card)&&ctx_side(t,obs_attr(t,card,"side"))==0&&contains_string(source,tc_hand(source,0),id),"Grey Company absent from FP hand");require(ev_requirement(source,t,card),"Grey Company requirement false");
    const auto die=ev_payment(source,t,0,index,card);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace"));
    
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());
    am_Builder ef_events;std::vector<uint64_t> ef_annotations;
    apply_grey_company_event_inplace(*result.state,t,result.random.get(),side,index,id,false,ef_events,ef_annotations);
    ef_events.out.root=ef_events.container(WOTR_LIST,ef_annotations);result.events=std::make_unique<Snapshot>(std::move(ef_events.out));return result;
}
uint64_t egc_validate_pending(const Snapshot& s,const Snapshot& t){
    require(egc_pending(s),"No Grey Company choice pending");const auto p=obs_attr(s,s.root,"pending_card_option"),card=ev_card(t,text(s,field(s,p,"card_id")));
    require(egc_supported(t,card)&&text_is(s,field(s,p,"side"),"fp")&&text_is(s,field(s,p,"decider"),"fp")&&!truth(s,field(s,p,"direct",false)),"Grey Company pending identity or owner mismatch");
    require(text_is(s,field(s,p,"deck"),"character")&&(ef_pending_is_free(s,p)||text_is(s,field(s,p,"die_face"),"CHARACTER")||text_is(s,field(s,p,"die_face"),"EVENT")||text_is(s,field(s,p,"die_face"),"WILL_OF_WEST")),"Grey Company pending funding invalid");
    const auto choices=child_edges(s,field(s,p,"options"));require(choices.size()>1,"Grey Company modal options invalid");std::set<std::pair<std::string,std::string>> seen;
    for(auto choice:choices){require(typed(s,choice,WOTR_DICT,"Grey Company choice must be dictionary").count==4,"Grey Company stored choice shape invalid");const std::string region(text(s,field(s,choice,"strider_region"))),nation(text(s,field(s,choice,"nation")));
        require(!region.empty()&&sanitize_name(region)==region&&(nation=="dwarves"||nation=="elves"||nation=="gondor"||nation=="rohan"||nation=="the north")&&seen.insert({region,nation}).second,"Grey Company stored choice malformed");
    }return p;
}
std::unique_ptr<Snapshot> native_grey_company_choices(const Snapshot& source,const Snapshot& t,int side){
    require(side==0,"Grey Company choice belongs to FP");egc_validate_pending(source,t);Snapshot s(source);const auto p=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t index=0;
    for(auto choice:child_edges(s,field(s,p,"options")))rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",choice}})}));s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
fm_Result apply_grey_company_choice(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index){
    const bool free_ents=ef_pending_is_free(source,obs_attr(source,source.root,"pending_card_option",false));

    require(side==0,"Grey Company choice belongs to FP");const auto pending=egc_validate_pending(source,t),options=field(source,pending,"options");require(index>=0&&static_cast<uint64_t>(index)<child_edges(source,options).size(),"Grey Company choice index invalid");
    const std::string id(text(source,field(source,pending,"card_id"))),face(ef_pending_face(source,pending));const int marker=ctx_side(source,obs_attr(source,source.root,"_battle_resolved_attacker",false));
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;const auto p=obs_attr(s,s.root,"pending_card_option"),choice=child_edges(s,field(s,p,"options"))[index];
    rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));quiet_action_gate(s,t,false,false,false,false,false,false,false,false,false,false,(false||free_ents),true);ring_waiver(s,0);ring_waiver(s,1);rr_set_attr(s,s.root,"_casualty_operations",cp_container(s,WOTR_LIST,{}));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    am_Builder events;std::vector<uint64_t> rows;
    if(ev_requirement(s,t,ev_card(t,id))){egc_effect(s,t,choice,events,rows);er_commit(s,t,id,"character",face);}
    cau_finish(s,t,result.random.get(),marker,events,rows);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
