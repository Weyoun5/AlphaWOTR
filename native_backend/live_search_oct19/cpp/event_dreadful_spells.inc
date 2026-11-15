// Dreadful's physical Nazgul dice and shared FP owner losses.
bool af_nazgul_figure(const Snapshot&,uint64_t);
bool eds_supported(const Snapshot& t,uint64_t card){return card!=missing&&text_is(t,obs_attr(t,card,"card_name"),"Dreadful Spells");}
bool eds_pending(const Snapshot& s){const auto p=obs_attr(s,s.root,"pending_card_option",false);return !ctx_none(s,p)&&!truth(s,field(s,p,"card_casualty",false))&&text_is(s,field(s,p,"card_name",false),"Dreadful Spells");}
int eds_figures(const Snapshot& s,uint64_t army){
    if(army==missing)return 0;bool shadow=false;for(auto u:child_edges(s,obs_attr(s,army,"units")))shadow|=ctx_unit_side(s,u)==1;
    if(!shadow)return 0;int count=0;for(auto figure:child_edges(s,obs_attr(s,army,"leaders")))count+=af_nazgul_figure(s,figure);return count;
}
uint64_t eds_options(Snapshot& s){
    std::vector<uint64_t> options;const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));const auto adj=obs_attr(s,obs_attr(s,s.root,"board"),"_adj");
    for(size_t i=0;i<armies.size();i+=2){
        if(!eds_figures(s,armies[i+1]))continue;const auto origin=sanitize_name(text(s,armies[i]));std::vector<std::string> targets;
        const auto adjacent=ctx_get(s,adj,origin);if(!ctx_none(s,adjacent))for(auto key:child_edges(s,adjacent))targets.emplace_back(sanitize_name(text(s,key)));
        std::sort(targets.begin(),targets.end());targets.insert(targets.begin(),origin);
        for(const auto& target:targets){const auto army=mu_army(s,target);if(army==missing)continue;bool fp=false;for(auto u:child_edges(s,obs_attr(s,army,"units")))fp|=ctx_unit_side(s,u)==0;
            if(fp)options.push_back(ev_dict(s,{{"target_region",rr_string(s,origin)},{"target_fp_army",rr_string(s,target)}}));
        }
    }return cp_container(s,WOTR_LIST,options);
}
void eds_effect(Snapshot& s,const Snapshot& t,std::string_view id,std::string_view origin,std::string_view target,void* rng,am_Builder& events,std::vector<uint64_t>& rows){
    const int count=std::min(5,eds_figures(s,mu_army(s,sanitize_name(origin))));if(!count)return;
    std::vector<uint32_t> rolls(count);hf_roll(rng,rolls.data(),rolls.size(),"card effect");
    int64_t hits=0;std::vector<uint64_t> faces;for(auto roll:rolls){hits+=roll>=5;faces.push_back(events.integer(roll));}
    rows.push_back(events.row(-1,"dice_roll",{events.str("faces"),events.container(WOTR_LIST,faces),events.str("ctx"),events.str("Dreadful Spells roll"),events.str("side"),rr_scalar(events.out,WOTR_NONE)}));
    const auto region=sanitize_name(target);if(hits>0&&mu_army(s,region)!=missing)ecc_route(s,t,region,0,hits,id,"Dreadful Spells",rng);
}
fm_Result apply_dreadful_event(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index,std::string_view id){
    require(side==1,"Dreadful Spells belongs to Shadow");quiet_stable_action_gate(source,t);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==1,"Dreadful violates retained Ring actor");
    const auto card=ev_card(t,id);require(eds_supported(t,card)&&contains_string(source,tc_hand(source,1),id),"Dreadful absent from Shadow hand");require(ev_requirement(source,t,card),"Dreadful needs Nazgul-led Shadow army with an FP target");
    const auto die=ev_payment(source,t,1,index,card);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace"));
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;
    rr_set_attr(s,s.root,"_casualty_operations",cp_container(s,WOTR_LIST,{}));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));rr_set_attr(s,ev_payment(s,t,1,index,card),"used",rr_scalar(s,WOTR_BOOL,1));
    const auto options=eds_options(s);const auto choices=child_edges(s,options);am_Builder events;std::vector<uint64_t> rows;
    if(choices.size()>1){rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,"Dreadful Spells")},{"side",rr_string(s,"sh")},{"decider",rr_string(s,"sh")},{"options",options},{"die_face",rr_string(s,face)},{"deck",rr_string(s,"character")}}));tc_open_hand_limit_discard(s);}
    else if(choices.size()==1){const std::string region(text(s,field(s,choices.front(),"target_region"))),target(text(s,field(s,choices.front(),"target_fp_army")));eds_effect(s,t,id,region,target,result.random.get(),events,rows);esr_commit(s,t,id,"character",face);}
    else esr_commit(s,t,id,"character",face); // Package W (P3): no target = the frozen zero-effect play
    cau_finish(s,t,result.random.get(),-1,events,rows);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
uint64_t eds_validate_pending(const Snapshot& s,const Snapshot& t){
    require(eds_pending(s),"No Dreadful target pending");const auto p=obs_attr(s,s.root,"pending_card_option"),card=ev_card(t,text(s,field(s,p,"card_id")));
    require(eds_supported(t,card)&&text_is(s,field(s,p,"side"),"sh")&&text_is(s,field(s,p,"decider"),"sh")&&!truth(s,field(s,p,"direct",false)),"Dreadful pending identity or owner mismatch");
    const auto face=text(s,field(s,p,"die_face"));require(text_is(s,field(s,p,"deck"),"character")&&(face=="CHARACTER"||face=="EVENT"),"Dreadful pending funding invalid");
    const auto choices=child_edges(s,field(s,p,"options"));require(choices.size()>1,"Dreadful modal options invalid");std::set<std::pair<std::string,std::string>> pairs;
    for(auto choice:choices){require(typed(s,choice,WOTR_DICT,"Dreadful choice must be dictionary").count==4,"Dreadful stored choice shape invalid");const std::string origin(text(s,field(s,choice,"target_region"))),target(text(s,field(s,choice,"target_fp_army")));require(!origin.empty()&&!target.empty()&&sanitize_name(origin)==origin&&sanitize_name(target)==target&&pairs.insert({origin,target}).second,"Dreadful stored pair malformed");}
    return p;
}
std::unique_ptr<Snapshot> native_dreadful_choices(const Snapshot& source,const Snapshot& t,int side){
    require(side==1,"Dreadful target belongs to Shadow");eds_validate_pending(source,t);Snapshot s(source);const auto p=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t index=0;
    for(auto choice:child_edges(s,field(s,p,"options")))rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",choice}})}));s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
fm_Result apply_dreadful_choice(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index){
    require(side==1,"Dreadful target belongs to Shadow");const auto pending=eds_validate_pending(source,t);const auto options=child_edges(source,field(source,pending,"options"));require(index>=0&&static_cast<uint64_t>(index)<options.size(),"Dreadful choice index invalid");
    const std::string id(text(source,field(source,pending,"card_id"))),face(text(source,field(source,pending,"die_face"))),region(text(source,field(source,options[index],"target_region"))),target(text(source,field(source,options[index],"target_fp_army")));const int marker=ctx_side(source,obs_attr(source,source.root,"_battle_resolved_attacker",false));
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;
    rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));quiet_stable_action_gate(s,t);ring_waiver(s,0);ring_waiver(s,1);rr_set_attr(s,s.root,"_casualty_operations",cp_container(s,WOTR_LIST,{}));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    am_Builder events;std::vector<uint64_t> rows;
    // Global requirement loss retains the card. A qualifying stored origin
    // rolls before looking up its stored FP target, matching the source.
    if(ev_requirement(s,t,ev_card(t,id))){eds_effect(s,t,id,region,target,result.random.get(),events,rows);esr_commit(s,t,id,"character",face);}
    cau_finish(s,t,result.random.get(),marker,events,rows);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
