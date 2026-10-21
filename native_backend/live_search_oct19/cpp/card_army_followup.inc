// Explicit free card funding reuses the validated ordinary movement/attack bodies.
// Mixed sieges and non-Palantir mandatory-option overlaps remain unported.
uint64_t af_pending(const Snapshot& s,int side){
    require(side==1,"Card army follow-up belongs to Shadow");
    const auto p=obs_attr(s,s.root,"pending_card_followup",false);
    require(!ctx_none(s,p)&&text_is(s,field(s,p,"side",false),"sh"),"No Shadow card army follow-up pending");
    require(text_is(s,field(s,p,"kind",false),"nazgul_armies")||text_is(s,field(s,p,"kind",false),"wk_army"),"Card army follow-up kind unported");
    const auto moves=field(s,p,"moves_left",false),exclude=field(s,p,"exclude",false),moved=field(s,p,"moved_any",false);
    if(moves!=missing)typed(s,moves,WOTR_INT,"Card follow-up counter must be integer");
    if(exclude!=missing){typed(s,exclude,WOTR_SET,"Card follow-up exclusion must be set");for(auto value:child_edges(s,exclude))text(s,value);}
    if(moved!=missing)typed(s,moved,WOTR_BOOL,"Card follow-up moved flag must be Boolean");return p;
}
void af_scope(const Snapshot& s,const Snapshot& t){
    const auto follow=af_pending(s,1);Snapshot view(s);rr_set_attr(view,view.root,"pending_card_followup",rr_scalar(view,WOTR_NONE));
    // The complete card includes its free army action. Its bonus stays queued
    // until that action settles; validate the originating paid card before
    // hiding this debt only in the read-only structural scope view.
    if(truth(s,obs_attr(s,s.root,"_palantir_draw_pending",false))){
        const char* id=text_is(s,field(s,follow,"kind"),"wk_army")?
            "SHADOW_Character_The_Black_Captain_Commands":"SHADOW_Character_The_Ringwraiths_Are_Abroad";
        epa_paid_continuation_debt(s,id,true);
        rr_set_attr(view,view.root,"_palantir_draw_pending",rr_scalar(view,WOTR_BOOL,0));
    }
    // Retain compatibility with older saved continuations that already opened
    // the bonus choice. Other mandatory windows still refuse.
    const auto option=obs_attr(s,s.root,"pending_card_option",false);
    if(!ctx_none(s,option)&&text_is(s,field(s,option,"card_id",false),"__palantir_draw__")){
        epa_pending(s,1);rr_set_attr(view,view.root,"pending_card_option",rr_scalar(view,WOTR_NONE));
    }
    // Mixed sieges remain outside the free-followup proof boundary.
    quiet_stable_action_gate(view,t);ring_waiver(s,0);ring_waiver(s,1);
}
bool af_nazgul_figure(const Snapshot& s,uint64_t figure){
    if(object_type(s,figure)=="game_state.Nazgul")return true;
    const auto name=obs_attr(s,figure,"name",false);if(ctx_none(s,name))return false;
    std::string normalized;for(char c:text(s,name)){if(c>='A'&&c<='Z')c=char(c-'A'+'a');if(c>='a'&&c<='z')normalized.push_back(c);}
    return normalized.find("witchking")!=std::string::npos;
}
std::unique_ptr<Snapshot> native_card_army_followup_actions(const Snapshot& s,const Snapshot& t,int side){
    const auto p=af_pending(s,side);af_scope(s,t);
    const bool wk=text_is(s,field(s,p,"kind"),"wk_army"),moved=truth(s,field(s,p,"moved_any",false));
    am_Builder move_builder;std::vector<uint64_t> move_rows;amg_append(move_builder,move_rows,s,t,side,-1,false,{},wk?2:1);
    move_builder.out.root=move_builder.container(WOTR_LIST,move_rows);
    auto attacks=native_army_attack_for_die(s,t,side,-1,false);
    am_Builder b;const auto moves=fm_import(b.out,move_builder.out),attack_rows=fm_import(b.out,*attacks);std::vector<uint64_t> rows;
    const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));
    for(size_t i=0;i<armies.size();i+=2){
        const auto key=sanitize_name(text(s,armies[i]));const auto army=armies[i+1];bool own=false,leader=false;
        for(auto unit:child_edges(s,obs_attr(s,army,"units")))own|=ctx_unit_side(s,unit)==side;
        for(auto figure:child_edges(s,obs_attr(s,army,"leaders")))leader|=af_nazgul_figure(s,figure)&&(!wk||object_type(s,figure)!="game_state.Nazgul");
        if(!own||!leader||contains_string(s,field(s,p,"exclude",false),key))continue;
        auto append=[&](const std::vector<uint64_t>& candidates,bool attack){for(auto row:candidates){
            const auto params=child_edges(b.out,row)[2];if(sanitize_name(text(b.out,field(b.out,params,"from")))!=key)continue;
            auto edges=child_edges(b.out,params);edges.push_back(b.str("card_followup"));edges.push_back(rr_scalar(b.out,WOTR_BOOL,1));
            if(attack){edges.push_back(b.str("followup_attack"));edges.push_back(rr_scalar(b.out,WOTR_BOOL,1));}
            set_edges(b.out,params,edges);rows.push_back(row);
        }};
        append(moves,false);if(!moved)append(attack_rows,true);
    }
    rows.push_back(b.row(-1,"followup_done",{}));b.out.root=b.container(WOTR_LIST,rows);return std::make_unique<Snapshot>(std::move(b.out));
}
void af_after_move(Snapshot& s,std::string_view destination){
    const auto p=af_pending(s,1),old=field(s,p,"exclude",false);auto values=ctx_none(s,old)?std::vector<uint64_t>{}:child_edges(s,old);
    const auto key=sanitize_name(destination);if(!contains_string(s,old,key))values.push_back(rr_string(s,key));
    cp_put(s,p,"exclude",cp_container(s,WOTR_SET,values));cp_put(s,p,"moved_any",rr_scalar(s,WOTR_BOOL,1));
    const auto counter=field(s,p,"moves_left",false);const int64_t remaining=(ctx_none(s,counter)?1:s.nodes[counter].integer)-1;cp_put(s,p,"moves_left",rr_scalar(s,WOTR_INT,remaining));
    if(remaining<=0||text_is(s,field(s,p,"kind"),"wk_army"))rr_set_attr(s,s.root,"pending_card_followup",rr_scalar(s,WOTR_NONE));
}
fm_Result apply_card_army_followup_descriptor(const Snapshot& source,const Snapshot& t,const Snapshot& actions,const void* rng,int side,uint64_t row){
    af_pending(source,side);auto current=native_card_army_followup_actions(source,t,side);bool legal=false;
    for(auto candidate:child_edges(*current,current->root))legal|=ap_equal(ap_freeze(*current,candidate),ap_freeze(actions,row));
    require(legal,"Card army follow-up descriptor is stale or illegal");
    const auto parts=child_edges(actions,row);require(typed(actions,parts[0],WOTR_INT,"Follow-up die must be integer").integer==-1,"Card army follow-up cannot name a physical die");
    const auto kind=text(actions,parts[1]);const auto p=parts[2];
    if(kind=="army_attack")return apply_army_attack(source,t,rng,side,-1,ad_text(actions,p,"from"),ad_text(actions,p,"to"),false,true);
    fm_Result result;result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());
    if(kind=="army_move"){
        ArmyMoveRequest req;req.side=side;req.die_index=-1;req.card_followup=true;req.from=ad_text(actions,p,"from");req.to=ad_text(actions,p,"to");
        req.regular=ad_int(actions,p,"n_regular");req.elite=ad_int(actions,p,"n_elite");req.leaders=ad_int(actions,p,"n_leaders");req.nazgul=ad_int(actions,p,"n_nazgul");
        const auto take=field(actions,p,"take_leaders",false);if(take!=missing)req.take_leaders=typed(actions,take,WOTR_BOOL,"take_leaders must be Boolean").integer!=0;
        result.state=apply_army_move(source,t,req);af_after_move(*result.state,req.to);
    }else{
        require(kind=="followup_done","Unknown card army follow-up action");result.state=action_clone(source);
        rr_set_attr(*result.state,result.state->root,"_casualty_operations",rr_scalar(*result.state,WOTR_LIST));rr_set_attr(*result.state,result.state->root,"_ring_free_side",rr_scalar(*result.state,WOTR_NONE));
        rr_set_attr(*result.state,result.state->root,"pending_card_followup",rr_scalar(*result.state,WOTR_NONE));
    }
    const int marker=ctx_side(source,obs_attr(source,source.root,"_battle_resolved_attacker",false));am_Builder events;std::vector<uint64_t> rows;
    cau_finish(*result.state,t,result.random.get(),marker,events,rows);result.state=ac_compact(*result.state);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
