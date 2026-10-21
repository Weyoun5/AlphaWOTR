// Turn-boundary physical reset. Card dealing is composed by the outer caller.
// Direct rule-helper semantics: copy resident graph, not ai.action_clone.
uint64_t tr_faces(Snapshot& s,int side){
    const auto name=side==0?"game_state.FP_FACES":"game_state.SH_FACES";
    for(uint64_t n=0;n<s.nodes.size();++n)
        if(s.nodes[n].tag==WOTR_LIST && s.nodes[n].flags==1 && text_is(s,s.nodes[n].integer,name))return n;
    const std::array<std::string_view,6> names=side==0
        ?std::array<std::string_view,6>{"CHARACTER","CHARACTER","MUSTER","EVENT","MUSTER_ARMY","WILL_OF_WEST"}
        :std::array<std::string_view,6>{"CHARACTER","ARMY","MUSTER","EVENT","MUSTER_ARMY","EYE"};
    std::vector<uint64_t> faces;for(auto face:names)faces.push_back(rr_enum(s,face));
    const auto binding=rr_string(s,name),result=rr_scalar(s,WOTR_LIST);
    set_edges(s,result,faces);s.nodes[result].flags=1;s.nodes[result].integer=static_cast<int64_t>(binding);return result;
}
void tr_adjust_pool(Snapshot& s,int side,size_t target){
    const auto pool=pa_pool(s,side);auto dice=child_edges(s,pool);
    if(dice.size()>target)dice.resize(target);
    while(dice.size()<target){
        const auto faces=tr_faces(s,side),current=rr_scalar(s,WOTR_NONE);
        const auto owner=mu_enum(s,"game_state.Side",side==0?"FREE_PEOPLES":"SHADOW"),used=rr_scalar(s,WOTR_BOOL,0);
        const auto cls=rr_string(s,"game_state.ActionDie");
        std::vector<uint64_t> slots{cls,rr_string(s,"faces"),faces,rr_string(s,"current"),current,
            rr_string(s,"side"),owner,rr_string(s,"used"),used};
        const auto die=rr_scalar(s,WOTR_OBJECT);set_edges(s,die,slots);dice.push_back(die);
    }
    set_edges(s,pool,dice);
}
void tr_bonus_dice(Snapshot& s){
    auto alive=[&](std::string_view name){
        const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));
        for(size_t i=1;i<armies.size();i+=2)
            for(auto leader:child_edges(s,obs_attr(s,armies[i],"leaders")))
                if(object_type(s,leader)=="game_state.Companion" && text_is(s,obs_attr(s,leader,"name"),name))return true;
        for(auto companion:child_edges(s,obs_attr(s,obs_attr(s,s.root,"fellowship"),"companions")))
            if(text_is(s,obs_attr(s,companion,"name"),name))return true;
        return false;
    };
    auto active=[&](const char* attr,const char* flag){
        const auto n=obs_attr(s,s.root,attr);return truth(s,n) && truth(s,obs_attr(s,n,flag));
    };
    const int fp=(active("gandalf_the_white","in_play") && alive("Gandalf the White"))+
                 (active("aragorn_crowned","crowned") && alive("Aragorn"));
    const int sh=active("witch_king","in_play")+active("saruman","in_play")+active("mouth_of_sauron","in_play");
    rr_set_attr(s,s.root,"fp_bonus_dice",rr_scalar(s,WOTR_INT,fp));
    rr_set_attr(s,s.root,"sh_bonus_dice",rr_scalar(s,WOTR_INT,sh));
    tr_adjust_pool(s,0,4+fp);tr_adjust_pool(s,1,7+sh);
}
void tr_check_pending(const Snapshot& s){
    const auto pending=obs_attr(s,s.root,"pending_card_option",false);
    require(ctx_none(s,obs_attr(s,s.root,"_gandalf_draw_pending",false)) &&
        (!truth(s,pending) || !truth(s,ctx_get(s,pending,"gandalf_draw"))),
        "Resolve the pending Gandalf draw before the next turn");
}
bool tr_inactivity(Snapshot& s){
    const auto fellowship=obs_attr(s,s.root,"fellowship");
    const auto corruption=typed(s,obs_attr(s,fellowship,"corruption"),WOTR_INT,"turn corruption must be integer").integer;
    if(truth(s,obs_attr(s,fellowship,"mordor_entered")) &&
       !truth(s,obs_attr(s,s.root,"fellowship_acted_in_mordor",false)) && corruption<12){
        rr_set_attr(s,fellowship,"corruption",rr_scalar(s,WOTR_INT,corruption+1));return corruption+1>=12;
    }
    return false;
}
bool tr_settle(Snapshot& s,const Snapshot& t){
    if(!ctx_none(s,obs_attr(s,s.root,"_search_endturn_victory",false)))return true;
    auto result=check_victory(s,t);
    if(!terminal_should_end(result,false)){tr_inactivity(s);result=check_victory(s,t);}
    if(!terminal_should_end(result,true))return false;
    std::string kind=result.kind==1?"military":result.kind==2?"ring_destroyed":"ring_corruption",message;
    if(result.kind==1)message=(result.winner==0?"Free Peoples":"Shadow")+std::string(" military victory! (")+std::to_string(result.winner==0?result.fp_vp:result.sh_vp)+" VP)";
    else if(result.kind==2)message="The Ring is destroyed! Free Peoples win!";
    else message="Ring-bearers corrupted! ("+std::to_string(typed(s,obs_attr(s,obs_attr(s,s.root,"fellowship"),"corruption"),WOTR_INT,"terminal corruption must be integer").integer)+" corruption)";
    const auto marker=rr_scalar(s,WOTR_DICT);
    set_edges(s,marker,{rr_string(s,"victory"),rr_string(s,result.winner==0?"free_peoples":"shadow"),
        rr_string(s,"victory_type"),rr_string(s,kind),rr_string(s,"message"),rr_string(s,message),
        rr_string(s,"free_peoples_vp"),rr_scalar(s,WOTR_INT,result.fp_vp),
        rr_string(s,"shadow_vp"),rr_scalar(s,WOTR_INT,result.sh_vp)});
    rr_set_attr(s,s.root,"_search_endturn_victory",marker);return true;
}
void tr_reset_physical(Snapshot& s,int64_t turn,bool bonus){
    rr_set_attr(s,s.root,"turn",rr_scalar(s,WOTR_INT,turn));
    rr_set_attr(s,s.root,"eyes_allocated",rr_scalar(s,WOTR_INT,0));
    const auto box=obs_attr(s,s.root,"hunt_box");auto hunted=child_edges(s,box);
    bool fp_hunted=false;for(auto die:hunted)fp_hunted|=enum_is(s,obs_attr(s,die,"side"),"game_state.Side","FREE_PEOPLES");
    rr_set_attr(s,s.root,"_fp_used_hunt_box_last_turn",rr_scalar(s,WOTR_BOOL,fp_hunted));
    auto recover=[&](uint64_t die){
        require(object_type(s,die)=="game_state.ActionDie","turn reset requires ActionDie");
        rr_set_attr(s,die,"used",rr_scalar(s,WOTR_BOOL,0));rr_set_attr(s,die,"current",rr_scalar(s,WOTR_NONE));
    };
    for(auto die:hunted)recover(die);set_edges(s,box,{});
    const auto pools=child_edges(s,obs_attr(s,s.root,"dice"));
    for(size_t i=1;i<pools.size();i+=2)for(auto die:child_edges(s,pools[i]))recover(die);
    for(auto key:{"fp_ring_used_this_turn","sh_ring_used_this_turn","fp_ring_waiver","sh_ring_waiver"})
        rr_set_attr(s,s.root,key,rr_scalar(s,WOTR_BOOL,0));
    for(auto key:{"fp_ring_waiver_face","sh_ring_waiver_face","_ring_free_side","_battle_resolved_attacker"})
        rr_set_attr(s,s.root,key,rr_scalar(s,WOTR_NONE));
    for(auto key:{"sh_mouth_muster_used_this_turn","fp_token_used_this_turn","sh_token_used_this_turn","dice_rolled","fellowship_acted_in_mordor"})
        rr_set_attr(s,s.root,key,rr_scalar(s,WOTR_BOOL,0));
    if(bonus)tr_bonus_dice(s);
}
// Retained legacy ABI: low-level physical rollover, without military tables.
bool tr_reset(Snapshot& s,int64_t turn,bool bonus){
    tr_check_pending(s);const bool won=tr_inactivity(s);tr_reset_physical(s,turn,bonus);return won;
}
bool tr_settled_reset(Snapshot& s,const Snapshot& t,int64_t turn,bool bonus){
    tr_check_pending(s);if(tr_settle(s,t))return true;tr_reset_physical(s,turn,bonus);return false;
}
