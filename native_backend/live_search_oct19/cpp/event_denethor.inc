// Denethor's printed loss belongs to the card player's nation choice.
bool edf_pending(const Snapshot& s){
    const auto p=obs_attr(s,s.root,"pending_card_option",false);
    return !ctx_none(s,p)&&text_is(s,field(s,p,"card_name",false),"Denethor's Folly");
}
std::vector<std::string> edf_nations(const Snapshot& s){
    std::set<std::string> nations;const auto army=mu_army(s,"minas tirith");
    if(army!=missing)for(auto leader:child_edges(s,obs_attr(s,army,"leaders"))){
        if(object_type(s,leader)!="game_state.Leader")continue;
        const auto nation=ctx_nation(s,obs_attr(s,leader,"nation"));
        if(fp_nation(nation))nations.insert(ev_lower(nation));
    }
    return {nations.begin(),nations.end()};
}
void edf_eliminate(Snapshot& s,std::string_view wanted){
    const auto army=mu_army(s,"minas tirith");
    if(army!=missing){
        const auto list=obs_attr(s,army,"leaders");auto leaders=child_edges(s,list);
        for(auto it=leaders.begin();it!=leaders.end();++it){
            if(object_type(s,*it)!="game_state.Leader")continue;
            const auto nation=obs_attr(s,*it,"nation");const auto name=ctx_nation(s,nation);
            if(!fp_nation(name)||(!wanted.empty()&&ev_lower(name)!=wanted))continue;
            cbp_book(s,nation,"leader");leaders.erase(it);set_edges(s,list,leaders);return;
        }
    }
    require(wanted.empty(),"Denethor's Folly leader target unavailable");
}
uint64_t edf_choice_pending(const Snapshot& s,const Snapshot& t){
    require(edf_pending(s),"No Denethor leader choice pending");
    const auto p=obs_attr(s,s.root,"pending_card_option");
    require(text_is(s,field(s,p,"side"),"sh")&&text_is(s,field(s,p,"decider"),"sh")&&!truth(s,field(s,p,"direct",false)),"Denethor pending owner or stage invalid");
    const auto card=ev_card(t,text(s,field(s,p,"card_id")));
    require(card!=missing&&text_is(t,obs_attr(t,card,"card_name"),"Denethor's Folly"),"Denethor pending card identity invalid");
    require(text_is(s,field(s,p,"deck"),"strategy"),"Denethor pending deck invalid");
    const auto face=text(s,field(s,p,"die_face"));
    require(face=="ARMY"||face=="MUSTER_ARMY"||face=="MUSTER"||face=="EVENT"||face=="WILL_OF_WEST","Denethor pending payment invalid");
    const auto options=field(s,p,"options");typed(s,options,WOTR_LIST,"Denethor choices must be a list");
    for(auto option:child_edges(s,options)){
        require(typed(s,option,WOTR_DICT,"Denethor choice must be a dictionary").count==2,"Denethor choice shape invalid");
        const auto nation=text(s,field(s,option,"nation"));
        require(nation=="gondor"||nation=="rohan"||nation=="elves"||nation=="dwarves"||nation=="the_north","Denethor choice nation invalid");
    }
    return p;
}
std::unique_ptr<Snapshot> native_denethor_choices(const Snapshot& source,const Snapshot& t,int side){
    require(side==1,"Denethor leader choice belongs to Shadow");edf_choice_pending(source,t);
    auto result=std::make_unique<Snapshot>(source);auto& s=*result;const auto p=obs_attr(s,s.root,"pending_card_option");
    std::vector<uint64_t> rows;int64_t index=0;
    for(auto option:child_edges(s,field(s,p,"options")))rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),
        ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",option}})}));
    s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
bool edf_character(const Snapshot& s){
    const auto army=mu_army(s,"minas tirith");if(army==missing)return false;
    for(auto figure:child_edges(s,obs_attr(s,army,"leaders"))){
        const auto name=obs_attr(s,figure,"name",false);if(ctx_none(s,name))continue;
        const auto value=text(s,name);
        if(value=="Gandalf the Grey"||value=="Gandalf the White"||value.substr(0,7)=="Aragorn")return true;
    }
    return false;
}
std::unique_ptr<Snapshot> native_denethor_kill_for_die(const Snapshot& s,int side,int64_t index){
    am_Builder b;std::vector<uint64_t> rows;const auto card=hpr_card(s,"DENETHORS_FOLLY");
    if(side==0&&card!=missing){
        const auto dice=child_edges(s,pa_pool(s,0));require(index>=0&&static_cast<uint64_t>(index)<dice.size(),"Denethor removal die index invalid");
        const bool will=enum_is(s,obs_attr(s,dice[index],"current"),"game_state.ActionDieFace","WILL_OF_WEST");
        if(will||edf_character(s))rows.push_back(b.row(index,"discard_table_card",{b.str("card_id"),b.str(text(s,obs_attr(s,card,"card_id"))),b.str("card_name"),b.str(text(s,obs_attr(s,card,"card_name"))),b.str("payment"),b.str(will?"will":"character")}));
    }
    b.out.root=b.container(WOTR_LIST,rows);return std::make_unique<Snapshot>(b.out);
}
uint64_t edf_kill_die(const Snapshot& s,int64_t index,std::string_view payment){
    const auto dice=child_edges(s,pa_pool(s,0));require(index>=0&&static_cast<uint64_t>(index)<dice.size(),"Denethor removal die index invalid");
    const auto die=dice[index];require(object_type(s,die)=="game_state.ActionDie"&&ctx_side(s,obs_attr(s,die,"side"))==0&&!truth(s,obs_attr(s,die,"used")),"Denethor removal die unavailable");
    const auto face=enum_member(s,obs_attr(s,die,"current"),"game_state.ActionDieFace");rr_face_index(face);
    require(payment=="will"||payment=="character","Denethor removal payment invalid");
    if(payment=="will")require(face=="WILL_OF_WEST","Denethor removal requires a Will die");
    else require(face!="WILL_OF_WEST"&&edf_character(s),"Denethor any-die removal is not currently offered");
    return die;
}
std::unique_ptr<Snapshot> apply_denethor_kill(const Snapshot& source,const Snapshot& t,int side,int64_t index,std::string_view id,std::string_view payment){
    require(side==0,"Denethor removal belongs to FP");quiet_stable_action_gate(source,t);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==0,"Denethor removal belongs to retained Ring actor");
    const auto card=hpr_card(source,"DENETHORS_FOLLY");require(card!=missing&&text_is(source,obs_attr(source,card,"card_id"),id),"Denethor removal target absent");edf_kill_die(source,index,payment);
    auto child=action_clone(source);auto& s=*child;rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    rr_set_attr(s,edf_kill_die(s,index,payment),"used",rr_scalar(s,WOTR_BOOL,1));
    const auto view=political_view(s,"ROHAN",false);discard_card(s,view,remove_card(s,view,id));
    su_cleanup(s,t);tc_open_hand_limit_discard(s);cau_gandalf(s);rr_set_attr(s,s.root,"_battle_resolved_attacker",rr_scalar(s,WOTR_NONE));return child;
}
