// Shared Power/Tom registration and learned removal, with canonical table
// admission and preserved source-specific movement/attack restrictions.
std::string_view ept_kind(const Snapshot& t,uint64_t card){
    if(card==missing)return {};const auto name=obs_attr(t,card,"card_name");
    if(text_is(t,name,"A Power too Great"))return "POWER_TOO_GREAT";
    if(text_is(t,name,"The Power of Tom Bombadil"))return "TOM_BOMBADIL";return {};
}
bool ept_supported(const Snapshot& t,uint64_t card){return !ept_kind(t,card).empty();}
uint64_t ept_first_kind(const Snapshot& s,std::string_view wanted){
    const auto table=obs_attr(s,s.root,"persistent_cards");
    for(const char* kind:{"POWER_TOO_GREAT","TOM_BOMBADIL"}){
        if(!wanted.empty()&&wanted!=kind)continue;
        for(const char* key:{"fp_cards","sh_cards"})for(auto card:child_edges(s,obs_attr(s,table,key)))if(enum_is(s,obs_attr(s,card,"card_type",false),"game_state.PersistentCardType",kind))return card;
    }
    return missing;
}
uint64_t ept_first(const Snapshot& s){return ept_first_kind(s,{});}
std::vector<uint64_t> ept_cards(const Snapshot& s){
    std::vector<uint64_t> result;for(const char* kind:{"POWER_TOO_GREAT","TOM_BOMBADIL"}){const auto card=ept_first_kind(s,kind);if(card!=missing)result.push_back(card);}return result;
}
uint64_t ept_by_id(const Snapshot& s,std::string_view id){for(auto card:ept_cards(s))if(text_is(s,obs_attr(s,card,"card_id"),id))return card;return missing;}
bool ept_blocks(const Snapshot& s,std::string_view region){
    const auto key=sanitize_name(region);
    return (ept_first_kind(s,"POWER_TOO_GREAT")!=missing&&(key=="lorien"||key=="rivendell"||key=="grey havens"||key=="the grey havens")) ||
        (ept_first_kind(s,"TOM_BOMBADIL")!=missing&&(key=="old forest"||key=="the old forest"||key=="shire"||key=="the shire"||key=="buckland"));
}
void ept_validate_card(const Snapshot& s,const Snapshot& t,uint64_t card){
    require(object_type(s,card)=="game_state.PersistentCard"&&ctx_side(s,obs_attr(s,card,"side"))==0&&ctx_none(s,obs_attr(s,card,"requires_companion",false)),"Power table identity malformed");
    const auto kind=ept_kind(s,card);
    require(!kind.empty()&&enum_is(s,obs_attr(s,card,"card_type"),"game_state.PersistentCardType",kind),"Power table name/type mismatch");
    require(ept_kind(t,ev_card(t,text(s,obs_attr(s,card,"card_id"))))==kind,"Power table ID does not name registered event");
}
void ept_scope(const Snapshot& source,const Snapshot& t,bool allow_payment=false,bool allow_stable_context=false){
    Snapshot view(source);const auto table=obs_attr(view,view.root,"persistent_cards");
    for(const char* key:{"fp_cards","sh_cards"}){
        const auto list=obs_attr(view,table,key);std::vector<uint64_t> keep;
        for(auto card:child_edges(view,list)){
            const auto type=obs_attr(view,card,"card_type",false);
            if(enum_is(view,type,"game_state.PersistentCardType","POWER_TOO_GREAT")||enum_is(view,type,"game_state.PersistentCardType","TOM_BOMBADIL")){require(std::string_view(key)=="fp_cards","Power belongs on FP table");ept_validate_card(view,t,card);}
            else keep.push_back(card);
        }
        set_edges(view,list,keep);
    }
    if(allow_payment)rr_set_attr(view,view.root,"pending_card_option",rr_scalar(view,WOTR_NONE));
    // Validated registration and explicit removal callers opt into stable
    // sieges after checking the exact table-card objects above.
    if(allow_stable_context)quiet_stable_action_gate(view,t);else quiet_action_gate(view,t);
    ring_waiver(source,0);ring_waiver(source,1);
}
std::vector<std::pair<std::string,std::string>> ept_pairs(const Snapshot& s,const Snapshot& t){
    std::vector<std::string> army,character;
    for(auto value:child_edges(s,tc_hand(s,1))){
        if(s.nodes[value].tag!=WOTR_STRING)continue;const std::string id(text(s,value));const auto card=ev_card(t,id);
        if(card!=missing&&text_is(t,obs_attr(t,card,"card_type"),"Army")){if(std::find(army.begin(),army.end(),id)==army.end())army.push_back(id);}
        else if(id.find("_Character_")!=std::string::npos&&std::find(character.begin(),character.end(),id)==character.end())character.push_back(id);
    }
    std::vector<std::pair<std::string,std::string>> result;for(const auto& a:army)for(const auto& c:character)result.emplace_back(a,c);return result;
}
uint64_t ept_pair(Snapshot& s,const std::pair<std::string,std::string>& pair){return ev_dict(s,{{"army_card",rr_string(s,pair.first)},{"character_card",rr_string(s,pair.second)}});}
std::unique_ptr<Snapshot> native_power_kill_for_die(const Snapshot& s,const Snapshot& t,int side,int64_t index){
    require(side==0||side==1,"Power kill menu actor invalid");am_Builder b;std::vector<uint64_t> rows;
    if(side==1&&!ept_pairs(s,t).empty())for(auto card:ept_cards(s)){ept_validate_card(s,t,card);rows.push_back(b.row(index,"discard_table_card",{b.str("card_id"),b.str(text(s,obs_attr(s,card,"card_id"))),b.str("card_name"),b.str(text(s,obs_attr(s,card,"card_name")))}));}
    b.out.root=b.container(WOTR_LIST,rows);return std::make_unique<Snapshot>(b.out);
}
std::unique_ptr<Snapshot> native_power_kill_actions(const Snapshot& s,const Snapshot& t,int side){
    require(side==0||side==1,"Power kill actor invalid");ept_scope(s,t,false,true);am_Builder b;std::vector<uint64_t> rows;std::set<std::string> faces;const auto dice=child_edges(s,pa_pool(s,side));
    for(size_t i=0;i<dice.size();++i){const auto die=dice[i],face=obs_attr(s,die,"current");if(truth(s,obs_attr(s,die,"used"))||ctx_none(s,face))continue;
        const std::string name(enum_member(s,face,"game_state.ActionDieFace"));if(!faces.insert(name).second)continue;
        if(side==1&&!ept_pairs(s,t).empty())for(auto card:ept_cards(s))rows.push_back(b.row(i,"discard_table_card",{b.str("card_id"),b.str(text(s,obs_attr(s,card,"card_id"))),b.str("card_name"),b.str(text(s,obs_attr(s,card,"card_name")))}));
    }
    b.out.root=b.container(WOTR_LIST,rows);return std::make_unique<Snapshot>(b.out);
}
uint64_t ept_payment_die(const Snapshot& s,int64_t index){
    const auto pool=child_edges(s,pa_pool(s,1));require(index>=0&&static_cast<uint64_t>(index)<pool.size(),"Power kill die index invalid");const auto die=pool[index];
    require(object_type(s,die)=="game_state.ActionDie"&&ctx_side(s,obs_attr(s,die,"side"))==1&&!truth(s,obs_attr(s,die,"used")),"Power kill die unavailable");rr_face_index(enum_member(s,obs_attr(s,die,"current"),"game_state.ActionDieFace"));return die;
}
std::unique_ptr<Snapshot> apply_power_event(const Snapshot& source,const Snapshot& t,int side,int64_t index,std::string_view id){
    require(side==0,"Power event belongs to FP");ept_scope(source,t,false,true);const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==0,"Power event belongs to retained Ring actor");
    const auto card=ev_card(t,id);require(ept_supported(t,card)&&ctx_side(t,obs_attr(t,card,"side"))==0&&contains_string(source,tc_hand(source,0),id),"Power event absent or invalid");require(ev_requirement(source,t,card),"Power event requirement false");
    const auto die=ev_payment(source,t,0,index,card);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace")),deck=ev_lower(text(t,obs_attr(t,card,"deck")));
    auto child=action_clone(source);auto& s=*child;rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));rr_set_attr(s,ev_payment(s,t,0,index,card),"used",rr_scalar(s,WOTR_BOOL,1));
    const auto kind=ept_kind(t,card),nation=kind=="TOM_BOMBADIL"?std::string_view("THE_NORTH"):std::string_view("ELVES");
    const auto boxes=obs_attr(s,obs_attr(s,s.root,"political"),"boxes");const int64_t current=political_view(s,nation,true).box,cap=active_nation(s,political_view(s,nation,true),nation,false)?3:2;
    if(current<cap)me_nation_assign(s,boxes,nation,rr_scalar(s,WOTR_INT,std::min<int64_t>(cap,current+1)));
    const auto persistent=mu_object(s,"game_state.PersistentCard",{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,text(t,obs_attr(t,card,"card_name")))},{"side",mu_enum(s,"game_state.Side","FREE_PEOPLES")},{"card_type",mu_enum(s,"game_state.PersistentCardType",kind)},{"requires_companion",rr_scalar(s,WOTR_NONE)}});
    const auto list=obs_attr(s,obs_attr(s,s.root,"persistent_cards"),"fp_cards");auto entries=child_edges(s,list);entries.push_back(persistent);set_edges(s,list,entries);
    const auto hand=tc_hand(s,0);auto held=child_edges(s,hand);held.erase(std::find_if(held.begin(),held.end(),[&](uint64_t c){return text_is(s,c,id);}));set_edges(s,hand,held);
    tc_open_hand_limit_discard(s);if(!contains_string(s,hand,id))ev_gandalf_after_event(s,id,deck,face);su_cleanup(s,t);cau_gandalf(s);rr_set_attr(s,s.root,"_battle_resolved_attacker",rr_scalar(s,WOTR_NONE));return child;
}
std::unique_ptr<Snapshot> apply_power_kill(const Snapshot& source,const Snapshot& t,int side,int64_t index,std::string_view id){
    require(side==1,"Power kill belongs to Shadow");ept_scope(source,t,false,true);const auto card=ept_by_id(source,id);require(card!=missing,"Power kill target absent");const std::string name(text(source,obs_attr(source,card,"card_name")));ept_payment_die(source,index);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==1,"Power kill belongs to retained Ring actor");const auto pairs=ept_pairs(source,t);require(!pairs.empty(),"Power kill needs Army and Character payment");
    auto child=action_clone(source);auto& s=*child;rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));rr_set_attr(s,ept_payment_die(s,index),"used",rr_scalar(s,WOTR_BOOL,1));
    std::vector<uint64_t> options;for(const auto& pair:pairs)options.push_back(ept_pair(s,pair));
    rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,name)},{"side",rr_string(s,"sh")},{"decider",rr_string(s,"sh")},{"direct",rr_scalar(s,WOTR_BOOL,1)},{"table_kill_payment",rr_scalar(s,WOTR_BOOL,1)},{"options",cp_container(s,WOTR_LIST,options)}}));
    const auto list=obs_attr(s,obs_attr(s,s.root,"persistent_cards"),"fp_cards");auto cards=child_edges(s,list);const auto found=std::find_if(cards.begin(),cards.end(),[&](uint64_t c){return text_is(s,obs_attr(s,c,"card_id"),id);});require(found!=cards.end(),"Power target vanished");cards.erase(found);set_edges(s,list,cards);
    const auto pile=field(s,obs_attr(s,s.root,"discard_face_up"),"fp_strategy");auto discarded=child_edges(s,pile);discarded.push_back(rr_string(s,id));set_edges(s,pile,discarded);
    su_cleanup(s,t);tc_open_hand_limit_discard(s);cau_gandalf(s);rr_set_attr(s,s.root,"_battle_resolved_attacker",rr_scalar(s,WOTR_NONE));return child;
}
uint64_t ept_pending(const Snapshot& s,const Snapshot& t){
    const auto p=obs_attr(s,s.root,"pending_card_option",false);require(!ctx_none(s,p)&&truth(s,field(s,p,"table_kill_payment",false))&&truth(s,field(s,p,"direct",false))&&text_is(s,field(s,p,"side",false),"sh")&&text_is(s,field(s,p,"decider",false),"sh"),"No Power payment pending");
    const auto card=ev_card(t,text(s,field(s,p,"card_id")));require(ept_supported(t,card)&&text(s,field(s,p,"card_name"))==text(t,obs_attr(t,card,"card_name")),"Power pending identity invalid");return p;
}
std::unique_ptr<Snapshot> native_power_payment_actions(const Snapshot& source,const Snapshot& t,int side){
    require(side==1,"Power payment belongs to Shadow");ept_pending(source,t);auto out=std::make_unique<Snapshot>(source);auto& s=*out;const auto p=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t i=0;
    // Source deliberately recomputes from the current hand, not stored options.
    for(const auto& pair:ept_pairs(s,t))rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,i++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",ept_pair(s,pair)}})}));
    s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
// Package W2 (P10): the pair comes from the row's `choices` as the frozen pay_table_card_kill (ai.py 10053-10071) reads it; the only
// refusal is the frozen one - the pair is not an (Army-type, Character) pair of the CURRENT Shadow hand (IllegalPlay, state untouched).
// The index entry apply_power_payment (the C export wotr_snapshot_apply_power_payment / native_backend/snapshot.py, the direct driver
// and tests) resolves the pair of the CURRENT menu at that index and pays it through the same body.
std::unique_ptr<Snapshot> apply_power_payment_pair(const Snapshot& source,const Snapshot& t,int side,const std::string& army_id,const std::string& character_id){
    require(side==1,"Power payment belongs to Shadow");ept_pending(source,t);ept_scope(source,t,true,true);const auto pairs=ept_pairs(source,t);
    require(std::find(pairs.begin(),pairs.end(),std::make_pair(army_id,character_id))!=pairs.end(),"table-card kill payment is not an (Army-type, Character) pair in the Shadow hand");
    auto child=action_clone(source);auto& s=*child;rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));const auto hand=tc_hand(s,1),pile=obs_attr(s,ctx_side_value(s,obs_attr(s,s.root,"hands"),1),"discarded_face_down");
    for(const auto& id:{army_id,character_id}){auto held=child_edges(s,hand);const auto found=std::find_if(held.begin(),held.end(),[&](uint64_t c){return text_is(s,c,id);});require(found!=held.end(),"Power payment card missing");const auto paid=*found;held.erase(found);set_edges(s,hand,held);auto discarded=child_edges(s,pile);discarded.push_back(paid);set_edges(s,pile,discarded);}
    rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));su_cleanup(s,t);tc_open_hand_limit_discard(s);cau_gandalf(s);return child;
}
std::unique_ptr<Snapshot> apply_power_payment(const Snapshot& source,const Snapshot& t,int side,int64_t index){
    require(side==1,"Power payment belongs to Shadow");ept_pending(source,t);const auto pairs=ept_pairs(source,t);require(index>=0&&static_cast<uint64_t>(index)<pairs.size(),"Power payment index invalid");
    return apply_power_payment_pair(source,t,side,pairs[static_cast<size_t>(index)].first,pairs[static_cast<size_t>(index)].second);
}
