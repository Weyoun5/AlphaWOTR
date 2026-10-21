// Ordinary event transactions. Legality and effect support are separate:
// an unported effect is never mistaken for a card with no legal effect.
std::string ev_lower(std::string_view value){std::string out(value);for(auto& c:out)if(c>='A' && c<='Z')c=c-'A'+'a';return out;}
#include "event_figure_requirements.inc"
#include "event_board_requirements.inc"
#include "event_target_requirements.inc"
uint64_t ev_card(const Snapshot& t,std::string_view id){return ctx_get(t,field(t,t.root,"cards"),id);}
bool ev_simple_supported(const Snapshot& t,uint64_t card){
    const auto name=obs_attr(t,card,"card_name");return text_is(t,name,"Bilbo's Song") || text_is(t,name,"Morgul Wound") || text_is(t,name,"The Day Without Dawn");
}
bool esr_saruman_present(const Snapshot& s);
bool eff_requirement(const Snapshot& s);
bool ev_requirement(const Snapshot& s,const Snapshot& t,uint64_t card){
    const auto name=std::string(text(t,obs_attr(t,card,"card_name")));
    if(name=="Faramir's Rangers")return eff_requirement(s);
    if(name=="The Spirit of Mordor")return evt_spirit(s);
    if(name=="Stormcrow")return evt_stormcrow(s,t);
    if(name=="Through a Day and a Night")return evt_through_day(s);
    const auto requirement=obs_attr(t,card,"event_req");const auto req=ctx_none(t,requirement)?std::string{}:ev_lower(text(t,requirement));auto has=[&](const char* term){return req.find(term)!=std::string::npos;};
    // The source's empty-requirement return precedes Woses and Horde.
    const bool empty_requirement=req.find_first_not_of(" \t\r\n\v\f")==std::string::npos;
    if(name=="Paths of the Woses")return empty_requirement||evt_paths(s,t);
    if(name=="Horde from the East"&&!empty_requirement&&!evt_horde(s))return false;
    const auto fs=obs_attr(s,s.root,"fellowship");
    auto companion=[&](const char* name){for(auto c:child_edges(s,obs_attr(s,fs,"companions")))if(text_is(s,obs_attr(s,c,"name"),name))return true;return false;};
    if(has("gimli or legolas are in the fellowship") && !companion("Gimli") && !companion("Legolas"))return false;
    if(has("boromir is in the fellowship") && !companion("Boromir"))return false;
    if(has("gandalf the grey is in the fellowship") && !companion("Gandalf the Grey"))return false;
    if(has("at least one companion is in the fellowship") && !truth(s,obs_attr(s,fs,"companions")))return false;
    if(has("fellowship is revealed") && truth(s,obs_attr(s,fs,"hidden")))return false;
    if(has("fellowship is not in a region") && has("free peoples settlement")){
        const auto meta=ctx_get(t,field(t,t.root,"event_region_metadata"),glb_fellowship_key(s,fs));
        if(meta!=missing && text_is(t,ctx_get(t,meta,"owner"),"free") && truth(t,ctx_get(t,meta,"is_settlement")))return false;
    }else require(!has("fellowship is not in a region"),"native event requirement has an unported location predicate");
    if(has("fellowship is on step 1 or higher") && (s.nodes[obs_attr(s,fs,"progress")].integer<1 || truth(s,obs_attr(s,fs,"mordor_entered"))))return false;
    if(!evf_requirement(s,t,req))return false;
    if(!evb_requirement(s,t,req))return false;
    if(has("gondor is active") && !active_nation(s,political_view(s,"GONDOR",true),"GONDOR",false))return false;
    for(const auto& entry:std::vector<std::pair<const char*,const char*>>{{"elves are \"at war\"","ELVES"},{"rohan is \"at war\"","ROHAN"},{"isengard is \"at war\"","ISENGARD"},{"sauron is \"at war\"","SAURON"},{"southrons & easterlings are \"at war\"","SOUTHRON_EASTERLINGS"}})
        if(has(entry.first) && political_view(s,entry.second,true).box<3)return false;
    if(has("all shadow nations are \"at war\""))for(const auto nation:{"SAURON","ISENGARD","SOUTHRON_EASTERLINGS"})if(political_view(s,nation,true).box<3)return false;
    if(has("saruman is in play")&&!esr_saruman_present(s))return false;
    require(!has("either the ring-bearers"),"native event requirement has an unported board or figure predicate");
    return true;
}
bool ev_category(const Snapshot& t,uint64_t card,std::string_view category){
    const auto kind=text(t,obs_attr(t,card,"card_type"));
    return category=="event"?(kind=="Character" || kind=="Army" || kind=="Muster"):
        category=="character"?kind=="Character":category=="army"?kind=="Army":category=="muster"?kind=="Muster":false;
}
std::unique_ptr<Snapshot> native_play_event_for_die(const Snapshot& s,const Snapshot& t,int32_t side,int64_t index,std::string_view category){
    require(side==0 || side==1,"event menu side must be FP or SH");
    // Requirement enumeration is read-only; legal siege card effects remain
    // separately checked by complete menus and each event executor.
    quiet_action_gate(s,t,false,false,false,false,false,false,false,false,false,true);ring_waiver(s,side);
    am_Builder b;std::vector<uint64_t> rows;
    for(auto id:child_edges(s,tc_hand(s,side))){
        const auto card=ev_card(t,text(s,id));if(card==missing || !ev_category(t,card,category))continue;
        if(ev_requirement(s,t,card))rows.push_back(b.row(index,"play_event",{b.str("card_id"),b.str(text(s,id)),b.str("side"),b.str(side==0?"fp":"sh")}));
    }
    b.out.root=b.container(WOTR_LIST,rows);return std::make_unique<Snapshot>(b.out);
}
uint64_t ev_payment(const Snapshot& s,const Snapshot& t,int32_t side,int64_t index,uint64_t card,bool mouth_muster=false){
    const auto dice=child_edges(s,pa_pool(s,side));require(index>=0 && static_cast<uint64_t>(index)<dice.size(),"event payment index is out of range");
    const auto die=dice[index];require(object_type(s,die)=="game_state.ActionDie" && ctx_side(s,obs_attr(s,die,"side"))==side && !truth(s,obs_attr(s,die,"used")),"event payment die is used or belongs to another side");
    const auto current=obs_attr(s,die,"current");const auto face=enum_member(s,current,"game_state.ActionDieFace");
    const auto kind=text(t,obs_attr(t,card,"card_type"));
    if(mouth_muster){
        require(side==1 && kind=="Army" && face=="MUSTER" && am_mouth_available(s,side),"Mouth Army-event requires available ability and real Muster die");
        return die;
    }
    const bool valid=face=="EVENT" || face=="WILL_OF_WEST" || (kind=="Character" && face=="CHARACTER") ||
        (kind=="Army" && (face=="ARMY" || face=="MUSTER_ARMY")) || (kind=="Muster" && (face=="MUSTER" || face=="MUSTER_ARMY"));
    require(valid,"event requires its named printed-type die");return die;
}
void ev_consume_payment(Snapshot& s,const Snapshot& t,int side,int64_t index,uint64_t card,bool mouth_muster){
    const auto die=ev_payment(s,t,side,index,card,mouth_muster);
    rr_set_attr(s,die,"used",rr_scalar(s,WOTR_BOOL,1));
    if(mouth_muster)rr_set_attr(s,s.root,"sh_mouth_muster_used_this_turn",rr_scalar(s,WOTR_BOOL,1));
}
uint64_t ev_dict(Snapshot& s,const std::vector<std::pair<const char*,uint64_t>>& values){
    const auto dict=rr_scalar(s,WOTR_DICT);std::vector<uint64_t> edges;for(const auto& kv:values){edges.push_back(rr_string(s,kv.first));edges.push_back(kv.second);}set_edges(s,dict,edges);return dict;
}
void ev_gandalf_after_event(Snapshot& s,std::string_view id,std::string_view deck,std::string_view face){
    if(face!="EVENT" && face!="WILL_OF_WEST")return;
    // A completed quiet event has no preceding blocking continuation. Queue
    // then resolve eligibility now, exactly as the common executor tail does.
    rr_set_attr(s,s.root,"_gandalf_draw_pending",rr_scalar(s,WOTR_NONE));
    const auto fs=obs_attr(s,s.root,"fellowship"),guide=obs_attr(s,fs,"guide");
    if(ctx_none(s,guide) || !text_is(s,obs_attr(s,guide,"name"),"Gandalf the Grey"))return;
    bool member=false;for(auto c:child_edges(s,obs_attr(s,fs,"companions")))member|=c==guide;
    if(!member || !truth(s,obs_attr(s,s.root,da_deck_name(0,deck),false)))return;
    const auto options=rr_scalar(s,WOTR_LIST);
    set_edges(s,options,{ev_dict(s,{{"gandalf_draw",rr_scalar(s,WOTR_BOOL,1)}}),ev_dict(s,{{"gandalf_draw",rr_scalar(s,WOTR_BOOL,0)}})});
    rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,"__gandalf_draw__")},{"card_name",rr_string(s,"Gandalf guide draw")},
        {"side",rr_string(s,"fp")},{"decider",rr_string(s,"fp")},{"direct",rr_scalar(s,WOTR_BOOL,1)},{"gandalf_draw",rr_scalar(s,WOTR_BOOL,1)},
        {"source_card_id",rr_string(s,id)},{"deck",rr_string(s,deck)},{"die_face",rr_string(s,face)},{"options",options}}));
}
#include "event_free_funding.inc"
void apply_simple_event_action_inplace(Snapshot& s,const Snapshot& t,void* rng,int side,int64_t index,std::string_view id,bool free_ents,am_Builder& ef_events,std::vector<uint64_t>& ef_annotations,bool mouth_muster=false){
    const auto& source=s;ef_validate_free(source,t,side,index,id,free_ents);
    const int ef_marker=ctx_side(source,obs_attr(source,source.root,"_battle_resolved_attacker",false));const auto ef_unused=ef_unused_dice(source);

    require(side==0 || side==1,"event action side must be FP or SH");quiet_action_gate(source,t,false,false,false,false,false,false,false,false,false,false,(false||free_ents),(side==1 && ev_card(t,id)!=missing && (text_is(t,obs_attr(t,ev_card(t,id),"card_name"),"Morgul Wound") || text_is(t,obs_attr(t,ev_card(t,id),"card_name"),"The Day Without Dawn")) || (side==0 && ev_card(t,id)!=missing && text_is(t,obs_attr(t,ev_card(t,id),"card_name"),"Bilbo's Song"))));ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained) || ctx_side(source,retained)==side,"event belongs to retained Ring actor");
    const auto card=ev_card(t,id);require(card!=missing,"event card is unknown");
    require(ctx_side(t,obs_attr(t,card,"side"))==side,"event card belongs to another side");
    require(contains_string(source,tc_hand(source,side),id),"event card is not in acting hand");
    require(ev_requirement(source,t,card),"event printed requirement is false");require(ev_simple_supported(t,card),"native event effect is not ported");
    const auto die=free_ents?missing:ev_payment(source,t,side,index,card,mouth_muster);const std::string face(free_ents?std::string_view{}:enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace"));
    const auto name=text(t,obs_attr(t,card,"card_name"));const auto deck=ev_lower(text(t,obs_attr(t,card,"deck")));
    
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    if(!free_ents)ev_consume_payment(s,t,side,index,card,mouth_muster);
    const auto fs=obs_attr(s,s.root,"fellowship"),corruption=obs_attr(s,fs,"corruption");
    if(name=="Bilbo's Song"){
        const auto guide=obs_attr(s,fs,"guide");const int heal=!ctx_none(s,guide) && text(s,obs_attr(s,guide,"name")).find("Gollum")!=std::string_view::npos?2:1;
        rr_set_attr(s,fs,"corruption",rr_scalar(s,WOTR_INT,std::max<int64_t>(0,typed(s,corruption,WOTR_INT,"corruption must be integer").integer-heal)));
    }else if(name=="Morgul Wound"){
        const auto value=typed(s,corruption,WOTR_INT,"corruption must be integer").integer;
        rr_set_attr(s,fs,"corruption",rr_scalar(s,WOTR_INT,std::min<int64_t>(12,value+(value<=3?2:1))));
    }else{
        const auto pool=pa_pool(s,0);std::vector<uint64_t> kept;
        for(auto d:child_edges(s,pool))if(truth(s,obs_attr(s,d,"used")) || !enum_is(s,obs_attr(s,d,"current"),"game_state.ActionDieFace","WILL_OF_WEST"))kept.push_back(d);
        set_edges(s,pool,kept); // quiet gate proved no pending integer-index owners
    }
    const auto hand=tc_hand(s,side);auto cards=child_edges(s,hand);auto pos=std::find_if(cards.begin(),cards.end(),[&](uint64_t c){return text_is(s,c,id);});
    require(pos!=cards.end(),"event card disappeared during effect");const auto played=*pos;cards.erase(pos);set_edges(s,hand,cards);
    const std::string pile=(side==0?"fp_":"sh_")+deck;const auto discarded=field(s,obs_attr(s,s.root,"discard_face_up"),pile.c_str());
    auto discard=child_edges(s,discarded);discard.push_back(played);set_edges(s,discarded,discard);
    tc_open_hand_limit_discard(s);
    if(side==0 && !contains_string(s,tc_hand(s,side),id))ev_gandalf_after_event(s,id,deck,face);
    epa_after_event(s,side,id,deck,face);
    rr_set_attr(s,s.root,"_battle_resolved_attacker",rr_scalar(s,WOTR_NONE));

    if(free_ents)ef_finish_free(s,t,rng,ef_marker,ef_unused,ef_events,ef_annotations);
}
std::unique_ptr<Snapshot> apply_simple_event_action(const Snapshot& s,const Snapshot& t,int32_t side,int64_t index,std::string_view id,bool mouth_muster=false){
    require(side==0 || side==1,"event action side must be FP or SH");quiet_action_gate(s,t,false,false,false,false,false,false,false,false,false,false,false,(side==1 && ev_card(t,id)!=missing && (text_is(t,obs_attr(t,ev_card(t,id),"card_name"),"Morgul Wound") || text_is(t,obs_attr(t,ev_card(t,id),"card_name"),"The Day Without Dawn")) || (side==0 && ev_card(t,id)!=missing && text_is(t,obs_attr(t,ev_card(t,id),"card_name"),"Bilbo's Song"))));ring_waiver(s,0);ring_waiver(s,1);
    const auto retained=obs_attr(s,s.root,"_ring_free_side",false);require(ctx_none(s,retained) || ctx_side(s,retained)==side,"event belongs to retained Ring actor");
    const auto card=ev_card(t,id);require(card!=missing,"event card is unknown");
    require(ctx_side(t,obs_attr(t,card,"side"))==side,"event card belongs to another side");
    require(contains_string(s,tc_hand(s,side),id),"event card is not in acting hand");
    require(ev_requirement(s,t,card),"event printed requirement is false");require(ev_simple_supported(t,card),"native event effect is not ported");
    const auto die=ev_payment(s,t,side,index,card,mouth_muster);const std::string face(enum_member(s,obs_attr(s,die,"current"),"game_state.ActionDieFace"));
    const auto name=text(t,obs_attr(t,card,"card_name"));const auto deck=ev_lower(text(t,obs_attr(t,card,"deck")));
    
    auto child=action_clone(s);am_Builder ef_events;std::vector<uint64_t> ef_annotations;
    apply_simple_event_action_inplace(*child,t,nullptr,side,index,id,false,ef_events,ef_annotations,mouth_muster);return child;
}
uint64_t ev_gandalf_pending(const Snapshot& s){
    const auto p=obs_attr(s,s.root,"pending_card_option",false);
    require(!ctx_none(s,p) && text_is(s,field(s,p,"card_id",false),"__gandalf_draw__") && truth(s,field(s,p,"gandalf_draw",false)),"no Gandalf draw is pending");
    require(text_is(s,field(s,p,"side"),"fp") && text_is(s,field(s,p,"decider"),"fp"),"Gandalf draw requires FP owner and decider");
    const auto deck=field(s,p,"deck");da_deck_name(0,text(s,deck));
    const auto opts=child_edges(s,field(s,p,"options"));
    require(opts.size()==2 && s.nodes[opts[0]].tag==WOTR_DICT && s.nodes[opts[0]].count==2 && s.nodes[opts[1]].tag==WOTR_DICT && s.nodes[opts[1]].count==2,
            "Gandalf draw requires original two boolean options");
    for(size_t i=0;i<2;++i){const auto v=field(s,opts[i],"gandalf_draw");require(typed(s,v,WOTR_BOOL,"Gandalf choice must be bool").integer==static_cast<int64_t>(1-i),"Gandalf option order mismatch");}
    return p;
}
std::unique_ptr<Snapshot> native_gandalf_draw_actions(const Snapshot& s,int32_t side){
    require(side==0,"Gandalf draw menu belongs to FP");const auto p=ev_gandalf_pending(s);
    am_Builder b;std::vector<uint64_t> rows;
    if(side==0)for(size_t i=0;i<2;++i){
        const auto choice=b.container(WOTR_DICT,{b.str("gandalf_draw"),rr_scalar(b.out,WOTR_BOOL,i==0)});
        rows.push_back(b.row(-1,"card_option",{b.str("idx"),b.integer(i),b.str("card_id"),b.str("__gandalf_draw__"),
            b.str("card_name"),b.str(text(s,field(s,p,"card_name"))),b.str("side"),b.str("fp"),b.str("choices"),choice,
            b.str("source_card_id"),b.str(text(s,field(s,p,"source_card_id"))),b.str("deck"),b.str(text(s,field(s,p,"deck")))}));
    }
    b.out.root=b.container(WOTR_LIST,rows);return std::make_unique<Snapshot>(b.out);
}
void bc_settle_draw(Snapshot& s,int side,std::string_view deck); // Package W (P9b): the owed-draw ledger (game_logic.settle_owed_draw)
std::unique_ptr<Snapshot> apply_gandalf_draw_action(const Snapshot& s,const Snapshot& t,int32_t side,bool draw){
    require(side==0,"Gandalf draw belongs to FP");const auto p=ev_gandalf_pending(s);const std::string deck(text(s,field(s,p,"deck")));
    // The pending itself is the only allowed exception to the ordinary tail
    // gate. Validate an unpublished clone with that pending temporarily gone.
    auto child=action_clone(s);auto& out=*child;rr_set_attr(out,out.root,"pending_card_option",rr_scalar(out,WOTR_NONE));
    quiet_action_gate(out,t); // Package W: the stable-siege origin lists and their proof branch are lifted (frozen: the draw reads neither the origin nor a siege)
    ring_waiver(out,0);ring_waiver(out,1);
    rr_set_attr(out,out.root,"_casualty_operations",rr_scalar(out,WOTR_LIST));rr_set_attr(out,out.root,"_ring_free_side",rr_scalar(out,WOTR_NONE));
    if(draw){tc_draw_event_card(out,0,deck);bc_settle_draw(out,0,deck);} // Package W (P9b): an engine-granted draw settles one unit of owed-draw debt (game_logic.grant_draw, ai.py:11805)
    tc_open_hand_limit_discard(out);return child;
}
