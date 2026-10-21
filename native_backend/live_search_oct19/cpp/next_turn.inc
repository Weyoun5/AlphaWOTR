// Turn-boundary settlement repair uses standing occupancy, not entry effects.
int nt_controller(const Snapshot& s,const Snapshot& t,std::string_view key){
    const auto controls=obs_attr(s,s.root,"enemy_controlled");
    if(contains_string(s,ctx_side_value(s,controls,1),key))return 1;
    if(contains_string(s,ctx_side_value(s,controls,0),key))return 0;
    return contains_string(t,field(t,t.root,"fp_region_keys"),key)?0:1;
}
void nt_capture(Snapshot& s,const Snapshot& t,std::string_view key,int side){
    const bool fp=contains_string(t,field(t,t.root,"fp_region_keys"),key);
    const bool sh=contains_string(t,field(t,t.root,"shadow_region_keys"),key);
    if(!fp&&!sh)return;
    const auto controls=obs_attr(s,s.root,"enemy_controlled");
    const auto own=ctx_side_value(s,controls,side),other=ctx_side_value(s,controls,1-side);
    if((side==1&&fp)||(side==0&&sh)){
        const bool first=!contains_string(s,own,key);me_set_member(s,own,key,true);me_set_member(s,other,key,false);
        if(!first)return;
        const auto nation=ctx_get(t,field(t,t.root,"movement_region_nations"),key);if(ctx_none(t,nation))return;
        const auto name=ctx_nation(t,nation);const auto political=obs_attr(s,s.root,"political");
        const auto boxes=obs_attr(s,political,"boxes"),active=obs_attr(s,political,"active");
        const auto value=ctx_nation_value(s,boxes,name);
        const auto current=ctx_none(s,value)?0:typed(s,value,WOTR_INT,"capture political box must be integer").integer;
        require(current>=0&&current<=3,"capture political box out of range");if(current>=3)return;
        bool activated=truth(s,ctx_nation_value(s,active,name));
        if(!activated){
            const bool blocked=!wt_can_activate(s,name,"enemy_entry",key);
            if(!blocked){me_activate(s,name);activated=true;}
        }
        if(current+1!=3||activated)me_nation_assign(s,boxes,name,rr_scalar(s,WOTR_INT,current+1));
    }else me_set_member(s,other,key,false);
}
void nt_enforce_control(Snapshot& s,const Snapshot& t){
    const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));
    for(size_t i=0;i<armies.size();i+=2){
        const auto units=child_edges(s,obs_attr(s,armies[i+1],"units"));if(units.empty())continue;
        const auto key=sanitize_name(text(s,armies[i]));const auto meta=ctx_get(t,field(t,t.root,"board_regions"),key);
        if(meta==missing||(!truth(t,ctx_get(t,meta,"is_city"))&&!truth(t,ctx_get(t,meta,"is_stronghold"))&&!truth(t,ctx_get(t,meta,"is_town"))))continue;
        std::array<bool,2> present{};
        for(auto unit:units){const auto nation=ctx_nation(s,obs_attr(s,unit,"nation"));validate_nation_name(nation);present[fp_nation(nation)?0:1]=true;}
        if(present[0]==present[1])continue;const int side=present[0]?0:1;
        if(nt_controller(s,t,key)!=side)nt_capture(s,t,key,side);
    }
}
void nt_deal(Snapshot& s,am_Builder& b,std::vector<uint64_t>& rows){
    for(int side=0;side<2;++side)for(const auto deck:{"character","strategy"}){
        uint64_t card;
        try{card=tc_draw_event_card(s,side,deck);}
        catch(const std::runtime_error& error){if(std::string_view(error.what()).find("deck empty")!=std::string_view::npos)continue;throw;}
        rows.push_back(b.row(-1,"card_drawn",{b.str("deck"),b.str(deck),b.str("card_id"),b.str(text(s,card)),b.str("side"),b.str(side==0?"FP":"SH")}));
    }
    tc_open_hand_limit_discard(s);
}
