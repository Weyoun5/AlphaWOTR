// Return to Valinor: fixed printed order with FP-owned Elven losses.
bool erv_supported(const Snapshot& t,uint64_t card){return card!=missing&&text_is(t,obs_attr(t,card,"card_name"),"Return to Valinor");}
void erv_effect(Snapshot& s,const Snapshot& t,std::string_view id,std::string_view after,void* rng,am_Builder& events,std::vector<uint64_t>& rows){
    bool skipping=!after.empty();
    for(const std::string region:{"rivendell","lorien","grey havens","woodland realm"}){
        if(skipping){if(region==sanitize_name(after))skipping=false;continue;}
        if(contains_string(s,obs_attr(s,s.root,"besieged_strongholds",false),region))continue;
        const auto army=mu_army(s,region);if(army==missing)continue;
        const auto units=child_edges(s,obs_attr(s,army,"units"));
        // The pinned source classifies this handler's army by its FIRST unit.
        if(units.empty()||ctx_unit_side(s,units.front())!=0)continue;
        int count=0;for(auto unit:units)count+=enum_member(s,obs_attr(s,unit,"nation"),"game_state.Nation")=="ELVES";
        count=std::min(5,count);if(!count)continue;
        std::vector<uint32_t> rolls(count);hf_roll(rng,rolls.data(),rolls.size(),"card effect");
        int64_t hits=0;std::vector<uint64_t> faces;for(auto roll:rolls){hits+=roll==6;faces.push_back(events.integer(roll));}
        rows.push_back(events.row(-1,"dice_roll",{events.str("faces"),events.container(WOTR_LIST,faces),events.str("ctx"),events.str("Return to Valinor roll"),events.str("side"),rr_scalar(events.out,WOTR_NONE)}));
        if(hits>0&&!ecc_route(s,t,region,0,hits,id,"Return to Valinor",rng))return;
    }
}
// package Q: mouth_muster = the Mouth of Sauron's once-per-turn Muster-as-Army funding (ai.py:11535-11565); payment only.
fm_Result apply_return_valinor_event(const Snapshot& source,const Snapshot& t,const void* rng,int side,int64_t index,std::string_view id,bool mouth_muster=false){
    require(side==1,"Return to Valinor belongs to Shadow");quiet_stable_action_gate(source,t);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==1,"Return to Valinor violates retained Ring actor");
    const auto card=ev_card(t,id);require(erv_supported(t,card)&&contains_string(source,tc_hand(source,1),id),"Return to Valinor absent from Shadow hand");require(ev_requirement(source,t,card),"Return to Valinor needs Shadow control of an Elven stronghold");
    const auto die=ev_payment(source,t,1,index,card,mouth_muster);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace"));
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(rng));if(!result.random)throw std::invalid_argument(wotr_random_last_error());auto& s=*result.state;
    rr_set_attr(s,s.root,"_casualty_operations",cp_container(s,WOTR_LIST,{}));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));ev_consume_payment(s,t,1,index,card,mouth_muster);
    am_Builder events;std::vector<uint64_t> rows;erv_effect(s,t,id,"",result.random.get(),events,rows);esr_commit(s,t,id,"strategy",face);
    cau_finish(s,t,result.random.get(),-1,events,rows);events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
