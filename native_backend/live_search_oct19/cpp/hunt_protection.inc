// Resume learned Hunt card windows without spending another action die.
uint64_t cbp_remove_card(Snapshot& s,uint64_t persistent,uint64_t card);
void cbp_discard_table(Snapshot& s,uint64_t card);
bool ehd_stash_kind(std::string_view kind);
void ehd_draw(Snapshot&,const Snapshot&,void*,std::string_view,uint64_t,am_Builder&,std::vector<uint64_t>&,bool,uint64_t);
void ehd_apply_tile(Snapshot&,const Snapshot&,void*,std::string_view,uint64_t,uint64_t,am_Builder&,std::vector<uint64_t>&);
void ehd_foul_confront(Snapshot&,const Snapshot&,void*,uint64_t,int64_t,am_Builder&,std::vector<uint64_t>&);
int64_t hpr_integer(const Snapshot& s,uint64_t dict,const char* name,int64_t fallback=0){const auto value=field(s,dict,name,false);return value==missing?fallback:typed(s,value,WOTR_INT,"Protection stash integer expected").integer;}
std::string hpr_family(const Snapshot& s){
    const auto pending=obs_attr(s,s.root,"pending_card_option");require(!ctx_none(s,pending),"Hunt protection choice missing");
    const bool balrog=text_is(s,field(s,pending,"card_name"),"Balrog of Moria");const char* owner=balrog?"sh":"fp";
    require(text_is(s,field(s,pending,"side"),owner) && text_is(s,field(s,pending,"decider"),owner) && truth(s,field(s,pending,"direct")),"Hunt protection requires its direct decision owner");
    if(balrog)return "balrog";
    if(truth(s,field(s,pending,"hunt_reduction",false)))return "reduction";
    if(text_is(s,field(s,pending,"card_name"),"Wizard's Staff"))return "wizard";
    if(text_is(s,field(s,pending,"card_name"),"Mithril Coat and Sting"))return "mithril";
    throw std::invalid_argument("Pending card is not a Hunt protection window");
}
void hpr_stash_gate(const Snapshot& s,uint64_t stash){
    require(!ctx_none(s,stash),"Hunt protection stash missing");typed(s,stash,WOTR_DICT,"Hunt protection stash must be dictionary");
    const auto kind=field(s,stash,"apply_kind",false);require(ctx_none(s,kind) || text_is(s,kind,"stronghold") || ehd_stash_kind(text(s,kind)),"Hunt card draw needs its event continuation");
    if(!ctx_none(s,kind) && ehd_stash_kind(text(s,kind))){
        const auto choices=field(s,stash,"choices");typed(s,choices,WOTR_DICT,"Hunt event resume choices must be dictionary");const auto entries=child_edges(s,choices);
        for(size_t i=0;i<entries.size();i+=2)require(text_is(s,entries[i],"_skip_wizard") || (text_is(s,kind,"breaking") && text_is(s,entries[i],"_breaking_card_id")),"Hunt event refuses teacher-forced continuation choices");
    }
    require(!truth(s,field(s,stash,"return_pending",false)),"Mithril choice already resolved");
    const auto automatic=field(s,stash,"skip_auto_sacrifice",false);require(automatic==missing || truth(s,automatic),"Hunt protection refuses automatic casualty policy");
}
std::unique_ptr<Snapshot> native_hunt_protection_actions(const Snapshot& source){
    hpr_family(source);auto out=std::make_unique<Snapshot>(source);auto& s=*out;
    const auto pending=obs_attr(s,s.root,"pending_card_option"),options=field(s,pending,"options");typed(s,options,WOTR_LIST,"Protection options must be list");
    std::vector<uint64_t> rows;int64_t index=0;
    for(auto choice:child_edges(s,options)){
        const auto params=hpr_dict(s,{rr_string(s,"idx"),rr_scalar(s,WOTR_INT,index++),rr_string(s,"card_id"),field(s,pending,"card_id"),rr_string(s,"card_name"),field(s,pending,"card_name"),rr_string(s,"side"),field(s,pending,"side"),rr_string(s,"choices"),choice});
        const auto row=rr_scalar(s,WOTR_TUPLE);set_edges(s,row,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),params});rows.push_back(row);
    }
    s.root=rr_scalar(s,WOTR_LIST);set_edges(s,s.root,rows);return ac_compact(s);
}
void hpr_discard(Snapshot& s,std::string_view type){
    const auto card=hpr_card(s,type);if(card!=missing)cbp_discard_table(s,cbp_remove_card(s,obs_attr(s,s.root,"persistent_cards"),card));
}
void hpr_apply_stashed(Snapshot& s,const Snapshot& t,void* rng,uint64_t stash,uint64_t tile,am_Builder& events,std::vector<uint64_t>& rows){
    const auto kind=field(s,stash,"apply_kind",false);
    if(!ctx_none(s,kind) && ehd_stash_kind(text(s,kind))){const auto name=std::string(text(s,kind));ehd_apply_tile(s,t,rng,name,field(s,stash,"choices"),tile,events,rows);return;}
    if(text_is(s,field(s,stash,"apply_kind",false),"stronghold")){fr_apply_stronghold(s,tile);return;}
    const bool track=truth(s,field(s,stash,"mordor_track",false));
    const auto value=ht_apply(s,t,rng,tile,hpr_integer(s,stash,"successes"),truth(s,field(s,stash,"mordor",false)),track,truth(s,field(s,stash,"eye_no_effect",false)),events,rows);
    if(track && value.stop){const auto fs=obs_attr(s,s.root,"fellowship");rr_set_attr(s,fs,"mordor_step",rr_scalar(s,WOTR_INT,std::max<int64_t>(0,ht_integer(s,fs,"mordor_step")-1)));}
}
fm_Result apply_hunt_protection(const Snapshot& source,const Snapshot& t,const void* random_source,int64_t index){
    const auto family=hpr_family(source);hc_gate(source,false,true,false,true);
    const auto pending=obs_attr(source,source.root,"pending_card_option"),options=field(source,pending,"options");typed(source,options,WOTR_LIST,"Protection options must be list");
    require(index>=0 && static_cast<uint64_t>(index)<source.nodes[options].count,"Protection choice index out of range");
    if(family=="wizard" || family=="mithril")hpr_stash_gate(source,obs_attr(source,source.root,family=="wizard"?"_wizardstaff_stash":"_mithril_stash"));
    fm_Result result;result.state=action_clone(source);result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());
    auto& s=*result.state;am_Builder events;std::vector<uint64_t> rows;const auto copied_pending=obs_attr(s,s.root,"pending_card_option"),copied_options=field(s,copied_pending,"options"),choice=child_edges(s,copied_options)[index];
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));
    if(family=="reduction"){
        if(truth(s,field(s,choice,"use_hunt_reduction",false))){
            const auto requested=field(s,choice,"card_id",false),id=truth(s,requested)?requested:field(s,copied_pending,"card_id"),persistent=obs_attr(s,s.root,"persistent_cards");
            uint64_t card=missing;for(const char* side:{"fp_cards","sh_cards"})for(auto candidate:child_edges(s,obs_attr(s,persistent,side)))if(text(s,obs_attr(s,candidate,"card_id"))==text(s,id)){card=candidate;break;}
            if(card!=missing)cbp_discard_table(s,cbp_remove_card(s,persistent,card));
            rr_set_attr(s,s.root,"_pending_hunt_damage",rr_scalar(s,WOTR_INT,std::max<int64_t>(0,ht_integer(s,s.root,"_pending_hunt_damage")-1)));
        }
        const auto resume=field(s,copied_pending,"foul_resume",false);
        if(!ctx_none(s,resume)){
            ehd_foul_confront(s,t,result.random.get(),resume,std::max<int64_t>(0,hpr_integer(s,resume,"damage")-(truth(s,field(s,choice,"use_hunt_reduction",false))?1:0)),events,rows);
            rr_set_attr(s,s.root,"_pending_hunt_damage",rr_scalar(s,WOTR_INT));
        }else hd_settle(s,t);
    }else if(family=="balrog"){
        if(truth(s,field(s,choice,"balrog_declare_draw",false)) && hpr_card(s,"BALROG_OF_MORIA")!=missing){
            ht_resolve(s,t,result.random.get(),1,false,false,true,events,rows);hpr_discard(s,"BALROG_OF_MORIA");
        }
    }else if(family=="wizard"){
        const auto stash=obs_attr(s,s.root,"_wizardstaff_stash");rr_set_attr(s,s.root,"_wizardstaff_stash",rr_scalar(s,WOTR_NONE));
        if(truth(s,field(s,choice,"wizardstaff_prevent",false)))hpr_discard(s,"WIZARD_STAFF");
        else if(!ctx_none(s,field(s,stash,"apply_kind",false)) && ehd_stash_kind(text(s,field(s,stash,"apply_kind")))){
            const auto kind=std::string(text(s,field(s,stash,"apply_kind")));auto entries=child_edges(s,field(s,stash,"choices"));
            entries.push_back(rr_string(s,"_skip_wizard"));entries.push_back(rr_scalar(s,WOTR_BOOL,1));const auto choices=hpr_dict(s,entries);
            ehd_draw(s,t,result.random.get(),kind,choices,events,rows,true,missing);
        }
        else if(text_is(s,field(s,stash,"apply_kind",false),"stronghold")){
            const auto region=std::string(text(s,field(s,stash,"region"))),reason=std::string(text(s,field(s,stash,"reason")));
            fr_stronghold_draw(s,t,result.random.get(),region,events,rows,true,true,reason);
        }else{
            const bool track=truth(s,field(s,stash,"mordor_track",false));const auto value=ht_resolve(s,t,result.random.get(),hpr_integer(s,stash,"successes"),track || truth(s,field(s,stash,"mordor",false)),track,truth(s,field(s,stash,"eye_no_effect",false)),events,rows,true);
            if(track && value.stop){const auto fs=obs_attr(s,s.root,"fellowship");rr_set_attr(s,fs,"mordor_step",rr_scalar(s,WOTR_INT,std::max<int64_t>(0,ht_integer(s,fs,"mordor_step")-1)));}
        }
    }else{
        const auto stash=obs_attr(s,s.root,"_mithril_stash");rr_set_attr(s,s.root,"_mithril_stash",rr_scalar(s,WOTR_NONE));
        if(!truth(s,field(s,choice,"mithril_redraw",false)))hpr_apply_stashed(s,t,result.random.get(),stash,field(s,stash,"tile"),events,rows);
        else{
            hpr_discard(s,"MITHRIL_COAT");const auto pool=obs_attr(s,s.root,"hunt_pool");if(!truth(s,pool))ht_recycle(s,result.random.get(),events,rows);
            auto entries=child_edges(s,stash);entries.push_back(rr_string(s,"return_pending"));entries.push_back(rr_scalar(s,WOTR_BOOL,1));const auto returning=hpr_dict(s,entries);rr_set_attr(s,s.root,"_mithril_stash",returning);
            if(truth(s,pool)){auto tiles=child_edges(s,pool);const auto pick=hf_pool(s,pool,result.random.get(),true);const auto replacement=tiles[pick];tiles.erase(tiles.begin()+pick);set_edges(s,pool,tiles);ht_trace_draw(s,replacement,"mithril redraw",events,rows);hpr_apply_stashed(s,t,result.random.get(),stash,replacement,events,rows);}
            hpr_return_mithril(s,result.random.get());
        }
    }
    if(family!="reduction" && ht_integer(s,s.root,"_pending_hunt_damage")>0 && ctx_none(s,obs_attr(s,s.root,"pending_hunt")) && !hpr_offer_reduction(s))hd_settle(s,t);
    epa_maybe(s);ew_open(s,result.random.get());fr_open(s,t,result.random.get(),events,rows);hpr_return_mithril(s,result.random.get());fm_finish(s,t);epa_maybe(s);cau_gandalf(s);
    events.out.root=events.container(WOTR_LIST,rows);result.events=std::make_unique<Snapshot>(std::move(events.out));return result;
}
