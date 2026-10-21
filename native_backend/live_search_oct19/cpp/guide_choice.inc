// Learned highest-level Guide selection. Compulsory Guide choices retain
// their preaction tail marker and never roll dice or draw cards here.
void cau_gandalf(Snapshot&);
std::unique_ptr<Snapshot> native_guide_actions(const Snapshot& s,int32_t side){
    require(side==0 || side==1,"Guide menu side must be FP or SH");
    require(ctx_none(s,obs_attr(s,s.root,"pending_combat")),"Combat decision precedes Guide selection");
    const auto pending=obs_attr(s,s.root,"pending_guide_change");
    require(!ctx_none(s,pending),"There is no pending Guide selection");
    am_Builder b;std::vector<uint64_t> rows;
    if(side==0){
        const auto candidates=field(s,pending,"candidates");typed(s,candidates,WOTR_LIST,"Guide candidates must be an ordered list");
        for(auto name:child_edges(s,candidates))rows.push_back(b.row(-1,"choose_guide",{b.str("companion"),b.str(text(s,name)),b.str("side"),b.str("fp")}));
    }
    b.out.root=b.container(WOTR_LIST,rows);return std::make_unique<Snapshot>(std::move(b.out));
}
void gc_choose_inner(Snapshot& s,std::string_view name){
    const auto pending=obs_attr(s,s.root,"pending_guide_change");
    require(!ctx_none(s,pending),"Guide choice requires pending legal candidates");
    const auto candidates=field(s,pending,"candidates",false);
    require(candidates!=missing && contains_string(s,candidates,name),"Guide choice is not among pending legal candidates");
    const auto fs=obs_attr(s,s.root,"fellowship"),comps=obs_attr(s,fs,"companions");
    typed(s,comps,WOTR_LIST,"Guide selection requires companions list");
    uint64_t chosen=missing;int64_t highest=std::numeric_limits<int64_t>::min();
    for(auto companion:child_edges(s,comps)){
        const auto level=typed(s,obs_attr(s,companion,"level"),WOTR_INT,"Guide levels must be integer").integer;
        if(level>highest){highest=level;chosen=missing;}
        if(level==highest && chosen==missing && text_is(s,obs_attr(s,companion,"name"),name))chosen=companion;
    }
    require(s.nodes[comps].count==0 || chosen!=missing,"Chosen Guide is not among highest-level companions");
    const auto new_fs=obs_attr(s,s.root,"fellowship"),new_comps=obs_attr(s,new_fs,"companions");
    if(s.nodes[new_comps].count==0)ft_guide(s);
    else for(auto companion:child_edges(s,new_comps))
        if(text_is(s,obs_attr(s,companion,"name"),name) &&
           typed(s,obs_attr(s,companion,"level"),WOTR_INT,"Guide level must be integer").integer==highest){
            rr_set_attr(s,new_fs,"guide",companion);break;
        }
    rr_set_attr(s,s.root,"pending_guide_change",rr_scalar(s,WOTR_NONE));
}
std::unique_ptr<Snapshot> apply_guide_choice(const Snapshot& s,const Snapshot& t,std::string_view name){
    const bool queued=!ctx_none(s,obs_attr(s,s.root,"_gandalf_draw_pending",false));
    // Package W: a queued grant never refuses the guide choice beside a card option / tracked overflow (frozen: none)
    quiet_action_gate(s,t,false,false,true,false,false,true,true,false,false,false,queued,true);
    auto child=action_clone(s);auto& out=*child;
    rr_set_attr(out,out.root,"_casualty_operations",rr_scalar(out,WOTR_LIST));
    rr_set_attr(out,out.root,"_ring_free_side",rr_scalar(out,WOTR_NONE));
    if(truth(out,obs_attr(out,out.root,"_declare_phase",false)))rr_set_attr(out,out.root,"_declare_phase",rr_scalar(out,WOTR_BOOL));
    gc_choose_inner(out,name);if(queued)cau_gandalf(out);return child;
}
