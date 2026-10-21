// Complete menus for proved combat continuations; all other phases refuse.
std::unique_ptr<Snapshot> native_combat_entry_actions(const Snapshot&,const Snapshot&,int);
std::unique_ptr<Snapshot> native_combat_phase_actions(const Snapshot&,const Snapshot&,int);
std::unique_ptr<Snapshot> native_combat_pending_actions(const Snapshot& source,const Snapshot& tables,int32_t side){
    require(side==0||side==1,"Combat actor must be FP or SH");
    const auto pc=obs_attr(source,source.root,"pending_combat");require(!ctx_none(source,pc),"No combat is pending");
    const auto owner=field(source,pc,"deciding_side",false);
    if(!ctx_none(source,owner)&&ctx_side(source,owner)!=side){am_Builder b;b.out.root=b.container(WOTR_LIST,{});return std::make_unique<Snapshot>(std::move(b.out));}
    const auto phase=field(source,pc,"phase");
    const auto effect_entry=field(source,pc,"effect_choice",false);
    if(text_is(source,phase,"attacker_combat_card")||text_is(source,phase,"defender_combat_card")||text_is(source,phase,"combat_card_reveal")||text_is(source,phase,"fp_shine")||(!ctx_none(source,effect_entry)&&text_is(source,field(source,effect_entry,"stage",false),"pre")))return native_combat_entry_actions(source,tables,side);
    if(text_is(source,phase,"casualty_allocation")){
        const auto stage=field(source,pc,"alloc_stage");require(text_is(source,stage,"attacker")||text_is(source,stage,"defender"),"Invalid casualty stage");
        const std::string role(text(source,stage));const auto count=field(source,pc,role=="attacker"?"atk_hits":"def_hits",false);
        const int64_t hits=ctx_none(source,count)?0:typed(source,count,WOTR_INT,"Casualty hits must be integer").integer;
        const std::string region(text(source,field(source,pc,(role+"_region").c_str())));
        auto menu=combat_casualty_menu(source,ctx_side(source,field(source,pc,(role+"_side").c_str())),region.c_str(),hits);
        return menu; // Package W: an empty casualty menu is returned as the frozen _casualty_operations returns []
    }
    if(!text_is(source,phase,"combat_effect_choice"))return native_combat_phase_actions(source,tables,side);
    auto out=std::make_unique<Snapshot>(source);auto& s=*out;const auto effect=field(s,pc,"effect_choice"),stage=field(s,effect,"stage",false);
    std::vector<uint64_t> rows;
    auto append=[&](uint64_t params){rows.push_back(cp_container(s,WOTR_TUPLE,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"combat_effect_choice"),params}));};
    const auto id=field(s,effect,"card_id",false);const auto card=id==missing?rr_string(s,""):id;
    if(text_is(s,stage,"round")||text_is(s,stage,"onslaught_payment")){
        const auto options=field(s,effect,"options");require(truth(s,options),"Paused combat round has no legal choices");
        for(auto option:child_edges(s,options)){
            typed(s,option,WOTR_DICT,"Round choice must be dictionary");
            const auto params=cp_container(s,WOTR_DICT,{rr_string(s,"card_id"),card});const auto pairs=child_edges(s,option);
            for(size_t i=0;i<pairs.size();i+=2)cp_put(s,params,std::string(text(s,pairs[i])).c_str(),pairs[i+1]);append(params);
        }
    }else if(text_is(s,stage,"post")&&text_is(s,field(s,effect,"effect",false),"onslaught")){
        const auto limit=field(s,effect,"max_count",false);const auto maximum=ctx_none(s,limit)?0:typed(s,limit,WOTR_INT,"Onslaught count must be integer").integer;
        require(maximum>=0&&maximum<=4,"Onslaught maximum exceeds native legal range");
        for(int64_t count=0;count<=maximum;++count)append(cp_container(s,WOTR_DICT,{rr_string(s,"card_id"),card,rr_string(s,"count"),rr_scalar(s,WOTR_INT,count)}));
    }else throw std::invalid_argument("Complete combat choice stage is not ported");
    s.root=cp_container(s,WOTR_LIST,rows);return ac_compact(s);
}
