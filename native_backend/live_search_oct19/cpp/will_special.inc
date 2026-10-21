// The two Will-of-the-West promotions. Board companion and global marker
// are distinct objects; neither promotion adds an immediate action die.
bool ws_named(const Snapshot& s,uint64_t leader,std::string_view name){return object_type(s,leader)=="game_state.Companion" && text_is(s,obs_attr(s,leader,"name"),name);}
bool ws_in_fellowship(const Snapshot& s,std::string_view name){for(auto c:child_edges(s,obs_attr(s,obs_attr(s,s.root,"fellowship"),"companions")))if(text_is(s,obs_attr(s,c,"name"),name))return true;return false;}
std::string ws_last_region(const Snapshot& s,std::string_view name){
    std::string out;const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));
    for(size_t i=0;i<armies.size();i+=2)for(auto l:child_edges(s,obs_attr(s,armies[i+1],"leaders")))if(ws_named(s,l,name)){out=std::string(text(s,armies[i]));break;}return out;
}
std::vector<std::string> ws_gandalf_regions(const Snapshot& s){
    const auto white=obs_attr(s,s.root,"gandalf_the_white",false);if(!ctx_none(s,white) && truth(s,obs_attr(s,white,"in_play")))return {};
    if(ws_in_fellowship(s,"Gandalf the Grey"))return {};
    bool previous=false;for(auto key:{"witch_king","saruman","mouth_of_sauron"}){const auto marker=obs_attr(s,s.root,key,false);previous|=!ctx_none(s,marker) && truth(s,obs_attr(s,marker,"was_ever_in_play"));}
    if(!previous)return {};
    const auto region=ws_last_region(s,"Gandalf the Grey");if(!region.empty())return {region};
    const auto captured=ctx_side_value(s,obs_attr(s,s.root,"enemy_controlled"),1);std::vector<std::string> out;
    for(auto key:{"fangorn","grey havens","rivendell","lorien","woodland realm"})if(!contains_string(s,captured,key))out.push_back(key);return out;
}
std::string ws_aragorn_region(const Snapshot& s){
    const auto crowned=obs_attr(s,s.root,"aragorn_crowned",false);if(!ctx_none(s,crowned) && truth(s,obs_attr(s,crowned,"crowned")))return {};
    if(ws_in_fellowship(s,"Strider"))return {};
    const auto region=ws_last_region(s,"Strider"),key=sanitize_name(region);
    if(key!="minas tirith" && key!="pelargir" && key!="dol amroth")return {};
    if(contains_string(s,ctx_side_value(s,obs_attr(s,s.root,"enemy_controlled"),1),key))return {};return region;
}
std::unique_ptr<Snapshot> native_will_special_actions(const Snapshot& s,const Snapshot& t,int32_t side){
    require(side==0 || side==1,"Will special menu actor invalid");quiet_stable_action_gate(s,t);ring_waiver(s,side);am_Builder b;std::vector<uint64_t> rows;
    if(side==0){
        const auto gandalf=ws_gandalf_regions(s);const bool aragorn=!ws_aragorn_region(s).empty();const auto dice=child_edges(s,pa_pool(s,0));
        for(size_t i=0;i<dice.size();++i){
            const auto die=dice[i];if(truth(s,obs_attr(s,die,"used")) || !enum_is(s,obs_attr(s,die,"current"),"game_state.ActionDieFace","WILL_OF_WEST"))continue;
            for(const auto& region:gandalf)rows.push_back(b.row(i,"transform_gandalf",{b.str("region"),b.str(region)}));
            if(aragorn)rows.push_back(b.row(i,"crown_aragorn",{}));break;
        }
    }
    b.out.root=b.container(WOTR_LIST,rows);return std::make_unique<Snapshot>(b.out);
}
uint64_t ws_payment(const Snapshot& s,int64_t index){
    const auto dice=child_edges(s,pa_pool(s,0));require(index>=0 && static_cast<uint64_t>(index)<dice.size(),"Promotion die index out of range");const auto die=dice[index];
    require(object_type(s,die)=="game_state.ActionDie" && ctx_side(s,obs_attr(s,die,"side"))==0 && !truth(s,obs_attr(s,die,"used")) && enum_is(s,obs_attr(s,die,"current"),"game_state.ActionDieFace","WILL_OF_WEST"),"Promotion needs named unused FP Will die");return die;
}
uint64_t ws_companion(Snapshot& s,bool gandalf){
    const auto none=rr_scalar(s,WOTR_NONE),no=rr_scalar(s,WOTR_BOOL),yes=rr_scalar(s,WOTR_BOOL,1),one=rr_scalar(s,WOTR_INT,1),zero=rr_scalar(s,WOTR_INT);
    return mu_object(s,"game_state.Companion",{{"name",rr_string(s,gandalf?"Gandalf the White":"Aragorn")},{"level",rr_scalar(s,WOTR_INT,3)},
        {"leadership",gandalf?one:rr_scalar(s,WOTR_INT,2)},{"captain_of_the_west",gandalf?no:yes},{"guide_ability",none},
        {"activates_nations",rr_scalar(s,WOTR_LIST)},{"activation_regions",rr_scalar(s,WOTR_LIST)},{"can_muster_nation",none},{"muster_regions",rr_scalar(s,WOTR_LIST)},
        {"bonus_movement",gandalf?one:zero},{"solo_hobbit_movement",gandalf?yes:no}});
}
std::unique_ptr<Snapshot> apply_will_special_action(const Snapshot& source,const Snapshot& t,int32_t side,int64_t index,bool gandalf,std::string_view raw_region){
    require(side==0,"Promotions belong to FP");quiet_stable_action_gate(source,t);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained) || ctx_side(source,retained)==0,"Promotion belongs to retained Ring actor");
    const auto marker=obs_attr(source,source.root,gandalf?"gandalf_the_white":"aragorn_crowned",false);require(!ctx_none(source,marker),"Promotion requires its registered marker");
    std::string region;
    if(gandalf){const auto valid=ws_gandalf_regions(source);region=sanitize_name(raw_region);bool okay=false;for(const auto& key:valid)okay|=sanitize_name(key)==region;require(okay,"Gandalf promotion unavailable or destination invalid");}
    else{region=ws_aragorn_region(source);require(!region.empty(),"Aragorn promotion unavailable");}
    ws_payment(source,index);auto child=action_clone(source);auto& s=*child;
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));rr_set_attr(s,ws_payment(s,index),"used",rr_scalar(s,WOTR_BOOL,1));
    const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));const auto companion=ws_companion(s,gandalf);
    if(gandalf){
        for(size_t i=1;i<armies.size();i+=2){std::vector<uint64_t> leaders;for(auto l:child_edges(s,obs_attr(s,armies[i],"leaders")))if(!ws_named(s,l,"Gandalf the Grey"))leaders.push_back(l);
            const auto next=rr_scalar(s,WOTR_LIST);set_edges(s,next,leaders);rr_set_attr(s,armies[i],"leaders",next);
        }
        rr_set_attr(s,obs_attr(s,s.root,"gandalf_the_white"),"in_play",rr_scalar(s,WOTR_BOOL,1));
        auto army=mu_army(s,region);if(army==missing){const auto reg=ctx_get(s,obs_attr(s,obs_attr(s,s.root,"board"),"_regions"),region);require(reg!=missing,"Gandalf placement region absent from Board");army=am_new_army(s,reg);am_insert(s,region,army);}
        am_extend(s,army,"leaders",{companion});
        cbp_companion_death(s,nullptr,false); // Grey left play: requirement discards, no casualty.
    }else{
        bool found=false;
        for(size_t i=0;i<armies.size() && !found;i+=2){const auto list=obs_attr(s,armies[i+1],"leaders");auto leaders=child_edges(s,list);
            for(auto& l:leaders)if(ws_named(s,l,"Strider")){l=companion;set_edges(s,list,leaders);found=true;break;}
        }
        require(found,"Strider disappeared during promotion clone");rr_set_attr(s,obs_attr(s,s.root,"aragorn_crowned"),"crowned",rr_scalar(s,WOTR_BOOL,1));
        if(!active_nation(s,political_view(s,"GONDOR",true),"GONDOR",false))me_activate(s,"GONDOR");
    }
    tc_open_hand_limit_discard(s);rr_set_attr(s,s.root,"_battle_resolved_attacker",rr_scalar(s,WOTR_NONE));return child;
}
