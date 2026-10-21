// Stable-point siege/control reconciliation, preserving learned overstack choices.
int su_side(const Snapshot& s,uint64_t unit){const auto nation=ctx_nation(s,obs_attr(s,unit,"nation"));validate_nation_name(nation);return fp_nation(nation)?0:1;}
std::string su_nation_value(std::string_view name){
    const std::array<std::string_view,8> keys={"ELVES","DWARVES","GONDOR","THE_NORTH","ROHAN","ISENGARD","SAURON","SOUTHRON_EASTERLINGS"};
    const std::array<std::string_view,8> values={"Elves","Dwarves","Gondor","The North","Rohan","Isengard","Sauron","Southrons & Easterlings"};
    for(size_t i=0;i<keys.size();++i)if(keys[i]==name)return std::string(values[i]);throw std::invalid_argument("Unknown nation value");
}
int su_cap(const Snapshot& s,const Snapshot& t,std::string_view region,int side){
    if(!contains_string(s,obs_attr(s,s.root,"besieged_strongholds",false),region))return 10;
    int holder=-1;const auto controls=child_edges(s,obs_attr(s,s.root,"enemy_controlled"));
    // event_effects._stack_cap deliberately uses dictionary order here.
    for(size_t i=0;i<controls.size();i+=2)if(contains_string(s,controls[i+1],region)){holder=ctx_side(s,controls[i]);break;}
    if(holder<0)holder=contains_string(t,field(t,t.root,"fp_region_keys"),region)?0:1;
    return side==holder?5:10;
}
void su_resolve_overstacks(Snapshot& s,const Snapshot& t){
    const auto regions=obs_attr(s,s.root,"_overstack_regions",false);if(!truth(s,regions))return;
    typed(s,regions,WOTR_SET,"Overstack regions must be a set");
    if(!ctx_none(s,obs_attr(s,s.root,"pending_card_option",false)))return;
    for(;;){
        std::vector<std::string> keys;for(auto key:child_edges(s,regions))keys.emplace_back(text(s,key));std::sort(keys.begin(),keys.end());bool repeat=false;
        for(const auto& region:keys){
            const auto army=mu_army(s,sanitize_name(region));if(army==missing){me_set_member(s,regions,region,false);continue;}
            bool resolved=true;
            for(int side=0;side<2;++side){
                const auto list=obs_attr(s,army,"units");auto all=child_edges(s,list);std::vector<uint64_t> mine;
                for(auto unit:all)if(su_side(s,unit)==side)mine.push_back(unit);
                const int cap=su_cap(s,t,sanitize_name(region),side);if(mine.size()<=static_cast<size_t>(cap))continue;
                std::set<std::pair<bool,std::string>> kinds;
                for(auto unit:mine)kinds.emplace(enum_is(s,obs_attr(s,unit,"kind"),"game_state.UnitKind","ELITE"),su_nation_value(ctx_nation(s,obs_attr(s,unit,"nation"))));
                if(kinds.size()==1){
                    all.erase(std::find(all.begin(),all.end(),mine.front()));set_edges(s,list,all);
                    if(mine.size()-1>static_cast<size_t>(cap))resolved=false;
                    continue;
                }
                std::vector<uint64_t> options;
                for(const auto& kind:kinds){
                    const auto spec=cp_container(s,WOTR_DICT,{rr_string(s,"region"),rr_string(s,region),rr_string(s,"nation"),rr_string(s,kind.second),rr_string(s,"kind"),rr_string(s,kind.first?"elite":"regular")});
                    options.push_back(cp_container(s,WOTR_DICT,{rr_string(s,"overstack_remove"),spec}));
                }
                const auto id=rr_string(s,"__overstack__"),owner=rr_string(s,side==0?"fp":"sh");
                const auto pending=cp_container(s,WOTR_DICT,{rr_string(s,"card_id"),id,rr_string(s,"card_name"),id,rr_string(s,"side"),owner,rr_string(s,"decider"),owner,rr_string(s,"direct"),rr_scalar(s,WOTR_BOOL,1),rr_string(s,"options"),cp_container(s,WOTR_LIST,options)});
                rr_set_attr(s,s.root,"pending_card_option",pending);return;
            }
            if(resolved)me_set_member(s,regions,region,false);else{repeat=true;break;}
        }
        if(!repeat)return;
    }
}
void su_cleanup(Snapshot& s,const Snapshot& t){
    nt_enforce_control(s,t);
    auto sieges=obs_attr(s,s.root,"besieged_strongholds",false);
    if(ctx_none(s,sieges)){sieges=rr_scalar(s,WOTR_SET);rr_set_attr(s,s.root,"besieged_strongholds",sieges);}
    typed(s,sieges,WOTR_SET,"Sieges must be a set");const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));
    for(size_t i=0;i<armies.size();i+=2){
        const auto key=sanitize_name(text(s,armies[i]));if(contains_string(s,sieges,key))continue;
        const auto meta=ctx_get(t,field(t,t.root,"board_regions"),key);if(meta==missing||!truth(t,ctx_get(t,meta,"is_stronghold")))continue;
        std::array<bool,2> sides{};for(auto unit:child_edges(s,obs_attr(s,armies[i+1],"units")))sides[su_side(s,unit)]=true;
        if(sides[0]&&sides[1])me_set_member(s,sieges,key,true);
    }
    std::vector<std::string> old;for(auto key:child_edges(s,sieges))old.emplace_back(text(s,key));
    for(const auto& key:old){
        const auto army=mu_army(s,sanitize_name(key));std::array<bool,2> sides{};
        if(army!=missing)for(auto unit:child_edges(s,obs_attr(s,army,"units")))sides[su_side(s,unit)]=true;
        if(!sides[0]||!sides[1])me_set_member(s,sieges,key,false);
    }
    const auto pc=obs_attr(s,s.root,"persistent_cards",false);
    if(!ctx_none(s,pc)&&!contains_string(s,sieges,"minas tirith")){
        uint64_t folly=missing;
        for(const char* name:{"fp_cards","sh_cards"}){for(auto card:child_edges(s,obs_attr(s,pc,name)))if(enum_is(s,obs_attr(s,card,"card_type"),"game_state.PersistentCardType","DENETHORS_FOLLY")){folly=card;break;}if(folly!=missing)break;}
        if(folly!=missing)cbp_discard_table(s,cbp_remove_card(s,pc,folly));
    }
    bool found=false;
    for(size_t i=0;i<armies.size();i+=2){
        const auto key=sanitize_name(text(s,armies[i]));std::array<size_t,2> counts{};
        for(auto unit:child_edges(s,obs_attr(s,armies[i+1],"units")))++counts[su_side(s,unit)];
        for(int side=0;side<2;++side)if(counts[side]>static_cast<size_t>(su_cap(s,t,key,side))){
            auto regions=obs_attr(s,s.root,"_overstack_regions",false);if(ctx_none(s,regions)){regions=rr_scalar(s,WOTR_SET);rr_set_attr(s,s.root,"_overstack_regions",regions);}
            me_set_member(s,regions,key,true);found=true;
        }
    }
    if((found || truth(s,obs_attr(s,s.root,"_overstack_regions",false))) && ctx_none(s,obs_attr(s,s.root,"pending_combat",false)))su_resolve_overstacks(s,t);
}
