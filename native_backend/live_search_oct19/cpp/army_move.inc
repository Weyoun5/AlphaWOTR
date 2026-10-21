// Ordinary noncombat movement, including settlement entry/capture effects.
// Ordinary/Mouth die funding and validated free card grants share this body.
// Ordinary stable sieges use acting-side partitions and controller checks.
// Combat advances and free-card followups retain their separate boundaries.
int at_controller(const Snapshot& s,const Snapshot& t,std::string_view key);
void su_cleanup(Snapshot& s,const Snapshot& t);
void cau_last_battle(Snapshot& s,const Snapshot& tables);
struct ArmyMoveRequest {
    int32_t side;int64_t die_index;
    std::string from,to;
    int64_t regular=-1,elite=-1,leaders=-1,nazgul=-1;
    bool take_leaders=true,pass=false,mouth_muster=false,card_followup=false; // -1 denotes absent optional count
};
bool am_plain(const Snapshot& s,uint64_t leader){
    const auto type=object_type(s,leader);
    if(type=="game_state.Nazgul" || type=="game_state.Companion")return false;
    require(type=="game_state.Leader" || type=="game_logic.Minion","unsupported army leader class");
    return !ctx_minion_name(s,leader);
}
int am_leader_side(const Snapshot& s,uint64_t leader){
    const auto type=object_type(s,leader);
    if(type=="game_state.Companion")return 0;
    if(type=="game_state.Nazgul" || type=="game_logic.Minion")return 1;
    require(type=="game_state.Leader","unsupported army leader class");
    const auto nation=obs_attr(s,leader,"nation",false);
    return ctx_none(s,nation)?0:(fp_nation(ctx_nation(s,nation))?0:1);
}
bool am_saruman(const Snapshot& s,uint64_t leader){
    if(object_type(s,leader)!="game_logic.Minion")return false;
    auto name=std::string(text(s,obs_attr(s,leader,"name")));
    for(auto& c:name)if(c>='A' && c<='Z')c=static_cast<char>(c-'A'+'a');
    return name.find("saruman")!=std::string::npos;
}
bool am_card_figure(const Snapshot& s,uint64_t leader,bool witch_king){
    if(object_type(s,leader)=="game_state.Nazgul")return !witch_king;
    const auto name=obs_attr(s,leader,"name",false);if(ctx_none(s,name))return false;
    std::string normalized;for(char c:text(s,name)){if(c>='A'&&c<='Z')c=char(c-'A'+'a');if(c>='a'&&c<='z')normalized.push_back(c);}
    return normalized.find("witchking")!=std::string::npos;
}
// Immutable nation/kind/character inventory for the second Army-die leg.
using am_FigureKey=std::tuple<std::string,std::string,std::string>;
am_FigureKey am_figure_key(const Snapshot& s,uint64_t figure,bool unit){
    if(unit)return {"unit",std::string(ctx_nation(s,obs_attr(s,figure,"nation"))),std::string(enum_member(s,obs_attr(s,figure,"kind"),"game_state.UnitKind"))};
    if(object_type(s,figure)=="game_state.Nazgul")return {"nazgul","SAURON",""};
    const auto name=obs_attr(s,figure,"name",false);if(truth(s,name))return {"named",std::string(text(s,name)),""};
    return {"leader",std::string(ctx_nation(s,obs_attr(s,figure,"nation"))),""};
}
std::pair<std::vector<uint64_t>,std::vector<uint64_t>> am_preexisting(const Snapshot& s,uint64_t army,uint64_t pre){
    std::map<am_FigureKey,int64_t> due;const auto inventory=ctx_get(s,pre,"figures");
    if(!ctx_none(s,inventory))for(auto row:child_edges(s,inventory)){
        const auto values=child_edges(s,row);require(values.size()==4,"Malformed preexisting figure row");
        am_FigureKey key{std::string(text(s,values[0])),std::string(text(s,values[1])),std::string(text(s,values[2]))};
        const auto n=typed(s,values[3],WOTR_INT,"Preexisting figure count must be integer").integer;
        require(n>0&&due.emplace(key,n).second,"Invalid preexisting figure count");
    }
    std::pair<std::vector<uint64_t>,std::vector<uint64_t>> out;
    for(auto u:child_edges(s,obs_attr(s,army,"units"))){auto& n=due[am_figure_key(s,u,true)];if(n>0){--n;out.first.push_back(u);}}
    for(auto l:child_edges(s,obs_attr(s,army,"leaders"))){auto& n=due[am_figure_key(s,l,false)];if(n>0){--n;out.second.push_back(l);}}
    for(const auto& [key,n]:due)require(n==0,"A preexisting army figure is missing");return out;
}
uint64_t am_inventory(Snapshot& s,const std::vector<uint64_t>& units,const std::vector<uint64_t>& leaders,int side){
    std::map<am_FigureKey,int64_t> counts;
    for(auto u:units)if(ctx_unit_side(s,u)==side)++counts[am_figure_key(s,u,true)];
    for(auto l:leaders)if(am_leader_side(s,l)==side)++counts[am_figure_key(s,l,false)];
    std::vector<uint64_t> rows;for(const auto& [key,n]:counts){const auto row=rr_scalar(s,WOTR_TUPLE);set_edges(s,row,{rr_string(s,std::get<0>(key)),rr_string(s,std::get<1>(key)),rr_string(s,std::get<2>(key)),rr_scalar(s,WOTR_INT,n)});rows.push_back(row);}
    const auto inventory=rr_scalar(s,WOTR_TUPLE);set_edges(s,inventory,rows);return inventory;
}
uint64_t am_list(Snapshot& s,const std::vector<uint64_t>& elements){
    const auto list=rr_scalar(s,WOTR_LIST);set_edges(s,list,elements);return list;
}
void am_assign_list(Snapshot& s,uint64_t object,const char* attr,const std::vector<uint64_t>& elements){
    rr_set_attr(s,object,attr,am_list(s,elements));
}
void am_extend(Snapshot& s,uint64_t object,const char* attr,const std::vector<uint64_t>& append){
    const auto list=obs_attr(s,object,attr);auto elements=child_edges(s,list);
    elements.insert(elements.end(),append.begin(),append.end());set_edges(s,list,elements);
}
std::string am_key(const Snapshot& s,std::string_view norm){
    const auto& dict=typed(s,obs_attr(s,s.root,"armies"),WOTR_DICT,"army mapping required");
    for(uint64_t i=0;i<dict.count;i+=2){
        const auto key=text(s,s.edges[dict.offset+i]);if(sanitize_name(key)==norm)return std::string(key);
    }
    return {};
}
void am_erase(Snapshot& s,std::string_view key){
    const auto dict=obs_attr(s,s.root,"armies");auto edges=child_edges(s,dict);
    for(size_t i=0;i<edges.size();i+=2)if(text_is(s,edges[i],key)){
        edges.erase(edges.begin()+i,edges.begin()+i+2);set_edges(s,dict,edges);return;
    }
}
void am_insert(Snapshot& s,std::string_view key,uint64_t army){
    const auto dict=obs_attr(s,s.root,"armies");auto edges=child_edges(s,dict);
    require(ctx_get(s,dict,key)==missing,"army insertion would replace existing key");
    edges.push_back(rr_string(s,key));edges.push_back(army);set_edges(s,dict,edges);
}
uint64_t am_new_army(Snapshot& s,uint64_t region){
    const auto units=rr_scalar(s,WOTR_LIST),leaders=rr_scalar(s,WOTR_LIST);
    return mu_object(s,"game_state.Army",{{"region",region},{"units",units},{"leaders",leaders}});
}
void am_orphans(Snapshot& s){
    const auto entries=child_edges(s,obs_attr(s,s.root,"armies"));
    for(size_t i=0;i<entries.size();i+=2){
        const auto army=entries[i+1];auto units=child_edges(s,obs_attr(s,army,"units"));
        auto leaders=child_edges(s,obs_attr(s,army,"leaders"));
        if(!units.empty() || leaders.empty())continue;
        std::vector<uint64_t> keep;
        for(auto l:leaders)if(!am_plain(s,l))keep.push_back(l);
        const auto key=std::string(text(s,entries[i]));am_assign_list(s,army,"leaders",keep);
        if(keep.empty())am_erase(s,key);
    }
}
bool am_mouth_available(const Snapshot& s,int side){
    const auto mouth=obs_attr(s,s.root,"mouth_of_sauron",false);
    return side==1 && !ctx_none(s,mouth) && truth(s,obs_attr(s,mouth,"in_play")) && !truth(s,obs_attr(s,s.root,"sh_mouth_muster_used_this_turn",false));
}
uint64_t am_die(const Snapshot& s,int side,int64_t index,bool character,bool muster){
    const auto& pool=typed(s,pa_pool(s,side),WOTR_LIST,"army dice list required");
    require(index>=0 && static_cast<uint64_t>(index)<pool.count,"army payment die index out of range");
    const auto die=s.edges[pool.offset+index];
    require(object_type(s,die)=="game_state.ActionDie","army payment requires ActionDie");
    require(ctx_side(s,obs_attr(s,die,"side"))==side,"army payment owner mismatch");
    require(!truth(s,obs_attr(s,die,"used")),"army payment die already used");
    const auto face=obs_attr(s,die,"current");
    require(enum_is(s,face,"game_state.ActionDieFace","ARMY") || enum_is(s,face,"game_state.ActionDieFace","MUSTER_ARMY") ||
            enum_is(s,face,"game_state.ActionDieFace","WILL_OF_WEST") ||
            (character && enum_is(s,face,"game_state.ActionDieFace","CHARACTER")) ||
            (muster && enum_is(s,face,"game_state.ActionDieFace","MUSTER")),"army payment face unsupported");
    return die;
}
std::unique_ptr<Snapshot> apply_army_move(const Snapshot& source,const Snapshot& tables,const ArmyMoveRequest& req){
    require(req.side==0 || req.side==1,"army moving side must be FP or SH");
    require(req.regular>=-1 && req.elite>=-1 && req.leaders>=-1 && req.nazgul>=-1,"negative army split count unsupported");
    if(req.card_followup)af_scope(source,tables);
    else{
        quiet_action_gate(source,tables,false,true,false,false,false,false,false,false,false,false,false,true);
    }
    const bool had_siege=truth(source,obs_attr(source,source.root,"besieged_strongholds",false));
    ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=ctx_side(source,obs_attr(source,source.root,"_ring_free_side",false));
    require(retained<0 || retained==req.side,"army movement belongs to retained ring actor");
    const auto pending=obs_attr(source,source.root,"pending_army_move_die");const bool second=!ctx_none(source,pending);
    require(!req.card_followup || (req.side==1 && req.die_index==-1 && !second && !req.pass && !req.mouth_muster),"Invalid free card movement funding");
    const int64_t payment=second?typed(source,pending,WOTR_INT,"pending army die must be integer").integer:req.die_index;
    if(second){
        require(ctx_side(source,obs_attr(source,source.root,"pending_army_move_side"))==req.side,"pending army owner differs");
        require(req.pass || req.die_index==payment,"second army move must name stored die");
        const auto pre=obs_attr(source,source.root,"pending_army_move_preexist");
        require(!ctx_none(source,pre),"pending army pre-existing snapshot missing"); // Package W: the frozen preexist dict has no `waived` key
    }
    require(!req.pass || second,"army pass requires pending pair");
    require(!req.mouth_muster || (!second && am_mouth_available(source,req.side)),"Mouth army ability is unavailable or already pending");
    const bool muster_mode=req.mouth_muster || second;
    const auto paid=req.card_followup?missing:am_die(source,req.side,payment,!second,muster_mode);
    require(!req.mouth_muster || enum_is(source,obs_attr(source,paid,"current"),"game_state.ActionDieFace","MUSTER"),"Mouth stamp requires a physical Muster die");
    const bool character=!req.card_followup && enum_is(source,obs_attr(source,paid,"current"),"game_state.ActionDieFace","CHARACTER");
    auto child=action_clone(source);auto& s=*child;
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));
    rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    if(req.pass){
        rr_set_attr(s,am_die(s,req.side,payment,false,muster_mode),"used",rr_scalar(s,WOTR_BOOL,1));
    }else{
        const auto from=sanitize_name(req.from),to=sanitize_name(req.to);
        const bool stationary=from==to;
        const auto sieges=obs_attr(s,s.root,"besieged_strongholds",false);
        const bool source_siege=contains_string(s,sieges,from),destination_siege=contains_string(s,sieges,to);
        require(stationary || !source_siege || at_controller(s,tables,from)!=req.side,"Garrison cannot leave siege by ordinary movement");
        require(stationary || !destination_siege || at_controller(s,tables,to)!=req.side,"Garrison relief requires an attack");
        const auto from_key=am_key(s,from);require(!from_key.empty(),"army source absent");
        const auto src=mu_army(s,from);auto src_units=child_edges(s,obs_attr(s,src,"units"));
        const auto src_leaders=child_edges(s,obs_attr(s,src,"leaders"));
        bool shared_source=false;size_t own_source_count=0;
        for(auto unit:src_units){
            const bool own=ctx_unit_side(s,unit)==req.side;own_source_count+=own;shared_source|=!own;
            const auto kind=obs_attr(s,unit,"kind");require(enum_is(s,kind,"game_state.UnitKind","REGULAR") || enum_is(s,kind,"game_state.UnitKind","ELITE"),"unsupported moving unit kind");
        }
        require(own_source_count>0,"ordinary army move needs army units");
        const auto board=obs_attr(s,s.root,"board");attributes(s,board,"board.Board");
        const auto region_map=obs_attr(s,board,"_regions"),adj=obs_attr(s,board,"_adj");
        const auto neighbors=ctx_get(s,adj,from);require(stationary || contains_string(s,neighbors,to),"army destination is not adjacent");
        const auto to_region=ctx_get(s,region_map,to);require(to_region!=missing,"army destination Board.region missing");
        attributes(s,to_region,"board.Region");
        require(text_is(s,obs_attr(s,to_region,"key"),to),"noncanonical destination Region.key unsupported");
        auto to_key=am_key(s,to);if(to_key.empty())to_key=std::string(text(s,obs_attr(s,to_region,"orig_name")));
        require(sanitize_name(to_key)==to,"destination original name does not match board key");
        auto dst=mu_army(s,to);std::vector<uint64_t> dst_units,dst_leaders;
        if(dst!=missing){dst_units=child_edges(s,obs_attr(s,dst,"units"));dst_leaders=child_edges(s,obs_attr(s,dst,"leaders"));}
        bool destination_enemy=false;size_t own_destination_count=0;
        for(auto unit:dst_units){
            const bool own=ctx_unit_side(s,unit)==req.side;own_destination_count+=own;destination_enemy|=!own;
            require(own || destination_siege,"noncombat move cannot enter enemy army");
            const auto kind=obs_attr(s,unit,"kind");require(enum_is(s,kind,"game_state.UnitKind","REGULAR") || enum_is(s,kind,"game_state.UnitKind","ELITE"),"unsupported destination unit kind");
        }
        require(!destination_enemy || own_destination_count>0,"noncombat move cannot enter enemy army");
        const auto regions=field(tables,tables.root,"board_regions"),meta=ctx_get(tables,regions,to);
        require(meta!=missing,"unknown movement metadata region");
        const auto dest_nat=ctx_get(tables,field(tables,tables.root,"region_nations"),to);
        const auto political=obs_attr(s,s.root,"political"),boxes=obs_attr(s,political,"boxes");
        auto atwar=[&](uint64_t nation){const auto n=ctx_nation_value(s,boxes,ctx_nation(s,nation));return ctx_num(s,n)>=3;};
        bool any_passive=false;
        for(auto u:src_units)if(ctx_unit_side(s,u)==req.side)any_passive|=!atwar(obs_attr(s,u,"nation"));
        for(auto l:src_leaders)if(am_plain(s,l) && am_leader_side(s,l)==req.side)any_passive|=!atwar(obs_attr(s,l,"nation"));
        const bool border=!stationary && any_passive && !ctx_none(tables,dest_nat);
        auto nation_ok=[&](uint64_t nation){return !border || atwar(nation) || ctx_nation(s,nation)==ctx_nation(tables,dest_nat);};
        const bool restricted=second&&sanitize_name(text(s,obs_attr(s,s.root,"pending_army_move_from")))==from;
        std::set<uint64_t> eligible_units,eligible_leaders;
        if(restricted){const auto eligible=am_preexisting(s,src,obs_attr(s,s.root,"pending_army_move_preexist"));eligible_units.insert(eligible.first.begin(),eligible.first.end());eligible_leaders.insert(eligible.second.begin(),eligible.second.end());}
        auto movable=[&](uint64_t l){return (!restricted||eligible_leaders.count(l)) && !am_saruman(s,l) && am_leader_side(s,l)==req.side && (!am_plain(s,l) || nation_ok(obs_attr(s,l,"nation")));};
        bool has_leader=false;for(auto l:src_leaders)has_leader|=am_leader_side(s,l)==req.side;
        if(character && !has_leader){
            // Saruman's in-play Isengard elite ability is supported via the
            // already ported original predicate; no Mouth funding accepted.
            bool saruman=false;const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));
            for(size_t i=1;i<armies.size();i+=2)for(auto l:child_edges(s,obs_attr(s,armies[i],"leaders")))saruman|=am_saruman(s,l);
            if(saruman)
                for(auto u:src_units)has_leader|=enum_is(s,obs_attr(s,u,"nation"),"game_state.Nation","ISENGARD") && enum_is(s,obs_attr(s,u,"kind"),"game_state.UnitKind","ELITE");
        }
        require(!character || has_leader,"Character move requires leader figure");
        int64_t regular=0,elite=0;
        for(auto u:src_units)(enum_is(s,obs_attr(s,u,"kind"),"game_state.UnitKind","REGULAR")?regular:elite)++;
        const bool explicit_counts=req.regular>=0 || req.elite>=0 || req.leaders>=0 || req.nazgul>=0;
        int64_t mr=explicit_counts?std::min(std::max<int64_t>(0,req.regular),regular):regular;
        int64_t me=explicit_counts?std::min(std::max<int64_t>(0,req.elite),elite):elite;
        bool split=explicit_counts && !(mr>=regular && me>=elite && req.leaders<0 && req.nazgul<0 && req.take_leaders);
        if((border || shared_source || restricted) && !split){mr=me=0;for(auto u:src_units)if(ctx_unit_side(s,u)==req.side && (!restricted||eligible_units.count(u)) && nation_ok(obs_attr(s,u,"nation")))(enum_is(s,obs_attr(s,u,"kind"),"game_state.UnitKind","REGULAR")?mr:me)++;split=true;}
        int64_t ml=req.leaders>=0?req.leaders:(req.take_leaders?static_cast<int64_t>(src_leaders.size()):0);
        if(!explicit_counts || (!split))ml=src_leaders.size();
        std::vector<uint64_t> moving_units,kept_units,moving_leaders,kept_leaders;
        if(!split){
            moving_units=src_units;
            for(auto l:src_leaders)(am_saruman(s,l) || am_leader_side(s,l)!=req.side?kept_leaders:moving_leaders).push_back(l);
        }else{
            std::vector<uint64_t> kept_elite,kept_regular,held,enemy,moved_elite,moved_regular;
            for(auto u:src_units){
                if(ctx_unit_side(s,u)!=req.side || (restricted&&!eligible_units.count(u))){enemy.push_back(u);continue;}
                if(!nation_ok(obs_attr(s,u,"nation"))){held.push_back(u);continue;}
                const bool reg=enum_is(s,obs_attr(s,u,"kind"),"game_state.UnitKind","REGULAR");
                auto& moved=reg?moved_regular:moved_elite;auto& kept=reg?kept_regular:kept_elite;
                (static_cast<int64_t>(moved.size())<(reg?mr:me)?moved:kept).push_back(u);
            }
            require(static_cast<int64_t>(moved_regular.size())==mr && static_cast<int64_t>(moved_elite.size())==me,"requested units cannot cross nation border");
            moving_units=moved_elite;moving_units.insert(moving_units.end(),moved_regular.begin(),moved_regular.end());
            kept_units=kept_elite;kept_units.insert(kept_units.end(),kept_regular.begin(),kept_regular.end());kept_units.insert(kept_units.end(),held.begin(),held.end());kept_units.insert(kept_units.end(),enemy.begin(),enemy.end());
            std::set<uint64_t> chosen;int64_t remaining=ml;
            for(bool plain:{true,false})for(auto l:src_leaders)
                if(remaining>0 && am_plain(s,l)==plain && movable(l) && (req.nazgul<0 || object_type(s,l)!="game_state.Nazgul")){chosen.insert(l);--remaining;}
            if(req.nazgul>=0){
                remaining=req.nazgul;for(auto l:src_leaders)if(remaining>0 && movable(l) && object_type(s,l)=="game_state.Nazgul"){chosen.insert(l);moving_leaders.push_back(l);--remaining;}
            }
            for(auto l:src_leaders)if(chosen.count(l)){if(req.nazgul<0 || object_type(s,l)!="game_state.Nazgul")moving_leaders.push_back(l);}else kept_leaders.push_back(l);
        }
        if(req.card_followup){
            const auto followup=obs_attr(s,s.root,"pending_card_followup");
            const bool witch_king=text_is(s,field(s,followup,"kind"),"wk_army");
            require(std::any_of(moving_leaders.begin(),moving_leaders.end(),[&](uint64_t leader){return am_card_figure(s,leader,witch_king);}),"Card army move requires participating Nazgul or Witch-king");
        }
        if(character){
            bool leadership=!moving_leaders.empty(),saruman=false;const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));
            for(size_t i=1;i<armies.size();i+=2)for(auto l:child_edges(s,obs_attr(s,armies[i],"leaders")))saruman|=am_saruman(s,l);
            if(saruman)for(auto u:moving_units)leadership|=enum_is(s,obs_attr(s,u,"nation"),"game_state.Nation","ISENGARD")&&enum_is(s,obs_attr(s,u,"kind"),"game_state.UnitKind","ELITE");
            require(leadership,"Character move requires participating leadership");
        }
        require(!moving_units.empty(),"ordinary movement must move army units");
        require(stationary || own_destination_count+moving_units.size()<=10,"army destination stacking cap exceeded");
        require(!stationary || (req.side==0 && me_companion_activation_available(s,tables,to,moving_leaders)),
                "Stationary army movement must activate a nation");
        bool needs_guard=false,has_guard=false;
        for(auto l:kept_leaders)needs_guard|=am_plain(s,l) && am_leader_side(s,l)==req.side;
        for(auto u:kept_units)has_guard|=ctx_unit_side(s,u)==req.side;
        require(stationary || !needs_guard || has_guard,"mandatory plain leader needs a unit left behind");
        if(second && sanitize_name(text(s,obs_attr(s,s.root,"pending_army_move_from")))==from){
            const auto pre=obs_attr(s,s.root,"pending_army_move_preexist");int64_t r=0,e=0;
            for(auto u:moving_units)(enum_is(s,obs_attr(s,u,"kind"),"game_state.UnitKind","REGULAR")?r:e)++;
            require(r<=ctx_num(s,ctx_get(s,pre,"regular")) && e<=ctx_num(s,ctx_get(s,pre,"elite")) && moving_leaders.size()<=ctx_num(s,ctx_get(s,pre,"leaders")),"second move exceeds pre-existing destination figures");
        }
        if(!second && !character && !req.card_followup){
            if(stationary){dst_units=kept_units;dst_leaders=kept_leaders;}
            int64_t r=0,e=0;for(auto u:dst_units)if(ctx_unit_side(s,u)==req.side)(enum_is(s,obs_attr(s,u,"kind"),"game_state.UnitKind","REGULAR")?r:e)++;
            const auto pre=rr_scalar(s,WOTR_DICT);std::vector<uint64_t> entries;
            for(auto pair:{std::pair<const char*,int64_t>{"regular",r},{"elite",e},{"leaders",static_cast<int64_t>(dst_leaders.size())}}){entries.push_back(rr_string(s,pair.first));entries.push_back(rr_scalar(s,WOTR_INT,pair.second));}
            entries.push_back(rr_string(s,"figures"));entries.push_back(am_inventory(s,dst_units,dst_leaders,req.side));
            entries.push_back(rr_string(s,"waived"));entries.push_back(rr_scalar(s,WOTR_BOOL,0));set_edges(s,pre,entries);
            rr_set_attr(s,s.root,"pending_army_move_preexist",pre);
        }
        // Public source spends Character/second-leg funding before the
        // primitive movement, including its political/capture callbacks.
        if(character || second)rr_set_attr(s,am_die(s,req.side,payment,!second,muster_mode),"used",rr_scalar(s,WOTR_BOOL,1));
        if(stationary){
            // A paid movement ending in place activates without moving figures.
        }else if(!split){
            const auto old_region=obs_attr(s,src,"region");if(!kept_leaders.empty())am_assign_list(s,src,"leaders",moving_leaders);
            am_erase(s,from_key);if(dst==missing){dst=src;am_insert(s,to_key,src);}rr_set_attr(s,src,"region",to_region);
            if(dst!=src){am_extend(s,dst,"units",moving_units);am_extend(s,dst,"leaders",moving_leaders);}
            if(!kept_leaders.empty()){const auto back=am_new_army(s,old_region);am_insert(s,from_key,back);am_extend(s,back,"leaders",kept_leaders);}
        }else{
            am_assign_list(s,src,"units",kept_units);am_assign_list(s,src,"leaders",kept_leaders);
            if(kept_units.empty() && kept_leaders.empty())am_erase(s,from_key);
            if(dst==missing){dst=am_new_army(s,to_region);am_insert(s,to_key,dst);}
            am_extend(s,dst,"units",moving_units);am_extend(s,dst,"leaders",moving_leaders);
        }
        movement_entry_effects(s,tables,to,req.side,moving_leaders);
        if(!stationary)am_orphans(s);
        if(!stationary && !destination_enemy)movement_capture_effects(s,tables,to,req.side);
        if(!character && !second && !req.card_followup){
            rr_set_attr(s,s.root,"pending_army_move_die",rr_scalar(s,WOTR_INT,payment));
            rr_set_attr(s,s.root,"pending_army_move_from",rr_string(s,req.to));
            rr_set_attr(s,s.root,"pending_army_move_side",mu_enum(s,"game_state.Side",req.side==0?"FREE_PEOPLES":"SHADOW"));
        }
    }
    if(req.mouth_muster)rr_set_attr(s,s.root,"sh_mouth_muster_used_this_turn",rr_scalar(s,WOTR_BOOL,1));
    if(second){for(auto key:{"pending_army_move_die","pending_army_move_from","pending_army_move_side","pending_army_move_preexist"})rr_set_attr(s,s.root,key,rr_scalar(s,WOTR_NONE));}
    if(second || character)rr_set_attr(s,s.root,"_battle_resolved_attacker",rr_scalar(s,WOTR_NONE));
    // The source primitive leaves siege reconciliation to public cleanup.
    // Card-granted moves retain source cleanup timing in their public wrapper.
    if(had_siege && !req.card_followup)su_cleanup(s,tables);
    if(!req.card_followup && ctx_none(s,obs_attr(s,s.root,"pending_combat",false)))cau_last_battle(s,tables);
    if(req.card_followup){
        // A departed besieger leaves stale siege flags until cau_finish.
        // Validate the reconciled view without changing the returned primitive.
        if(had_siege){Snapshot validation(s);su_cleanup(validation,tables);af_scope(validation,tables);}
        else af_scope(s,tables);
    }
    else quiet_action_gate(s,tables,false,true,false,false,false,false,false,false,false,false,false,true);
    return child;
}
