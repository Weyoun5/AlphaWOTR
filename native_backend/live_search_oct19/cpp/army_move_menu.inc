// Raw _get_army_move_actions/get_legal_actions descriptors. No AI projection.
bool at_shadow_blocked(const Snapshot& s,std::string_view key);
void amg_append(am_Builder& b,std::vector<uint64_t>& rows,const Snapshot& s,const Snapshot& t,
                int side,int64_t die,bool require_leader,std::string_view exclude,int followup_figure=0){
    // These views borrow only the immutable input graph. Every caller supplies
    // a distinct output builder, so growing b.out cannot invalidate them.
    // Match child_edges' range exactly; retain its existing admission boundary.
    using EdgeIterator=std::vector<uint64_t>::const_iterator;
    struct Edges {
        EdgeIterator first,last;
        EdgeIterator begin()const{return first;}
        EdgeIterator end()const{return last;}
        size_t size()const{return static_cast<size_t>(last-first);}
        uint64_t operator[](size_t i)const{return first[static_cast<ptrdiff_t>(i)];}
    };
    auto edges=[&](uint64_t node){const auto& n=s.nodes[node];return Edges{
        s.edges.begin()+static_cast<ptrdiff_t>(n.offset),
        s.edges.begin()+static_cast<ptrdiff_t>(n.offset+n.count)};};
    auto nation_view=[](const Snapshot& source,uint64_t node){
        return validate_nation_name(enum_member(source,node,"game_state.Nation"));
    };
    // Refill scratch arrays in the original traversal/validation order. Their
    // capacity survives each army/destination; no data survives this call.
    std::vector<uint64_t> p,own,pool;p.reserve(16);
    std::vector<std::string_view> neighbors;
    const auto armies=edges(obs_attr(s,s.root,"armies"));
    const auto adj=obs_attr(s,obs_attr(s,s.root,"board"),"_adj");
    const auto pre=obs_attr(s,s.root,"pending_army_move_preexist",false);
    const auto sieges=obs_attr(s,s.root,"besieged_strongholds",false);
    bool saruman=false;
    for(size_t i=1;i<armies.size();i+=2)for(auto l:edges(obs_attr(s,armies[i],"leaders")))saruman|=am_saruman(s,l);
    for(size_t i=0;i<armies.size();i+=2){
        const auto akey=text(s,armies[i]);const auto army=armies[i+1];
        auto units=child_edges(s,obs_attr(s,army,"units")),leaders=child_edges(s,obs_attr(s,army,"leaders"));
        own.clear();for(auto u:units)if(ctx_unit_side(s,u)==side)own.push_back(u);
        if(own.empty())continue;
        bool figure=false;for(auto l:leaders)figure|=am_leader_side(s,l)==side;
        if(saruman)for(auto u:own)figure|=enum_is(s,obs_attr(s,u,"nation"),"game_state.Nation","ISENGARD") && enum_is(s,obs_attr(s,u,"kind"),"game_state.UnitKind","ELITE");
        if(require_leader && !figure)continue;
        const auto from=sanitize_name(akey);const bool capped=!exclude.empty() && from==exclude;
        const bool source_garrison=contains_string(s,sieges,from) && at_controller(s,t,from)==side;
        auto cap=[&](const char* name){const auto n=ctx_get(s,pre,name);return ctx_none(s,n)?int64_t(0):typed(s,n,WOTR_INT,"army menu preexisting count must be integer").integer;};
        if(capped && (!truth(s,pre) || cap("regular")+cap("elite")<=0))continue;
        const auto physical_own_count=own.size();
        if(capped){const auto allowed=am_preexisting(s,army,pre);units=allowed.first;leaders=allowed.second;own.clear();for(auto u:units)if(ctx_unit_side(s,u)==side)own.push_back(u);if(own.empty())continue;}
        int64_t plain=0,must=0,nazgul=0;
        for(auto l:leaders){
            plain+=object_type(s,l)!="game_state.Nazgul" && !am_saruman(s,l) && am_leader_side(s,l)==side;
            must+=am_plain(s,l) && am_leader_side(s,l)==side;
            nazgul+=side==1 && object_type(s,l)=="game_state.Nazgul";
        }
        const int64_t plain_all=plain;   // D2 mirror (static_r4): the UNCAPPED non-Nazgul count = the figures a take_leaders row really moves
        if(capped){plain=std::min(plain,cap("leaders"));must=std::min(must,plain);}
        const auto political=obs_attr(s,s.root,"political"),boxes=obs_attr(s,political,"boxes");
        auto atwar=[&](uint64_t n){return ctx_num(s,ctx_nation_value(s,boxes,nation_view(s,n)))>=3;};
        bool passive=false;for(auto u:own)passive|=!atwar(obs_attr(s,u,"nation"));
        for(auto l:leaders)if(am_plain(s,l) && am_leader_side(s,l)==side)passive|=!atwar(obs_attr(s,l,"nation"));
        neighbors.clear();if(!source_garrison)for(auto n:edges(ctx_get(s,adj,from)))neighbors.emplace_back(text(s,n));
        if(side==0 && me_companion_activation_available(s,t,from,leaders))neighbors.emplace_back(from);
        std::sort(neighbors.begin(),neighbors.end());
        for(const auto& to:neighbors){
            const bool stationary=to==from;
            if(side==1 && at_shadow_blocked(s,to))continue;
            if(!stationary && contains_string(s,sieges,to) && at_controller(s,t,to)==side)continue;
            const auto nation=ctx_get(t,field(t,t.root,"region_nations"),to);
            const bool border=!stationary && passive && !ctx_none(t,nation);
            auto eligible=[&](uint64_t n){return !border || atwar(n) || nation_view(s,n)==nation_view(t,nation);};
            pool.clear();for(auto u:own)if(eligible(obs_attr(s,u,"nation")))pool.push_back(u);
            if(pool.empty())continue;
            int64_t pl=plain,mg=must,pl_all=plain_all;
            if(border){
                pl=mg=0;
                for(auto l:leaders)if(am_leader_side(s,l)==side){
                    const bool crosses=!am_plain(s,l) || eligible(obs_attr(s,l,"nation"));
                    pl+=object_type(s,l)!="game_state.Nazgul" && !am_saruman(s,l) && crosses;
                    mg+=am_plain(s,l) && crosses;
                }
                pl_all=pl;
                if(capped){pl=std::min(pl,cap("leaders"));mg=std::min(mg,pl);}
                if(require_leader && pl==0 && plain>0)continue;
            }
            bool held_plain=false;for(auto l:leaders)held_plain|=am_plain(s,l) && am_leader_side(s,l)==side && !eligible(obs_attr(s,l,"nation"));
            const bool stay=mg==0 && pl>0;
            const auto dst=mu_army(s,to);int64_t existing=0;bool enemy=false;
            if(dst!=missing)for(auto u:edges(obs_attr(s,dst,"units"))){if(ctx_unit_side(s,u)==side)++existing;else enemy=true;}
            if(enemy && !existing)continue;
            if(stationary)existing=0;
            int64_t nr=0,ne=0;bool isengard_elite=false;
            for(auto u:pool){
                const auto kind=obs_attr(s,u,"kind");nr+=enum_is(s,kind,"game_state.UnitKind","REGULAR");ne+=enum_is(s,kind,"game_state.UnitKind","ELITE");
                isengard_elite|=side==1 && saruman && enum_is(s,kind,"game_state.UnitKind","ELITE") && enum_is(s,obs_attr(s,u,"nation"),"game_state.Nation","ISENGARD");
            }
            if(capped){nr=std::min(nr,cap("regular"));ne=std::min(ne,cap("elite"));}
            const bool take_ok=!capped || cap("leaders")>0;
            const int64_t capL=capped?cap("leaders"):int64_t(-1);   // D2 mirror (static_r4): every leader figure a row moves (Nazgul included) must fit the pre-existing count - the executor rule of army_move.inc (rulebook p.25 "no figure is moved twice using the same action")
            {   // speed pass (growth policy): an upper bound of this block's rows - (nr+1)(ne+1) x take x (plain + nazgul-count + leader-count variants) - reserved before emitting; capacity only
                const int64_t variants=1+(nazgul>0?nazgul+1:0)+(pl>1?(pl-1)*(nazgul>0?nazgul+1:1):0);const size_t bound=static_cast<size_t>((nr+1)*(ne+1)*2*variants);
                b.grow(bound*8+16,bound*19+8,64);am_grow(rows,bound);
            }
            for(int64_t mr=0;mr<=nr;++mr)for(int64_t me=0;me<=ne;++me){
                if(mr+me==0 || existing+mr+me>10 || (held_plain && mr+me==static_cast<int64_t>(physical_own_count)))continue;
                for(bool take:{true,false}){
                    if(take ? !take_ok : (require_leader || !(stationary || mr+me<static_cast<int64_t>(physical_own_count) || stay)))continue;
                    auto parameters=[&](int64_t nl,int64_t nn,bool late_naz){
                        if(stationary){
                            int64_t remaining=nl>=0?nl:(take?static_cast<int64_t>(leaders.size()):0);
                            std::vector<uint64_t> selected;
                            for(bool is_plain:{true,false})for(auto leader:leaders)
                                if(remaining>0 && am_plain(s,leader)==is_plain && !am_saruman(s,leader) && am_leader_side(s,leader)==side){
                                    selected.push_back(leader);--remaining;
                                }
                            if(!me_companion_activation_available(s,t,from,selected))return;
                        }
                        if(followup_figure){
                            // Select the actual moving figures in kernel order.
                            // A qualifying figure elsewhere in the source does not suffice.
                            int64_t remaining=nl>=0?nl:(take?static_cast<int64_t>(leaders.size()):0);
                            bool qualifies=false;
                            for(bool is_plain:{true,false})for(auto leader:leaders){
                                if(remaining<=0||am_plain(s,leader)!=is_plain||am_saruman(s,leader)||am_leader_side(s,leader)!=side)continue;
                                if(am_plain(s,leader)&&!eligible(obs_attr(s,leader,"nation")))continue;
                                if(nn>=0&&object_type(s,leader)=="game_state.Nazgul")continue;
                                --remaining;qualifies|=am_card_figure(s,leader,followup_figure==2);
                            }
                            if(followup_figure==1&&nn>0&&nazgul>0)qualifies=true;
                            if(!qualifies)return;
                        }
                        if(require_leader){
                            bool leadership=(take&&(nl<0?pl>0:nl>0))||(nn>=0?nn>0:(take&&nl<0&&nazgul>0));
                            int64_t picked=0;for(auto u:pool)if(enum_is(s,obs_attr(s,u,"kind"),"game_state.UnitKind","ELITE")&&picked++<me)leadership|=side==1&&saruman&&enum_is(s,obs_attr(s,u,"nation"),"game_state.Nation","ISENGARD");
                            if(!leadership)return;
                        }
                        p.clear();p.push_back(b.str("from"));p.push_back(b.str(akey));p.push_back(b.str("to"));p.push_back(b.str(to));p.push_back(b.str("n_regular"));p.push_back(b.integer(mr));p.push_back(b.str("n_elite"));p.push_back(b.integer(me));p.push_back(b.str("take_leaders"));p.push_back(rr_scalar(b.out,WOTR_BOOL,take));
                        if(nl>=0){p.push_back(b.str("n_leaders"));p.push_back(b.integer(nl));}
                        if(nn>=0 && !late_naz){p.push_back(b.str("n_nazgul"));p.push_back(b.integer(nn));}
                        p.push_back(b.str("side"));p.push_back(b.str(side==0?"fp":"sh"));
                        if(nn>=0 && late_naz){p.push_back(b.str("n_nazgul"));p.push_back(b.integer(nn));}
                        rows.push_back(b.row(die,"army_move",p));
                    };
                    if(capL<0 || (take?pl_all+nazgul:int64_t(0))<=capL)parameters(-1,-1,false);
                    if(nazgul>0)for(int64_t nn=0;nn<=nazgul;++nn){
                        if(nn==(take?nazgul:0))continue;
                        if(require_leader && nn==0 && pl==0 && !(me>0 && isengard_elite))continue;
                        if(capL>=0 && (take?pl_all:int64_t(0))+nn>capL)continue;
                        parameters(-1,nn,false);
                    }
                    if(take && pl>1)for(int64_t nl=1;nl<pl;++nl){
                        if(!(stationary || mr+me<static_cast<int64_t>(physical_own_count) || nl>=mg))continue;
                        if(capL>=0 && nl>capL)continue;
                        if(nazgul>0)for(int64_t nn=0;nn<=nazgul;++nn){if(capL>=0 && nl+nn>capL)continue;parameters(nl,nn,true);}
                        else parameters(nl,-1,false);
                    }
                }
            }
        }
    }
}
std::unique_ptr<Snapshot> native_army_move_actions(const Snapshot& s,const Snapshot& t,int32_t side){
    require(side==0 || side==1,"army menu side must be FP or SH");quiet_action_gate(s,t,false,true,false,false,false,false,false,false,false,false,false,true);
    am_Builder b;std::vector<uint64_t> rows;
    const auto pending=obs_attr(s,s.root,"pending_army_move_die"),owner=obs_attr(s,s.root,"pending_army_move_side",false);
    if(!ctx_none(s,pending) && (ctx_none(s,owner) || ctx_side(s,owner)==side)){
        const auto die=typed(s,pending,WOTR_INT,"pending army menu index must be integer").integer;
        const auto prev=obs_attr(s,s.root,"pending_army_move_from",false);
        amg_append(b,rows,s,t,side,die,false,truth(s,prev)?sanitize_name(text(s,prev)):std::string{});
        rows.push_back(b.row(die,"pass",{}));
    }else{
        ring_waiver(s,side);
        const bool mouth_mode=am_mouth_available(s,side);
        const auto dice=child_edges(s,pa_pool(s,side));std::array<bool,7> seen{};
        for(size_t i=0;i<dice.size();++i){
            const auto die=dice[i];require(object_type(s,die)=="game_state.ActionDie","army menu requires ActionDie");
            if(truth(s,obs_attr(s,die,"used")))continue;
            const auto current=obs_attr(s,die,"current");if(ctx_none(s,current))continue;
            const int face=rr_face_index(enum_member(s,current,"game_state.ActionDieFace"));
            if(seen[face])continue;seen[face]=true;
            if(face==1 || face==4 || face==6)amg_append(b,rows,s,t,side,i,false,{});
            else if(face==0)amg_append(b,rows,s,t,side,i,true,{});
            else if(face==2 && mouth_mode){
                const size_t begin=rows.size();amg_append(b,rows,s,t,side,i,false,{});
                for(size_t j=begin;j<rows.size();++j){const auto p=child_edges(b.out,rows[j])[2];auto edges=child_edges(b.out,p);edges.push_back(b.str("mouth_muster"));edges.push_back(rr_scalar(b.out,WOTR_BOOL,1));set_edges(b.out,p,edges);}
            }
        }
    }
    b.out.root=b.container(WOTR_LIST,rows);return std::make_unique<Snapshot>(std::move(b.out));
}
