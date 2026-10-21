// Fixed-region Shadow recruitment; no hidden defaults replace learned regions.
int esr_kind(const Snapshot& t,uint64_t card){
    if(card==missing)return 0;const auto name=obs_attr(t,card,"card_name");
    if(text_is(t,name,"Orcs Multiplying Again"))return 1;
    if(text_is(t,name,"Monsters Roused"))return 2;
    if(text_is(t,name,"Musterings of Long-planned War"))return 3;
    if(text_is(t,name,"A New Power is Rising"))return 4;
    if(text_is(t,name,"The King is Revealed"))return 5;
    if(text_is(t,name,"Return of the Witch-king"))return 6;return 0;
}
bool esr_saruman_present(const Snapshot& s){
    const auto marker=obs_attr(s,s.root,"saruman",false);
    if(!ctx_none(s,marker) && truth(s,obs_attr(s,marker,"in_play",false)))return true;
    const auto armies=child_edges(s,obs_attr(s,s.root,"armies"));
    for(size_t i=1;i<armies.size();i+=2)for(auto figure:child_edges(s,obs_attr(s,armies[i],"leaders")))
        if(object_type(s,figure)=="game_logic.Minion" && ev_lower(text(s,obs_attr(s,figure,"name"))).find("saruman")!=std::string::npos)return true;
    return false;
}
void esr_effect(Snapshot& s,const Snapshot& t,int kind){
    auto regulars=[&](std::string_view nation,std::string_view region,int count){for(int i=0;i<count;++i)er_place(s,t,nation,region,"regular");};
    if(kind==1){regulars("SAURON","dol guldur",3);regulars("SAURON","mount gundabad",3);}
    else if(kind==2){for(const char* region:{"angmar","ettenmoors","weather hills"})regulars("SAURON",region,1);er_place(s,t,"SAURON","trollshaws","elite");}
    else if(kind==3){regulars("SOUTHRON_EASTERLINGS","gorgoroth",5);regulars("SAURON","nurn",5);}
    else if(kind==4){
        regulars("ISENGARD","north dunland",2);regulars("ISENGARD","south dunland",2);
        const auto available=mu_available(s,t,"ISENGARD","elite");
        if(available>=2){er_place(s,t,"ISENGARD","orthanc","elite");er_place(s,t,"ISENGARD","orthanc","elite");}
        else if(available==1){er_place(s,t,"ISENGARD","orthanc","regular");er_place(s,t,"ISENGARD","orthanc","elite");}
        else regulars("ISENGARD","orthanc",2);
    }else if(kind==5){
        regulars("SAURON","minas morgul",5);
        // Card Nazgul recruitment may stand alone. Ordinary leader placement
        // intentionally requires units, so use the shared figure primitive.
        if(mu_available(s,t,"SAURON","leader")>0&&er_region_open(s,"minas morgul",1))mu_place(s,"SAURON","minas morgul","leader");
    }else if(kind==6){
        // Source matches physical name, irrespective of figure class or tracker.
        const auto armies=obs_attr(s,s.root,"armies");const auto entries=child_edges(s,armies);uint64_t moved=missing;
        for(size_t i=0;i<entries.size()&&moved==missing;i+=2){
            const auto army=entries[i+1],list=obs_attr(s,army,"leaders");auto leaders=child_edges(s,list);
            for(size_t j=0;j<leaders.size();++j){
                const auto name=obs_attr(s,leaders[j],"name",false);if(ctx_none(s,name))continue;
                std::string letters;for(auto ch:text(s,name)){if(ch>='A'&&ch<='Z')ch=ch-'A'+'a';if(ch>='a'&&ch<='z')letters+=ch;}
                if(letters.find("witchking")==std::string::npos)continue;
                moved=leaders[j];leaders.erase(leaders.begin()+j);set_edges(s,list,leaders);
                if(leaders.empty()&&!truth(s,obs_attr(s,army,"units")))am_erase(s,text(s,entries[i]));break;
            }
        }
        require(moved!=missing,"Return of the Witch-king physical requirement changed");
        auto destination=ctx_get(s,armies,"angmar");
        if(destination==missing){
            destination=am_new_army(s,ctx_get(s,obs_attr(s,obs_attr(s,s.root,"board"),"_regions"),"angmar"));
            am_insert(s,"angmar",destination);
        }
        am_extend(s,destination,"leaders",{moved});
        // The effect deliberately creates the literal Angmar key before
        // recruit_unit's exact-key-preferred lookup. Shared placement keeps
        // the new target until the public wrapper deduplicates afterward.
        for(int i=0;i<2;++i)er_place(s,t,"SAURON","angmar","regular",destination);
        er_place(s,t,"SAURON","angmar","elite",destination);
    }else throw std::invalid_argument("Unknown fixed Shadow recruitment effect");
}
void esr_commit(Snapshot& s,const Snapshot& t,std::string_view id,std::string_view deck,std::string_view face){
    const auto hand=tc_hand(s,1);auto cards=child_edges(s,hand);const auto it=std::find_if(cards.begin(),cards.end(),[&](uint64_t c){return text_is(s,c,id);});
    require(it!=cards.end(),"Shadow recruitment event absent at commit");const auto played=*it;cards.erase(it);set_edges(s,hand,cards);
    const std::string key="sh_"+std::string(deck);const auto pile=field(s,obs_attr(s,s.root,"discard_face_up"),key.c_str());auto discarded=child_edges(s,pile);discarded.push_back(played);set_edges(s,pile,discarded);
    su_resolve_overstacks(s,t);epa_after_event(s,1,id,deck,face);tc_open_hand_limit_discard(s);if(ctx_none(s,obs_attr(s,s.root,"pending_combat")))su_cleanup(s,t);cau_gandalf(s);
    rr_set_attr(s,s.root,"_battle_resolved_attacker",rr_scalar(s,WOTR_NONE));
}
std::unique_ptr<Snapshot> apply_shadow_recruit_event(const Snapshot& source,const Snapshot& t,int side,int64_t index,std::string_view id){
    if(esra_has_choices(source,t,id))return esra_begin(source,t,side,index,id);
    require(side==1,"Fixed Shadow recruitment belongs to Shadow");quiet_action_gate(source,t,false,false,false,false,false,false,false,false,false,false,false,esr_kind(t,ev_card(t,id))>0);ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained)||ctx_side(source,retained)==side,"Shadow recruitment belongs to retained Ring actor");
    const auto card=ev_card(t,id);const int kind=esr_kind(t,card);require(kind && ctx_side(t,obs_attr(t,card,"side"))==1,"Unsupported fixed Shadow recruitment event");
    require(contains_string(source,tc_hand(source,side),id),"Shadow recruitment event absent from hand");
    require(kind==4?esr_saruman_present(source):ev_requirement(source,t,card),"Shadow recruitment printed requirement false");
    const auto paid=ev_payment(source,t,side,index,card);const std::string face(enum_member(source,obs_attr(source,paid,"current"),"game_state.ActionDieFace"));const auto deck=ev_lower(text(t,obs_attr(t,card,"deck")));
    auto child=action_clone(source);auto& s=*child;rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    rr_set_attr(s,ev_payment(s,t,side,index,card),"used",rr_scalar(s,WOTR_BOOL,1));esr_effect(s,t,kind);esr_commit(s,t,id,deck,face);if(kind==6){cau_dedupe(s);su_cleanup(s,t);}return child;
}
