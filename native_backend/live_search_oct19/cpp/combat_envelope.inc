void crs_witch_king_draw(Snapshot&,uint64_t,am_Builder* = nullptr,std::vector<uint64_t>* = nullptr);
// Battle-envelope publication. Inner RoundCursor choices remain native; these
// helpers preserve the outer result/pending graph and observation provenance.
uint64_t cbe_descriptor_copy(Snapshot& s,uint64_t descriptor){
    auto entries=child_edges(s,descriptor);for(size_t i=0;i<entries.size();i+=2)if(text_is(s,entries[i],"options")){std::vector<uint64_t> options;for(auto option:child_edges(s,entries[i+1]))options.push_back(cp_container(s,WOTR_DICT,child_edges(s,option)));entries[i+1]=cp_container(s,WOTR_LIST,options);}return cp_container(s,WOTR_DICT,entries);
}
void cbe_annotation_key(Snapshot& s,uint64_t payload,uint64_t round,std::string_view context,uint64_t rolls){
    auto emitted=field(s,payload,"annotated_rolls",false);if(emitted==missing){emitted=cp_container(s,WOTR_LIST,{});cp_put(s,payload,"annotated_rolls",emitted);}
    const auto key=cp_container(s,WOTR_TUPLE,{round,rr_string(s,context),cp_container(s,WOTR_TUPLE,child_edges(s,rolls))});auto entries=child_edges(s,emitted);for(auto old:entries)if(tail_equal(s,old,key))return;entries.push_back(key);set_edges(s,emitted,entries);
}
void cbe_annotations(Snapshot& s,uint64_t combat,uint64_t payload,bool siege,std::string_view defender_region){
    const auto round=obs_attr(s,combat,"siege_round");const auto context=std::string(siege?"siege ":"")+"battle at "+std::string(defender_region)+" r"+std::to_string(cp_number(s,combat,"siege_round"));
    for(int side=0;side<2;++side)for(const char* suffix:{"_combat_rolls","_rerolls"}){
        const auto rolls=cp_attr(s,combat,cp_role(side)+suffix);if(!truth(s,rolls))continue;const auto label=context+" "+cp_role(side)+(std::string(suffix)=="_rerolls"?" re-roll":"");cbe_annotation_key(s,payload,round,label,rolls);
    }
    for(int side=0;side<2;++side){const auto rolls=cp_attr(s,combat,cp_role(side)+"_pre_combat_rolls");if(!truth(s,rolls))continue;const auto card=cp_attr(s,combat,cp_role(side)+"_card");const auto name=ctx_none(s,card)?missing:obs_attr(s,card,"combat_name",false);const std::string label=(ctx_none(s,name)||!truth(s,name)?"pre-combat":std::string(text(s,name)))+" pre-combat attack";cbe_annotation_key(s,payload,round,label,rolls);}
}
uint64_t cbe_result(Snapshot& s,uint64_t combat,uint64_t args,int64_t rounds,int64_t atk_hits,int64_t def_hits){
    const auto parameters=child_edges(s,args);require(parameters.size()==7,"Battle resume arguments require seven entries");
    return mu_object(s,"game_state.CombatResult",{{"attacker_region",parameters[0]},{"defender_region",parameters[1]},
        {"rounds_fought",rr_scalar(s,WOTR_INT,rounds)},{"attacker_total_hits",rr_scalar(s,WOTR_INT,atk_hits)},{"defender_total_hits",rr_scalar(s,WOTR_INT,def_hits)},
        {"attacker_casualties",obs_attr(s,combat,"attacker_casualties")},{"defender_casualties",obs_attr(s,combat,"defender_casualties")},
        {"attacker_destroyed",obs_attr(s,combat,"attacker_destroyed")},{"defender_destroyed",obs_attr(s,combat,"defender_destroyed")},
        {"attacker_retreated",obs_attr(s,combat,"attacker_retreated")},{"defender_retreated",obs_attr(s,combat,"defender_retreated")},
        {"siege_continues",rr_scalar(s,WOTR_BOOL,0)},{"deferred_effect",rr_scalar(s,WOTR_NONE)}});
}
void cbe_accept_pause(Snapshot& s,uint64_t pc,uint64_t payload,uint64_t combat,uint64_t pending,void* rng){
    const auto args=field(s,payload,"args"),totals=field(s,payload,"totals");const auto arg_values=child_edges(s,args),values=child_edges(s,totals);require(values.size()==4,"Battle resume totals require four entries");
    auto number=[&](size_t index){return typed(s,values[index],WOTR_INT,"Battle totals must be integer").integer;};
    cp_put(s,payload,"totals",cp_container(s,WOTR_TUPLE,values));cbp_publish_armies(s,combat);cbp_publish_casualties(s,combat,payload,rng);
    require(arg_values.size()==7,"Battle resume arguments require seven entries");const std::string defender(text(s,arg_values[1]));cbe_annotations(s,combat,payload,truth(s,arg_values[4]),defender);
    const auto result=cbe_result(s,combat,args,number(2),number(0),number(1));rr_set_attr(s,result,"round_paused",rr_scalar(s,WOTR_BOOL,1));rr_set_attr(s,result,"round_resume",payload);rr_set_attr(s,result,"round_pending",pending);rr_set_attr(s,s.root,"_last_battle_result",result);
    const auto descriptor=cbe_descriptor_copy(s,pending),owner=field(s,descriptor,"owner");require((text_is(s,owner,"atk")||text_is(s,owner,"def"))&&truth(s,field(s,descriptor,"options")),"Invalid combat round pending descriptor");
    cp_put(s,descriptor,"round_stage",field(s,descriptor,"stage"));cp_put(s,descriptor,"stage",rr_string(s,"round"));const bool victim=text_is(s,field(s,descriptor,"effect"),"pre_combat_casualty");const bool attacker=text_is(s,owner,"atk");
    const auto cards=field(s,pc,"_round_cards");const auto ids=child_edges(s,cards);require(ids.size()==2,"Battle round card IDs require two entries");cp_put(s,descriptor,"card_id",ids[(attacker!=victim)?0:1]);
    cp_put(s,pc,"_round_resume",payload);cp_put(s,pc,"_round_cards",cp_container(s,WOTR_TUPLE,ids));cp_put(s,pc,"effect_choice",descriptor);cp_put(s,pc,"phase",rr_string(s,"combat_effect_choice"));cp_put(s,pc,"deciding_side",field(s,pc,attacker?"attacker_side":"defender_side"));
}
void cbe_reconcile(Snapshot& s,std::string_view region,int side,void* rng){
    const auto army=mu_army(s,sanitize_name(region));if(army==missing)return;const auto roster=ca_partition(s,army,region,side);
    // All participating characters die in an ordinary combat destruction.
    for(auto figure:roster.leaders)if(object_type(s,figure)=="game_logic.Minion"){
        const auto name=cp_lower(text(s,obs_attr(s,figure,"name")));const char* attr=cp_has(name,"saruman")?"saruman":cp_has(name,"witch")?"witch_king":cp_has(name,"mouth")?"mouth_of_sauron":nullptr;
        if(attr){const auto sheet=obs_attr(s,s.root,attr,false);if(!ctx_none(s,sheet))rr_set_attr(s,sheet,"in_play",rr_scalar(s,WOTR_BOOL,0));}
        // Shared card-kill bookkeeping needs only a live-state reference.
        const auto context=mu_object(s,"game_state.CombatState",{{"_live_state",s.root}});cs_minion_bookkeeping(s,context,figure);
    }
    for(auto figure:roster.leaders)if(cs_plain_fp(s,figure))cbp_book(s,obs_attr(s,figure,"nation"),"leader");
    for(auto figure:roster.leaders)if(object_type(s,figure)=="game_state.Companion")cbp_companion_death(s,rng,false);
    rr_set_attr(s,army,"leaders",cp_container(s,WOTR_LIST,roster.rear_leaders));
    if(!truth(s,obs_attr(s,army,"units"))&&roster.rear_leaders.empty()){
        const auto map=obs_attr(s,s.root,"armies");const auto entries=child_edges(s,map);for(size_t i=0;i<entries.size();i+=2)if(entries[i+1]==army&&sanitize_name(text(s,entries[i]))==sanitize_name(region)){const std::string key(text(s,entries[i]));cp_erase(s,map,key);break;}
    }
}
bool cbe_palantir_blocked(const Snapshot& s){
    for(const char* key:{"pending_combat","pending_hunt","pending_guide_change","pending_card_option","pending_card_followup","pending_discard","pending_army_move_die","pending_muster_die","_fellowship_move_die_pending"})
        if(!ctx_none(s,obs_attr(s,s.root,key,false)))return true;
    for(const char* key:{"_pending_hunt_damage","_hunt_reveal_pending","_reveal_place_due","_pending_reveal_strongholds","_fellowship_route_event","_hobbit_place_queue","_overstack_regions","_wwsat_owed","_mithril_stash","_wizardstaff_stash"})
        if(truth(s,obs_attr(s,s.root,key,false)))return true;
    return false;
}
void cbe_palantir(Snapshot& s){
    if(!truth(s,obs_attr(s,s.root,"_palantir_draw_pending",false))||cbe_palantir_blocked(s))return;
    rr_set_attr(s,s.root,"_palantir_draw_pending",rr_scalar(s,WOTR_BOOL,0));const auto persistent=obs_attr(s,s.root,"persistent_cards");bool live=false;for(const char* side:{"fp_cards","sh_cards"})for(auto card:child_edges(s,obs_attr(s,persistent,side)))live|=enum_is(s,obs_attr(s,card,"card_type"),"game_state.PersistentCardType","PALANTIR_OF_ORTHANC");if(!live)return;
    if(!ctx_none(s,obs_attr(s,s.root,"pending_card_option",false))){rr_set_attr(s,s.root,"_palantir_draw_pending",rr_scalar(s,WOTR_BOOL,1));return;}
    std::vector<uint64_t> options;for(const char* deck:{"character","strategy"})if(truth(s,obs_attr(s,s.root,std::string(deck)=="character"?"sh_char_deck":"sh_strat_deck")))options.push_back(cp_container(s,WOTR_DICT,{rr_string(s,"palantir_deck"),rr_string(s,deck)}));if(options.empty())return;
    rr_set_attr(s,s.root,"pending_card_option",cp_container(s,WOTR_DICT,{rr_string(s,"card_id"),rr_string(s,"__palantir_draw__"),rr_string(s,"card_name"),rr_string(s,"Palantir draw"),rr_string(s,"side"),rr_string(s,"sh"),rr_string(s,"decider"),rr_string(s,"sh"),rr_string(s,"direct"),rr_scalar(s,WOTR_BOOL,1),rr_string(s,"palantir_draw"),rr_scalar(s,WOTR_BOOL,1),rr_string(s,"options"),cp_container(s,WOTR_LIST,options)}));
}
void cv_register_overstack(Snapshot&,const Snapshot&,std::string_view,int);
void cbe_finish(Snapshot& s,const Snapshot& tables,uint64_t pc,bool attacker_dead,bool defender_dead,am_Builder* events=nullptr,std::vector<uint64_t>* rows=nullptr){
    crs_witch_king_draw(s,pc,events,rows);
    const std::string attacker(text(s,field(s,pc,"attacker_region"))),defender(text(s,field(s,pc,"defender_region")));
    if(defender_dead&&!attacker_dead&&sanitize_name(attacker)!=sanitize_name(defender)){
        const auto army=mu_army(s,sanitize_name(attacker));if(army!=missing&&truth(s,obs_attr(s,army,"units"))){su_cleanup(s,tables);const auto rear=field(s,pc,"attacker_rearguard",false),rounds=field(s,pc,"rounds_fought",false);
            rr_set_attr(s,s.root,"pending_combat",cp_container(s,WOTR_DICT,{rr_string(s,"phase"),rr_string(s,"victor_advance"),rr_string(s,"attacker_rearguard"),rear==missing?rr_scalar(s,WOTR_NONE):rear,rr_string(s,"attacker_region"),field(s,pc,"attacker_region"),rr_string(s,"defender_region"),field(s,pc,"defender_region"),rr_string(s,"attacker_side"),field(s,pc,"attacker_side"),rr_string(s,"defender_side"),field(s,pc,"defender_side"),rr_string(s,"deciding_side"),field(s,pc,"attacker_side"),rr_string(s,"is_siege"),rr_scalar(s,WOTR_BOOL,0),rr_string(s,"rounds_fought"),rounds==missing?rr_scalar(s,WOTR_INT,0):rounds}));return;
        }
    }
    su_cleanup(s,tables);rr_set_attr(s,s.root,"pending_combat",rr_scalar(s,WOTR_NONE));if(defender_dead&&truth(s,field(s,pc,"corsairs_relief_garrison",false)))cv_register_overstack(s,tables,defender,ctx_side(s,field(s,pc,"attacker_side")));cbe_palantir(s);
}
bool cbe_onslaught(Snapshot& s,uint64_t pc,int role,uint64_t card,bool attacker_dead,bool defender_dead,am_Builder* events=nullptr,std::vector<uint64_t>* rows=nullptr){
    if(role<0||role>1||(role==0?attacker_dead:defender_dead))return false;const auto own=role==0?"attacker":"defender";const auto side_node=field(s,pc,(std::string(own)+"_side").c_str());const int side=enum_is(s,side_node,"game_state.Side","FREE_PEOPLES")?0:1;const auto region=text(s,field(s,pc,(std::string(own)+"_region").c_str()));int64_t capacity=0;for(auto unit:ca_units(s,region,side))capacity+=enum_is(s,obs_attr(s,unit,"kind"),"game_state.UnitKind","ELITE")?2:1;const auto maximum=std::min<int64_t>(4,capacity);if(maximum<=0)return false;
    if(attacker_dead||defender_dead){
        if(events&&rows)rows->push_back(events->row(-1,"onslaught_deferred_on_destroyed",{events->str("region"),events->str(text(s,field(s,pc,"defender_region"))),events->str("card"),events->str(ctx_none(s,card)?"None":text(s,card)),events->str("role"),events->str(own)}));
        cp_put(s,pc,"_finish_atk_dead",rr_scalar(s,WOTR_BOOL,attacker_dead));cp_put(s,pc,"_finish_def_dead",rr_scalar(s,WOTR_BOOL,defender_dead));}
    cp_put(s,pc,"effect_choice",cp_container(s,WOTR_DICT,{rr_string(s,"stage"),rr_string(s,"post"),rr_string(s,"owner"),rr_string(s,role==0?"atk":"def"),rr_string(s,"effect"),rr_string(s,"onslaught"),rr_string(s,"card_id"),ctx_none(s,card)||!truth(s,card)?rr_string(s,""):card,rr_string(s,"max_count"),rr_scalar(s,WOTR_INT,maximum)}));cp_put(s,pc,"phase",rr_string(s,"combat_effect_choice"));cp_put(s,pc,"deciding_side",side_node);return true;
}
void cbe_next_round(Snapshot& s,uint64_t pc,am_Builder* events=nullptr,std::vector<uint64_t>* rows=nullptr){
    crs_witch_king_draw(s,pc,events,rows);
    require(ctx_none(s,field(s,pc,"_round_resume",false)),"Paused combat must resume with its observed-stage choice");
    const auto rounds=field(s,pc,"rounds_fought",false);require(!ctx_none(s,rounds)&&typed(s,rounds,WOTR_INT,"Round count must be integer").integer>0,"Next-round opener requires an already completed round");
    cp_erase(s,pc,"_observed_round_completed");for(const char* key:{"atk_casualty_policy","def_casualty_policy"})if(field(s,pc,key,false)==missing)cp_put(s,pc,key,rr_string(s,"preserve_elites"));cp_put(s,pc,"is_siege",rr_scalar(s,WOTR_BOOL,1));
    if(!truth(s,field(s,pc,"shine_asked",false))){bool white=false,nazgul=false;for(int role=0;role<2;++role){const std::string region(text(s,field(s,pc,role==0?"attacker_region":"defender_region")));const int side=enum_is(s,field(s,pc,role==0?"attacker_side":"defender_side"),"game_state.Side","FREE_PEOPLES")?0:1;const auto army=mu_army(s,sanitize_name(region));require(army!=missing,"Next round requires both live armies");for(auto figure:ca_partition(s,army,region,side).leaders){const auto name=obs_attr(s,figure,"name",false);if(side==0&&object_type(s,figure)=="game_state.Companion")white|=truth(s,obs_attr(s,figure,"can_cancel_nazgul_leadership",false))||text_is(s,name,"Gandalf the White");if(side==1){std::string letters;if(!ctx_none(s,name))for(char c:cp_lower(text(s,name)))if(c>='a'&&c<='z')letters+=c;nazgul|=object_type(s,figure)=="game_state.Nazgul"||letters.find("witchking")!=std::string::npos;}}}
        if(white&&nazgul){cp_put(s,pc,"phase",rr_string(s,"fp_shine"));cp_put(s,pc,"deciding_side",mu_enum(s,"game_state.Side","FREE_PEOPLES"));return;}cp_put(s,pc,"shine_asked",rr_scalar(s,WOTR_BOOL,1));
    }
    require(field(s,pc,"atk_combat_card",false)==missing,"Completed round must consume attacker card commitment before next-round entry");
    cp_put(s,pc,"phase",rr_string(s,"attacker_combat_card"));cp_put(s,pc,"deciding_side",field(s,pc,"attacker_side"));
}
void cbe_post_round(Snapshot& s,uint64_t pc,am_Builder* events=nullptr,std::vector<uint64_t>* rows=nullptr){
    const bool siege=truth(s,field(s,pc,"is_siege",false));const auto mandatory=field(s,pc,"mandatory_rounds",false),rounds=field(s,pc,"rounds_fought",false);
    if(siege&&(ctx_none(s,mandatory)?0:s.nodes[mandatory].integer)>(ctx_none(s,rounds)?0:s.nodes[rounds].integer)){cbe_next_round(s,pc,events,rows);return;}
    cp_put(s,pc,"phase",rr_string(s,siege?"combat_round_result":"attacker_continue"));cp_put(s,pc,"deciding_side",field(s,pc,"attacker_side"));
}
void cbe_complete_allocated(crr_Result& owned,const Snapshot& tables,am_Builder* events=nullptr,std::vector<uint64_t>* rows=nullptr){
    for(int iteration=0;iteration<4;++iteration){
        auto& before=*owned.state;const auto old_pc=obs_attr(before,before.root,"pending_combat");const std::array<std::string,2> region={std::string(text(before,field(before,old_pc,"attacker_region"))),std::string(text(before,field(before,old_pc,"defender_region")))};std::array<int,2> side{};std::array<bool,2> dead{};
        for(int role=0;role<2;++role){side[role]=enum_is(before,field(before,old_pc,role==0?"attacker_side":"defender_side"),"game_state.Side","FREE_PEOPLES")?0:1;dead[role]=ca_units(before,region[role],side[role]).empty();}
        const auto prior=field(before,old_pc,"_p9",false);const bool has_cards=!ctx_none(before,prior)&&(truth(before,field(before,prior,"atk_card",false))||truth(before,field(before,prior,"def_card",false)));
        if(!truth(before,field(before,old_pc,"_p9_done",false))&&has_cards&&!(dead[0]&&dead[1])){
            auto post=combat_deferred_post(before,tables,owned.random.get());owned=std::move(post.owned);auto& s=*owned.state;const auto pc=obs_attr(s,s.root,"pending_combat");cp_put(s,pc,"_p9_done",rr_scalar(s,WOTR_BOOL,1));cp_put(s,pc,"_p9_role",post.role<0?rr_scalar(s,WOTR_NONE):rr_string(s,post.role==0?"attacker":"defender"));dead[0]|=post.attacker_destroyed;dead[1]|=post.defender_destroyed;
            const auto dh=dead[1]?0:post.hits_on_defender,ah=dead[0]?0:post.hits_on_attacker;if(dh>0||ah>0){cp_put(s,pc,"def_hits",rr_scalar(s,WOTR_INT,dh));cp_put(s,pc,"atk_hits",rr_scalar(s,WOTR_INT,ah));cp_put(s,pc,"alloc_stage",rr_string(s,"attacker"));if(ca_enter(s,tables,pc))return;continue;}
        }
        auto& s=*owned.state;const auto pc=obs_attr(s,s.root,"pending_combat"),p9=field(s,pc,"_p9",false),role_node=field(s,pc,"_p9_role",false);const int role=text_is(s,role_node,"attacker")?0:text_is(s,role_node,"defender")?1:-1;const auto card=ctx_none(s,p9)?missing:field(s,p9,role==0?"atk_card":"def_card",false);const bool onslaught_done=truth(s,field(s,pc,"_onslaught_done",false));
        cp_erase(s,pc,"_onslaught_done");cp_erase(s,pc,"_p9");cp_erase(s,pc,"_p9_done");cp_erase(s,pc,"_p9_role");for(int who=0;who<2;++who)if(dead[who])cbe_reconcile(s,region[who],side[who],owned.random.get());
        // Onslaught may remove the last besieger after the defender is gone.
        // Preserve ownership and siege provenance until its payment resolves.
        if(!onslaught_done&&cbe_onslaught(s,pc,role,card,dead[0],dead[1],events,rows))return;
        if(dead[1]&&!dead[0]&&sanitize_name(region[0])==sanitize_name(region[1])&&c9_garrison(s,tables,sanitize_name(region[1]))==side[1]){
            const auto key=sanitize_name(region[1]);const auto meta=field(tables,field(tables,tables.root,"board_regions"),key.c_str(),false);if(!ctx_none(tables,meta)&&(truth(tables,field(tables,meta,"is_city",false))||truth(tables,field(tables,meta,"is_town",false))||truth(tables,field(tables,meta,"is_stronghold",false))))nt_capture(s,tables,key,side[0]);
        }
        su_cleanup(s,tables);
        if(dead[0]||dead[1]){cbe_finish(s,tables,pc,dead[0],dead[1],events,rows);return;}
        cbe_post_round(s,pc,events,rows);return;
    }
    throw std::invalid_argument("Deferred combat aftermath failed to settle");
}
crr_Result combat_allocation_settle(const Snapshot& source,const Snapshot& tables,const void* random_source,bool enter){
    const auto pc=obs_attr(source,source.root,"pending_combat");typed(source,pc,WOTR_DICT,"Casualty settlement requires pending combat");
    crr_Result owned;owned.state=std::make_unique<Snapshot>(source);owned.random.reset(wotr_random_clone(random_source));if(!owned.random)throw std::invalid_argument(wotr_random_last_error());
    if(!enter||!ca_enter(*owned.state,tables,pc))cbe_complete_allocated(owned,tables);
    owned.state=ac_compact(*owned.state);return owned;
}
