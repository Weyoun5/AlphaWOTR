// RoundCursor stages0..4. Explicit battle_context uses the resume payload's
// frozen replacement baseline and consumption ledger, never module globals.
bool ce_defender_first(const Snapshot& s,uint64_t combat){
    int64_t initiative[2];for(int side=0;side<2;++side){const auto card=cp_card(s,combat,side);initiative[side]=card==missing?99:cp_number(s,card,"combat_value");}return initiative[1]<=initiative[0];
}
void ce_begin(Snapshot& s,uint64_t combat){
    cp_set_number(s,combat,"attacker_final_hits",0);cp_set_number(s,combat,"defender_final_hits",0);
    rr_set_attr(s,combat,"_round_paid_leadership",cp_container(s,WOTR_LIST,{}));rr_set_attr(s,combat,"_round_nazgul_present",obs_attr(s,combat,"nazgul_in_battle"));
    const auto shined=obs_attr(s,combat,"_shined_companion",false),cancelled=obs_attr(s,combat,"_wop_cancelled",false);
    const auto forfeited=cp_container(s,WOTR_SET,ctx_none(s,shined)?std::vector<uint64_t>{}:std::vector<uint64_t>{obs_attr(s,shined,"name")});rr_set_attr(s,combat,"_leadership_forfeited",forfeited);
    const auto companions=child_edges(s,obs_attr(s,combat,"companions_in_battle"));const bool defender_first=ce_defender_first(s,combat);
    for(int step=0;step<2;++step){const int side=(step==0?defender_first:!defender_first)?1:0;const auto effect=cp_card_text(s,combat,side,"combat_effect");
        if(!cp_has(effect,"cancel the effects of the combat card"))continue;
        const bool cost=cp_has(effect,"forfeit")&&cp_has(effect,"leadership");
        // Almanac p38: Daring Defiance pays no Leadership when no enemy card exists.
        if(cost&&ctx_none(s,cp_attr(s,combat,cp_role(1-side)+"_card")))continue;
        if(cost){bool already=false;for(auto c:companions)for(auto name:child_edges(s,forfeited))already|=text(s,name)==text(s,obs_attr(s,c,"name"));if(already)continue;}
        if(truth(s,cp_attr(s,combat,cp_role(1-side)+"_card")))rr_set_attr(s,combat,cp_role(1-side)+"_card_cancelled",rr_scalar(s,WOTR_BOOL,1));
        if(cost){int64_t sum=0;for(auto c:companions)if(c!=shined&&c!=cancelled){sum+=cp_number(s,c,"leadership",1,true);const std::string name(text(s,obs_attr(s,c,"name")));crr_mark(s,forfeited,name);}cs_drop_leadership(s,combat,side,sum);}
    }
}
void ce_words(Snapshot& s,uint64_t cursor,uint64_t combat){
    for(int side=0;side<2;++side){const auto effect=cp_card_text(s,combat,side,"combat_effect");
        if(!cp_has(effect,"companion")||!cp_has(effect,"leadership")||!cp_has(effect,"cancelled")||cp_number(s,combat,"nazgul_in_battle")<1)continue;
        const auto companions=child_edges(s,obs_attr(s,combat,"companions_in_battle"));if(companions.empty())continue;
        const auto choices=obs_attr(s,cursor,"choices"),requested=field(s,choices,(cp_short(side)+"_effect_target").c_str(),false);uint64_t target=missing;
        if(!ctx_none(s,requested)&&truth(s,requested)&&!text_is(s,requested,"leader"))for(auto c:companions)if(text(s,obs_attr(s,c,"name"))==text(s,requested)){target=c;break;}
        if(target==missing)for(auto c:companions)if(target==missing||cp_number(s,c,"leadership",0,true)>cp_number(s,target,"leadership",0,true))target=c;
        if(target==obs_attr(s,combat,"_shine_companion",false)){const int shadow=enum_is(s,obs_attr(s,combat,"attacker_side"),"game_state.Side","SHADOW")?0:1;cp_add(s,combat,cp_role(shadow)+"_leadership",cp_number(s,combat,"_shine_cancelled_leadership",0,true));}
        if(target!=obs_attr(s,combat,"_shined_companion",false))cs_drop_leadership(s,combat,1-side,cp_number(s,target,"leadership",0,true));
        rr_set_attr(s,combat,"_wop_cancelled",target);rr_set_attr(s,combat,"_wop_cancelled_is_attacker",rr_scalar(s,WOTR_BOOL,side==1));
    }
}
void ce_modifiers(Snapshot& s,uint64_t combat){
    for(int side=0;side<2;++side){const auto effect=cp_card_text(s,combat,side,"combat_effect"),role=cp_role(side),enemy=cp_role(1-side);
        int both=cp_has(effect,"both armies add 2")?2:cp_has(effect,"both armies add 1")?1:0;
        if(both){for(int owner=0;owner<2;++owner){cp_add(s,combat,cp_role(owner)+"_dice_mod",both);cp_add(s,combat,cp_role(owner)+"_reroll_mod",both);}}
        else if(cp_has(effect,"subtract 1")&&cp_has(effect,"nazg")&&cp_has(effect,"leader re-roll")){}
        else if(cp_has(effect,"subtract 2")&&cp_has(effect,"nazg")&&cp_has(effect,"add 1")){if(ctx_none(s,obs_attr(s,combat,"_shine_companion",false))){cs_drop_leadership(s,combat,side,2);cp_add(s,combat,role+"_dice_mod",1);}}
        else if(cp_has(effect,"inflict")&&cp_has(effect,"up to two hits")){}
        else{
            // Literal equivalent of add ([12]) to all dice on (your|their)
            // followed by every character before the first period.
            size_t best=std::string::npos,start=0;int amount=0;
            for(int n=1;n<=2;++n)for(const char* owner:{"your","their"}){const auto prefix=std::string("add ")+std::to_string(n)+" to all dice on "+owner+" ";const auto p=effect.find(prefix);if(p<best){best=p;start=p+prefix.size();amount=n;}}
            if(amount){const auto stop=effect.find('.',start);const auto target=effect.substr(start,stop==std::string::npos?stop:stop-start);
                if(cp_has(target,"combat"))cp_add(s,combat,role+"_dice_mod",amount);
                if(cp_has(target,"re-roll")&&(!cp_has(effect,"did not play a card")||ctx_none(s,cp_attr(s,combat,enemy+"_card"))))cp_add(s,combat,role+"_reroll_mod",amount);
            }
        }
        if(cp_has(effect,"subtract 1 from all dice")&&cp_has(effect,"shadow"))cp_add(s,combat,enemy+"_dice_mod",-1);
    }
    for(int side=0;side<2;++side){const auto effect=cp_card_text(s,combat,side,"combat_effect"),enemy=cp_role(1-side);
        if(cp_has(effect,"maximum of two dice"))cp_set_number(s,combat,enemy+"_dice_cap",2);else if(cp_has(effect,"maximum of three dice"))cp_set_number(s,combat,enemy+"_dice_cap",3);
        if(cp_has(effect,"rolls one die less")&&cp_has(effect,"companion"))cp_set_number(s,combat,enemy+"_dice_reduction",cp_number(s,combat,enemy+"_dice_reduction",0,true)+child_edges(s,obs_attr(s,combat,"companions_in_battle")).size());
    }
}
bool ce_dread(Snapshot& s,uint64_t cursor,uint64_t combat){
    const auto choices=obs_attr(s,cursor,"choices");
    for(int side=0;side<2;++side)if(cp_card_text(s,combat,side,"combat_name")=="dread and despair"&&ctx_none(s,obs_attr(s,combat,"_shine_companion",false))){
        // The one-die floor limits the result, not the legal leadership cost.
        const auto maximum=std::max<int64_t>(0,cp_number(s,combat,"nazgul_in_battle"));
        if(!maximum)continue;
        std::vector<uint64_t> options;for(int64_t n=1;n<=maximum;++n)options.push_back(cs_option(s,n,nullptr));if(!cs_choose(s,cursor,"dread_and_despair",side,options))return false;
    }
    for(int side=0;side<2;++side){const auto effect=cp_card_text(s,combat,side,"combat_effect");
        if(!cp_has(effect,"forfeit")||!cp_has(effect,"nazg")||!cp_has(effect,"one die less")||!ctx_none(s,obs_attr(s,combat,"_shine_companion",false)))continue;
        const auto available=cp_number(s,combat,"nazgul_in_battle");if(available<=0)continue;const auto strength=cp_strength(s,cp_attr(s,combat,cp_role(1-side)+"_army"),missing);
        const auto count=field(s,choices,(cp_short(side)+"_effect_count").c_str(),false);const auto forfeit=ctx_none(s,count)?std::min(std::max<int64_t>(1,strength-1),std::max<int64_t>(1,available-1)):s.nodes[count].integer;
        require(forfeit>=1&&forfeit<=available,"Dread and Despair must forfeit between 1 and the available Nazgul leadership");
        if(forfeit>0){cp_add(s,combat,cp_role(side)+"_leadership",-forfeit);cp_set_number(s,combat,cp_role(1-side)+"_dice_cap",std::max<int64_t>(1,strength-forfeit));cp_add(s,combat,"nazgul_in_battle",-forfeit);}
    }return true;
}
void ce_enum_put(Snapshot& s,uint64_t map,uint64_t nation,uint64_t value){
    const std::string name(enum_member(s,nation,"game_state.Nation"));auto entries=child_edges(s,map);for(size_t i=0;i<entries.size();i+=2)if(enum_is(s,entries[i],"game_state.Nation",name)){entries[i+1]=value;set_edges(s,map,entries);return;}entries.push_back(nation);entries.push_back(value);set_edges(s,map,entries);
}
bool ce_replacement(Snapshot& s,uint64_t resume,uint64_t nation,bool battle){
    if(!battle)return true;
    const std::string name(enum_member(s,nation,"game_state.Nation"));const auto ledger=field(s,resume,"downgrades"),baseline=field(s,resume,"reinforcement_baseline");
    const auto counts=ctx_nation_value(s,obs_attr(s,s.root,"fp_casualties"),name),regular=counts==missing?missing:field(s,counts,"regular",false);
    const bool from_box=fp_nation(name)&&regular!=missing&&s.nodes[regular].integer>0;
    const auto supply=ctx_nation_value(s,baseline,name),used=ctx_nation_value(s,ledger,name);
    require(supply!=missing,"Battle baseline must include every downgraded nation");
    const auto consumed=used==missing?0:typed(s,used,WOTR_INT,"Downgrade count must be integer").integer;
    const bool from_pool=!from_box&&typed(s,supply,WOTR_INT,"Reinforcement baseline must be integer").integer>consumed;
    if(from_box)cp_put(s,counts,"regular",rr_scalar(s,WOTR_INT,s.nodes[regular].integer-1));
    else if(from_pool)ce_enum_put(s,ledger,nation,rr_scalar(s,WOTR_INT,consumed+1));
    auto log=obs_attr(s,s.root,"_dg_src_log",false);if(ctx_none(s,log)){log=cp_container(s,WOTR_LIST,{});rr_set_attr(s,s.root,"_dg_src_log",log);}
    auto entries=child_edges(s,log);entries.push_back(cp_container(s,WOTR_TUPLE,{nation,from_box?rr_string(s,"cas"):from_pool?rr_string(s,"pool"):rr_scalar(s,WOTR_NONE)}));set_edges(s,log,entries);return from_box||from_pool;
}
bool ce_payment(Snapshot& s,uint64_t cursor,uint64_t combat,uint64_t resume,int side,int64_t remaining,const char* effect,bool battle,int64_t& cost){
    const auto role=cp_role(side);const auto units=obs_attr(s,cp_attr(s,combat,role+"_army"),"units");std::vector<uint64_t> options;
    for(auto unit:child_edges(s,units)){
        const bool elite=enum_is(s,obs_attr(s,unit,"kind"),"game_state.UnitKind","ELITE");const std::string nation(enum_member(s,obs_attr(s,unit,"nation"),"game_state.Nation"));
        for(int op=0;op<(elite&&remaining>=2?2:1);++op){const auto option=cp_container(s,WOTR_DICT,{rr_string(s,"nation"),rr_string(s,nation),rr_string(s,"kind"),rr_string(s,elite?"elite":"regular"),rr_string(s,"operation"),rr_string(s,elite&&op==0?"downgrade":"remove")});bool duplicate=false;for(auto old:options)duplicate|=cs_option_equal(s,old,option);if(!duplicate)options.push_back(option);}
    }
    const auto selection=obs_attr(s,cursor,"selection");if(ctx_none(s,selection)){
        rr_set_attr(s,cursor,"pending",cp_container(s,WOTR_DICT,{rr_string(s,"stage"),obs_attr(s,cursor,"stage"),rr_string(s,"effect"),rr_string(s,effect),rr_string(s,"owner"),rr_string(s,cp_short(side)),rr_string(s,"options"),cp_container(s,WOTR_LIST,options)}));cost=0;return false;
    }
    bool legal=false;for(auto option:options)legal|=cs_option_equal(s,option,selection);require(legal,"Illegal observed casualty payment");rr_set_attr(s,cursor,"selection",rr_scalar(s,WOTR_NONE));
    const std::string nation(text(s,field(s,selection,"nation")));const bool elite=text_is(s,field(s,selection,"kind"),"elite"),downgrade=text_is(s,field(s,selection,"operation"),"downgrade");uint64_t unit=missing;
    for(auto candidate:child_edges(s,units))if(enum_is(s,obs_attr(s,candidate,"nation"),"game_state.Nation",nation)&&(enum_is(s,obs_attr(s,candidate,"kind"),"game_state.UnitKind","ELITE")==elite)){unit=candidate;break;}
    require(unit!=missing,"Casualty payment figure is missing");cs_remove(s,units,unit);cost=elite&&!downgrade?2:1;
    // Net pool consumption includes Shadow Regulars returned before a later
    // replacement, even while split battle rosters await publication.
    if(battle&&!elite&&!fp_nation(nation)){
        const auto ledger=field(s,resume,"downgrades"),used=ctx_nation_value(s,ledger,nation);
        const auto consumed=used==missing?0:typed(s,used,WOTR_INT,"Downgrade count must be integer").integer;
        ce_enum_put(s,ledger,obs_attr(s,unit,"nation"),rr_scalar(s,WOTR_INT,consumed-1));
    }
    if(downgrade&&ce_replacement(s,resume,obs_attr(s,unit,"nation"),battle)){
        const auto replacement=mu_object(s,"game_state.Unit",{{"nation",obs_attr(s,unit,"nation")},{"kind",mu_enum(s,"game_state.UnitKind","REGULAR")}});auto survivors=child_edges(s,units);survivors.push_back(replacement);set_edges(s,units,survivors);rr_set_attr(s,unit,"_downgraded",rr_scalar(s,WOTR_BOOL,1));
    }
    const auto casualties=cp_attr(s,combat,role+"_casualties");auto dead=child_edges(s,casualties);dead.push_back(unit);set_edges(s,casualties,dead);
    if(elite&&nation=="ISENGARD"&&truth(s,obs_attr(s,combat,"_saruman_in_play",false)))cs_drop_leadership(s,combat,side,1);return true;
}
void ce_pre_roll(Snapshot& s,uint64_t combat,int side,void* rng){
    const auto effect=cp_card_text(s,combat,side,"combat_effect"),role=cp_role(side);
    if(!cp_has(effect,"before the combat roll")||!cp_has(effect,"additional attack"))return;
    int64_t threshold=side==0&&(truth(s,obs_attr(s,combat,"is_siege"))||(truth(s,obs_attr(s,combat,"city_defense"))&&cp_number(s,combat,"siege_round",1,true)==1))?6:5,count=0;
    if(cp_has(effect,"leadership"))count=std::min<int64_t>(5,cp_number(s,combat,role+"_leadership"));
    else if(cp_has(effect,"elite units")){for(auto u:child_edges(s,obs_attr(s,cp_attr(s,combat,role+"_army"),"units")))if(enum_is(s,obs_attr(s,u,"kind"),"game_state.UnitKind","ELITE"))++count;count=std::min<int64_t>(5,count);}
    else if(cp_has(effect,"three combat dice")){count=3;threshold=4;}
    int64_t hits=0;std::array<uint32_t,5> rolls{};if(count>0){cf_d6(rng,rolls.data(),static_cast<size_t>(count),threshold,0,true);for(int64_t i=0;i<count;++i)if(crr_hit(rolls[i],threshold,0))++hits;}
    cp_add(s,combat,role+"_pre_combat_hits",hits);const auto list=cp_attr(s,combat,role+"_pre_combat_rolls");auto entries=child_edges(s,list);for(int64_t i=0;i<count;++i)entries.push_back(rr_scalar(s,WOTR_INT,rolls[i]));set_edges(s,list,entries);
}
bool ce_pre_attacks(Snapshot& s,uint64_t cursor,uint64_t combat,uint64_t resume,void* rng,bool battle){
    const bool defender_first=ce_defender_first(s,combat);while(cp_number(s,cursor,"pre_attack_index")<2){
        const auto index=cp_number(s,cursor,"pre_attack_index");require(index>=0,"Pre-attack index must be nonnegative");const int side=(index==0?defender_first:!defender_first)?1:0,victim=1-side;const auto role=cp_role(side),enemy=cp_role(victim);const auto units=obs_attr(s,cp_attr(s,combat,enemy+"_army"),"units");
        if(ctx_none(s,obs_attr(s,cursor,"pre_remaining"))){const auto before=cp_number(s,combat,role+"_pre_combat_hits");ce_pre_roll(s,combat,side,rng);const auto hits=cp_number(s,combat,role+"_pre_combat_hits")-before;
            cp_set_number(s,cursor,"pre_remaining",hits);rr_set_attr(s,cursor,"pre_total",obs_attr(s,cursor,"pre_remaining"));cp_set_number(s,cursor,"pre_paid",0);rr_set_attr(s,cursor,"pre_victim_attacker",rr_scalar(s,WOTR_BOOL,victim==0));crr_mark(s,obs_attr(s,cursor,"observed_rolls"),"pre_combat");crr_mark(s,obs_attr(s,cursor,"observed_roll_sides"),"pre_combat",role);
        }
        while(cp_number(s,cursor,"pre_remaining")&&truth(s,units)){int64_t cost=0;if(!ce_payment(s,cursor,combat,resume,victim,cp_number(s,cursor,"pre_remaining"),"pre_combat_casualty",battle,cost))return false;cp_add(s,cursor,"pre_remaining",-cost);cp_add(s,cursor,"pre_paid",cost);}
        if(!truth(s,units)){rr_set_attr(s,combat,enemy+"_destroyed",rr_scalar(s,WOTR_BOOL,1));rr_set_attr(s,combat,"battle_over",rr_scalar(s,WOTR_BOOL,1));return true;}
        cp_add(s,cursor,"pre_attack_index",1);rr_set_attr(s,cursor,"pre_remaining",rr_scalar(s,WOTR_NONE));
    }return true;
}
bool ce_relentless(Snapshot& s,uint64_t cursor,uint64_t combat,uint64_t resume,bool battle){
    for(int side=0;side<2;++side)if(cp_card_text(s,combat,side,"combat_name")=="relentless assault"){
        const auto role=cp_role(side);const auto units=obs_attr(s,cp_attr(s,combat,role+"_army"),"units");
        if(ctx_none(s,obs_attr(s,cursor,"ra_remaining"))){int64_t capacity=0;for(auto u:child_edges(s,units))capacity+=enum_is(s,obs_attr(s,u,"kind"),"game_state.UnitKind","ELITE")?2:1;std::vector<uint64_t> options;for(int64_t n=0;n<=std::min<int64_t>(2,capacity);++n)options.push_back(cs_option(s,n,nullptr));if(!cs_choose(s,cursor,"relentless_assault",side,options))return false;
            rr_set_attr(s,cursor,"ra_remaining",field(s,obs_attr(s,cursor,"choices"),(cp_short(side)+"_effect_count").c_str()));rr_set_attr(s,cursor,"ra_attacker",rr_scalar(s,WOTR_BOOL,side==0));
        }
        while(cp_number(s,cursor,"ra_remaining")&&truth(s,units)){int64_t cost=0;if(!ce_payment(s,cursor,combat,resume,side,cp_number(s,cursor,"ra_remaining"),"relentless_assault_payment",battle,cost))return false;cp_add(s,cursor,"ra_remaining",-cost);cp_add(s,cursor,"ra_paid",cost);cp_add(s,combat,role+"_dice_mod",cost);}
        if(!truth(s,units)){rr_set_attr(s,combat,role+"_destroyed",rr_scalar(s,WOTR_BOOL,1));rr_set_attr(s,combat,"battle_over",rr_scalar(s,WOTR_BOOL,1));}
    }return true;
}
bool combat_pre_stage_inplace(Snapshot& s,void* random_source,int64_t selection_index,bool battle){
    const auto& source=s;
    attributes(source,source.root,"game_state.GameState");const auto pc=obs_attr(source,source.root,"pending_combat"),resume=field(source,pc,"_round_resume"),cursor=field(source,resume,"cursor");attributes(source,cursor,"combat_round_resume.RoundCursor");
    require(text_is(source,obs_attr(source,cursor,"schema"),"combat_round_cursor_v1"),"Unknown combat round cursor schema");auto stage=cp_number(source,cursor,"stage");require(stage>=0&&stage<=4,"Pre-combat requires cursor stage0..4");require(ctx_none(source,obs_attr(source,cursor,"selection")),"Pre-combat refuses preloaded selection");
    const auto combat=obs_attr(source,cursor,"combat"),pending=obs_attr(source,cursor,"pending");attributes(source,combat,"game_state.CombatState");
    if(battle){require(text_is(source,field(source,resume,"schema"),"combat_battle_resume_v1"),"Native battle context requires resume schema");require(obs_attr(source,combat,"_live_state",false)==source.root,"Native battle context must own its live GameState");
        for(const char* key:{"downgrades","reinforcement_baseline"}){const auto map=field(source,resume,key);typed(source,map,WOTR_DICT,"Native battle context requires explicit supply ledgers");const auto entries=child_edges(source,map);for(size_t i=0;i<entries.size();i+=2){validate_nation_name(enum_member(source,entries[i],"game_state.Nation"));const auto n=typed(source,entries[i+1],WOTR_INT,"Supply ledger must contain integer counts").integer;require(n>=(std::string_view(key)=="downgrades"?-1048576:0)&&n<=1048576,"Supply ledger count out of range");}}
    }
    require(selection_index>=-1,"Pre-combat choice index must be -1 or nonnegative");
    if(selection_index<0&&!ctx_none(s,pending))return false;
    if(battle)for(const char* key:{"downgrades","reinforcement_baseline"})cp_put(s,resume,key,cp_container(s,WOTR_DICT,child_edges(s,field(s,resume,key))));
    if(selection_index>=0){require(!ctx_none(s,pending),"No pending pre-combat owner choice");const auto options=field(s,pending,"options");typed(s,options,WOTR_LIST,"Pending choices must be list");require(static_cast<uint64_t>(selection_index)<s.nodes[options].count,"Pre-combat choice index out of range");const auto option=s.edges[s.nodes[options].offset+selection_index];rr_set_attr(s,cursor,"selection",cp_container(s,WOTR_DICT,child_edges(s,option)));rr_set_attr(s,cursor,"pending",rr_scalar(s,WOTR_NONE));}
    while(stage<=4){
        if(truth(s,obs_attr(s,combat,"battle_over"))){cp_set_number(s,cursor,"stage",14);break;}
        if(stage==0){ce_begin(s,combat);ce_words(s,cursor,combat);}
        else if(stage==1){if(!ce_pre_attacks(s,cursor,combat,resume,random_source,battle))break;}
        else if(stage==2)ce_modifiers(s,combat);
        else if(stage==3){if(!ce_dread(s,cursor,combat))break;}
        else if(!ce_relentless(s,cursor,combat,resume,battle))break;
        cp_set_number(s,cursor,"stage",++stage);
    }
    // Original advance immediately observes a wipe before the next stage.
    if(truth(s,obs_attr(s,combat,"battle_over")))cp_set_number(s,cursor,"stage",14);
    return true;
}

crr_Result combat_pre_stage(const Snapshot& source,const void* random_source,int64_t selection_index,bool battle){
    crr_Result result;result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());result.state=std::make_unique<Snapshot>(source);
    if(combat_pre_stage_inplace(*result.state,result.random.get(),selection_index,battle))result.state=ac_compact(*result.state);return result;
}
