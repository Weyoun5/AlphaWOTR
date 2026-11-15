// combat_round_resume stages5..8: primary dice, reroll setup, Mighty/Anduril.
// Stops at an unresolved owner choice; stage9 belongs to combat_reroll_stage.
std::string cp_lower(std::string_view source){std::string out(source);for(auto& c:out)if(c>='A'&&c<='Z')c=char(c-'A'+'a');return out;}
bool cp_has(const std::string& value,std::string_view part){return value.find(part)!=std::string::npos;}
std::string cp_role(int side){return side==0?"attacker":"defender";}
std::string cp_short(int side){return side==0?"atk":"def";}
uint64_t cp_attr(const Snapshot& s,uint64_t object,const std::string& key,bool required=true){return obs_attr(s,object,key.c_str(),required);}
int64_t cp_number(const Snapshot& s,uint64_t object,const std::string& key,int64_t fallback=0,bool optional=false){
    const auto value=cp_attr(s,object,key,!optional);if(value==missing)return fallback;
    const auto& node=s.nodes[value];require(node.tag==WOTR_INT || node.tag==WOTR_BOOL,"Primary combat requires integer statistics");return node.integer;
}
void cp_set_number(Snapshot& s,uint64_t object,const std::string& key,int64_t value){rr_set_attr(s,object,key,rr_scalar(s,WOTR_INT,value));}
void cp_add(Snapshot& s,uint64_t object,const std::string& key,int64_t value){
    const auto old=cp_number(s,object,key);
    require(value>=0?old<=std::numeric_limits<int64_t>::max()-value:old>=std::numeric_limits<int64_t>::min()-value,"Combat statistic exceeds int64 storage");
    cp_set_number(s,object,key,old+value);
}
uint64_t cp_card(const Snapshot& s,uint64_t combat,int side){
    if(truth(s,cp_attr(s,combat,cp_role(side)+"_card_cancelled")))return missing;
    const auto card=cp_attr(s,combat,cp_role(side)+"_card");return ctx_none(s,card)?missing:card;
}
std::string cp_card_text(const Snapshot& s,uint64_t combat,int side,const char* key){
    const auto card=cp_card(s,combat,side);return card==missing?std::string():cp_lower(text(s,obs_attr(s,card,key)));
}
int cp_leader_side(const Snapshot& s,uint64_t leader){
    const auto type=object_type(s,leader);
    if(type=="game_state.Nazgul" || type=="game_logic.Minion")return 1;
    if(type=="game_state.Companion")return 0;
    const auto nation=obs_attr(s,leader,"nation",false);
    return nation==missing?0:(fp_nation(validate_nation_name(enum_member(s,nation,"game_state.Nation")))?0:1);
}
int64_t cp_strength(const Snapshot& s,uint64_t army,uint64_t cancelled){
    attributes(s,army,"game_state.Army");const auto units=obs_attr(s,army,"units"),leaders=obs_attr(s,army,"leaders");
    const auto count=typed(s,units,WOTR_LIST,"Combat army units must be list").count;
    typed(s,leaders,WOTR_LIST,"Combat army leaders must be list");
    int64_t strength=static_cast<int64_t>(std::min<uint64_t>(count,5));
    const int own=count?ctx_unit_side(s,s.edges[s.nodes[units].offset]):-1;
    for(auto leader:child_edges(s,leaders)){
        if(own>=0 && cp_leader_side(s,leader)!=own)continue;
        if(object_type(s,leader)!="game_state.Companion" || leader==cancelled)continue;
        const auto name=text(s,obs_attr(s,leader,"name"));
        if(name=="Gandalf the Grey" || name=="Strider" || name=="Legolas" || name=="Gimli" || name=="Boromir" || name=="Aragorn")++strength;
    }
    return std::min<int64_t>(5,strength);
}
void cp_primary_side(Snapshot& s,uint64_t cursor,uint64_t combat,void* rng,int side);
void cp_primary(Snapshot& s,uint64_t cursor,uint64_t combat,void* rng){
    const bool siege=truth(s,obs_attr(s,combat,"is_siege"));
    const bool city=truth(s,obs_attr(s,combat,"city_defense")) && cp_number(s,combat,"siege_round")==1;
    cp_set_number(s,combat,"_resume_atk_threshold",siege||city?6:5);cp_set_number(s,combat,"_resume_def_threshold",5);
    for(int side:{1,0}){
        const auto effect=cp_card_text(s,combat,side,"combat_effect");
        if(cp_has(effect,"result of") && cp_has(effect,"1") && cp_has(effect,"shadow army"))
            rr_set_attr(s,combat,cp_role(1-side)+"_confusion",rr_scalar(s,WOTR_BOOL,1));
    }
    cp_primary_side(s,cursor,combat,rng,0);
}
void cp_primary_side(Snapshot& s,uint64_t cursor,uint64_t combat,void* rng,int side){
    if(crr_observed(s,cursor,"combat",cp_role(side)))return;
    const auto cancelled=obs_attr(s,combat,"_wop_cancelled",false);
    const int cancelled_side=truth(s,obs_attr(s,combat,"_wop_cancelled_is_attacker",false))?0:1;
    int64_t hits=0,ones=0;
        const auto role=cp_role(side),short_role=cp_short(side);
        int64_t count=cp_strength(s,cp_attr(s,combat,role+"_army"),side==cancelled_side?cancelled:missing);
        const auto reduction=cp_number(s,combat,role+"_dice_reduction",0,true);
        require(reduction>=-1048576 && reduction<=1048576,"Combat reduction exceeds supported dice domain");count-=reduction;
        const auto cap=cp_attr(s,combat,role+"_dice_cap");if(!ctx_none(s,cap)){
            require(s.nodes[cap].tag==WOTR_INT || s.nodes[cap].tag==WOTR_BOOL,"Combat die cap must be integer or None");count=std::min(count,s.nodes[cap].integer);
        }
        count=std::max<int64_t>(1,count);require(count<=1048576,"Combat dice count exceeds supported batch bound");
        std::vector<uint32_t> rolls(static_cast<size_t>(count));
        const auto threshold=cp_number(s,combat,"_resume_"+short_role+"_threshold"),modifier=cp_number(s,combat,role+"_dice_mod");
        cf_d6(rng,rolls.data(),rolls.size(),threshold,modifier,true);
        for(auto face:rolls){if(crr_hit(face,threshold,modifier))++hits;if(face==1)++ones;}
        rr_set_attr(s,combat,role+"_combat_rolls",crr_list(s,rolls.data(),rolls.size()));
    if(truth(s,cp_attr(s,combat,role+"_confusion")))cp_add(s,combat,cp_role(1-side)+"_hits",ones);
    cp_add(s,combat,role+"_hits",hits);
    crr_mark(s,obs_attr(s,cursor,"observed_rolls"),"combat");
    crr_mark(s,obs_attr(s,cursor,"observed_roll_sides"),"combat",role);
}
void cp_setup(Snapshot& s,uint64_t combat){
    const bool shine=!ctx_none(s,obs_attr(s,combat,"_shine_companion",false));
    const auto nazgul=shine?0:cp_number(s,combat,"nazgul_in_battle");
    for(int side=0;side<2;++side){
        const auto effect=cp_card_text(s,combat,side,"combat_effect");
        if(cp_has(effect,"leader re-roll is cancelled") && nazgul>=cp_number(s,combat,cp_role(1-side)+"_leadership")){
            rr_set_attr(s,combat,cp_role(1-side)+"_reroll_cancelled",rr_scalar(s,WOTR_BOOL,1));
            rr_set_attr(s,combat,"_foul_stench_cancel_"+cp_short(1-side),rr_scalar(s,WOTR_BOOL,1));
        }
    }
    for(int side=0;side<2;++side){
        const auto effect=cp_card_text(s,combat,side,"combat_effect");
        if(cp_has(effect,"subtract 1") && cp_has(effect,"nazg") && cp_has(effect,"leader re-roll") && !shine && cp_number(s,combat,"nazgul_in_battle")>=1){
            cp_add(s,combat,"nazgul_in_battle",-1);cp_add(s,combat,cp_role(side)+"_leadership",-1);cp_add(s,combat,cp_role(side)+"_reroll_mod",1);
        }
    }
}
std::vector<uint64_t> cp_companions(const Snapshot& s,uint64_t combat){
    const auto companions=obs_attr(s,combat,"companions_in_battle");typed(s,companions,WOTR_LIST,"Battle companions must be list");
    const auto shined=obs_attr(s,combat,"_shined_companion",false),cancelled=obs_attr(s,combat,"_wop_cancelled",false);
    std::vector<uint64_t> eligible;for(auto companion:child_edges(s,companions))if(companion!=shined && companion!=cancelled)eligible.push_back(companion);return eligible;
}
int64_t cp_misses(const Snapshot& s,uint64_t combat,int side){
    const auto role=cp_role(side);const auto rolls=cp_attr(s,combat,role+"_combat_rolls");
    typed(s,rolls,WOTR_LIST,"Combat rolls must be list");const auto modifier=cp_number(s,combat,role+"_dice_mod"),threshold=cp_number(s,combat,"_resume_"+cp_short(side)+"_threshold");
    require(modifier<=std::numeric_limits<int64_t>::max()-6,"Combat modifier exceeds supported integer range");int64_t misses=0;
    // Deliberately preserve current Might/Anduril eligibility arithmetic:
    // unlike leader_reroll it does not special-case a modified natural one.
    for(auto value:child_edges(s,rolls)){const auto face=typed(s,value,WOTR_INT,"Combat face must be integer").integer;
        require(face>=1&&face<=6,"Combat face must be 1..6");if(face!=6 && face+modifier<threshold)++misses;}
    return misses;
}
uint64_t cp_container(Snapshot& s,uint32_t tag,const std::vector<uint64_t>& entries){const auto node=rr_scalar(s,tag);set_edges(s,node,entries);return node;}
void cp_erase(Snapshot& s,uint64_t dictionary,std::string_view key){
    typed(s,dictionary,WOTR_DICT,"Combat choices must be dictionary");auto old=child_edges(s,dictionary);std::vector<uint64_t> fresh;fresh.reserve(old.size());
    for(size_t i=0;i<old.size();i+=2)if(!text_is(s,old[i],key)){fresh.push_back(old[i]);fresh.push_back(old[i+1]);}set_edges(s,dictionary,fresh);
}
void cp_put(Snapshot& s,uint64_t dictionary,std::string_view key,uint64_t value){
    auto entries=child_edges(s,dictionary);for(size_t i=0;i<entries.size();i+=2)if(text_is(s,entries[i],key)){entries[i+1]=value;set_edges(s,dictionary,entries);return;}
    entries.push_back(rr_string(s,key));entries.push_back(value);set_edges(s,dictionary,entries);
}
bool cp_mighty(Snapshot& s,uint64_t cursor,uint64_t combat,int side){
    const auto role=cp_role(side),short_role=cp_short(side);const auto card=cp_card(s,combat,side);
    const bool blocked=truth(s,cp_attr(s,combat,"_foul_stench_cancel_"+short_role,false));
    const auto misses=card==missing||blocked?0:cp_misses(s,combat,side);
    auto companions=card==missing||blocked?std::vector<uint64_t>():cp_companions(s,combat);
    if(card!=missing && !blocked && misses && cp_card_text(s,combat,side,"combat_name")=="mighty attack" && !companions.empty()){
        const auto selection=obs_attr(s,cursor,"selection");std::vector<uint64_t> options;
        for(auto companion:companions)options.push_back(cp_container(s,WOTR_DICT,{rr_string(s,"target"),obs_attr(s,companion,"name")}));
        if(ctx_none(s,selection)){
            const auto pending=cp_container(s,WOTR_DICT,{rr_string(s,"stage"),obs_attr(s,cursor,"stage"),rr_string(s,"effect"),rr_string(s,"mighty_attack"),
                rr_string(s,"owner"),rr_string(s,short_role),rr_string(s,"options"),cp_container(s,WOTR_LIST,options)});
            rr_set_attr(s,cursor,"pending",pending);return false;
        }
        const auto target=field(s,selection,"target");const auto target_name=text(s,target);bool eligible=false;
        for(auto companion:companions)eligible|=text_is(s,obs_attr(s,companion,"name"),target_name);
        require(eligible,"Choice is not legal at observed Mighty stage");
        rr_set_attr(s,cursor,"selection",rr_scalar(s,WOTR_NONE));
        const auto choices=obs_attr(s,cursor,"choices");cp_erase(s,choices,short_role+"_effect_count");cp_erase(s,choices,short_role+"_effect_target");cp_put(s,choices,short_role+"_effect_target",target);
    }
    int64_t automatic=0;
    if(card!=missing && !blocked && misses && !companions.empty()){
        const auto effect=cp_card_text(s,combat,side,"combat_effect");
        if(cp_has(effect,"forfeit") && cp_has(effect,"leadership") && cp_has(effect,"change") && cp_has(effect,"missed")){
            uint64_t payer=missing;
            if(cp_has(effect,"aragorn") || cp_has(effect,"strider")){
                for(auto companion:companions){const auto name=cp_lower(text(s,obs_attr(s,companion,"name")));if(cp_has(name,"aragorn")||cp_has(name,"strider")){payer=companion;break;}}
                if(payer!=missing){const bool crowned=truth(s,obs_attr(s,payer,"crowned",false)) || cp_number(s,payer,"leadership",1,true)>=2;automatic=std::min<int64_t>(crowned?2:1,misses);}
            }else{
                const auto choices=obs_attr(s,cursor,"choices"),target=field(s,choices,(short_role+"_effect_target").c_str(),false);
                if(!ctx_none(s,target)){
                    const auto name=text(s,target);if(!name.empty() && name!="leader")for(auto companion:companions)if(text_is(s,obs_attr(s,companion,"name"),name)){payer=companion;break;}
                }else for(auto companion:companions)if(payer==missing || cp_number(s,companion,"leadership",1,true)<cp_number(s,payer,"leadership",1,true))payer=companion;
                require(payer!=missing,"Mighty Attack payer is not currently eligible");automatic=1;
            }
            if(payer!=missing){
                const auto payment=std::max<int64_t>(0,cp_number(s,payer,"leadership",1,true)),leadership=cp_number(s,combat,role+"_leadership");
                cp_set_number(s,combat,role+"_leadership",leadership<=payment?0:leadership-payment);
                auto paid=obs_attr(s,combat,"_round_paid_leadership",false);if(ctx_none(s,paid)){paid=rr_scalar(s,WOTR_LIST);rr_set_attr(s,combat,"_round_paid_leadership",paid);}
                typed(s,paid,WOTR_LIST,"Paid leadership must be list");auto entries=child_edges(s,paid);entries.push_back(obs_attr(s,payer,"name"));set_edges(s,paid,entries);
                auto forfeited=obs_attr(s,combat,"_leadership_forfeited",false);if(ctx_none(s,forfeited)){forfeited=rr_scalar(s,WOTR_SET);rr_set_attr(s,combat,"_leadership_forfeited",forfeited);}
                // Copy name before rr_string can grow its owning byte vector.
                const std::string name(text(s,obs_attr(s,payer,"name")));crr_mark(s,forfeited,name);
            }
        }
    }
    const auto paid=obs_attr(s,combat,"_round_paid_leadership",false);
    rr_set_attr(s,cursor,"paid_leadership_targets",cp_container(s,WOTR_LIST,paid==missing?std::vector<uint64_t>():child_edges(s,paid)));
    cp_set_number(s,combat,"_resume_"+short_role+"_auto",automatic);cp_add(s,combat,role+"_hits",automatic);return true;
}
bool combat_primary_stage_inplace(Snapshot& changed,void* random_source,const char* target){
    const auto& source=changed;
    attributes(source,source.root,"game_state.GameState");const auto pc=obs_attr(source,source.root,"pending_combat"),resume=field(source,pc,"_round_resume"),cursor=field(source,resume,"cursor");
    attributes(source,cursor,"combat_round_resume.RoundCursor");require(text_is(source,obs_attr(source,cursor,"schema"),"combat_round_cursor_v1"),"Unknown combat round cursor schema");
    const auto stage=cp_number(source,cursor,"stage");require(stage>=5&&stage<=8,"Primary combat requires cursor stage5..8");
    const auto combat=obs_attr(source,cursor,"combat");attributes(source,combat,"game_state.CombatState");require(!truth(source,obs_attr(source,combat,"battle_over")),"Primary combat rejects completed battle");
    require(ctx_none(source,obs_attr(source,cursor,"selection")),"Primary combat rejects preloaded selection");
    const auto pending=obs_attr(source,cursor,"pending");
    if(target){require(!ctx_none(source,pending),"No pending Mighty choice for supplied target");
        bool allowed=false;for(auto option:child_edges(source,field(source,pending,"options"))){
            if(source.nodes[option].tag==WOTR_DICT && source.nodes[option].count==2 && text_is(source,field(source,option,"target",false),target))allowed=true;
        }require(allowed,"No matching pending Mighty target");
    }
    if(!ctx_none(source,pending) && !target)return false;
    if(target){rr_set_attr(changed,cursor,"selection",cp_container(changed,WOTR_DICT,{rr_string(changed,"target"),rr_string(changed,target)}));rr_set_attr(changed,cursor,"pending",rr_scalar(changed,WOTR_NONE));}
    for(int current=static_cast<int>(stage);current<=8;++current){
        if(current==5)cp_primary(changed,cursor,combat,random_source);
        else if(current==6)cp_setup(changed,combat);
        else {
            if(current==8){
                combat_reroll_side_inplace(changed,cursor,combat,random_source,0);
                cp_primary_side(changed,cursor,combat,random_source,1);
            }
            if(!cp_mighty(changed,cursor,combat,current-7))break;
        }
        cp_set_number(changed,cursor,"stage",current+1);
    }
    return true;
}

crr_Result combat_primary_stage(const Snapshot& source,const void* random_source,const char* target){
    crr_Result result;result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());result.state=std::make_unique<Snapshot>(source);
    if(combat_primary_stage_inplace(*result.state,result.random.get(),target))result.state=ac_compact(*result.state);return result;
}
