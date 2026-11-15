// Exact RoundCursor stages 10..13, stopping at each owner decision.
// Immediate casualty application remains a separate context-bearing operation.
void cs_remove(Snapshot& s,uint64_t list,uint64_t value){
    auto entries=child_edges(s,list);auto found=std::find(entries.begin(),entries.end(),value);
    require(found!=entries.end(),"Combat figure is missing from its roster");entries.erase(found);set_edges(s,list,entries);
}
void cs_remove_name(Snapshot& s,uint64_t list,std::string_view name){
    auto entries=child_edges(s,list);for(auto i=entries.begin();i!=entries.end();++i)if(text_is(s,*i,name)){entries.erase(i);set_edges(s,list,entries);return;}
}
uint64_t cs_leaders(const Snapshot& s,uint64_t combat,int side){return obs_attr(s,cp_attr(s,combat,cp_role(side)+"_army"),"leaders");}
bool cs_plain_fp(const Snapshot& s,uint64_t figure){return object_type(s,figure)=="game_state.Leader"&&cp_leader_side(s,figure)==0;}
void cs_drop_leadership(Snapshot& s,uint64_t combat,int side,int64_t count){
    const auto key=cp_role(side)+"_leadership";cp_set_number(s,combat,key,std::max<int64_t>(0,cp_number(s,combat,key)-count));
}
void cs_bonus(Snapshot& s,uint64_t combat){
    for(int side:{1,0}){
        const auto effect=cp_card_text(s,combat,side,"combat_effect"),role=cp_role(side),enemy=cp_role(1-side);
        const auto total=cp_number(s,combat,role+"_hits")+cp_number(s,combat,role+"_pre_combat_hits")+cp_number(s,combat,role+"_bonus_hits");
        const auto other=cp_number(s,combat,enemy+"_hits")+cp_number(s,combat,enemy+"_pre_combat_hits")+cp_number(s,combat,enemy+"_bonus_hits");
        if((cp_has(effect,"score at least one hit")||cp_has(effect,"scores at least one hit"))&&total>=1){
            if(cp_has(effect,"score one additional hit"))cp_add(s,combat,role+"_bonus_hits",1);
            if(cp_has(effect,"score two additional hits"))cp_add(s,combat,role+"_bonus_hits",2);
        }
        if(cp_has(effect,"more total hits")&&cp_has(effect,"score one additional hit")&&total>other)cp_add(s,combat,role+"_bonus_hits",1);
    }
    for(int side=0;side<2;++side){const auto effect=cp_card_text(s,combat,side,"combat_effect");
        if(cp_has(effect,"two or more hits")&&cp_has(effect,"cancel one hit")&&cp_number(s,combat,cp_role(1-side)+"_hits")>=2)cp_add(s,combat,cp_role(side)+"_hits_cancelled",1);
    }
}
uint64_t cs_option(Snapshot& s,int64_t count,const std::string* target){
    std::vector<uint64_t> entries;if(count>=0){entries.push_back(rr_string(s,"count"));entries.push_back(rr_scalar(s,WOTR_INT,count));}
    if(target){entries.push_back(rr_string(s,"target"));entries.push_back(rr_string(s,*target));}return cp_container(s,WOTR_DICT,entries);
}
bool cs_option_equal(const Snapshot& s,uint64_t a,uint64_t b){
    if(s.nodes[a].tag!=WOTR_DICT||s.nodes[b].tag!=WOTR_DICT||s.nodes[a].count!=s.nodes[b].count)return false;
    const auto entries=child_edges(s,a);for(size_t i=0;i<entries.size();i+=2){
        const auto value=field(s,b,std::string(text(s,entries[i])).c_str(),false);if(value==missing)return false;
        const auto x=entries[i+1];if(s.nodes[x].tag!=s.nodes[value].tag)return false;
        if(s.nodes[x].tag==WOTR_STRING){if(text(s,x)!=text(s,value))return false;}
        else if(s.nodes[x].tag==WOTR_INT){if(s.nodes[x].integer!=s.nodes[value].integer)return false;}else return false;
    }return true;
}
bool cs_choose(Snapshot& s,uint64_t cursor,const std::string& effect,int side,const std::vector<uint64_t>& options){
    const auto selection=obs_attr(s,cursor,"selection");if(ctx_none(s,selection)){
        rr_set_attr(s,cursor,"pending",cp_container(s,WOTR_DICT,{rr_string(s,"stage"),obs_attr(s,cursor,"stage"),rr_string(s,"effect"),rr_string(s,effect),rr_string(s,"owner"),rr_string(s,cp_short(side)),rr_string(s,"options"),cp_container(s,WOTR_LIST,options)}));return false;
    }
    bool legal=false;for(auto option:options)legal|=cs_option_equal(s,selection,option);require(legal,"Choice is not legal at observed post-reroll stage");
    rr_set_attr(s,cursor,"selection",rr_scalar(s,WOTR_NONE));const auto choices=obs_attr(s,cursor,"choices");
    cp_erase(s,choices,cp_short(side)+"_effect_count");cp_erase(s,choices,cp_short(side)+"_effect_target");
    for(const auto& item:{std::pair<const char*,const char*>{"count","_effect_count"},{"target","_effect_target"}}){const auto value=field(s,selection,item.first,false);if(value!=missing)cp_put(s,choices,cp_short(side)+item.second,value);}return true;
}
void cs_minion_bookkeeping(Snapshot& s,uint64_t combat,uint64_t victim){
    const auto state=obs_attr(s,combat,"_live_state",false);if(ctx_none(s,state))return;
    attributes(s,state,"game_state.GameState");const auto name=cp_lower(text(s,obs_attr(s,victim,"name")));
    const char* attr=cp_has(name,"saruman")?"saruman":cp_has(name,"witch")?"witch_king":cp_has(name,"mouth")?"mouth_of_sauron":nullptr;
    if(!attr)return;const auto sheet=obs_attr(s,state,attr,false);if(!ctx_none(s,sheet))rr_set_attr(s,sheet,"in_play",rr_scalar(s,WOTR_BOOL,0));
    if(std::string(attr)!="saruman"||ctx_none(s,sheet))return;
    const auto persistent=obs_attr(s,state,"persistent_cards",false);if(ctx_none(s,persistent)||object_type(s,persistent)!="game_state.PersistentCardState")return;
    std::vector<uint64_t> discard;
    for(const char* type:{"PALANTIR_OF_ORTHANC","WORMTONGUE"}){
        uint64_t found=missing;for(const char* role:{"fp_cards","sh_cards"}){for(auto card:child_edges(s,obs_attr(s,persistent,role)))if(enum_is(s,obs_attr(s,card,"card_type"),"game_state.PersistentCardType",type)){found=card;break;}if(found!=missing)break;}
        if(found!=missing)discard.push_back(found);
    }
    for(auto card:discard){
        const std::string id(text(s,obs_attr(s,card,"card_id")));bool removed=false;
        for(const char* role:{"fp_cards","sh_cards"}){const auto list=obs_attr(s,persistent,role);for(auto entry:child_edges(s,list))if(text_is(s,obs_attr(s,entry,"card_id"),id)){card=entry;cs_remove(s,list,entry);removed=true;break;}if(removed)break;}
        const auto faceup=obs_attr(s,state,"discard_face_up",false);if(faceup==missing||s.nodes[faceup].tag!=WOTR_DICT)continue;
        const std::string actual_id(text(s,obs_attr(s,card,"card_id")));const auto key=std::string(enum_is(s,obs_attr(s,card,"side",false),"game_state.Side","FREE_PEOPLES")?"fp_":"sh_")+(cp_has(actual_id,"Character")?"character":"strategy");
        const auto pile=field(s,faceup,key.c_str(),false);if(pile!=missing){auto entries=child_edges(s,pile);entries.push_back(obs_attr(s,card,"card_id"));set_edges(s,pile,entries);}
    }
}
std::vector<uint64_t> cs_targets(const Snapshot& s,uint64_t combat,int side,const std::string& name,int64_t hits){
    std::vector<uint64_t> result;const auto enemy=cs_leaders(s,combat,1-side);
    if(name=="fateful strike"){
        if(hits<1)return result;for(auto l:child_edges(s,enemy))if(object_type(s,l)=="game_state.Nazgul"){result.push_back(l);break;}
        if(hits>=2)for(auto l:child_edges(s,enemy))if(object_type(s,l)=="game_logic.Minion")result.push_back(l);
    }else if(name=="blade of westernesse"){
        if(hits>0)for(auto l:child_edges(s,enemy))if(object_type(s,l)=="game_logic.Minion")result.push_back(l);
    }else if(name=="black breath"){
        if(hits>0&&cp_number(s,combat,"_round_nazgul_present",0,true)>0){for(auto c:child_edges(s,obs_attr(s,combat,"companions_in_battle")))if(cp_number(s,c,"level")<=hits)result.push_back(c);
            for(auto l:child_edges(s,enemy))if(cs_plain_fp(s,l)){result.push_back(l);break;}}
    }return result;
}
std::string cs_target_name(const Snapshot& s,uint64_t figure){const auto type=object_type(s,figure);return type=="game_state.Nazgul"?"nazgul":type=="game_state.Leader"?"leader":std::string(text(s,obs_attr(s,figure,"name")));}
void cs_remove_companion(Snapshot& s,uint64_t combat,uint64_t companion,int army_side){
    const std::string name(text(s,obs_attr(s,companion,"name")));cs_remove(s,obs_attr(s,combat,"companions_in_battle"),companion);
    const auto leaders=cs_leaders(s,combat,army_side);for(auto leader:child_edges(s,leaders))if(object_type(s,leader)=="game_state.Companion"&&text_is(s,obs_attr(s,leader,"name"),name)){cs_remove(s,leaders,leader);break;}
    cs_drop_leadership(s,combat,army_side,cp_number(s,companion,"leadership",1,true));
}
bool cs_initiative(Snapshot& s,uint64_t cursor,uint64_t combat,bool first){
    int64_t initiative[2];for(int side=0;side<2;++side){const auto card=cp_card(s,combat,side);initiative[side]=card==missing?99:cp_number(s,card,"combat_value");}
    const bool defender_first=initiative[1]<=initiative[0];const int side=(first?defender_first:!defender_first)?1:0;
    const auto name=cp_card_text(s,combat,side,"combat_name"),role=cp_role(side),short_role=cp_short(side),enemy=cp_role(1-side);
    const auto hits=cp_number(s,combat,"_resume_"+short_role+"_reroll_hits");const auto targets=cs_targets(s,combat,side,name,hits);
    std::vector<uint64_t> options;
    if(!targets.empty()){
        if(name=="black breath")options.push_back(cs_option(s,0,nullptr));
        for(auto target:targets){const auto text=cs_target_name(s,target);options.push_back(cs_option(s,-1,&text));}
    }
    const auto incoming=std::max<int64_t>(0,cp_number(s,combat,enemy+"_hits")+cp_number(s,combat,enemy+"_bonus_hits")-cp_number(s,combat,role+"_hits_cancelled"));
    if(name=="heroic death"){
        options.push_back(cs_option(s,0,nullptr));if(incoming){
            for(auto leader:child_edges(s,cs_leaders(s,combat,side)))if(cs_plain_fp(s,leader)){const std::string target="leader";options.push_back(cs_option(s,1,&target));break;}
            for(auto companion:child_edges(s,obs_attr(s,combat,"companions_in_battle"))){const std::string target(text(s,obs_attr(s,companion,"name")));const auto level=cp_number(s,companion,"level");require(level<=1048576,"Companion level exceeds native choice bound");for(int64_t n=1;n<=std::min(incoming,level);++n)options.push_back(cs_option(s,n,&target));}
        }
    }
    std::string effect=name;std::replace(effect.begin(),effect.end(),' ','_');if(!options.empty()&&!cs_choose(s,cursor,effect,side,options))return false;
    const auto choices=obs_attr(s,cursor,"choices"),choice=field(s,choices,(short_role+"_effect_target").c_str(),false),count=field(s,choices,(short_role+"_effect_count").c_str(),false);
    const std::string selected=ctx_none(s,choice)?std::string():std::string(text(s,choice));
    const auto effect_text=cp_card_text(s,combat,side,"combat_effect");
    if(name=="fateful strike"||name=="blade of westernesse"){
        const bool active=name=="fateful strike"?(cp_has(effect_text,"re-roll")&&cp_has(effect_text,"nazg")):(cp_has(effect_text,"eliminate")&&cp_has(effect_text,"minion"));
        if(active&&!targets.empty()){
            uint64_t victim=missing;for(auto target:targets)if(cs_target_name(s,target)==selected){victim=target;break;}require(victim!=missing,"Post-reroll Minion choice is not eligible");
            cs_remove(s,cs_leaders(s,combat,1-side),victim);
            if(object_type(s,victim)=="game_logic.Minion"){const std::string victim_name(text(s,obs_attr(s,victim,"name")));cs_remove_name(s,obs_attr(s,combat,"minions_in_battle"),victim_name);cs_minion_bookkeeping(s,combat,victim);}
            else cp_set_number(s,combat,"nazgul_in_battle",std::max<int64_t>(0,cp_number(s,combat,"nazgul_in_battle")-1));
            if(name=="blade of westernesse")cp_set_number(s,combat,role+"_hits",std::max<int64_t>(0,cp_number(s,combat,role+"_hits")-1));
        }return true;
    }
    auto before=child_edges(s,cs_leaders(s,combat,0));const auto before_def=child_edges(s,cs_leaders(s,combat,1));before.insert(before.end(),before_def.begin(),before_def.end());
    if(name=="black breath"&&!targets.empty()&&(count==missing||s.nodes[count].integer!=0)&&cp_has(effect_text,"re-roll")&&cp_has(effect_text,"eliminate")&&cp_has(effect_text,"leader")){
        uint64_t victim=missing;for(auto target:targets)if(cs_target_name(s,target)==selected){victim=target;break;}require(victim!=missing,"Black Breath choice is not eligible");
        if(object_type(s,victim)=="game_state.Companion")cs_remove_companion(s,combat,victim,1-side);else{cs_remove(s,cs_leaders(s,combat,1-side),victim);cs_drop_leadership(s,combat,1-side,1);}
    }
    if(name=="heroic death"&&count!=missing&&s.nodes[count].integer>0&&incoming>0){
        const auto number=s.nodes[count].integer;uint64_t victim=missing;
        if(selected=="leader"){for(auto leader:child_edges(s,cs_leaders(s,combat,side)))if(cs_plain_fp(s,leader)){victim=leader;break;}require(victim!=missing,"Heroic Death leader is missing");cs_remove(s,cs_leaders(s,combat,side),victim);cs_drop_leadership(s,combat,side,1);cp_add(s,combat,role+"_hits_cancelled",1);}
        else{for(auto companion:child_edges(s,obs_attr(s,combat,"companions_in_battle")))if(text_is(s,obs_attr(s,companion,"name"),selected)){victim=companion;break;}require(victim!=missing,"Heroic Death companion is missing");const auto level=cp_number(s,victim,"level");cs_remove_companion(s,combat,victim,side);cp_add(s,combat,role+"_hits_cancelled",std::min({std::max<int64_t>(number,1),level,incoming}));}
    }
    auto after=child_edges(s,cs_leaders(s,combat,0));const auto after_def=child_edges(s,cs_leaders(s,combat,1));after.insert(after.end(),after_def.begin(),after_def.end());
    auto dead=obs_attr(s,combat,"_round_card_fp_dead",false);if(ctx_none(s,dead)){dead=cp_container(s,WOTR_LIST,{});rr_set_attr(s,combat,"_round_card_fp_dead",dead);}auto deaths=child_edges(s,dead);
    for(auto figure:before)if(cp_leader_side(s,figure)==0&&std::find(after.begin(),after.end(),figure)==after.end())deaths.push_back(figure);set_edges(s,dead,deaths);return true;
}
bool combat_post_stage_inplace(Snapshot& s,int64_t selection_index){
    const auto& source=s;
    attributes(source,source.root,"game_state.GameState");const auto pending_combat=obs_attr(source,source.root,"pending_combat"),resume=field(source,pending_combat,"_round_resume"),cursor=field(source,resume,"cursor");
    attributes(source,cursor,"combat_round_resume.RoundCursor");require(text_is(source,obs_attr(source,cursor,"schema"),"combat_round_cursor_v1"),"Unknown combat round cursor schema");
    auto stage=cp_number(source,cursor,"stage");require(stage>=10&&stage<=14,"Post-reroll combat requires cursor stage10..14");require(ctx_none(source,obs_attr(source,cursor,"selection")),"Post-reroll combat refuses preloaded selection");
    const auto combat=obs_attr(source,cursor,"combat"),pending=obs_attr(source,cursor,"pending");attributes(source,combat,"game_state.CombatState");
    // Real statistics are bounded by a handful of figures/dice. Reject forged
    // huge graph integers before the fixed-width arithmetic below can overflow.
    for(int side=0;side<2;++side)for(const char* suffix:{"_hits","_pre_combat_hits","_bonus_hits","_hits_cancelled","_leadership"}){
        const auto number=cp_number(source,combat,cp_role(side)+suffix);require(number>=-1048576&&number<=1048576,"Combat statistic exceeds native post-reroll range");
    }
    require(selection_index>=-1,"Post-reroll choice index must be -1 or nonnegative");
    if(selection_index>=0){require(!ctx_none(s,pending),"No pending post-reroll choice");const auto options=field(s,pending,"options");typed(s,options,WOTR_LIST,"Pending choices must be a list");require(static_cast<uint64_t>(selection_index)<s.nodes[options].count,"Post-reroll choice index out of range");const auto option=s.edges[s.nodes[options].offset+selection_index];rr_set_attr(s,cursor,"selection",cp_container(s,WOTR_DICT,child_edges(s,option)));rr_set_attr(s,cursor,"pending",rr_scalar(s,WOTR_NONE));}
    else if(!ctx_none(s,pending))return false;
    while(stage<14){
        if(truth(s,obs_attr(s,combat,"battle_over"))){cp_set_number(s,cursor,"stage",14);break;}
        if(stage==10)cs_bonus(s,combat);
        else if(stage<=12){if(!cs_initiative(s,cursor,combat,stage==11))break;}
        else{
            const auto choices=obs_attr(s,cursor,"choices");require(truth(s,field(s,choices,"defer_casualties",false)),"Immediate casualties require explicit native battle context");
            const auto attacker=std::max<int64_t>(0,cp_number(s,combat,"attacker_hits")+cp_number(s,combat,"attacker_bonus_hits")-cp_number(s,combat,"defender_hits_cancelled"));
            const auto defender=std::max<int64_t>(0,cp_number(s,combat,"defender_hits")+cp_number(s,combat,"defender_bonus_hits")-cp_number(s,combat,"attacker_hits_cancelled"));
            cp_set_number(s,combat,"attacker_final_hits",attacker);cp_set_number(s,combat,"defender_final_hits",defender);rr_set_attr(s,combat,"casualties_deferred",rr_scalar(s,WOTR_BOOL,1));cp_set_number(s,combat,"deferred_hits_on_defender",attacker);cp_set_number(s,combat,"deferred_hits_on_attacker",defender);rr_set_attr(s,combat,"battle_over",rr_scalar(s,WOTR_BOOL,1));
        }
        cp_set_number(s,cursor,"stage",++stage);
    }return true;
}

std::unique_ptr<Snapshot> combat_post_stage(const Snapshot& source,int64_t selection_index){
    auto result=std::make_unique<Snapshot>(source);if(combat_post_stage_inplace(*result,selection_index))return ac_compact(*result);return result;
}
