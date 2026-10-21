// Shared casualty/persistent bookkeeping and paused army publication.
uint64_t cbp_fp_counts(Snapshot& s,uint64_t nation){
    auto box=obs_attr(s,s.root,"fp_casualties",false);if(ctx_none(s,box)){box=cp_container(s,WOTR_DICT,{});rr_set_attr(s,s.root,"fp_casualties",box);}
    const std::string name(enum_member(s,nation,"game_state.Nation"));auto counts=ctx_nation_value(s,box,name);
    if(counts==missing){counts=cp_container(s,WOTR_DICT,{rr_string(s,"regular"),rr_scalar(s,WOTR_INT,0),rr_string(s,"elite"),rr_scalar(s,WOTR_INT,0),rr_string(s,"leader"),rr_scalar(s,WOTR_INT,0)});ce_enum_put(s,box,nation,counts);}return counts;
}
void cbp_book(Snapshot& s,uint64_t nation,const char* kind){
    const auto counts=cbp_fp_counts(s,nation),old=field(s,counts,kind,false);const auto number=old==missing?0:typed(s,old,WOTR_INT,"Casualty box counts must be integer").integer;
    require(number>=0&&number<1048576,"Casualty box count exceeds native bound");cp_put(s,counts,kind,rr_scalar(s,WOTR_INT,number+1));
}
uint64_t cbp_remove_card(Snapshot& s,uint64_t persistent,uint64_t card){
    const std::string id(text(s,obs_attr(s,card,"card_id")));for(const char* side:{"fp_cards","sh_cards"}){const auto list=obs_attr(s,persistent,side);for(auto candidate:child_edges(s,list))if(text_is(s,obs_attr(s,candidate,"card_id"),id)){cs_remove(s,list,candidate);return candidate;}}return card;
}
void cbp_discard_table(Snapshot& s,uint64_t card){
    const auto id=obs_attr(s,card,"card_id");const std::string name(text(s,id));const auto key=std::string(enum_is(s,obs_attr(s,card,"side",false),"game_state.Side","FREE_PEOPLES")?"fp_":"sh_")+(cp_has(name,"Character")?"character":"strategy");
    const auto faceup=obs_attr(s,s.root,"discard_face_up",false);if(faceup==missing||s.nodes[faceup].tag!=WOTR_DICT)return;const auto pile=field(s,faceup,key.c_str(),false);if(pile==missing)return;auto entries=child_edges(s,pile);entries.push_back(id);set_edges(s,pile,entries);
}
std::string cbp_strip(std::string value){
    // Companion requirement metadata is ASCII names and comma whitespace.
    const auto first=value.find_first_not_of(" \t\n\r\v\f");if(first==std::string::npos)return {};const auto last=value.find_last_not_of(" \t\n\r\v\f");return value.substr(first,last-first+1);
}
void cbp_wwsat_resolve(Snapshot& s,void* rng,uint64_t source=missing,uint64_t card_id=missing){
    const auto forced=obs_attr(s,s.root,"_wwsat_forced",false);
    if(ctx_none(s,source) && !ctx_none(s,forced)){source=field(s,forced,"source",false);if(ctx_none(s,card_id))card_id=field(s,forced,"card_id",false);}
    rr_set_attr(s,s.root,"_wwsat_forced",rr_scalar(s,WOTR_NONE));
    if(text_is(s,source,"decline"))return;
    require(ctx_none(s,source)||text_is(s,source,"hand")||text_is(s,source,"table"),"Invalid Worn source");
    const auto persistent=obs_attr(s,s.root,"persistent_cards",false),player=ctx_side_value(s,obs_attr(s,s.root,"hands"),0),hand=obs_attr(s,player,"hand");std::vector<uint64_t> chars,table;
    for(auto id:child_edges(s,hand)){const auto name=text(s,id);if(name.find("_Character_")!=std::string_view::npos || (name.substr(0,3)=="HU_" && name.find("_CHAR_")!=std::string_view::npos))chars.push_back(id);}
    if(!ctx_none(s,persistent))for(auto card:child_edges(s,obs_attr(s,persistent,"fp_cards")))if(text(s,obs_attr(s,card,"card_id")).find("Character")!=std::string_view::npos)table.push_back(card);
    require(!text_is(s,source,"table")||!table.empty(),"Worn table target unavailable");
    require(!text_is(s,source,"hand")||!chars.empty(),"Worn hand source unavailable");
    const bool use_table=text_is(s,source,"table") || (ctx_none(s,source) && chars.empty());
    if(use_table && !table.empty()){
        uint64_t target=missing;
        if(!ctx_none(s,card_id)){
            for(auto card:table)if(text(s,obs_attr(s,card,"card_id"))==text(s,card_id)){target=card;break;}
            require(target!=missing,"Worn named table target unavailable");
        }else{
            require(!text_is(s,source,"table")||table.size()==1,"Worn requires a named table target");
            target=table.front();
        }
        cbp_discard_table(s,cbp_remove_card(s,persistent,target));return;
    }
    if(!chars.empty()){const auto index=hf_choice(rng,chars.size(),false);const auto victim=chars[index];cs_remove(s,hand,victim);const auto discard=obs_attr(s,player,"discarded_face_down");auto entries=child_edges(s,discard);entries.push_back(victim);set_edges(s,discard,entries);}
}
void cbp_companion_death(Snapshot& s,void* rng,bool casualty=true){
    const auto persistent=obs_attr(s,s.root,"persistent_cards",false);if(ctx_none(s,persistent))return;
    attributes(s,persistent,"game_state.PersistentCardState");const auto companions=child_edges(s,obs_attr(s,obs_attr(s,s.root,"fellowship"),"companions"));std::vector<uint64_t> gone;
    for(auto card:child_edges(s,obs_attr(s,persistent,"fp_cards"))){const auto required=obs_attr(s,card,"requires_companion");if(!truth(s,required))continue;const std::string names(text(s,required));bool present=false;size_t begin=0;
        while(begin<=names.size()){const auto comma=names.find(',',begin);const auto name=cbp_strip(names.substr(begin,comma==std::string::npos?comma:comma-begin));for(auto companion:companions)present|=text_is(s,obs_attr(s,companion,"name"),name);if(comma==std::string::npos)break;begin=comma+1;}if(!present)gone.push_back(card);
    }
    for(auto card:gone)cbp_discard_table(s,cbp_remove_card(s,persistent,card));
    if(!casualty)return;
    bool worn=false;for(const char* side:{"fp_cards","sh_cards"})for(auto card:child_edges(s,obs_attr(s,persistent,side)))worn|=enum_is(s,obs_attr(s,card,"card_type"),"game_state.PersistentCardType","WORN_WITH_SORROW");if(!worn)return;
    const auto hand_owner=ctx_side_value(s,obs_attr(s,s.root,"hands"),0),hand=obs_attr(s,hand_owner,"hand");std::vector<uint64_t> chars,table;
    for(auto cid:child_edges(s,hand)){const auto name=text(s,cid);if(name.find("_Character_")!=std::string_view::npos||(name.substr(0,3)=="HU_"&&name.find("_CHAR_")!=std::string_view::npos))chars.push_back(cid);}
    for(auto card:child_edges(s,obs_attr(s,persistent,"fp_cards")))if(text(s,obs_attr(s,card,"card_id")).find("Character")!=std::string_view::npos)table.push_back(card);
    if(chars.empty()&&table.empty())return;
    const auto forced=obs_attr(s,s.root,"_wwsat_forced",false);
    if(ctx_none(s,forced)){auto owed=obs_attr(s,s.root,"_wwsat_owed",false);if(ctx_none(s,owed)){owed=cp_container(s,WOTR_LIST,{});rr_set_attr(s,s.root,"_wwsat_owed",owed);}auto entries=child_edges(s,owed);entries.push_back(rr_scalar(s,WOTR_BOOL,1));set_edges(s,owed,entries);return;}
    cbp_wwsat_resolve(s,rng);
}
void cbp_publish_casualties(Snapshot& s,uint64_t combat,uint64_t payload,void* rng){
    const auto published=field(s,payload,"published");typed(s,published,WOTR_LIST,"Published casualties must be a two-item list");auto counts=child_edges(s,published);require(counts.size()==2,"Published casualties require both sides");
    for(int side=0;side<2;++side){const auto role=cp_role(side);const auto list=cp_attr(s,combat,role+"_casualties");const auto casualties=child_edges(s,list);const auto start=typed(s,counts[side],WOTR_INT,"Published count must be integer").integer;require(start>=0&&static_cast<uint64_t>(start)<=casualties.size(),"Published casualties exceed the casualty roster");
        if(enum_is(s,cp_attr(s,combat,role+"_side"),"game_state.Side","FREE_PEOPLES")){if(ctx_none(s,obs_attr(s,s.root,"fp_casualties",false)))rr_set_attr(s,s.root,"fp_casualties",cp_container(s,WOTR_DICT,{}));for(size_t i=static_cast<size_t>(start);i<casualties.size();++i){const auto unit=casualties[i];cbp_book(s,obs_attr(s,unit,"nation"),enum_is(s,obs_attr(s,unit,"kind"),"game_state.UnitKind","REGULAR")?"regular":"elite");}}
        counts[side]=rr_scalar(s,WOTR_INT,casualties.size());set_edges(s,published,counts);
    }
    const auto dead_node=obs_attr(s,combat,"_round_card_fp_dead",false);const auto dead=dead_node==missing?std::vector<uint64_t>{}:child_edges(s,dead_node);const auto old=field(s,payload,"published_card_figures",false);const auto start=old==missing?0:typed(s,old,WOTR_INT,"Published figure count must be integer").integer;require(start>=0&&static_cast<uint64_t>(start)<=dead.size(),"Published figures exceed dead roster");
    // Original book_eliminated_characters executes all Minion bookkeeping,
    // then all plain leader bookkeeping, then every Companion trigger.
    for(size_t i=static_cast<size_t>(start);i<dead.size();++i)if(object_type(s,dead[i])=="game_logic.Minion")cs_minion_bookkeeping(s,combat,dead[i]);
    for(size_t i=static_cast<size_t>(start);i<dead.size();++i)if(cs_plain_fp(s,dead[i]))cbp_book(s,obs_attr(s,dead[i],"nation"),"leader");
    for(size_t i=static_cast<size_t>(start);i<dead.size();++i)if(object_type(s,dead[i])=="game_state.Companion")cbp_companion_death(s,rng,false);
    cp_put(s,payload,"published_card_figures",rr_scalar(s,WOTR_INT,dead.size()));
}
void cbp_join(Snapshot& s,uint64_t parent,const char* attribute,const std::vector<uint64_t>& first,const std::vector<uint64_t>& second,const std::vector<uint64_t>& rear){
    auto entries=first;entries.insert(entries.end(),second.begin(),second.end());entries.insert(entries.end(),rear.begin(),rear.end());rr_set_attr(s,parent,attribute,cp_container(s,WOTR_LIST,entries));
}
void cbp_publish_armies(Snapshot& s,uint64_t combat){
    const auto parent=obs_attr(s,combat,"coloc_parent",false);if(!ctx_none(s,parent)){
        const auto rear=child_edges(s,obs_attr(s,combat,"coloc_rear"));require(rear.size()==2,"Co-located rearguard requires unit and leader lists");
        for(int i=0;i<2;++i){const char* name=i==0?"units":"leaders";cbp_join(s,parent,name,child_edges(s,obs_attr(s,obs_attr(s,combat,"defender_army"),name)),child_edges(s,obs_attr(s,obs_attr(s,combat,"attacker_army"),name)),child_edges(s,rear[i]));}
    }else for(int side=0;side<2;++side){const auto prefix=side==0?"away_atk":"away_def";const auto away=cp_attr(s,combat,std::string(prefix)+"_parent",false);if(ctx_none(s,away))continue;const auto rear=child_edges(s,cp_attr(s,combat,std::string(prefix)+"_rest"));require(rear.size()==2,"Away rearguard requires unit and leader lists");for(int i=0;i<2;++i){const char* name=i==0?"units":"leaders";cbp_join(s,away,name,child_edges(s,obs_attr(s,cp_attr(s,combat,cp_role(side)+"_army"),name)),{},child_edges(s,rear[i]));}}
}
crr_Result combat_publish_paused(const Snapshot& source,const void* random_source){
    attributes(source,source.root,"game_state.GameState");const auto payload=field(source,obs_attr(source,source.root,"pending_combat"),"_round_resume"),cursor=field(source,payload,"cursor"),combat=obs_attr(source,cursor,"combat");
    require(text_is(source,field(source,payload,"schema"),"combat_battle_resume_v1"),"Paused publication requires battle resume schema");require(obs_attr(source,combat,"_live_state",false)==source.root,"Paused publication requires its owned live GameState");
    crr_Result result;result.state=std::make_unique<Snapshot>(source);result.random.reset(wotr_random_clone(random_source));if(!result.random)throw std::invalid_argument(wotr_random_last_error());
    cbp_publish_armies(*result.state,combat);cbp_publish_casualties(*result.state,combat,payload,result.random.get());result.state=ac_compact(*result.state);return result;
}
