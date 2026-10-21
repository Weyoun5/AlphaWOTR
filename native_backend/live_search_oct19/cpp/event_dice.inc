// Event dice transformations and the learned Lidless Eye count continuation.
bool ed_supported(const Snapshot& t,uint64_t card){
    if(card==missing)return false;const auto name=obs_attr(t,card,"card_name");
    return text_is(t,name,"Mirror of Galadriel") || text_is(t,name,"The Lidless Eye");
}
std::vector<std::vector<std::string>> ed_selections(const Snapshot& s){
    std::vector<std::string> faces;for(auto die:child_edges(s,pa_pool(s,1))){const auto face=obs_attr(s,die,"current");if(!truth(s,obs_attr(s,die,"used"))&&!ctx_none(s,face)&&!enum_is(s,face,"game_state.ActionDieFace","EYE"))faces.emplace_back(enum_member(s,face,"game_state.ActionDieFace"));}
    std::sort(faces.begin(),faces.end());std::vector<std::vector<std::string>> result{{}};
    for(size_t count=1;count<=std::min<size_t>(3,faces.size());++count){std::set<std::vector<std::string>> unique;std::vector<std::string> choice;
        auto visit=[&](auto&& self,size_t start){if(choice.size()==count){unique.insert(choice);return;}for(size_t i=start;i<faces.size();++i){choice.push_back(faces[i]);self(self,i+1);choice.pop_back();}};visit(visit,0);result.insert(result.end(),unique.begin(),unique.end());}return result;
}
void ed_lidless(Snapshot& s,const std::vector<std::string>& selected){
    const auto legal=ed_selections(s);require(std::find(legal.begin(),legal.end(),selected)!=legal.end(),"Lidless face selection unavailable");
    const auto box=obs_attr(s,s.root,"hunt_box");
    for(const auto& face:selected){
        uint64_t die=missing;for(auto candidate:child_edges(s,pa_pool(s,1)))if(!truth(s,obs_attr(s,candidate,"used"))&&enum_is(s,obs_attr(s,candidate,"current"),"game_state.ActionDieFace",face)){die=candidate;break;}
        require(die!=missing,"Lidless selected face vanished");
        rr_set_attr(s,die,"current",mu_enum(s,"game_state.ActionDieFace","EYE"));rr_set_attr(s,die,"used",rr_scalar(s,WOTR_BOOL,1));
        auto entries=child_edges(s,box);entries.push_back(die);set_edges(s,box,entries);
    }
}
void ed_commit(Snapshot& s,int32_t side,std::string_view id,std::string_view deck,std::string_view face){
    const auto hand=tc_hand(s,side);auto cards=child_edges(s,hand);const auto it=std::find_if(cards.begin(),cards.end(),[&](uint64_t c){return text_is(s,c,id);});
    require(it!=cards.end(),"event commit card absent from hand");const auto played=*it;cards.erase(it);set_edges(s,hand,cards);
    const std::string key=(side==0?"fp_":"sh_")+std::string(deck);const auto pile=field(s,obs_attr(s,s.root,"discard_face_up"),key.c_str());
    auto discarded=child_edges(s,pile);discarded.push_back(played);set_edges(s,pile,discarded);tc_open_hand_limit_discard(s);
    if(side==0 && !contains_string(s,tc_hand(s,side),id))ev_gandalf_after_event(s,id,deck,face);
    epa_after_event(s,side,id,deck,face);
}
void apply_dice_event_action_inplace(Snapshot& s,const Snapshot& t,void* rng,int side,int64_t index,std::string_view id,bool free_ents,am_Builder& ef_events,std::vector<uint64_t>& ef_annotations){
    const auto& source=s;ef_validate_free(source,t,side,index,id,free_ents);
    const int ef_marker=ctx_side(source,obs_attr(source,source.root,"_battle_resolved_attacker",false));const auto ef_unused=ef_unused_dice(source);

    require(side==0 || side==1,"dice event actor invalid");quiet_action_gate(source,t,false,false,false,false,false,false,false,false,false,false,(false||free_ents),(side==0 && ev_card(t,id)!=missing && text_is(t,obs_attr(t,ev_card(t,id),"card_name"),"Mirror of Galadriel") || (side==1 && ev_card(t,id)!=missing && text_is(t,obs_attr(t,ev_card(t,id),"card_name"),"The Lidless Eye"))));ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained) || ctx_side(source,retained)==side,"dice event belongs to retained Ring actor");
    const auto card=ev_card(t,id);require(card!=missing && ctx_side(t,obs_attr(t,card,"side"))==side && ed_supported(t,card),"unsupported or wrong-side dice event");
    require(contains_string(source,tc_hand(source,side),id),"dice event absent from hand");require(ev_requirement(source,t,card),"dice event requirement false");
    const auto die=free_ents?missing:ev_payment(source,t,side,index,card);const std::string face(free_ents?std::string_view{}:enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace"));
    const std::string name(text(t,obs_attr(t,card,"card_name"))),deck=ev_lower(text(t,obs_attr(t,card,"deck")));
    
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    if(!free_ents)rr_set_attr(s,ev_payment(s,t,side,index,card),"used",rr_scalar(s,WOTR_BOOL,1));
    if(name=="The Lidless Eye"){
        const auto selections=ed_selections(s); // payment precedes choice enumeration
        if(selections.size()>=2){
            const auto options=rr_scalar(s,WOTR_LIST);std::vector<uint64_t> list;for(const auto& selected:selections){std::vector<uint64_t> names;for(const auto& name:selected)names.push_back(rr_string(s,name));const auto face_list=rr_scalar(s,WOTR_LIST);set_edges(s,face_list,names);list.push_back(ev_dict(s,{{"dice_count",rr_scalar(s,WOTR_INT,selected.size())},{"dice_faces",face_list}}));}set_edges(s,options,list);
            rr_set_attr(s,s.root,"pending_card_option",ev_dict(s,{{"card_id",rr_string(s,id)},{"card_name",rr_string(s,name)},{"side",rr_string(s,"sh")},{"decider",rr_string(s,"sh")},
                {"options",options},{"die_face",ef_pending_face_node(s,face)},{"deck",rr_string(s,deck)}}));
            tc_open_hand_limit_discard(s);rr_set_attr(s,s.root,"_battle_resolved_attacker",rr_scalar(s,WOTR_NONE));return;
        }
        ed_lidless(s,selections.front());
    }else{
        for(auto d:child_edges(s,pa_pool(s,0)))if(!truth(s,obs_attr(s,d,"used")) && enum_is(s,obs_attr(s,d,"current"),"game_state.ActionDieFace","CHARACTER")){
            rr_set_attr(s,d,"current",mu_enum(s,"game_state.ActionDieFace","WILL_OF_WEST"));break;
        }
        const auto fs=obs_attr(s,s.root,"fellowship");
        if(glb_fellowship_key(s,fs)=="lorien" && !contains_string(s,ctx_side_value(s,obs_attr(s,s.root,"enemy_controlled"),1),"lorien"))
            rr_set_attr(s,fs,"corruption",rr_scalar(s,WOTR_INT,std::max<int64_t>(0,typed(s,obs_attr(s,fs,"corruption"),WOTR_INT,"corruption must be integer").integer-1)));
    }
    ed_commit(s,side,id,deck,face);rr_set_attr(s,s.root,"_battle_resolved_attacker",rr_scalar(s,WOTR_NONE));

    if(free_ents)ef_finish_free(s,t,rng,ef_marker,ef_unused,ef_events,ef_annotations);
}
std::unique_ptr<Snapshot> apply_dice_event_action(const Snapshot& source,const Snapshot& t,int32_t side,int64_t index,std::string_view id){
    require(side==0 || side==1,"dice event actor invalid");quiet_action_gate(source,t,false,false,false,false,false,false,false,false,false,false,false,(side==0 && ev_card(t,id)!=missing && text_is(t,obs_attr(t,ev_card(t,id),"card_name"),"Mirror of Galadriel") || (side==1 && ev_card(t,id)!=missing && text_is(t,obs_attr(t,ev_card(t,id),"card_name"),"The Lidless Eye"))));ring_waiver(source,0);ring_waiver(source,1);
    const auto retained=obs_attr(source,source.root,"_ring_free_side",false);require(ctx_none(source,retained) || ctx_side(source,retained)==side,"dice event belongs to retained Ring actor");
    const auto card=ev_card(t,id);require(card!=missing && ctx_side(t,obs_attr(t,card,"side"))==side && ed_supported(t,card),"unsupported or wrong-side dice event");
    require(contains_string(source,tc_hand(source,side),id),"dice event absent from hand");require(ev_requirement(source,t,card),"dice event requirement false");
    const auto die=ev_payment(source,t,side,index,card);const std::string face(enum_member(source,obs_attr(source,die,"current"),"game_state.ActionDieFace"));
    const std::string name(text(t,obs_attr(t,card,"card_name"))),deck=ev_lower(text(t,obs_attr(t,card,"deck")));
    
    auto child=action_clone(source);am_Builder ef_events;std::vector<uint64_t> ef_annotations;
    apply_dice_event_action_inplace(*child,t,nullptr,side,index,id,false,ef_events,ef_annotations);return child;
}
uint64_t ed_pending(const Snapshot& s){
    const auto p=obs_attr(s,s.root,"pending_card_option",false);
    require(!ctx_none(s,p) && text_is(s,field(s,p,"card_name",false),"The Lidless Eye") && text_is(s,field(s,p,"side",false),"sh") && text_is(s,field(s,p,"decider",false),"sh"),"no Shadow Lidless Eye choice pending");
    const auto options=child_edges(s,field(s,p,"options"));const auto legal=ed_selections(s);require(options.size()>=2&&options.size()==legal.size(),"Lidless Eye choices stale");
    for(size_t i=0;i<options.size();++i){require(typed(s,options[i],WOTR_DICT,"Lidless option must be dict").count==4,"Lidless option has extra fields");require(typed(s,field(s,options[i],"dice_count"),WOTR_INT,"Lidless count must be integer").integer==static_cast<int64_t>(legal[i].size()),"Lidless option count invalid");const auto faces=child_edges(s,field(s,options[i],"dice_faces"));require(faces.size()==legal[i].size(),"Lidless faces length mismatch");for(size_t j=0;j<faces.size();++j)require(text_is(s,faces[j],legal[i][j]),"Lidless face order invalid");}
    return p;
}
std::unique_ptr<Snapshot> native_dice_event_choice_actions(const Snapshot& source,int32_t side){
    require(side==1,"Lidless Eye choice belongs to Shadow");ed_pending(source);
    auto out=std::make_unique<Snapshot>(source);auto& s=*out;const auto p=obs_attr(s,s.root,"pending_card_option");std::vector<uint64_t> rows;int64_t index=0;
    for(auto option:child_edges(s,field(s,p,"options"))){
        const auto params=ev_dict(s,{{"idx",rr_scalar(s,WOTR_INT,index++)},{"card_id",field(s,p,"card_id")},{"card_name",field(s,p,"card_name")},{"side",field(s,p,"side")},{"choices",option}});
        const auto row=rr_scalar(s,WOTR_TUPLE);set_edges(s,row,{rr_scalar(s,WOTR_INT,-1),rr_string(s,"card_option"),params});rows.push_back(row);
    }
    const auto root=rr_scalar(s,WOTR_LIST);set_edges(s,root,rows);s.root=root;return ac_compact(s);
}
std::unique_ptr<Snapshot> apply_dice_event_choice(const Snapshot& source,const Snapshot& t,int32_t side,int64_t option){
    require(side==1,"Lidless Eye choice belongs to Shadow");const auto p=ed_pending(source);const auto options=child_edges(source,field(source,p,"options"));
    require(option>=0 && static_cast<uint64_t>(option)<options.size(),"Lidless Eye choice index out of range");
    const auto selected=ed_selections(source)[option];
    const std::string id(text(source,field(source,p,"card_id"))),deck(text(source,field(source,p,"deck"))),face(ef_pending_face(source,p));
    const auto card=ev_card(t,id);require(card!=missing && text_is(t,obs_attr(t,card,"card_name"),"The Lidless Eye") && ctx_side(t,obs_attr(t,card,"side"))==1,"Lidless pending card ID mismatch");
    require(contains_string(source,tc_hand(source,1),id),"Lidless pending card absent from hand");
    require(deck=="character" && (face=="CHARACTER" || face=="EVENT" || face=="WILL_OF_WEST"),"Lidless pending original funding metadata invalid");
    auto child=action_clone(source);auto& s=*child;rr_set_attr(s,s.root,"pending_card_option",rr_scalar(s,WOTR_NONE));quiet_action_gate(s,t,false,false,false,false,false,false,false,false,false,false,false,true);ring_waiver(s,0);ring_waiver(s,1);
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));
    ed_lidless(s,selected);ed_commit(s,1,id,deck,face);rr_set_attr(s,s.root,"_battle_resolved_attacker",rr_scalar(s,WOTR_NONE));return child;
}
