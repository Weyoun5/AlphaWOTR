// Movement entry effects, matching game_logic's two activation
// helpers and combat._check_settlement_capture. Called on unpublished child.
void me_nation_assign(Snapshot& s,uint64_t dict,std::string_view nation,uint64_t value){
    typed(s,dict,WOTR_DICT,"movement political dictionary required");
    auto entries=child_edges(s,dict);
    for(size_t i=0;i<entries.size();i+=2)if(enum_is(s,entries[i],"game_state.Nation",nation)){
        entries[i+1]=value;set_edges(s,dict,entries);return;
    }
    entries.push_back(mu_enum(s,"game_state.Nation",nation));entries.push_back(value);set_edges(s,dict,entries);
}
void me_activate(Snapshot& s,std::string_view nation){
    const auto active=obs_attr(s,obs_attr(s,s.root,"political"),"active");
    if(!truth(s,ctx_nation_value(s,active,nation)))me_nation_assign(s,active,nation,rr_scalar(s,WOTR_BOOL,1));
}
#include "political_activation.inc"
void me_set_member(Snapshot& s,uint64_t collection,std::string_view key,bool add){
    typed(s,collection,WOTR_SET,"movement control set required");auto entries=child_edges(s,collection);
    for(size_t i=0;i<entries.size();++i)if(text_is(s,entries[i],key)){
        if(!add){entries.erase(entries.begin()+i);set_edges(s,collection,entries);}return;
    }
    if(add){entries.push_back(rr_string(s,key));set_edges(s,collection,entries);}
}
bool me_companion_activation_available(const Snapshot& s,const Snapshot& t,std::string_view key,const std::vector<uint64_t>& companions){
    const auto meta=ctx_get(t,field(t,t.root,"board_regions"),key);
    const auto nation=ctx_get(t,field(t,t.root,"movement_region_nations"),key);
    if(meta==missing || ctx_none(t,nation) || !fp_nation(ctx_nation(t,nation)) ||
       (!truth(t,ctx_get(t,meta,"is_city")) && !truth(t,ctx_get(t,meta,"is_stronghold"))))return false;
    if(wt_active(s,ctx_nation(t,nation)) ||
       contains_string(s,ctx_side_value(s,obs_attr(s,s.root,"enemy_controlled"),1),key))return false;
    const auto names=field(t,t.root,"companion_nations");
    for(auto companion:companions)if(object_type(s,companion)=="game_state.Companion"){
        const auto allowed=ctx_get(t,names,text(s,obs_attr(s,companion,"name")));
        if(allowed!=missing && t.nodes[allowed].tag==WOTR_NONE)return true;
        if(allowed!=missing)for(auto n:child_edges(t,allowed))
            if(ctx_nation(t,n)==ctx_nation(t,nation))return true;
    }
    return false;
}
void movement_entry_effects(Snapshot& s,const Snapshot& t,std::string_view key,int side,const std::vector<uint64_t>& arriving_leaders){
    const auto meta=ctx_get(t,field(t,t.root,"board_regions"),key);
    require(meta!=missing,"movement metadata missing region");
    const auto nation=ctx_get(t,field(t,t.root,"movement_region_nations"),key);
    const auto controls=obs_attr(s,s.root,"enemy_controlled");
    const auto sh=ctx_side_value(s,controls,1);
    // Enemy entry activates before companion checks and before capture.
    if(!ctx_none(t,nation) && (fp_nation(ctx_nation(t,nation))?0:1)!=side)wt_activate_if_inactive(s,ctx_nation(t,nation),"attack");
    if(!ctx_none(t,nation) && fp_nation(ctx_nation(t,nation)) &&
       (truth(t,ctx_get(t,meta,"is_city")) || truth(t,ctx_get(t,meta,"is_stronghold"))) && !contains_string(s,sh,key)){
        const auto names=field(t,t.root,"companion_nations");
        for(auto leader:arriving_leaders)if(object_type(s,leader)=="game_state.Companion"){
            const auto allowed=ctx_get(t,names,text(s,obs_attr(s,leader,"name")));
            bool activates=allowed!=missing && t.nodes[allowed].tag==WOTR_NONE;
            if(allowed!=missing && t.nodes[allowed].tag!=WOTR_NONE)
                for(auto n:child_edges(t,allowed))activates|=ctx_nation(t,n)==ctx_nation(t,nation);
            if(activates){wt_activate_if_inactive(s,ctx_nation(t,nation),"companion_move",key);break;}
        }
    }
}
void movement_capture_effects(Snapshot& s,const Snapshot& t,std::string_view key,int side){
    const auto meta=ctx_get(t,field(t,t.root,"board_regions"),key);
    if(!truth(t,ctx_get(t,meta,"is_city")) && !truth(t,ctx_get(t,meta,"is_stronghold")) && !truth(t,ctx_get(t,meta,"is_town")))return;
    const bool fp=contains_string(t,field(t,t.root,"fp_region_keys"),key);
    const bool sh=contains_string(t,field(t,t.root,"shadow_region_keys"),key);
    if(!fp && !sh)return;
    const auto controls=obs_attr(s,s.root,"enemy_controlled");
    const auto own=ctx_side_value(s,controls,side),other=ctx_side_value(s,controls,1-side);
    if((side==1 && fp) || (side==0 && sh)){
        const bool first=!contains_string(s,own,key);
        me_set_member(s,own,key,true);me_set_member(s,other,key,false);
        if(first){
            const auto nation=ctx_get(t,field(t,t.root,"movement_region_nations"),key);
            if(ctx_none(t,nation))return;
            const auto name=ctx_nation(t,nation);
            const auto political=obs_attr(s,s.root,"political"),boxes=obs_attr(s,political,"boxes");
            const auto value=ctx_nation_value(s,boxes,name);
            const auto current=ctx_none(s,value)?0:typed(s,value,WOTR_INT,"movement political box must be integer").integer;
            require(current>=0 && current<=3,"movement political box out of range");
            if(current<3){
                wt_activate_if_inactive(s,name,"enemy_entry",key);
                if(current+1<3 || wt_active(s,name))me_nation_assign(s,boxes,name,rr_scalar(s,WOTR_INT,current+1));
            }
        }
    }else me_set_member(s,other,key,false);
}
