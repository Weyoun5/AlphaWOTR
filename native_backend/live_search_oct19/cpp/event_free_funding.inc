// Private funding for a validated Ents direct choice. No snapshot field or
// external descriptor can turn on this in-place entry parameter by itself.
void ef_validate_free(const Snapshot& s,const Snapshot& t,int side,int64_t index,std::string_view id,bool granted){
    if(!granted)return;
    require(side==0&&index==-1,"Free event requires its FP Ents grant");
    const auto card=ev_card(t,id);require(card!=missing&&ctx_side(t,obs_attr(t,card,"side"))==0&&ev_lower(text(t,obs_attr(t,card,"deck")))=="character"&&id.find("_Character_")!=std::string_view::npos,"Ents grant requires an FP Character card");
    require(ctx_none(s,obs_attr(s,s.root,"pending_card_option",false)),"Nested event requires a consumed Ents choice");
}
uint64_t ef_pending_face_node(Snapshot& s,std::string_view face){return face.empty()?rr_scalar(s,WOTR_NONE):rr_string(s,face);}
bool ef_pending_is_free(const Snapshot& s,uint64_t pending){
    if(ctx_none(s,pending))return false;const auto face=field(s,pending,"die_face",false);
    if(face==missing||s.nodes[face].tag!=WOTR_NONE)return false;
    require(text_is(s,field(s,pending,"side"),"fp")&&text_is(s,field(s,pending,"deck"),"character")&&text(s,field(s,pending,"card_id")).find("_Character_")!=std::string_view::npos,"None event funding requires FP Character metadata");return true;
}
std::string_view ef_pending_face(const Snapshot& s,uint64_t pending){
    const auto node=field(s,pending,"die_face");
    if(s.nodes[node].tag==WOTR_NONE){require(ef_pending_is_free(s,pending),"Invalid free event funding");return {};}
    const auto face=text(s,node);require(!face.empty(),"Empty event funding face is not None");return face;
}
int64_t ef_unused_dice(const Snapshot& s){
    int64_t count=0;for(int side=0;side<2;++side)for(auto die:child_edges(s,pa_pool(s,side)))count+=!truth(s,obs_attr(s,die,"used"))&&!ctx_none(s,obs_attr(s,die,"current"));return count;
}
int ef_final_marker(const Snapshot& s,int marker,int64_t before){return ef_unused_dice(s)<before?-1:marker;}
bool eea_supported(const Snapshot&,uint64_t);
bool ef_ents_queued_grant(const Snapshot& s,const Snapshot& t,uint64_t queued){
    if(ctx_none(s,queued))return false;
    const auto source=field(s,queued,"source_card_id",false),counter=obs_attr(s,s.root,"_ents_chain_n",false);
    if(ctx_none(s,source)||ctx_none(s,counter)||s.nodes[counter].tag!=WOTR_INT||s.nodes[counter].integer<1||s.nodes[counter].integer>3)return false;
    const auto id=text(s,source);return eea_supported(t,ev_card(t,id))&&text_is(s,field(s,queued,"deck",false),"character")&&contains_string(s,field(s,obs_attr(s,s.root,"discard_face_up"),"fp_character"),id);
}
// Defined after common action upkeep; invokes that exact tail once.
void ef_finish_free(Snapshot& s,const Snapshot& t,void* rng,int marker,int64_t before,am_Builder& events,std::vector<uint64_t>& rows);
