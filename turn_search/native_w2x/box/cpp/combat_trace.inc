// Copy newly published combat roll annotations into explicit native trace rows.
// This is the same once-only ledger used by _emit_round_annotations.
void ctr_annotations(const Snapshot& s,uint64_t combat,uint64_t payload,am_Builder& events,std::vector<uint64_t>& rows){
    const auto args=child_edges(s,field(s,payload,"args"));const auto round=obs_attr(s,combat,"siege_round");const auto emitted=field(s,payload,"annotated_rolls",false);
    const auto context=std::string(truth(s,args[4])?"siege ":"")+"battle at "+std::string(text(s,args[1]))+" r"+std::to_string(s.nodes[round].integer);
    auto emit=[&](int role,uint64_t rolls,const std::string& label){if(!truth(s,rolls))return;
        if(!ctx_none(s,emitted))for(auto key:child_edges(s,emitted)){const auto parts=child_edges(s,key);if(parts.size()!=3||!tail_equal(s,parts[0],round)||!text_is(s,parts[1],label))continue;const auto faces=child_edges(s,parts[2]),actual=child_edges(s,rolls);if(faces.size()!=actual.size())continue;bool same=true;for(size_t i=0;i<faces.size();++i)same&=tail_equal(s,faces[i],actual[i]);if(same)return;}
        std::vector<uint64_t> faces;for(auto face:child_edges(s,rolls))faces.push_back(events.integer(typed(s,face,WOTR_INT,"Combat face must be integer").integer));const bool shadow=enum_is(s,cp_attr(s,combat,cp_role(role)+"_side"),"game_state.Side","SHADOW");
        rows.push_back(events.row(-1,"dice_roll",{events.str("faces"),events.container(WOTR_LIST,faces),events.str("ctx"),events.str(label),events.str("side"),events.str(shadow?"SH":"FP")}));
    };
    for(int role=0;role<2;++role){emit(role,cp_attr(s,combat,cp_role(role)+"_combat_rolls"),context+" "+cp_role(role));emit(role,cp_attr(s,combat,cp_role(role)+"_rerolls"),context+" "+cp_role(role)+" re-roll");}
    for(int role=0;role<2;++role){const auto card=cp_attr(s,combat,cp_role(role)+"_card");const auto name=ctx_none(s,card)?missing:obs_attr(s,card,"combat_name",false);emit(role,cp_attr(s,combat,cp_role(role)+"_pre_combat_rolls"),(truth(s,name)?std::string(text(s,name)):"pre-combat")+" pre-combat attack");}
}
