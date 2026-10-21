// Native continuation for the pinned Python search turn boundary.
// The continuation owns all scheduling metadata outside encoded GameState.
enum class sd_Stage {Discard,AutoDeclare,Declare,BeginTail,Tail,Guide,Eye,Roll,Finish,Done};
struct sd_Driver {
    std::unique_ptr<Snapshot> state,raw,policy;
    std::shared_ptr<const Snapshot> tables;
    std::unique_ptr<void,void(*)(void*)> random{nullptr,wotr_random_destroy};
    am_Builder annotations;
    std::vector<uint64_t> annotation_rows;
    std::string forced_reason;
    std::vector<std::string> reasons;
    sd_Stage stage=sd_Stage::Discard;
    int side=-1,hand_steps=0,tail_steps=0;
    uint64_t pre_unused=0;
    sd_Driver(){annotations.out.root=0;}
    explicit sd_Driver(const am_Builder& events):annotations(events){}
};
std::unique_ptr<sd_Driver> sd_clone(const sd_Driver& before){
    auto out=std::make_unique<sd_Driver>(before.annotations);
    out->state=std::make_unique<Snapshot>(*before.state);out->tables=before.tables;
    out->random.reset(wotr_random_clone(before.random.get()));if(!out->random)throw std::invalid_argument(wotr_random_last_error());
    if(before.raw)out->raw=std::make_unique<Snapshot>(*before.raw);
    if(before.policy)out->policy=std::make_unique<Snapshot>(*before.policy);
    out->annotation_rows=before.annotation_rows;
    out->forced_reason=before.forced_reason;out->reasons=before.reasons;
    out->stage=before.stage;out->side=before.side;out->hand_steps=before.hand_steps;out->tail_steps=before.tail_steps;out->pre_unused=before.pre_unused;
    return out;
}
void sd_append(sd_Driver& d,const Snapshot* events){
    if(!events)return;auto rows=fm_import(d.annotations.out,*events);
    d.annotation_rows.insert(d.annotation_rows.end(),rows.begin(),rows.end());
}
std::unique_ptr<Snapshot> sd_events(const sd_Driver& d){
    auto out=std::make_unique<Snapshot>(d.annotations.out);
    out->root=cp_container(*out,WOTR_LIST,d.annotation_rows);return ac_compact(*out);
}
// Python float str: shortest correctly-rounded digits, fixed for exponent
// [-4,16), otherwise scientific with two exponent digits. No inference/RNG.
std::string sd_number_string(const Snapshot& s,uint64_t node){
    const auto& value=s.nodes[node];
    if(value.tag==WOTR_INT)return std::to_string(value.integer);
    if(value.tag==WOTR_BOOL)return value.integer?"True":"False";
    require(value.tag==WOTR_FLOAT,"Search victory corruption must be numeric");
    char buffer[128];const auto converted=std::to_chars(buffer,buffer+sizeof(buffer),value.real,std::chars_format::scientific);
    require(converted.ec==std::errc{},"Cannot format search victory corruption");
    std::string text(buffer,converted.ptr);const auto e=text.find('e');if(e==std::string::npos)return text;
    const int exponent=std::stoi(text.substr(e+1));std::string digits=text.substr(0,e),sign;
    if(!digits.empty()&&digits[0]=='-'){sign="-";digits.erase(0,1);}digits.erase(std::remove(digits.begin(),digits.end(),'.'),digits.end());
    if(exponent>=-4&&exponent<16){
        const int split=exponent+1;
        if(split<=0)return sign+"0."+std::string(-split,'0')+digits;
        if(split>=static_cast<int>(digits.size()))return sign+digits+std::string(split-digits.size(),'0')+".0";
        return sign+digits.substr(0,split)+"."+digits.substr(split);
    }
    std::string suffix=std::to_string(exponent<0?-exponent:exponent);if(suffix.size()<2)suffix="0"+suffix;
    return sign+digits.substr(0,1)+(digits.size()>1?"."+digits.substr(1):"")+"e"+(exponent<0?"-":"+")+suffix;
}
void sd_set_terminal(sd_Driver& d,const TerminalResult& result){
    require(result.winner>=0,"Search terminal marker requires an actual winner");auto& s=*d.state;
    std::string kind=result.kind==1?"military":result.kind==2?"ring_destroyed":"ring_corruption",message;
    if(result.kind==1)message=(result.winner==0?"Free Peoples":"Shadow")+std::string(" military victory! (")+std::to_string(result.winner==0?result.fp_vp:result.sh_vp)+" VP)";
    else if(result.kind==2)message="The Ring is destroyed! Free Peoples win!";
    else message="Ring-bearers corrupted! ("+sd_number_string(s,obs_attr(s,obs_attr(s,s.root,"fellowship"),"corruption"))+" corruption)";
    rr_set_attr(s,s.root,"_search_endturn_victory",ev_dict(s,{
        {"victory",rr_string(s,result.winner==0?"free_peoples":"shadow")},{"victory_type",rr_string(s,kind)},
        {"message",rr_string(s,message)},{"free_peoples_vp",rr_scalar(s,WOTR_INT,result.fp_vp)},{"shadow_vp",rr_scalar(s,WOTR_INT,result.sh_vp)}}));
}
void sd_inner_begin(Snapshot& s){if(truth(s,obs_attr(s,s.root,"_declare_phase",false)))rr_set_attr(s,s.root,"_declare_phase",rr_scalar(s,WOTR_BOOL));}
void sd_declare_inner(sd_Driver& d,std::string_view destination){
    auto& s=*d.state;const auto& t=*d.tables;const std::string key(sanitize_name(destination));
    const auto fs=obs_attr(s,s.root,"fellowship");
    require(truth(s,obs_attr(s,fs,"hidden"))&&!truth(s,obs_attr(s,fs,"mordor_entered")),"Search declaration requires hidden Fellowship outside Mordor");
    require(fd_reachable(s,fs).count(key),"Search declaration is unreachable");
    const auto route=fr_path(s,t,key,true);sd_inner_begin(s);fd_declare_inner(s,t,key);
    fbr_register_route(s,t,d.random.get(),route,d.annotations,d.annotation_rows);
}
bool sd_auto_gateway(sd_Driver& d){
    auto& s=*d.state;const auto fs=obs_attr(s,s.root,"fellowship");
    if(truth(s,obs_attr(s,fs,"mordor_entered"))||!truth(s,obs_attr(s,fs,"hidden"))||ht_integer(s,fs,"progress")<=0)return false;
    const auto reachable=fd_reachable(s,fs);
    for(const char* gateway:{"morannon","minas morgul"}){
        if(!reachable.count(gateway))continue;
        sd_declare_inner(d,gateway);
        auto& b=d.annotations;const auto row=b.row(-1,"auto_declare_gateway",{b.str("region"),b.str(gateway)});
        const auto fields=child_edges(b.out,row);
        d.annotation_rows.push_back(b.container(WOTR_TUPLE,{fields[0],fields[1],fields[2],b.str("FP")}));
        return true;
    }
    return false;
}
bool sd_has_declaration(const Snapshot& s){
    const auto fs=obs_attr(s,s.root,"fellowship");
    if(!truth(s,obs_attr(s,fs,"hidden"))||truth(s,obs_attr(s,fs,"mordor_entered"))||ht_integer(s,fs,"progress")<0)return false;
    return !fd_reachable(s,fs).empty();
}
void sd_offer(sd_Driver& d,int side){
    d.side=side;d.raw=native_full_actions(*d.state,*d.tables,side);
    auto projected=project_native_ai_actions(*d.state,*d.raw,side);
    d.policy=std::move(projected.actions);d.forced_reason=std::move(projected.forced_reason);d.reasons=std::move(projected.reasons);
    require(truth(*d.policy,d.policy->root),"Required search-turn selector has no legal action");
}
void sd_tail(sd_Driver& d,bool begin){
    auto tail=fellowship_tail(*d.state,*d.tables,d.random.get(),begin);
    sd_append(d,tail.events.get());d.state=std::move(tail.state);d.random=std::move(tail.random);
}
void sd_pump(sd_Driver& d){
    while(true){
        auto& s=*d.state;const auto& t=*d.tables;
        switch(d.stage){
        case sd_Stage::Discard:
            if(d.hand_steps<4&&!ctx_none(s,obs_attr(s,s.root,"pending_discard",false))){sd_offer(d,dc_owner(s));return;}
            d.stage=sd_Stage::AutoDeclare;break;
        case sd_Stage::AutoDeclare:
            if(!sd_auto_gateway(d)&&sd_has_declaration(s)){
                rr_set_attr(s,s.root,"_declare_phase",rr_scalar(s,WOTR_BOOL,1));d.stage=sd_Stage::Declare;sd_offer(d,0);return;
            }
            d.stage=sd_Stage::BeginTail;break;
        case sd_Stage::Declare:throw std::invalid_argument("Search declaration selection is still pending");
        case sd_Stage::BeginTail:sd_tail(d,true);d.stage=sd_Stage::Tail;break;
        case sd_Stage::Tail:
            if(truth(s,obs_attr(s,s.root,"_fellowship_tail_phase",false))){
                require(d.tail_steps<128,"Fellowship tail exceeded 128 decisions");const auto actor=native_pending_actor(s,-1);
                require(actor.side>=0,"Unresolved Fellowship tail has no decision owner");sd_offer(d,actor.side);return;
            }
            d.stage=terminal_should_end(check_victory(s,t),false)?sd_Stage::Finish:sd_Stage::Guide;break;
        case sd_Stage::Guide:
            if(!ctx_none(s,obs_attr(s,s.root,"pending_guide_change"))){sd_offer(d,0);return;}
            d.stage=sd_Stage::Eye;break;
        case sd_Stage::Eye:{
            bool unused=false;for(auto die:child_edges(s,pa_pool(s,1)))unused|=!truth(s,obs_attr(s,die,"used"));
            if(unused){rr_set_attr(s,s.root,"_eye_allocation_phase",rr_scalar(s,WOTR_BOOL,1));sd_offer(d,1);return;}
            d.stage=sd_Stage::Roll;break;
        }
        case sd_Stage::Roll:{auto rolled=sr_roll_current(s,d.random.get());d.state=std::move(rolled.state);d.random=std::move(rolled.random);d.stage=sd_Stage::Finish;break;}
        case sd_Stage::Finish:{
            // Source outer finally preserves the current inner marker unless
            // this enclosing action spent a previously rolled unused die.
            const int marker=ctx_side(s,obs_attr(s,s.root,"_battle_resolved_attacker",false));
            cau_finish(s,t,d.random.get(),marker,d.annotations,d.annotation_rows);
            if(ea_unused_rolled(s)<d.pre_unused)rr_set_attr(s,s.root,"_battle_resolved_attacker",rr_scalar(s,WOTR_NONE));
            d.raw.reset();d.policy.reset();d.side=-1;d.stage=sd_Stage::Done;return;
        }
        case sd_Stage::Done:return;
        }
    }
}
std::unique_ptr<sd_Driver> sd_start(const Snapshot& before,const Snapshot& tables,const void* random){
    require(ctx_none(before,obs_attr(before,before.root,"pending_combat")),"Search end_turn requires no pending combat");
    hpr_card_gate(before); // No unported persistent upkeep may be bypassed by automatic declaration.
    auto d=std::make_unique<sd_Driver>();d->state=action_clone(before);d->tables=std::make_shared<const Snapshot>(tables);
    d->random.reset(wotr_random_clone(random));if(!d->random)throw std::invalid_argument(wotr_random_last_error());
    auto& s=*d->state;d->pre_unused=ea_unused_rolled(s);
    rr_set_attr(s,s.root,"_casualty_operations",rr_scalar(s,WOTR_LIST));cau_dedupe(s);rr_set_attr(s,s.root,"_ring_free_side",rr_scalar(s,WOTR_NONE));sd_inner_begin(s);
    if(st_safe_next_turn(s,tables,d->annotations,d->annotation_rows)){d->stage=sd_Stage::Finish;}
    else{
        const auto old=obs_attr(s,s.root,"_search_turn_crossings",false);const int64_t count=old==missing?0:typed(s,old,WOTR_INT,"Search turn crossings must be integer").integer;
        require(count<std::numeric_limits<int64_t>::max(),"Search turn crossing counter overflow");rr_set_attr(s,s.root,"_search_turn_crossings",rr_scalar(s,WOTR_INT,count+1));
    }
    sd_pump(*d);return d;
}
std::unique_ptr<sd_Driver> sd_select(const sd_Driver& before,int64_t index){
    require(before.stage!=sd_Stage::Done&&before.policy,"Search turn is not waiting for a selection");
    const auto options=child_edges(*before.policy,before.policy->root);
    if(index==-1&&before.stage==sd_Stage::Discard&&!options.empty())index=0; // source selector-None fallback
    require(index>=0&&static_cast<uint64_t>(index)<options.size(),"Search turn selection index is invalid");
    auto d=sd_clone(before);const auto& menu=*d->policy;const auto row=options[index];const auto fields=child_edges(menu,row);const auto kind=text(menu,fields[1]);const auto params=fields[2];
    auto& s=*d->state;
    switch(d->stage){
    case sd_Stage::Discard:
        require(kind=="discard_card","Required discard selection has wrong action type");sd_inner_begin(s);dc_discard_inner(s,d->side,ad_text(menu,params,"card_id"));++d->hand_steps;break;
    case sd_Stage::Declare:
        rr_set_attr(s,s.root,"_declare_phase",rr_scalar(s,WOTR_BOOL));
        require(kind=="fellowship_declare"||kind=="pass"||kind=="declare_decline","Required Fellowship declaration selection is missing");
        if(kind=="fellowship_declare")sd_declare_inner(*d,ad_text(menu,params,"region"));d->stage=sd_Stage::BeginTail;break;
    case sd_Stage::Tail:{
        auto action=apply_native_random_descriptor(s,*d->tables,menu,d->random.get(),d->side,index);
        sd_append(*d,action.events.get());d->state=std::move(action.state);d->random=std::move(action.random);++d->tail_steps;sd_tail(*d,false);break;
    }
    case sd_Stage::Guide:
        require(kind=="choose_guide","Required Fellowship guide selection is missing");sd_inner_begin(s);gc_choose_inner(s,ad_text(menu,params,"companion"));
        require(ctx_none(s,obs_attr(s,s.root,"pending_guide_change")),"Fellowship guide selection did not resolve");d->stage=sd_Stage::Eye;break;
    case sd_Stage::Eye:
        rr_set_attr(s,s.root,"_eye_allocation_phase",rr_scalar(s,WOTR_BOOL));require(kind=="eye_allocate","Required Shadow Hunt allocation selection is missing");
        sd_inner_begin(s);ea_allocate_inner(s,ad_int(menu,params,"count"));d->stage=sd_Stage::Roll;break;
    default:throw std::invalid_argument("Search turn is not at a selectable stage");
    }
    d->raw.reset();d->policy.reset();d->side=-1;d->forced_reason.clear();d->reasons.clear();sd_pump(*d);return d;
}

int sd_turn_start_actor(const Snapshot& s){
    const auto pending=native_pending_actor(s,0);if(pending.side>=0)return pending.side;
    for(int side:{0,1})if(ar_has_unused(s,side))return side;return -1;
}
