// Fixed AI policy projection, not legality or action generation. Mirrors the
// pinned ai_policies.project_ai_actions and retains descriptor row aliases.
// Provenance denotes forcedness only when the caller supplied a COMPLETE menu.
struct NativePolicyProjection {
    std::unique_ptr<Snapshot> actions;
    std::string forced_reason;
    std::vector<std::string> reasons;
};
constexpr const char* ap_recruit="recruit_elite_if_available";
constexpr const char* ap_draw="bonus_draw_always_take";
constexpr const char* ap_victim="combat_victim_priority";
constexpr const char* ap_mighty="mighty_payer_lowest_leadership";

struct ap_Frozen {
    uint32_t tag=WOTR_NONE;
    int64_t integer=0;
    double real=0;
    std::string value;
    std::vector<ap_Frozen> children;
};
bool ap_equal(const ap_Frozen& a,const ap_Frozen& b){
    const bool an=a.tag==WOTR_INT || a.tag==WOTR_BOOL || a.tag==WOTR_FLOAT;
    const bool bn=b.tag==WOTR_INT || b.tag==WOTR_BOOL || b.tag==WOTR_FLOAT;
    if(an && bn){
        if(a.tag!=WOTR_FLOAT && b.tag!=WOTR_FLOAT)return a.integer==b.integer;
        if(a.tag==WOTR_FLOAT && b.tag==WOTR_FLOAT)return a.real==b.real;
        const auto f=a.tag==WOTR_FLOAT?a.real:b.real;
        const auto i=a.tag==WOTR_FLOAT?b.integer:a.integer;
        return std::isfinite(f) && f>=-9223372036854775808.0 && f<9223372036854775808.0 &&
            std::trunc(f)==f && static_cast<int64_t>(f)==i;
    }
    if(a.tag!=b.tag || a.value!=b.value || a.children.size()!=b.children.size())return false;
    for(size_t i=0;i<a.children.size();++i)if(!ap_equal(a.children[i],b.children[i]))return false;
    return true;
}
// repr is used only by the source's sorted(set, key=repr). Legal descriptor
// sets use strings/integers/tuples. Refuse unsupported repr types explicitly.
std::string ap_repr(const ap_Frozen& f){
    if(f.tag==WOTR_NONE)return "None";
    if(f.tag==WOTR_BOOL)return f.integer?"True":"False";
    if(f.tag==WOTR_INT)return std::to_string(f.integer);
    if(f.tag==WOTR_BIGINT)return f.value;
    if(f.tag==WOTR_STRING){
        const char quote=f.value.find('\'')!=std::string::npos && f.value.find('"')==std::string::npos?'"':'\'';
        std::string out(1,quote);constexpr const char* hex="0123456789abcdef";
        for(unsigned char c:f.value){
            if(c=='\\' || c==quote){out.push_back('\\');out.push_back(c);}
            else if(c=='\n')out+="\\n";else if(c=='\r')out+="\\r";else if(c=='\t')out+="\\t";
            else if(c<32 || c==127){out+="\\x";out+=hex[c>>4];out+=hex[c&15];}
            else {require(c<128,"native policy set repr requires ASCII strings");out.push_back(c);}
        }
        out.push_back(quote);return out;
    }
    if(f.tag==WOTR_TUPLE){
        std::string out="(";for(size_t i=0;i<f.children.size();++i){if(i)out+=", ";out+=ap_repr(f.children[i]);}
        if(f.children.size()==1)out+=",";return out+")";
    }
    throw std::invalid_argument("native policy set repr type is not implemented");
}
ap_Frozen ap_freeze(const Snapshot& s,uint64_t node,size_t depth=0){
    require(depth<200,"native policy choices are cyclic or nested too deeply");
    ap_Frozen out;
    if(node==missing)return out;
    const auto& n=s.nodes[node];out.tag=n.tag;out.integer=n.integer;out.real=n.real;
    if(n.tag==WOTR_NONE || n.tag==WOTR_BOOL || n.tag==WOTR_INT)return out;
    if(n.tag==WOTR_FLOAT){require(std::isfinite(n.real),"native policy requires finite choice numbers");return out;}
    if(n.tag==WOTR_STRING || n.tag==WOTR_BYTES){
        out.value.assign(reinterpret_cast<const char*>(s.bytes.data()+n.offset),n.count);return out;
    }
    if(n.tag==WOTR_ENUM){out.value=std::string(text(s,s.edges[n.offset]))+":"+std::string(text(s,s.edges[n.offset+1]));return out;}
    if(n.tag==WOTR_DICT){
        std::vector<std::pair<std::string,uint64_t>> ordered;
        for(uint64_t i=0;i<n.count;i+=2){
            // Engine choice keys are strings; never silently reinterpret an
            // unsupported object key using a different str() representation.
            ordered.emplace_back(std::string(text(s,s.edges[n.offset+i])),s.edges[n.offset+i+1]);
        }
        std::sort(ordered.begin(),ordered.end(),[](const auto& a,const auto& b){return a.first<b.first;});
        out.tag=WOTR_TUPLE;
        for(const auto& item:ordered){
            ap_Frozen pair,key;pair.tag=WOTR_TUPLE;key.tag=WOTR_STRING;key.value=item.first;
            pair.children.push_back(std::move(key));pair.children.push_back(ap_freeze(s,item.second,depth+1));out.children.push_back(std::move(pair));
        }
        return out;
    }
    if(n.tag==WOTR_LIST || n.tag==WOTR_TUPLE || n.tag==WOTR_SET){
        out.tag=WOTR_TUPLE;
        for(uint64_t i=0;i<n.count;++i)out.children.push_back(ap_freeze(s,s.edges[n.offset+i],depth+1));
        if(n.tag==WOTR_SET){
            // Precompute all reprs, including the one-element case.
            std::vector<std::pair<std::string,ap_Frozen>> ordered;
            for(auto& child:out.children)ordered.emplace_back(ap_repr(child),std::move(child));
            std::stable_sort(ordered.begin(),ordered.end(),[](const auto& a,const auto& b){return a.first<b.first;});
            out.children.clear();for(auto& item:ordered)out.children.push_back(std::move(item.second));
        }
        return out;
    }
    throw std::invalid_argument("native policy choice value type is not implemented");
}
uint64_t ap_get(const Snapshot& s,uint64_t dict,const char* key){
    if(dict==missing || !truth(s,dict))return missing;
    typed(s,dict,WOTR_DICT,"native policy params/continuation must be dictionary");
    return field(s,dict,key,false);
}
// Rows are read-only views. The descriptor graph remains owned and unchanged
// throughout projection, menu-key construction and reply-panel admission.
struct ap_Row {uint64_t node,params;int64_t die;std::string_view kind;};
std::vector<ap_Row> ap_rows(const Snapshot& s){
    const auto& list=typed(s,s.root,WOTR_LIST,"native policy input must be a descriptor list");
    std::vector<ap_Row> rows;
    for(uint64_t i=0;i<list.count;++i){
        const auto node=s.edges[list.offset+i];const auto& tuple=typed(s,node,WOTR_TUPLE,"native policy descriptor must be a tuple");
        require(tuple.count==3,"native policy descriptor must have three fields");
        const auto die=typed(s,s.edges[tuple.offset],WOTR_INT,"native policy die index must be integer").integer;
        const auto params=s.edges[tuple.offset+2];typed(s,params,WOTR_DICT,"native policy params must be dictionary");
        rows.push_back({node,params,die,text(s,s.edges[tuple.offset+1])});
    }
    return rows;
}
bool ap_ends(std::string_view value,std::string_view suffix){return value.size()>=suffix.size() && value.substr(value.size()-suffix.size())==suffix;}
struct ap_Recruit {bool found=false;int score=0;ap_Frozen rest;};
ap_Recruit ap_recruit_choice(const Snapshot& s,uint64_t choices){
    ap_Recruit result;
    if(choices==missing || s.nodes[choices].tag!=WOTR_DICT)return result;
    result.rest=ap_freeze(s,choices);
    for(auto& item:result.rest.children){
        const auto& key=item.children[0].value;auto& value=item.children[1];
        if(value.tag!=WOTR_STRING)continue;
        if((key=="unit_type" || ap_ends(key,"_unit_type")) && (value.value=="regular" || value.value=="elite")){
            result.found=true;result.score+=value.value=="elite";value.value="<regular-or-elite>";
        }else if((key=="unit_combo" || ap_ends(key,"_unit_combo")) && (value.value=="2regular" || value.value=="mixed" || value.value=="2elite")){
            result.found=true;result.score+=value.value=="2elite"?2:value.value=="mixed"?1:0;value.value="<two-recruits>";
        }
    }
    return result;
}
int ap_target_rank(const Snapshot& s,uint64_t target){
    if(target==missing || s.nodes[target].tag!=WOTR_STRING)return -1;
    std::string normalized;
    for(unsigned char c:text(s,target)){
        // Actual legal target identifiers are ASCII (names and Nazgul).
        require(c<128,"native combat victim identifier requires ASCII");
        if(c>='A' && c<='Z')c=c-'A'+'a';
        if((c>='a' && c<='z') || (c>='0' && c<='9'))normalized.push_back(c);
    }
    const std::array<std::string_view,4> names={"witchking","mouthofsauron","saruman","nazgul"};
    for(size_t i=0;i<names.size();++i)if(normalized==names[i])return static_cast<int>(i);
    return -1;
}
std::string ap_payer_name(std::string_view name){
    // Engine Companion names are ASCII. Match strip().casefold() on this
    // supported domain, and refuse an unsupported name rather than misrank it.
    for(unsigned char c:name)require(c<128,"native Mighty Attack payer name requires ASCII");
    auto space=[](unsigned char c){return c==' ' || (c>=9 && c<=13) || (c>=28 && c<=31);};
    size_t first=0,last=name.size();while(first<last && space(name[first]))++first;
    while(last>first && space(name[last-1]))--last;
    return cp_lower(name.substr(first,last-first));
}
NativePolicyProjection ap_project(const Snapshot& state,const Snapshot& descriptors,int32_t side,std::unique_ptr<Snapshot> owned){
    require(side==0 || side==1,"native policy side must be FP or SH");
    const auto rows=ap_rows(descriptors);std::vector<bool> keep(rows.size(),true);std::vector<std::string_view> tags(rows.size());
    NativePolicyProjection result;
    const auto pending=obs_attr(state,state.root,"pending_combat",false);const auto effect=ap_get(state,pending,"effect_choice");
    const auto effect_name=ap_get(state,effect,"effect");
    if(text_is(state,ap_get(state,pending,"phase"),"combat_effect_choice") && text_is(state,effect_name,"dread_and_despair")){
        const auto resume=ap_get(state,pending,"_round_resume"),cursor=ap_get(state,resume,"cursor");
        const auto battle=cursor==missing?missing:obs_attr(state,cursor,"combat",false),owner=ap_get(state,effect,"owner");
        if(battle!=missing && (text_is(state,owner,"atk") || text_is(state,owner,"def"))){
            const auto enemy=cp_attr(state,battle,text_is(state,owner,"atk")?"defender_army":"attacker_army");
            const auto maximum=std::max<int64_t>(1,cp_strength(state,enemy,obs_attr(state,battle,"_wop_cancelled",false))-1);
            bool exact=true,any=false;std::vector<bool> useful(rows.size(),false);
            for(size_t i=0;i<rows.size();++i){
                const auto count=ap_get(descriptors,rows[i].params,"count"),card=ap_get(descriptors,rows[i].params,"card_id");
                if(rows[i].kind!="combat_effect_choice" || descriptors.nodes[rows[i].params].count!=4 || count==missing || descriptors.nodes[count].tag!=WOTR_INT || card==missing ||
                   !ap_equal(ap_freeze(descriptors,card),ap_freeze(state,ap_get(state,effect,"card_id")))){exact=false;break;}
                useful[i]=descriptors.nodes[count].integer>=1 && descriptors.nodes[count].integer<=maximum;any|=useful[i];
            }
            if(exact&&any){keep=std::move(useful);for(auto& tag:tags)tag="dread_no_excess_forfeit";}
        }
    }
    if(text_is(state,ap_get(state,pending,"phase"),"combat_effect_choice") &&
       (text_is(state,effect_name,"blade_of_westernesse") || text_is(state,effect_name,"fateful_strike"))){
        int best=4;size_t index=rows.size();
        for(size_t i=0;i<rows.size();++i){
            // Source normalizes each target before checking action type.
            const int rank=ap_target_rank(descriptors,ap_get(descriptors,rows[i].params,"target"));
            if(rows[i].kind=="combat_effect_choice" && rank>=0 && rank<best){best=rank;index=i;}
        }
        if(index!=rows.size()){
            std::fill(keep.begin(),keep.end(),false);keep[index]=true;result.forced_reason=ap_victim;result.reasons={ap_victim};
        }
    }
    if(text_is(state,ap_get(state,pending,"phase"),"combat_effect_choice") && text_is(state,effect_name,"mighty_attack")){
        const auto resume=ap_get(state,pending,"_round_resume"),cursor=ap_get(state,resume,"cursor");
        const auto battle=ctx_none(state,cursor)?missing:obs_attr(state,cursor,"combat",false);
        const auto companions=ctx_none(state,battle)?missing:obs_attr(state,battle,"companions_in_battle",false);
        std::unordered_map<std::string,uint64_t> figures;
        if(!ctx_none(state,companions))for(auto figure:child_edges(state,companions))
            figures[std::string(text(state,obs_attr(state,figure,"name")))]=figure;
        size_t selected=rows.size();std::tuple<int64_t,bool,std::string,size_t> best;
        for(size_t i=0;i<rows.size();++i){
            const auto target=ap_get(descriptors,rows[i].params,"target");
            // A separate use/decline or unknown target preserves the entire
            // menu, matching _mighty_payer_priority's all-row requirement.
            if(rows[i].kind!="combat_effect_choice" || target==missing || descriptors.nodes[target].tag!=WOTR_STRING){selected=rows.size();break;}
            const auto found=figures.find(std::string(text(descriptors,target)));
            if(found==figures.end()){selected=rows.size();break;}
            const auto name=ap_payer_name(text(descriptors,target));
            const auto candidate=std::make_tuple(cp_number(state,found->second,"leadership"),name=="aragorn",name,i);
            if(selected==rows.size() || candidate<best){best=candidate;selected=i;}
        }
        if(selected!=rows.size()){
            std::fill(keep.begin(),keep.end(),false);keep[selected]=true;
            result.forced_reason=ap_mighty;result.reasons={ap_mighty};
        }
    }
    if(result.forced_reason.empty()){
        struct Family{std::string_view kind;int64_t die;ap_Frozen outer,rest;std::vector<std::pair<size_t,int>> members;};
        std::vector<Family> families;
        for(size_t i=0;i<rows.size();++i){
            const auto& row=rows[i];const auto p=row.params;
            if(row.kind=="muster_recruit" && text_is(descriptors,ap_get(descriptors,p,"type"),"regular") && !truth(descriptors,ap_get(descriptors,p,"saruman"))){
                for(const auto& elite:rows){
                    if(elite.kind!="muster_elite" || elite.die!=row.die || truth(descriptors,ap_get(descriptors,elite.params,"saruman")))continue;
                    if(ap_equal(ap_freeze(descriptors,ap_get(descriptors,p,"nation")),ap_freeze(descriptors,ap_get(descriptors,elite.params,"nation"))) &&
                       ap_equal(ap_freeze(descriptors,ap_get(descriptors,p,"region")),ap_freeze(descriptors,ap_get(descriptors,elite.params,"region")))){
                        keep[i]=false;tags[i]=ap_recruit;break;
                    }
                }
            }
            if(row.kind!="card_option" && row.kind!="play_event")continue;
            const auto recruit=ap_recruit_choice(descriptors,ap_get(descriptors,p,"choices"));if(!recruit.found)continue;
            ap_Frozen outer;outer.tag=WOTR_TUPLE;
            const std::array<std::string_view,6> ignore={"choices","card_name","side","idx","source_card_id","deck"};
            const auto& parameters=descriptors.nodes[p];
            for(uint64_t j=0;j<parameters.count;j+=2){
                const auto key=descriptors.edges[parameters.offset+j];const auto name=text(descriptors,key);
                if(std::find(ignore.begin(),ignore.end(),name)!=ignore.end())continue;
                ap_Frozen item;item.tag=WOTR_TUPLE;item.children={ap_freeze(descriptors,key),ap_freeze(descriptors,descriptors.edges[parameters.offset+j+1])};outer.children.push_back(std::move(item));
            }
            std::sort(outer.children.begin(),outer.children.end(),[](const ap_Frozen& a,const ap_Frozen& b){return a.children[0].value<b.children[0].value;});
            auto family=std::find_if(families.begin(),families.end(),[&](const Family& f){return f.kind==row.kind && f.die==row.die && ap_equal(f.outer,outer) && ap_equal(f.rest,recruit.rest);});
            if(family==families.end()){families.push_back({row.kind,row.die,std::move(outer),recruit.rest,{}});family=families.end()-1;}
            family->members.emplace_back(i,recruit.score);tags[i]=ap_recruit;
        }
        for(const auto& family:families){int best=0;for(const auto& m:family.members)best=std::max(best,m.second);for(const auto& m:family.members)if(m.second<best)keep[m.first]=false;}
        const auto card_pending=obs_attr(state,state.root,"pending_card_option",false);
        if(truth(state,ap_get(state,card_pending,"gandalf_draw"))){
            std::vector<std::pair<size_t,bool>> draws;bool any=false;
            for(size_t i=0;i<rows.size();++i){
                const auto& row=rows[i];const auto choice=ap_get(descriptors,ap_get(descriptors,row.params,"choices"),"gandalf_draw");
                if(row.kind=="card_option" && text_is(descriptors,ap_get(descriptors,row.params,"card_id"),"__gandalf_draw__") && choice!=missing && descriptors.nodes[choice].tag==WOTR_BOOL){
                    const bool take=truth(descriptors,choice);draws.emplace_back(i,take);any|=take;
                }
            }
            if(any)for(const auto& draw:draws){keep[draw.first]=draw.second;tags[draw.first]=ap_draw;}
        }else if(truth(state,ap_get(state,card_pending,"palantir_draw"))){
            for(size_t i=0;i<rows.size();++i)if(rows[i].kind=="card_option" && ap_get(descriptors,ap_get(descriptors,rows[i].params,"choices"),"palantir_deck")!=missing)tags[i]=ap_draw;
        }
        std::set<std::string_view> unique;size_t selected=0,last=0;
        for(size_t i=0;i<rows.size();++i){if(!tags[i].empty())unique.insert(tags[i]);if(keep[i]){++selected;last=i;}}
        result.reasons.assign(unique.begin(),unique.end());
        if(selected==1 && rows[last].kind=="card_option")result.forced_reason=tags[last];
    }
    // A freshly generated search menu has one owner and no derived caches.
    // Transfer it only after all descriptor reads; the public const entry point
    // below continues to copy so its caller's graph is never modified.
    result.actions=owned?std::move(owned):std::make_unique<Snapshot>(descriptors);
    std::vector<uint64_t> selected;for(size_t i=0;i<rows.size();++i)if(keep[i])selected.push_back(rows[i].node);
    // A fresh root avoids mutating an existing list that a descriptor references.
    const auto offset=result.actions->edges.size();result.actions->edges.insert(result.actions->edges.end(),selected.begin(),selected.end());
    result.actions->root=result.actions->nodes.size();result.actions->nodes.push_back({WOTR_LIST,0,0,0,offset,selected.size()});
    return result;
}
NativePolicyProjection project_native_ai_actions(const Snapshot& state,const Snapshot& descriptors,int32_t side){
    return ap_project(state,descriptors,side,{});
}
NativePolicyProjection project_native_ai_actions_owned(const Snapshot& state,std::unique_ptr<Snapshot> descriptors,int32_t side){
    const auto& source=*descriptors;
    return ap_project(state,source,side,std::move(descriptors));
}
