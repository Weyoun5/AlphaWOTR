// Exact v45--v52 append-only observation contract. Included in snapshot.cpp.
// Source oracle: the nine frozen static_r3 context modules; never executes Python.
bool tail_none(const Snapshot& s,uint64_t n){return n==missing || s.nodes[n].tag==WOTR_NONE;}
uint64_t tail_get(const Snapshot& s,uint64_t n,const char* key){
    if(tail_none(s,n))return missing;
    return field(s,n,key,false);
}
uint64_t tail_attr(const Snapshot& s,uint64_t n,const char* key){
    if(tail_none(s,n))return missing;
    const auto& o=typed(s,n,WOTR_OBJECT,"expected observation object");
    for(uint64_t i=1;i<o.count;i+=2)if(text_is(s,s.edges[o.offset+i],key))return s.edges[o.offset+i+1];
    for(uint64_t i=1;i<o.count;i+=2)if(text_is(s,s.edges[o.offset+i],"__dict__"))return tail_get(s,s.edges[o.offset+i+1],key);
    return missing;
}
bool tail_seq(const Snapshot& s,uint64_t n){return n!=missing&&(s.nodes[n].tag==WOTR_LIST||s.nodes[n].tag==WOTR_TUPLE);}
std::vector<uint64_t> tail_items(const Snapshot& s,uint64_t n){
    if(tail_none(s,n))return {};
    const auto tag=s.nodes[n].tag;
    require(tag==WOTR_LIST||tag==WOTR_TUPLE||tag==WOTR_SET||tag==WOTR_FROZENSET,"expected observation collection");
    return child_edges(s,n);
}
double tail_number(const Snapshot& s,uint64_t n,double absent=0){
    if(n==missing)return absent;
    const auto& v=s.nodes[n];
    if(v.tag==WOTR_INT||v.tag==WOTR_BOOL)return static_cast<double>(v.integer);
    if(v.tag==WOTR_FLOAT)return v.real;
    if(v.tag==WOTR_STRING||v.tag==WOTR_BIGINT){
        const std::string value(reinterpret_cast<const char*>(s.bytes.data()+v.offset),v.count);
        size_t used=0;const double number=std::stod(value,&used);
        while(used<value.size() && (value[used]==' '||value[used]=='\t'||value[used]=='\n'||value[used]=='\r'))++used;
        require(used==value.size(),"invalid numeric observation value");return number;
    }
    throw std::invalid_argument("invalid numeric observation value");
}
int64_t tail_integer(const Snapshot& s,uint64_t n,const char* message){return typed(s,n,WOTR_INT,message).integer;}
bool tail_equal(const Snapshot& s,uint64_t a,uint64_t b,size_t depth=0){
    if(a==b)return true;
    if(a==missing||b==missing)return false;
    require(depth<100,"cyclic observation history equality");
    const auto& x=s.nodes[a];const auto& y=s.nodes[b];
    const bool xn=x.tag==WOTR_INT||x.tag==WOTR_BOOL||x.tag==WOTR_FLOAT;
    const bool yn=y.tag==WOTR_INT||y.tag==WOTR_BOOL||y.tag==WOTR_FLOAT;
    if(xn&&yn){
        if(x.tag!=WOTR_FLOAT&&y.tag!=WOTR_FLOAT)return x.integer==y.integer;
        return static_cast<long double>(x.tag==WOTR_FLOAT?x.real:static_cast<long double>(x.integer))==
               static_cast<long double>(y.tag==WOTR_FLOAT?y.real:static_cast<long double>(y.integer));
    }
    if(x.tag!=y.tag)return false;
    if(x.tag==WOTR_NONE)return true;
    if(x.tag==WOTR_STRING||x.tag==WOTR_BYTES||x.tag==WOTR_BIGINT)
        return x.count==y.count&&(!x.count||std::memcmp(s.bytes.data()+x.offset,s.bytes.data()+y.offset,x.count)==0);
    if(x.count!=y.count)return false;
    if(x.tag==WOTR_DICT){
        for(uint64_t i=0;i<x.count;i+=2){bool found=false;
            for(uint64_t j=0;j<y.count;j+=2)if(tail_equal(s,s.edges[x.offset+i],s.edges[y.offset+j],depth+1)){
                found=tail_equal(s,s.edges[x.offset+i+1],s.edges[y.offset+j+1],depth+1);break;}
            if(!found)return false;
        }return true;
    }
    for(uint64_t i=0;i<x.count;++i)if(!tail_equal(s,s.edges[x.offset+i],s.edges[y.offset+i],depth+1))return false;
    return true;
}
template<size_t N> int tail_index(const Snapshot& s,uint64_t n,const char* const (&names)[N]){
    for(size_t i=0;i<N;++i)if(text_is(s,n,names[i]))return static_cast<int>(i);return -1;
}
const char* const tail_companions[]={"Gandalf the Grey","Strider","Legolas","Gimli","Boromir","Meriadoc","Peregrin","Aragorn","Gandalf the White","Gollum"};
const char* const tail_strongholds[]={"barad dur","dol guldur","minas morgul","morannon","moria","mount gundabad","orthanc","umbar"};
const char* const tail_sources[]={"hunt","mordor","balrog","orc_patrol","foul_thing","isildur","stronghold","breaking"};
void tail_keys(const Snapshot& s,uint64_t n,std::initializer_list<const char*> allowed,bool exact,const char* message){
    const auto& d=typed(s,n,WOTR_DICT,message);
    require(!exact||d.count==allowed.size()*2,message);
    for(uint64_t i=0;i<d.count;i+=2){bool found=false;
        for(auto key:allowed)if(text_is(s,s.edges[d.offset+i],key))found=true;
        require(found,message);
    }
}
int tail_region(const Snapshot& t,uint64_t keys,std::string_view key){
    const auto& list=t.nodes[keys];for(uint64_t i=0;i<list.count;++i)if(text_is(t,t.edges[list.offset+i],key))return static_cast<int>(i);return -1;
}
uint64_t tail_army(const Snapshot& s,uint64_t attrs,std::string_view key){
    const auto armies=field(s,attrs,"armies");const auto exact=tail_get(s,armies,std::string(key).c_str());
    if(!tail_none(s,exact))return exact;
    const auto& d=s.nodes[armies];
    for(uint64_t i=0;i<d.count;i+=2)if(sanitize_name(text(s,s.edges[d.offset+i]))==key)return s.edges[d.offset+i+1];
    return missing;
}
void tail_commitments(const Snapshot& s,const Snapshot& t,uint64_t keys,uint64_t attrs,uint64_t pending,float* out){
    const bool tsl=text_is(s,tail_get(s,pending,"card_id"),"SHADOW_Strategy_The_Shadow_Lengthens");
    const bool tsim=text_is(s,tail_get(s,pending,"card_id"),"SHADOW_Strategy_The_Shadow_is_Moving");
    if(!tsl&&!tsim)return;
    const auto options=tail_items(s,tail_get(s,pending,"options"));const char* hkey=tsl?"_tsl_moves":"_tsim_moves";
    uint64_t history=missing;bool any=false,all=true;
    for(auto option:options){
        if(tsl)tail_keys(s,option,{"from_region","to_region","n_regular","n_elite","_tsl_moves","_tsl_done","_from_pending","cargo","count","_cargo_key","_cargo_index","_cargo_continue","_cargo_complete"},false,"TSL choice includes a group distinction outside the movement contract");
        else tail_keys(s,option,{"from_region","to_region","_tsim_moves","_tsim_done","_from_pending","cargo","count","_cargo_key","_cargo_index","_cargo_continue","_cargo_complete"},false,"TSiM pending includes unsupported group or legacy immediate history");
        const auto h=tail_get(s,option,hkey);all=all&&h!=missing;
        if(h!=missing){if(any)require(tail_equal(s,history,h),"Conflicting movement commitment histories");else history=h;any=true;}
    }
    if(!any)return;
    if(tail_seq(s,history)&&tail_items(s,history).empty())return;
    require(all&&truth(s,tail_get(s,pending,"direct")),"Incomplete movement commitment provenance");
    require(tail_seq(s,history),"Malformed movement commitment history");
    auto moves=tail_items(s,history);require(tsl?moves.size()==1:moves.size()>=1&&moves.size()<=3,"Invalid committed movement count");
    struct Move{std::string source,target;int source_index,target_index,regular,elite;};std::vector<Move> result;
    for(auto move:moves){
        require(tail_seq(s,move),"Malformed movement commitment");auto parts=tail_items(s,move);
        require(tsl?(parts.size()==2||parts.size()==4||parts.size()==5):(parts.size()==2||parts.size()==3),"Malformed movement commitment");
        const auto source=sanitize_name(text(s,parts[0])),target=sanitize_name(text(s,parts[1]));
        int si=tail_region(t,keys,source),ti=tail_region(t,keys,target);
        require(si>=0&&ti>=0&&si!=ti,"Invalid movement commitment regions");
        const auto army=tail_army(s,attrs,source);require(!tail_none(s,army),"Staged source army missing before movement");
        int counts[2]={0,0};for(auto unit:tail_items(s,tail_attr(s,army,"units"))){
            auto nat=tail_attr(s,unit,"nation");bool shadow=enum_is(s,nat,"game_state.Nation","SAURON")||enum_is(s,nat,"game_state.Nation","ISENGARD")||enum_is(s,nat,"game_state.Nation","SOUTHRON_EASTERLINGS");
            if(shadow){const auto kind=tail_attr(s,unit,"kind");counts[0]+=enum_is(s,kind,"game_state.UnitKind","REGULAR");counts[1]+=enum_is(s,kind,"game_state.UnitKind","ELITE");}
        }
        if(tsl&&parts.size()>=4)for(int i=0;i<2;++i)if(!tail_none(s,parts[i+2])){
            auto n=tail_integer(s,parts[i+2],"Malformed committed count");require(n>=0&&n<=counts[i],"Committed count exceeds source");counts[i]=static_cast<int>(n);}
        const auto cargo=(tsl&&parts.size()==5)?parts[4]:(!tsl&&parts.size()==3)?parts[2]:missing;
        if(!tail_none(s,cargo)){counts[0]=counts[1]=0;for(auto row:tail_items(s,cargo)){const auto values=tail_items(s,row);require(values.size()==4,"Malformed committed cargo");if(text_is(s,values[0],"unit")){const auto n=tail_integer(s,values[3],"Cargo count integer");require(n>0&&n<=10,"Invalid committed cargo count");const int kind=text_is(s,values[2],"REGULAR")?0:text_is(s,values[2],"ELITE")?1:-1;require(kind>=0,"Unknown committed cargo kind");counts[kind]+=static_cast<int>(n);}}}
        require(counts[0]+counts[1]>0,"Cannot commit empty Shadow army");
        result.push_back({source,target,si,ti,counts[0],counts[1]});
    }
    if(tsim){std::sort(result.begin(),result.end(),[](const Move&a,const Move&b){return a.source<b.source;});
        for(size_t i=1;i<result.size();++i)require(result[i-1].source!=result[i].source,"Duplicate staged TSiM source");}
    for(size_t i=0;i<result.size();++i){const auto&m=result[i];auto p=out+108*i;p[0]=1;p[1+m.source_index]=1;p[1+m.target_index]=-1;p[106]=static_cast<float>(m.regular/10.0);p[107]=static_cast<float>(m.elite/10.0);}
}
void tail_nazgul(const Snapshot& s,const Snapshot& t,uint64_t keys,uint64_t pending,float* out){
    if(!text_is(s,tail_get(s,pending,"card_id"),"__nazgul_moves__")&&!text_is(s,tail_get(s,pending,"card_name"),"__nazgul_moves__"))return;
    require(truth(s,tail_get(s,pending,"direct")),"Nazgul continuation lacks direct movement context");
    const auto opts=tail_get(s,pending,"options");require(tail_seq(s,opts)&&truth(s,opts),"Nazgul continuation has no public options");
    uint64_t history=missing;
    for(auto option:tail_items(s,opts)){
        typed(s,option,WOTR_DICT,"Invalid Nazgul continuation option");if(truth(s,tail_get(s,option,"_naz_done")))continue;
        const auto h=tail_get(s,option,"_naz_arr");typed(s,h,WOTR_DICT,"Nazgul flight lacks public arrival provenance");
        if(history!=missing)require(tail_equal(s,history,h),"Conflicting Nazgul arrival histories");else history=h;
    }
    require(history!=missing,"Nazgul continuation has no remaining flight");out[0]=1;
    const char* const flags[]={"__wk_ok__","__wk__","__mo_ok__","__mo__"};const auto& d=s.nodes[history];
    for(uint64_t i=0;i<d.count;i+=2){auto k=s.edges[d.offset+i],v=s.edges[d.offset+i+1];auto n=tail_integer(s,v,"Invalid Nazgul history count");
        const int flag=tail_index(s,k,flags);if(flag>=0){require(n==0||n==1,"Invalid Nazgul Minion eligibility flag");out[106+flag]=static_cast<float>(n);}
        else{const auto region=tail_region(t,keys,text(s,k));require(region>=0&&n>=0&&n<=9,"Invalid Nazgul arrival region or count");out[1+region]=static_cast<float>(n/9.0);}
    }
    require(!truth(s,tail_get(s,history,"__wk__"))||truth(s,tail_get(s,history,"__wk_ok__")),"Moved Witch-king was not eligible for this chain");
    require(!truth(s,tail_get(s,history,"__mo__"))||truth(s,tail_get(s,history,"__mo_ok__")),"Moved Mouth was not eligible for this chain");
}
void tail_companion_moves(const Snapshot& s,uint64_t pending,float* out){
    if(!text_is(s,tail_get(s,pending,"card_id"),"__companion_moves__"))return;
    require(truth(s,tail_get(s,pending,"direct")),"Companion continuation lacks direct movement provenance");
    const auto opts=tail_get(s,pending,"options");require(tail_seq(s,opts)&&truth(s,opts),"Companion continuation has no public options");
    unsigned first=0;bool found=false;
    for(auto option:tail_items(s,opts)){
        typed(s,option,WOTR_DICT,"Malformed Companion movement option");if(truth(s,tail_get(s,option,"_comp_done")))continue;
        const auto h=tail_get(s,option,"_comp_moved");require(tail_seq(s,h)&&truth(s,h),"Missing already-moved Companion names");unsigned mask=0;
        for(auto name:tail_items(s,h)){int i=tail_index(s,name,tail_companions);require(i>=0&&i<9&&!(mask&(1u<<i)),"Malformed already-moved Companion names");mask|=1u<<i;}
        require(!found||mask==first,"Conflicting Companion movement histories");first=mask;found=true;
    }
    require(found,"Companion continuation has missing histories");out[0]=1;for(int i=0;i<9;++i)out[1+i]=(first>>i)&1;
}
void tail_die(const Snapshot& s,uint64_t attrs,float* out){
    const auto index=tail_get(s,attrs,"_fellowship_move_die_pending");if(tail_none(s,index))return;
    auto idx=tail_integer(s,index,"Invalid pending Fellowship movement die index");
    const auto dice=field(s,attrs,"dice");const auto edge=enum_value_edge(s,dice,"game_state.Side","FREE_PEOPLES");require(edge!=missing,"Missing FP dice");auto items=tail_items(s,s.edges[edge]);
    require(idx>=0&&static_cast<uint64_t>(idx)<items.size(),"Invalid pending Fellowship movement die index");auto die=items[static_cast<size_t>(idx)];
    require(enum_is(s,tail_attr(s,die,"side"),"game_state.Side","FREE_PEOPLES")&&truth(s,tail_attr(s,die,"used"))&&!tail_none(s,tail_attr(s,die,"current")),"Pending Fellowship movement die must be consumed");
    for(auto other:tail_items(s,field(s,attrs,"hunt_box")))require(other!=die,"Pending Fellowship movement die is already in the Hunt Box");out[0]=1;
}
int tail_source(const Snapshot& s,uint64_t stash){
    const auto kind=tail_get(s,stash,"apply_kind");if(!tail_none(s,kind)){int k=tail_index(s,kind,tail_sources);require(k>=3,"Unknown Hunt protection draw source");return k;}
    if(truth(s,tail_get(s,stash,"mordor_track")))return 1;return truth(s,tail_get(s,stash,"eye_no_effect"))?2:0;
}
void tail_stash(const Snapshot& s,uint64_t stash,float* out){
    if(tail_none(s,stash))return;typed(s,stash,WOTR_DICT,"Malformed Hunt protection continuation");require(truth(s,stash),"Malformed Hunt protection continuation");
    int source=tail_source(s,stash);auto count=tail_get(s,stash,"successes");auto n=count==missing?0:tail_integer(s,count,"Hunt successes must be an integer from zero to five");
    require(n>=0&&n<=5,"Hunt successes must be an integer from zero to five");if(source<7)out[source]=1;out[7]=static_cast<float>(n/5.0);
}
void tail_stronghold_queue(const Snapshot& s,uint64_t queue,float* out,const char* message){
    if(tail_none(s,queue))return;require(tail_seq(s,queue)&&s.nodes[queue].count<=8,message);unsigned seen=0;size_t position=0;
    for(auto item:tail_items(s,queue)){int k=tail_index(s,item,tail_strongholds);require(k>=0&&!(seen&(1u<<k)),message);seen|=1u<<k;out[position++*8+k]=1;}
}
void tail_hunt(const Snapshot& s,uint64_t attrs,uint64_t pending,float* protection,float* route,float* event,float* breaking){
    const auto mithril=tail_get(s,attrs,"_mithril_stash"),wizard=tail_get(s,attrs,"_wizardstaff_stash");tail_stash(s,mithril,protection);tail_stash(s,wizard,protection+9);
    if(!tail_none(s,mithril)){auto returning=tail_get(s,mithril,"return_pending");if(returning!=missing)protection[8]=static_cast<float>(typed(s,returning,WOTR_BOOL,"Malformed Mithril return status").integer);}
    if(truth(s,tail_get(s,attrs,"_reveal_place_due"))){const auto seen=tail_get(s,attrs,"_sh_tile_drawn_regions");if(truth(s,seen))for(auto item:tail_items(s,seen)){int i=tail_index(s,item,tail_strongholds);require(i>=0,"Unknown stronghold in reveal-event history");protection[17+i]=1;}}
    tail_stronghold_queue(s,tail_get(s,attrs,"_pending_reveal_strongholds"),route,"Malformed pending reveal stronghold draws");
    const auto e=tail_get(s,attrs,"_fellowship_route_event");if(!tail_none(s,e)){
        tail_keys(s,e,{"kind","balrog_due","conditional_strongholds"},true,"Malformed Fellowship route event");const char* const kinds[]={"reveal","declare"};int k=tail_index(s,tail_get(s,e,"kind"),kinds);require(k>=0,"Malformed Fellowship route event");event[k]=1;
        event[2]=static_cast<float>(typed(s,tail_get(s,e,"balrog_due"),WOTR_BOOL,"Malformed Fellowship route event").integer);
        const auto q=tail_get(s,e,"conditional_strongholds");require(tail_seq(s,q),"Malformed conditional stronghold order");tail_stronghold_queue(s,q,event+3,"Malformed conditional stronghold order");
    }
    breaking[0]=truth(s,mithril)&&tail_source(s,mithril)==7;breaking[1]=truth(s,wizard)&&tail_source(s,wizard)==7;
    if(tail_none(s,pending))return;typed(s,pending,WOTR_DICT,"Malformed pending card context");const auto sep=tail_get(s,pending,"bof_separation"),count=tail_get(s,pending,"bof_n_left");
    if(!truth(s,sep)){require(count==missing,"Breaking count without a separation context");return;}
    require(s.nodes[sep].tag==WOTR_BOOL&&s.nodes[sep].integer==1,"Malformed Breaking separation context");auto n=tail_integer(s,count,"Breaking remaining choices must be an integer from one to eight");require(n>=1&&n<=8,"Breaking remaining choices must be an integer from one to eight");
    int eligible=0;auto fship=field(s,attrs,"fellowship");for(auto c:tail_items(s,tail_attr(s,fship,"companions"))){auto name=tail_attr(s,c,"name");eligible+=!text_is(s,name,"Frodo")&&!text_is(s,name,"Sam")&&!text_is(s,name,"Gollum");}
    require(n<=eligible,"Breaking remaining choices exceed eligible Companions");breaking[2]=static_cast<float>(n/8.0);
}
std::string tail_normal(const Snapshot& s,uint64_t node){
    if(node==missing)return {};const auto& n=s.nodes[node];std::string input;
    if(n.tag==WOTR_STRING)input=std::string(text(s,node));else if(n.tag==WOTR_INT)input=std::to_string(n.integer);else return {};
    std::string result;for(size_t i=0;i<input.size();++i){unsigned char c=input[i];
        if(c>='A'&&c<='Z')c=c-'A'+'a';if((c>='a'&&c<='z')||(c>='0'&&c<='9'))result.push_back(static_cast<char>(c));
        // Python lower() expands dotted capital I to ASCII i + combining dot,
        // and maps Kelvin sign to k; subsequent ASCII regex keeps those bytes.
        else if(c==0xc4&&i+1<input.size()&&static_cast<unsigned char>(input[i+1])==0xb0){result+='i';++i;}
        else if(c==0xe2&&i+2<input.size()&&static_cast<unsigned char>(input[i+1])==0x84&&static_cast<unsigned char>(input[i+2])==0xaa){result+='k';i+=2;}
    }return result;
}
int tail_target(const Snapshot& s,uint64_t node){
    const auto normal=tail_normal(s,node);const char* names[]={"gandalfthegrey","strider","legolas","gimli","boromir","meriadoc","peregrin","aragorn","gandalfthewhite","gollum","witchking","saruman","mouthofsauron","leader","nazgul"};
    for(int i=0;i<15;++i)if(normal==names[i])return i;
    if(normal=="merry"||normal=="meriadocbrandybuck")return 5;if(normal=="pippin"||normal=="peregrintook")return 6;if(normal=="witchkingofangmar")return 10;return -1;
}
int tail_owner(const Snapshot& s,uint64_t owner){if(text_is(s,owner,"atk"))return 0;require(text_is(s,owner,"def"),"Unknown combat effect owner");return 1;}
void tail_combat(const Snapshot& s,uint64_t attrs,float* out){
    auto pc=tail_get(s,attrs,"pending_combat");if(!truth(s,pc))pc=missing;
    auto payload=tail_get(s,pc,"_round_resume");if(!truth(s,payload))payload=missing;
    const auto cursor=tail_get(s,payload,"cursor");
    if(tail_none(s,cursor)){
        auto effect=tail_get(s,pc,"effect_choice");if(!truth(s,effect))effect=missing;
        if(text_is(s,tail_get(s,effect,"effect"),"onslaught")&&text_is(s,tail_get(s,effect,"stage"),"post"))out[125]=out[127]=1;
        else if(text_is(s,tail_get(s,effect,"stage"),"onslaught_payment")){
            out[126]=out[127]=1;auto payment=tail_get(s,pc,"_onslaught_payment");const char* names[]={"remaining","paid","count"};
            for(int i=0;i<3;++i)out[128+i]=static_cast<float>(tail_number(s,field(s,payment,names[i]))/10.0);
        }
        if(out[127])out[24+tail_owner(s,tail_get(s,effect,"owner"))]=1;
        return; // Python intentionally does not finite-check this legacy branch.
    }
    require(text_is(s,tail_attr(s,cursor,"schema"),"combat_round_cursor_v1"),"Unknown combat continuation schema");
    const auto combat=tail_attr(s,cursor,"combat");const auto stage_node=tail_attr(s,cursor,"stage");auto stage=tail_integer(s,stage_node,"Invalid combat continuation stage");
    if(stage==4&&truth(s,tail_attr(s,cursor,"ra_remaining")))out[14]=1;
    else if(stage==1&&truth(s,tail_attr(s,cursor,"pre_remaining")))out[120]=1;
    else if(stage>=14)out[15]=1;
    else{if(stage<0)stage+=14;require(stage>=0&&stage<14,"Unknown combat continuation stage");out[stage]=1;}
    auto pending=tail_attr(s,cursor,"pending");if(truth(s,pending)){
        const auto effect=tail_get(s,pending,"effect");if(text_is(s,effect,"pre_combat_casualty")){out[121]=1;out[122]=static_cast<float>(tail_number(s,tail_attr(s,cursor,"pre_remaining"))/10);out[truth(s,tail_attr(s,cursor,"pre_victim_attacker"))?123:124]=1;}
        else{const char* const effects[]={"dread_and_despair","relentless_assault","mighty_attack","fateful_strike","blade_of_westernesse","black_breath","heroic_death","relentless_assault_payment"};int i=tail_index(s,effect,effects);require(i>=0,"Unknown combat continuation effect");out[16+i]=1;}
        out[24+tail_owner(s,tail_get(s,pending,"owner"))]=1;
    }
    const char* const phases[]={"pre_combat","combat","reroll"};bool observed[3]={false,false,false},owners[2][3]={{false,false,false},{false,false,false}};
    for(auto p:tail_items(s,tail_attr(s,cursor,"observed_rolls"))){int i=tail_index(s,p,phases);require(i>=0,"Unknown observed combat roll phase");observed[i]=true;}
    require(!observed[2]||observed[1],"Noncausal observed combat roll phases");
    for(auto p:tail_items(s,tail_attr(s,cursor,"observed_roll_sides"))){require(s.nodes[p].tag==WOTR_TUPLE&&s.nodes[p].count==2,"Unknown observed combat roll owner");auto a=child_edges(s,p);int phase=tail_index(s,a[0],phases);int role=text_is(s,a[1],"attacker")?0:text_is(s,a[1],"defender")?1:-1;require(phase>=0&&role>=0,"Unknown observed combat roll owner");owners[role][phase]=true;}
    const char* roles[]={"attacker","defender"};const char* rolls[]={"pre_combat_rolls","combat_rolls","rerolls"};
    const char* nums[]={"combat_strength","leadership","dice_mod","reroll_mod","hits","pre_combat_hits","bonus_hits","hits_cancelled","final_hits"};
    for(int r=0;r<2;++r){
        for(int p=0;p<3;++p)if(owners[r][p]){int slot=r*3+p;out[26+slot]=1;const auto key=std::string(roles[r])+"_"+rolls[p];
            for(auto die:tail_items(s,tail_attr(s,combat,key.c_str()))){auto d=tail_integer(s,die,"Invalid observed d6");require(d>=1&&d<=6,"Invalid observed d6");auto& bin=out[32+6*slot+d-1];bin+=0.2f;}}
        for(int j=0;j<9;++j){if((j==5&&!observed[0])||((j==4||j==6||j==7||j==8)&&!observed[1]))continue;const auto key=std::string(roles[r])+"_"+nums[j];out[68+r*10+j]=static_cast<float>(tail_number(s,tail_attr(s,combat,key.c_str()))/10);}
        if(observed[2])out[68+r*10+9]=static_cast<float>(tail_number(s,tail_attr(s,combat,r==0?"_resume_atk_reroll_hits":"_resume_def_reroll_hits"))/5);
    }
    std::vector<uint64_t> leaders;for(const char* key:{"attacker_army","defender_army"}){auto part=tail_items(s,tail_attr(s,tail_attr(s,combat,key),"leaders"));leaders.insert(leaders.end(),part.begin(),part.end());}
    auto companions=tail_items(s,tail_attr(s,combat,"companions_in_battle"));leaders.insert(leaders.end(),companions.begin(),companions.end());
    unsigned paid=0;for(auto target:tail_items(s,tail_attr(s,cursor,"paid_leadership_targets"))){int i=tail_target(s,target);if(i>=0)paid|=1u<<i;}
    const auto suppressed=tail_attr(s,combat,"_wop_cancelled"),shined=tail_attr(s,combat,"_shined_companion"),forfeited=tail_attr(s,combat,"_leadership_forfeited");
    for(auto leader:leaders){auto name=tail_attr(s,leader,"name");int i=tail_target(s,name);if(i<0||i>=10)continue;
        out[88+i]=1;out[98+i]=leader==suppressed;bool forfeit=false;for(auto f:tail_items(s,forfeited))if(tail_equal(s,name,f))forfeit=true;
        out[108+i]=leader==shined||forfeit||(paid&(1u<<i));
    }
    auto remaining=tail_attr(s,cursor,"ra_remaining");out[118]=truth(s,remaining)?static_cast<float>(tail_number(s,remaining)/10):0;out[119]=static_cast<float>(tail_number(s,tail_attr(s,cursor,"ra_paid"))/10);
    for(size_t i=0;i<131;++i)require(std::isfinite(out[i]),"Nonfinite public combat observation");
}
void append_tails(const Snapshot& s,const Snapshot& tables,int32_t side,std::vector<float>& out){
    (void)side;const auto attrs=attributes(s,s.root,"game_state.GameState");const auto keys=field(tables,tables.root,"region_keys");require(tables.nodes[keys].count==105,"Tail contract requires 105 ordered regions");
    const size_t start=out.size();out.resize(start+735,0.0f);float* tail=out.data()+start;
    const auto raw_pending=tail_get(s,attrs,"pending_card_option");const auto pending=truth(s,raw_pending)?raw_pending:missing;
    tail_commitments(s,tables,keys,attrs,pending,tail);
    tail_nazgul(s,tables,keys,pending,tail+324);
    tail_companion_moves(s,pending,tail+434);
    tail_die(s,attrs,tail+444);
    tail_hunt(s,attrs,raw_pending,tail+445,tail+470,tail+534,tail+601);
    tail_combat(s,attrs,tail+604);
}
