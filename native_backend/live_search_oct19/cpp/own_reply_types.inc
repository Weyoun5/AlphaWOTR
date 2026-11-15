// Included inside namespace sc, after Node/Menu and before Controller.
// Numerical selector implementation remains in the existing native RNG module.
// SEARCHFIX FIX R (ai._REPLY_SHARED; twin: reply_selection.SharedReplySelector / HandReplies): one reply node's statistics
// across all sampled hands.  Arms are semantic action keys; an arm's position is a 64-bit priority drawn once, at first
// sight (menu order), from the table's own seeded stream, so a key first seen in a later hand takes a stable place in one
// fixed random order.  Width and the lower-confidence rule are rs::Selector's, on this node's completed count.
struct OwnSharedReplies {
    struct Arm {uint64_t priority=0,panels=0;double utility_sum=0.0;double mean()const{return utility_sum/static_cast<double>(panels);}};
    static constexpr size_t none=std::numeric_limits<size_t>::max();
    RngPtr rng{nullptr,wotr_random_destroy};
    std::unordered_map<std::string,size_t> index;
    std::vector<Arm> arms;
    uint64_t completed=0,wire_bytes=13;   // len(actions_bytes(all arms)): the node's cache charge
    size_t pending=none;
    const double exploration=1.25,widening=2.0;
    explicit OwnSharedReplies(const std::array<uint32_t,4>& seed):rng(wotr_random_create_seeded(seed.data(),seed.size()),wotr_random_destroy){
        require(bool(rng),wotr_random_last_error());
    }
    size_t admit(const std::string& key){
        const auto found=index.find(key);if(found!=index.end())return found->second;
        std::array<uint32_t,2> limbs{};require(wotr_random_getrandbits(rng.get(),64,limbs.data(),limbs.size())==0,wotr_random_last_error());
        Arm arm;arm.priority=uint64_t(limbs[0])|(uint64_t(limbs[1])<<32);arms.push_back(arm);
        const size_t at=arms.size()-1;index.emplace(key,at);wire_bytes+=8+key.size();return at;
    }
};
// One visit's hand menu over a node's table: rs::Selector's select / complete / cancel / best on this hand's rows.
struct OwnHandReplies {
    std::shared_ptr<OwnSharedReplies> table;std::vector<size_t> arm_of_row,order;
    OwnHandReplies(std::shared_ptr<OwnSharedReplies> shared,const std::vector<std::string>& keys):table(std::move(shared)){
        arm_of_row.reserve(keys.size());for(const auto& key:keys)arm_of_row.push_back(table->admit(key));
        require(!keys.empty()&&std::unordered_set<size_t>(arm_of_row.begin(),arm_of_row.end()).size()==keys.size(),"A nonempty unique legal action menu is required");
        order.resize(keys.size());std::iota(order.begin(),order.end(),size_t(0));
        std::sort(order.begin(),order.end(),[&](size_t a,size_t b){
            const auto pa=table->arms[arm_of_row[a]].priority,pb=table->arms[arm_of_row[b]].priority;
            return pa!=pb?pa<pb:arm_of_row[a]<arm_of_row[b];});
    }
    uint64_t select(){
        auto& t=*table;require(t.pending==OwnSharedReplies::none,"Complete or cancel the pending reply before selecting again");
        const double span=t.widening*std::sqrt(static_cast<double>(t.completed+1));require(std::isfinite(span),"cannot convert float infinity to integer");
        const size_t count=order.size(),width=span>=static_cast<double>(count)?count:std::max<size_t>(1,static_cast<size_t>(std::ceil(span)));
        size_t chosen=OwnSharedReplies::none;
        for(size_t i=0;i<width;++i)if(t.arms[arm_of_row[order[i]]].panels==0){chosen=order[i];break;}
        if(chosen==OwnSharedReplies::none){
            double best_score=0.0;
            for(size_t i=0;i<width;++i){const auto row=order[i];const auto& arm=t.arms[arm_of_row[row]];
                const double score=arm.mean()-t.exploration*std::sqrt(std::log(static_cast<double>(t.completed+1))/static_cast<double>(arm.panels));
                if(i==0||score<best_score){best_score=score;chosen=row;}
            }
        }
        t.pending=arm_of_row[chosen];return chosen;
    }
    void complete(uint64_t row,const double* values,size_t count,const double* weights,size_t wcount){
        auto& t=*table;require(t.pending!=OwnSharedReplies::none&&row<arm_of_row.size()&&t.pending==arm_of_row[size_t(row)],"Completion does not identify the pending reply");
        double value=0.0;require(wotr_reply_selector_expected(values,count,weights,wcount,&value)==0,wotr_random_last_error());
        auto& arm=t.arms[t.pending];const double updated=arm.utility_sum+value;require(std::isfinite(updated),"Accumulated reply utility is not finite");
        arm.utility_sum=updated;++arm.panels;++t.completed;t.pending=OwnSharedReplies::none;
    }
    void cancel(){table->pending=OwnSharedReplies::none;}
    uint64_t best()const{
        size_t found=OwnSharedReplies::none;double value=0.0;
        for(size_t row=0;row<arm_of_row.size();++row){const auto& arm=table->arms[arm_of_row[row]];
            if(arm.panels&&(found==OwnSharedReplies::none||arm.mean()<value)){found=row;value=arm.mean();}}
        require(found!=OwnSharedReplies::none,"No completed reply evaluations");return found;
    }
};
struct OwnReplies {
    using SelectorPtr=std::unique_ptr<void,void(*)(void*)>;
    struct Entry {
        std::string key;
        uint64_t charge=0;
        SelectorPtr selector{nullptr,wotr_reply_selector_destroy};
        std::shared_ptr<OwnSharedReplies> table;   // SEARCHFIX FIX R, instead of selector
    };
    RngPtr stream{nullptr,wotr_random_destroy};
    std::array<uint32_t,4> world_seed{};
    uint64_t panels=0,world_count=0,next_node_token=1,cache_bytes=0;
    bool reply_shared=false;   // SEARCHFIX FIX R
    std::list<Entry> cache;
    std::map<std::string,int64_t> stats;
    explicit OwnReplies(const WotrOwnValueConfig& cfg):panels(cfg.reply_panels),world_count(cfg.belief_worlds){
        require(cfg.struct_size==sizeof(WotrOwnValueConfig)&&cfg.version==1,"Own-value config version/size mismatch");
        require((cfg.flags&~uint32_t(1))==0&&cfg.seed_words<=8,"Invalid own-value seed configuration");
        reply_shared=(cfg.flags&1)!=0;
        require(panels>0&&world_count>0,"Own-value panels and worlds must be positive");
        stream.reset(wotr_random_create_seeded(cfg.seed,cfg.seed_words));
        require(bool(stream),wotr_random_last_error());
        require(wotr_random_getrandbits(stream.get(),128,world_seed.data(),4)==0,wotr_random_last_error());
    }
    static RngPtr seeded(const uint32_t* limbs,size_t count){
        RngPtr result(wotr_random_create_seeded(limbs,count),wotr_random_destroy);
        require(bool(result),wotr_random_last_error());return result;
    }
    template<size_t N> static std::array<uint32_t,N> bits(void* rng){
        std::array<uint32_t,N> out{};
        require(wotr_random_getrandbits(rng,32*N,out.data(),N)==0,wotr_random_last_error());
        return out;
    }
    static std::vector<bw::World> worlds(const Snapshot& state,const Snapshot& tables,int actor,
                                        uint64_t count,const std::array<uint32_t,4>& seed){
        auto local=seeded(seed.data(),seed.size());std::vector<bw::World> out;
        require(count<=out.max_size(),"Belief world count exceeds native storage");out.reserve(size_t(count));
        for(uint64_t i=0;i<count;++i){
            auto words=bits<8>(local.get());auto rng=seeded(words.data(),words.size());
            out.push_back(bw::sample(state,tables,actor,rng.get()));
        }
        return out;
    }
};
