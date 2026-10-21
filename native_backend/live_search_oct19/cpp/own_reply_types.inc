// Included inside namespace sc, after Node/Menu and before Controller.
// Numerical selector implementation remains in the existing native RNG module.
struct OwnReplies {
    using SelectorPtr=std::unique_ptr<void,void(*)(void*)>;
    struct Entry {
        std::string key;
        uint64_t charge=0;
        SelectorPtr selector{nullptr,wotr_reply_selector_destroy};
    };
    RngPtr stream{nullptr,wotr_random_destroy};
    std::array<uint32_t,4> world_seed{};
    uint64_t panels=0,world_count=0,next_node_token=1,cache_bytes=0;
    std::list<Entry> cache;
    std::map<std::string,int64_t> stats;
    explicit OwnReplies(const WotrOwnValueConfig& cfg):panels(cfg.reply_panels),world_count(cfg.belief_worlds){
        require(cfg.struct_size==sizeof(WotrOwnValueConfig)&&cfg.version==1,"Own-value config version/size mismatch");
        require(cfg.reserved==0&&cfg.seed_words<=8,"Invalid own-value seed configuration");
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
