// Explicit RNG import after the unchanged Python model selector.
// Existing wotr_search_turn_select and its owned-stream behavior stay intact.
void* wotr_search_turn_select_random(const void* handle,int64_t index,const void* random){
    return checked<void*>([&]()->void*{
        require(handle,"Missing search-turn driver");
        auto input=sd_clone(*static_cast<const sd_Driver*>(handle));
        input->random.reset(wotr_random_clone(random));
        if(!input->random)throw std::invalid_argument(wotr_random_last_error());
        return sd_select(*input,index).release();
    },nullptr);
}
