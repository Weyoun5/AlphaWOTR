// Opaque native handles. Serialization is confined to the external test harness.
int wotr_belief_world_sample(const void* state,const void* tables,int32_t actor,const void* rng,void** out_state,void** out_rng){
    if(out_state)*out_state=nullptr;if(out_rng)*out_rng=nullptr;
    return checked<int>([&](){require(out_state&&out_rng,"Belief output pointers are null");
        auto world=bw::sample(snapshot(state),snapshot(tables),actor,rng);
        *out_state=world.state.release();*out_rng=world.rng.release();return 0;
    },-1);
}
