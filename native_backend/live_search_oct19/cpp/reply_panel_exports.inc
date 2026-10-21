extern "C" {
int wotr_snapshot_reply_panel(const void* handle,const void* tables,const void* menu,const void* random,
                             int32_t actor,int32_t root_side,int64_t index,void** state_out,
                             void** random_out,void** terminal_out,float* pre_obs,float* successor_obs,
                             size_t count,uint64_t* forced_count){
    return checked<int>([&](){
        require(state_out&&random_out&&terminal_out&&forced_count,"Reply panel outputs are required");
        require(state_out!=random_out&&state_out!=terminal_out&&random_out!=terminal_out,"Reply panel handle outputs must be distinct");
        require((pre_obs||successor_obs)?count==rp::observation_size:count==0,"Reply observation size is invalid");
        if(pre_obs&&successor_obs){
            const auto a=reinterpret_cast<uintptr_t>(pre_obs),b=reinterpret_cast<uintptr_t>(successor_obs);
            const size_t width=count*sizeof(float);require(a>b?a-b>=width:b-a>=width,"Reply observation outputs overlap");
        }
        auto result=rp::evaluate(snapshot(handle),snapshot(tables),snapshot(menu),random,actor,root_side,index,pre_obs!=nullptr,successor_obs!=nullptr);
        if(pre_obs)std::copy(result.pre_action_root_obs.begin(),result.pre_action_root_obs.end(),pre_obs);
        if(successor_obs)std::copy(result.successor_root_obs.begin(),result.successor_root_obs.end(),successor_obs);
        *forced_count=result.forced_count;*state_out=result.state.release();*random_out=result.random.release();*terminal_out=result.terminal.release();return 0;
    },-1);
}
}
