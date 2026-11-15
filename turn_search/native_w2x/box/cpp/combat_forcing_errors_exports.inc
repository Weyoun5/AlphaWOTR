// New entry only: old forced-combat ABI and error ordering remain unchanged.
void* wotr_snapshot_apply_descriptor_forced_combat_compatible(const void* handle,const void* tables,const void* menu,const void* random,int32_t side,int64_t index,const int64_t* plan,const uint8_t* tuples,size_t count,int32_t confusion,void** random_out,void** events,void** failure){
    return checked<void*>([&]()->void*{
        require(random_out&&events&&failure&&random_out!=events&&random_out!=failure&&events!=failure,"Compatible forced action needs distinct outputs");
        *random_out=nullptr;*events=nullptr;*failure=nullptr;
        require(count<=256&&(confusion==0||confusion==1),"Invalid forced combat plan domain");
        try{
            cfc_Scope compatible(tuples,count);cf_Scope scope(plan,count,confusion!=0);
            auto result=apply_native_random_descriptor(snapshot(handle),snapshot(tables),snapshot(menu),random,side,index);
            if(!cf_same_rng(random,result.random.get()))throw cfc_Error(4,0,"",0,0,0);
            if(scope.plan.next!=scope.plan.size)throw cfc_Error(5,0,"",0,0,0);
            *random_out=result.random.release();*events=result.events.release();return result.state.release();
        }catch(const cfc_Error& e){
            am_Builder b;b.out.root=b.container(WOTR_TUPLE,{b.integer(e.kind),b.integer(e.index),b.str(e.context),b.integer(e.count),b.integer(e.threshold),b.integer(e.modifier)});
            *failure=std::make_unique<Snapshot>(b.out).release();error(e.what());return nullptr;
        }
    },nullptr);
}
