// R2-D5-V53-NATIVE: encoding generation v53 entry points (encoding_v53.inc), included inside
// snapshot.cpp's extern "C" block; identical in both native trees. The v52 exports
// (wotr_snapshot_encode_state, wotr_snapshot_encode_actions_v52,
// wotr_snapshot_prepare_action_tables_v52) are untouched and stay bit-identical.
// Full v53 observation: exactly 4745 floats = the v52 observation (4647) ++ the 98-wide v53
// state tail; requires the v53 observation tables. Refusal leaves the output untouched.
WOTR_API int wotr_snapshot_encode_state_v53(const void* handle,const void* tables,int32_t side,float* output,size_t count){
    return checked<int>([&](){encode_state_v53(snapshot(handle),snapshot(tables),side,output,count);return 0;},-1);
}
// v53 action matrix: 199 float32 values per row (the v52 row ++ the 42-wide v53 tail) and the
// v52 int64 primary region per row; requires v53 action tables.
WOTR_API int wotr_snapshot_encode_actions_v53(const void* state,const void* tables,const void* menu,const void* action_tables,int32_t side,float* output,size_t count,int64_t* regions,size_t rows){
    return checked<int>([&](){af_encode_v53(snapshot(state),snapshot(tables),snapshot(menu),snapshot(action_tables),side,output,count,regions,rows);return 0;},-1);
}
// Validates the pinned v53 layout, then builds the same owned dictionary indexes as v52.
WOTR_API int wotr_snapshot_prepare_action_tables_v53(void* handle){
    return checked<int>([&](){snapshot(handle);auto& s=*static_cast<Snapshot*>(handle);v53_check_action_tables(s);af_prepare_tables(s);return 0;},-1);
}
