#pragma once
#include "wotr_native.h"
// Public legal reply + source forced-policy closure. All outputs are assigned
// only on success; nullable terminal handle means no terminal. Observation
// buffers are optional; terminal successors do not write successor observation.
extern "C" {
WOTR_API int wotr_snapshot_reply_panel(const void*,const void*,const void*,const void*,
    int32_t,int32_t,int64_t,void**,void**,void**,float*,float*,size_t,uint64_t*);
}
