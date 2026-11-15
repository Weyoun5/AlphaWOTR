#pragma once
#include "wotr_native.h"
// Exclusive-owned selector, root utilities, positive finite allocation settings.
// Keys preserve caller order; ties use shuffled active order or input order for best().
extern "C" {
WOTR_API void* wotr_reply_selector_create(const uint64_t*,size_t,const uint32_t*,size_t,double,double);
WOTR_API void wotr_reply_selector_destroy(void*);
WOTR_API int wotr_reply_selector_select(void*,uint64_t*);
WOTR_API int wotr_reply_selector_complete(void*,uint64_t,const double*,size_t,const double*,size_t,double*);
WOTR_API int wotr_reply_selector_cancel(void*);
WOTR_API int wotr_reply_selector_best(void*,uint64_t*);
WOTR_API int wotr_reply_selector_expected(const double*,size_t,const double*,size_t,double*);
WOTR_API int wotr_reply_selector_stats(void*,uint64_t*,double*,uint64_t*,size_t,uint64_t*,uint64_t*);
WOTR_API void* wotr_reply_selector_rng_clone(void*);
}
