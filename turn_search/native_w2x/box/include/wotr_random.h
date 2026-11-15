#pragma once
#include "wotr_native.h"
// Independent CPython 3.11 Random streams; unsigned integer seed API, no NumPy API.
// Word order is Random.getstate()[1][:624]. Gaussian payload is untouched.
extern "C" {
WOTR_API int wotr_random_begin_forbidden_scope();
WOTR_API void wotr_random_end_forbidden_scope();
WOTR_API const char* wotr_random_last_error();
WOTR_API void* wotr_random_create(const uint32_t*,size_t,uint32_t,int32_t,uint64_t);
// Create Random(unsigned_integer): least-significant uint32 limb first.
// Redundant high zero limbs are ignored; count=0 denotes zero (null allowed).
// Limit: 1048576 limbs. Inputs are borrowed only for this call, output is owned.
WOTR_API void* wotr_random_create_seeded(const uint32_t*,size_t);
WOTR_API void* wotr_random_clone(const void*);
WOTR_API void wotr_random_destroy(void*);
WOTR_API int wotr_random_export(const void*,uint32_t*,size_t,uint32_t*,int32_t*,uint64_t*);
WOTR_API int wotr_random_setstate(void*,const uint32_t*,size_t,uint32_t,int32_t,uint64_t);
WOTR_API int wotr_random_double(void*,double*);
// Little-endian limbs, with the final partial word right shifted, as CPython.
WOTR_API int wotr_random_getrandbits(void*,uint32_t bits,uint32_t*,size_t);
WOTR_API int wotr_random_randbelow(void*,uint64_t,uint64_t*);
WOTR_API int wotr_random_randint(void*,int64_t,int64_t,int64_t*);
WOTR_API int wotr_random_d6(void*,uint32_t*,size_t);
WOTR_API int wotr_random_shuffle(void*,uint64_t*,size_t);
WOTR_API int wotr_random_sample(void*,uint64_t population,size_t k,uint64_t*,size_t);
// null cumulative weights selects uniform random()*population behavior.
WOTR_API int wotr_random_choices(void*,uint64_t population,const double* cumulative,size_t weights_count,
                                 size_t k,uint64_t*,size_t);
}
