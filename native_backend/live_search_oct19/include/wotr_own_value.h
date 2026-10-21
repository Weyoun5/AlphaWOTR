#pragma once
#include "wotr_native.h"
// Additive own-value entry configuration; the existing search ABI is unchanged.
struct WotrOwnValueConfig {
    uint32_t struct_size;
    uint32_t version;
    uint64_t reply_panels;
    uint64_t belief_worlds;
    uint32_t seed_words;
    uint32_t reserved;
    uint32_t seed[8]; // unsigned integer, least-significant uint32 limb first
};
struct WotrSearchConfig;
extern "C" {
WOTR_API uint32_t wotr_search_controller_own_value_version();
WOTR_API void* wotr_search_controller_create_own_value(
    const void*,const void*,const void*,const void*,const void*,const void*,
    const WotrSearchConfig*,const WotrOwnValueConfig*);
}
