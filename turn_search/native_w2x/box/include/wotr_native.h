#pragma once
#include <stdint.h>
#include <stddef.h>
#if defined(_WIN32)
#define WOTR_API __declspec(dllexport)
#else
#define WOTR_API __attribute__((visibility("default")))
#endif
// ABI v1 is deliberately a regional/political state slice, not a complete game.
// Integer records retain piece order for future native transitions.
extern "C" {
struct WotrUnit { int32_t region; int32_t nation; int32_t kind; }; // kind: regular=0, elite=1
struct WotrLeader { int32_t region; int32_t side; int32_t named; int32_t nazgul; };
struct WotrRegion { int32_t occupied; int32_t besieged; int32_t controlled_fp; int32_t controlled_sh; int32_t fellowship; };
struct WotrNation { int32_t side; int32_t box; int32_t active; };
WOTR_API uint32_t wotr_abi_version();
WOTR_API const char* wotr_last_error();
WOTR_API void* wotr_position_create(const WotrRegion*, size_t, const WotrNation*, size_t,
                                  const WotrUnit*, size_t, const WotrLeader*, size_t);
WOTR_API void* wotr_position_clone(const void*);
WOTR_API void wotr_position_destroy(void*);
WOTR_API size_t wotr_observation_size(const void*);
WOTR_API int wotr_encode_regions_politics(const void*, int32_t side, float* output, size_t output_count);
// Checked mutation primitive for independent clone/encoding tests; not a legal-action API.
WOTR_API int wotr_set_political_box(void*, size_t nation, int32_t box);
}
