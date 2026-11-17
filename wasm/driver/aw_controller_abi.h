// The search controller's extern "C" surface as the module's own C++ uses it. The structs are declared inside their
// translation unit (cpp/search_controller.inc, SHARED_ABI section 2); these are the same declarations, field for field
// (identical definitions in two translation units are one type), and the exports of cpp/search_controller_exports.inc.
#pragma once
#include <cstddef>
#include <cstdint>
#include "wotr_snapshot.h"
#include "wotr_own_value.h"
#include "wotr_random.h"

extern "C" {
struct WotrSearchConfig {
    uint32_t struct_size;
    int32_t  root_side;
    int64_t  num_simulations;
    int32_t  sims_menu_scale; int64_t sims_per_child; int64_t sims_floor;
    double   c_puct;
    int64_t  leaf_batch;
    int64_t  turn_horizon;
    int64_t  max_children;
    int64_t  chance_k_max;
    int64_t  kernel_k_max;
    int64_t  chance_draws;
    int32_t  free_cands; int64_t free_cands_n; int64_t free_force;
    int32_t  opponent_policy_mode;
    double   opponent_policy_epsilon;
    int32_t  conf_leaf; int32_t conf_track; int32_t conf_ref;
    int32_t  determinize;
    int32_t  add_noise;
    int32_t  overlay_assert;
    int32_t  reward_contract;
    int32_t  fp_profile;
    int32_t  sh_profile;
    double   phi_rings_rho;
    int32_t  trace;
    int32_t  root_turn_owner;
    int32_t  fpu_parent;
};
struct WotrSearchRequest {
    uint32_t struct_size;
    uint64_t request_id;
    int32_t  operation;
    int32_t  model_side;
    int32_t  purpose;
    uint64_t batch;
    uint64_t rows;
    const uint64_t* leaf_ids;
    const float*    observations;
    const float*    actions;
    const int64_t*  regions;
    const uint64_t* offsets;
};
struct WotrSearchReply {
    uint32_t struct_size;
    uint64_t request_id;
    const float*  priors;
    const double* values;
    const double* conf;
    const double* mixed;
};
struct WotrSearchRoot {
    uint32_t struct_size;
    uint64_t children;
    const uint64_t* visits; const uint64_t* term_visits; const uint64_t* forced_visits;
    const double* value_sum; const double* value_sq; const double* conf_sum; const double* phi_sum;
    const float*  priors_clean;
    const int8_t* proven;
    const double* proven_value;
    const int32_t* side;
    int32_t root_side;
    uint64_t simulations_run;
    uint64_t budget;
};

WOTR_API void wotr_search_controller_destroy(void* ctl);
WOTR_API int wotr_search_controller_advance(void* ctl, int32_t* status);
WOTR_API int wotr_search_controller_request(const void* ctl, WotrSearchRequest* out);
WOTR_API int wotr_search_controller_submit(void* ctl, const WotrSearchReply* reply);
WOTR_API int wotr_search_controller_root(const void* ctl, WotrSearchRoot* out);
WOTR_API void* wotr_search_controller_root_menu(const void* ctl);
WOTR_API void* wotr_search_controller_rng(const void* ctl);
WOTR_API void* wotr_search_controller_request_payload(const void* ctl);
WOTR_API int wotr_search_controller_submit_payload(void* ctl, uint64_t request_id, const void* reply);
// wasm patch 0004 (single-thread controller)
WOTR_API int wotr_search_controller_set_sync_provider(void* ctl, int (*provider)(void*, void*), void* user);
}

namespace awabi {
constexpr int STATUS_REQUEST = 0, STATUS_COMPLETE = 1;
constexpr int OP_PV = 1, OP_PVM = 2, OP_EVAL = 3, OP_DIST = 4;
}
