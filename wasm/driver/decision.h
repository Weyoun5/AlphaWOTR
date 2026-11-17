// The AI server's search for one decision on the module (their OwnModelSearch on the native backend at strength 1),
// from a state graph alone: the forced check (project_ai_actions), the controller with this module's network, value
// nets and combat kernel, the root's pick (build_visit_info + select_root_action + THE SAME BUT BETTER); and the net's
// own choice among a menu (ai._greedy_prior_action, decision=True).
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "awnet.h"

namespace awdec {

struct Box {
    void* obs = nullptr; void* act = nullptr; void* menu_tables = nullptr; void* hunt = nullptr;
    awnet::Net net[2], vnet[2];
    // tables.awg (observation, actions_v53, menu, hunt) and the four nets; false and `error` on failure
    bool load(const std::string& tables_awg, const std::string& nets_dir, std::string& error);
};

// the search's pick: a row of `menu` (owned: wotr_snapshot_destroy), or a forced row; error when it failed
struct Pick { void* menu = nullptr; int64_t index = -1; bool forced = false; std::string error; double seconds = 0; };

// OwnModelSearch.choose(state, side, num_simulations=sims) with the seat seed and search index (the own-value seed)
// and the search stream Random(search_seed); turn_owner 0 none, 1 FP, 2 SH
Pick search(Box& box, const void* state, int side, int64_t sims, uint64_t seat_seed, uint32_t index, uint64_t search_seed,
            int turn_owner = 0);

// ai._greedy_prior_action(state, side, net, menu, decision=True) -> the row of `menu` (-1: none)
int64_t raw(Box& box, const void* state, int side, const void* menu);

// a menu snapshot of the rows `indices` of `menu` (native_menu_rows), owned
void* menu_rows(const void* menu, const std::vector<int64_t>& indices);

// _get_priors_and_value's priors of the rows of `menu` (no projection); user: the Box. 0 on success
int priors(void* user, const void* state, int32_t side, const void* menu, float* out, size_t n);

// THE CERTAIN LINE at strength 1 (driver/certain_line.inc): the row of `menu` to play outright instead of `own`, -1
int64_t certain_line(Box& box, const void* state, int side, const void* menu, int64_t own);

// ai_units.pick (driver/ai_units.inc) for the row `index` of `menu`: the action naming the units the seat's rule takes,
// as a one-row menu (owned), or null: *changed 0 played as it came, -1 an error (the rule's failure keeps the pick)
void* units(Box& box, const void* state, int side, const void* menu, int64_t index, int* changed);

}  // namespace awdec
