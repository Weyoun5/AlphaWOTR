#pragma once
#include "wotr_native.h"

// A lossless typed object graph for migration. This is distinct from the
// regional observation projection. It owns values/edges/bytes, never PyObjects.
// Container edges retain order. Objects: [type-name, field-name, value, ...].
// Enum: [type-name, member-name]. Integer values outside int64 use decimal bytes.
enum WotrValueTag : uint32_t {
    WOTR_NONE=0, WOTR_BOOL=1, WOTR_INT=2, WOTR_FLOAT=3,
    WOTR_STRING=4, WOTR_BYTES=5, WOTR_LIST=6, WOTR_TUPLE=7,
    WOTR_DICT=8, WOTR_SET=9, WOTR_FROZENSET=10,
    WOTR_OBJECT=11, WOTR_ENUM=12, WOTR_BIGINT=13
};
struct WotrValue {
    uint32_t tag;
    uint32_t flags; // 1: registered external LIST binding; otherwise zero
    int64_t integer; // with flags=1, STRING node naming the registered binding
    double real;
    uint64_t offset;
    uint64_t count;
};
extern "C" {
WOTR_API void* wotr_snapshot_apply_half_orcs_event(const void*,const void*,int32_t,int64_t,const char*);
WOTR_API void* wotr_snapshot_apply_half_orcs_choice(const void*,const void*,int32_t,int64_t);
WOTR_API void* wotr_snapshot_half_orcs_choice_actions(const void*,const void*,int32_t);
WOTR_API void* wotr_snapshot_apply_shadow_lengthens_event(const void*,const void*,const void*,int32_t,int64_t,const char*,void**,void**);
WOTR_API void* wotr_snapshot_apply_shadow_lengthens_choice(const void*,const void*,const void*,int32_t,int64_t,void**,void**);
WOTR_API void* wotr_snapshot_shadow_lengthens_choice_actions(const void*,int32_t);
WOTR_API void* wotr_snapshot_apply_nazgul_hunt_event(const void*,const void*,const void*,int32_t,int64_t,const char*,void**,void**);
WOTR_API void* wotr_snapshot_apply_nazgul_hunt_choice(const void*,const void*,const void*,int32_t,int64_t,void**,void**);
WOTR_API void* wotr_snapshot_nazgul_hunt_choice_actions(const void*,int32_t);
WOTR_API void* wotr_snapshot_palantir_draw_actions(const void* handle,int32_t side);
WOTR_API void* wotr_snapshot_apply_palantir_draw(const void* handle,const void* tables,const void* random,int32_t side,int64_t index,void** random_out,void** events);
WOTR_API void* wotr_snapshot_apply_faramir_event(const void*,const void*,const void*,int32_t,int64_t,const char*,void**,void**);
WOTR_API void* wotr_snapshot_apply_faramir_choice(const void*,const void*,const void*,int32_t,int64_t,void**,void**);
WOTR_API void* wotr_snapshot_faramir_choice_actions(const void*,const void*,int32_t);
WOTR_API void* wotr_snapshot_apply_card_casualty_choice(const void*,const void*,const void*,int32_t,int64_t,void**,void**);
WOTR_API void* wotr_snapshot_card_casualty_actions(const void*,const void*,int32_t);
WOTR_API void* wotr_snapshot_apply_pits_event(const void*,const void*,int32_t,int64_t,const char*);
WOTR_API void* wotr_snapshot_apply_pits_choice(const void*,const void*,int32_t,int64_t);
WOTR_API void* wotr_snapshot_pits_choice_actions(const void*,const void*,int32_t);
WOTR_API void* wotr_snapshot_apply_gwaihir_event(const void*,const void*,const void*,int32_t,int64_t,const char*,void**,void**);
WOTR_API void* wotr_snapshot_apply_gwaihir_choice(const void*,const void*,const void*,int32_t,int64_t,void**,void**);
WOTR_API void* wotr_snapshot_gwaihir_choice_actions(const void*,int32_t);
WOTR_API void* wotr_snapshot_apply_power_event(const void*,const void*,int32_t,int64_t,const char*);
WOTR_API void* wotr_snapshot_apply_power_kill(const void*,const void*,int32_t,int64_t,const char*);
WOTR_API void* wotr_snapshot_apply_power_payment(const void*,const void*,int32_t,int64_t);
WOTR_API void* wotr_snapshot_power_kill_actions(const void*,const void*,int32_t);
WOTR_API void* wotr_snapshot_power_payment_actions(const void*,const void*,int32_t);
WOTR_API uint32_t wotr_snapshot_abi_version();
// Build owned immutable indexes once, before publishing observation metadata.
WOTR_API int wotr_snapshot_prepare_observation_tables(void*);
// Physical die conversion helper; not the complete ai.apply_action envelope.
// Message includes a NUL terminator; any error leaves state/output unchanged.
WOTR_API int wotr_snapshot_use_elven_ring(void*, int32_t side, int64_t die_index,
    const char* face_name, char* message, size_t capacity);
WOTR_API int wotr_snapshot_ring_waiver_active(const void*, int32_t side);
// Full frozen v52 observation from resident state and immutable metadata graph.
// Requires exactly 4647 floats (v52 observation v3). Refusal leaves the output untouched.
WOTR_API int wotr_snapshot_encode_state(const void*, const void* tables,
    int32_t side, float* output, size_t output_count);
WOTR_API const char* wotr_snapshot_last_error();
WOTR_API void* wotr_snapshot_create(const WotrValue*, size_t,
    const uint64_t*, size_t, const uint8_t*, size_t, uint64_t root);
WOTR_API void* wotr_snapshot_clone(const void*);
// Exact ordinary and paused-combat ai.clone_state semantics.
WOTR_API void* wotr_snapshot_action_clone(const void*);
// Derived per-handle cache diagnostics. Bytes counts retained graph payload.
// Concurrent readonly clones are supported; callers serialize mutation/close.
WOTR_API int wotr_snapshot_action_clone_cache_info(const void*, uint64_t* hits,
    uint64_t* misses, uint64_t* bytes);
// Read-only victory: winner(-1/0/1), kind(0/1military/2destroyed/3corruption),
// FP VP, SH VP, ended. Provisional military result is distinct from end timing.
WOTR_API int wotr_snapshot_victory(const void*, const void* tables, int32_t end_of_turn,
    int32_t search, int64_t* output, size_t count);
// Owned builtin descriptor graph; raw legal ordinary political/ring subset.
WOTR_API void* wotr_snapshot_political_ring_actions(const void*, const void* tables, int32_t side);
WOTR_API void* wotr_snapshot_muster_actions(const void*, const void* tables, int32_t side);
WOTR_API void* wotr_snapshot_army_move_actions(const void*, const void* tables, int32_t side);
WOTR_API void* wotr_snapshot_companion_political_actions(const void*, const void* tables, int32_t side);
WOTR_API void* wotr_snapshot_apply_companion_political_action(const void*, const void* tables,
    int32_t side, int64_t index, const char* nation, const char* name);
WOTR_API void* wotr_snapshot_apply_descriptor(const void*, const void* tables,
    const void* menu, int32_t side, int64_t index);
WOTR_API void* wotr_snapshot_project_policy(const void*, const void* menu, int32_t side, void** metadata);
WOTR_API void* wotr_snapshot_full_actions(const void*, const void* tables, int32_t side);
WOTR_API void* wotr_snapshot_next_turn(const void*,const void* tables,int32_t* won,void** annotations);
WOTR_API void* wotr_snapshot_enforce_control(const void*,const void* tables);
WOTR_API void* wotr_snapshot_minion_muster_actions(const void*,const void* tables,int32_t side);
WOTR_API void* wotr_snapshot_apply_minion_muster(const void*,const void* tables,int32_t side,int64_t index,const char* name,const char* region);
WOTR_API void* wotr_snapshot_companion_actions(const void*,const void* tables,int64_t index,int32_t mode);
WOTR_API void* wotr_snapshot_apply_companion(const void*,const void* tables,int64_t index,int32_t mode,const char* from,const char* to,const char* const* names,size_t count);
WOTR_API void* wotr_snapshot_combat_deferred_post(const void*,const void* tables,const void* random,void** random_out,void** info);
WOTR_API void* wotr_snapshot_cleanup_sieges(const void*,const void* tables);
WOTR_API void* wotr_snapshot_apply_combat_round_choice(const void*,const void* tables,const void* random,int64_t index,void** random_out,void** events);
WOTR_API void* wotr_snapshot_apply_combat_casualty(const void*,const void*,const void*,int64_t,void**,void**);
WOTR_API void* wotr_snapshot_apply_combat_onslaught(const void*,const void*,const void*,int64_t,void**,void**);
WOTR_API void* wotr_snapshot_apply_combat_onslaught_payment(const void*,const void*,const void*,int64_t,void**,void**);
WOTR_API void* wotr_snapshot_combat_round_begin(const void*,const void*,const void*,int64_t,void**,void**);
WOTR_API void* wotr_snapshot_army_attack_menu(const void*,const void*,int32_t,int64_t,int32_t);
WOTR_API void* wotr_snapshot_apply_army_attack(const void*,const void*,const void*,int32_t,int64_t,const char*,const char*,void**,void**);
WOTR_API void* wotr_snapshot_nazgul_move_actions(const void*,const void*,int32_t);
WOTR_API void* wotr_snapshot_nazgul_chain_actions(const void*,int32_t);
WOTR_API void* wotr_snapshot_apply_nazgul_move(const void*,const void*,int32_t,int64_t,const char*,const char*,int64_t,int32_t);
WOTR_API void* wotr_snapshot_apply_nazgul_chain(const void*,const void*,int32_t,int64_t);
WOTR_API void* wotr_snapshot_will_special_actions(const void*,const void*,int32_t);
WOTR_API void* wotr_snapshot_apply_will_special(const void*,const void*,int32_t,int64_t,int32_t,const char*);
WOTR_API void* wotr_snapshot_apply_combat_card(const void*,const void*,const void*,const char*,int32_t,void**,void**);
WOTR_API void* wotr_snapshot_apply_combat_shine(const void*,const void*,const void*,int64_t,void**,void**);
WOTR_API void* wotr_snapshot_apply_combat_pre_target(const void*,const void*,const void*,int64_t,void**,void**);
WOTR_API void* wotr_snapshot_combat_entry_actions(const void*,const void*,int32_t);
WOTR_API void* wotr_snapshot_apply_combat_retreat(const void* handle,const void* tables,const void* random,const char* region,void** random_out,void** events);
WOTR_API void* wotr_snapshot_apply_hunt_table_event(const void* handle,const void* tables,const void* random,int32_t side,int64_t index,const char* id,void** random_out,void** events);
WOTR_API void* wotr_snapshot_apply_token(const void* handle,const void* tables,const void* random,int32_t side,const char* kind,const char* value,void** random_out,void** events);
WOTR_API void* wotr_snapshot_apply_dwarven_ring(const void* handle,const void* tables,const void* random,int32_t side,int64_t index,void** random_out,void** events);
WOTR_API void* wotr_snapshot_token_actions(const void* handle,const void* tables,int32_t side);
WOTR_API void* wotr_snapshot_dwarven_ring_actions(const void* handle,int32_t side);
WOTR_API void* wotr_snapshot_apply_recruit_event(const void* handle,const void* tables,int32_t side,int64_t index,const char* id);
WOTR_API void* wotr_snapshot_apply_recruit_event_traced(const void*,const void*,const void*,int32_t,int64_t,const char*,void**,void**);
WOTR_API void* wotr_snapshot_recruit_event_choice_actions(const void* handle,const void* tables,int32_t side);
WOTR_API void* wotr_snapshot_apply_recruit_event_choice(const void* handle,const void* tables,int32_t side,int64_t index);
WOTR_API void* wotr_snapshot_apply_hill_trolls_event(const void*,const void*,int32_t,int64_t,const char*);
WOTR_API void* wotr_snapshot_hill_trolls_choice_actions(const void*,const void*,int32_t);
WOTR_API void* wotr_snapshot_apply_hill_trolls_choice(const void*,const void*,int32_t,int64_t);
WOTR_API void* wotr_snapshot_apply_target_recruit_event(const void*,const void*,int32_t,int64_t,const char*);
WOTR_API void* wotr_snapshot_apply_shadow_recruit_event(const void*,const void*,int32_t,int64_t,const char*);
WOTR_API void* wotr_snapshot_apply_shadows_gather_event(const void*,const void*,int32_t,int64_t,const char*);
WOTR_API void* wotr_snapshot_apply_target_recruit_choice(const void*,const void*,int32_t,int64_t);
WOTR_API void* wotr_snapshot_apply_shadows_gather_choice(const void*,const void*,int32_t,int64_t);
WOTR_API void* wotr_snapshot_apply_crebain_choice(const void*,const void*,const void*,int32_t,int64_t,void**,void**);
WOTR_API void* wotr_snapshot_apply_companion_event_choice(const void*,const void*,const void*,int32_t,int64_t,void**,void**);
WOTR_API void* wotr_snapshot_target_recruit_choice_actions(const void*,const void*,int32_t);
WOTR_API void* wotr_snapshot_shadows_gather_choice_actions(const void*,const void*,int32_t);
WOTR_API void* wotr_snapshot_crebain_choice_actions(const void*,int32_t);
WOTR_API void* wotr_snapshot_companion_event_choice_actions(const void*,int32_t);
WOTR_API void* wotr_snapshot_apply_companion_event(const void*,const void*,const void*,int32_t,int64_t,const char*,void**,void**);
WOTR_API void* wotr_snapshot_worn_choice_actions(const void*,int32_t);
WOTR_API void* wotr_snapshot_apply_worn_choice(const void*,const void*,const void*,int32_t,int64_t,void**,void**);
WOTR_API void* wotr_snapshot_overstack_choice_actions(const void* handle,int32_t side);
WOTR_API void* wotr_snapshot_apply_overstack_choice(const void* handle,const void* tables,int32_t side,int64_t index);
WOTR_API void* wotr_snapshot_combat_relocate(const void*,const void*,const char*,const char*,int32_t,int64_t,int64_t,int64_t,int64_t,int32_t,int32_t);
WOTR_API void* wotr_snapshot_combat_phase_actions(const void*,const void*,int32_t);
WOTR_API void* wotr_snapshot_apply_combat_phase(const void*,const void*,const void*,const char*,int64_t,void**,void**);
WOTR_API void* wotr_snapshot_apply_combat_advance(const void*,const void*,const void*,int64_t,void**,void**);
WOTR_API void* wotr_snapshot_lure_choice_actions(const void*);
WOTR_API void* wotr_snapshot_apply_lure_event(const void*,const void*,const void*,int32_t,int64_t,const char*,void**,void**);
WOTR_API void* wotr_snapshot_apply_lure_event_choice(const void*,const void*,const void*,int64_t,void**,void**);
WOTR_API void* wotr_snapshot_apply_hunt_event(const void*,const void*,const void*,int32_t,int64_t,const char*,void**,void**);
WOTR_API void* wotr_snapshot_hunt_event_choice_actions(const void*);
WOTR_API void* wotr_snapshot_apply_hunt_event_choice(const void*,const void*,const void*,int64_t,void**,void**);
WOTR_API void* wotr_snapshot_apply_misc_event(const void*,const void*,const void*,int32_t,int64_t,const char*,void**,void**);
WOTR_API void* wotr_snapshot_misc_event_choice_actions(const void*,int32_t);
WOTR_API void* wotr_snapshot_apply_misc_event_choice(const void*,const void*,int32_t,int64_t);
WOTR_API void* wotr_snapshot_combat_setup(const void*,const void*,const char*,const char*,const char*,const char*,int32_t,int32_t);
WOTR_API void* wotr_snapshot_apply_dice_event(const void*,const void*,int32_t,int64_t,const char*);
WOTR_API void* wotr_snapshot_dice_event_choice_actions(const void*,int32_t);
WOTR_API void* wotr_snapshot_apply_dice_event_choice(const void*,const void*,int32_t,int64_t);
WOTR_API void* wotr_snapshot_fellowship_declare_with_rng(const void*,const void*,const void*,const char*,int32_t,void**,void**);
WOTR_API void* wotr_snapshot_fellowship_tail_with_events(const void*,const void*,const void*,int32_t,void**,int32_t*,void**);
WOTR_API void* wotr_snapshot_resume_preaction_with_events(const void*,const void*,const void*,const void*,const char*,void**,void**);
WOTR_API void* wotr_snapshot_apply_special_tile_event(const void*,const void* tables,const void* random,int32_t side,int64_t index,const char* id,void** random_out,void** events);
WOTR_API void* wotr_snapshot_combat_resume_settle(const void*,const void* tables,const void* random,int64_t selection,void** random_out);
WOTR_API void* wotr_snapshot_resume_preaction(const void* before,const void*,const void* tables,const void* random,const char* kind,void** random_out);
WOTR_API void* wotr_snapshot_hunt_protection_actions(const void*);
WOTR_API void* wotr_snapshot_apply_hunt_protection(const void*,const void* tables,const void* random,int64_t index,void** random_out,void** events);
WOTR_API void* wotr_snapshot_play_event_menu(const void*,const void* tables,int32_t side,int64_t index,const char* category);
WOTR_API void* wotr_snapshot_apply_simple_event(const void*,const void* tables,int32_t side,int64_t index,const char* id);
WOTR_API void* wotr_snapshot_gandalf_draw_actions(const void*,int32_t side);
WOTR_API void* wotr_snapshot_apply_gandalf_draw(const void*,const void* tables,int32_t side,int32_t draw);
WOTR_API int wotr_snapshot_apply_descriptor_hunt_plan(const void*,const void*,const void*,const void*,const void*,const int64_t*,size_t,int32_t,int64_t,int32_t,void**,void**,void**,void**,double*);
WOTR_API void* wotr_snapshot_apply_descriptor_forced_combat(const void*,const void*,const void*,const void*,int32_t,int64_t,const int64_t*,size_t,int32_t,void**,void**);
WOTR_API void* wotr_snapshot_apply_descriptor_random(const void*,const void* tables,const void* menu,const void* random,int32_t side,int64_t index,void** random_out,void** events);
WOTR_API void* wotr_snapshot_combat_allocation_settle(const void*,const void* tables,const void* random,int32_t enter,void** random_out);
WOTR_API void* wotr_snapshot_safe_next_turn(const void*,const void* tables,int32_t* won,void** annotations);
WOTR_API void* wotr_snapshot_mouth_move_actions(const void*,const void* tables,int32_t side);
WOTR_API void* wotr_snapshot_apply_mouth_move(const void*,const void* tables,int32_t side,int64_t index,const char* from,const char* to);
WOTR_API void* wotr_snapshot_combat_casualty_menu(const void*,int32_t side,const char* region,int64_t hits);
WOTR_API void* wotr_snapshot_combat_casualty_operation(const void*,const void* tables,int32_t side,const char* region,int64_t hits,const char* nation,const char* operation);
WOTR_API void* wotr_snapshot_hunt_menu(const void*,const void* tables,int32_t mode);
WOTR_API void* wotr_snapshot_fellowship_actions(const void*,int64_t index);
WOTR_API void* wotr_snapshot_apply_fellowship_hide(const void*,const void* tables,int64_t index,int32_t strider);
WOTR_API void* wotr_snapshot_hunt_decision(const void*,const void* tables,const void* random,int32_t mode,const char* selection,void** random_out,void** events);
WOTR_API void* wotr_snapshot_hunt_params(const void*, const void* tables);
WOTR_API void* wotr_snapshot_hunt_tile(const void*, const void* tables,const void* random,int64_t successes,int32_t mordor,int32_t track,int32_t eye_no_effect,void** random_out,void** value,void** events);
WOTR_API void* wotr_snapshot_draw_card_actions(const void*, const void* tables,int32_t side);
WOTR_API void* wotr_snapshot_apply_draw_card(const void*, const void* tables,int32_t side,int64_t index,const char* deck);
WOTR_API void* wotr_snapshot_combat_round_stage(const void*,const void* random,int64_t selection,int32_t battle,void** random_out);
WOTR_API void* wotr_snapshot_combat_publish_paused(const void*,const void* random,void** random_out);
WOTR_API void* wotr_snapshot_apply_fellowship_move(const void*,const void* tables,const void* random,int64_t index,void** random_out,void** events);
WOTR_API void* wotr_snapshot_hunt_roll(const void*, const void* tables,const void* random,
    int64_t count,int64_t threshold,int64_t modifier,void** random_out,int64_t* successes,void** events);
WOTR_API void* wotr_snapshot_combat_pre_stage(const void*,const void* random,int64_t selection,
    int32_t battle,void** random_out);
// Read-only all-phase ownership; -1 no actor, -2 error. It does not certify menus.
WOTR_API int wotr_snapshot_next_actor(const void*, int32_t just_acted);
WOTR_API void* wotr_snapshot_pending_actor(const void*, int32_t fallback);
WOTR_API void* wotr_snapshot_combat_post_stage(const void*, int64_t selection);
WOTR_API void* wotr_snapshot_fellowship_tail(const void*, const void* tables, const void* random_source,
    int32_t begin, void** random_out, int32_t* finished);
WOTR_API void* wotr_snapshot_fellowship_declare_actions(const void*, const void* tables, int32_t side);
WOTR_API void* wotr_snapshot_guide_actions(const void*, int32_t side);
WOTR_API void* wotr_snapshot_apply_guide_choice(const void*, const void* tables, const char* name);
WOTR_API void* wotr_snapshot_apply_fellowship_declare(const void*, const void* tables,
    const char* destination, int32_t pass);
WOTR_API void* wotr_snapshot_discard_card_actions(const void*, const void* tables, int32_t side);
WOTR_API void* wotr_snapshot_apply_discard_card(const void*, const void* tables, int32_t side, const char* card);
WOTR_API void* wotr_snapshot_saruman_muster_actions(const void*, const void* tables, int32_t side);
WOTR_API void* wotr_snapshot_apply_saruman_muster_action(const void*, const void* tables,
    int32_t side, int64_t index, const char* mode);
WOTR_API void* wotr_snapshot_eye_allocation_actions(const void*, int32_t side);
WOTR_API void* wotr_snapshot_apply_eye_allocation(const void*, const void* tables, int64_t count);
// Transaction produces three owned handles; both inputs and output slots stay
// unchanged on refusal. Caller releases snapshot, random and face graph handles.
WOTR_API void* wotr_snapshot_roll_action_dice(const void*, const void* random_source,
    int32_t side, void** random_out, void** faces_out);
WOTR_API void* wotr_snapshot_combat_reroll_stage(const void*, const void* random_source, void** random_out);
WOTR_API void* wotr_snapshot_combat_primary_stage(const void*, const void* random_source,
    const char* target_or_null, void** random_out);
WOTR_API void* wotr_snapshot_draw_event_card(const void*, int32_t side,
    const char* deck, void** draw_out);
WOTR_API void* wotr_snapshot_open_hand_limit_discard(const void*, int32_t* opened);
WOTR_API void* wotr_snapshot_begin_turn_reset(const void*, int64_t turn,
    int32_t deal, int32_t bonus, int32_t* won, void** annotations);
WOTR_API void* wotr_snapshot_begin_turn_reset_settled(const void*, const void* tables, int64_t turn,
    int32_t deal, int32_t bonus, int32_t* won, void** annotations);
WOTR_API void* wotr_snapshot_apply_army_move(const void*, const void* tables,
    int32_t side, int64_t index, const char* from, const char* to,
    int64_t regular, int64_t elite, int64_t leaders, int64_t nazgul,
    int32_t take_leaders, int32_t pass);
WOTR_API void* wotr_snapshot_apply_muster_action(const void*, const void* tables,
    int32_t side, int64_t index, const char* nation, const char* region, const char* kind);
// Complete ordinary political action inside an explicit quiet-state gate.
// Returns a new owned child or null with error; input is never modified.
WOTR_API void* wotr_snapshot_apply_political_action(const void*, const void* tables,
    int32_t side, int64_t die_index, const char* nation_name);
WOTR_API void* wotr_snapshot_apply_elven_ring_action(const void*, const void* tables,
    int32_t side, int64_t die_index, const char* face_name);
// Quiet post-action routing: 0=FP, 1=SH, 2=no remaining actor, -1=error.
WOTR_API int wotr_snapshot_next_quiet_actor(const void*, const void* tables, int32_t just_acted);
WOTR_API void wotr_snapshot_destroy(void*);
WOTR_API size_t wotr_snapshot_node_count(const void*);
WOTR_API size_t wotr_snapshot_edge_count(const void*);
WOTR_API size_t wotr_snapshot_byte_count(const void*);
WOTR_API uint64_t wotr_snapshot_root(const void*);
WOTR_API int wotr_snapshot_export(const void*, WotrValue*, size_t,
    uint64_t*, size_t, uint8_t*, size_t);
// Checked scalar-storage primitive, not a game action. Refuses a non-int node
// or unexpected old value without changing the snapshot.
WOTR_API int wotr_snapshot_replace_integer(void*, uint64_t node,
    int64_t expected, int64_t replacement);
// GameState political-track rule only: no action-die consumption or activation.
// Nation names are Nation enum member names. Methods are muster_die, attack,
// companion_ability or event_card. can returns 0/1, advance returns the new box;
// either returns -1 on malformed/unsupported input; illegal advances also -1.
WOTR_API int wotr_snapshot_can_advance_political(const void*, const char* nation_name,
                                               const char* method);
WOTR_API int wotr_snapshot_advance_political(void*, const char* nation_name,
                                           const char* method);
// Read the regional/political observation directly from the complete graph.
// Region keys and Nation enum member names fix the original model feature order.
// Normalization matches the pinned Unicode NFKD-to-ASCII mapping.
// Owned resident-menu metadata, descriptors and compound alias components.
WOTR_API void* wotr_snapshot_search_menu_view(const void*, const void*);
WOTR_API void* wotr_snapshot_search_menu_row(const void*, int64_t);
WOTR_API void* wotr_snapshot_search_menu_rows(const void*, const void*);
WOTR_API void* wotr_snapshot_search_menu_alias_groups(const void*);
// Read-only source lattice partition for a matching complete policy menu.
WOTR_API void* wotr_snapshot_search_lattice_context(const void*, const void*, const void*);
WOTR_API int wotr_snapshot_encode_prefix(const void*, int32_t side,
    const char* const* region_keys, size_t region_count,
    const char* const* nation_names, size_t nation_count,
    float* output, size_t output_count);
// UTF-8 with surrogatepass, then Python board.sanitize semantics. Input length
// is authoritative, including embedded NUL. Output is not NUL-terminated.
// Null output with zero capacity queries required length; no partial writes.
WOTR_API int64_t wotr_snapshot_normalize_name(const char* input, size_t length,
                                            char* output, size_t capacity);
// Exact logical search projections and small metadata; root -1=None, 0=FP, 1=SH.
WOTR_API void* wotr_snapshot_search_metadata(const void*,int32_t);
// Experimental explicit mutable search worlds, not gameplay transitions.
// Isolated owned belief world: input state and RNG remain unchanged.
WOTR_API int wotr_belief_world_sample(const void*,const void*,int32_t,const void*,void**,void**);
WOTR_API void* wotr_search_overlay_sample(const void*,const void*,int32_t,void*);
WOTR_API void* wotr_search_overlay_import(const void*);
WOTR_API void* wotr_search_overlay_export(void*);
WOTR_API void wotr_search_overlay_destroy(void*);
WOTR_API int wotr_search_overlay_require(void*,void*,const char*);
WOTR_API int wotr_search_overlay_patch(void*,void*,void*);
WOTR_API int wotr_search_overlay_recon(void*,void*,int32_t,const void*);
WOTR_API void* wotr_search_overlay_root(const void*,int32_t);
WOTR_API void* wotr_search_overlay_mixture(const void*,const void*);
WOTR_API int wotr_search_overlay_prepare(void*,void*,void*,int32_t,const void*,int32_t);
// Exact compact ExchangeContext constructor values; no distribution math.
WOTR_API void* wotr_snapshot_kernel_context(const void*,const void*,const void*,const char*,const char*,int32_t);
// Compact resident search presence and Hunt diagnostic facts.
WOTR_API void* wotr_snapshot_search_value_facts(const void*);
// Compact combat-search facts and ordered allocation units.
WOTR_API void* wotr_snapshot_combat_search_facts(const void*);
WOTR_API void* wotr_snapshot_combat_search_units(const void*,const void*);
// Exact filtered hand for the optional search overlay assertion.
WOTR_API void* wotr_snapshot_overlay_assert_hand(const void*,int32_t);
// Native data-only chance-key transfer; Python retains its own hash.
WOTR_API void* wotr_snapshot_chance_signature_wire(const void*,int32_t);
WOTR_API void* wotr_snapshot_metadata_wire(const void*);
// Explicit source-compatible PlanMismatch metadata; old forcing ABI is unchanged.
WOTR_API void* wotr_snapshot_apply_descriptor_forced_combat_compatible(const void*,const void*,const void*,const void*,int32_t,int64_t,const int64_t*,const uint8_t*,size_t,int32_t,void**,void**,void**);
WOTR_API void* wotr_snapshot_combat_search_army(const void*,const void*);
WOTR_API void* wotr_snapshot_search_driver_facts(const void*);
WOTR_API void* wotr_search_turn_select_random(const void*, int64_t, const void*);
// Start/select return independent owned search-turn continuations.
// Snapshot kinds: 0 state, 1 raw menu, 2 policy menu, 3 annotations,
// 4 scheduler metadata, 5 combined raw/policy graph. Snapshot/RNG exports
// are independently owned; destroy handles with their respective APIs.
WOTR_API void* wotr_search_turn_start(const void*,const void*,const void*);
WOTR_API void* wotr_search_turn_select(const void*,int64_t);
WOTR_API void wotr_search_turn_destroy(void*);
WOTR_API void* wotr_search_turn_random(const void*);
WOTR_API void* wotr_search_turn_snapshot(const void*,int32_t);
// Resident v52 action v7 matrix: 157 float32 values and one int64 region per row.
WOTR_API int wotr_snapshot_prepare_action_tables_v52(void*);
WOTR_API int wotr_snapshot_encode_actions_v52(const void*,const void*,const void*,const void*,
    int32_t,float*,size_t,int64_t*,size_t);

}
