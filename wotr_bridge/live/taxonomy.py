"""ONE mismatch taxonomy for every surface (BL-37, USER 2026-08-25
manual test: "turn 1 there is already an engine divergence" - the
panel printed an informational finding, `die_rebound_to_staged`, under
'!!! DIVERGENCES !!!').

The live server publishes `mismatches` (recent findings + alarms) and
counts them into `client_divergences` / `error_count` with these word
lists; the panel must partition the SAME list with the SAME lists so a
row is a divergence on every surface or on none.  Three classes:

  divergence  the tester's STOP signal / the human's "board disagrees"
  error       a harness/protocol error to file, keep playing unless wedged
  info        protocol bookkeeping - never shown as a divergence

Pure functions, no imports beyond the stdlib: the panel process must
be able to import this without the engine.
"""

DIVERGENCE_WORDS = ('WITNESS ESCALATION', 'INVARIANT', 'desync',
                    'DIVERGENCE', 'COVERAGE GAP',
                    'ADOPTION DECLINED', 'OWED TRACK MARKER STALE',
                    # BL-141/142: a wedged human path is a STOP signal
                    'PENDING BUFFER WEDGED', 'REFUSAL LOOP',
                    # FLC H1: an off-engine political-activation write
                    'FLIPPED OUTSIDE ANY COMMIT')
ERROR_WORDS = ('undecodable', 'lenient_accept', 'unconsumed_rng',
               'UNDO reached a lineless', 'forced_rows_hold', 'FAILED')


# ---------------------------------------------------------------------------
# THE KIND TABLE (L6, 2026-08-25): every finding kind the live lane files
# is classified HERE, by name, at birth.  test_live_taxonomy_kinds.py
# AST-walks the lane and fails the build on an unlisted kind.
#
#   divergence  the client board and the engine are known to disagree,
#               the lane is wedged (no progress possible), the lane
#               HALTED, or the engine cannot express the client's act
#   error       a harness/protocol defect to file; the lane continues
#   info        bookkeeping, echoes, protocol traffic, refusals of a
#               human's own mistake (undo + redo is the protocol)
# ---------------------------------------------------------------------------
_DIV = 'divergence'
_ERR = 'error'
_INF = 'info'

FINDING_KINDS = {
    # -- divergence-class
    'armies_key_collision_adopt': _DIV,
    # BRCMP (2026-08-29): the SAME board rows held across quiescent
    # reads for the confirm window - the one census STOP signal
    'census_divergence_confirmed': _DIV,
    # BRJAVA-7 (2026-08-29): a dump / request from ANOTHER client
    # session under this server - the lane halts, nothing is adopted
    'client_session_mismatch': _DIV,
    'chat_apply_halt': _DIV, 'commit_replay_halt': _DIV, 'draw_halt': _DIV,
    'halt': _DIV, 'client_illegal_muster': _DIV,
    'engine_gap_army_composition': _DIV,
    'engine_gap_draw_grant_declined': _DIV,
    'engine_gap_hunt_reroll_declined': _DIV,
    'engine_gap_partial_advance': _DIV, 'engine_gap_reroll_declined': _DIV,
    'engine_gap_ring_waiver_pick_die': _DIV,
    'hold_engine_gap_draw_grant': _DIV, 'hold_engine_gap_hunt_reroll': _DIV,
    'pending_buffer_orphaned': _DIV, 'pending_buffer_stall': _DIV,
    'pending_unresolved': _DIV, 'reconcile_erased_paid_delta': _DIV,
    'refusal_loop': _DIV, 'prefix_short': _DIV,
    # BL-150 D2: a committed chain discarded a card the human never
    # removed - the engine's hand copy may now disagree with the human's
    'unvoiced_discard_leg': _DIV,
    # -- error-class
    'adopt_blocked_mid_battle': _ERR, 'adopt_deferred_stale_dump': _ERR,
    # BRCMP-4/-9 (2026-08-29): an adoption dropped an open pending the
    # new board cannot answer (loud, filed); the witness gated too
    # long (instrument health - never a divergence); the census ask's
    # chat builder is missing (ask on the status line only)
    'adopt_pending_dropped': _ERR, 'census_aging': _ERR,
    'census_ask_panel_only': _ERR,
    'ambiguous_hold_voiced': _ERR, 'battle_quarantined': _ERR,
    'card_play_region_chat_held': _ERR, 'chat_answer_unmatched': _ERR,
    'chat_deferred_expired': _ERR, 'client_unowed_draw': _ERR,
    'd6_fed_from_refused_window': _ERR, 'dead_candidate_ask_buried': _ERR,
    'deferred_expired': _ERR, 'discard_claim_miss': _ERR,
    'discard_forced': _ERR, 'event_play_orphan_refused': _ERR,
    'eye_alloc_failed': _ERR, 'eye_alloc_unbound': _ERR,
    'eye_before_ai_declare': _ERR, 'eye_menu_unavailable': _ERR,
    'eye_minimum_not_met': _ERR, 'feed_error': _ERR,
    'fellowship_death_bank_err': _ERR, 'forced_rows_hold': _ERR,
    'guide_reconcile_refused': _ERR, 'lenient_accept': _ERR,
    'lenient_empty_window_unstaged': _ERR, 'lenient_hand_fact_unapplied': _ERR,
    'lenient_hand_facts_unread': _ERR, 'lenient_hand_title_unresolved': _ERR,
    'mirror_echo_unclaimed': _ERR, 'pending_forced': _ERR,
    'pending_reminder_failed': _ERR, 'pending_silent_noop': _ERR,
    'pick_identity_stale': _ERR, 'political_sweep_hold': _ERR,
    'prefix_attack_from_noask': _ERR, 'prefix_attack_no_window': _ERR,
    'prefix_attack_refused': _ERR, 'prefix_pass_illegal': _ERR,
    'prefix_pass_nobody': _ERR, 'prefix_refusal': _ERR,
    # DR-5 (2026-08-29): the quiet-tick census witness threw - the board
    # is NOT known clean, the reconcile stays due; a harness defect to
    # file, not a known board disagreement
    'reconcile_census_failed': _ERR,
    'refusal_draw_not_reissued': _ERR, 'refusal_explain_failed': _ERR,
    'reissue_token_expired': _ERR, 'reveal_failed': _ERR,
    'route_error': _ERR, 'sidecar_gap': _ERR, 'siege_movein_undemanded': _ERR,
    'step_wedge_broken': _ERR, 'unconsumed_rng': _ERR,
    'unconsumed_rng_kept_for_open_step': _ERR, 'unowed_draw_press_held': _ERR,
    'window_before_action_phase': _ERR, 'window_out_of_turn': _ERR,
    'window_while_pending': _ERR,
    # -- info-class (bookkeeping, echoes, protocol, refusals of the
    #    human's own mistake)
    'ai_companion_death_emitted': _INF, 'battle_gap_buffered': _INF,
    'board_adopt': _INF, 'capture_marker': _INF,
    # BRCMP design 3 (2026-08-29): the census ASK protocol traffic -
    # offered / consented ('yes') / declined ('no') / withdrawn
    'census_adopt_offered': _INF, 'census_adopt_consented': _INF,
    'census_adopt_declined': _INF, 'census_offer_withdrawn': _INF,
    # SQF-11 (2026-08-29): the decline's standing chat routes - the
    # 'no' reply and the later yes/no reminder are protocol traffic
    'census_declined_chat': _INF, 'census_route_reminder': _INF,
    'cc_ai_reveal_restored': _INF, 'cc_hidden_commit': _INF,
    # A delayed old reveal's chooser preamble was cancelled successfully.
    'cc_prior_reveal': _INF,
    'cc_refused_returned': _INF, 'cc_reveal': _INF, 'cc_reveal_dup_echo': _INF,
    'cc_reveal_echo': _INF, 'cc_reveal_physical': _INF,
    'cc_reveal_refused': _INF, 'chat_attack': _INF,
    'corruption_sweep_emitted': _INF, 'declare_no_pass': _INF,
    'demanded_draw_fed': _INF, 'dice_resync': _INF,
    'gandalf_draw_fed': _INF,
    'palantir_draw_fed': _INF,
    'discard_identity_from_title': _INF,
    'forced_siege_step': _INF,
    'political_activation_emitted': _INF,
    'stall_repair_incomplete': _ERR,
    'resync_engine_not_adopted': _ERR,
    'die_rebound_to_staged': _INF, 'die_self_undo': _INF,
    'discard_identity_from_dump': _INF, 'discard_identity_pending': _INF,
    'discard_owed_claimed': _INF, 'discard_owed_claimed_late': _INF,
    # DR-6 (2026-08-29): a depleted deck's turn-start draw is not owed
    # (rulebook 1763) - announced to the human, pure protocol bookkeeping
    'draw_skipped_deck_depleted': _INF,
    'early_recover_refused': _INF, 'echo_in_self_closed': _INF,
    'echo_move_rebase_exempt': _INF, 'echo_move_rebased': _INF,
    'echo_move_self_closed': _INF, 'eye_alloc_refused': _INF,
    'eye_chat_refused': _INF, 'forced_rows_emitted': _INF,
    'fp_ok_held_for_discard': _INF, 'game_over': _INF,
    'guide_box_recon': _INF, 'guide_recon_skip_dead': _INF,
    'guards_recruit_refused': _INF,
    'hunt_corruption_partial': _INF, 'hunt_corruption_chat_completed': _INF,
    'guide_reconciled': _INF, 'hunt_box_die_refused': _INF,
    'mirror_apply': _INF, 'mirror_kill_echo': _INF, 'mordor_entry': _INF,
    'paid_noop': _INF, 'pawn_sweep_emitted': _INF, 'political_recon': _INF,
    'political_sweep_emitted': _INF, 'political_sweep_skipped_inflight': _INF,
    'attack_activation_emitted': _INF, 'card_activation_emitted': _INF,
    'post_terminator_draw_fed': _INF, 'progress_sweep_emitted': _INF,
    'random_casualty_withheld': _INF, 'refusal_draw_reissued': _INF,
    'refused_reveal_restored': _INF, 'reissued_draw_consumed': _INF,
    'render_dup_dropped': _INF, 'resync_adopt': _INF,
    'resync_engine_adopted': _INF, 'resync_feed_drained': _INF,
    'resync_recovery': _INF, 'resync_uncommitted_cleared': _INF,
    'ring_refused': _INF, 'ring_standalone': _INF, 'rng_card_feed': _INF,
    'rng_companion_feed': _INF, 'rng_d6_feed': _INF, 'rng_tile_feed': _INF,
    'roll_oversize_refused': _INF, 'roll_restore_demanded': _INF,
    'scenario_flip': _INF, 'siege_seat_demanded': _INF,
    'undo_of_ai_line': _INF, 'undo_of_ai_line_redone': _INF,
    'undo_rebased': _INF, 'unowed_draw': _INF,
    'unowed_draw_corrective_owed': _INF, 'unowed_draw_refused': _INF,
    'unowed_draw_undone': _INF,
    # -- self-repair (USER 2026-08-28): a stall happened (error-class,
    #    filed) and the lane recovered; the protocol traffic is info
    'stall_repair_proposed': _ERR, 'stall_repair_done': _ERR,
    'stall_repair_kill_unconfirmed': _ERR,
    'stall_repair_declined': _INF, 'stall_repair_withdrawn': _INF,
    'stall_repair_kill_echo': _INF, 'stall_log_written': _INF,
    # -- BRLANE fix wave (2026-08-29): the lanes between client lines
    #    and engine actions.  divergence = the client and the engine are
    #    known to disagree (a refused / unrendered AI line, a held gesture
    #    that expired); error = a protocol defect to file; info = the
    #    loud bookkeeping that replaced a silent lane.
    'emission_unacknowledged': _DIV, 'emission_refused': _DIV,
    'hold_expired': _DIV, 'battle_resolved_by_hand': _DIV,
    'client_error_line': _DIV,
    'undo_of_ai_line_unreassertable': _ERR, 'chat_unrecognised': _ERR,
    'dieless_gesture_refused': _ERR, 'orphan_die_kill': _ERR,
    # SQF-6 (2026-08-29): the client recover parked a live die in a
    # reinforcements box (frecover Controls.784 artifact) - a client
    # defect to file; the counter-drag demand keeps play moving
    'die_parked_in_reinforcements': _ERR,
    'pending_ambiguous_voiced': _ERR, 'card_play_extra_rows': _ERR,
    'table_kill_payment_ambiguous': _ERR, 'card_target_hint_unmatched': _ERR,
    'ai_declare_parked': _INF, 'hold_retried': _INF,
    'hold_retry_matched': _INF, 'dieless_commit': _INF, 'eye_undone': _INF,
    'undo_of_ai_line_reasserted': _INF, 'emission_echo_expected': _INF,
    'emission_echo_released': _INF, 'reconcile_parked_window': _INF,
    'human_draw_asked': _INF, 'roll_side_from_sidecar': _INF,
    'table_kill_payment_from_dump': _INF, 'card_play_identity_commit': _INF,
    'card_target_from_chat': _INF, 'siege_seat_performed': _INF,
    'echo_expect_cleared': _INF,
    # -- FLA wave (2026-08-30, par5 fleet Families A/B/D)
    # a window-collected record closed an open echo (follow-through)
    'mirror_echo_in_window': _INF,
    # an adoption deferred while open owed: demands explain the rows
    # (erasing them would delete a paid engine delta)
    'adopt_deferred_owed_demands': _INF,
    # pieces staged in an EARLIER turn excluded from the repair kill
    # list - a client-hygiene defect worth filing
    'repair_kill_scope_excluded': _ERR,
    # the gate held (never undid) a gesture while a confirmed
    # divergence stood - the divergence is the stop signal, not this
    'gesture_hold_divergence_open': _INF,
    # a die-less drag accepted as repair traffic during a divergence
    'repair_drag_accepted': _INF,
    # an AI step abandoned because the board resynced under it
    'step_abandoned_after_resync': _INF,
    # a journal mismatch degraded to step-abandon (harness defect to
    # file; the lane continues)
    'journal_mismatch_step_abandoned': _ERR,
    # an unrenderable AI card_option degraded to say-pick + sweep
    # (the engine could not express the effect as client lines)
    'emission_hold_degraded': _ERR,
    # the client watchdog's WWSAT confirm line, consumed as telemetry
    'wwsat_modal_telemetry': _INF,
    # the one evidence-gated wwsat re-send
    'wwsat_ask_resent': _INF,
    # Activation without an engine commit is a tripwire error. A change
    # accompanying a committed action is ordinary diagnostic bookkeeping.
    'political_activation_changed': _ERR,
    'political_activation_committed': _INF,
    # FLD H1 (2026-08-30): the resync adoption lane WAS the off-engine
    # activation writer - the corroboration gate's protocol traffic
    # (kept = blocked de-activation; unadopted = blocked activation
    # with no engine trigger; face_recon = the counter flip walked)
    'political_activation_kept': _ERR,
    'political_activation_unadopted': _ERR,
    'political_face_recon': _INF,
    # FLD-2 (2026-08-30): the performed table-card seat releases its
    # owed row (bookkeeping; the demand itself stays BL-67/BL-121)
    'table_seat_released': _INF,
    # FLD-6a (2026-08-30): an expired deferred window whose die the
    # client killed debits the engine die (1.230 completes-not-
    # refunds); the face-miss twin is a real dice desync to file
    'deferred_die_spent': _INF,
    'deferred_die_spend_miss': _ERR,
    # FLD-10e (2026-08-30): an engine-side die removal is rendered
    # and voiced (the silent shrink was the phantom-die roll wedge)
    'die_removed_rendered': _INF,
    # FLC H5: a card-initiated battle's declaration chat
    'card_battle_announced': _INF,
}


def kind_of(row):
    """The finding kind a published row starts with ('kind: detail'),
    or None for alarms / free text."""
    s = str(row)
    head = s.split(':', 1)[0].strip()
    return head if head in FINDING_KINDS else None


def classify(row):
    """'divergence' | 'error' | 'info' for one published mismatch row:
    a known finding KIND classifies by the table; anything else
    (alarms, free text, an unlisted kind) by the word lists."""
    s = str(row)
    k = kind_of(s)
    if k is not None:
        return FINDING_KINDS[k]
    if any(w in s for w in DIVERGENCE_WORDS):
        return 'divergence'
    if any(w in s for w in ERROR_WORDS):
        return 'error'
    return 'info'


def partition(rows):
    """(divergences, errors, info) - each a list, order preserved."""
    div, err, info = [], [], []
    for r in rows or ():
        k = classify(r)
        (div if k == 'divergence' else err if k == 'error'
         else info).append(str(r))
    return div, err, info
