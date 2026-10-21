"""Explicit operator escape hatch. Never called by search or automatic repair.

Prefer an ordinary decline. If that cannot advance the game, the user's skip
authorizes abandoning the unfinished interaction, taking the visible board as
given. It does not invent rolls, cards, casualties or a new turn.
"""


def capture(driver):
    """Identify the current interaction, independently of export/transport lag."""
    import ai
    try:
        pending = ai.describe_pending(driver.es)
        kind = pending['kind'] if pending is not None else None
        side = pending['side'].name if pending is not None else None
    except Exception:
        kind, side = 'unreadable', None
    phase = driver.phase
    if (getattr(driver, '_turn_start_draws_open', False)
            and phase in ('draws', 'declare') and kind in (None, 'declare')):
        # The bridge opens declare while the AI's compulsory draws are still
        # outstanding. That preparatory step is not yet FP's interaction.
        kind, side = 'turn_start_draws', driver.ai_side.name
    if phase in ('declare', 'guide') and kind in (None, 'declare', 'guide_change', 'eye'):
        # The core can already offer eyes while the bridge is still waiting
        # for FP's completion signal. The skip belongs to that visible phase.
        kind, side = phase, 'FREE_PEOPLES'
    label = {'declare': 'Fellowship declaration', 'guide': 'Guide choice'}.get(
        kind, str(kind or phase).replace('_', ' '))
    gesture = None
    if phase == 'actions' and kind is None:
        human_work = (driver._human_window_open() or driver.pending_buf is not None
                      or driver.deferred is not None or driver.chat_deferred is not None
                      or driver.human_physical_work_pending()
                      or bool(driver.holds) or bool(driver.uncommitted))
        if human_work:
            side = driver.human_side.name
        elif driver.last_actor is not None:
            # A speculative ordinary AI action has no authoritative pending
            # until its random result arrives. Its current actor still owns
            # the action, including its issued native requests.
            actor = ai._next_acting_side(driver.es, driver.last_actor)
            side = getattr(actor, 'name', None)
        # A skip typed while idle must not attach to a die staged later in
        # the same feed batch, or to work opened after an acknowledgment.
        group = getattr(driver, '_grouped_die_selection', None)
        selection_record = (group['records'][0] if group and group.get('records')
                            else getattr(driver, '_grouped_die_identity_wait', None))
        gesture = (id(driver.seg.window) if driver.seg.window is not None else None,
                   id(driver.pending_buf) if driver.pending_buf is not None else None,
                   id(driver.deferred) if driver.deferred is not None else None,
                   id(driver.chat_deferred) if driver.chat_deferred is not None else None,
                   driver.echo_pending(), driver.human_physical_work_pending(),
                   bool(driver.forced_ai_rows_pending()),
                   getattr(driver.last_actor, 'name', None),
                   # Native identity delivery refines the same selection;
                   # it must not expire a Skip already bound to that group.
                   id(selection_record) if selection_record is not None else None)
    return {'key': (int(driver.es.turn), phase, kind, side,
                    int(driver.n_commits_total), int(getattr(driver, 'undo_epoch', 0)),
                    getattr(driver, '_cc_battle_serial', 0), bool(driver.game_over), gesture),
            'label': label}


def arm(server, target=None):
    """Capture once; repeated delivery cannot retarget an outstanding skip."""
    if not getattr(server, '_skip_requested', False):
        server._skip_target = target if target is not None else capture(server.driver)
        server._skip_message = None
    server._skip_requested = True


def _evidence(driver):
    from .repair_evidence import dump_signature, same_dump
    from .recovery_board import validate
    from .recovery_epoch import prepare
    if not driver.view.dump_path:
        raise ValueError('waiting for the client board export')
    if driver._feed_seq != len(driver.feed.lines) or driver.feed._buf:
        raise ValueError('waiting for the rest of the native feed')
    evidence = dump_signature(driver)
    if evidence['header'] is None:
        raise ValueError('waiting for the native board position header')
    driver._check_dump_session(evidence['header'])
    hold = driver._dump_position_hold(evidence['header'])
    if hold:
        raise ValueError(hold)
    boundary = prepare(driver, evidence)
    if not driver.view.refresh_dump() or not same_dump(driver, evidence):
        raise ValueError('waiting for a complete stable native board')
    validate(driver, driver.view.piece_region)
    from .choice_recovery import _track
    if any(_track(driver.view.piece_region, pid) is None for pid in ('248', '252')):
        raise ValueError('waiting for readable Fellowship progress and corruption tracks')
    return evidence, boundary


def request(driver):
    """Return (items, message, complete, forced) for one explicit skip."""
    import ai
    missing = getattr(driver, '_sidecar_wait_seq', None)
    if missing is not None:
        return [], ('Skip could not proceed: native gesture metadata is missing '
                    'at feed line %d. Restore the client feed metadata, then '
                    'retry Skip; the current interaction is retained.' % missing), True, False
    # Skip is an explicit operator recovery control, including a stuck AI
    # interaction. Its captured key still prevents skipping the successor.
    interaction = capture(driver)
    # Native gestures can append after tick's poll. Route them before even a
    # legal decline, so a skip cannot abandon an action that has just finished.
    # Preview only: a refusal must not consume that unread evidence.
    import copy
    try:
        reader = copy.deepcopy(driver.feed)
        reader.poll()
        if (reader._buf or reader.lines != driver.feed.lines
                or driver._feed_seq != len(driver.feed.lines)):
            raise ValueError('new native gestures must be routed before skip')
    except (OSError, ValueError) as error:
        driver.findings.append(('skip_waiting_for_feed', str(error)))
        return [], 'Skip requested; waiting for the latest board gestures. I will retry automatically.', False, False
    if getattr(driver, '_allocation_roll_wait', None) is not None:
        # A physical roll is already an observed outcome. Abandoning its
        # allocation window and scheduling another roll would replace that
        # randomness. Complete the retained roll from its original identities.
        from .allocation_roll import complete as complete_allocation_roll
        items = complete_allocation_roll(driver)
        if getattr(driver, '_allocation_roll_wait', None) is not None:
            return items, ('Skip requested; retaining your action-dice roll while '
                           'its matching board and dice identities arrive.'), False, False
        return items, ('Skip complete: the observed action-dice roll is retained; '
                       'continue with the next interaction.'), True, False
    broken_menu = False
    try:
        pending = ai.describe_pending(driver.es)
    except Exception as error:
        pending = None
        broken_menu = True
        driver.findings.append(('skip_unreadable_menu', str(error)))
    halted = bool(getattr(driver.rand, 'sticky_halt', None))
    # Completion is the normal decline for the entire Fellowship phase.
    if (not halted and not broken_menu and driver.phase in ('declare', 'guide')
            and driver.human_side.name == 'FREE_PEOPLES'):
        original = capture(driver)
        items = driver._on_chat({'body': 'fp ok', 'nick': 'skip control'})
        if driver.phase == 'eyes':
            return items, 'Skipped declaration; the Fellowship phase is complete.', True, False
        if getattr(driver, '_fellowship_completion_requested', False):
            return [], 'Skip requested; waiting for the current board export.', False, False
        if capture(driver)['key'] != original['key']:
            return items, 'Skipped %s; continue with the next interaction.' % original['label'], True, False
    elif (not halted and pending is not None and pending['side'] is driver.human_side
          and not getattr(driver.pending_buf, 'records', None)
          and not driver.human_physical_work_pending() and not driver.echo_pending()
          and any(action.action_type in driver._SKIP_SAFE_TYPES
                  or (isinstance((action.params or {}).get('choices'), dict)
                      and (action.params or {})['choices']
                      and (not any(bool(value) for value in action.params['choices'].values())
                           or action.params['choices'].get('action') == 'none'))
                  for action in pending.get('actions', ()))):
        before = driver.n_commits_total
        items, message = driver.skip_open_decision()
        if driver.n_commits_total != before:
            return items, message, True, False
    blocked = (broken_menu or halted or pending is not None or driver.phase != 'actions'
               or driver.pending_buf is not None or driver._human_window_open()
               or driver.deferred is not None or driver.chat_deferred is not None
               or driver.echo_pending() or driver.human_physical_work_pending()
               or driver.forced_ai_rows_pending() or driver.holds or driver.alarms
               or driver._repair is not None
               or (driver.rand is not None and driver.rand.step_open))
    if not blocked:
        return [], 'No pending interaction to skip; continue your turn.', True, False
    try:
        evidence, boundary = _evidence(driver)
    except (OSError, ValueError) as error:
        driver.findings.append(('skip_waiting_for_board', str(error)))
        return [], 'Skip requested; waiting for the current complete board export. I will retry automatically.', False, False

    name = interaction['label']
    phase = driver.phase
    result = []
    def adopt():
        from .repair_evidence import adopt_measured, same_dump
        from .segmenter import Segmenter
        from .undo_model import RestoreWatch
        old = driver.es
        driver.es = ai.clone_state(old)
        # Hunt separation temporarily owns figures outside both the Fellowship
        # and armies. Preserve their actual objects for this candidate's board
        # adoption before discarding the abandoned placement decisions.
        hunt = getattr(driver.es, 'pending_hunt', None) or {}
        pending_companions = list(getattr(driver.es, '_hobbit_place_queue', None) or ())
        if hunt.get('companion_obj') is not None:
            pending_companions.append(hunt['companion_obj'])
        # Only the abandoned interaction is waived; persistent cards, tracks,
        # units, dice and hidden inventories are retained/adopted below.
        for field in driver._PENDING_FIELDS + ('_fellowship_tail_phase', '_wwsat_owed',
                                               '_wormtongue_attack_pending',
                                               '_pending_reveal_strongholds',
                                               '_fellowship_route_event'):
            value = getattr(driver.es, field, None)
            if value is not None or field in ('_hunt_reveal_pending', '_pending_hunt_damage'):
                setattr(driver.es, field, 0 if field in ('_hunt_reveal_pending', '_pending_hunt_damage')
                        else False if isinstance(value, bool) else None)
        driver.es._hobbit_place_queue = []
        for hand in driver.es.hands.values():
            hand.needs_discard = False
        for model in driver.hand_models.values():
            model.es = driver.es
        driver.hands.es = driver.es
        inverse = {str(pid): str(cid) for cid, pid in driver.view.card_pid_map.items()}
        inverse.update(boundary['card_ids'])
        report, _, _ = adopt_measured(driver, evidence,
            driver.es, driver.view.piece_region, dynamic_types=driver.view.dynamic_types,
            ai_module=ai, manual=True, pid_to_cid=inverse,
            guide_hook=driver.reconcile_guide_from_client,
            pending_companions=pending_companions,
            mint=getattr(driver.hands, '_mint', None), human_side=driver.human_side,
            pending_fields=driver._PENDING_FIELDS,
            dice_pids=driver._resync_dice_pids(), dice_phase=phase)
        outcome = driver._adoption_outcome(report, pid_to_cid=inverse)
        if not outcome['complete'] or not same_dump(driver, evidence):
            raise ValueError('client board could not be adopted completely: %s' %
                             (outcome.get('remaining') or '; '.join(report[-2:])))
        # The physical die's paid/pool/Hunt location was adopted above.
        # Replaying the abandoned move's delayed payment would move it again
        # or leave a stale index pointing into the next turn's recovered pool.
        driver.es._fellowship_move_die_pending = None
        from .table_recovery import cleanup_seats, rebase_cleanup
        recovered_table_seats = cleanup_seats(driver.es, driver.view.piece_region, inverse)
        # The adopter may derive a new guide or hand-limit choice. This
        # explicit override waives it for this boundary as well.
        driver.es.pending_discard = None
        driver.es.pending_guide_change = None
        for hand in driver.es.hands.values():
            hand.needs_discard = False
        # This override abandons the whole Hunt, including its protection
        # continuation. Staff has not drawn a tile. An unresolved Mithril
        # original has been drawn, so retire it to the ordinary used pile
        # without applying damage/reveal or using the table card. Keeping it
        # orphaned would block the next Fellowship phase and Ring victory.
        driver.es._wizardstaff_stash = None
        stash = getattr(driver.es, '_mithril_stash', None)
        if stash and not stash.get('return_pending'):
            driver.es.hunt_discard.append(stash['tile'])
            driver.es._mithril_stash = None
        # A resolved redraw instead returns its rejected original to the
        # pool after this action. Preserve that existing settlement helper.
        if stash and stash.get('return_pending'):
            import game_logic
            if driver.rand is not None:
                # The helper shuffles the returned tile. Its residual stream
                # belongs to this rollback-covered recovery candidate.
                with driver.rand.advance():
                    game_logic.return_mithril_tile_if_settled(driver.es)
            else:
                random_state = game_logic.random.getstate()
                try:
                    game_logic.return_mithril_tile_if_settled(driver.es)
                finally:
                    game_logic.random.setstate(random_state)
        driver.seg = Segmenter(nick=driver.nick)
        for field in ('pending_buf', 'deferred', 'chat_deferred', '_repair', '_census_offer',
                      '_attack_ask', '_chat_attack_window', '_chat_attack_identity_wait',
                      '_staged_house_window', '_staged_event_window', '_uncommitted_card_selection',
                      '_grouped_die_selection', '_grouped_die_identity_wait',
                      '_ring_gesture_records', '_human_casualty_prefix', '_ai_casualty_prefix',
                      '_siege_seat_owed', '_siege_seat_receipts', '_siege_seat_redo',
                      '_guide_seat_receipts', '_guide_seat_redo', '_guide_seat_commit_epoch',
                      '_pawn_owed', '_last_blocked_kind', '_last_pending_match'):
            setattr(driver, field, None)
        for field in ('holds', 'deferred_tail', 'uncommitted', 'echo_ledger', 'alarms',
                      '_out_queue', 'owed_die_kills', 'owed_card_kills'):
            setattr(driver, field, [])
        for field in ('echo_expect', '_track_owed', '_pol_recon_inflight', '_cc_reveal_owed'):
            setattr(driver, field, {})
        driver._forced_ai_rows_holds = ()
        driver._post_commit_tail = ()
        driver._corsairs_arrival = None
        driver.human_cc_committed = False
        driver._fellowship_completion_requested = False
        driver._turn_start_draws_open = False
        driver.restore_watch = RestoreWatch()
        driver._last_resync_adoption = outcome
        # A staged action die is already paid in this explicit recovery. Let
        # the existing renderer remove it; never refund or spend another die.
        from .action_die_zones import temporary
        for pid, (side, index) in driver.view.die_id_map.items():
            if temporary(driver.view.piece_region.get(str(pid), '')):
                driver.owed_die_kills.append((side.name, index))
                driver.last_actor = side
        if interaction['key'][2] == 'turn_start_draws' or phase == 'draws':
            # Waive only the unfinished draw transaction. Declaration is the
            # next interaction and remains available to its owner.
            driver.phase = 'declare'
            driver._ensure_declare_armed()
            if driver.ai_side.name == 'FREE_PEOPLES':
                result.append(('ai_declare', None))
        elif phase in ('declare', 'guide', 'fellowship_tail'):
            driver._declare_done_turn = driver.es.turn
            driver.es._eye_allocation_phase = True
            driver.phase = 'eyes'
            if driver.ai_side.name == 'SHADOW':
                result.append(('ai_eyes', None))
        elif phase == 'eyes':
            driver.phase = 'roll'
            result.append(('ai_roll', None))
        elif phase == 'roll':
            # The same automatic scheduler owns normal and operator recovery
            # rolls; skipping a delayed acknowledgment must not roll twice.
            driver.phase = 'roll'
        else:
            driver.phase = 'actions'
        driver.findings.append(('explicit_skip_override', dict(
            decision=name, phase=phase, turn=int(driver.es.turn), board=evidence['sha256'])))
        result.extend(('note', 'skip: ' + row) for row in report)
        from .recovery_epoch import commit
        commit(driver, boundary)
        rebase_cleanup(driver, recovered_table_seats)
        return True
    accepted, error = driver._run_resync_transaction(adopt)
    if not accepted:
        driver.findings.append(('skip_adoption_wait', error))
        return [], 'Skip requested; waiting for a readable board. The AI panel has the recovery details.', False, False
    if phase == 'roll':
        return result, 'Skipped the manual-roll wait; the remaining action dice are rolled in Java.', True, True
    return result, ('Skipped %s; its remaining effects were abandoned. Continuing from '
                    'the pieces currently on the board; turn %d is retained.' %
                    (name, driver.es.turn)), True, True


def run(server):
    """Shared chat/panel route, after feed routing, before new AI work."""
    if not getattr(server, '_skip_requested', False):
        return []
    driver = server.driver
    server._skip_replaced_tick_items = False
    target = getattr(server, '_skip_target', None)
    if target is None:
        target = server._skip_target = capture(driver)
    if target['key'] != capture(driver)['key']:
        # A feed update (including a deferred fp ok) already completed the
        # requested interaction. Its successor is not covered by this skip.
        items, message, complete, forced = (
            [], 'Skip complete: %s has finished.' % target['label'], True, False)
    else:
        items, message, complete, forced = request(driver)
    if complete:
        server._skip_requested = False
        server._skip_target = None
        server._skip_message = None
    elif message == getattr(server, '_skip_message', None):
        return items
    else:
        server._skip_message = message
    if forced:
        server._skip_replaced_tick_items = True
        # The operator explicitly abandoned this step. Do not emit the old
        # unstage operations, reroll a retained request, or replay paid effects.
        server._em_sent = False
        server._step_dead_men_arrival = None
        server._abandon_step('explicit skip: adopted visible board')
        server._step_queue = []
        server._held_nc = None
        server._live_asks = {}
        server._window_asks_sent = set()
        server._ask_meta = {}
        server.rand.reset_feed()
        server.rand.sticky_halt = None
        from .cycle_recovery import rewind_completed
        rewind_completed(server)
    from .chat import skip_status
    return items + [('chat', 'prompt', skip_status(message))]
