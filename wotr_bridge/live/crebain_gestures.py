"""Prove the human move that opens Shadow's pre-Hunt Crebain decision."""


def interposition_match(state, side, window):
    """Match a net one-step FSP move and its die transport to the Hunt Box.

    The returned engine state still owes the Shadow's discard/keep decision.
    It has neither moved the Fellowship nor consumed its die yet; that happens
    when the ordinary card-option executor resumes the saved human action.
    """
    import ai
    from game_state import Side, PersistentCardType, ActionDieFace
    from . import gesture_match
    from .segmenter import fellowship_track_step, _matches_staged_die

    if (side is not Side.FREE_PEOPLES
            or window.die_side != side.name
            or window.closed != 'die_to_hunt'
            or state.fellowship.mordor_entered
            or getattr(state, 'pending_card_option', None) is not None):
        return None
    persistent = getattr(state, 'persistent_cards', None)
    card = persistent.get_by_type(PersistentCardType.FLOCKS_OF_CREBAIN) if persistent else None
    if card is None or not any(d.side is Side.SHADOW for d in state.hunt_box):
        return None
    facts = gesture_match.window_facts(window)
    ring_records = [r for r in window.records if r.get('op') == 'move'
                    and gesture_match.is_ring_track_move(r.get('obj', ''), r.get('from'), r.get('to'))]
    if ring_records and (len(ring_records) != 1 or dict(facts.rings) != {'FP': -1, 'SH': 1}):
        return None
    if not ring_records and window.die_face not in ('C', 'W'):
        return None
    changes = [r for r in window.records if r.get('op') == 'die_change']
    if changes and (not ring_records or len(changes) != 1):
        return None
    records = [r for r in window.records if r not in ring_records and r not in changes]
    stages = [r for r in records if r.get('die_stage')]
    steps = [r for r in records if fellowship_track_step(r) is not None]
    if len(stages) != 1 or not steps:
        return None
    progress = state.fellowship.progress
    if (facts.fs_progress != 1 or fellowship_track_step(steps[0])[0] != progress
            or fellowship_track_step(steps[-1])[1] != progress + 1):
        return None
    stage = stages[0]
    if (tuple(stage.get('die_stage', ())) != (side.name, window.die_face)
            or str(stage.get('from', '')).strip().lower() != 'fp dice'):
        return None
    destination = str(stage.get('to', '')).strip().lower()
    from .action_die_zones import temporary
    ending = None
    if destination != 'the hunt for the ring' and temporary(destination):
        endings = [r for r in records if r.get('op') == 'move'
                   and temporary(r.get('from', ''))
                   and str(r.get('to', '')).strip().lower() == 'the hunt for the ring'
                   and _matches_staged_die(window, r)]
        if len(endings) != 1:
            return None
        ending = endings[0]
        if (ending.get('op') != 'move'
                or not temporary(ending.get('from', ''))
                or str(ending.get('to', '')).strip().lower() != 'the hunt for the ring'
                or not _matches_staged_die(window, ending)):
            return None
    elif destination != 'the hunt for the ring':
        return None
    # Draft/correction moves have no extra rule effect when their net delta
    # is empty. Do not demand exactly three paragraphs or a particular order.
    neutral = [r for r in records if r is not stage and r is not ending and r not in steps]
    neutral_facts = gesture_match.window_facts(gesture_match._PrefixShim(neutral, None, None))
    if not neutral_facts.is_empty() or neutral_facts.unmapped:
        return None
    chain = []
    candidate_state = state
    face = window.die_face
    target = None
    if ring_records:
        if changes:
            changed = changes[0]
            if (not temporary(destination) or changed.get('side') != side.name
                    or changed.get('frm') != window.die_face or changed.get('to') not in ('C', 'W')
                    or not window.records.index(stage) < window.records.index(changed) < window.records.index(ending)):
                return None
            from .native_die_identity import pids as die_pids
            stage_pids, changed_pids = die_pids(stage), die_pids(changed)
            if stage_pids is not None and changed_pids is not None and set(stage_pids) != set(changed_pids):
                return None
            targets = getattr(window, '_explicit_ring_targets', None)
            if targets is None:
                targets = {i for i, die in enumerate(state.dice[side]) if not die.used
                           and gesture_match.FACE_LETTER.get(getattr(die.current, 'name', '')) == changed['frm']}
            if len(targets) != 1:
                return None
            target = next(iter(targets))
            face = changed['to']
        else:
            target, error = gesture_match._inferred_ring_target(state, side, window, facts)
            if error or target is None:
                return None
            face = 'C'
        rings = [a for a in ai.get_legal_actions(state, side)
                 if a.action_type == 'elven_ring'
                 and a.params.get('target_die_idx') == target
                 and gesture_match.FACE_LETTER.get(ActionDieFace(a.params.get('new_face')).name) == face]
        if len(rings) != 1:
            return None
        chain.append(rings[0])
        candidate_state = ai.apply_action(state, rings[0])
    candidates = [a for a in ai.get_legal_actions(candidate_state, side)
                  if a.action_type == 'fellowship_move' and a.die_idx >= 0
                  and gesture_match.FACE_LETTER.get(candidate_state.dice[side][a.die_idx].current.name) == face]
    if len(candidates) != 1:
        return None
    action = candidates[0]
    if target is not None:
        # The menu deduplicates equal faces; the ring's physical die must pay.
        action = ai.Action(target, action.action_type, dict(action.params))
    chain.append(action)
    post = ai.apply_action(candidate_state, action)
    pending = getattr(post, 'pending_card_option', None) or {}
    resume = pending.get('resume_action') or {}
    if (pending.get('card_id') != card.card_id or pending.get('decider') != 'sh'
            or pending.get('options') != [{'crebain_discard': True}, {'crebain_discard': False}]
            or resume.get('action_type') != 'fellowship_move'
            or resume.get('die_idx') != action.die_idx):
        return None
    result = gesture_match.MatchResult('match', chain=chain, post=post, applies=len(chain),
                                       note='Fellowship move awaits Shadow Crebain decision')
    if target is not None:
        window._inferred_ring_target = target
    result.crebain_interposition = True
    return result


def remember_transport(driver, window):
    """Keep already-performed human transport beside its committed pending."""
    from .segmenter import fellowship_track_step
    pending = getattr(driver.es, 'pending_card_option', None) or {}
    resume = pending.get('resume_action') or {}
    steps = [fellowship_track_step(r) for r in window.records]
    steps = [step for step in steps if step is not None]
    if (pending.get('card_name') != 'Flocks of Crebain'
            or resume.get('action_type') != 'fellowship_move' or not steps
            or sum(target - source for source, target in steps) != 1):
        raise ValueError('Crebain transport requires its committed Fellowship pending')
    driver._crebain_human_transport = {
        'state': driver.es, 'window': window, 'step': (steps[0][0], steps[-1][1]),
        'die_idx': resume['die_idx'], 'card_id': pending['card_id']}


def resumed_transport(driver, action, post):
    """Account for only the two human gestures when the saved move resumes."""
    import ai
    from game_state import Side
    witness = getattr(driver, '_crebain_human_transport', None)
    if not witness or witness['state'] is not driver.es or post is None:
        return None
    pre = driver.es
    pending = getattr(pre, 'pending_card_option', None) or {}
    resume = pending.get('resume_action') or {}
    params = action.params or {}
    choices = params.get('choices') or {}
    if (action.action_type != 'card_option'
            or params.get('card_id') != witness['card_id']
            or pending.get('card_id') != witness['card_id']
            or type(choices.get('crebain_discard')) is not bool
            or resume.get('action_type') != 'fellowship_move'
            or resume.get('die_idx') != witness['die_idx']):
        return None
    index = witness['die_idx']
    if (pre.fellowship.progress != witness['step'][0]
            or post.fellowship.progress != witness['step'][1]
            or pre.dice[Side.FREE_PEOPLES][index].used
            or not post.dice[Side.FREE_PEOPLES][index].used):
        return None
    visible = ai.clone_state(pre)
    visible.fellowship.progress = witness['step'][1]
    visible.dice[Side.FREE_PEOPLES][index].used = True
    return {'pre': visible, 'gesture': witness['window'], 'witness': witness}


def _observed_ai_move(driver, pre):
    """Prove the pending AI move from its public progress marker and paid die."""
    from game_state import Side
    from board import sanitize
    if driver.ai_side is not Side.FREE_PEOPLES or driver.human_side is not Side.SHADOW:
        return None
    pending = getattr(pre, 'pending_card_option', None) or {}
    resume = pending.get('resume_action') or {}
    if (pending.get('card_name') != 'Flocks of Crebain'
            or pending.get('decider') != 'sh'
            or resume.get('action_type') != 'fellowship_move'
            or not (resume.get('params') or {}).get('_crebain_resolved')):
        return None
    index = resume.get('die_idx')
    if (type(index) is not int or index < 0 or index >= len(pre.dice[Side.FREE_PEOPLES])
            or pre.dice[Side.FREE_PEOPLES][index].used):
        return None
    target = pre.fellowship.progress + 1
    pieces = driver.view.piece_region
    progress_zone = sanitize(str(pieces.get('248', '')))
    if progress_zone not in ('fellowship track %d' % target, 'track %d' % target):
        return None
    paying = [pid for pid, identity in driver.view.die_id_map.items()
              if identity == (Side.FREE_PEOPLES, index)]
    if not any(sanitize(str(pieces.get(pid, ''))) in ('the hunt for the ring', 'hunt box')
               for pid in paying):
        return None
    return pending, index, target


def observed_ai_progress(driver, pre, action, post):
    """Cover only the proved AI progress when human Shadow resumes Crebain."""
    from game_state import Side
    proof = _observed_ai_move(driver, pre)
    if proof is None:
        return None
    pending, index, target = proof
    params = getattr(action, 'params', None) or {}
    if (getattr(action, 'action_type', '') != 'card_option'
            or params.get('card_id') != pending.get('card_id')
            or type((params.get('choices') or {}).get('crebain_discard')) is not bool
            or not post.dice[Side.FREE_PEOPLES][index].used
            or post.fellowship.progress != target):
        return None
    return [('track_marker', {'piece': '248',
                             'frm': 'Game.%d' % (2014 + pre.fellowship.progress),
                             'to': 'Game.%d' % (2014 + post.fellowship.progress)})]


def pending_discard_window(driver, window):
    """Supplement an actual public Flocks discard with the AI's observed move.

    The human's original records and history remain intact. The ordinary
    matcher still proves the discard choice and consumes real Hunt outcomes.
    """
    import copy
    from game_state import Side
    pending = getattr(driver.es, 'pending_card_option', None) or {}
    if (getattr(driver, 'spectator', False)
            or driver.ai_side is not Side.FREE_PEOPLES or driver.human_side is not Side.SHADOW
            or pending.get('card_name') != 'Flocks of Crebain'):
        return window
    proof = _observed_ai_move(driver, driver.es)
    if proof is None:
        return None
    pending, index, target = proof
    discards = []
    match_records = []
    for record in window.records:
        if record.get('op') == 'roll_num':
            match_records.append(record)
            continue
        if record.get('op') in ('move', 'kill') and record.get('_table_discards') == [pending['card_id']]:
            discards.append(record)
            match_records.append(record)
            continue
        if record.get('op') == 'move':
            import re
            from .gesture_match import title_to_cid, zone_key
            def public_slot(value):
                zone = zone_key(value)
                return (zone in (('play', 'FP'), ('play', 'SH'))
                        or zone[0] == 'other' and re.fullmatch(r'board\d*', zone[1]))
            if (discards or not public_slot(record.get('from'))
                    or not public_slot(record.get('to'))
                    or title_to_cid(record.get('obj', '')) != pending['card_id']):
                return None
            # Either current-card area and the table slots are public display
            # locations. Omit their neutral transit only from this match copy;
            # the following kill still must prove this exact card's discard.
            continue
        return None
    if len(discards) != 1:
        return None
    derived = copy.copy(window)
    from .gesture_match import FACE_LETTER
    derived.die_side = Side.FREE_PEOPLES.name
    derived.die_face = FACE_LETTER[driver.es.dice[Side.FREE_PEOPLES][index].current.name]
    derived.closed = 'die_to_hunt'
    # This is the AI's already observed Fellowship payment, supplied only
    # so the matcher can resume that move after the human's card choice.
    # It is not a new human action-die completion boundary.
    derived._borrowed_payment_context = True
    origin = driver.es.fellowship.progress
    derived.records = match_records + [{
        'op': 'move', 'obj': 'FSP', 'from': 'Fellowship Track %d' % origin,
        'to': 'Fellowship Track %d' % target,
        'norm': 'observed AI FSP from fellowship track %d to fellowship track %d' % (origin, target)}]
    return derived
