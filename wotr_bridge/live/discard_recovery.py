"""Recover completed hand-limit discards from identified native card endpoints."""


def plan(driver, dump, pending):
    import ai
    import game_logic as gl
    from .combat_recovery import Recovery, _preserve_randomness
    from .decision_endpoint import filter as filter_endpoint
    from .gesture_match import MatchResult
    side = pending['side']
    hand = list(driver.es.hands[side].hand)
    mapping = driver.view.card_pid_map
    prefix = 'FP' if side.name == 'FREE_PEOPLES' else 'SA'
    if any(cid.startswith('HU_') or cid not in mapping for cid in hand):
        return Recovery('hold', 'waiting for the identities of the current hand cards')
    if len({str(mapping[cid]) for cid in hand}) != len(hand):
        return Recovery('hold', 'waiting for distinct native identities for the hand cards')
    shown = {str(pid) for pid, zone in dump.items() if zone == prefix + ' Hand'}
    lost = [cid for cid in hand if dump.get(str(mapping[cid])) == prefix + ' Dead']
    remaining = {str(mapping[cid]) for cid in hand if cid not in lost}
    if not lost or shown != remaining:
        return Recovery('hold', 'the hand and discard pile do not yet show a completed hand-limit payment')
    # Build the actual kill receipt, rather than adopt a smaller hand and erase
    # the pending effects that follow its last required discard.
    window = driver._new_phase_window()
    chain, post = [], driver.es
    with _preserve_randomness(driver.rctl):
        for cid in lost:
            choice = ai.describe_pending(post)
            if not choice or choice['kind'] != 'discard' or choice['side'] is not side:
                return Recovery('hold', 'more cards were removed than this discard decision requires')
            action = next((a for a in choice['actions'] if a.action_type == 'discard_card'
                           and a.params.get('card_id') == cid), None)
            if action is None:
                return Recovery('hold', 'the discarded card is not legal for the retained decision')
            chain.append(action)
            post = ai.apply_action(post, action)
            window.records.append(dict(op='kill', pieces=[
                '%s %s Card' % ('Free' if prefix == 'FP' else 'Shadow', gl.card_deck_of(cid).title())],
                zone='Free Peoples Hand' if prefix == 'FP' else 'Shadow Hand',
                norm='', meta={'pids': [str(mapping[cid])]}, _combat_recovery=True))
    resolved = filter_endpoint(driver, window, [dict(chain=chain, post=post, execution=())])
    if resolved.status != 'matched':
        return Recovery('hold', resolved.reason)
    return Recovery('match', 'recovered %d hand-limit discard(s) from the native board' % len(lost),
                    window=window, match=MatchResult('match', chain=chain, post=post), added=len(lost))


def poll(driver, new_lines=()):
    """Finish a human's visible hand-limit payment at the complete feed boundary.

    Card deletion, direct drags, grouped drags and temporary parking have the
    same result. Only known held cards that now occupy their own discard pile
    may satisfy this obligation; no unseen card or extra discard is selected.
    """
    if (getattr(driver, 'spectator', False) or not driver.view.dump_path
            or driver.deferred is not None or driver.chat_deferred is not None
            or getattr(driver.seg, 'window', None) is not None):
        return False
    import ai
    pending = ai.describe_pending(driver.es)
    if (not pending or pending['kind'] != 'discard'
            or pending['side'] is not driver.human_side):
        driver._discard_endpoint_trace = None
        return False
    key = (driver.es.turn, driver.n_commits_total, pending['side'].name,
           tuple(driver.es.hands[pending['side']].hand))
    trace = getattr(driver, '_discard_endpoint_trace', None)
    if trace is None or trace[0] != key:
        trace = (key, [])
        driver._discard_endpoint_trace = trace
    from .live_parser import parse
    for line in new_lines:
        record = parse(line)
        if (record.get('op') in ('move', 'kill', 'flip')
                and record.get('player') != driver.nick):
            norm = record.get('norm', line)
            if norm and norm not in trace[1]:
                trace[1].append(norm)
    from .decision_endpoint import read_complete
    from .repair_evidence import same_dump
    try:
        dump, _markers, evidence = read_complete(driver)
    except (OSError, ValueError):
        return False
    # Undo followed by a replacement draw can share this board export. Repair
    # only its authenticated identity swap before identifying the payment.
    from .hand_redraw import reconcile
    reconcile(driver)
    result = plan(driver, dump, pending)
    if result.status != 'match' or not same_dump(driver, evidence):
        return False
    original = driver.pending_buf
    norms = list(trace[1])
    for record in getattr(original, 'records', ()):
        if record.get('norm') and record['norm'] not in norms:
            norms.append(record['norm'])
    # Keep native receipt strings for Undo/checkpoint provenance without
    # counting intermediate drags as extra payments in the canonical facts.
    result.window.records.extend(dict(op='noop', norm=norm) for norm in norms)
    accepted, reason = driver._commit_recovered_combat(
        result, evidence, label='pending:discard')
    if accepted:
        driver._discard_endpoint_trace = None
        driver._buf_stall = None
        driver._last_pending_match = result.match
        driver._ensure_declare_armed()
        driver.findings.append(('hand_discard_endpoint_committed', reason))
    return accepted
