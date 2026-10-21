"""Reconcile replacement draws after native Undo from the settled human hand.

Undo may retract a physical draw inside an already committed event without
retracting the event's other effects.  A subsequent draw then replaces that
grant, rather than creating an extra hand slot.  Use authenticated draw IDs
and the actual hand/deck endpoints; never choose an unknown card by count.
"""


def reconcile(driver):
    model = driver.hands
    if not getattr(model, '_native_draw_undo_seen', False):
        return False
    if (getattr(driver, 'spectator', False) or driver.seg.window is not None
            or driver.deferred is not None or driver.chat_deferred is not None
            or getattr(driver, 'holds', ())):
        return False
    import ai
    pending = ai.describe_pending(driver.es)
    if pending and pending.get('kind') not in ('discard', 'declare'):
        return False
    buf = driver.pending_buf
    if buf is not None and (not pending or pending.get('side') is not driver.human_side):
        return False

    side = driver.human_side
    tag = 'FP' if side.name == 'FREE_PEOPLES' else 'SH'
    hand = model.es.hands[side].hand
    mapping = dict(driver.view.card_pid_map)
    observations = {}
    for entry in getattr(model, '_draw_receipts', ()):
        decoded = entry.get('observation')
        if (decoded and decoded.get('cid') and not entry.get('undone')
                and not entry.get('cancelled') and decoded.get('side') == side.name):
            mapping[decoded['cid']] = str(decoded['pid'])
            observations[decoded['cid']] = entry
    if any(str(cid).startswith('HU_') or cid not in mapping for cid in hand):
        return False

    # A fresh complete export at the routed feed boundary is the witness.
    from .repair_evidence import dump_signature, same_dump
    from .client_protocol import dump_at_feed_position
    import copy
    try:
        evidence = dump_signature(driver)
        header = evidence['header']
        if not header or int(header.get('session', 0)) <= 0:
            return False
        driver._check_dump_session(header)
        reader = copy.deepcopy(driver.feed)
        reader.poll()
        if (reader._buf or driver._feed_seq != len(reader.lines)
                or not dump_at_feed_position(header, len(reader.lines))
                or not driver.view.refresh_dump() or not same_dump(driver, evidence)):
            return False
    except (OSError, ValueError, TypeError):
        return False
    zones = driver.view.piece_region
    hand_zone = 'FP Hand' if tag == 'FP' else 'SA Hand'
    dead_zone = 'FP Dead' if tag == 'FP' else 'SA Dead'
    in_hand = {str(pid) for pid, zone in zones.items() if zone == hand_zone}
    inverse = {}
    for cid, pid in mapping.items():
        if str(pid) in inverse and inverse[str(pid)] != cid:
            return False
        inverse[str(pid)] = cid
    if not in_hand or any(pid not in inverse for pid in in_hand):
        return False
    physical = {inverse[pid] for pid in in_hand}
    if any(not cid.startswith(side.name + '_') for cid in physical):
        return False
    # A current hand-limit payment is already on the board, but still awaits
    # engine commit. Retain exactly those proven discard cards in the prestate.
    paid = {cid for cid in hand if zones.get(str(mapping[cid])) == dead_zone}
    if paid:
        import game_logic as gl
        available = {action.params.get('card_id') for action in pending.get('actions', ())
                     if action.action_type == 'discard_card'} if pending else set()
        if (not pending or pending.get('kind') != 'discard'
                or pending.get('side') is not side or not paid <= available
                or len(paid) > len(hand) - gl.HAND_LIMIT):
            return False
    actual = physical | paid
    if len(actual) != len(hand):
        return False                    # An overdraw is not an identity repair.
    old, new = set(hand) - actual, actual - set(hand)
    if not old or len(old) != len(new) or any(cid not in observations for cid in new):
        return False
    import game_logic as gl
    from collections import Counter
    if Counter(gl.card_deck_of(cid) for cid in old) != Counter(gl.card_deck_of(cid) for cid in new):
        return False
    replacements = []
    for deck in ('character', 'strategy'):
        removed = sorted(cid for cid in old if gl.card_deck_of(cid) == deck)
        added = sorted(cid for cid in new if gl.card_deck_of(cid) == deck)
        attr = ('fp' if tag == 'FP' else 'sh') + ('_char_deck' if deck == 'character' else '_strat_deck')
        engine_deck = getattr(driver.es, attr)
        native_deck = ('FP' if tag == 'FP' else 'SA') + (' Char Deck' if deck == 'character' else ' Strat Deck')
        for previous, current in zip(removed, added):
            if zones.get(str(mapping[previous])) != native_deck:
                return False
            if current not in engine_deck or previous in engine_deck:
                return False
            replacements.append((previous, current, engine_deck))
    for previous, current, deck in replacements:
        hand[hand.index(previous)] = current
        deck[deck.index(current)] = previous
        entry = observations[current]
        entry['resolved'] = True
        entry['board_reconciled'] = True
        driver.view.card_pid_map[current] = mapping[current]
        entry['published'] = True
    driver.findings.append(('human_redraw_reconciled', '%d returned draw identity(s)' % len(replacements)))
    return True
