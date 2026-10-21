"""Describe a pending physical discard identity check without accepting a card."""


def _key(driver, pending):
    from . import gesture_match
    import game_logic
    buf = driver.pending_buf
    if (not pending or pending.get('kind') != 'discard'
            or pending.get('side') is not driver.human_side
            or buf is None or not buf.records):
        return None
    side = driver.human_side
    zone = ('hand', 'FP' if side.name == 'FREE_PEOPLES' else 'SH')
    if any(rec.get('op') != 'kill'
           or gesture_match.zone_key(rec.get('zone')) != zone
           for rec in buf.records):
        return None
    facts = gesture_match.window_facts(buf).hand_discards
    hand = tuple(driver.es.hands[side].hand)
    count = sum(facts.values())
    if not 0 < count <= len(hand) - game_logic.HAND_LIMIT:
        return None
    legal = [str(a.params.get('card_id')) for a in pending.get('actions', ())
             if a.action_type == 'discard_card']
    for (owner, deck), amount in facts.items():
        if (owner != side.name or deck is None
                or sum(game_logic.card_deck_of(cid) == deck for cid in legal) < amount):
            return None
    return (id(driver.es), driver.es.turn, side, hand, repr(buf.records))


def remember(driver, pending):
    driver.pending_buf._discard_wait_status_key = _key(driver, pending)


def waiting(driver):
    buf = driver.pending_buf
    proof = getattr(buf, '_discard_wait_status_key', None)
    if proof is None or not getattr(buf, '_awaits_discard_identity', False):
        return False
    import ai
    return proof == _key(driver, ai.describe_pending(driver.es))


def text(driver):
    if waiting(driver):
        return ('your discard is recorded; waiting for the client board snapshot '
                'to identify the discarded card')
    return None
