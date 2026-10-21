"""Ownership and input budget for an actually issued human roll request."""


def _call_key(call):
    if call is None or getattr(call, 'kind', None) != 'd6' or not hasattr(call, '_request_cursor'):
        return None
    return (call.kind, tuple(sorted((key, str(value)) for key, value in call.kw.items())),
            repr(call._request_cursor))


def _window_key(driver):
    pc = getattr(driver.es, 'pending_combat', None) or {}
    return (driver._pending_rng_signature(), repr(pc.get('effect_choice')))


def _starved_match(driver):
    result = getattr(driver, '_last_pending_match', None)
    return (result is not None and result.status == 'hold' and result.starved
            and not getattr(result, 'proven_starve_conflict', False)
            and driver.pending_buf is not None and bool(driver.pending_buf.records))


def remember_issued(server, call, emitted):
    """Called only after an actual outgoing roll ask has been constructed."""
    driver = server.driver
    if not emitted or _call_key(call) is None:
        return
    from .randomness import _side_name
    if _side_name(call.kw.get('side')) != _side_name(server.human_side):
        return
    if _call_key(driver.pending_random_call()) != _call_key(call):
        return
    mark = len(driver.feed.lines)
    asks = [(nonce, entry) for nonce, entry in server._live_asks.items()
            if entry.get('kind') == 'd6' and entry.get('need') == call.kw.get('need')
            and entry.get('feed_mark') == mark and entry.get('epoch') == driver.n_commits_total]
    if not asks:
        return
    nonce = max(nonce for nonce, _ in asks)
    receipt = getattr(driver, '_live_human_roll_receipt', None)
    if (getattr(server, 'script', None) is None
            and (receipt is None or receipt['nonce'] != nonce)):
        # Keep the exhausted receipt until another ask is actually emitted:
        # a second native paragraph must not become next round's dice.
        # Corpus prefeeding without a live request retains its semantics.
        driver._live_human_roll_receipt = {
            'nonce': nonce, 'remaining': int(call.kw['need']),
            'feed_mark': mark, 'call': _call_key(call),
        }
    server._issued_human_roll_wait = {
        'driver_id': id(driver), 'window': _window_key(driver),
        'call': _call_key(call), 'feed_mark': mark,
        'nonce': max(nonce for nonce, _ in asks), 'epoch': driver.n_commits_total,
    }
    if not _starved_match(driver):
        return
    driver._issued_pending_human_roll = {'window': _window_key(driver), 'call': _call_key(call),
                                         'feed_mark': mark, 'nonce': max(nonce for nonce, _ in asks)}


def admit_human_roll(driver, record):
    """Return the requested prefix and ignored rightmost faces, without feeding."""
    values = list(record.get('ints', ()))
    receipt = getattr(driver, '_live_human_roll_receipt', None)
    from .randomness import _side_name
    side = _side_name(driver._d6_roll_side(record))
    if (not receipt or getattr(driver, 'spectator', False)
            or str(record.get('player', '')) == driver.nick
            or side not in (None, _side_name(driver.human_side))):
        return values, []
    if any(not isinstance(value, int) or not 1 <= value <= 6 for value in values):
        return values, []  # the funnel must still reject invalid native outcomes
    keep = min(len(values), max(0, receipt['remaining']))
    receipt['remaining'] -= keep
    return values[:keep], values[keep:]


def retire_obsolete_roll(server):
    """Explicit retry may retire proven surplus from an already closed round.

    This supports saved pre-fix sessions. It requires the actual emitted ask,
    its rendered answer and extra faces, and exact residual queue agreement.
    An unfinished random continuation or an ambiguous tape is left intact.
    """
    driver, rand = server.driver, server.rand
    pc = getattr(driver.es, 'pending_combat', None) or {}
    if (getattr(server, 'script', None) is not None
            or pc.get('phase') not in ('combat_round_result', 'attacker_continue', 'defender_retreat')
            or pc.get('_round_resume') is not None or rand.pending() is not None
            or rand._feed_pos['d6'] or driver.chat_deferred is not None
            or driver.deferred is not None):
        return []
    asks = [(nonce, row) for nonce, row in server._live_asks.items()
            if row.get('kind') == 'd6' and row.get('epoch', driver.n_commits_total)
            < driver.n_commits_total]
    if not asks:
        return []
    nonce, ask = max(asks)
    from .live_parser import parse
    from .randomness import _side_name
    values = []
    for line in driver.feed.lines[ask['feed_mark']:driver._feed_seq]:
        rec = parse(str(line))
        if (rec.get('op') == 'roll_num' and rec.get('player') != driver.nick
                and _side_name(driver._d6_roll_side(rec)) in
                (None, _side_name(driver.human_side))):
            values.extend(rec.get('ints', ()))
    surplus = values[int(ask['need']):]
    candidates = [(index, value) for index, (side, value) in enumerate(rand._feed['d6'])
                  if side in (None, _side_name(driver.human_side))]
    if not surplus or [value for _, value in candidates] != surplus:
        return []
    remove = {index for index, _ in candidates}
    rand._feed['d6'] = [entry for index, entry in enumerate(rand._feed['d6'])
                       if index not in remove]
    driver._live_human_roll_receipt = {'nonce': nonce, 'remaining': 0,
                                      'feed_mark': ask['feed_mark'], 'call': None}
    driver.findings.append(('obsolete_human_dice_ignored', tuple(surplus)))
    return [('chat', 'excess_dice', 'roll already complete; ignoring extra dice: %s'
             % ' '.join(map(str, surplus)))]


def issued_waiting(server):
    """An actual unanswered human ask, including deferred card-reveal rolls.

    This is only diagnostic ownership. It cannot legalize a gesture, hide an
    already rendered result, or discard an outstanding AI acknowledgement.
    """
    driver = server.driver
    proof = getattr(server, '_issued_human_roll_wait', None)
    if (not proof or proof['driver_id'] != id(driver)
            or proof['window'] != _window_key(driver)
            or proof['epoch'] != driver.n_commits_total):
        return None
    call = driver.pending_random_call()
    if _call_key(call) != proof['call']:
        return None
    ask = (getattr(server, '_live_asks', None) or {}).get(proof['nonce'])
    if (not ask or ask.get('kind') != 'd6' or ask.get('need') != call.kw.get('need')
            or ask.get('feed_mark') != proof['feed_mark'] or ask.get('epoch') != proof['epoch']):
        return None
    if server._rng_result_rendered('d6', driver.feed.lines[proof['feed_mark']:]) is not None:
        return None
    return call


def waiting(driver):
    """Return the current issued call only while its result is still absent."""
    proof = getattr(driver, '_issued_pending_human_roll', None)
    if not proof or not _starved_match(driver) or proof['window'] != _window_key(driver):
        return None
    call = driver.pending_random_call()
    if _call_key(call) != proof['call']:
        return None
    # A rendered outcome that was not harvested must remain a diagnostic
    # failure. Do not suppress it or issue another roll through this helper.
    from .live_parser import parse
    for line in driver.feed.lines[proof['feed_mark']:]:
        try:
            if parse(str(line)).get('op') == 'roll_num':
                return None
        except (ValueError, TypeError, KeyError):
            return None
    return call
