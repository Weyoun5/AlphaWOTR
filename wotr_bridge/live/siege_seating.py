"""Physical completion of an already chosen stronghold withdrawal."""


def waiting(driver):
    region = getattr(driver, '_siege_seat_owed', None)
    if not region:
        return False
    from .shadow_siege_slots import garrison_seated
    if (driver.view.refresh_dump()
            and garrison_seated(driver.view, driver.es, region, driver.human_side)):
        driver._siege_seat_owed = None
        driver.findings.append(('siege_seat_complete', str(region)))
        return False
    return True


def consume(driver, record):
    """Seating belongs to its established siege, never to the next choice.

    Only the human garrison's field -> same stronghold movement is a
    receipt. Losses, returns, different regions and attacking pieces retain
    their ordinary routing. No engine action or casualty is applied here.
    """
    from board import sanitize
    from combat import garrison_side_for, side_of_nation
    from game_state import Side, Nation
    from . import gesture_match as gm

    if getattr(driver, 'spectator', False):
        return False
    receipts = getattr(driver, '_siege_seat_receipts', None) or []
    redo = getattr(driver, '_siege_seat_redo', None) or []
    context = (driver.es.turn, getattr(driver, '_cc_battle_serial', 0))
    op = record.get('op')
    if op in ('undo', 'redo'):
        from . import live_parser
        _, norm = live_parser.normalize(str(record.get('inner', '')).strip())
        if op == 'redo':
            if (not redo or (context, norm) != redo[-1]['receipt'][:2]
                    or driver.n_commits_total != redo[-1]['commit']):
                driver._siege_seat_redo = []
                return False
            entry = redo.pop()
            receipt = entry['receipt']
            driver._siege_seat_receipts = (list(receipts) + [receipt])[-32:]
            driver._siege_seat_redo = redo
            # Native Undo can remove a shared-box projection during census.
            # Restore only origins witnessed in this exact undone movement;
            # the fresh dump still removes any piece no longer in that box.
            bindings = dict(getattr(driver.view, 'shadow_stronghold_regions', {}) or {})
            bindings.update({pid: (region, code) for pid, region, code in receipt[3]})
            driver.view.shadow_stronghold_regions = bindings
            driver._siege_seat_owed = receipt[2]
            waiting(driver)
            driver.findings.append(('siege_seat_redone', norm))
            return True
        if not receipts or (context, norm) != receipts[-1][:2]:
            driver._siege_seat_redo = []
            return False
        receipt = receipts.pop()
        driver._siege_seat_receipts = receipts
        region = receipt[2]
        driver._siege_seat_redo = (list(redo) + [
            dict(receipt=receipt, commit=driver.n_commits_total)])[-32:]
        if (region in {sanitize(str(r)) for r in driver.es.besieged_strongholds}
                and garrison_side_for(driver.es, region) is driver.human_side):
            driver._siege_seat_owed = region
        driver.findings.append(('siege_seat_undone', norm))
        return True
    if op in ('move', 'kill', 'flip', 'draw', 'alloc_eye', 'roll_dice', 'roll_num', 'choose'):
        driver._siege_seat_redo = []
    if (op != 'move' or str(record.get('player', '')) == driver.nick):
        return False
    source, target = gm.zone_key(record.get('from')), gm.zone_key(record.get('to'))
    if source[0] != 'region' or target != ('stronghold', source[1]):
        return False
    region = source[1]
    pc = driver.es.pending_combat or {}
    advancing = (pc.get('phase') == 'victor_advance' and pc.get('post_siege_retreat')
                 and sanitize(str(pc.get('defender_region', ''))) == region
                 and pc.get('defender_side') is driver.human_side)
    established = region in {sanitize(str(r)) for r in driver.es.besieged_strongholds}
    if not (advancing or established):
        return False
    if not advancing and garrison_side_for(driver.es, region) is not driver.human_side:
        return False
    obj = str(record.get('obj', ''))
    tokens = gm.rr.split_group(obj[7:-1]) if obj.startswith('group (') and obj.endswith(')') else [obj]
    if not tokens:
        return False
    for token in tokens:
        piece = gm.rr.parse_piece(token)
        if piece is None or piece[3] <= 0:
            return False
        kind, name, _, _ = piece
        if kind in ('unit', 'leader'):
            side = side_of_nation(Nation[name])
        elif kind == 'nazgul':
            side = Side.SHADOW
        elif kind == 'named':
            from .emission_card import _named_pid_side
            _, owner = _named_pid_side(name)
            side = Side[owner] if owner else None
        else:
            return False
        if side is not driver.human_side:
            return False
    driver.findings.append(('siege_seat_performed', record.get('norm', '')))
    receipts = list(receipts)
    receipts.append((context, record.get('norm', ''), region,
                     tuple(record.get('_shadow_slot_bindings', ()))))
    driver._siege_seat_receipts = receipts[-32:]
    waiting(driver)
    return True
