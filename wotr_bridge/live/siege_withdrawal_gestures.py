"""Keep an early garrison reserve return across the attacker's real choice."""
from collections import Counter


def _nonzero(rows):
    return {key: value for key, value in rows.items() if value}


def _move_rows(record, inverse=False):
    from . import gesture_match as gm
    if record.get('op') != 'move':
        return None
    obj = str(record.get('obj', ''))
    tokens = gm.rr.split_group(obj[7:-1]) if obj.startswith('group (') and obj.endswith(')') else [obj]
    pieces = [gm.rr.parse_piece(token) for token in tokens]
    if not pieces or any(p is None or p[0] not in ('unit', 'leader', 'nazgul', 'named') for p in pieces):
        return None
    src, dst = gm.zone_key(record.get('from')), gm.zone_key(record.get('to'))
    if inverse:
        src, dst = dst, src
    rows = Counter()
    for kind, first, second, count in pieces:
        if count <= 0:
            return None
        rows[(src, dst, (kind, first, second))] += count
    return rows


def _stayed_here(driver, retained):
    """A committed actual attacker stay, not merely an absent combat pending."""
    from board import sanitize
    if not driver.commits:
        return False
    entry = driver.commits[-1]
    action = entry[4] if len(entry) > 4 else None
    pc = getattr(entry[0], 'pending_combat', None) or {}
    return (getattr(action, 'action_type', None) == 'victor_stay'
            and str(entry[2]).startswith('ai:')
            and pc.get('phase') == 'victor_advance' and pc.get('post_siege_retreat')
            and pc.get('attacker_side') is driver.ai_side
            and pc.get('defender_side') is driver.human_side
            and sanitize(str(pc.get('defender_region', ''))) == retained['region'])


def _corrected_endpoint(driver, buf, retained, record=None):
    """An actual restored board completes correction, whatever its drag route."""
    if not retained.get('endpoint') or not retained.get('attacker_stayed'):
        return False
    if record is not None:
        retained['last_correction_record'] = record
    record = retained.get('last_correction_record')
    if record is None:
        return False
    from .decision_endpoint import read_complete
    from . import combat_recovery as recovery, repair_evidence
    revision = driver._decision_endpoint_revision()
    try:
        dump, _markers, signature = read_complete(driver)
        expected, labels, owners = recovery._engine_counts(driver.es)
        if (expected != recovery._dump_counts(dump, driver.view.dynamic_types, labels, owners)
                or not recovery._permanent_losses_match(driver.es, driver.human_side,
                                                       dump, driver.view.dynamic_types)
                or not repair_evidence.same_dump(driver, signature)):
            return False
    except (OSError, ValueError):
        buf._endpoint_revision = revision
        driver._tag_hold(buf, 'decision_endpoint')
        return False
    retained.setdefault('corrections', []).append(
        {'record': record, 'rows': _move_rows(record) or Counter(), 'pids': {}})
    retained['remaining'], retained['remaining_pids'] = Counter(), {}
    driver.pending_buf = None
    driver._siege_correction_receipt = (buf, retained)
    driver._buf_stall = None
    driver._last_pending_match = None
    driver.findings.append(('early_siege_return_corrected', retained['region']))
    return True


def consume_correction(driver, record):
    """Accept only the saved physical inverse after the actual attacker stays.

    This changes evidence bookkeeping, never engine state, adoption or RNG.
    Exact native Undo also works while correcting, including its final line.
    """
    from . import gesture_match as gm, live_parser
    from .native_die_identity import pids as record_pids
    from .shadow_siege_slots import _same_pieces

    if getattr(driver.seg, 'window', None) is not None:
        return None
    buf = driver.pending_buf
    retained = getattr(buf, '_siege_awaiting_advance', None)
    receipt = getattr(driver, '_siege_correction_receipt', None)
    if buf is None and record.get('op') == 'undo' and receipt is not None:
        buf, retained = receipt
    if (retained and not retained.get('attacker_stayed')
            and driver.es.pending_combat is None and _stayed_here(driver, retained)):
        # A physical correction can render before the first idle tick after
        # the AI reply; the same committed choice authenticates that order.
        retained['attacker_stayed'] = True
        retained['correction_state'] = driver.es
    if (not retained or not retained.get('attacker_stayed')
            or retained.get('correction_state') is not driver.es):
        return None
    done = retained.setdefault('corrections', [])
    if record.get('op') == 'undo':
        _, norm = live_parser.normalize(str(record.get('inner', '')).strip())
        if not done or norm != done[-1]['record'].get('norm'):
            return None
        undone = done.pop()
        retained['remaining'].update(undone['rows'])
        retained['remaining_pids'].update(undone['pids'])
        driver.pending_buf = buf
        driver._siege_correction_receipt = None
        return []
    if record.get('op') != 'move' or driver.pending_buf is not buf:
        return None
    if _corrected_endpoint(driver, buf, retained, record):
        return []
    if retained.get('endpoint') and _move_rows(record):
        # A restoration can take several drags. Keep it in this physical
        # correction until the complete board agrees; it is not a new unpaid
        # action just because the first leg uses a different return route.
        retained.setdefault('remaining', Counter())
        retained.setdefault('remaining_pids', {})
        retained.setdefault('corrections', []).append(
            {'record': record, 'rows': _move_rows(record), 'pids': {}})
        return []
    rows = _move_rows(record)
    if not rows:
        return None
    if 'remaining' not in retained:
        remaining, expected_pids = Counter(), {}
        for original in list(buf.records) + retained['seats']:
            inverse = _move_rows(original, inverse=True)
            if not inverse:
                return None
            remaining.update(inverse)
            route = (gm.zone_key(original.get('to')), gm.zone_key(original.get('from')))
            for pid in record_pids(original) or ():
                expected_pids[str(pid)] = route
        retained['remaining'] = remaining
        retained['remaining_pids'] = expected_pids
    if any(count > retained['remaining'][key] for key, count in rows.items()):
        return None
    pids = [str(pid) for pid in record_pids(record) or ()]
    route = (gm.zone_key(record.get('from')), gm.zone_key(record.get('to')))
    if pids and (len(pids) != len(set(pids))
                 or not _same_pieces(driver.view, pids, str(record.get('obj', '')))
                 or (retained['remaining_pids'] and any(
                     retained['remaining_pids'].get(pid) != route for pid in pids))):
        return None
    known_pids = {pid: retained['remaining_pids'][pid] for pid in pids
                  if pid in retained['remaining_pids']}
    for pid in known_pids:
        del retained['remaining_pids'][pid]
    retained['remaining'].subtract(rows)
    done.append({'record': record, 'rows': rows, 'pids': known_pids})
    if not any(retained['remaining'].values()):
        driver.pending_buf = None
        driver._siege_correction_receipt = (buf, retained)
        driver._buf_stall = None
        driver._last_pending_match = None
        driver.findings.append(('early_siege_return_corrected', retained['region']))
    return []


def endpoint_record_needed(driver, record):
    """Keep zero-map-delta legs of the current garrison's physical decision."""
    if getattr(driver, 'spectator', False) or record.get('op') != 'move':
        return False
    pc = getattr(driver.es, 'pending_combat', None) or {}
    if (pc.get('phase') != 'defender_siege_choice'
            or pc.get('defender_side') is not driver.human_side
            or driver.pending_buf is None or not driver.pending_buf.records):
        return False
    rows = _move_rows(record)
    return bool(rows and all(piece[0] in ('unit', 'leader', 'nazgul', 'named')
                             for (_source, _target, piece) in rows))


def _endpoint_records(driver, dp):
    """Prove the whole garrison endpoint, independent of temporary parking.

    This supplies evidence for the existing withdrawal/trim split; it never
    chooses the attacker's advance or removes reserve units from the engine.
    """
    import ai
    import re
    from . import gesture_match as gm, combat_recovery as recovery
    from .decision_endpoint import read_complete
    from . import repair_evidence
    from board import sanitize
    pc = getattr(driver.es, 'pending_combat', None) or {}
    buf = driver.pending_buf
    if (dp is None or dp['kind'] != 'combat:defender_siege_choice'
            or dp['side'] is not driver.human_side or buf is None or not buf.records
            or pc.get('attacker_side') is not driver.ai_side):
        return None
    region = sanitize(str(pc.get('defender_region', '')))
    if region == sanitize(str(pc.get('attacker_region', ''))):
        return None
    reserve = 'FP Reinforcements' if driver.human_side.name == 'FREE_PEOPLES' else 'SA Reinforcements'
    destination_names = {gm.zone_key(reserve): reserve, ('stronghold', region): region + ' Stronghold'}
    flow, labels, last = Counter(), {}, {}
    for record in buf.records:
        rows = _move_rows(record)
        if rows is None:
            return None
        obj = str(record.get('obj', ''))
        tokens = gm.rr.split_group(obj[7:-1]) if obj.startswith('group (') and obj.endswith(')') else [obj]
        for token in tokens:
            piece = gm.rr.parse_piece(token)
            labels[piece[:3]] = re.sub(r'^\[\d+\]\s*', '', token.strip())
        for (source, target, piece), count in rows.items():
            flow[(piece, source)] -= count
            flow[(piece, target)] += count
            last[(piece, target)] = record
    flow = Counter({key: count for key, count in flow.items() if count})
    if not flow or any((count < 0 and zone != ('region', region)) or
                       (count > 0 and zone not in destination_names)
                       for (_piece, zone), count in flow.items()):
        return None
    moves = []
    for (piece, target), count in flow.items():
        if count <= 0:
            continue
        label = labels[piece]
        record = dict(last[(piece, target)])
        record.update(obj=(f'[{count}]{label}' if count > 1 else label),
                      **{'from': region, 'to': destination_names[target]})
        record.pop('meta', None)  # A grouped raw PID list is not this derived single-type row.
        moves.append(record)
    if not moves:
        return None
    revision = driver._decision_endpoint_revision()
    try:
        dump, _markers, signature = read_complete(driver)
        expected, figure_labels, owners = recovery._engine_counts(driver.es)
        actual = recovery._dump_counts(dump, driver.view.dynamic_types, figure_labels, owners)
        for (piece, zone), count in flow.items():
            if zone[0] not in ('region', 'stronghold'):
                continue
            name = recovery._key(labels[piece])
            place = zone[1] + (' stronghold' if zone[0] == 'stronghold' else '')
            expected[(place, name)] += count
        if (+expected != actual or any(value < 0 for value in expected.values())
                or not recovery._permanent_losses_match(driver.es, driver.human_side,
                                                       dump, driver.view.dynamic_types)
                or not repair_evidence.same_dump(driver, signature)):
            return None
    except (OSError, ValueError):
        buf._endpoint_revision = revision
        driver._tag_hold(buf, 'decision_endpoint')
        return None
    return moves


def _split(driver, dp, records=None):
    import ai
    import game_logic as gl
    from board import sanitize
    from . import gesture_match as gm

    state, buf = driver.es, driver.pending_buf
    pc = state.pending_combat or {}
    if (dp is None or dp['kind'] != 'combat:defender_siege_choice'
            or dp['side'] is not driver.human_side
            or pc.get('attacker_side') is not driver.ai_side
            or buf is None or not buf.records):
        return None
    region = sanitize(str(pc.get('defender_region', '')))
    if region == sanitize(str(pc.get('attacker_region', ''))):
        return None  # Co-located combat has no intervening attacker choice.
    army = ai._armies_lookup(state, region)
    if army is None:
        return None
    garrison = [u for u in army.units if ai._side_of_nation(u.nation) is driver.human_side]
    excess = len(garrison) - gl.MAX_SIEGE_UNITS
    if excess <= 0:
        return None
    reserve = 'FP Reinforcements' if driver.human_side.name == 'FREE_PEOPLES' else 'SA Reinforcements'
    reserve_zone = gm.zone_key(reserve)
    returns, seats, lifted = [], [], []
    for record in (buf.records if records is None else records):
        if record.get('op') != 'move' or gm.zone_key(record.get('from')) != ('region', region):
            return None
        destination = gm.zone_key(record.get('to'))
        if destination == reserve_zone:
            obj = str(record.get('obj', ''))
            tokens = gm.rr.split_group(obj[7:-1]) if obj.startswith('group (') and obj.endswith(')') else [obj]
            pieces = [gm.rr.parse_piece(token) for token in tokens]
            if not pieces or any(p is None or p[0] != 'unit' for p in pieces):
                return None  # Capacity returns units alive, never leaders/cards.
            returns.append(record)
        elif destination == ('stronghold', region):
            seats.append(record)
        else:
            return None
        lifted.append(dict(record, to=reserve))
    if not returns or not seats:
        return None
    returned = driver._new_phase_window()
    returned.records = list(returns)
    loss = gm.window_facts(returned)
    if loss.unmapped or -sum(loss.armies.values()) != excess:
        return None
    expected_units = Counter((region, gm._unit_type_key(u.nation, u.kind)) for u in garrison)
    if any(value >= 0 or -value > expected_units[key] for key, value in loss.armies.items() if value):
        return None
    # Seat plus return must account for the exact garrison, including leaders.
    # The hypothetical projection only measures its roster; no action or AI
    # choice is executed to authorize the later trim.
    lifted_window = driver._new_phase_window()
    lifted_window.records = lifted
    physical = gm.window_facts(lifted_window)
    empty = ai.clone_state(state)
    empty_army = ai._armies_lookup(empty, region)
    empty_army.units = [u for u in empty_army.units if ai._side_of_nation(u.nation) is not driver.human_side]
    empty_army.leaders = [l for l in empty_army.leaders if gl._leader_side(l) is not driver.human_side]
    expected = gm.delta_facts(state, empty)
    if physical.unmapped or any(_nonzero(getattr(physical, key)) != _nonzero(getattr(expected, key))
                               for key in ('armies', 'leaders', 'named')):
        return None
    seated = driver._new_phase_window()
    seated.records = list(seats)
    result = gm.match_window(state, dp['side'], seated,
                            menu=[a for a in dp['actions'] if a.action_type == 'retreat_into_siege'],
                            pin=driver._attempt_pin(), rng=driver.rctl, advance_tolerance=True)
    if result.status != 'match' or [a.action_type for a in result.chain] != ['retreat_into_siege']:
        return None
    next_choice = ai.describe_pending(result.post)
    if (next_choice is None or next_choice['kind'] != 'combat:victor_advance'
            or next_choice['side'] is not driver.ai_side):
        return None
    return region, seated, returned, result


def handle_pending(driver, dp):
    """None delegates to the normal matcher; bool owns this pending attempt."""
    if getattr(driver, 'spectator', False):
        return None
    buf = driver.pending_buf
    retained = getattr(buf, '_siege_awaiting_advance', None)
    if retained:
        from board import sanitize
        pc = driver.es.pending_combat or {}
        same = sanitize(str(pc.get('defender_region', ''))) == retained['region']
        if (same and dp is not None and dp['side'] is driver.human_side
                and dp['kind'] in ('combat:siege_trim', 'combat:defender_siege_choice')):
            # The real advance opened trim, or native Undo restored the
            # withdrawal decision. The ordinary legal matcher owns both.
            del buf._siege_awaiting_advance
            return None
        if (same and dp is not None and dp['kind'] == 'combat:victor_advance'
                and dp['side'] is driver.ai_side and pc.get('post_siege_retreat')):
            return False
        if _stayed_here(driver, retained):
            retained['attacker_stayed'] = True
            retained['correction_state'] = driver.es
        if _corrected_endpoint(driver, buf, retained):
            return True
        reason = ('The attacker did not establish this siege. Restore the early '
                  'reserve return and move the garrison out of the %s Stronghold; '
                  'the engine has not removed those units.' % retained['region'])
        if not retained.get('correction'):
            retained['correction'] = reason
            if _stayed_here(driver, retained):
                retained['attacker_stayed'] = True
                retained['correction_state'] = driver.es
            driver.findings.append(('early_siege_return_needs_correction', reason))
            driver._out_queue.append(('note', reason))
            driver._out_queue.extend(driver._propose_repair(reason))
        return False
    planned = _split(driver, dp)
    endpoint_norms = None
    original_returns = None
    if planned is None:
        records = _endpoint_records(driver, dp)
        if records is not None:
            planned = _split(driver, dp, records)
            if planned is not None:
                from . import gesture_match as gm
                region = planned[0]
                seat_records = [r for r in buf.records
                                if gm.zone_key(r.get('to')) == ('stronghold', region)]
                endpoint_norms = [r.get('norm', '') for r in seat_records]
                # The original return route remains independently undoable.
                # It is a later trim, not part of the withdrawal's commit.
                candidate = driver._new_phase_window()
                candidate.records = [r for r in buf.records if r not in seat_records]
                actual, derived = gm.window_facts(candidate), gm.window_facts(planned[2])
                if not actual.unmapped and all(_nonzero(getattr(actual, key)) ==
                                              _nonzero(getattr(derived, key))
                                              for key in ('armies', 'leaders', 'named', 'sieges')):
                    original_returns = candidate.records
    if planned is None:
        return None
    region, seated, returned, result = planned
    if not driver._commit_chain(result.chain, endpoint_norms or [r.get('norm', '') for r in seated.records],
                                'pending:combat:defender_siege_choice', gesture=seated):
        return False
    if original_returns is not None:
        returned.records = original_returns
    returned._siege_awaiting_advance = {'region': region, 'seats': list(seated.records)}
    if endpoint_norms is not None:
        returned._siege_awaiting_advance['endpoint'] = True
    driver.pending_buf = returned
    driver._last_pending_match = result
    driver.pending_stats['matched'] += 1
    driver._bury_dead_candidate_asks()
    driver.findings.append(('siege_return_retained', '%s: waiting for the attacker advance decision' % region))
    return True


def retry(driver):
    """Resume only the saved human evidence after an actual engine decision."""
    buf = driver.pending_buf
    retained = getattr(buf, '_siege_awaiting_advance', None)
    if not retained:
        return False
    driver._try_pending()
    if (driver.pending_buf is None
            and getattr(driver, '_siege_seat_owed', None) == retained['region']):
        from .shadow_siege_slots import garrison_seated
        driver.view.refresh_dump()
        if garrison_seated(driver.view, driver.es, retained['region'], driver.human_side):
            driver._siege_seat_owed = None
    return bool(getattr(driver.pending_buf, '_siege_awaiting_advance', None))


def trim_before_seating_prompt(driver, post, region):
    """The saved five-unit seating cannot satisfy a transient six-unit census."""
    from board import sanitize
    retained = getattr(driver.pending_buf, '_siege_awaiting_advance', None)
    pc = post.pending_combat or {}
    return bool(retained and retained['region'] == sanitize(str(region))
                and pc.get('phase') == 'siege_trim'
                and pc.get('defender_side') is driver.human_side)
