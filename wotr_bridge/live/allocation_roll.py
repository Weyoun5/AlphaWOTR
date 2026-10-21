"""Close a human Hunt allocation at its actual native roll boundary."""


def defer(driver, rec, pids):
    """Retain the observed faces while the matching board export arrives."""
    if (rec.get('_allocation_closed') or driver.phase != 'eyes'
            or driver.human_side.name != 'SHADOW' or getattr(driver, 'spectator', False)
            or not driver.view.dump_path
            or not (pids or (driver.sidecar and driver.sidecar.present))):
        return False
    pending = getattr(driver, '_allocation_roll_wait', None)
    if pending is None:
        pending = driver._allocation_roll_wait = dict(turn=int(driver.es.turn), rolls=[])
    # Several disjoint selections can belong to one pool roll. Keep each
    # native command until identities arrive; deduplicate by die, not event.
    pending['rolls'].append(dict(record=dict(rec), pids=list(pids) if pids else None))
    driver._eye_completion_requested = None
    return True


def complete(driver):
    """Use non-rolled dice in Hunt as allocation; rolled Eyes stay separate."""
    pending = getattr(driver, '_allocation_roll_wait', None)
    if pending is None:
        return []
    if int(driver.es.turn) != pending['turn'] or driver.phase != 'eyes':
        driver._allocation_roll_wait = None
        return []
    if driver._feed_seq != len(driver.feed.lines) or getattr(driver.feed, '_buf', ''):
        return []
    from . import client_board
    from .repair_evidence import dump_signature, same_dump
    from game_state import Side, ActionDieFace
    import game_logic as gl
    record = dict(pending['rolls'][0]['record'])
    retained, overlapping = {}, False
    for roll in pending['rolls']:
        faces = list(roll['record']['faces'])
        pids = roll['pids']
        if not pids:
            witness = driver._take_public_roll_witness(faces, None)
            pids = witness['pids'] if witness else None
            if pids is None and len(pending['rolls']) == 1:
                pids = driver._pending_roll_pids.get('SHADOW')
            if not pids:
                return []
            roll['pids'] = pids = list(pids)
        if len(pids) != len(faces) or len(set(pids)) != len(pids):
            pending['waiting'] = 'The recorded roll has incomplete die identities; use Resync or Rewind Turn to recover.'
            return []
        for pid, face in zip(pids, faces):
            if pid in retained:
                overlapping = True
            else:
                retained[pid] = face
    pids, faces = list(retained), list(retained.values())
    record['faces'] = faces
    try:
        evidence = dump_signature(driver)
        header = evidence['header']
        if header is None or driver._dump_position_hold(header):
            return []
        driver._check_dump_session(header)
        dump = client_board.read_board(driver.view.dump_path)
        if not dump or not same_dump(driver, evidence):
            return []
    except (OSError, ValueError):
        return []
    dice = driver.es.dice[Side.SHADOW]
    entries = {str(pid): int(index) for pid, (side, index) in driver.view.die_id_map.items()
               if side is Side.SHADOW}
    if (sorted(entries.values()) != list(range(len(dice)))
            or len(pids) != len(faces) or len(set(pids)) != len(pids)
            or not set(pids).issubset(entries)):
        pending['waiting'] = 'The recorded roll includes an unrecognized die; use Resync or Rewind Turn to recover the dice pool.'
        return []
    # This is the first Shadow roll, not permission to erase settled faces.
    if any(d.current is not None and not (d.used and d.current is ActionDieFace.EYE)
           for d in dice):
        return []
    hunt = {'hunt box', 'the hunt for the ring'}
    zones = {pid: str(dump.get(pid, '')).strip().lower() for pid in entries}
    allocated = [entries[pid] for pid in entries if pid not in pids and zones[pid] in hunt]
    if any(zone not in hunt | {'sa dice'} for zone in zones.values()):
        return []
    if any(zones[pid] not in (hunt if face == 'E' else {'sa dice'})
           for pid, face in zip(pids, faces)):
        if overlapping:
            return _correct_extra_roll(pending, faces)
        return []  # Wait for the native move of rolled Eyes to finish.
    if overlapping:
        from .client_protocol import read_action_die_faces
        try:
            actual = read_action_die_faces(driver.view.dump_path)
        except (OSError, ValueError):
            return []
        if any(pid in actual and actual[pid] != face for pid, face in retained.items()):
            return _correct_extra_roll(pending, faces)
    counts = driver._eye_menu_counts(refresh=True)
    if counts and len(allocated) > max(counts):
        if pending.get('excess_reported') != len(allocated):
            pending['excess_reported'] = len(allocated)
            return [('chat', 'prompt',
                     'The roll is recorded. Return %d voluntarily allocated eye(s) '
                     'to SA Dice to meet the allocation limit; keep the rolled Eyes in Hunt.'
                     % (len(allocated) - max(counts)))]
        return []
    if not same_dump(driver, evidence):
        return []
    sh_ids = {id(die) for die in dice}
    driver.es.hunt_box = [die for die in driver.es.hunt_box if id(die) not in sh_ids]
    for die in dice:
        die.used, die.current = False, None
    driver.es.eyes_allocated = 0
    gl.allocate_eyes(driver.es, len(allocated), indices=allocated)
    driver._allocation_roll_wait = None
    driver._eye_completion_requested = None
    driver._eye_completion_rejected = None
    record['_allocation_closed'] = True
    record['_allocation_roll_indices'] = [entries[pid] for pid in pids]
    record['meta'] = dict(record.get('meta') or {}, segs=[[], pids])
    driver.findings.append(('allocation_from_completed_roll',
                            'allocated=%d rolled_eyes=%d' % (len(allocated), faces.count('E'))))
    output = driver._on_action_roll(record)
    if overlapping:
        output.insert(0, ('chat', 'prompt',
                         'The first result for each action die is retained; the extra roll is not counted.'))
    return output


def _correct_extra_roll(pending, faces):
    message = ('The extra roll is not counted. Restore the original Shadow result: %s. '
               'Put only its Eye dice in Hunt and the other dice in SA Dice; do not roll again. '
               'Rewind Turn is available if you prefer to restart this turn.' % ' '.join(faces))
    pending['waiting'] = message
    if pending.get('correction_reported') == message:
        return []
    pending['correction_reported'] = message
    return [('chat', 'prompt', message)]
