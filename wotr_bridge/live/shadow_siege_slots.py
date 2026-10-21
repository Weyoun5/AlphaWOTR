"""Bind the client's shared Shadow Stronghold slot to observed piece origins."""

SHADOW_STRONGHOLDS = frozenset(('orthanc', 'moria', 'dol guldur',
    'mount gundabad', 'morannon', 'minas morgul', 'barad dur', 'umbar'))
GENERIC_SLOTS = {'shadow stronghold': 'Game.1996', 'shadow stronghold 2': 'Game.2028'}


def bind_record(view, state, record):
    from board import sanitize
    from combat import side_of_nation, garrison_side_for
    from game_state import Side
    from .sidecar import meta_pids
    bindings = getattr(view, 'shadow_stronghold_regions', None)
    if bindings is None:
        bindings = view.shadow_stronghold_regions = {}
    pids = [str(p) for p in (meta_pids(record.get('meta')) or ())]
    _bind_departure(view, record, pids)
    if record.get('op') != 'move':
        if record.get('op') == 'kill':
            for pid in pids:bindings.pop(pid, None)
        return
    source, target = (sanitize(str(record.get(key, ''))).lower()
                      for key in ('from', 'to'))
    if source in GENERIC_SLOTS and pids:
        known = {bindings.get(pid) for pid in pids}
        if len(known) == 1 and None not in known:
            region, code = next(iter(known))
            if code == GENERIC_SLOTS[source]:record['from'] = region + ' stronghold'
    if target in GENERIC_SLOTS:
        pc = getattr(state, 'pending_combat', None) or {}
        region = sanitize(str(pc.get('defender_region', ''))).lower()
        defender = pc.get('defender_side')
        if (pc.get('phase') != 'defender_siege_choice'
                and not (pc.get('phase') == 'victor_advance' and pc.get('post_siege_retreat'))):
            # The first transfer can commit withdrawal before later pieces
            # reach the shared box. Its established siege still owns them.
            region = source
            if region not in {sanitize(str(r)).lower() for r in state.besieged_strongholds}:
                return
            defender = garrison_side_for(state, region)
        if (defender is None or region not in SHADOW_STRONGHOLDS or source != region):
            return
        if not pids:
            # Old hooks expose no raw IDs. Public dump transitions provide
            # exact identities, but never infer occupants from an initial slot.
            view.refresh_dump()
            arrivals = getattr(view, 'shadow_slot_arrivals', {})
            pids = _select_pieces(view, [pid for pid, transition in arrivals.items()
                    if transition == (region, target)], _record_pieces(record))
            if not pids or not _same_pieces(view, pids, _record_pieces(record)):
                return
        if not _same_pieces(view, pids, _record_pieces(record)):
            return
        from .client_board import unit_type
        for pid in pids:
            nation, kind = unit_type(pid, view.dynamic_types)
            if kind in ('minion', 'nazgul', 'companion'):
                piece_side = Side.FREE_PEOPLES if kind == 'companion' else Side.SHADOW
                if piece_side is not defender:
                    return
                continue
            if (kind not in ('regular', 'elite', 'leader') or nation is None
                    or side_of_nation(nation) is not defender):
                return
        record['to'] = region + ' stronghold'
        record['_shadow_slot_bindings'] = tuple(
            (pid, region, GENERIC_SLOTS[target]) for pid in pids)
        for pid in pids:
            bindings[pid] = (region, GENERIC_SLOTS[target])
            origins = getattr(view, 'shadow_slot_origins', None)
            if origins is None:
                origins = view.shadow_slot_origins = {}
            origins[pid] = bindings[pid]
            view.piece_region[pid] = region + ' stronghold'
            getattr(view, 'shadow_slot_arrivals', {}).pop(pid, None)
    else:
        for pid in pids:bindings.pop(pid, None)


def project_dump(view, dump):
    """Only previously witnessed exact pieces acquire a semantic region."""
    from board import sanitize
    result = dict(dump)
    previous = getattr(view, 'piece_region', {})
    arrivals = getattr(view, 'shadow_slot_arrivals', None)
    if arrivals is None:
        arrivals = view.shadow_slot_arrivals = {}
    for pid, value in dump.items():
        zone = sanitize(str(value)).lower()
        old = sanitize(str(previous.get(pid, ''))).lower()
        if zone in GENERIC_SLOTS and old in SHADOW_STRONGHOLDS:
            arrivals[pid] = (old, zone)
        elif zone not in GENERIC_SLOTS:
            arrivals.pop(pid, None)
    bindings = getattr(view, 'shadow_stronghold_regions', {})
    if getattr(view, '_shadow_slot_restore', False):
        for pid, (region, code) in getattr(view, 'shadow_slot_origins', {}).items():
            if GENERIC_SLOTS.get(sanitize(str(dump.get(pid, ''))).lower()) == code:
                bindings[pid] = (region, code)
        view._shadow_slot_restore = False
    departures = getattr(view, 'shadow_slot_departures', None)
    if departures is None:
        departures = view.shadow_slot_departures = {}
    for pid, (_, _, destination) in list(departures.items()):
        if sanitize(str(dump.get(pid, ''))).lower() != destination:
            departures.pop(pid, None)
    for pid, (region, code) in list(bindings.items()):
        if GENERIC_SLOTS.get(sanitize(str(dump.get(pid, ''))).lower()) == code:
            result[pid] = region + ' stronghold'
        else:
            departures[pid] = (region, code, sanitize(str(dump.get(pid, ''))).lower())
            bindings.pop(pid, None)
    return result


def _parsed_counts(text):
    from collections import Counter
    import rendered_replay as rr
    text = str(text)
    for full, canonical in rr.COMMA_NAME_FIX:
        text = text.replace(full, canonical)
    tokens = rr.split_group(text[7:-1]) if text.startswith('group (') and text.endswith(')') else [text]
    expected = Counter()
    for token in tokens:
        parsed = rr.parse_piece(token)
        if not parsed:
            return Counter()
        kind, name, subtype, count = parsed
        if kind == 'unit':kind = str(subtype).lower()
        if kind == 'named':kind = 'minion'
        expected[(str(name).lower(), kind)] += count
    return expected


def _pid_key(view, pid):
    from .client_board import unit_type
    nation, kind = unit_type(pid, view.dynamic_types)
    name = 'SAURON' if kind == 'nazgul' else getattr(nation, 'name', nation)
    return str(name).lower(), kind


def _same_pieces(view, pids, text):
    from collections import Counter
    expected = _parsed_counts(text)
    return bool(expected) and expected == Counter(_pid_key(view, pid) for pid in pids)


def garrison_seated(view, state, region, side):
    """The whole public garrison already occupies its specific or bound slot."""
    from collections import Counter
    from board import sanitize
    from combat import side_of_nation, _combat_leader_side
    from game_state import Side
    from .client_board import unit_type
    region = sanitize(str(region)).lower()
    expected, inside, outside = Counter(), Counter(), Counter()
    for key, army in state.armies.items():
        if sanitize(str(key)).lower() != region:
            continue
        for unit in army.units:
            if side_of_nation(unit.nation) is side:
                expected[(unit.nation.name, str(unit.kind.value).lower())] += 1
        for leader in army.leaders:
            if _combat_leader_side(leader) is side:
                expected[('leader', 'leader')] += 1
    for pid, zone in view.piece_region.items():
        zone = sanitize(str(zone)).lower()
        if zone not in (region, region + ' stronghold'):
            continue
        nation, kind = unit_type(pid, view.dynamic_types)
        if kind in ('regular', 'elite', 'leader'):
            if nation is None or side_of_nation(nation) is not side:
                continue
            key = (nation.name, kind) if kind != 'leader' else ('leader', 'leader')
        elif kind in ('nazgul', 'minion', 'companion'):
            piece_side = Side.FREE_PEOPLES if kind == 'companion' else Side.SHADOW
            if piece_side is not side:
                continue
            key = ('leader', 'leader')
        else:
            continue
        (inside if zone.endswith(' stronghold') else outside)[key] += 1
    return bool(expected) and inside == expected and not outside


def _bind_departure(view, record, pids):
    from board import sanitize
    key = 'zone' if record.get('op') == 'kill' else 'from'
    source = sanitize(str(record.get(key, ''))).lower()
    if record.get('op') not in ('move', 'kill') or source not in GENERIC_SLOTS:
        return
    code = GENERIC_SLOTS[source]
    # A dump can lead its feed; preserve exits before bindings are cleared.
    view.refresh_dump()
    departures = getattr(view, 'shadow_slot_departures', {})
    rows = {}
    for pid, (region, slot, destination) in departures.items():
        if slot != code:
            continue
        if record.get('op') == 'move':
            if destination != sanitize(str(record.get('to', ''))).lower():
                continue
        elif destination not in ('dead', 'sa casualties', 'fp casualties', 'casualties', 'sa reinforcements', 'sa reinf'):
            continue
        rows.setdefault(region, []).append(pid)
    if pids:
        # Raw IDs also work when the final dump has not landed yet.
        bindings = getattr(view, 'shadow_stronghold_regions', {})
        regions = {((bindings.get(pid) or departures.get(pid) or (None, None))[0])
                   for pid in pids}
        slots = {((bindings.get(pid) or departures.get(pid) or (None, None))[1])
                 for pid in pids}
        matches = [(next(iter(regions)), pids)] if len(regions) == 1 and None not in regions and slots == {code} else []
    else:
        matches = []
        # Count/type selection is allowed only within a uniquely proven region;
        # interchangeable same-type IDs never invent a region or an extra loss.
        for region, candidates in rows.items():
            selected = _select_pieces(view, candidates, _record_pieces(record))
            if selected:
                matches.append((region, selected))
    if len(matches) != 1:
        return
    region, selected = matches[0]
    if not _same_pieces(view, selected, _record_pieces(record)):
        return
    record[key] = region + ' stronghold'
    for pid in selected:
        departures.pop(pid, None)
        getattr(view, 'shadow_stronghold_regions', {}).pop(pid, None)


def _record_pieces(record):
    if record.get('op') == 'kill':
        return 'group (' + ', '.join(record.get('pieces') or ()) + ')'
    return record.get('obj', '')


def _select_pieces(view, candidates, text):
    needed = _parsed_counts(text)
    selected = []
    for pid in sorted(candidates):
        piece_key = _pid_key(view, pid)
        if needed[piece_key] > 0:
            selected.append(pid)
            needed[piece_key] -= 1
    return selected if selected and not any(needed.values()) else []
