"""Literal die/leader targets for events whose old labels lost identity."""

FACE_NAMES = {'A': 'ARMY', 'C': 'CHARACTER', 'M': 'MUSTER',
              'H': 'MUSTER_ARMY', 'P': 'EVENT', 'E': 'EYE'}


def _event_window(batch, title):
    import rendered_extract as rx
    plays = [(int(line), name) for line, name in rx._card_play_titles(batch)
             if str(name).split('/')[0].strip().lower() == title]
    if len(plays) != 1:
        return None
    start = plays[0][0]
    ends = [int(r['line']) for r in batch.get('records', ())
            if int(r.get('line', 0)) > start and (r.get('die_stage') or r.get('op') == 'turn')]
    return start, min(ends, default=10**12)


def lidless_evidence(batch):
    """Exact pool-to-Hunt deposits, or a count with identity explicitly unknown."""
    import rendered_extract as rx
    from collections import Counter
    window = _event_window(batch, 'the lidless eye')
    if window is None:
        return None, None, 'cardint_lidless_play_unproven'
    start, end = window
    net = Counter()
    refaced = Counter()
    eye_deposits = 0
    for record in batch.get('records', ()):
        if not start < int(record.get('line', 0)) < end:
            continue
        if record.get('op') == 'other':
            change = rx._DIECHANGE_PARSE_RE.search(str(record.get('raw', '')))
            if change and change.group(1).lower() == 'shadow':
                old, new = change.group(2).upper(), change.group(3).upper()
                if new == 'E' and old in FACE_NAMES and old != 'E':
                    refaced[FACE_NAMES[old]] += 1
                elif old == 'E':
                    return None, None, 'cardint_lidless_reface_retraction_unproven'
        if record.get('op') != 'move':
            continue
        # The spent funding die is not eligible, even if physically dragged.
        source, target = (str(record.get(k, '')).strip().lower() for k in ('from', 'to'))
        sign = (1 if source == 'sa dice' and 'hunt' in target
                else -1 if target == 'sa dice' and 'hunt' in source else 0)
        if not sign:
            continue
        tokens = ([('SHADOW', face) for face in record['_rx_lidless_original_faces']]
                  if record.get('_rx_lidless_original_faces') else rx._die_tokens(record.get('obj', '')))
        for side, face in tokens:
            if side == 'SHADOW' and face in FACE_NAMES and face != 'E':
                net[FACE_NAMES[face]] += sign
            elif side == 'SHADOW' and face == 'E':
                eye_deposits += sign
            elif side == 'SHADOW':
                return None, None, 'cardint_lidless_original_face_unproven'
    if eye_deposits != sum(refaced.values()):
        return None, None, 'cardint_lidless_refaced_deposit_unproven'
    net.update(refaced)
    if any(n < 0 for n in net.values()):
        return None, None, 'cardint_lidless_retraction_unproven'
    faces = sorted(net.elements())
    counts = {int(item['n']) for item in batch.get('rng', ())
              if item.get('kind') == 'add_eye' and 'n' in item
              and start < int(item.get('line', 0)) < end}
    if len(counts) > 1 or (faces and counts and counts != {len(faces)}):
        return None, None, 'cardint_lidless_conflicting_evidence'
    if faces:
        return sorted(faces), len(faces), None
    if counts:
        return None, counts.pop(), None
    return None, None, 'cardint_lidless_no_evidence'


def match_lidless(options, batch):
    faces, count, why = lidless_evidence(batch)
    if why:
        return None, why
    matches = [index for index, action in options
               if (action.params.get('choices') or {}).get('dice_count') == count
               and (faces is None or sorted((action.params.get('choices') or {}).get('dice_faces', [])) == faces)]
    return (matches[0], None) if len(matches) == 1 else (
        None, 'cardint_lidless_face_subset_unproven')


def match_denethor(options, batch):
    import rendered_extract as rx
    import rendered_replay as rr
    from collections import Counter
    victims = []
    staged = Counter()
    window = _event_window(batch, "denethor's folly")
    if window is None:
        return None, 'cardint_denethor_play_unproven'
    start, end = window
    # A subsequent battle has its own casualties; it cannot identify this
    # immediate event victim. Dice evidence marks that boundary conservatively.
    rolls = [int(item['line']) for item in batch.get('rng', ())
             if item.get('kind') not in ('add_eye', 'action_roll', 'alloc_eye')
             and 'line' in item and int(item['line']) > start]
    cutoff = min(rolls + [end])
    for record in batch.get('records', ()):
        if not start < int(record.get('line', 0)) < cutoff:
            continue
        if record.get('op') not in ('kill', 'move'):
            continue
        source = record.get('from', record.get('zone', ''))
        zone, name = rr.zone_kind(source)
        from_mt = zone in ('region', 'stronghold') and rr.sanitize(name) == rr.sanitize('Minas Tirith')
        tokens = (record.get('pieces', []) if record.get('op') == 'kill'
                  else rx._group_tokens(str(record.get('obj', ''))))
        for token in tokens:
            piece = rr.parse_piece(token)
            if piece and piece[0] == 'leader' and piece[1] in (
                    'ELVES', 'DWARVES', 'GONDOR', 'THE_NORTH', 'ROHAN'):
                nation, count = piece[1].lower(), int(piece[3])
                target = str(record.get('to', '')).lower()
                if from_mt and (record['op'] == 'kill' or 'casualt' in target):
                    victims.extend([nation] * count)
                elif from_mt and target in ('current action', 'current sa card', 'current fp card'):
                    staged[nation] += count
                elif str(source).lower() in ('current action', 'current sa card', 'current fp card'):
                    if record['op'] == 'kill' or 'casualt' in target:
                        if count > staged[nation]:
                            return None, 'cardint_denethor_staged_leader_unproven'
                        staged[nation] -= count
                        victims.extend([nation] * count)
                    elif rr.zone_kind(target)[0] in ('region', 'stronghold'):
                        staged[nation] -= count
    if len(victims) != 1 or any(staged.values()):
        return None, 'cardint_denethor_leader_unproven'
    matches = [index for index, action in options
               if (action.params.get('choices') or {}).get('nation') == victims[0]]
    return (matches[0], None) if len(matches) == 1 else (
        None, 'cardint_denethor_leader_unmatched')


def match_cirdan(options, batch):
    """Completed literal two-unit recruitment, never infer the Elite default."""
    import rendered_extract as rx
    import rendered_replay as rr
    from collections import Counter
    window = _event_window(batch, "cirdan's ships")
    if window is None:
        return None, 'cardint_cirdan_play_unproven'
    start, end = window
    records = [r for r in batch.get('records', ()) if start < int(r.get('line', 0)) < end]
    for record in records:
        if record.get('op') != 'move' or rr.zone_kind(record.get('from', ''))[0] != 'reinf':
            continue
        if (str(record.get('from', '')).strip().lower() != 'fp reinforcements'
                and any((rr.parse_piece(token) or ('', ''))[:2] == ('unit', 'ELVES')
                        for token in rx._group_tokens(record.get('obj', '')))):
            return None, 'cardint_cirdan_reinforcement_source_unproven'
    placements = rx._muster_placements({'records': records})
    elves = [(region, kind, n) for nation, region, kind, n in placements
             if nation == 'ELVES' and kind in ('REGULAR', 'ELITE')]
    if sum(n for _, _, n in elves) != 2 or len({region for region, _, _ in elves}) != 1:
        # Supply-limited single recruit and overstack returns require a richer
        # timed supply witness; do not mistake them for a free count decision.
        return None, 'cardint_cirdan_two_recruits_unproven'
    last_move = max((int(r['line']) for r in records if r.get('op') == 'move'
                     and any((rr.parse_piece(token) or ('', ''))[:2] == ('unit', 'ELVES')
                             for token in rx._group_tokens(r.get('obj', '')))), default=start)
    closed = end < 10**12 or any(r.get('op') == 'kill' and int(r['line']) > last_move
            and any(str(piece).split('/')[0].strip().lower() == "cirdan's ships"
                    for piece in r.get('pieces', ())) for r in records)
    if not closed:
        return None, 'cardint_cirdan_recruitment_incomplete'
    amounts = Counter()
    for _, kind, n in elves:
        amounts[kind] += n
    combo = 'mixed' if amounts['REGULAR'] == amounts['ELITE'] == 1 else (
        '2elite' if amounts['ELITE'] == 2 else '2regular')
    target = rr.sanitize(elves[0][0])
    matches = [index for index, action in options
               if (action.params.get('choices') or {}).get('unit_combo') == combo
               and rr.sanitize((action.params.get('choices') or {}).get('region', '')) == target]
    return (matches[0], None) if len(matches) == 1 else (None, 'cardint_cirdan_combo_unmatched')


def match_stormcrow_eliminate(options, batch):
    """Identify the FP-owned casualty, retaining its original region through staging."""
    import rendered_extract as rx
    import rendered_replay as rr
    from collections import Counter
    window = _event_window(batch, 'stormcrow')
    if window is None:
        return None, 'cardint_stormcrow_play_unproven'
    start, end = window
    staged, victims = Counter(), []
    closed = []
    pending_nations = {rx._nat_key((a.params.get('choices') or {}).get('eliminate', {}).get('nation'))
                       for _, a in options}
    card_areas = ('current action', 'current sa card', 'current fp card')
    for record in batch.get('records', ()):
        line = int(record.get('line', 0))
        if not start < line < end:
            continue
        op = record.get('op')
        if op == 'kill' and any(str(piece).split('/')[0].strip().lower() == 'stormcrow'
                                for piece in record.get('pieces', ())):
            closed.append(line)
        if op not in ('kill', 'move'):
            continue
        source = str(record.get('from', record.get('zone', ''))).lower()
        target = str(record.get('to', '')).lower()
        zone, region = rr.zone_kind(source)
        from_board = zone in ('region', 'stronghold')
        tokens = record.get('pieces', ()) if op == 'kill' else rx._group_tokens(record.get('obj', ''))
        for token in tokens:
            piece = rr.parse_piece(token)
            if not piece or piece[0] not in ('unit', 'leader') or piece[1] not in pending_nations:
                continue
            nation, kind, count = piece[1], ('leader' if piece[0] == 'leader' else piece[2].lower()), int(piece[3])
            if from_board:
                key = (nation, kind, rr.sanitize(region))
                if op == 'kill' or rr.zone_kind(target)[0] == 'cas':
                    victims.extend([key] * count)
                elif target in card_areas:
                    staged[key] += count
            elif source in card_areas:
                origins = [key for key, n in staged.items() if n > 0 and key[:2] == (nation, kind)]
                if not origins:
                    return None, 'cardint_stormcrow_staged_origin_missing'
                if len(origins) != 1 or staged[origins[0]] < count:
                    return None, 'cardint_stormcrow_staged_origin_ambiguous'
                key = origins[0]
                if op == 'kill' or rr.zone_kind(target)[0] == 'cas':
                    staged[key] -= count
                    victims.extend([key] * count)
                elif rr.zone_kind(target)[0] in ('region', 'stronghold'):
                    # Only a return to its original region retracts this gesture.
                    if rr.sanitize(rr.zone_kind(target)[1]) != key[2]:
                        return None, 'cardint_stormcrow_staged_piece_moved_elsewhere'
                    staged[key] -= count
    if len(victims) != 1 or any(staged.values()):
        return None, 'cardint_stormcrow_exact_victim_unproven'
    # Players may clean up the spent card before removing the casualty.
    # Both settled gestures in this net action window prove completion;
    # imposing their physical order would discard the same final choice.
    if not closed and end == 10**12:
        return None, 'cardint_stormcrow_completion_unproven'
    wanted = victims[0]
    matches = []
    for index, action in options:
        choice = (action.params.get('choices') or {}).get('eliminate') or {}
        candidate = (rx._nat_key(choice.get('nation')), choice.get('kind'), rr.sanitize(choice.get('region', '')))
        if candidate == wanted:
            matches.append(index)
    return (matches[0], None) if len(matches) == 1 else (None, 'cardint_stormcrow_victim_unmatched')
