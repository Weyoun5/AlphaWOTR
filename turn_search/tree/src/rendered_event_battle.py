"""Identify an event's battle from literal casualties and army transfers."""
import rendered_replay as rr


def funded_event_line(batch):
    """Prove the first Help hand exit is its Event, not No Quarter.

    An Event die cannot buy an ordinary army attack. A Will of the West die
    can, so require the additional literal Board placement in that case.
    Ring face conversion uses the extractor's existing funding evidence.
    """
    import rendered_extract as rx
    if batch.get('die_side') != 'FREE_PEOPLES':
        return None
    plays = rx._card_play_titles(batch)
    if not plays or str(plays[0][1]).split('/')[0].strip().lower() != 'help unlooked for':
        return None
    line, title = plays[0]
    first_roll = min((int(r['line']) for r in batch.get('rng', [])
                      if r.get('kind') == 'roll_num'), default=None)
    if first_roll is None or line >= first_roll:
        return None
    face = rx._effective_lead_letter(batch)
    if face not in ('P', 'W'):
        return None
    # A prior opponent commit or combat roll would make this a combat card
    # even if a malformed log claims an Event die funded the action.
    for record in batch['records']:
        if int(record.get('line', 0)) >= line:
            continue
        if record.get('op') == 'move' and rx._zone_norm(record.get('from', '')) == 'shadow hand':
            return None
    if face == 'P':
        return line
    seated = any(r.get('op') == 'move' and str(r.get('obj', '')) == str(title)
                 and line <= int(r.get('line', 0)) < first_roll
                 and str(r.get('to', '')).startswith('Board') for r in batch['records'])
    return line if seated else None


def match(options, batch):
    import rendered_extract as rx
    first_roll = min((int(r['line']) for r in batch.get('rng', [])
                      if r.get('kind') == 'roll_num'), default=None)
    if first_roll is None:
        return None, 'cardint_battle_no_roll'
    losses = {'fp': set(), 'sh': set()}
    advances = set()
    for record in batch['records']:
        if int(record['line']) <= first_roll:
            continue
        op = record.get('op')
        origin = rr.zone_kind(record.get('from', record.get('zone', '')))
        target = rr.zone_kind(record.get('to', ''))
        tokens = (record.get('pieces', []) if op == 'kill'
                  else rx._group_tokens(record.get('obj', '')) if op == 'move' else [])
        sides = set()
        for token in tokens:
            piece = rr.parse_piece(token)
            if piece and piece[0] == 'unit':
                sides.add('sh' if piece[1] in rx._SHADOW_NATS else 'fp')
        if origin[0] not in ('region', 'stronghold'):
            continue
        source = rr.sanitize(origin[1]).removesuffix(' stronghold')
        if op == 'kill' or target[0] in ('cas', 'reinf'):
            for side in sides:
                losses[side].add(source)
        elif op == 'move' and sides == {'fp'} and target[0] in ('region', 'stronghold'):
            destination = rr.sanitize(target[1]).removesuffix(' stronghold')
            if source != destination:
                advances.add((source, destination))
    matches = []
    for index, action in options:
        choice = action.params.get('choices') or {}
        target = rr.sanitize(str(choice.get('stronghold', '')))
        source = rr.sanitize(str(choice.get('attack_from', '')))
        if source and target and ((source, target) in advances or
                                  (source in losses['fp'] and target in losses['sh'])):
            matches.append(index)
    return (matches[0], None) if len(matches) == 1 else (None, 'cardint_battle_target_unproven')
