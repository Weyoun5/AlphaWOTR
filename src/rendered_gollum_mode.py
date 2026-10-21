"""There Is Another Way: observe the mode before its Hunt can reveal again."""


def mode(batch):
    import rendered_extract as rx
    import rendered_replay as rr
    plays = rx._card_play_titles(batch)
    title = 'there is another way'
    if len(plays) != 1 or plays[0][1].split('/')[0].strip().lower() != title:
        return None, 'cardint_another_way:play_unproven'
    start = int(plays[0][0])
    boundaries = [int(r['line']) for r in batch['records']
                  if int(r.get('line', 0)) > start and r.get('die_stage')]
    end = min(boundaries, default=10**12)
    hunts = [int(r['line']) for r in batch.get('rng', ())
             if start < int(r.get('line', 0)) < end and r.get('kind') in ('draw_tile', 'roll_num')]
    cutoff = min(hunts + [end])
    progress = 0
    flips = 0
    closed = False
    for record in batch['records']:
        line = int(record.get('line', 0))
        if not start < line < end:
            continue
        if record.get('op') == 'kill' and any(
                token.split('/')[0].strip().lower() == title
                for piece in record.get('pieces', ()) for token in rx._group_tokens(piece)):
            closed = True
        if line >= cutoff:
            continue
        obj = str(record.get('obj', '')).lower()
        if record.get('op') == 'flip' and obj == 'the fellowship':
            flips += 1
        if record.get('op') == 'move' and obj in ('fsp', 'the fellowship'):
            source, target = rr.zone_kind(record.get('from', '')), rr.zone_kind(record.get('to', ''))
            if source[0] == target[0] and source[0] in ('feltrack', 'mordor'):
                progress += target[1] - source[1]
    hidden = batch['truth_before']['fel'].get('hidden')
    if progress == 1 and hidden is True and flips % 2 == 0:
        return 'move', None
    if not progress and flips % 2 == 1 and hidden is False and not hunts:
        return 'hide', None
    if not progress and flips % 2 == 0 and closed and not hunts:
        return 'heal_only', None
    return None, 'cardint_another_way:mode_unproven'


def match(options, batch):
    chosen, why = mode(batch)
    if why:
        return None, why
    matches = [i for i, action in options
               if (action.params.get('choices') or {}).get('action') == chosen]
    return (matches[0], None) if len(matches) == 1 else (None, 'cardint_another_way:mode_not_legal')
