"""Literal, closed Regular/Elite transfers for Rage of the Dunlendings."""
from collections import Counter


class Unproven(ValueError):
    pass


def literal_split(batch, destination):
    import rendered_extract as rx
    import rendered_replay as rr
    title = 'rage of the dunlendings'
    plays = rx._card_play_titles(batch)
    if len(plays) != 1 or plays[0][1].split('/')[0].strip().lower() != title:
        raise Unproven('missing_or_multiple_card_plays')
    start = int(plays[0][0])
    dest = rr.sanitize(destination)
    sources = {'north dunland', 'south dunland'} - {dest}
    counts = Counter()
    last_move = start
    closes = []
    for record in batch['records']:
        line = int(record.get('line', 0))
        if line <= start:
            continue
        if record.get('op') == 'kill':
            if any(token.split('/')[0].strip().lower() == title
                   for piece in record.get('pieces', ()) for token in rx._group_tokens(piece)):
                closes.append(line)
        if record.get('op') != 'move':
            continue
        source = rr.zone_kind(record.get('from', ''))
        target = rr.zone_kind(record.get('to', ''))
        if source[0] not in ('region', 'stronghold') or target[0] not in ('region', 'stronghold'):
            continue  # Recruitment from the supply is a separate clause.
        if source[1] == target[1]:
            continue
        if source[1] in sources and target[1] == dest:
            home, direction = source[1], 1
        elif source[1] == dest and target[1] in sources:
            home, direction = target[1], -1
        else:
            raise Unproven('movement_outside_card_route')
        for token in rx._group_tokens(record.get('obj', '')):
            piece = rr.parse_piece(token)
            if not piece or piece[0] != 'unit' or piece[1] != 'ISENGARD':
                raise Unproven('ineligible_or_unrecognized_figure')
            counts[home, piece[2]] += direction * int(piece[3])
            last_move = line
    if not any(line >= last_move for line in closes):
        raise Unproven('card_not_closed_after_movement')
    if any(value < 0 for value in counts.values()) or sum(counts.values()) > 4:
        raise Unproven('invalid_net_movement')
    return {'n_north': counts['north dunland', 'REGULAR'] + counts['north dunland', 'ELITE'],
            'n_south': counts['south dunland', 'REGULAR'] + counts['south dunland', 'ELITE'],
            'n_north_elite': counts['north dunland', 'ELITE'],
            'n_south_elite': counts['south dunland', 'ELITE']}


def match(copts, batch):
    options = [(i, action.params.get('choices') or {}) for i, action in copts
               if 'rage_split' in (action.params.get('choices') or {})]
    destinations = {choice.get('_rage_dest') for _, choice in options}
    if len(destinations) != 1:
        return None, 'rage_split:ambiguous_destination'
    try:
        split = literal_split(batch, next(iter(destinations)))
    except Unproven as exc:
        return None, 'rage_split:' + str(exc)
    matches = [i for i, choice in options if choice['rage_split'] == split]
    return (matches[0], None) if len(matches) == 1 else (None, 'rage_split:no_unique_composition')
