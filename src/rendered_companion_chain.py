"""Closed, named companion transfers for Book of Mazarbul and Fear! Fire! Foes!."""
from collections import OrderedDict

NAMES = {'book of mazarbul', 'fear! fire! foes!'}


class Unproven(ValueError):
    pass


def literal_plan(batch, title):
    import rendered_extract as rx
    import rendered_replay as rr
    name = str(title).split('/')[0].strip().lower()
    if name not in NAMES:
        return None
    plays = rx._card_play_titles(batch)
    if len(plays) != 1:
        raise Unproven('multiple_or_missing_plays')
    start = int(plays[0][0])
    routes = OrderedDict()
    closes = []
    last = start
    for record in batch['records']:
        line = int(record.get('line', 0))
        if line <= start:
            continue
        if record.get('op') == 'kill':
            tokens = [token for piece in record.get('pieces', ()) for token in rx._group_tokens(str(piece))]
            if any(token.split('/')[0].strip().lower() == name for token in tokens):
                closes.append(line)
        if record.get('op') != 'move':
            continue
        source, target = rr.zone_kind(record.get('from', '')), rr.zone_kind(record.get('to', ''))
        names = []
        for token in rx._group_tokens(record.get('obj', '')):
            piece = rr.parse_piece(token)
            if piece and piece[0] == 'named' and piece[1] in rr.COMP_ALIAS.values():
                if int(piece[3]) != 1:
                    raise Unproven('repeated_companion_count')
                names.append(piece[1])
        if not names:
            continue
        if source[0] not in ('region', 'stronghold') or target[0] not in ('region', 'stronghold'):
            raise Unproven('companion_not_on_board')
        if source[1] == target[1]:
            continue
        if len(names) != len(set(names)):
            raise Unproven('repeated_group_identity')
        last = line
        for companion in names:
            if companion not in routes:
                routes[companion] = [source[1], target[1]]
            elif routes[companion][-1] == source[1]:
                routes[companion].append(target[1])
            else:
                raise Unproven('discontinuous_companion_route')
    if not any(line > last for line in closes):
        raise Unproven('named_card_not_closed_after_movement')
    groups = OrderedDict()
    for companion, route in routes.items():
        if route[0] == route[-1]:
            continue  # Fully undone movement has no residual decision.
        groups.setdefault((route[0], route[-1]), []).append(companion)
    return [dict(source=source, target=target, companions=sorted(names))
            for (source, target), names in groups.items()]


def match(copts, context):
    import rendered_replay as rr
    plan = context['_companion_card_plan']
    index = context.get('_companion_card_cursor', 0)
    if index == len(plan):
        matches = [i for i, action in copts if action.params.get('choices', {}).get('_ffm_done')]
    else:
        wanted = plan[index]
        matches = []
        for i, action in copts:
            group = action.params.get('choices', {}).get('move_group')
            if (group and sorted(group.get('companions') or []) == wanted['companions']
                    and rr.sanitize(str(group.get('from'))) == wanted['source']
                    and rr.sanitize(str(group.get('to'))) == wanted['target']):
                matches.append(i)
    if len(matches) != 1:
        return None, 'cardint_companion_chain:no_unique_proven_group'
    if index < len(plan):
        context['_companion_card_cursor'] = index + 1
    return matches[0], None
