"""Literal RWA/BCC army follow-ups, separated from independent Nazgul flights.

Optional completion requires a disposed named card and a spent funding die.
An unfinished window, combat, correction, or ambiguous leading leader drag
cannot manufacture a movement or decline target.
"""
from collections import Counter, OrderedDict

NAMES = {'the ringwraiths are abroad': 2, 'the black captain commands': 1}


class Unproven(ValueError):
    pass


def literal_plan(batch, title):
    import rendered_extract as rx
    import rendered_replay as rr
    from rendered_staged_movement import token_cargo
    name = str(title).split('/')[0].strip().lower()
    if name not in NAMES:
        return None
    # Keep the existing independently proven attack matcher for battles.
    if batch.get('kind') in ('combat', 'combat?') or any(
            row.get('kind') in ('roll_num', 'draw_tile') for row in batch.get('rng', ())):
        return None
    for record in batch['records']:
        if record.get('op') == 'choose':
            return None
        if record.get('op') == 'move':
            origin, target = rr.zone_kind(record.get('from', '')), rr.zone_kind(record.get('to', ''))
            if (origin[0] == 'region' and target[0] == 'stronghold' and origin[1] == target[1]
                    and any((rr.parse_piece(token) or (None,))[0] == 'unit'
                            for token in rx._group_tokens(record.get('obj', '')))):
                return None  # Visible defensive siege retreat proves combat.
    plays = rx._card_play_titles(batch)
    if len(plays) != 1 or str(plays[0][1]).split('/')[0].strip().lower() != name:
        raise Unproven('play_window_ambiguous')
    start = int(plays[0][0])
    movements, card_closed, die_closed, recruits = [], [], [], []
    for record in batch['records']:
        line = int(record.get('line', 0))
        if line <= start:
            continue
        if record.get('op') == 'kill':
            tokens = [token for piece in record.get('pieces', ()) for token in rx._group_tokens(str(piece))]
            if any(token.split('/')[0].strip().lower() == name for token in tokens):
                card_closed.append(line)
            if str(record.get('zone', '')).strip().lower() == 'current action' and any(
                    'Shadow Action Die' in token for token in tokens):
                die_closed.append(line)
        if record.get('op') != 'move':
            continue
        origin, target = rr.zone_kind(record.get('from', '')), rr.zone_kind(record.get('to', ''))
        if origin[0] == 'reinf' and target[0] in ('region', 'stronghold'):
            cargo = Counter()
            for token in rx._group_tokens(record.get('obj', '')):
                key, count = token_cargo(token)
                if key is not None:
                    cargo[key] += count
            if cargo:
                if name != 'the black captain commands' or set(cargo) != {('nazgul', 'SAURON', None)}:
                    raise Unproven('unexpected_recruitment')
                recruits.append(dict(region=target[1], count=cargo[('nazgul', 'SAURON', None)], line=line))
            continue
        if origin[0] not in ('region', 'stronghold') or target[0] not in ('region', 'stronghold'):
            continue
        if origin[1] == target[1]:
            continue
        cargo = Counter()
        for token in rx._group_tokens(record.get('obj', '')):
            key, count = token_cargo(token)
            if key is not None:
                cargo[key] += count
        if cargo:
            movements.append(dict(edge=(origin[1], target[1]), cargo=cargo, line=line, record=record))
    last = max([start] + [row['line'] for row in movements] + [row['line'] for row in recruits])
    if not card_closed or not die_closed or max(card_closed + die_closed) <= last:
        raise Unproven('card_action_not_closed_after_effects')
    army_edges = {row['edge'] for row in movements if any(key[0] == 'unit' for key in row['cargo'])}
    first_army = min([row['line'] for row in movements if row['edge'] in army_edges
                      and any(key[0] == 'unit' for key in row['cargo'])], default=last + 1)
    edges, flights = OrderedDict(), []
    for row in movements:
        if row['line'] < first_army:
            # A separate Nazgul drag along an army route could be either an
            # independent flight or the first fragment of that army's move.
            if row['edge'] in army_edges:
                raise Unproven('leading_leader_flight_or_army_ambiguous')
            if any(key[0] not in ('nazgul', 'named') or
                   (key[0] == 'named' and key[1] != 'Witch King') for key in row['cargo']):
                raise Unproven('non_nazgul_before_army')
            flights.append(row['record'])
            continue
        if row['edge'] not in army_edges:
            raise Unproven('independent_flight_after_army_started')
        edge = edges.setdefault(row['edge'], dict(cargo=Counter(), line=row['line']))
        edge['cargo'].update(row['cargo'])
    if recruits and (flights or any(row['line'] >= first_army for row in recruits)
                     or len({row['region'] for row in recruits}) != 1):
        raise Unproven('recruitment_or_flight_stage_ambiguous')
    if len(edges) > NAMES[name]:
        raise Unproven('too_many_army_movements')
    sources = set()
    moves = []
    for (source, target), row in edges.items():
        if source in sources or any(move['to'] == source or move['from'] == target for move in moves):
            raise Unproven('correction_or_forwarded_army')
        sources.add(source)
        moves.append(dict(row, **{'from': source, 'to': target}))
    return {'moves': moves, 'hops': rx._nazgul_hops(dict(batch, records=flights)),
            'recruits': recruits, 'closed': True}


def match_bcc_mode(state, copts, plan):
    """Recruit-vs-flight follows reinforcement gestures, not army-transport deltas."""
    import ai
    from rendered_staged_movement import army_cargo
    wanted = 'recruit' if plan['recruits'] else 'nazgul_moves'
    matches = [(index, action) for index, action in copts
               if action.params.get('choices', {}).get('action') == wanted]
    if len(matches) != 1:
        raise Unproven('no_unique_bcc_mode')
    index, action = matches[0]
    if plan['recruits']:
        region = plan['recruits'][0]['region']
        expected = sum(row['count'] for row in plan['recruits'])
        if action.params['choices'].get('witch_king_region') != region:
            raise Unproven('recruitment_wrong_region')
        after = ai.apply_action(state, action)
        before_cargo, after_cargo = army_cargo(state, region), army_cargo(after, region)
        if (after_cargo - before_cargo != Counter({('nazgul', 'SAURON', None): expected})
                or before_cargo - after_cargo):
            raise Unproven('recruitment_count_mismatch')
    return index


def match_move(state, side, menu, expected):
    """Match transported identities/counts, not merely source/destination."""
    import ai
    from rendered_staged_movement import army_cargo
    source, target = expected['from'], expected['to']
    before_source, before_target = army_cargo(state, source), army_cargo(state, target)
    matches = []
    for index, action in enumerate(menu):
        if action.action_type != 'army_move' or not action.params.get('card_followup'):
            continue
        if ai.sanitize(str(action.params.get('from'))) != source or ai.sanitize(str(action.params.get('to'))) != target:
            continue
        after = ai.apply_action(state, action)
        after_source, after_target = army_cargo(after, source), army_cargo(after, target)
        if (before_source - after_source == expected['cargo'] and
                after_target - before_target == expected['cargo'] and
                not after_source - before_source and not before_target - after_target):
            matches.append(index)
    # Multiple encodings can describe the same unit transfer. Reuse the
    # existing legal-action identity policy; do not pick arbitrarily here.
    if len(matches) != 1:
        raise Unproven('no_unique_exact_army_transfer')
    return matches[0]
