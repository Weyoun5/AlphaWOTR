"""Authenticate a temporary Mighty Attack Companion park, never a casualty."""
from copy import copy
from collections import Counter


def park_witness(state, side, window, action):
    from cards import get_card
    import rendered_replay as rr
    pc = getattr(state, 'pending_combat', None) or {}
    effect = pc.get('effect_choice') or {}
    owner = effect.get('owner')
    if (pc.get('phase') != 'combat_effect_choice' or effect.get('stage') != 'round'
            or effect.get('effect') != 'mighty_attack' or owner not in ('atk', 'def')
            or action.action_type != 'combat_effect_choice'):
        return None
    role = 'attacker' if owner == 'atk' else 'defender'
    if pc.get(role + '_side') != side or pc.get('deciding_side') != side:
        return None
    params = action.params or {}
    card = get_card(effect.get('card_id'))
    if (card is None or card.combat_name.casefold() != 'mighty attack'
            or params.get('card_id') != card.card_id or set(params) != {'card_id', 'target'}
            or {'target': params['target']} not in effect.get('options', ())):
        return None
    cursor = (pc.get('_round_resume') or {}).get('cursor')
    actual = getattr(getattr(cursor, 'combat', None), role + '_card', None)
    if actual is None or actual.card_id != card.card_id:
        return None
    source_region = rr.sanitize(str(pc[role + '_region']))
    from .action_die_zones import temporary
    import re
    changes = Counter()
    sources, arrivals = {}, {}
    for record in window.records:
        if record.get('op') not in ('move', 'kill'):
            continue
        obj = str(record.get('obj', ''))
        tokens = ((record.get('pieces') or ()) if record['op'] == 'kill' else
                  (rr.split_group(obj[len('group ('):-1]) if obj.startswith('group (') and obj.endswith(')') else [obj]))
        named = [p for p in (rr.parse_piece(t) for t in tokens) if p and p[0] == 'named']
        if not named:
            continue
        if record['op'] != 'move':
            return None
        for piece in named:
            source = rr.sanitize(str(record.get('from', '')))
            destination = rr.sanitize(str(record.get('to', '')))
            changes[(piece[1], source)] -= piece[3]
            changes[(piece[1], destination)] += piece[3]
            sources.setdefault((piece[1], source), record)
            arrivals[(piece[1], destination)] = record
    changes = {key: count for key, count in changes.items() if count}
    losses = [key for key, count in changes.items() if count == -1]
    gains = [key for key, count in changes.items() if count == 1]
    if len(changes) != 2 or len(losses) != 1 or len(gains) != 1:
        return None
    leaving, joining = losses[0], gains[0]
    source = rr.zone_kind(leaving[1])
    if (leaving[0] != params['target'] or joining[0] != params['target']
            or source[0] not in ('region', 'stronghold') or rr.sanitize(source[1]) != source_region
            or not (temporary(joining[1]) or re.fullmatch(r'board\d+', joining[1]))):
        return None
    record = arrivals[joining]
    return {'name': params['target'], 'target': params['target'], 'source_region': source_region,
            'source_zone': sources[leaving]['from'], 'destination_zone': record['to'],
            'card_id': card.card_id, 'rounds_fought': pc.get('rounds_fought', 0),
            'owner': owner, 'record': record}


def permits_choice(state, side, window, action):
    """A wrong physical park cannot select a zero-board-delta payer."""
    effect = (getattr(state, 'pending_combat', None) or {}).get('effect_choice') or {}
    if effect.get('effect') != 'mighty_attack' or action.action_type != 'combat_effect_choice':
        return True
    return park_witness(state, side, window, action) is not None


def payment_coverage(state, side, window, action, coverage):
    witness = park_witness(state, side, window, action)
    if witness is None:
        return coverage
    result = copy(coverage)
    result.named = Counter(coverage.named)
    result.named[(witness['name'], witness['source_region'])] -= 1
    return result
