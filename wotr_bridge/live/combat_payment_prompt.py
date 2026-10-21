"""Public physical payment instructions from the active combat round cursor."""


def payment_context(dp, state):
    """Return only a current, owner-bound physical casualty payment."""
    import game_state as gs
    pc = getattr(state, 'pending_combat', None) or {}
    if (pc.get('phase') != 'combat_effect_choice'
            or str(dp.get('kind', '')).split(':')[-1] != 'combat_effect_choice'):
        return None
    effect = pc.get('effect_choice') or {}
    name = effect.get('effect')
    spec = {'pre_combat_casualty': (1, 'pre_remaining', 'pre_victim_attacker'),
            'relentless_assault_payment': (4, 'ra_remaining', 'ra_attacker')}.get(name)
    owner = effect.get('owner')
    if name == 'onslaught_payment':
        ledger = pc.get('_onslaught_payment') or {}
        if (effect.get('stage') != 'onslaught_payment'
                or any(ledger.get(k) != effect.get(k) for k in ('owner', 'card_id', 'remaining', 'count', 'paid'))
                or not ledger or owner not in ('atk', 'def')):
            return None
        hits = ledger.get('remaining');stage = 'onslaught_payment';pending = effect
    else:
        if (spec is None or effect.get('stage') != 'round'
                or effect.get('round_stage') != spec[0]):
            return None
        cursor = (pc.get('_round_resume') or {}).get('cursor')
        if cursor is None or getattr(cursor, 'stage', None) != spec[0]:
            return None
        pending = getattr(cursor, 'pending', None) or {}
        if (owner not in ('atk', 'def') or pending.get('owner') != owner
                or pending.get('effect') != name or pending.get('stage') != spec[0]
                or getattr(cursor, spec[2], None) is not (owner == 'atk')):
            return None
        hits = getattr(cursor, spec[1], None);stage = spec[0]
    if type(hits) is not int or hits <= 0:
        return None
    attacker = pc.get('attacker_side')
    if attacker not in (gs.Side.FREE_PEOPLES, gs.Side.SHADOW):
        return None
    side = attacker if owner == 'atk' else (
        gs.Side.SHADOW if attacker is gs.Side.FREE_PEOPLES else gs.Side.FREE_PEOPLES)
    region = pc.get('attacker_region' if owner == 'atk' else 'defender_region')
    if dp.get('side') is not side or not isinstance(region, str) or not region:
        return None
    choices = []
    for action in dp.get('actions') or ():
        p = action.params or {}
        option = {key: p.get(key) for key in ('nation', 'kind', 'operation')}
        if (action.action_type != 'combat_effect_choice'
                or p.get('card_id') != effect.get('card_id')
                or option not in pending.get('options', ())
                or option not in effect.get('options', ())):
            return None
        try:
            nation_key = gs.Nation[option['nation']]
            nation = nation_key.value
        except (KeyError, TypeError):
            return None
        shadow = nation_key in (gs.Nation.SAURON, gs.Nation.ISENGARD, gs.Nation.SOUTHRON_EASTERLINGS)
        if shadow != (side is gs.Side.SHADOW):
            return None
        kind, operation = option['kind'], option['operation']
        if kind == 'regular' and operation == 'remove':
            label = 'remove one %s Regular (1 hit)' % nation
        elif kind == 'elite' and operation == 'downgrade':
            label = 'reduce one %s Elite (1 hit)' % nation
        elif kind == 'elite' and operation == 'remove' and hits >= 2:
            label = 'remove one %s Elite entirely (2 hits)' % nation
        else:
            return None
        if label not in choices:
            choices.append(label)
    if not choices:
        return None
    title = {'pre_combat_casualty': 'Pre-combat attack',
             'relentless_assault_payment': 'Relentless Assault payment',
             'onslaught_payment': 'Onslaught payment'}[name]
    if name == 'pre_combat_casualty':
        from cards import get_card
        card = get_card(effect.get('card_id'))
        if card is not None and card.combat_name:
            title = card.combat_name + ' pre-combat attack'
    return {'effect': name, 'owner': owner, 'stage': stage, 'side': side,
            'region': region, 'hits': hits, 'choices': tuple(choices), 'title': title}


def instructions(context):
    hits = context['hits']
    timing = 'before the additional attack' if context['effect'] == 'onslaught_payment' else 'before combat rolls'
    return ('%s: take %d remaining hit%s in %s %s by removing or reducing '
            'your pieces on the board: %s. When reducing an Elite, place its same-nation '
            'Regular replacement if one is available') % (
                context['title'], hits, '' if hits == 1 else 's', context['region'], timing,
                '; or '.join(context['choices']))
