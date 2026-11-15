"""Live Onslaught self-hit payments, followed by its additional attack."""


def _roles(payment):
    owner = payment['owner']
    if owner not in ('atk', 'def'):
        raise ValueError('Invalid Onslaught owner')
    role = 'attacker' if owner == 'atk' else 'defender'
    return role, 'defender' if role == 'attacker' else 'attacker'


def _options(state, pc, payment):
    import ai
    role, _ = _roles(payment)
    choices = ai._casualty_operations(state, pc[role + '_region'],
                                      pc[role + '_side'], payment['remaining'])
    return [{'nation': a.params['nation'].name,
             'kind': 'regular' if a.params['operation'] == 'remove_regular' else 'elite',
             'operation': 'downgrade' if a.params['operation'] == 'reduce_elite' else 'remove'}
            for a in choices]


def _advance(state, pc):
    import ai
    import combat
    payment = pc['_onslaught_payment']
    role, other = _roles(payment)
    options = _options(state, pc, payment)
    if payment['remaining'] > 0 and options:
        pc['phase'] = 'combat_effect_choice'
        pc['deciding_side'] = pc[role + '_side']
        pc['effect_choice'] = dict(payment, stage='onslaught_payment',
                                   effect='onslaught_payment', options=options)
        return
    # A self-wipe eliminates its accompanying figures immediately (Almanac
    # p44), before the enemy sees an allocation choice. The paid additional
    # attack still occurs; battle control/finalization remains in the tail.
    if not ai._alloc_side_units(state, pc[role + '_region'], pc[role + '_side']):
        combat.reconcile_destroyed_army(state, pc[role + '_region'], pc[role + '_side'])
    count = payment['paid']
    rolls, hits = combat.combat_roll(count, 4, 0, side=pc[role + '_side']) if count else ([], 0)
    from game_trace import emit_annotation
    if rolls:
        emit_annotation(state, 'dice_roll', {'faces': list(rolls),
                        'context': 'Onslaught additional attack',
                        'side': 'SH' if pc[role + '_side'] == ai.Side.SHADOW else 'FP'})
    enemy_present = bool(ai._alloc_side_units(state, pc[other + '_region'], pc[other + '_side']))
    pc.pop('_onslaught_payment')
    pc.pop('effect_choice', None)
    pc.pop('_finish_atk_dead', None)
    pc.pop('_finish_def_dead', None)
    pc['def_hits'] = int(hits) if other == 'defender' and enemy_present else 0
    pc['atk_hits'] = int(hits) if other == 'attacker' and enemy_present else 0
    pc['alloc_stage'] = 'attacker'
    pc['_p9_done'] = True
    pc['_onslaught_done'] = True
    ai._enter_casualty_allocation(state, pc)


def begin(state, pc, params):
    """Validate the chosen count, then expose exact unit payments."""
    ec = pc.get('effect_choice') or {}
    count = params.get('count')
    if (ec.get('stage') != 'post' or ec.get('effect') != 'onslaught'
            or params.get('card_id') != ec.get('card_id')
            or set(params) != {'card_id', 'count'}
            or type(count) is not int or not 0 <= count <= ec.get('max_count', 0)):
        raise ValueError('Invalid Onslaught hit count')
    import combat
    role, _ = _roles(ec)
    capacity = combat.onslaught_max_sacrifice(state, pc[role + '_region'], pc[role + '_side'])
    if count > capacity:
        raise ValueError('Onslaught hit count exceeds the participating army')
    pc['_onslaught_payment'] = {'owner': ec['owner'], 'card_id': ec['card_id'],
                                'count': count, 'remaining': count, 'paid': 0}
    _advance(state, pc)


def pay(state, pc, params):
    """Execute one legal payment without replaying the round or earlier losses."""
    import ai
    payment = pc.get('_onslaught_payment')
    if not payment:
        raise ValueError('No pending Onslaught payment')
    selected = {k: v for k, v in params.items() if k != 'card_id'}
    if params.get('card_id') != payment['card_id'] or selected not in _options(state, pc, payment):
        raise ValueError('Invalid Onslaught casualty payment')
    role, _ = _roles(payment)
    operation = ('remove_regular' if selected['kind'] == 'regular' else
                 'reduce_elite' if selected['operation'] == 'downgrade' else 'remove_elite')
    cost = ai._apply_casualty_operation(state, pc[role + '_region'], pc[role + '_side'],
                                       payment['remaining'],
                                       {'nation': ai.Nation[selected['nation']], 'operation': operation})
    payment['paid'] += cost
    payment['remaining'] -= cost
    _advance(state, pc)
