"""Separate a shown Onslaught payment from its compulsory extra-attack losses."""
from collections import Counter
from copy import copy


def is_payment_action(action):
    """Only the explicit unit-payment action, never the count or another card."""
    params = getattr(action, 'params', None) or {}
    if (getattr(action, 'action_type', None) != 'combat_effect_choice'
            or set(params) != {'card_id', 'nation', 'kind', 'operation'}):
        return False
    from cards import get_card
    card = get_card(params['card_id'])
    return card is not None and card.combat_name.casefold() == 'onslaught'


def _payment_prefix(root, chain, side):
    """Replay only legal loss primitives, never the extra attack or its RNG."""
    pc = getattr(root, 'pending_combat', None) or {}
    effect = pc.get('effect_choice') or {}
    payment = pc.get('_onslaught_payment') or {}
    owner = payment.get('owner')
    if (pc.get('phase') != 'combat_effect_choice'
            or effect.get('stage') != 'onslaught_payment'
            or effect.get('effect') != 'onslaught_payment'
            or owner not in ('atk', 'def') or effect.get('owner') != owner
            or payment.get('card_id') != effect.get('card_id') or not chain):
        return None
    role = 'attacker' if owner == 'atk' else 'defender'
    if pc.get('deciding_side') is not side or pc.get(role + '_side') is not side:
        return None
    import ai
    paid = ai.clone_state(root)
    paid._casualty_operations = []
    remaining = payment.get('remaining')
    if type(remaining) is not int or remaining <= 0:
        return None
    try:
        for action in chain:
            params = action.params or {}
            if (not is_payment_action(action)
                    or params['card_id'] != payment['card_id'] or remaining <= 0):
                return None
            operations = {('regular', 'remove'): 'remove_regular',
                          ('elite', 'remove'): 'remove_elite',
                          ('elite', 'downgrade'): 'reduce_elite'}
            operation = operations[(params['kind'], params['operation'])]
            cost = ai._apply_casualty_operation(
                paid, pc[role + '_region'], side, remaining,
                {'nation': ai.Nation[params['nation']], 'operation': operation})
            remaining -= cost
            ledger = paid.pending_combat['_onslaught_payment']
            ledger['paid'] += cost
            ledger['remaining'] = remaining
    except (KeyError, TypeError, ValueError, StopIteration):
        return None
    # An unreplaceable Elite can exhaust the army before the declared count.
    # The engine still rolls for the hits actually paid in that legal case.
    if remaining and ai._casualty_operations(paid, pc[role + '_region'], side, remaining):
        return None
    return paid


def completed_payment_key(root, side, window, chain, call):
    """Bind a grouped payment's unanimous pending roll to its actual window."""
    paid = _payment_prefix(root, chain, side)
    if paid is None:
        return None
    pc = paid.pending_combat
    ledger = pc['_onslaught_payment']
    role = 'attacker' if ledger['owner'] == 'atk' else 'defender'
    expected_call = dict(need=ledger['paid'], ctx='post-casualty Onslaught',
                         side=side, role=role)
    if getattr(call, 'kind', None) != 'd6' or getattr(call, 'kw', {}) != expected_call:
        return None
    import ai
    import combat
    from . import gesture_match as gm
    from .casualty_gestures import event_flow, prefix_matches, equivalence_state
    events = []
    for event in paid._casualty_operations:
        events.extend(event_flow(event, gm._unit_type_key, gm.sanitize))
    from .casualty_gestures import observed_unit_flow
    records, _required = observed_unit_flow(window.records)
    observed = sum(records, Counter())
    if Counter(events) != observed or not prefix_matches(events, records):
        return None
    if not gm._covers(gm.window_facts(window), gm.delta_facts(root, paid)):
        return None
    if not ai._alloc_side_units(paid, pc[role + '_region'], side):
        combat.reconcile_destroyed_army(paid, pc[role + '_region'], side)
    return gm._semantic_state_key(equivalence_state(paid, chain))


def payment_coverage(root, post, chain, facts, side):
    """Authenticate the paid prefix without rolling or changing the post-state.

    Only the final payment can cause this problem: its recorded extra attack
    may also resolve unique enemy casualties and finish the battle. Those
    compulsory rows remain owed by the normal commit/emission machinery.
    """
    paid = _payment_prefix(root, chain, side)
    if paid is None:
        return facts, False
    import ai
    from board import sanitize
    from .gesture_match import delta_facts, _unit_type_key
    from .casualty_gestures import event_flow
    pc = root.pending_combat
    role = 'attacker' if pc['_onslaught_payment']['owner'] == 'atk' else 'defender'
    other = 'defender' if role == 'attacker' else 'attacker'

    # Each apply has fresh witnesses: first its selected payment, then only
    # forced enemy allocations. No later selected casualty action is allowed.
    actual = list(getattr(post, '_casualty_operations', None) or ())
    expected = paid._casualty_operations[-1]
    fields = ('region', 'side', 'nation', 'operation', 'cost',
              'remaining_before', 'replacement_source')
    if not actual or any(actual[0].get(k) != expected.get(k) for k in fields):
        return facts, False
    expected_followthrough = ai.clone_state(paid)
    import combat
    if not ai._alloc_side_units(expected_followthrough, pc[role + '_region'], side):
        combat.reconcile_destroyed_army(expected_followthrough, pc[role + '_region'], side)
    forced = Counter()
    for event in actual[1:]:
        if (event.get('side') is not pc[other + '_side']
                or sanitize(str(event.get('region'))).lower()
                   != sanitize(str(pc[other + '_region'])).lower()
                or event.get('operation') not in
                   ('remove_regular', 'remove_elite', 'reduce_elite')):
            return facts, False
        # Replay only these deterministic casualty primitives on a private
        # clone. No extra attack, random draw, advance or capture is run.
        try:
            ai._apply_casualty_operation(
                expected_followthrough, event['region'], event['side'],
                event['remaining_before'],
                {'nation': event['nation'], 'operation': event['operation']})
        except (KeyError, TypeError, ValueError, StopIteration):
            return facts, False
        witness = expected_followthrough._casualty_operations[-1]
        if any(witness.get(k) != event.get(k) for k in fields):
            return facts, False
        for key, sign in event_flow(event, _unit_type_key, sanitize):
            forced[key] += sign

    # Every army difference beyond the independently proved payment must be
    # exactly the compulsory enemy witnesses, including any replacements.
    payment_rows = delta_facts(root, paid).armies
    difference = {key: facts.armies.get(key, 0) - payment_rows.get(key, 0)
                  for key in set(facts.armies) | set(payment_rows)}
    if ({key: value for key, value in difference.items() if value}
            != {key: value for key, value in forced.items() if value}):
        return facts, False
    if not ai._alloc_side_units(expected_followthrough, pc[other + '_region'], pc[other + '_side']):
        combat.reconcile_destroyed_army(expected_followthrough, pc[other + '_region'], pc[other + '_side'])
    # Combat-flux matching permits compulsory figure removal. Bind that
    # permission to the exact figures of the armies actually wiped here,
    # rather than accepting arbitrary removals elsewhere or among survivors.
    expected_figures = delta_facts(root, expected_followthrough)
    for field in ('named', 'leaders'):
        if ({key: value for key, value in getattr(facts, field).items() if value}
                != {key: value for key, value in getattr(expected_figures, field).items() if value}):
            return facts, False
    result = copy(facts)
    result.armies = payment_rows.copy()
    return result, True
