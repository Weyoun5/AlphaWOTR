"""Authenticate completed pre-roll unit payments without advancing any dice."""
from collections import Counter

EFFECTS = frozenset(('relentless_assault_payment', 'pre_combat_casualty'))


def is_action(action):
    params = getattr(action, 'params', None) or {}
    return (getattr(action, 'action_type', None) == 'combat_effect_choice'
            and set(params) == {'card_id', 'nation', 'kind', 'operation'})


def payment_prefix(root, side, chain):
    """Use the executor's deterministic payment primitive, never round advance."""
    import ai
    import combat
    import combat_round_resume
    from . import gesture_match as gm
    pc = getattr(root, 'pending_combat', None) or {}
    effect = pc.get('effect_choice') or {}
    payload = pc.get('_round_resume') or {}
    cursor = payload.get('cursor')
    owner = effect.get('owner')
    if (pc.get('phase') != 'combat_effect_choice' or effect.get('stage') != 'round'
            or effect.get('effect') not in EFFECTS or owner not in ('atk', 'def')
            or pc.get('deciding_side') is not side or cursor is None or not chain
            or not cursor.pending or cursor.pending.get('effect') != effect['effect']
            or cursor.pending.get('owner') != owner
            or getattr(cursor.combat, '_live_state', None) is not root):
        return None
    role = 'attacker' if owner == 'atk' else 'defender'
    if pc.get(role+'_side') is not side:
        return None
    is_ra = effect['effect'] == 'relentless_assault_payment'
    remaining = cursor.ra_remaining if is_ra else cursor.pre_remaining
    if type(remaining) is not int or remaining <= 0:
        return None
    clone = ai.clone_state(root)
    payload = clone.pending_combat['_round_resume']
    cursor = payload['cursor']
    old_active = combat.set_active_state(clone)
    previous = combat._BATTLE_DG, combat._DG_AVAIL0
    combat._BATTLE_DG = dict(payload['downgrades'])
    combat._DG_AVAIL0 = dict(payload['reinforcement_baseline'])
    events = []
    try:
        for action in chain:
            if (not is_action(action) or action.params['card_id'] != effect['card_id']
                    or remaining <= 0):
                return None
            before = ai.clone_state(clone)
            cursor.selection = {key:value for key,value in action.params.items() if key != 'card_id'}
            pending, cost = combat_round_resume._payment(
                cursor, effect['effect'], owner == 'atk', remaining)
            if pending is not None or cost <= 0 or cost > remaining:
                return None
            remaining -= cost
            if is_ra:
                cursor.ra_remaining = remaining
                cursor.ra_paid += cost
                setattr(cursor.combat, role+'_dice_mod', getattr(cursor.combat, role+'_dice_mod')+cost)
            else:
                cursor.pre_remaining = remaining
                cursor.pre_paid += cost
            payload['downgrades'] = dict(combat._BATTLE_DG)
            combat._publish_paused_combat(clone, cursor.combat, payload)
            delta = gm.delta_facts(before, clone).armies
            for sign in (-1,1):
                for key,value in delta.items():
                    if value*sign > 0:
                        events.extend([(key,sign)]*abs(value))
        if remaining:
            return None
        return clone, events
    except (KeyError, TypeError, ValueError, StopIteration):
        return None
    finally:
        combat._BATTLE_DG, combat._DG_AVAIL0 = previous
        combat.set_active_state(old_active)


def completed_payment_key(root, side, window, chain, call):
    from . import gesture_match as gm
    from .casualty_gestures import prefix_matches, equivalence_state
    if (getattr(call, 'kind', None) != 'd6'
            or not getattr(call, '_round_payment_prefix', False)):
        return None
    stopped = getattr(call, '_casualty_pre_request_state', None)
    if stopped is None:
        return None
    prefix = payment_prefix(root, side, chain)
    if prefix is None:
        return None
    paid, events = prefix
    from .casualty_gestures import observed_unit_flow
    records, _required = observed_unit_flow(window.records)
    observed = sum(records, Counter())
    if Counter(events) != observed or not prefix_matches(events, records):
        return None
    facts = gm.window_facts(window)
    # Both the deterministic proof and the real stopped executor must expose
    # precisely the paid board changes, including any Regular replacement.
    if not gm._covers(facts, gm.delta_facts(root, paid)):
        return None
    if not gm._covers(facts, gm.delta_facts(root, stopped)):
        return None
    return gm._semantic_state_key(equivalence_state(paid, chain))
