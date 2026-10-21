"""Prove a fully shown casualty payment before exposing its continuation."""
from collections import Counter


def is_round_unit_payment(state, action):
    """A staged combat-card casualty, chosen by its actual owner."""
    if action.action_type != 'combat_effect_choice':
        return False
    pending = (getattr(state, 'pending_combat', None) or {}).get('effect_choice') or {}
    return (pending.get('stage') == 'round'
            and pending.get('effect') in ('pre_combat_casualty', 'relentless_assault_payment')
            and all(key in action.params for key in ('nation', 'kind', 'operation')))


def apply_observed_casualty(state, action):
    """The engine's clone/execute operation, retaining its stopped prefix.

    A stopped clone is only evidence; callers must prove the full human
    payment and unanimous continuation before using any of its rows.
    """
    import ai
    round_payment = is_round_unit_payment(state, action)
    clone = ai.clone_state(state)
    try:
        ai._execute_action(clone, action)
    except BaseException as error:
        if type(error).__name__ == 'NeedClientCall':
            if round_payment:
                # Same-region sieges use split participant armies. A random
                # request interrupts resume_combat before its ordinary paused
                # writeback, so publish the *stopped clone* for physical rows.
                # The authoritative state and journal remain uncommitted.
                import combat
                payload = clone.pending_combat['_round_resume']
                combat._publish_paused_combat(clone, payload['cursor'].combat, payload)
                error._round_payment_prefix = True
            error._casualty_pre_request_state = clone
        raise
    return clone


def completed_payment_key(state, side, window, chain, call):
    """No speculative dice: apply only deterministic legal loss primitives."""
    import ai
    import emitter
    from . import gesture_match as gm
    from .casualty_gestures import event_flow, prefix_matches, equivalence_state
    effect = ((getattr(state, 'pending_combat', None) or {}).get('effect_choice') or {}).get('effect')
    from .round_payment_gestures import EFFECTS
    if effect in EFFECTS:
        from .round_payment_gestures import completed_payment_key as round_key
        return round_key(state, side, window, chain, call)
    if ((getattr(state, 'pending_combat', None) or {}).get(
            'effect_choice', {}).get('effect') == 'onslaught_payment'):
        from .onslaught_gestures import completed_payment_key as onslaught_key
        return onslaught_key(state, side, window, chain, call)
    context = emitter.casualty_context(state)
    if (context is None or context['kind'] != 'combat' or context['side'] is not side
            or not chain or any(a.action_type != 'casualty_allocate' for a in chain)):
        return None
    clone = ai.clone_state(state)
    clone._casualty_operations = []
    remaining = context['hits']
    flow = Counter(); events = []
    try:
        for action in chain:
            if int(action.params.get('casualties', -1)) != remaining:
                return None
            remaining -= ai._apply_casualty_operation(clone, context['region'], side, remaining, action.params)
            rows = event_flow(clone._casualty_operations[-1], gm._unit_type_key, gm.sanitize)
            events.extend(rows); flow.update(rows)
    except (KeyError, TypeError, ValueError, StopIteration):
        return None
    if remaining != 0:
        return None
    from .casualty_gestures import observed_unit_flow
    records, _required = observed_unit_flow(window.records)
    observed = sum(records, Counter())
    if flow != observed or not prefix_matches(events, records):
        return None
    facts = gm.window_facts(window)
    if not gm._covers(facts, gm.delta_facts(state, clone)):
        return None
    if not _authenticated_continuation(state, call):
        return None
    return gm._semantic_state_key(equivalence_state(clone, chain))


def _authenticated_continuation(state, call):
    """A proved loss can expose only its exact engine-owned continuation."""
    kind, kw = getattr(call, 'kind', None), getattr(call, 'kw', {})
    if kind == 'd6':
        return str(kw.get('ctx', '')).startswith('post-casualty ')
    if kind != 'draws':
        return False
    pc = getattr(state, 'pending_combat', None) or {}
    deck = pc.get('_wk_sorcerer_deck')
    key = ('SHADOW', deck)
    stopped = getattr(call, '_casualty_pre_request_state', None)
    if (deck not in ('character', 'strategy') or list(kw.get('items', ())) != [key]
            or stopped is None or getattr(call, '_physical_pre_request_state', None) is not stopped
            or getattr(call, '_physical_request_owner', None) is None):
        return False
    # The real grant site consumes the queued Sorcerer trigger and stamps one
    # owed draw on this stopped clone. A matching-shaped speculative request
    # without that grant is never authority to draw a physical card.
    stopped_pc = getattr(stopped, 'pending_combat', None) or {}
    before_debt = (getattr(state, '_owed_draws', None) or {}).get(key, 0)
    after_debt = (getattr(stopped, '_owed_draws', None) or {}).get(key, 0)
    return '_wk_sorcerer_deck' not in stopped_pc and after_debt == before_debt + 1
