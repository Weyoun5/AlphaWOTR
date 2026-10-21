"""Physical Relentless Assault payments, without changing public game state."""
from collections import Counter


def payment_coverage(root, post, chain, facts):
    """Exclude only engine-forced later combat losses from the paid prefix.

    The actual post-state remains unchanged. The commit's ordinary physical
    obligation ledger still requires these separately forced losses on board.
    No casualty action chosen by this candidate path may use this allowance.
    """
    pc = getattr(root, 'pending_combat', None) or {}
    if ((pc.get('effect_choice') or {}).get('effect') != 'relentless_assault'
            or not chain or any(a.action_type != 'combat_effect_choice'
                or not ('count' in a.params or 'operation' in a.params)
                for a in chain)):
        return facts
    operations = getattr(post, '_casualty_operations', None) or []
    after_pc = getattr(post, 'pending_combat', None) or {}
    completed = after_pc.get('_observed_round_completed') or {}
    if (not operations or after_pc.get('_round_resume')
            or completed.get('schema') != 'observed_round_v1'
            or completed.get('round') != int(pc.get('rounds_fought', 0))
            or int(after_pc.get('rounds_fought', 0)) != int(pc.get('rounds_fought', 0)) + 1):
        return facts
    import copy
    from board import sanitize
    result = copy.copy(facts)
    result.armies = facts.armies.copy()
    for op in operations:
        region = sanitize(str(op['region'])).lower()
        nation = op['nation'].name.lower()
        operation = op['operation']
        if operation not in ('remove_regular', 'remove_elite', 'reduce_elite'):
            return facts
        kind = 'regular' if operation == 'remove_regular' else 'elite'
        result.armies[(region, nation + ':' + kind)] += 1
        if operation == 'reduce_elite' and op.get('replacement_source') is not None:
            result.armies[(region, nation + ':regular')] -= 1
    return result

def permits_payment(root, current, action, facts, window):
    root_pc = getattr(root, 'pending_combat', None) or {}
    root_effect = root_pc.get('effect_choice') or {}
    if root_effect.get('effect') != 'relentless_assault':
        return True
    pc = getattr(current, 'pending_combat', None) or {}
    effect = pc.get('effect_choice') or {}
    if (action.action_type != 'combat_effect_choice'
            or effect.get('effect') != 'relentless_assault_payment'):
        return True
    before = (root_pc.get('_round_resume') or {}).get('cursor')
    now = (pc.get('_round_resume') or {}).get('cursor')
    if before is None or now is None:
        return False
    role = 'attacker' if effect.get('owner') == 'atk' else 'defender'
    from board import sanitize
    region = sanitize(str(pc[role + '_region'])).lower()
    def rows(cursor):
        return Counter((unit.nation.name.lower() + ':' + unit.kind.name.lower())
                       for unit in getattr(cursor.combat, role + '_army').units)
    baseline, after = rows(before), rows(now)
    choice = action.params
    key = str(choice['nation']).lower() + ':' + str(choice['kind'])
    after[key] -= 1
    if choice.get('operation') == 'downgrade':
        after[str(choice['nation']).lower() + ':regular'] += 1
    from .gesture_match import _PrefixShim, window_facts
    positive, negative = Counter(), Counter()
    for record in window.records:
        for (where, kind), amount in window_facts(_PrefixShim([record], None, None)).armies.items():
            if where == region:
                if amount > 0:positive[kind] += amount
                else:negative[kind] -= amount
    for kind in set(baseline) | set(after):
        delta = after[kind] - baseline[kind]
        if delta > positive[kind] or delta < -negative[kind]:
            return False
    return True
