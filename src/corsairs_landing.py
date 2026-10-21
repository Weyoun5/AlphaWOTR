"""Corsairs landing completes, including overstack, before combat starts."""
from game_state import Side
import copy


def begin(state, card, context):
    import ai, event_effects as ee
    from combat import side_of_nation
    target = context['defender_region']
    army = ai._armies_lookup(state, target)
    ours = [unit for unit in army.units if side_of_nation(unit.nation) == Side.SHADOW]
    if not context.get('corsairs_relief_garrison') and len(ours) > 10:
        types = sorted({(u.nation.value, u.kind.name.lower()) for u in ours})
        state.pending_card_option = {
            'card_id': card.card_id, 'card_name': card.card_name,
            'side': 'sh', 'decider': 'sh', 'direct': True,
            'options': [{'overstack_remove': {'region': target, 'nation': nation, 'kind': kind},
                         '_corsairs_landing_pc': dict(context)} for nation, kind in types]}
        return
    state.pending_combat = context
    ai._post_rearguard_transition(state, context)


def remove(state, card, choices):
    import ai, event_effects as ee
    from combat import side_of_nation
    context = copy.deepcopy(choices['_corsairs_landing_pc'])
    target = context['defender_region']
    army = ai._armies_lookup(state, target)
    spec = choices.get('overstack_remove') or {}
    if (spec.get('region') != target or context.get('corsairs_relief_garrison')
            or sum(side_of_nation(u.nation) == Side.SHADOW for u in army.units) <= 10):
        raise ValueError('No Corsairs landing overstack to remove')
    if not ee._overstack_remove_one(state, army, Side.SHADOW, spec):
        raise ValueError('Selected Corsairs overstack figure absent')
    begin(state, card, context)
    return {'effects': ['Corsairs landing: excess unit returned before battle']}
