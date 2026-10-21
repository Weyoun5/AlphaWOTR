"""Independent physical-outcome acceptance for Rulebook p30 casualties.

Fixtures roll a real public combat round with literal dice. Expected armies
and supply totals are written here, never obtained from another allocator.
Menus may contain complete plans or sequential operations; follow the same
owner only, stopping before the defender/next-round decision. The GAME04
composition is a captured missing choice; supply fixtures are isolated rules
examples, not claims about historical log incidence.
"""
from collections import Counter
from copy import deepcopy
import unittest

import ai
import board
import game_state as gs
from test_audit0905_casualty_order import position, rolled, SOURCE, TARGET, FP, SH, REGULAR, ELITE

S, I, G, E = gs.Nation.SAURON, gs.Nation.ISENGARD, gs.Nation.GONDOR, gs.Nation.ELVES
# Printed component counts, independent of ai.NATION_MAX_FIGURES.
REGULAR_CAP = {S: 36, I: 12, G: 15}


def army_counts(state, region=SOURCE):
    return Counter((u.nation, u.kind) for key, army in state.armies.items()
                   if board.sanitize(key) == region for u in army.units)


def box(state, nation, kind):
    return (state.fp_casualties.get(nation) or {}).get(kind, 0)


def reserve(state, nation):
    deployed = sum(u.nation == nation and u.kind == REGULAR
                   for army in state.armies.values() for u in army.units)
    return REGULAR_CAP[nation] - deployed - box(state, nation, 'regular')


def set_regular_supply(state, nation, available):
    """Deploy physical figures elsewhere, <=10 per army; never mock supply."""
    missing = reserve(state, nation) - available
    for region in ('gorgoroth', 'nurn', 'barad dur', 'minas morgul', 'morannon'):
        if missing <= 0:
            break
        army = gs.Army(state.board.region(region))
        count = min(missing, 10)
        army.units = [gs.Unit(nation, REGULAR) for _ in range(count)]
        state.armies[region] = army
        missing -= count
    if missing:
        raise AssertionError(('invalid physical supply fixture', missing))
    if reserve(state, nation) != available:
        raise AssertionError('literal physical supply fixture is inconsistent')


def allocation_leaves(state, owner, depth=0):
    pc = state.pending_combat or {}
    if pc.get('phase') != 'casualty_allocation' or pc.get('deciding_side') != owner:
        return [((), state)]
    if depth >= 12:
        raise AssertionError('casualty sequence did not finish within 12 operations')
    actions = [a for a in ai.get_legal_actions(state, owner)
               if a.action_type == 'casualty_allocate']
    if not actions:
        raise AssertionError('pending casualty owner has no allocation actions')
    result = []
    for action in actions:
        successor = ai.apply_action(state, action)
        result.extend(((action,) + path, leaf)
                      for path, leaf in allocation_leaves(successor, owner, depth + 1))
    return result


def signature(state):
    return (tuple(sorted((str(k), tuple(sorted((u.nation.name, u.kind.name)
                    for u in a.units)), tuple(sorted(repr(l) for l in a.leaders)))
                    for k, a in state.armies.items())),
            deepcopy(state.fp_casualties), deepcopy(state.pending_combat))


class CasualtyComposition(unittest.TestCase):
    def outcomes(self, state, side=SH):
        return allocation_leaves(rolled(state, attacker_hits=0, defender_hits=2), side)

    def test_game04_whole_isengard_elite_preserves_five_sauron_regulars(self):
        state = position(attack_units=(5, 1))
        state.armies[SOURCE].units[-1].nation = I
        state.political.boxes[I] = 3
        expected = Counter({(S, REGULAR): 5})
        leaves = self.outcomes(state)
        self.assertTrue(any(army_counts(leaf) == expected for _, leaf in leaves),
                        [army_counts(leaf) for _, leaf in leaves])
        self.assertTrue(any(army_counts(leaf) == Counter({(S, REGULAR): 3, (I, ELITE): 1})
                            for _, leaf in leaves))
        for _, leaf in leaves:
            self.assertGreaterEqual(reserve(leaf, S), 0)
            self.assertGreaterEqual(reserve(leaf, I), 0)

    def test_same_nation_whole_elite_and_two_regular_choices_both_exist(self):
        leaves = self.outcomes(position(attack_units=(2, 1)))
        outcomes = [army_counts(leaf) for _, leaf in leaves]
        for expected in (Counter({(S, REGULAR): 2}), Counter({(S, ELITE): 1})):
            self.assertIn(expected, outcomes)

    def test_distinct_result_paths_have_distinct_encoded_action_sequences(self):
        state = position(attacker=FP, attack_units=(2, 1))
        pending = rolled(state, attacker_hits=0, defender_hits=2)
        encodings = {}
        results = set()
        for path, leaf in allocation_leaves(pending, FP):
            current = pending
            codes = []
            for action in path:
                codes.append(ai.encode_action(action, current, FP).tobytes())
                current = ai.apply_action(current, action)
            physical = (tuple(sorted((n.name, k.name, c) for (n, k), c in army_counts(leaf).items())),
                        box(leaf, G, 'regular'), box(leaf, G, 'elite'))
            key = tuple(codes)
            if key in encodings:
                self.assertEqual(encodings[key], physical, 'different physical results collide in policy input')
            encodings[key] = physical
            results.add(physical)
        self.assertGreaterEqual(len(results), 3,
            'FP direct Elite loss, reduce-then-remove Regular, and two Regular losses differ physically')

    def test_one_hit_without_replacement_eliminates_elite_without_minting(self):
        state = position(attack_units=(0, 1))
        set_regular_supply(state, S, 0)
        pending = rolled(state, attacker_hits=0, defender_hits=1)
        leaves = allocation_leaves(pending, SH)
        self.assertEqual(len(leaves), 1)
        for path, leaf in leaves:
            self.assertEqual(path, (), 'a sole mandatory loss must not be learned')
            self.assertEqual(army_counts(leaf), Counter())
            self.assertEqual(reserve(leaf, S), 0)

    def test_two_elites_one_replacement_offers_surviving_elite_or_regular(self):
        state = position(attack_units=(0, 2))
        set_regular_supply(state, S, 1)
        leaves = self.outcomes(state)
        outcomes = [army_counts(leaf) for _, leaf in leaves]
        self.assertIn(Counter({(S, ELITE): 1}), outcomes)
        self.assertIn(Counter({(S, REGULAR): 1}), outcomes)
        for _, leaf in leaves:
            self.assertGreaterEqual(reserve(leaf, S), 0)

    def test_shadow_removed_regular_can_replace_elite_in_same_allocation(self):
        state = position(attack_units=(1, 1))
        set_regular_supply(state, S, 0)
        leaves = self.outcomes(state)
        survivors = [leaf for _, leaf in leaves if army_counts(leaf) == Counter({(S, REGULAR): 1})]
        self.assertTrue(survivors, [army_counts(leaf) for _, leaf in leaves])
        for leaf in survivors:
            self.assertEqual(reserve(leaf, S), 0)
            self.assertEqual(box(leaf, S, 'regular'), 0)

    def test_fp_whole_elite_is_permanent_casualty(self):
        leaves = self.outcomes(position(attacker=FP, attack_units=(2, 1)), FP)
        whole_elite = [leaf for _, leaf in leaves
                       if army_counts(leaf) == Counter({(G, REGULAR): 2})
                       and box(leaf, G, 'regular') == 0]
        self.assertTrue(whole_elite)
        for leaf in whole_elite:
            self.assertEqual(box(leaf, G, 'elite'), 1)
            self.assertEqual(box(leaf, G, 'regular'), 0)
            self.assertEqual(reserve(leaf, G), 13)

    def test_fp_prior_casualty_replacement_is_reused_once(self):
        state = position(attacker=FP, attack_units=(0, 2))
        state.fp_casualties[G] = {'regular': 1, 'elite': 0}
        set_regular_supply(state, G, 0)
        leaves = self.outcomes(state, FP)
        regular = [leaf for _, leaf in leaves if army_counts(leaf) == Counter({(G, REGULAR): 1})]
        self.assertTrue(regular)
        for leaf in regular:
            self.assertEqual(box(leaf, G, 'regular'), 0)
            self.assertEqual(box(leaf, G, 'elite'), 2)
            self.assertEqual(reserve(leaf, G), 0)

    def test_excess_hits_leave_frozen_peaceful_leader_guard_untouched(self):
        state = position(attacker=FP, attack_units=(1, 1))
        state.armies[SOURCE].leaders = [gs.Leader(E)]
        state.political.boxes[E] = 0
        # A literal frozen formation: the sole Gondor Regular guards the
        # peaceful Elven leader; only the Gondor Elite participated.
        state.pending_combat['attacker_rearguard'] = (SOURCE, FP.name,
            ('forced_rearguard_v1', (('GONDOR', 'REGULAR', 1),), (('ELVES', 1),)))
        pending = rolled(state, attacker_hits=0, defender_hits=3)
        leaves = allocation_leaves(pending, FP)
        self.assertTrue(leaves)
        self.assertTrue(any(box(leaf, G, 'regular') == 0 for _, leaf in leaves))
        for path, leaf in leaves:
            self.assertEqual(army_counts(leaf), Counter({(G, REGULAR): 1}))
            self.assertEqual([l.nation for l in leaf.armies[SOURCE].leaders], [E])
            self.assertEqual(box(leaf, G, 'elite'), 1)
            # FP may deliberately reduce then lose its replacement Regular.
            # That changes its permanent pool, so the board wipe alone does
            # not make every such allocation a unique physical outcome.
            self.assertIn(box(leaf, G, 'regular'), (0, 1))

    def test_unique_wipe_is_forced(self):
        pending = rolled(position(attack_units=(1, 1)), attacker_hits=0, defender_hits=3)
        leaves = allocation_leaves(pending, SH)
        self.assertEqual(len(leaves), 1)
        self.assertEqual(leaves[0][0], ())
        self.assertEqual(army_counts(leaves[0][1]), Counter())

    def test_unknown_operation_rejects_without_mutating_input(self):
        pending = rolled(position(attack_units=(2, 1)), attacker_hits=0, defender_hits=2)
        offered = next(a for a in ai.get_legal_actions(pending, SH) if a.action_type == 'casualty_allocate')
        bad = deepcopy(offered)
        bad.params['operation'] = 'invent_a_casualty'
        before = signature(pending)
        with self.assertRaises((ValueError, RuntimeError)):
            ai.apply_action(pending, bad)
        self.assertEqual(signature(pending), before)

    def test_opposing_nation_loss_rejects_without_mutating_input(self):
        pending = rolled(position(attack_units=(2, 1)), attacker_hits=0, defender_hits=2)
        offered = next(a for a in ai.get_legal_actions(pending, SH) if a.action_type == 'casualty_allocate')
        bad = deepcopy(offered)
        bad.params.update(operation='remove_regular', nation=G)
        before = signature(pending)
        with self.assertRaises((ValueError, RuntimeError)):
            ai.apply_action(pending, bad)
        self.assertEqual(signature(pending), before)

    def test_whole_elite_cannot_absorb_one_hit_when_replacement_exists(self):
        pending = rolled(position(attack_units=(2, 1)), attacker_hits=0, defender_hits=1)
        offered = next(a for a in ai.get_legal_actions(pending, SH) if a.action_type == 'casualty_allocate')
        bad = deepcopy(offered)
        bad.params.update(operation='remove_elite', nation=S)
        before = signature(pending)
        with self.assertRaises((ValueError, RuntimeError)):
            ai.apply_action(pending, bad)
        self.assertEqual(signature(pending), before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
