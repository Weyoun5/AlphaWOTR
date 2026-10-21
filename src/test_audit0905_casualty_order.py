"""RB30 / Almanac36,43: attacker casualties precede defender casualties.

The fixture begins at the public round card-declaration boundary. Only dice
are scripted; combat, card effects, legal menus and actions run production
code. Literal piece counts and printed timing are the expected oracle.
Whole-Elite and nationality composition choices are separate audit work.
"""
from collections import Counter
from math import comb
import unittest
from unittest.mock import patch

import ai
import board
import combat
import game_logic as gl
import game_state as gs

FP, SH = gs.Side.FREE_PEOPLES, gs.Side.SHADOW
REGULAR, ELITE = gs.UnitKind.REGULAR, gs.UnitKind.ELITE
SOURCE, TARGET = 'west harondor', 'pelargir'
GREAT_HOST = 'SHADOW_Strategy_Stormcrow'


def pieces(state, region):
    army = next(a for k, a in state.armies.items() if board.sanitize(k) == region)
    return Counter(u.kind for u in army.units)


def position(attacker=SH, attack_units=(2, 1), defend_units=(2, 1), great_host=False):
    state = gs.setup_initial_game(board.load_board())
    gl.initialize_decks(state)
    defender = FP if attacker == SH else SH
    # Isolated legal midgame armies, below physical supply caps; all other
    # armies removed so replacement Regular availability is unambiguous.
    state.armies = {}
    for region, side, counts in ((SOURCE, attacker, attack_units),
                                 (TARGET, defender, defend_units)):
        army = gs.Army(state.board.region(region))
        nation = gs.Nation.SAURON if side == SH else gs.Nation.GONDOR
        army.units = ([gs.Unit(nation, REGULAR) for _ in range(counts[0])]
                      + [gs.Unit(nation, ELITE) for _ in range(counts[1])])
        army.leaders = []
        state.armies[region] = army
        state.political.boxes[nation] = 3
        state.political.active[nation] = True
    for side in (FP, SH):
        state.hands[side].hand = [GREAT_HOST] if great_host and side == SH else []
    state.pending_combat = dict(attacker_region=SOURCE, defender_region=TARGET,
        attacker_side=attacker, defender_side=defender,
        phase='attacker_combat_card', deciding_side=attacker,
        rounds_fought=0, is_siege=False)
    return state


def play(state, side, kind, **params):
    matches = [a for a in ai.get_legal_actions(state, side)
               if a.action_type == kind
               and all(a.params.get(k) == v for k, v in params.items())]
    if len(matches) != 1:
        raise AssertionError((side, kind, params, matches, state.pending_combat))
    return ai.apply_action(state, matches[0])


def rolled(state, *, attacker_hits=1, defender_hits=1):
    """One literal six per requested hit, all remaining faces one.

    Both attack strengths are <=5 and no leaders are present. Natural six
    hits even against the first-round city defense, natural one misses.
    The returned sample is validated against each actual dice request.
    """
    atk, deff = state.pending_combat['attacker_side'], state.pending_combat['defender_side']
    packets = {atk: attacker_hits, deff: defender_hits}
    seen = []
    def dice(count, reason, side=None):
        if side not in packets or side in seen or reason != 'combat roll':
            raise AssertionError(('unexpected dice request', count, reason, side, seen))
        seen.append(side)
        hits = packets[side]
        if hits > count:
            raise AssertionError(('invalid fixture hit count', hits, count))
        return [6] * hits + [1] * (count - hits)
    with patch.object(combat, 'roll_d6_batch', side_effect=dice):
        for side in (atk, deff):
            if GREAT_HOST in state.hands[side].hand:
                state = play(state, side, 'combat_card_play', card_id=GREAT_HOST)
            else:
                state = play(state, side, 'combat_card_none')
    if seen != [atk, deff]:
        raise AssertionError(('both rolls must precede casualty choices', seen))
    return state


class CasualtyOrder(unittest.TestCase):
    def assert_owner(self, state, side, stage):
        pc = state.pending_combat or {}
        self.assertEqual((pc.get('phase'), pc.get('deciding_side'), pc.get('alloc_stage')),
                         ('casualty_allocation', side, stage))
        self.assertEqual(ai._pending_decision_side(state), side)

    def test_shadow_attacker_allocates_first_after_both_rolls(self):
        state = rolled(position())
        self.assert_owner(state, SH, 'attacker')
        self.assertEqual(pieces(state, SOURCE), Counter({REGULAR: 2, ELITE: 1}))
        self.assertEqual(pieces(state, TARGET), Counter({REGULAR: 2, ELITE: 1}))

    def test_standalone_battle_applies_attacker_then_defender_losses(self):
        # Observe the application boundary while executing the real casualty
        # implementation. Expected order is RB30, never another engine path.
        for attacker in (SH, FP):
            with self.subTest(attacker=attacker):
                state = position(attacker=attacker)
                defender = FP if attacker == SH else SH
                original = combat.apply_casualties
                calls = []
                def observe(army, hits, *args, **kwargs):
                    nation = army.units[0].nation
                    side = SH if nation == gs.Nation.SAURON else FP
                    before = Counter(u.kind for u in army.units)
                    result = original(army, hits, *args, **kwargs)
                    calls.append((side, hits, before, Counter(u.kind for u in army.units)))
                    return result
                with patch.object(combat, 'roll_d6_batch',
                                  side_effect=lambda n, *a, **k: [6] + [1] * (n - 1)), \
                     patch.object(combat, 'apply_casualties', side_effect=observe):
                    gl.battle(state, SOURCE, TARGET, max_rounds=1,
                              choices={'attacker_side': attacker})
                self.assertEqual(calls, [
                    (attacker, 1, Counter({REGULAR: 2, ELITE: 1}), Counter({REGULAR: 1, ELITE: 1})),
                    (defender, 1, Counter({REGULAR: 2, ELITE: 1}), Counter({REGULAR: 1, ELITE: 1}))])
                self.assertEqual(pieces(state, SOURCE), Counter({REGULAR: 1, ELITE: 1}))
                self.assertEqual(pieces(state, TARGET), Counter({REGULAR: 1, ELITE: 1}))

    def test_exact_kernel_branches_open_attacker_first_for_both_real_choices(self):
        # Each side has three figures (2R+1E), hence three dice. Pelargir
        # is a city: attacker hits only on 6 in round one, defender on 5/6.
        # Sixteen independent binomial pairs, each hit count in 0..3;
        # nine pairs have real allocation choices for BOTH armies.
        for attacker in (SH, FP):
            with self.subTest(attacker=attacker):
                defender = FP if attacker == SH else SH
                state = play(position(attacker=attacker), attacker, 'combat_card_none')
                action = next(a for a in ai.get_legal_actions(state, defender)
                              if a.action_type == 'combat_card_none')
                with patch.object(combat, 'roll_d6_batch',
                                  side_effect=lambda n, *a, **k: [1] * n):
                    probe = ai.apply_action(state, action)
                parent = ai.MCTSNode(state, defender)
                node = ai.MCTSNode(None, defender, parent=parent, action=action)
                node.is_chance = True
                with patch.object(ai, '_KERNEL_K_MAX', 64), patch.object(ai, 'CHANCE_K_MAX', 0):
                    self.assertTrue(ai._kernel_expand_combat(node, probe))
                masses = {}
                both_choice = 0
                for branch in node.outcomes:
                    pc = branch.state.pending_combat
                    hits_to_attacker = pc.get('atk_hits', 0)
                    hits_to_defender = pc.get('def_hits', 0)
                    key = hits_to_attacker, hits_to_defender
                    self.assertNotIn(key, masses)
                    masses[key] = branch.prior
                    if hits_to_attacker and hits_to_defender:
                        both_choice += 1
                        self.assert_owner(branch.state, attacker, 'attacker')
                        self.assertEqual(pieces(branch.state, SOURCE), Counter({REGULAR: 2, ELITE: 1}))
                        self.assertEqual(pieces(branch.state, TARGET), Counter({REGULAR: 2, ELITE: 1}))
                        after_attacker = branch.state
                        # Literal Regular-first path; each hit is now a
                        # separate owner choice while alternatives remain.
                        while (after_attacker.pending_combat or {}).get('alloc_stage') == 'attacker':
                            after_attacker = play(after_attacker, attacker, 'casualty_allocate', operation='remove_regular')
                        self.assert_owner(after_attacker, defender, 'defender')
                        expected_survivors = {1: Counter({REGULAR: 1, ELITE: 1}),
                                              2: Counter({ELITE: 1}),
                                              3: Counter({REGULAR: 1})}
                        self.assertEqual(pieces(after_attacker, SOURCE), expected_survivors[hits_to_attacker])
                        self.assertEqual(pieces(after_attacker, TARGET), Counter({REGULAR: 2, ELITE: 1}))
                    elif hits_to_attacker:
                        self.assert_owner(branch.state, attacker, 'attacker')
                    elif hits_to_defender:
                        self.assert_owner(branch.state, defender, 'defender')
                    else:
                        self.assertEqual(pc['phase'], 'attacker_continue')
                self.assertEqual(both_choice, 9)
                self.assertEqual(set(masses), {(ha, hd) for ha in range(4) for hd in range(4)})
                for (ha, hd), mass in masses.items():
                    expected = (comb(3, ha) * (1 / 3) ** ha * (2 / 3) ** (3 - ha)
                                * comb(3, hd) * (1 / 6) ** hd * (5 / 6) ** (3 - hd))
                    self.assertAlmostEqual(mass, expected)
                self.assertAlmostEqual(node.kernel_cover, 1.)

    def test_free_peoples_attacker_allocates_first_after_both_rolls(self):
        self.assert_owner(rolled(position(attacker=FP)), FP, 'attacker')

    def test_defender_observes_applied_attacker_losses_before_choosing(self):
        state = rolled(position())
        self.assert_owner(state, SH, 'attacker')
        state = play(state, SH, 'casualty_allocate', operation='remove_regular')
        self.assert_owner(state, FP, 'defender')
        self.assertEqual(pieces(state, SOURCE), Counter({REGULAR: 1, ELITE: 1}))
        self.assertEqual(pieces(state, TARGET), Counter({REGULAR: 2, ELITE: 1}))
        state = play(state, FP, 'casualty_allocate', operation='reduce_elite')
        self.assertEqual(pieces(state, SOURCE), Counter({REGULAR: 1, ELITE: 1}))
        self.assertEqual(pieces(state, TARGET), Counter({REGULAR: 3}))
        self.assertEqual(state.pending_combat['phase'], 'attacker_continue')

    def test_forced_attacker_loss_applied_before_defender_menu(self):
        state = rolled(position(attack_units=(3, 0)))
        self.assert_owner(state, FP, 'defender')
        self.assertEqual(pieces(state, SOURCE), Counter({REGULAR: 2}))
        self.assertEqual(pieces(state, TARGET), Counter({REGULAR: 2, ELITE: 1}))

    def test_forced_defender_loss_waits_for_attacker_choice(self):
        state = rolled(position(defend_units=(3, 0)))
        self.assert_owner(state, SH, 'attacker')
        self.assertEqual(pieces(state, TARGET), Counter({REGULAR: 3}))
        state = play(state, SH, 'casualty_allocate', operation='reduce_elite')
        self.assertEqual(pieces(state, SOURCE), Counter({REGULAR: 3}))
        self.assertEqual(pieces(state, TARGET), Counter({REGULAR: 2}))
        self.assertEqual(state.pending_combat['phase'], 'attacker_continue')

    def test_both_forced_allocations_finish_without_a_choice(self):
        state = rolled(position(attack_units=(3, 0), defend_units=(3, 0)))
        self.assertEqual(pieces(state, SOURCE), Counter({REGULAR: 2}))
        self.assertEqual(pieces(state, TARGET), Counter({REGULAR: 2}))
        self.assertEqual(state.pending_combat['phase'], 'attacker_continue')

    def test_no_attacker_damage_opens_only_defender_choice(self):
        state = rolled(position(), defender_hits=0)
        self.assert_owner(state, FP, 'defender')
        state = play(state, FP, 'casualty_allocate', operation='remove_regular')
        self.assertEqual(pieces(state, SOURCE), Counter({REGULAR: 2, ELITE: 1}))
        self.assertEqual(pieces(state, TARGET), Counter({REGULAR: 1, ELITE: 1}))
        self.assertEqual(state.pending_combat['phase'], 'attacker_continue')

    def test_great_host_uses_defender_choice_after_attacker_loss(self):
        # SH starts with six figures, FP three. Each suffers one rolled hit.
        # SH removes a Regular -> five figures. FP can preserve three by
        # downgrading (5 < 2*3: no GH hit), or remove a Regular -> two
        # (5 >= 2*2: GH inflicts one separate hit only on FP).
        for defender_downgrades in (0, 1):
            with self.subTest(defender_downgrades=defender_downgrades):
                state = rolled(position(attack_units=(5, 1), great_host=True))
                self.assert_owner(state, SH, 'attacker')
                state = play(state, SH, 'casualty_allocate', operation='remove_regular')
                self.assert_owner(state, FP, 'defender')
                self.assertEqual(pieces(state, SOURCE), Counter({REGULAR: 4, ELITE: 1}))
                self.assertEqual(pieces(state, TARGET), Counter({REGULAR: 2, ELITE: 1}))
                self.assertEqual(state.pending_combat['def_hits'], 1)
                state = play(state, FP, 'casualty_allocate', operation=('reduce_elite' if defender_downgrades else 'remove_regular'))
                if defender_downgrades:
                    self.assertEqual(state.pending_combat['phase'], 'attacker_continue')
                    self.assertEqual(pieces(state, TARGET), Counter({REGULAR: 3}))
                else:
                    self.assert_owner(state, FP, 'defender')
                    self.assertEqual(pieces(state, TARGET), Counter({REGULAR: 1, ELITE: 1}))
                    self.assertEqual(state.pending_combat['def_hits'], 1)
                    state = play(state, FP, 'casualty_allocate', operation='reduce_elite')
                    self.assertEqual(pieces(state, TARGET), Counter({REGULAR: 2}))
                    self.assertEqual(state.pending_combat['phase'], 'attacker_continue')
                self.assertEqual(pieces(state, SOURCE), Counter({REGULAR: 4, ELITE: 1}))

    def test_defending_great_host_extra_hit_returns_to_attacker_only(self):
        state = rolled(position(attacker=FP, defend_units=(5, 1), great_host=True))
        self.assert_owner(state, FP, 'attacker')
        state = play(state, FP, 'casualty_allocate', operation='remove_regular')
        self.assert_owner(state, SH, 'defender')
        state = play(state, SH, 'casualty_allocate', operation='remove_regular')
        self.assert_owner(state, FP, 'attacker')
        self.assertEqual(pieces(state, SOURCE), Counter({REGULAR: 1, ELITE: 1}))
        self.assertEqual(pieces(state, TARGET), Counter({REGULAR: 4, ELITE: 1}))
        self.assertEqual(state.pending_combat['atk_hits'], 1)
        state = play(state, FP, 'casualty_allocate', operation='reduce_elite')
        self.assertEqual(pieces(state, SOURCE), Counter({REGULAR: 2}))
        self.assertEqual(pieces(state, TARGET), Counter({REGULAR: 4, ELITE: 1}))
        self.assertEqual(state.pending_combat['phase'], 'attacker_continue')


if __name__ == '__main__':
    unittest.main(verbosity=2)
