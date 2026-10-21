"""Encoder optimization parity, plus independent checks on mutable armies.

The frozen function predates the optimization; it establishes compatibility
with trained weights, not correctness of all game rules or all features.
"""
import os
from pathlib import Path
import random
import unittest

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('WOTR_ACT_GEN', 'v41')
ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import ai
import board
import game_logic
import game_state as gs
import game_trace

REFERENCE = ROOT / 'audit/2026-09-04/encoding_baseline/encode_action.py'
TRACE = ROOT / 'data/v42s_selfplay_500/traces/w1_g0_sh_military.trace.gz'


def reference_encoder():
    # Keep the actual runtime's constants/types. Only the function body is
    # frozen, so this test checks old/new code under the same configuration.
    scope = dict(vars(ai))
    exec(compile(REFERENCE.read_text(encoding='utf-8'), str(REFERENCE), 'exec'),
         scope)
    return scope['encode_action']


def initial_menus():
    random.seed(904)
    state = gs.setup_initial_game(board.load_board())
    game_logic.initialize_decks(state)
    faces = (gs.ActionDieFace.CHARACTER, gs.ActionDieFace.ARMY,
             gs.ActionDieFace.MUSTER, gs.ActionDieFace.EVENT,
             gs.ActionDieFace.MUSTER_ARMY, gs.ActionDieFace.WILL_OF_WEST)
    for side in (gs.Side.FREE_PEOPLES, gs.Side.SHADOW):
        for i, die in enumerate(state.dice[side]):
            die.current = faces[i % len(faces)]
    return [(state, side, ai.get_legal_actions(state, side))
            for side in (gs.Side.FREE_PEOPLES, gs.Side.SHADOW)]


def trace_menus():
    trace = game_trace.load_trace(str(TRACE))
    steps = trace['steps']
    # Across the game, plus each recorded action family and pending phase.
    # Use recorded states as inputs; never manufacture expected encodings
    # by round-tripping the optimized function.
    indices = set(np.linspace(0, len(steps) - 1, 24, dtype=int))
    kinds = set()
    for i, row in enumerate(steps):
        state = row.get('state')
        if state is None:
            continue
        pending = tuple((name, str(getattr(state, name, {}).get('phase', '')))
                        for name in ('pending_combat', 'pending_hunt',
                                     'pending_card_option')
                        if isinstance(getattr(state, name, None), dict))
        key = row.get('action_type'), pending
        if key not in kinds:
            kinds.add(key)
            indices.add(max(0, i - 1))
            indices.add(i)
    for i in sorted(indices):
        state = steps[i].get('state')
        if state is None:
            continue
        owner = ai._pending_decision_side(state)
        sides = (owner,) if owner is not None else (gs.Side.FREE_PEOPLES,
                                                    gs.Side.SHADOW)
        legacy = tuple(side for side, attr in (
            (gs.Side.FREE_PEOPLES, 'fp_ring_waiver'),
            (gs.Side.SHADOW, 'sh_ring_waiver')) if getattr(state, attr, False))
        for side in sides:
            if legacy:
                # The old trace lost the physical target. Its recorded action
                # remains an encoder input, but cannot certify a current menu.
                row = steps[i]
                menu = ([ai.Action(row['die_idx'], row['action_type'], row['params'])]
                        if row.get('action_type') else [])
            else:
                menu = ai.get_legal_actions(state, side)
            yield i, state, side, menu, legacy


class ActionEncodingReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_encode = staticmethod(reference_encoder())

    def assert_parity(self, action, state, side):
        before = self.old_encode(action, state, side)
        after = ai.encode_action(action, state, side)
        self.assertEqual(before.dtype, after.dtype)
        self.assertEqual(before.shape, after.shape)
        self.assertEqual(before.tobytes(), after.tobytes(),
                         '%s %r %s' % (action.action_type, action.params, side))
        return after

    def test_initial_legal_menus_byte_identical_both_sides(self):
        count = 0
        for state, side, menu in initial_menus():
            self.assertTrue(menu)
            for action in menu:
                self.assert_parity(action, state, side)
                count += 1
        self.assertGreater(count, 100)

    def test_recorded_actions_parity_and_legacy_snapshot_refusal(self):
        count = 0
        legacy_count = 0
        revised_casualties = 0
        kinds = set()
        for i, state, side, menu, legacy in trace_menus():
            with self.subTest(step=i, side=side):
                for legacy_side in legacy:
                    with self.assertRaisesRegex(RuntimeError, 'no reliable physical target'):
                        game_logic.ring_waiver_active(state, legacy_side)
                legacy_count += bool(legacy)
                for action in menu:
                    choice = action.params.get('choices') or {}
                    if action.action_type == 'card_option' and 'cas_downgrades' in choice:
                        with self.assertRaisesRegex(ValueError, 'Legacy card casualty'):
                            ai.encode_action(action, state, side)
                        kinds.add(action.action_type)
                        continue
                    if action.action_type == 'card_option' and 'casualty_operation' in choice:
                        expected = {'remove_regular': (0., 0.),
                                    'reduce_elite': (1., 0.),
                                    'remove_elite': (0., 1.)}
                        np.testing.assert_array_equal(ai.encode_action(action, state, side)[23:25],
                                                      expected[choice['casualty_operation']])
                        kinds.add(action.action_type)
                        continue
                    if action.action_type == 'casualty_allocate':
                        # Current menus reconstructed from old boards use a
                        # deliberately new schema. The frozen count encoder
                        # cannot certify nation-specific casualty operations.
                        if 'operation' not in action.params:
                            with self.assertRaisesRegex(ValueError, 'casualty'):
                                ai.encode_action(action, state, side)
                        else:
                            expected = {'remove_regular': (0., 0.),
                                        'reduce_elite': (1., 0.),
                                        'remove_elite': (0., 1.)}
                            np.testing.assert_array_equal(
                                ai.encode_action(action, state, side)[23:25],
                                expected[action.params['operation']])
                            revised_casualties += 1
                        kinds.add(action.action_type)
                        continue
                    self.assert_parity(action, state, side)
                    count += 1
                    kinds.add(action.action_type)
        self.assertGreater(count, 1000)
        self.assertGreater(legacy_count, 0)
        self.assertGreater(revised_casualties, 0)
        self.assertTrue({'army_move', 'army_attack', 'card_option',
                         'casualty_allocate', 'nazgul_move'} <= kinds)
        print('encoding parity: %d saved-state actions, %d action types; '
              '%d legacy observations checked for refusal'
              % (count, len(kinds), legacy_count))

    def test_same_action_observes_unit_leader_and_army_replacement(self):
        state = gs.setup_initial_game(board.load_board())
        side = gs.Side.SHADOW
        action = ai.Action(0, 'army_move',
                           {'from': 'Morannon', 'to': 'Dagorlad',
                            'n_regular': 1, 'n_elite': 0,
                            'take_leaders': True})
        # Hand-counted mixed-stack test: enemy figures must not enter
        # Shadow source counts, even when all occupy one Army object.
        army = state.armies['Morannon']
        army.units[:] = [gs.Unit(gs.Nation.SAURON, gs.UnitKind.REGULAR),
                         gs.Unit(gs.Nation.GONDOR, gs.UnitKind.ELITE)]
        army.leaders[:] = []
        first = self.assert_parity(action, state, side)
        self.assertEqual(first[16], np.float32(.1))
        self.assertEqual(first[ai._ACT_SRC_OFF], np.float32(.1))
        army.units.append(gs.Unit(gs.Nation.SAURON, gs.UnitKind.ELITE))
        army.leaders.append(gs.Nazgul(gs.Nation.SAURON))
        second = self.assert_parity(action, state, side)
        self.assertEqual(second[16], np.float32(.2))
        self.assertEqual(second[ai._ACT_SRC_OFF + 1], np.float32(.1))
        self.assertEqual(second[ai._ACT_SRC_OFF + 2], np.float32(1 / 8))
        self.assertEqual(second[ai._ACT_COMP_OFF + 3], np.float32(1 / 8))
        state.armies['Morannon'] = gs.Army(army.region)
        third = self.assert_parity(action, state, side)
        self.assertEqual(third[16], 0)
        self.assertEqual(third[ai._ACT_COMP_OFF + 3], 0)
        # Returned arrays also remain independent of subsequent calls.
        self.assertEqual(first[16], np.float32(.1))

    def test_source_case_variants_and_exact_key_precedence(self):
        state = gs.setup_initial_game(board.load_board())
        # Compare lookup behavior where the old whole-army and current
        # compulsory-participant encodings legitimately agree.
        state.political.boxes[gs.Nation.SAURON] = 3
        for key in ('Morannon', 'morannon', 'MORANNON'):
            for at in ('army_move', 'army_attack', 'combat_retreat',
                       'nazgul_move', 'companion_move', 'victor_advance'):
                action = ai.Action(0, at, {'from': key, 'to': key,
                                           'n_regular': 1, 'n_elite': 0})
                for side in gs.Side:
                    self.assert_parity(action, state, side)
        # Case variants can coexist in legacy snapshots; the exact key
        # must win over a normalized match earlier in dictionary order.
        state.armies['morannon'] = gs.Army(state.armies['Morannon'].region)
        state.armies['morannon'].units.append(
            gs.Unit(gs.Nation.SAURON, gs.UnitKind.ELITE))
        action = ai.Action(0, 'army_attack', {'from': 'morannon',
                                              'to': 'Morannon'})
        enc = self.assert_parity(action, state, gs.Side.SHADOW)
        self.assertEqual(enc[16], np.float32(.1))
        self.assertEqual(enc[ai._ACT_SRC_OFF], 0)

    def test_legacy_empty_missing_and_non_string_source_semantics(self):
        state = gs.setup_initial_game(board.load_board())
        template = state.armies['Morannon']
        for key in ('', 'None', 7, '7'):
            state.armies[key] = gs.Army(template.region)
        state.armies[7].units.append(gs.Unit(gs.Nation.SAURON,
                                           gs.UnitKind.REGULAR))
        state.armies['7'].units.extend([
            gs.Unit(gs.Nation.SAURON, gs.UnitKind.ELITE) for _ in range(2)])
        for source_params in ({}, {'from': None}, {'from': ''}, {'from': 7}):
            for at in ('army_move', 'nazgul_move', 'companion_move'):
                action = ai.Action(0, at, dict(source_params, to='Morannon',
                                                n_regular=1, n_elite=0))
                self.assert_parity(action, state, gs.Side.SHADOW)

    def test_missing_attack_source_never_borrows_the_target_army(self):
        # The pre-army reference borrowed 'to' when 'from' was absent. The
        # current participant contract deliberately has no such attackers.
        state = gs.setup_initial_game(board.load_board())
        self.assertTrue(state.armies['Morannon'].units)
        for source_params in ({}, {'from': None}, {'from': ''}):
            with self.subTest(params=source_params):
                action = ai.Action(0, 'army_attack', dict(source_params, to='Morannon'))
                enc = ai.encode_action(action, state, gs.Side.SHADOW)
                np.testing.assert_array_equal(enc[ai._ACT_COMP_OFF:ai._ACT_COMP_OFF + 4],
                                              np.zeros(4, dtype=np.float32))
                self.assertEqual(enc[16], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
