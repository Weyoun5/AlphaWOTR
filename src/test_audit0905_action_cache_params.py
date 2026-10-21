"""Literal action-cache distinctions, independent of legal-menu generation.

These test encoding semantics; the original complete-menu Saruman witness is
separately recorded under performance_pass/lookup_index_menus_r3_20260905.
"""
import os
import unittest
import numpy as np

os.environ.setdefault('WOTR_ACT_GEN', 'v41')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
import ai
import board
import game_state as gs

FP, SH = gs.Side.FREE_PEOPLES, gs.Side.SHADOW


class ActionCacheParameters(unittest.TestCase):
    def setUp(self):
        self.state = gs.setup_initial_game(board.load_board())
        die = self.state.dice[SH][0]
        die.current, die.used = gs.ActionDieFace.MUSTER, False
        self.state.dice[SH] = [die]
        self.state.armies['orthanc'] = gs.Army(region=ai._armies_lookup(self.state, 'orthanc').region, units=[
            gs.Unit(gs.Nation.ISENGARD, gs.UnitKind.REGULAR),
            gs.Unit(gs.Nation.ISENGARD, gs.UnitKind.REGULAR)])
        self.old_cache = ai._ENC_CACHE
        ai._ENC_CACHE = {}
        self.addCleanup(setattr, ai, '_ENC_CACHE', self.old_cache)

    def pair(self, action_type, saruman):
        params = {'nation': gs.Nation.ISENGARD, 'region': 'orthanc'}
        if action_type == 'muster_recruit':
            params['type'] = 'regular'
        special = {**params, 'saruman': saruman}
        if saruman == 'settlements':
            special['placements'] = ['orthanc', 'north dunland', 'south dunland']
        return ai.Action(0, action_type, params), ai.Action(0, action_type, special)

    def assert_pair(self, plain, special, column, expected):
        for order in ((plain, special), (special, plain)):
            ai._ENC_CACHE.clear()
            memo = {}
            self.assertNotEqual(ai._action_cache_key(plain, self.state, SH, memo),
                                ai._action_cache_key(special, self.state, SH, memo))
            # Literal composition counts, including the encoder's /10 scale.
            self.assertAlmostEqual(float(ai.encode_action(plain, self.state, SH)[column]), expected[0])
            self.assertAlmostEqual(float(ai.encode_action(special, self.state, SH)[column]), expected[1])
            for _ in range(2):  # cold first pass, then hot entries, both orders
                for action in order:
                    np.testing.assert_array_equal(ai.encode_action(action, self.state, SH),
                        ai._encode_action_cached(action, self.state, SH, memo))

    def test_saruman_two_elites_differ_from_one_elite_in_both_cache_orders(self):
        self.assert_pair(*self.pair('muster_elite', 'swap2'), ai._ACT_COMP_OFF + 1, (.1, .2))

    def test_saruman_three_placements_differ_from_one_regular_in_both_orders(self):
        self.assert_pair(*self.pair('muster_recruit', 'settlements'), ai._ACT_COMP_OFF, (.1, .3))

    def test_changed_enemy_army_invalidates_hot_muster_cache(self):
        plain, special = self.pair('muster_elite', 'swap2')
        for action in (plain, special):
            ai._encode_action_cached(action, self.state, SH, {})
        before_key = ai._action_cache_key(special, self.state, SH, {})
        self.state.armies['orthanc'].units.append(gs.Unit(gs.Nation.GONDOR, gs.UnitKind.REGULAR))
        self.assertNotEqual(before_key, ai._action_cache_key(special, self.state, SH, {}))
        for action, count in ((plain, .1), (special, .2)):
            value = ai._encode_action_cached(action, self.state, SH, {})
            self.assertAlmostEqual(float(value[ai._ACT_COMP_OFF + 1]), count)
            self.assertAlmostEqual(float(value[ai._ACT_TGT_OFF]), .1)
            np.testing.assert_array_equal(value, ai.encode_action(action, self.state, SH))

    def test_source_composition_and_paid_die_face_stay_in_cache_key(self):
        action = ai.Action(0, 'army_move', {'from': 'orthanc', 'to': 'westemnet'})
        key = ai._action_cache_key(action, self.state, SH, {})
        ai._encode_action_cached(action, self.state, SH, {})
        self.state.armies['orthanc'].units.append(gs.Unit(gs.Nation.ISENGARD, gs.UnitKind.REGULAR))
        self.assertNotEqual(key, ai._action_cache_key(action, self.state, SH, {}))
        np.testing.assert_array_equal(ai.encode_action(action, self.state, SH),
            ai._encode_action_cached(action, self.state, SH, {}))
        key = ai._action_cache_key(action, self.state, SH, {})
        self.state.dice[SH][0].current = gs.ActionDieFace.CHARACTER
        self.assertNotEqual(key, ai._action_cache_key(action, self.state, SH, {}))
        np.testing.assert_array_equal(ai.encode_action(action, self.state, SH),
            ai._encode_action_cached(action, self.state, SH, {}))

    def test_explicit_mordor_adapter_flag_does_not_alias_plain_declare(self):
        plain = ai.Action(-1, 'fellowship_declare', {'region': 'rivendell'})
        explicit = ai.Action(-1, 'fellowship_declare', {'region': 'rivendell', 'mordor_entry': True})
        # This exercises the encoder's explicit adapter flag, not the legality
        # of entering Mordor from Rivendell. Canonical gateway regions already
        # identify entry independently of the flag.
        for order in ((plain, explicit), (explicit, plain)):
            ai._ENC_CACHE.clear()
            self.assertNotEqual(ai._action_cache_key(plain, self.state, FP),
                                ai._action_cache_key(explicit, self.state, FP))
            for action, bit in ((plain, 0.), (explicit, 1.)):
                self.assertEqual(float(ai.encode_action(action, self.state, FP)[ai._ACT_MORDOR_OFF]), bit)
            for action in order * 2:
                np.testing.assert_array_equal(ai.encode_action(action, self.state, FP),
                    ai._encode_action_cached(action, self.state, FP, {}))

    def test_faramir_choice_unit_and_region_fields_remain_distinct(self):
        actions = [ai.Action(-1, 'card_option', {
            'card_id': "FREE_PEOPLES_Strategy_Faramir's_Rangers",
            'choices': {'unit_type': unit, 'region': region}})
            for unit, region in [('regular', 'minas tirith'), ('elite', 'minas tirith'),
                                 ('elite', 'pelargir')]]
        self.assertEqual(len({ai._action_cache_key(a, self.state, FP) for a in actions}), 3)
        encs = [ai.encode_action(a, self.state, FP) for a in actions]
        self.assertEqual(len({a.tobytes() for a in encs}), 3)
        for action, expected in zip(actions, encs):
            np.testing.assert_array_equal(ai._encode_action_cached(action, self.state, FP, {}), expected)


if __name__ == '__main__':
    unittest.main()
