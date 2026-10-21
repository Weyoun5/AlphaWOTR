"""Literal feature controls and saved expert-state regression controls."""
from pathlib import Path
import gzip
import importlib.util
import json
import os
import pickle
import sys
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REVIEW = ROOT / 'audit/2026-09-04/decision_pass/encoded_menu_review'
RUNTIME = ROOT
os.environ.update(WOTR_ACT_GEN='v45', WOTR_NATIVE='0', CUDA_VISIBLE_DEVICES='-1',
                  KMP_DUPLICATE_LIB_OK='TRUE', WOTR_REWARD_CONTRACT='predecision_profile_v1')
sys.path.insert(0, str(RUNTIME / 'src'))
import ai
import action_choice_tail as tail
import numpy as np
assert Path(ai.__file__).parent == RUNTIME / 'src'
assert Path(tail.__file__).parent == RUNTIME / 'src'
spec = importlib.util.spec_from_file_location('frozen_census', REVIEW / 'corpus10_r1/run_audit.py')
census = importlib.util.module_from_spec(spec)
spec.loader.exec_module(census)


def action(choices):
    return ai.Action(-1, 'card_option', {'choices': choices})


class CompleteTailTests(unittest.TestCase):
    def test_literal_multiregion_and_rage_fields(self):
        index = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
        np.testing.assert_array_equal(tail.encode_tail(action({'regions': ['a', 'a', 'c']}), index),
                                      [.25, .75, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(tail.encode_tail(action({'rage_split': {'n_north': 1,
            'n_south': 3}, '_rage_dest': 'd'}), index),
            [0, 0, .25, .75, 1, 0, 0, 0, 0, 0, 0, 0])

    def test_literal_movement_subset_and_target(self):
        expected = np.array([0, 0, 0, 0, 0, 1, .1, .2, .5, 0, 0, 0], np.float32)
        np.testing.assert_array_equal(tail.encode_tail(action({'n_regular': 1, 'n_elite': 2,
            'target_fp_army': 'a'}), {'a': 0, 'b': 1}), expected)
        full = tail.encode_tail(action({}), {})
        explicit = tail.encode_tail(action({'n_regular': 0, 'n_elite': 0}), {})
        self.assertEqual(full[5], 0)
        self.assertEqual(explicit[5], 1)

    def test_literal_binary_and_deck_choices(self):
        a = tail.encode_tail(action({'use_hunt_reduction': 0, 'palantir_deck': 'character',
            'wizardstaff_prevent': False}), {})
        b = tail.encode_tail(action({'use_hunt_reduction': 1, 'palantir_deck': 'strategy',
            'wizardstaff_prevent': True}), {})
        np.testing.assert_array_equal(a[9:], [-1, -1, -1])
        np.testing.assert_array_equal(b[9:], [1, 1, 1])

    def test_malformed_choices_refuse(self):
        choices = [{'regions': ['a'] * 4}, {'regions': []}, {'target_fp_army': 'unknown'},
                   {'n_regular': -1}, {'n_elite': 11}, {'n_regular': True},
                   {'use_hunt_reduction': 2}, {'wizardstaff_prevent': 'yes'},
                   {'palantir_deck': 'event'}]
        for choice in choices:
            with self.subTest(choice=choice), self.assertRaises(ValueError):
                tail.encode_tail(action(choice), {'a': 0})

    def test_saved_expert_menus_no_longer_hide_distinct_choices(self):
        checked, families = 0, set()
        for path in sorted((REVIEW / 'corpus10_r1/games').glob('*/state_*_public.pickle.gz')):
            bundle = pickle.load(gzip.open(path, 'rb'))
            state, side, menu = bundle['state'], bundle['side'], bundle['menu']
            direct = np.asarray([ai.encode_action(a, state, side) for a in menu])
            cached = np.asarray([ai._encode_action_cached(a, state, side, {}) for a in menu])
            np.testing.assert_array_equal(direct, cached)
            self.assertEqual(direct.shape[1], 105)
            for dtype in (np.float32, np.float16):
                grouped = {}
                for a, vector in zip(menu, direct.astype(dtype)):
                    key = vector.tobytes(), ai._get_action_region_index(a)
                    if a.action_type == 'dwarven_ring':
                        # This menu also contains ordinary paid actions, so
                        # normalize only the proven physical reroll aliases.
                        die = state.dice[side][a.params['die_idx']]
                        params = dict(a.params, die_idx=census.signature({
                            'faces': census.norm(die.faces), 'current': census.norm(die.current),
                            'side': census.norm(die.side), 'used': bool(die.used)}))
                        a = ai.Action(a.die_idx, a.action_type, params)
                    grouped.setdefault(key, set()).add(census.semantic_action(a, state, side))
                with self.subTest(path=str(path), dtype=str(dtype)):
                    self.assertTrue(all(len(values) == 1 for values in grouped.values()))
            checked += 1
            families.add((getattr(state, 'pending_card_option', None) or {}).get('card_id'))
        self.assertGreaterEqual(checked, 12)
        self.assertGreaterEqual(len(families), 12)


if __name__ == '__main__':
    unittest.main(verbosity=2)
