from pathlib import Path
import copy
import gzip
import os
import pickle
import sys
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REVIEW = ROOT / 'audit/2026-09-04/decision_pass/encoded_menu_review'
RUNTIME = ROOT
os.environ.update(WOTR_ACT_GEN='v45', WOTR_NATIVE='0', CUDA_VISIBLE_DEVICES='-1',
                  KMP_DUPLICATE_LIB_OK='TRUE')
sys.path.insert(0, str(RUNTIME / 'src'))
import ai
import numpy as np
from game_state import Side
from nazgul_context import movement_tail
assert Path(ai.__file__).resolve().parent == ROOT / 'src'


def states():
    path = REVIEW / 'deferred_movement_review/naz_context_states.pickle.gz'
    with gzip.open(path, 'rb') as stream:
        return pickle.load(stream)


def encode(state, side):
    return ai.encode_state(state, side)


class NazgulContextTests(unittest.TestCase):
    def test_actual_public_cycle_preserves_board_but_changes_remaining_flights(self):
        first, last = states()
        for side in (Side.SHADOW, Side.FREE_PEOPLES):
            np.testing.assert_array_equal(ai.encode_state(first, side)[:3746], ai.encode_state(last, side)[:3746])
            self.assertEqual(encode(first, side).shape, (4180,))
            np.testing.assert_array_equal(encode(last, side)[4070:], movement_tail(last, ai.REGION_INDEX))
            self.assertFalse(np.array_equal(encode(first, side), encode(last, side)))
        self.assertEqual([len(ai.get_legal_actions(s, Side.SHADOW)) for s in (first, last)], [289, 97])
        before, after = (movement_tail(s, ai.REGION_INDEX) for s in (first, last))
        self.assertEqual(len(after), 110)
        expected = before.copy()
        expected[1 + ai.REGION_INDEX['orthanc']] = 1 / 9.
        expected[1 + ai.REGION_INDEX['fords of isen']] = 1 / 9.
        np.testing.assert_array_equal(after, expected)
        self.assertEqual(np.count_nonzero(after - before), 2)
        self.assertFalse(np.array_equal(before.astype(np.float16), after.astype(np.float16)))

    def test_literal_minion_flags_are_independent(self):
        initial, _ = states()
        changed = copy.deepcopy(initial)
        for option in changed.pending_card_option['options']:
            if '_naz_arr' in option:
                option['_naz_arr'].update(__wk__=1, __mo_ok__=1, __mo__=1)
        np.testing.assert_array_equal(movement_tail(initial, ai.REGION_INDEX)[-4:], [1, 0, 0, 0])
        np.testing.assert_array_equal(movement_tail(changed, ai.REGION_INDEX)[-4:], [1, 1, 1, 1])

    def test_actual_done_clears_context(self):
        _, state = states()
        done = next(a for a in ai.get_legal_actions(state, Side.SHADOW)
                    if a.params.get('choices', {}).get('_naz_done'))
        after = ai.apply_action(state, done)
        np.testing.assert_array_equal(movement_tail(after, ai.REGION_INDEX), np.zeros(110))

    def test_option_order_clone_pickle_and_no_mutation(self):
        _, state = states()
        before = pickle.dumps(state, protocol=5)
        expected = movement_tail(state, ai.REGION_INDEX)
        encode(state, Side.SHADOW)
        self.assertEqual(before, pickle.dumps(state, protocol=5))
        for clone in (ai.clone_state(state), pickle.loads(before)):
            clone.pending_card_option['options'].reverse()
            np.testing.assert_array_equal(movement_tail(clone, ai.REGION_INDEX), expected)

    def test_incomplete_or_conflicting_history_refuses(self):
        _, base = states()
        for mutation in ('missing', 'conflict', 'unknown_region', 'bad_count', 'invalid_flag'):
            state = copy.deepcopy(base)
            flights = [o for o in state.pending_card_option['options'] if '_naz_arr' in o]
            if mutation == 'missing':
                flights[0].pop('_naz_arr')
            elif mutation == 'conflict':
                flights[0]['_naz_arr']['orthanc'] = 2
            else:
                for option in flights:
                    option['_naz_arr'][{'unknown_region': 'unknown', 'bad_count': 'orthanc',
                                         'invalid_flag': '__wk__'}[mutation]] = 12
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                movement_tail(state, ai.REGION_INDEX)

    def test_unrelated_pending_card_has_zero_context(self):
        _, state = states()
        state.pending_card_option = {'card_id': 'SHADOW_Strategy_Pits_of_Mordor', 'options': []}
        np.testing.assert_array_equal(movement_tail(state, ai.REGION_INDEX), np.zeros(110))


if __name__ == '__main__':
    unittest.main(verbosity=2)
