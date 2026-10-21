"""Public-state oracles for operation encoding and search identity."""
import unittest
from unittest.mock import patch
import numpy as np
import ai
import game_state as gs
from test_audit0905_casualty_order import position, rolled, play, SH, FP, SOURCE


class CasualtyOperationIdentity(unittest.TestCase):
    def state(self):
        return rolled(position(), attacker_hits=1, defender_hits=2)

    def test_remaining_budget_changes_observation_and_search_identity(self):
        state = self.state()
        other = ai.clone_state(state)
        other.pending_combat['atk_hits'] = 1
        self.assertFalse(np.array_equal(ai.encode_state(state, SH), ai.encode_state(other, SH)))
        self.assertNotEqual(ai._state_sig(state), ai._state_sig(other))
        # Literal v44 tail: the three casualty lanes retain their v43 positions;
        # the appended Fellowship continuation lane is inactive during battle.
        np.testing.assert_allclose(ai.encode_state(state, SH)[3742:], [.2, .1, 1., 0.])

    def test_owner_stage_changes_observation_and_search_identity(self):
        state = self.state()
        other = ai.clone_state(state)
        other.pending_combat.update(alloc_stage='defender', deciding_side=FP)
        self.assertFalse(np.array_equal(ai.encode_state(state, SH), ai.encode_state(other, SH)))
        self.assertNotEqual(ai._state_sig(state), ai._state_sig(other))

    def test_permanent_regular_box_changes_search_identity(self):
        state = self.state()
        other = ai.clone_state(state)
        other.fp_casualties.setdefault(gs.Nation.GONDOR, {})['regular'] = 1
        self.assertNotEqual(ai._state_sig(state), ai._state_sig(other))

    def test_all_three_operations_encode_differently_and_cache_agrees(self):
        state = self.state()
        actions = ai.get_legal_actions(state, SH)
        self.assertEqual({a.params['operation'] for a in actions},
                         {'remove_regular', 'reduce_elite', 'remove_elite'})
        encodings = [ai.encode_action(a, state, SH) for a in actions]
        self.assertEqual(len({a.tobytes() for a in encodings}), 3)
        keys = [ai._action_cache_key(a, state, SH) for a in actions]
        self.assertEqual(len(set(keys)), 3)
        for action, expected in zip(actions, encodings):
            np.testing.assert_array_equal(ai._encode_action_cached(action, state, SH), expected)

    def test_nationality_changes_encoding(self):
        state = self.state()
        state.armies[SOURCE].units[0].nation = gs.Nation.ISENGARD
        actions = [a for a in ai.get_legal_actions(state, SH)
                   if a.params['operation'] == 'remove_regular']
        self.assertEqual(len(actions), 2)
        self.assertFalse(np.array_equal(*(ai.encode_action(a, state, SH) for a in actions)))

    def test_same_owner_continues_and_witness_reports_exact_loss(self):
        state = self.state()
        following = play(state, SH, 'casualty_allocate', operation='remove_regular', nation=gs.Nation.SAURON)
        self.assertEqual(ai._pending_decision_side(following), SH)
        self.assertEqual(following.pending_combat['atk_hits'], 1)
        self.assertEqual(len(following._casualty_operations), 1)
        loss = following._casualty_operations[0]
        self.assertEqual((loss['operation'], loss['nation'], loss['cost'], loss['remaining_before']),
                         ('remove_regular', gs.Nation.SAURON, 1, 2))
        self.assertEqual(state.pending_combat['atk_hits'], 2)

    def test_whole_elite_advances_owner_without_replacement_stamp(self):
        state = self.state()
        following = play(state, SH, 'casualty_allocate', operation='remove_elite', nation=gs.Nation.SAURON)
        self.assertEqual(ai._pending_decision_side(following), FP)
        self.assertEqual(following.pending_combat['atk_hits'], 0)
        self.assertEqual([x['operation'] for x in following._casualty_operations], ['remove_elite'])
        self.assertEqual(getattr(following, '_dg_src_log', []), getattr(state, '_dg_src_log', []))

    def test_legacy_action_encoding_is_rejected(self):
        state = self.state()
        with self.assertRaisesRegex(ValueError, 'casualty'):
            ai.encode_action(ai.Action(-1, 'casualty_allocate', {'count': 1}), state, SH)


if __name__ == '__main__':
    unittest.main(verbosity=2)
