"""Independent profile-utility proof and root-currency search regressions.

No game rollout or checkpoint is needed: exact terminal utilities and small
trees establish whether the solver is allowed to stop searching alternatives.
"""
import contextlib
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
import numpy as np
import ai
from game_state import Side
from reward_contract import PROFILE, LEGACY, WINLOSS


FP, SH = Side.FREE_PEOPLES, Side.SHADOW


class ProfileSearchProofTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, {'WOTR_REWARD_CONTRACT': PROFILE}))
        self.stack.enter_context(patch.dict(ai.PLAYER_PROFILES, {'fp': 'ring', 'sh': 'military'}))
        self.stack.enter_context(patch.object(ai, '_SIM_OVERLAY', None))
        self.stack.enter_context(patch.object(ai, '_PENDING_EVAL_BOX', [None]))

    def settle(self, victory, route, root_side):
        node = ai.MCTSNode(SimpleNamespace(), root_side)
        result = {'victory': victory, 'victory_type': route}
        with patch.object(ai.game_logic, 'check_all_victory_conditions', return_value=result), \
                patch.object(ai.game_logic, 'should_game_end', return_value=True):
            value = ai._settle_new_node(node, root_side, None, None, root_side)
        self.assertTrue(node.is_terminal)
        self.assertEqual(value, node.terminal_value)
        return node

    def stats(self, *, proof=(1, 0), exact=(0.0, 0.0), q=(0.0, 0.8), n=(10, 90)):
        result = {'p': [0.5, 0.5], 'q': q, 'n': n,
                  'tf': [1.0, 0.0], 'pv': proof}
        if exact is not None:
            result['pvv'] = exact
        return result

    def test_off_target_wins_remain_exact_terminals_without_dominating_proofs(self):
        fp = self.settle('free_peoples', 'military', FP)
        sh = self.settle('shadow', 'ring_corruption', SH)
        self.assertEqual(fp.terminal_value, 0.0)
        self.assertEqual(sh.terminal_value, 0.2)
        self.assertIsNone(fp.proven)
        self.assertIsNone(sh.proven)

    def test_on_target_wins_and_flat_losses_retain_boundary_proofs(self):
        for victory, route, side, mark, value in (
                ('free_peoples', 'ring_destroyed', FP, 'W', 1.0),
                ('shadow', 'military', SH, 'W', 1.0),
                ('shadow', 'ring_corruption', FP, 'L', -1.0),
                ('free_peoples', 'military', SH, 'L', -1.0)):
            with self.subTest(victory=victory, side=side):
                node = self.settle(victory, route, side)
                self.assertEqual(node.proven, mark)
                self.assertEqual(node.proven_value, value)

    def test_root_chooser_keeps_better_profile_line_eligible_than_off_target_win(self):
        for off in (0.0, 0.2):
            with self.subTest(off=off):
                picked, target, info = ai.select_root_action(
                    self.stats(exact=(off, 0.0), q=(off, 0.8)), 0.0)
                self.assertEqual(picked, 1)
                self.assertGreater(target[1], 0.0)
                self.assertTrue(info['elig'][1])

    def test_boundary_win_still_dominates_uncertain_values_and_stale_q(self):
        picked, target, _ = ai.select_root_action(
            self.stats(exact=(1.0, 0.0), q=(-0.2, 0.8)), 0.0)
        self.assertEqual(picked, 0)
        np.testing.assert_array_equal(target, [1.0, 0.0])

    def test_two_winner_marks_do_not_make_off_target_target_mass_equal_to_on_target(self):
        picked, target, _ = ai.select_root_action(
            self.stats(proof=(1, 1), exact=(0.0, 1.0), q=(0.9, 0.1)), 0.0)
        self.assertEqual(picked, 1)
        np.testing.assert_array_equal(target, [0.0, 1.0])

    def test_profile_proof_without_exact_value_is_not_treated_as_full_reward(self):
        picked, target, _ = ai.select_root_action(self.stats(exact=None), 0.0)
        self.assertEqual(picked, 1)
        self.assertGreater(target[1], 0.0)

    def test_off_target_terminal_does_not_solve_parent_and_hide_unexplored_alternative(self):
        root = ai.MCTSNode(SimpleNamespace(), FP)
        off = self.settle('free_peoples', 'military', FP)
        better = ai.MCTSNode(SimpleNamespace(), FP, prior=0.9)
        better.visit_count, better.value_sum = 10, 8.0
        root.children = [off, better]
        root.visit_count = 11
        ai._propagate_proofs([root, off], FP)
        self.assertIsNone(root.proven)
        self.assertIs(ai._select_child(root, 1.0), better)

    def test_propagation_rejects_nonboundary_marks_supplied_by_an_older_tree(self):
        for side in (FP, SH):
            with self.subTest(side=side):
                root = ai.MCTSNode(SimpleNamespace(), side)
                off = ai.MCTSNode(SimpleNamespace(), side)
                off.proven, off.proven_value = 'W', 0.2
                alternative = ai.MCTSNode(SimpleNamespace(), side)
                root.children = [off, alternative]
                ai._propagate_proofs([root], FP)
                self.assertIsNone(root.proven)

    def test_mixed_exact_chance_outcomes_do_not_gain_full_win_proof(self):
        chance = ai.MCTSNode(SimpleNamespace(), FP)
        chance.is_chance, chance.kernel_cover = True, 1.0
        on = self.settle('free_peoples', 'ring_destroyed', FP)
        off = self.settle('free_peoples', 'military', FP)
        on.prior, off.prior = 0.5, 0.5
        chance.outcomes = [on, off]
        self.assertFalse(ai._kernel_chance_proof(chance))
        self.assertIsNone(chance.proven)
        chance.outcomes = [on, self.settle('free_peoples', 'ring_destroyed', FP)]
        chance.outcomes[1].prior = 0.5
        self.assertTrue(ai._kernel_chance_proof(chance))
        self.assertEqual(chance.proven_value, 1.0)

    def test_legacy_and_pure_winloss_keep_existing_winner_dominance(self):
        for contract in (LEGACY, WINLOSS):
            with self.subTest(contract=contract), patch.dict(
                    os.environ, {'WOTR_REWARD_CONTRACT': contract}):
                result = {'victory': 'free_peoples', 'victory_type': 'military'}
                self.assertEqual(ai._proof_mark(result, FP), 'W')
                picked, target, _ = ai.select_root_action(self.stats(), 0.0)
                self.assertEqual(picked, 0)
                np.testing.assert_array_equal(target, [1.0, 0.0])

    def test_opponent_node_uses_root_value_net_instead_of_negating_opponent_objective(self):
        node = ai.MCTSNode(SimpleNamespace(), SH)
        fp_net, sh_net = object(), object()
        action = ai.Action(-1, 'literal_choice')
        with patch.object(ai, 'get_legal_actions', return_value=[action]), \
                patch.object(ai, '_get_priors_and_value', return_value=([1.0], -0.7)) as priors, \
                patch.object(ai, '_mcts_eval_state', return_value=0.25) as own_eval:
            actual = ai._expand_node(node, fp_net, sh_net, FP)
        self.assertEqual(actual, 0.25)  # Opponent's -0.7 must not become +0.7.
        self.assertIs(priors.call_args.args[2], sh_net)
        own_eval.assert_called_once_with(node.state, FP, fp_net, sh_net)

    def test_batched_opponent_node_backs_up_root_currency_with_correct_sign(self):
        root = ai.MCTSNode(SimpleNamespace(), FP)
        node = ai.MCTSNode(SimpleNamespace(), SH, parent=root)
        fp_net, sh_net = object(), object()
        coll = ai._PendingEvals()
        action = ai.Action(-1, 'literal_choice')
        with patch.object(ai, '_PENDING_EVAL_BOX', [coll]), \
                patch.object(ai, 'get_legal_actions', return_value=[action]), \
                patch.object(ai, 'encode_state', return_value=np.array([0.25], dtype=np.float32)), \
                patch.object(ai, '_encode_actions_batch', return_value=(
                    np.zeros((1, ai.ACTION_ENCODING_DIM), dtype=np.float32),
                    np.zeros(1, dtype=np.int64))):
            ai._expand_node(node, fp_net, sh_net, FP)
        coll.recs[0]['paths'] = [[root, node]]
        # Install the virtual loss which the real simulation adds before flush.
        root.visit_count, root.value_sum = 1, -1.0
        node.visit_count, node.value_sum = 1, 1.0
        with patch.object(ai, '_pv_forward_many', return_value=[(np.array([1.0]), -0.7)]), \
                patch.object(ai, '_eval_forward_many', return_value=[0.25]) as own_eval:
            ai._flush_pending(coll, fp_net, sh_net, FP)
        self.assertEqual(root.value_sum, 0.25)
        self.assertEqual(node.value_sum, -0.25)
        self.assertIs(own_eval.call_args.args[0], fp_net)

    def test_exact_proof_utilities_survive_trace_payload_and_keep_selector_target(self):
        from selector_contract import make_evidence
        live = self.stats(exact=(1.0, 0.0), q=(0.1, 0.8))
        info = {'action_encs': np.zeros((2, ai.ACTION_ENCODING_DIM)),
                'region_indices': [0, 0], 'priors_clean': live['p'],
                'q_values': live['q'], 'visit_counts': live['n'],
                'term_frac': live['tf'], 'proven': live['pv'],
                'proven_value': live['pvv'], 'played': 0, 'regret': 0.0}
        info.update(make_evidence(ai.SELECT_CFG, 0., [1., 0.], eligible=[True, False],
                                  reward_contract='predecision_profile_v1'))
        with patch.object(ai, '_STORE_SEARCH_STATS', True):
            payload, _ = ai._search_stats_payload(info)
        self.assertIn('pvv', payload)
        np.testing.assert_array_equal(payload['pvv'], [1.0, 0.0])
        played, expected, _ = ai.select_root_action(live, 0.0)
        replayed, actual, _ = ai.select_root_action(payload, 0.0)
        self.assertEqual(played, replayed)
        np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
