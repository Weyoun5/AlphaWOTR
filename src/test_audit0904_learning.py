"""Independent accounting oracles for real game-state reward transitions.

These tests use hand-calculated VP/corruption deltas, not the trainer as
its own oracle.  They do not claim to establish playing strength.
"""
import os
import contextlib
import io
import unittest
from unittest.mock import patch

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

import ai
import board
import game_state as gs
import run_td_v33loop as loop


def row(state, side=None, action=None):
    return {'state': state, 'side': side, 'action_type': action,
            'kind': 'action' if action else 'initial',
            'params': {}, 'die_idx': -1}


class RewardAccountingTests(unittest.TestCase):
    def setUp(self):
        self.profiles = patch.dict(ai.PLAYER_PROFILES,
                                   {'fp': 'ring', 'sh': 'corruption'})
        self.profiles.start()
        self.addCleanup(self.profiles.stop)
        self.state = gs.setup_initial_game(board.load_board())

    def replay(self, *rows):
        return loop._replay_reward_stream({'steps': list(rows)},
                                         dict(ai.PLAYER_PROFILES))

    def test_final_opponent_capture_reaches_defender_without_another_decision(self):
        final = ai.clone_state(self.state)
        final.enemy_controlled[gs.Side.SHADOW].add('minas tirith')
        per = self.replay(row(self.state),
                          row(self.state, 'FP', 'combat_card_none'),
                          row(final, 'SH', 'victor_advance'))
        # Minas Tirith is 2 VP. FP defensive potential drops by (2/10)^2.
        self.assertAlmostEqual(sum(per['fp'][1]), -0.04)

    def test_ninth_to_tenth_vp_tail_is_not_lost(self):
        self.state.enemy_controlled[gs.Side.SHADOW].update(
            ['minas tirith', 'lorien', 'woodland realm', 'helm s deep',
             'edoras'])
        final = ai.clone_state(self.state)
        final.enemy_controlled[gs.Side.SHADOW].add('pelargir')
        self.assertEqual(ai.game_logic.calculate_victory_points(self.state)['shadow'], 9)
        self.assertEqual(ai.game_logic.calculate_victory_points(final)['shadow'], 10)
        per = self.replay(row(self.state),
                          row(self.state, 'FP', 'combat_card_none'),
                          row(final, 'SH', 'victor_advance'))
        self.assertAlmostEqual(sum(per['fp'][1]), -0.19)

    def test_final_corruption_reaches_shadow_without_extra_decision_cost(self):
        self.state.fellowship.corruption = 11
        final = ai.clone_state(self.state)
        final.fellowship.corruption = 12
        per = self.replay(row(self.state),
                          row(self.state, 'SH', 'eye_allocate'),
                          row(final, 'FP', 'hunt_take_corruption'))
        # One SH decision costs .005; final hunt is not another SH decision.
        self.assertAlmostEqual(sum(per['sh'][1]), 23 / 144 - 0.005)

    def test_final_annotation_state_also_settles_unobserved_delta(self):
        self.state.fellowship.corruption = 11
        final = ai.clone_state(self.state)
        final.fellowship.corruption = 12
        per = self.replay(row(self.state),
                          row(self.state, 'SH', 'eye_allocate'), row(final))
        self.assertAlmostEqual(sum(per['sh'][1]), 23 / 144 - 0.005)

    def test_unchanged_tail_does_not_charge_an_extra_shadow_decision(self):
        per = self.replay(row(self.state),
                          row(self.state, 'SH', 'eye_allocate'), row(self.state))
        self.assertAlmostEqual(sum(per['sh'][1]), -0.005)

    def test_already_banked_capture_is_paid_once_at_stream_end(self):
        final = ai.clone_state(self.state)
        final.enemy_controlled[gs.Side.SHADOW].add('minas tirith')
        per = self.replay(row(self.state),
                          row(self.state, 'FP', 'combat_card_none'),
                          row(final, 'SH', 'victor_advance'),
                          row(final, 'FP', 'fellowship_hide'))
        self.assertAlmostEqual(sum(per['fp'][1]), -0.04)

    def test_no_fabricated_training_sample_for_side_that_never_acted(self):
        final = ai.clone_state(self.state)
        final.enemy_controlled[gs.Side.SHADOW].add('minas tirith')
        per = self.replay(row(self.state), row(final, 'SH', 'victor_advance'))
        self.assertEqual(per['fp'], ([], [], []))


class LambdaTargetTests(unittest.TestCase):
    def test_two_step_return_and_residual_have_hand_calculated_targets(self):
        import numpy as np
        import torch
        import value_trainer_v2

        # The artificial net is an identity, independent of WOTR encodings.
        net = torch.nn.Identity().eval()
        enc = np.array([[0.2], [0.4]], dtype=np.float32)
        seq = (enc, np.array([0.1, -0.2], dtype=np.float32),
               np.array([0.05, 0.10], dtype=np.float32), 1.0)
        targets = value_trainer_v2.build_targets(net, [seq], lam=0.8, clip=2.0)
        # G1 = -.2 + 1 = .8; G0 = .1 + .2*(.4+.1) + .8*.8 = .84.
        np.testing.assert_allclose(targets[0], [0.79, 0.70], atol=1e-7)


class SelfPlayCompletionTests(unittest.TestCase):
    def invoke_runner(self, args, result):
        import run_td_v33sp as runner
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(runner.sys, 'argv', ['runner'] + args))
            stack.enter_context(patch.object(runner, 'FP_NET', __file__))
            stack.enter_context(patch.object(runner, 'SH_NET', __file__))
            stack.enter_context(patch.dict(os.environ, {'WOTR_SP_NET_DIR': ''}))
            stack.enter_context(patch.object(runner, '_frozen_env', return_value=None))
            stack.enter_context(patch.object(runner, '_fresh_run_weight_guard'))
            stack.enter_context(patch.object(runner, 'run_phase', return_value=result))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            runner.main()

    def test_one_success_does_not_hide_one_failed_game(self):
        with self.assertRaises(SystemExit) as caught:
            self.invoke_runner(['--games', '2'], ([{'winner': 'fp'}], [('e', 0, 'failed')]))
        self.assertEqual(caught.exception.code, 1)

    def test_dead_worker_missing_result_is_not_complete(self):
        with self.assertRaises(SystemExit) as caught:
            self.invoke_runner(['--games', '2'], ([{'winner': 'fp'}], []))
        self.assertEqual(caught.exception.code, 1)

    def test_failed_smoke_has_nonzero_exit(self):
        with self.assertRaises(SystemExit) as caught:
            self.invoke_runner(['--smoke'], ([], [('e', 0, 'failed')]))
        self.assertEqual(caught.exception.code, 1)

    def test_complete_batch_still_succeeds(self):
        self.invoke_runner(['--games', '2'], ([{'winner': 'fp'}, {'winner': 'sh'}], []))


if __name__ == '__main__':
    unittest.main()
