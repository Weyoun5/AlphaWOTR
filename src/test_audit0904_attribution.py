"""Literal-value oracles for trace attribution, without loading game models.

The expected TD edges and GAE sums below are hand calculated. The production
reward-stream reader is exercised for win/loss traces; game states only need
the event fields used by this view. No checkpoint, corpus, or GUI is written.
"""
import contextlib
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

import numpy as np
import torch
import ai
import game_logic
import game_trace
import run_td_v33loop as trainer
import td_attribution as attribution
from reward_contract import LEGACY, WINLOSS


MISSING = object()


def state(corruption=0, progress=0, fp_vp=0, sh_vp=0):
    return SimpleNamespace(
        fellowship=SimpleNamespace(corruption=corruption, progress=progress),
        vp={'free_peoples': fp_vp, 'shadow': sh_vp})


def decision(side, value=MISSING, *, pre=None, post=None):
    context = {'contract': WINLOSS, 'encoding': np.array([0.25, -0.5]),
               'potential': 0.0, 'state': pre if pre is not None else state()}
    if value is not MISSING:
        context['value'] = value
    return {'kind': 'action', 'side': side, 'action_type': 'literal_choice',
            'params': {}, 'turn': 1, 'state': post if post is not None else state(),
            'decision_context': context}


class ConstantNet(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, inputs):
        return torch.full((len(inputs), 1), self.value, device=inputs.device)


class AttributionProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(trainer, '_apply_loop_config'))
        self.stack.enter_context(patch.object(trainer, 'LAM', 0.8))
        self.stack.enter_context(patch.object(game_logic, 'calculate_victory_points',
                                              side_effect=lambda snapshot: snapshot.vp))
        self.stack.enter_context(patch.dict(ai.PLAYER_PROFILES,
                                            {'fp': 'original-fp', 'sh': 'original-sh'}))
        self.stack.enter_context(patch.dict(os.environ,
                                            {'WOTR_REWARD_CONTRACT': LEGACY}))
        self.newest = self.stack.enter_context(patch.object(
            attribution, '_load_net', side_effect=AssertionError('newest model must not load')))
        self.loader = self.stack.enter_context(patch.object(
            ai, 'load_value_net', side_effect=AssertionError('no implicit checkpoint')))

    def compute(self, rows, *, winner='fp', contract=WINLOSS, **kwargs):
        trace = {'meta': {'reward_contract': contract, 'winner': winner,
                          'victory_type': 'ring'}, 'steps': rows}
        with patch.object(game_trace, 'load_trace', return_value=trace):
            return attribution.compute_attribution('unwritten.trace.gz', **kwargs)

    def test_own_side_next_decisions_give_positive_and_negative_literal_credit(self):
        result = self.compute([decision('FP', 0.2), decision('SH', -0.3),
                               decision('FP', 0.8), decision('SH', -0.7)])
        rows = result['steps']
        # Own-side edges: .8-.2=.6; -.7-(-.3)=-.4; results 1-.8=.2,
        # -1-(-.7)=-.3. Lambda .8 gives .6+.8*.2=.76 and -.4+.8*(-.3)=-.64.
        for row, delta, advantage in zip(rows, [0.6, -0.4, 0.2, -0.3],
                                         [0.76, -0.64, 0.2, -0.3]):
            self.assertAlmostEqual(row['delta'], delta)
            self.assertAlmostEqual(row['adv'], advantage)
            self.assertEqual(row['value_source'], 'recorded_behavior')
            self.assertEqual(row['reward'], 0.0)
        self.assertEqual(result['outcomes'], {'fp': 1.0, 'sh': -1.0})
        self.assertEqual(result['value_mode'], 'recorded_behavior')
        self.assertEqual(result['lambda_source'], 'current_loop_configuration')
        self.assertEqual(result['top_credit'][0]['i'], 0)
        self.assertEqual(result['top_blame'][0]['i'], 1)
        self.newest.assert_not_called()
        self.loader.assert_not_called()

    def test_all_missing_historical_values_stay_unavailable(self):
        result = self.compute([decision('FP'), decision('SH')])
        self.assertEqual(result['unavailable_values'], 2)
        self.assertEqual(result['value_mode'], 'unavailable')
        self.assertEqual(result['top_credit'], [])
        self.assertEqual(result['top_blame'], [])
        self.assertTrue(all(row['v'] is None and row['adv'] is None
                            for row in result['steps']))
        self.newest.assert_not_called()
        self.loader.assert_not_called()

    def test_missing_values_do_not_jump_to_later_own_side_decisions(self):
        result = self.compute([decision('FP', 0.2), decision('FP', 0.4),
                               decision('FP'), decision('FP', 0.7)])
        rows = result['steps']
        self.assertAlmostEqual(rows[0]['delta'], 0.2)
        self.assertIsNone(rows[0]['adv'])  # Complete GAE suffix is unknown.
        self.assertIsNone(rows[1]['delta'])  # Cannot jump .4 -> .7.
        self.assertIsNone(rows[2]['delta'])
        self.assertAlmostEqual(rows[3]['delta'], 0.3)
        self.assertEqual([row['i'] for row in result['top_credit']], [3])

    def test_invalid_recorded_values_are_unavailable(self):
        for value in (float('nan'), float('inf'), True, '0.4'):
            with self.subTest(value=value):
                result = self.compute([decision('FP', value)])
                self.assertIsNone(result['steps'][0]['v'])
                self.assertIsNone(result['steps'][0]['delta'])
        self.newest.assert_not_called()
        self.loader.assert_not_called()

    def test_zero_lambda_keeps_known_one_step_advantage_before_a_later_gap(self):
        with patch.object(trainer, 'LAM', 0.0):
            result = self.compute([decision('FP', 0.2), decision('FP', 0.4),
                                   decision('FP'), decision('FP', 1.0)])
        self.assertAlmostEqual(result['steps'][0]['adv'], 0.2)
        self.assertIsNone(result['steps'][1]['adv'])
        self.assertEqual(result['steps'][3]['adv'], 0.0)
        self.assertEqual([row['i'] for row in result['top_credit']], [0])
        self.assertEqual(result['top_blame'], [])

    def test_recorded_total_value_does_not_add_potential_twice(self):
        row = decision('FP', 0.6)
        row['decision_context']['potential'] = 0.2
        result = self.compute([row])
        self.assertEqual(result['steps'][0]['v'], 0.6)
        self.assertAlmostEqual(result['steps'][0]['delta'], 0.4)

    def test_events_compare_exact_decision_prestate(self):
        unrelated = {'kind': 'annotation', 'state': state(corruption=0)}
        row = decision('FP', 0.2, pre=state(corruption=4, fp_vp=1),
                       post=state(corruption=5, fp_vp=3))
        result = self.compute([unrelated, row])
        self.assertEqual(result['steps'][0]['events'], ['+2 FP VP', '+1 corr'])
        self.assertEqual(result['steps'][0]['event_source'], 'recorded_predecision')

    def test_explicit_checkpoint_reanalysis_is_labeled_and_side_scoped(self):
        self.loader.side_effect = None
        self.loader.return_value = ConstantNet(0.5)
        result = self.compute([decision('FP', 0.9), decision('SH', -0.3)],
                               checkpoint_paths={'fp': 'chosen_by_user.pt'})
        fp, sh = result['steps']
        self.assertEqual(fp['v'], 0.5)
        self.assertEqual(fp['value_source'], 'retrospective_checkpoint')
        self.assertEqual(sh['v'], -0.3)
        self.assertEqual(sh['value_source'], 'recorded_behavior')
        self.assertEqual(result['value_mode'], 'mixed')
        self.assertEqual(result['nets'], {'fp': 'chosen_by_user.pt', 'sh': None})
        self.loader.assert_called_once_with('chosen_by_user.pt', side='fp')
        self.newest.assert_not_called()

    def test_success_restores_existing_profiles_and_reward_environment(self):
        before = dict(ai.PLAYER_PROFILES)
        real_replay = trainer._replay_reward_stream
        def observe_contract(*args, **kwargs):
            self.assertEqual(os.environ['WOTR_REWARD_CONTRACT'], WINLOSS)
            self.assertEqual(ai.PLAYER_PROFILES['fp'], 'ring')
            return real_replay(*args, **kwargs)
        with patch.object(trainer, '_replay_reward_stream', side_effect=observe_contract):
            self.compute([decision('FP', 0.0)])
        self.assertEqual(ai.PLAYER_PROFILES, before)
        self.assertEqual(os.environ['WOTR_REWARD_CONTRACT'], LEGACY)

    def test_failure_restores_profiles_and_absent_reward_environment(self):
        os.environ.pop('WOTR_REWARD_CONTRACT', None)
        before = dict(ai.PLAYER_PROFILES)
        with patch.object(trainer, '_replay_reward_stream', side_effect=RuntimeError('fixture')):
            with self.assertRaisesRegex(RuntimeError, 'fixture'):
                self.compute([decision('FP', 0.0)])
        self.assertNotIn('WOTR_REWARD_CONTRACT', os.environ)
        self.assertEqual(ai.PLAYER_PROFILES, before)

    def test_unknown_contract_is_refused_and_environment_restored(self):
        before = dict(ai.PLAYER_PROFILES)
        with self.assertRaisesRegex(ValueError, 'Unknown WOTR_REWARD_CONTRACT'):
            self.compute([decision('FP', 0.0)], contract='future_unknown_contract')
        self.assertEqual(os.environ['WOTR_REWARD_CONTRACT'], LEGACY)
        self.assertEqual(ai.PLAYER_PROFILES, before)

    def test_undecided_game_has_provisional_bootstrap_without_result_label(self):
        result = self.compute([decision('FP', 0.2), decision('FP', 0.6)], winner=None)
        self.assertTrue(result['untrained_timeout'])
        self.assertEqual(result['outcomes'], {'fp': None, 'sh': None})
        self.assertAlmostEqual(result['steps'][0]['delta'], 0.4)
        self.assertEqual(result['steps'][1]['delta'], 0.0)

    def test_legacy_newest_models_are_retrospective_and_bind_legacy_contract(self):
        os.environ['WOTR_REWARD_CONTRACT'] = WINLOSS
        self.newest.side_effect = [(ConstantNet(0.6), 'recent-fp.pt'),
                                   (ConstantNet(-0.4), 'recent-sh.pt')]
        def legacy_stream(trace, profiles, detail, *, for_learning):
            self.assertEqual(os.environ['WOTR_REWARD_CONTRACT'], LEGACY)
            detail.extend([{'step_idx': 0, 'tag': 'fp', 'k': 0, 'paid': 0.0},
                           {'step_idx': 1, 'tag': 'sh', 'k': 0, 'paid': 0.0}])
            return {'fp': ([[0.25]], [0.1], [0.1]),
                    'sh': ([[0.25]], [-0.1], [-0.1])}
        def legacy_terminal(tag, profile, victory, won, final_state):
            self.assertEqual(os.environ['WOTR_REWARD_CONTRACT'], LEGACY)
            return 1.0 if won else -1.0
        with patch.object(trainer, '_replay_reward_stream', side_effect=legacy_stream), \
                patch.object(ai, '_terminal_for', side_effect=legacy_terminal):
            result = self.compute([decision('FP', 0.9), decision('SH', -0.9)], contract=LEGACY)
        self.assertEqual(result['value_mode'], 'retrospective_checkpoint')
        self.assertAlmostEqual(result['steps'][0]['v'], 0.7)
        self.assertAlmostEqual(result['steps'][1]['v'], -0.5)
        self.assertEqual(self.newest.call_count, 2)
        self.assertEqual(os.environ['WOTR_REWARD_CONTRACT'], WINLOSS)


if __name__ == '__main__':
    unittest.main()
