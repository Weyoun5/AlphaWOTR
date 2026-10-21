"""Literal target calculations and state/contract boundaries, independent of play strength."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
ROOT = Path(__file__).resolve().parents[1]
import numpy as np
import torch
import ai
import board
import game_state as gs
import game_trace
import run_td_v33loop as loop
import delta_trainer
import value_trainer_v2 as value
from reward_contract import WINLOSS, LEGACY, current_contract, terminal


class FirstFeature(torch.nn.Module):
    def forward(self, x):
        return x[:, :1]


class DecisionContractTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'WOTR_REWARD_CONTRACT': WINLOSS})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_every_legal_victory_has_equal_value_and_opposite_loss(self):
        for side in ('fp', 'sh'):
            for profile in ('ring', 'corruption', 'military', None):
                for victory in ('ring', 'military', 'corruption'):
                    with self.subTest(side=side, profile=profile, victory=victory):
                        self.assertEqual(ai._terminal_for(side, profile, victory, True), 1)
                        self.assertEqual(ai._terminal_for(side, profile, victory, False), -1)

    def test_no_reward_for_extra_microdecisions_or_recruit_category(self):
        state = gs.setup_initial_game(board.load_board())
        state.fellowship.corruption = 11
        state.enemy_controlled[gs.Side.SHADOW].add('minas tirith')
        with patch.dict(ai.PLAYER_PROFILES, {'fp': 'military', 'sh': 'corruption'}):
            for side in (gs.Side.SHADOW, gs.Side.FREE_PEOPLES):
                for _ in range(5):
                    self.assertEqual(ai._compute_step_reward_split({}, state, side), (0, 0))
            self.assertEqual(ai._profile_shaping(state), 0)

    def test_actual_predecision_context_survives_execution_and_disk(self):
        state = gs.setup_initial_game(board.load_board())
        with tempfile.TemporaryDirectory() as folder:
            recorder = game_trace.TraceRecorder(str(Path(folder)/'game.trace.gz'))
            recorder.record_initial(state)
            context = ai._decision_context(state, gs.Side.FREE_PEOPLES,
                                           recorder, '_declare_phase')
            expected = context['encoding'].copy()
            self.assertTrue(context['state']._declare_phase)
            self.assertFalse(getattr(state, '_declare_phase', False))
            state.fellowship.corruption = 7
            recorder.record_action(state, gs.Side.FREE_PEOPLES,
                                   ai.Action(-1, 'declare_decline', {}),
                                   decision_context=context)
            recorder.save(reward_contract=WINLOSS)
            tr = game_trace.load_trace(recorder.path)
            saved = tr['steps'][1]['decision_context']
            self.assertEqual(saved['state'].fellowship.corruption, 0)
            self.assertEqual(tr['steps'][1]['state'].fellowship.corruption, 7)
            np.testing.assert_array_equal(saved['encoding'], expected)
            np.testing.assert_array_equal(
                ai.encode_state(saved['state'], gs.Side.FREE_PEOPLES), expected)

    def test_replay_uses_literal_predecision_vectors_not_poststate(self):
        rows = [{'kind': 'initial', 'state': object()}]
        for side, vec in [('FP', [1., 2.]), ('SH', [3., 4.]), ('FP', [5., 6.])]:
            rows.append({'kind': 'action', 'side': side, 'state': object(),
                         'decision_context': {'contract': WINLOSS, 'encoding': vec}})
        result = loop._replay_reward_stream(
            {'meta': {'reward_contract': WINLOSS, 'action_encoding_contract': 'casualty_operations_v2'}, 'steps': rows}, {})
        np.testing.assert_array_equal(result['fp'][0], [[1, 2], [5, 6]])
        np.testing.assert_array_equal(result['sh'][0], [[3, 4]])
        self.assertEqual(result['fp'][1:], ([0., 0.], [0., 0.]))

    def test_missing_context_cannot_fall_back_to_poststate(self):
        with self.assertRaisesRegex(ValueError, 'pre-decision context'):
            loop._replay_reward_stream({'meta': {'reward_contract': WINLOSS, 'action_encoding_contract': 'casualty_operations_v2'},
                'steps': [{'kind': 'action', 'side': 'FP', 'state': object()}]}, {})

    def test_legacy_trace_cannot_be_reinterpreted_under_new_environment(self):
        with self.assertRaisesRegex(ValueError, 'Legacy replay requires'):
            loop._replay_reward_stream({'steps': []}, {})

    def test_losing_game_search_targets_have_positive_training_weight(self):
        tr = {'meta': {'reward_contract': WINLOSS, 'winner': 'sh'}, 'steps': []}
        self.assertEqual(delta_trainer._axis_outcome_weight(tr, 'fp'), 1)
        self.assertEqual(delta_trainer._axis_outcome_weight(tr, 'sh'), 1)
        tr['meta']['winner'] = 'timeout'
        self.assertEqual(delta_trainer._axis_outcome_weight(tr, 'fp'), 0)

    def test_unknown_contract_and_timeout_are_not_win_labels(self):
        with self.assertRaises(ValueError):
            terminal('timeout', 'fp')
        with patch.dict(os.environ, {'WOTR_REWARD_CONTRACT': 'typo'}):
            with self.assertRaises(ValueError):
                current_contract()

    def test_new_contract_rejects_legacy_training_operators(self):
        with patch.dict(os.environ, {'WOTR_OPS_V2': '0'}):
            with self.assertRaisesRegex(ValueError, 'requires the v2'):
                loop._resolve_ops()

    def test_old_buffer_cannot_enter_new_training(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict(os.environ, {'WOTR_REWARD_CONTRACT': LEGACY}):
                old = loop.ReplayBuffer(folder, [('fp', 'ring')])
                old.add(0, {('fp', 'ring'): {'value': [1], 'delta': [],
                    'observation_contract': 'v44_fellowship_tail_v1',
                    'action_encoding_contract': 'casualty_operations_v2',
                    'ai_policy_contract': 'elite_and_bonus_draw_v1'}})
            new = loop.ReplayBuffer(folder, [('fp', 'ring')])
            with self.assertRaisesRegex(ValueError, 'reward contract mismatch'):
                new.load(('fp', 'ring'))

    def test_old_payload_cannot_be_added_to_new_buffer(self):
        with tempfile.TemporaryDirectory() as folder:
            buffer = loop.ReplayBuffer(folder, [('fp', 'ring')])
            with self.assertRaisesRegex(ValueError, 'reward contract mismatch'):
                buffer.add(0, {('fp', 'ring'): {'value': [], 'delta': []}})
            self.assertEqual(buffer.batches(('fp', 'ring')), [])

    def test_new_buffer_preserves_both_sides_of_result(self):
        from selector_contract import training_identity
        with tempfile.TemporaryDirectory() as folder:
            keys = [('fp', 'ring'), ('sh', 'military')]
            buffer = loop.ReplayBuffer(folder, keys)
            buffer.add(0, {k: {'reward_contract': WINLOSS,
                              **training_identity(),
                              'observation_contract': 'v44_fellowship_tail_v1',
                    'action_encoding_contract': 'casualty_operations_v2',
                              'ai_policy_contract': 'elite_and_bonus_draw_v1',
                              'value': [terminal('sh', k[0])], 'delta': []}
                           for k in keys})
            self.assertEqual(buffer.load(keys[0])['value'], [-1])
            self.assertEqual(buffer.load(keys[1])['value'], [1])

    def test_checkpoint_stamp_refuses_wrong_objective_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'model.pt'
            torch.save({'reward_contract': LEGACY}, path)
            with self.assertRaisesRegex(ValueError, 'reward contract'):
                ai.load_value_net(str(path), 'fp')


class ReturnOracleTests(unittest.TestCase):
    def seq(self, outcome):
        return (np.array([[.2], [-.4], [.5]], dtype=np.float32),
                np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), outcome)

    def test_lambda_one_assigns_eventual_loss_to_every_decision(self):
        got = value.build_targets(FirstFeature(), [self.seq(-1)], lam=1, clip=1)[0]
        np.testing.assert_allclose(got, [-1, -1, -1])

    def test_lambda_zero_bootstraps_next_own_decision_not_current(self):
        got = value.build_targets(FirstFeature(), [self.seq(1)], lam=0, clip=1)[0]
        np.testing.assert_allclose(got, [-.4, .5, 1])

    def test_lambda_point_eight_has_hand_calculated_backward_credit(self):
        # G2=-1; G1=.2*.5 + .8*(-1)=-.7;
        # G0=.2*(-.4) + .8*(-.7)=-.64.
        got = value.build_targets(FirstFeature(), [self.seq(-1)], lam=.8, clip=1)[0]
        np.testing.assert_allclose(got, [-.64, -.7, -1], atol=1e-7)

    def test_nonfinite_and_misaligned_sequences_are_rejected(self):
        for slot in range(4):
            seq = list(self.seq(1))
            if slot == 3:
                seq[slot] = float('nan')
            else:
                seq[slot].flat[0] = float('nan')
            with self.subTest(slot=slot), self.assertRaises(ValueError):
                value.build_targets(FirstFeature(), [seq], lam=.8, clip=1)
        seq = list(self.seq(1)); seq[1] = np.zeros(2)
        with self.assertRaises(ValueError):
            value.build_targets(FirstFeature(), [seq], lam=.8, clip=1)

    def test_bootstrap_uses_eval_mode_and_restores_caller_mode(self):
        class Dropout(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.drop = torch.nn.Dropout(1)
            def forward(self, x):
                return self.drop(x[:, :1])
        net = Dropout().train()
        got = value.build_targets(net, [self.seq(1)], lam=0, clip=1)[0]
        np.testing.assert_allclose(got, [-.4, .5, 1])
        self.assertTrue(net.training)

    def test_final_singleton_minibatch_is_actually_trained(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.value_head = torch.nn.Sequential(
                    torch.nn.Linear(1, 1), torch.nn.Tanh())
                self.sizes = []
            def forward(self, x):
                if torch.is_grad_enabled(): self.sizes.append(len(x))
                return self.value_head(x)
        net = Tiny()
        # This one-dimensional optimizer fixture has no observation projection.
        with patch.dict(os.environ, WOTR_OBSERVATION_TRAINING='head_only_legacy'), \
                patch.object(value, 'VALUE_BS', 2), patch.object(value, 'VALUE_EPOCHS', 1):
            result = value.value_update_for_driver(net, [self.seq(1)], .001)
        self.assertTrue(np.isfinite(result))
        self.assertEqual(net.sizes, [2, 1])


class AdmissionIntegrationTests(unittest.TestCase):
    def test_audit_banner_cannot_override_failed_process(self):
        from types import SimpleNamespace
        result = SimpleNamespace(returncode=1, stdout=b'AUDIT PASS\n')
        with patch.object(loop.subprocess, 'run', return_value=result), \
                patch.object(loop, 'log_line'), patch('builtins.print'):
            self.assertFalse(loop.audit_batch('unused'))

    def test_empty_admission_never_retrains_on_old_buffer(self):
        from unittest.mock import Mock
        ctx = {'state': object(), 'play': lambda idx: 'unused',
               'audit': lambda path: True, 'census': lambda path: (True, [], {}),
               'extract': lambda path: {k: {'value': [], 'delta': []} for k in loop.NET_KEYS},
               'buffer': Mock(), 'train': Mock(), 'log': Mock()}
        with patch.object(loop, '_log_frozen_telemetry'), \
                self.assertRaisesRegex(RuntimeError, 'no admitted training data'):
            loop.run_batch_cycle(0, ctx)
        ctx['buffer'].add.assert_not_called()
        ctx['train'].assert_not_called()

    def test_historical_diagnostics_do_not_admit_old_observations_to_fresh_training(self):
        source = ROOT/'data/v42sx_selfplay_500/traces/w1_g0_sh_military.trace.gz'
        tr = game_trace.load_trace(str(source))
        tr['meta']['profiles'] = {'fp': 'ring', 'sh': 'military'}
        import gzip, pickle, copy, json
        bad = copy.deepcopy(tr)
        bad['meta']['source_integrity_version'] = 1
        bad['meta']['source_errors'] = [{'code': 'action_exec_failed', 'detail': 'injected real-action failure'}]
        # This is genuine historical data, not a current observation fixture.
        # Keep its diagnostic validity distinct from current learner admission.
        from trace_admission import validate_trace
        self.assertTrue(validate_trace(tr)['accepted'])
        self.assertFalse(validate_trace(bad)['accepted'])
        with tempfile.TemporaryDirectory() as folder:
            for name, trace in [('good.trace.gz', tr), ('bad.trace.gz', bad)]:
                with gzip.open(Path(folder)/name, 'wb') as handle:
                    pickle.dump(trace, handle)
            with patch.dict(os.environ, {'WOTR_REWARD_CONTRACT': LEGACY}), \
                    patch.object(loop, 'DELTA_COLLECTOR', lambda tr, tag: []):
                result = loop.extract_batch_streams(folder)
            report = json.loads((Path(folder)/'training_admission.json').read_text())
            self.assertEqual((report['accepted'], report['rejected']), (0, 2))
            self.assertEqual(len(result[('fp', 'ring')]['value']), 0)
            self.assertEqual(len(result[('sh', 'military')]['value']), 0)
            good = next(item for item in report['traces'] if item['file'] == 'good.trace.gz')
            self.assertIn('observation_contract_mismatch', [e['code'] for e in good['errors']])


if __name__ == '__main__':
    unittest.main(verbosity=2)
