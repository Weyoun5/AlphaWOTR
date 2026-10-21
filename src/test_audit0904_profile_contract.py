"""Profile rewards and learning inputs with literal numerical oracles."""
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
import numpy as np
import torch
import ai
import board
import game_state as gs
import game_trace
import run_td_v33loop as loop
import delta_trainer
import trace_admission
import td_attribution
import value_trainer_v2 as value
import reward_contract as rc
from test_audit0904_admission import terminal_trace


class ProfileContractTests(unittest.TestCase):
    def setUp(self):
        scope = patch.dict(os.environ, {'WOTR_REWARD_CONTRACT': rc.PROFILE})
        scope.start(); self.addCleanup(scope.stop)
        profiles = patch.dict(ai.PLAYER_PROFILES, {'fp': 'ring', 'sh': 'military'})
        profiles.start(); self.addCleanup(profiles.stop)

    def test_literal_victory_matrix_preserves_mission_preferences(self):
        cases = [('fp', 'ring', 'ring_destroyed', 1),
                 ('fp', 'ring', 'military', 0),
                 ('fp', 'military', 'military', 1),
                 ('fp', 'military', 'ring_destroyed', 0),
                 ('sh', 'corruption', 'ring_corruption', 1),
                 ('sh', 'corruption', 'military', .2),
                 ('sh', 'military', 'military', 1),
                 ('sh', 'military', 'ring_corruption', .2)]
        for side, profile, route, expected in cases:
            with self.subTest(side=side, profile=profile, route=route):
                self.assertEqual(ai._terminal_for(side, profile, route, True), expected)
                self.assertEqual(rc.terminal(side, side, profile=profile,
                    victory_type=route), expected)

    def test_mixed_profile_values_all_own_wins_equally(self):
        for side, routes in [('fp', ['military', 'ring_destroyed']),
                             ('sh', ['military', 'ring_corruption'])]:
            for route in routes:
                self.assertEqual(ai._terminal_for(side, 'mixed', route, True), 1)

    def test_loss_is_negative_even_if_loser_almost_achieved_its_goal(self):
        state = gs.setup_initial_game(board.load_board())
        state.fellowship.corruption = 11
        with patch.object(ai, '_TERM_CLOSENESS', 100):
            self.assertEqual(ai._terminal_for('sh', 'corruption', 'ring_destroyed', False, state), -1)
            self.assertEqual(ai._terminal_for('fp', 'ring', 'ring_corruption', False, state), -1)

    def test_fixed_contract_cannot_be_changed_by_legacy_environment_weights(self):
        with patch.object(ai, '_OFFMISSION_FP', .9), patch.object(ai, '_OFFMISSION_SH', .9):
            self.assertEqual(ai._terminal_for('fp', 'ring', 'military', True), 0)
            self.assertEqual(ai._terminal_for('sh', 'military', 'ring_corruption', True), .2)

    def test_invalid_profile_and_impossible_winner_route_are_rejected(self):
        for args in [('fp', 'corruption', 'military', True),
                     ('sh', 'ring', 'military', True),
                     ('fp', 'ring', 'ring_corruption', True),
                     ('sh', 'military', 'ring_corruption', False)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                rc.profile_terminal(*args)

    def test_microdecisions_do_not_change_profile_return(self):
        state = gs.setup_initial_game(board.load_board())
        state.fellowship.corruption = 11
        for side in (gs.Side.FREE_PEOPLES, gs.Side.SHADOW):
            self.assertEqual([ai._compute_step_reward_split({}, state, side)
                              for _ in range(10)], [(0., 0.)] * 10)
        self.assertEqual(ai._profile_shaping(state), 0)

    def test_profile_replay_keeps_exact_predecision_inputs(self):
        tr = {'meta': {'reward_contract': rc.PROFILE, 'action_encoding_contract': 'casualty_operations_v2'}, 'steps': [
            {'kind': 'action', 'side': 'FP', 'state': object(),
             'decision_context': {'contract': rc.PROFILE, 'encoding': [2., 3.]}}]}
        per = loop._replay_reward_stream(tr, {'fp': 'ring', 'sh': 'military'})
        np.testing.assert_array_equal(per['fp'][0], [[2., 3.]])
        self.assertEqual(per['fp'][1:], ([0.], [0.]))

    def test_off_target_wins_and_losses_still_train_search_policy(self):
        tr = {'meta': {'reward_contract': rc.PROFILE, 'winner': 'fp',
                       'victory_type': 'military'}, 'steps': []}
        self.assertEqual(delta_trainer._axis_outcome_weight(tr, 'fp'), 1)
        self.assertEqual(delta_trainer._axis_outcome_weight(tr, 'sh'), 1)

    def test_missing_trace_success_is_not_omitted_from_attempt_denominator(self):
        attempt = {'trace_path': 'w0_g1.trace.gz', 'status': 'completed'}
        joined = loop._join_attempt_records([], [attempt])
        self.assertEqual(len(joined), 1)
        self.assertFalse(joined[0]['admission']['accepted'])
        self.assertEqual(joined[0]['admission']['errors'][0]['code'], 'trace_missing')

    def test_partial_error_trace_and_error_attempt_are_one_game(self):
        trace = {'trace_path': 'w0_g1_sh_military.trace.gz',
                 'admission': {'accepted': False}}
        attempt = {'trace_path': 'w0_g1.trace.gz', 'status': 'error', 'error': 'failed execution'}
        joined = loop._join_attempt_records([trace], [attempt])
        self.assertEqual(len(joined), 1)
        self.assertEqual(joined[0]['status'], 'error')

    def test_off_target_shadow_return_propagates_with_literal_lambda_values(self):
        class First(torch.nn.Module):
            def forward(self, x): return x[:, :1]
        # G1=.2; G0=.2*.5 + .8*.2=.26, independent of engine rewards.
        seq = (np.array([[.7], [.5]], dtype=np.float32), np.zeros(2), np.zeros(2), .2)
        target = value.build_targets(First(), [seq], lam=.8, clip=2)[0]
        np.testing.assert_allclose(target, [.26, .2], atol=1e-7)

    def test_scaled_head_bootstrap_cannot_exceed_best_possible_profile_reward(self):
        class First(torch.nn.Module):
            def forward(self, x): return x[:, :1]
        seq = (np.array([[1.8], [1.8], [1.8]], dtype=np.float32),
               np.zeros(3), np.zeros(3), 1.)
        target = value.build_targets(First(), [seq], lam=.8, clip=2)[0]
        np.testing.assert_array_equal(target, [1., 1., 1.])

    def test_accidentally_shaped_profile_buffer_is_rejected(self):
        class First(torch.nn.Module):
            def forward(self, x): return x[:, :1]
        seq = (np.array([[0.]], dtype=np.float32), np.array([.01]), np.zeros(1), 1.)
        with self.assertRaisesRegex(ValueError, 'zero rewards'):
            value.build_targets(First(), [seq], clip=2)

    def test_wrong_profile_model_is_rejected_before_episode_start(self):
        fp = SimpleNamespace(reward_profile='military')
        with self.assertRaisesRegex(ValueError, 'checkpoint profile'):
            ai._run_single_episode(fp, object(), {})

    def test_checkpoint_profile_is_validated_before_model_state_load(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'model.pt'
            torch.save({'reward_contract': rc.PROFILE, 'reward_profile': 'corruption',
                        'action_encoding_contract': 'casualty_operations_v2'}, path)
            with self.assertRaisesRegex(ValueError, 'Checkpoint profile'):
                ai.load_value_net(str(path), side='sh', profile='military')

    def test_auxiliary_checkpoint_writer_preserves_objective_identity(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value_head = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Tanh())
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder)/'profile.pt')
            value.save_v2_checkpoint(Tiny(), 'sh', path, profile='military')
            with self.assertRaisesRegex(ValueError, 'Checkpoint profile'):
                ai.load_value_net(path, side='sh', profile='corruption')
            with patch.dict(os.environ, {'WOTR_REWARD_CONTRACT': rc.WINLOSS}):
                with self.assertRaisesRegex(ValueError, 'Checkpoint reward contract'):
                    ai.load_value_net(path, side='sh')

    def test_old_and_winloss_buffers_cannot_be_reused_for_profile_training(self):
        from selector_contract import training_identity
        for contract in (rc.LEGACY, rc.WINLOSS):
            with self.subTest(contract=contract), tempfile.TemporaryDirectory() as folder:
                with patch.dict(os.environ, {'WOTR_REWARD_CONTRACT': contract}):
                    buffer = loop.ReplayBuffer(folder, [('fp', 'ring')])
                    buffer.add(0, {('fp', 'ring'): {'reward_contract': contract, 'value': [1], 'delta': [],
                        **training_identity(),
                        'observation_contract': 'v44_fellowship_tail_v1',
                    'action_encoding_contract': 'casualty_operations_v2',
                        'ai_policy_contract': 'elite_and_bonus_draw_v1'}})
                with self.assertRaisesRegex(ValueError, 'reward contract mismatch'):
                    loop.ReplayBuffer(folder, [('fp', 'ring')]).load(('fp', 'ring'))

    def new_trace(self):
        from selector_contract import current_identity, make_evidence
        tr = terminal_trace()
        tr['meta'].update(reward_contract=rc.PROFILE,
                          profiles={'fp': 'ring', 'sh': 'military'})
        context = ai._decision_context(tr['steps'][0]['state'],
                                      gs.Side.FREE_PEOPLES, tracer=object())
        context['value'] = .3
        tr['steps'][1]['decision_context'] = context
        tr['meta'].update(current_identity())
        ss = dict(ae=np.zeros((1, ai.ACTION_ENCODING_DIM)), ri=np.zeros(1, dtype=int),
                  p=np.ones(1), q=np.zeros(1), n=np.ones(1), played=0)
        ss.update(make_evidence(ai.SELECT_CFG, 0., [1.], eligible=[True],
                                reward_contract=rc.PROFILE))
        tr['steps'][1]['search_stats'] = ss
        return tr

    def test_profile_admission_requires_objective_metadata(self):
        tr = self.new_trace()
        self.assertTrue(trace_admission.validate_trace(tr, decision_contract=rc.PROFILE)['accepted'])
        del tr['meta']['profiles']
        report = trace_admission.validate_trace(tr, decision_contract=rc.PROFILE)
        self.assertFalse(report['accepted'])
        self.assertIn('profile_objective_missing', [e['code'] for e in report['errors']])

    def test_attribution_uses_recorded_profile_without_loading_newest_network(self):
        tr = self.new_trace()
        with patch.object(game_trace, 'load_trace', return_value=tr), \
                patch.object(loop, '_apply_loop_config'), \
                patch.object(td_attribution, '_load_net', side_effect=AssertionError('newest model')):
            result = td_attribution.compute_attribution('unused')
        self.assertEqual(result['outcomes'], {'fp': -1., 'sh': .2})
        self.assertEqual(result['value_mode'], 'recorded_behavior')
        self.assertAlmostEqual(result['steps'][0]['adv'], -1.3)

    def test_attribution_cannot_relabel_the_recorded_profile(self):
        with patch.object(game_trace, 'load_trace', return_value=self.new_trace()):
            with self.assertRaisesRegex(ValueError, 'recorded reward objective'):
                td_attribution.compute_attribution('unused', sh_profile='corruption')


class PackagedProfileLoadingTests(unittest.TestCase):
    """Real temporary checkpoints expose the bridge's objective initialization."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        for scope in (
                patch.dict(os.environ, {'WOTR_REWARD_CONTRACT': rc.PROFILE}),
                patch.dict(ai.PLAYER_PROFILES, {'fp': None, 'sh': None}),
                patch.object(ai, '_PROJECT_ROOT', self.folder.name),
                patch.object(ai, 'DEVICE', torch.device('cpu'))):
            scope.start()
            self.addCleanup(scope.stop)

    def write_pair(self, shadow_profile='military', omit_shadow_profile=False):
        from action_schema import ACTION_ENCODING_CONTRACT
        # No archived model dependency and no mocked loader/model/validation.
        # Preserve the surrounding suite's random stream while making real nets.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(31)
            for side, profile in [('fp', 'ring'), ('sh', shadow_profile)]:
                net = ai.WOTRValueNet(ai.get_feature_dim(), side=side)
                checkpoint = dict(model_state=net.state_dict(),
                    input_dim=ai.get_feature_dim(), action_dim=ai.ACTION_ENCODING_DIM,
                    observation_contract=ai.OBSERVATION_CONTRACT,
                    action_encoding_contract=ACTION_ENCODING_CONTRACT,
                    reward_contract=rc.PROFILE, reward_profile=profile)
                if side == 'sh' and omit_shadow_profile:
                    checkpoint.pop('reward_profile')
                torch.save(checkpoint, Path(self.folder.name)/('wotr_%s_loader_test.pt' % side))

    def test_pair_load_initializes_actual_search_terminal_profiles(self):
        import engine_play
        self.write_pair()
        fp, sh = engine_play.load_nets('loader_test')
        self.assertEqual((fp.reward_profile, sh.reward_profile), ('ring', 'military'))
        # Literal outcomes, not values calculated by another reward helper.
        cases = [('free_peoples', 'ring_destroyed', 1., -1., 'W', 'L'),
                 ('free_peoples', 'military', 0., -1., None, 'L'),
                 ('shadow', 'military', -1., 1., 'L', 'W'),
                 ('shadow', 'ring_corruption', -1., .2, 'L', None)]
        for winner, route, fp_value, sh_value, fp_mark, sh_mark in cases:
            with self.subTest(winner=winner, route=route):
                result = {'victory': winner, 'victory_type': route}
                self.assertEqual(ai._search_terminal(result, gs.Side.FREE_PEOPLES), fp_value)
                self.assertEqual(ai._search_terminal(result, gs.Side.SHADOW), sh_value)
                self.assertEqual(ai._proof_mark(result, gs.Side.FREE_PEOPLES), fp_mark)
                self.assertEqual(ai._proof_mark(result, gs.Side.SHADOW), sh_mark)
        self.assertEqual(ai.PLAYER_PROFILES, {'fp': 'ring', 'sh': 'military'})

    def test_invalid_second_profile_preserves_previous_pair(self):
        import engine_play
        self.write_pair(shadow_profile='ring')
        previous = {'fp': 'military', 'sh': 'corruption'}
        ai.PLAYER_PROFILES.update(previous)
        with self.assertRaisesRegex(ValueError, 'Invalid profile for sh'):
            engine_play.load_nets('loader_test')
        self.assertEqual(ai.PLAYER_PROFILES, previous)

    def test_missing_second_profile_preserves_previous_pair(self):
        import engine_play
        self.write_pair(omit_shadow_profile=True)
        previous = {'fp': 'military', 'sh': 'corruption'}
        ai.PLAYER_PROFILES.update(previous)
        with self.assertRaisesRegex(ValueError, 'Invalid profile for sh'):
            engine_play.load_nets('loader_test')
        self.assertEqual(ai.PLAYER_PROFILES, previous)


if __name__ == '__main__':
    unittest.main(verbosity=2)
