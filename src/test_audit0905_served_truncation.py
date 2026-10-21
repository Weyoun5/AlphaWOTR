"""Actual episode boundary controls: a turn cap is never a made-up draw."""
import contextlib
import os
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import ai
import game_logic
import game_trace
from infer_server import RemoteNet


class LiteralClient:
    def __init__(self, values):
        self.values, self.calls = values, []

    def eval(self, key, x):
        self.calls.append((key, x.copy()))
        if isinstance(self.values, Exception):
            raise self.values
        return self.values


class EpisodeTruncationTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, {
            'WOTR_REWARD_CONTRACT': 'predecision_profile_v1',
            'WOTR_ACT_GEN': 'v41'}))
        self.stack.enter_context(patch.dict(ai.PLAYER_PROFILES, fp='ring', sh='military'))
        self.stack.enter_context(patch.object(ai, 'DEVICE', torch.device('cpu')))
        self.addCleanup(random.setstate, random.getstate())
        self.addCleanup(np.random.set_state, np.random.get_state())
        self.old_vp = game_logic.SHADOW_VP_VICTORY
        self.old_tracer = game_trace.ACTIVE_TRACER
        self.addCleanup(setattr, game_logic, 'SHADOW_VP_VICTORY', self.old_vp)
        self.addCleanup(setattr, game_trace, 'ACTIVE_TRACER', self.old_tracer)

    def episode(self, f, s, **config):
        return ai._run_single_episode(RemoteNet('fp', f), RemoteNet('sh', s),
            dict(max_turns=0, mcts_sims=4, shadow_vp=9, **config))

    def test_turn_cap_requests_each_own_value_and_keeps_timeout_identity(self):
        f, s = LiteralClient([.6]), LiteralClient([-.4])
        result = self.episode(f, s)
        self.assertEqual((result['winner'], result['victory_type']), ('timeout', 'timeout'))
        self.assertAlmostEqual(result['outcome_fp'], .6)
        self.assertAlmostEqual(result['outcome_sh'], -.4)
        self.assertEqual(len(result['states']), 0)
        self.assertEqual([c[0] for c in f.calls], ['fp'])
        self.assertEqual([c[0] for c in s.calls], ['sh'])
        self.assertEqual(f.calls[0][1].shape, (1, ai.get_feature_dim()))
        self.assertEqual(game_logic.SHADOW_VP_VICTORY, self.old_vp)

    def test_cap_preserves_existing_point_nine_bound(self):
        result = self.episode(LiteralClient([1.5]), LiteralClient([-1.5]))
        self.assertEqual(result['outcome_fp'], .9)
        self.assertEqual(result['outcome_sh'], -.9)

    def test_explicit_absent_net_keeps_untrained_timeout_control(self):
        for f, s, expected in ((None, None, (0., 0.)),
                              (None, RemoteNet('sh', LiteralClient([.6])), (0., .6))):
            with self.subTest(expected=expected):
                result = ai._run_single_episode(f, s, dict(max_turns=0, shadow_vp=9))
                self.assertEqual((result['outcome_fp'], result['outcome_sh']), expected)
                self.assertEqual(result['winner'], 'timeout')
                self.assertEqual(game_logic.SHADOW_VP_VICTORY, self.old_vp)

    def test_failed_value_never_returns_zero_outcomes_and_restores_globals(self):
        for bad in ([float('nan')], [], [.2, .3], RuntimeError('server gone')):
            with self.subTest(bad=repr(bad)):
                with self.assertRaises((ValueError, RuntimeError)):
                    self.episode(LiteralClient([.6]), LiteralClient(bad))
                self.assertEqual(game_logic.SHADOW_VP_VICTORY, self.old_vp)
                self.assertIs(game_trace.ACTIVE_TRACER, self.old_tracer)

    def test_aborted_search_cannot_leave_tracer_or_victory_override_in_next_game(self):
        with tempfile.TemporaryDirectory(prefix='wotr_abort_episode_') as folder:
            path = os.path.join(folder, 'aborted.trace.gz')
            with patch.object(ai, 'mcts_action_with_policy', side_effect=RuntimeError('search failed')):
                with self.assertRaisesRegex(RuntimeError, 'search failed'):
                    ai._run_single_episode(RemoteNet('fp', LiteralClient([.5])),
                        RemoteNet('sh', LiteralClient([.5])),
                        dict(max_turns=1, mcts_sims=4, shadow_vp=9, trace_path=path))
            self.assertEqual(game_logic.SHADOW_VP_VICTORY, self.old_vp)
            self.assertIs(game_trace.ACTIVE_TRACER, self.old_tracer)
            self.assertTrue(os.path.isfile(path))
            failed = game_trace.load_trace(path)
            self.assertEqual(failed['meta']['status'], 'error')
            self.assertIsNone(failed['meta']['winner'])
            self.assertIn('search failed', failed['meta']['error_traceback'])
            from trace_admission import validate_trace
            admission = validate_trace(failed, require_decided=False)
            self.assertFalse(admission['accepted'])
            self.assertIn('source_episode_exception', [e['code'] for e in admission['errors']])
            self.assertFalse(any(row.get('action_type') == 'pass' for row in failed['steps']))
            # Exercise the next episode too, rather than only inspecting globals.
            following = self.episode(LiteralClient([.6]), LiteralClient([-.4]))
            self.assertEqual(following['winner'], 'timeout')
            self.assertAlmostEqual(following['outcome_fp'], .6)
            self.assertAlmostEqual(following['outcome_sh'], -.4)
            self.assertEqual(game_logic.SHADOW_VP_VICTORY, self.old_vp)
            self.assertIs(game_trace.ACTIVE_TRACER, self.old_tracer)


if __name__ == '__main__':
    unittest.main()
