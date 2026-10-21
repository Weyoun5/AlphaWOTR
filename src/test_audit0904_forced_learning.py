"""Fixed policies remain observable transitions, not fabricated learning targets."""
import copy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import ai
import ai_policies
import board
import delta_trainer as delta
import game_state as gs
import game_trace
import run_td_v33loop as loop
import td_attribution as attribution
import trace_admission as admission
from reward_contract import LEGACY, WINLOSS

REASON = 'bonus_draw_always_take'


def snapshot(score=0):
    return SimpleNamespace(score=score, turn=1,
        fellowship=SimpleNamespace(corruption=0, progress=0))


def action_row(value=.2, score=0, *, forced=False, encoding=0, side='FP'):
    st = snapshot(score)
    row = dict(kind='action', turn=1, side=side, action_type='literal_choice',
        die_idx=-1, params={}, state=st,
        decision_context=dict(contract=WINLOSS, state=st, encoding=np.array([encoding], np.float32),
                              reward_stats={}, potential=0., value=value))
    if forced:
        row.update(policy_forced=True, forced_policy_reason=REASON)
    return row


def stats():
    return dict(ae=np.array([[0.], [1.]], np.float32), ri=np.array([-1, -1]),
                p=np.array([.5, .5]), q=np.array([0., 1.]), n=np.array([1, 2]), played=1)


class ForcedRecorderTests(unittest.TestCase):
    def test_forced_transition_round_trips_without_search_targets(self):
        state = gs.setup_initial_game(board.load_board())
        with tempfile.TemporaryDirectory(prefix='forced-trace-') as folder:
            path = Path(folder) / 'record.trace.gz'
            recorder = game_trace.TraceRecorder(str(path))
            recorder.record_initial(state)
            recorder.record_action(state, gs.Side.FREE_PEOPLES,
                ai.Action(-1, 'card_option', {'choices': {'gandalf_draw': True}}),
                forced_policy=REASON)
            recorder.save(winner='timeout', victory_type='timeout')
            trace = game_trace.load_trace(path)
        row = trace['steps'][1]
        self.assertTrue(row['policy_forced'])
        self.assertEqual(row['forced_policy_reason'], REASON)
        self.assertEqual(row['kind'], 'action')
        self.assertIsNotNone(row['state'].board)
        self.assertNotIn('search_stats', row)
        self.assertNotIn('regret', row)

    def test_recorder_refuses_forced_action_with_policy_target(self):
        state = gs.setup_initial_game(board.load_board())
        recorder = game_trace.TraceRecorder('never-written.trace.gz')
        with self.assertRaisesRegex(ValueError, 'forced'):
            recorder.record_action(state, gs.Side.FREE_PEOPLES,
                ai.Action(-1, 'card_option'), forced_policy=REASON,
                search_stats=stats())
        self.assertEqual(recorder.steps, [])


class ForcedLearningTests(unittest.TestCase):
    def projected_menu_trace(self, version=True):
        # The first Regular is excluded only when the same location/die
        # offers an Elite. The other ordinary choice remains learned.
        legal = [ai.Action(0, 'muster_recruit', {'nation': 'gondor',
                    'region': 'minas tirith', 'type': 'regular'}),
                 ai.Action(0, 'muster_elite', {'nation': 'gondor',
                    'region': 'minas tirith'}),
                 ai.Action(1, 'pass')]
        row = action_row()
        row['search_stats'] = stats()
        row['search_stats']['ae'] = np.array([[1.], [2.]], np.float32)
        tr = dict(meta={}, steps=[dict(kind='initial', state=snapshot()), row])
        if version:
            tr['meta']['ai_policy_contract'] = ai_policies.POLICY_VERSION
        return tr, legal

    def test_versioned_policy_replay_excludes_forbidden_regular_from_softmax(self):
        for version, extras in ((True, []), (False, [[0.]])):
            with self.subTest(version=version):
                trace, legal = self.projected_menu_trace(version)
                with patch.object(ai, 'get_legal_actions', return_value=legal), \
                        patch.object(ai, 'encode_action',
                            side_effect=lambda a, *_: np.array([legal.index(a)], np.float32)), \
                        patch.object(ai, '_get_action_region_index', return_value=-1):
                    rows, diagnostics = delta.collect_delta_samples(trace, 'fp')
                self.assertEqual(diagnostics['rows'], 1)
                self.assertEqual(rows[0]['extra_ae'].tolist(), extras)
                self.assertEqual(rows[0]['sm'].tolist(), [True, True])

    def test_unknown_policy_version_is_refused_before_learning(self):
        trace, _ = self.projected_menu_trace()
        trace['meta']['ai_policy_contract'] = 'unknown-policy-v999'
        trace['meta']['reward_contract'] = WINLOSS
        with self.assertRaisesRegex(ValueError, 'policy contract'):
            delta.collect_delta_samples(trace, 'fp')
        with self.assertRaisesRegex(ValueError, 'policy contract'):
            loop._replay_reward_stream(trace, {'fp': 'ring', 'sh': 'military'})

    def test_corrected_value_stream_skips_fixed_rows_and_preserves_learned_order(self):
        rows = [action_row(encoding=1), action_row(forced=True, encoding=999),
                action_row(encoding=2)]
        details = []
        result = loop._replay_reward_stream(
            dict(meta={'reward_contract': WINLOSS,
                       'action_encoding_contract': 'casualty_operations_v2'}, steps=rows),
            {'fp': 'ring', 'sh': 'military'}, details)
        self.assertEqual([float(x[0]) for x in result['fp'][0]], [1, 2])
        self.assertEqual(result['fp'][1:], ([0., 0.], [0., 0.]))
        self.assertEqual([d['step_idx'] for d in details], [0, 2])

    def test_legacy_reward_interval_and_terminal_tail_include_forced_effects_once(self):
        rows = [dict(kind='initial', side=None, state=snapshot(0))]
        rows += [action_row(score=0), action_row(score=3, forced=True),
                 action_row(score=5), action_row(score=7, forced=True)]
        # Literal accounting: two learned decisions cost1 each; total
        # state progress is7. Forced transitions cost0, so total reward5.
        def split(pre, post, side, include_decision_cost=True):
            return post.score - pre['score'] - int(include_decision_cost), 0.
        with patch.dict(os.environ, {'WOTR_REWARD_CONTRACT': LEGACY}), \
                patch.object(ai, '_reward_stats', side_effect=lambda st: {'score': st.score}), \
                patch.object(ai, '_compute_step_reward_split', side_effect=split), \
                patch.object(ai, '_is_military_action', return_value=False), \
                patch.object(ai, '_profile_shaping', return_value=0.), \
                patch.object(ai, 'encode_state', side_effect=lambda st, side: np.array([st.score])):
            result = loop._replay_reward_stream(dict(steps=rows), {'fp': 'ring', 'sh': 'military'})
        self.assertEqual(result['fp'][1], [-1., 6.])
        self.assertEqual(sum(result['fp'][1]), 5.)

    def test_both_policy_collectors_exclude_forced_rows_even_with_stale_search_stats(self):
        forced = action_row(forced=True)
        forced['search_stats'] = stats()
        trace = dict(steps=[dict(kind='initial', state=snapshot()), forced])
        with patch.object(ai, 'get_legal_actions', side_effect=AssertionError('must skip before menu')):
            collected, diagnostics = delta.collect_delta_samples(trace, 'fp')
            legacy = loop._collect_search_samples(trace, 'fp')
        self.assertEqual(collected, [])
        self.assertEqual(legacy, [])
        self.assertEqual(diagnostics['forced_dropped'], 1)
        self.assertEqual(diagnostics['error'], 0)

    def test_preceding_learned_choices_keep_continuation_and_terminal_credit(self):
        trace = dict(meta={'reward_contract': WINLOSS, 'winner': 'fp',
                           'victory_type': 'ring_destroyed'},
            steps=[action_row(value=.2), action_row(value=-.99, forced=True),
                   action_row(value=.8), action_row(value=-.99, forced=True)])
        with patch.object(game_trace, 'load_trace', return_value=trace), \
                patch.object(loop, '_apply_loop_config'), patch.object(loop, 'LAM', .8), \
                patch('game_logic.calculate_victory_points',
                      return_value={'free_peoples': 0, 'shadow': 0}), \
                patch.object(ai, 'load_value_net', side_effect=AssertionError('no checkpoint')):
            result = attribution.compute_attribution('unwritten.trace.gz')
        self.assertEqual(len(result['steps']), 2)
        self.assertEqual(len(result['forced_steps']), 2)
        self.assertAlmostEqual(result['steps'][0]['delta'], .6)
        self.assertAlmostEqual(result['steps'][0]['adv'], .76)
        self.assertAlmostEqual(result['steps'][1]['delta'], .2)
        self.assertAlmostEqual(result['steps'][1]['adv'], .2)
        self.assertEqual(result['stream_extra'], 0)
        self.assertEqual(result['unmapped'], 0)
        self.assertTrue(all('adv' not in row and 'regret' not in row
                            for row in result['forced_steps']))


class ForcedAdmissionTests(unittest.TestCase):
    def terminal(self):
        state = gs.setup_initial_game(board.load_board())
        initial = copy.deepcopy(state)
        state.fellowship.corruption = 12
        row = dict(kind='action', turn=1, side='FP', action_type='card_option',
                   params={}, state=state, policy_forced=True, forced_policy_reason=REASON)
        return dict(meta={'winner': 'sh', 'victory_type': 'ring_corruption',
                          'source_integrity_version': 1, 'source_errors': []},
                    steps=[dict(kind='initial', turn=1, side=None, state=initial), row])

    def test_valid_forced_transition_is_counted_separately_from_decisions(self):
        report = admission.validate_trace(self.terminal(), audit_rules=False)
        self.assertTrue(report['accepted'], report)
        self.assertEqual(report['decisions'], {'fp': 0, 'sh': 0})
        self.assertEqual(report['forced_transitions'], {'fp': 1, 'sh': 0})

    def test_invalid_role_metadata_cannot_hide_or_create_training_decisions(self):
        for fields in ({'policy_forced': 'true'}, {'forced_policy_reason': ''},
                       {'forced_policy_reason': 'unreviewed_new_policy'},
                       {'policy_forced': False}):
            with self.subTest(fields=fields):
                trace = self.terminal()
                trace['steps'][-1].update(fields)
                report = admission.validate_trace(trace, audit_rules=False)
                self.assertFalse(report['accepted'])
                self.assertIn('forced_policy_schema', [e['code'] for e in report['errors']])

    def test_forced_row_with_a_policy_target_is_rejected(self):
        trace = self.terminal()
        trace['steps'][-1]['search_stats'] = stats()
        report = admission.validate_trace(trace, audit_rules=False)
        self.assertFalse(report['accepted'])
        self.assertIn('forced_policy_target', [e['code'] for e in report['errors']])

    def test_explicit_unknown_policy_version_is_rejected(self):
        trace = self.terminal()
        trace['meta']['ai_policy_contract'] = 'unknown-policy-v999'
        report = admission.validate_trace(trace, audit_rules=False)
        self.assertFalse(report['accepted'])
        self.assertIn('ai_policy_contract', [e['code'] for e in report['errors']])

    def test_versioned_forced_action_requires_exact_prestate_and_literal_projected_choice(self):
        from test_audit0904_gandalf_draw import opening_for
        source = 'FREE_PEOPLES_Strategy_Guards_of_the_Citadel'
        trace = self.terminal()
        trace['meta']['ai_policy_contract'] = ai_policies.POLICY_VERSION
        row = trace['steps'][-1]
        row.update(die_idx=-1, action_type='card_option', params={
            'card_id': '__gandalf_draw__', 'source_card_id': source, 'side': 'fp',
            'deck': 'strategy', 'choices': {'gandalf_draw': True}})
        row['decision_context'] = dict(state=opening_for(source, 'strategy'),
            contract=WINLOSS, encoding=np.array([0.]), reward_stats={}, potential=0.)
        report = admission.validate_trace(trace, audit_rules=False)
        self.assertTrue(report['accepted'], report)
        for name in ('decline', 'wrong-reason', 'missing-prestate'):
            with self.subTest(name=name):
                bad = copy.deepcopy(trace)
                row = bad['steps'][-1]
                if name == 'decline':
                    row['params']['choices']['gandalf_draw'] = False
                elif name == 'wrong-reason':
                    row['forced_policy_reason'] = 'recruit_elite_if_available'
                else:
                    del row['decision_context']['state']
                report = admission.validate_trace(bad, audit_rules=False)
                self.assertFalse(report['accepted'])
                self.assertIn('forced_policy_semantics', [e['code'] for e in report['errors']])


if __name__ == '__main__':
    unittest.main(verbosity=2)
