"""Data-integrity gates with independent corruptions and real source failures."""
import contextlib
import copy
import io
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

import numpy as np
import ai
import board
import game_state as gs
import game_logic
import game_trace
import state_invariants
import trace_admission
import trace_audit

ROOT = Path(__file__).resolve().parents[1]


def state_row(state, side=None, atype=None, kind=None, **extra):
    return dict({'kind': kind or ('action' if atype else 'initial'),
                 'state': state, 'turn': state.turn, 'side': side,
                 'action_type': atype, 'params': {}, 'die_idx': -1}, **extra)


def terminal_trace():
    before = gs.setup_initial_game(board.load_board())
    before.fellowship.corruption = 11
    after = ai.clone_state(before)
    after.fellowship.corruption = 12
    return {'meta': {'action_encoding_contract': 'casualty_operations_v2',
                     'winner': 'sh', 'victory_type': 'ring_corruption',
                     'source_integrity_version': 1, 'source_errors': []},
            'steps': [state_row(before),
                      state_row(after, 'FP', 'hunt_take_corruption')]}


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.trace = terminal_trace()

    def validate(self, **kwargs):
        return trace_admission.validate_trace(self.trace, **kwargs)

    def assertRejected(self, code, **kwargs):
        report = self.validate(**kwargs)
        self.assertFalse(report['accepted'], report)
        self.assertIn(code, [e['code'] for e in report['errors']], report)

    def test_ordinary_losing_decision_is_admitted(self):
        report = self.validate()
        self.assertTrue(report['accepted'], report)
        self.assertEqual(report['decisions'], {'fp': 1, 'sh': 0})

    def test_missing_snapshot_is_rejected(self):
        self.trace['steps'][1]['state'] = None
        self.assertRejected('snapshot_missing')

    def test_missing_initial_row_is_rejected(self):
        self.trace['steps'] = self.trace['steps'][1:]
        self.assertRejected('initial_missing')

    def test_missing_terminal_label_is_rejected(self):
        del self.trace['meta']['winner']
        self.assertRejected('termination_missing')

    def test_wrong_winner_is_rejected_even_if_profile_reward_is_positive(self):
        self.trace['meta'].update(winner='fp', victory_type='ring_destroyed', outcome_fp=1.0)
        self.assertRejected('termination_mismatch')

    def test_nonfinite_state_is_rejected(self):
        self.trace['steps'][1]['state'].fellowship.corruption = float('nan')
        self.assertRejected('nonfinite')

    def test_nonfinite_regret_is_rejected(self):
        self.trace['steps'][1]['regret'] = float('inf')
        self.assertRejected('nonfinite')

    def test_finite_large_regret_is_not_a_quality_gate(self):
        self.trace['steps'][1]['regret'] = 100.0
        self.assertTrue(self.validate()['accepted'])

    def test_recovered_execution_annotation_is_rejected(self):
        self.trace['steps'].insert(1, state_row(
            self.trace['steps'][0]['state'], atype='exec_failure',
            params={'error': 'RuntimeError: failed to apply move'}))
        self.assertRejected('execution_recovered')

    def test_persisted_source_failure_is_rejected(self):
        self.trace['meta']['source_errors'].append(
            {'code': 'action_exec_failed', 'step': 1, 'detail': 'invalid move recovered'})
        self.assertRejected('source_action_exec_failed')

    def test_deliberate_zero_effect_card_telemetry_is_not_invalidity(self):
        self.trace['meta']['source_errors'].append(
            {'code': 'card_play_no_legal_choice', 'step': 1,
             'detail': 'allowed zero-effect play'})
        report = self.validate()
        self.assertTrue(report['accepted'], report)
        self.assertEqual(report['warnings'][0]['code'], 'source_card_play_no_legal_choice')

    def test_unknown_source_error_is_rejected_until_explicitly_classified(self):
        self.trace['meta']['source_errors'].append(
            {'code': 'future_execution_recovery', 'detail': 'unrecognized failure'})
        self.assertRejected('source_future_execution_recovery')

    def test_real_event_dryrun_failure_is_rejected(self):
        self.trace['meta']['source_errors'].append(
            {'code': 'event_dryrun_failed', 'detail': 'card handler raised'})
        self.assertRejected('source_event_dryrun_failed')

    def test_timeout_is_distinguished_from_invalid_data(self):
        self.trace['steps'][1]['state'].fellowship.corruption = 11
        self.trace['meta'].update(winner='timeout', victory_type='timeout')
        report = self.validate()
        self.assertTrue(report['valid'], report)
        self.assertFalse(report['accepted'], report)
        self.assertEqual([e['code'] for e in report['errors']], ['truncated'])
        self.assertTrue(self.validate(require_decided=False)['accepted'])

    def test_search_array_shape_and_chosen_index_are_validated(self):
        row = self.trace['steps'][1]
        row['search_stats'] = {'ae': np.zeros((2, 3), np.float32),
                               'ri': np.array([-1, -1]), 'p': np.array([.4, .6]),
                               'q': np.array([-.2, -.4]), 'n': np.array([2, 3]),
                               'played': 1}
        self.assertTrue(self.validate()['accepted'])
        row['search_stats']['played'] = 2
        self.assertRejected('search_schema')
        row['search_stats']['played'] = 1
        row['search_stats']['p'] = np.array([1.0])
        self.assertRejected('search_schema')

    def test_new_learner_requires_explicit_decision_context_contract(self):
        from selector_contract import current_identity, make_evidence
        self.assertRejected('decision_context_schema', decision_contract='predecision_winloss_v1')
        self.trace['meta']['reward_contract'] = 'predecision_winloss_v1'
        self.trace['steps'][1]['decision_context'] = {
            'contract': 'predecision_winloss_v1', 'encoding': np.array([1, 2], np.float32),
            'reward_stats': {'corruption': 11}, 'potential': 0.,
            'state': self.trace['steps'][0]['state']}
        self.trace['meta'].update(current_identity())
        ss = dict(ae=np.zeros((1, 3)), ri=np.zeros(1, dtype=int), p=np.ones(1),
                  q=np.zeros(1), n=np.ones(1), played=0)
        ss.update(make_evidence(ai.SELECT_CFG, 0., [1.], eligible=[True],
                                reward_contract='predecision_winloss_v1'))
        self.trace['steps'][1]['search_stats'] = ss
        self.assertTrue(self.validate(decision_contract='predecision_winloss_v1')['accepted'])
        self.trace['steps'][1]['decision_context']['potential'] = .1
        self.assertRejected('decision_context_schema', decision_contract='predecision_winloss_v1')
        self.assertRejected('decision_context_schema', decision_contract='future_contract')

    def test_context_without_exact_prestate_cannot_train_new_contract(self):
        self.trace['meta']['reward_contract'] = 'predecision_winloss_v1'
        self.trace['steps'][1]['decision_context'] = {
            'contract': 'predecision_winloss_v1', 'encoding': np.array([1, 2], np.float32),
            'reward_stats': {}, 'potential': 0.0}
        self.assertRejected('decision_context_schema', decision_contract='predecision_winloss_v1')

    def test_validation_does_not_mutate_profiles_state_or_arrays(self):
        initial_profiles = dict(ai.PLAYER_PROFILES)
        state = self.trace['steps'][1]['state']
        values = (state.fellowship.corruption, tuple(state.armies))
        self.assertTrue(self.validate()['accepted'])
        self.assertEqual(ai.PLAYER_PROFILES, initial_profiles)
        self.assertEqual((state.fellowship.corruption, tuple(state.armies)), values)


class RecorderIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='wotr_admission_')
        self.addCleanup(self.temp.cleanup)
        self.recorder = game_trace.TraceRecorder(str(Path(self.temp.name) / 'trace.gz'))
        self.trace = terminal_trace()
        self.before = self.trace['steps'][0]['state']
        self.after = self.trace['steps'][1]['state']
        self.recorder.record_initial(self.before)
        self.action = ai.Action(-1, 'pass', {})
        self.old_sink = len(state_invariants.VIOLATIONS)
        self.addCleanup(lambda: state_invariants.VIOLATIONS.__delitem__(slice(self.old_sink, None)))

    def report(self):
        self.recorder.save(winner='sh', victory_type='ring_corruption')
        return trace_admission.validate_trace(game_trace.load_trace(self.recorder.path))

    def test_actual_exception_recovery_is_persisted_after_active_tracer_clears(self):
        with patch.object(ai, '_execute_action', side_effect=RuntimeError('injected apply failure')):
            with self.assertRaises(RuntimeError) as caught:
                ai._traced_execute(self.before, self.action, self.recorder)
        self.assertIsNone(game_trace.ACTIVE_TRACER)
        with contextlib.redirect_stdout(io.StringIO()):
            ai._record_exec_failure(self.before, self.action, caught.exception)
        self.recorder.record_action(self.after, gs.Side.FREE_PEOPLES, self.action)
        report = self.report()
        self.assertFalse(report['accepted'], report)
        self.assertIn('source_action_exec_failed', [e['code'] for e in report['errors']])
        self.assertNotIn('exec_failure', [r.get('action_type') for r in self.recorder.steps])

    def test_search_failure_before_real_execution_does_not_taint_game(self):
        with contextlib.redirect_stdout(io.StringIO()):
            state_invariants.VIOLATIONS.append({'code': 'action_exec_failed', 'detail': 'simulated'})
        self.recorder.begin_action_buffer()
        self.recorder.record_action(self.after, gs.Side.FREE_PEOPLES, self.action)
        report = self.report()
        self.assertTrue(report['accepted'], report)

    def _rollover_report(self):
        self.recorder.record_new_turn(self.before)
        self.recorder.save(winner='timeout', victory_type='timeout')
        return trace_admission.validate_trace(
            game_trace.load_trace(self.recorder.path), require_decided=False)

    def test_real_rollover_cleanup_failure_is_rejected(self):
        game_logic.initialize_decks(self.before)
        self.recorder.record_action(self.before, gs.Side.FREE_PEOPLES, self.action)
        with patch.object(game_logic, 'cleanup_sieges',
                          side_effect=RuntimeError('injected siege cleanup failure')):
            ai._safe_next_turn(self.before)
        report = self._rollover_report()
        self.assertFalse(report['accepted'], report)
        self.assertIn('source_turn_cleanup_failed', [e['code'] for e in report['errors']])

    def test_nonempty_rollover_deck_failure_is_rejected(self):
        game_logic.initialize_decks(self.before)
        self.recorder.record_action(self.before, gs.Side.FREE_PEOPLES, self.action)
        with patch.object(game_logic, 'draw_event_card',
                          side_effect=RuntimeError('injected draw implementation failure')):
            ai._safe_next_turn(self.before)
        report = self._rollover_report()
        self.assertFalse(report['accepted'], report)
        self.assertIn('source_turn_draw_failed', [e['code'] for e in report['errors']])
        self.assertEqual(len(self.recorder.meta['source_errors']), 4)

    def test_draw_that_removes_last_card_then_fails_is_not_deck_exhaustion(self):
        game_logic.initialize_decks(self.before)
        self.before.fp_char_deck = self.before.fp_char_deck[:1]
        self.recorder.record_action(self.before, gs.Side.FREE_PEOPLES, self.action)
        original_draw = game_logic.draw_event_card

        def fail_after_pop(state, side, deck):
            if side is gs.Side.FREE_PEOPLES and deck == 'character':
                state.fp_char_deck.pop()
                raise RuntimeError('injected failure after removing last card')
            return original_draw(state, side, deck)

        with patch.object(game_logic, 'draw_event_card', side_effect=fail_after_pop):
            ai._safe_next_turn(self.before)
        report = self._rollover_report()
        self.assertFalse(report['accepted'], report)
        self.assertEqual([e['code'] for e in report['errors']], ['source_turn_draw_failed'])

    def test_exhausted_rollover_decks_remain_valid(self):
        game_logic.initialize_decks(self.before)
        for attr in ('fp_char_deck', 'fp_strat_deck', 'sh_char_deck', 'sh_strat_deck'):
            setattr(self.before, attr, [])
        self.recorder.record_action(self.before, gs.Side.FREE_PEOPLES, self.action)
        ai._safe_next_turn(self.before)
        report = self._rollover_report()
        self.assertTrue(report['accepted'], report)
        self.assertEqual(self.recorder.meta['source_errors'], [])

    def test_simulated_rollover_failure_does_not_taint_actual_play(self):
        simulation = ai.clone_state(self.before)
        game_logic.initialize_decks(simulation)
        with patch.object(game_logic, 'cleanup_sieges',
                          side_effect=RuntimeError('injected simulated cleanup failure')):
            ai._safe_next_turn(simulation)
        self.assertIn('turn_cleanup_failed',
                      [e['code'] for e in state_invariants.VIOLATIONS[self.old_sink:]])
        self.recorder.begin_action_buffer()
        self.recorder.record_action(self.after, gs.Side.FREE_PEOPLES, self.action)
        report = self.report()
        self.assertTrue(report['accepted'], report)
        self.assertEqual(self.recorder.meta['source_errors'], [])

    def test_initial_draw_exception_survives_later_recorder_creation(self):
        old_random, old_numpy = random.getstate(), np.random.get_state()
        self.addCleanup(random.setstate, old_random)
        self.addCleanup(np.random.set_state, old_numpy)
        random.seed(90904)
        np.random.seed(90904)
        original_draw = game_logic.draw_event_card
        calls = []

        def fail_first_draw(state, side, deck):
            calls.append((side, deck))
            if len(calls) == 1:
                raise RuntimeError('injected initial draw failure')
            return original_draw(state, side, deck)

        def uniform_priors(state, side, net, actions, features_vec=None):
            return [1.0 / len(actions)] * len(actions), 0.0

        trace_path = Path(self.temp.name) / 'initial_draw.gz'
        with patch.dict(os.environ, {'WOTR_REWARD_CONTRACT': 'predecision_winloss_v1'}), \
             patch.object(game_logic, 'draw_event_card', side_effect=fail_first_draw), \
             patch.object(ai, '_get_priors_and_value', side_effect=uniform_priors), \
             patch.object(ai, '_mcts_eval_state', return_value=0.0), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'injected initial draw failure'):
                ai._run_single_episode(None, None, {
                    'trace_path': str(trace_path), 'max_turns': 1, 'mcts_sims': 1,
                    'temperature_override': 0.0, 'add_noise': False})
        self.assertEqual(len(calls), 1)
        loaded = game_trace.load_trace(str(trace_path))
        self.assertEqual(loaded['meta']['status'], 'error')
        self.assertEqual(len(loaded['steps']), 1)
        report = trace_admission.validate_trace(loaded, require_decided=False)
        self.assertFalse(report['accepted'], report)
        self.assertIn('source_initial_draw_failed', [e['code'] for e in report['errors']])
        failure = next(e for e in loaded['meta']['source_errors']
                       if e['code'] == 'initial_draw_failed')
        self.assertEqual(failure['step'], 0)
        self.assertIn('injected initial draw failure', failure['detail'])

    def test_abandoned_annotation_buffer_keeps_schema_and_records_failure(self):
        self.recorder.begin_action_buffer()
        self.recorder._ann_buffer.append((None, SimpleNamespace(
            action_type='dice_roll', params={'faces': [6]}, die_idx=-1)))
        self.recorder.begin_action_buffer()
        self.recorder.record_action(self.after, gs.Side.FREE_PEOPLES, self.action)
        self.assertTrue(all(isinstance(s, dict) for s in self.recorder.steps))
        report = self.report()
        self.assertFalse(report['accepted'], report)
        self.assertIn('source_uncommitted_action', [e['code'] for e in report['errors']])

    def test_annotation_failure_becomes_rejectable_source_metadata(self):
        with patch.object(game_trace, 'ACTIVE_TRACER', self.recorder):
            with patch.object(self.recorder, 'record_action', side_effect=ValueError('bad annotation')):
                game_trace.emit_annotation(self.before, 'dice_roll', {'faces': [6]})
        self.recorder.record_action(self.after, gs.Side.FREE_PEOPLES, self.action)
        report = self.report()
        self.assertFalse(report['accepted'], report)
        self.assertIn('source_annotation_failed', [e['code'] for e in report['errors']])

    def test_later_phase_anchor_does_not_hide_uncommitted_player_action(self):
        self.recorder.begin_action_buffer()
        self.recorder.record_action(self.after, None, ai.Action(-1, 'turn_state', {}))
        self.recorder.record_action(self.after, gs.Side.FREE_PEOPLES, self.action)
        report = self.report()
        self.assertFalse(report['accepted'], report)
        self.assertIn('source_uncommitted_action', [e['code'] for e in report['errors']])

    def test_decision_context_is_copied_before_future_mutations(self):
        context = {'contract': 'predecision_v1', 'encoding': np.array([1, 2], np.float32),
                   'reward_stats': {'corruption': 11}, 'potential': .25}
        self.recorder.record_action(self.after, gs.Side.FREE_PEOPLES, self.action,
                                    decision_context=context)
        context['encoding'][0] = 999
        context['reward_stats']['corruption'] = 999
        saved = self.recorder.steps[-1]['decision_context']
        np.testing.assert_array_equal(saved['encoding'], [1, 2])
        self.assertEqual(saved['reward_stats']['corruption'], 11)

    def test_loading_reattaches_board_to_exact_predecision_snapshot(self):
        context = {'contract': 'predecision_winloss_v1',
                   'state': game_trace.snapshot_state(self.before)}
        self.recorder.record_action(self.after, gs.Side.FREE_PEOPLES, self.action,
                                    decision_context=context)
        self.recorder.save(winner='sh', victory_type='ring_corruption')
        loaded = game_trace.load_trace(self.recorder.path)
        self.assertIsNotNone(loaded['steps'][1]['decision_context']['state'].board)
        self.assertEqual(loaded['steps'][1]['decision_context']['state'].fellowship.corruption, 11)


class RuleAuditRegressionTests(unittest.TestCase):
    def test_eye_annotation_compares_against_pre_action_hidden_state(self):
        tr = terminal_trace()
        tr['steps'][0]['state'].fellowship.hidden = True
        tr['steps'][1]['state'].fellowship.hidden = False
        tr['steps'].append(state_row(tr['steps'][1]['state'], atype='hunt_tile_draw',
                                    params={'tile_name': 'Eye', 'ctx': "Isildur's Bane"}))
        violations = trace_audit.audit_loaded_trace(tr)
        self.assertTrue(any('[A2]' in v for v in violations), violations)

    def test_shadow_card_ownership_uses_recorded_player_side(self):
        tr = terminal_trace()
        card_id = 'SHADOW_Character_The_Palantir_of_Orthanc'
        tr['steps'][0]['state'].hands[gs.Side.SHADOW].hand.append(card_id)
        tr['steps'][1].update(side='SH', action_type='play_event', params={'card_id': card_id})
        violations = trace_audit.audit_loaded_trace(tr)
        self.assertFalse(any('[A3]' in v for v in violations), violations)

    def test_existing_real_game_passes_and_corrupted_copy_is_rejected(self):
        path = ROOT / 'data/v42sx_selfplay_500/traces/w1_g0_sh_military.trace.gz'
        if not path.exists():
            self.skipTest('optional archived game is not distributed with this checkout')
        tr = game_trace.load_trace(str(path))
        report = trace_admission.validate_trace(tr)
        self.assertTrue(report['accepted'], report)
        self.assertIn('source_integrity_unrecorded', [w['code'] for w in report['warnings']])
        tr['steps'][-1]['state'].fellowship.corruption = float('nan')
        self.assertFalse(trace_admission.validate_trace(tr)['accepted'])


if __name__ == '__main__':
    unittest.main()
