"""Literal consumer controls for Gandalf's delayed optional guide draw.

These tests do not use the encoder, legal menu, or action executor as an oracle.
The admission fixture isolates end-of-turn obligations from tactical play.
"""
import copy
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import ai
import board
import game_state as gs
import game_trace
import run_td_v33loop as trainer
import td_attribution as attribution
import trace_admission as admission
from reward_contract import WINLOSS


if os.environ.get('WOTR_AUDIT_GANDALF_CONSUMER_BASELINE'):
    folder = Path(os.environ['WOTR_AUDIT_GANDALF_CONSUMER_BASELINE'])
    for variable, filename in (('admission', 'trace_admission'),
                               ('attribution', 'td_attribution')):
        spec = importlib.util.spec_from_file_location(
            'gandalf_consumer_baseline_' + filename, folder / (filename + '.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        globals()[variable] = module


def military_trace():
    initial = gs.setup_initial_game(board.load_board())
    final = copy.deepcopy(initial)
    # Two Shadow strongholds are four FP victory points (RB44).
    final.enemy_controlled[gs.Side.FREE_PEOPLES] = {'orthanc', 'moria'}
    return {'meta': {'winner': 'fp', 'victory_type': 'military',
                     'source_integrity_version': 1, 'source_errors': []},
            'steps': [dict(kind='initial', side=None, action_type=None,
                           params={}, state=initial, turn=1),
                      dict(kind='action', side='FP', action_type='pass',
                           params={}, state=final, turn=1)]}


class GandalfAdmissionTests(unittest.TestCase):
    def test_completed_military_result_without_draw_debt_is_admitted(self):
        trace = military_trace()
        trace['steps'][-1]['state']._gandalf_draw_pending = None
        report = admission.validate_trace(trace, audit_rules=False)
        self.assertTrue(report['accepted'], report)

    def test_military_result_with_queued_optional_draw_is_not_finished(self):
        trace = military_trace()
        trace['steps'][-1]['state']._gandalf_draw_pending = {
            'source_card_id': 'FREE_PEOPLES_Character_Athelas',
            'deck': 'character', 'die_face': 'EVENT'}
        report = admission.validate_trace(trace, audit_rules=False)
        self.assertFalse(report['accepted'], report)
        self.assertEqual([e['code'] for e in report['errors']],
                         ['unfinished_terminal_decision', 'unfinished_terminal_turn'])
        for error in report['errors']:
            self.assertIn('_gandalf_draw_pending', error['detail'])

    def test_immediate_ring_victory_does_not_wait_for_optional_draw(self):
        trace = military_trace()
        final = trace['steps'][-1]['state']
        final.enemy_controlled[gs.Side.FREE_PEOPLES] = set()
        final.fellowship.corruption = 12
        final._gandalf_draw_pending = {'source_card_id': 'literal interrupted event',
                                       'deck': 'strategy', 'die_face': 'EVENT'}
        trace['meta'].update(winner='sh', victory_type='ring_corruption')
        report = admission.validate_trace(trace, audit_rules=False)
        self.assertTrue(report['accepted'], report)


class GandalfAttributionTests(unittest.TestCase):
    def label(self, params, action_type='card_option'):
        # A display/credit fixture, not an asserted legal game: independent
        # zero-change board metrics and recorded value eliminate net loading.
        snapshot = SimpleNamespace(fellowship=SimpleNamespace(corruption=0, progress=0))
        row = dict(kind='action', side='FP', action_type=action_type,
                   params=params, state=snapshot, turn=1,
                   decision_context=dict(contract=WINLOSS,
                       encoding=np.array([.25, -.5], np.float32),
                       potential=0.0, state=snapshot, value=.2))
        trace = dict(meta=dict(reward_contract=WINLOSS, winner='fp',
                               victory_type='ring_destroyed'), steps=[row])
        with patch.object(game_trace, 'load_trace', return_value=trace), \
                patch.object(trainer, '_apply_loop_config'), \
                patch('game_logic.calculate_victory_points',
                      return_value={'free_peoples': 0, 'shadow': 0}), \
                patch.object(ai, 'load_value_net', side_effect=AssertionError('no model read')):
            result = attribution.compute_attribution('unwritten_gandalf.trace.gz')
        step = result['steps'][0]
        self.assertEqual(step['reward'], 0.0)
        self.assertAlmostEqual(step['delta'], .8)
        return step['label']

    def test_take_decline_and_fixed_deck_are_visible_in_actual_report(self):
        for take, deck, expected in (
                (True, 'character', 'Gandalf draw: take (character)'),
                (False, 'character', 'Gandalf draw: decline (character)'),
                (True, 'strategy', 'Gandalf draw: take (strategy)'),
                (False, 'strategy', 'Gandalf draw: decline (strategy)')):
            with self.subTest(take=take, deck=deck):
                self.assertEqual(self.label(dict(card_id='__gandalf_draw__',
                    deck=deck, choices={'gandalf_draw': take})), expected)

    def test_missing_or_malformed_choice_is_not_reported_as_decline(self):
        for choice in ({}, {'gandalf_draw': 0}, {'gandalf_draw': 'false'}):
            with self.subTest(choice=choice):
                self.assertEqual(self.label(dict(card_id='__gandalf_draw__',
                    choices=choice)), 'Gandalf draw: unknown choice (unknown deck)')

    def test_unrelated_card_option_keeps_its_existing_label(self):
        self.assertEqual(self.label(dict(card_id='__palantir_draw__',
            choices={'palantir_deck': 'strategy'})), 'card_option __palantir_draw__')


if __name__ == '__main__':
    unittest.main()
