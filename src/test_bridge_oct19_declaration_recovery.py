"""Declaration completion/recovery against complete native board files."""
from pathlib import Path
import os
import pickle
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'wotr_bridge'), str(ROOT)]
os.environ.setdefault('WOTR_NATIVE', '0')
import ai
import board
import game_logic as gl
import test_bridge_oct11_combat_recovery as fixture
from live import choice_recovery
from live.driver import PhaseWindow


class DeclarationRecovery(unittest.TestCase):
    setUp = fixture.LiveNoncombatRecovery.setUp
    write_dump = fixture.LiveCombatRecovery.write_dump
    companion_pid = fixture.LiveNoncombatRecovery.companion_pid

    def prepare(self, *, completed=False):
        state = self.driver.es
        state.turn = 3
        state.fellowship.companions = [c for c in state.fellowship.companions
                                      if c.name != 'Gandalf the Grey']
        gl.change_guide(state, chosen='Strider')
        self.dump[self.companion_pid('Gandalf the Grey')] = 'FP Dead'
        self.dump[self.companion_pid('Strider')] = 'Guide Box'
        state.fellowship.hidden = True
        state.fellowship.region = state.board.region('rivendell')
        state.fellowship.progress = 4
        self.dump['247'] = 'Dimrill Dale'
        self.dump['248'] = 'Track 0'
        self.marker_states = {'247': 0}
        self.driver.phase = 'declare'
        self.driver._ensure_declare_armed()
        if completed:
            action = next(a for a in ai.describe_pending(state)['actions']
                          if a.action_type == 'fellowship_declare'
                          and a.params['region'] == 'dimrill dale')
            self.driver.es = ai.apply_action(state, action)
            gl.begin_fellowship_tail(self.driver.es)
            self.driver.phase = 'guide'
        self.write_dump()

    def assert_declared(self):
        state = self.driver.es
        self.assertEqual(board.sanitize(str(gl.fellowship_region(state))), 'dimrill dale')
        self.assertEqual(state.fellowship.progress, 0)
        self.assertTrue(state.fellowship.hidden)
        self.assertEqual(state.fellowship.guide.name, 'Strider')
        self.assertEqual(ai.describe_pending(state)['kind'], 'eye')
        self.assertIsNone(state.pending_guide_change)

    def test_resync_already_accepted_declaration_preserves_next_eye_decision(self):
        self.prepare(completed=True)
        def phase_state():
            state = self.driver.es
            return pickle.dumps((state.fellowship, state.turn, state._eye_allocation_phase,
                                 state.pending_hunt, state.pending_guide_change,
                                 state.eyes_allocated, state.dice, state.hunt_box))
        before = phase_state()
        _, report = self.driver.reset_and_resync()
        self.assertTrue(self.driver._last_resync_adoption['complete'], report)
        self.assert_declared()
        self.assertEqual(self.driver.phase, 'guide')
        self.assertEqual(self.driver.n_commits_total, 0)
        self.assertEqual(self.driver.es.eyes_allocated, 0)
        # Resync may rename opaque human-hand placeholders; no phase, die,
        # Fellowship choice or visible game result may change here.
        self.assertEqual(phase_state(), before)
        self.driver._on_chat({'body': 'fp ok', 'nick': 'Wey'})
        self.assertEqual(self.driver.phase, 'eyes')
        self.assert_declared()

    def test_resync_recovers_missing_declaration_once_from_native_result(self):
        self.prepare()
        _, report = self.driver.reset_and_resync()
        self.assert_declared()
        self.assertEqual(self.driver.n_commits_total, 1, report)
        self.assertEqual(self.driver.commits[-1][2], 'pending:declare')
        self.assertEqual(self.driver.phase, 'guide')
        self.assertFalse(getattr(self.driver, '_track_owed', {}).get('248'))
        self.driver.reset_and_resync()
        self.assertEqual(self.driver.n_commits_total, 1)
        self.driver._on_chat({'body': 'fp ok', 'nick': 'Wey'})
        self.assertEqual(self.driver.phase, 'eyes')

    def test_fp_ok_uses_final_board_not_duplicate_resets(self):
        self.prepare()
        self.driver.pending_buf = PhaseWindow()
        self.driver.pending_buf.records = [
            dict(op='move', obj='The Fellowship', **{'from':'Rivendell', 'to':'Dimrill Dale'}),
            dict(op='flip', obj='The Fellowship', region='Dimrill Dale',
                 zone='Dimrill Dale', revealed=True, prev_track=4),
            dict(op='flip', obj='The Fellowship', zone='Dimrill Dale'),
            dict(op='move', obj='FSP', **{'from':'Fellowship Track 4', 'to':'Fellowship Track 0'}),
        ]
        out = self.driver._on_chat({'body': 'fp ok', 'nick': 'Wey'})
        self.assert_declared()
        self.assertEqual(self.driver.phase, 'eyes', out)
        self.assertEqual(self.driver.n_commits_total, 1)
        self.assertIsNone(self.driver.pending_buf)
        self.assertFalse(any(x[0:2] == ('chat','refusal') for x in out), out)

    def test_fp_ok_recovers_declaration_when_gesture_lines_are_missing(self):
        self.prepare()
        self.assertIsNone(self.driver.pending_buf)
        out = self.driver._on_chat({'body': 'fp ok', 'nick': 'Wey'})
        self.assert_declared()
        self.assertEqual(self.driver.phase, 'eyes', out)
        self.assertEqual(self.driver.n_commits_total, 1)
        self.assertEqual(self.driver.commits[-1][2], 'pending:declare')

    def test_fp_ok_cannot_decline_unseen_invalid_or_incomplete_board(self):
        self.prepare()
        for destination, progress, hidden in (('Far Harad','Track 0',0),
                ('Dimrill Dale','Track 4',0), ('Dimrill Dale','Track 0',1)):
            with self.subTest(destination=destination, progress=progress, hidden=hidden):
                self.dump['247'] = destination; self.dump['248'] = progress
                self.marker_states['247'] = hidden; self.write_dump()
                before = pickle.dumps(self.driver.es)
                out = self.driver._on_chat({'body': 'fp ok', 'nick': 'Wey'})
                self.assertEqual(self.driver.phase, 'declare', out)
                self.assertEqual(self.driver.n_commits_total, 0)
                self.assertEqual(pickle.dumps(self.driver.es), before)
                self.assertTrue(any(x[0:2] == ('chat','refusal') for x in out), out)

    def test_fp_ok_declines_measured_unchanged_native_board(self):
        self.prepare()
        self.dump['247'] = 'Rivendell'; self.dump['248'] = 'Track 4'; self.write_dump()
        out = self.driver._on_chat({'body': 'fp ok', 'nick': 'Wey'})
        self.assertEqual(self.driver.phase, 'eyes', out)
        self.assertEqual(self.driver.es.fellowship.progress, 4)
        self.assertEqual(self.driver.commits[-1][2], 'declare-decline')

    def test_fp_ok_waits_for_unflip_after_engine_declaration_commit(self):
        self.prepare(completed=True)
        self.marker_states['247'] = 1; self.write_dump()
        out = self.driver._on_chat({'body': 'fp ok', 'nick': 'Wey'})
        self.assertEqual(self.driver.phase, 'guide', out)
        self.assertEqual(self.driver.n_commits_total, 0)
        self.assertEqual(self.driver.es.eyes_allocated, 0)
        self.assertTrue(any(x[0:2] == ('chat','refusal') for x in out), out)
        self.marker_states['247'] = 0; self.write_dump()
        self.driver._on_chat({'body': 'fp ok', 'nick': 'Wey'})
        self.assertEqual(self.driver.phase, 'eyes')

    def test_incomplete_or_illegal_native_result_cannot_erase_pending(self):
        self.prepare()
        good = dict(self.dump)
        cases = [('unreachable', '247', 'Far Harad'),
                 ('progress not reset', '248', 'Track 4'),
                 ('unknown progress', '248', 'Spare'),
                 ('unknown corruption', '252', 'Spare'),
                 ('wrong guide', self.companion_pid('Strider'), 'Fellowship Box'),
                 ('companion gone', self.companion_pid('Gimli'), 'FP Dead')]
        for label, pid, target in cases:
            with self.subTest(label=label):
                self.dump = dict(good); self.dump[pid] = target
                before = pickle.dumps(self.driver.es)
                rng = pickle.dumps(vars(self.rand))
                result = choice_recovery.plan(self.driver, self.dump, self.marker_states)
                self.assertEqual(result.status, 'hold', result.note)
                self.assertEqual(pickle.dumps(self.driver.es), before)
                self.assertEqual(pickle.dumps(vars(self.rand)), rng)
        self.dump = good
        for markers in ({}, {'247': 1}):
            self.assertEqual(choice_recovery.plan(self.driver, good, markers).status, 'hold')
        missing = dict(good); del missing[self.companion_pid('Gimli')]
        self.assertEqual(choice_recovery.plan(self.driver, missing, self.marker_states).status, 'hold')

    def test_unrelated_pending_evidence_is_not_replaced_by_declaration(self):
        self.prepare()
        self.driver.pending_buf = PhaseWindow()
        self.driver.pending_buf.records = [dict(op='kill', pieces=['Gimli'], zone='Fellowship Box')]
        before = pickle.dumps(self.driver.es)
        result = choice_recovery.plan(self.driver, self.dump, self.marker_states)
        self.assertEqual(result.status, 'hold')
        self.assertEqual(pickle.dumps(self.driver.es), before)
        _, message = self.driver.skip_open_decision()
        self.assertIn('cannot be declined', message)
        self.assertEqual(self.driver.n_commits_total, 0)


if __name__ == '__main__':
    unittest.main()
