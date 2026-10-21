"""Independent native Undo witnesses for deferred and held, uncommitted work."""
from pathlib import Path
import pickle
import tempfile
import unittest
from types import SimpleNamespace

import ai
import game_state as gs
import test_live_driver as td
from wotr_bridge.live.driver import LiveDriver
from wotr_bridge.live.randomness import ClientRandomness

ROOT = Path(__file__).resolve().parents[1]


class RetainedUndoAdversarial(unittest.TestCase):
    def rig(self, state=None, ai_side=gs.Side.SHADOW):
        folder = tempfile.TemporaryDirectory(prefix='retained_undo_', dir=ROOT)
        self.addCleanup(folder.cleanup)
        path = Path(folder.name)/'feed.txt'
        path.write_text('')
        rand = ClientRandomness(seed=7)
        rand.install()
        driver = LiveDriver(state or td._hunt_armed_state(), ai_side, str(path), randomness=rand)
        driver.phase = 'actions'
        def feed(*lines):
            with path.open('a', encoding='utf8') as stream:
                stream.write('\n'.join(lines)+'\n')
            return driver.tick()
        return SimpleNamespace(driver=driver, rand=rand, feed=feed)

    def undo(self, rig, line):
        return rig.feed('$Wey undoes last action : ' + line)

    def test_undo_retracts_record_queued_after_a_randomness_deferral(self):
        rig = self.rig()
        rig.feed(*td._FS_WINDOW)
        driver = rig.driver
        self.assertIsNotNone(driver.deferred)
        line = '$Wey flips The Fellowship in Rivendell.'
        rig.feed(line)
        self.assertEqual(len(driver.deferred_tail), 1)
        before = pickle.dumps(driver.es)
        self.undo(rig, line)
        self.assertEqual(driver.deferred_tail, [], driver.alarms)
        self.assertEqual(pickle.dumps(driver.es), before)
        self.assertIsNotNone(driver.deferred)

    def test_late_roll_cannot_commit_a_window_whose_die_closure_was_undone(self):
        rig = self.rig()
        rig.feed(*td._FS_WINDOW)
        driver = rig.driver
        self.assertIsNotNone(driver.deferred)
        self.undo(rig, td._FS_WINDOW[-1])
        self.assertEqual(driver.es.fellowship.progress, 0)
        rig.feed('<game> Wey rolls 1')
        self.assertEqual(driver.es.fellowship.progress, 0,
                         (driver.window_stats, driver.alarms, driver.findings[-5:]))
        self.assertEqual(driver.n_commits_total, 0)
        self.assertFalse(driver.es.dice[gs.Side.FREE_PEOPLES][0].used)

    def test_fully_undone_held_window_is_not_retried_when_pending_clears(self):
        state = td.adopted_state()
        action, lines = td._army_window_lines(state)
        self.assertIsNotNone(action)
        state.dice[gs.Side.FREE_PEOPLES][0].current = gs.ActionDieFace.ARMY
        first_leg = next(a for a in ai.get_legal_actions(state, gs.Side.FREE_PEOPLES)
                         if a.action_type == 'army_move')
        state = ai.apply_action(state, first_leg)
        self.assertIsNotNone(state.pending_army_move_die)
        rig = self.rig(state, gs.Side.FREE_PEOPLES)
        driver = rig.driver
        driver.last_actor = gs.Side.FREE_PEOPLES
        rig.feed(*lines)
        self.assertEqual(len(driver.holds), 1, (driver.findings, driver.alarms))
        self.assertEqual(driver.n_commits_total, 0)
        for line in reversed(lines):
            self.undo(rig, line)
        self.assertEqual(driver.holds, [], driver.alarms)
        state.pending_army_move_die = None  # The AI later completes its die.
        driver._retry_holds('choice settled')
        self.assertEqual(driver.n_commits_total, 0)

    def test_control_complete_deferral_abort_remains_uncommitted(self):
        rig = self.rig()
        before = pickle.dumps(rig.driver.es)
        rig.feed(*td._FS_WINDOW)
        self.assertIsNotNone(rig.driver.deferred)
        for line in reversed(td._FS_WINDOW):
            self.undo(rig, line)
        self.assertIsNone(rig.driver.deferred)
        self.assertEqual(pickle.dumps(rig.driver.es), before)
        self.assertEqual(rig.driver.n_commits_total, 0)

    def test_reclose_uses_late_native_roll_for_all_fellowship_orders(self):
        direct = ('$Wey moves Free Action Die (C) from FP Dice to The Hunt '
                  'For The Ring.')
        for lines in (td._FS_WINDOW, [direct, td._FS_WINDOW[1]],
                      [td._FS_WINDOW[1], direct]):
            with self.subTest(lines=lines):
                rig = self.rig()
                rig.feed(*lines)
                self.assertIsNotNone(rig.driver.deferred)
                self.undo(rig, lines[-1])
                self.assertIsNone(rig.driver.deferred)
                self.assertIsNone(rig.driver.pending_random_call())
                self.assertIsNotNone(rig.driver.seg.window)
                rig.feed('<game> Wey rolls 1')
                self.assertEqual(rig.driver.n_commits_total, 0)
                rig.feed(lines[-1])
                self.assertEqual(rig.driver.es.fellowship.progress, 1,
                                 (rig.driver.alarms, rig.driver.findings[-5:]))
                self.assertEqual(rig.driver.n_commits_total, 1)
                self.assertIsNone(rig.driver.deferred)
                self.assertEqual(rig.rand.unconsumed()['d6'], 0)

    def test_reclose_does_not_refeed_roll_already_in_the_old_window(self):
        rig = self.rig()
        rig.feed(*td._FS_WINDOW[:-1], '<game> Wey rolls 6', td._FS_WINDOW[-1])
        self.assertIsNotNone(rig.driver.deferred)
        self.assertEqual(rig.driver.deferred[0]._starve_call.kind, 'tile')
        bank = list(rig.rand._feed['d6'])
        self.assertEqual(len(bank), 1)
        self.undo(rig, td._FS_WINDOW[-1])
        rig.feed(td._FS_WINDOW[-1])
        self.assertIsNotNone(rig.driver.deferred)
        self.assertEqual(rig.driver.deferred[0]._starve_call.kind, 'tile')
        self.assertEqual(rig.rand._feed['d6'], bank)

    def test_full_abort_preserves_banked_and_consumed_native_randomness(self):
        rig = self.rig()
        rig.feed(*td._FS_WINDOW)
        rig.rand.feed('d6', [2, 5])
        self.assertEqual(rig.rand._consume_d6(1), [2])
        before = pickle.dumps((rig.rand._feed, rig.rand._feed_pos))
        for line in reversed(td._FS_WINDOW):
            self.undo(rig, line)
        self.assertEqual(pickle.dumps((rig.rand._feed, rig.rand._feed_pos)), before)
        self.assertEqual(rig.driver.n_commits_total, 0)

    def test_partial_held_undo_requires_a_new_completion_after_gate_clears(self):
        state = td.adopted_state()
        _action, lines = td._army_window_lines(state)
        state.dice[gs.Side.FREE_PEOPLES][0].current = gs.ActionDieFace.ARMY
        first_leg = next(a for a in ai.get_legal_actions(state, gs.Side.FREE_PEOPLES)
                         if a.action_type == 'army_move')
        state = ai.apply_action(state, first_leg)
        rig = self.rig(state, gs.Side.FREE_PEOPLES)
        rig.driver.last_actor = gs.Side.FREE_PEOPLES
        rig.feed(*lines)
        self.assertEqual(len(rig.driver.holds), 1)
        self.undo(rig, lines[-1])
        state.pending_army_move_die = None
        rig.driver._retry_holds('choice settled')
        self.assertEqual(rig.driver.n_commits_total, 0)
        self.assertEqual(rig.driver.holds, [])
        rig.feed(lines[-1])
        self.assertEqual(rig.driver.n_commits_total, 1,
                         (rig.driver.alarms, rig.driver.findings[-5:]))

    def test_non_tail_undo_does_not_retract_a_newer_queued_gesture(self):
        rig = self.rig()
        rig.feed(*td._FS_WINDOW)
        line = '$Wey flips The Fellowship in Rivendell.'
        rig.feed(line)
        window = rig.driver.deferred[0]
        records = list(window.records)
        self.undo(rig, td._FS_WINDOW[-1])
        self.assertIs(rig.driver.deferred[0], window)
        self.assertEqual(window.records, records)
        self.assertEqual(len(rig.driver.deferred_tail), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
