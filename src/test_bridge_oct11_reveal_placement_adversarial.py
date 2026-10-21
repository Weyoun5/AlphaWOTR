"""Lifecycle checks against the actual held native reveal checkpoint."""
from pathlib import Path
import pickle
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'wotr_bridge'), str(ROOT)]
import ai
import emitter
import game_logic as gl
import test_bridge_oct11_native_reveal_placement as fixture


class RevealLifecycle(unittest.TestCase):
    rig = fixture.NativeRevealPlacementTests.rig
    route = fixture.NativeRevealPlacementTests.route
    alter_piece = fixture.NativeRevealPlacementTests.alter_piece

    def test_native_undo_destination_then_alternate_destination_reuses_same_reset(self):
        driver, feed, dump = self.rig()
        self.route(driver, feed, fixture.FLIP, fixture.MOVE)
        self.assertEqual(driver.n_commits_total, 1)
        self.assertIsNone(ai.describe_pending(driver.es))
        # Undo only the human pawn move. The older AI reveal/reset remains
        # physically on the native board and is not an undone game choice.
        self.alter_piece(dump, '247', zone='<~Game.1911~>', marker=1)
        self.route(driver, feed, '$Wey undoes last action : '+fixture.MOVE.lstrip('$'))
        self.assertEqual(ai.describe_pending(driver.es)['kind'], 'reveal_place')
        self.assertEqual(driver.es.fellowship.progress, 1)
        self.assertEqual(gl.fellowship_region(driver.es), 'rivendell')
        self.assertEqual(driver.undo_epoch, 1)
        self.alter_piece(dump, '247', zone='<~'+emitter.zone_code_for('trollshaws')+'~>', marker=1)
        self.route(driver, feed, fixture.MOVE.replace('Fords of Bruinen', 'Trollshaws'))
        self.assertEqual(gl.fellowship_region(driver.es), 'trollshaws', driver.stall_description())
        self.assertEqual(driver.es.fellowship.progress, 0)
        self.assertIsNone(ai.describe_pending(driver.es))

    def test_native_undo_of_reveal_invalidates_echo_even_while_dump_lags(self):
        driver, feed, _dump = self.rig()
        self.route(driver, feed, fixture.FLIP)
        # Native feed events may precede the next asynchronous dump write.
        # The inverse of the exact own reveal says its reset no longer stands;
        # the older dump still shows the earlier Track0/revealed snapshot.
        self.route(driver, feed, '$Wey undoes last action : '+fixture.FLIP.lstrip('$'))
        before = pickle.dumps(driver.es)
        self.route(driver, feed, fixture.MOVE)
        self.assertTrue(pickle.dumps(driver.es) == before,
                         (driver.n_commits_total, driver.alarms))
        self.assertEqual(driver.n_commits_total, 0)
        self.assertIsNotNone(driver.pending_buf)

    def test_late_native_reset_dump_allows_retained_destination_on_next_tick(self):
        driver, feed, dump = self.rig()
        self.alter_piece(dump, '248', zone='<~Game.2015~>')
        self.route(driver, feed, fixture.FLIP, fixture.MOVE)
        self.assertEqual(driver.n_commits_total, 0)
        self.assertIsNotNone(driver.pending_buf)
        self.alter_piece(dump, '248', zone='<~Game.2014~>')
        # Model a new exporter snapshot, not an in-place edit to one version.
        dump.write_text(re.sub(r'^#SEQ dump=(\d+)',
                    lambda m: '#SEQ dump='+str(int(m.group(1))+1), dump.read_text()), encoding='utf8')
        for _ in range(5):
            driver.tick()
        self.assertEqual(gl.fellowship_region(driver.es), 'fords of bruinen')
        self.assertEqual(driver.n_commits_total, 1)
        self.assertIsNone(driver.pending_buf)

    def test_destination_undo_does_not_accept_physically_unrevealed_replacement(self):
        driver, feed, dump = self.rig()
        self.route(driver, feed, fixture.FLIP, fixture.MOVE)
        self.route(driver, feed, '$Wey undoes last action : '+fixture.MOVE.lstrip('$'))
        self.alter_piece(dump, '247', zone='<~Game.1911~>', marker=0)
        before = pickle.dumps(driver.es)
        self.route(driver, feed, fixture.MOVE.replace('Fords of Bruinen', 'Trollshaws'))
        self.assertTrue(pickle.dumps(driver.es) == before)

    def test_undo_of_uncommitted_wrong_destination_retains_earlier_native_reset(self):
        driver, feed, _dump = self.rig()
        wrong = fixture.MOVE.replace('Fords of Bruinen', 'Moria')
        self.route(driver, feed, fixture.FLIP, wrong)
        self.assertEqual(driver.n_commits_total, 0)
        self.assertIsNotNone(driver.pending_buf)
        self.route(driver, feed, '$Wey undoes last action : '+wrong.lstrip('$'))
        self.assertIsNone(driver.pending_buf)
        self.route(driver, feed, fixture.MOVE)
        self.assertEqual(gl.fellowship_region(driver.es), 'fords of bruinen')
        self.assertEqual(driver.n_commits_total, 1)


if __name__ == '__main__':
    unittest.main()
