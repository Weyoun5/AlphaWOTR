"""Native Undo must retract uncommitted gestures without undoing the battle."""
from pathlib import Path
import pickle
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT), str(ROOT / 'wotr_bridge')]

import ai
import game_logic
import game_state as gs
from test_bridge_oct11_combat_recovery import state_and_driver
from test_live_driver import Rig


class PendingUndo(unittest.TestCase):
    def rig(self, hits=1):
        state, _, _ = state_and_driver(hits=hits)
        game_logic.initialize_decks(state)
        rig = Rig(state, gs.Side.SHADOW, phase='actions')
        target = Path(rig.td).resolve()
        self.assertEqual(target.parent, Path(tempfile.gettempdir()).resolve())
        self.assertTrue(target.name.startswith('tmp'))
        self.addCleanup(shutil.rmtree, target)
        rig.drv._snapshot([], 'chat', post=state)
        rig.drv.last_actor = gs.Side.SHADOW
        return rig

    def test_bad_piece_move_undo_leaves_battle_and_next_casualty_usable(self):
        rig = self.rig()
        before = pickle.dumps(rig.drv.es)
        commits = rig.drv.n_commits_total
        bad = '$Wey moves Nazgul from Orthanc to FP Reinforcements.'
        rig.feed(bad)
        self.assertEqual(len(rig.drv.pending_buf.records), 1)
        alarms = list(rig.drv.alarms)
        rig.feed('$Wey undoes last action : ' + bad.lstrip('$'))
        self.assertIsNone(rig.drv.pending_buf)
        self.assertEqual(rig.drv.alarms, alarms)
        self.assertEqual(pickle.dumps(rig.drv.es), before)
        self.assertEqual(rig.drv.n_commits_total, commits)
        rig.feed('$Wey kills Rohan Regular in Fords of Isen.')
        self.assertEqual(len(rig.drv.es.armies['fords of isen'].units), 2)
        self.assertEqual(rig.drv.es.fp_casualties[gs.Nation.ROHAN]['regular'], 1)
        self.assertEqual(rig.drv.n_commits_total, commits + 1)

    def test_tail_undo_keeps_earlier_uncommitted_gesture_and_existing_roll(self):
        rig = self.rig()
        first = '$Wey moves Nazgul from Orthanc to FP Reinforcements.'
        second = '$Wey moves Nazgul from Morannon to FP Reinforcements.'
        rig.feed(first)
        rig.feed(second)
        self.assertEqual(len(rig.drv.pending_buf.records), 2)
        before = pickle.dumps(rig.drv.es)
        rig.feed('$Wey undoes last action : ' + second.lstrip('$'))
        self.assertEqual(len(rig.drv.pending_buf.records), 1)
        self.assertEqual(pickle.dumps(rig.drv.es), before)
        self.assertEqual(rig.drv.es.pending_combat['_recovery_test_rolls'], (6, 4, 1))

    def test_undo_other_line_does_not_drop_the_current_pending_tail(self):
        rig = self.rig()
        rig.feed('$Wey moves Nazgul from Orthanc to FP Reinforcements.')
        records = list(rig.drv.pending_buf.records)
        before = pickle.dumps(rig.drv.es)
        rig.feed('$Wey undoes last action : Wey moves Nazgul from Orthanc to Morannon.')
        self.assertEqual(rig.drv.pending_buf.records, records)
        self.assertEqual(pickle.dumps(rig.drv.es), before)


if __name__ == '__main__':
    unittest.main()
