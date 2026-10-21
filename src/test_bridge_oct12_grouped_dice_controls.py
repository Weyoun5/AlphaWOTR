"""Boundary controls for authenticated multi-die selection, not gameplay."""
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_bridge_oct12_staged_dice_adversarial as actual
from wotr_bridge.live.gesture_match import FACE_LETTER


class GroupedDiceControls(unittest.TestCase):
    setUp = actual.StagedDicePublicEvidence.setUp
    physical = actual.StagedDicePublicEvidence.physical
    start_group = actual.StagedDicePublicEvidence.start_group

    def test_native_undo_redo_return_reopens_exact_remaining_die(self):
        self.start_group()
        self.physical(actual.RETURN_THREE, ['210', '212', '213'])
        self.physical('$Wey undoes last action : ' + actual.RETURN_THREE, [])
        self.assertIsNone(self.driver.seg.window)
        self.physical('$Wey redoes last action : ' + actual.RETURN_THREE, [])
        self.assertEqual(self.driver.seg.window._staged_die_pid, '211')
        self.assertEqual(self.driver.seg.window.die_face, 'W')
        self.physical(actual.gandalf.ARRIVE, ['240'])
        self.physical(actual.gandalf.FLIP, ['240'])
        self.physical(actual.gandalf.BONUS, ['358'])
        self.physical(actual.gandalf.CLOSE, ['211', '358'])
        self.assertEqual(self.driver.n_commits_total, 1)
        self.assertEqual([die.used for die in self.driver.es.dice[self.driver.human_side]],
                         [False, True, False, False])
        self.assertFalse(self.driver.alarms)

    def test_no_action_window_while_two_dice_remain(self):
        self.start_group()
        self.physical('$Wey moves group (Free Action Die (P), Free Action Die (C)) '
                      'from Current Action to FP Dice.', ['210', '213'])
        self.assertIsNone(self.driver.seg.window)
        self.assertEqual(len(self.driver._grouped_die_selection['active']), 2)
        self.assertEqual(self.driver.n_commits_total, 0)

    def test_equal_faces_preserve_the_exact_remaining_native_identity(self):
        side = self.driver.human_side
        self.driver.es.dice[side][0].current = actual.gandalf.gs.ActionDieFace.CHARACTER
        self.physical('$Wey moves group ([2]Free Action Die (C)) '
                      'from FP Dice to Current Action.', ['210', '213'])
        self.physical('$Wey moves Free Action Die (C) from Current Action to FP Dice.', ['210'])
        window = self.driver.seg.window
        self.assertIsNotNone(window)
        self.assertEqual(window._staged_die_pid, '213')
        self.assertEqual(self.driver._staged_die_idx(window), (side, 3))
        self.assertEqual(self.driver.n_commits_total, 0)

    def test_unknown_group_identity_cannot_select_a_payment(self):
        self.physical(actual.STAGE_GROUP, ['210', '211', '212', '999'])
        self.physical(actual.RETURN_THREE, ['210', '212', '999'])
        self.assertIsNone(self.driver.seg.window)
        self.assertEqual(self.driver.n_commits_total, 0)
        self.assertFalse(any(key == 'early_recover_refused' for key, _ in self.driver.findings))

    def test_wrong_side_identity_cannot_select_a_payment(self):
        self.physical(actual.STAGE_GROUP, ['210', '211', '212', '099'])
        self.physical(actual.RETURN_THREE, ['210', '212', '099'])
        self.assertIsNone(self.driver.seg.window)
        self.assertEqual(self.driver.n_commits_total, 0)

    def test_group_of_used_dice_still_requires_real_turn_end(self):
        self.driver.es.dice[self.driver.human_side][0].used = True
        turn = self.driver.es.turn
        items = self.physical('$Wey moves group (Free Action Die (P), Free Action Die (C)) '
                              'from Current Action to FP Dice.', ['210', '213'])
        self.assertEqual(self.driver.es.turn, turn)
        self.assertTrue(self.driver.es.dice[self.driver.human_side][0].used)
        self.assertTrue(any(key == 'hunt_box_die_refused' for key, _ in self.driver.findings), items)

    def test_hunt_return_with_other_live_dice_remains_refused(self):
        self.driver.es.dice[self.driver.human_side][0].used = True
        turn = self.driver.es.turn
        items = self.physical('$Wey moves Free Action Die (P) from The Hunt For The Ring to FP Dice.', ['210'])
        self.assertEqual(self.driver.es.turn, turn)
        self.assertTrue(any(key == 'hunt_box_die_refused' for key, _ in self.driver.findings), items)

    def test_old_turn_selection_cannot_open_a_new_window(self):
        self.start_group()
        self.driver.es.turn += 1
        self.physical(actual.RETURN_THREE, ['210', '212', '213'])
        self.assertIsNone(self.driver.seg.window)
        self.assertIsNone(self.driver._grouped_die_selection)

    def recover(self, human):
        self.driver.human_side = human
        self.driver.ai_side = next(side for side in actual.gandalf.gs.Side if side is not human)
        for pool in self.driver.es.dice.values():
            for die in pool:
                die.used = True
        pids = [pid for pid, binding in self.driver.view.die_id_map.items() if binding[0] is human]
        label = 'Free' if human.name == 'FREE_PEOPLES' else 'Shadow'
        pool_name = 'FP Dice' if human.name == 'FREE_PEOPLES' else 'SA Dice'
        tokens = [label + ' Action Die (' + FACE_LETTER[
                  self.driver.es.dice[human][self.driver.view.die_id_map[pid][1]].current.name] + ')'
                  for pid in pids]
        turn = self.driver.es.turn
        items = self.physical('$Wey moves group (' + ', '.join(tokens) + ') '
                              'from Current Action to ' + pool_name + '.', pids)
        self.assertEqual(self.driver.es.turn, turn + 1, (items, self.driver.alarms))
        self.assertEqual(self.driver.phase, 'draws')
        self.assertEqual(sum(item[0] == 'recover_ai' for item in items), 1)

    def test_genuine_fp_recovery_still_advances_once(self):
        self.recover(actual.gandalf.gs.Side.FREE_PEOPLES)

    def test_genuine_shadow_recovery_still_advances_once(self):
        self.recover(actual.gandalf.gs.Side.SHADOW)


if __name__ == '__main__':
    unittest.main()
