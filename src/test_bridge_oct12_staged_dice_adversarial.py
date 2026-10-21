"""Independent regressions from the visible Java Gandalf test on October 12.

The two group lines and identities below are literal native evidence in
audit/bridge_oct11/Gandalf_group_physical_r3/player_bridge_failure. A group
returned from Current Action is not necessarily a turn-end Recover command.
"""
import unittest

import test_bridge_oct11_gandalf_group_live as gandalf


STAGE_GROUP = ('$Wey moves group (Free Action Die (P), Free Action Die (W), '
               'Free Action Die (H), Free Action Die (C)) from FP Dice to Current Action.')
RETURN_THREE = ('$Wey moves group (Free Action Die (C), Free Action Die (H), '
                'Free Action Die (P)) from Current Action to FP Dice.')
RETURN_ALL = ('$Wey moves group (Free Action Die (P), Free Action Die (W), '
              'Free Action Die (H), Free Action Die (C)) from Current Action to FP Dice.')


class StagedDicePublicEvidence(unittest.TestCase):
    def setUp(self):
        self.fixture = gandalf.GandalfGroupedLive()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.driver = self.fixture.driver
        self.driver.enforce_human = True

    def physical(self, text, pids):
        return self.fixture.physical(text, pids)

    def assert_no_action_or_recovery(self, before, items):
        state = self.driver.es
        self.assertEqual(state.turn, before[0])
        self.assertEqual(self.driver.phase, 'actions')
        self.assertEqual(self.driver.n_commits_total, 0)
        self.assertEqual([(d.current, d.used) for d in state.dice[gandalf.gs.Side.FREE_PEOPLES]], before[1])
        self.assertFalse(self.driver.alarms, self.driver.alarms)
        self.assertFalse(any(k == 'early_recover_refused' for k, _ in self.driver.findings))
        self.assertFalse(any('undoe' in str(row).lower() or 'cannot recover' in str(row).lower()
                             for row in items), items)

    def start_group(self):
        before = (self.driver.es.turn,
                  [(d.current, d.used) for d in self.driver.es.dice[gandalf.gs.Side.FREE_PEOPLES]])
        self.physical(STAGE_GROUP, ['213', '212', '211', '210'])
        return before

    def test_return_three_unused_dice_is_not_a_recover_command(self):
        before = self.start_group()
        items = self.physical(RETURN_THREE, ['210', '212', '213'])
        self.assert_no_action_or_recovery(before, items)

    def test_return_all_unused_dice_cancels_without_a_turn_or_payment(self):
        before = self.start_group()
        items = self.physical(RETURN_ALL, ['213', '212', '211', '210'])
        self.assert_no_action_or_recovery(before, items)
        self.assertIsNone(self.driver.seg.window)

    def test_narrow_to_west_then_actual_promotion_has_one_payment(self):
        self.start_group()
        self.physical(RETURN_THREE, ['210', '212', '213'])
        self.physical(gandalf.ARRIVE, ['240'])
        self.physical(gandalf.FLIP, ['240'])
        self.physical(gandalf.BONUS, ['358'])
        items = self.physical(gandalf.CLOSE, ['211', '358'])
        self.assertTrue(self.driver.es.gandalf_the_white.in_play, (items, self.driver.findings))
        self.assertEqual(self.driver.n_commits_total, 1)
        self.assertEqual([d.used for d in self.driver.es.dice[gandalf.gs.Side.FREE_PEOPLES]],
                         [False, True, False, False])
        self.assertEqual(len(self.driver.es.dice[gandalf.gs.Side.FREE_PEOPLES]), 4)
        self.assertFalse(self.driver.alarms, self.driver.alarms)
        self.assertIsNone(self.driver.seg.window)
        self.assertIsNone(self.driver.deferred)

    def test_group_cancellation_does_not_choose_a_later_action_die(self):
        self.start_group()
        self.physical(RETURN_ALL, ['213', '212', '211', '210'])
        self.physical('$Wey moves Free Action Die (C) from FP Dice to Current Action.', ['213'])
        window = self.driver.seg.window
        self.assertIsNotNone(window)
        self.assertEqual(window.die_face, 'C')
        self.physical('$Wey moves Free Action Die (C) from Current Action to FP Dice.', ['213'])
        self.assertIsNone(self.driver.seg.window)
        self.assertEqual(self.driver.n_commits_total, 0)
        self.assertFalse(any(d.used for d in self.driver.es.dice[gandalf.gs.Side.FREE_PEOPLES]))

    def test_native_undo_of_group_allows_ordinary_single_west_promotion(self):
        self.start_group()
        self.physical('$Wey undoes last action : '+STAGE_GROUP, [])
        items = self.fixture.promote()
        self.assertEqual(self.driver.n_commits_total, 1, (items, self.driver.findings))
        self.assertTrue(self.driver.es.gandalf_the_white.in_play)
        self.assertEqual([d.used for d in self.driver.es.dice[gandalf.gs.Side.FREE_PEOPLES]],
                         [False, True, False, False])
        self.assertIsNone(self.driver.seg.window)
        self.assertFalse(self.driver.alarms, self.driver.alarms)

    def test_undo_narrowing_restores_selection_without_spending_any_die(self):
        self.start_group()
        self.physical(RETURN_THREE, ['210', '212', '213'])
        self.physical('$Wey undoes last action : '+RETURN_THREE, [])
        self.assertIsNone(self.driver.seg.window)
        self.assertEqual(self.driver.n_commits_total, 0)
        self.assertFalse(any(d.used for d in self.driver.es.dice[gandalf.gs.Side.FREE_PEOPLES]))
        self.physical(RETURN_THREE, ['210', '212', '213'])
        self.assertIsNotNone(self.driver.seg.window)
        self.assertEqual(self.driver.seg.window.die_face, 'W')
        self.physical(gandalf.ARRIVE, ['240'])
        self.physical(gandalf.FLIP, ['240'])
        self.physical(gandalf.BONUS, ['358'])
        items = self.physical(gandalf.CLOSE, ['211', '358'])
        self.assertEqual(self.driver.n_commits_total, 1, (items, self.driver.findings))
        self.assertTrue(self.driver.es.gandalf_the_white.in_play)
        self.assertFalse(self.driver.alarms, self.driver.alarms)


if __name__ == '__main__':
    unittest.main()
