"""Rejected draw windows restore through native Undo without ordering cards."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_bridge_oct12_event_draw_adversarial as original
from wotr_bridge.live.live_server import LiveServer


class EventDrawRollback(unittest.TestCase):
    def setUp(self):
        self.case = original.EventDrawEvidence()
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.fixture = self.case.fixture
        self.driver = self.case.driver

    def check_refusal(self, decks):
        driver = self.driver
        state = driver.es
        before_hands = {side: list(hand.hand) for side, hand in state.hands.items()}
        before_decks = {name: len(getattr(state, name)) for name in
                        ('fp_char_deck', 'fp_strat_deck', 'sh_char_deck', 'sh_strat_deck')}
        turn = state.turn
        items = self.case.draw_window(decks)
        # Exercise the actual server output dispatcher: rejected in-window
        # draws must never become either direct CALLs or draw requests.
        server = LiveServer(self.fixture.rig.td, driver.ai_side, load_nets=False)
        server.driver, server.rand = driver, self.fixture.rand
        server.rand.install()
        output = [line for item in items for line in server._dispatch(item)]
        undo = [line for line in output if '<~Controls.819~>' in line]
        self.assertEqual(len(undo), len(decks) + 2, output)
        self.assertFalse(any(line.startswith('!CALL') for line in output), output)
        self.assertFalse(any('draw a free' in line.lower() for line in output), output)
        self.assertFalse(getattr(driver, '_reissued_draws94', {}))
        self.assertIsNone(driver.deferred)
        self.assertIsNone(self.fixture.rand.pending())
        self.assertFalse(driver.alarms)
        rows = list(driver.feed.lines[-(len(decks) + 2):])
        for row in reversed(rows):
            self.fixture.physical('$WOTR_AI undoes last action : ' + row, [])
        self.assertFalse(driver.uncommitted)
        self.assertEqual(driver.n_commits_total, 0)
        self.assertFalse(driver.es.dice[driver.human_side][0].used)
        self.assertEqual(driver.es.turn, turn)
        self.assertEqual({side: list(hand.hand) for side, hand in driver.es.hands.items()}, before_hands)
        self.assertEqual({name: len(getattr(driver.es, name)) for name in before_decks}, before_decks)
        self.assertFalse(any(self.fixture.rand.unconsumed().values()))
        self.assertFalse(driver.alarms)
        self.fixture.write_dump(feed_position=self.fixture.seq)
        driver.view.dump_path = str(self.fixture.dump_path)
        driver._check_restore_watch()
        self.assertFalse(driver.restore_watch.armed)
        self.assertFalse(driver.alarms)

    def test_two_strategy_draws_restore_without_another_draw(self):
        self.check_refusal(['Strategy', 'Strategy'])

    def test_two_character_draws_restore_without_another_draw(self):
        self.check_refusal(['Character', 'Character'])

    def test_mixed_draws_restore_without_another_draw(self):
        self.check_refusal(['Character', 'Strategy'])


if __name__ == '__main__':
    unittest.main()
