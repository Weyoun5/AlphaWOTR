"""Ordinary human choices take time; unmatched gestures remain diagnostic."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'wotr_bridge'), str(ROOT)]
import ai
import event_effects as ee
import game_logic as gl
import game_state as gs
from live import live_parser
from live.live_server import LiveServer
from live.segmenter import Window

FP, SH = gs.Side.FREE_PEOPLES, gs.Side.SHADOW


class HumanChoiceWaitStatus(unittest.TestCase):
    def server(self, human):
        folder = tempfile.TemporaryDirectory(prefix='human_wait_', dir=ROOT)
        self.addCleanup(folder.cleanup)
        server = LiveServer(folder.name, SH if human is FP else FP, load_nets=False, seed=7)
        driver = server.driver
        gl.initialize_decks(driver.es)
        driver.es.turn = 3
        driver.enforce_human = True
        driver._turn_start_draws_open = False
        driver.turn_draws = {(side, deck): 1 for side in (FP, SH)
                            for deck in ('character', 'strategy')}
        Path(folder.name, 'ai_feed.txt').write_text('', encoding='utf8')
        return server

    def discard(self):
        server = self.server(SH)
        driver = server.driver
        driver.phase = 'declare'
        driver.last_actor = SH
        card = ee._card_by_name('The Day Without Dawn')
        # An isolated public fixture, not identities from the player's live hand.
        hand = list(driver.es.sh_strat_deck[:6])
        hand = [cid for cid in hand if cid != card.card_id]
        hand += driver.es.sh_char_deck[:6-len(hand)]
        driver.es.hands[SH].hand = hand[:6]+[card.card_id]
        gl.open_hand_limit_discard(driver.es)
        self.assertEqual(ai.describe_pending(driver.es)['kind'], 'discard')
        server._open_step = ('ai_declare', None)
        return server, card

    def test_idle_seven_card_discard_is_not_stall_or_repair_then_native_discard_finishes(self):
        server, card = self.discard()
        driver = server.driver
        before = driver.es
        responses = []
        for _ in range(30):
            responses.extend(server.handle({'lines': ()}))
        server._stalled = 60  # exercise the real elapsed status without sleeping
        self.assertIn('hand-limit discard', server._waiting_on())
        self.assertNotIn('stalled', server._waiting_on())
        self.assertEqual(sum('hand-limit discard is still open' in line for line in responses), 1)
        self.assertFalse(any(kind == 'repair_not_offered' for kind, _ in driver.findings))
        self.assertFalse(any('OPEN STEP STARVED' in note for note in server.notes))
        self.assertIs(driver.es, before)
        self.assertEqual(driver.n_commits_total, 0)
        self.assertIsNone(driver.pending_buf)
        self.assertIsNone(driver._repair)
        self.assertFalse(driver.alarms)
        Path(server.client_dir, 'ai_feed.txt').write_text(
            '$Wey kills The Day Without Dawn in Shadow Hand.\n', encoding='utf8')
        completed = server.handle({'lines': ()})
        for _ in range(3):
            completed += server.handle({'lines': ()})
        self.assertEqual(len(driver.es.hands[SH].hand), 6)
        self.assertNotIn(card.card_id, driver.es.hands[SH].hand)
        self.assertIsNone(driver.es.pending_discard)
        self.assertEqual(sum('> fp ok' in line for line in completed), 1, completed)
        self.assertEqual(driver.phase, 'eyes')
        count = driver.n_commits_total
        self.assertGreater(count, 1)  # Human discard and normal AI Fellowship decisions.
        self.assertEqual(server.handle({'lines': ()}), [])
        self.assertEqual(driver.n_commits_total, count)
        self.assertFalse(driver.alarms)

    def test_guide_before_ai_eyes_is_parked_not_rng_starvation(self):
        server = self.server(FP)
        driver = server.driver
        driver.phase = 'eyes'
        driver.es._declare_phase = False
        gl.change_guide(driver.es, allow_pending=True)
        server._open_step = ('ai_eyes', None)
        self.assertEqual(ai.describe_pending(driver.es)['kind'], 'guide_change')
        for _ in range(30):
            self.assertEqual(server.handle({'lines': ()}), [])
        server._stalled = 60
        self.assertIn('Guide', server._waiting_on())
        self.assertNotIn('stalled', server._waiting_on())
        self.assertFalse(any('OPEN STEP STARVED' in note for note in server.notes))
        self.assertEqual(driver.n_commits_total, 0)
        self.assertIsNone(server.rand.pending())
        self.assertFalse(driver.alarms)
        Path(server.client_dir, 'ai_feed.txt').write_text(
            '$Wey moves Strider from Fellowship Of The Ring to Guide Of The Fellowship.\n',
            encoding='utf8')
        completed = server.handle({'lines': ()})
        self.assertEqual(driver.es.fellowship.guide.name, 'Strider')
        self.assertIsNone(driver.es.pending_guide_change)
        self.assertEqual(driver.phase, 'roll')
        self.assertEqual(driver.n_commits_total, 2)  # Guide and AI eye allocation.
        self.assertEqual(sum(line.startswith('!CALL roll_action_dice') for line in completed), 1)
        for _ in range(3):
            self.assertEqual(server.handle({'lines': ()}), [])
            self.assertEqual(driver.n_commits_total, 2)
        self.assertFalse(driver.alarms)

    def test_unanswered_hunt_and_eye_choices_have_truthful_status(self):
        for kind in ('hunt', 'eye'):
            with self.subTest(kind=kind):
                server = self.server(FP if kind == 'hunt' else SH)
                driver = server.driver
                driver.es._declare_phase = False
                if kind == 'hunt':
                    driver.phase = 'actions'
                    driver.es.pending_hunt = {'phase': 'hunt_sacrifice', 'damage': 1}
                else:
                    driver.phase = 'eyes'
                    driver.es._eye_allocation_phase = True
                dp = ai.describe_pending(driver.es)
                self.assertIs(dp['side'], driver.human_side)
                self.assertTrue(dp['actions'])
                server._stalled = 60
                self.assertNotIn('stalled', server._waiting_on())
                self.assertEqual(driver.n_commits_total, 0)
                self.assertIsNone(server.rand.pending())

    def test_unrelated_buffer_still_produces_unmatched_diagnostic(self):
        server, _ = self.discard()
        driver = server.driver
        row = live_parser.parse('$Wey moves S&E Regular from North Rhun to Dale.')
        driver.pending_buf = Window('SHADOW', None, row)
        for _ in range(30):
            server.handle({'lines': ()})
        server._stalled = 60
        self.assertIn('stalled', server._waiting_on())
        self.assertTrue(any(kind == 'pending_buffer_stall' for kind, _ in driver.findings))
        self.assertEqual(driver.n_commits_total, 0)
        self.assertIsNotNone(driver.es.pending_discard)


if __name__ == '__main__':
    unittest.main()
