"""Real AI event suspension: native echoes/draws must resume the same pick.

Only the search choice and private draw decoder are supplied by the test.
Menus, clone apply, randomness, staging, feed routing and cleanup are real.
"""
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'wotr_bridge'), str(ROOT)]
import ai
import game_logic as gl
import game_state as gs
from live.live_server import LiveServer
from live.live_parser import parse
from live.segmenter import Window

FP, SH = gs.Side.FREE_PEOPLES, gs.Side.SHADOW
CARD = 'FREE_PEOPLES_Strategy_Thranduil\'s_Archers'
DRAW = 'FREE_PEOPLES_Strategy_The_Red_Arrow'
BONUS = 'FREE_PEOPLES_Strategy_A_Power_too_Great'
STAGE = '$WOTR_AI_FP moves Free Action Die (P) from FP Dice to Current Action.'
REVEAL = "<auto> moves Thranduil's Archers/Valour from Free Peoples Hand (chosen) to Current FP Card."


class AIStagedResume(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(prefix='ai_resume_', dir=ROOT)
        self.addCleanup(folder.cleanup)
        self.server = LiveServer(folder.name, FP, load_nets=False, seed=7)
        self.feed = Path(folder.name)/'ai_feed.txt'
        self.feed.write_text('', encoding='utf8')
        self.driver = self.server.driver
        state = self.driver.es
        gl.initialize_decks(state)
        state._declare_phase = False
        state.dice_rolled = True
        self.driver.phase = 'actions'
        self.driver.last_actor = SH
        self.driver.enforce_human = True
        for pool in state.dice.values():
            for die in pool:
                die.current = gs.ActionDieFace.CHARACTER
                die.used = True
        state.dice[FP][2].current = gs.ActionDieFace.EVENT
        state.dice[FP][2].used = False
        state.dice[SH][0].used = False
        state.hands[FP].hand = [CARD]
        if CARD in state.fp_strat_deck:
            state.fp_strat_deck.remove(CARD)
        self.assertIn(DRAW, state.fp_strat_deck)
        self.assertIn(BONUS, state.fp_strat_deck)
        self.driver.view.card_pid_map[CARD] = '129'
        self.driver.view.piece_region['129'] = 'Free Peoples Hand'
        self.driver.view.piece_region['212'] = 'FP Dice'
        self.searches = 0

        def choose(state, side, **kwargs):
            self.searches += 1
            menu = ai.get_legal_actions(state, side)
            return next(a for a in menu if a.action_type == 'play_event'
                        and a.params.get('card_id') == CARD), None

        self.server.own_net = object()
        self.server.own_search = SimpleNamespace(choose=choose)

    def cycle(self, *lines, card=None):
        if lines:
            with self.feed.open('a', encoding='utf8') as stream:
                stream.write('\n'.join(lines)+'\n')
        decoded = ([] if card is None else [dict(side='FREE_PEOPLES', deck='strategy',
                   cid=card, pid='145' if card == DRAW else '146', name='test own draw')])
        with patch.object(self.server.cards, 'harvest', return_value=decoded):
            return self.server.handle({'lines': ()})

    def start(self):
        output = self.cycle()
        self.assertEqual(output.count('!CALL play_card 129'), 1, output)
        self.assertEqual(output.count('!CALL draw_fp_strat'), 1, output)
        self.assertEqual(self.searches, 1)
        self.assertEqual(self.driver.n_commits_total, 0)
        self.assertTrue(self.server.rand.step_open)
        self.assertEqual(self.server.rand.pending().kind, 'draws')
        self.assertEqual(self.driver.owed_die_kills,
                         [('FREE_PEOPLES', 2, '129', CARD)])
        return output

    def test_waiting_event_replays_without_restage_or_search_then_distinct_bonus(self):
        output = self.start()
        journal = list(self.server.rand._journal)
        for rows in ((STAGE, REVEAL), (), (), ()):
            output += self.cycle(*rows)
            self.assertIsNotNone(self.server._open_step)
            self.assertEqual(self.server.rand._journal, journal)
            self.assertEqual(self.searches, 1)
            self.assertEqual(self.driver.n_commits_total, 0)
            self.assertFalse(self.driver.alarms)
        output += self.cycle('$Wey draws Free Strategy Card.', card=DRAW)
        self.assertEqual(self.driver.n_commits_total, 1)
        self.assertEqual(self.driver.es.pending_card_option['card_id'], '__gandalf_draw__')
        self.assertEqual(self.driver.es.hands[FP].hand, [DRAW])
        self.assertTrue(self.driver.es.dice[FP][2].used)
        self.assertEqual(output.count('!CALL play_card 129'), 1)
        self.assertEqual(output.count('!CALL draw_fp_strat'), 1)
        self.assertEqual(sum('Scribe.141' in row and '264' in row for row in output), 1)
        bonus_request = self.cycle()
        self.assertEqual(bonus_request.count('!CALL draw_fp_strat'), 1, bonus_request)
        self.assertFalse(any('play_card' in row or 'HighlightLabel.86' in row for row in bonus_request))
        self.assertEqual(self.searches, 1)  # forced Gandalf policy does not reselect the event
        self.assertEqual(self.cycle(), [])
        cleanup = self.cycle('$Wey draws Free Strategy Card.', card=BONUS)
        self.assertEqual(self.driver.n_commits_total, 2)
        self.assertEqual(self.driver.es.hands[FP].hand, [DRAW, BONUS])
        self.assertEqual(sum('Controls.644' in row and '212 129' in row for row in cleanup), 1, cleanup)
        self.assertFalse(self.driver.owed_die_kills)
        self.assertEqual(self.cycle(), [])
        self.assertFalse(self.driver.alarms)

    def test_result_in_first_echo_poll_still_resumes_the_same_pick(self):
        output = self.start()
        output += self.cycle(STAGE, REVEAL, '$Wey draws Free Strategy Card.', card=DRAW)
        self.assertEqual(self.driver.n_commits_total, 1)
        self.assertEqual(self.searches, 1)
        self.assertEqual(output.count('!CALL play_card 129'), 1)
        self.assertEqual(output.count('!CALL draw_fp_strat'), 1)
        self.assertFalse(self.driver.alarms)

    def test_new_action_query_remains_blocked_by_the_staged_die(self):
        self.start()
        self.assertTrue(self.driver.action_completion_pending())
        self.assertIsNone(self.driver.ai_action_dp())

    def test_unrelated_or_mismatched_owed_die_still_holds_the_retry(self):
        self.start()
        owned = self.driver.owed_die_kills[0]
        journal = list(self.server.rand._journal)
        for owed in (owned, (owned[0], owned[1], '130', owned[3])):
            with self.subTest(known_card=owed[2]):
                self.driver.owed_die_kills[:] = [owed, ('FREE_PEOPLES', 1, None, None)]
                self.assertEqual(self.cycle(), [])
                self.assertEqual(self.driver.n_commits_total, 0)
                self.assertEqual(self.searches, 1)
                self.assertEqual(self.server.rand._journal, journal)
                self.assertTrue(self.server._step_parked)
                self.assertFalse(self.driver.alarms)
        # A different card PID alone also cannot use the old step's credit.
        self.driver.owed_die_kills[:] = [(owned[0], owned[1], '130', owned[3])]
        self.assertEqual(self.cycle(), [])
        self.assertEqual(self.driver.n_commits_total, 0)
        self.driver.owed_die_kills[:] = [owned]
        self.cycle('$Wey draws Free Strategy Card.', card=DRAW)
        self.assertEqual(self.driver.n_commits_total, 1)
        self.assertEqual(self.searches, 1)

    def test_human_window_is_not_exempted_by_ai_staging(self):
        self.start()
        record = parse('$Wey moves Shadow Action Die (C) from SA Dice to Current Action.')
        self.driver.seg.window = Window('SHADOW', 'C', record)
        self.assertTrue(self.driver.action_completion_pending(self.server._resuming_owed_dice()))
        self.assertEqual(self.cycle(), [])
        self.assertEqual(self.driver.n_commits_total, 0)
        self.assertTrue(self.server.rand.step_open)
        self.assertFalse(self.driver.alarms)
        self.driver.seg.window = None
        self.cycle('$Wey draws Free Strategy Card.', card=DRAW)
        self.assertEqual(self.driver.n_commits_total, 1)

    def test_staging_without_current_emitted_journal_does_not_bypass_direct_gate(self):
        dp = self.driver.ai_action_dp()
        self.start()
        journal = self.server.rand._journal
        for field, value in (('_em_sent', False), ('_open_step', None)):
            old = getattr(self.server, field)
            setattr(self.server, field, value)
            self.assertEqual(self.server._resuming_owed_dice(), ())
            self.assertEqual(self.server._ai_decide(dp), [])
            setattr(self.server, field, old)
        self.server.rand._journal = []
        self.assertEqual(self.server._resuming_owed_dice(), ())
        self.assertEqual(self.server._ai_decide(dp), [])
        self.server.rand._journal = journal
        self.assertEqual(self.searches, 1)
        self.assertEqual(self.driver.n_commits_total, 0)

    def test_real_phase_change_still_abandons_the_obsolete_step(self):
        self.start()
        self.driver.phase = 'eyes'
        self.cycle()
        self.assertIsNone(self.server._open_step)
        self.assertFalse(self.server.rand.step_open)
        self.assertTrue(any('ABANDONED STEP RESIDUE' in alarm for alarm in self.driver.alarms))
        self.assertEqual(self.searches, 1)
        self.assertEqual(self.driver.n_commits_total, 0)

    def test_shadow_stock_draw_waits_and_commits_once(self):
        folder = tempfile.TemporaryDirectory(prefix='shadow_draw_', dir=ROOT)
        self.addCleanup(folder.cleanup)
        server = LiveServer(folder.name, SH, load_nets=False, seed=13)
        driver = server.driver
        state = driver.es
        gl.initialize_decks(state)
        state._declare_phase = False
        state.dice_rolled = True
        driver.phase = 'actions'
        driver.last_actor = FP
        for pool in state.dice.values():
            for die in pool:
                die.current = gs.ActionDieFace.CHARACTER
                die.used = True
        state.dice[SH][0].current = gs.ActionDieFace.EVENT
        state.dice[SH][0].used = False
        state.dice[FP][0].used = False
        searches = []

        def choose(state, side, **kwargs):
            searches.append(True)
            return next(a for a in ai.get_legal_actions(state, side)
                        if a.action_type == 'draw_card' and a.params['deck'] == 'strategy'), None

        server.own_net = object()
        server.own_search = SimpleNamespace(choose=choose)
        output = server.handle({'lines': ()})
        self.assertEqual(output.count('!CALL draw_sa_strat'), 1, output)
        for _ in range(3):
            self.assertEqual(server.handle({'lines': ()}), [])
            self.assertTrue(server.rand.step_open)
            self.assertEqual(len(searches), 1)
            self.assertEqual(driver.n_commits_total, 0)
        card = state.sh_strat_deck[-1]
        count = len(state.hands[SH].hand)
        Path(folder.name, 'ai_feed.txt').write_text(
            '$WOTR_AI_Shadow moves Shadow Action Die (P) from SA Dice to Current Action.\n'
            '$Wey draws Shadow Strategy Card.\n', encoding='utf8')
        decoded = [dict(side='SHADOW', deck='strategy', cid=card, pid='174', name='test own draw')]
        with patch.object(server.cards, 'harvest', return_value=decoded):
            output += server.handle({'lines': ()})
        self.assertEqual(driver.n_commits_total, 1)
        self.assertEqual(len(driver.es.hands[SH].hand), count+1)
        self.assertEqual(len(searches), 1)
        self.assertEqual(output.count('!CALL draw_sa_strat'), 1)
        self.assertEqual(sum('Controls.644' in row for row in output), 1)
        self.assertFalse(driver.owed_die_kills)
        self.assertFalse(driver.alarms)

    def test_hunt_chance_retry_retains_its_one_pick_and_one_native_roll(self):
        state = self.driver.es
        state.hands[FP].hand = []
        state.dice[FP][2].current = gs.ActionDieFace.CHARACTER
        state.dice[SH][1].used = False
        state.dice[SH][1].current = None
        gl.allocate_eyes(state, 1, indices=[1])

        def choose(state, side, **kwargs):
            self.searches += 1
            return next(a for a in ai.get_legal_actions(state, side)
                        if a.action_type == 'fellowship_move'), None

        self.server.own_search.choose = choose
        output = self.cycle()
        self.assertEqual(self.server.rand.pending().kind, 'd6')
        self.assertEqual(sum('roll 1 die' in row for row in output), 1, output)
        journal = list(self.server.rand._journal)
        for _ in range(3):
            self.assertEqual(self.cycle(), [])
            self.assertEqual(self.server.rand._journal, journal)
            self.assertEqual(self.searches, 1)
        output += self.cycle('<game> Wey rolls 1')
        self.assertEqual(self.driver.n_commits_total, 1)
        self.assertEqual(self.driver.es.fellowship.progress, 1)
        self.assertTrue(self.driver.es.dice[FP][2].used)
        self.assertEqual(self.searches, 1)
        self.assertEqual(sum('roll 1 die' in row for row in output), 1)
        self.assertFalse(self.driver.alarms)


if __name__ == '__main__':
    unittest.main()
