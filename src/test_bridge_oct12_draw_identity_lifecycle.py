"""Real raw-card harvesting and rendered-feed handling around refused draws.

The decoder's commitment result is initialized as fixture data; harvesting,
matching, hand mutation, native Undo handling, and status publication are real.
"""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_bridge_oct11_gandalf_group_live as fixture_module
from wotr_bridge.live.live_server import LiveServer
from wotr_bridge.live.card_identity import name_to_cid
import card_decoder


STAGE = '$Wey moves Free Action Die (P) from FP Dice to Current Action.'
DRAW = '$Wey draws Free Strategy Card.'
CLOSE = '$Wey kills Free Action Die (P) in Current Action.'


class DrawIdentityLifecycle(unittest.TestCase):
    def setUp(self):
        self.f = fixture_module.GandalfGroupedLive()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.d = self.f.driver
        self.d.enforce_human = True
        self.s = LiveServer(self.f.rig.td, self.d.ai_side, load_nets=False)
        self.s.driver, self.s.rand = self.d, self.f.rand
        self.s.rand.install()
        self.human = self.d.human_side
        self.before = list(self.d.es.hands[self.human].hand)
        self.raw = []
        self.cids = {}
        for pid, number in [('120', 8), ('124', 0), ('129', 1)]:
            self.s.cards.decoder.fp_strat_deck[card_decoder._PID_TO_KI[int(pid)]] = number
            self.cids[pid] = name_to_cid(card_decoder.FP_STRATEGY_CARDS[number], 'FP Strategy')

    def cycle(self, line=None, pids=(), raw_pids=()):
        if line is not None:
            self.f.seq += 1
            meta = dict(seq=self.f.seq, gen=1, raw_new=True, nick='Wey',
                        pids='redacted' if line == DRAW else list(pids), side='FP',
                        tags=['Controls.509', 'Controls.510', 'Game.2005', 'Controls.511', 'Controls.45']
                        if line == DRAW else ['Scribe.141', 'Scribe.142', 'Scribe.143', 'Scribe.144'])
            with self.f.meta.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(meta)+'\n')
            with self.f.feed.open('a', encoding='utf-8') as stream:
                stream.write(line+'\n')
        for pid in raw_pids:
            self.raw.append('$Wey<~Controls.509~>'+pid+
                            '<~Controls.510~><~Game.2005~><~Controls.511~><~Controls.45~>')
        out = self.s.handle({'lines': self.raw})
        self.s.status_menu()
        return out

    def reject(self, raw_pids=('120', '124')):
        self.cycle(STAGE, ['210'])
        self.cycle(DRAW, raw_pids=raw_pids[:1])
        self.cycle(DRAW, raw_pids=raw_pids[1:])
        out = self.cycle(CLOSE, ['210'])
        self.assertEqual(sum('Controls.819' in line for line in out), 4, out)
        for row in (CLOSE, DRAW, DRAW, STAGE):
            self.cycle('$WOTR_AI undoes last action : '+row)

    def test_actual_harvest_refusal_preserves_hand_and_known_pid_bindings(self):
        mapping = dict(self.d.view.card_pid_map)
        drawn = set(self.d.view.ai_drawn_pids)
        self.reject()
        for _ in range(31):
            self.cycle()
        self.assertEqual(self.d.es.hands[self.human].hand, self.before)
        self.assertEqual(self.d.view.card_pid_map, mapping)
        self.assertEqual(self.d.view.ai_drawn_pids, drawn)
        self.assertEqual(self.d.n_commits_total, 0)
        self.assertFalse(self.d.alarms)
        self.assertFalse(self.d.hands._draw_receipts)

    def test_separate_grant_identity_survives_interleaved_refused_observations(self):
        # A genuine turn-start grant entered the hand before the bad action;
        # its private observation arrives in the same poll as the overdraw.
        self.d.turn_draws[(self.human, 'strategy')] = 0
        self.cycle(DRAW)
        self.assertEqual(self.d.n_commits_total, 1)
        self.cycle(STAGE, ['210'])
        self.cycle(DRAW)
        self.cycle(DRAW, raw_pids=['129', '120', '124'])
        self.assertEqual(self.d.es.hands[self.human].hand, self.before+[self.cids['129']])
        out = self.cycle(CLOSE, ['210'])
        self.assertEqual(sum('Controls.819' in line for line in out), 4)
        for row in (CLOSE, DRAW, DRAW, STAGE):
            self.cycle('$WOTR_AI undoes last action : '+row)
        self.assertEqual(self.d.es.hands[self.human].hand, self.before+[self.cids['129']])
        self.assertEqual(self.d.view.card_pid_map[self.cids['129']], '129')
        self.assertNotIn(self.cids['120'], self.d.view.card_pid_map)
        self.assertNotIn(self.cids['124'], self.d.view.card_pid_map)
        self.assertFalse(self.d.alarms)

    def test_completed_draw_undo_restores_old_hand_and_redo_remains_explicitly_held(self):
        self.cycle(STAGE, ['210'])
        self.cycle(DRAW, raw_pids=['120'])
        self.cycle(CLOSE, ['210'])
        self.assertEqual(self.d.es.hands[self.human].hand, self.before+[self.cids['120']])
        # AI search is not run: the server waits for a native boundary first.
        for row in (CLOSE, DRAW, STAGE):
            self.cycle('$Wey undoes last action : '+row)
        self.assertEqual(self.d.es.hands[self.human].hand, self.before)
        self.assertFalse(self.d.alarms)
        self.cycle('$Wey redoes last action : '+STAGE)
        self.assertTrue(any('REDO' in alarm for alarm in self.d.alarms))
        self.assertEqual(self.d.es.hands[self.human].hand, self.before)
        self.assertFalse(self.d.hands._draw_receipts)


if __name__ == '__main__':
    unittest.main()
