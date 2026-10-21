"""Independent server-path checks for the r5 physical draw rollback defect.

Only private commitment decoding is substituted. Rendered input, native
sidecar metadata, server handling, driver matching and hand updates are real.
These unit tests supplement, and cannot replace, actual Java acceptance.
"""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_bridge_oct11_gandalf_group_live as fixture_module
from wotr_bridge.live.live_server import LiveServer


STRATEGY = 'FREE_PEOPLES_Strategy_Riders_of_Théoden'
CHARACTER = 'FREE_PEOPLES_Character_Elven_Rope'


class ObservedDrawIdentity(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.GandalfGroupedLive()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.driver = self.fixture.driver
        self.driver.enforce_human = True
        self.server = LiveServer(self.fixture.rig.td, self.driver.ai_side,
                                 load_nets=False, seed=11)
        self.server.driver = self.driver
        self.server.rand = self.fixture.rand
        self.server.rand.install()
        self.human = self.driver.human_side
        self.initial = list(self.driver.es.hands[self.human].hand)

    def cycle(self, text=None, *, pids=(), decoded=()):
        f = self.fixture
        if text is not None:
            f.seq += 1
            if ' draws Free ' in text:
                deck = 'Strategy' if 'Strategy' in text else 'Character'
                tags = (['Controls.509', 'Controls.510', 'Game.2005', 'Controls.511', 'Controls.45']
                        if deck == 'Strategy' else
                        ['Controls.491', 'Controls.492', 'Game.2004', 'Controls.493', 'Controls.45'])
                meta = dict(seq=f.seq, gen=1, raw_new=True, nick='Wey',
                            pids='redacted', side='FP', tags=tags)
            else:
                meta = dict(seq=f.seq, gen=1, attest=True, nick='Wey', pids=list(pids),
                            tags=['Scribe.141', 'Scribe.142', 'Scribe.143', 'Scribe.144'])
            with f.meta.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(meta)+'\n')
            with f.feed.open('a', encoding='utf-8') as stream:
                stream.write(text+'\n')
        with patch.object(self.server.cards, 'harvest', return_value=list(decoded)):
            return self.server.handle({'lines': ()})

    @staticmethod
    def identity(deck, cid=None, pid='120'):
        return dict(side='FREE_PEOPLES', deck=deck, pid=pid,
                    cid=cid or (STRATEGY if deck == 'strategy' else CHARACTER), name='fixture draw')

    def stage(self):
        self.cycle('$Wey moves Free Action Die (P) from FP Dice to Current Action.', pids=['210'])

    def test_strategy_observation_cannot_replace_preexisting_card_before_payment(self):
        self.stage()
        self.cycle('$Wey draws Free Strategy Card.', decoded=[self.identity('strategy')])
        self.assertEqual(self.driver.es.hands[self.human].hand, self.initial)
        self.assertEqual(self.driver.n_commits_total, 0)
        for _ in range(3):
            self.cycle()
        self.assertEqual(self.driver.es.hands[self.human].hand, self.initial)

    def test_character_observation_cannot_replace_preexisting_card_before_payment(self):
        self.stage()
        self.cycle('$Wey draws Free Character Card.', decoded=[self.identity('character', pid='144')])
        self.assertEqual(self.driver.es.hands[self.human].hand, self.initial)
        self.assertEqual(self.driver.n_commits_total, 0)

    def test_legal_draw_binds_new_card_and_preserves_both_old_cards(self):
        self.stage()
        self.cycle('$Wey draws Free Strategy Card.', decoded=[self.identity('strategy')])
        self.cycle('$Wey kills Free Action Die (P) in Current Action.', pids=['210'])
        self.assertEqual(self.driver.es.hands[self.human].hand, self.initial+[STRATEGY])
        self.assertFalse(self.driver.alarms)

    def test_decode_arriving_after_commit_binds_that_draw(self):
        self.stage()
        self.cycle('$Wey draws Free Strategy Card.')
        self.cycle('$Wey kills Free Action Die (P) in Current Action.', pids=['210'])
        self.cycle(decoded=[self.identity('strategy')])
        self.assertEqual(self.driver.es.hands[self.human].hand, self.initial+[STRATEGY])
        self.assertFalse(self.driver.alarms)

    def rejected_window(self, *, decoded=True):
        rows = [
            '$Wey moves Free Action Die (P) from FP Dice to Current Action.',
            '$Wey draws Free Strategy Card.',
            '$Wey draws Free Strategy Card.',
            '$Wey kills Free Action Die (P) in Current Action.',
        ]
        other = 'FREE_PEOPLES_Strategy_A_Power_too_Great'
        identities = [self.identity('strategy'), self.identity('strategy', other, '122')]
        self.cycle(rows[0], pids=['210'])
        for row, identity in zip(rows[1:3], identities):
            self.cycle(row, decoded=[identity] if decoded else ())
        output = self.cycle(rows[3], pids=['210'])
        self.assertEqual(sum('Controls.819' in line for line in output), 4, output)
        for row in reversed(rows):
            self.cycle('$WOTR_AI undoes last action : '+row)
        self.assertEqual(self.driver.es.hands[self.human].hand, self.initial)
        self.assertEqual(self.driver.n_commits_total, 0)
        return identities

    def test_rejected_draws_cannot_steal_the_next_legitimate_draw(self):
        old = self.rejected_window()
        for _ in range(31):
            self.cycle()
        replacement = 'FREE_PEOPLES_Strategy_Cirdan\'s_Ships'
        self.stage()
        self.cycle('$Wey draws Free Strategy Card.',
                   decoded=[self.identity('strategy', replacement, '123')])
        self.cycle('$Wey kills Free Action Die (P) in Current Action.', pids=['210'])
        self.assertEqual(self.driver.es.hands[self.human].hand, self.initial+[replacement])
        self.assertFalse(any(d['cid'] in self.driver.es.hands[self.human].hand for d in old))
        self.assertFalse(self.driver.alarms)

    def test_delayed_decoding_of_undone_draws_does_not_change_hand(self):
        old = self.rejected_window(decoded=False)
        self.cycle(decoded=old)
        for _ in range(31):
            self.cycle()
        self.assertEqual(self.driver.es.hands[self.human].hand, self.initial)
        self.assertFalse(self.driver.alarms)


if __name__ == '__main__':
    unittest.main()
