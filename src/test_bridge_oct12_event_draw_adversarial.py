"""A rejected human overdraw must not request cards for hypothetical actions.

Physical evidence: Gandalf_group_physical_r4/draw_failure. The human mistakenly
drew two Strategy cards with one Event die; the bridge then requested Character
and drew a third card while testing a different candidate. Rulebook PDF p21
allows one Event card from either deck for this action.
"""
import json
import unittest

import test_bridge_oct11_gandalf_group_live as gandalf


class EventDrawEvidence(unittest.TestCase):
    def setUp(self):
        self.fixture = gandalf.GandalfGroupedLive()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.driver = self.fixture.driver
        self.driver.enforce_human = True

    def draw_window(self, decks):
        f = self.fixture
        items = f.physical('$Wey moves Free Action Die (P) from FP Dice to Current Action.', ['210'])
        for index, deck in enumerate(decks):
            # Native private draw records deliberately redact identities. A
            # public invented card PID would weaken this reproduction.
            f.seq += 1
            tags = (['Controls.509', 'Controls.510', 'Game.2005', 'Controls.511', 'Controls.45']
                    if deck == 'Strategy' else
                    ['Controls.491', 'Controls.492', 'Game.2004', 'Controls.493', 'Controls.45'])
            meta = {'seq': f.seq, 'gen': 1, 'raw_new': True, 'nick': 'Wey',
                    'tags': tags, 'pids': 'redacted', 'side': 'FP'}
            with f.meta.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(meta)+'\n')
            items += f.rig.feed('$Wey draws Free '+deck+' Card.')
        items += f.physical('$Wey kills Free Action Die (P) in Current Action.', ['210'])
        return items

    def check_one(self, deck):
        before = len(self.driver.es.hands[gandalf.gs.Side.FREE_PEOPLES].hand)
        items = self.draw_window([deck])
        self.assertEqual(self.driver.n_commits_total, 1, (items, self.driver.findings))
        self.assertEqual(len(self.driver.es.hands[gandalf.gs.Side.FREE_PEOPLES].hand), before+1)
        self.assertEqual([d.used for d in self.driver.es.dice[gandalf.gs.Side.FREE_PEOPLES]],
                         [True, False, False, False])
        self.assertIsNone(self.driver.deferred)
        self.assertFalse(self.driver.alarms)

    def test_one_strategy_is_a_complete_event_die_action(self):
        self.check_one('Strategy')

    def test_one_character_is_a_complete_event_die_action(self):
        self.check_one('Character')

    def check_overdraw(self, decks):
        before = list(self.driver.es.hands[gandalf.gs.Side.FREE_PEOPLES].hand)
        items = self.draw_window(decks)
        self.assertEqual(self.driver.n_commits_total, 0)
        self.assertEqual(self.driver.es.hands[gandalf.gs.Side.FREE_PEOPLES].hand, before)
        self.assertFalse(self.driver.es.dice[gandalf.gs.Side.FREE_PEOPLES][0].used)
        self.assertFalse(any(kind == 'human_draw_asked' for kind, _ in self.driver.findings), items)
        self.assertFalse(any(item[0] == 'rng_ask' for item in items), items)
        self.assertFalse(any(item[0] == 'call' and 'draw_' in str(item[1:]) for item in items), items)
        self.assertTrue(any(item[0] == 'line' and 'Controls.819' in str(item[1:]) for item in items),
                        'The untouched, reversible native window should be undone')
        self.assertIsNone(self.driver.deferred, 'An impossible overdraw cannot await another random card')

    def test_two_strategy_cards_do_not_request_a_third_character_card(self):
        self.check_overdraw(['Strategy', 'Strategy'])

    def test_two_character_cards_do_not_request_a_third_strategy_card(self):
        self.check_overdraw(['Character', 'Character'])

    def test_mixed_overdraw_does_not_request_any_further_card(self):
        self.check_overdraw(['Character', 'Strategy'])


if __name__ == '__main__':
    unittest.main()
