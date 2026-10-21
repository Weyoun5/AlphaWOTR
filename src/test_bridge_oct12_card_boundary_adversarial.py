"""Independent completion-boundary probes over captured public native gestures."""
import json
import unittest

import test_bridge_oct12_rage_split_live as fixture


class CardBoundaryAdversarial(unittest.TestCase):
    def setUp(self):
        self.case = fixture.RageSplitLive()
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.case.recruit()
        self.base_commits = self.case.driver.n_commits_total

    def literal(self, line, pids=(), moves=None, driver_only=False):
        c = self.case
        c.seq += 1
        with c.feed.open('a', encoding='utf8') as stream:
            stream.write(line+'\n')
        with c.meta_path.open('a', encoding='utf8') as stream:
            stream.write(json.dumps({'seq': c.seq, 'gen': 1,
                'nick': 'Wey', 'pids': list(pids)})+'\n')
        if moves:
            c.dump.update(moves)
        c.write_dump()
        return c.driver.tick() if driver_only else c.server.handle({'lines': ()})

    def test_grouped_card_and_die_disposal_commits_whole_split_once(self):
        c = self.case
        c.native(103)
        c.native(104)
        self.assertEqual(c.driver.n_commits_total, self.base_commits)
        self.literal('$Wey kills group (Shadow Action Die (M), Rage of the Dunlendings/Relentless Assault) in Current Action.',
            ('102', '185'), {'102': 'SA Casualties', '185': 'SA Casualties'}, driver_only=True)
        self.assertEqual(c.regulars('gap of rohan'), 4)
        self.assertEqual(c.regulars('south dunland'), 0)
        self.assertEqual(c.driver.n_commits_total, self.base_commits+1)
        self.assertIsNone(fixture.ai.describe_pending(c.driver.es))
        c.assert_native_census()

    def test_idle_and_prefix_expiry_do_not_invent_completion(self):
        c = self.case
        c.native(103)
        for _ in range(3):
            c.server.handle({'lines': ()})
        c.driver._try_pending_prefixes()
        self.assertEqual(c.driver.n_commits_total, self.base_commits)
        self.assertIsNotNone(fixture.ai.describe_pending(c.driver.es))
        self.assertEqual(c.regulars('gap of rohan'), 2)
        c.native(104, 107)
        self.assertEqual(c.regulars('gap of rohan'), 4)
        self.assertEqual(c.driver.n_commits_total, self.base_commits+1)

    def test_an_unrelated_discard_does_not_close_rage(self):
        c = self.case
        c.native(103)
        self.literal('$Wey kills Pits of Mordor/Desperate Battle in Current SA Card.',
            ('190',), driver_only=True)
        self.assertEqual(c.driver.n_commits_total, self.base_commits)
        self.assertIsNotNone(fixture.ai.describe_pending(c.driver.es))

    def test_corpus_matcher_still_accepts_complete_historical_gesture(self):
        c = self.case
        c.native(103, 104)
        from live import gesture_match
        dp = fixture.ai.describe_pending(c.driver.es)
        res = gesture_match.match_window(c.driver.es, dp['side'], c.driver.pending_buf,
            menu=dp['actions'], pin=c.driver._attempt_pin(), rng=c.driver.rctl,
            advance_tolerance=False, fs_prefilter=False)
        self.assertEqual(res.status, 'match', res.note)
        # Pure probing must not consume the live player's unfinished choice.
        self.assertEqual(c.driver.n_commits_total, self.base_commits)
        self.assertIsNotNone(fixture.ai.describe_pending(c.driver.es))


if __name__ == '__main__':
    unittest.main()
