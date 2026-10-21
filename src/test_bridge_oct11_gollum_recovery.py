"""Board-proved Hunt payment when Gollum guides outside the companion list."""
import pickle
import unittest

import test_bridge_oct11_combat_recovery as fixtures
import log_replay


class GollumRecovery(unittest.TestCase):
    write_dump = fixtures.LiveNoncombatRecovery.write_dump

    def setUp(self):
        fixtures.LiveNoncombatRecovery.setUp(self)
        state = self.driver.es
        state.fellowship.companions.clear()
        state.fellowship.guide = state.gollum
        for pid in log_replay.COMPANIONS:
            self.dump[pid] = 'Guide Box' if pid == '364' else 'FP Dead'
        state.pending_hunt = {'damage': 2, 'phase': 'hunt_gollum'}
        self.marker_states = {'247': 0}
        self.write_dump()

    def test_full_visible_payment_proves_gollum_no_reveal_once(self):
        self.dump['252'] = 'Track 2'
        self.write_dump()
        native_results = pickle.dumps((self.rand._feed, self.rand._feed_pos))
        items, report = self.driver.reset_and_resync()
        self.assertIsNone(self.driver.es.pending_hunt, (items, report))
        self.assertEqual(self.driver.es.fellowship.corruption, 2)
        self.assertTrue(self.driver.es.fellowship.hidden)
        self.assertEqual(self.driver.es.fellowship.guide.name, 'Gollum')
        self.assertEqual(self.driver.es.fellowship.companions, [])
        self.assertEqual(self.driver.n_commits_total, 1)
        self.assertEqual(pickle.dumps((self.rand._feed, self.rand._feed_pos)), native_results)
        self.assertIsNone(self.rand._pending)
        self.assertFalse(any(item[0] == 'rng_ask' for item in items))
        self.driver.reset_and_resync()
        self.assertEqual(self.driver.n_commits_total, 1)

    def test_partial_damage_without_reveal_does_not_choose_for_human(self):
        self.dump['252'] = 'Track 1'
        self.write_dump()
        before = pickle.dumps(self.driver.es)
        self.driver.reset_and_resync()
        self.assertEqual(pickle.dumps(self.driver.es), before)
        self.assertEqual(self.driver.n_commits_total, 0)

    def test_unchanged_board_does_not_mean_declining_gollum_reveal(self):
        before = pickle.dumps(self.driver.es)
        self.driver.reset_and_resync()
        self.assertEqual(pickle.dumps(self.driver.es), before)
        self.assertEqual(self.driver.n_commits_total, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
