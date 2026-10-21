"""Actual RA recovery and server controls report the retained transaction."""
from pathlib import Path
import copy
import importlib.util
import json
import pickle
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'audit/bridge_oct11/ra_resync_checkpoint'
sys.path[:0] = [str(AUDIT), str(ROOT / 'src'), str(ROOT / 'wotr_bridge'), str(ROOT)]
spec = importlib.util.spec_from_file_location('ra_reporting_checkpoint', AUDIT / 'test_checkpoint.py')
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
from live.live_server import LiveServer
from live import repair_evidence


class CombatRecoveryReporting(unittest.TestCase):
    def rig(self, count):
        case = fixtures.RACheckpointTests()
        self.addCleanup(case.doCleanups)
        driver, path, dump, markers, manifest = case.rig(count)
        server = LiveServer(str(path.parent), driver.ai_side, load_nets=False)
        server.driver, server.rand = driver, driver.rand
        server.rand.install()
        return case, server, driver, path, dump, markers, manifest

    def reset(self, server, epoch):
        server.on_control({'reset_resync': epoch})
        out = server.handle({'lines': ()})
        return out, server.status_menu()['recovery']

    def test_partial_payment_and_repeat_report_reconciled_with_pending_choice(self):
        _, server, driver, *_ = self.rig(1)
        old = driver.n_commits_total
        actor, turn = driver.last_actor, driver.es.turn
        for epoch in (2000000001, 2000000002):
            _, status = self.reset(server, epoch)
            self.assertTrue(status['complete'], status)
            self.assertEqual(status['waiting_for'], 'combat_step')
            self.assertIn('board reconciled; combat and turn order retained', status['summary'])
            self.assertIn('1 remaining hit', server._waiting_on())
            cursor = driver.es.pending_combat['_round_resume']['cursor']
            self.assertEqual((cursor.ra_paid, cursor.ra_remaining), (1, 1))
            self.assertEqual(driver.n_commits_total, old+1)
            self.assertEqual((driver.last_actor, driver.es.turn), (actor, turn))
            self.assertEqual(server._live_asks, {})
            self.assertTrue(driver._last_reset_preserved_combat)
            self.assertIs(driver.seg.window, driver._chat_attack_window)

    def test_complete_payment_and_repeat_report_existing_roll_without_committing(self):
        _, server, driver, *_ = self.rig(2)
        before = pickle.dumps(driver.es)
        _, status = self.reset(server, 2000000001)
        self.assertFalse(status['complete'])
        self.assertEqual(status['waiting_for'], 'client_roll')
        self.assertIn('waiting for the existing client roll', status['summary'])
        self.assertNotIn('could not finish', status['summary'])
        asks = copy.deepcopy(server._live_asks)
        self.assertEqual([entry['need'] for entry in asks.values()], [3])
        _, repeated = self.reset(server, 2000000002)
        self.assertEqual(repeated['waiting_for'], 'client_roll')
        self.assertEqual(server._live_asks, asks)
        self.assertEqual(pickle.dumps(driver.es), before)
        self.assertEqual(len(driver.pending_buf.records), 2)

    def test_real_regular_payment_wait_uses_proved_three_not_alternative_four(self):
        case, server, driver, path, dump, markers, _ = self.rig(1)
        self.reset(server, 2000000001)
        # The remaining hit can legally remove a Regular (3 dice) or reduce
        # an Elite (4 dice). This actual native Regular move proves the first.
        line = '$Wey moves Isengard Regular from Orthanc to SA Reinforcements.'
        with Path(driver.feed.path).open('a') as stream:
            stream.write(line+'\n')
        with Path(driver.sidecar.path).open('a') as stream:
            stream.write(json.dumps({'seq': driver._feed_seq+1, 'gen': 1,
                                     'pids': ['003'], 'nick': 'Wey'})+'\n')
        dump['003'] = 'SA Reinforcements'
        case.write_dump(path, dump, markers, driver._feed_seq+1)
        server.handle({'lines': ()})
        proved = driver.pending_random_call()
        self.assertEqual(proved.kw['need'], 3)
        self.assertEqual(server.rand.pending().kw['need'], 4,
                         'This must reproduce the real competing-candidate funnel residue')
        asks = copy.deepcopy(server._live_asks)
        self.assertEqual([entry['need'] for entry in asks.values()], [3])
        server._stalled = 25
        text = server._waiting_on()
        self.assertIn('Shadow (you) - normal combat roll: 3 dice', text)
        self.assertNotIn('4 dice', text)
        self.assertNotIn('[stalled', text)
        server._mirror_last = (None, 0)
        server._write_mirror_snapshot()
        live = json.loads((path.parent / 'ai_live_status.json').read_text())
        self.assertEqual(live['last'], text)
        self.assertEqual(server._live_asks, asks)

    def test_failed_commit_never_reports_reconciled(self):
        _, server, driver, *_ = self.rig(1)
        before = pickle.dumps(driver.es)
        with patch.object(driver, '_commit_chain', return_value=False):
            _, status = self.reset(server, 2000000001)
        self.assertFalse(status['complete'])
        self.assertIsNone(status['waiting_for'])
        self.assertIn('incomplete', status['summary'])
        self.assertEqual(pickle.dumps(driver.es), before)

    def test_unstable_dump_never_reports_reconciled(self):
        _, server, _, *_ = self.rig(1)
        with patch.object(repair_evidence, 'same_dump', return_value=False):
            _, status = self.reset(server, 2000000001)
        self.assertFalse(status['complete'])
        self.assertIsNone(status['waiting_for'])

    def test_wrong_side_reserve_clears_previous_success(self):
        case, server, driver, path, dump, markers, _ = self.rig(1)
        self.assertTrue(self.reset(server, 2000000001)[1]['complete'])
        dump['003'] = 'FP Reinforcements'
        case.write_dump(path, dump, markers, driver._feed_seq)
        _, status = self.reset(server, 2000000002)
        self.assertFalse(status['complete'])
        self.assertIsNone(status['waiting_for'])


if __name__ == '__main__':
    unittest.main()
