"""The panel must describe the same partial Hunt choice as the player chat.

Uses the real captured opening and the real server status method. The actual
Java Hunt smoke independently checks this boundary in the composed application.
"""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

import test_audit0904_bridge_followup as capture
from wotr_bridge.live.live_server import LiveServer
from game_state import Side


if os.environ.get('WOTR_AUDIT_STATUS_BASELINE'):
    spec = importlib.util.spec_from_file_location(
        'wotr_bridge.live._audit_status_before', os.environ['WOTR_AUDIT_STATUS_BASELINE'])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    LiveServer = module.LiveServer


class PartialHuntStatusTests(unittest.TestCase):
    # Share the captured-input setup, without inheriting/counting its tests again.
    rig = capture.RecordedHuntRecoveryTests.rig
    feed = staticmethod(capture.RecordedHuntRecoveryTests.feed)
    replay_to_hunt = capture.RecordedHuntRecoveryTests.replay_to_hunt

    def server_at_partial(self):
        driver, feed, random = self.replay_to_hunt()
        self.feed(driver, feed, [capture.PARTIAL])
        scratch = tempfile.TemporaryDirectory(prefix='wotr-hunt-status-')
        self.addCleanup(scratch.cleanup)
        server = LiveServer(scratch.name, Side.SHADOW, load_nets=False, seed=7)
        server.driver, server.rand = driver, random
        return server, driver, feed, random

    def test_panel_explains_partial_damage_and_completion_without_mutation(self):
        server, driver, _, _ = self.server_at_partial()
        commits, findings = driver.n_commits_total, list(driver.findings)
        for _ in range(3):
            # FileBridge persists only the first 140 characters.
            status = server._waiting_on()[:140]
            self.assertIn('2 damage', status)
            self.assertIn('from 1 to 2', status)
            self.assertIn('say corruption', status)
            self.assertNotIn('match none', status)
        self.assertEqual(driver.es.fellowship.corruption, 0)
        self.assertEqual(driver.es.pending_hunt['damage'], 2)
        self.assertEqual(driver.n_commits_total, commits)
        self.assertEqual(driver.findings, findings)

    def test_completed_hunt_no_longer_reports_partial_status(self):
        server, driver, feed, _ = self.server_at_partial()
        self.feed(driver, feed, [capture.COMPLETE])
        self.assertEqual(driver.es.fellowship.corruption, 2)
        self.assertIsNone(driver.es.pending_hunt)
        self.assertNotIn('from 1 to 2', server._waiting_on())

    def test_actual_requested_random_draw_keeps_status_priority(self):
        server, driver, feed, random = self.server_at_partial()
        out = self.feed(driver, feed, ['<Wey> random'])
        self.assertEqual([i[1].kind for i in out if i[0] == 'rng_ask'], ['randomcomp'])
        self.assertEqual(random.pending().kind, 'randomcomp')
        self.assertIn('client randomcomp', server._waiting_on())
        self.assertNotIn('from 1 to 2', server._waiting_on())


if __name__ == '__main__':
    unittest.main(verbosity=2)
