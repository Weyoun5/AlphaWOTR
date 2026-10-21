"""Recovery feedback must reach the real panel status, including failure."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'wotr_bridge'), str(ROOT)]
import ai_panel
import game_state as gs
from live.live_server import LiveServer
from live.transport import FileBridge


class RecoveryStatusTests(unittest.TestCase):
    def server(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name)
        (path / 'ai_feed.txt').write_text('', encoding='utf8')
        server = LiveServer(str(path), gs.Side.SHADOW, load_nets=False, seed=81)
        return server, path

    def test_actual_reset_control_failure_is_published_persistently(self):
        server, path = self.server()
        epoch = 2000000000.0
        detail = "choice recovery unavailable: cannot pickle 'module' object; pending retained"
        with patch.object(server.driver, 'reset_and_resync', return_value=(
                [('note', detail)], ['resync incomplete; existing action retained'])):
            server.on_control({'reset_resync': epoch})
            server.handle({'lines': ()})
        bridge = FileBridge(str(path), 'recovery-status-test', status_extra=server.status_menu)
        bridge.write_status(state='idle', note='')
        status = json.loads((path / 'ai_bridge_status.json').read_text())
        recovery = status['recovery']
        self.assertEqual(recovery['epoch'], epoch)
        self.assertFalse(recovery['complete'])
        text = ai_panel.note_text(status, '', 'RESYNC requested', epoch)
        self.assertIn('RESYNC incomplete', text)
        self.assertIn('could not save its rollback state', text)
        # A heartbeat must not turn a failed result back into 'requested'.
        bridge.write_status(state='idle', note='')
        later = json.loads((path / 'ai_bridge_status.json').read_text())
        self.assertEqual(ai_panel.note_text(later, '', '', epoch), text)

    def test_previous_result_does_not_answer_a_newer_click(self):
        status = {'recovery': {'epoch': 10, 'summary': 'RESYNC complete'}}
        self.assertEqual(ai_panel.note_text(status, '', 'RESYNC requested', 11),
                         'RESYNC requested')
        status['recovery']['epoch'] = 11
        self.assertEqual(ai_panel.note_text(status, '', 'RESYNC requested', 11),
                         'RESYNC complete')
        self.assertEqual(ai_panel.note_text(status, '', 'skip-decision sent', 0),
                         'skip-decision sent')

    def test_actual_reset_control_success_is_acknowledged(self):
        server, path = self.server()
        epoch = 2000000001.0
        def complete():
            server.driver._last_resync_adoption = {'accepted': True, 'complete': True}
            return [], ['resynced to the client board', 'turn order retained']
        with patch.object(server.driver, 'reset_and_resync', side_effect=complete):
            server.on_control({'reset_resync': epoch})
            server.handle({'lines': ()})
        bridge = FileBridge(str(path), 'recovery-status-test', status_extra=server.status_menu)
        bridge.write_status(state='idle', note='')
        status = json.loads((path / 'ai_bridge_status.json').read_text())
        self.assertTrue(status['recovery']['complete'])
        self.assertIn('RESYNC complete', ai_panel.note_text(status, '', 'RESYNC requested', epoch))

    def test_recovery_expires_after_gameplay_and_on_foreign_server(self):
        server, path = self.server()
        server._panel_recovery_result = {'epoch': 10, 'complete': True,
            'summary': 'RESYNC complete', 'commits': server.driver.n_commits_total}
        bridge = FileBridge(str(path), 'recovery-status-test', status_extra=server.status_menu)
        bridge.write_status(state='idle')
        self.assertTrue(json.loads((path / 'ai_bridge_status.json').read_text())['recovery'])
        server.driver._snapshot([], 'subsequent legal action')
        bridge.write_status(state='idle')
        self.assertEqual(json.loads((path / 'ai_bridge_status.json').read_text())['recovery'], {})
        stale = {'pid': -1, 'recovery': {'epoch': 10, 'summary': 'old failed run'}}
        (path / 'ai_bridge_status.json').write_text(json.dumps(stale))
        bridge.status_extra = None
        bridge.write_status(state='starting')
        self.assertNotIn('recovery', json.loads((path / 'ai_bridge_status.json').read_text()))


if __name__ == '__main__':
    unittest.main()
