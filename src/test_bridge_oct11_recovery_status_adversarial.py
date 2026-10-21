"""A failed reset cleanup cannot acknowledge successful board recovery."""
from pathlib import Path
import importlib
import json
import pickle
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'wotr_bridge'), str(ROOT)]
import ai_panel
import test_bridge_oct11_combat_recovery as board_fixture
import test_bridge_oct11_recovery_status as status_fixture


class RecoveryAcknowledgement(unittest.TestCase):
    server = status_fixture.RecoveryStatusTests.server

    def test_real_adoption_followed_by_reset_cleanup_exception_is_not_success(self):
        server, _path = self.server()
        fixture = board_fixture.LiveNoncombatRecovery('runTest')
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        server.driver, server.rand = fixture.driver, fixture.rand
        self.assertTrue(server.driver._last_resync_adoption['complete'])
        undo_module = importlib.import_module(server.driver.__class__.__module__.rsplit('.', 1)[0]+'.undo_model')
        epoch = 2000000010.0
        # Exercise the actual full native board adoption. RestoreWatch is
        # constructed afterwards by reset_and_resync, outside adoption's
        # transaction, so this proves the caught late-exception path.
        with patch.object(undo_module, 'RestoreWatch', side_effect=RuntimeError('reset cleanup failed')), \
                patch.object(server, '_abandon_step', wraps=server._abandon_step) as abandon:
            server.on_control({'reset_resync': epoch})
            server.handle({'lines': ()})
        abandon.assert_not_called()
        self.assertTrue(any('reset failed:' in note for note in server.notes), server.notes[-4:])
        status = server.status_menu()
        self.assertFalse(status['recovery']['complete'], status['recovery'])
        text = ai_panel.note_text(status, '', 'RESYNC requested', epoch)
        self.assertIn('incomplete', text.lower())
        self.assertIn('could not finish', text.lower())
        self.assertNotIn('cleanup failed', text.lower())

    def test_real_choice_recovery_exception_cannot_publish_hidden_card_identity(self):
        server, _path = self.server()
        fixture = board_fixture.LiveNoncombatRecovery('runTest')
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        server.driver, server.rand = fixture.driver, fixture.rand
        server.driver.es.pending_hunt = dict(damage=2, phase='hunt_sacrifice', casualty_taken=False)
        before = pickle.dumps(server.driver.es)
        random_before = pickle.dumps(vars(server.rand))
        actor, turn = server.driver.last_actor, server.driver.es.turn
        choice_module = importlib.import_module(server.driver.__class__.__module__.rsplit('.', 1)[0]+'.choice_recovery')
        secret = 'SHADOW_Character_PRIVATE_TEST_CARD'
        with patch.object(choice_module, 'plan', side_effect=ValueError('hidden AI card '+secret)) as plan:
            server.on_control({'reset_resync': 2000000011.0})
            server.handle({'lines': ()})
        plan.assert_called_once()
        self.assertTrue(any(secret in note for note in server.notes))  # diagnostic only
        status = server.status_menu()
        self.assertFalse(status['recovery']['complete'])
        self.assertNotIn(secret, json.dumps(status))
        self.assertIn('could not finish', status['recovery']['summary'])
        self.assertTrue(pickle.dumps(server.driver.es) == before)
        self.assertTrue(pickle.dumps(vars(server.rand)) == random_before)
        self.assertEqual((server.driver.last_actor, server.driver.es.turn), (actor, turn))


if __name__ == '__main__':
    unittest.main()
