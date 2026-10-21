"""Independent physical-emission checks for typed board-recovery records."""
import unittest

import test_bridge_oct11_combat_recovery as fixtures
import game_state as gs
import log_replay


class ReconstructedEmissions(unittest.TestCase):
    write_dump = fixtures.LiveNoncombatRecovery.write_dump
    companion_pid = fixtures.LiveNoncombatRecovery.companion_pid

    def setUp(self):
        fixtures.LiveNoncombatRecovery.setUp(self)

    def prepare_capture(self, physical_box, *, active=True, physical_face=1):
        """The legal post-battle entry captures Orthanc and advances Isengard."""
        state = self.driver.es
        state.armies.pop('orthanc')
        for pid in self.shadow_regulars:
            self.dump[pid] = 'SA Reinforcements'
        state.pending_combat = {
            'phase': 'victor_advance',
            'attacker_region': 'fords of isen',
            'defender_region': 'orthanc',
            'attacker_side': gs.Side.FREE_PEOPLES,
            'defender_side': gs.Side.SHADOW,
            'deciding_side': gs.Side.FREE_PEOPLES,
            'is_siege': False,
            'rounds_fought': 1,
        }
        state.political.boxes[gs.Nation.ISENGARD] = 1
        state.political.active[gs.Nation.ISENGARD] = active
        self.political_pid = next(str(pid) for pid, nation in
                                  log_replay.POLITICAL_MARKERS.items()
                                  if nation is gs.Nation.ISENGARD)
        for pid, zone in list(self.dump.items()):
            if zone.lower() == 'fords of isen':
                self.dump[pid] = 'Orthanc'
        # Decoded labels count distance to war; native Game.2000..2003 and
        # engine boxes count advancement in the opposite direction.
        self.dump[self.political_pid] = 'Political %d' % (3 - physical_box)
        self.marker_states = {self.political_pid: physical_face}
        self.write_dump()
        from wotr_bridge.live import client_board
        measured = client_board.read_board(str(self.dump_path))
        self.assertEqual(measured[self.political_pid], 'Political %d' % (3 - physical_box))
        rows = self.dump_path.read_text(encoding='utf-8').splitlines()
        marker_row = next(row for row in rows[1:]
                          if self.political_pid in row.split('\t')[2].split())
        self.assertIn('<~Game.%d~>' % (2000 + physical_box), marker_row)

    def recover_capture(self, physical_box, **kwargs):
        self.prepare_capture(physical_box, **kwargs)
        items, report = self.driver.reset_and_resync()
        self.assertEqual(self.driver.n_commits_total, 1, report)
        self.assertEqual(self.driver.es.political.boxes[gs.Nation.ISENGARD], 2)
        self.assertIn('orthanc', self.driver.es.enemy_controlled[gs.Side.FREE_PEOPLES])
        self.assertIsNone(self.driver.es.pending_combat)
        return items

    def political_lines(self, items):
        return [item[1] for item in items if item[0] == 'line'
                and '<~HighlightLabel.' in item[1]
                and self.political_pid in item[1]]

    def test_recovered_capture_does_not_repeat_already_visible_political_step(self):
        items = self.recover_capture(physical_box=2)
        self.assertEqual(self.political_lines(items), [])
        self.assertFalse(any(key == 'political_sweep_emitted' and value == 'ISENGARD 1->2'
                             for key, value in self.driver.findings))

    def test_recovered_capture_emits_political_step_still_absent_from_board(self):
        items = self.recover_capture(physical_box=1)
        lines = self.political_lines(items)
        self.assertEqual(len(lines), 1, lines)
        self.assertIn('<~Game.2001~>', lines[0])
        self.assertIn('<~Game.2002~>', lines[0])

    def test_visible_political_step_still_requires_its_missing_activation_flip(self):
        items = self.recover_capture(physical_box=2, active=False, physical_face=0)
        self.assertTrue(self.driver.es.political.active[gs.Nation.ISENGARD])
        self.assertEqual(self.political_lines(items), [])
        flips = [item[1] for item in items if item[0] == 'line'
                 and '<~Controls.323~>%s' % self.political_pid in item[1]]
        self.assertEqual(len(flips), 1, flips)

    def test_ordinary_ai_sweep_does_not_credit_board_recovery_without_transaction(self):
        self.prepare_capture(physical_box=2)
        self.driver.view.refresh_dump()
        proof = fixtures.recovery.plan(self.driver, self.dump)
        self.assertEqual(proof.status, 'match', proof.note)
        self.assertTrue(any(record.get('_combat_recovery') for record in proof.window.records))
        self.driver.human_side = gs.Side.SHADOW
        self.driver.ai_side = gs.Side.FREE_PEOPLES
        self.assertFalse(getattr(self.driver, '_resync_commit_transaction', False))
        items = self.driver.post_commit_sweep(
            self.driver.es, proof.match.post, gesture=proof.window,
            actions=proof.match.chain, label='ai:combat:victor_advance')
        lines = self.political_lines(items)
        self.assertEqual(len(lines), 1, lines)
        self.assertIn('<~Game.2001~>', lines[0])
        self.assertIn('<~Game.2002~>', lines[0])

    def test_gollum_payment_does_not_repeat_visible_corruption_drag(self):
        state = self.driver.es
        state.fellowship.companions.clear()
        state.fellowship.guide = state.gollum
        state.pending_hunt = {'damage': 2, 'phase': 'hunt_gollum'}
        for pid in log_replay.COMPANIONS:
            self.dump[pid] = 'Guide Box' if pid == '364' else 'FP Dead'
        self.dump['252'] = 'Track 2'
        self.marker_states = {'247': 0}
        self.write_dump()
        items, report = self.driver.reset_and_resync()
        self.assertEqual(self.driver.n_commits_total, 1, report)
        self.assertIsNone(self.driver.es.pending_hunt)
        self.assertFalse(any(item[0] == 'line' for item in items), items)
        self.assertNotIn('252', getattr(self.driver, '_track_owed', {}))

    def test_reconstructed_casualty_does_not_create_another_physical_debt(self):
        self.driver.es.pending_combat = fixtures.state_and_driver()[0].pending_combat
        self.dump[self.fp_regulars[0]] = 'FP Dead'
        self.write_dump()
        items, report = self.driver.reset_and_resync()
        self.assertEqual(self.driver.n_commits_total, 1, report)
        self.assertEqual(self.driver.es.fp_casualties[gs.Nation.ROHAN]['regular'], 1)
        self.assertFalse(any(item[0] == 'line' for item in items), items)
        self.assertEqual(self.driver.echo_ledger, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
