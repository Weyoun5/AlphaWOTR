"""Real cold combat recovery must emit from measured political marker boxes."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
import re
import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_bridge_oct11_resync_emission_independent as fixtures
import game_state as gs


class RecoveredPoliticalSources(unittest.TestCase):
    def fixture(self, box, face=1):
        case = fixtures.ReconstructedEmissions()
        case.setUp()
        self.addCleanup(case.doCleanups)
        case.prepare_capture(box, active=True, physical_face=face)
        # The native runtime has an ACK channel. Repeated clicks before
        # injection must retain the emitted tail, not enter legacy adoption.
        case.driver.ack_path = str(case.dump_path.parent / 'ai_inject_ack.jsonl')
        Path(case.driver.ack_path).write_text('', encoding='utf8')
        return case

    def marker_moves(self, items, pid='263'):
        rows = []
        for item in items:
            if item[0] != 'line' or not re.search(r'~>%s\s*<~' % pid, item[1]):
                continue
            zones = re.findall(r'<~Game\.(\d+)~>', item[1])
            if zones:
                self.assertEqual(len(zones), 2, item[1])
                rows.append(tuple(map(int, zones)))
        return rows

    def acknowledge_native_operations(self, case, items):
        """Simulate only the emitted operations and their ACKs, offline."""
        import board
        import emitter
        from bridge_evidence_fixture import publish
        response = [item[1] for item in items if item[0] == 'line']
        header = publish(case.driver, response)
        acks = []
        for item in items:
            if item[0] != 'line':
                continue
            raw = item[1]
            facts = case.driver._op_line_facts(raw)
            if facts is None:
                continue
            op, pids, target = facts
            _, fields = emitter.classify_line(emitter.strip_tail(raw))
            for pid in pids:
                if op == 'flip':
                    case.marker_states[pid] = 1 - case.marker_states[pid]
                else:
                    measured = emitter.zone_code_for(board.sanitize(case.dump[pid]))
                    self.assertEqual(measured, fields['frm'], (pid, fields))
                    self.assertIsNotNone(target, raw)
                    case.dump[pid] = target
            acks.append(dict(raw=raw, ok=True, paragraphs_before=len(acks),
                             paragraphs_after=len(acks)+1, i=response.index(raw), **header))
        case.write_dump()
        Path(case.driver.ack_path).write_text(
            ''.join(json.dumps(row)+'\n' for row in acks), encoding='utf8')
        self.assertEqual(case.driver._consume_inject_acks(), [])
        self.assertFalse(case.driver.echo_pending())

    def test_capped_capture_corrects_all_boxes_and_survives_ack_then_repeat(self):
        for box in range(4):
            with self.subTest(box=box):
                case = self.fixture(box)
                case.driver.es.political.boxes[gs.Nation.ISENGARD] = 3
                actor, turn = case.driver.last_actor, case.driver.es.turn
                items, report = case.driver.reset_and_resync()
                current = 2000 + box
                for source, target in self.marker_moves(items):
                    self.assertEqual(source, current, (items, report))
                    current = target
                self.assertEqual(current, 2003, (items, report))
                self.assertEqual(case.driver.n_commits_total, 1)
                before_ack, _ = case.driver.reset_and_resync()
                self.assertEqual(self.marker_moves(before_ack), [])
                self.acknowledge_native_operations(case, items)
                for _ in range(2):
                    again, report = case.driver.reset_and_resync()
                    self.assertFalse(any(item[0] == 'line' for item in again), again)
                    self.assertTrue(case.driver._last_resync_adoption['complete'], report)
                    self.assertEqual(case.driver.es.political.boxes[gs.Nation.ISENGARD], 3)
                    self.assertEqual(case.driver.n_commits_total, 1)
                    self.assertEqual((case.driver.last_actor, case.driver.es.turn), (actor, turn))
                    self.assertIsNone(case.driver.rand.pending())

    def test_all_physical_boxes_emit_only_from_native_source_to_rules_target(self):
        for box in range(4):
            with self.subTest(box=box):
                case = self.fixture(box)
                actor, turn = case.driver.last_actor, case.driver.es.turn
                items, report = case.driver.reset_and_resync()
                moves = self.marker_moves(items)
                current = 2000 + box
                for source, target in moves:
                    self.assertEqual(source, current, (moves, report))
                    current = target
                self.assertEqual(current, 2002, (moves, report))
                self.assertEqual(case.driver.es.political.boxes[gs.Nation.ISENGARD], 2)
                self.assertEqual(case.driver.n_commits_total, 1)
                self.assertEqual((case.driver.last_actor, case.driver.es.turn), (actor, turn))
                self.assertIsNone(case.driver.rand.pending())
                # An unapplied correction is already in flight: no second
                # battle advance or physical marker command on another click.
                again, _ = case.driver.reset_and_resync()
                self.assertEqual(self.marker_moves(again), [])
                self.assertEqual(case.driver.n_commits_total, 1)
                self.assertEqual(case.driver.es.political.boxes[gs.Nation.ISENGARD], 2)
                case.dump['263'] = 'Political 1'
                case.write_dump()
                settled, _ = case.driver.reset_and_resync()
                self.assertEqual(self.marker_moves(settled), [])
                self.assertEqual(case.driver.n_commits_total, 1)

    def rohan_capture_fixture(self, engine_box=1):
        case = self.fixture(1)
        state = case.driver.es
        state.armies['westemnet'] = gs.Army('westemnet', [
            gs.Unit(gs.Nation.ISENGARD, gs.UnitKind.REGULAR),
            gs.Unit(gs.Nation.ISENGARD, gs.UnitKind.REGULAR)], [])
        state.pending_combat.update(attacker_region='westemnet', defender_region='helm s deep',
                                   attacker_side=gs.Side.SHADOW, defender_side=gs.Side.FREE_PEOPLES,
                                   deciding_side=gs.Side.SHADOW)
        state.political.boxes[gs.Nation.ROHAN] = engine_box
        state.political.active[gs.Nation.ROHAN] = True
        case.driver.human_side, case.driver.ai_side = gs.Side.SHADOW, gs.Side.FREE_PEOPLES
        # Restore the FP figures that prepare_capture moved to Orthanc.
        for pid, zone in list(case.dump.items()):
            if zone == 'Orthanc':
                case.dump[pid] = 'Fords of Isen'
        for pid in case.shadow_regulars:
            case.dump[pid] = "Helm's Deep"
        case.dump['259'] = 'Political 0'
        case.marker_states = {'259': 0}
        case.write_dump()
        return case

    def test_pre_active_engine_still_corrects_proved_passive_native_face_once(self):
        case = self.rohan_capture_fixture()
        items, _ = case.driver.reset_and_resync()
        flips = lambda rows: [item for item in rows if item[0] == 'line'
                             and '<~Controls.323~>259' in item[1]]
        self.assertEqual(len(flips(items)), 1, items)
        self.assertEqual(self.marker_moves(items, '259'), [(2003, 2002)])
        self.assertTrue(case.driver.es.political.active[gs.Nation.ROHAN])
        again, _ = case.driver.reset_and_resync()
        self.assertEqual(flips(again), [])
        self.assertEqual(case.driver.n_commits_total, 1)

    def test_capped_capture_corrects_passive_native_face_then_ack_is_complete(self):
        case = self.rohan_capture_fixture(engine_box=3)
        items, report = case.driver.reset_and_resync()
        flips = [item for item in items if item[0] == 'line'
                 and '<~Controls.323~>259' in item[1]]
        self.assertEqual(len(flips), 1, (items, report))
        self.assertEqual(self.marker_moves(items, '259'), [])
        self.assertEqual(case.driver.n_commits_total, 1)
        self.acknowledge_native_operations(case, items)
        again, report = case.driver.reset_and_resync()
        self.assertTrue(case.driver._last_resync_adoption['complete'], report)
        self.assertFalse(any(item[0] == 'line' for item in again), again)
        self.assertEqual(case.driver.es.political.boxes[gs.Nation.ROHAN], 3)
        self.assertTrue(case.driver.es.political.active[gs.Nation.ROHAN])
        self.assertEqual(case.driver.n_commits_total, 1)

    def test_stale_or_incomplete_census_cannot_emit_or_commit_correction(self):
        for engine_box in (1, 3):
            for mode in ('stale', 'incomplete'):
                with self.subTest(engine_box=engine_box, mode=mode):
                    case = self.fixture(2)
                    case.driver.es.political.boxes[gs.Nation.ISENGARD] = engine_box
                    case.write_dump(**({'feed_position': 99} if mode == 'stale' else {'complete': False}))
                    items, _ = case.driver.reset_and_resync()
                    self.assertEqual(self.marker_moves(items), [])
                    self.assertEqual(case.driver.n_commits_total, 0)
                    self.assertEqual(case.driver.es.political.boxes[gs.Nation.ISENGARD], engine_box)
                    self.assertFalse(case.driver._last_resync_adoption['complete'])

    def test_missing_affected_marker_does_not_guess_an_emission_source(self):
        case = self.fixture(3)
        del case.dump['263']
        case.write_dump()
        items, _ = case.driver.reset_and_resync()
        self.assertEqual(self.marker_moves(items), [])
        self.assertFalse(case.driver._last_resync_adoption['complete'])

    def test_unrelated_political_drift_is_not_a_recovered_combat_consequence(self):
        for engine_box in (1, 3):
            with self.subTest(engine_box=engine_box):
                case = self.fixture(2)
                case.driver.es.political.boxes[gs.Nation.ISENGARD] = engine_box
                old_rohan = case.driver.es.political.boxes[gs.Nation.ROHAN]
                case.dump['259'] = 'Political 0' if old_rohan != 3 else 'Political 3'
                case.write_dump()
                items, _ = case.driver.reset_and_resync()
                self.assertEqual(self.marker_moves(items, '259'), [])
                self.assertEqual(case.driver.es.political.boxes[gs.Nation.ROHAN], old_rohan)

    def test_existing_control_is_not_a_new_capture_proof(self):
        case = self.fixture(2)
        case.driver.es.political.boxes[gs.Nation.ISENGARD] = 3
        case.driver.es.enemy_controlled[gs.Side.FREE_PEOPLES].add('orthanc')
        case.dump['405'] = 'Orthanc'
        case.dump['422'] = 'Track 2'
        case.write_dump()
        items, _ = case.driver.reset_and_resync()
        self.assertEqual(case.driver.n_commits_total, 1)
        self.assertEqual(self.marker_moves(items), [])
        self.assertFalse(case.driver._last_resync_adoption['complete'])

    def test_ordinary_manual_adoption_still_uses_measured_political_box(self):
        case = self.fixture(2)
        case.driver.es.political.boxes[gs.Nation.ISENGARD] = 3
        case.driver.es.pending_combat = None
        items, report = case.driver.reset_and_resync()
        self.assertTrue(case.driver._last_resync_adoption['complete'], report)
        self.assertEqual(case.driver.es.political.boxes[gs.Nation.ISENGARD], 2)
        self.assertEqual(self.marker_moves(items), [])
        self.assertEqual(case.driver.n_commits_total, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
