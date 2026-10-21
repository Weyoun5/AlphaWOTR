"""Captured Morannon return-before-seat gestures must stop at the AI choice."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import pickle
import shutil
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'wotr_bridge'), str(ROOT)]
import ai
import game_logic as gl
import game_state as gs
import test_live_driver as td
from wotr_bridge.live import gesture_match as gm, live_parser
from wotr_bridge.live.driver import PhaseWindow
from wotr_bridge.live import client_board, client_protocol, scenario

CAPTURE = ROOT/'audit/bridge_oct11/player/Morannon_original_gesture_pinned'
RETURN = '$Wey moves Gondor Regular from Morannon to FP Reinforcements.'
SEAT = '$Wey moves group ([5]Gondor Regular) from Morannon to Morannon Stronghold.'


class SiegeWithdrawalPrefix(unittest.TestCase):
    def rig(self):
        spec = importlib.util.spec_from_file_location('oct11_cold_setup',
                    ROOT/'audit/bridge_oct11/player/cold_setup.py')
        setup = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(setup)
        manifest = json.loads((CAPTURE/'setup_manifest.json').read_text())
        fixture = ROOT/'audit/bridge_oct11/player/scenarios/captured_morannon'
        stub = scenario.replay_prefix(str(fixture/'setup_feed.txt'), manifest['base_cut'],
                        'SHADOW', meta_path=str(fixture/'setup_feed.meta.jsonl'), seed=42)
        setup.apply_position(stub, manifest)
        rig = td.Rig(stub.es, gs.Side.SHADOW, phase='actions')
        self.addCleanup(shutil.rmtree, rig.td)
        rig.drv.last_actor = gs.Side.SHADOW
        rig.drv.view.dump_path = str(CAPTURE/'ai_client_board.txt')
        self.assertTrue(rig.drv.view.refresh_dump())
        self.assertEqual(ai.describe_pending(rig.drv.es)['kind'], 'combat:defender_siege_choice')
        self.assertEqual(len(rig.drv.es.armies['dagorlad'].units), 10)
        self.assertEqual(len(rig.drv.es.armies['dagorlad'].leaders), 1)
        self.assertEqual(len(rig.drv.es.armies['morannon'].units), 6)
        return rig

    def choose(self, rig, kind):
        dp = ai.describe_pending(rig.drv.es)
        self.assertIs(dp['side'], gs.Side.SHADOW)
        actions = [a for a in dp['actions'] if a.action_type == kind]
        if kind == 'victor_advance':
            actions = [a for a in actions if a.params.get('n_regular') == 9
                       and a.params.get('n_elite') == 1
                       and a.params.get('n_nazgul') == 1]
        self.assertTrue(actions, [(a.action_type, a.params) for a in dp['actions']])
        self.assertTrue(rig.drv._commit_chain([actions[0]], [], 'ai:combat:victor_advance'))
        return actions[0]

    def assert_ai_boundary(self, rig):
        dp = ai.describe_pending(rig.drv.es)
        self.assertEqual(dp['kind'], 'combat:victor_advance',
                         (dp['kind'], rig.drv._last_pending_match.note))
        self.assertIs(dp['side'], gs.Side.SHADOW)
        self.assertEqual(len(rig.drv.es.armies['morannon'].units), 6)
        self.assertEqual(len(rig.drv.es.armies['dagorlad'].units), 10)
        self.assertNotIn('morannon', rig.drv.es.besieged_strongholds)
        self.assertEqual(rig.drv.pending_buf.records[0]['norm'],
                         live_parser.parse(RETURN)['norm'])
        self.assertEqual(len(rig.drv.pending_buf.records), 1)
        return dp

    def test_captured_return_before_seat_exposes_real_ai_choice(self):
        rig = self.rig()
        before = pickle.dumps(rig.drv.es)
        rig.feed(RETURN)
        self.assertEqual(pickle.dumps(rig.drv.es), before)
        rig.feed(SEAT)
        self.assert_ai_boundary(rig)
        self.assertEqual(rig.drv.alarms, [])
        self.assertEqual(rig.drv.holds, [])

    def test_actual_wrong_box_undo_then_correct_box_sequence(self):
        rig = self.rig()
        wrong = SEAT.replace('to Morannon Stronghold', 'to Dol Amroth Stronghold')
        rig.feed(RETURN, wrong)
        self.assertEqual(ai.describe_pending(rig.drv.es)['kind'], 'combat:defender_siege_choice')
        rig.feed('$Wey undoes last action : '+wrong)
        self.assertEqual(len(rig.drv.pending_buf.records), 1)
        rig.feed(SEAT)
        self.assert_ai_boundary(rig)

    def test_no_return_control_still_matches_ordinary_seating(self):
        rig = self.rig()
        rig.feed(SEAT.replace('[5]', '[6]'))
        self.assertEqual(ai.describe_pending(rig.drv.es)['kind'], 'combat:victor_advance')
        self.assertIsNone(rig.drv.pending_buf)

    def test_unrelated_or_incomplete_returns_do_not_invent_withdrawal(self):
        for returned, seat in ((RETURN, None),
                (RETURN.replace('FP Reinforcements', 'FP Casualties'), SEAT),
                (RETURN.replace('Gondor Regular', 'Rohan Regular'), SEAT),
                (RETURN.replace('Gondor Regular', 'Gondor Elite'), SEAT),
                (RETURN.replace('Gondor Regular', 'group ([2]Gondor Regular)'), SEAT),
                (RETURN.replace('from Morannon', 'from Dagorlad'), SEAT),
                (RETURN, SEAT.replace('to Morannon Stronghold', 'to Dol Amroth Stronghold'))):
            with self.subTest(returned=returned, seat=seat):
                rig = self.rig()
                before = pickle.dumps(rig.drv.es)
                rig.feed(returned, *([seat] if seat else []))
                self.assertEqual(pickle.dumps(rig.drv.es), before)
                self.assertEqual(ai.describe_pending(rig.drv.es)['kind'], 'combat:defender_siege_choice')

    def test_real_advance_opens_and_consumes_only_the_retained_legal_trim(self):
        rig = self.rig()
        casualties = pickle.dumps(rig.drv.es.fp_casualties)
        reserve = ai.figures_available(rig.drv.es, gs.Nation.GONDOR, 'regular')
        rig.feed(RETURN, SEAT)
        self.assert_ai_boundary(rig)
        self.choose(rig, 'victor_advance')
        self.assertEqual(ai.describe_pending(rig.drv.es)['kind'], 'combat:siege_trim')
        self.assertFalse(any(key == 'siege_seat_demanded' for key, _ in rig.drv.findings))
        rig.drv.tick()
        self.assertIsNone(rig.drv.es.pending_combat)
        self.assertIsNone(rig.drv.pending_buf)
        army = rig.drv.es.armies['morannon']
        self.assertEqual(sum(u.nation is gs.Nation.GONDOR for u in army.units), 5)
        self.assertEqual(sum(u.nation is gs.Nation.SAURON for u in army.units), 10)
        self.assertEqual(len(army.leaders), 1)
        self.assertEqual(ai.figures_available(rig.drv.es, gs.Nation.GONDOR, 'regular'), reserve+1)
        self.assertEqual(pickle.dumps(rig.drv.es.fp_casualties), casualties)
        # Start from the independent complete native capture. The only
        # subsequent physical change here is the chosen AI army advance.
        dump = client_board.read_board(str(CAPTURE/'ai_client_board.txt'))
        self.assertEqual(dump['081'].lower(), 'fp reinf')
        self.assertTrue(all(dump[pid].lower() == 'morannon stronghold'
                            for pid in ('075','076','077','082','083')))
        movers = [pid for pid, zone in dump.items() if zone.lower() == 'dagorlad'
                  and (client_board.unit_type(pid)[0] is gs.Nation.SAURON or pid == '038')]
        self.assertEqual(len(movers), 11)
        for pid in movers:
            dump[pid] = 'morannon'
        issues = client_board.diff_engine_vs_dump(rig.drv.es, dump, ai_module=ai,
                    dynamic_types=rig.drv.view.dynamic_types,
                    granted_pids=rig.drv._granted_die_pids(dump),
                    marker_states=client_protocol.read_marker_states(str(CAPTURE/'ai_client_board.txt')))
        self.assertEqual(issues, [])
        self.assertIsNone(getattr(rig.drv, '_siege_seat_owed', None))

    def test_real_stay_keeps_early_return_for_correction_without_removing_units(self):
        rig = self.rig()
        rig.feed(RETURN, SEAT)
        self.choose(rig, 'victor_stay')
        before = pickle.dumps(rig.drv.es)
        rig.drv.tick()
        self.assertEqual(pickle.dumps(rig.drv.es), before)
        self.assertIsNone(rig.drv.es.pending_combat)
        self.assertEqual(len(rig.drv.es.armies['morannon'].units), 6)
        self.assertEqual(len(rig.drv.es.armies['dagorlad'].units), 10)
        self.assertEqual(len(rig.drv.pending_buf.records), 1)
        self.assertTrue(any(k == 'early_siege_return_needs_correction' for k, _ in rig.drv.findings))

    def test_native_undo_of_seating_cancels_ai_step_and_retains_early_return(self):
        rig = self.rig()
        rig.feed(RETURN, SEAT)
        self.assert_ai_boundary(rig)
        epoch = rig.drv.undo_epoch
        rig.feed('$Wey undoes last action : '+SEAT)
        self.assertEqual(rig.drv.undo_epoch, epoch+1)
        self.assertEqual(ai.describe_pending(rig.drv.es)['kind'], 'combat:defender_siege_choice')
        self.assertEqual(len(rig.drv.pending_buf.records), 1)
        self.assertEqual(rig.drv.pending_buf.records[0]['norm'], live_parser.parse(RETURN)['norm'])
        rig.feed(SEAT)
        self.assert_ai_boundary(rig)


if __name__ == '__main__':
    unittest.main(verbosity=2)
