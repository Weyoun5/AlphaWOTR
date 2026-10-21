"""Independent negative witnesses at the real resync entry points."""
import pickle
import unittest
from unittest.mock import patch

import ai
import board
import game_state as gs
from test_bridge_oct11_combat_recovery import LiveCombatRecovery, LiveNoncombatRecovery
from live import adopt, client_board
from live.driver import PhaseWindow


class AdversarialRecovery(unittest.TestCase):
    def rig(self, combat=True):
        fixture = (LiveCombatRecovery if combat else LiveNoncombatRecovery)('runTest')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return fixture

    def test_partial_native_elite_reduction_retains_unfinished_payment(self):
        rig = self.rig()
        driver = rig.driver
        elite = next(pid for pid, zone in rig.dump.items()
                     if board.sanitize(zone) == 'fords of isen'
                     and client_board.unit_type(pid) == (gs.Nation.ROHAN, 'elite'))
        rig.dump[elite] = 'FP Dead'
        driver.pending_buf = PhaseWindow()
        driver.pending_buf.records = [{'op': 'kill', 'pieces': ['Rohan Elite'],
            'zone': 'Fords of Isen', 'norm': '$Human kills Rohan Elite in Fords of Isen.',
            'meta': {'tags': ['Controls.304']}}]
        pending = driver.es.pending_combat
        buffer = driver.pending_buf
        before = pickle.dumps(driver.es)
        rig.write_dump()
        driver.reset_and_resync()
        self.assertEqual(driver.n_commits_total, 0)
        self.assertIs(driver.es.pending_combat, pending)
        self.assertIs(driver.pending_buf, buffer)
        self.assertEqual(len(buffer.records), 1)
        self.assertEqual(pickle.dumps(driver.es), before)

    def test_one_missing_native_piece_cannot_be_inferred_as_shadow_casualty(self):
        rig = self.rig()
        driver = rig.driver
        driver.human_side = gs.Side.SHADOW
        driver.ai_side = gs.Side.FREE_PEOPLES
        driver.es.pending_combat.update(deciding_side=gs.Side.SHADOW,
            alloc_stage='defender', atk_hits=0, def_hits=1)
        driver._initial_dump = dict(rig.dump)
        before = pickle.dumps(driver.es)
        del rig.dump[rig.shadow_regulars[0]]
        self.assertEqual(len(rig.dump), 427)  # Still exceeds the coarse 400-piece threshold.
        rig.write_dump()
        items, report = driver.reset_and_resync()
        self.assertEqual(driver.n_commits_total, 0, (items, report))
        self.assertEqual(pickle.dumps(driver.es), before)

    def test_missing_canonical_figure_is_rejected_without_a_startup_census(self):
        rig = self.rig(combat=False)
        driver = rig.driver
        driver._initial_dump = None
        before = pickle.dumps(driver.es)
        del rig.dump[rig.fp_regulars[0]]
        rig.write_dump()
        items, report = driver.reset_and_resync()
        self.assertFalse(driver._last_resync_adoption['complete'], (items, report))
        self.assertEqual(pickle.dumps(driver.es), before)

    def test_shadow_casualty_requires_a_physical_reserve_destination(self):
        for count in (1, 2):
            for missing in (False, True):
                with self.subTest(losses=count, missing=missing):
                    rig = self.rig()
                    driver = rig.driver
                    driver.human_side = gs.Side.SHADOW
                    driver.ai_side = gs.Side.FREE_PEOPLES
                    driver.es.pending_combat.update(deciding_side=gs.Side.SHADOW,
                        alloc_stage='defender', atk_hits=0, def_hits=count)
                    before = pickle.dumps(driver.es)
                    for pid in rig.shadow_regulars[:count]:
                        if missing:
                            del rig.dump[pid]
                        else:
                            rig.dump[pid] = 'SA Reinforcements'
                    rig.write_dump()
                    items, report = driver.reset_and_resync()
                    if missing:
                        self.assertEqual(driver.n_commits_total, 0, (items, report))
                        self.assertEqual(pickle.dumps(driver.es), before)
                    else:
                        self.assertEqual(driver.n_commits_total, 1, (items, report))
                        army = driver.es.armies.get('orthanc')
                        self.assertEqual(len(army.units) if army else 0, 2-count)

    def test_valid_own_reserve_return_is_adopted(self):
        rig = self.rig(combat=False)
        driver = rig.driver
        rig.dump[rig.fp_regulars[0]] = 'FP Reinforcements'
        rig.write_dump()
        items, report = driver.reset_and_resync()
        self.assertTrue(driver._last_resync_adoption['complete'], (items, report))
        self.assertEqual(len(driver.es.armies['fords of isen'].units), 2)
        self.assertEqual(driver.es.fp_casualties.get(gs.Nation.ROHAN, {}).get('regular', 0), 0)

    def test_replacement_id_can_complete_an_elite_reduction(self):
        rig = self.rig()
        driver = rig.driver
        elite = next(pid for pid, zone in rig.dump.items()
                     if board.sanitize(zone) == 'fords of isen'
                     and client_board.unit_type(pid) == (gs.Nation.ROHAN, 'elite'))
        rig.dump[elite] = 'FP Dead'
        rig.dump['900'] = 'Fords of Isen'
        driver.view.dynamic_types['900'] = (gs.Nation.ROHAN, gs.UnitKind.REGULAR)
        rig.write_dump()
        items, report = driver.reset_and_resync()
        self.assertEqual(driver.n_commits_total, 1, (items, report))
        units = driver.es.armies['fords of isen'].units
        self.assertEqual(len(units), 3)
        self.assertTrue(all(unit.kind is gs.UnitKind.REGULAR for unit in units))
        self.assertEqual(driver.es.fp_casualties[gs.Nation.ROHAN]['elite'], 1)

    def test_unanswered_hidden_card_choice_survives_complete_board(self):
        rig = self.rig(combat=False)
        driver = rig.driver
        pending = {'card_id': '__gandalf_draw__', 'card_name': 'Gandalf guide draw',
            'side': 'fp', 'decider': 'fp', 'direct': True, 'gandalf_draw': True,
            'source_card_id': 'FREE_PEOPLES_Character_House_of_the_Stewards',
            'deck': 'character', 'options': [{'gandalf_draw': True}, {'gandalf_draw': False}]}
        driver.es.pending_card_option = pending
        before = pickle.dumps(driver.es)
        rng = pickle.dumps(vars(rig.rand))
        items, report = driver.reset_and_resync()
        self.assertIs(driver.es.pending_card_option, pending, (items, report))
        self.assertEqual(pickle.dumps(driver.es), before)
        self.assertEqual(pickle.dumps(vars(rig.rand)), rng)
        self.assertFalse(driver._last_resync_adoption['complete'])

    def test_fp_unit_in_shadow_reserve_is_not_successful_sync(self):
        rig = self.rig(combat=False)
        driver = rig.driver
        before = pickle.dumps(driver.es)
        rig.dump[rig.fp_regulars[0]] = 'SA Reinforcements'
        rig.write_dump()
        items, report = driver.reset_and_resync()
        self.assertFalse(driver._last_resync_adoption['complete'], (items, report))
        self.assertEqual(pickle.dumps(driver.es), before)

    def test_board_change_during_adoption_cannot_publish_success(self):
        rig = self.rig(combat=False)
        driver = rig.driver
        before = pickle.dumps(driver.es)
        original = adopt.adopt_client_board
        def board_arrival(*args, **kwargs):
            result = original(*args, **kwargs)
            rig.dump[rig.fp_regulars[0]] = 'Westemnet'
            rig.write_dump()
            return result
        with patch.object(adopt, 'adopt_client_board', side_effect=board_arrival):
            items, report = driver.reset_and_resync()
        self.assertFalse(driver._last_resync_adoption['complete'], (items, report))
        self.assertEqual(pickle.dumps(driver.es), before)


if __name__ == '__main__':
    # Import fixtures without re-running their suites as inherited test cases.
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AdversarialRecovery)
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
