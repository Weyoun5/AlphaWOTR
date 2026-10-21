"""Literal formation oracles for the pure compulsory-participant helper.

No public engine is patched or used to derive expected participants.
Public engine transitions are covered separately in test_audit0904_forced_army.py.
"""
from collections import Counter
import copy
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
import unittest

import game_state as gs
from army_participants import select_forced_attack, FORMATION_VERSION

FP, SH = gs.Side.FREE_PEOPLES, gs.Side.SHADOW
NORTH, ELVES, GONDOR = gs.Nation.THE_NORTH, gs.Nation.ELVES, gs.Nation.GONDOR
REGULAR, ELITE = gs.UnitKind.REGULAR, gs.UnitKind.ELITE


def fixture(units=None, leaders=None, *, at_war=(NORTH,)):
    army = gs.Army(region='dale', units=list(units or []), leaders=list(leaders or []))
    state = SimpleNamespace(
        armies={'dale': army},
        political=gs.PoliticalTrack(
            boxes={n: 3 if n in at_war else 2 for n in gs.Nation},
            active={n: True for n in gs.Nation}))
    return state, army


def counts(units):
    return Counter((u.nation, u.kind) for u in units)


class PureForcedArmySelection(unittest.TestCase):
    def test_literal_dale_reserves_one_north_regular_with_peaceful_elf_leader(self):
        first, second, elf = gs.Unit(NORTH, REGULAR), gs.Unit(NORTH, REGULAR), gs.Leader(ELVES)
        state, army = fixture([first, second], [elf])
        formed = select_forced_attack(state, army, FP)
        self.assertTrue(formed.available)
        self.assertTrue(formed.guard_added)
        self.assertFalse(formed.character_action_ok)
        self.assertEqual(counts(formed.attacking_units), Counter({(NORTH, REGULAR): 1}))
        self.assertEqual(formed.attacking_leaders, ())
        self.assertEqual(counts(formed.rearguard_units), Counter({(NORTH, REGULAR): 1}))
        self.assertIs(formed.rearguard_leaders[0], elf)
        self.assertEqual(formed.rearguard_signature,
            (FORMATION_VERSION, (('THE_NORTH', 'REGULAR', 1),), (('ELVES', 1),)))

    def test_active_but_not_at_war_elf_cannot_supply_character_attack(self):
        state, army = fixture([gs.Unit(NORTH, REGULAR), gs.Unit(NORTH, REGULAR)], [gs.Leader(ELVES)])
        formed = select_forced_attack(state, army, FP, require_leader=True)
        self.assertFalse(formed.available)
        self.assertEqual(formed.unavailable_reason, 'no_participating_leader_for_character_action')
        self.assertEqual(formed.attacking_leaders, ())

    def test_last_eligible_unit_must_guard_so_attack_is_unavailable(self):
        state, army = fixture([gs.Unit(NORTH, REGULAR)], [gs.Leader(ELVES)])
        formed = select_forced_attack(state, army, FP)
        self.assertFalse(formed.available)
        self.assertEqual(formed.unavailable_reason, 'no_attacking_unit_after_guard')
        self.assertEqual(formed.attacking_units, ())
        self.assertEqual(len(formed.rearguard_units), 1)
        self.assertEqual(len(formed.rearguard_leaders), 1)

    def test_peaceful_unit_already_supplies_guard_and_every_north_unit_attacks(self):
        state, army = fixture([gs.Unit(NORTH, REGULAR), gs.Unit(NORTH, REGULAR),
                               gs.Unit(ELVES, REGULAR)], [gs.Leader(ELVES)])
        formed = select_forced_attack(state, army, FP)
        self.assertTrue(formed.available)
        self.assertFalse(formed.guard_added)
        self.assertEqual(counts(formed.attacking_units), Counter({(NORTH, REGULAR): 2}))
        self.assertEqual(counts(formed.rearguard_units), Counter({(ELVES, REGULAR): 1}))

    def test_existing_peaceful_elite_guard_does_not_reserve_an_extra_regular(self):
        state, army = fixture([gs.Unit(NORTH, REGULAR), gs.Unit(ELVES, ELITE)], [gs.Leader(ELVES)])
        formed = select_forced_attack(state, army, FP)
        self.assertTrue(formed.available)
        self.assertFalse(formed.guard_added)
        self.assertEqual(counts(formed.attacking_units), Counter({(NORTH, REGULAR): 1}))
        self.assertEqual(counts(formed.rearguard_units), Counter({(ELVES, ELITE): 1}))

    def test_guard_prefers_regular_before_elite(self):
        state, army = fixture([gs.Unit(NORTH, ELITE), gs.Unit(NORTH, REGULAR)], [gs.Leader(ELVES)])
        formed = select_forced_attack(state, army, FP)
        self.assertEqual(counts(formed.attacking_units), Counter({(NORTH, ELITE): 1}))
        self.assertEqual(counts(formed.rearguard_units), Counter({(NORTH, REGULAR): 1}))

    def test_guard_falls_back_to_one_elite_when_no_regular_exists(self):
        state, army = fixture([gs.Unit(NORTH, ELITE), gs.Unit(GONDOR, ELITE)],
                               [gs.Leader(ELVES)], at_war=(NORTH, GONDOR))
        formed = select_forced_attack(state, army, FP)
        self.assertEqual(counts(formed.rearguard_units), Counter({(GONDOR, ELITE): 1}))
        self.assertEqual(counts(formed.attacking_units), Counter({(NORTH, ELITE): 1}))

    def test_nation_tie_break_is_stable_under_input_order_and_deep_copy(self):
        state, army = fixture([gs.Unit(NORTH, REGULAR), gs.Unit(GONDOR, REGULAR)],
                               [gs.Leader(ELVES)], at_war=(NORTH, GONDOR))
        first = select_forced_attack(state, army, FP)
        copied = copy.deepcopy(army)
        copied.units.reverse()
        second = select_forced_attack(state, copied, FP)
        self.assertEqual(first.rearguard_signature, second.rearguard_signature)
        self.assertEqual(counts(second.rearguard_units), Counter({(GONDOR, REGULAR): 1}))
        self.assertEqual(counts(second.attacking_units), Counter({(NORTH, REGULAR): 1}))

    def test_one_guard_supports_multiple_peaceful_leaders(self):
        state, army = fixture([gs.Unit(NORTH, REGULAR), gs.Unit(NORTH, REGULAR)],
                               [gs.Leader(ELVES), gs.Leader(GONDOR)])
        formed = select_forced_attack(state, army, FP)
        self.assertEqual(len(formed.rearguard_units), 1)
        self.assertEqual(len(formed.rearguard_leaders), 2)
        self.assertEqual(len(formed.attacking_units), 1)

    def test_eligible_north_leader_joins_attack_while_elf_stays(self):
        north, elf = gs.Leader(NORTH), gs.Leader(ELVES)
        state, army = fixture([gs.Unit(NORTH, REGULAR), gs.Unit(NORTH, REGULAR)], [north, elf])
        formed = select_forced_attack(state, army, FP, require_leader=True)
        self.assertTrue(formed.available)
        self.assertTrue(formed.character_action_ok)
        self.assertEqual(formed.attacking_leaders, (north,))
        self.assertEqual(formed.rearguard_leaders, (elf,))

    def test_at_war_elf_leader_attacks_with_every_unit_without_guard(self):
        elf = gs.Leader(ELVES)
        state, army = fixture([gs.Unit(NORTH, REGULAR), gs.Unit(NORTH, ELITE)],
                               [elf], at_war=(NORTH, ELVES))
        formed = select_forced_attack(state, army, FP, require_leader=True)
        self.assertTrue(formed.available)
        self.assertEqual(len(formed.attacking_units), 2)
        self.assertEqual(formed.attacking_leaders, (elf,))
        self.assertEqual(formed.rearguard_units, ())
        self.assertEqual(formed.rearguard_leaders, ())

    def test_legolas_is_a_character_not_a_peaceful_elf_leader(self):
        legolas = gs.Companion('Legolas', level=2, leadership=1)
        state, army = fixture([gs.Unit(NORTH, REGULAR)], [legolas])
        formed = select_forced_attack(state, army, FP, require_leader=True)
        self.assertTrue(formed.available)
        self.assertEqual(formed.attacking_leaders, (legolas,))
        self.assertFalse(formed.guard_added)

    def test_nazgul_and_actual_board_minions_ignore_sauron_politics(self):
        from game_logic import Minion
        figures = [gs.Nazgul(gs.Nation.SAURON),
                   Minion(gs.Nation.SAURON, 'Witch King'),
                   Minion(gs.Nation.SAURON, 'Mouth of Sauron')]
        state, army = fixture([gs.Unit(gs.Nation.ISENGARD, REGULAR)], figures,
                               at_war=(gs.Nation.ISENGARD,))
        formed = select_forced_attack(state, army, SH, require_leader=True)
        self.assertTrue(formed.available)
        self.assertEqual(formed.attacking_leaders, tuple(figures))
        self.assertEqual(formed.rearguard_leaders, ())

    def test_saruman_can_lead_attack_although_he_cannot_advance(self):
        from game_logic import Minion
        saruman = Minion(gs.Nation.ISENGARD, 'Saruman')
        state, army = fixture([gs.Unit(gs.Nation.ISENGARD, REGULAR)], [saruman],
                               at_war=(gs.Nation.ISENGARD,))
        army.region = 'orthanc'
        formed = select_forced_attack(state, army, SH, require_leader=True)
        self.assertTrue(formed.available)
        self.assertEqual(formed.attacking_leaders, (saruman,))

    def test_saruman_grants_character_eligibility_only_to_participating_isengard_elite(self):
        from game_logic import Minion
        state, army = fixture([gs.Unit(gs.Nation.ISENGARD, ELITE)],
                               at_war=(gs.Nation.ISENGARD,))
        state.armies['orthanc'] = gs.Army('orthanc', leaders=[Minion(gs.Nation.ISENGARD, 'Saruman')])
        self.assertTrue(select_forced_attack(state, army, SH, require_leader=True).available)
        state.political.boxes[gs.Nation.ISENGARD] = 2
        state.political.boxes[gs.Nation.SAURON] = 3
        army.units.append(gs.Unit(gs.Nation.SAURON, REGULAR))
        formed = select_forced_attack(state, army, SH, require_leader=True)
        self.assertFalse(formed.available)
        self.assertEqual(formed.unavailable_reason, 'no_participating_leader_for_character_action')
        self.assertEqual(counts(formed.rearguard_units), Counter({(gs.Nation.ISENGARD, ELITE): 1}))

    def test_enemy_figures_in_shared_entry_cannot_supply_guard_or_leadership(self):
        enemy = gs.Unit(gs.Nation.SAURON, REGULAR)
        nazgul = gs.Nazgul(gs.Nation.SAURON)
        state, army = fixture([gs.Unit(NORTH, REGULAR), enemy], [gs.Leader(ELVES), nazgul])
        formed = select_forced_attack(state, army, FP)
        self.assertFalse(formed.available)
        self.assertEqual(formed.bystander_units, (enemy,))
        self.assertEqual(formed.bystander_leaders, (nazgul,))
        self.assertEqual(counts(formed.rearguard_units), Counter({(NORTH, REGULAR): 1}))

    def test_all_peaceful_units_cannot_attack_even_with_a_companion(self):
        state, army = fixture([gs.Unit(ELVES, REGULAR)], [gs.Companion('Legolas', level=2)])
        formed = select_forced_attack(state, army, FP)
        self.assertFalse(formed.available)
        self.assertEqual(formed.unavailable_reason, 'no_eligible_attacking_unit')
        self.assertFalse(formed.character_action_ok)

    def test_selection_is_read_only_and_returns_original_figures(self):
        state, army = fixture([gs.Unit(NORTH, REGULAR), gs.Unit(NORTH, ELITE)], [gs.Leader(ELVES)])
        before = copy.deepcopy(state)
        unit_ids = [id(u) for u in army.units]
        leader_ids = [id(l) for l in army.leaders]
        formed = select_forced_attack(state, army, FP)
        self.assertEqual(state, before)
        self.assertEqual([id(u) for u in army.units], unit_ids)
        self.assertEqual([id(l) for l in army.leaders], leader_ids)
        self.assertEqual(sorted(id(u) for u in formed.attacking_units + formed.rearguard_units), sorted(unit_ids))
        with self.assertRaises(FrozenInstanceError):
            formed.guard_added = False

    def test_captured_signature_does_not_change_if_politics_later_changes(self):
        state, army = fixture([gs.Unit(NORTH, REGULAR), gs.Unit(NORTH, ELITE)], [gs.Leader(ELVES)])
        formed = select_forced_attack(state, army, FP)
        signature = formed.rearguard_signature
        state.political.boxes[ELVES] = 3
        self.assertEqual(formed.rearguard_signature, signature)
        self.assertEqual(len(formed.rearguard_leaders), 1)
        # A NEW attack declaration has different eligibility; do not recompute
        # the earlier battle's frozen plan after a later state change.
        self.assertEqual(select_forced_attack(state, army, FP).rearguard_leaders, ())

    def test_invalid_metadata_is_rejected_instead_of_inventing_eligibility(self):
        state, army = fixture([gs.Unit(NORTH, REGULAR)])
        del state.political.boxes[NORTH]
        with self.assertRaisesRegex(ValueError, 'Missing political'):
            select_forced_attack(state, army, FP)
        state.political.boxes[NORTH] = True
        with self.assertRaisesRegex(ValueError, 'Invalid political'):
            select_forced_attack(state, army, FP)
        with self.assertRaisesRegex(ValueError, 'actual Side'):
            select_forced_attack(state, army, 'fp')


if __name__ == '__main__':
    unittest.main()
