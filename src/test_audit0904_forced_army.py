"""Compulsory rearguards under the user's all-eligible-figures AI policy.

RB27: a plain FP Leader cannot remain without Army units. RB28: peaceful
figures must remain in the rearguard, which has at least one Army unit;
Character attack needs participating leadership. RB35: peaceful Leaders
cannot attack even inside their own nation, while Companions/Minions/Nazgul
are always belligerent. RB31/AL36: rearguard cannot take losses or advance.

The user specifically requires the peaceful Elven Leader in Dale to stay
with a guard when North attacks, even into Woodland Realm. AI uses every
eligible figure otherwise, preferring a Regular as the compulsory guard.
This suite does not request voluntary rearguard learning or alter human rules.
"""
from collections import Counter
import tempfile
from pathlib import Path
import unittest

import ai
import board
import combat
import game_logic as gl
import game_state as gs
import game_trace

FP, SH = gs.Side.FREE_PEOPLES, gs.Side.SHADOW
NORTH, ELVES = gs.Nation.THE_NORTH, gs.Nation.ELVES
REGULAR, ELITE = gs.UnitKind.REGULAR, gs.UnitKind.ELITE
SOURCE, TARGET = 'dale', 'woodland realm'


def army(state, region):
    return next((a for k, a in state.armies.items() if board.sanitize(k) == region), None)


def position(*, units=2, elite=False, peaceful_unit=False,
             north_leader=False, elf_at_war=False, companion=False,
             face=gs.ActionDieFace.MUSTER_ARMY):
    """Later board using real pieces: North marches, Shadow occupies Woodland.

    Woodland's Elven force has withdrawn; its plain Leader now shares Dale
    with North. The Elf political marker remains below At War. No Elven
    Army unit is in Dale unless explicitly requested by a control.
    """
    state = gs.setup_initial_game(board.load_board())
    gl.initialize_decks(state)
    for side in (FP, SH):
        state.hands[side].hand = []
    dale, carrock = army(state, SOURCE), army(state, 'carrock')
    woodland, lorien = army(state, TARGET), army(state, 'lorien')
    if not north_leader:
        # Carrock supplies its only Regular to the two-unit fixture below.
        # Keep the displaced Leader with Bree's unchanged North Army.
        army(state, 'bree').leaders.extend(dale.leaders)
        dale.leaders = []
    elf_leader = woodland.leaders.pop()
    if companion:
        lorien.leaders.append(elf_leader)
        legolas = next(c for c in state.fellowship.companions if c.name == 'Legolas')
        state.fellowship.companions.remove(legolas)
        dale.leaders.append(legolas)
    else:
        dale.leaders.append(elf_leader)
    lorien.units.extend(woodland.units)
    woodland.units = []
    shadow_source = army(state, 'dol guldur')
    shadow = next(u for u in shadow_source.units if u.kind == REGULAR)
    shadow_source.units.remove(shadow)
    woodland.units.append(shadow)
    state.enemy_controlled[SH].add(TARGET)
    if units == 2:
        source = army(state, 'north downs') if elite else carrock
        unit = next(u for u in source.units if u.nation == NORTH)
        source.units.remove(unit)
        dale.units.append(unit)
    if peaceful_unit:
        unit = next(u for u in lorien.units if u.nation == ELVES and u.kind == REGULAR)
        lorien.units.remove(unit)
        dale.units.append(unit)
    state.political.boxes[NORTH] = 3
    state.political.active[NORTH] = True
    state.political.boxes[ELVES] = 3 if elf_at_war else 2
    state.political.active[ELVES] = True
    die = gs.ActionDie([face], FP)
    die.current = face
    shadow_die = gs.ActionDie([gs.ActionDieFace.MUSTER], SH)
    shadow_die.current = gs.ActionDieFace.MUSTER
    state.dice = {FP: [die], SH: [shadow_die]}
    return state


def attack(state):
    return ai.apply_action(state, ai.Action(0, 'army_attack', {
        'from': SOURCE, 'to': TARGET, 'side': 'fp'}))


def target_menu(state):
    return [a for a in ai.get_ai_actions(state, FP)
            if a.action_type == 'army_attack'
            and board.sanitize(a.params['from']) == SOURCE
            and board.sanitize(a.params['to']) == TARGET]


def participants(state):
    view = combat.setup_combat(state, SOURCE, TARGET, attacker_side=FP)
    return (Counter((u.nation, u.kind) for u in view.attacker_army.units),
            list(view.attacker_army.leaders), view.attacker_leadership)


def north_count(state, region):
    a = army(state, region)
    return sum(u.nation == NORTH for u in (a.units if a else []))


def elf_leaders(state, region):
    a = army(state, region)
    return sum(isinstance(l, gs.Leader) and l.nation == ELVES
               for l in (a.leaders if a else []))


class ForcedArmyPolicy(unittest.TestCase):
    def test_fixture_never_orphans_a_plain_leader_elsewhere_on_the_board(self):
        for options in ({}, {'units': 1}, {'elite': True},
                        {'north_leader': True}, {'companion': True},
                        {'peaceful_unit': True}):
            with self.subTest(options=options):
                state = position(**options)
                for region, a in state.armies.items():
                    if any(isinstance(l, gs.Leader) and not getattr(l, 'name', None)
                           for l in a.leaders):
                        self.assertTrue(a.units, 'Plain Leader orphaned at %s' % region)

    def test_peaceful_elf_leader_keeps_regular_guard_even_attacking_woodland(self):
        state = position()
        self.assertEqual(len(target_menu(state)), 1, 'One target, no rearguard policy branches')
        state = attack(state)
        self.assertNotEqual(state.pending_combat['phase'], 'attacker_rearguard')
        units, leaders, leadership = participants(state)
        self.assertEqual(units, Counter({(NORTH, REGULAR): 1}))
        self.assertEqual(leaders, [])
        self.assertEqual(leadership, 0)
        self.assertEqual(north_count(state, SOURCE), 2, 'Battle formation does not move board pieces')
        self.assertEqual(elf_leaders(state, SOURCE), 1)

    def test_character_die_cannot_use_only_the_peaceful_leaders_leadership(self):
        self.assertEqual(target_menu(position(face=gs.ActionDieFace.CHARACTER)), [])

    def test_invalid_character_attack_rejects_before_die_or_political_payment(self):
        state = position(face=gs.ActionDieFace.CHARACTER)
        political = dict(state.political.boxes)
        with self.assertRaises((ValueError, RuntimeError)):
            ai._execute_action(state, ai.Action(0, 'army_attack', {
                'from': SOURCE, 'to': TARGET, 'side': 'fp'}))
        self.assertFalse(state.dice[FP][0].used)
        self.assertEqual(state.political.boxes, political)
        self.assertEqual(north_count(state, SOURCE), 2)
        self.assertEqual(elf_leaders(state, SOURCE), 1)

    def test_only_army_unit_cannot_leave_the_peaceful_leader_unguarded(self):
        self.assertEqual(target_menu(position(units=1)), [])

    def test_existing_peaceful_unit_supplies_guard_without_holding_back_north(self):
        state = position(peaceful_unit=True)
        self.assertEqual(len(target_menu(state)), 1)
        state = attack(state)
        units, leaders, leadership = participants(state)
        self.assertEqual(units, Counter({(NORTH, REGULAR): 2}))
        self.assertEqual(leaders, [])

    def test_guard_prefers_regular_so_the_eligible_elite_attacks(self):
        state = attack(position(elite=True))
        units, leaders, leadership = participants(state)
        self.assertEqual(units, Counter({(NORTH, ELITE): 1}))
        self.assertEqual(leaders, [])

    def test_character_attack_uses_eligible_north_leader_while_elf_stays(self):
        state = position(north_leader=True, face=gs.ActionDieFace.CHARACTER)
        self.assertEqual(len(target_menu(state)), 1)
        units, leaders, leadership = participants(attack(state))
        self.assertEqual(units, Counter({(NORTH, REGULAR): 1}))
        self.assertEqual([l.nation for l in leaders], [NORTH])
        self.assertEqual(leadership, 1)

    def test_at_war_elf_leader_and_all_units_participate_without_guard(self):
        units, leaders, leadership = participants(attack(position(elf_at_war=True)))
        self.assertEqual(units, Counter({(NORTH, REGULAR): 2}))
        self.assertEqual([l.nation for l in leaders], [ELVES])
        self.assertEqual(leadership, 1)

    def test_legolas_is_belligerent_even_though_elves_are_not_at_war(self):
        state = position(companion=True, face=gs.ActionDieFace.CHARACTER)
        self.assertEqual(len(target_menu(state)), 1)
        units, leaders, leadership = participants(attack(state))
        self.assertEqual(units, Counter({(NORTH, REGULAR): 2}))
        self.assertEqual([l.name for l in leaders], ['Legolas'])
        self.assertEqual(leadership, 1)

    def test_excess_combat_hits_do_not_kill_guard_or_peaceful_leader(self):
        state = attack(position())
        ai._apply_allocation(state, SOURCE, FP, hits=2, k_downgrades=0)
        self.assertEqual(north_count(state, SOURCE), 1)
        self.assertEqual(elf_leaders(state, SOURCE), 1)
        self.assertEqual(ai._pending_battle_wiped(state, state.pending_combat), (True, False))

    def test_advance_menu_contains_only_participating_unit_counts(self):
        state = attack(position())
        state.pending_combat['phase'] = 'victor_advance'
        state.pending_combat['deciding_side'] = FP
        army(state, TARGET).units = []  # literal defender wipe, pending advance
        advances = [a for a in ai.get_legal_actions(state, FP) if a.action_type == 'victor_advance']
        self.assertTrue(advances)
        self.assertEqual({a.params['n_regular'] for a in advances}, {1})

    def test_advance_into_elf_homeland_cannot_take_protected_elf_leader(self):
        state = attack(position())
        state.pending_combat['phase'] = 'victor_advance'
        state.pending_combat['deciding_side'] = FP
        army(state, TARGET).units = []
        state = ai.apply_action(state, ai.Action(-1, 'victor_advance', {
            'from': SOURCE, 'to': TARGET, 'side': 'fp',
            'n_regular': 1, 'n_elite': 0, 'take_leaders': True}))
        self.assertEqual(north_count(state, SOURCE), 1)
        self.assertEqual(elf_leaders(state, SOURCE), 1)
        self.assertEqual(north_count(state, TARGET), 1)
        self.assertEqual(elf_leaders(state, TARGET), 0)

    def test_clone_and_trace_keep_the_same_protected_formation(self):
        state = attack(position())
        cloned = ai.clone_state(state)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'formation.trace.gz')
            recorder = game_trace.TraceRecorder(path)
            recorder.record_initial(state)
            recorder.save()
            restored = game_trace.load_trace(path)['steps'][0]['state']
        for variant in (cloned, restored):
            with self.subTest(kind='clone' if variant is cloned else 'trace'):
                units, leaders, leadership = participants(variant)
                self.assertEqual(units, Counter({(NORTH, REGULAR): 1}))
                self.assertEqual(leaders, [])


if __name__ == '__main__':
    unittest.main()
