"""Printed army choices: read-only, tests-first investigation.

RB28: an attacker may divide into two nonempty armies before the defender
chooses field/siege, freely assigning figures; non-At-War figures MUST stay
in the rearguard. RB31: a victorious attacker may advance all/part, while a
defender retreats with its entire army. AL36 confirms rearguard timing and
exclusion from combat; AL46 gives the literal Dale->Erebor retreat example.

These are independent desired behaviors, deliberately failing where the
public engine omits a choice. Prospective attack-count encoding uses the
existing movement field names; it is not a claim those fields are already
supported for attacks. No production code or other test class is imported
as the oracle. No game/model/corpus writes.
"""
import unittest

import ai
import board
import game_logic as gl
import game_state as gs

FP, SH = gs.Side.FREE_PEOPLES, gs.Side.SHADOW
GONDOR, ROHAN = gs.Nation.GONDOR, gs.Nation.ROHAN
REGULAR, ELITE = gs.UnitKind.REGULAR, gs.UnitKind.ELITE
SOURCE, TARGET = 'fords of isen', 'orthanc'


def army(state, region):
    return next((a for k, a in state.armies.items()
                 if board.sanitize(k) == board.sanitize(region)), None)


def fresh():
    state = gs.setup_initial_game(board.load_board())
    gl.initialize_decks(state)
    for side in (FP, SH):
        state.hands[side].hand = []
    return state


def attack_position(*, mixed_peace=False, face=gs.ActionDieFace.ARMY):
    """Later legal local position, reusing real setup pieces.

    Fords of Isen is a Rohan region next to Orthanc. At-war Gondor has
    marched there. The mixed control leaves a native, not-at-war Rohan
    Regular in place; it need not have crossed a border while at peace.
    """
    state = fresh()
    source = army(state, SOURCE)
    edoras = army(state, 'edoras')
    edoras.leaders.extend(source.leaders)
    source.leaders = []
    if not mixed_peace:
        edoras.units.extend(source.units)
        source.units = []
    else:
        # Setup has two Rohan Regulars here; keep exactly one in this
        # local two-unit fixture and move the other within Rohan.
        edoras.units.extend(source.units[1:])
        source.units = source.units[:1]
    minas = army(state, 'minas tirith')
    selected = [u for u in minas.units if u.kind == REGULAR][:1 if mixed_peace else 2]
    for unit in selected:
        minas.units.remove(unit)
        source.units.append(unit)
    source.leaders.extend(minas.leaders)
    minas.leaders = []
    for nation in (GONDOR, gs.Nation.ISENGARD):
        state.political.boxes[nation] = 3
        state.political.active[nation] = True
    state.political.boxes[ROHAN] = 2
    state.political.active[ROHAN] = True
    fp = gs.ActionDie([face], FP)
    fp.current = face
    sh = gs.ActionDie([gs.ActionDieFace.MUSTER], SH)
    sh.current = gs.ActionDieFace.MUSTER
    state.dice = {FP: [fp], SH: [sh]}
    return state


def target_attacks(state):
    return [a for a in ai.get_legal_actions(state, FP)
            if a.action_type == 'army_attack'
            and board.sanitize(a.params['from']) == SOURCE
            and board.sanitize(a.params['to']) == TARGET]


def battle_pending(state, phase='attacker_rearguard'):
    state.pending_combat = {
        'phase': phase, 'attacker_region': SOURCE, 'defender_region': TARGET,
        'attacker_side': FP, 'defender_side': SH, 'deciding_side': FP,
        'is_siege': False, 'stronghold_region': True, 'rounds_fought': 0}
    return state


def victorious_position(*, mixed_nations=False, named_companions=False):
    state = attack_position(mixed_peace=mixed_nations)
    # Shadow casualties return to reinforcement supply. A cleared Orthanc
    # remains Shadow-controlled until an attacker actually advances.
    army(state, TARGET).units = []
    army(state, TARGET).leaders = []
    state.political.boxes[ROHAN] = 3
    if named_companions:
        source = army(state, SOURCE)
        army(state, 'minas tirith').leaders.extend(source.leaders)
        source.leaders = []
        for name in ('Boromir', 'Gimli'):
            companion = next(c for c in state.fellowship.companions if c.name == name)
            state.fellowship.companions.remove(companion)
            source.leaders.append(companion)
    battle_pending(state, 'victor_advance')
    state.pending_combat['rounds_fought'] = 1
    state.dice[FP][0].used = True
    return state


class ArmyDecisionExposure(unittest.TestCase):
    def test_public_attack_offers_rearguard_before_defender_siege_choice(self):
        state = attack_position()
        self.assertEqual(len(army(state, SOURCE).units), 2)
        attack, = target_attacks(state)
        state = ai.apply_action(state, attack)
        self.assertEqual(state.pending_combat['phase'], 'attacker_rearguard',
                         'RB28/AL36: choose participating figures before the defender reacts')
        self.assertEqual(ai._pending_decision_side(state), FP)

    def test_existing_rearguard_phase_offers_split_as_well_as_full_army(self):
        state = battle_pending(attack_position())
        types = {a.action_type for a in ai.get_legal_actions(state, FP)}
        self.assertIn('rearguard_none', types)
        self.assertIn('rearguard_split', types,
                      'One Regular attacks while the second Regular stays: both armies nonempty')

    def test_mixed_war_army_can_attack_with_its_at_war_portion(self):
        state = attack_position(mixed_peace=True)
        self.assertEqual([(u.nation, u.kind) for u in army(state, SOURCE).units],
                         [(ROHAN, REGULAR), (GONDOR, REGULAR)])
        self.assertEqual(state.political.boxes[ROHAN], 2)
        self.assertTrue(target_attacks(state),
                        'RB28 requires Gondor to attack with Rohan held back; it does not ban the attack')

    def test_prospective_attack_counts_need_distinct_encodings(self):
        state = attack_position()
        actions = [ai.Action(0, 'army_attack', {
            'from': SOURCE, 'to': TARGET, 'side': 'fp',
            'n_regular': n, 'n_elite': 0, 'n_leaders': 1}) for n in (1, 2)]
        encodings = [ai.encode_action(a, state, FP) for a in actions]
        self.assertFalse((encodings[0] == encodings[1]).all(),
                         'One participating Regular and two are different decisions')

    def test_single_unit_has_no_nonempty_rearguard_to_choose(self):
        state = attack_position()
        army(state, 'minas tirith').units.append(army(state, SOURCE).units.pop())
        attack, = target_attacks(state)
        state = ai.apply_action(state, attack)
        self.assertEqual(state.pending_combat['phase'], 'defender_siege_choice')
        self.assertEqual(ai._pending_decision_side(state), SH)

    def test_character_attack_without_a_leader_is_unavailable(self):
        state = attack_position(face=gs.ActionDieFace.CHARACTER)
        army(state, 'minas tirith').leaders.extend(army(state, SOURCE).leaders)
        army(state, SOURCE).leaders = []
        self.assertEqual(target_attacks(state), [])

    def test_victor_can_choose_which_nations_regular_advances(self):
        state = victorious_position(mixed_nations=True)
        one_unit = [a for a in ai.get_legal_actions(state, FP)
                    if a.action_type == 'victor_advance'
                    and a.params.get('n_regular') == 1
                    and a.params.get('n_elite', 0) == 0]
        reachable = set()
        for action in one_unit:
            result = ai.apply_action(state, action)
            units = army(result, TARGET).units
            self.assertEqual(len(units), 1)
            reachable.add(units[0].nation)
        self.assertEqual(reachable, {GONDOR, ROHAN},
                         'RB31 all/part includes either participating nationality')

    def test_victor_can_choose_which_named_companion_advances(self):
        state = victorious_position(named_companions=True)
        one_unit = [a for a in ai.get_legal_actions(state, FP)
                    if a.action_type == 'victor_advance'
                    and a.params.get('n_regular') == 1
                    and a.params.get('n_elite', 0) == 0]
        reachable = set()
        for action in one_unit:
            result = ai.apply_action(state, action)
            reachable.add(frozenset(l.name for l in army(result, TARGET).leaders
                                    if isinstance(l, gs.Companion)))
        self.assertIn(frozenset({'Boromir'}), reachable)
        self.assertIn(frozenset({'Gimli'}), reachable,
                      'Boromir and Gimli are distinct figures, not interchangeable Leader counts')

    def test_literal_dale_retreat_to_erebor_crosses_border_while_not_at_war(self):
        state = fresh()
        boromir = next(c for c in state.fellowship.companions if c.name == 'Boromir')
        state.fellowship.companions.remove(boromir)
        army(state, 'dale').leaders.append(boromir)
        state.political.boxes[gs.Nation.THE_NORTH] = 1
        state.political.active[gs.Nation.THE_NORTH] = True
        state.political.boxes[gs.Nation.DWARVES] = 0
        source = army(state, 'dol guldur')
        attacker = gl.get_or_create_army(state, 'northern rhovanion')
        unit = next(u for u in source.units if u.kind == REGULAR)
        source.units.remove(unit)
        attacker.units.append(unit)
        state.pending_combat = {
            'phase': 'defender_retreat', 'attacker_region': 'northern rhovanion',
            'defender_region': 'dale', 'attacker_side': SH, 'defender_side': FP,
            'deciding_side': FP, 'is_siege': False, 'rounds_fought': 1}
        choices = ai.get_legal_actions(state, FP)
        self.assertFalse(any(a.action_type.startswith('rearguard') for a in choices))
        retreat, = [a for a in choices if a.action_type == 'combat_retreat'
                    and board.sanitize(a.params['to_region']) == 'erebor']
        before = dict(state.political.boxes)
        state = ai.apply_action(state, retreat)
        remaining = army(state, 'dale')
        self.assertFalse(remaining and (remaining.units or remaining.leaders))
        erebor = army(state, 'erebor')
        self.assertEqual(sum(u.nation == gs.Nation.THE_NORTH for u in erebor.units), 1)
        self.assertEqual(sum(u.nation == gs.Nation.DWARVES for u in erebor.units), 3)
        self.assertIn('Boromir', [l.name for l in erebor.leaders if isinstance(l, gs.Companion)])
        self.assertEqual(state.political.boxes, before)

    def test_retreat_cannot_enter_empty_enemy_moria_or_captured_lorien(self):
        state = fresh()
        # AL46's second literal example: Dimrill Dale has an FP army;
        # empty Moria is still enemy-owned, as is captured Lorien.
        dale = army(state, 'dale')
        gl.get_or_create_army(state, 'dimrill dale').units.extend(dale.units)
        dale.units = []
        army(state, 'dol guldur').units.extend(army(state, 'moria').units)
        army(state, 'moria').units = []
        army(state, 'grey havens').units.extend(army(state, 'lorien').units)
        army(state, 'grey havens').leaders.extend(army(state, 'lorien').leaders)
        army(state, 'lorien').units = []
        army(state, 'lorien').leaders = []
        state.enemy_controlled[SH].add('lorien')
        source = army(state, 'dol guldur')
        attacker = gl.get_or_create_army(state, 'parth celebrant')
        unit = next(u for u in source.units if u.kind == REGULAR)
        source.units.remove(unit)
        attacker.units.append(unit)
        state.pending_combat = {
            'phase': 'defender_retreat', 'attacker_region': 'parth celebrant',
            'defender_region': 'dimrill dale', 'attacker_side': SH, 'defender_side': FP,
            'deciding_side': FP, 'is_siege': False, 'rounds_fought': 1}
        actions = ai.get_legal_actions(state, FP)
        destinations = {board.sanitize(a.params['to_region']) for a in actions
                        if a.action_type == 'combat_retreat'}
        self.assertNotIn('moria', destinations)
        self.assertNotIn('lorien', destinations)
        self.assertIn('combat_no_retreat', {a.action_type for a in actions})


if __name__ == '__main__':
    unittest.main(verbosity=2)
