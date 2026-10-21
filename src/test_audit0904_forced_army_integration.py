"""Battle consequences use the declared force, with literal RB28/31/35 oracles.

Only setup helpers are imported. Expected outcomes below are piece counts
from the printed rules, not regenerated action menus or engine constants.
"""
from collections import Counter
import copy
from unittest.mock import patch
import unittest

import numpy as np
import ai
import combat
import game_logic as gl
import game_state as gs
from cards import get_card
from test_audit0904_forced_army import (
    position, attack, army, participants, north_count, elf_leaders,
    FP, SH, NORTH, ELVES, REGULAR, ELITE, SOURCE, TARGET)


def counted(state, region):
    a = army(state, region)
    return Counter((u.nation, u.kind) for u in a.units) if a else Counter()


class ForcedArmyIntegration(unittest.TestCase):
    def test_actual_combat_round_rolls_one_attacker_die_and_no_elf_reroll(self):
        state = attack(position())
        with patch.object(combat, 'combat_roll', return_value=([1], 0)) as rolls:
            result = gl.battle(state, SOURCE, TARGET, choices={'attacker_side': FP})
        self.assertEqual([call.args[0] for call in rolls.call_args_list], [1, 1])
        self.assertEqual(north_count(state, SOURCE), 2)
        self.assertEqual(elf_leaders(state, SOURCE), 1)
        self.assertFalse(result.attacker_destroyed)

    def test_actual_combat_wipe_eliminates_participating_leader_only(self):
        state = attack(position(north_leader=True))
        with patch.object(combat, 'combat_roll', side_effect=[([1], 0), ([6], 1), ([1], 0)]):
            result = gl.battle(state, SOURCE, TARGET, choices={'attacker_side': FP})
        self.assertTrue(result.attacker_destroyed)
        self.assertEqual(counted(state, SOURCE), Counter({(NORTH, REGULAR): 1}))
        self.assertEqual([l.nation for l in army(state, SOURCE).leaders], [ELVES])

    def test_deferred_round_wipe_preserves_guard_leader_and_tracks_only_one_loss(self):
        state = attack(position(north_leader=True))
        pc = state.pending_combat
        before = state.fp_casualties.get(NORTH, {}).get('regular', 0)
        pc.update(atk_hits=5, def_hits=0, alloc_stage='attacker')
        ai._enter_casualty_allocation(state, pc)
        self.assertIsNone(state.pending_combat)
        self.assertEqual(counted(state, SOURCE), Counter({(NORTH, REGULAR): 1}))
        self.assertEqual([l.nation for l in army(state, SOURCE).leaders], [ELVES])
        self.assertEqual(state.fp_casualties[NORTH]['regular'] - before, 1)

    def test_political_change_after_declaration_does_not_release_protected_elf(self):
        state = attack(position())
        state.political.boxes[ELVES] = 3
        units, leaders, leadership = participants(state)
        self.assertEqual(units, Counter({(NORTH, REGULAR): 1}))
        self.assertEqual(leaders, [])
        self.assertEqual(leadership, 0)

    def test_heroic_death_cannot_use_a_peaceful_leader_in_the_rear(self):
        state = attack(position())
        cid = 'FREE_PEOPLES_Character_There_is_Another_Way'
        state.hands[FP].hand = [cid]
        state.pending_combat.update(phase='attacker_combat_card', deciding_side=FP)
        self.assertEqual([a for a in ai.get_legal_actions(state, FP)
                          if a.action_type == 'combat_card_play' and a.params.get('card_id') == cid], [])
        positive = attack(position(north_leader=True))
        positive.hands[FP].hand = [cid]
        positive.pending_combat.update(phase='attacker_combat_card', deciding_side=FP)
        self.assertTrue([a for a in ai.get_legal_actions(positive, FP)
                         if a.action_type == 'combat_card_play' and a.params.get('card_id') == cid])

    def test_great_host_counts_fighting_armies_and_its_hit_cannot_reach_guard(self):
        # One Shadow vs one participating North is NOT twice as many.
        state = attack(position())
        cid = 'SHADOW_Strategy_Stormcrow'
        self.assertEqual(get_card(cid).combat_name, 'Great Host')
        out = {}
        combat.resume_deferred_post_casualty(state, SOURCE, TARGET, FP, SH,
                                            def_card_id=cid, choices={'defer_post_hits': True}, out=out)
        self.assertEqual(out['post_hits_on_attacker'], 0)
        # Two Shadow vs ONE fighter is twice as many, regardless of guard.
        army(state, TARGET).units.append(gs.Unit(gs.Nation.SAURON, REGULAR))
        out = {}
        combat.resume_deferred_post_casualty(state, SOURCE, TARGET, FP, SH,
                                            def_card_id=cid, choices={'defer_post_hits': True}, out=out)
        self.assertEqual(out['post_hits_on_attacker'], 1)
        ai._apply_allocation(state, SOURCE, FP, out['post_hits_on_attacker'], 0)
        self.assertEqual(counted(state, SOURCE), Counter({(NORTH, REGULAR): 1}))
        self.assertEqual(elf_leaders(state, SOURCE), 1)

    def test_onslaught_enemy_hits_stop_at_last_participant(self):
        state = attack(position())
        army(state, TARGET).units.append(gs.Unit(gs.Nation.SAURON, REGULAR))
        with patch.object(combat, 'combat_roll', return_value=([6, 6], 2)):
            own_dead, enemy_dead, _, hits = combat.apply_onslaught_choice(
                state, TARGET, SOURCE, SH, FP, 2, enemy_is_defender=False)
        self.assertTrue(own_dead)
        self.assertTrue(enemy_dead)
        self.assertEqual(hits, 2)
        self.assertEqual(counted(state, SOURCE), Counter({(NORTH, REGULAR): 1}))
        self.assertEqual(elf_leaders(state, SOURCE), 1)

    def test_defender_withdrawal_keeps_partition_through_advance_and_siege(self):
        state = attack(position())
        state = ai.apply_action(state, ai.Action(-1, 'retreat_into_siege', {}))
        self.assertEqual(state.pending_combat['phase'], 'victor_advance')
        advances = [a for a in ai.get_legal_actions(state, FP) if a.action_type == 'victor_advance']
        self.assertEqual({(a.params['n_regular'], a.params['n_elite']) for a in advances}, {(1, 0)})
        state = ai.apply_action(state, ai.Action(-1, 'victor_advance', {
            'from': SOURCE, 'to': TARGET, 'side': 'fp',
            'n_regular': 1, 'n_elite': 0, 'take_leaders': True}))
        self.assertEqual(counted(state, SOURCE), Counter({(NORTH, REGULAR): 1}))
        self.assertEqual(elf_leaders(state, SOURCE), 1)
        self.assertEqual(north_count(state, TARGET), 1)
        self.assertEqual(elf_leaders(state, TARGET), 0)
        self.assertIn(TARGET, state.besieged_strongholds)

    def test_won_round_creates_advance_pending_with_the_same_rear(self):
        state = attack(position())
        pc = state.pending_combat
        army(state, TARGET).units = []
        ai._finish_battle(state, pc, False, True)
        menu = [a for a in ai.get_legal_actions(state, FP) if a.action_type == 'victor_advance']
        self.assertEqual({a.params['n_regular'] for a in menu}, {1})

    def test_kernel_uses_one_fighter_and_zero_leaders(self):
        state = attack(position())
        spec = ai._kernel_army_spec(state, SOURCE, FP)
        self.assertEqual((spec.regulars, spec.elites, spec.elites2, spec.leaders), (1, 0, 0, ()))

    def test_action_encoding_and_cache_reflect_forced_composition(self):
        state = position()
        action = ai.Action(0, 'army_attack', {'from': SOURCE, 'to': TARGET, 'side': 'fp'})
        before = ai.encode_action(action, state, FP)
        key_before = ai._action_cache_key(action, state, FP)
        self.assertAlmostEqual(float(before[ai._ACT_COMP_OFF]), .1)
        self.assertAlmostEqual(float(before[ai._ACT_COMP_OFF + 2]), 0.)
        state.political.boxes[ELVES] = 3
        after = ai.encode_action(action, state, FP)
        self.assertAlmostEqual(float(after[ai._ACT_COMP_OFF]), .2)
        self.assertAlmostEqual(float(after[ai._ACT_COMP_OFF + 2]), 1 / 7)
        self.assertNotEqual(key_before, ai._action_cache_key(action, state, FP))

    def test_protected_peaceful_elite_cannot_pay_for_a_siege_press(self):
        state = position(peaceful_unit=True)
        source = army(state, SOURCE)
        protected = next(u for u in source.units if u.nation == ELVES)
        protected.kind = ELITE
        state = attack(state)
        state.pending_combat.update(phase='combat_round_result', is_siege=True, deciding_side=FP)
        menu = ai.get_legal_actions(state, FP)
        self.assertEqual([a.action_type for a in menu], ['siege_no_press'])
        with self.assertRaises((ValueError, RuntimeError)):
            ai._execute_action(state, ai.Action(-1, 'siege_press', {}))
        self.assertEqual(counted(state, SOURCE)[ELVES, ELITE], 1)

    def test_co_located_siege_mutual_wipe_rear_takes_stronghold(self):
        # Almanac36 explicitly gives the surviving rearguard control after
        # dual elimination in a siege or sortie; it is still a real Army.
        state = position()
        source, target = army(state, SOURCE), army(state, TARGET)
        target.units.extend(source.units)
        target.leaders.extend(source.leaders)
        state.armies = {k: a for k, a in state.armies.items() if a is not source}
        state.besieged_strongholds.add(TARGET)
        state = ai.apply_action(state, ai.Action(0, 'army_attack', {
            'from': TARGET, 'to': TARGET, 'side': 'fp', 'siege': True}))
        with patch.object(combat, 'combat_roll', return_value=([6], 1)):
            result = gl.battle(state, TARGET, TARGET, is_siege=True, choices={'attacker_side': FP})
        self.assertTrue(result.attacker_destroyed)
        self.assertTrue(result.defender_destroyed)
        self.assertEqual(counted(state, TARGET), Counter({(NORTH, REGULAR): 1}))
        self.assertEqual(elf_leaders(state, TARGET), 1)
        self.assertNotIn(TARGET, state.enemy_controlled[SH])

    def test_new_card_battle_entry_freezes_compulsory_partition(self):
        # Help Unlooked For and other printed battle events enter this
        # engine path directly; there is no regular army_attack action.
        state = position()
        pc = dict(attacker_region=SOURCE, defender_region=TARGET,
                  attacker_side=FP, defender_side=SH, deciding_side=FP,
                  phase='combat_setup', rounds_fought=0, is_siege=False,
                  def_dice_reduction=1)
        state.pending_combat = pc
        state.hands[FP].hand = ['FREE_PEOPLES_Character_There_is_Another_Way']
        ai._run_combat_round(state, pc)
        self.assertEqual(participants(state)[0], Counter({(NORTH, REGULAR): 1}))
        self.assertEqual(participants(state)[1], [])

    def test_post_victory_advance_encoding_keeps_the_physical_guard_in_source_features(self):
        state = attack(position())
        state.pending_combat.update(phase='victor_advance', deciding_side=FP)
        action = ai.Action(-1, 'victor_advance', {'from': SOURCE, 'to': TARGET,
            'side': 'fp', 'n_regular': 1, 'n_elite': 0, 'take_leaders': True})
        encoded = ai.encode_action(action, state, FP)
        self.assertAlmostEqual(float(encoded[ai._ACT_LEAVE_OFF]), .1)
        self.assertEqual(float(encoded[ai._ACT_EMPTIES_OFF]), 0.)

    def test_same_composition_battles_from_different_sources_have_distinct_observations(self):
        state = attack(position())
        state.armies['carrock'] = copy.deepcopy(army(state, SOURCE))
        state.armies['carrock'].region = state.board.region('carrock')
        # Literal rear in both equally composed source armies; this also
        # probes an older encoder that does not understand the new field.
        signature = ('forced_rearguard_v1', (('THE_NORTH', 'REGULAR', 1),), (('ELVES', 1),))
        state.pending_combat['attacker_rearguard'] = (SOURCE, 'FREE_PEOPLES', signature)
        other = ai.clone_state(state)
        other.pending_combat['attacker_region'] = 'carrock'
        other.pending_combat['attacker_rearguard'] = ('carrock', 'FREE_PEOPLES', signature)
        first, second = ai.encode_state(state, FP), ai.encode_state(other, FP)
        self.assertFalse(np.array_equal(first, second))
        start = ai.get_pending_context_start()
        self.assertEqual(first[start + ai.REGION_INDEX[SOURCE]], -1.)
        self.assertEqual(first[start + ai.REGION_INDEX[TARGET]], 1.)
        self.assertEqual(second[start + ai.REGION_INDEX['carrock']], -1.)
        self.assertEqual(second[start + ai.REGION_INDEX[TARGET]], 1.)

    def test_kernel_exact_four_outcomes_and_engine_witnesses_preserve_rear(self):
        # Field battle, one die per Army, hit on 5/6, no Leadership.
        # The four independent Bernoulli outcomes have masses 4/9,2/9,2/9,1/9.
        state = attack(position())
        state.pending_combat.update(phase='defender_combat_card', deciding_side=SH,
                                    atk_combat_card=None, is_siege=False)
        action = ai.Action(-1, 'combat_card_none', {'side': 'sh'})
        probe = ai.apply_action(state, action)
        parent = ai.MCTSNode(state, SH)
        node = ai.MCTSNode(None, SH, parent=parent, action=action)
        node.is_chance = True
        self.assertTrue(ai._kernel_expand_combat(node, probe))
        expected = {(2, 1): 4 / 9, (2, 0): 2 / 9,
                    (1, 1): 2 / 9, (1, 0): 1 / 9}
        seen = {}
        for branch in node.outcomes:
            target = army(branch.state, TARGET)
            key = (north_count(branch.state, SOURCE), len(target.units) if target else 0)
            seen[key] = seen.get(key, 0.) + branch.prior
            self.assertEqual(elf_leaders(branch.state, SOURCE), 1)
        self.assertEqual(set(seen), set(expected))
        for key in expected:
            self.assertAlmostEqual(seen[key], expected[key])
        self.assertAlmostEqual(node.kernel_cover, 1.)

    def test_round_metadata_can_supply_kernel_formation_without_a_live_pending(self):
        state = attack(position())
        state.pending_combat.update(phase='defender_combat_card', deciding_side=SH,
                                    atk_combat_card=None, is_siege=False)
        probe = ai.apply_action(state, ai.Action(-1, 'combat_card_none', {'side': 'sh'}))
        info = probe._last_round_call
        state.pending_combat = None
        context = ai._kernel_ctx(state, info, None, None, defer=True)
        self.assertEqual((context.attacker.regulars, context.attacker.leaders), (1, ()))

    def test_help_unlooked_for_checks_that_an_attacker_remains_after_guard(self):
        import card_options
        import event_effects
        cid = 'FREE_PEOPLES_Strategy_Help_Unlooked_For'
        for n in (1, 2):
            with self.subTest(north_regulars=n):
                state = position(units=n)
                state.enemy_controlled[SH].discard(TARGET)
                state.besieged_strongholds.add(TARGET)
                # One Elven unit is inside; the Sauron unit besieges it.
                elf = next(u for u in army(state, 'lorien').units
                           if u.nation == ELVES and u.kind == REGULAR)
                army(state, 'lorien').units.remove(elf)
                army(state, TARGET).units.insert(0, elf)
                options = card_options.CARD_OPTION_SPECS['help unlooked for'](state, FP)
                self.assertEqual({'stronghold': TARGET, 'attack_from': SOURCE} in options, n == 2)
                if n == 1:
                    event_effects._effect_help_unlooked_for(state, get_card(cid),
                        {'stronghold': TARGET, 'attack_from': SOURCE})
                    self.assertIsNone(state.pending_combat)
                else:
                    state.hands[FP].hand = [cid]
                    state = ai.apply_action(state, ai.Action(0, 'play_event', {'card_id': cid}))
                    self.assertIsNotNone(state.pending_combat)
                    self.assertEqual(participants(state)[0], Counter({(NORTH, REGULAR): 1}))
                    self.assertEqual(participants(state)[1], [])
                    self.assertTrue(state.dice[FP][0].used)
                    self.assertNotIn(cid, state.hands[FP].hand)

    def test_shadow_peaceful_elite_neither_enables_card_nor_pays_onslaught(self):
        state = position()
        state.political.boxes[gs.Nation.SAURON] = 3
        state.political.boxes[gs.Nation.ISENGARD] = 2
        elite = next(u for u in army(state, 'orthanc').units if u.kind == ELITE)
        army(state, 'orthanc').units.remove(elite)
        army(state, TARGET).units.append(elite)
        state.dice[SH][0].current = gs.ActionDieFace.ARMY
        state = ai.apply_action(state, ai.Action(0, 'army_attack', {
            'from': TARGET, 'to': SOURCE, 'side': 'sh'}))
        cid = 'SHADOW_Strategy_Olog-hai'
        self.assertEqual(get_card(cid).combat_name, 'We Come to Kill')
        state.hands[SH].hand = [cid]
        state.pending_combat.update(phase='attacker_combat_card', deciding_side=SH)
        self.assertEqual([a for a in ai.get_legal_actions(state, SH)
                          if a.action_type == 'combat_card_play' and a.params.get('card_id') == cid], [])
        self.assertEqual(combat.onslaught_max_sacrifice(state, TARGET, SH), 1)
        with patch.object(combat, 'combat_roll', return_value=([1], 0)):
            own_dead, _, _, _ = combat.apply_onslaught_choice(state, TARGET, SOURCE, SH, FP, 1)
        self.assertTrue(own_dead)
        self.assertEqual(counted(state, TARGET), Counter({(gs.Nation.ISENGARD, ELITE): 1}))

    def test_frozen_partition_is_visible_to_state_encoding_and_both_fingerprints(self):
        state = attack(position())
        other = ai.clone_state(state)
        other.pending_combat['attacker_rearguard'] = (SOURCE, 'FREE_PEOPLES',
            ('forced_rearguard_v1', (), ()))
        self.assertFalse(np.array_equal(ai.encode_state(state, FP), ai.encode_state(other, FP)))
        self.assertNotEqual(ai._state_sig(state), ai._state_sig(other))
        self.assertNotEqual(ai._chance_sig(state), ai._chance_sig(other))


class VictoryAdvanceStacking(unittest.TestCase):
    """Amendment 1.222: a relief advance merges, then returns excess.

    Literal counts are the oracle; executing a generated action alone is
    insufficient. These positions do not depend on retained pilot files.
    """

    def position(self, side=FP, destination_count=3):
        import board
        state = gs.setup_initial_game(board.load_board())
        gl.initialize_decks(state)
        for player in (FP, SH):
            state.hands[player].hand = []
        nation = gs.Nation.GONDOR if side == FP else gs.Nation.ISENGARD
        # Redistribute this nation's available figures into a relief army
        # (5 Regulars + 3 Elites) and a friendly surviving garrison.
        # Both individual armies are below ten; their merge can exceed it.
        for value in state.armies.values():
            value.units = [u for u in value.units if u.nation != nation]
        for region in ('fords of isen', 'orthanc'):
            value = gl.get_or_create_army(state, region)
            value.units = []
            value.leaders = []
        army(state, 'fords of isen').units = ([gs.Unit(nation, REGULAR) for _ in range(5)]
                                    + [gs.Unit(nation, ELITE) for _ in range(3)])
        army(state, 'orthanc').units = [gs.Unit(nation, REGULAR)
                                     for _ in range(destination_count)]
        state.political.boxes[nation] = 3
        state.political.active[nation] = True
        if side == FP:
            state.enemy_controlled[FP].add('orthanc')
        state.pending_combat = {
            'phase': 'victor_advance', 'attacker_region': 'fords of isen',
            'defender_region': 'orthanc', 'attacker_side': side,
            'defender_side': SH if side == FP else FP,
            'deciding_side': side, 'is_siege': False,
            'stronghold_region': True, 'rounds_fought': 1,
            'post_retreat': True}
        action, = [a for a in ai.get_legal_actions(state, side)
                   if a.action_type == 'victor_advance'
                   and a.params.get('n_regular') == 5
                   and a.params.get('n_elite') == 3]
        return state, action, nation

    def test_eleven_merge_requires_owner_trim_and_returns_to_reinforcements(self):
        from collections import Counter
        for side in (FP, SH):
            with self.subTest(side=side):
                state, action, nation = self.position(side)
                before = Counter((u.nation, u.kind) for a in state.armies.values()
                                 for u in a.units)
                supply = ai.figures_available(state, nation, 'regular')
                casualties = {n: dict(v) for n, v in state.fp_casualties.items()}
                moved = ai.apply_action(state, action)
                self.assertEqual(Counter((u.nation, u.kind)
                                         for a in moved.armies.values()
                                         for u in a.units), before)
                self.assertEqual(len(army(moved, 'orthanc').units), 11)
                self.assertFalse(army(moved, 'fords of isen') and army(moved, 'fords of isen').units)
                self.assertEqual(len(army(state, 'fords of isen').units), 8)
                self.assertEqual(len(army(state, 'orthanc').units), 3)
                self.assertIsNone(moved.pending_combat)
                self.assertEqual(moved.pending_card_option['card_id'], '__overstack__')
                self.assertEqual(ai._pending_decision_side(moved), side)
                choices = ai.get_legal_actions(moved, side)
                self.assertTrue(choices)
                self.assertTrue(all(a.action_type == 'card_option' for a in choices))
                # No pass or ordinary die action can bypass the excess.
                for option in choices:
                    trimmed = ai.apply_action(moved, option)
                    self.assertEqual(len(army(trimmed, 'orthanc').units), 10)
                    self.assertIsNone(trimmed.pending_card_option)
                    self.assertEqual(trimmed.fp_casualties, casualties)
                    after = Counter((u.nation, u.kind) for a in trimmed.armies.values()
                                    for u in a.units)
                    removed = before - after
                    self.assertEqual(sum(removed.values()), 1)
                    self.assertEqual({n for n, _ in removed}, {nation})
                    kind = next(iter(removed))[1]
                    typ = 'regular' if kind == REGULAR else 'elite'
                    self.assertEqual(ai.figures_available(trimmed, nation, typ),
                                     ai.figures_available(state, nation, typ) + 1)
                self.assertEqual(ai.figures_available(moved, nation, 'regular'), supply)

    def test_exactly_ten_merge_has_no_trim(self):
        for side in (FP, SH):
            with self.subTest(side=side):
                state, action, _ = self.position(side, destination_count=2)
                moved = ai.apply_action(state, action)
                self.assertEqual(len(army(moved, 'orthanc').units), 10)
                self.assertIsNone(moved.pending_card_option)
                self.assertIsNone(moved.pending_combat)

    def test_ordinary_full_and_partial_moves_still_reject_eleven(self):
        for split in (False, True):
            with self.subTest(split=split):
                state, _, _ = self.position()
                state.pending_combat = None
                before = {k: tuple(id(u) for u in a.units)
                          for k, a in state.armies.items()}
                kwargs = dict(move_units=5, move_elite=3, move_leaders=0) if split else {}
                with self.assertRaisesRegex(RuntimeError, 'stacking limit'):
                    gl.move_army(state, 'fords of isen', 'orthanc', side=FP, **kwargs)
                self.assertEqual({k: tuple(id(u) for u in a.units)
                                  for k, a in state.armies.items()}, before)

    def test_deferred_cap_requires_victory_advance_context(self):
        for context in (None, {'phase': 'defender_retreat'}):
            with self.subTest(context=context):
                state, _, _ = self.position()
                with self.assertRaisesRegex(ValueError, 'victory-advance context'):
                    gl.move_army(state, 'fords of isen', 'orthanc', side=FP,
                                 defer_stacking=True, combat_context=context)


if __name__ == '__main__':
    unittest.main()
