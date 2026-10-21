"""Literal Faramir's Rangers rules: RB22/27 and supplied Almanac18.

Choose a Shadow Army in Osgiliath/North/South Ithilien and strike it,
then recruit one Gondor Regular OR Elite and one Leader if an FP Army
stands in Osgiliath. Recruitment-only requires that FP Army. Available
effects are compulsory; exhausted supplies do not erase positional legality.
The AI prefers Elite, while the human unit choice follows any strike.
"""
from collections import Counter
import copy
import unittest
from unittest.mock import patch

import numpy as np
import ai
import board
import card_options
import combat
import event_effects as ee
import game_logic as gl
import game_state as gs
import state_invariants

FP, SH = gs.Side.FREE_PEOPLES, gs.Side.SHADOW
GONDOR, SAURON = gs.Nation.GONDOR, gs.Nation.SAURON
R, E = gs.UnitKind.REGULAR, gs.UnitKind.ELITE
CID = "FREE_PEOPLES_Strategy_Faramir's_Rangers"
TITLE = "Faramir's Rangers"


def army(state, region):
    return next((a for k, a in state.armies.items() if board.sanitize(k) == region), None)


def before_play(*, fp_present=True, targets=(), target_elite=False):
    state = gs.setup_initial_game(board.load_board())
    gl.initialize_decks(state)
    state.hands[FP].hand = [CID]
    state.hands[SH].hand = []
    for name in ('fp_char_deck', 'fp_strat_deck', 'sh_char_deck', 'sh_strat_deck'):
        deck = getattr(state, name)
        if CID in deck:
            deck.remove(CID)
    if not fp_present:
        src, dst = army(state, 'osgiliath'), army(state, 'minas tirith')
        dst.units.extend(src.units)
        src.units.clear()
    for region in targets:
        dest = army(state, region)
        if dest is None:
            dest = gs.Army(region=region)
            state.armies[region] = dest
        for kind in ((R, E) if target_elite else (R,)):
            src = next(a for a in state.armies.values()
                       if a is not dest and any(u.nation == SAURON and u.kind == kind for u in a.units))
            unit = next(u for u in src.units if u.nation == SAURON and u.kind == kind)
            src.units.remove(unit)
            dest.units.append(unit)
    die = gs.ActionDie([gs.ActionDieFace.MUSTER_ARMY], FP)
    die.current = gs.ActionDieFace.MUSTER_ARMY
    unused = gs.ActionDie([gs.ActionDieFace.CHARACTER], FP)
    unused.current = gs.ActionDieFace.CHARACTER
    shadow = gs.ActionDie([gs.ActionDieFace.MUSTER], SH)
    shadow.current = gs.ActionDieFace.MUSTER
    state.dice = {FP: [die, unused], SH: [shadow]}
    return state


def exhaust(state, *kinds):
    # Literal component counts RB7, independent of engine cap helpers.
    caps = {'regular': 15, 'elite': 5, 'leader': 4}
    for kind in kinds:
        if kind == 'leader':
            deployed = sum(isinstance(l, gs.Leader) and l.nation == GONDOR
                           for a in state.armies.values() for l in a.leaders)
        else:
            wanted = R if kind == 'regular' else E
            deployed = sum(u.nation == GONDOR and u.kind == wanted
                           for a in state.armies.values() for u in a.units)
        state.fp_casualties.setdefault(GONDOR, {})[kind] = caps[kind] - deployed


def counts(state, region='osgiliath', nation=GONDOR):
    a = army(state, region)
    if a is None:
        return 0, 0, 0
    return (sum(u.nation == nation and u.kind == R for u in a.units),
            sum(u.nation == nation and u.kind == E for u in a.units),
            sum(isinstance(l, gs.Leader) and l.nation == nation for l in a.leaders))


def accounting(state):
    return (list(state.hands[FP].hand), copy.deepcopy(state.discard_face_up),
            {s: [d.used for d in ds] for s, ds in state.dice.items()},
            copy.deepcopy(getattr(state, 'pending_card_option', None)),
            {k: Counter((u.nation, u.kind) for u in a.units) for k, a in state.armies.items()},
            dict(state.political.boxes))


def play(state, choices=None):
    params = {'card_id': CID, 'side': 'fp'}
    if choices is not None:
        params['choices'] = choices
    return ai.apply_action(state, ai.Action(0, 'play_event', params))


class FaramirRules(unittest.TestCase):
    def recruitment_menu(self, state):
        pending = state.pending_card_option
        self.assertIsNotNone(pending, 'Human recruitment needs a real post-strike choice')
        self.assertEqual(pending['card_id'], CID)
        self.assertTrue(pending.get('faramir_recruit'))
        self.assertTrue(pending.get('direct'))
        self.assertEqual(ai._pending_decision_side(state), FP)
        menu = ai.get_legal_actions(state, FP)
        self.assertEqual({a.params['choices'].get('unit_type') for a in menu}, {'regular', 'elite'})
        self.assertEqual(len(menu), 2)
        self.assertTrue(all(a.params['choices'].get('_far_recruit') is True for a in menu))
        return menu

    def choose(self, state, kind):
        self.recruitment_menu(state)
        return ai.apply_action(state, ai.Action(-1, 'card_option', {
            'card_id': CID, 'side': 'fp', 'choices': {'_far_recruit': True, 'unit_type': kind}}))

    def paid_once(self, state):
        self.assertEqual(state.hands[FP].hand, [])
        self.assertEqual(state.discard_face_up['fp_strategy'].count(CID), 1)
        self.assertEqual([d.used for d in state.dice[FP]], [True, False])
        self.assertEqual([d.used for d in state.dice[SH]], [False])
        self.assertIsNone(state.pending_card_option)
        self.assertIsNone(state.pending_muster_die)
        self.assertIsNone(state.pending_army_move_die)

    def test_empty_area_fails_printed_requirement(self):
        self.assertFalse(ee.check_event_requirement(before_play(fp_present=False), CID)[0])

    def test_empty_area_has_no_public_card_play(self):
        self.assertFalse(any(a.action_type == 'play_event' and a.params.get('card_id') == CID
                             for a in ai.get_legal_actions(before_play(fp_present=False), FP)))

    def test_empty_area_direct_execution_rejects_before_card_and_die_payment(self):
        state = before_play(fp_present=False)
        initial = accounting(state)
        with self.assertRaises((ValueError, RuntimeError)):
            ai._execute_action(state, ai.Action(0, 'play_event', {'card_id': CID, 'side': 'fp'}))
        self.assertEqual(accounting(state), initial)

    def test_empty_area_direct_card_api_keeps_card(self):
        state = before_play(fp_present=False)
        initial = accounting(state)
        self.assertFalse(ee.play_event_card(state, FP, CID)['success'])
        self.assertEqual(accounting(state), initial)

    def test_recruitment_only_is_legal_without_zero_choice_notice(self):
        state = before_play()
        self.assertTrue(ee.check_event_requirement(state, CID)[0])
        mark = len(state_invariants.VIOLATIONS)
        with patch.object(ee, '_d6_batch', side_effect=AssertionError('Recruitment-only must not roll')):
            state = play(state)
        self.recruitment_menu(state)
        self.assertEqual(counts(state), (2, 0, 0))
        self.assertFalse(state_invariants.VIOLATIONS[mark:])

    def test_human_regular_places_regular_and_one_leader_once(self):
        state = self.choose(play(before_play()), 'regular')
        self.assertEqual(counts(state), (3, 0, 1))
        self.paid_once(state)

    def test_human_elite_places_elite_and_one_leader_once(self):
        state = self.choose(play(before_play()), 'elite')
        self.assertEqual(counts(state), (2, 1, 1))
        self.paid_once(state)

    def test_ai_projects_elite_and_preserves_stage_marker(self):
        state = play(before_play())
        self.recruitment_menu(state)
        projection = ai.get_ai_policy(state, FP)
        self.assertEqual(len(projection.actions), 1)
        self.assertEqual(projection.forced_reason, 'recruit_elite_if_available')
        self.assertEqual(projection.actions[0].params['choices'],
                         {'_far_recruit': True, 'unit_type': 'elite'})
        self.assertEqual(counts(ai.apply_action(state, projection.actions[0])), (2, 1, 1))

    def test_regular_and_elite_encodings_are_distinct(self):
        state = play(before_play())
        menu = self.recruitment_menu(state)
        self.assertFalse(np.array_equal(ai.encode_action(menu[0], state, FP), ai.encode_action(menu[1], state, FP)))

    def test_single_remaining_regular_pool_is_forced(self):
        state = before_play()
        exhaust(state, 'elite')
        state = play(state)
        self.assertEqual(counts(state), (3, 0, 1))
        self.paid_once(state)

    def test_leader_only_when_both_unit_pools_exhausted(self):
        state = before_play()
        exhaust(state, 'regular', 'elite')
        state = play(state)
        self.assertEqual(counts(state), (2, 0, 1))
        self.paid_once(state)

    def test_all_supplies_exhausted_preserves_positional_playability(self):
        state = before_play()
        exhaust(state, 'regular', 'elite', 'leader')
        self.assertTrue(ee.check_event_requirement(state, CID)[0])
        mark = len(state_invariants.VIOLATIONS)
        state = play(state)
        self.assertEqual(counts(state), (2, 0, 0))
        self.paid_once(state)
        self.assertFalse(state_invariants.VIOLATIONS[mark:])

    def test_missing_leader_supply_keeps_both_human_unit_alternatives(self):
        state = before_play()
        exhaust(state, 'leader')
        state = self.choose(play(state), 'regular')
        self.assertEqual(counts(state), (3, 0, 0))
        self.paid_once(state)

    def test_recruitment_only_accepts_an_fp_army_without_gondor_units(self):
        state = before_play(fp_present=False)
        source, dest = army(state, 'dale'), army(state, 'osgiliath')
        dest.units.extend(source.units)
        dest.leaders.extend(source.leaders)
        source.units.clear()
        source.leaders.clear()
        self.assertEqual(counts(state), (0, 0, 0))
        politics = dict(state.political.boxes)
        state = self.choose(play(state), 'regular')
        self.assertEqual(counts(state), (1, 0, 1))
        self.assertEqual(counts(state, nation=gs.Nation.THE_NORTH), (1, 0, 1))
        self.assertEqual(state.political.boxes, politics)
        self.paid_once(state)

    def test_gandalf_bonus_waits_until_unit_recruitment_has_finished(self):
        state = before_play()
        state.dice[FP][0].current = gs.ActionDieFace.EVENT
        remaining_strategy = len(state.fp_strat_deck)
        state = play(state)
        self.recruitment_menu(state)
        self.assertEqual(len(state.fp_strat_deck), remaining_strategy)
        self.assertEqual(state.discard_face_up['fp_strategy'].count(CID), 1)
        state = self.choose(state, 'regular')
        self.assertEqual(counts(state), (3, 0, 1))
        self.assertEqual(state.pending_card_option['card_id'], '__gandalf_draw__')
        self.assertEqual(state.pending_card_option['deck'], 'strategy')
        self.assertEqual(len(state.fp_strat_deck), remaining_strategy,
                         'Human bonus acceptance is a later pending')

    def test_strike_only_miss_still_spends_card_without_recruitment(self):
        state = before_play(fp_present=False, targets=('north ithilien',))
        politics = dict(state.political.boxes)
        with patch.object(ee, '_d6_batch', return_value=[1, 1, 1]) as rolls:
            state = play(state)
        self.assertEqual(rolls.call_count, 1)
        self.assertEqual(counts(state, 'north ithilien', SAURON), (1, 0, 0))
        self.assertEqual(counts(state), (0, 0, 0))
        self.assertEqual(state.political.boxes, politics)
        self.paid_once(state)

    def test_strike_finishes_before_human_recruitment_choice(self):
        with patch.object(ee, '_d6_batch', return_value=[6, 1, 1]) as rolls:
            state = play(before_play(targets=('north ithilien',)))
            self.assertEqual(counts(state, 'north ithilien', SAURON), (0, 0, 0))
            self.assertEqual(counts(state), (2, 0, 0))
            state = self.choose(state, 'regular')
        self.assertEqual(rolls.call_count, 1, 'Recruitment continuation never repeats the hit roll')
        self.assertEqual(counts(state), (3, 0, 1))
        self.paid_once(state)

    def test_shadow_casualty_then_fp_regular_choice_preserves_owner_order(self):
        with patch.object(ee, '_d6_batch', return_value=[6, 1, 1]) as rolls:
            state = play(before_play(targets=('north ithilien',), target_elite=True))
            self.assertTrue(state.pending_card_option.get('card_casualty'))
            self.assertEqual(ai._pending_decision_side(state), SH)
            self.assertEqual(counts(state), (2, 0, 0))
            state = ai.apply_action(state, ai.Action(-1, 'card_option', {
                'card_id': CID, 'side': 'sh',
                'choices': {'casualty_operation': 'remove_regular', 'nation': SAURON,
                            'target_region': 'north ithilien'}}))
            self.assertEqual(counts(state, 'north ithilien', SAURON), (0, 1, 0))
            state = self.choose(state, 'regular')
        self.assertEqual(rolls.call_count, 1)
        self.assertEqual(counts(state), (3, 0, 1))
        self.paid_once(state)

    def test_multiple_targets_remain_a_real_choice_before_recruitment(self):
        with patch.object(ee, '_d6_batch', return_value=[1, 1, 1]) as rolls:
            state = play(before_play(targets=('north ithilien', 'south ithilien')))
            self.assertEqual(rolls.call_count, 0)
            menu = ai.get_legal_actions(state, FP)
            self.assertEqual([a.params['choices'] for a in menu],
                             [{'target_region': 'south ithilien'}, {'target_region': 'north ithilien'}])
            self.assertIsNone(ai.get_ai_policy(state, FP).forced_reason)
            state = ai.apply_action(state, ai.Action(-1, 'card_option', {
                'card_id': CID, 'side': 'fp', 'choices': {'target_region': 'north ithilien'}}))
            self.recruitment_menu(state)
        self.assertEqual(rolls.call_count, 1)

    def test_invalid_target_rejects_before_payment(self):
        state = before_play(targets=('north ithilien',))
        initial = accounting(state)
        with self.assertRaises((ValueError, RuntimeError)):
            ai._execute_action(state, ai.Action(0, 'play_event', {
                'card_id': CID, 'side': 'fp', 'choices': {'target_region': 'morannon'}}))
        self.assertEqual(accounting(state), initial)

    def test_invalid_recruitment_kind_preserves_paid_pending(self):
        state = play(before_play())
        self.recruitment_menu(state)
        initial = accounting(state)
        with self.assertRaises((ValueError, RuntimeError)):
            ai._execute_action(state, ai.Action(-1, 'card_option', {
                'card_id': CID, 'side': 'fp', 'choices': {'_far_recruit': True, 'unit_type': 'dragon'}}))
        self.assertEqual(accounting(state), initial)

    def test_stale_elite_choice_cannot_fall_back_to_regular_or_lose_pending(self):
        state = play(before_play())
        self.recruitment_menu(state)
        exhaust(state, 'elite')
        initial = accounting(state)
        with self.assertRaises((ValueError, RuntimeError)):
            ai._execute_action(state, ai.Action(-1, 'card_option', {
                'card_id': CID, 'side': 'fp', 'choices': {'_far_recruit': True, 'unit_type': 'elite'}}))
        self.assertEqual(accounting(state), initial)

    def test_stage_marker_cannot_skip_a_required_strike(self):
        state = before_play(targets=('north ithilien',))
        initial = accounting(state)
        with self.assertRaises((ValueError, RuntimeError)):
            ai._execute_action(state, ai.Action(0, 'play_event', {
                'card_id': CID, 'side': 'fp', 'choices': {'_far_recruit': True, 'unit_type': 'regular'}}))
        self.assertEqual(accounting(state), initial)


if __name__ == '__main__':
    unittest.main()
