"""Independent public-card casualty choices, supply, timing and representations."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from collections import Counter
import unittest
from unittest.mock import patch
import numpy as np
import ai, board, event_effects as ee, game_logic as gl, game_state as gs
FP, SH = gs.Side.FREE_PEOPLES, gs.Side.SHADOW
S, I, G = gs.Nation.SAURON, gs.Nation.ISENGARD, gs.Nation.GONDOR
R, E = gs.UnitKind.REGULAR, gs.UnitKind.ELITE


def played(card_name, faces, *, all_regular=False, mutate=None):
    state = gs.setup_initial_game(board.load_board())
    gl.initialize_decks(state)
    state.armies = {}
    if card_name == 'The Spirit of Mordor':
        owner = FP
        armies = [('orthanc', [(S, R)] * (2 if all_regular else 5)
                   + ([(I, R)] * 2 if all_regular else [(I, E)]))]
    else:
        owner = SH
        armies = [('orthanc', [(I, R)]), ('fords of isen', [(G, R)] * 5 + [(G, E)])]
    for region, figures in armies:
        army = gs.Army(state.board.region(region))
        army.units = [gs.Unit(nation, kind) for nation, kind in figures]
        army.leaders = []
        state.armies[region] = army
    if owner == SH:
        state.armies['orthanc'].leaders = [gs.Nazgul(nation=S), gs.Nazgul(nation=S)]
    card = ee._card_by_name(card_name)
    state.hands[owner].hand = [card.card_id]
    for side in (FP, SH):
        for die in state.dice[side]:
            die.used = True
    state.dice[owner][0].current = gs.ActionDieFace.EVENT
    state.dice[owner][0].used = False
    if mutate is not None:
        mutate(state)
    # Patch only the card dice. The play is selected from the real legal menu,
    # card payment, discard, requirements, effect and owner routing are live.
    with patch.object(ee, '_d6_batch', return_value=faces) as dice:
        actions = [a for a in ai.get_legal_actions(state, owner)
                   if a.action_type == 'play_event' and a.params['card_id'] == card.card_id]
        if len(actions) != 1:
            raise AssertionError(('fixture needs exactly one public card play', actions))
        after = ai.apply_action(state, actions[0])
        if dice.call_args_list != [unittest.mock.call(len(faces))]:
            raise AssertionError(('unexpected random requests', dice.call_args_list))
    if not after.dice[owner][0].used or card.card_id in after.hands[owner].hand:
        raise AssertionError('public play did not spend the die and card')
    return after


def outcomes(state, owner, region, depth=0):
    pending = state.pending_card_option or {}
    if not pending.get('card_casualty') or pending.get('cas_side') != owner:
        army = ai._armies_lookup(state, region)
        counts = Counter((u.nation, u.kind) for u in army.units) if army else Counter()
        return [((), state, counts)]
    if depth >= 12:
        raise AssertionError('Casualty allocation failed to terminate')
    result = []
    actions = ai.get_legal_actions(state, owner)
    if not actions:
        raise AssertionError('Casualty owner has no legal operation')
    for action in actions:
        successor = ai.apply_action(state, action)
        next_pending = successor.pending_card_option or {}
        if next_pending.get('card_casualty'):
            if next_pending['cas_hits'] >= pending['cas_hits']:
                raise AssertionError('Casualty operation did not reduce the budget')
        for path, leaf, counts in outcomes(successor, owner, region, depth + 1):
            result.append(((action,) + path, leaf, counts))
    return result


class CardCasualtyReview(unittest.TestCase):
    def test_different_casualty_choices_have_different_encodings(self):
        state = played('The Spirit of Mordor', [6, 5, 1, 2, 3])
        choices = ai.get_legal_actions(state, SH)
        self.assertGreaterEqual(len(choices), 2)
        encodings = [ai.encode_action(a, state, SH).tobytes() for a in choices]
        self.assertEqual(len(encodings), len(set(encodings)))

    def test_actual_one_and_two_hit_plays_have_different_observations(self):
        one = played('The Spirit of Mordor', [6, 1, 1, 2, 3])
        two = played('The Spirit of Mordor', [6, 5, 1, 2, 3])
        self.assertEqual(one.pending_card_option['cas_hits'], 1)
        self.assertEqual(two.pending_card_option['cas_hits'], 2)
        self.assertFalse(np.array_equal(ai.encode_state(one, SH), ai.encode_state(two, SH)))

    def test_shadow_can_reduce_isengard_elite_and_lose_sauron_regular(self):
        state = played('The Spirit of Mordor', [6, 5, 1, 2, 3])
        expected = Counter({(S, R): 4, (I, R): 1})
        actual = [counts for action, after, counts in outcomes(state, SH, 'orthanc')]
        self.assertIn(expected, actual)

    def test_fp_can_remove_whole_elite_without_spending_regular_reserve(self):
        state = played('Dreadful Spells', [5, 6])
        expected = Counter({(G, R): 5})
        actual = [(counts, (after.fp_casualties.get(G) or {}).get('regular', 0),
                   (after.fp_casualties.get(G) or {}).get('elite', 0))
                  for action, after, counts in outcomes(state, FP, 'fords of isen')]
        self.assertIn((expected, 0, 1), actual)

    def test_all_regular_mixed_nation_owner_keeps_nationality_choice(self):
        state = played('The Spirit of Mordor', [6, 1, 1, 2, 3], all_regular=True)
        self.assertIsNotNone(state.pending_card_option)
        self.assertTrue(state.pending_card_option.get('card_casualty'))
        self.assertEqual(state.pending_card_option['decider'], 'sh')

    def test_literal_existing_two_regular_removal_control(self):
        state = played('Dreadful Spells', [5, 6])
        expected = Counter({(G, R): 3, (G, E): 1})
        actual = [(counts, (after.fp_casualties.get(G) or {}).get('regular', 0))
                  for action, after, counts in outcomes(state, FP, 'fords of isen')]
        self.assertIn((expected, 2), actual)


    def test_dreadful_wipe_spares_companion_but_removes_plain_leader(self):
        def setup(state):
            army = state.armies['fords of isen']
            army.units = [gs.Unit(G, R)]
            companion = next(c for c in state.fellowship.companions if c.name == 'Legolas')
            state.fellowship.companions.remove(companion)
            army.leaders = [companion, gs.Leader(G)]
        state = played('Dreadful Spells', [5, 6], mutate=setup)
        army = ai._armies_lookup(state, 'fords of isen')
        self.assertIsNotNone(army)
        self.assertEqual(army.units, [])
        self.assertEqual([getattr(l, 'name', None) for l in army.leaders], ['Legolas'])
        self.assertEqual(state.fp_casualties[G]['regular'], 1)
        self.assertEqual(state.fp_casualties[G]['leader'], 1)
        self.assertFalse((state.pending_card_option or {}).get('card_casualty'))

    def test_spirit_wipe_spares_nazgul(self):
        def setup(state):
            state.armies['orthanc'].units = [gs.Unit(S, R), gs.Unit(I, R)]
            state.armies['orthanc'].leaders = [gs.Nazgul(S)]
        state = played('The Spirit of Mordor', [6, 6, 6, 6, 6], mutate=setup)
        army = ai._armies_lookup(state, 'orthanc')
        self.assertIsNotNone(army)
        self.assertEqual(army.units, [])
        self.assertEqual(len(army.leaders), 1)
        self.assertIsInstance(army.leaders[0], gs.Nazgul)

    def test_invalid_opposite_nation_choice_refuses_without_mutation(self):
        from copy import deepcopy
        state = played('The Spirit of Mordor', [6, 5, 1, 2, 3])
        before = (deepcopy(state.pending_card_option),
                  Counter((u.nation, u.kind) for u in state.armies['orthanc'].units))
        original = ai.get_legal_actions(state, SH)[0]
        params = dict(original.params)
        params['choices'] = dict(params['choices'], nation=G)
        invalid = ai.Action(die_idx=-1, action_type='card_option', params=params)
        with self.assertRaises(ValueError):
            ai.apply_action(state, invalid)
        self.assertEqual((state.pending_card_option,
                         Counter((u.nation, u.kind) for u in state.armies['orthanc'].units)), before)

    def test_real_operation_witness_has_card_context_and_exact_budget(self):
        state = played('The Spirit of Mordor', [6, 5, 1, 2, 3])
        action = next(a for a in ai.get_legal_actions(state, SH)
                      if a.params['choices']['casualty_operation'] == 'remove_regular')
        after = ai.apply_action(state, action)
        self.assertEqual(after.pending_card_option['cas_hits'], 1)
        self.assertEqual(len(after._casualty_operations), 1)
        witness = after._casualty_operations[0]
        self.assertEqual((witness['stage'], witness['region'], witness['side'],
                          witness['nation'], witness['operation'], witness['cost']),
                         ('card', 'orthanc', SH, S, 'remove_regular', 1))
        self.assertEqual(witness['remaining_before'], 2)

    def test_faramir_resumes_recruitment_once_after_last_casualty(self):
        import test_audit0905_faramir as fixture
        state = fixture.before_play(targets=('north ithilien',), target_elite=True)
        before = fixture.counts(state)
        with patch.object(ee, '_d6_batch', return_value=[6, 5, 1]) as dice:
            state = fixture.play(state)
            for _ in range(4):
                if (state.pending_card_option or {}).get('card_casualty'):
                    break
                menu = ai.get_legal_actions(state, FP)
                target = next(a for a in menu if a.action_type == 'card_option'
                              and a.params['choices'].get('target_region') == 'north ithilien')
                state = ai.apply_action(state, target)
            self.assertTrue((state.pending_card_option or {}).get('card_casualty'))
            self.assertEqual(fixture.counts(state), before)
            # Whole-Elite removal spends both hits; final choice resumes the
            # distinct human recruitment decision without another card roll.
            casualty = next(a for a in ai.get_legal_actions(state, SH)
                            if a.params['choices']['casualty_operation'] == 'remove_elite')
            state = ai.apply_action(state, casualty)
            self.assertFalse((state.pending_card_option or {}).get('card_casualty'))
            self.assertEqual(fixture.counts(state), before)
            recruit = next(a for a in ai.get_legal_actions(state, FP)
                           if a.action_type == 'card_option'
                           and a.params['choices'].get('unit_type') == 'elite')
            state = ai.apply_action(state, recruit)
            self.assertEqual(fixture.counts(state), (before[0], before[1] + 1, before[2] + 1))
            self.assertEqual(dice.call_args_list, [unittest.mock.call(3)])


if __name__ == '__main__':
    unittest.main(verbosity=2)
