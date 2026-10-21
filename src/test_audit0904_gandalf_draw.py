"""Shared optional Gandalf draw contract: Almanac4/16/46.

Literal card/hand/guide oracles, with no model/checkpoint/corpus writes.
House's two Strategy draws and Guards' no-draw event exercise the same
end-of-action mechanism. Source and queue must survive search and traces.
"""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import ai
import game_logic as gl
import game_state as gs
import game_trace
import state_invariants
from test_audit0904_stewards_choices import (
    before_play, exhaust_unit_pool, army, figures, FP, SH, CID, TITLE,
    S1, S2, BONUS, FILLERS)

ALONE = 'FREE_PEOPLES_Character_I_Will_Go_Alone'
MITHRIL = 'FREE_PEOPLES_Character_Mithril_Coat_and_Sting'


def finish_house(*, full_hand=False, face=gs.ActionDieFace.EVENT):
    state = before_play(face=face, fillers=FILLERS if full_hand else ())
    exhaust_unit_pool(state, 'elite')
    return ai.apply_action(state, ai.Action(0, 'play_event', {
        'card_id': CID, 'side': 'fp'}))


def gandalf_action(state, take):
    actions = ai.get_legal_actions(state, FP)
    action, = [a for a in actions if a.params.get('choices') == {'gandalf_draw': take}]
    return action


def discard(state, cid):
    return ai.apply_action(state, ai.Action(-1, 'discard_card', {
        'card_id': cid, 'side': 'fp'}))


def opening_for(card_id, deck):
    state = before_play(face=gs.ActionDieFace.EVENT)
    state.pending_card_option = {
        'card_id': '__gandalf_draw__', 'card_name': 'Gandalf guide draw',
        'side': 'fp', 'decider': 'fp', 'direct': True, 'gandalf_draw': True,
        'source_card_id': card_id, 'deck': deck, 'die_face': 'EVENT',
        'options': [{'gandalf_draw': True}, {'gandalf_draw': False}]}
    return state


class GandalfGuideDraw(unittest.TestCase):
    def test_queue_clone_and_trace_round_trip_preserve_first_discard_future(self):
        state = finish_house(full_hand=True)
        expected = {'source_card_id': CID, 'deck': 'character', 'die_face': 'EVENT'}
        self.assertEqual(state._gandalf_draw_pending, expected)
        self.assertEqual(ai._pending_decision_kind_side(state), ('discard', FP))
        self.assertEqual(state.hands[FP].hand, FILLERS + [S1, S2])
        cloned = ai.clone_state(state)
        self.assertEqual(cloned._gandalf_draw_pending, expected)
        cloned._gandalf_draw_pending['deck'] = 'strategy'
        self.assertEqual(state._gandalf_draw_pending, expected)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'gandalf.trace.gz')
            recorder = game_trace.TraceRecorder(path)
            recorder.record_initial(state)
            recorder.save()
            loaded = game_trace.load_trace(path)['steps'][0]['state']
        self.assertEqual(loaded._gandalf_draw_pending, expected)
        self.assertIsNotNone(loaded.board)
        loaded = discard(loaded, S1)
        self.assertEqual(loaded.pending_card_option['card_id'], '__gandalf_draw__')
        self.assertEqual(loaded.hands[FP].hand, FILLERS + [S2])

    def test_queue_changes_state_encoding_and_both_search_signatures(self):
        state = finish_house(full_hand=True)
        absent = ai.clone_state(state)
        absent._gandalf_draw_pending = None
        self.assertFalse(np.array_equal(ai.encode_state(state, FP),
                                        ai.encode_state(absent, FP)))
        self.assertIsNotNone(ai._state_sig(state))
        self.assertNotEqual(ai._state_sig(state), ai._state_sig(absent))
        self.assertNotEqual(ai._chance_sig(state), ai._chance_sig(absent))

    def test_queue_remains_encoded_under_another_cards_resumable_option(self):
        state = finish_house(full_hand=True)
        state.pending_card_option = {
            'card_id': MITHRIL, 'card_name': 'Mithril Coat and Sting',
            'side': 'fp', 'decider': 'fp', 'direct': True,
            'resume_action': {'action_type': 'fellowship_move', 'params': {}},
            'options': [{'mithril_redraw': True}, {'mithril_redraw': False}]}
        absent = ai.clone_state(state)
        absent._gandalf_draw_pending = None
        no_resume = ai.clone_state(state)
        no_resume.pending_card_option.pop('resume_action')
        vectors = [ai.encode_state(s, FP) for s in (state, absent, no_resume)]
        self.assertNotEqual(vectors[0].tobytes(), vectors[1].tobytes())
        self.assertNotEqual(vectors[0].tobytes(), vectors[2].tobytes())
        changed, = np.flatnonzero(vectors[0] != vectors[1])
        self.assertGreater(float(vectors[0][changed]), -1.0)
        self.assertLess(float(vectors[0][changed]), 0.0)
        self.assertEqual(float(vectors[1][changed]), 1.0)

    def test_open_context_distinguishes_source_and_deck_without_dimension_change(self):
        char = opening_for(CID, 'character')
        strat = opening_for(S1, 'strategy')
        self.assertEqual(len(ai.encode_state(char, FP)), ai.get_feature_dim())
        self.assertFalse(np.array_equal(ai.encode_state(char, FP), ai.encode_state(strat, FP)))
        self.assertNotEqual(ai._state_sig(char), ai._state_sig(strat))
        self.assertNotEqual(ai._chance_sig(char), ai._chance_sig(strat))

    def test_action_cache_and_actual_scorer_features_keep_source_and_choice(self):
        states = [opening_for(CID, 'character'), opening_for(ALONE, 'character'),
                  opening_for(S1, 'strategy')]
        actions = [gandalf_action(s, True) for s in states]
        keys = [ai._action_cache_key(a, s, FP) for a, s in zip(actions, states)]
        self.assertEqual(len(set(keys)), 3)
        encs = [ai._encode_action_cached(a, s, FP) for a, s in zip(actions, states)]
        encs.append(ai.encode_action(gandalf_action(states[0], False), states[0], FP))
        self.assertEqual(len({e.tobytes() for e in encs}), 4)
        self.assertTrue(all(len(e) == ai.ACTION_ENCODING_DIM for e in encs))
        for e in encs:
            self.assertLess(float(e[23]), 0.0)
            self.assertGreaterEqual(float(e[ai._ACTION_CARD_IDX_DIM]), 0.0)
        self.assertEqual([float(e[24]) for e in encs], [0.5, 0.5, 1.0, 0.5])
        # Exercise the scorer's real preprocessing: negative card indices
        # alone map to padding and would not establish a new decision.
        dummy = SimpleNamespace(card_embedding=lambda idx: idx.unsqueeze(-1).float())
        feats, embedded = ai.WOTRValueNet._embed_action_encs(
            dummy, torch.from_numpy(np.asarray(encs)))
        self.assertFalse(torch.equal(feats[0], feats[3]))
        self.assertFalse(torch.equal(feats[0], feats[1]))
        self.assertFalse(torch.equal(embedded[0], embedded[1]))

    def test_missing_source_or_wrong_deck_is_not_silently_encoded_as_unknown(self):
        for source, deck in [(None, 'character'), (CID, 'strategy')]:
            with self.subTest(source=source, deck=deck):
                state = before_play()
                action = ai.Action(-1, 'card_option', {
                    'card_id': '__gandalf_draw__', 'source_card_id': source,
                    'deck': deck, 'choices': {'gandalf_draw': True}})
                with self.assertRaises(ValueError):
                    ai.encode_action(action, state, FP)

    def test_shared_strategy_event_offers_same_deck_optional_draw(self):
        state = before_play(face=gs.ActionDieFace.EVENT)
        state.fp_char_deck.append(CID)
        state.hands[FP].hand = [S1]
        state.fp_strat_deck.remove(S1)
        initial_char = list(state.fp_char_deck)
        initial_strategy = list(state.fp_strat_deck)
        state = ai.apply_action(state, ai.Action(0, 'play_event', {
            'card_id': S1, 'side': 'fp'}))
        regular, = [a for a in ai.get_legal_actions(state, FP)
                    if a.params['choices'].get('unit_type') == 'regular']
        state = ai.apply_action(state, regular)
        self.assertEqual(figures(state), (4, 1, 2))
        self.assertEqual(state.hands[FP].hand, [])
        self.assertEqual(state.pending_card_option['deck'], 'strategy')
        state = ai.apply_action(state, gandalf_action(state, True))
        self.assertEqual(state.hands[FP].hand, [S2])
        self.assertEqual(state.fp_strat_deck, initial_strategy[:-1])
        self.assertEqual(state.fp_char_deck, initial_char)

    def test_guide_eligibility_is_checked_after_separation_not_at_commitment(self):
        for separates, starts_guide, expected_guide, offered in [
                ('Gandalf the Grey', 'Gandalf the Grey', 'Strider', False),
                ('Strider', 'Strider', 'Gandalf the Grey', True)]:
            with self.subTest(separates=separates):
                state = before_play(region=None, face=gs.ActionDieFace.EVENT)
                state.fp_char_deck.append(CID)
                state.fp_char_deck.remove(ALONE)
                state.hands[FP].hand = [ALONE]
                state.fellowship.guide = next(c for c in state.fellowship.companions
                                             if c.name == starts_guide)
                state = ai.apply_action(state, ai.Action(0, 'play_event', {
                    'card_id': ALONE, 'side': 'fp'}))
                self.assertEqual(state.fellowship.guide.name, starts_guide)
                choice, = [a for a in ai.get_legal_actions(state, FP)
                           if a.params['choices'] == {
                               'companions': [separates], 'destination': 'rivendell'}]
                state = ai.apply_action(state, choice)
                self.assertEqual(state.fellowship.guide.name, expected_guide)
                self.assertEqual(state.hands[FP].hand, [])
                self.assertEqual(bool((getattr(state, 'pending_card_option', None)
                                       or {}).get('gandalf_draw')), offered)
                self.assertIsNone(state._gandalf_draw_pending)

    def test_empty_same_deck_suppresses_unfulfillable_bonus_only(self):
        state = before_play(face=gs.ActionDieFace.EVENT)
        exhaust_unit_pool(state, 'elite')
        state.discard_face_up['fp_character'].extend(state.fp_char_deck)
        state.fp_char_deck = []
        state = ai.apply_action(state, ai.Action(0, 'play_event', {
            'card_id': CID, 'side': 'fp'}))
        self.assertEqual(figures(state), (4, 1, 1))
        self.assertEqual(state.hands[FP].hand, [S1, S2])
        self.assertIsNone(getattr(state, 'pending_card_option', None))
        self.assertIsNone(state._gandalf_draw_pending)

    def test_recruit_at_siege_cap_resolves_overstack_before_optional_bonus(self):
        state = before_play(face=gs.ActionDieFace.EVENT)
        # A later legal siege: Gondor has consolidated its Pelargir Regular
        # in Minas Tirith (4R1E, exactly the five-unit garrison limit).
        army(state, 'minas tirith').units.append(army(state, 'pelargir').units.pop())
        source = army(state, 'dol guldur')
        for unit in [u for u in source.units if u.kind == gs.UnitKind.REGULAR][:2]:
            source.units.remove(unit)
            army(state, 'minas tirith').units.append(unit)
        for nation in (gs.Nation.GONDOR, gs.Nation.SAURON):
            state.political.active[nation] = True
            state.political.boxes[nation] = 3
        state.besieged_strongholds.add('minas tirith')
        state = ai.apply_action(state, ai.Action(0, 'play_event', {
            'card_id': CID, 'side': 'fp'}))
        menu = ai.get_legal_actions(state, FP)
        self.assertEqual({a.params['choices']['unit_type'] for a in menu},
                         {'regular', 'elite'})
        regular, = [a for a in menu if a.params['choices']['unit_type'] == 'regular']
        state = ai.apply_action(state, regular)
        self.assertEqual(figures(state), (5, 1, 1))
        self.assertEqual(state.hands[FP].hand, [S1, S2])
        self.assertEqual(state.pending_card_option['card_id'], '__overstack__')
        self.assertEqual(state._gandalf_draw_pending['source_card_id'], CID)
        remove_elite, = [a for a in ai.get_legal_actions(state, FP)
                         if a.params['choices']['overstack_remove']['kind'] == 'elite']
        selected = remove_elite.params['choices']['overstack_remove']
        self.assertEqual(selected['region'].casefold(), 'minas tirith')
        self.assertEqual(selected['nation'], 'Gondor')
        state = ai.apply_action(state, remove_elite)
        self.assertEqual(figures(state), (5, 0, 1))
        self.assertEqual(state.pending_card_option['card_id'], '__gandalf_draw__')
        state = ai.apply_action(state, gandalf_action(state, False))
        self.assertEqual(state.hands[FP].hand, [S1, S2])
        self.assertEqual(state.discard_face_up['fp_character'].count(CID), 1)
        self.assertEqual([d.used for d in state.dice[FP]], [True, False])

    def test_guide_tie_is_resolved_before_end_of_action_eligibility(self):
        state = before_play(region=None, face=gs.ActionDieFace.EVENT)
        state.fp_char_deck.append(CID)
        state.fp_char_deck.remove(ALONE)
        state.hands[FP].hand = [ALONE]
        state = ai.apply_action(state, ai.Action(0, 'play_event', {
            'card_id': ALONE, 'side': 'fp'}))
        choice, = [a for a in ai.get_legal_actions(state, FP)
                   if a.params['choices'] == {'companions': ['Gandalf the Grey', 'Strider'],
                                               'destination': 'rivendell'}]
        state = ai.apply_action(state, choice)
        self.assertEqual(ai._pending_decision_kind_side(state), ('guide_change', FP))
        self.assertEqual(state._gandalf_draw_pending['source_card_id'], ALONE)
        menu = ai.get_legal_actions(state, FP)
        self.assertEqual({a.params['companion'] for a in menu},
                         {'Boromir', 'Legolas', 'Gimli'})
        boromir, = [a for a in menu if a.params['companion'] == 'Boromir']
        state = ai.apply_action(state, boromir)
        self.assertEqual(state.fellowship.guide.name, 'Boromir')
        self.assertIsNone(state._gandalf_draw_pending)
        self.assertIsNone(getattr(state, 'pending_card_option', None))
        self.assertEqual(state.hands[FP].hand, [])

    def test_character_funding_does_not_trigger_but_will_as_event_does(self):
        plain = finish_house(face=gs.ActionDieFace.CHARACTER)
        will = finish_house(face=gs.ActionDieFace.WILL_OF_WEST)
        self.assertIsNone(getattr(plain, 'pending_card_option', None))
        self.assertEqual(will.pending_card_option['card_id'], '__gandalf_draw__')
        self.assertEqual(plain.hands[FP].hand, [S1, S2])
        self.assertEqual(will.hands[FP].hand, [S1, S2])

    def test_last_die_still_routes_first_discard_and_optional_choice(self):
        state = finish_house(full_hand=True)
        for pool in state.dice.values():
            for die in pool:
                die.used = True
        self.assertEqual(ai._next_acting_side(state, FP), FP)
        state = discard(state, S1)
        self.assertEqual(ai._next_acting_side(state, FP), FP)
        state = ai.apply_action(state, gandalf_action(state, False))
        self.assertIsNone(ai._next_acting_side(state, FP))
        self.assertEqual(state.hands[FP].hand, FILLERS + [S2])

    def test_deferred_physical_draw_keeps_pending_then_grants_once(self):
        class NeedPhysicalCard(BaseException):
            pass
        state = finish_house()
        action = gandalf_action(state, True)
        pending = state.pending_card_option
        deck = list(state.fp_char_deck)
        with patch.object(gl, 'draw_event_card', side_effect=NeedPhysicalCard):
            with self.assertRaises(NeedPhysicalCard):
                ai._execute_action(state, action)
        self.assertIs(state.pending_card_option, pending)
        self.assertEqual(state.fp_char_deck, deck)
        self.assertEqual(state.hands[FP].hand, [S1, S2])
        ai._execute_action(state, action)
        self.assertEqual(state.hands[FP].hand, [S1, S2, BONUS])
        self.assertEqual(state.fp_char_deck, deck[:-1])
        self.assertEqual(state._owed_draws.get((FP.name, 'character'), 0), 0)
        self.assertIsNone(state.pending_card_option)
        with self.assertRaises(ValueError):
            ai._execute_action(state, action)
        self.assertEqual(state.hands[FP].hand, [S1, S2, BONUS])
        self.assertEqual([d.used for d in state.dice[FP]], [True, False])

    def test_invalid_optional_choice_preserves_pending_and_paid_resources(self):
        for choices in ({}, {'gandalf_draw': 1}, {'gandalf_draw': None},
                        {'gandalf_draw': False, 'deck': 'strategy'}):
            with self.subTest(choices=choices):
                state = finish_house()
                pending = state.pending_card_option
                deck = list(state.fp_char_deck)
                with self.assertRaises(ValueError):
                    ai._execute_action(state, ai.Action(-1, 'card_option', {
                        'card_id': '__gandalf_draw__', 'side': 'fp', 'choices': choices}))
                self.assertIs(state.pending_card_option, pending)
                self.assertEqual(state.fp_char_deck, deck)
                self.assertEqual(state.hands[FP].hand, [S1, S2])
                self.assertEqual([d.used for d in state.dice[FP]], [True, False])

    def test_normal_turn_reset_refuses_unfinished_draw_before_any_mutation(self):
        for full_hand in (False, True):
            with self.subTest(full_hand=full_hand):
                state = finish_house(full_hand=full_hand)
                turn = state.turn
                dice = [[(d.current, d.used) for d in state.dice[s]] for s in (FP, SH)]
                hand = list(state.hands[FP].hand)
                with self.assertRaisesRegex(RuntimeError, 'Gandalf'):
                    gl.begin_turn_reset(state, turn + 1)
                self.assertEqual(state.turn, turn)
                self.assertEqual(state.hands[FP].hand, hand)
                self.assertEqual([[(d.current, d.used) for d in state.dice[s]]
                                  for s in (FP, SH)], dice)

    def test_abnormal_selfplay_rollover_refuses_pending_draw(self):
        state = finish_house(full_hand=True)
        before = ai.clone_state(state)
        with self.assertRaisesRegex(RuntimeError, '_gandalf_draw_pending'):
            ai._safe_next_turn(state)
        self.assertEqual(state.turn, before.turn)
        self.assertEqual(state._gandalf_draw_pending, before._gandalf_draw_pending)
        self.assertEqual(state.hands[FP].hand, before.hands[FP].hand)



if __name__ == '__main__':
    unittest.main(verbosity=2)
