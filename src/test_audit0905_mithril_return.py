"""Literal Almanac p.11 outcomes through publicly enumerated player actions."""
from collections import Counter
from pathlib import Path
import os
import pickle
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get('WOTR_TIMING_SOURCE', str(HERE))).resolve()
sys.path.insert(0, str(SOURCE))
os.environ.update(WOTR_ACT_GEN=os.environ.get('WOTR_TIMING_GENERATION', 'v46'), WOTR_NATIVE='0', CUDA_VISIBLE_DEVICES='-1')
import ai
import board
import game_logic as gl
import game_state as gs
import numpy as np
assert Path(ai.__file__).resolve().parent == SOURCE
FP, SH = gs.Side.FREE_PEOPLES, gs.Side.SHADOW
CID = 'FREE_PEOPLES_Character_Mithril_Coat_and_Sting'
ORDINARY = Counter({'3': 3, '2': 2, '1': 2, 'Eye': 4, '2R': 1, '1R': 2, '0R': 2})

def choose(state, kind, side=FP, die_idx=None, **params):
    actions = [a for a in ai.get_legal_actions(state, side)
               if a.action_type == kind and (die_idx is None or a.die_idx == die_idx)
               and all(a.params.get(k) == v for k, v in params.items())]
    if len(actions) != 1:
        raise AssertionError((kind, params, [(a.action_type, a.params) for a in actions]))
    return ai.apply_action(state, actions[0])

def setup(replacement='1R', original='3', one_left=False):
    state = gs.setup_initial_game(board.load_board())
    gl.initialize_decks(state)
    state.fellowship.guide = next(c for c in state.fellowship.companions if c.name == 'Strider')
    state.fellowship.region = state.board.region('dimrill dale')
    state.fellowship.progress = 0
    for name in ('fp_char_deck', 'fp_strat_deck', 'sh_char_deck', 'sh_strat_deck'):
        deck = getattr(state, name)
        if CID in deck:
            deck.remove(CID)
    state.hands[FP].hand = [CID]
    def die(face, side):
        d = gs.ActionDie([face], side)
        d.current = face
        return d
    state.dice = {FP: [die(gs.ActionDieFace.EVENT, FP), die(gs.ActionDieFace.CHARACTER, FP)],
                  SH: [die(gs.ActionDieFace.EVENT, SH), die(gs.ActionDieFace.EYE, SH)]}
    state.hunt_box = [state.dice[SH][1]]
    state.dice[SH][1].used = True
    bag = [gs.HuntTile(3), gs.HuntTile(3), gs.HuntTile(3),
           gs.HuntTile(2), gs.HuntTile(2), gs.HuntTile(1), gs.HuntTile(1),
           gs.HuntTile(is_eye=True), gs.HuntTile(is_eye=True),
           gs.HuntTile(is_eye=True), gs.HuntTile(is_eye=True),
           gs.HuntTile(2, reveal=True), gs.HuntTile(1, reveal=True),
           gs.HuntTile(1, reveal=True), gs.HuntTile(0, reveal=True), gs.HuntTile(0, reveal=True)]
    first = next(t for t in bag if repr(t) == original)
    bag.remove(first)
    second = next(t for t in bag if repr(t) == replacement)
    bag.remove(second)
    state.hunt_pool = [first] if one_left else [second, first]
    state.hunt_discard = bag + ([second] if one_left else [])
    return choose(state, 'play_event', die_idx=0, card_id=CID)

def move(state):
    with patch.object(gl, 'roll_d6_batch', side_effect=lambda n, *args: [6] * n):
        return choose(state, 'fellowship_move')

def redraw(state, yes=True):
    with patch.object(gl.random, 'shuffle', side_effect=lambda values: None):
        return choose(state, 'card_option', choices={'mithril_redraw': yes})

def place(state, region):
    # Deterministic chance only: choose an actually eligible physical tile.
    with patch.object(gl.random, 'choice', side_effect=lambda values: values[-1]), \
            patch.object(gl.random, 'shuffle', side_effect=lambda values: None):
        return choose(state, 'fellowship_reveal_place', region=region)

def inventory(state):
    held = getattr(state, '_mithril_stash', None)
    return Counter(map(repr, state.hunt_pool + state.hunt_discard + ([held['tile']] if held else [])))

def add_protection_through_public_play(state, title):
    from cards import get_cards
    cid = next(cid for cid, card in get_cards().items() if card.card_name == title)
    for name in ('fp_char_deck', 'fp_strat_deck'):
        deck = getattr(state, name)
        if cid in deck:
            deck.remove(cid)
    state.hands[FP].hand.append(cid)
    d = gs.ActionDie([gs.ActionDieFace.EVENT], FP)
    d.current = gs.ActionDieFace.EVENT
    state.dice[FP].append(d)
    return choose(state, 'play_event', die_idx=2, card_id=cid), cid

class MithrilTimingTests(unittest.TestCase):
    def test_protection_choice_applies_to_replacement_before_return(self):
        for title in ('Axe and Bow', 'Horn of Gondor'):
            for use in (True, False):
                with self.subTest(card=title, use=use):
                    state, cid = add_protection_through_public_play(setup(), title)
                    state = redraw(move(state))
                    self.assertEqual(state.hunt_pool, [])
                    self.assertTrue(state.pending_card_option['hunt_reduction'])
                    option = {'use_hunt_reduction': 1, 'card_id': cid} if use else {'use_hunt_reduction': 0}
                    state = choose(state, 'card_option', choices=option)
                    self.assertEqual(state.hunt_pool, [])
                    if not use:
                        state = choose(state, 'hunt_take_corruption')
                    state = place(state, 'dimrill dale')
                    self.assertEqual(state.fellowship.corruption, 0 if use else 1)
                    self.assertEqual(list(map(repr, state.hunt_pool)), ['3'])
                    self.assertEqual(inventory(state), ORDINARY)

    def test_random_hobbit_placement_finishes_before_original_returns(self):
        state = redraw(move(setup(replacement='1')))
        with patch.object(gl, 'pick_random_companion',
                          side_effect=lambda candidates, *args: next(c for c in candidates if c.name == 'Meriadoc')):
            state = choose(state, 'hunt_sacrifice_random')
        self.assertEqual(state.hunt_pool, [])
        self.assertEqual(state.pending_hunt['phase'], 'hobbit_place')
        if state.pending_guide_change is not None:
            state = choose(state, 'choose_guide', companion='Strider', side=FP)
        action = next(a for a in ai.get_legal_actions(state, FP)
                      if a.action_type == 'companion_separate' and a.params.get('to_region') == 'dimrill dale')
        state = ai.apply_action(state, action)
        self.assertEqual(list(map(repr, state.hunt_pool)), ['3'])
        self.assertEqual(state.fellowship.corruption, 0)
        self.assertEqual(inventory(state), ORDINARY)

    def test_completed_redraw_cannot_be_used_twice(self):
        import event_effects as effects
        state = redraw(move(setup()))
        before = pickle.dumps(state)
        with self.assertRaisesRegex(ValueError, 'already resolved'):
            effects._effect_mithril_coat(state, None, {'mithril_redraw': True})
        self.assertEqual(pickle.dumps(state), before)
        state = place(choose(state, 'hunt_take_corruption'), 'dimrill dale')
        before = pickle.dumps(state)
        self.assertFalse(gl.return_mithril_tile_if_settled(state))
        self.assertEqual(pickle.dumps(state), before)

    def test_reveal_in_moria_recycles_used_tiles_without_held_original(self):
        state = redraw(move(setup()))
        self.assertEqual(state.hunt_pool, [])
        self.assertEqual(len(state.hunt_discard), 15)
        self.assertEqual(inventory(state), ORDINARY)
        self.assertEqual(state.pending_hunt['damage'], 1)
        state = choose(state, 'hunt_take_corruption')
        self.assertEqual(state.hunt_pool, [])
        state = place(state, 'moria')
        self.assertEqual(state.pending_hunt['damage'], 1)
        self.assertEqual(len(state.hunt_pool), 14)
        self.assertEqual([repr(t) for t in state.hunt_discard], ['1R'])
        self.assertEqual(inventory(state), ORDINARY)
        final = choose(state, 'hunt_take_corruption')
        self.assertEqual(final.fellowship.corruption, 2)
        self.assertEqual(len(final.hunt_pool), 15)
        self.assertIsNone(final._mithril_stash)
        self.assertEqual(inventory(final), ORDINARY)

    def test_no_extra_draw_returns_only_original_without_refilling(self):
        state = choose(redraw(move(setup())), 'hunt_take_corruption')
        final = place(state, 'dimrill dale')
        self.assertEqual(list(map(repr, final.hunt_pool)), ['3'])
        self.assertEqual(len(final.hunt_discard), 15)
        self.assertEqual(final.fellowship.corruption, 1)
        self.assertIsNone(final._mithril_stash)
        self.assertEqual(inventory(final), ORDINARY)

    def test_decline_keeps_card_and_applies_original_once(self):
        state = redraw(move(setup()), yes=False)
        self.assertIsNone(state._mithril_stash)
        self.assertEqual(state.pending_hunt['damage'], 3)
        self.assertTrue(state.persistent_cards.has_card_type(gs.PersistentCardType.MITHRIL_COAT))
        final = choose(state, 'hunt_take_corruption')
        self.assertEqual(final.fellowship.corruption, 3)
        self.assertEqual(list(map(repr, final.hunt_pool)), ['1R'])
        self.assertEqual(inventory(final), ORDINARY)

    def test_zero_damage_replacement_still_waits_for_reveal_placement(self):
        state = redraw(move(setup(replacement='0R')))
        self.assertIsNone(state.pending_hunt)
        self.assertTrue(state._reveal_place_due)
        self.assertEqual(state.hunt_pool, [])
        final = place(state, 'dimrill dale')
        self.assertEqual(list(map(repr, final.hunt_pool)), ['3'])
        self.assertEqual(final.fellowship.corruption, 0)
        self.assertEqual(inventory(final), ORDINARY)

    def test_replacement_draw_can_recycle_an_already_empty_pool(self):
        state = redraw(move(setup(one_left=True)))
        self.assertEqual(len(state.hunt_pool), 14)
        self.assertEqual(list(map(repr, state.hunt_discard)), ['1R'])
        self.assertEqual(state.pending_hunt['damage'], 1)
        self.assertEqual(inventory(state), ORDINARY)
        final = place(choose(state, 'hunt_take_corruption'), 'dimrill dale')
        self.assertEqual(len(final.hunt_pool), 15)
        self.assertEqual(inventory(final), ORDINARY)

    def test_clones_and_saved_states_keep_held_tile_and_do_not_mutate_parent(self):
        state = redraw(move(setup()))
        before = pickle.dumps(state)
        for copy in (ai.clone_state(state), pickle.loads(before)):
            self.assertEqual(repr(copy._mithril_stash['tile']), '3')
            final = place(choose(copy, 'hunt_take_corruption'), 'dimrill dale')
            self.assertIsNone(final._mithril_stash)
            self.assertEqual(inventory(final), ORDINARY)
        self.assertEqual(pickle.dumps(state), before)

    def test_held_tile_remains_visible_in_existing_observation_slots(self):
        states = [redraw(move(setup(original=face))) for face in ('3', '2')]
        for side in (FP, SH):
            arrays = [ai.encode_state(s, side) for s in states]
            expected_width = 4481 if os.environ.get('WOTR_ACT_GEN') == 'v52' else 4190
            self.assertEqual(arrays[0].shape, (expected_width,))
            self.assertFalse(np.array_equal(arrays[0], arrays[1]))
            # Removing only the known held-tile field must change the encoding.
            altered = ai.clone_state(states[0])
            altered._mithril_stash = None
            self.assertFalse(np.array_equal(arrays[0], ai.encode_state(altered, side)))

    def test_orc_patrol_and_isildur_redraw_do_not_refill_without_a_new_draw(self):
        for title, slug in [('Orc Patrol', 'Orc_Patrol'), ("Isildur's Bane", 'Isildurs_Bane')]:
            with self.subTest(card=title):
                state = setup(replacement='1')
                from cards import get_cards
                cid = next(cid for cid, card in get_cards().items() if card.card_name == title)
                state.hands[SH].hand = [cid]
                for name in ('sh_char_deck', 'sh_strat_deck'):
                    deck = getattr(state, name)
                    if cid in deck:
                        deck.remove(cid)
                with patch.object(gl.random, 'choice', side_effect=lambda values: values[-1]):
                    state = choose(state, 'play_event', side=SH, card_id=cid)
                state = redraw(state)
                self.assertEqual(len(state.hunt_discard), 15)
                if state.pending_hunt is not None:
                    self.assertEqual(state.hunt_pool, [])
                    state = choose(state, 'hunt_take_corruption')
                self.assertEqual(list(map(repr, state.hunt_pool)), ['3'])
                self.assertEqual(inventory(state), ORDINARY)

if __name__ == '__main__':
    unittest.main(verbosity=2)
