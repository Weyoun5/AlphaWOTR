"""Book of Mazarbul: literal captured no-effect play and movement controls.

RB22 and Almanac18/46: no printed play condition; previously separated
companions may move, none need move, and the Dwarven trigger is independent.
No requirement permits using this card to separate Fellowship companions.
Expected outcomes come from those clauses, not emitter/engine roundtrips.
"""
import copy
import dataclasses
import hashlib
import os
from pathlib import Path
import unittest

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('WOTR_NATIVE', '0')
os.environ.setdefault('WOTR_ACT_GEN', 'v41')
import torch
torch.set_num_threads(1)
import ai
import board
import game_logic as gl
import game_state as gs
from card_options import get_card_options
from event_effects import check_event_requirement
from game_trace import load_trace
import state_invariants

ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / 'audit/2026-09-04/performance_pass/served_gpu_selector_20260905/batch/traces/w1_g0_sh_military.trace.gz'
TRACE_SHA = '7a7ee5fe0ae48d1686d83d5a2f101d3a751ddfab0397b14bd01daf7a0d42a6ee'
CID, TITLE = 'FREE_PEOPLES_Strategy_Book_of_Mazarbul', 'Book of Mazarbul'
FP, SH = gs.Side.FREE_PEOPLES, gs.Side.SHADOW


def world(state):
    armies = {board.sanitize(k): {
        'units': sorted((u.nation.name, u.kind.name) for u in a.units),
        'leaders': sorted((type(c).__name__, getattr(c, 'name', ''),
                           getattr(getattr(c, 'nation', None), 'name', '')) for c in a.leaders)}
        for k, a in state.armies.items()}
    return {'armies': armies, 'fellowship': dataclasses.asdict(state.fellowship),
            'political': dataclasses.asdict(state.political),
            'control': copy.deepcopy(state.enemy_controlled),
            'rings': (state.fp_elven_rings, state.sh_elven_rings)}


class MazarbulCapturedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if hashlib.sha256(TRACE.read_bytes()).hexdigest() != TRACE_SHA:
            raise AssertionError('Original captured trace changed')
        cls.row = load_trace(str(TRACE))['steps'][101]
        assert cls.row['action_type'] == 'play_event'
        assert cls.row['params'] == {'card_id': CID, 'side': 'fp'}

    def before(self):
        return ai.clone_state(self.row['decision_context']['state'])

    def play(self, state):
        return ai.apply_action(state, ai.Action(0, 'play_event', {'card_id': CID, 'side': 'fp'}))

    def assert_spent_once(self, before, after):
        hands = {s: list(h.hand) for s, h in before.hands.items()}
        hands[FP].remove(CID)
        self.assertEqual({s: list(h.hand) for s, h in after.hands.items()}, hands)
        discard = copy.deepcopy(before.discard_face_up)
        discard.setdefault('fp_strategy', []).append(CID)
        self.assertEqual(after.discard_face_up, discard)
        expected_dice = {s: [(d.current, d.used) for d in dice] for s, dice in before.dice.items()}
        expected_dice[FP][0] = (gs.ActionDieFace.MUSTER_ARMY, True)
        self.assertEqual({s: [(d.current, d.used) for d in dice] for s, dice in after.dice.items()}, expected_dice)
        self.assertIsNone(after.pending_card_option)

    def test_recorded_prior_has_no_separated_companion_or_hidden_war_trigger(self):
        state = self.before()
        self.assertEqual(state.turn, 3)
        self.assertEqual([c.name for c in state.fellowship.companions],
                         ['Strider', 'Legolas', 'Gimli', 'Boromir', 'Meriadoc', 'Peregrin'])
        self.assertEqual(state.fellowship.guide.name, 'Strider')
        self.assertFalse(any(isinstance(c, gs.Companion) for a in state.armies.values() for c in a.leaders))
        self.assertEqual((state.political.boxes[gs.Nation.DWARVES], state.political.active[gs.Nation.DWARVES]), (0, True))
        self.assertEqual(self.row['die_idx'], 0)

    def test_no_effect_play_remains_legal_and_does_not_separate_anyone(self):
        state = self.before()
        self.assertTrue(check_event_requirement(state, CID)[0])
        legal = ai.get_legal_actions(state, FP)
        self.assertTrue(any(a.action_type == 'play_event' and a.die_idx == 0
                            and a.params.get('card_id') == CID for a in legal))
        options = get_card_options(state, FP, TITLE)
        self.assertFalse(any('move_group' in option for option in options))

    def test_recorded_outcome_is_exactly_card_and_die_consumption(self):
        before, after = self.before(), self.row['state']
        self.assert_spent_once(before, after)
        self.assertEqual(world(before), world(after))

    def test_current_execution_preserves_the_literal_zero_effect_outcome(self):
        before = self.before()
        after = self.play(before)
        self.assert_spent_once(before, after)
        self.assertEqual(world(before), world(after))

    def test_legitimate_automatic_completion_does_not_report_missing_legal_choice(self):
        start = len(state_invariants.VIOLATIONS)
        self.play(self.before())
        emitted = state_invariants.VIOLATIONS[start:]
        self.assertFalse(any(row.get('code') == 'card_play_no_legal_choice' for row in emitted), emitted)

    def test_zero_effect_event_play_still_offers_gandalf_bonus_after_discard(self):
        before = gs.setup_initial_game(board.load_board())
        gl.initialize_decks(before)
        before.hands[FP].hand = [CID]
        before.fp_strat_deck.remove(CID)
        die = gs.ActionDie([gs.ActionDieFace.EVENT], FP)
        die.current = gs.ActionDieFace.EVENT
        before.dice[FP] = [die]
        deck = list(before.fp_strat_deck)
        after = self.play(before)
        self.assertEqual(after.fellowship.guide.name, 'Gandalf the Grey')
        self.assertEqual(after.pending_card_option['card_id'], '__gandalf_draw__')
        self.assertEqual(after.pending_card_option['source_card_id'], CID)
        self.assertEqual(after.pending_card_option['deck'], 'strategy')
        self.assertEqual(after.hands[FP].hand, [])
        self.assertEqual(after.fp_strat_deck, deck, 'Bonus choice must precede its draw')
        self.assertEqual(after.discard_face_up['fp_strategy'].count(CID), 1)
        self.assertTrue(after.dice[FP][0].used)

    def separated(self, region):
        state = self.before()
        gimli = next(c for c in state.fellowship.companions if c.name == 'Gimli')
        state.fellowship.companions.remove(gimli)
        gl.get_or_create_army(state, region).leaders.append(gimli)
        return state

    def resolve(self, state, choice):
        pending = self.play(state)
        self.assertIsNotNone(pending.pending_card_option)
        actions = ai.get_legal_actions(pending, FP)
        selected = [a for a in actions if a.action_type == 'card_option' and a.params.get('choices') == choice]
        self.assertEqual(len(selected), 1)
        return ai.apply_action(pending, selected[0])

    def test_optional_no_movement_still_applies_independent_dwarven_trigger(self):
        before = self.separated('erebor')
        choices = get_card_options(before, FP, TITLE)
        self.assertIn({'_ffm_done': True}, choices)
        self.assertTrue(any('move_group' in item for item in choices))
        after = self.resolve(before, {'_ffm_done': True})
        self.assertEqual(world(before)['armies'], world(after)['armies'])
        self.assertEqual(after.political.boxes[gs.Nation.DWARVES], 3)
        self.assertTrue(after.political.active[gs.Nation.DWARVES])
        self.assert_spent_once(before, after)

    def test_real_movement_destinations_and_declining_movement_are_preserved(self):
        before = self.separated('dale')
        to_erebor = {'move_group': {'from': 'dale', 'to': 'erebor', 'companions': ['Gimli']}}
        to_iron_hills = {'move_group': {'from': 'dale', 'to': 'iron hills', 'companions': ['Gimli']}}
        choices = get_card_options(before, FP, TITLE)
        for literal in (to_erebor, to_iron_hills, {'_ffm_done': True}):
            self.assertIn(literal, choices)
        pending = self.play(before)
        for menu in (ai.get_legal_actions(pending, FP), ai.get_ai_actions(pending, FP)):
            exposed = [a.params.get('choices') for a in menu if a.action_type == 'card_option']
            for literal in (to_erebor, to_iron_hills, {'_ffm_done': True}):
                self.assertIn(literal, exposed)
        stopped = self.resolve(before, {'_ffm_done': True})
        self.assertEqual(world(stopped), world(before))
        moved = self.resolve(before, to_erebor)
        self.assertEqual(moved.political.boxes[gs.Nation.DWARVES], 3)
        positions = {board.sanitize(k) for k, a in moved.armies.items()
                     if any(getattr(c, 'name', '') == 'Gimli' for c in a.leaders)}
        self.assertEqual(positions, {'erebor'})
        self.assert_spent_once(before, moved)


if __name__ == '__main__':
    unittest.main()
