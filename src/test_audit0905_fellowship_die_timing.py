"""Rulebook p.18/p.44: only previous movement dice affect the current Hunt."""
from pathlib import Path
import os
import pickle
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get('WOTR_TIMING_SOURCE', str(Path(__file__).resolve().parent))).resolve()
sys.path.insert(0, str(SOURCE))
os.environ.update(WOTR_ACT_GEN='v47', WOTR_TIMING_GENERATION='v47', WOTR_NATIVE='0', CUDA_VISIBLE_DEVICES='-1')
import test_audit0905_mithril_return as fixture
ai, gl, gs = fixture.ai, fixture.gl, fixture.gs
FP, SH = fixture.FP, fixture.SH
assert Path(ai.__file__).resolve().parent == SOURCE

def mordor(original='Eye', previous=0, corruption=10):
    state = fixture.setup(original=original, replacement='Eye')
    gl.get_or_create_army(state, 'dimrill dale').leaders.extend(state.fellowship.companions)
    state.fellowship.companions = []
    state.fellowship.guide = state.gollum
    state.fellowship.mordor_entered = True
    state.fellowship.mordor_step = 4
    state.fellowship.corruption = corruption
    state.fellowship.hidden = True
    state.fellowship.region = state.board.region('morannon')
    for _ in range(previous):
        d = gs.ActionDie([gs.ActionDieFace.CHARACTER], FP)
        d.current = gs.ActionDieFace.CHARACTER
        d.used = True
        state.dice[FP].append(d)
        state.hunt_box.append(d)
    return state

class FellowshipDieTimingTests(unittest.TestCase):
    def test_last_battle_final_state_controls_payment_after_mithril_reveal(self):
        for replacement, revealed in [('1', False), ('1R', True)]:
            with self.subTest(replacement=replacement):
                state = fixture.setup(replacement=replacement)
                # A legal late-game board: crowned Aragorn with a Gondor unit
                # in Dagorlad, outside every Free Peoples Nation.
                army = gl.get_or_create_army(state, 'dagorlad')
                for name in ('Strider', 'Gandalf the Grey'):
                    person = next(c for c in state.fellowship.companions if c.name == name)
                    state.fellowship.companions.remove(person)
                    if name == 'Strider':
                        person.name = 'Aragorn'
                        person.leadership = 2
                    army.leaders.append(person)
                origin = gl.get_or_create_army(state, 'minas tirith')
                unit = next(u for u in origin.units if u.kind is gs.UnitKind.REGULAR)
                origin.units.remove(unit)
                army.units.append(unit)
                state.political.boxes[gs.Nation.GONDOR] = 3
                state.political.active[gs.Nation.GONDOR] = True
                state.aragorn_crowned.crowned = True
                state.fellowship.guide = next(c for c in state.fellowship.companions if c.name == 'Legolas')
                state, cid = fixture.add_protection_through_public_play(state, 'The Last Battle')
                state = fixture.redraw(fixture.move(state))
                self.assertEqual(len(state.hunt_box), 1)
                state = fixture.choose(state, 'hunt_take_corruption')
                if revealed:
                    self.assertEqual(len(state.hunt_box), 1)
                    state = fixture.place(state, 'dimrill dale')
                self.assertEqual(state.fellowship.hidden, not revealed)
                self.assertEqual(state.persistent_cards.has_card_type(gs.PersistentCardType.LAST_BATTLE), not revealed)
                self.assertEqual(len(state.hunt_box), 2 if revealed else 1)
                self.assertIsNone(state._fellowship_move_die_pending)

    def test_mithril_keep_and_redraw_award_correct_final_mordor_winner(self):
        for original, use in [('Eye', False), ('3', True)]:
            with self.subTest(original=original, redraw=use):
                state = fixture.move(mordor(original))
                self.assertEqual(len(state.hunt_box), 1)
                self.assertIsNone(gl.check_ring_victory(state)['victory'])
                final = fixture.redraw(state, yes=use)
                self.assertEqual(final.fellowship.corruption, 11)
                self.assertEqual(gl.check_ring_victory(final)['victory'], 'free_peoples')
                self.assertEqual(len(final.hunt_box), 2)
                self.assertIs(final.hunt_box[-1], final.dice[FP][1])
                self.assertIsNone(final._fellowship_move_die_pending)
                self.assertEqual(fixture.inventory(final), fixture.ORDINARY)

    def test_eye_counts_previous_movement_dice_but_not_current_one(self):
        for previous in (0, 1, 2):
            with self.subTest(previous=previous):
                state = fixture.move(mordor(previous=previous, corruption=6))
                final = fixture.redraw(state, yes=False)
                self.assertEqual(final.fellowship.corruption, 7 + previous)
                self.assertEqual(len(final.hunt_box), 2 + previous)

    def test_ordinary_eye_without_interpose_has_same_damage_and_final_payment(self):
        state = mordor()
        from persistent_effects import discard_table_card
        card = state.persistent_cards.get_by_type(gs.PersistentCardType.MITHRIL_COAT)
        discard_table_card(state, state.persistent_cards.remove_card(card.card_id))
        final = fixture.move(state)
        self.assertEqual(final.fellowship.corruption, 11)
        self.assertEqual(gl.check_ring_victory(final)['victory'], 'free_peoples')
        self.assertEqual(len(final.hunt_box), 2)
        self.assertIsNone(final._fellowship_move_die_pending)

    def test_payment_waits_through_reveal_placement_and_extra_damage(self):
        state = fixture.redraw(fixture.move(fixture.setup()))
        self.assertEqual(len(state.hunt_box), 1)
        state = fixture.choose(state, 'hunt_take_corruption')
        self.assertEqual(len(state.hunt_box), 1)
        state = fixture.place(state, 'moria')
        self.assertEqual(len(state.hunt_box), 1)
        final = fixture.choose(state, 'hunt_take_corruption')
        self.assertEqual(len(final.hunt_box), 2)
        self.assertIs(final.hunt_box[-1], final.dice[FP][1])
        self.assertIsNone(final._fellowship_move_die_pending)

    def test_staff_prevention_finishes_payment_without_drawing(self):
        state, cid = fixture.add_protection_through_public_play(fixture.setup(), "Wizard's Staff")
        before_pool = list(map(repr, state.hunt_pool))
        state = fixture.move(state)
        self.assertEqual(len(state.hunt_box), 1)
        self.assertEqual(list(map(repr, state.hunt_pool)), before_pool)
        final = fixture.choose(state, 'card_option', choices={'wizardstaff_prevent': True})
        self.assertEqual(list(map(repr, final.hunt_pool)), before_pool)
        self.assertEqual(len(final.hunt_box), 2)
        self.assertIsNone(final._fellowship_move_die_pending)

    def test_clones_pickle_and_repeated_settlement_preserve_physical_die(self):
        import fellowship_action_timing as timing
        state = fixture.move(mordor())
        before = pickle.dumps(state)
        for child in (ai.clone_state(state), pickle.loads(before)):
            self.assertIs(timing.pending_die(child), child.dice[FP][1])
            final = fixture.redraw(child, yes=False)
            saved = pickle.dumps(final)
            self.assertFalse(timing.finish_if_settled(final))
            self.assertEqual(pickle.dumps(final), saved)
        self.assertEqual(pickle.dumps(state), before)

    def test_pending_observation_and_turn_guard_include_die_zero(self):
        import numpy as np
        state = fixture.move(mordor())
        # The physical index is an execution detail; the public observation
        # shows one already-spent die awaiting placement after the action.
        for side in (FP, SH):
            encoded = ai.encode_state(state, side)
            self.assertEqual(encoded.shape, (4191,))
            self.assertEqual(encoded[-1], 1)
            other = ai.clone_state(state)
            other._fellowship_move_die_pending = None
            absent = ai.encode_state(other, side)
            np.testing.assert_array_equal(encoded[:-1], absent[:-1])
            self.assertEqual(absent[-1], 0)
        self.assertIn('_fellowship_move_die_pending', ai._turn_end_blockers(state))
        state.dice[FP][0], state.dice[FP][1] = state.dice[FP][1], state.dice[FP][0]
        state._fellowship_move_die_pending = 0
        self.assertIn('_fellowship_move_die_pending', ai._turn_end_blockers(state))
        final = fixture.redraw(state, yes=False)
        self.assertIs(final.hunt_box[-1], final.dice[FP][0])

    def test_bad_pending_die_refused_instead_of_silent_payment(self):
        import fellowship_action_timing as timing
        for invalid in (-1, 99, True, '1'):
            state = fixture.move(mordor())
            state._fellowship_move_die_pending = invalid
            with self.subTest(index=invalid), self.assertRaises(ValueError):
                timing.finish_if_settled(state)

    def test_search_does_not_merge_different_public_mithril_tiles(self):
        import random
        states = []
        for original in ('2', '3'):
            random.seed(95471)
            states.append(fixture.move(fixture.setup(original=original)))
        for signature in (ai._state_sig, ai._chance_sig):
            with self.subTest(signature=signature.__name__):
                values = [signature(state) for state in states]
                self.assertIsNotNone(values[0])
                self.assertNotEqual(values[0], values[1])

    def test_search_preserves_pending_die_and_original_return_obligations(self):
        state = fixture.redraw(fixture.move(fixture.setup()))
        for attribute in ('_fellowship_move_die_pending', '_mithril_stash'):
            other = ai.clone_state(state)
            setattr(other, attribute, None)
            for signature in (ai._state_sig, ai._chance_sig):
                with self.subTest(attribute=attribute, signature=signature.__name__):
                    self.assertNotEqual(signature(state), signature(other))

if __name__ == '__main__':
    unittest.main(verbosity=2)
