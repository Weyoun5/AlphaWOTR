"""A dead ordinary die is paid, not turned into an impossible pending action."""
import os
import unittest

os.environ.setdefault('WOTR_NATIVE', '0')
os.environ.setdefault('WOTR_ACT_GEN', 'v52')

import ai
import board
import game_state as gs


def exhausted_muster(side):
    state = gs.setup_initial_game(board.load_board())
    state.armies = {}
    for nation in state.political.boxes:
        state.political.boxes[nation] = 3
        state.political.active[nation] = True
    other = gs.Side.SHADOW if side == gs.Side.FREE_PEOPLES else gs.Side.FREE_PEOPLES
    # Every home settlement has been lost; no recruitment or political move
    # exists. This position does not depend on the menu implementation.
    state.enemy_controlled[other] = set(ai._get_settlement_regions())
    for minion in (state.saruman, state.witch_king, state.mouth_of_sauron):
        minion.in_play = True
    state.sh_mouth_muster_used_this_turn = True
    state.fp_ring_used_this_turn = state.sh_ring_used_this_turn = True
    state.dice = {gs.Side.FREE_PEOPLES: [], gs.Side.SHADOW: []}
    die = gs.ActionDie([gs.ActionDieFace.MUSTER], side)
    die.current = gs.ActionDieFace.MUSTER
    state.dice[side] = [die]
    state.dice_rolled = True
    state._declare_phase = state._eye_allocation_phase = False
    state._ring_free_side = side
    return state


class DeadDieDiscard(unittest.TestCase):
    def test_both_sides_can_finish_unusable_ring_action(self):
        for side in (gs.Side.FREE_PEOPLES, gs.Side.SHADOW):
            with self.subTest(side=side):
                state = exhausted_muster(side)
                actions = ai.get_legal_actions(state, side)
                self.assertEqual(len(actions), 1)
                action = actions[0]
                self.assertEqual((action.action_type, action.die_idx), ('skip', 0))
                self.assertEqual(action.params['face'], 'MUSTER')
                successor = ai.apply_action(state, action)
                self.assertTrue(successor.dice[side][0].used)
                self.assertFalse(state.dice[side][0].used)
                self.assertIsNone(successor._ring_free_side)
                self.assertIsNone(ai._next_acting_side(successor, side))

    def test_political_move_prevents_voluntary_discard(self):
        state = exhausted_muster(gs.Side.SHADOW)
        state.political.boxes[gs.Nation.SAURON] = 2
        actions = ai.get_legal_actions(state, gs.Side.SHADOW)
        self.assertIn('advance_political', [a.action_type for a in actions])
        self.assertNotIn('skip', [a.action_type for a in actions])

    def test_unavailable_pending_choice_still_fails(self):
        state = exhausted_muster(gs.Side.SHADOW)
        state.pending_card_option = dict(card_id='fixture', card_name='fixture', side='sh', options=[])
        self.assertEqual(ai.get_legal_actions(state, gs.Side.SHADOW), [])
        with self.assertRaisesRegex(RuntimeError, 'pending decision'):
            ai.apply_action(state, ai.Action(0, 'skip', {'face': 'MUSTER', 'side': 'sh'}))

    def test_eye_cannot_be_discarded(self):
        state = exhausted_muster(gs.Side.SHADOW)
        state.dice[gs.Side.SHADOW][0].current = gs.ActionDieFace.EYE
        self.assertEqual(ai.get_legal_actions(state, gs.Side.SHADOW), [])
        with self.assertRaisesRegex(RuntimeError, 'non-Eye'):
            ai.apply_action(state, ai.Action(0, 'skip', {'face': 'EYE', 'side': 'sh'}))


if __name__ == '__main__':
    unittest.main()
