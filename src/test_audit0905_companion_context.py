"""Public moves with identical boards must expose different moved Companions."""
from pathlib import Path
import os
import pickle
import sys
import unittest

SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))
os.environ.update(WOTR_ACT_GEN='v46', WOTR_NATIVE='0', CUDA_VISIBLE_DEVICES='-1')
import ai
import board
import game_logic as gl
import game_state as gs
import numpy as np
import companion_context as context
assert Path(ai.__file__).resolve().parent == SOURCE
assert Path(context.__file__).resolve().parent == SOURCE


def public_states():
    base = gs.setup_initial_game(board.load_board())
    gl.initialize_decks(base)
    base.armies = {}
    figures = {}
    for name in ('Gimli', 'Legolas', 'Boromir'):
        figures[name] = next(c for c in base.fellowship.companions if c.name == name)
        base.fellowship.companions.remove(figures[name])
    for dice in base.dice.values():
        for die in dice:
            die.used = True
    base.dice[gs.Side.FREE_PEOPLES][0].used = False
    base.dice[gs.Side.FREE_PEOPLES][0].current = gs.ActionDieFace.CHARACTER
    states = []
    for name, origin in (('Gimli', 'iron hills'), ('Legolas', 'woodland realm')):
        state = ai.clone_state(base)
        for person, figure in figures.items():
            where = origin if person == name else ('minas tirith' if person == 'Boromir' else 'dale')
            gl.get_or_create_army(state, where).leaders.append(figure)
        move = next(a for a in ai.get_legal_actions(state, gs.Side.FREE_PEOPLES)
                    if a.action_type == 'companion_move' and a.params.get('from') == origin
                    and a.params.get('to') == 'dale'
                    and (not a.params.get('companions') or a.params['companions'] == [name]))
        states.append(ai.apply_action(state, move))
    return states


class CompanionContextTests(unittest.TestCase):
    def test_public_transposition_literal_history_and_remaining_moves(self):
        first, second = public_states()
        for side in (gs.Side.FREE_PEOPLES, gs.Side.SHADOW):
            arrays = [ai.encode_state(s, side) for s in (first, second)]
            self.assertEqual(arrays[0].shape, (4190,))
            np.testing.assert_array_equal(arrays[0][:4180], arrays[1][:4180])
            # Literal schema positions: active, Gandalf, Strider, Legolas, Gimli, ...
            np.testing.assert_array_equal(arrays[0][4180:], [1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
            np.testing.assert_array_equal(arrays[1][4180:], [1, 0, 0, 1, 0, 0, 0, 0, 0, 0])
            self.assertFalse(np.array_equal(arrays[0].astype(np.float16), arrays[1].astype(np.float16)))
        for state, expected in zip((first, second), ({'Boromir', 'Legolas'}, {'Boromir', 'Gimli'})):
            menu = ai.get_legal_actions(state, gs.Side.FREE_PEOPLES)
            self.assertEqual({n for a in menu for n in a.params.get('choices', {}).get('companions', [])}, expected)

    def test_public_done_clears_and_consumes_no_second_die(self):
        state = public_states()[0]
        before = [d.used for d in state.dice[gs.Side.FREE_PEOPLES]]
        done = next(a for a in ai.get_legal_actions(state, gs.Side.FREE_PEOPLES)
                    if a.params.get('choices', {}).get('_comp_done'))
        final = ai.apply_action(state, done)
        np.testing.assert_array_equal(ai.encode_state(final, gs.Side.FREE_PEOPLES)[4180:], np.zeros(10))
        self.assertEqual([d.used for d in final.dice[gs.Side.FREE_PEOPLES]], before)

    def test_clone_pickle_order_and_no_mutation(self):
        state = public_states()[0]
        before = pickle.dumps(state)
        expected = context.movement_tail(state)
        self.assertEqual(pickle.dumps(state), before)
        for other in (ai.clone_state(state), pickle.loads(before)):
            other.pending_card_option['options'].reverse()
            np.testing.assert_array_equal(context.movement_tail(other), expected)

    def test_invalid_or_conflicting_history_refused(self):
        for bad in (None, [], ['Gollum'], ['Gimli', 'Gimli'], ['Unknown']):
            state = public_states()[0]
            for option in state.pending_card_option['options']:
                if not option.get('_comp_done'):
                    option['_comp_moved'] = bad
            with self.subTest(history=bad), self.assertRaises(ValueError):
                ai.encode_state(state, gs.Side.FREE_PEOPLES)
        state = public_states()[0]
        state.pending_card_option['options'][0]['_comp_moved'] = ['Boromir']
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            ai.encode_state(state, gs.Side.FREE_PEOPLES)


if __name__ == '__main__':
    unittest.main(verbosity=2)
