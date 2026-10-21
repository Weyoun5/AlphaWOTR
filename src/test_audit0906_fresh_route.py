"""Fresh live Fellowship routes must not depend on clone-created fields."""
from collections import Counter
from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch

SOURCE = Path(os.environ.get('WOTR_TIMING_SOURCE', str(Path(__file__).resolve().parent))).resolve()
os.environ.update(WOTR_ACT_GEN='v51', WOTR_NATIVE='0', CUDA_VISIBLE_DEVICES='-1')
sys.path.insert(0, str(SOURCE))
import ai
import board
import game_logic as gl
import game_state as gs
assert all(Path(module.__file__).resolve().parent == SOURCE for module in (ai, gl, gs))
FP, SH = gs.Side.FREE_PEOPLES, gs.Side.SHADOW
ORDINARY = Counter({'3': 3, '2': 2, '1': 2, 'Eye': 4, '2R': 1, '1R': 2, '0R': 2})


def die(face, side):
    item = gs.ActionDie([face], side)
    item.current = face
    return item


class FreshLiveRoute(unittest.TestCase):
    def setUp(self):
        self.state = gs.setup_initial_game(board.load_board())
        gl.initialize_decks(self.state)
        self.assertFalse(hasattr(self.state, 'pending_card_option'))
        self.state.fellowship.guide = next(c for c in self.state.fellowship.companions if c.name == 'Strider')
        self.state.fellowship.region = self.state.board.region('hollin')
        self.state.fellowship.progress = 0
        self.state.dice = {FP: [die(gs.ActionDieFace.CHARACTER, FP)],
                           SH: [die(gs.ActionDieFace.EYE, SH)]}
        self.state.dice[SH][0].used = True
        self.state.hunt_box = [self.state.dice[SH][0]]
        # Literal physically conserved bag: primary 1R, extra stronghold 3.
        bag = list(self.state.hunt_pool)
        first = next(tile for tile in bag if repr(tile) == '1R')
        bag.remove(first)
        second = next(tile for tile in bag if repr(tile) == '3')
        bag.remove(second)
        self.state.hunt_pool = [second, first]
        self.state.hunt_discard = bag
        # Chance controls return eligible physical tiles; player menus and
        # effect execution remain real, with no clone/apply normalization.
        with patch.object(gl, 'roll_d6_batch', side_effect=lambda count, *args: [6] * count), \
             patch.object(gl, 'pick_hunt_tile', side_effect=lambda current: current.hunt_pool[-1]):
            self.play('fellowship_move')
            self.play('hunt_take_corruption')
        self.assertFalse(hasattr(self.state, 'pending_card_option'))
        self.assertEqual(gl.fellowship_region(self.state), 'hollin')
        self.assertEqual(self.state.fellowship.progress, 1)
        self.assertEqual(self.state.fellowship.corruption, 1)
        self.assertEqual(list(map(repr, self.state.hunt_pool)), ['3'])
        self.inventory()

    def play(self, kind, **params):
        legal = [action for action in ai.get_legal_actions(self.state, FP)
                 if action.action_type == kind and
                 all(action.params.get(key) == value for key, value in params.items())]
        self.assertEqual(len(legal), 1, (kind, params))
        ai._execute_action(self.state, legal[0])

    def inventory(self):
        self.assertEqual(Counter(map(repr, self.state.hunt_pool + self.state.hunt_discard)), ORDINARY)

    def test_missing_card_field_retains_damage_choice_and_die_until_resolution(self):
        self.play('fellowship_reveal_place', region='moria')
        self.assertEqual(self.state.pending_hunt, {'damage': 3, 'phase': 'hunt_sacrifice'})
        self.assertEqual(self.state.fellowship.corruption, 1)
        self.assertEqual(len(self.state.hunt_box), 1)
        self.assertEqual(self.state._fellowship_move_die_pending, 0)
        self.assertFalse(hasattr(self.state, 'pending_card_option'))
        self.inventory()
        self.play('hunt_take_corruption')
        self.assertEqual(self.state.fellowship.corruption, 4)
        self.assertIsNone(self.state.pending_hunt)
        self.assertEqual(self.state._pending_hunt_damage, 0)
        self.assertIsNone(self.state._fellowship_route_event)
        self.assertFalse(self.state._pending_reveal_strongholds)
        self.assertIsNone(self.state._fellowship_move_die_pending)
        self.assertEqual(len(self.state.hunt_box), 2)
        self.inventory()

    def test_clone_initialized_card_field_has_same_damage_choice(self):
        self.state = ai.clone_state(self.state)
        self.assertTrue(hasattr(self.state, 'pending_card_option'))
        self.play('fellowship_reveal_place', region='moria')
        self.assertEqual(self.state.pending_hunt, {'damage': 3, 'phase': 'hunt_sacrifice'})
        self.inventory()

    def test_extra_eye_resolves_with_zero_damage_without_card_field(self):
        tile = next(tile for tile in self.state.hunt_discard if tile.is_eye)
        self.state.hunt_discard.remove(tile)
        self.state.hunt_discard.extend(self.state.hunt_pool)
        self.state.hunt_pool = [tile]
        self.play('fellowship_reveal_place', region='moria')
        self.assertEqual(self.state.fellowship.corruption, 1)
        self.assertIsNone(self.state.pending_hunt)
        self.assertIsNone(self.state._fellowship_route_event)
        self.assertEqual(len(self.state.hunt_box), 2)
        self.inventory()


if __name__ == '__main__':
    unittest.main(verbosity=2)
