"""The Mordor Track has no map region (supplied rulebook p43)."""
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import board
import cards
import combat
import game_state as gs


class MordorCombatCardTests(unittest.TestCase):
    def fixture(self, gateway, entered):
        state = gs.setup_initial_game(board.load_board())
        origin = "dagorlad" if gateway == "morannon" else "north ithilien"
        state.armies = {
            origin: gs.Army(state.board.region(origin),
                           [gs.Unit(gs.Nation.GONDOR, gs.UnitKind.REGULAR)], []),
            gateway: gs.Army(state.board.region(gateway),
                            [gs.Unit(gs.Nation.SAURON, gs.UnitKind.REGULAR)], []),
        }
        for nation in gs.Nation:
            state.political.boxes[nation] = 3
            state.political.active[nation] = True
        state.fellowship.region = state.board.region(gateway)
        state.fellowship.mordor_entered = entered
        state.fellowship.mordor_step = int(entered)
        battle = combat.setup_combat(state, origin, gateway, attacker_side=gs.Side.FREE_PEOPLES)
        return state, battle

    def test_all_eight_region_cards_reject_both_former_gateways(self):
        affected = [card for card in cards.get_cards().values() if card.requires_fellowship_region]
        self.assertEqual(len(affected), 8)
        for gateway in ("morannon", "minas morgul"):
            state, battle = self.fixture(gateway, True)
            for card in affected:
                with self.subTest(gateway=gateway, card=card.card_name):
                    valid, reason = cards.can_play_combat_card(
                        card, battle, card.side == gs.Side.FREE_PEOPLES, state)
                    self.assertFalse(valid)
                    self.assertIn("Mordor Track", reason)

    def test_map_fellowship_all_eight_cards_remain_legal(self):
        for gateway in ("morannon", "minas morgul"):
            state, battle = self.fixture(gateway, False)
            for card in cards.get_cards().values():
                if card.requires_fellowship_region:
                    with self.subTest(gateway=gateway, card=card.card_name):
                        valid, reason = cards.can_play_combat_card(
                            card, battle, card.side == gs.Side.FREE_PEOPLES, state)
                        self.assertTrue(valid, reason)

    def test_other_combat_cards_are_unaffected_by_track_entry(self):
        for gateway in ("morannon", "minas morgul"):
            state, battle = self.fixture(gateway, False)
            for card in cards.get_cards().values():
                if card.requires_fellowship_region:
                    continue
                side = card.side == gs.Side.FREE_PEOPLES
                state.fellowship.mordor_entered = False
                before = cards.can_play_combat_card(card, battle, side, state)
                state.fellowship.mordor_entered = True
                after = cards.can_play_combat_card(card, battle, side, state)
                self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
