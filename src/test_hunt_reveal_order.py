"""Independent Rulebook pp41-42 / Almanac pp6,11,23 Hunt ordering tests."""
import os
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import ai
import board
import cards
import game_logic as gl
import game_state as gs

FP, SH = gs.Side.FREE_PEOPLES, gs.Side.SHADOW


def fixture(*, mordor=False, card=None, tile=None, companions=("Meriadoc",)):
    state = gs.setup_initial_game(board.load_board())
    state.armies = {}
    state.persistent_cards = gs.PersistentCardState()
    state.dice_rolled = True
    state._declare_phase = state._eye_allocation_phase = False
    state._fellowship_tail_phase = state._reveal_place_due = False
    state.fp_elven_rings = state.sh_elven_rings = 0
    for hand in state.hands.values():
        hand.hand = []
        hand.needs_discard = False
    state.fellowship.region = state.board.region("hollin")
    state.fellowship.progress = 0 if mordor else 1
    state.fellowship.corruption = 0
    state.fellowship.hidden = True
    state.fellowship.mordor_entered = mordor
    state.fellowship.mordor_step = 1 if mordor else 0
    available = {c.name: c for c in state.fellowship.companions}
    state.fellowship.companions = [available[name] for name in companions]
    state.fellowship.guide = (state.fellowship.companions[0] if companions else
                             gs.Companion("Gollum", 0, 0, guide_ability="no_reveal_on_damage"))
    state.hunt_pool = [tile or gs.HuntTile(damage=2, reveal=True)]
    state.hunt_discard = []
    actor = SH if card else FP
    for side, dice in state.dice.items():
        for index, die in enumerate(dice):
            die.used = not (side == actor and index == 1)
            die.current = gs.ActionDieFace.EVENT if card else gs.ActionDieFace.CHARACTER
    state.hunt_box = []
    if card:
        cid = next(cid for cid, obj in cards.get_cards().items() if obj.card_name == card)
        state.hands[SH].hand = [cid]
    elif not mordor:
        eye = state.dice[SH][0]
        eye.current = gs.ActionDieFace.EYE
        state.hunt_box = [eye, state.dice[FP][0], state.dice[FP][2]]
    return state, actor


def rows(state, side=FP):
    return ai.get_legal_actions(state, side)


def action(state, kind=None, *, side=FP, choice=None):
    choices = rows(state, side)
    selected = next(a for a in choices if
                    (kind is None or a.action_type == kind) and
                    (choice is None or choice(a.params.get("choices", {}))))
    return ai.apply_action(state, selected)


def start(state, actor):
    random.seed(0)
    return action(state, "play_event" if actor == SH else "fellowship_move", side=actor)


def table_card(state, name, kind):
    cid = next(cid for cid, obj in cards.get_cards().items() if obj.card_name == name)
    state.persistent_cards.add_card(gs.PersistentCard(cid, name, FP, kind))


def palantir_bonus(state):
    card = next(c for c in cards.get_cards().values()
                if "Palant" in c.card_name and "Orthanc" in c.card_name)
    state.persistent_cards.add_card(gs.PersistentCard(
        card.card_id, card.card_name, SH, gs.PersistentCardType.PALANTIR_OF_ORTHANC))
    state.saruman.in_play = state.saruman.was_ever_in_play = True
    gl._add_minion_to_region(state, state.saruman, "orthanc")
    state.sh_char_deck = [next(cid for cid, c in cards.get_cards().items()
                              if c.card_name == "Morgul Wound")]
    state.sh_strat_deck = []


class HuntRevealOrderTests(unittest.TestCase):
    def test_last_hobbit_ordinary_reveal_all_draw_sources(self):
        for mordor, card in ((False, None), (True, None),
                             (False, "Orc Patrol"), (False, "Foul Thing from the Deep")):
            with self.subTest(mordor=mordor, card=card):
                state, actor = fixture(mordor=mordor, card=card)
                drawn = start(state, actor)
                self.assertTrue(drawn.fellowship.hidden)
                self.assertEqual(drawn._hunt_reveal_pending, 1)
                if card == "Foul Thing from the Deep":
                    after = action(drawn, "card_option", choice=lambda c: c.get("use_foul_guide") == 1)
                    self.assertTrue(after.pending_card_option["foul_guide"])
                    self.assertEqual(after.pending_card_option["foul_resume"]["damage"], 1)
                else:
                    after = action(drawn, "hunt_guide_separate")
                    self.assertEqual(after.pending_hunt["phase"], "hunt_gollum")
                    self.assertEqual(after.pending_hunt["damage"], 1)
                self.assertTrue(after.fellowship.hidden)
                self.assertEqual(after.fellowship.corruption, 0)
                self.assertEqual(after.fellowship.guide.name, "Gollum")

    def test_last_nonhobbit_casualty_suppresses_ordinary_icon(self):
        for kind in ("hunt_sacrifice_guide", "hunt_sacrifice_random"):
            state, actor = fixture(companions=("Boromir",))
            after = action(start(state, actor), kind)
            self.assertEqual(after.fellowship.guide.name, "Gollum")
            self.assertTrue(after.fellowship.hidden)
            self.assertEqual(after.fellowship.corruption, 0)
            self.assertFalse(after._hunt_reveal_pending)
            self.assertFalse(after._reveal_place_due)

    def test_ordinary_reveal_waits_for_damage_then_places(self):
        state, actor = fixture(companions=("Boromir", "Meriadoc"))
        drawn = start(state, actor)
        self.assertTrue(drawn.fellowship.hidden)
        after = action(drawn, "hunt_take_corruption")
        self.assertFalse(after.fellowship.hidden)
        self.assertEqual(after.fellowship.corruption, 2)
        self.assertIn("fellowship_reveal_place", [a.action_type for a in rows(after)])

    def test_unavoidable_icons_never_offer_gollum_damage_reduction(self):
        for tile in (gs.HuntTile(is_eye=True),
                     gs.HuntTile(damage=2, reveal=True, is_special=True,
                                 special_name="Shelob's Lair", stops_movement=True)):
            state, actor = fixture(mordor=True, tile=tile)
            state.hunt_box = [state.dice[SH][0], state.dice[SH][2]]
            drawn = start(state, actor)
            self.assertTrue(drawn.fellowship.hidden)
            self.assertEqual(drawn._hunt_reveal_pending, 2)
            after = action(drawn, "hunt_guide_separate")
            self.assertFalse(after.fellowship.hidden)
            self.assertEqual(after.fellowship.corruption, 1)
            self.assertIsNone(after.pending_hunt)
            self.assertFalse(after._hunt_reveal_pending)

    def test_zero_reveal_and_gollum_already_guide(self):
        for companions in ((), ("Boromir",)):
            state, actor = fixture(mordor=True, companions=companions,
                                   tile=gs.HuntTile(damage=0, reveal=True))
            after = start(state, actor)
            self.assertEqual(after.fellowship.hidden, not companions)
            self.assertEqual(after.fellowship.corruption, 0)
            self.assertFalse(getattr(after, "_hunt_reveal_pending", 0))

    def test_already_revealed_card_draw_does_not_create_new_route(self):
        state, actor = fixture(card="Orc Patrol", companions=("Boromir",))
        state.fellowship.hidden = False
        state.fellowship.progress = 0
        after = action(start(state, actor), "hunt_take_corruption")
        self.assertFalse(after.fellowship.hidden)
        self.assertFalse(after._reveal_place_due)
        self.assertFalse(after._hunt_reveal_pending)
        self.assertEqual(len(after.hunt_discard), 1)

    def test_axe_choice_precedes_new_guide_and_same_tile_reveal(self):
        state, actor = fixture(companions=("Gimli",))
        table_card(state, "Axe and Bow", gs.PersistentCardType.AXE_AND_BOW)
        drawn = start(state, actor)
        self.assertTrue(drawn.pending_card_option["hunt_reduction"])
        self.assertIsNone(drawn.pending_hunt)
        self.assertTrue(drawn.fellowship.hidden)
        reduced = action(drawn, "card_option", choice=lambda c: c.get("use_hunt_reduction") == 1)
        after = action(reduced, "hunt_sacrifice_guide")
        self.assertTrue(after.fellowship.hidden)
        self.assertEqual(after.fellowship.guide.name, "Gollum")
        self.assertFalse(after._hunt_reveal_pending)
        self.assertEqual(after.fellowship.corruption, 0)

    def test_mithril_rejected_reveal_is_not_staged(self):
        state, actor = fixture(mordor=True, companions=("Boromir",))
        table_card(state, "Mithril Coat and Sting", gs.PersistentCardType.MITHRIL_COAT)
        # Movement pops the last tile. Replacement is the nonrevealing tile.
        state.hunt_pool.insert(0, gs.HuntTile(damage=1))
        drawn = start(state, actor)
        self.assertEqual(drawn.pending_card_option["card_name"], "Mithril Coat and Sting")
        self.assertFalse(drawn._hunt_reveal_pending)
        replaced = action(drawn, "card_option", choice=lambda c: c.get("mithril_redraw") is True)
        self.assertTrue(replaced.fellowship.hidden)
        self.assertFalse(replaced._hunt_reveal_pending)
        after = action(replaced, "hunt_take_corruption")
        self.assertTrue(after.fellowship.hidden)
        self.assertEqual(after.fellowship.corruption, 1)
        self.assertEqual([repr(t) for t in after.hunt_pool], ["2R"])

    def test_staff_prevention_never_stages_tile_reveal(self):
        state, actor = fixture(mordor=True, companions=("Gandalf the Grey",))
        table_card(state, "Wizard's Staff", gs.PersistentCardType.WIZARD_STAFF)
        drawn = start(state, actor)
        after = action(drawn, "card_option", choice=lambda c: c.get("wizardstaff_prevent") is True)
        self.assertTrue(after.fellowship.hidden)
        self.assertFalse(after._hunt_reveal_pending)
        self.assertEqual(len(after.hunt_pool), 1)
        self.assertEqual(after.hunt_discard, [])
        self.assertEqual(after.fellowship.mordor_step, 2)

    def test_foul_gollum_voluntary_reveal_then_separate_stronghold_tile(self):
        state, actor = fixture(card="Foul Thing from the Deep", companions=(),
                               tile=gs.HuntTile(damage=2))
        state.fellowship.progress = 2
        state.hunt_pool.append(gs.HuntTile(damage=2))
        drawn = start(state, actor)
        after = action(drawn, "card_option", choice=lambda c: c.get("use_foul_guide") == 1)
        self.assertFalse(after.fellowship.hidden)
        self.assertEqual(after.fellowship.corruption, 1)
        self.assertEqual(len(after.hunt_discard), 1)
        placement = next(a for a in rows(after) if a.params.get("region") == "moria")
        placed = ai.apply_action(after, placement)
        self.assertEqual(gl.fellowship_region(placed), "moria")
        self.assertEqual(placed.fellowship.progress, 0)
        self.assertEqual(placed.fellowship.corruption, 3)
        self.assertEqual(len(placed.hunt_discard), 2)
        self.assertFalse(placed._reveal_place_due)

    def test_foul_gollum_reveal_on_mordor_has_no_map_placement(self):
        state, actor = fixture(mordor=True, card="Foul Thing from the Deep",
                               companions=(), tile=gs.HuntTile(damage=2))
        after = action(start(state, actor), "card_option",
                       choice=lambda c: c.get("use_foul_guide") == 1)
        self.assertFalse(after.fellowship.hidden)
        self.assertFalse(after._reveal_place_due)
        self.assertFalse(after._hunt_reveal_pending)
        self.assertEqual(after.fellowship.corruption, 1)
        self.assertEqual(after.fellowship.mordor_step, 1)

    def test_palantir_waits_for_hunt_damage_reveal_and_route_tile(self):
        state, actor = fixture(card="Orc Patrol", companions=("Boromir",),
                               tile=gs.HuntTile(damage=1, reveal=True))
        palantir_bonus(state)
        state.fellowship.progress = 2
        state.hunt_pool.append(gs.HuntTile(damage=1, reveal=True))
        drawn = start(state, actor)
        self.assertIsNone(drawn.pending_card_option, "Palantir waits for the first damage")
        self.assertTrue(drawn._palantir_draw_pending)
        after = action(drawn, "hunt_take_corruption")
        self.assertIsNone(after.pending_card_option, "Palantir waits for reveal placement")
        self.assertTrue(after._reveal_place_due)
        placement = next(a for a in rows(after) if a.params.get("region") == "moria")
        placed = ai.apply_action(after, placement)
        self.assertIsNone(placed.pending_card_option, "Palantir waits for route tile damage")
        self.assertEqual(placed.pending_hunt["damage"], 1)
        final = action(placed, "hunt_take_corruption")
        self.assertTrue(final.pending_card_option["palantir_draw"])
        self.assertFalse(final._hunt_reveal_pending)
        self.assertFalse(final._reveal_place_due)
        self.assertIsNone(final._fellowship_route_event)
        self.assertEqual(final.fellowship.corruption, 2)
        self.assertEqual(len(final.hunt_discard), 2)

    def test_palantir_foul_gollum_choice_waits_for_reveal_route(self):
        state, actor = fixture(card="Foul Thing from the Deep", companions=(),
                               tile=gs.HuntTile(damage=2))
        palantir_bonus(state)
        state.fellowship.progress = 2
        state.hunt_pool.append(gs.HuntTile(damage=2))
        drawn = start(state, actor)
        self.assertTrue(drawn.pending_card_option["foul_guide"])
        self.assertTrue(drawn._palantir_draw_pending)
        after = action(drawn, "card_option", choice=lambda c: c.get("use_foul_guide") == 1)
        self.assertIsNone(after.pending_card_option)
        self.assertTrue(after._reveal_place_due)
        placement = next(a for a in rows(after) if a.params.get("region") == "moria")
        final = ai.apply_action(after, placement)
        self.assertTrue(final.pending_card_option["palantir_draw"])
        self.assertFalse(final._reveal_place_due)
        self.assertIsNone(final._fellowship_route_event)
        self.assertEqual(final.fellowship.corruption, 3)
        self.assertEqual(len(final.hunt_discard), 2)

    def test_palantir_nazgul_search_waits_for_movement_and_reveal(self):
        name = next(c.card_name for c in cards.get_cards().values()
                    if c.card_name.startswith("Nazg") and c.card_name.endswith("Search"))
        state, actor = fixture(card=name, companions=("Boromir",))
        palantir_bonus(state)
        state.armies["morannon"] = gs.Army(state.board.region("morannon"), [],
                                          [gs.Nazgul(gs.Nation.SAURON)])
        drawn = start(state, actor)
        self.assertTrue(drawn._palantir_draw_pending)
        moved = action(drawn, "card_option", side=SH,
                       choice=lambda c: c.get("to_region") == "hollin")
        self.assertIsNone(moved.pending_card_option)
        self.assertTrue(moved._reveal_place_due)
        placement = next(a for a in rows(moved) if a.params.get("region") == "hollin")
        after = ai.apply_action(moved, placement)
        self.assertTrue(after.pending_card_option["palantir_draw"])
        self.assertFalse(after._reveal_place_due)
        self.assertEqual(after.fellowship.progress, 0)

    def test_palantir_direct_corruption_card_and_complete_event(self):
        state, actor = fixture(card="Isildur's Bane", companions=("Boromir",),
                               tile=gs.HuntTile(damage=1, reveal=True))
        palantir_bonus(state)
        after = start(state, actor)
        self.assertIsNone(after.pending_card_option)
        self.assertTrue(after._reveal_place_due)
        self.assertEqual(after.fellowship.corruption, 1)
        placement = next(a for a in rows(after) if a.params.get("region") == "hollin")
        final = ai.apply_action(after, placement)
        self.assertTrue(final.pending_card_option["palantir_draw"])
        state, actor = fixture(card="Orc Patrol", companions=("Boromir",),
                               tile=gs.HuntTile(damage=0))
        palantir_bonus(state)
        final = start(state, actor)
        self.assertTrue(final.pending_card_option["palantir_draw"])
        self.assertFalse(final._palantir_draw_pending)


if __name__ == "__main__":
    unittest.main()
