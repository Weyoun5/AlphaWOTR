"""Literal companion movement destinations and complete same-die groups."""
from itertools import combinations
from pathlib import Path
import os
import pickle
import sys
import unittest

SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import ai
import board
import cards
import event_effects as ee
import game_logic as gl
import game_state as gs

FP, SH = gs.Side.FREE_PEOPLES, gs.Side.SHADOW
SEVEN = {"Gandalf the Grey", "Strider", "Legolas", "Gimli", "Boromir",
         "Meriadoc", "Peregrin"}


def ready(placements):
    state = gs.setup_initial_game(board.load_board())
    state.armies = {}
    for name, region in placements.items():
        companion = next(c for c in state.fellowship.companions if c.name == name)
        state.fellowship.companions.remove(companion)
        if region not in state.armies:
            state.armies[region] = gs.Army(state.board.region(region))
        state.armies[region].leaders.append(companion)
    if state.fellowship.companions:
        chosen = max(state.fellowship.companions, key=lambda c: c.level)
        gl.change_guide(state, chosen=chosen.name)
    else:
        gl.change_guide(state)
    for nation in gs.Nation:
        state.political.boxes[nation] = 3
        state.political.active[nation] = True
    state.dice_rolled = True
    for dice in state.dice.values():
        for die in dice:
            die.used = True
            die.current = gs.ActionDieFace.CHARACTER
    state.dice[FP][0].used = False
    return state


def shadow(state, region, besieged=False):
    army = state.armies.setdefault(region, gs.Army(state.board.region(region)))
    army.units.append(gs.Unit(gs.Nation.SAURON, gs.UnitKind.REGULAR))
    if besieged:
        army.units.append(gs.Unit(gs.Nation.ELVES, gs.UnitKind.REGULAR))
        state.besieged_strongholds.add(region)


def first_move(state, source, destination, names):
    return next(a for a in ai.get_legal_actions(state, FP)
                if a.action_type == "companion_move"
                and a.params["from"] == source and a.params["to"] == destination
                and (a.params.get("companions") == sorted(names)
                     or ("companions" not in a.params
                         and sorted(c.name for c in state.armies[source].leaders
                                    if isinstance(c, gs.Companion)) == sorted(names))))


def continuation(state, source, destination, names):
    return next((a for a in ai.get_legal_actions(state, FP)
                 if a.action_type == "card_option"
                 and a.params["choices"].get("from_region") == source
                 and a.params["choices"].get("to_region") == destination
                 and a.params["choices"].get("companions") == sorted(names)), None)


def group_options(state):
    return {(o["move_group"]["from"], o["move_group"]["to"],
             tuple(o["move_group"]["companions"]))
            for o in ee._companion_group_move_options(state)}


def locations(state):
    return {c.name: board.sanitize(region)
            for region, army in state.armies.items()
            for c in army.leaders if isinstance(c, gs.Companion)}


def field_position():
    state = ready({"Legolas": "bree", "Meriadoc": "bree"})
    shadow(state, "buckland")
    return state, first_move(state, "bree", "north downs", ["Legolas"])


def pairs_position():
    state = ready({name: "bree" for name in SEVEN})
    return state, first_move(state, "bree", "arnor", ["Legolas", "Peregrin"])


def stronghold_position(besieged=False):
    state = ready({"Legolas": "bree", "Gimli": "hollin", "Meriadoc": "hollin"})
    shadow(state, "moria", besieged=besieged)
    return state, first_move(state, "bree", "north downs", ["Legolas"])


def event_position(name):
    state = ready({person: "bree" for person in SEVEN})
    shadow(state, "buckland")
    card = next(c for c in cards.get_cards().values() if c.card_name == name)
    state.hands[FP].hand.append(card.card_id)
    state.dice[FP][0].current = gs.ActionDieFace.MUSTER
    action = next(a for a in ai.get_legal_actions(state, FP)
                  if a.action_type == "play_event" and a.params["card_id"] == card.card_id)
    return state, action


class CompanionContinuationTests(unittest.TestCase):
    def test_shadow_field_destination_and_once_per_die(self):
        state, action = field_position()
        before = pickle.dumps(state, protocol=5)
        after = ai.apply_action(state, action)
        move = continuation(after, "bree", "buckland", ["Meriadoc"])
        self.assertIsNotNone(move)
        final = ai.apply_action(after, move)
        self.assertEqual(locations(final), {"Legolas": "north downs", "Meriadoc": "buckland"})
        self.assertEqual(len(final.armies["buckland"].units), 1)
        self.assertIsNone(final.pending_card_option)
        self.assertEqual(pickle.dumps(state, protocol=5), before)

    def test_two_escorted_hobbits_leave_other_three_stationary(self):
        state, action = pairs_position()
        after = ai.apply_action(state, action)
        move = continuation(after, "bree", "cardolan", ["Gimli", "Meriadoc"])
        self.assertIsNotNone(move)
        self.assertIsNone(continuation(after, "bree", "cardolan", ["Meriadoc"]))
        final = ai.apply_action(after, move)
        done = next(a for a in ai.get_legal_actions(final, FP)
                    if a.action_type == "card_option" and a.params["choices"].get("_comp_done"))
        final = ai.apply_action(final, done)
        self.assertEqual(locations(final), {
            "Legolas": "arnor", "Peregrin": "arnor",
            "Gimli": "cardolan", "Meriadoc": "cardolan",
            "Gandalf the Grey": "bree", "Strider": "bree", "Boromir": "bree"})
        self.assertIsNone(final.pending_card_option)

    def test_every_subset_survives_four_five_and_seven_boundaries(self):
        names = sorted(SEVEN)
        for count in (4, 5, 7):
            with self.subTest(count=count):
                state = ready({name: "bree" for name in names[:count]})
                actual = {group for source, dest, group in group_options(state)
                          if source == "bree" and dest == "buckland"}
                expected = {group for size in range(1, count + 1)
                            for group in combinations(names[:count], size)}
                self.assertEqual(actual, expected)

    def test_shadow_stronghold_is_enterable_but_stops_movement(self):
        for besieged in (False, True):
            with self.subTest(besieged=besieged):
                state, action = stronghold_position(besieged)
                after = ai.apply_action(state, action)
                move = continuation(after, "hollin", "moria", ["Gimli", "Meriadoc"])
                self.assertIsNotNone(move)
                self.assertIsNone(continuation(
                    after, "hollin", "dimrill dale", ["Gimli", "Meriadoc"]))
                final = ai.apply_action(after, move)
                self.assertEqual(locations(final)["Gimli"], "moria")
                self.assertEqual(locations(final)["Meriadoc"], "moria")
                self.assertNotIn("moria", final.enemy_controlled[FP])
                # They may leave a Shadow-controlled stronghold on a later move.
                self.assertIn(("moria", "dimrill dale", ("Gimli", "Meriadoc")),
                              group_options(final))

    def test_friendly_besieged_stronghold_cannot_be_entered_or_left(self):
        state = ready({"Gimli": "dimrill dale"})
        shadow(state, "lorien", besieged=True)
        self.assertFalse(any(dest == "lorien" for _, dest, _ in group_options(state)))
        state = ready({"Gimli": "lorien"})
        shadow(state, "lorien", besieged=True)
        self.assertEqual(group_options(state), set())

    def test_both_companion_event_menus_include_field_and_escorted_subset(self):
        for name in ("Book of Mazarbul", "Fear! Fire! Foes!"):
            with self.subTest(card=name):
                state, action = event_position(name)
                after = ai.apply_action(state, action)
                offered = {(
                    c.get("move_group", {}).get("to"),
                    tuple(c.get("move_group", {}).get("companions", [])))
                    for c in after.pending_card_option["options"]}
                self.assertIn(("buckland", ("Meriadoc",)), offered)
                self.assertIn(("cardolan", ("Gimli", "Meriadoc")), offered)


if __name__ == "__main__":
    unittest.main()
