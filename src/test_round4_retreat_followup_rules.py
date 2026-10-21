"""Literal retreat ownership/politics and card-required participant bounds."""
from pathlib import Path
import os
import pickle
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import ai
import board
import event_effects as ee
import game_logic as gl
import game_state as gs
from army_move_participants import selected
from persistent_effects import create_persistent_card

FP, SH = gs.Side.FREE_PEOPLES, gs.Side.SHADOW


def unit(nation):
    return gs.Unit(nation, gs.UnitKind.REGULAR)


def army(state, region):
    return next((a for k, a in state.armies.items()
                 if board.sanitize(k) == region), None)


def ready():
    state = gs.setup_initial_game(board.load_board())
    for nation in gs.Nation:
        state.political.boxes[nation] = 3
        state.political.active[nation] = True
    state.dice_rolled = True
    for dice in state.dice.values():
        for die in dice:
            die.current = gs.ActionDieFace.ARMY
    return state


def retreat_position(side=FP, count=10):
    state = ready()
    merry = next(c for c in state.fellowship.companions if c.name == "Meriadoc")
    state.fellowship.companions.remove(merry)
    gimli = next(c for c in state.fellowship.companions if c.name == "Gimli")
    state.fellowship.companions.remove(gimli)
    nation = gs.Nation.GONDOR if side == FP else gs.Nation.SAURON
    destination = "minas tirith" if side == FP else "north ithilien"
    attacker = "north ithilien" if side == FP else "minas tirith"
    foe = SH if side == FP else FP
    leaders = [merry, gs.Leader(gs.Nation.GONDOR), gs.Nazgul(gs.Nation.SAURON)]
    state.armies = {
        "osgiliath": gs.Army(state.board.region("osgiliath"), [unit(nation)], leaders),
        destination: gs.Army(state.board.region(destination),
                             [unit(nation) for _ in range(count)], [gimli]),
        attacker: gs.Army(state.board.region(attacker),
                          [unit(gs.Nation.SAURON if side == FP else gs.Nation.GONDOR)]),
    }
    if side == SH:
        # A plain Gondor Leader cannot coexist with only Shadow units.
        state.armies["osgiliath"].leaders = [merry, gs.Nazgul(gs.Nation.SAURON)]
        state.enemy_controlled[SH].add("osgiliath")
    state.pending_combat = dict(
        phase="defender_retreat", attacker_region=attacker,
        defender_region="osgiliath", attacker_side=foe, defender_side=side,
        deciding_side=side, is_siege=False, rounds_fought=1)
    return state, destination


def politics_position():
    state = ready()
    merry = next(c for c in state.fellowship.companions if c.name == "Meriadoc")
    state.fellowship.companions.remove(merry)
    state.armies = {
        "westemnet": gs.Army(state.board.region("westemnet"),
                            [unit(gs.Nation.ROHAN)], [merry]),
        "eastemnet": gs.Army(state.board.region("eastemnet"),
                            [unit(gs.Nation.SAURON)]),
    }
    state.political.active[gs.Nation.ROHAN] = False
    state.persistent_cards.add_card(create_persistent_card(
        "Shadow_Character_Wormtongue", "Wormtongue", SH))
    state.pending_combat = dict(
        phase="defender_retreat", attacker_region="eastemnet",
        defender_region="westemnet", attacker_side=SH, defender_side=FP,
        deciding_side=FP, is_siege=False, rounds_fought=1)
    return state


def followup_position(kind, leaders=None):
    state = ready()
    if leaders is None:
        leaders = ([gs.Nazgul(gs.Nation.SAURON)] if kind == "nazgul_armies"
                   else [gl.Minion(gs.Nation.SAURON, "Witch King")])
    state.armies = {"minas morgul": gs.Army(state.board.region("minas morgul"),
                                           [unit(gs.Nation.SAURON) for _ in range(2)],
                                           leaders)}
    if any(getattr(l, "name", "") == "Witch King" for l in leaders):
        state.witch_king.in_play = True
        state.witch_king.was_ever_in_play = True
    ee._grant_card_followup(state, SH, kind, 2 if kind == "nazgul_armies" else 1)
    return state


def retreat_action(state, destination, side=FP):
    return next(a for a in ai.get_legal_actions(state, side)
                if a.action_type == "combat_retreat"
                and a.params["to_region"] == destination)


class RetreatOwnershipTests(unittest.TestCase):
    def test_both_sides_preserve_enemy_and_destination_characters_at_cap_boundary(self):
        for side in (FP, SH):
            for count in (9, 10):
                with self.subTest(side=side, count=count):
                    state, destination = retreat_position(side, count)
                    before = pickle.dumps(state, protocol=5)
                    after = ai.apply_action(state, retreat_action(state, destination, side))
                    origin, dest = army(after, "osgiliath"), army(after, destination)
                    self.assertEqual(pickle.dumps(state, protocol=5), before)
                    self.assertIn("Gimli", [getattr(l, "name", "") for l in dest.leaders])
                    if side == FP:
                        self.assertTrue(any(isinstance(l, gs.Nazgul) for l in origin.leaders))
                        self.assertFalse(any(isinstance(l, gs.Nazgul) for l in dest.leaders))
                        self.assertIn("Meriadoc", [getattr(l, "name", "") for l in dest.leaders])
                        self.assertTrue(any(type(l) is gs.Leader for l in dest.leaders))
                    else:
                        self.assertIn("Meriadoc", [getattr(l, "name", "") for l in origin.leaders])
                        self.assertNotIn("Meriadoc", [getattr(l, "name", "") for l in dest.leaders])
                        self.assertTrue(any(isinstance(l, gs.Nazgul) for l in dest.leaders))

    def test_retreat_into_character_only_destination_preserves_every_character(self):
        state, destination = retreat_position(FP, 0)
        after = ai.apply_action(state, retreat_action(state, destination))
        self.assertEqual({getattr(l, "name", "") for l in army(after, destination).leaders},
                         {"Gimli", "Meriadoc", ""})
        self.assertIs(army(after, destination).region, state.board.region(destination))

    def test_saruman_stays_in_orthanc_when_shadow_retreats(self):
        state = ready()
        state.armies = {
            "orthanc": gs.Army(state.board.region("orthanc"), [unit(gs.Nation.ISENGARD)],
                               [gl.Minion(gs.Nation.ISENGARD, "Saruman"), gs.Nazgul(gs.Nation.SAURON)]),
            "fords of isen": gs.Army(state.board.region("fords of isen"),
                                     [unit(gs.Nation.ISENGARD) for _ in range(10)]),
        }
        gl_source = army(state, "orthanc")
        ai._retreat_army(state, "orthanc", "fords of isen", SH)
        self.assertEqual([getattr(l, "name", "") for l in gl_source.leaders], ["Saruman"])
        self.assertTrue(any(isinstance(l, gs.Nazgul) for l in army(state, "fords of isen").leaders))

    def test_retreat_does_not_activate_rohan_or_discard_wormtongue(self):
        state = politics_position()
        after = ai.apply_action(state, retreat_action(state, "edoras"))
        self.assertFalse(after.political.active[gs.Nation.ROHAN])
        self.assertTrue(after.persistent_cards.has_card_type(gs.PersistentCardType.WORMTONGUE))

    def test_ordinary_companion_move_still_activates_rohan(self):
        state = politics_position()
        state.pending_combat = None
        gl.move_army(state, "westemnet", "edoras", side=FP)
        self.assertTrue(state.political.active[gs.Nation.ROHAN])
        self.assertFalse(state.persistent_cards.has_card_type(gs.PersistentCardType.WORMTONGUE))


class CardFollowupTests(unittest.TestCase):
    def move_rows(self, state):
        return [a for a in ai.get_legal_actions(state, SH)
                if a.action_type == "army_move" and a.params["to"] == "south ithilien"]

    def test_every_menu_split_carries_required_figure(self):
        for kind in ("nazgul_armies", "wk_army"):
            with self.subTest(kind=kind):
                state = followup_position(kind)
                rows = self.move_rows(state)
                self.assertTrue(rows)
                for row in rows:
                    moving = selected(army(state, "minas morgul"), SH, row.params)
                    self.assertTrue(any(gl.is_nazgul_figure(l) for l in moving.leaders))
                    after = ai.apply_action(state, row)
                    self.assertTrue(any(gl.is_nazgul_figure(l)
                                        for l in army(after, "south ithilien").leaders))

    def test_rwa_can_leave_one_nazgul_but_must_take_another(self):
        state = followup_position("nazgul_armies",
                                  [gs.Nazgul(gs.Nation.SAURON) for _ in range(2)])
        row = next(a for a in self.move_rows(state) if a.params.get("n_nazgul") == 1)
        after = ai.apply_action(state, row)
        self.assertEqual(sum(isinstance(l, gs.Nazgul) for l in army(after, "minas morgul").leaders), 1)
        self.assertEqual(sum(isinstance(l, gs.Nazgul) for l in army(after, "south ithilien").leaders), 1)

    def test_bcc_plain_nazgul_cannot_substitute_for_witch_king(self):
        state = followup_position("wk_army", [gs.Nazgul(gs.Nation.SAURON),
                                             gl.Minion(gs.Nation.SAURON, "Witch King")])
        for row in self.move_rows(state):
            moving = selected(army(state, "minas morgul"), SH, row.params)
            self.assertIn("Witch King", [getattr(l, "name", "") for l in moving.leaders])

    def test_bcc_count_selects_the_actual_named_minion(self):
        for first in ("Mouth of Sauron", "Witch King"):
            second = "Witch King" if first == "Mouth of Sauron" else "Mouth of Sauron"
            state = followup_position("wk_army", [gl.Minion(gs.Nation.SAURON, first),
                                                 gl.Minion(gs.Nation.SAURON, second)])
            one = [a for a in self.move_rows(state) if a.params.get("n_leaders") == 1]
            self.assertEqual(bool(one), first == "Witch King")
            for row in one:
                after = ai.apply_action(state, row)
                self.assertEqual([l.name for l in army(after, "south ithilien").leaders], ["Witch King"])

    def test_executor_rejects_nonparticipating_figure_without_changing_source(self):
        for kind in ("nazgul_armies", "wk_army"):
            state = followup_position(kind)
            before = pickle.dumps(state, protocol=5)
            action = ai.Action(-1, "army_move", {
                "from": "minas morgul", "to": "south ithilien",
                "n_regular": 1, "n_elite": 0, "take_leaders": False,
                "side": "sh", "card_followup": True})
            with self.assertRaisesRegex(RuntimeError, "participating"):
                ai.apply_action(state, action)
            self.assertEqual(pickle.dumps(state, protocol=5), before)


if __name__ == "__main__":
    unittest.main()
