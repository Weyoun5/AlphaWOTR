"""Retained-tree and public chance identity at paid-action boundaries."""
import copy
import unittest

import ai
import board
import game_state as gs

FP, SH = gs.Side.FREE_PEOPLES, gs.Side.SHADOW


def ready(regions=None, nation=gs.Nation.GONDOR, face=gs.ActionDieFace.MUSTER):
    state = gs.setup_initial_game(board.load_board())
    if regions is not None:
        state.armies = {r: gs.Army(state.board.region(r), [
            gs.Unit(nation, gs.UnitKind.REGULAR) for _ in range(n)])
            for r, n in regions.items()}
    for n in gs.Nation:
        state.political.boxes[n] = 3
        state.political.active[n] = True
    state.dice_rolled = True
    for pool in state.dice.values():
        for die in pool:
            die.current = face
    return state


def muster_histories():
    states = []
    for first in ("minas tirith", "pelargir"):
        state = ready({r: int(r != first) for r in ("minas tirith", "pelargir")})
        action = next(a for a in ai.get_legal_actions(state, FP)
            if a.action_type == "muster_recruit" and
            board.sanitize(a.params["region"]) == first and
            a.params["type"] == "regular")
        states.append(ai.apply_action(state, action))
    return states


def army_histories():
    states = []
    for origin, target in (("minas morgul", "north ithilien"),
                           ("north ithilien", "minas morgul")):
        state = ready({origin: 2, target: 0}, gs.Nation.SAURON, gs.ActionDieFace.ARMY)
        action = next(a for a in ai.get_legal_actions(state, SH)
            if a.action_type == "army_move" and
            board.sanitize(a.params["from"]) == origin and
            board.sanitize(a.params["to"]) == target and
            a.params.get("n_regular") == 1 and a.params.get("n_elite") == 0)
        states.append(ai.apply_action(state, action))
    return states


def decision_cases():
    """Each entry changes a pending value without changing board or hand."""
    base = ready()
    cases = [("base", base)]
    values = {
        "pending_army_move_die": 1, "pending_army_move_from": "minas tirith",
        "pending_army_move_side": FP,
        "pending_army_move_preexist": {"regular": 1, "figures": {("unit", "GONDOR", "REGULAR"): 1}},
        "pending_muster_die": 1, "pending_muster_first_type": "regular",
        "pending_muster_region": "minas tirith", "pending_muster_side": FP,
        "pending_combat": {"phase": "defender_retreat", "defender_region": "minas tirith"},
        "pending_hunt": {"phase": "hunt_sacrifice", "damage": 2, "casualty_taken": True},
        "pending_guide_change": {"candidates": ["Strider"]},
        "pending_card_option": {"card_id": "__companion_moves__", "direct": True,
                                "options": [{"_comp_moved": ["Legolas"]}]},
        "pending_discard": {"side": SH}, "_pending_hunt_damage": 2,
        "_hobbit_place_queue": [next(c for c in base.fellowship.companions if c.name == "Meriadoc")],
        "_wwsat_owed": [True], "_overstack_regions": {"minas tirith"},
        "_palantir_draw_pending": True, "_owed_draws": {(FP.name, "character"): 1},
        "_battle_resolved_attacker": FP, "_crebain_hunt_bonus": 1, "_ents_chain_n": 2,
        "_wizardstaff_stash": {"apply_kind": "stronghold", "region": "moria", "reason": "reveal"},
    }
    for key, value in values.items():
        state = copy.deepcopy(base)
        setattr(state, key, value)
        cases.append((key, state))
    for name, alias in (("shared_alias", True), ("separate_lists", False)):
        state = copy.deepcopy(base)
        shared = [1, 2]
        payload = {"one": shared, "two": shared if alias else list(shared)}
        payload["cycle"] = payload
        state.pending_card_option = {"card_id": "test", "options": [payload]}
        cases.append((name, state))
    for cid in ("SHADOW_Character_One", "SHADOW_Character_Two"):
        state = copy.deepcopy(base)
        state.pending_combat = {"phase": "defender_combat_card", "attacker_side": SH,
                                "atk_combat_card": cid}
        cases.append((cid, state))
    return cases


class SearchPendingContext(unittest.TestCase):
    def check_histories(self, states, side):
        first, second = states
        census = lambda s: sorted((r, sorted((u.nation.name, u.kind.name)
            for u in a.units)) for r, a in s.armies.items())
        self.assertEqual(census(first), census(second))
        self.assertNotEqual(sorted(map(repr, ai.get_legal_actions(first, side))),
                            sorted(map(repr, ai.get_legal_actions(second, side))))
        for sorted_dice in (False, True):
            left, right = (ai._state_sig(s, sorted_dice) for s in states)
            self.assertIsNotNone(left)
            self.assertIsNotNone(right)
            self.assertNotEqual(left, right)

    def test_generated_muster_histories(self):
        self.check_histories(muster_histories(), FP)

    def test_generated_army_histories(self):
        self.check_histories(army_histories(), SH)

    def test_complete_pending_context_fields(self):
        base = ai._state_sig(ready())
        for name, state in decision_cases()[1:-4]:
            with self.subTest(name=name):
                self.assertIsNotNone(ai._state_sig(state))
                self.assertNotEqual(base, ai._state_sig(state))

    def test_aliases_cycles_and_none_remain_distinct(self):
        cases = dict(decision_cases())
        self.assertNotEqual(ai._state_sig(cases["shared_alias"]),
                            ai._state_sig(cases["separate_lists"]))
        empty = ready()
        empty.pending_card_option = {}
        self.assertNotEqual(ai._state_sig(ready()), ai._state_sig(empty))
        self.assertEqual(ai._state_sig(cases["shared_alias"]),
                         ai._state_sig(copy.deepcopy(cases["shared_alias"])))

    def test_equal_presence_different_public_obligations(self):
        for key, left, right in (
            ("pending_hunt", {"phase": "hunt_sacrifice", "damage": 1},
             {"phase": "hunt_sacrifice", "damage": 2}),
            ("pending_guide_change", {"candidates": ["Strider"]},
             {"candidates": ["Gandalf the Grey"]}),
            ("pending_discard", {"side": FP}, {"side": SH}),
            ("pending_combat", {"phase": "defender_retreat", "defender_region": "minas tirith"},
             {"phase": "defender_retreat", "defender_region": "pelargir"}),
            ("pending_card_option", {"card_id": "public", "options": [{"_comp_moved": ["Legolas"]}]},
             {"card_id": "public", "options": [{"_comp_moved": ["Gimli"]}]}),
        ):
            with self.subTest(field=key):
                first, second = ready(), ready()
                setattr(first, key, left)
                setattr(second, key, right)
                self.assertNotEqual(ai._state_sig(first), ai._state_sig(second))

    def test_unrevealed_commitment_reuse_fails_closed(self):
        for name, state in decision_cases()[-2:]:
            with self.subTest(name=name):
                self.assertIsNone(ai._state_sig(state))
                self.assertIsNotNone(ai._state_sig(state, True))
        for field in ("atk_hidden_cc", "def_hidden_cc"):
            state = ready()
            state.pending_combat = {field: {"deck": "character"}}
            self.assertIsNone(ai._state_sig(state))

    def test_private_hands_and_chance_menu_projection_remain_blind(self):
        first = ready()
        second = copy.deepcopy(first)
        first.hands[SH].hand = ["SHADOW_Character_One"]
        second.hands[SH].hand = ["SHADOW_Character_Two"]
        self.assertEqual(ai._state_sig(first), ai._state_sig(second))
        old_root = ai._SEARCH_ROOT_SIDE
        ai._SEARCH_ROOT_SIDE = FP
        try:
            self.assertEqual(ai._chance_sig(first), ai._chance_sig(second))
            first.pending_card_option = {"card_id": "public", "options": [{"card_id": "One"}]}
            second.pending_card_option = {"card_id": "public", "options": [{"card_id": "Two"}]}
            # Retention sees the acting player's exact choices, whereas chance
            # identity retains its prior root-relative hidden-menu projection.
            self.assertNotEqual(ai._state_sig(first), ai._state_sig(second))
            self.assertEqual(ai._chance_sig(first), ai._chance_sig(second))
        finally:
            ai._SEARCH_ROOT_SIDE = old_root


if __name__ == "__main__":
    unittest.main()
