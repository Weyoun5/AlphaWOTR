"""Public action semantics and observer-safe search identities."""
import copy
import unittest

import ai
import event_effects
import game_logic as gl
import game_state as gs
import persistent_effects
from test_round6_search_pending_context import ready, FP, SH


def guide_histories():
    state = ready(face=gs.ActionDieFace.MUSTER)
    state.fellowship.hidden = False
    gl.change_guide(state, allow_pending=True)
    return [ai.apply_action(state, action)
            for action in ai.get_legal_actions(state, FP)]


def public_cases():
    """Explicit observable deltas used by both source and native verification."""
    base = ready()
    cases = [("base", base)]
    for key in (
        "fp_dwarven_rings", "sh_dwarven_rings", "fp_political_token",
        "fp_card_token", "sh_action_tokens", "fp_token_used_this_turn",
        "sh_token_used_this_turn", "sh_mouth_muster_used_this_turn",
        "eyes_allocated", "fellowship_acted_in_mordor",
        "_fp_used_hunt_box_last_turn", "game_over",
    ):
        child = copy.deepcopy(base)
        setattr(child, key, True if key not in (
            "fp_dwarven_rings", "sh_dwarven_rings", "sh_action_tokens",
            "eyes_allocated") else 1)
        cases.append((key, child))
    child = copy.deepcopy(base)
    child.dice_rolled = False
    cases.append(("dice_rolled", child))
    for key in ("witch_king", "saruman", "mouth_of_sauron"):
        for field in ("in_play", "was_ever_in_play"):
            child = copy.deepcopy(base)
            setattr(getattr(child, key), field, True)
            cases.append((key + "." + field, child))
    for key, field in (("gandalf_the_white", "in_play"),
                       ("aragorn_crowned", "crowned")):
        child = copy.deepcopy(base)
        setattr(getattr(child, key), field, True)
        cases.append((key + "." + field, child))
    for key in ("hunt_pool", "hunt_discard", "hunt_removed"):
        child = copy.deepcopy(base)
        getattr(child, key).append(gs.HuntTile(1))
        cases.append((key, child))
    for key in ("fp_char_deck", "fp_strat_deck", "sh_char_deck", "sh_strat_deck"):
        child = copy.deepcopy(base)
        setattr(child, key, ["unseen"])
        cases.append((key, child))
    for side in (FP, SH):
        child = copy.deepcopy(base)
        child.hands[side].hand = ["HU_side_CHAR_1"]
        cases.append((side.name + "_hand_type", child))
        child = copy.deepcopy(base)
        child.hands[side].discarded_face_down = ["unseen"]
        cases.append((side.name + "_discard_count", child))
    for label, update in (
        ("control", lambda s: s.enemy_controlled[SH].add("minas tirith")),
        ("siege", lambda s: s.besieged_strongholds.add("minas tirith")),
        ("winner", lambda s: setattr(s, "winner", "FP")),
        ("endturn_victory", lambda s: setattr(s, "_search_endturn_victory",
                                             {"victory": "free_peoples"})),
        ("public_discard", lambda s: s.discard_face_up["fp_character"].append(
            "FP_Character_Athelas")),
        ("table_card", lambda s: s.persistent_cards.add_card(gs.PersistentCard(
            "FP_Strategy_A_Power_too_Great", "A Power too Great", FP,
            gs.PersistentCardType.POWER_TOO_GREAT))),
        ("guide", lambda s: setattr(s.fellowship, "guide",
            next(c for c in s.fellowship.companions if c.name == "Strider"))),
    ):
        child = copy.deepcopy(base)
        update(child)
        cases.append((label, child))
    for side in (FP, SH):
        child = copy.deepcopy(base)
        die = gs.ActionDie(gs.FP_FACES if side is FP else gs.SH_FACES, side)
        die.current, die.used = gs.ActionDieFace.EYE, True
        child.hunt_box = [die]
        cases.append(("hunt_box_" + side.name, child))
    return cases


class SearchPublicContext(unittest.TestCase):
    def different(self, first, second):
        for sort_dice in (False, True):
            left, right = ai._state_sig(first, sort_dice), ai._state_sig(second, sort_dice)
            self.assertIsNotNone(left)
            self.assertIsNotNone(right)
            self.assertNotEqual(left, right)

    def test_generated_guide_choices_have_different_legal_actions(self):
        first, second = guide_histories()
        self.assertEqual([first.fellowship.guide.name, second.fellowship.guide.name],
                         ["Gandalf the Grey", "Strider"])
        hide = lambda s: [a for a in ai.get_legal_actions(s, FP)
                          if a.action_type == "fellowship_hide"]
        self.assertFalse(hide(first))
        self.assertTrue(hide(second))
        self.different(first, second)

    def test_control_and_table_cards_change_rule_permissions(self):
        first = ready()
        second = copy.deepcopy(first)
        second.enemy_controlled[SH].add("minas tirith")
        self.assertNotEqual(gl.calculate_victory_points(first),
                            gl.calculate_victory_points(second))
        self.different(first, second)
        second = copy.deepcopy(first)
        second.persistent_cards.add_card(gs.PersistentCard(
            "FP_Strategy_A_Power_too_Great", "A Power too Great", FP,
            gs.PersistentCardType.POWER_TOO_GREAT))
        self.assertTrue(persistent_effects.can_shadow_move_to(first, "rivendell")[0])
        self.assertFalse(persistent_effects.can_shadow_move_to(second, "rivendell")[0])
        self.different(first, second)

    def test_minion_history_changes_gandalf_eligibility(self):
        first = ready(face=gs.ActionDieFace.WILL_OF_WEST)
        first.fellowship.companions = [c for c in first.fellowship.companions
                                      if c.name != "Gandalf the Grey"]
        gl.change_guide(first, chosen="Strider")
        for key in ("witch_king", "saruman", "mouth_of_sauron"):
            second = copy.deepcopy(first)
            getattr(second, key).was_ever_in_play = True
            self.assertFalse(gl.can_transform_gandalf_white(first)[0])
            self.assertTrue(gl.can_transform_gandalf_white(second)[0])
            self.different(first, second)

    def test_tokens_rings_and_mouth_permissions(self):
        for side, field in ((FP, "fp_dwarven_rings"), (SH, "sh_dwarven_rings")):
            first, second = ready(), ready()
            setattr(second, field, 1)
            self.assertFalse(gl.can_use_dwarven_ring(first, side))
            self.assertTrue(gl.can_use_dwarven_ring(second, side))
            self.different(first, second)
            for used in ("fp_token_used_this_turn", "sh_token_used_this_turn"):
                spent = copy.deepcopy(second)
                setattr(spent, used, True)
                self.assertFalse(gl.can_use_dwarven_ring(spent, side))
                self.different(second, spent)
        for side, field, kind in ((FP, "fp_card_token", "draw"),
                                  (FP, "fp_political_token", "muster"),
                                  (SH, "sh_action_tokens", "draw")):
            first, second = ready(), ready()
            setattr(second, field, 1)
            self.assertFalse(gl.has_action_token(first, side, kind))
            self.assertTrue(gl.has_action_token(second, side, kind))
            self.different(first, second)
        first = ready(face=gs.ActionDieFace.MUSTER)
        first.mouth_of_sauron.in_play = True
        first.mouth_of_sauron.was_ever_in_play = True
        second = copy.deepcopy(first)
        second.sh_mouth_muster_used_this_turn = True
        stamped = lambda s: [a for a in ai.get_legal_actions(s, SH)
                             if a.params.get("mouth_muster")]
        self.assertTrue(stamped(first))
        self.assertFalse(stamped(second))
        self.different(first, second)

    def test_mordor_activity_changes_real_turn_penalty(self):
        first = ready()
        first.fellowship.mordor_entered = True
        first.fellowship.corruption = 11
        first.fellowship.hidden = False
        second = copy.deepcopy(first)
        gl.hide_fellowship(second)
        # Compare equal hidden status, retaining the action's activity marker.
        first.fellowship.hidden = second.fellowship.hidden
        self.different(first, second)
        self.assertTrue(gl.begin_turn_reset(first, 2, deal_cards=False, bonus_dice=False))
        self.assertFalse(gl.begin_turn_reset(second, 2, deal_cards=False, bonus_dice=False))
        self.assertEqual((first.fellowship.corruption, second.fellowship.corruption), (12, 11))

    def test_prior_hunt_changes_generated_eye_menu(self):
        first = ready()
        first._eye_allocation_phase = True
        second = copy.deepcopy(first)
        second._fp_used_hunt_box_last_turn = True
        self.assertEqual(ai.get_legal_actions(first, SH)[0].params["count"], 0)
        self.assertEqual(ai.get_legal_actions(second, SH)[0].params["count"], 1)
        self.different(first, second)

    def test_public_hunt_zones_and_deck_counts_change_future_draws(self):
        first = ready()
        second = copy.deepcopy(first)
        event_effects.stage_special_hunt_tile(second, "Elven Rope")
        self.assertEqual(len(second.hunt_discard), len(first.hunt_discard) + 1)
        self.different(first, second)
        box_cases = dict(public_cases())
        fp_box, sh_box = (box_cases["hunt_box_" + side.name] for side in (FP, SH))
        self.assertEqual(gl.get_hunt_params(fp_box)["num_dice"], 0)
        self.assertEqual(gl.get_hunt_params(sh_box)["num_dice"], 1)
        self.different(fp_box, sh_box)
        first.fp_char_deck = []
        second = copy.deepcopy(first)
        second.fp_char_deck = ["FP_Character_Athelas"]
        self.assertFalse([a for a in ai.get_legal_actions(first, FP)
                          if a.action_type == "draw_card" and a.params["deck"] == "character"])
        # An Event die funds the explicit draw surface.
        for state in (first, second):
            state.dice[FP][0].current = gs.ActionDieFace.EVENT
        self.assertTrue([a for a in ai.get_legal_actions(second, FP)
                         if a.action_type == "draw_card" and a.params["deck"] == "character"])
        self.different(first, second)

    def test_all_public_projection_groups_are_distinct(self):
        cases = public_cases()
        for label, child in cases[1:]:
            with self.subTest(label=label):
                self.different(cases[0][1], child)
        self.different(dict(cases)["hunt_box_FREE_PEOPLES"],
                       dict(cases)["hunt_box_SHADOW"])

    def test_unseen_identities_and_orders_do_not_split_public_keys(self):
        first = ready()
        first.sh_char_deck = ["SHADOW_Character_One", "SHADOW_Character_Two"]
        first.hands[SH].hand = ["SHADOW_Character_Three"]
        first.hands[SH].discarded_face_down = ["SHADOW_Strategy_One"]
        second = copy.deepcopy(first)
        second.sh_char_deck = ["SHADOW_Character_Other", "SHADOW_Character_Unknown"]
        second.hands[SH].hand = ["SHADOW_Character_Four"]
        second.hands[SH].discarded_face_down = ["SHADOW_Strategy_Two"]
        second.hunt_pool.reverse()
        for sort_dice in (False, True):
            self.assertEqual(ai._state_sig(first, sort_dice),
                             ai._state_sig(second, sort_dice))
        old = ai._SEARCH_ROOT_SIDE
        try:
            ai._SEARCH_ROOT_SIDE = FP
            self.assertEqual(ai._chance_sig(first), ai._chance_sig(second))
        finally:
            ai._SEARCH_ROOT_SIDE = old

    def test_unsupported_public_context_refuses_reuse(self):
        state = ready()
        state.persistent_cards = b"unsupported public payload"
        self.assertIsNone(ai._state_sig(state))
        self.assertIsNone(ai._state_sig(state, True))


if __name__ == "__main__":
    unittest.main()
