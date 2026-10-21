"""Delayed public Hunt icons remain visible without widening stored features."""
import copy
import os
from pathlib import Path
import sys
import unittest
from collections import Counter
from unittest.mock import patch

os.environ.setdefault("WOTR_ACT_GEN", "v52")
os.environ.setdefault("WOTR_NATIVE", "0")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ai
import board
import game_state as gs
import numpy as np
import rendered_extract as rx
import rendered_census as rc

FP, SH = gs.Side.FREE_PEOPLES, gs.Side.SHADOW
REVEAL_SLOT = 3373  # global start3182 + companion start160 + Gollum7*4 + location3


def positions():
    """Public pending contexts independent of the Hunt implementation."""
    base = gs.setup_initial_game(board.load_board())
    base.fellowship.hidden = True
    base._pending_hunt_damage = 2
    cases = []
    for phase in ("hunt", "guide", "card"):
        state = copy.deepcopy(base)
        if phase == "hunt":
            state.pending_hunt = {"phase": "hunt_sacrifice", "damage": 2}
        elif phase == "guide":
            state.pending_guide_change = {"candidates": ["Legolas", "Gimli"]}
        else:
            state.pending_card_option = {
                "card_id": "SHADOW_Character_Foul_Thing_from_the_Deep",
                "side": "fp", "options": [{"foul_gollum_reveal": True},
                                          {"foul_gollum_reveal": False}]}
        for pending in (0, 1, 2):
            child = copy.deepcopy(state)
            child._hunt_reveal_pending = pending
            cases.append((phase, pending, child))
    return cases


class HuntRevealEncoding(unittest.TestCase):
    def test_only_unused_slot_changes_and_both_observers_see_drawn_icon(self):
        baselines = {}
        for phase, pending, state in positions():
            for side in (FP, SH):
                encoded = ai.encode_state(state, side)
                self.assertEqual(encoded.shape, (4647,))
                self.assertEqual(encoded[REVEAL_SLOT], pending / 2.)
                self.assertTrue(state.fellowship.hidden)
                if pending == 0:
                    baselines[phase, side] = encoded
                else:
                    changed = np.flatnonzero(encoded != baselines[phase, side])
                    np.testing.assert_array_equal(changed, [REVEAL_SLOT])

    def test_absent_and_zero_preserve_old_observation(self):
        state = gs.setup_initial_game(board.load_board())
        state.__dict__.pop("_hunt_reveal_pending", None)
        before = {side: ai.encode_state(state, side).tobytes() for side in (FP, SH)}
        before_keys = (ai._state_sig(state), ai._state_sig(state, True), ai._chance_sig(state))
        for inactive in (None, 0):
            state._hunt_reveal_pending = inactive
            for side in (FP, SH):
                self.assertEqual(before[side], ai.encode_state(state, side).tobytes())
            self.assertEqual(before_keys, (ai._state_sig(state), ai._state_sig(state, True),
                                          ai._chance_sig(state)))

    def test_retained_and_chance_search_keys_distinguish_all_three_values(self):
        for phase in ("hunt", "guide", "card"):
            states = [state for name, _, state in positions() if name == phase]
            for sorted_dice in (False, True):
                keys = [ai._state_sig(state, sorted_dice) for state in states]
                self.assertNotIn(None, keys)
                self.assertEqual(len(set(keys)), 3)
            self.assertEqual(len({ai._chance_sig(state) for state in states}), 3)

    def test_actual_teacher_sample_preserves_pending_icon_and_action_width(self):
        state = next(st for phase, pending, st in positions()
                     if phase == "hunt" and pending == 1)
        state._rx_deck_prefix = 10
        menu = ai.get_legal_actions(state, FP)
        index = next(i for i, action in enumerate(menu)
                     if action.action_type == "hunt_take_corruption")
        sample = rx._emit_sample(state, FP, menu, index, "hunt_sacrifice", 1)
        self.assertEqual(sample["x"].shape, (4647,))
        self.assertEqual(sample["x"][REVEAL_SLOT], .5)
        self.assertEqual(sample["encs"].shape[1], 157)
        self.assertEqual(sample["human"], index)
        self.assertNotIn("ai_policy_exclusion", sample)
        self.assertTrue(state.fellowship.hidden)

    def test_teacher_keeps_actual_guide_prestate_after_final_reveal_tile(self):
        state = gs.setup_initial_game(board.load_board())
        state.fellowship.companions = [
            c for c in state.fellowship.companions
            if c.name in ("Gandalf the Grey", "Legolas", "Gimli")]
        state.fellowship.guide = state.fellowship.companions[0]
        state._rx_deck_prefix = 10
        before = dict(comps=["Gandalf the Grey", "Legolas", "Gimli"],
                      guide="Gandalf the Grey", corruption=0, hidden=True,
                      mordor=False, step=0, progress=0, region="rivendell")
        guide_truth = dict(before, comps=["Legolas", "Gimli"], progress=1)
        after = dict(guide_truth, guide="Legolas", hidden=False,
                     guide_provenance="declared")
        nomination = dict(op="move", obj="Legolas", line=45,
                          **{"from": "Fellowship Of The Ring",
                             "to": "Guide Of The Fellowship"},
                          _guide_gesture=True, decision_truth={"fel": guide_truth},
                          guide_truth_after={"fel": after})
        batch = dict(kind="hunt", die_side="FREE_PEOPLES", die_face="C", turn=1,
                     start_line=10, end_line=55,
                     _rx_pools={"FREE_PEOPLES": {"unused": ["C"], "used": []},
                                "SHADOW": {"unused": [], "used": ["E"]}},
                     _rx_fellowship_roll_known=True, _rx_shadow_rolled_eyes=0,
                     truth_before={"fel": before}, truth={"fel": after},
                     rng=[dict(kind="roll_num", ints=[6], line=25),
                          dict(kind="draw_tile", value="2r", line=30)],
                     records=[
                         dict(op="move", obj="FSP", line=20,
                              **{"from": "Fellowship Track 0", "to": "Fellowship Track 1"}),
                         dict(op="move", obj="Gandalf the Grey", line=40,
                              **{"from": "Guide Of The Fellowship", "to": "FP Casualties"}),
                         nomination,
                         dict(op="flip", obj="The Fellowship", line=50)])
        stats = {key: Counter() for key in ("detected", "drop_reasons", "kinds_emit")}
        # Supply the literal initial board; all Hunt/Guide rule transitions
        # and the actual teacher sample encoder remain live.
        with patch.object(rx, "_mk_state_with_die", return_value=(state, FP)), \
                patch.object(rx, "_RX_EYEALLOC", {1: 1}), \
                patch.object(rx, "_RX_EYEROLL", {1: 0}), \
                patch.object(rx, "_FP_HUNTBOX_FACES", []):
            census = copy.deepcopy(state)
            rx._prepare_fellowship_dice(census, batch, pending=True)
            status, diffs, _ = rc._drive_hunt(census, batch)
            self.assertEqual(status, "drove")
            self.assertTrue(census.fellowship.hidden)
            self.assertEqual(census._hunt_reveal_pending, 1)
            self.assertNotIn("fel.hidden", [entry["field"] for entry in diffs])
            samples = rx._emit_hunt_resolution(batch, stats)
        self.assertEqual(len(samples), 1, stats)
        self.assertEqual(samples[0]["x"][REVEAL_SLOT], .5)
        captured = batch.get("_rx_hunt_guide_predecision", {})
        self.assertIn(45, captured, stats)
        pending = captured[45]
        self.assertTrue(pending.fellowship.hidden)
        self.assertEqual(pending._hunt_reveal_pending, 1)
        self.assertIsNotNone(pending.pending_guide_change)
        with patch.object(rx, "_recon_state", side_effect=AssertionError("must use driven Hunt prestate")):
            guides = rx._emit_guides([batch], stats)
        self.assertEqual(len(guides), 1, stats)
        self.assertEqual(guides[0]["x"][REVEAL_SLOT], .5)


if __name__ == "__main__":
    unittest.main()
