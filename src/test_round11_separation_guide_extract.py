"""Bounded literal guide gestures, with real card execution and no games."""
import collections
from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("WOTR_NATIVE", "0")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import ai
import board
import event_effects as ee
import game_logic as gl
import game_state as gs
import rendered_extract as rx
import rendered_replay as rr

FP = gs.Side.FREE_PEOPLES
CARDS = ("Gwaihir the Windlord", "We Prove Swifter", "I Will Go Alone", "There and Back Again")
OPEN = ("%<replay> Turn 1\n%SH allocates an eye.\n%SH rolls A A M P H H .\n"
        "%FP rolls P C H M .\n%FP moves Free Action Die (P) from FP Dice to Current Action.\n")
CHANGE = ("%FP moves Gandalf the Grey from Guide Of The Fellowship to Fellowship Of The Ring.\n"
          "%FP moves Strider from Fellowship Of The Ring to Guide Of The Fellowship.\n")


def batch(name, change=False, departure=True):
    text = OPEN + f"%FP moves {name} from Free Peoples Hand to Current FP Card.\n"
    if departure:
        text += "%FP moves Boromir from Fellowship Of The Ring to Rivendell.\n"
    if change:
        text += CHANGE
    text += f"%FP kills {name} in Current FP Card.\n"
    text += "%SH moves Shadow Action Die (A) from SA Dice to Current Action.\n"
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "literal.rendered.txt"
        path.write_text(text, encoding="utf-8")
        return rr.parse_game(path)["batches"][0]


def position(name):
    state = gs.setup_initial_game(board.load_board())
    gl.initialize_decks(state)
    state._declare_phase = False
    state._eye_allocation_phase = False
    state.dice_rolled = True
    for pool in state.dice.values():
        for die in pool:
            die.used = True
    state.dice[FP][0].used = False
    state.dice[FP][0].current = gs.ActionDieFace.EVENT
    state.hands[FP].hand = [ee._card_by_name(name).card_id]
    return state


def stats():
    return {key: collections.Counter() for key in ("detected", "drop_reasons", "kinds_emit")}


def capture(state, side, legal, index, phase, turn):
    return dict(state=state, side=side, chosen=legal[index], phase=phase, legal=legal)


class SeparationGuideExtractionTests(unittest.TestCase):
    def test_four_card_interiors_resolve_retention_and_literal_switch(self):
        for name in CARDS:
            for change in (False, True):
                with self.subTest(card=name, change=change):
                    literal, counters = batch(name, change), stats()
                    applied = []
                    real_apply = ai.apply_action

                    def apply(state, action):
                        result = real_apply(state, action)
                        applied.append((action.action_type, result))
                        return result

                    with patch.object(rx, "_mk_state_with_die", return_value=(position(name), FP)), \
                            patch.object(rx, "_emit_sample", side_effect=capture), \
                            patch.object(ai, "apply_action", side_effect=apply), \
                            patch.object(rx, "_LEDGER", None), patch.object(rx, "_TABLE", None):
                        rows = rx._emit_card_interiors(literal, counters)
                    self.assertEqual(len(rows), 1, counters)
                    self.assertEqual(rows[0]["phase"], "card_option")
                    self.assertIn("choose_guide", [kind for kind, _ in applied], counters)
                    final = applied[-1][1]
                    self.assertIsNone(final.pending_guide_change)
                    self.assertEqual(final.fellowship.guide.name,
                                     "Strider" if change else "Gandalf the Grey")
                    # A retained guide's deferred fixed bonus draw is reached
                    # after guide resolution, instead of leaving that tail blocked.
                    if not change:
                        self.assertEqual(counters["detected"]["fixed_policy_bonus_draw"], 1, counters)

    def test_post_separation_switch_has_a_real_guide_row_and_silence_has_none(self):
        for change in (False, True):
            literal, counters = batch("I Will Go Alone", change), stats()
            # Omit the separately owned turn-opening opportunity in this test.
            literal.pop("_guide_phase_start", None)
            with patch.object(rx, "_emit_sample", side_effect=capture), \
                    patch.object(rx, "_LEDGER", None), patch.object(rx, "_TABLE", None):
                rows = rx._emit_guides([literal], counters)
            self.assertEqual(len(rows), int(change), counters)
            if change:
                row = rows[0]
                self.assertEqual(row["chosen"].params["companion"], "Strider")
                self.assertFalse(row["state"]._eye_allocation_phase)
                self.assertEqual(row["guide_evidence"]["kind"], "explicit_post_separation_nomination")
                self.assertNotIn("Boromir", {c.name for c in row["state"].fellowship.companions})

    def test_without_separation_or_with_contradictory_guide_no_label_is_invented(self):
        literal, counters = batch("I Will Go Alone", change=True, departure=False), stats()
        literal.pop("_guide_phase_start", None)
        with patch.object(rx, "_emit_sample", side_effect=capture), \
                patch.object(rx, "_LEDGER", None), patch.object(rx, "_TABLE", None):
            self.assertEqual(rx._emit_guides([literal], counters), [])
        literal = batch("I Will Go Alone")
        state = position("I Will Go Alone")
        state.fellowship.companions = [c for c in state.fellowship.companions if c.name != "Boromir"]
        state.pending_guide_change = {"candidates": ["Gandalf the Grey", "Strider"]}
        legal = ai.get_legal_actions(state, FP)
        literal["truth"]["fel"]["guide"] = "Strider"
        with self.assertRaisesRegex(ValueError, "unproven_post_separation_keep"):
            rx._card_guide_index(state, literal, legal)


if __name__ == "__main__":
    unittest.main()
