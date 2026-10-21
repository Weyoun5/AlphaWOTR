"""Public FP downgrade source proofs, using literal native piece/zone IDs.

Run directly with Python; no model, Java process, or live state is used.
"""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "wotr_bridge"))

import board
import game_state as gs
from live import emission


class ForcedDowngradeSources(unittest.TestCase):
    def fixture(self, sources=(("rohan", "cas"),), regular_count=1):
        state = gs.setup_initial_game(board.load_board())
        army = gs.Army(state.board.region("Folde"))
        army.units = [gs.Unit(gs.Nation.ROHAN, gs.UnitKind.ELITE)
                      for _ in range(2)]
        state.armies = {"folde": army}
        view = SimpleNamespace(
            piece_region={"288": "Folde", "289": "Folde",
                          "290": "FP Casualties", "291": "FP Reinf"},
            piece_types=None, state=state)
        rows = {
            "units": {
                "rohan:elite": {"minus": [("folde", 1)], "plus": []},
                "rohan:regular": {"minus": [],
                                  "plus": [("folde", regular_count)]}},
            "naz": {"minus": [], "plus": []},
            "named": {"leaves": {}, "joins": {}},
            "cas": {"rohan:elite": 1}, "unexplained": [],
            "downgrade_sources": list(sources)}
        return state, view, rows

    def render(self, state, view, rows):
        result, _ = emission.render_forced_rows(
            view, "WOTR_AI_FP", gs.Side.FREE_PEOPLES, rows, state)
        return result

    @staticmethod
    def moves(result):
        return [fields for operation, fields in result.ops
                if operation == "move"]

    def test_cas_source_not_available_pool(self):
        state, view, rows = self.fixture()
        result = self.render(state, view, rows)
        self.assertFalse(result.hold)
        move = self.moves(result)[0]
        self.assertEqual((move["pieces"], move["frm"], move["to"]),
                         (["290"], "Game.2039", "Game.1933"))

    def test_pool_source_does_not_resurrect_cas(self):
        state, view, rows = self.fixture((("rohan", "pool"),))
        result = self.render(state, view, rows)
        self.assertFalse(result.hold)
        move = self.moves(result)[0]
        self.assertEqual((move["pieces"], move["frm"]),
                         (["291"], "Game.2038"))

    def test_missing_cas_never_rescued_from_pool(self):
        state, view, rows = self.fixture()
        view.piece_region["290"] = "FP Reinf"
        self.assertTrue(self.render(state, view, rows).hold)

    def test_none_source_holds(self):
        state, view, rows = self.fixture((("rohan", None),))
        self.assertTrue(self.render(state, view, rows).hold)

    def test_pool_shortage_never_rescued_from_cas(self):
        state, view, rows = self.fixture((("rohan", "pool"),))
        view.piece_region["291"] = "FP Casualties"
        self.assertTrue(self.render(state, view, rows).hold)

    def test_sources_consumed_once(self):
        state, view, rows = self.fixture(
            (("rohan", "cas"), ("rohan", "cas")), regular_count=2)
        self.assertTrue(self.render(state, view, rows).hold)

    def test_ordinary_recruit_keeps_pool_policy(self):
        state, view, rows = self.fixture(())
        result = self.render(state, view, rows)
        self.assertFalse(result.hold)
        self.assertEqual(self.moves(result)[0]["frm"], "Game.2038")

    def test_mixed_sources_are_each_consumed_once(self):
        state, view, rows = self.fixture(
            (("rohan", "cas"), ("rohan", "pool")), regular_count=2)
        result = self.render(state, view, rows)
        self.assertFalse(result.hold)
        self.assertEqual([(move["pieces"], move["frm"])
                          for move in self.moves(result)],
                         [(["290"], "Game.2039"),
                          (["291"], "Game.2038")])

    def test_cas_source_keeps_besieged_garrison_destination(self):
        state, view, rows = self.fixture()
        state.besieged_strongholds = {"helm's deep"}
        rows["units"] = {"rohan:regular": {
            "minus": [], "plus": [("helm's deep", 1)]}}
        rows["cas"] = {}
        result = self.render(state, view, rows)
        self.assertFalse(result.hold)
        move = self.moves(result)[0]
        self.assertEqual((move["frm"], move["to"]),
                         ("Game.2039", "Game.1993"))

    def test_unknown_source_holds(self):
        state, view, rows = self.fixture((("rohan", "invented"),))
        self.assertTrue(self.render(state, view, rows).hold)

    def test_source_proofs_consumed_across_distinct_arrivals(self):
        state, view, rows = self.fixture(
            (("rohan", "cas"), ("rohan", "pool")), regular_count=2)
        rows["units"]["rohan:regular"]["plus"] = [
            ("folde", 1), ("westemnet", 1)]
        result = self.render(state, view, rows)
        self.assertFalse(result.hold)
        self.assertEqual([(move["pieces"], move["frm"])
                          for move in self.moves(result)],
                         [(["290"], "Game.2039"),
                          (["291"], "Game.2038")])


if __name__ == "__main__":
    unittest.main()
