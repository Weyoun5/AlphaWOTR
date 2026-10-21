"""Rulebook pp43-44: phase-5 inactivity precedes phase-6 military victory."""
import copy
import os
from pathlib import Path
import pickle
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import ai
import board
import game_logic as gl
import game_state as gs

FP, SH = gs.Side.FREE_PEOPLES, gs.Side.SHADOW


def fixture(owner=FP, corruption=11, active=False):
    s = gs.setup_initial_game(board.load_board())
    s.turn = 8
    s.dice_rolled = True
    s.armies = {}
    gl.initialize_decks(s)
    for hand in s.hands.values():
        hand.hand = []
        hand.needs_discard = False
    for dice in s.dice.values():
        for die in dice:
            die.current = gs.ActionDieFace.CHARACTER
            die.used = True
    s.hunt_box = []
    s.fp_elven_rings = s.sh_elven_rings = 0
    s.fellowship.region = s.board.region("minas morgul")
    s.fellowship.mordor_entered = True
    s.fellowship.mordor_step = 3
    s.fellowship.corruption = corruption
    s.fellowship_acted_in_mordor = active
    targets = ({"morannon", "minas morgul"} if owner == FP else
               {"minas tirith", "dol amroth", "erebor", "lorien", "rivendell"}
               if owner == SH else set())
    s.enemy_controlled = {FP: targets if owner == FP else set(),
                          SH: targets if owner == SH else set()}
    nation = gs.Nation.GONDOR if owner == FP else gs.Nation.SAURON
    for target in targets:
        s.armies[target] = gs.Army(s.board.region(target),
                                  [gs.Unit(nation, gs.UnitKind.REGULAR)], [])
    for nation in gs.Nation:
        s.political.boxes[nation] = 3
        s.political.active[nation] = True
    return s


class InactivityOrderingTests(unittest.TestCase):
    def assert_terminal(self, s, owner, corruption, active):
        want_corruption = corruption + (not active)
        expected = ("shadow", "ring_corruption") if want_corruption >= 12 else (
            "free_peoples" if owner == FP else "shadow", "military")
        self.assertEqual((s._search_endturn_victory["victory"],
                          s._search_endturn_victory["victory_type"]), expected)
        self.assertEqual(s.fellowship.corruption, want_corruption)
        self.assertEqual(s.turn, 8)
        self.assertTrue(s.dice_rolled)
        self.assertTrue(all(d.used for pool in s.dice.values() for d in pool))
        self.assertEqual(s.hands[FP].hand, [])
        self.assertEqual(s.hands[SH].hand, [])

    def test_public_search_end_turn_literal_priority_matrix(self):
        for owner in (FP, SH):
            for corruption, active in ((11, False), (11, True), (10, False)):
                with self.subTest(owner=owner, corruption=corruption, active=active):
                    s = fixture(owner, corruption, active)
                    self.assertEqual(ai._turn_end_blockers(s), [])
                    child = ai.apply_action(s, ai.Action(-1, "end_turn", {}))
                    self.assert_terminal(child, owner, corruption, active)
                    self.assertEqual(s.fellowship.corruption, corruption)

    def test_direct_reset_and_next_turn_terminal_priority(self):
        for boundary in (lambda s: gl.begin_turn_reset(s, 9, deal_cards=True),
                         gl.next_turn, ai._safe_next_turn):
            for owner in (FP, SH):
                for corruption, active in ((11, False), (11, True), (10, False)):
                    with self.subTest(boundary=boundary, owner=owner,
                                      corruption=corruption, active=active):
                        s = fixture(owner, corruption, active)
                        self.assertTrue(boundary(s))
                        self.assert_terminal(s, owner, corruption, active)
                        after = pickle.dumps(s, protocol=5)
                        self.assertTrue(boundary(s))
                        self.assertEqual(pickle.dumps(s, protocol=5), after)

    def test_terminal_marker_survives_public_search_clone(self):
        s = ai.apply_action(fixture(corruption=10), ai.Action(-1, "end_turn", {}))
        again = ai.apply_action(s, ai.Action(-1, "end_turn", {}))
        self.assert_terminal(again, FP, 10, False)
        self.assertEqual(again._search_endturn_victory, s._search_endturn_victory)
        self.assertIsNot(again._search_endturn_victory, s._search_endturn_victory)

    def test_normal_turn_advances_once_without_double_penalty(self):
        for boundary in (lambda s: gl.begin_turn_reset(s, 9),
                         gl.next_turn, ai._safe_next_turn):
            s = fixture(owner=None, corruption=10)
            with self.subTest(boundary=boundary):
                self.assertFalse(boundary(s))
                self.assertEqual((s.turn, s.fellowship.corruption), (9, 11))
                self.assertFalse(s.dice_rolled)
                self.assertFalse(any(d.used for pool in s.dice.values() for d in pool))
                self.assertEqual(len(s.hands[FP].hand), 2)
                self.assertEqual(len(s.hands[SH].hand), 2)

    def test_finished_ring_destruction_is_not_reversed_by_inactivity(self):
        for boundary in (gl.next_turn, ai._safe_next_turn):
            s = fixture(corruption=11)
            s.fellowship.mordor_step = 5
            self.assertTrue(boundary(s))
            self.assertEqual(s.fellowship.corruption, 11)
            self.assertEqual(s._search_endturn_victory["victory_type"], "ring_destroyed")

    def test_existing_corruption_terminal_does_not_reset(self):
        s = fixture(corruption=12)
        self.assertTrue(ai._safe_next_turn(s))
        self.assertEqual((s.turn, s.fellowship.corruption), (8, 12))
        self.assertEqual(s._search_endturn_victory["victory_type"], "ring_corruption")

    def test_unspent_die_and_pending_obligation_reject_before_settlement(self):
        for field, value in (("pending_guide_change", {"candidates": ["Strider"]}),
                             ("_pending_hunt_damage", 1), ("unused", None)):
            s = fixture()
            if field == "unused":
                s.dice[FP][0].used = False
            else:
                setattr(s, field, value)
            before = pickle.dumps(s, protocol=5)
            with self.assertRaisesRegex(RuntimeError, "unfinished_turn"):
                ai._safe_next_turn(s)
            self.assertEqual(before, pickle.dumps(s, protocol=5))
            with self.assertRaisesRegex(RuntimeError, "unfinished_turn"):
                ai.apply_action(s, ai.Action(-1, "end_turn", {}))
            self.assertEqual(before, pickle.dumps(s, protocol=5))

    def test_environment_terminal_precedes_turn_and_transition_caps(self):
        from selfplay_actor_critic.environment import PythonEnvironment
        for cap in ("normal", "turn", "transition"):
            s = fixture()
            env = PythonEnvironment(ai, "inactivity-order", 17, state=s,
                                    max_turns=1 if cap == "turn" else 2,
                                    max_transitions=1)
            if cap == "transition":
                env.committed = 1
            self.assertIsNone(env.prepare())
            self.assertEqual((env.end.status, env.end.winner, env.end.victory_route),
                             ("terminated", "sh", "ring_corruption"))
            self.assert_terminal(s, FP, 11, False)


if __name__ == "__main__":
    unittest.main()
