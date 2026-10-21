"""Literal shortage placements through actual card-interior extraction; no games."""
from collections import Counter
import os
from pathlib import Path
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
from location_codes_authoritative import AUTH_LOCATION_CODES
from piece_table_authoritative import AUTH_ARMY_PIECES

FP, SH = gs.Side.FREE_PEOPLES, gs.Side.SHADOW
CASES = (
    ("Orcs Multiplying Again", gs.Nation.SAURON, "Sauron",
     (("Dol Guldur", 1), ("Mount Gundabad", 1)), ()),
    ("Monsters Roused", gs.Nation.SAURON, "Sauron",
     (("Angmar", 1), ("Weather Hills", 1)), (("Trollshaws", 1),)),
    ("A New Power is Rising", gs.Nation.ISENGARD, "Isengard",
     (("North Dunland", 1), ("South Dunland", 1)), (("Orthanc", 2),)),
    ("Many Kings to the Service of Mordor", gs.Nation.SOUTHRON_EASTERLINGS, "Southron",
     (("Far Harad", 1), ("South Rhun", 1)), ()),
    ("Pits of Mordor", gs.Nation.SAURON, "Sauron",
     (("Barad-Dur", 1), ("Minas Morgul", 1)), ()),
)


def position(name, nation, reserve=2):
    state = gs.setup_initial_game(board.load_board())
    gl.initialize_decks(state)
    state.dice_rolled = True
    state._declare_phase = state._eye_allocation_phase = False
    for pool in state.dice.values():
        for die in pool:
            die.current = gs.ActionDieFace.EVENT
    for n in gs.Nation:
        state.political.active[n], state.political.boxes[n] = True, 3
    state.hands[FP].hand = []
    state.hands[SH].hand = [ee._card_by_name(name).card_id]
    for key in ("fp_char_deck", "fp_strat_deck", "sh_char_deck", "sh_strat_deck"):
        setattr(state, key, [])
    state.armies = {}
    remaining = ai.NATION_MAX_FIGURES[nation]["regular"] - reserve
    for region in ("gorgoroth", "nurn", "dagorlad", "south ithilien"):
        count = min(9, remaining)
        if count:
            state.armies[region] = gs.Army(state.board.region(region),
                [gs.Unit(nation, gs.UnitKind.REGULAR) for _ in range(count)], [])
            remaining -= count
    assert remaining == 0
    if name == "A New Power is Rising":
        state.saruman.in_play = state.saruman.was_ever_in_play = True
        gl._add_minion_to_region(state, state.saruman, "orthanc")
    return state


def literal(case, *, complete=True, recruits=None, elite_recruits=None, correction=False):
    name, nation, label, regulars, elites = case
    regulars = regulars if recruits is None else recruits
    elites = elites if elite_recruits is None else elite_recruits
    text = ("%<replay> Turn 1\n%SH allocates an eye.\n%SH rolls P A M P H H .\n"
            "%FP rolls P C H M .\n%SH moves Shadow Action Die (P) from SA Dice to Current Action.\n"
            f"%SH moves {name} from Shadow Hand to Current SA Card.\n")
    def raw_move(ids, origin, target):
        return f"$SH<~Controls.271~>{' '.join(ids)} <~Controls.272~><~{origin}~><~Controls.273~><~{target}~>"

    raw = ["$SH<~Controls.345~>099 <~Controls.346~>1",
           "$FP<~Controls.345~>210 <~Controls.346~>1",
           raw_move(["099"], "Game.2043", "Game.2035")]
    used = set()
    for kind, placements in (("Regular", regulars), ("Elite", elites)):
        for region, count in placements:
            text += f"%SH moves group ([{count}]{label} {kind}) from SA Reinforcements to {region}.\n"
            ids = [pid for pid, piece in AUTH_ARMY_PIECES.items()
                   if piece == (nation.name, kind.upper()) and pid not in used][:count]
            assert len(ids) == count
            used.update(ids)
            target = next(code for code, display in AUTH_LOCATION_CODES.items()
                          if rr.zone_kind(display) == ("region", board.sanitize(region)))
            raw.append(raw_move(ids, "Game.2040", target))
    if correction:
        text += f"%SH moves {label} Regular from {regulars[0][0]} to SA Reinforcements.\n"
    if complete:
        text += f"%SH kills {name} in Current SA Card.\n"
    text += "%FP moves Free Action Die (P) from FP Dice to Current Action.\n"
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "literal.rendered.txt"
        path.write_text(text, encoding="utf-8")
        raw_path = Path(folder) / "literal.log"
        raw_path.write_text("\n".join(raw), encoding="utf-8")
        return rr.parse_game(path, raw_path=raw_path)["batches"][0]


def extract(case, batch, *, reserve=2):
    counters = {key: Counter() for key in ("detected", "drop_reasons", "kinds_emit")}
    applied = []
    original = ai.apply_action

    def apply(state, action):
        child = original(state, action)
        applied.append((action, child))
        return child

    def emit(state, side, legal, index, phase, turn):
        return dict(chosen=legal[index], phase=phase)

    with patch.object(rx, "_mk_state_with_die", return_value=(position(case[0], case[1], reserve), SH)), \
            patch.object(rx, "_emit_sample", side_effect=emit), patch.object(ai, "apply_action", side_effect=apply), \
            patch.object(rx, "_LEDGER", None), patch.object(rx, "_TABLE", None):
        rows = rx._emit_card_interiors(batch, counters)
    return rows, counters, applied


class RecruitmentAllocationExtractionTests(unittest.TestCase):
    def test_all_five_cards_match_literal_counts_through_actual_menu_and_apply(self):
        for case in CASES:
            with self.subTest(card=case[0]):
                rows, counters, applied = extract(case, literal(case))
                self.assertEqual(len(rows), 1, counters)
                choice = rows[0]["chosen"].params["choices"]
                key = next(k for k in ("regions", "settlements", "strongholds") if k in choice)
                self.assertEqual(dict(zip(choice[key], choice["recruit_counts"])),
                                 {board.sanitize(r): n for r, n in case[3]})
                self.assertIsNone(applied[-1][1].pending_card_option)

    def test_same_regions_different_counts_select_distinct_exact_allocations(self):
        case = CASES[0]
        seen = []
        for placements in ((("Dol Guldur", 1), ("Mount Gundabad", 2)),
                           (("Dol Guldur", 2), ("Mount Gundabad", 1)),
                           (("Mount Gundabad", 3),)):
            rows, counters, _ = extract(case, literal(case, recruits=placements), reserve=3)
            self.assertEqual(len(rows), 1, counters)
            choice = rows[0]["chosen"].params["choices"]
            self.assertEqual(dict(zip(choice["regions"], choice["recruit_counts"])),
                             {board.sanitize(r): n for r, n in placements})
            seen.append(choice)
        self.assertEqual(len({str(choice) for choice in seen}), 3)

    def test_existing_full_supply_many_kings_and_pits_rows_keep_their_payloads(self):
        for case, regions in ((CASES[3], ("Far Harad", "South Rhun", "Umbar")),
                              (CASES[4], ("Barad-Dur", "Dol Guldur", "Minas Morgul"))):
            with self.subTest(card=case[0]):
                batch = literal(case, recruits=tuple((region, 2) for region in regions))
                rows, counters, _ = extract(case, batch, reserve=6)
                self.assertEqual(len(rows), 1, counters)
                self.assertNotIn("recruit_counts", rows[0]["chosen"].params["choices"])

    def test_incomplete_corrections_and_unavailable_counts_never_supply_labels(self):
        for case in CASES:
            for kwargs in (dict(complete=False), dict(correction=True),
                           dict(recruits=((case[3][0][0], 3),))):
                with self.subTest(card=case[0], kwargs=kwargs):
                    rows, counters, _ = extract(case, literal(case, **kwargs))
                    self.assertEqual(rows, [], counters)
                    self.assertTrue(counters["drop_reasons"], counters)

    def test_missing_fixed_elite_effect_is_not_silently_invented(self):
        for case in (CASES[1], CASES[2]):
            rows, counters, _ = extract(case, literal(case, elite_recruits=()))
            self.assertEqual(rows, [], counters)
            self.assertTrue(counters["drop_reasons"], counters)


if __name__ == "__main__":
    unittest.main()
