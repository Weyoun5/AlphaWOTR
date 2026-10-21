"""Native public Fellowship faces and explicit board recovery, outside search."""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_live_driver import fresh_state
from test_audit0905_political_resync import NATIVE
from live import adopt, client_board, client_protocol
import game_logic


class FellowshipRecovery(unittest.TestCase):
    def test_native_face_is_compared_without_selection_flag_guessing(self):
        state = fresh_state()
        dump = {'247': 'rivendell'}
        state.fellowship.hidden = True
        rows = client_board.diff_engine_vs_dump(state, dump, marker_states={'247': 1})
        self.assertTrue(any('fellowship visibility: engine=hidden client=revealed' in row for row in rows))
        for faces in ({}, {'247': 0}, {'247': 'U'}, {'247': 2}):
            rows = client_board.diff_engine_vs_dump(state, dump, marker_states=faces)
            self.assertFalse(any('fellowship visibility' in row for row in rows), rows)
        state.fellowship.hidden = False
        rows = client_board.diff_engine_vs_dump(state, dump, marker_states={'247': 0})
        self.assertTrue(any('fellowship visibility: engine=revealed client=hidden' in row for row in rows))

    def test_complete_native_board_adopts_public_face_and_rolls_back_on_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'board.txt'
            shutil.copy2(NATIVE, path)
            dump = client_board.read_board(path)
            faces = client_protocol.read_marker_states(path)
            self.assertIn(faces['247'], (0, 1))
            physical_hidden = faces['247'] == 0
            for fail in (False, True):
                state = fresh_state()
                state.fellowship.hidden = not physical_hidden
                if fail:
                    with patch.object(adopt, 'adopt_hands', side_effect=RuntimeError('test interrupted adoption')):
                        report, _, _ = adopt.adopt_client_board(state, dump, dump_path=path, manual=True)
                    self.assertEqual(state.fellowship.hidden, not physical_hidden)
                    self.assertTrue(any('rolled back' in row for row in report))
                else:
                    report, _, _ = adopt.adopt_client_board(state, dump, dump_path=path, manual=True)
                    self.assertEqual(state.fellowship.hidden, physical_hidden, report)
                    self.assertTrue(any('visibility adopted' in row for row in report))

    def test_mordor_exit_is_visible_and_only_explicit_recovery_adopts_it(self):
        state = fresh_state()
        state.fellowship.mordor_entered = True
        state.fellowship.mordor_step = 3
        dump = {'247': 'North Ithilien'}
        rows = client_board.diff_engine_vs_dump(state, dump)
        self.assertTrue(any('pawn mordor exit' in row for row in rows))
        adopt.adopt_fellowship_position(state, dump)
        self.assertTrue(state.fellowship.mordor_entered)
        before = adopt._snapshot_armies(state)
        adopt.adopt_fellowship_position(state, dump, manual=True)
        self.assertFalse(state.fellowship.mordor_entered)
        self.assertEqual(state.fellowship.mordor_step, 0)
        self.assertEqual(game_logic.fellowship_region(state).lower(), 'north ithilien')
        adopt._restore_armies(state, before)
        self.assertTrue(state.fellowship.mordor_entered)
        self.assertEqual(state.fellowship.mordor_step, 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
