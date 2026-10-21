"""Causal extraction contracts derived from the turn order, not replay output.

Rulebook PDF/printed page 18: Fellowship phase precedes Hunt allocation,
which precedes the Action roll. Changing those future events must not change
the input for an earlier decision. Run with Python's unittest runner.
"""
import os
import sys
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('WOTR_ACT_GEN', 'v41')
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import ai
import board
import game_state as gs
import rendered_extract as rx
import extract_corpus


class DecisionTimeInformationTests(unittest.TestCase):
    def emit(self, phase, allocated, rolled, before_allocation=False):
        state = gs.setup_initial_game(board.load_board())
        state.turn = 2
        state._declare_phase = phase == 'fellowship_declare'
        state._eye_allocation_phase = phase == 'eye_allocate'
        state._rx_before_hunt_allocation = before_allocation
        side = gs.Side.SHADOW if phase == 'eye_allocate' else gs.Side.FREE_PEOPLES
        # The label and pre-decision state stay fixed. Only future log facts
        # differ; no expected output is computed using the engine's rules.
        action = ai.Action(-1, 'pass', {})
        with patch.multiple(rx, _RX_EYEALLOC={2: allocated},
                            _RX_EYEROLL={2: rolled}, _RX_RINGS_CUR=None,
                            _RX_LINE_CUR=None, _RX_STATS=None,
                            _RX_ALT_HIDX=None, _RX_ALIAS_V41=None):
            sample = rx._emit_sample(state, side, [action], 0, phase, 2)
        return sample['x'], state

    def test_declaration_input_does_not_see_future_hunt_allocation(self):
        x0, _ = self.emit('fellowship_declare', 0, 0)
        x1, _ = self.emit('fellowship_declare', 3, 0)
        np.testing.assert_array_equal(x0, x1)

    def test_declaration_input_does_not_see_future_eye_roll(self):
        x0, _ = self.emit('fellowship_declare', 0, 0)
        x1, _ = self.emit('fellowship_declare', 0, 4)
        np.testing.assert_array_equal(x0, x1)

    def test_turn_start_guide_input_does_not_see_future_hunt(self):
        x0, _ = self.emit('choose_guide', 0, 0, before_allocation=True)
        x1, _ = self.emit('choose_guide', 2, 3, before_allocation=True)
        np.testing.assert_array_equal(x0, x1)

    def test_eye_allocation_input_does_not_see_its_answer_or_roll(self):
        x0, _ = self.emit('eye_allocate', 0, 0)
        x1, _ = self.emit('eye_allocate', 2, 3)
        np.testing.assert_array_equal(x0, x1)

    def test_action_phase_keeps_public_hunt_information(self):
        x0, _ = self.emit('fellowship_move', 0, 0)
        x1, state = self.emit('fellowship_move', 2, 3)
        self.assertEqual(state.eyes_allocated, 2)
        self.assertEqual(len(state.hunt_box), 5)
        self.assertFalse(np.array_equal(x0, x1))

    def test_pack_keeps_decision_phase_for_coverage_audits(self):
        # Hand-written rows: expectations describe the storage contract,
        # without using the same packer to create its expected output.
        samples = [dict(x=np.zeros(3), encs=np.zeros((2, 2)),
                        regs=np.array([0, 1]), side='fp', human=1,
                        z=1, phi=0.0, turn=2, phase=phase)
                   for phase in ('fellowship_declare', 'choose_guide')]
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'game.npz')
            extract_corpus.pack_npz(samples, path)
            with np.load(path, allow_pickle=False) as packed:
                self.assertEqual(packed['decision_phase'].tolist(),
                                 ['fellowship_declare', 'choose_guide'])
                self.assertEqual(packed['human'].tolist(), [1, 1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
