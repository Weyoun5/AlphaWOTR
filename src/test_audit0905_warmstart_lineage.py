"""Initialization saves and real optimizer calls retain distinct provenance."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import run_td_v33loop as loop
import value_trainer_v2 as value


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value_head = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Tanh())
        self.warm_start = {'source_sha256': 'a' * 64, 'new_input_columns': 3,
                           'new_columns_initialization': 'zero',
                           'replay_or_optimizer_migrated': False}
        self.training_status = 'warm_start_not_retrained'
        self.warm_start_updates = {}

    def forward(self, x):
        return self.value_head(x)


class WarmStartLineageTests(unittest.TestCase):
    def setUp(self):
        # Tiny isolates optimizer provenance and has no WOTR observation layout.
        env = patch.dict(os.environ, WOTR_REWARD_CONTRACT='predecision_profile_v1',
                         WOTR_OBSERVATION_TRAINING='head_only_legacy')
        env.start()
        self.addCleanup(env.stop)

    def sequence(self):
        return (np.array([[.2], [.3], [.4]], np.float32), np.zeros(3), np.zeros(3), -1.)

    def test_initial_save_keeps_untrained_status_and_source_for_both_writers(self):
        for writer in (loop.save_net, value.save_v2_checkpoint):
            with self.subTest(writer=writer), tempfile.TemporaryDirectory() as folder:
                net = Tiny()
                original = copy.deepcopy(net.warm_start)
                path = str(Path(folder) / 'initial.pt')
                writer(net, 'fp', path, profile='ring')
                saved = torch.load(path, weights_only=True)
                self.assertEqual(saved.get('warm_start'), original)
                self.assertEqual(saved.get('training_status'), 'warm_start_not_retrained')
                self.assertEqual(saved.get('warm_start_updates'), {})
                self.assertEqual(net.warm_start, original)

    def test_real_value_steps_are_counted_and_persisted_without_skill_claim(self):
        net = Tiny()
        net.training_status = 'updated_after_warm_start'
        net.warm_start_updates = {'policy': 7}
        with patch.object(value, 'VALUE_BS', 2), patch.object(value, 'VALUE_EPOCHS', 2):
            value.value_update_for_driver(net, [self.sequence()], .001)
        # Three samples, batches of two, two epochs: four completed steps.
        self.assertEqual(net.warm_start_updates, {'policy': 7, 'value': 4})
        self.assertEqual(net.training_status, 'updated_after_warm_start')
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'updated.pt')
            loop.save_net(net, 'fp', path, profile='ring')
            saved = torch.load(path, weights_only=True)
            self.assertEqual(saved['warm_start_updates'], {'policy': 7, 'value': 4})
            self.assertEqual(saved['warm_start']['source_sha256'], 'a' * 64)

    def test_no_samples_or_failed_optimizer_never_marks_training_completed(self):
        for failed in (False, True):
            with self.subTest(failed=failed):
                net = Tiny()
                if failed:
                    with patch.object(torch.optim.Adam, 'step', side_effect=RuntimeError('literal failure')):
                        with self.assertRaisesRegex(RuntimeError, 'literal failure'):
                            value.value_update_for_driver(net, [self.sequence()], .001)
                else:
                    self.assertIsNone(value.value_update_for_driver(net, [], .001))
                self.assertEqual(net.training_status, 'warm_start_not_retrained')
                self.assertEqual(net.warm_start_updates, {})

    def test_inconsistent_status_cannot_be_written_as_a_verified_checkpoint(self):
        for writer in (loop.save_net, value.save_v2_checkpoint):
            with self.subTest(writer=writer), tempfile.TemporaryDirectory() as folder:
                net = Tiny()
                net.training_status = 'updated_after_warm_start'
                path = str(Path(folder) / 'invalid.pt')
                with self.assertRaisesRegex(ValueError, 'status and optimizer counts disagree'):
                    writer(net, 'fp', path, profile='ring')
                self.assertFalse(Path(path).exists())


if __name__ == '__main__':
    unittest.main()
