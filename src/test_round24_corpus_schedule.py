"""Public corpus schedule checks with tiny CPU models and no optimizer updates."""
import os
import unittest
from unittest.mock import patch

import numpy as np
import torch

import test_round22_corpus_admission as fixture
from test_round22_corpus_admission import TinyNet, ai, tc


class CorpusScheduleTests(unittest.TestCase):
    _make_data = fixture.CorpusFitAdmissionTests._make_data

    def setUp(self):
        fixture.CorpusFitAdmissionTests.setUp(self)
        self.rates = []

        def step(optimizer, *args, **kwargs):
            self.rates.append(float(optimizer.param_groups[0]["lr"]))
            gradients = [p.grad for group in optimizer.param_groups
                         for p in group["params"] if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.steps.append(sum(g.numel() for g in gradients))
            # Intentionally no parameter update.
        self.stack.enter_context(patch.object(torch.optim.Adam, "step", step))
        self.stack.enter_context(patch.object(
            tc, "batched_scores", lambda net, h, ne, c, r:
            torch.stack((h[:, 0], -h[:, 0]), dim=1)))

    def test_four_tiny_epochs_keep_positive_decay_through_last_epoch(self):
        tc.train(epochs=4, lr=.001, value_weight=.5)
        np.testing.assert_allclose(self.rates,
            [.001, .0008535533905932737, .0005, .00014644660940672628],
            rtol=0, atol=1e-15)
        self.assertEqual(len(self.steps), 4)

    def test_budget_remainder_receives_all_four_scheduled_updates(self):
        self._make_data(3)
        with patch.object(tc, "CAND_BUDGET", 4):
            tc.train(epochs=2, lr=.001, value_weight=.5)
        np.testing.assert_allclose(self.rates,
            [.001, .0008535533905932737, .0005, .00014644660940672628],
            rtol=0, atol=1e-15)

    def test_row_limit_is_counted_even_below_candidate_budget(self):
        self._make_data(1025)
        tc.train(epochs=1, lr=.001, value_weight=.5)
        np.testing.assert_allclose(self.rates, [.001, .0005], rtol=0, atol=1e-15)

    def _interrupt_after_two_epochs(self, value_weight):
        saved_step = torch.optim.Adam.step

        def interrupted(optimizer, *args, **kwargs):
            if len(self.rates) == 2:
                raise RuntimeError("synthetic interruption before third update")
            return saved_step(optimizer, *args, **kwargs)
        with patch.object(tc, "SNAP_EVERY", 1), \
                patch.object(torch.optim.Adam, "step", interrupted):
            with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                tc.train(epochs=4, lr=.001, value_weight=value_weight)

    def test_mid_schedule_resume_matches_uninterrupted_rates_rng_and_buffers(self):
        self._interrupt_after_two_epochs(.5)
        with patch.dict(os.environ, {"CORPUS_RESUME": "1"}):
            tc.train(epochs=4, lr=.001, value_weight=.5)
        np.testing.assert_allclose(self.rates,
            [.001, .0008535533905932737, .0005, .00014644660940672628],
            rtol=0, atol=1e-15)
        resumed = torch.load(self.root / "wotr_fp_corpus_synthetic.pt", weights_only=True)
        self.rates.clear()
        tc.train(epochs=4, lr=.001, value_weight=.5)
        fresh = torch.load(self.root / "wotr_fp_corpus_synthetic.pt", weights_only=True)
        self.assertTrue(all(torch.equal(value, fresh["model_state"][name])
                            for name, value in resumed["model_state"].items()))

    def test_policy_only_resume_value_status_follows_positive_rate_update(self):
        self._interrupt_after_two_epochs(0.)
        snap = self.root / "wotr_fp_corpus_synthetic_ep02.pt.resume"
        self.assertIs(torch.load(snap, weights_only=False)["value_head_trained"], False)
        self.rates.clear()
        with patch.dict(os.environ, {"CORPUS_RESUME": "1"}):
            tc.train(epochs=4, lr=.001, value_weight=.5)
        np.testing.assert_allclose(self.rates, [.0005, .00014644660940672628],
                                   rtol=0, atol=1e-15)
        final = torch.load(self.root / "wotr_fp_corpus_synthetic.pt", weights_only=True)
        self.assertIs(final["value_head_trained"], True)

    def test_legacy_zero_rate_resume_rebuilds_progress_instead_of_reusing_floor(self):
        net = TinyNet(ai.get_feature_dim(), "fp")
        opt = torch.optim.Adam(net.parameters(), lr=0.)
        torch.save({**self.contract, "model_state": net.state_dict(), "head_act": "tanh",
                    "value_head_trained": False, "epoch": 2, "opt_state": opt.state_dict(),
                    "sched_state": {"T_max": 1, "last_epoch": 1, "_last_lr": [0.]},
                    "np_rng_state": np.random.RandomState(4).get_state(),
                    "torch_rng_state": torch.get_rng_state(), "cuda_rng_state": None},
                   self.root / "wotr_fp_corpus_synthetic_ep02.pt.resume")
        with patch.dict(os.environ, {"CORPUS_RESUME": "1"}):
            tc.train(epochs=4, lr=.001, value_weight=.5)
        np.testing.assert_allclose(self.rates, [.0005, .00014644660940672628],
                                   rtol=0, atol=1e-15)

    def test_current_schedule_mismatch_refuses_before_any_further_update(self):
        self._interrupt_after_two_epochs(.5)
        self.rates.clear()
        with patch.dict(os.environ, {"CORPUS_RESUME": "1"}):
            with self.assertRaisesRegex(ValueError, "schedule differs"):
                tc.train(epochs=5, lr=.001, value_weight=.5)
        self.assertEqual(self.rates, [])
        self.assertFalse((self.root / "wotr_fp_corpus_synthetic.pt").exists())

