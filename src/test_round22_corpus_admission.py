"""Bounded corpus fit admission; optimizer updates are explicitly substituted."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
os.environ.update(KMP_DUPLICATE_LIB_OK="TRUE", OMP_NUM_THREADS="1",
                  CUDA_VISIBLE_DEVICES="-1", WOTR_ACT_GEN="v52", WOTR_NATIVE="0")
import numpy as np
import torch
import torch._dynamo
import ai
import corpus_feed
import train_corpus as tc
os.chdir(ROOT)


class TinyNet(torch.nn.Module):
    def __init__(self, width, side):
        super().__init__()
        self.trunk = torch.nn.Linear(2, 2)
        self.value_head = torch.nn.Sequential(
            torch.nn.BatchNorm1d(2), torch.nn.Linear(2, 1), torch.nn.Tanh())

    def _process_input(self, x):
        h = self.trunk(x[:, :2])
        return h, h.unsqueeze(1)


class CorpusFitAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix="r22_corpus_test_")))
        self.stack.callback(corpus_feed._G.clear)
        self.ds = self.root / "ds"
        self.ds.mkdir()
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(patch.dict(os.environ, {
            "WOTR_FEED_WORKERS": "0", "CORPUS_TRAIN_SIDES": "fp", "WOTR_ANCHOR_SIDES": "fp",
            "WOTR_VALUE_HEAD_ACT": "tanh", "WOTR_BC_CERT": "0", "WOTR_JT_CERT": "0",
            "CORPUS_RESUME": "0", "VALUE_CTX_DROPOUT": "0"}))
        for key, value in dict(ROOT=str(self.root), DS_DIR=str(self.ds),
                               SPLIT_JSON=str(self.ds / "split.json"), NET_TAG="synthetic",
                               CORPUS=str(self.root / "corpus"), SNAP_EVERY=0).items():
            self.stack.enter_context(patch.object(tc, key, value))
        self.stack.enter_context(patch.object(ai, "DEVICE", torch.device("cpu")))
        self.stack.enter_context(patch.object(ai, "WOTRValueNet", TinyNet))
        self.contract = tc._current_learning_contract()
        (self.ds / "split.json").write_text(json.dumps(dict(
            train=["synthetic_train.npz"], holdout=["synthetic_hold.npz"],
            learning_contract=self.contract)), encoding="utf-8")
        self.steps = []
        self._make_data(2)

        def step(optimizer, *args, **kwargs):
            grads = [p.grad for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.steps.append(sum(g.numel() for g in grads))
            # Deliberately no parameter update.
        self.stack.enter_context(patch.object(torch.optim.Adam, "step", step))

    def _make_data(self, n):
        x = np.zeros((n, ai.get_feature_dim()), np.float32)
        x[:, 0] = np.arange(n) + 1
        arrays = dict(X=x, z=np.ones(n, np.float32), phi=np.zeros(n, np.float32),
                      off=np.arange(n + 1) * 2,
                      cands=np.zeros((2 * n, ai.ACTION_ENCODING_DIM), np.float16),
                      regs=np.full(2 * n, -1, np.int32), pos=np.tile([1, 0], n).astype(np.uint8),
                      dw=np.zeros(n, np.uint8))
        for key, value in arrays.items():
            np.save(self.ds / ("fp_" + key + ".npy"), value)

    def _anchor_model(self):
        net = TinyNet(ai.get_feature_dim(), "fp")
        path = self.root / "wotr_fp_corpus_syntheticjt.pt"
        torch.save({**self.contract, "model_state": net.state_dict(), "side": "fp",
                    "head_act": "tanh", "input_dim": ai.get_feature_dim(),
                    "action_dim": ai.ACTION_ENCODING_DIM}, path)
        self.stack.enter_context(patch.object(ai, "load_value_net", return_value=net))
        return net, path

    def test_public_cli_rejects_zero_epochs_before_dataset_or_model_publication(self):
        with patch.object(sys, "argv", ["train_corpus.py", "--stage", "train", "--epochs", "0"]):
            with self.assertRaisesRegex(ValueError, "positive epoch"):
                tc.main()
        self.assertFalse(list(self.root.glob("*.pt")))
        self.assertEqual(self.steps, [])

    def test_public_stages_refuse_empty_schedules(self):
        for fn in (lambda: tc.train(-1, 1e-3, .5),
                   lambda: tc.value_td(.8, 0, 1, 1e-3, 2),
                   lambda: tc.value_td(.8, 1, 0, 1e-3, 2),
                   lambda: tc.value_anchor(epochs=0),
                   lambda: tc.value_anchor(batch=0),
                   lambda: tc.value_anchor(n_train=0)):
            with self.subTest(fn=fn), self.assertRaises(ValueError):
                fn()
        self.assertFalse(list(self.root.glob("*.pt")))
        self.assertEqual(self.steps, [])

    def test_default_anchor_fits_two_rows_each_epoch(self):
        net, path = self._anchor_model()
        before = {name: value.clone() for name, value in net.state_dict().items()}
        tc.value_anchor(epochs=2)
        saved = torch.load(path, weights_only=True)
        self.assertEqual(len(self.steps), 2)
        self.assertEqual(saved["value_anchor"], dict(epochs=2, lr=3e-4, n=2, optimizer_steps=2))
        self.assertTrue(all(torch.equal(value, saved["model_state"][name]) for name, value in before.items()))

    def test_anchor_keeps_partial_last_batch_and_one_row(self):
        for n, batch, expected in ((3, 2, 2), (1, 4096, 1)):
            with self.subTest(n=n):
                self._make_data(n)
                self._anchor_model()
                self.steps.clear()
                tc.value_anchor(epochs=1, batch=batch)
                self.assertEqual(len(self.steps), expected)

    def test_failed_anchor_update_preserves_checkpoint_without_stamp(self):
        _, path = self._anchor_model()
        before = path.read_bytes()
        with patch.object(torch.optim.Adam, "step", side_effect=RuntimeError("substituted failure")):
            with self.assertRaisesRegex(RuntimeError, "substituted failure"):
                tc.value_anchor(epochs=1)
        self.assertEqual(path.read_bytes(), before)

    def test_budget_batches_preserve_every_row_including_singletons(self):
        self.assertEqual(list(tc._budget_batches([0, 1, 2], [0, 2, 4, 6], 4)), [[0, 1], [2]])
        self.assertEqual(list(tc._budget_batches([0, 1], [0, 5, 10], 2)), [[0], [1]])

    def test_singleton_bc_completes_gradients_and_publication(self):
        self._make_data(1)
        def scores(net, h, ne, cands, regs):
            return h[:, :1].expand(-1, cands.shape[1])
        with patch.object(tc, "batched_scores", scores):
            result = tc.train(epochs=1, lr=1e-3, value_weight=.5)
        self.assertEqual(len(self.steps), 1)
        self.assertEqual(len(result["fp"]["curve"]), 1)
        saved = torch.load(self.root / "wotr_fp_corpus_synthetic.pt", weights_only=True)
        self.assertTrue(saved["value_head_trained"])
        self.assertEqual(saved["model_state"]["value_head.0.num_batches_tracked"].item(), 0)

    def test_singleton_batch_norm_restores_training_flags(self):
        net = TinyNet(2, "fp").train()
        with tc._singleton_batch_norm(net, 1):
            self.assertFalse(net.value_head[0].training)
            net.value_head(torch.ones(1, 2)).sum().backward()
        self.assertTrue(net.value_head[0].training)
        self.assertIsNotNone(net.value_head[0].weight.grad)

    def test_policy_only_bc_never_labels_fresh_value_head_as_trained(self):
        with patch.object(tc, "batched_scores", lambda net, h, ne, c, r: h[:, :1].expand(-1, c.shape[1])), \
                patch.object(tc, "SNAP_EVERY", 1):
            tc.train(epochs=2, lr=1e-3, value_weight=0.)
        for name in ("wotr_fp_corpus_synthetic.pt", "wotr_fp_corpus_synthetic_ep01.pt",
                     "wotr_fp_corpus_synthetic_ep01.pt.resume"):
            saved = torch.load(self.root / name, weights_only=False)
            self.assertIs(saved["value_head_trained"], False)
        self.assertEqual(len(self.steps), 2)

    def test_frozen_joint_value_head_preserves_seed_training_status(self):
        for previously_trained in (False, True):
            with self.subTest(previously_trained=previously_trained):
                net = TinyNet(ai.get_feature_dim(), "fp")
                torch.save({**self.contract, "model_state": net.state_dict(), "side": "fp",
                            "head_act": "tanh", "value_head_trained": previously_trained},
                           self.root / "wotr_fp_corpus_synthetic.pt")
                np.save(self.ds / "fp_ztd.npy", np.zeros(2, np.float32))
                (self.ds / "fp_ztd_meta.json").write_text(json.dumps(
                    {**self.contract, "currency": "residual-z-phi"}), encoding="utf-8")
                with patch.dict(os.environ, {"WOTR_JT_FREEZE_VALUE": "1"}), \
                        patch.object(tc, "batched_scores", lambda net, h, ne, c, r: h[:, :1].expand(-1, c.shape[1])):
                    tc.train(epochs=1, lr=1e-3, value_weight=.5, joint=True)
                saved = torch.load(self.root / "wotr_fp_corpus_syntheticjt.pt", weights_only=True)
                self.assertIs(saved["value_head_trained"], previously_trained)

    def test_policy_only_resume_preserves_trained_seed_status(self):
        net = TinyNet(ai.get_feature_dim(), "fp")
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1)
        torch.save({**self.contract, "model_state": net.state_dict(), "head_act": "tanh",
                    "value_head_trained": True, "epoch": 1, "opt_state": opt.state_dict(),
                    "sched_state": sched.state_dict(), "np_rng_state": np.random.RandomState(4).get_state(),
                    "torch_rng_state": torch.get_rng_state(), "cuda_rng_state": None},
                   self.root / "wotr_fp_corpus_synthetic_ep01.pt.resume")
        with patch.dict(os.environ, {"CORPUS_RESUME": "1"}), \
                patch.object(tc, "batched_scores", lambda net, h, ne, c, r: h[:, :1].expand(-1, c.shape[1])):
            tc.train(epochs=2, lr=1e-3, value_weight=0.)
        saved = torch.load(self.root / "wotr_fp_corpus_synthetic.pt", weights_only=True)
        self.assertIs(saved["value_head_trained"], True)
        self.assertEqual(len(self.steps), 1)

    def test_td_keeps_partial_last_batch_and_exports_after_steps(self):
        for side in ("fp", "sh"):
            net = TinyNet(ai.get_feature_dim(), side)
            torch.save({**self.contract, "model_state": net.state_dict(), "side": side,
                        "head_act": "tanh", "input_dim": ai.get_feature_dim()},
                       self.root / f"wotr_{side}_corpus_synthetic.pt")
        synthetic = (np.ones((3, 2), np.float32), np.ones(3, np.float32),
                     np.array([0, 0, 0]), np.array([1, 2, 3]),
                     np.arange(3), np.zeros(3, np.float32))
        def collect(nets, names, sc, tag):
            return synthetic if names == ["synthetic_train.npz"] else None
        with patch.object(tc, "_td_collect_side", side_effect=collect), patch.object(tc, "_td_target_clip", return_value=1.):
            tc.value_td(.8, rounds=1, epochs=1, lr=1e-3, batch=2)
        self.assertEqual(len(self.steps), 4)  # Two batches for each of two sides.
        self.assertTrue((self.ds / "fp_ztd.npy").exists())
        self.assertTrue((self.root / "wotr_fp_corpus_synthetictd.pt").exists())


if __name__ == "__main__":
    unittest.main()
