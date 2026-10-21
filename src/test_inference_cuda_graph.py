"""Literal CUDA Graph PV, cache lifetime and budget regressions.

Run explicitly with WOTR_TEST_CUDA_GRAPH=1 on the v52 base runtime.
No audit paths/checkpoints or running inference server are required.
"""
import os
import unittest
import torch
import numpy as np
import ai
from graph_pv import GraphPV, GraphCacheBudgetExceeded

@unittest.skipUnless(os.environ.get('WOTR_TEST_CUDA_GRAPH') == '1' and torch.cuda.is_available(), 'Explicit CUDA test opt-in required')
class InferenceCudaGraphTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.old_device=ai.DEVICE;cls.old_threads=torch.get_num_threads();cls.old_deterministic=torch.are_deterministic_algorithms_enabled()
        torch.set_num_threads(1);torch.use_deterministic_algorithms(False)
        ai.DEVICE = torch.device('cuda:0')
        cls.net = ai.WOTRValueNet(4481, side='fp').to(ai.DEVICE).eval()
        with torch.no_grad():
            for p in cls.net.parameters():
                p.zero_()
            base = cls.net.spatial_action_scorer
            index = ai.NET_ARCH_CFG['bb2'] + ai.NET_ARCH_CFG['gnn_out']
            base[0].weight[0, index] = 2
            base[0].bias[0] = 1
            base[3].weight[0, 0] = 1
            base[5].weight[0, 0] = 1
            delta = cls.net.policy_delta
            delta[0].weight[0, index] = -1
            delta[0].bias[0] = 4
            delta[2].weight[0, 0] = 1
        cls.vec = np.zeros(4481, np.float32)
        cls.enc = np.zeros((3, 133), np.float32)
        cls.enc[:, 0] = [0, 1, 2]
        cls.enc[:, 33] = -1
        cls.reg = np.array([-1, 0, 1], np.int64)

    @classmethod
    def tearDownClass(cls):
        del cls.net
        ai.DEVICE=cls.old_device
        torch.set_num_threads(cls.old_threads)
        torch.use_deterministic_algorithms(cls.old_deterministic)

    def setUp(self):
        self.cache = GraphPV(self.net, max_graphs=2)

    def tearDown(self):
        self.cache.close()

    def item(self, n=3):
        return (self.vec, self.enc[:n], self.reg[:n])

    def test_literal_delta_padding_empty_and_stale_rows(self):
        expected = np.exp(np.array([0.0, 1.0, 2.0]))
        expected /= expected.sum()
        out = self.cache.pv_many([self.item(), self.item(0)], True)
        np.testing.assert_allclose(out[0][0], expected, atol=1e-07)
        self.assertEqual(len(out[1][0]), 0)
        self.assertEqual(out[1][1], 0.0)
        out = self.cache.pv_many([self.item(0), self.item()], True)
        self.assertEqual(len(out[0][0]), 0)
        np.testing.assert_allclose(out[1][0], expected, atol=1e-07)
        self.assertEqual(self.cache.captures, 1)
        base = np.exp(np.array([0.0, 2.0, 4.0]))
        base /= base.sum()
        np.testing.assert_allclose(self.cache.pv_many([self.item()], False)[0][0], base, atol=1e-07)

    def test_eviction_lifetime_interleaved_allocations(self):
        for n in [3, 1, 2, 3, 1, 3]:
            out = self.cache.pv_many([self.item(n)], True)
            rubbish = [torch.empty((4096, 64), device='cuda') for _ in range(3)]
            for x in rubbish:
                x.fill_(999)
            expected = np.exp(np.arange(n, dtype=np.float64))
            expected /= expected.sum()
            np.testing.assert_allclose(out[0][0], expected, atol=1e-07)
            self.assertLessEqual(len(self.cache.graphs), 2)
        self.assertGreater(self.cache.evictions, 0)

    def test_invalid_region_width_finite_and_budget_refuse_before_capture(self):
        for item in [(self.vec, self.enc, np.array([-2, 0, 1])), (self.vec, self.enc[:, :132], self.reg), (self.vec * np.nan, self.enc, self.reg)]:
            with self.assertRaises(ValueError):
                self.cache.pv_many([item])
        self.cache.max_rows = 1
        with self.assertRaisesRegex(ValueError, 'budget'):
            self.cache.pv_many([self.item()])
        self.assertEqual(self.cache.captures, 0)

    def test_revision_math_and_training_refusal(self):
        self.cache.pv_many([self.item()])
        self.net.train()
        with self.assertRaisesRegex(RuntimeError, 'Eval'):
            self.cache.pv_many([self.item()])
        self.net.eval()
        torch.use_deterministic_algorithms(True)
        try:
            with self.assertRaisesRegex(RuntimeError, 'math'):
                self.cache.pv_many([self.item()])
            with self.assertRaisesRegex(RuntimeError, 'deterministic'):
                GraphPV(self.net)
        finally:
            torch.use_deterministic_algorithms(False)
        with torch.no_grad():
            next(self.net.parameters()).add_(0)
        with self.assertRaisesRegex(RuntimeError, 'revision'):
            self.cache.pv_many([self.item()])

    def test_accounted_memory_eviction_and_clear(self):
        self.cache.close()
        self.cache = GraphPV(self.net, max_graphs=32)
        self.cache.pv_many([self.item(1)] * 8)
        budget = self.cache.cached_bytes + self.cache._estimated_bytes(8, 2) - 1
        self.cache.max_bytes = budget
        for n in [1, 2, 3, 1, 2, 3]:
            self.cache.pv_many([self.item(n)] * 8)
            stats = self.cache.stats()
            self.assertLessEqual(stats['accounted_cached_bytes'], budget)
        self.assertGreater(self.cache.evictions, 0)
        self.assertGreater(self.cache.capture_seconds, 0.0)
        self.cache.close()
        self.assertEqual(self.cache.cached_bytes, 0)

    def test_tiny_budget_refuses_before_capture(self):
        self.cache.close()
        self.cache = GraphPV(self.net, max_cached_bytes=1024)
        with self.assertRaises(GraphCacheBudgetExceeded):
            self.cache.pv_many([self.item()])
        self.assertEqual(self.cache.captures, 0)

class InferenceGraphRevisionGuardTests(unittest.TestCase):
    """CPU-only cache-guard state; CUDA construction/capture is tested above."""
    def seeded_guard(self):
        model=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.ReLU()).eval()
        model.register_buffer('literal_buffer',torch.zeros(2))
        cache=object.__new__(GraphPV);cache.net=model
        cache.parameters=tuple(model.parameters());cache.buffers=tuple(model.buffers())
        cache.versions=tuple(t._version for t in cache.parameters+cache.buffers)
        cache.storage=tuple(t.data_ptr() for t in cache.parameters+cache.buffers)
        cache.layouts=tuple((m,tuple(m._parameters.items()),tuple(m._buffers.items()),tuple(m._modules.items())) for m in model.modules())
        cache.math=cache._math_settings();cache.guard()
        return model,cache

    def test_parameter_storage_replacement_refused(self):
        model,cache=self.seeded_guard()
        model[0].weight.data=model[0].weight.detach().clone()
        with self.assertRaises(RuntimeError):cache.guard()

    def test_buffer_storage_replacement_refused(self):
        model,cache=self.seeded_guard()
        model.literal_buffer.data=model.literal_buffer.detach().clone()
        with self.assertRaises(RuntimeError):cache.guard()

    def test_parameter_free_child_replacement_refused(self):
        model,cache=self.seeded_guard()
        model[1]=torch.nn.Sigmoid().eval()
        with self.assertRaises(RuntimeError):cache.guard()

if __name__ == '__main__':
    unittest.main(verbosity=2)
