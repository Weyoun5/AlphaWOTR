"""CPU-only adapter tests for raw graph values; no model or audit inputs."""
import unittest
import types
import numpy as np
from graph_pv import GraphPV, GraphCacheBudgetExceeded

class InferenceGraphEvalTests(unittest.TestCase):

    def setUp(self):
        self.cache = object.__new__(GraphPV)
        self.cache.net = types.SimpleNamespace(input_dim=3, spatial_action_scorer=[types.SimpleNamespace(in_features=468)], backbone=[None] * 4 + [types.SimpleNamespace(out_features=256)], board_gnn=types.SimpleNamespace(layers=[types.SimpleNamespace(linear=types.SimpleNamespace(out_features=64))]), card_embedding=types.SimpleNamespace(embedding_dim=16))
        self.calls = []
        self.guards = []
        self.cache.guard = lambda: self.guards.append(True)

        def pv(items, with_delta=True):
            self.calls.append((items, with_delta))
            return [(np.empty(0, np.float32), float(v[0])) for v, a, r in items]
        self.cache.pv_many = pv

    def test_order_full_observation_and_empty_action_contract(self):
        x = np.array([[0.25, 9, 8], [0.75, 7, 6]], np.float32)
        out = self.cache.eval_values(x)
        np.testing.assert_array_equal(out, np.array([0.25, 0.75], np.float32))
        items, delta = self.calls[0]
        self.assertFalse(delta)
        for i, (v, a, r) in enumerate(items):
            np.testing.assert_array_equal(v, x[i])
            self.assertEqual(a.shape, (0, 133))
            self.assertEqual(a.dtype, np.float32)
            self.assertEqual(r.shape, (0,))
            self.assertEqual(r.dtype, np.int64)

    def test_empty_batch_still_checks_revision_without_capture(self):
        out = self.cache.eval_values(np.empty((0, 3), np.float32))
        self.assertEqual(out.shape, (0,))
        self.assertEqual(len(self.guards), 1)
        self.assertEqual(self.calls, [])

    def test_bad_inputs_refuse_before_inference(self):
        for x in [np.zeros((2, 4), np.float32), np.zeros((2, 3), np.float64), np.full((2, 3), np.nan, np.float32)]:
            with self.assertRaises(ValueError):
                self.cache.eval_values(x)
        self.assertEqual(self.calls, [])

    def test_graph_failure_propagates_unchanged(self):

        def fail(*a, **kw):
            raise GraphCacheBudgetExceeded('literal budget')
        self.cache.pv_many = fail
        with self.assertRaisesRegex(GraphCacheBudgetExceeded, 'literal budget'):
            self.cache.eval_values(np.zeros((1, 3), np.float32))

if __name__=='__main__':unittest.main(verbosity=2)
