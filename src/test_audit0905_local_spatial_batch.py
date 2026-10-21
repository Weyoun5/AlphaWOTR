"""Independent hand-computed logits across differently sized policy menus."""
import math
import os
import unittest
from unittest.mock import patch
import numpy as np
import torch
from inference_batch import spatial_policy_value_batch


class LiteralNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_action_scorer = torch.nn.Linear(4, 1, bias=False)
        self.policy_delta = torch.nn.Linear(4, 1, bias=False)
        with torch.no_grad():
            self.spatial_action_scorer.weight.copy_(torch.tensor([[1., 2., 3., 5.]]))
            self.policy_delta.weight.copy_(torch.tensor([[-.5, 1., 0., 2.]]))

    def forward_backbone(self, x):
        return x[:, :1], x[:, 1:].unsqueeze(-1)

    def value_head(self, bb):
        return bb

    def _embed_action_encs(self, ae):
        return ae[:, :1], ae[:, 1:]


def literal_softmax(logits):
    vals = [math.exp(v-max(logits)) for v in logits]
    return np.array([v/sum(vals) for v in vals])


class SpatialBatchTests(unittest.TestCase):
    def setUp(self):
        self.net = LiteralNet().eval()
        self.vecs = [np.array([.2, .1, .3], np.float32), np.array([.7, .2, .6], np.float32)]
        self.aes = [np.array([[.1, 0], [.2, 1], [.3, 0]], np.float32),
                    np.array([[.4, 0], [.5, 1]], np.float32)]
        self.ris = [np.array([0, 1, -1], np.int64), np.array([-1, 0], np.int64)]

    def run_batch(self, **kw):
        return spatial_policy_value_batch(self.net, self.vecs, self.aes, self.ris,
                                          'cpu', kw.pop('with_delta', False), **kw)

    def test_literal_region_card_and_independent_softmax(self):
        output = self.run_batch()
        for (p, v), logits, value in zip(output, [[.7, 6.4, 1.5], [2.7, 7.6]], [.2, .7]):
            np.testing.assert_allclose(p, literal_softmax(logits), rtol=1e-6, atol=1e-7)
            self.assertAlmostEqual(v, value, places=6)
            self.assertAlmostEqual(float(p.sum()), 1., places=6)

    def test_literal_residual_and_chunks_cross_leaf_boundary(self):
        for limit in [1, 2, 4, 16384]:
            with self.subTest(limit=limit):
                output = self.run_batch(with_delta=True, max_rows=limit)
                for (p, _), logits in zip(output, [[.7, 8.6, 1.6], [2.75, 9.45]]):
                    np.testing.assert_allclose(p, literal_softmax(logits), rtol=1e-6, atol=1e-7)

    def test_candidate_permutation_retains_order(self):
        self.aes[0] = self.aes[0][[2, 0, 1]]
        self.ris[0] = self.ris[0][[2, 0, 1]]
        p = self.run_batch()[0][0]
        np.testing.assert_allclose(p, literal_softmax([1.5, .7, 6.4]), rtol=1e-6, atol=1e-7)

    def test_missing_delta_and_singleton_menu(self):
        self.net.policy_delta = None
        self.aes[1] = self.aes[1][:1]
        self.ris[1] = self.ris[1][:1]
        output = self.run_batch(with_delta=True)
        np.testing.assert_allclose(output[0][0], literal_softmax([.7, 6.4, 1.5]), rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(output[1][0], [1.])

    def test_invalid_counts_shapes_empty_menu_and_chunk_limit_refused(self):
        for mutate in ['counts', 'regions', 'matrix', 'empty', 'limit']:
            with self.subTest(mutate=mutate):
                self.setUp()
                if mutate == 'counts': self.ris.pop()
                if mutate == 'regions': self.ris[0] = self.ris[0][:-1]
                if mutate == 'matrix': self.ris[0] = self.ris[0][:, None]
                if mutate == 'empty': self.aes[0], self.ris[0] = self.aes[0][:0], self.ris[0][:0]
                with self.assertRaises(ValueError):
                    self.run_batch(max_rows=0 if mutate == 'limit' else 16384)

    def test_no_gradients_or_parameter_updates(self):
        before = {k: v.clone() for k, v in self.net.state_dict().items()}
        self.run_batch(with_delta=True)
        self.assertTrue(all(p.grad is None for p in self.net.parameters()))
        for key, value in self.net.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)


class SearchBoundaryTests(unittest.TestCase):
    def setUp(self):
        SpatialBatchTests.setUp(self)

    def test_gpu_dispatch_preserves_output_validation(self):
        import ai
        invalid = [([(np.array([float('nan'), 0., 0.]), .2), (np.array([.5, .5]), .7)]),
                   ([(np.array([.5, .5]), .2), (np.array([.5, .5]), .7)]),
                   ([(np.array([.2, .3, .5]), float('inf')), (np.array([.5, .5]), .7)]),
                   ([(np.array([.2, .3, .5]), .2)])]
        with patch.dict(os.environ, WOTR_REWARD_CONTRACT='predecision_profile_v1', WOTR_LOCAL_SCORER_BATCH='1'), \
             patch.object(ai, 'WOTRValueNet', LiteralNet), patch.object(ai, 'DEVICE', torch.device('cuda')), \
             patch.object(ai, 'INFER_CLIENT', None):
            for output in invalid:
                with self.subTest(output=repr(output)), \
                     patch('inference_batch.spatial_policy_value_batch', return_value=output):
                    with self.assertRaises(ValueError):
                        ai._pv_forward_many(self.net, self.vecs, self.aes, self.ris)

    def test_gpu_helper_failure_is_not_silently_replaced(self):
        import ai
        with patch.dict(os.environ, WOTR_REWARD_CONTRACT='predecision_profile_v1', WOTR_LOCAL_SCORER_BATCH='1'), \
             patch.object(ai, 'WOTRValueNet', LiteralNet), patch.object(ai, 'DEVICE', torch.device('cuda')), \
             patch.object(ai, 'INFER_CLIENT', None), \
             patch('inference_batch.spatial_policy_value_batch', side_effect=RuntimeError('scorer failed')):
            with self.assertRaisesRegex(RuntimeError, 'scorer failed'):
                ai._pv_forward_many(self.net, self.vecs, self.aes, self.ris)


if __name__ == '__main__':
    unittest.main(verbosity=2)
