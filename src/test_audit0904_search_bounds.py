"""Corrected reward contracts bound search leaves, not differentiable nets."""
import contextlib
import os
import unittest
from unittest.mock import patch

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

import numpy as np
import torch
import ai
import board
import game_state as gs
from reward_contract import LEGACY, WINLOSS, PROFILE


class RawNet(torch.nn.Module):
    """Known network values beyond the corrected terminal range."""
    def __init__(self, value, *, client=None, broken=False):
        super().__init__()
        self.raw = torch.nn.Parameter(torch.tensor(float(value)))
        self.broken = broken
        self._client = client
        if client is not None:
            self.remote_key = 'test-network'

    def forward(self, x):
        if self.broken:
            raise RuntimeError('injected inference failure')
        return self.raw.expand(len(x), 1)

    def forward_backbone(self, x):
        if self.broken:
            raise RuntimeError('injected inference failure')
        return x, x.unsqueeze(1)

    def value_head(self, backbone):
        return self.raw.expand(len(backbone), 1)

    def score_actions_spatial(self, backbone, nodes, encs, regions):
        return torch.full((len(encs),), 1 / len(encs), device=encs.device)


class RawClient:
    def __init__(self, value, *, broken_batch=False, broken_all=False):
        self.value = value
        self.broken_batch = broken_batch
        self.broken_all = broken_all
        self.calls = []

    def pv(self, key, vec, encs, regions):
        self.calls.append('pv')
        if self.broken_all:
            raise RuntimeError('injected remote failure')
        return np.full(len(encs), 1 / len(encs)), self.value

    def pv_many(self, key, items):
        self.calls.append('pv_many')
        if self.broken_all or self.broken_batch:
            raise RuntimeError('injected remote batch failure')
        return [(np.full(len(encs), 1 / len(encs)), self.value)
                for _, encs, _ in items]

    def eval(self, key, vecs):
        self.calls.append('eval')
        if self.broken_all:
            raise RuntimeError('injected remote value failure')
        return np.full(len(vecs), self.value)


class SearchValueBoundsTests(unittest.TestCase):
    def setUp(self):
        self.state = gs.setup_initial_game(board.load_board())
        self.actions = [ai.Action(-1, 'pass', {})]
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        for name, value in (('DEVICE', torch.device('cpu')),
                            ('INFER_CLIENT', None), ('_CONF_LEAF', False),
                            ('_CONF_TRACK', False), ('_PHI_RINGS_RHO', 0)):
            self.stack.enter_context(patch.object(ai, name, value))
        self.stack.enter_context(patch.dict(ai.PLAYER_PROFILES,
                                            {'fp': None, 'sh': None}))

    def contracts_and_values(self):
        for contract in (WINLOSS, PROFILE):
            for raw, expected in ((1.8, 1.0), (-1.8, -1.0), (.2, .2)):
                yield contract, raw, expected

    def assert_value(self, actual, expected):
        self.assertAlmostEqual(actual, expected, places=6)
        self.assertGreaterEqual(actual, -1)
        self.assertLessEqual(actual, 1)

    def test_local_priors_and_side_value_obey_corrected_range(self):
        for contract, raw, expected in self.contracts_and_values():
            with self.subTest(contract=contract, raw=raw), patch.dict(
                    os.environ, WOTR_REWARD_CONTRACT=contract):
                net = RawNet(raw)
                for side in gs.Side:
                    priors, value = ai._get_priors_and_value(
                        self.state, side, net, self.actions)
                    self.assertEqual(priors, [1.0])
                    self.assert_value(value, expected)
                    self.assert_value(ai._mcts_eval_state(
                        self.state, side, net, net), expected)

    def test_remote_priors_and_side_value_obey_corrected_range(self):
        for contract, raw, expected in self.contracts_and_values():
            with self.subTest(contract=contract, raw=raw), patch.dict(
                    os.environ, WOTR_REWARD_CONTRACT=contract):
                client = RawClient(raw)
                net = RawNet(-.4, client=client)
                for side in gs.Side:
                    self.assert_value(ai._get_priors_and_value(
                        self.state, side, net, self.actions)[1], expected)
                    self.assert_value(ai._mcts_eval_state(
                        self.state, side, net, net), expected)
                self.assertEqual(client.calls, ['pv', 'eval', 'pv', 'eval'])

    def test_global_remote_client_uses_the_same_bounds(self):
        with patch.dict(os.environ, WOTR_REWARD_CONTRACT=PROFILE):
            client = RawClient(1.8)
            net = RawNet(-.4)
            net.remote_key = 'global-network'
            with patch.object(ai, 'INFER_CLIENT', client):
                self.assert_value(ai._get_priors_and_value(
                    self.state, gs.Side.SHADOW, net, self.actions)[1], 1)
                self.assert_value(ai._mcts_eval_state(
                    self.state, gs.Side.SHADOW, net, net), 1)
            self.assertEqual(client.calls, ['pv', 'eval'])

    def test_provided_model_failures_propagate_without_heuristic_substitution(self):
        for contract, raw, expected in self.contracts_and_values():
            with self.subTest(contract=contract, raw=raw), patch.dict(
                    os.environ, WOTR_REWARD_CONTRACT=contract), patch.object(
                    ai, 'state_value_fp', return_value=.4) as heuristic:
                for net in (RawNet(raw, client=RawClient(100, broken_all=True)),
                            RawNet(raw, broken=True)):
                    with self.assertRaisesRegex(RuntimeError, 'injected'):
                        ai._get_priors_and_value(
                            self.state, gs.Side.FREE_PEOPLES, net, self.actions)
                    with self.assertRaisesRegex(RuntimeError, 'injected'):
                        ai._mcts_eval_state(self.state, gs.Side.FREE_PEOPLES, net, net)
                heuristic.assert_not_called()

    def test_explicit_missing_net_heuristic_values_are_bounded(self):
        for contract, raw, expected in self.contracts_and_values():
            with self.subTest(contract=contract, raw=raw), patch.dict(
                    os.environ, WOTR_REWARD_CONTRACT=contract), patch.object(
                    ai, 'state_value_fp', return_value=raw):
                for side, sign in ((gs.Side.FREE_PEOPLES, 1), (gs.Side.SHADOW, -1)):
                    self.assert_value(ai._get_priors_and_value(
                        self.state, side, None, self.actions)[1], sign * expected)
                    self.assert_value(ai._mcts_eval_state(
                        self.state, side, None, None), sign * expected)
                    root, leaf = self.flush_one(None, root_side=side)
                    self.assert_value(root.value_sum, sign * expected)

    def flush_one(self, net, *, root_side, opposite=False, root_net=None):
        leaf_side = (gs.Side.SHADOW if root_side == gs.Side.FREE_PEOPLES
                     else gs.Side.FREE_PEOPLES) if opposite else root_side
        root = ai.MCTSNode(self.state, root_side)
        leaf = ai.MCTSNode(self.state, leaf_side, parent=root)
        # One path already carries the real batching lane's virtual loss.
        root.value_sum = -1
        leaf.value_sum = 1 if opposite else -1
        root.visit_count = leaf.visit_count = 1
        coll = ai._PendingEvals()
        with patch.object(ai, '_PENDING_EVAL_BOX', [coll]):
            ai._defer_expand(leaf, self.actions, net, root_side, None)
        coll.recs[0]['paths'].append([root, leaf])
        root_net = net if root_net is None else root_net
        fp = net if leaf_side == gs.Side.FREE_PEOPLES else root_net
        sh = net if leaf_side == gs.Side.SHADOW else root_net
        ai._flush_pending(coll, fp, sh, root_side)
        self.assertEqual(len(leaf.children), 1)
        return root, leaf

    def test_local_batch_backups_and_variance_obey_bounds(self):
        for contract, raw, expected in self.contracts_and_values():
            with self.subTest(contract=contract, raw=raw), patch.dict(
                    os.environ, WOTR_REWARD_CONTRACT=contract):
                for side in gs.Side:
                    root, leaf = self.flush_one(RawNet(raw), root_side=side)
                    self.assert_value(root.value_sum, expected)
                    self.assert_value(leaf.value_sum, expected)
                    self.assertAlmostEqual(root.value_sq, expected ** 2, places=6)
                    self.assertAlmostEqual(leaf.value_sq, expected ** 2, places=6)

    def test_remote_batch_opponent_uses_bounded_root_net_and_sign(self):
        for contract, raw, expected in self.contracts_and_values():
            with self.subTest(contract=contract, raw=raw), patch.dict(
                    os.environ, WOTR_REWARD_CONTRACT=contract):
                # Root evaluation must come from the root's own network,
                # not the opposing leaf's conflicting value prediction.
                client = RawClient(raw)
                for side in gs.Side:
                    root, leaf = self.flush_one(
                        RawNet(-raw, client=RawClient(-raw)), root_side=side,
                        opposite=True, root_net=RawNet(.3, client=client))
                    self.assert_value(root.value_sum, expected)
                    self.assert_value(leaf.value_sum, -expected)
                    self.assertAlmostEqual(root.value_sq, expected ** 2, places=6)
                self.assertEqual(client.calls, ['eval', 'eval'])

    def test_failed_batch_propagates_in_corrected_modes(self):
        for contract, raw, expected in self.contracts_and_values():
            with self.subTest(contract=contract, raw=raw), patch.dict(
                    os.environ, WOTR_REWARD_CONTRACT=contract):
                net = RawNet(raw, client=RawClient(raw, broken_batch=True))
                with self.assertRaisesRegex(RuntimeError, 'batch failure'):
                    self.flush_one(net, root_side=gs.Side.FREE_PEOPLES)
                self.assertEqual(net._client.calls, ['pv_many'])
                with patch.object(ai, '_eval_forward_many',
                                  side_effect=RuntimeError('batch value failure')):
                    with self.assertRaisesRegex(RuntimeError, 'batch value failure'):
                        self.flush_one(RawNet(raw), root_side=gs.Side.SHADOW,
                                       opposite=True, root_net=RawNet(raw))

    def test_legacy_values_and_profile_shaping_remain_unclipped(self):
        with patch.dict(os.environ, WOTR_REWARD_CONTRACT=LEGACY):
            for raw in (1.8, -1.8):
                for net in (RawNet(raw), RawNet(0, client=RawClient(raw))):
                    self.assertAlmostEqual(ai._get_priors_and_value(
                        self.state, gs.Side.FREE_PEOPLES, net, self.actions)[1], raw)
                    self.assertAlmostEqual(ai._mcts_eval_state(
                        self.state, gs.Side.FREE_PEOPLES, net, net), raw)
                    root, leaf = self.flush_one(net, root_side=gs.Side.FREE_PEOPLES)
                    self.assertAlmostEqual(root.value_sum, raw)
                with patch.object(ai, 'state_value_fp', return_value=raw):
                    self.assertEqual(ai._mcts_eval_state(
                        self.state, gs.Side.FREE_PEOPLES, None, None), raw)
            with patch.object(ai, '_profile_shaping', return_value=.3), patch.dict(
                    ai.PLAYER_PROFILES, {'fp': 'ring'}):
                self.assertAlmostEqual(ai._get_priors_and_value(
                    self.state, gs.Side.FREE_PEOPLES, RawNet(1.8), self.actions)[1], 2.1)

    def test_legacy_supplied_model_and_batch_failures_keep_existing_fallbacks(self):
        with patch.dict(os.environ, WOTR_REWARD_CONTRACT=LEGACY), patch.object(
                ai, 'state_value_fp', return_value=.4):
            broken = RawNet(0, broken=True)
            self.assertEqual(ai._get_priors_and_value(
                self.state, gs.Side.FREE_PEOPLES, broken, self.actions)[1], .4)
            self.assertEqual(ai._mcts_eval_state(
                self.state, gs.Side.FREE_PEOPLES, broken, broken), .4)
            net = RawNet(1.8, client=RawClient(0, broken_all=True))
            self.assertAlmostEqual(ai._get_priors_and_value(
                self.state, gs.Side.FREE_PEOPLES, net, self.actions)[1], 1.8)
            self.assertAlmostEqual(ai._mcts_eval_state(
                self.state, gs.Side.FREE_PEOPLES, net, net), 1.8)
            batch = RawNet(.2, client=RawClient(1.8, broken_batch=True))
            root, leaf = self.flush_one(batch, root_side=gs.Side.FREE_PEOPLES)
            self.assertAlmostEqual(root.value_sum, 1.8)
            self.assertEqual(batch._client.calls, ['pv_many', 'pv'])
            with patch.object(ai, '_eval_forward_many',
                              side_effect=RuntimeError('batch value failure')):
                root, leaf = self.flush_one(RawNet(.2), root_side=gs.Side.SHADOW,
                                            opposite=True, root_net=RawNet(1.8))
            self.assertAlmostEqual(root.value_sum, 1.8)
            self.assertAlmostEqual(leaf.value_sum, -1.8)

    def test_corrected_search_does_not_clamp_raw_training_forward_or_gradients(self):
        for contract in (WINLOSS, PROFILE):
            with self.subTest(contract=contract), patch.dict(
                    os.environ, WOTR_REWARD_CONTRACT=contract):
                net = RawNet(1.8)
                self.assert_value(ai._mcts_eval_state(
                    self.state, gs.Side.FREE_PEOPLES, net, net), 1)
                # A differentiable training call still sees the overshoot,
                # and its gradient pushes the seed back toward the target.
                raw = net(torch.zeros(1, 3))
                self.assertAlmostEqual(raw.item(), 1.8)
                ((raw - 1) ** 2).mean().backward()
                self.assertAlmostEqual(net.raw.grad.item(), 1.6, places=6)

    def test_nonfinite_predictions_raise_instead_of_using_plausible_fallbacks(self):
        for contract in (WINLOSS, PROFILE):
            for raw in (float('nan'), float('inf'), -float('inf')):
                with self.subTest(contract=contract, raw=raw), patch.dict(
                        os.environ, WOTR_REWARD_CONTRACT=contract), patch.object(
                        ai, 'state_value_fp', return_value=.4) as heuristic:
                    for net in (RawNet(raw), RawNet(.3, client=RawClient(raw))):
                        with self.assertRaisesRegex(ValueError, 'Non-finite search value'):
                            ai._get_priors_and_value(
                                self.state, gs.Side.FREE_PEOPLES, net, self.actions)
                        with self.assertRaisesRegex(ValueError, 'Non-finite search value'):
                            ai._mcts_eval_state(
                                self.state, gs.Side.FREE_PEOPLES, net, net)
                    heuristic.assert_not_called()

    def test_nonfinite_batched_and_heuristic_values_raise(self):
        for contract in (WINLOSS, PROFILE):
            for raw in (float('nan'), float('inf'), -float('inf')):
                with self.subTest(contract=contract, raw=raw), patch.dict(
                        os.environ, WOTR_REWARD_CONTRACT=contract):
                    for net in (RawNet(raw), RawNet(.3, client=RawClient(raw))):
                        with self.assertRaisesRegex(ValueError, 'Non-finite search value'):
                            self.flush_one(net, root_side=gs.Side.FREE_PEOPLES)
                        with self.assertRaisesRegex(ValueError, 'Non-finite search value'):
                            self.flush_one(RawNet(.4), root_side=gs.Side.SHADOW,
                                           opposite=True, root_net=net)
                        with self.assertRaisesRegex(ValueError, 'Non-finite search value'):
                            self.flush_one(net, root_side=gs.Side.SHADOW,
                                           opposite=True, root_net=RawNet(.4))
                    with patch.object(ai, 'state_value_fp', return_value=raw):
                        with self.assertRaisesRegex(ValueError, 'Non-finite search value'):
                            ai._mcts_eval_state(
                                self.state, gs.Side.FREE_PEOPLES, None, None)
                        with self.assertRaisesRegex(ValueError, 'Non-finite search value'):
                            ai._get_priors_and_value(
                                self.state, gs.Side.FREE_PEOPLES,
                                None, self.actions)


if __name__ == '__main__':
    unittest.main(verbosity=2)
