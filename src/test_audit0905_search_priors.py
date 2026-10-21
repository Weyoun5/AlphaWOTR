"""Literal malformed model outputs at search inference boundaries; no training."""
import contextlib
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import ai
import board
import game_state as gs
from reward_contract import LEGACY, PROFILE, WINLOSS

FP = gs.Side.FREE_PEOPLES
BAD = {
    'nan': [float('nan'), .5],
    'infinity': [float('inf'), .5],
    'negative': [-.2, 1.2],
    'all_negative': [-1., -3.],
    'zero_mass': [0., 0.],
    'tiny_positive_mass': [1e-40, 3e-40],
    'short': [1.],
    'long': [.2, .3, .5],
    'matrix': [[.25, .75]],
    'scalar': .5,
    'empty': [],
}


class LiteralNet(torch.nn.Module):
    def __init__(self, prior, client=None, broken=False):
        super().__init__()
        self.prior = prior
        self.broken = broken
        self._client = client
        if client is not None:
            self.remote_key = 'literal-policy'

    def forward_backbone(self, x):
        return x, x.unsqueeze(1)

    def value_head(self, x):
        return torch.full((len(x), 1), .25, device=x.device)

    def score_actions_spatial(self, backbone, nodes, encs, regions):
        if self.broken:
            raise RuntimeError('literal scorer failure')
        return torch.tensor(self.prior, dtype=torch.float32, device=encs.device)


class LiteralClient:
    def __init__(self, prior, replies=None):
        self.prior = prior
        self.replies = replies
        self.pv_calls = 0
        self.batch_calls = 0

    def pv(self, key, vec, encs, regions):
        self.pv_calls += 1
        return np.asarray(self.prior, dtype=np.float32), .25

    def pv_many(self, key, items):
        self.batch_calls += 1
        if self.replies is not None:
            return self.replies
        return [(np.asarray(self.prior, dtype=np.float32), .25) for _ in items]


class SearchPolicyIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.state = gs.setup_initial_game(board.load_board())
        die = self.state.dice[FP][0]
        die.current = gs.ActionDieFace.MUSTER
        die.used = False
        self.state.dice[FP] = [die]
        self.actions = [ai.Action(-1, 'elven_ring', dict(side='fp', target_die_idx=0,
            old_face='Muster', new_face=face)) for face in ('Character', 'Event')]
        self.vec = ai.encode_state(self.state, FP)
        self.ae, self.ri = ai._encode_actions_batch(self.actions, self.state, FP, {})
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        for key, value in [('DEVICE', torch.device('cpu')), ('INFER_CLIENT', None),
                           ('_CONF_LEAF', False), ('_CONF_TRACK', False),
                           ('_PHI_RINGS_RHO', 0), ('_SIM_OVERLAY', None)]:
            self.stack.enter_context(patch.object(ai, key, value))
        self.stack.enter_context(patch.dict(ai.PLAYER_PROFILES, fp=None, sh=None))
        self.stack.enter_context(patch.object(ai, '_conf_shrink', side_effect=lambda v, s, x: x))

    def net(self, prior, remote):
        return LiteralNet(prior, LiteralClient(prior) if remote else None)

    def pending(self, net, count=2):
        coll = ai._PendingEvals()
        nodes = []
        with patch.object(ai, '_PENDING_EVAL_BOX', [coll]):
            for _ in range(count):
                node = ai.MCTSNode(self.state, FP)
                node.value_sum, node.visit_count = -1., 1  # existing virtual loss
                ai._defer_expand(node, self.actions, net, FP, None)
                coll.recs[-1]['paths'].append([node])
                nodes.append(node)
        return coll, nodes

    def assert_rejected_without_publication(self, coll, nodes, net):
        try:
            with self.assertRaises(ValueError):
                ai._flush_pending(coll, net, net, FP)
        finally:
            for node in nodes:
                self.assertEqual(node.children, [], 'invalid batch published children')
                self.assertEqual(node.value_sum, -1., 'invalid batch published a backup')

    def test_local_and_remote_synchronous_priors_reject_malformed_vectors(self):
        for contract in (PROFILE, WINLOSS):
            with patch.dict(os.environ, WOTR_REWARD_CONTRACT=contract):
                for remote in (False, True):
                    for name, prior in BAD.items():
                        with self.subTest(contract=contract, remote=remote, bad=name):
                            with self.assertRaises(ValueError):
                                ai._get_priors_and_value(self.state, FP, self.net(prior, remote), self.actions)

    def test_global_remote_route_rejects_invalid_policy(self):
        client = LiteralClient([float('nan'), .5])
        net = LiteralNet([.25, .75]); net.remote_key = 'global-key'
        with patch.dict(os.environ, WOTR_REWARD_CONTRACT=PROFILE), \
                patch.object(ai, 'INFER_CLIENT', client), self.assertRaises(ValueError):
            ai._get_priors_and_value(self.state, FP, net, self.actions)

    def test_local_and_remote_batch_forward_reject_malformed_vectors(self):
        with patch.dict(os.environ, WOTR_REWARD_CONTRACT=PROFILE):
            for remote in (False, True):
                for name, prior in BAD.items():
                    with self.subTest(remote=remote, bad=name), self.assertRaises(ValueError):
                        ai._pv_forward_many(self.net(prior, remote), [self.vec]*2,
                                            [self.ae]*2, [self.ri]*2)

    def test_old_remote_client_sequential_batch_path_validates_policy(self):
        client = LiteralClient([-.2, 1.2])
        old_client = SimpleNamespace(pv=client.pv)
        with patch.dict(os.environ, WOTR_REWARD_CONTRACT=WINLOSS), self.assertRaises(ValueError):
            ai._pv_forward_many(LiteralNet(None, old_client), [self.vec]*2,
                                [self.ae]*2, [self.ri]*2)

    def test_remote_batch_reply_count_must_equal_request_count(self):
        with patch.dict(os.environ, WOTR_REWARD_CONTRACT=PROFILE):
            for n in (0, 1, 3):
                client = LiteralClient([.25, .75], replies=[(np.array([.25, .75]), .25)]*n)
                with self.subTest(replies=n), self.assertRaises(ValueError):
                    ai._pv_forward_many(LiteralNet(None, client), [self.vec]*2,
                                        [self.ae]*2, [self.ri]*2)

    def test_batch_request_components_cannot_silently_zip_to_shorter_input(self):
        for remote in (False, True):
            with self.subTest(remote=remote), patch.dict(os.environ, WOTR_REWARD_CONTRACT=PROFILE):
                with self.assertRaises(ValueError):
                    ai._pv_forward_many(self.net([.25, .75], remote), [self.vec]*2,
                                        [self.ae], [self.ri]*2)

    def test_valid_ragged_batch_checks_each_leaf_against_its_own_menu(self):
        ae3 = np.vstack([self.ae, self.ae[:1]])
        ri3 = np.concatenate([self.ri, self.ri[:1]])
        client = LiteralClient(None, replies=[(np.array([.25, .75]), .25),
                                              (np.array([.2, .3, .5]), .25)])
        with patch.dict(os.environ, WOTR_REWARD_CONTRACT=PROFILE):
            out = ai._pv_forward_many(LiteralNet(None, client), [self.vec]*2,
                                      [self.ae, ae3], [self.ri, ri3])
        np.testing.assert_allclose(out[0][0], [.25, .75])
        np.testing.assert_allclose(out[1][0], [.2, .3, .5])

    def test_value_only_batches_also_require_one_result_per_request(self):
        for count in (0, 1, 3):
            for remote in (False, True):
                with self.subTest(count=count, remote=remote), patch.dict(
                        os.environ, WOTR_REWARD_CONTRACT=PROFILE):
                    net = LiteralNet([.25, .75])
                    if remote:
                        net.remote_key = 'values'
                        net._client = SimpleNamespace(eval=lambda *a: np.full(count, .25))
                    else:
                        net.forward = lambda x: torch.full((count, 1), .25)
                    with self.assertRaises(ValueError):
                        ai._eval_forward_many(net, [self.vec]*2)

    def test_local_joint_forward_checks_value_count_before_returning_batch(self):
        for count in (1, 3):
            with self.subTest(count=count), patch.dict(os.environ, WOTR_REWARD_CONTRACT=WINLOSS):
                net = LiteralNet([.25, .75])
                net.value_head = lambda x: torch.full((count, 1), .25)
                with self.assertRaises(ValueError):
                    ai._pv_forward_many(net, [self.vec]*2, [self.ae]*2, [self.ri]*2)

    def test_flush_checks_batch_boundary_even_when_forward_helper_is_replaced(self):
        for reply in [[(np.array([.25, .75]), .25)],
                      [(np.array([.25, .75]), .25), (np.array([1.]), .25)]]:
            with self.subTest(reply=reply), patch.dict(os.environ, WOTR_REWARD_CONTRACT=PROFILE):
                net = LiteralNet([.25, .75])
                coll, nodes = self.pending(net)
                with patch.object(ai, '_pv_forward_many', return_value=reply):
                    self.assert_rejected_without_publication(coll, nodes, net)

    def test_flush_rejects_bad_second_reply_before_any_tree_publication(self):
        for name in ('nan', 'negative', 'zero_mass', 'tiny_positive_mass', 'short', 'matrix'):
            with self.subTest(bad=name), patch.dict(os.environ, WOTR_REWARD_CONTRACT=PROFILE):
                client = LiteralClient([.25, .75], replies=[(np.array([.25, .75]), .25),
                                       (np.asarray(BAD[name]), .25)])
                net = LiteralNet(None, client)
                coll, nodes = self.pending(net)
                self.assert_rejected_without_publication(coll, nodes, net)

    def test_flush_does_not_disguise_missing_remote_replies_as_sync_success(self):
        client = LiteralClient([.25, .75], replies=[])
        net = LiteralNet(None, client)
        with patch.dict(os.environ, WOTR_REWARD_CONTRACT=WINLOSS):
            coll, nodes = self.pending(net)
            self.assert_rejected_without_publication(coll, nodes, net)
            self.assertEqual(client.pv_calls, 0, 'protocol error silently fell back to sync')

    def test_valid_local_remote_and_batched_policy_preserves_alignment_and_backup(self):
        for contract in (PROFILE, WINLOSS):
            for remote in (False, True):
                with self.subTest(contract=contract, remote=remote), patch.dict(
                        os.environ, WOTR_REWARD_CONTRACT=contract):
                    net = self.net([.25, .75], remote)
                    priors, value = ai._get_priors_and_value(self.state, FP, net, self.actions)
                    np.testing.assert_allclose(priors, [.25, .75], rtol=1e-6)
                    self.assertEqual(value, .25)
                    coll, nodes = self.pending(net)
                    ai._flush_pending(coll, net, net, FP)
                    for node in nodes:
                        self.assertEqual([c.action for c in node.children], self.actions)
                        np.testing.assert_allclose([c.prior for c in node.children], [.25, .75], rtol=1e-6)
                        self.assertEqual(node.value_sum, .25)

    def test_explicit_missing_net_still_uses_uniform_policy(self):
        for contract in (PROFILE, WINLOSS):
            with patch.dict(os.environ, WOTR_REWARD_CONTRACT=contract), \
                    patch.object(ai, 'state_value_fp', return_value=.25):
                priors, value = ai._get_priors_and_value(self.state, FP, None, self.actions)
                np.testing.assert_allclose(priors, [.5, .5])
                coll, nodes = self.pending(None)
                ai._flush_pending(coll, None, None, FP)
                for node in nodes:
                    np.testing.assert_allclose([c.prior for c in node.children], [.5, .5])

    def test_legacy_scorer_failure_keeps_existing_uniform_fallback(self):
        with patch.dict(os.environ, WOTR_REWARD_CONTRACT=LEGACY), \
                patch.object(ai, 'state_value_fp', return_value=.25):
            priors, value = ai._get_priors_and_value(self.state, FP,
                LiteralNet(None, broken=True), self.actions)
            np.testing.assert_allclose(priors, [.5, .5])
            self.assertEqual(value, .25)

    def test_legacy_negative_outputs_are_not_silently_redefined_by_corrected_gate(self):
        with patch.dict(os.environ, WOTR_REWARD_CONTRACT=LEGACY):
            priors, _ = ai._get_priors_and_value(self.state, FP, LiteralNet([-1., -3.]), self.actions)
            np.testing.assert_allclose(priors, [.25, .75])

    def test_legacy_extra_reply_behavior_is_preserved(self):
        replies = [(np.array([.25, .75]), .25)]*3
        with patch.dict(os.environ, WOTR_REWARD_CONTRACT=LEGACY):
            client = LiteralClient(None, replies=replies)
            out = ai._pv_forward_many(LiteralNet(None, client), [self.vec]*2,
                                      [self.ae]*2, [self.ri]*2)
            self.assertEqual(len(out), 3)

    def test_legacy_tiny_mass_normalization_is_preserved(self):
        with patch.dict(os.environ, WOTR_REWARD_CONTRACT=LEGACY):
            priors, _ = ai._get_priors_and_value(self.state, FP,
                LiteralNet([1e-40, 3e-40]), self.actions)
            self.assertGreater(sum(priors), 0)
            self.assertLess(sum(priors), 1e-20)


if __name__ == '__main__':
    unittest.main()
