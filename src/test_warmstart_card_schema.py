"""Seeded checkpoint predictions, learning access, strict lineage and file safety."""
import os
os.environ.setdefault('WOTR_ACT_GEN', 'v52')
os.environ.setdefault('WOTR_NATIVE', '0')
os.environ.setdefault('WOTR_OBSERVATION_TRAINING', 'full_network_v1')
import copy
import hashlib
import io
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
import ai
import optimizer_continuity as oc
import warmstart_card_schema as migration
from reward_contract import current_contract, PROFILE
from trace_decisions import current_learning_context, validate_action_encoding_contract


class Migration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(5237)
        target = ai.WOTRValueNet(4647, side='fp')
        target.observation_contract = ai.OBSERVATION_CONTRACT
        target.action_encoding_contract = current_learning_context()['action_encoding_contract']
        target.reward_contract = current_contract()
        target.reward_profile = 'ring' if current_contract() == PROFILE else None
        ident = oc._identity(target)
        with patch.object(ai, 'ACTION_ENCODING_DIM', 133):
            cls.source_net = ai.WOTRValueNet(4481, side='fp').eval()
        cls.source = {'model_state': cls.source_net.state_dict(), 'arch_name': ai.NET_ARCH_NAME,
            'input_dim': 4481, 'action_dim': 133, 'reward_profile': target.reward_profile,
            **{key: copy.deepcopy(ident[key]) for key in ('side', 'head_act', 'reward_contract',
                'ai_policy_contract', 'selector_contract', 'selector_config', 'selector_config_sha256')},
            'observation_contract': 'v52_observed_combat_round_v2',
            'action_encoding_contract': 'observed_combat_choices_v5'}

    def convert(self, source=None):
        return migration.convert_checkpoint(source or self.source, '1' * 64, reset_optimizer=True)

    def test_compatible_predictions_and_changed_lanes_reset(self):
        for version in ('v5', 'v6'):
            source = dict(self.source, action_encoding_contract='observed_combat_choices_' + version)
            result = self.convert(source)
            target = ai.WOTRValueNet(4647, side='fp').eval()
            target.load_state_dict(result['model_state'], strict=True)
            # Independent compatible subspace: all changing scalar lanes are
            # absent; remaining numeric and real card embeddings are nonzero.
            generator = torch.Generator().manual_seed(902)
            x = torch.rand((1, 4481), generator=generator)
            x[:, ai._CARD_IDX_START:ai._CARD_IDX_END] = torch.tensor([1, 12, 26, 43, -1, -1])
            enc = torch.rand((4, 133), generator=generator)
            enc[:, 33] = torch.tensor([3, 20, -1, 48])
            affected = [ai._ACT_OPTVAL_OFF, ai._ACT_OPTKIND_OFF, *range(93, 101)]
            if version == 'v5':
                enc[:, affected] = 0
            regions = torch.tensor([-1, 0, 17, 73])
            expanded_x = torch.nn.functional.pad(x, (0, 166))
            expanded_enc = torch.nn.functional.pad(enc, (0, 24))
            with torch.no_grad():
                a, h, n = self.source_net.forward_spatial(x)
                b, hh, nn = target.forward_spatial(expanded_x)
                p = self.source_net.action_scores(h, n, enc, regions)
                q = target.action_scores(hh, nn, expanded_enc, regions)
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-6)
            for name in ('spatial_action_scorer.0.weight', 'policy_delta.0.weight'):
                old = source['model_state'][name]
                new = result['model_state'][name]
                card = target.card_embedding.embedding_dim
                start = old.shape[1] - card - 132
                for raw in range(133):
                    if raw == 33:
                        continue
                    index = start + raw - (raw > 33)
                    if version == 'v5' and raw in affected:
                        self.assertEqual(torch.count_nonzero(new[:, index]).item(), 0)
                    else:
                        self.assertTrue(torch.equal(old[:, index], new[:, index]))
                self.assertTrue(torch.equal(old[:, -card:], new[:, -card:]))
                self.assertEqual(torch.count_nonzero(new[:, -card-24:-card]).item(), 0)
            self.assertTrue(torch.equal(source['model_state']['card_embedding.weight'], result['model_state']['card_embedding.weight']))
            # New observations/actions initially have no effect but remain
            # trainable; a real forward/backward reaches their projections.
            expanded_x[:, -166:] = .5
            expanded_enc[:, -24:] = .5
            value, hh, nn = target.forward_spatial(expanded_x)
            (value.sum() + target.action_scores(hh, nn, expanded_enc, regions).sum()).backward()
            projection = target.tok_global if target.arch_kind == 'transformer' else target.backbone[0]
            self.assertGreater(torch.count_nonzero(projection.weight.grad[:, -166:]).item(), 0)
            card = target.card_embedding.embedding_dim
            self.assertGreater(torch.count_nonzero(target.spatial_action_scorer[0].weight.grad[:, -card-24:-card]).item(), 0)

    def test_fresh_adam_bound_and_trainable_scope(self):
        from warmstart_combat_schema import _attach
        from observation_training import ObservationTrainingScope
        result = self.convert()
        net = ai.WOTRValueNet(4647, side='fp')
        net.load_state_dict(result['model_state'])
        _attach(net, result)
        with patch.dict(os.environ, {'WOTR_OPTIMIZER_STATE': oc.VERSION, 'WOTR_OPTIMIZER_ALLOW_FRESH': '0'}):
            oc.restore_checkpoint(net, result)
            self.assertEqual(net._optimizer_continuity['components'], {'value': None, 'policy': None})
        with patch.dict(os.environ, {'WOTR_OBSERVATION_TRAINING': 'full_network_v1'}):
            with ObservationTrainingScope(net):
                self.assertTrue(all(p.requires_grad for p in net.parameters()))
        self.assertEqual(result['warm_start_updates'], {'value': 0, 'policy': 0})
        self.assertFalse(result['warm_start']['historical_rows_migrated'])

    def test_old_data_cannot_be_relabeled_by_migration(self):
        for version in ('v5', 'v6'):
            with self.assertRaises(ValueError):
                validate_action_encoding_contract({'action_encoding_contract': 'observed_combat_choices_' + version}, required=True)
        self.assertEqual(self.source['input_dim'], 4481)
        self.assertEqual(self.source['action_encoding_contract'], 'observed_combat_choices_v5')
        with self.assertRaises(ValueError):
            migration.convert_checkpoint(self.source, '1' * 64)
        for key, value in [('observation_contract', 'future'), ('action_encoding_contract', 'observed_combat_choices_v7'), ('action_dim', 157), ('side', 'invalid')]:
            source = dict(self.source); source[key] = value
            with self.assertRaises(ValueError):
                self.convert(source)
        malformed = dict(self.source, model_state=dict(self.source['model_state']))
        malformed['model_state']['policy_delta.0.weight'] = malformed['model_state']['policy_delta.0.weight'][:, :-1]
        with self.assertRaises(ValueError):
            self.convert(malformed)
        corrupt = dict(self.source, optimizer_continuity={'version': oc.VERSION, 'sha256': '0'*64, 'blob': b'bad'})
        with self.assertRaisesRegex(ValueError, 'optimizer envelope'):
            self.convert(corrupt)

    def test_source_adam_is_explicitly_reset_and_bad_binding_rejected(self):
        # Nonempty old moments must not survive reset. Only their source hash
        # is retained, so stale momentum cannot repopulate zeroed lanes.
        payload = {'version': oc.VERSION, 'model_sha256': oc._model_hash(self.source['model_state']),
                   'record': {'old_moments': torch.ones(7)}}
        stream = io.BytesIO(); torch.save(payload, stream); blob = stream.getvalue()
        source = dict(self.source, optimizer_continuity={'version': oc.VERSION,
                      'sha256': hashlib.sha256(blob).hexdigest(), 'blob': blob})
        result = self.convert(source)
        fresh = torch.load(io.BytesIO(result[oc.FIELD]['blob']), weights_only=True)
        self.assertEqual(fresh['record']['components'], {'value': None, 'policy': None})
        self.assertEqual(fresh['model_sha256'], oc._model_hash(result['model_state']))
        self.assertEqual(result['warm_start']['source_optimizer_sha256'], source[oc.FIELD]['sha256'])
        payload['model_sha256'] = '0'*64
        stream = io.BytesIO(); torch.save(payload, stream); blob = stream.getvalue()
        source[oc.FIELD] = {'version': oc.VERSION, 'sha256': hashlib.sha256(blob).hexdigest(), 'blob': blob}
        with self.assertRaisesRegex(ValueError, 'binding mismatch'):
            self.convert(source)

    def test_source_file_immutable_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder)/'source.pt', Path(folder)/'target.pt'
            torch.save(self.source, source)
            sha = hashlib.sha256(source.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):
                migration.convert_file(source, target, '0'*64, reset_optimizer=True)
            self.assertFalse(target.exists())
            migration.convert_file(source, target, sha, reset_optimizer=True)
            with patch.object(ai, 'DEVICE', torch.device('cpu')):
                loaded = ai.load_value_net(str(target), side='fp', profile=self.source['reward_profile'])
            self.assertEqual(loaded.input_dim, 4647)
            self.assertEqual(loaded.training_status, 'warm_start_not_retrained')
            with self.assertRaises(ValueError):
                ai.load_value_net(str(source), side='fp')
            with self.assertRaises(ValueError):
                migration.convert_file(source, target, sha, reset_optimizer=True)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), sha)


if __name__ == '__main__':
    unittest.main()
