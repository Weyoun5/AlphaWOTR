"""Weights-only action conversion, including nonzero suffix and delta controls."""
import copy
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.update(WOTR_ACT_GEN='v45', WOTR_NATIVE='0',
                  WOTR_REWARD_CONTRACT='predecision_profile_v1')
import numpy as np
import torch
import ai
import board
import card_options
import game_state as gs
from warmstart_action_schema import convert_checkpoint, convert_file, EXPANDED_PARAMETERS


class ActionWarmStart(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(945)
        with patch.object(ai, 'ACTION_ENCODING_DIM', 93):
            cls.old = ai.WOTRValueNet(3746, side='sh').eval()
        with torch.no_grad():
            cls.old.policy_delta[2].weight.fill_(.07)
            for index, name in enumerate(EXPANDED_PARAMETERS):
                matrix = cls.old.state_dict()[name]
                suffix = cls.old.card_embedding.embedding_dim
                matrix[:, -suffix:] = torch.arange(1, suffix + 1) / (suffix * (index + 2))
        cls.source = dict(model_state=copy.deepcopy(cls.old.state_dict()), input_dim=3746,
                          action_dim=93, side='sh', arch_name=ai.NET_ARCH_NAME,
                          head_act='scaled2', reward_contract='predecision_profile_v1',
                          reward_profile='military', observation_contract='v44_fellowship_tail_v1',
                          action_encoding_contract='casualty_operations_v2',
                          training_status='updated_after_warm_start', warm_start_updates={'value': 2, 'policy': 20},
                          warm_start={'source_sha256': 'earlier', 'nested': {'untouched': True}},
                          optimizer={'state': 'old'}, replay=[1], validation_score=.99)

    def test_exact_splice_all_other_tensors_and_nested_provenance(self):
        converted = convert_checkpoint(self.source, 'a' * 64)
        suffix = self.old.card_embedding.embedding_dim
        for name, old in self.source['model_state'].items():
            new = converted['model_state'][name]
            if name in EXPANDED_PARAMETERS:
                cut = old.shape[1] - suffix
                torch.testing.assert_close(new[:, :cut], old[:, :cut], rtol=0, atol=0)
                torch.testing.assert_close(new[:, cut + 12:], old[:, cut:], rtol=0, atol=0)
                self.assertEqual(torch.count_nonzero(new[:, cut:cut + 12]).item(), 0)
            elif name == 'backbone.0.weight':
                torch.testing.assert_close(new[:, :old.shape[1]], old, rtol=0, atol=0)
                self.assertEqual(torch.count_nonzero(new[:, old.shape[1]:]).item(), 0)
            else:
                torch.testing.assert_close(new, old, rtol=0, atol=0)
            self.assertNotEqual(new.data_ptr(), old.data_ptr())
        self.assertEqual(converted['warm_start_updates'], {'value': 0, 'policy': 0})
        self.assertEqual(converted['training_status'], 'warm_start_not_retrained')
        self.assertEqual(converted['warm_start']['source_warm_start_updates'], {'value': 2, 'policy': 20})
        self.assertEqual(converted['warm_start']['parent_warm_start'], self.source['warm_start'])
        converted['warm_start']['parent_warm_start']['nested']['untouched'] = False
        self.assertTrue(self.source['warm_start']['nested']['untouched'])
        for field in ('optimizer', 'replay', 'validation_score', 'selector_contract'):
            self.assertNotIn(field, converted)

    def test_source_training_scope_preserved_without_target_relabel(self):
        source = copy.deepcopy(self.source)
        width = source['model_state']['backbone.0.weight'].shape[1]
        source['training_scope'] = {
            'version': 'observation_training_scope_v1',
            'observation_contract': 'v44_fellowship_tail_v1', 'input_dim': 3746,
            'modes': {'appended_columns_v1': {
                'value_steps': 2, 'policy_steps': 20,
                'value_projection_changes': 1, 'policy_projection_changes': 17,
                'parameter': 'backbone.0.weight', 'column_start': width - 4,
                'column_end': width, 'raw_start': 3742, 'raw_end': 3746}}}
        converted = convert_checkpoint(source, 'c' * 64)
        self.assertEqual(converted['warm_start']['source_training_scope'], source['training_scope'])
        self.assertNotIn('training_scope', converted)
        self.assertEqual(converted['warm_start_updates'], {'value': 0, 'policy': 0})
        self.assertEqual(converted['training_status'], 'warm_start_not_retrained')
        converted['warm_start']['source_training_scope']['modes']['appended_columns_v1']['policy_steps'] = 0
        self.assertEqual(source['training_scope']['modes']['appended_columns_v1']['policy_steps'], 20)
        for field, value in [('parameter', 'tok_global.weight'), ('column_start', width - 3),
                             ('policy_projection_changes', 21), ('raw_end', 4180)]:
            bad = copy.deepcopy(source)
            bad['training_scope']['modes']['appended_columns_v1'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                convert_checkpoint(bad, 'c' * 64)
        bad = copy.deepcopy(source)
        bad['training_scope']['modes']['appended_columns_v1'].update(column_start=width - 3, column_end=width + 1)
        with self.assertRaisesRegex(ValueError, 'projection layout'):
            convert_checkpoint(bad, 'c' * 64)

    def test_real_public_card_menu_forward_with_nonzero_tail_and_delta(self):
        converted = convert_checkpoint(self.source, 'b' * 64)
        net = ai.WOTRValueNet(4180, side='sh').eval()
        net.load_state_dict(converted['model_state'])
        state = gs.setup_initial_game(board.load_board())
        options = card_options.get_card_options(state, gs.Side.SHADOW, 'Pits of Mordor')
        actions = [ai.Action(-1, 'card_option', {'card_id': 'SHADOW_Strategy_Pits_of_Mordor', 'choices': option}) for option in options]
        enc = np.asarray([ai.encode_action(action, state, gs.Side.SHADOW) for action in actions])
        self.assertEqual(enc.shape[1], 105)
        self.assertGreater(np.count_nonzero(enc[:, 93:]), 0)
        x = torch.from_numpy(ai.encode_state(state, gs.Side.SHADOW)[None])
        x[:, 3746:] = torch.linspace(.01, .99, 434)
        regions = torch.tensor([ai._get_action_region_index(action) for action in actions])
        with torch.no_grad():
            old_h, old_nodes = self.old.forward_backbone(x[:, :3746])
            new_h, new_nodes = net.forward_backbone(x)
            torch.testing.assert_close(self.old(x[:, :3746]), net(x), rtol=2e-6, atol=2e-6)
            for delta in (False, True):
                expected = self.old.action_scores(old_h, old_nodes, torch.from_numpy(enc[:, :93]), regions, with_delta=delta)
                actual = net.action_scores(new_h, new_nodes, torch.from_numpy(enc), regions, with_delta=delta)
                torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
        wrong = copy.deepcopy(converted['model_state'])
        for name in EXPANDED_PARAMETERS:
            old = self.source['model_state'][name]
            wrong[name] = torch.cat((old, old.new_zeros((old.shape[0], 12))), dim=1)
        net.load_state_dict(wrong)
        with torch.no_grad():
            wrong_scores = net.action_scores(new_h, new_nodes, torch.from_numpy(enc), regions, with_delta=True)
        self.assertFalse(torch.allclose(wrong_scores, expected, atol=2e-6, rtol=2e-6))

    def test_missing_wrong_or_already_new_contract_refused(self):
        changes = [dict(observation_contract=None), dict(action_encoding_contract=None),
                   dict(action_encoding_contract='casualty_card_choices_v3'), dict(action_dim=105),
                   dict(input_dim=3745), dict(action_dim=True), dict(side=None), dict(head_act='unknown')]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                convert_checkpoint(dict(self.source, **change), 'c' * 64)
        for digest in ('', 'A' * 64, 'z' * 64):
            with self.subTest(digest=digest), self.assertRaises(ValueError):
                convert_checkpoint(self.source, digest)

    def test_incompatible_target_generation_refused(self):
        with patch.object(ai, 'ACT_GEN', 'v41'), self.assertRaisesRegex(ValueError, 'v45'):
            convert_checkpoint(self.source, 'c' * 64)

    def test_all_shapes_keys_and_finiteness_checked_before_copy(self):
        for mode in ('missing', 'extra', 'short_scorer', 'changed_value', 'nonfinite'):
            state = dict(self.source['model_state'])
            if mode == 'missing':
                state.pop('policy_delta.2.bias')
            elif mode == 'extra':
                state['optimizer_tensor'] = torch.zeros(1)
            elif mode == 'short_scorer':
                state['spatial_action_scorer.0.weight'] = state['spatial_action_scorer.0.weight'][:, :-1]
            elif mode == 'changed_value':
                state['value_head.0.weight'] = state['value_head.0.weight'][:-1]
            else:
                state['policy_delta.2.bias'] = torch.full_like(state['policy_delta.2.bias'], float('nan'))
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                convert_checkpoint(dict(self.source, model_state=state), 'd' * 64)
        with self.assertRaises(ValueError):
            convert_checkpoint(dict(self.source, model_state_dict=self.source['model_state']), 'd' * 64)

    def test_architecture_stamp_mismatch_refused(self):
        other = 'base' if ai.NET_ARCH_NAME == 'xformer' else 'xformer'
        with self.assertRaisesRegex(ValueError, 'architecture'):
            convert_checkpoint(dict(self.source, arch_name=other), 'd' * 64)

    def test_exclusive_files_and_old_loader_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.pt'
            destination = Path(directory) / 'new.pt'
            torch.save(self.source, source)
            before = source.read_bytes()
            with self.assertRaisesRegex(ValueError, 'action'):
                ai.load_value_net(str(source), side='sh', profile='military')
            result = convert_file(source, destination)
            self.assertEqual(result['warm_start']['source_sha256'], hashlib.sha256(before).hexdigest())
            after = destination.read_bytes()
            for path in (source, destination):
                with self.assertRaisesRegex(ValueError, 'new file'):
                    convert_file(source, path)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(destination.read_bytes(), after)

    def test_transformer_preserves_nonzero_suffix_and_model_outputs(self):
        with patch.object(ai, 'NET_ARCH_NAME', 'xformer'), patch.object(ai, 'NET_ARCH_CFG', ai._NET_ARCHS['xformer']):
            with patch.object(ai, 'ACTION_ENCODING_DIM', 93):
                old = ai.WOTRValueNet(3746, side='sh').eval()
            with torch.no_grad():
                old.policy_delta[2].weight.fill_(.03)
                for name in EXPANDED_PARAMETERS:
                    old.state_dict()[name][:, -old.card_embedding.embedding_dim:].fill_(.02)
            source = dict(self.source, model_state=old.state_dict(), arch_name='xformer')
            converted = convert_checkpoint(source, 'e' * 64)
            new = ai.WOTRValueNet(4180, side='sh').eval()
            new.load_state_dict(converted['model_state'])
            for name, tensor in old.state_dict().items():
                if name == 'tok_global.weight':
                    matrix = new.state_dict()[name]
                    torch.testing.assert_close(matrix[:, :tensor.shape[1]], tensor, rtol=0, atol=0)
                    self.assertEqual(torch.count_nonzero(matrix[:, tensor.shape[1]:]).item(), 0)
                elif name not in EXPANDED_PARAMETERS:
                    torch.testing.assert_close(new.state_dict()[name], tensor, rtol=0, atol=0)
            state = gs.setup_initial_game(board.load_board())
            x = torch.from_numpy(ai.encode_state(state, gs.Side.SHADOW)[None])
            x[:, 3746:] = torch.linspace(.01, .99, 434)
            action = ai.Action(-1, 'card_option', {'choices': {'regions': ['moria', 'dol guldur']}})
            enc = torch.from_numpy(ai.encode_action(action, state, gs.Side.SHADOW)[None])
            self.assertGreater(torch.count_nonzero(enc[:, 93:]).item(), 0)
            region = torch.tensor([ai._get_action_region_index(action)])
            with torch.no_grad():
                h1, n1 = old.forward_backbone(x[:, :3746])
                h2, n2 = new.forward_backbone(x)
                torch.testing.assert_close(old(x[:, :3746]), new(x), rtol=2e-6, atol=2e-6)
                torch.testing.assert_close(old.action_scores(h1, n1, enc[:, :93], region),
                                           new.action_scores(h2, n2, enc, region), rtol=2e-6, atol=2e-6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
