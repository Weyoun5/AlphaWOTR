"""BC admission controls: no optimizer, original corpus, or checkpoint writes."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import train_corpus as tc

OBS = 'v44_fellowship_tail_v1'
POLICY = 'elite_and_bonus_draw_v1'
ACTION = 'casualty_operations_v2'


class CorpusIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.corpus = self.root / 'corpus'
        self.ds = self.root / 'ds'
        self.corpus.mkdir()
        self.ds.mkdir()
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        for name, value in [('CORPUS', str(self.corpus)), ('DS_DIR', str(self.ds)),
                            ('SPLIT_JSON', str(self.ds / 'split.json'))]:
            self.stack.enter_context(mock.patch.object(tc, name, value))
        self.stack.enter_context(mock.patch.object(tc, '_check_phi_provenance'))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def write_game(self, name='one.npz', *, obs=OBS, policy=POLICY, phase=('pass', 'pass')):
        import ai
        enc = np.zeros((2, ai.ACTION_ENCODING_DIM), np.float32)
        enc[:, 10] = 1  # Literal pass-category fixture; no menu regeneration.
        data = dict(X=np.zeros((2, ai.get_feature_dim()), np.float32),
                    side=np.array([0, 1], np.uint8), offsets=np.array([0, 1, 2]),
                    human=np.array([0, 0]), z=np.array([1, -1]), phi=np.zeros(2),
                    turn=np.array([1, 1]), cand_encs=enc, cand_regs=np.array([-1, -1]))
        for key, value in [('observation_contract', obs), ('ai_policy_version', policy),
                           ('action_encoding_contract', ACTION)]:
            if value is not None:
                data[key] = np.asarray([value, value] if isinstance(value, str) else value)
        if phase is not None:
            data['decision_phase'] = np.asarray(phase)
        np.savez(self.corpus / name, **data)
        return self.corpus / name

    def split(self, contract=None):
        data = dict(train=['one.npz'], holdout=['two.npz'])
        if contract is not None:
            data['learning_contract'] = contract
        (self.ds / 'split.json').write_text(json.dumps(data), encoding='utf-8')
        return data

    def test_legacy_consolidation_rejected_before_deleting_targets_or_opening_memmaps(self):
        self.write_game(obs=None, policy=None, phase=None)
        self.write_game('two.npz')
        target = self.ds / 'fp_ztd.npy'
        target.write_bytes(b'original derived targets')
        with mock.patch.object(np.lib.format, 'open_memmap', side_effect=AssertionError('write reached')):
            with self.assertRaisesRegex(ValueError, 'observation_contract'):
                tc.consolidate()
        self.assertEqual(target.read_bytes(), b'original derived targets')

    def test_one_legacy_row_in_current_file_is_rejected(self):
        self.write_game(obs=[OBS, 'legacy'])
        self.write_game('two.npz')
        with mock.patch.object(np.lib.format, 'open_memmap', side_effect=AssertionError('write reached')):
            with self.assertRaisesRegex(ValueError, 'observation_contract'):
                tc.consolidate()

    def test_previous_ring_only_schema_is_rejected_by_npz_and_dataset_gates(self):
        previous = 'v41_ring_state_v1'
        self.write_game(obs=previous)
        self.write_game('two.npz')
        with mock.patch.object(np.lib.format, 'open_memmap', side_effect=AssertionError('write reached')):
            with self.assertRaisesRegex(ValueError, 'observation_contract'):
                tc.consolidate()
        self.split(dict(observation_contract=previous, ai_policy_contract=POLICY))
        with mock.patch.object(tc, '_train_impl') as operator:
            with self.assertRaisesRegex(ValueError, 'observation_contract'):
                tc.train(1, .001, .5)
            operator.assert_not_called()

    def test_legacy_policy_in_holdout_cannot_bypass_preflight(self):
        self.write_game(policy=[POLICY, 'legacy'])
        self.write_game('two.npz')
        with mock.patch.object(np.lib.format, 'open_memmap', side_effect=AssertionError('write reached')):
            with self.assertRaisesRegex(ValueError, 'ai_policy_version'):
                tc.consolidate()

    def test_scalar_or_misaligned_contract_and_empty_phase_are_rejected(self):
        for kwargs, field in [({'obs': np.asarray(OBS)}, 'observation_contract'),
                              ({'policy': [POLICY]}, 'ai_policy_version'),
                              ({'phase': ('pass', '')}, 'decision_phase')]:
            with self.subTest(kwargs=kwargs):
                self.write_game(**kwargs)
                self.write_game('two.npz')
                with mock.patch.object(np.lib.format, 'open_memmap', side_effect=AssertionError('write reached')):
                    with self.assertRaisesRegex(ValueError, field):
                        tc.consolidate()

    def test_current_tiny_consolidation_stamps_both_identities(self):
        self.write_game()
        self.write_game('two.npz')
        tc.consolidate()
        split = json.loads((self.ds / 'split.json').read_text(encoding='utf-8'))
        self.assertEqual(split['learning_contract'],
                         {'observation_contract': OBS, 'ai_policy_contract': POLICY, 'action_encoding_contract': ACTION})
        self.assertEqual(split['contract_rows_checked'], 4)

    def test_train_rejects_unversioned_split_before_operator(self):
        self.split()
        with mock.patch.object(tc, '_train_impl') as operator:
            with self.assertRaisesRegex(ValueError, 'learning_contract'):
                tc.train(1, .001, .5)
        operator.assert_not_called()

    def test_train_rejects_explicit_other_observation_or_policy(self):
        for key in ('observation_contract', 'ai_policy_contract'):
            contract = dict(observation_contract=OBS, ai_policy_contract=POLICY, action_encoding_contract=ACTION)
            contract[key] = 'legacy'
            self.split(contract)
            with self.subTest(key=key), mock.patch.object(tc, '_train_impl') as operator:
                with self.assertRaisesRegex(ValueError, key):
                    tc.train(1, .001, .5)
                operator.assert_not_called()

    def test_current_split_reaches_operator_without_training_in_test(self):
        self.split(dict(observation_contract=OBS, ai_policy_contract=POLICY, action_encoding_contract=ACTION))
        with mock.patch.object(tc, '_train_impl', return_value='reached') as operator:
            self.assertEqual(tc.train(1, .001, .5), 'reached')
            operator.assert_called_once()

    def test_direct_training_impl_refuses_legacy_split_before_arrays(self):
        with mock.patch.object(np, 'load', side_effect=AssertionError('array reached')):
            with self.assertRaisesRegex(ValueError, 'learning_contract'):
                tc._train_impl({'train': []}, {}, '', False, 1, .001, .5)

    def test_other_dataset_stages_refuse_legacy_before_model_load(self):
        self.split()
        calls = [tc.evaluate, lambda: tc.value_td(.8, 1, 1, .001, 2),
                 tc.value_anchor, tc.conf_stamp]
        import torch
        for call in calls:
            with self.subTest(call=call), mock.patch.object(torch, 'load', side_effect=AssertionError('model reached')):
                with self.assertRaisesRegex(ValueError, 'learning_contract'):
                    call()

    def test_direct_evaluation_refuses_legacy_before_any_forward(self):
        self.write_game(obs=None)
        with self.assertRaisesRegex(ValueError, 'observation_contract'):
            tc.eval_games({}, ['one.npz'], 'negative')

    def test_direct_td_collection_refuses_legacy_before_any_forward(self):
        self.write_game(policy=None)
        net = mock.Mock()
        with self.assertRaisesRegex(ValueError, 'ai_policy_version'):
            tc._td_collect_side({'fp': net}, ['one.npz'], 0, 'fp')
        net._process_input.assert_not_called()

    def test_legacy_inspection_reports_missing_versions_without_writes(self):
        path = self.write_game(obs=None, policy=None, phase=None)
        before = path.read_bytes()
        report = tc.inspect_contracts()
        self.assertEqual(report['accepted_for_current_learning'], False)
        self.assertEqual(report['rows'], 2)
        self.assertEqual(report['invalid_files'], 1)
        self.assertIn('observation_contract', report['files'][0]['errors'][0])
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(self.ds.iterdir()), [])

    def test_current_dataset_does_not_relabel_legacy_checkpoint_during_eval(self):
        self.split(dict(observation_contract=OBS, ai_policy_contract=POLICY, action_encoding_contract=ACTION))
        import torch
        import ai
        with mock.patch.object(torch, 'load', return_value={'model_state': {}, 'input_dim': 1}), \
                mock.patch.object(ai, 'WOTRValueNet') as constructor:
            with self.assertRaisesRegex(ValueError, 'observation_contract'):
                tc.evaluate()
            constructor.assert_not_called()

    def test_actual_snapshot_writer_stamps_validated_dataset_contract(self):
        self.split(dict(observation_contract=OBS, ai_policy_contract=POLICY, action_encoding_contract=ACTION))
        import torch
        net = mock.Mock()
        net.state_dict.return_value = {'literal': 'not a trained model'}
        jc = SimpleNamespace(cert_report=lambda *a: {'pass': True},
                             summarize=lambda r: {'pass': True, 'fails': []})
        with mock.patch.dict(sys.modules, jt_cert=jc), \
                mock.patch.object(tc, '_bc_cert_hist_save'), \
                mock.patch.object(tc, '_head_act_of', return_value='tanh'), \
                mock.patch.object(torch, 'save') as save:
            tc._bc_cert_epoch('fp', 0, net, None, str(self.root/'snapshot.pt'), [], [])
        saved = save.call_args.args[0]
        self.assertEqual(saved['observation_contract'], OBS)
        self.assertEqual(saved['ai_policy_contract'], POLICY)
        self.assertEqual(saved['model_state'], {'literal': 'not a trained model'})

    def test_resume_contract_validator_refuses_missing_or_mixed_identity(self):
        for ck in [{}, {'observation_contract': OBS},
                   {'observation_contract': 'v41_ring_state_v1', 'ai_policy_contract': POLICY},
                   {'observation_contract': OBS, 'ai_policy_contract': 'legacy'}]:
            with self.subTest(ck=ck), self.assertRaisesRegex(ValueError, 'contract'):
                tc._require_learning_contract(ck, 'optimizer resume')

    def test_empty_inspection_does_not_claim_training_admission(self):
        report = tc.inspect_contracts()
        self.assertFalse(report['accepted_for_current_learning'])
        self.assertEqual(report['rows'], 0)


if __name__ == '__main__':
    unittest.main()
