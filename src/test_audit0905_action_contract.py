"""Same-width historical actions must not enter fresh casualty learning.

Literal identities and in-memory old arrays provide negative controls;
these tests never regenerate an expected label from the current encoder.
"""
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ACTION = 'casualty_operations_v2'
OBS = 'v44_fellowship_tail_v1'
POLICY = 'elite_and_bonus_draw_v1'
PROFILE = 'predecision_profile_v1'


def metadata(action=ACTION):
    out = dict(observation_contract=OBS, ai_policy_contract=POLICY)
    if action is not None:
        out['action_encoding_contract'] = action
    return out


class ActionContractControls(unittest.TestCase):
    def test_missing_same_width_npz_is_refused(self):
        import train_corpus as tc
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'literal.npz'
            for action in (None, 'previous_casualty_counts', 'casualty_operations_v1', ACTION):
                import ai
                data = dict(X=np.zeros((1, ai.get_feature_dim()), np.float32),
                            cand_encs=np.zeros((1, 93), np.float32),
                            observation_contract=np.array([OBS]),
                            ai_policy_version=np.array([POLICY]),
                            decision_phase=np.array(['combat_casualty']))
                if action is not None:
                    data['action_encoding_contract'] = np.array([action])
                np.savez(path, **data)
                with np.load(path, allow_pickle=False) as pack, self.subTest(action=action):
                    if action == ACTION:
                        self.assertEqual(tc._require_npz_contract(pack, 'literal')['errors'], [])
                    else:
                        with self.assertRaisesRegex(ValueError, 'action_encoding_contract'):
                            tc._require_npz_contract(pack, 'literal')

    def test_dataset_and_checkpoint_gate_refuses_missing_or_other_contract(self):
        import train_corpus as tc
        for action in (None, 'previous_casualty_counts', 'casualty_operations_v1', ACTION):
            with self.subTest(action=action):
                if action == ACTION:
                    self.assertEqual(tc._require_learning_contract(metadata(action), 'literal'),
                                     metadata(action))
                else:
                    with self.assertRaisesRegex(ValueError, 'action_encoding_contract'):
                        tc._require_learning_contract(metadata(action), 'literal')

    def test_new_recorders_and_context_stamp_explicit_action_semantics(self):
        from game_trace import TraceRecorder
        from trace_decisions import current_learning_context
        self.assertEqual(current_learning_context().get('action_encoding_contract'), ACTION)
        self.assertEqual(TraceRecorder('unused').meta.get('action_encoding_contract'), ACTION)

    def test_fifo_rejects_old_semantics_before_any_batch_write(self):
        import run_td_v33loop as loop
        from selector_contract import current_identity
        with patch.dict(os.environ, WOTR_REWARD_CONTRACT=PROFILE):
            for action in (None, 'previous_casualty_counts', 'casualty_operations_v1'):
                with self.subTest(action=action), tempfile.TemporaryDirectory() as temp:
                    key = ('fp', 'ring')
                    buffer = loop.ReplayBuffer(temp, net_keys=[key])
                    stream = dict(metadata(action), **current_identity(),
                                  reward_contract=PROFILE, value=[], delta=[])
                    before = copy.deepcopy(stream)
                    with self.assertRaisesRegex(ValueError, 'action_encoding_contract'):
                        buffer.add(0, {key: stream})
                    self.assertEqual(list(Path(temp).rglob('*.gz')), [])
                    self.assertEqual(stream, before)

    def test_served_corrected_identity_refuses_old_semantics(self):
        import ai
        import infer_server
        from test_audit0905_served_identity import LiteralPipe, protocol_exchange
        # Contract unit fixture; real transport is covered by stalled-peer tests.
        with patch.object(infer_server, "_exchange", side_effect=protocol_exchange), \
                patch.dict(os.environ, WOTR_REWARD_CONTRACT=PROFILE):
            for action in (None, 'previous_casualty_counts', 'casualty_operations_v1', ACTION):
                desc = dict(metadata(action), checkpoint_id='a' * 64,
                            reward_profile='ring', reward_contract=PROFILE,
                            input_dim=ai.get_feature_dim(), action_dim=ai.ACTION_ENCODING_DIM)
                pipe = LiteralPipe([(1, {'fp': desc})])
                with self.subTest(action=action):
                    if action == ACTION:
                        handle = infer_server.attach_client(pipe, ['fp'], set_global=False,
                                                            require_metadata=True)['fp']
                        self.assertEqual(handle.action_encoding_contract, ACTION)
                    else:
                        with self.assertRaisesRegex(ValueError, 'action_encoding_contract'):
                            infer_server.attach_client(pipe, ['fp'], set_global=False,
                                                       require_metadata=True)

    def test_old_trace_diagnostics_remain_readable_but_fresh_admission_refuses(self):
        from trace_admission import validate_trace
        from trace_decisions import trace_action_encoding_contract
        trace = {'meta': {}, 'steps': []}
        self.assertIsNone(trace_action_encoding_contract(trace))
        with self.assertRaisesRegex(ValueError, 'action_encoding_contract'):
            trace_action_encoding_contract(trace, required=True)
        report = validate_trace(trace, decision_contract=PROFILE, audit_rules=False)
        self.assertIn('action_encoding_contract', [error['code'] for error in report['errors']])

    def test_direct_corrected_collectors_refuse_before_reading_old_rows(self):
        import delta_trainer
        import run_td_v33loop as loop
        trace = {'meta': {'reward_contract': PROFILE}, 'steps': []}
        with patch.dict(os.environ, WOTR_REWARD_CONTRACT=PROFILE):
            for consumer in (lambda: delta_trainer.collect_delta_samples(trace, 'fp'),
                             lambda: loop._collect_search_samples(trace, 'fp'),
                             lambda: loop._replay_reward_stream(trace, {})):
                with self.subTest(consumer=consumer), self.assertRaisesRegex(
                        ValueError, 'action_encoding_contract'):
                    consumer()

    def test_checkpoint_validator_preserves_explicit_legacy_warmstart(self):
        from trace_decisions import validate_action_encoding_contract
        self.assertIsNone(validate_action_encoding_contract({}))
        for value in (None, '', 'previous_casualty_counts', 'casualty_operations_v1'):
            with self.subTest(value=value), self.assertRaisesRegex(
                    ValueError, 'action_encoding_contract'):
                validate_action_encoding_contract({'action_encoding_contract': value})
        self.assertEqual(validate_action_encoding_contract(metadata(), required=True), ACTION)

    def test_direct_policy_operator_refuses_unversioned_rows_before_model_mutation(self):
        import delta_trainer
        from test_audit0905_selector_contract import literal_row
        class UntouchedModel:
            def parameters(self):
                raise AssertionError('parameters accessed before action identity validation')
        with patch.dict(os.environ, WOTR_REWARD_CONTRACT=PROFILE):
            row = literal_row()
            row.pop('action_encoding_contract', None)
            with self.assertRaisesRegex(ValueError, 'action_encoding_contract'):
                delta_trainer.train_policy_delta(UntouchedModel(), [row])

    def test_npz_writer_preserves_missing_historical_identity_instead_of_upgrading_it(self):
        import ai
        import extract_corpus
        import train_corpus
        sample = dict(x=np.zeros(ai.get_feature_dim(), np.float32), side='fp',
                      human=0, z=1, turn=1, phi=0., encs=np.zeros((1, 93), np.float32),
                      regs=np.array([-1]), phase='pass', ai_policy_version=POLICY,
                      observation_contract=OBS)
        with tempfile.TemporaryDirectory() as temp:
            for action in (None, ACTION):
                if action is not None:
                    sample['action_encoding_contract'] = action
                path = str(Path(temp) / 'literal.npz')
                extract_corpus.pack_npz([sample], path)
                with np.load(path) as data, self.subTest(action=action):
                    self.assertEqual(data['action_encoding_contract'].tolist(), [action or 'legacy'])
                    if action is None:
                        with self.assertRaisesRegex(ValueError, 'action_encoding_contract'):
                            train_corpus._require_npz_contract(data, path)
                    else:
                        self.assertEqual(train_corpus._require_npz_contract(data, path)['errors'], [])

    def test_readonly_historical_attribution_does_not_authorize_training(self):
        import run_td_v33loop as loop
        trace = dict(meta={'reward_contract': PROFILE,
                           'observation_contract': 'v41_ring_rearguard_state_v1'},
                     steps=[{'kind': 'action', 'side': 'FP',
                             'decision_context': {'contract': PROFILE, 'encoding': [2., 3.]}}])
        original = copy.deepcopy(trace)
        with patch.dict(os.environ, WOTR_REWARD_CONTRACT=PROFILE):
            inspected = loop._replay_reward_stream(trace, {}, for_learning=False)
            np.testing.assert_array_equal(inspected['fp'][0], [[2., 3.]])
            with self.assertRaisesRegex(ValueError, 'observation contract'):
                loop._replay_reward_stream(trace, {})
        self.assertEqual(trace, original)


if __name__ == '__main__':
    unittest.main()
