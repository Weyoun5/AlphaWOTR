"""Literal schema refusal controls at the server/worker boundary."""
import copy
import os
import unittest
from unittest.mock import patch
import numpy as np
import ai
import infer_server as server
from reward_contract import PROFILE
from trace_decisions import ACTION_ENCODING_CONTRACT


class Pipe:
    def __init__(self, metadata):
        self.metadata = metadata
        self.sent = []
        self.closed = False

    def send(self, message):
        self.sent.append(message)

    def poll(self, seconds):
        return True

    def recv(self):
        return (1, self.metadata)

    def close(self):
        self.closed = True


def metadata():
    return {'fp': dict(checkpoint_id='a' * 64, reward_profile='ring',
                       reward_contract=PROFILE,
                       observation_contract=ai.OBSERVATION_CONTRACT,
                       action_encoding_contract=ACTION_ENCODING_CONTRACT,
                       input_dim=ai.get_feature_dim(), action_dim=ai.ACTION_ENCODING_DIM)}



def protocol_exchange(pipe, message, deadline):
    # Literal metadata/reply-ID unit fixture, deliberately not transport coverage.
    pipe.send(message)
    if not pipe.poll(0):
        raise TimeoutError('literal absent protocol reply')
    return pipe.recv()


class HandoffTests(unittest.TestCase):
    def setUp(self):
        transport = patch.object(server, '_exchange', side_effect=protocol_exchange)
        transport.start()
        self.addCleanup(transport.stop)
        env = patch.dict(os.environ, WOTR_REWARD_CONTRACT=PROFILE)
        env.start()
        self.addCleanup(env.stop)

    def refuse(self, desc, message):
        pipe = Pipe(desc)
        sentinel = object()
        with patch.object(ai, 'INFER_CLIENT', sentinel, create=True):
            with self.assertRaisesRegex(ValueError, message):
                server.attach_client(pipe, ['fp'], require_metadata=True)
            self.assertIs(ai.INFER_CLIENT, sentinel)
        self.assertEqual(pipe.sent, [('describe', 1)])
        self.assertTrue(pipe.closed)

    def test_observation_semantics_mismatch_same_shape_refused(self):
        desc = metadata()
        desc['fp']['observation_contract'] = 'literal_incompatible_observation'
        self.refuse(desc, 'observation contract')

    def test_missing_modern_observation_refused(self):
        desc = metadata()
        del desc['fp']['observation_contract']
        self.refuse(desc, 'observation contract')

    def test_input_dimension_mismatch_refused(self):
        desc = metadata()
        desc['fp']['input_dim'] += 1
        self.refuse(desc, 'input_dim')

    def test_action_dimension_mismatch_refused(self):
        desc = metadata()
        desc['fp']['action_dim'] += 1
        self.refuse(desc, 'action_dim')

    def test_missing_modern_dimensions_refused(self):
        for field in ['input_dim', 'action_dim']:
            with self.subTest(field=field):
                desc = metadata()
                del desc['fp'][field]
                self.refuse(desc, field)

    def test_nonintegral_dimension_claims_refused(self):
        for field in ['input_dim', 'action_dim']:
            for value in [True, np.bool_(True), '93', float(ai.ACTION_ENCODING_DIM), None]:
                with self.subTest(field=field, value=value):
                    desc = metadata()
                    desc['fp'][field] = value
                    self.refuse(desc, field)

    def test_matching_metadata_is_retained_on_handle(self):
        desc = metadata()
        pipe = Pipe(desc)
        handle = server.attach_client(pipe, ['fp'], set_global=False, require_metadata=True)['fp']
        for field in ['checkpoint_id', 'input_dim', 'action_dim', 'observation_contract',
                      'action_encoding_contract']:
            self.assertEqual(getattr(handle, field), desc['fp'][field])
        self.assertFalse(pipe.closed)
        self.assertEqual(pipe.sent, [('describe', 1)])

    def test_explicit_legacy_unstamped_seed_remains_supported(self):
        desc = {'fp': dict(checkpoint_id='a' * 64, reward_profile=None, reward_contract=None)}
        pipe = Pipe(desc)
        handle = server.attach_client(pipe, ['fp'], set_global=False, require_metadata=True)['fp']
        self.assertIsNone(handle.reward_contract)
        self.assertFalse(pipe.closed)
        desc['fp']['input_dim'] = ai.get_feature_dim() + 1
        self.refuse(desc, 'input_dim')

    def test_unbound_legacy_path_remains_no_handshake(self):
        pipe = Pipe({})
        server.attach_client(pipe, ['fp'], set_global=False)
        self.assertEqual(pipe.sent, [])

    def test_later_key_refusal_does_not_publish_partial_handles(self):
        desc = metadata()
        desc['sh'] = copy.deepcopy(desc['fp'])
        desc['sh']['reward_profile'] = 'military'
        desc['sh']['input_dim'] += 1
        pipe = Pipe(desc)
        sentinel = object()
        with patch.object(ai, 'INFER_CLIENT', sentinel, create=True):
            with self.assertRaisesRegex(ValueError, 'input_dim'):
                server.attach_client(pipe, ['fp', 'sh'], require_metadata=True)
            self.assertIs(ai.INFER_CLIENT, sentinel)
        self.assertTrue(pipe.closed)
        self.assertEqual(pipe.sent, [('describe', 1)])


if __name__ == '__main__':
    unittest.main(verbosity=2)
