"""Literal served checkpoint identity and profile boundaries; no training."""
import hashlib
import os
from pathlib import Path
import queue
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import infer_server as server
import run_sp_served as runner
from reward_contract import PROFILE, validate_net_profile
from trace_decisions import ACTION_ENCODING_CONTRACT


class LiteralPipe:
    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []
        self.closed = False

    def close(self):
        self.closed = True

    def send(self, msg):
        self.sent.append(msg)

    def recv(self):
        return self.replies.pop(0)

    def poll(self, timeout=0):
        return bool(self.replies)


def description(profile='ring'):
    import ai
    return {'fp': {'checkpoint_id': 'a' * 64, 'reward_profile': profile,
                   'reward_contract': PROFILE, 'action_encoding_contract': ACTION_ENCODING_CONTRACT,
                   'observation_contract': ai.OBSERVATION_CONTRACT,
                   'input_dim': ai.get_feature_dim(), 'action_dim': ai.ACTION_ENCODING_DIM}}



def protocol_exchange(pipe, message, deadline):
    # Literal metadata/reply-ID unit fixture, deliberately not transport coverage.
    pipe.send(message)
    if not pipe.poll(0):
        raise TimeoutError('literal absent protocol reply')
    return pipe.recv()


class ServedIdentityTests(unittest.TestCase):
    def setUp(self):
        transport = patch.object(server, '_exchange', side_effect=protocol_exchange)
        transport.start()
        self.addCleanup(transport.stop)
        env = patch.dict(os.environ, WOTR_REWARD_CONTRACT=PROFILE)
        env.start()
        self.addCleanup(env.stop)

    def test_wrong_reply_id_is_never_used_as_a_current_prediction(self):
        calls = [lambda c: c.eval('fp', np.zeros((1, 2))),
                 lambda c: c.pv('fp', np.zeros(2), np.zeros((2, 3)), np.zeros(2)),
                 lambda c: c.pv_many('fp', [(np.zeros(2), np.zeros((2, 3)), np.zeros(2))])]
        for call, reply in zip(calls, [(9, [.5]), (9, [.2, .8], .5), (9, [([.2, .8], .5)])]):
            with self.subTest(reply=reply):
                pipe = LiteralPipe([reply])
                with self.assertRaisesRegex(ValueError, 'request|reply'):
                    call(server.InferClient(pipe))
                self.assertTrue(pipe.closed)

    def test_described_handle_carries_loaded_profile_to_existing_episode_guard(self):
        pipe = LiteralPipe([(1, description())])
        handles = server.attach_client(pipe, ['fp'], set_global=False, require_metadata=True)
        handle = handles['fp']
        self.assertEqual(handle.checkpoint_id, 'a' * 64)
        self.assertEqual(handle.reward_profile, 'ring')
        self.assertEqual(handle.reward_contract, PROFILE)
        validate_net_profile(handle, 'fp', 'ring')
        with self.assertRaisesRegex(ValueError, 'profile'):
            validate_net_profile(handle, 'fp', 'military')
        self.assertEqual(pipe.sent, [('describe', 1)])

    def test_unstamped_bc_seed_is_explicit_and_remains_usable(self):
        desc = description(None)
        desc['fp']['reward_contract'] = None
        desc['fp'].pop('action_encoding_contract')
        desc['fp'].pop('observation_contract')
        handle = server.attach_client(LiteralPipe([(1, desc)]), ['fp'],
                                       set_global=False, require_metadata=True)['fp']
        self.assertIsNone(handle.reward_profile)
        validate_net_profile(handle, 'fp', 'military')
        self.assertEqual(handle.checkpoint_id, 'a' * 64)

    def test_missing_or_malformed_descriptor_is_not_an_unstamped_seed(self):
        malformed = [{}, {'fp': {}}, {'fp': dict(description()['fp'], checkpoint_id='wrong')},
                     {'fp': dict(description()['fp'], reward_profile='corruption')},
                     {'fp': dict(description()['fp'], reward_profile=None)},
                     {'fp': dict(description()['fp'], reward_contract='predecision_winloss_v1')}]
        for desc in malformed:
            with self.subTest(desc=desc):
                with self.assertRaises(ValueError):
                    server.attach_client(LiteralPipe([(1, desc)]), ['fp'],
                                         set_global=False, require_metadata=True)

    def test_describe_timeout_is_a_startup_failure(self):
        pipe = LiteralPipe([])
        with self.assertRaises(TimeoutError):
            server.attach_client(pipe, ['fp'], set_global=False, require_metadata=True)
        self.assertTrue(pipe.closed)

    def test_original_unbound_handle_has_no_unsolicited_roundtrip(self):
        pipe = LiteralPipe([])
        handle = server.attach_client(pipe, ['fp'], set_global=False)['fp']
        self.assertEqual(pipe.sent, [])
        self.assertEqual(handle.remote_key, 'fp')

    def test_loader_fingerprints_actual_file_and_refuses_during_load_change(self):
        with tempfile.TemporaryDirectory(prefix='wotr_served_identity_') as temp:
            path = Path(temp) / 'seed.pt'
            path.write_bytes(b'literal seed checkpoint')
            net = SimpleNamespace(eval=lambda: None)
            fake_ai = SimpleNamespace(load_value_net=lambda *a, **kw: net,
                                      _resolve_model_path=os.path.abspath)
            result = server._load_net(fake_ai, None, str(path), 'fp')
            self.assertEqual(result.checkpoint_id, hashlib.sha256(b'literal seed checkpoint').hexdigest())
            def changing_load(*args, **kw):
                path.write_bytes(b'changed while loading')
                return net
            fake_ai.load_value_net = changing_load
            with self.assertRaisesRegex(ValueError, 'changed'):
                server._load_net(fake_ai, None, str(path), 'fp')

    def test_loader_uses_project_resolver_for_relative_model_paths(self):
        with tempfile.TemporaryDirectory(prefix='wotr_served_relative_') as temp:
            path = Path(temp) / 'seed.pt'
            path.write_bytes(b'project relative checkpoint')
            seen = []
            net = SimpleNamespace(eval=lambda: None)
            def resolve(relative):
                self.assertEqual(relative, 'seed.pt')
                return str(path)
            def load(resolved, side):
                seen.append(resolved)
                return net
            server._load_net(SimpleNamespace(_resolve_model_path=resolve, load_value_net=load),
                             None, 'seed.pt', 'fp')
            self.assertEqual(seen, [str(path)])
            self.assertEqual(net.checkpoint_id, hashlib.sha256(b'project relative checkpoint').hexdigest())

    def test_worker_records_actual_handle_ids_in_episode_config_and_stats(self):
        ids = {'fp': 'a' * 64, 'sh': 'b' * 64}
        handles = {k: SimpleNamespace(checkpoint_id=v) for k, v in ids.items()}
        calls, attached = [], []
        def attach(*a, **kw):
            attached.append(kw)
            return handles
        def episode(fp, sh, config):
            calls.append(config)
            return {'game_stats': {'winner': 'fp'}}
        # The worker installs observation wrappers on the engine's search/apply entry
        # points (served_progress since 2026-09-08); the stub carries both.
        fake_ai = SimpleNamespace(PLAYER_PROFILES={}, FP_PROFILES=['ring'], SH_PROFILES=['military'],
                                  _run_single_episode=episode, mcts_search=lambda *a, **k: None,
                                  _traced_execute=lambda *a, **k: None)
        modules = {'ai': fake_ai, 'infer_server': SimpleNamespace(attach_client=attach),
                   'numpy': SimpleNamespace(random=SimpleNamespace(seed=lambda _: None)),
                   'torch': SimpleNamespace(set_num_threads=lambda _: None, device=lambda x: x)}
        with tempfile.TemporaryDirectory(prefix='wotr_served_worker_') as temp, \
                patch.dict(sys.modules, modules), patch.dict(os.environ, {
                    'WOTR_SP_FP_PROFILES': 'ring', 'WOTR_SP_SH_PROFILES': 'military'}):
            sink = queue.Queue()
            runner._worker(0, object(), 1, 4, temp, 123, 'base', sink)
        self.assertEqual(attached, [{'require_metadata': True}])
        self.assertEqual(calls[0]['checkpoint_ids'], ids)
        record = sink.get_nowait()
        while record[0] == 'p':            # throttled worker progress precedes the game record
            record = sink.get_nowait()
        self.assertEqual(record[0], 'g')
        self.assertEqual(record[2]['checkpoint_ids'], ids)


if __name__ == '__main__':
    unittest.main()
