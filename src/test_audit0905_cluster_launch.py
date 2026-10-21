"""Literal launcher boundaries; no GPUs, real games, or cluster allocation."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
import infer_server
import run_sp_served as runner

SRC = Path(__file__).resolve().parent


def source_functions(filename, names, namespace):
    tree = ast.parse((SRC / filename).read_text(encoding='utf-8-sig'))
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(nodes) != len(names):
        raise AssertionError('Missing source function')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, 'exec'), namespace)
    return namespace


class ClusterLaunchTests(unittest.TestCase):
    def args(self, temp, **changes):
        root = Path(temp)
        for side in ('fp', 'sh'):
            (root / (side + '.pt')).write_bytes(('literal ' + side).encode())
        values = dict(fp_net=str(root / 'fp.pt'), sh_net=str(root / 'sh.pt'),
                      out_dir=str(root / 'run'), games=2, workers=2, sims=4,
                      seed0=43000, arch='base', shard_id=None)
        values.update(changes)
        return types.SimpleNamespace(**values)

    def test_all_server_entries_preserve_every_scheduler_mask(self):
        assigned = {'CUDA_VISIBLE_DEVICES': '2', 'HIP_VISIBLE_DEVICES': '0',
                    'ROCR_VISIBLE_DEVICES': '3'}
        for filename in ('run_sp_served.py', 'arena_scale.py', 'bench_scale.py'):
            with self.subTest(filename=filename):
                seen = []
                stub = types.SimpleNamespace(server_main=lambda *a: seen.append(dict(os.environ)))
                scope = source_functions(filename, ['_server_entry'], {'os': os, 'sys': sys, 'SRC': str(SRC)})
                with patch.dict(os.environ, assigned, clear=True), patch.dict(sys.modules, {'infer_server': stub}):
                    scope['_server_entry']([], {}, 'base')
                self.assertEqual(seen, [assigned])

    def test_requested_unavailable_gpu_refuses_before_loading(self):
        with patch.object(torch.cuda, 'is_available', return_value=False), \
                patch.object(torch.cuda, 'device_count', return_value=0), \
                patch.object(infer_server, '_load_net') as loader:
            with self.assertRaisesRegex(RuntimeError, 'assigned GPU'):
                infer_server.server_main([], {'fp': 'unused'}, 'base')
            loader.assert_not_called()

    def test_zero_devices_and_out_of_range_logical_device_refuse(self):
        for count, device in ((0, 'cuda'), (1, 'cuda:1')):
            with self.subTest(count=count, device=device), \
                    patch.object(torch.cuda, 'is_available', return_value=True), \
                    patch.object(torch.cuda, 'device_count', return_value=count), \
                    patch.object(infer_server, '_load_net') as loader:
                with self.assertRaisesRegex(RuntimeError, 'assigned GPU'):
                    infer_server.server_main([], {'fp': 'unused'}, 'base', device)
                loader.assert_not_called()

    def test_explicit_cpu_and_assigned_gpu_are_selected_before_loading(self):
        import ai
        class Observed(Exception):
            pass
        for device in ('cpu', 'cuda:0'):
            def load(*args):
                self.assertEqual(str(ai.DEVICE), device)
                raise Observed()
            with self.subTest(device=device), patch.object(ai, 'DEVICE', torch.device('cpu')), \
                    patch.object(torch.cuda, 'is_available', return_value=True), \
                    patch.object(torch.cuda, 'device_count', return_value=1), \
                    patch.object(infer_server, '_load_net', side_effect=load):
                with self.assertRaises(Observed):
                    infer_server.server_main([], {'fp': 'unused'}, 'base', device)

    def test_relative_output_resolved_before_engine_cwd_change(self):
        original = os.getcwd()
        self.addCleanup(os.chdir, original)
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {}, clear=True):
            args = self.args(temp, out_dir='relative_output')
            os.chdir(temp)
            paths, seed = runner._prepare_run(args)
            os.chdir(SRC)
            self.assertEqual(args.out_dir, str(Path(temp) / 'relative_output'))
            self.assertTrue(all(Path(path).is_absolute() for path in paths.values()))
            manifest = json.loads((Path(args.out_dir) / 'launch.json').read_text())
            self.assertEqual(manifest['worker_seed_base'], 43000)
            self.assertEqual(manifest['checkpoints']['fp']['sha256'], hashlib.sha256(b'literal fp').hexdigest())

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {}, clear=True):
            args = self.args(temp)
            output = Path(args.out_dir)
            output.mkdir()
            sentinel = output / 'existing.trace.gz'
            sentinel.write_bytes(b'original game')
            with self.assertRaises(FileExistsError):
                runner._prepare_run(args)
            self.assertEqual(sentinel.read_bytes(), b'original game')
            self.assertEqual(list(output.iterdir()), [sentinel])

    def test_shards_have_literal_disjoint_seed_ranges_and_directories(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, FLUX_TASK_RANK='0'):
            first = self.args(temp, shard_id=0, workers=24)
            _, seed0 = runner._prepare_run(first)
            second = self.args(temp, shard_id=1, workers=16)
            _, seed1 = runner._prepare_run(second)
            self.assertEqual((seed0, seed1), (43000, 108536))
            self.assertEqual(Path(first.out_dir).name, 'shard_000000')
            self.assertEqual(Path(second.out_dir).name, 'shard_000001')
            self.assertFalse(set(range(seed0, seed0 + 24)) & set(range(seed1, seed1 + 16)))
            duplicate = self.args(temp, shard_id=0)
            with self.assertRaises(FileExistsError):
                runner._prepare_run(duplicate)

    def test_flux_requires_explicit_shard_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, FLUX_TASK_RANK='1'):
            args = self.args(temp)
            with self.assertRaisesRegex(ValueError, 'shard-id'):
                runner._prepare_run(args)
            self.assertFalse(Path(args.out_dir).exists())

    def test_invalid_resource_or_seed_range_does_not_reserve_output(self):
        for changes in ({'games': 0}, {'workers': 0}, {'workers': 65537},
                        {'sims': 0}, {'seed0': -1}, {'shard_id': -1},
                        {'seed0': 2 ** 32 - 1}, {'shard_id': 65536}):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as temp, \
                    patch.dict(os.environ, {}, clear=True):
                args = self.args(temp, **changes)
                with self.assertRaises(ValueError):
                    runner._prepare_run(args)
                self.assertFalse(Path(args.out_dir).exists())

    def test_loop_to_arena_restores_only_its_own_inserted_mask(self):
        class ReachedSubprocess(Exception):
            pass
        cases = [(True, {'CUDA_VISIBLE_DEVICES': '-1'}, {}),
                 (False, {'CUDA_VISIBLE_DEVICES': '-1'}, {'CUDA_VISIBLE_DEVICES': '-1'}),
                 (False, {'CUDA_VISIBLE_DEVICES': '2', 'ROCR_VISIBLE_DEVICES': '3'},
                         {'CUDA_VISIBLE_DEVICES': '2', 'ROCR_VISIBLE_DEVICES': '3'}),
                 (True, {'CUDA_VISIBLE_DEVICES': '1'}, {'CUDA_VISIBLE_DEVICES': '1'})]
        for inserted, assigned, expected in cases:
            seen = []
            def launch(*args, **kwargs):
                seen.append(kwargs['env'])
                raise ReachedSubprocess()
            scope = source_functions('run_td_v33loop.py', ['_arena_gpu_environment', 'run_arena_one'],
                                     {'os': os, '_LOOP_INSERTED_CPU_MASK': inserted,
                                      'ARENA_PARTNER': {'fp': ('sh', 'military')},
                                      '_nm': lambda *a: 'unused.pt', 'LOOP': 'unused',
                                      'SRC': str(SRC), 'ROOT': str(SRC.parent), 'PY': sys.executable,
                                      'ARENA_WORKERS': 2, 'ARENA_GAMES_PER_WORKER': 1, 'ARENA_SIMS': 4,
                                      'subprocess': types.SimpleNamespace(run=launch, PIPE=-1, STDOUT=-2)})
            with self.subTest(inserted=inserted, assigned=assigned), \
                    patch.dict(os.environ, assigned, clear=True), patch.object(os, 'remove'), \
                    self.assertRaises(ReachedSubprocess):
                scope['run_arena_one']('fp', 'ring')
            actual = {key: value for key, value in seen[0].items() if key.endswith('VISIBLE_DEVICES')}
            self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
