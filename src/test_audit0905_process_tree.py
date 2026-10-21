"""Actual Windows trees plus explicit unexecuted-POSIX branch controls."""
import ast
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import shutil
import time
import types
import unittest
from unittest.mock import patch

import process_tree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def extract(path, names, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def phase_scope(subprocess_module, factory=None, output=None):
    scope = dict(os=os, subprocess=subprocess_module, ROOT=str(ROOT), SRC=str(ROOT / 'src'),
                 LOOP=str(output or HERE / 'never_created'), TRACE_DIR=str(HERE / 'never_created/traces'),
                 GAMES_JSONL=str(HERE / 'never_created/games.jsonl'), NET_KEYS=[('fp', 'ring'), ('sh', 'military')],
                 _frozen_cfg=lambda: None, PY=sys.executable, BATCH_WALL_SEC=0.01,
                 log_line=lambda text: None, json=json,
                 _PRE_READY_ATTEMPT=None)  # Ordinary backend; no active recovery attempt.
    if factory is not None:
        scope['ProcessTree'] = factory
    return scope


class ProcessTreeControls(unittest.TestCase):

    def test_posix_launch_owns_new_session_and_escalates_after_root_exit(self):
        sent = []
        alive = [True]
        fake = types.SimpleNamespace(pid=1234, poll=lambda: 0, wait=lambda **kw: 0)
        def killpg(pid, sig):
            sent.append((pid, sig))
            if not alive[0]:
                raise ProcessLookupError()
            if sig == 9:
                alive[0] = False
        with patch.object(process_tree.os, 'name', 'posix'), \
                patch.object(process_tree.subprocess, 'Popen', return_value=fake) as launch:
            tree = process_tree.ProcessTree(['literal-child'], cwd='literal-cwd', env={'K': 'V'})
        self.assertEqual(launch.call_args.kwargs, {'cwd': 'literal-cwd', 'env': {'K': 'V'}, 'start_new_session': True})
        with patch.object(process_tree.os, 'killpg', side_effect=killpg, create=True), \
                patch.object(process_tree.os, 'getpgrp', return_value=999, create=True), \
                patch.object(process_tree.signal, 'SIGKILL', 9, create=True):
            result = tree.terminate(grace_seconds=0)
        self.assertIn((1234, signal.SIGTERM), sent)
        self.assertIn((1234, 9), sent)
        self.assertTrue(result['sent_sigkill'])

    def test_posix_surviving_group_is_not_certified_by_exited_root(self):
        fake = types.SimpleNamespace(pid=1234, poll=lambda: 0, wait=lambda **kw: 0)
        with patch.object(process_tree.os, 'name', 'posix'), \
                patch.object(process_tree.subprocess, 'Popen', return_value=fake):
            tree = process_tree.ProcessTree(['literal-child'])
        with patch.object(process_tree.os, 'killpg', create=True), \
                patch.object(process_tree.os, 'getpgrp', return_value=999, create=True), \
                patch.object(process_tree.signal, 'SIGKILL', 9, create=True):
            with self.assertRaisesRegex(process_tree.ProcessTreeError, 'group still exists'):
                tree.terminate(grace_seconds=0, kill_timeout=.01)

    def test_posix_never_targets_callers_process_group(self):
        fake = types.SimpleNamespace(pid=1234, poll=lambda: None, wait=lambda **kw: 0)
        with patch.object(process_tree.os, 'name', 'posix'), \
                patch.object(process_tree.subprocess, 'Popen', return_value=fake):
            tree = process_tree.ProcessTree(['literal-child'])
        with patch.object(process_tree.os, 'getpgrp', return_value=1234, create=True), \
                patch.object(process_tree.os, 'killpg', create=True) as kill:
            with self.assertRaisesRegex(process_tree.ProcessTreeError, 'caller process group'):
                tree.terminate(grace_seconds=0)
            kill.assert_not_called()

    def test_posix_absent_group_is_bounded_and_does_not_signal_other_pids(self):
        fake = types.SimpleNamespace(pid=1234, poll=lambda: 0, wait=lambda **kw: 0)
        with patch.object(process_tree.os, 'name', 'posix'), \
                patch.object(process_tree.subprocess, 'Popen', return_value=fake):
            tree = process_tree.ProcessTree(['literal-child'])
        with patch.object(process_tree.os, 'getpgrp', return_value=999, create=True), \
                patch.object(process_tree.os, 'killpg', side_effect=ProcessLookupError, create=True) as kill:
            result = tree.terminate(grace_seconds=0)
        self.assertFalse(result['sent_sigterm'])
        self.assertFalse(result['sent_sigkill'])
        self.assertTrue(all(call.args[0] == 1234 for call in kill.call_args_list))

    @unittest.skipUnless(os.name == 'nt', 'Windows native command control')
    def test_windows_taskkill_failure_is_not_success_or_unbounded_wait(self):
        fake = types.SimpleNamespace(pid=1234, poll=lambda: None, wait=lambda **kw: self.fail('No root wait after failed taskkill'))
        with patch.object(process_tree.subprocess, 'Popen', return_value=fake):
            tree = process_tree.ProcessTree(['literal-child'])
        with patch.object(process_tree.subprocess, 'run', return_value=types.SimpleNamespace(
                returncode=1, stdout=b'', stderr=b'literal refusal')) as run:
            with self.assertRaisesRegex(process_tree.ProcessTreeError, 'literal refusal'):
                tree.terminate(kill_timeout=.25)
        self.assertEqual(run.call_args.kwargs['timeout'], .25)
        self.assertNotIn('shell', run.call_args.kwargs)

    def test_timeout_cleanup_failure_stops_before_trace_archiving(self):
        class Refusal:
            def __init__(self, *a, **kw):
                pass
            def wait(self, **kw):
                raise subprocess.TimeoutExpired('literal batch', .01)
            def terminate(self):
                raise process_tree.ProcessTreeError('literal cleanup failed')
        scope = phase_scope(subprocess, Refusal)
        extract(HERE / 'run_td_v33loop.py', ['play_batch'], scope)
        with self.assertRaisesRegex(process_tree.ProcessTreeError, 'literal cleanup failed'), \
                patch.object(os, 'makedirs') as create:
            scope['play_batch'](0, 1, 1, 4)
        create.assert_not_called()

    @unittest.skipUnless(os.name == 'nt', 'Windows native command control')
    def test_windows_command_timeout_is_bounded_and_not_certified(self):
        fake = types.SimpleNamespace(pid=1234, poll=lambda: None)
        with patch.object(process_tree.subprocess, 'Popen', return_value=fake):
            tree = process_tree.ProcessTree(['literal-child'])
        with patch.object(process_tree.subprocess, 'run', side_effect=subprocess.TimeoutExpired(
                'taskkill', .25)) as run:
            with self.assertRaisesRegex(process_tree.ProcessTreeError, 'termination failed'):
                tree.terminate(kill_timeout=.25)
        self.assertEqual(run.call_args.kwargs['timeout'], .25)

    def test_interrupt_cleans_up_once_and_preserves_interrupt(self):
        calls = []
        class Interrupted:
            def __init__(self, *a, **kw):
                pass
            def wait(self, **kw):
                raise KeyboardInterrupt()
            def terminate(self):
                calls.append('terminated')
        scope = phase_scope(subprocess, Interrupted)
        extract(HERE / 'run_td_v33loop.py', ['play_batch'], scope)
        with self.assertRaises(KeyboardInterrupt), patch.object(os, 'makedirs') as create:
            scope['play_batch'](0, 1, 1, 4)
        self.assertEqual(calls, ['terminated'])
        create.assert_not_called()

    @unittest.skipUnless(os.name == 'nt', 'Windows native command control')
    def test_windows_dead_root_does_not_target_reused_pid_or_claim_success(self):
        fake = types.SimpleNamespace(pid=1234, poll=lambda: 0)
        with patch.object(process_tree.subprocess, 'Popen', return_value=fake):
            tree = process_tree.ProcessTree(['literal-child'])
        with patch.object(process_tree.subprocess, 'run') as run:
            with self.assertRaisesRegex(process_tree.ProcessTreeError, 'root already exited'):
                tree.terminate()
        run.assert_not_called()


@unittest.skipUnless(os.name == 'nt', 'Actual Windows process handles are required')
class RealWindowsTreeTests(unittest.TestCase):
    def setUp(self):
        from ctypes import wintypes
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.kernel.WaitForSingleObject.restype = wintypes.DWORD
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.handles = []
        self.trees = []

    def tearDown(self):
        for tree in self.trees:
            if tree.poll() is None:
                tree.terminate(kill_timeout=5)
        for handle in self.handles:
            self.kernel.CloseHandle(handle)

    def spawn(self, label, role='root'):
        output = Path(tempfile.mkdtemp(prefix='wotr_tree_' + label + '_'))
        self.addCleanup(shutil.rmtree, output)
        tree = process_tree.ProcessTree([sys.executable, str(Path(__file__).resolve()), '--fixture-tree', role, str(output)])
        self.trees.append(tree)
        names = ['root', 'child', 'grandchild'] if role == 'root' else [role]
        deadline = time.monotonic() + 10
        while not all((output / (name + '.json')).is_file() for name in names):
            if time.monotonic() > deadline:
                self.fail('Fixture tree did not become ready')
            time.sleep(.02)
        rows = [json.loads((output / (name + '.json')).read_text()) for name in names]
        self.assertEqual(rows[0]['pid'], tree.pid)
        if role == 'root':
            self.assertEqual(rows[1]['parent'], rows[0]['pid'])
            self.assertEqual(rows[2]['parent'], rows[1]['pid'])
        handles = []
        for row in rows:
            handle = self.kernel.OpenProcess(0x00100000 | 0x1000, False, row['pid'])
            self.assertTrue(handle)
            self.handles.append(handle)
            handles.append(handle)
            self.assertEqual(self.kernel.WaitForSingleObject(handle, 0), 258)
        return tree, rows, handles

    def test_real_child_grandchild_and_unrelated_control(self):
        tree, rows, handles = self.spawn('direct_tree')
        control, control_rows, control_handles = self.spawn('unrelated_control', role='control')
        with self.assertRaises(subprocess.TimeoutExpired):
            tree.wait(timeout=.05)
        start = time.monotonic()
        result = tree.terminate(kill_timeout=5)
        states = [self.kernel.WaitForSingleObject(handle, 1000) for handle in handles]
        self.assertEqual(states, [0, 0, 0])
        self.assertEqual(self.kernel.WaitForSingleObject(control_handles[0], 0), 258)

    def test_actual_loop_timeout_kills_tree_before_archiving(self):
        tree, rows, handles = self.spawn('loop_tree')
        scope = phase_scope(subprocess, lambda *a, **kw: tree)
        extract(HERE / 'run_td_v33loop.py', ['play_batch'], scope)
        class AtArchive(Exception):
            pass
        def archive_boundary(*args, **kwargs):
            states = [self.kernel.WaitForSingleObject(handle, 1000) for handle in handles]
            self.assertEqual(states, [0, 0, 0])
            raise AtArchive()
        with patch.object(os, 'makedirs', side_effect=archive_boundary), self.assertRaises(AtArchive):
            scope['play_batch'](0, 1, 1, 4)



def _fixture_tree(role, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if role in ('root', 'child'):
        following = 'child' if role == 'root' else 'grandchild'
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                          '--fixture-tree', following, str(output)])
    (output / (role + '.json')).write_text(json.dumps({'pid': os.getpid(), 'parent': os.getppid()}))
    while True:
        time.sleep(.2)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--fixture-tree':
        _fixture_tree(sys.argv[2], sys.argv[3])
    else:
        unittest.main(verbosity=2)
