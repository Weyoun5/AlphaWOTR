"""Actual main_v2 startup control; no game play or optimizer is performed."""
import ast
import glob
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'src'))
import loop_meters



def exercise(path, reference, generated=None):
    with tempfile.TemporaryDirectory(prefix='wotr_probe_startup_') as temp:
        directory = Path(temp)
        probes = directory / 'actual_snapshot_placeholder.pkl.gz'
        probes.write_bytes(b'Not read by this startup-only control')
        reference_path = directory / 'probe_ref_seed.json'
        if reference is not None:
            reference_path.write_text(json.dumps({'rows': reference}))
        original = reference_path.read_bytes() if reference_path.exists() else None
        events = []
        ns = {'os': os, 'json': json, 'glob': glob, 'time': time, 'LOOP': temp,
              'NET_KEYS': [('fp', 'ring'), ('sh', 'military')], 'BUFFER_K': 1, 'BATCH_GAMES': 1,
              'ARENA_ENABLED': False, 'ARENA_EVERY_BATCHES': 4,
              '_probeset_path': lambda: str(probes), '_side_key': lambda side: (side, 'ring' if side == 'fp' else 'military'),
              '_net_names': lambda: ['fp_ring', 'sh_military'], '_nm': lambda *a: 'never_loaded.pt',
              'ReplayBuffer': lambda *a: object(), '_next_batch_idx': lambda dirs: 0,
              '_unbuffered_last_batch': lambda dirs: None, 'log_line': lambda *a: None,
              'run_batch_cycle': lambda index, context: events.append('batch_started'),
              '_batch_journal': lambda: None}
        for name in ('_resolve_ops', 'setup', '_pid_lock', 'preflight_generation', 'integrity_check',
                     'quarantine_stale_traces', 'play_batch', 'audit_batch', 'stage_a_gate',
                     'extract_batch_streams', 'train_nets_v2', '_probe_live_rows', 'stage_b_gate',
                     '_write_probe_cache', 'gate_all_v2'):
            ns[name] = lambda *a, **kw: None
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main_v2')
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
        error = None
        with patch.object(loop_meters, 'LoopState', return_value=object()), \
                patch.object(loop_meters, 'probe_rows', return_value=generated):
            try:
                ns['main_v2'](types.SimpleNamespace(workers=1, sims=4, total_games=1))
            except Exception as exc:
                error = exc
        after = reference_path.read_bytes() if reference_path.exists() else None
        return error, events, original, after


class StartupReferenceTests(unittest.TestCase):

    def test_candidate_refuses_legacy_mixed_and_empty_before_batch(self):
        current = {'id': 'new', 'action_identity_contract': 'complete_action_identity_v1'}
        for rows in ([{'id': 'old'}], [], [current, {'id': 'old'}]):
            error, events, before, after = exercise(HERE / 'run_td_v33loop.py', rows)
            self.assertIsInstance(error, ValueError)
            self.assertIn('probe reference', str(error))
            self.assertEqual(events, [])
            self.assertEqual(before, after)

    def test_current_existing_reference_reaches_batch_unchanged(self):
        rows = [{'id': 'current', 'action_identity_contract': 'complete_action_identity_v1'}]
        error, events, before, after = exercise(HERE / 'run_td_v33loop.py', rows)
        self.assertIsNone(error)
        self.assertEqual(events, ['batch_started'])
        self.assertEqual(before, after)

    def test_fresh_invalid_reference_is_not_published(self):
        error, events, before, after = exercise(HERE / 'run_td_v33loop.py', None, [])
        self.assertIsInstance(error, ValueError)
        self.assertEqual(events, [])
        self.assertIsNone(after)

    def test_generated_legacy_rows_are_not_published(self):
        error, events, before, after = exercise(HERE / 'run_td_v33loop.py', None, [{'id': 'legacy'}])
        self.assertIsInstance(error, ValueError)
        self.assertEqual(events, [])
        self.assertIsNone(after)

    def test_fresh_current_reference_is_published_and_reaches_batch(self):
        rows = [{'id': 'current', 'action_identity_contract': 'complete_action_identity_v1'}]
        error, events, before, after = exercise(HERE / 'run_td_v33loop.py', None, rows)
        self.assertIsNone(error)
        self.assertEqual(events, ['batch_started'])
        self.assertEqual(json.loads(after), {'rows': rows})



if __name__ == '__main__':
    unittest.main(verbosity=2)
