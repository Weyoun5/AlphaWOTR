"""Audit imports preserve the GPU assignment used by the served launcher."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


SOURCE = Path(os.environ.get('WOTR_TEST_TRACE_AUDIT_SOURCE',
                            Path(__file__).with_name('trace_audit.py'))).resolve()


class AuditDeviceEnvironment(unittest.TestCase):
    def probe(self, visibility, cli=False):
        env = dict(os.environ)
        env.pop('CUDA_VISIBLE_DEVICES', None)
        if visibility is not None:
            env['CUDA_VISIBLE_DEVICES'] = visibility
        program = """
import json, os, runpy, sys
source, mode, source_dir = sys.argv[1:]
sys.path.insert(0, source_dir)
sys.argv = [source]
try:
    runpy.run_path(source, run_name=mode)
except SystemExit as exc:
    # The CLI deliberately rejects an empty target list; no trace is read.
    assert mode == '__main__' and exc.code == 1
print(json.dumps({'visibility': os.environ.get('CUDA_VISIBLE_DEVICES')}))
"""
        result = subprocess.run(
            [sys.executable, '-B', '-X', 'utf8', '-c', program,
             str(SOURCE), '__main__' if cli else 'trace_audit_import_probe',
             str(Path(__file__).resolve().parent)],
            env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])
        return json.loads(result.stdout.splitlines()[-1])['visibility']

    def test_import_does_not_hide_unassigned_gpu(self):
        self.assertIsNone(self.probe(None))

    def test_import_preserves_scheduler_assignment(self):
        self.assertEqual(self.probe('3,7'), '3,7')

    def test_standalone_audit_keeps_default_cpu_mode(self):
        self.assertEqual(self.probe(None, cli=True), '-1')


if __name__ == '__main__':
    unittest.main()
