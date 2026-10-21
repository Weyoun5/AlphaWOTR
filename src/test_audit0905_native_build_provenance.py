"""A native rebuild must not certify untouched stale mirror bodies."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import build_wotr_native as build


class NativeBuildProvenance(unittest.TestCase):
    def attempt_stale(self, declaration):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'wotr_native.pyx'
            original = ('def encode_state(state, side):\n    return [0] * 3742\n'
                        + declaration).encode('ascii')
            source.write_bytes(original)
            binary = root / 'wotr_native.old.pyd'
            binary.write_bytes(b'preserved existing binary')
            artifacts = root / 'build_native'
            artifacts.mkdir()
            sentinel = artifacts / 'keep.txt'
            sentinel.write_text('preserved build output')
            with patch.object(build, 'HERE', str(root)), patch.object(build, 'PYX', str(source)), \
                    patch.object(build.subprocess, 'call', side_effect=AssertionError('compiler reached')):
                with self.assertRaisesRegex(SystemExit, 'mirror provenance'):
                    build.main()
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(binary.read_bytes(), b'preserved existing binary')
            self.assertEqual(sentinel.read_text(), 'preserved build output')

    def test_unstamped_old_body_refused_before_any_mutation(self):
        self.attempt_stale('')

    def test_old_fingerprint_cannot_be_restamped_current(self):
        self.attempt_stale('\n' + build.FP_BEGIN + "\nMIRROR_FINGERPRINT = '" + '0' * 40 + "'\n" + build.FP_END + '\n')

    def test_duplicate_marker_refused(self):
        self.attempt_stale('\n' + build.FP_BEGIN + '\n' + build.FP_BEGIN + '\n' + build.FP_END + '\n')

    def test_matching_existing_declaration_is_checked_without_rewriting(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'mirror.pyx'
            fingerprint = build.mirror_fingerprint()
            original = (build.FP_BEGIN + "\nMIRROR_FINGERPRINT = '" + fingerprint + "'\n" + build.FP_END + '\n').encode('ascii')
            source.write_bytes(original)
            self.assertEqual(build.require_mirror_provenance(source, fingerprint), fingerprint)
            self.assertEqual(source.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
