"""Actual child generation selection plus the existing build-failure controls."""
from pathlib import Path
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from native_build_candidate import build_candidate
spec = importlib.util.spec_from_file_location('original_native_controls', ROOT / 'src/test_audit0905_native_generated_build.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)


class GenerationControls(unittest.TestCase):
    fixture = original.NativeGeneratedBuild.fixture

    def test_explicit_generation_reaches_actual_child_and_is_recorded(self):
        for generation in ('v41', 'v45', 'v46', 'v47', 'v48', 'v49', 'v50', 'v51'):
            with self.subTest(generation=generation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, tooling = self.fixture(root)
                generator = tooling / 'native_codegen.py'
                with generator.open('a') as stream:
                    stream.write("\n(out/'selected_generation.txt').write_text(os.environ['WOTR_ACT_GEN'])\n")
                output = root / 'output'
                with patch.dict(os.environ, {'WOTR_ACT_GEN': 'conflicting_parent'}):
                    with self.assertRaisesRegex(RuntimeError, 'compile failed rc=19'):
                        build_candidate(source, output, tooling, action_generation=generation)
                    self.assertEqual(os.environ['WOTR_ACT_GEN'], 'conflicting_parent')
                self.assertEqual((output / 'selected_generation.txt').read_text(), generation)
                self.assertEqual(json.loads((output / 'build_manifest.json').read_text())['action_generation'], generation)

    def test_unknown_generation_refused_before_output_or_subprocess(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, tooling = self.fixture(root)
            output = root / 'output'
            with self.assertRaisesRegex(ValueError, 'generation'):
                build_candidate(source, output, tooling, action_generation='v999')
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
