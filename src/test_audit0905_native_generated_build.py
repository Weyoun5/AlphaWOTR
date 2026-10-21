"""Candidate failures must preserve original source and installed binaries."""

from pathlib import Path
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from native_build_candidate import RUNTIME_DATA_FILES, build_candidate


class NativeGeneratedBuild(unittest.TestCase):
    def fixture(self, base):
        source = base / "source"
        (source / "src").mkdir(parents=True)
        (source / "data").mkdir()
        for name in RUNTIME_DATA_FILES:
            (source / "data" / name).write_bytes(b"required runtime fixture")
        for name in ("ai.py", "board.py"):
            (source / "src" / name).write_text("# literal source fixture\n")
        (source / "src/wotr_native.old.pyd").write_bytes(b"installed binary preserved")
        (source / "src/wotr_native.pyx").write_bytes(b"old body preserved")
        tooling = base / "tooling"
        tooling.mkdir()
        for name in ("build_wotr_native.py", "native_build_candidate.py"):
            (tooling / name).write_text("# provenance fixture\n")
        # Execute real child processes, with a deliberately failing compiler
        # command. No model/menu result is manufactured by this fixture.
        (tooling / "native_codegen.py").write_text(
            """from pathlib import Path
import hashlib,json,os,sys
out=Path(sys.argv[1]);src=out/'engine/src'
pyx=src/'wotr_native.pyx';pyx.write_text('def example(): return 1\\n')
(src/'setup_generated.py').write_text('raise SystemExit(19)\\n')
(out/'generation.json').write_text(json.dumps({'generated_pyx_sha256':hashlib.sha256(pyx.read_bytes()).hexdigest()}))
(out/'generator_native_flag.txt').write_text(os.environ.get('WOTR_NATIVE','missing'))
(out/'generator_optimization_flag.txt').write_text(os.environ.get('PYTHONOPTIMIZE','missing'))
"""
        )
        return source, tooling

    def test_actual_compiler_failure_preserves_installed_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            source, tooling = self.fixture(base)
            before = {p: p.read_bytes() for p in (source / "src").iterdir()}
            output = base / "candidate"
            with patch.dict(os.environ, {"WOTR_NATIVE": "1", "PYTHONOPTIMIZE": "1"}):
                with self.assertRaisesRegex(RuntimeError, "compile failed rc=19"):
                    build_candidate(source, output, tooling)
                self.assertEqual(os.environ["WOTR_NATIVE"], "1")
                self.assertEqual(os.environ["PYTHONOPTIMIZE"], "1")
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            self.assertEqual((output / "generator_native_flag.txt").read_text(), "0")
            self.assertEqual(
                (output / "generator_optimization_flag.txt").read_text(), "0"
            )
            result = json.loads((output / "build_manifest.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertTrue(result["originals_unchanged"])

    def test_existing_candidate_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            source, tooling = self.fixture(base)
            output = base / "candidate"
            output.mkdir()
            sentinel = output / "prior_result.json"
            sentinel.write_bytes(b"prior evidence preserved")
            with self.assertRaises(FileExistsError):
                build_candidate(source, output, tooling)
            self.assertEqual(sentinel.read_bytes(), b"prior evidence preserved")
            self.assertEqual(list(output.iterdir()), [sentinel])

    def test_missing_runtime_data_refused_before_creating_output(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            source, tooling = self.fixture(base)
            output = base / "candidate"
            for name in RUNTIME_DATA_FILES:
                path = source / "data" / name
                original = path.read_bytes()
                path.unlink()
                with self.subTest(name=name), self.assertRaisesRegex(
                    FileNotFoundError, name
                ):
                    build_candidate(source, output, tooling)
                self.assertFalse(output.exists())
                path.write_bytes(original)

    def test_unrelated_data_excluded_from_snapshot_and_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            source, tooling = self.fixture(base)
            unrelated = source / "data" / "unrelated_corpus.csv"
            unrelated.write_bytes(
                b"unrelated fixture must stay in its original directory"
            )
            output = base / "candidate"
            with self.assertRaisesRegex(RuntimeError, "compile failed rc=19"):
                build_candidate(source, output, tooling)
            copied = {path.name for path in (output / "engine/data").iterdir()}
            self.assertEqual(copied, set(RUNTIME_DATA_FILES))
            result = json.loads((output / "build_manifest.json").read_text())
            self.assertNotIn(str(unrelated), result["source_sha256_before"])
            self.assertEqual(
                unrelated.read_bytes(),
                b"unrelated fixture must stay in its original directory",
            )

    def test_candidate_cannot_be_created_inside_live_source_or_data(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            source, tooling = self.fixture(base)
            for name in ("src", "data"):
                output = source / name / "candidate"
                with self.subTest(name=name), self.assertRaisesRegex(
                    ValueError, "inside source"
                ):
                    build_candidate(source, output, tooling)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
