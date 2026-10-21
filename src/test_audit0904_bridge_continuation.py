"""Real JVM/Python protocol acceptance and truthful capture verdicts.

No GUI is created. Java is forced to a Windows-1252 default charset to exercise
the deployment environment that differs from Python's UTF-8 wire protocol.
"""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from wotr_bridge.tests import java_protocol_smoke as smoke
import test_live_session_census as census


class JavaProtocolIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='wotr-protocol-test-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.classes = Path(cls.temporary.name) / 'classes'
        cls.classes.mkdir()
        smoke.compile_probe(cls.classes, smoke.HOOK_SOURCE)

    def check_exchange(self, side, stale=False, jar=smoke.JAR, classes=None):
        with contextlib.redirect_stdout(io.StringIO()):
            result = smoke.exchange(classes or self.classes, side, stale_first=stale, jar=jar)
        self.assertEqual(result['errors'], [])
        self.assertEqual(len(result['requests']), 1)
        request = result['requests'][0]
        self.assertEqual(request.side, 'LIVE_' + side)
        self.assertEqual(request.log_lines[0],
                         '#SEQ req=7 dump=1 feed_paragraphs=2 raw_lines=2 session=9004')
        self.assertEqual(request.log_lines[1:], smoke.REQUEST_LOG.splitlines())
        self.assertEqual(result['response'], smoke.RESPONSE_LINES)
        self.assertIn('DEFAULT_CHARSET=windows-1252', result['java_stdout'])
        self.assertIn('reply_req=7', result['java_stdout'])
        self.assertNotIn('STALE_MUST_NOT_BE_ACCEPTED', result['response'])

    def test_shadow_java_request_and_python_response_preserve_unicode(self):
        self.check_exchange('SHADOW')

    def test_fp_java_request_and_python_response_preserve_unicode(self):
        self.check_exchange('FREE_PEOPLES')

    def test_shadow_drops_stale_response_then_accepts_current_sequence(self):
        self.check_exchange('SHADOW', stale=True)

    def test_fp_drops_stale_response_then_accepts_current_sequence(self):
        self.check_exchange('FREE_PEOPLES', stale=True)

    def test_reverting_utf8_boundary_reproduces_both_corruptions(self):
        """Sensitivity: restoring default-charset I/O must break both directions."""
        source = smoke.HOOK_SOURCE.read_text(encoding='utf-8')
        replacements = {
            'java.nio.file.Files.newBufferedWriter(\n                    tmp.toPath(), java.nio.charset.StandardCharsets.UTF_8)':
                'new FileWriter(tmp, false)',
            'java.nio.file.Files.newBufferedReader(\n                responseClaim.toPath(), java.nio.charset.StandardCharsets.UTF_8)':
                'new BufferedReader(new FileReader(responseClaim))',
        }
        for before, after in replacements.items():
            self.assertIn(before, source, 'The mutation must actually alter the wire boundary')
            source = source.replace(before, after)
        with tempfile.TemporaryDirectory(prefix='wotr-protocol-mutation-') as scratch:
            path = Path(scratch)
            mutant = path / 'AiOpp.java'
            mutant.write_text(source, encoding='utf-8')
            smoke.compile_probe(path, mutant)
            with contextlib.redirect_stdout(io.StringIO()):
                result = smoke.exchange(path, 'SHADOW')
        self.assertEqual(result['errors'], [])
        self.assertEqual(len(result['requests']), 1)
        self.assertNotEqual(result['requests'][0].log_lines[1:], smoke.REQUEST_LOG.splitlines())
        self.assertIn('\ufffd', result['requests'][0].raw)
        self.assertNotEqual(result['response'], smoke.RESPONSE_LINES)

    def test_candidate_jar_contains_fix_and_preserves_other_class_families(self):
        with tempfile.TemporaryDirectory(prefix='wotr-candidate-test-') as scratch:
            path = Path(scratch)
            candidate = path / 'candidate.jar'
            manifest = smoke.build_candidate(candidate)
            self.assertFalse(manifest['installed'])
            with zipfile.ZipFile(smoke.JAR) as old, zipfile.ZipFile(candidate) as new:
                shared = set(old.namelist()) & set(new.namelist())
                changed = [name for name in shared if old.read(name) != new.read(name)]
                self.assertIn('AiOpp.class', changed)
                self.assertTrue(all(name.startswith('AiOpp') and name.endswith('.class')
                                    for name in changed), changed)
                self.assertNotIn('JavaProtocolProbe.class', new.namelist())
            helper = path / 'helper'
            helper.mkdir()
            smoke.compile_probe(helper, jar=candidate)
            self.check_exchange('SHADOW', stale=True, jar=candidate, classes=helper)


class CensusVerdictTests(unittest.TestCase):
    @staticmethod
    def record(board=(), staging=(), invariants=()):
        return {'board': list(board), 'staging': list(staging),
                'invariants': list(invariants)}

    def test_reviewed_zero_difference_capture_passes(self):
        self.assertEqual(census.verdict(self.record(), {'census': 'clean'}), ('PASS', None))

    def test_packaged_real_captures_keep_clean_known_and_unverified_distinct(self):
        fixtures = ROOT / 'wotr_bridge/tests/fixtures/census_samples'
        for name, expected in (('fpgame10', 'PASS'), ('fpgame11', 'KNOWN_FAIL'),
                               ('fpgame1', 'UNVERIFIED')):
            with self.subTest(capture=name):
                record = census.census_fixture(name, str(fixtures))
                self.assertNotIn('error', record)
                self.assertEqual(census.verdict(record, census.expected_for(name, str(fixtures)))[0],
                                 expected)

    def test_known_discrepancy_never_counts_as_pass(self):
        verdict, why = census.verdict(self.record(board=['missing leader']),
                                     {'census': 'known', 'rows': 1, 'bug': 'example'})
        self.assertEqual(verdict, 'KNOWN_FAIL')
        self.assertIn('remain', why)

    def test_unsigned_clean_and_dirty_captures_are_unverified(self):
        for board in ([], ['wrong progress']):
            with self.subTest(board=board):
                self.assertEqual(census.verdict(self.record(board=board), None)[0], 'UNVERIFIED')

    def test_unmatched_dice_or_deck_snapshot_cannot_certify_game(self):
        for row in ('dice shadow pool: engine=5 client=8',
                    'deck fp char deck: engine=16 client=13'):
            with self.subTest(row=row):
                self.assertEqual(census.verdict(self.record(staging=[row]),
                                               {'census': 'clean'})[0], 'UNVERIFIED')

    def test_unexpected_board_difference_fails(self):
        self.assertEqual(census.verdict(self.record(board=['wrong pawn']),
                                       {'census': 'clean'})[0], 'FAIL')

    def test_known_discrepancy_disappearing_requires_review(self):
        self.assertEqual(census.verdict(self.record(),
                                       {'census': 'known', 'rows': 1})[0], 'UNVERIFIED')

    def test_missing_requested_fixture_is_nonzero(self):
        with tempfile.TemporaryDirectory() as path, contextlib.redirect_stdout(io.StringIO()):
            self.assertNotEqual(census.main(['--fixtures', path, 'does-not-exist']), 0)

    def test_incomplete_fixture_directory_is_not_silently_omitted(self):
        with tempfile.TemporaryDirectory() as path, contextlib.redirect_stdout(io.StringIO()):
            (Path(path) / 'missing-pickle').mkdir()
            self.assertNotEqual(census.main(['--fixtures', path]), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
