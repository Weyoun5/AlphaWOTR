"""Test the test-process boundary with small, independent child programs."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_test_tiers as runner


class TestResultContracts(unittest.TestCase):
    def test_dedicated_gpu_child_preserves_scheduler_masks(self):
        import os
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / 'gpu_child.py'
            script.write_text(
                "import os\nassert os.environ['CUDA_VISIBLE_DEVICES']=='2,3'\n"
                "assert os.environ['HIP_VISIBLE_DEVICES']=='1'\n"
                "assert os.environ['ROCR_VISIBLE_DEVICES']=='7'\nprint('TOTAL=1 FAIL=0')\n",
                encoding='utf-8')
            with patch.dict(runner.TIERS, {'gpu-training': [str(script)]}), \
                 patch.dict(os.environ, CUDA_VISIBLE_DEVICES='2,3',
                            HIP_VISIBLE_DEVICES='1', ROCR_VISIBLE_DEVICES='7'):
                self.assertTrue(runner.run_suite(str(script))['ok'])

    def test_cli_lists_every_imported_suite(self):
        import subprocess
        result = subprocess.run([sys.executable, runner.__file__, 'list'],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        listed = {fields[0]: fields[1:] for line in result.stdout.splitlines()
                  if (fields := line.split())}
        self.assertEqual(listed, runner.TIERS)

    def test_empty_success_is_not_a_test_run(self):
        self.assertFalse(runner.parse_result('', 0)['ok'])
        self.assertFalse(runner.parse_result('ALL PASS\n', 0)['ok'])

    def test_zero_unittest_cases_is_not_a_test_run(self):
        self.assertFalse(runner.parse_result('Ran 0 tests in 0.001s\nOK', 0)['ok'])

    def test_actual_unittest_cases_count(self):
        result = runner.parse_result('Ran 7 tests in 0.001s\nOK', 0)
        self.assertTrue(result['ok'])
        self.assertEqual(result['total'], 7)

    def test_skips_are_incomplete(self):
        result = runner.parse_result('Ran 7 tests in 0.001s\nOK (skipped=2)', 0)
        self.assertFalse(result['ok'])
        self.assertEqual(result['skipped'], 2)

    def test_legacy_skip_inside_passing_case_is_incomplete(self):
        result = runner.parse_result('SKIP fixture missing\nPASS test_fixture\nALL PASS', 0)
        self.assertFalse(result['ok'])

    def test_legacy_scenario_counts_cases_not_failures(self):
        result = runner.parse_result('-- test_one\n   ok\n-- test_two\n   ok\nscenario: 0 failure(s)', 0)
        self.assertTrue(result['ok'])
        self.assertEqual(result['total'], 2)

    def test_equals_summary_cannot_hide_failure(self):
        self.assertFalse(runner.parse_result('TOTAL=500 FAIL=1', 0)['ok'])

    def test_nonzero_exit_overrides_passing_summary(self):
        self.assertFalse(runner.parse_result('TOTAL 500 FAIL 0', 1)['ok'])

    def test_unverified_observation_is_not_a_passing_test(self):
        result = runner.parse_result('[MEASURED 1-board] fixture\nTOTAL 1 FAIL 0', 0)
        self.assertFalse(result['ok'])
        self.assertEqual(result['unverified'], 1)

    def test_reported_error_overrides_zero_exit(self):
        self.assertFalse(runner.parse_result('TOTAL 5 PASS 4 FAIL 0 SKIP 0 ERROR 1', 0)['ok'])

    def test_known_census_failure_cannot_count_as_success(self):
        result = runner.parse_result('[PASS] clean\n[KNOWN_FAIL] known drift', 0)
        self.assertEqual((result['total'], result['fail']), (2, 1))
        self.assertFalse(result['ok'])

    def test_legacy_failure_detail_does_not_count_as_another_case(self):
        # Captured taxonomy runner format: one failed case, one detail row.
        output = ("  [FAIL] test_every_kind_classified\n"
                  "    FAIL: ('unclassified_kinds', ['cc_prior_reveal'])\n"
                  "  [PASS] test_classify_uses_kind_before_words\n"
                  "  [PASS] test_partition_counts_match_classes\n"
                  "TOTAL 3  FAIL 1\n")
        result = runner.parse_result(output, 1)
        self.assertEqual((result['total'], result['fail']), (3, 1))
        self.assertFalse(result['ok'])
        # The same case records also work without a totals banner.
        result = runner.parse_result(output.rsplit('TOTAL', 1)[0], 1)
        self.assertEqual((result['total'], result['fail']), (3, 1))

    def test_multiple_details_belong_to_the_announced_failed_case(self):
        result = runner.parse_result(
            '[FAIL] test_one\n  FAIL: first assertion\n'
            '  FAIL: second assertion\n[PASS] test_two\nTOTAL 2 FAIL 1', 1)
        self.assertEqual((result['total'], result['fail']), (2, 1))

    def test_failing_case_marker_overrides_zero_failure_summary(self):
        result = runner.parse_result('[FAIL] test_one\nTOTAL 1 FAIL 0', 0)
        self.assertEqual(result['fail'], 1)
        self.assertFalse(result['ok'])

    def test_failure_detail_cannot_hide_behind_passing_case_markers(self):
        result = runner.parse_result(
            '[PASS] test_one\n  FAIL: lost assertion\nTOTAL 1 FAIL 0', 0)
        self.assertEqual(result['fail'], 1)
        self.assertFalse(result['ok'])

    def test_named_subchecks_count_the_parent_case_once(self):
        output = ('PASS test_one\nFAIL test_two :: sub_a :: first problem\n'
                  'FAIL test_two :: sub_b :: second problem\n'
                  'FAIL test_two :: sub_c :: third problem\nTOTAL=2 FAIL=1')
        result = runner.parse_result(output, 1)
        self.assertEqual((result['total'], result['fail']), (2, 1))
        self.assertFalse(result['ok'])
        result = runner.parse_result(output.rsplit('TOTAL', 1)[0], 1)
        self.assertEqual((result['total'], result['fail']), (2, 1))

    def test_plain_legacy_failure_cases_remain_counted(self):
        result = runner.parse_result('PASS test_one\nFAIL test_two\nFAIL test_three', 0)
        self.assertEqual((result['total'], result['fail']), (3, 2))
        self.assertFalse(result['ok'])

    def test_unverified_census_case_counts_as_incomplete(self):
        result = runner.parse_result('[PASS] clean\n[UNVERIFIED] unsigned', 0)
        self.assertEqual((result['total'], result['unverified']), (2, 1))
        self.assertFalse(result['ok'])

    def test_real_child_output_and_exit_are_both_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / 'child.py'
            script.write_text("print('TOTAL=500 FAIL=1')\n", encoding='utf-8')
            result = runner.run_suite(str(script))
            self.assertEqual(result['rc'], 0)
            self.assertFalse(result['ok'])

    def test_real_child_timeout_is_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / 'child.py'
            script.write_text('import time\ntime.sleep(5)\n', encoding='utf-8')
            result = runner.run_suite(str(script), timeout=0.1)
            self.assertTrue(result['timed_out'])
            self.assertFalse(result['ok'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
