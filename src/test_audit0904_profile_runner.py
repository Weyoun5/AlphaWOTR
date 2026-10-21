"""Worker accounting and identity without games, subprocesses, or training."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import queue
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_td_v33sp as runner


class ProfileRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='wotr_runner_accounting_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.addCleanup(random.setstate, random.getstate())
        self.env = patch.dict(os.environ, {
            'WOTR_REWARD_CONTRACT': 'predecision_profile_v1',
            'WOTR_SP_FP_PROFILES': 'ring', 'WOTR_SP_SH_PROFILES': 'military',
            'WOTR_SP_FROZEN_FP': '', 'WOTR_SP_FROZEN_SH': '',
            'WOTR_SP_FROZEN_P': '', 'WOTR_SP_ATTEMPTS_LOG': '',
            'WOTR_SP_PHASE_SUMMARY': '', 'WOTR_SP_REPORT': str(self.root / 'report.md')})
        self.env.start()
        self.addCleanup(self.env.stop)
        runner._CHECKPOINT_IDS.clear()
        self.addCleanup(runner._CHECKPOINT_IDS.clear)

    def checkpoint(self, name):
        path = self.root / name
        path.write_bytes(name.encode('ascii'))
        return str(path)

    def expected_id(self, name):
        return hashlib.sha256(name.encode('ascii')).hexdigest()

    def fake_modules(self, episode):
        fake_ai = SimpleNamespace(
            FP_PROFILES=['ring'], SH_PROFILES=['military'], PLAYER_PROFILES={},
            load_value_net=lambda path, side: SimpleNamespace(path=path, eval=lambda: None),
            _run_single_episode=episode)
        return {'ai': fake_ai,
                'numpy': SimpleNamespace(random=SimpleNamespace(seed=lambda value: None)),
                'torch': SimpleNamespace(set_num_threads=lambda value: None,
                                         device=lambda value: value)}

    def test_checkpoint_content_hash_is_read_once_per_worker_path(self):
        path = self.checkpoint('checkpoint.pt')
        import builtins
        with patch('builtins.open', wraps=builtins.open) as opened:
            first = runner._checkpoint_id(path)
            second = runner._checkpoint_id(os.path.join(self.temp.name, '.', 'checkpoint.pt'))
        self.assertEqual(first, self.expected_id('checkpoint.pt'))
        self.assertEqual(first, second)
        self.assertEqual(opened.call_count, 1)

    def test_worker_uses_actual_live_or_frozen_pair_ids_in_config_and_stats(self):
        names = ('live_fp.pt', 'live_sh.pt', 'frozen_fp.pt', 'frozen_sh.pt')
        paths = {name: self.checkpoint(name) for name in names}
        calls = []

        def episode(fp_net, sh_net, config):
            calls.append((fp_net.path, sh_net.path, config))
            return {'game_stats': {'winner': 'fp', 'victory_type': 'ring_destroyed'}}

        sink = queue.Queue()
        with patch.dict(sys.modules, self.fake_modules(episode)), \
             patch.dict(os.environ, {'WOTR_SP_FROZEN_FP': paths['frozen_fp.pt'],
                                     'WOTR_SP_FROZEN_SH': paths['frozen_sh.pt'],
                                     'WOTR_SP_FROZEN_P': '1.0'}), \
             patch.object(runner, '_frozen_pick', side_effect=['fp', 'sh', None]):
            runner._worker(2, 3, 1, str(self.root / 'traces'), 123,
                           paths['live_fp.pt'], paths['live_sh.pt'], sink)
        expected = [('frozen_fp.pt', 'live_sh.pt'), ('live_fp.pt', 'frozen_sh.pt'),
                    ('live_fp.pt', 'live_sh.pt')]
        self.assertEqual(len(calls), 3)
        for index, (fp_name, sh_name) in enumerate(expected):
            fp_path, sh_path, config = calls[index]
            message = sink.get_nowait()
            identities = {'fp': self.expected_id(fp_name), 'sh': self.expected_id(sh_name)}
            self.assertEqual((fp_path, sh_path), (paths[fp_name], paths[sh_name]))
            self.assertEqual(config['checkpoint_ids'], identities)
            self.assertEqual(message[0], 'g')
            self.assertEqual(message[2]['checkpoint_ids'], identities)
            self.assertEqual(message[2]['reward_contract'], 'predecision_profile_v1')

    def test_per_profile_roster_selects_the_profile_checkpoint_identity(self):
        names = ('live_fp_ring.pt', 'live_fp_military.pt',
                 'live_sh_corruption.pt', 'live_sh_military.pt')
        for name in names:
            self.checkpoint(name)
        configs = []

        def episode(fp_net, sh_net, config):
            configs.append(config)
            return {'game_stats': {'winner': 'fp', 'victory_type': 'military'}}

        with patch.dict(sys.modules, self.fake_modules(episode)), \
             patch.dict(os.environ, {'WOTR_SP_FP_PROFILES': 'ring,military',
                                     'WOTR_SP_SH_PROFILES': 'corruption,military'}), \
             patch.object(random, 'choice', side_effect=['ring', 'corruption', 'military', 'military']):
            runner._worker(0, 2, 1, str(self.root / 'traces'), 123,
                           None, None, queue.Queue(), net_dir=str(self.root))
        self.assertEqual([c['checkpoint_ids'] for c in configs], [
            {'fp': self.expected_id('live_fp_ring.pt'),
             'sh': self.expected_id('live_sh_corruption.pt')},
            {'fp': self.expected_id('live_fp_military.pt'),
             'sh': self.expected_id('live_sh_military.pt')}])

    def test_episode_error_preserves_full_detail_profiles_and_ids(self):
        fp_path, sh_path = self.checkpoint('fp.pt'), self.checkpoint('sh.pt')
        failure_text = 'bad "gesture"\n' + 'detail ' * 60

        def episode(*args):
            raise RuntimeError(failure_text)

        sink = queue.Queue()
        with patch.dict(sys.modules, self.fake_modules(episode)):
            runner._worker(7, 1, 1, str(self.root / 'traces'), 123, fp_path, sh_path, sink)
        message = sink.get_nowait()
        self.assertEqual(message[:2], ('e', 7))
        self.assertIsInstance(message[2], str)
        self.assertLessEqual(len(message[2]), 200)
        record, failure = runner._message_record(message, 'phase-A', 'batch 1')
        self.assertIsNone(failure)
        self.assertEqual(record['error'], repr(RuntimeError(failure_text)))
        self.assertEqual(record['profiles'], {'fp': 'ring', 'sh': 'military'})
        self.assertEqual(record['checkpoint_ids'], {
            'fp': self.expected_id('fp.pt'), 'sh': self.expected_id('sh.pt')})
        self.assertEqual(record['episode'], 0)
        self.assertNotIn('winner', record)
        path = str(self.root / 'attempts.jsonl')
        runner._append_attempt(path, record)
        lines = Path(path).read_text().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0]), record)

    def test_net_load_failure_has_no_fabricated_game_or_profile(self):
        sink = queue.Queue()
        with patch.dict(sys.modules, self.fake_modules(lambda *args: None)), \
             patch.object(runner, '_load_net', side_effect=OSError('checkpoint unreadable')):
            runner._worker(4, 8, 1, str(self.root / 'traces'), 123, 'bad.pt', 'other.pt', sink)
        message = sink.get_nowait()
        record, failure = runner._message_record(message, 'phase-A', 'batch 1')
        self.assertIsNone(record)
        self.assertEqual(failure['stage'], 'net_load')
        self.assertEqual(failure['assigned_games'], 8)
        self.assertNotIn('profiles', failure)
        self.assertNotIn('episode', failure)
        self.assertIn('checkpoint unreadable', failure['error'])

    def test_legacy_error_tuple_remains_worker_failure_with_unknown_attempt(self):
        record, failure = runner._message_record(('e', 3, 'old diagnostic'), 'p', 'b')
        self.assertIsNone(record)
        self.assertEqual(failure['error'], 'old diagnostic')
        self.assertNotIn('profiles', failure)

    def test_phase_summary_exposes_missing_coverage_without_inventing_results(self):
        attempts = [
            {'status': 'completed', 'profiles': {'fp': 'ring', 'sh': 'military'},
             'winner': 'fp', 'victory_type': 'ring_destroyed'},
            {'status': 'error', 'profiles': {'fp': 'ring', 'sh': 'military'},
             'error': 'actual attempted episode failed'}]
        path = str(self.root / 'summary.json')
        summary = runner._write_phase_summary(path, 'phase-A', 'batch 1', 5, attempts,
                                              [{'stage': 'net_load', 'worker_id': 2}],
                                              str(self.root / 'attempts.jsonl'), finished=True)
        self.assertEqual((summary['planned'], summary['observed'], summary['unreported']), (5, 2, 3))
        self.assertFalse(summary['coverage_complete'])
        self.assertEqual((summary['completed'], summary['error_attempts']), (1, 1))
        fp = summary['profile_metrics']['fp']['cohorts'][0]
        self.assertEqual(fp['counts']['attempted'], 2)
        self.assertEqual(fp['rates']['on_target_win_rate_attempted'],
                         {'numerator': 1, 'denominator': 2, 'value': .5})
        self.assertEqual(json.loads(Path(path).read_text()), summary)
        self.assertFalse(Path(path + '.tmp').exists())

    def test_report_paths_follow_isolated_report_or_games_log(self):
        attempts, summary = runner._report_paths()
        self.assertEqual(attempts, str(self.root / 'report.attempts.jsonl'))
        self.assertEqual(summary, str(self.root / 'report.phase.json'))
        with patch.dict(os.environ, {'WOTR_SP_REPORT': '',
                                     'WOTR_SP_GAMES_LOG': str(self.root / 'isolated.jsonl')}):
            self.assertEqual(runner._report_paths(), (
                str(self.root / 'isolated.attempts.jsonl'), str(self.root / 'isolated.phase.json')))

    def test_phase_collector_preserves_success_only_game_log_and_error_attempts(self):
        good = {'winner': 'fp', 'victory_type': 'ring_destroyed',
                'profiles': {'fp': 'ring', 'sh': 'military'}, 'turns': 1}
        failed = {'record_type': 'game_attempt', 'status': 'error', 'episode': 1,
                  'profiles': {'fp': 'ring', 'sh': 'military'}, 'error': 'engine failed'}
        sink = queue.Queue()
        sink.put(('g', 0, good))
        sink.put(('e', 0, 'engine failed', failed))
        logged = []
        fake_ai = SimpleNamespace(append_game_log=lambda path, data, label: logged.append(data),
                                  write_td_window_report=lambda *args: None)
        process = SimpleNamespace(start=lambda: None, join=lambda timeout: None,
                                  is_alive=lambda: False, exitcode=0)
        context = SimpleNamespace(Queue=lambda: sink, Process=lambda **kwargs: process)
        with patch('multiprocessing.get_context', return_value=context), \
             patch.dict(sys.modules, {'ai': fake_ai}), contextlib.redirect_stdout(io.StringIO()):
            stats, errors = runner.run_phase('batch 1', 2, 1, 1, str(self.root / 'traces'), 123)
        self.assertEqual(logged, [good])
        self.assertEqual(stats, [good])
        self.assertEqual(errors, [('e', 0, 'engine failed', failed)])
        attempts_path, summary_path = runner._report_paths()
        saved = [json.loads(line) for line in Path(attempts_path).read_text().splitlines()]
        self.assertEqual([row['status'] for row in saved], ['completed', 'error'])
        summary = json.loads(Path(summary_path).read_text())
        self.assertEqual((summary['planned'], summary['observed'], summary['unreported']), (2, 2, 0))
        self.assertTrue(summary['finished'])
        self.assertTrue(summary['coverage_complete'])
        self.assertEqual({row['phase_id'] for row in saved}, {summary['phase_id']})

    def test_dead_worker_leaves_unreported_gap_without_attempt_record(self):
        class EmptySink:
            def get(self, timeout):
                raise queue.Empty

        process = SimpleNamespace(start=lambda: None, join=lambda timeout: None,
                                  is_alive=lambda: False, exitcode=7)
        context = SimpleNamespace(Queue=EmptySink, Process=lambda **kwargs: process)
        fake_ai = SimpleNamespace(write_td_window_report=lambda *args: None)
        with patch('multiprocessing.get_context', return_value=context), \
             patch.dict(sys.modules, {'ai': fake_ai}), contextlib.redirect_stdout(io.StringIO()):
            stats, errors = runner.run_phase('batch 1', 3, 1, 1, str(self.root / 'traces'), 123)
        self.assertEqual((stats, errors), ([], []))
        attempts_path, summary_path = runner._report_paths()
        self.assertFalse(Path(attempts_path).exists())
        summary = json.loads(Path(summary_path).read_text())
        self.assertEqual((summary['planned'], summary['observed'], summary['unreported']), (3, 0, 3))
        self.assertEqual(summary['worker_failures'][0]['exitcode'], 7)
        self.assertEqual(summary['profile_metrics']['fp']['cohorts'], [])


if __name__ == '__main__':
    unittest.main()
