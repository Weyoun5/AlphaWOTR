"""Startup-only callers and real process cleanup; no fabricated training data."""
from pathlib import Path
import contextlib
import copy
import io
import multiprocessing as mp
import os
import queue
import sys
import tempfile
import unittest
from unittest.mock import patch

NETS = None
os.environ.update(CUDA_VISIBLE_DEVICES='-1', WOTR_NATIVE='0',
                  WOTR_REWARD_CONTRACT='predecision_profile_v1', WOTR_NET_ARCH='base')
os.environ.setdefault('WOTR_ACT_GEN', 'v41')
import ai
import arena_scale
import bench_scale
import infer_server
import served_lifecycle
from test_audit0905_served_handoff import Pipe, metadata, protocol_exchange


def cpu_server(conns, paths, arch):
    infer_server.server_main(conns, paths, 'base', device='cpu')


def dead_worker(*args):
    raise ValueError('literal worker startup refusal')


def reported_error_worker(wid, conn_a, conn_b, n_games, sims, out_q, seed0=4000, episode_stride=1):
    for game in range(n_games):
        out_q.put(('e', wid, True, 'literal episode refusal', None, None))


def pair_metadata():
    desc = metadata()
    desc['sh'] = dict(desc['fp'], reward_profile='military')
    return desc


class CallerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global NETS
        import torch
        from trace_decisions import ACTION_ENCODING_CONTRACT
        torch.set_num_threads(1)
        cls.models = tempfile.TemporaryDirectory(prefix='wotr_handoff_models_')
        NETS = Path(cls.models.name)
        with torch.random.fork_rng():
            torch.manual_seed(945)
            for side, profile in [('fp', 'ring'), ('sh', 'military')]:
                net = ai.WOTRValueNet(ai.get_feature_dim(), side=side)
                torch.save(dict(model_state=net.state_dict(), side=side, reward_profile=profile,
                                reward_contract='predecision_profile_v1',
                                observation_contract=ai.OBSERVATION_CONTRACT,
                                action_encoding_contract=ACTION_ENCODING_CONTRACT,
                                input_dim=ai.get_feature_dim(), action_dim=ai.ACTION_ENCODING_DIM,
                                head_act='scaled2', arch_name=ai.NET_ARCH_NAME),
                           NETS / ('live_%s_%s.pt' % (side, profile)))

    @classmethod
    def tearDownClass(cls):
        cls.models.cleanup()

    def setUp(self):
        self.env = patch.dict(os.environ, WOTR_REWARD_CONTRACT='predecision_profile_v1')
        self.env.start()
        self.addCleanup(self.env.stop)
        self.old_client = getattr(ai, 'INFER_CLIENT', None)
        self.old_mcts = ai.mcts_action_with_policy
        self.addCleanup(setattr, ai, 'INFER_CLIENT', self.old_client)
        self.addCleanup(setattr, ai, 'mcts_action_with_policy', self.old_mcts)

    # Metadata-only fake pipe; actual subprocess cases below use real transport.
    @patch.object(infer_server, "_exchange", new=protocol_exchange)
    def test_arena_matching_metadata_allows_startup(self):
        a, b = Pipe(pair_metadata()), Pipe(pair_metadata())
        arena_scale._arena_worker(0, a, b, 0, 1, queue.Queue())
        self.assertEqual(a.sent, [('describe', 1)])
        self.assertEqual(b.sent, [('describe', 1)])

    # Metadata-only fake pipe; actual subprocess cases below use real transport.
    @patch.object(infer_server, "_exchange", new=protocol_exchange)
    def test_arena_mismatch_refused_before_episode(self):
        desc = pair_metadata()
        desc['fp']['input_dim'] += 1
        a, b = Pipe(desc), Pipe(pair_metadata())
        with patch.object(ai, '_run_single_episode', side_effect=AssertionError('episode must not start')):
            with self.assertRaisesRegex(ValueError, 'input_dim'):
                arena_scale._arena_worker(0, a, b, 1, 1, queue.Queue())
        self.assertTrue(a.closed)
        self.assertEqual(b.sent, [])

    def bench_start(self, desc):
        stop = mp.get_context('spawn').Event()
        stop.set()
        output = queue.Queue()
        pipe = Pipe(desc)
        bench_scale._bench_worker(0, 'server-base', pipe, 'base', 1, '', '', stop, output)
        return pipe, output

    # Metadata-only fake pipe; actual subprocess cases below use real transport.
    @patch.object(infer_server, "_exchange", new=protocol_exchange)
    def test_benchmark_matching_metadata_allows_startup(self):
        pipe, output = self.bench_start(pair_metadata())
        self.assertEqual(pipe.sent, [('describe', 1)])
        self.assertTrue(output.empty())

    # Metadata-only fake pipe; actual subprocess cases below use real transport.
    @patch.object(infer_server, "_exchange", new=protocol_exchange)
    def test_benchmark_mismatch_reports_startup_error(self):
        desc = pair_metadata()
        desc['fp']['action_dim'] += 1
        pipe, output = self.bench_start(desc)
        error = output.get_nowait()
        self.assertEqual(error[0], 'e')
        self.assertIn('worker init failed', error[3])
        self.assertIn('action_dim', error[3])
        self.assertTrue(pipe.closed)

    def parity(self, mismatch):
        cleanup_reports = []
        cleanup = served_lifecycle.cleanup
        def capture(*args, **kwargs):
            report = cleanup(*args, **kwargs)
            cleanup_reports.append(report)
            return report
        attach = infer_server.attach_client
        def handoff(*args, **kwargs):
            self.assertIs(kwargs.get('require_metadata'), True)
            if mismatch:
                with patch.object(ai, 'OBSERVATION_CONTRACT', 'literal_other_generation'):
                    return attach(*args, **kwargs)
            return attach(*args, **kwargs)
        with patch.object(bench_scale, '_server_entry', cpu_server), patch.object(
                served_lifecycle, 'cleanup', capture), patch.object(infer_server, 'attach_client', handoff):
            if mismatch:
                with self.assertRaisesRegex(ValueError, 'observation contract'):
                    bench_scale.parity_check(str(NETS / 'live_fp_ring.pt'), str(NETS / 'live_sh_military.pt'))
            else:
                self.assertTrue(bench_scale.parity_check(str(NETS / 'live_fp_ring.pt'), str(NETS / 'live_sh_military.pt')))
        self.assertEqual(len(cleanup_reports), 1)
        report = cleanup_reports[0]
        self.assertTrue(report['stop_sent'])
        self.assertEqual(report['server_exitcode'], 0)
        self.assertEqual(report['alive'], [])
        self.assertFalse(report['server_terminated'])
        self.assertFalse(report['server_killed'])
        self.assertIs(ai.INFER_CLIENT, self.old_client)

    def test_actual_cpu_parity_normal_success_cleans_server(self):
        self.parity(False)

    def test_actual_cpu_parity_mismatch_cleans_server(self):
        self.parity(True)

    def failed_arena(self, worker, exitcode):
        reports = []
        cleanup = served_lifecycle.cleanup
        def capture(*args, **kwargs):
            report = cleanup(*args, **kwargs)
            reports.append(report)
            return report
        argv = ['arena_scale', '--a-arch', 'wide', '--b-arch', 'wide',
                '--a-fp', str(NETS / 'live_fp_ring.pt'), '--a-sh', str(NETS / 'live_sh_military.pt'),
                '--b-fp', str(NETS / 'live_fp_ring.pt'), '--b-sh', str(NETS / 'live_sh_military.pt'),
                '--workers', '1', '--games-per-worker', '1', '--sims', '1']
        output = io.StringIO()
        with patch.object(sys, 'argv', argv), patch.object(arena_scale, '_server_entry', cpu_server), patch.object(
                arena_scale, '_arena_worker', worker), patch.object(served_lifecycle, 'cleanup', capture), contextlib.redirect_stdout(output):
            with self.assertRaisesRegex(SystemExit, 'evaluation incomplete'):
                arena_scale.main()
        self.assertIn('ARENA INCOMPLETE', output.getvalue())
        self.assertNotIn('Elo(A-B)', output.getvalue())
        self.assertNotIn('A score =', output.getvalue())
        self.assertEqual(len(reports), 2)
        for report in reports:
            self.assertTrue(report['stop_sent'])
            self.assertEqual(report['server_exitcode'], 0)
            self.assertEqual(report['alive'], [])
            self.assertFalse(report['server_terminated'])
        self.assertEqual(reports[0]['worker_exitcodes'], {0: exitcode})

    def test_actual_arena_worker_startup_failure_has_no_score_and_no_orphans(self):
        self.failed_arena(dead_worker, 1)

    def test_actual_arena_all_reported_errors_has_no_score_and_no_orphans(self):
        self.failed_arena(reported_error_worker, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
