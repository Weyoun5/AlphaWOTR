"""Real spawned CPU processes; all reported results are lifecycle fixtures."""
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
import queue
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ.update(CUDA_VISIBLE_DEVICES='-1', WOTR_NATIVE='0')
import run_sp_served as runner
# Load NumPy-backed RNG state before tests temporarily replace sys.modules.
import rng_streams



def _fixture_server_entry(conns, paths, arch):
    output = Path(os.environ['WOTR_LIFECYCLE_FIXTURE_DIR'])
    mode = os.environ['WOTR_LIFECYCLE_FIXTURE_MODE']
    (output / 'server_started.json').write_text(json.dumps({'pid': os.getpid()}))
    if mode == 'server_start_failure':
        raise RuntimeError('literal server startup failure')
    if mode == 'ignore_stop':
        while True:
            time.sleep(.05)
    # Exercise the real stop/EOF protocol without models, inference or GPU.
    import infer_server
    infer_server.server_main(conns, {}, 'base', device='cpu')
    (output / 'server_graceful.json').write_text(json.dumps({'pid': os.getpid(), 'returned': True}))


def _fixture_worker(wid, conn, count, sims, trace_dir, seed0, arch, out_q, worker_stride=1):
    mode = os.environ['WOTR_LIFECYCLE_FIXTURE_MODE']
    if mode == 'crash' and wid == 0:
        os._exit(7)
    if mode == 'parent_error' and wid == 1:
        while True:
            time.sleep(.05)
    if mode == 'worker_start_failure':
        while True:
            time.sleep(.05)
    if mode == 'init_error':
        out_q.put(('e', wid, 'literal model initialization error', {'stage': 'worker_init'}))
        return
    for episode in range(count):
        record = {'worker_id': wid, 'episode': episode, 'lifecycle_fixture': True}
        if mode == 'episode_error' and episode == 0:
            record.update(record_type='game_attempt', error='literal episode error', stage='episode')
            out_q.put(('e', wid, 'literal episode error', record))
        else:
            out_q.put(('g', wid, record))
            if mode == 'duplicate':
                out_q.put(('g', wid, record))
    conn.close()


def _emergency_cleanup(context):
    # A regression must fail the test without leaving its fixture children.
    for process in reversed(context.processes):
        try:
            if process.pid is not None and process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
        except ValueError:
            pass  # The passing test already closed this process handle.


class Context:
    def __init__(self, fail_worker_start=False):
        self.real = mp.get_context('spawn')
        self.processes = []
        self.fail_worker_start = fail_worker_start
    def Queue(self):
        return self.real.Queue()
    def Pipe(self):
        return self.real.Pipe()
    def Process(self, **kwargs):
        process = self.real.Process(**kwargs)
        self.processes.append(process)
        if self.fail_worker_start and len(self.processes) == 3:
            def refuse():
                raise OSError('literal second worker spawn failed')
            process.start = refuse
        return process


class ServedLifecycleTests(unittest.TestCase):
    def run_case(self, mode, games, workers, expected_exception=None, env=None):
        temporary = tempfile.TemporaryDirectory(prefix='wotr_served_' + mode + '_')
        self.addCleanup(temporary.cleanup)
        output = Path(temporary.name)
        for side in ('fp', 'sh'):
            (output / (side + '.pt')).write_bytes(b'lifecycle-only fake model; never loaded')
        args = types.SimpleNamespace(fp_net=str(output / 'fp.pt'), sh_net=str(output / 'sh.pt'),
                                     out_dir=str(output / 'operator'), games=games, workers=workers,
                                     sims=4, seed0=123, shard_id=0, arch='base')
        ctx = Context(mode == 'worker_start_failure')
        self.addCleanup(_emergency_cleanup, ctx)
        def append(*args):
            if mode == 'parent_error':
                raise OSError('literal parent log write failed')
        fake_ai = types.SimpleNamespace(append_game_log=append, write_td_window_report=lambda *a: None)
        actual_error = None
        with patch.dict(os.environ, dict(env or {}), WOTR_LIFECYCLE_FIXTURE_DIR=str(output),
                        WOTR_LIFECYCLE_FIXTURE_MODE=mode), \
                patch.object(runner, '_server_entry', _fixture_server_entry), \
                patch.object(runner, '_worker', _fixture_worker), \
                patch.object(runner, '_CLEANUP_JOIN_SECONDS', 2.0), \
                patch.object(runner, '_CLEANUP_KILL_SECONDS', 1.0), \
                patch.object(mp, 'get_context', return_value=ctx), \
                patch.dict(sys.modules, {'ai': fake_ai}):
            try:
                runner.run(args)
            except BaseException as exc:
                actual_error = exc
        if expected_exception is None:
            self.assertIsNone(actual_error)
        else:
            self.assertIsInstance(actual_error, expected_exception)
        phase = json.loads((Path(args.out_dir) / 'phase.json').read_text())
        states = []
        for process in ctx.processes:
            if process.pid is not None:
                self.assertFalse(process.is_alive(), 'orphan PID %s' % process.pid)
                self.assertEqual(wait([process.sentinel], timeout=0), [process.sentinel])
                states.append({'pid': process.pid, 'exitcode': process.exitcode, 'sentinel_signaled': True})
                process.close()
        self.assertFalse(list(output.rglob('*.trace.gz')))
        self.assertEqual(phase['completed'] + phase['error_attempts'] + phase['unreported'], games)
        self.assertEqual(phase['cleanup']['alive'], [])
        return phase, output

    def test_normal_completion_stops_real_cpu_server_gracefully(self):
        phase, output = self.run_case('normal', 3, 2)
        self.assertEqual((phase['completed'], phase['error_attempts'], phase['unreported']), (3, 0, 0))
        self.assertTrue((output / 'server_graceful.json').is_file())
        self.assertTrue(phase['cleanup']['stop_sent'])
        self.assertFalse(phase['cleanup']['server_terminated'])
        self.assertEqual(phase['cleanup']['worker_terminated'], [])

    def test_episode_failure_has_no_invented_unreported_result(self):
        # Graceful per-game failure (USER 2026-09-11): the failed episode is an error
        # attempt, the worker continues, the batch completes with exit 0 and the
        # failure is ledgered; nothing is invented for the failed game.
        phase, output = self.run_case('episode_error', 2, 1)
        self.assertEqual((phase['completed'], phase['error_attempts'], phase['unreported']), (1, 1, 0))
        self.assertEqual((phase['game_failures'], phase['max_game_failures']), (1, 2))
        self.assertTrue((output / 'server_graceful.json').is_file())
        ledger = (Path(phase['failures_ledger'])).read_text(encoding='utf-8').splitlines()
        self.assertEqual(len(ledger), 1)
        row = json.loads(ledger[0])
        self.assertEqual((row['record_type'], row['worker_id'], row['episode']), ('game_failure', 0, 0))
        self.assertIn('literal episode error', row['error'])

    def test_episode_failure_over_budget_stops_batch_without_invented_result(self):
        # WOTR_SP_MAX_GAME_FAILURES=0 restores the previous stop-on-first-error policy.
        phase, output = self.run_case('episode_error', 2, 1, RuntimeError,
                                      env={'WOTR_SP_MAX_GAME_FAILURES': '0'})
        self.assertEqual((phase['completed'], phase['error_attempts'], phase['unreported']), (0, 1, 1))
        self.assertIn('WOTR_SP_MAX_GAME_FAILURES=0', phase['parent_error']['error'])
        self.assertTrue((output / 'server_graceful.json').is_file())

    def test_initialization_failure_leaves_all_games_unreported(self):
        phase, _ = self.run_case('init_error', 2, 1, RuntimeError)
        self.assertEqual((phase['completed'], phase['error_attempts'], phase['unreported']), (0, 0, 2))
        self.assertEqual(phase['unreported_episodes'], [{'worker_id': 0, 'episode': 0}, {'worker_id': 0, 'episode': 1}])

    def test_crashed_worker_does_not_hide_its_missing_games(self):
        phase, _ = self.run_case('crash', 3, 2, RuntimeError)
        self.assertEqual((phase['completed'], phase['error_attempts'], phase['unreported']), (1, 0, 2))
        self.assertEqual(phase['cleanup']['worker_exitcodes']['0'], 7)

    def test_parent_exception_terminates_waiting_worker_and_stops_server(self):
        phase, output = self.run_case('parent_error', 2, 2, OSError)
        self.assertIn('literal parent log write failed', phase['parent_error']['error'])
        self.assertEqual((phase['completed'], phase['error_attempts'], phase['unreported']), (1, 0, 1))
        self.assertEqual(phase['cleanup']['worker_terminated'], [1])
        self.assertTrue((output / 'server_graceful.json').is_file())

    def test_partial_spawn_failure_cleans_previously_started_children(self):
        phase, _ = self.run_case('worker_start_failure', 2, 2, OSError)
        self.assertEqual((phase['completed'], phase['error_attempts'], phase['unreported']), (0, 0, 2))
        self.assertEqual(phase['cleanup']['worker_terminated'], [0])

    def test_ignored_stop_uses_bounded_fallback_and_fails_clean_exit(self):
        phase, output = self.run_case('ignore_stop', 1, 1, RuntimeError)
        self.assertTrue(phase['cleanup']['stop_sent'])
        self.assertTrue(phase['cleanup']['server_terminated'])
        self.assertFalse((output / 'server_graceful.json').exists())

    def test_server_startup_failure_creates_no_workers(self):
        phase, _ = self.run_case('server_start_failure', 2, 2, RuntimeError)
        self.assertEqual(phase['cleanup']['worker_exitcodes'], {})
        self.assertEqual(phase['unreported'], 2)

    def test_duplicate_after_last_expected_result_is_not_hidden(self):
        phase, output = self.run_case('duplicate', 1, 1, ValueError)
        self.assertEqual(phase['completed'], 1)
        self.assertIn('Duplicate', phase['parent_error']['error'])
        self.assertTrue((output / 'server_graceful.json').is_file())

    def test_actual_worker_episode_failure_preserves_attempt_identity(self):
        def fail(*args):
            raise RuntimeError('literal engine failure')
        ai = types.SimpleNamespace(PLAYER_PROFILES={}, FP_PROFILES=['ring'], SH_PROFILES=['military'],
                                   _run_single_episode=fail,mcts_search=fail,_traced_execute=fail)
        handles = {side: types.SimpleNamespace(checkpoint_id=side[0] * 64) for side in ('fp', 'sh')}
        modules = {'ai': ai, 'infer_server': types.SimpleNamespace(attach_client=lambda *a, **kw: handles),
                   'torch': types.SimpleNamespace(set_num_threads=lambda _: None, device=lambda v: v),
                   'numpy': types.SimpleNamespace(random=types.SimpleNamespace(seed=lambda _: None))}
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, modules), \
                patch.dict(os.environ, WOTR_SP_FP_PROFILES='ring', WOTR_SP_SH_PROFILES='military'):
            sink = queue.Queue()
            runner._worker(9, object(), 1, 4, temp, 123, 'base', sink)
        progress = [sink.get_nowait(), sink.get_nowait()]
        self.assertEqual([row[:2] for row in progress], [('p',9),('p',9)])
        self.assertEqual([row[2]['stage'] for row in progress], ['worker_started','episode_started'])
        self.assertEqual([row[2]['sequence'] for row in progress], [1,2])
        event = sink.get_nowait()
        self.assertEqual(event[:2], ('e', 9))
        self.assertEqual(event[3]['record_type'], 'game_attempt')
        self.assertEqual(event[3]['episode'], 0)
        self.assertEqual(event[3]['profiles'], {'fp': 'ring', 'sh': 'military'})
        self.assertIn('literal engine failure', event[3]['error'])
        self.assertTrue(sink.empty(), 'Failed work must not emit completed progress')



if __name__ == '__main__':
    unittest.main(verbosity=2)
