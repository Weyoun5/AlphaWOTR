# -*- coding: utf-8 -*-
"""Graceful per-game failure, short-batch acceptance and the native-route hook
(package Y, loop_resilience, 2026-09-11).  Every fixture here is literal test
data; no engine game is played and no model is loaded.

  python src/test_loop_resilience_served.py
"""
import contextlib
import copy
import importlib.util
import json
import os
import pickle
import queue
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('WOTR_NATIVE', '0')
sys.path.insert(0, str(HERE))
import run_sp_served as runner   # noqa: E402
import rng_streams               # noqa: E402  (NumPy-backed RNG state before sys.modules patches)

BACKUP_RUNNER = (ROOT / 'native_backend/experiments/native_search_core/loop_resilience'
                 / 'backup_20260911_034757/src/run_sp_served.py')


class FakeState:
    """Picklable stand-in for GameState: a board to detach and turn-start flags."""

    def __init__(self):
        self.turn = 3
        self._declare_phase = False
        self._eye_allocation_phase = False
        self.board = object()          # not picklable on purpose: the capture must detach it


def _fake_ai(episode):
    return types.SimpleNamespace(PLAYER_PROFILES={}, FP_PROFILES=['ring'], SH_PROFILES=['military'],
                                 _run_single_episode=episode, mcts_search=None, _traced_execute=lambda *a: None,
                                 _SEARCH_FAILURE_DIAGNOSTIC=None, __file__='fake_ai.py')


def _worker_modules(ai):
    handles = {side: types.SimpleNamespace(checkpoint_id=side[0] * 64) for side in ('fp', 'sh')}
    return {'ai': ai, 'infer_server': types.SimpleNamespace(attach_client=lambda *a, **kw: handles),
            'torch': types.SimpleNamespace(set_num_threads=lambda _: None, device=lambda v: v),
            'numpy': types.SimpleNamespace(random=types.SimpleNamespace(seed=lambda _: None)),
            'ai_policies': types.SimpleNamespace(),
            'policy_keys_backend': types.SimpleNamespace(activate=lambda policy: {'backend': 'python'})}


def _run_worker(module, ai, temp, games, wid=7):
    sink = queue.Queue()
    with patch.dict(sys.modules, _worker_modules(ai)), \
            patch.dict(os.environ, WOTR_SP_FP_PROFILES='ring', WOTR_SP_SH_PROFILES='military',
                       WOTR_COMBAT_FORECAST_EVAL='0', WOTR_NATIVE_SEARCH='0'):
        module._worker(wid, object(), games, 4, os.path.join(temp, 'traces'), 123, 'base', sink)
    rows = []
    while not sink.empty():
        rows.append(sink.get_nowait())
    return rows


class GracefulWorkerTests(unittest.TestCase):
    def test_failed_episode_is_recorded_with_fixture_and_worker_continues(self):
        state = FakeState()

        def search(state_, side, fp, sh, num_simulations=100, **kwargs):
            raise RuntimeError('Empty sampled menu during required decision')

        calls = []

        def episode(fp, sh, config):
            calls.append(config)
            if len(calls) == 1:
                state._declare_phase = True          # turn-start flag as the frozen loop sets it
                try:
                    ai.mcts_search(state, 'sh', fp, sh, num_simulations=config['mcts_sims'])
                finally:
                    state._declare_phase = False     # the frozen loop's finally clears it before we see the error
            return {'game_stats': {'winner': 'fp', 'literal_fixture': True}}
        ai = _fake_ai(episode)
        ai.mcts_search = search
        ai._SEARCH_FAILURE_DIAGNOSTIC = {'schema': 'search_failure_diagnostic_v1',
                                         'error': 'RuntimeError: Empty sampled menu during required decision'}
        with tempfile.TemporaryDirectory(prefix='wotr_resilience_') as temp:
            rows = _run_worker(runner, ai, temp, 2)
            kinds = [row[0] for row in rows]
            self.assertEqual(kinds, ['p', 'p', 'e', 'p', 'g', 'p'])
            stages = [row[2]['stage'] for row in rows if row[0] == 'p']
            # failed work emits NO progress (the stall deadline is renewed by real work only)
            self.assertEqual(stages, ['worker_started', 'episode_started', 'episode_started', 'episode_completed'])
            error = rows[2][3]
            self.assertEqual((error['record_type'], error['status'], error['episode']), ('game_attempt', 'error', 0))
            self.assertIn('Empty sampled menu', error['error'])
            self.assertIn('RuntimeError', error['traceback'])
            failure = error['failure']
            self.assertEqual((failure['decision_index'], failure['side'], failure['turn']), (0, 'sh', 3))
            self.assertTrue(failure['phase_flags_at_search']['_declare_phase'])
            self.assertTrue(failure['state_captured'] and failure['search_failure_captured'])
            self.assertEqual(failure['search_failure_error'], ai._SEARCH_FAILURE_DIAGNOSTIC['error'])
            fixture = Path(failure['fixture_path'])
            self.assertEqual(fixture.parent, Path(temp) / 'failures')
            self.assertEqual(fixture.name, 'w7_g0_failure_d0.pkl')
            payload = pickle.loads(fixture.read_bytes())
            self.assertEqual(payload['schema'], runner.FAILURE_FIXTURE_SCHEMA)
            restored = pickle.loads(payload['state'])
            self.assertTrue(restored._declare_phase)          # captured AT the failing search
            self.assertIsNone(restored.board)
            self.assertEqual(payload['phase_flags_at_search']['_declare_phase'], True)
            self.assertEqual(payload['search_failure']['schema'], 'search_failure_diagnostic_v1')
            self.assertEqual(payload['attempt']['episode'], 0)
            self.assertEqual(payload['error']['type'], 'RuntimeError')
            self.assertFalse(state._declare_phase)            # the live object was left as the loop left it
            self.assertIsNotNone(state.board)
            game = rows[4][2]
            self.assertEqual((game['status'], game['episode'], game['worker_id']), ('completed', 1, 7))
            self.assertNotIn('search_route', game)
            self.assertNotIn('search_route', error)
            self.assertEqual(len(calls), 2)
            # the JSON ledger row the parent writes is serialisable
            json.dumps(runner._failure_ledger_row(error, 'label', 'phase'))

    def test_python_route_records_match_pre_edit_runner_byte_for_byte(self):
        """Bookkeeping parity: the same fake episodes through the backup (pre-edit) runner and
        the current runner emit identical progress and game rows (wall_s excluded)."""
        self.assertTrue(BACKUP_RUNNER.is_file(), 'pre-edit backup missing: %s' % BACKUP_RUNNER)
        spec = importlib.util.spec_from_file_location('legacy_run_sp_served', str(BACKUP_RUNNER))
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        legacy.SRC = str(HERE)

        def make_ai():
            def episode(fp, sh, config):
                return {'game_stats': {'winner': 'sh', 'turns': 9, 'config_seed': config.get('rng_episode_seed')}}
            ai = _fake_ai(episode)
            ai.mcts_search = lambda *a, **k: None
            return ai
        outputs = []
        with tempfile.TemporaryDirectory(prefix='wotr_parity_') as temp:
            for module in (legacy, runner):
                rows = _run_worker(module, make_ai(), temp, 2, wid=1)
                outputs.append(rows)
        for index, rows in enumerate(outputs):
            cleaned = []
            for row in rows:
                if row[0] == 'g':
                    record = {k: v for k, v in row[2].items() if k != 'wall_s'}
                    cleaned.append(('g', row[1], json.dumps(record, sort_keys=True)))
                else:
                    cleaned.append((row[0], row[1], json.dumps(row[2], sort_keys=True)))
            outputs[index] = cleaned
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual([r[0] for r in outputs[1]], ['p', 'p', 'g', 'p', 'p', 'g', 'p'])

    def test_capture_wrapper_is_pass_through_and_draws_no_randomness(self):
        sentinel = object()
        ai = types.SimpleNamespace(mcts_search=lambda state, side, *a, **k: sentinel)
        capture = runner.SearchFailureCapture(ai)
        original = capture.install()
        self.assertIs(ai.mcts_search.__wrapped__, original)
        before = random.getstate()
        self.assertIs(ai.mcts_search(FakeState(), 'fp', None, None, num_simulations=3), sentinel)
        self.assertEqual(random.getstate(), before)
        self.assertEqual((capture.decisions, capture.captured), (1, None))
        capture.reset()
        self.assertEqual((capture.decisions, capture.captured), (0, None))

    def test_max_game_failures_defaults_to_batch_size(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('WOTR_SP_MAX_GAME_FAILURES', None)
            self.assertEqual(runner.max_game_failures(25), 25)
        with patch.dict(os.environ, WOTR_SP_MAX_GAME_FAILURES='0'):
            self.assertEqual(runner.max_game_failures(25), 0)
        with patch.dict(os.environ, WOTR_SP_MAX_GAME_FAILURES='-1'):
            with self.assertRaises(ValueError):
                runner.max_game_failures(25)
        self.assertEqual(runner.failures_dir_for('/x/out/traces'), os.path.abspath('/x/out/failures'))


class NativeRouteHookTests(unittest.TestCase):
    def test_pin_check_requires_the_pinned_runtime_and_exact_hashes(self):
        with tempfile.TemporaryDirectory(prefix='wotr_pin_') as temp:
            runtime = Path(temp) / 'runtime'
            (runtime / 'src').mkdir(parents=True)
            (runtime / 'data').mkdir()
            (runtime / 'src' / 'ai.py').write_bytes(b'# literal engine\n')
            (runtime / 'data' / 'wotr_map.txt').write_bytes(b'map\n')
            reference = {'runtime': str(runtime),
                         'source_sha256': {'ai.py': runner._file_sha256(runtime / 'src' / 'ai.py')},
                         'data_sha256': {'wotr_map.txt': runner._file_sha256(runtime / 'data' / 'wotr_map.txt')}}
            pin = Path(temp) / 'reference.json'
            pin.write_text(json.dumps(reference), encoding='utf-8')
            ai = types.SimpleNamespace(__file__=str(runtime / 'src' / 'ai.py'))
            result = runner.verify_native_reference(ai, str(pin))
            self.assertEqual((result['pinned_files'], result['reference_runtime']), (1, str(runtime)))
            other = types.SimpleNamespace(__file__=str(HERE / 'ai.py'))
            with self.assertRaisesRegex(RuntimeError, 'not the pinned runtime'):
                runner.verify_native_reference(other, str(pin))
            (runtime / 'src' / 'ai.py').write_bytes(b'# drifted engine\n')
            with self.assertRaisesRegex(RuntimeError, 'source drift'):
                runner.verify_native_reference(ai, str(pin))

    def test_route_wrapper_forwards_kwargs_and_restores_the_original(self):
        calls, drops = [], []

        def native_mcts_search(ai_module, state, side, fp, sh, **kwargs):
            calls.append((ai_module, state, side, fp, sh, kwargs))
            return ('action', {'visits': 1})
        orchestrator = types.SimpleNamespace(native_mcts_search=native_mcts_search,
                                             drop_cached_child=lambda ai_module: drops.append(ai_module))
        original = lambda *a, **k: 'python'    # noqa: E731
        ai = types.SimpleNamespace(mcts_search=original)
        state = FakeState()
        with runner.native_search_route(ai, orchestrator) as wrapper:
            self.assertIs(ai.mcts_search, wrapper)
            self.assertEqual(wrapper.__wotr_route__, 'native')
            result = ai.mcts_search(state, 'fp', 'FP', 'SH', num_simulations=8, c_puct=1.5,
                                    temperature=0.5, add_noise=False, hot_regions={'x'}, bonus_sims=0)
        self.assertEqual(result, ('action', {'visits': 1}))
        self.assertIs(ai.mcts_search, original)
        self.assertEqual(len(drops), 2)
        module, state_, side, fp, sh, kwargs = calls[0]
        self.assertIs(module, ai)
        self.assertIs(state_, state)
        self.assertEqual((side, fp, sh), ('fp', 'FP', 'SH'))
        self.assertEqual(kwargs, {'num_simulations': 8, 'c_puct': 1.5, 'temperature': 0.5, 'add_noise': False,
                                  'hot_regions': {'x'}, 'bonus_sims': 0, 'determinize': True})

    def test_native_setup_failure_is_a_worker_init_failure_and_stamps_records(self):
        ai = _fake_ai(lambda fp, sh, config: {'game_stats': {'winner': 'fp'}})
        ai.mcts_search = lambda *a, **k: None
        sink = queue.Queue()
        with tempfile.TemporaryDirectory(prefix='wotr_native_init_') as temp, \
                patch.dict(sys.modules, _worker_modules(ai)), \
                patch.dict(os.environ, WOTR_SP_FP_PROFILES='ring', WOTR_SP_SH_PROFILES='military',
                           WOTR_COMBAT_FORECAST_EVAL='0', WOTR_NATIVE_SEARCH='1',
                           WOTR_NATIVE_SEARCH_REFERENCE=os.path.join(temp, 'missing_reference.json')):
            runner._worker(2, object(), 1, 4, os.path.join(temp, 'traces'), 123, 'base', sink)
        rows = []
        while not sink.empty():
            rows.append(sink.get_nowait())
        errors = [row for row in rows if row[0] == 'e']
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0][3]['stage'], 'worker_init')
        self.assertEqual(errors[0][3]['search_route'], 'native')
        self.assertFalse(any(row[0] == 'g' for row in rows))

    def test_native_stamp_is_applied_when_the_setup_succeeds(self):
        drops = []
        stamp = {'search_route': 'native', 'controller_binary_sha256': 'c' * 64, 'orchestrator_sha256': 'o' * 64,
                 'reference_sha256': 'r' * 64, 'capabilities': {'tree_reuse': True}}
        orchestrator = types.SimpleNamespace(
            native_mcts_search=lambda ai_module, state, side, fp, sh, **kw: ('a', {}),
            drop_cached_child=lambda ai_module: drops.append(1))
        calls = []

        def episode(fp, sh, config):
            calls.append(config)
            if len(calls) == 1:
                raise NotImplementedError('Unsupported: combat request without observer events')
            return {'game_stats': {'winner': 'sh'}}
        ai = _fake_ai(episode)
        ai.mcts_search = lambda *a, **k: None
        sink = queue.Queue()
        with tempfile.TemporaryDirectory(prefix='wotr_native_stamp_') as temp, \
                patch.dict(sys.modules, _worker_modules(ai)), \
                patch.object(runner, 'native_search_setup', lambda ai_module: (orchestrator, dict(stamp))), \
                patch.dict(os.environ, WOTR_SP_FP_PROFILES='ring', WOTR_SP_SH_PROFILES='military',
                           WOTR_COMBAT_FORECAST_EVAL='0', WOTR_NATIVE_SEARCH='1'):
            runner._worker(0, object(), 2, 4, os.path.join(temp, 'traces'), 123, 'base', sink)
            rows = []
            while not sink.empty():
                rows.append(sink.get_nowait())
            errors = [row for row in rows if row[0] == 'e']
            games = [row for row in rows if row[0] == 'g']
            self.assertEqual((len(errors), len(games)), (1, 1))
            error = errors[0][3]
            self.assertEqual((error['search_route'], error['native_refusal'], error['episode']), ('native', True, 0))
            self.assertEqual(error['native_search'], stamp)
            self.assertTrue(error['failure']['fixture_path'].endswith('w0_g0_failure_dna.pkl'))
            game = games[0][2]
            self.assertEqual(game['search_route'], 'native')
            self.assertEqual(game['search_acceleration']['native_search'], stamp)
            self.assertEqual(game['episode'], 1)
        # drop on route entry, after the failed episode, and on route exit
        self.assertEqual(len(drops), 3)

    def test_loop_and_runner_agree_on_the_native_source_identity(self):
        import run_td_v33loop as driver
        if not os.path.isfile(driver.ROOT + '/native_backend/reference.json'):
            self.skipTest('no native reference on this box')
        with patch.dict(os.environ, {}, clear=False):
            for key in ('WOTR_NATIVE_SEARCH_BOX', 'WOTR_NATIVE_SEARCH_REFERENCE', 'WOTR_NATIVE_SEARCH_CORE'):
                os.environ.pop(key, None)
            self.assertEqual(driver._native_search_source_identity(), runner.native_search_source_identity())
        for key in ('WOTR_LOOP_MIN_BATCH_GAMES', 'WOTR_SP_MAX_GAME_FAILURES', 'WOTR_NATIVE_SEARCH',
                    'WOTR_NATIVE_SEARCH_REFERENCE'):
            self.assertIn(key, driver.JOURNAL_ENVIRONMENT_EXCLUDED)


def _served_output(temp, planned=2, completed=1, errors=1, worker_failures=(), rc_hint=0):
    """A literal served output directory: launch/phase/attempts/games + traces/failures."""
    output = Path(temp) / 'incoming_served_b007'
    (output / 'traces').mkdir(parents=True)
    if errors:
        (output / 'failures').mkdir()      # the runner creates it only when a fixture is written
    phase_id = 'f' * 32
    attempts, games = [], []
    episode = 0
    for _ in range(errors):
        trace = str(output / 'traces' / ('w0_g%d.trace.gz' % episode))
        (output / 'traces' / ('w0_g%d.trace.gz' % episode)).write_bytes(b'failed trace bytes')
        fixture = output / 'failures' / ('w0_g%d_failure_d3.pkl' % episode)
        fixture.write_bytes(pickle.dumps({'schema': runner.FAILURE_FIXTURE_SCHEMA, 'decision': 3}))
        attempts.append({'record_type': 'game_attempt', 'worker_id': 0, 'episode': episode, 'status': 'error',
                         'phase_id': phase_id, 'batch': 'served-base selfplay', 'trace_path': trace,
                         'error': "RuntimeError('Empty sampled menu during required decision')",
                         'error_type': 'RuntimeError', 'traceback': 'Traceback ... RuntimeError',
                         'failure': {'fixture_path': str(fixture), 'fixture_sha256': 'a' * 64,
                                     'decision_index': 3, 'side': 'sh', 'turn': 5,
                                     'phase_flags_at_search': {'_declare_phase': False}}})
        episode += 1
    for _ in range(completed):
        name = 'w0_g%d_sh_military.trace.gz' % episode
        (output / 'traces' / name).write_bytes(b'completed trace bytes')
        row = {'record_type': 'game_attempt', 'worker_id': 0, 'episode': episode, 'status': 'completed',
               'phase_id': phase_id, 'batch': 'served-base selfplay', 'trace_path': str(output / 'traces' / name),
               'winner': 'sh'}
        attempts.append(row)
        games.append({k: v for k, v in row.items() if k not in ('record_type', 'phase_id', 'batch')})
        episode += 1
    with (output / 'attempts.jsonl').open('w', encoding='utf-8') as handle:
        for row in attempts:
            handle.write(json.dumps(row) + '\n')
    with (output / 'games.jsonl').open('w', encoding='utf-8') as handle:
        for row in games:
            handle.write(json.dumps(row) + '\n')
    if errors:
        with (output / 'failures.jsonl').open('w', encoding='utf-8') as handle:
            for row in attempts:
                if row['status'] == 'error':
                    handle.write(json.dumps({'record_type': 'game_failure', 'episode': row['episode']}) + '\n')
    observed = len(attempts)
    phase = {'phase_id': phase_id, 'batch': 'served-base selfplay', 'finished': True, 'planned': planned,
             'observed': observed, 'unreported': planned - observed, 'completed': completed,
             'error_attempts': errors, 'coverage_complete': observed == planned,
             'worker_failures': list(worker_failures), 'unreported_episodes': [], 'parent_error': None,
             'cleanup': {'alive': []}, 'game_failures': errors, 'max_game_failures': planned}
    (output / 'phase.json').write_text(json.dumps(phase), encoding='utf-8')
    (output / 'launch.json').write_text(json.dumps({'games': planned, 'workers': 1}), encoding='utf-8')
    return output


class ShortBatchAcceptanceTests(unittest.TestCase):
    def setUp(self):
        import run_td_v33loop as driver
        self.driver = driver
        self.temp = tempfile.TemporaryDirectory(prefix='wotr_loop_short_')
        self.addCleanup(self.temp.cleanup)
        self.loop = os.path.join(self.temp.name, 'loop')
        os.makedirs(self.loop)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(driver, 'LOOP', self.loop))
        self.stack.enter_context(patch.object(driver, 'FEED_JSONL', os.path.join(self.loop, 'all_games.jsonl')))
        self.stack.enter_context(patch.dict(os.environ, {}, clear=False))
        os.environ.pop('WOTR_LOOP_MIN_BATCH_GAMES', None)
        os.environ.pop('WOTR_PRE_READY_RECOVERY', None)
        for key in ('planned', 'completed', 'game_failures', 'batches_short'):
            driver._SHORTFALL_TOTAL[key] = 0

    def test_short_batch_with_one_failure_is_archived_and_accepted(self):
        output = _served_output(self.temp.name)
        bdir = self.driver._archive_served_batch(7, str(output), 0, False, None)
        self.assertEqual(os.path.basename(bdir), 'batch_007')
        root_traces = sorted(p.name for p in Path(bdir).glob('*.trace.gz'))
        self.assertEqual(root_traces, ['w0_g1_sh_military.trace.gz'])
        failures = Path(bdir) / 'failures'
        self.assertTrue((failures / 'w0_g0.trace.gz').is_file())
        self.assertTrue((failures / 'w0_g0_failure_d3.pkl').is_file())
        self.assertTrue((failures / 'failures.jsonl').is_file())
        self.assertTrue((output / 'failures' / 'w0_g0_failure_d3.pkl').is_file())   # originals preserved
        rows = [json.loads(line) for line in (Path(bdir) / 'attempts.jsonl').read_text().splitlines()]
        error = next(row for row in rows if row['status'] == 'error')
        self.assertEqual(Path(error['trace_path']).parent, failures)
        ledger = [json.loads(line) for line in
                  Path(self.loop, self.driver.FAILURES_LEDGER).read_text().splitlines()]
        self.assertEqual([row['record_type'] for row in ledger], ['game_failure', 'batch_shortfall'])
        self.assertEqual((ledger[0]['batch'], ledger[0]['episode'], ledger[0]['decision_index']), (7, 0, 3))
        self.assertEqual(ledger[0]['fixture_path'], str(failures / 'w0_g0_failure_d3.pkl'))
        self.assertEqual((ledger[1]['planned'], ledger[1]['completed'], ledger[1]['game_failures']), (2, 1, 1))
        log = Path(self.loop, 'loop_log.md').read_text(encoding='utf-8')
        self.assertIn('GAME FAILURE batch 007 worker 0 episode 0', log)
        self.assertIn('BATCH 007 SHORT: 1/2 games completed', log)
        feed = Path(self.loop, 'all_games.jsonl').read_text().splitlines()
        self.assertEqual(len(feed), 1)

    def test_batch_below_minimum_completed_games_is_refused_after_archiving(self):
        output = _served_output(self.temp.name, planned=2, completed=1, errors=1)
        with patch.dict(os.environ, WOTR_LOOP_MIN_BATCH_GAMES='2'):
            with self.assertRaisesRegex(SystemExit, 'WOTR_LOOP_MIN_BATCH_GAMES=2'):
                self.driver._archive_served_batch(7, str(output), 0, False, None)
        self.assertTrue(Path(self.loop, 'batch_007', 'failures', 'w0_g0_failure_d3.pkl').is_file())
        self.assertIn('BATCH 007 REFUSED', Path(self.loop, 'loop_log.md').read_text(encoding='utf-8'))

    def test_zero_completed_games_is_refused_by_default(self):
        output = _served_output(self.temp.name, planned=1, completed=0, errors=1)
        with self.assertRaisesRegex(SystemExit, 'WOTR_LOOP_MIN_BATCH_GAMES=1'):
            self.driver._archive_served_batch(7, str(output), 0, False, None)

    def test_worker_level_failure_on_exit_zero_is_still_refused(self):
        output = _served_output(self.temp.name, worker_failures=[{'stage': 'reporting_gap', 'worker_id': 0}])
        with self.assertRaisesRegex(ValueError, 'incomplete/malformed'):
            self.driver._archive_served_batch(7, str(output), 0, False, None)
        errors = json.loads(Path(self.loop, 'batch_007', 'served_archive_errors.json').read_text())
        self.assertTrue(any('no worker failure' in e['error'] for e in errors['errors']))

    def test_full_batch_without_failures_writes_no_ledger(self):
        output = _served_output(self.temp.name, planned=1, completed=1, errors=0)
        bdir = self.driver._archive_served_batch(7, str(output), 0, False, None)
        self.assertFalse(Path(self.loop, self.driver.FAILURES_LEDGER).exists())
        self.assertFalse((Path(bdir) / 'failures').exists())
        self.assertEqual(self.driver.min_batch_games(), 1)


if __name__ == '__main__':
    unittest.main()
