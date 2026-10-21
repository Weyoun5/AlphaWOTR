# -*- coding: utf-8 -*-
"""Per-search budget guard (package Y2, loop_resilience, 2026-09-11): the orchestrator's
SearchBudgetGuard (checked between native controller advances), the runner's SearchBudgetWatchdog
thread (both routes), the fixture / attempt-row / ledger plumbing and the loop's journal
exclusions.  Every controller here is a literal fake that never converges; no engine game is
played, no model is loaded, the native DLL is never touched.

  python src/test_search_budget_guard.py
"""
import functools
import importlib.util
import json
import os
import pickle
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('WOTR_NATIVE', '0')
sys.path.insert(0, str(HERE))
import run_sp_served as runner                    # noqa: E402
from test_loop_resilience_served import FakeState, _fake_ai, _run_worker   # noqa: E402

ORCHESTRATOR = ROOT / 'native_backend/experiments/native_search_core/inference/orchestrator.py'
BUDGET_ENV = {'WOTR_SEARCH_MAX_SECONDS': None, 'WOTR_SEARCH_MAX_PRIVATE_GB': None, 'WOTR_SEARCH_MAX_ADVANCES': None,
              'WOTR_SEARCH_WATCHDOG_INTERRUPT': None, 'WOTR_SEARCH_BUDGET_EXPORT': None}


def load_orchestrator():
    spec = importlib.util.spec_from_file_location('wotr_native_search_orchestrator_under_test', str(ORCHESTRATOR))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clean_budget_env(**values):
    """patch.dict that REMOVES the budget variables and sets only `values`."""
    env = dict(os.environ)
    for name in BUDGET_ENV:
        env.pop(name, None)
    env.update({k: v for k, v in values.items() if v is not None})
    return patch.dict(os.environ, env, clear=True)


class FakeController:
    """A controller that never completes: advance() always reports a pending request."""

    def __init__(self, nodes=200, events=300, counters=None, fail_export=False):
        self.calls = 0
        self._counters = counters if counters is not None else {'search': {'leaf_batches': 3}, 'kernel': {}}
        self._export = {'nodes': [{'id': i, 'visit_count': i} for i in range(nodes)],
                        'events': [('SELECT', i, 0, i) for i in range(events)], 'extra': 1}
        self.fail_export = fail_export
        self.export_calls = 0

    def advance(self):
        self.calls += 1
        return 0

    def counters(self, ai):
        return self._counters

    def export(self, ai, with_states=False):
        self.export_calls += 1
        if self.fail_export:
            raise RuntimeError('export refused after a detach')
        return self._export


class Clock:
    def __init__(self, step):
        self.t, self.step = 0.0, step

    def __call__(self):
        self.t += self.step
        return self.t


class Memory:
    def __init__(self, step_gb):
        self.value, self.step = 0, int(step_gb * 1024 ** 3)

    def __call__(self):
        value = self.value
        self.value += self.step
        return value


class OrchestratorGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.orch = load_orchestrator()

    def test_limits_from_env_defaults_factor_and_disable(self):
        orch = self.orch
        with clean_budget_env():
            limits = orch.search_budget_limits(1000)
            self.assertEqual((limits['max_seconds'], limits['max_private_gb'], limits['max_advances_per_sim']),
                             (300.0, 4.0, 20.0))
            self.assertEqual((limits['max_advances'], limits['export']), (20000, 'auto'))
        with clean_budget_env(WOTR_SEARCH_MAX_ADVANCES='0.1', WOTR_SEARCH_MAX_SECONDS='0',
                              WOTR_SEARCH_MAX_PRIVATE_GB='0', WOTR_SEARCH_BUDGET_EXPORT='never'):
            limits = orch.search_budget_limits(8)
            self.assertEqual((limits['max_seconds'], limits['max_private_gb'], limits['max_advances']), (0.0, 0.0, 1))
            self.assertEqual(limits['export'], 'never')
        with clean_budget_env(WOTR_SEARCH_MAX_ADVANCES='0'):
            self.assertEqual(orch.search_budget_limits(8)['max_advances'], 0)
        for bad in ({'WOTR_SEARCH_MAX_SECONDS': '-1'}, {'WOTR_SEARCH_MAX_PRIVATE_GB': 'four'},
                    {'WOTR_SEARCH_MAX_ADVANCES': 'nan'}, {'WOTR_SEARCH_BUDGET_EXPORT': 'sometimes'}):
            with clean_budget_env(**bad):
                with self.assertRaises(ValueError):
                    orch.search_budget_limits(8)

    def _guard(self, limits, sims=1, **kw):
        base = {'max_seconds': 0, 'max_private_gb': 0, 'max_advances': 0, 'export': 'auto'}
        base.update(limits)
        return self.orch.SearchBudgetGuard(object(), base, sims, **kw)

    def test_never_converging_controller_breaches_the_advance_limit_with_attributes(self):
        orch = self.orch
        ctl = FakeController()
        root = FakeState()
        kwargs = {'side': 'sh', 'num_simulations': 1, 'c_puct': 4.0, 'temperature': 0.6, 'add_noise': True,
                  'hot_regions': {'gondor'}, 'bonus_sims': 0, 'determinize': True}
        guard = self._guard({'max_advances': 5}, root_state=root, search_kwargs=kwargs,
                            clock=Clock(0.001), memory=lambda: 0)
        with self.assertRaises(RuntimeError) as caught:          # a RuntimeError, as the runner expects
            for _ in range(100):
                guard.before_advance(ctl)
                ctl.advance()
        error = caught.exception
        self.assertIsInstance(error, orch.SearchBudgetExceeded)
        self.assertEqual(str(error), 'native search budget exceeded: advances=6 (limit 5) sims=1 advances=6')
        self.assertEqual(ctl.calls, 5)                            # the 6th advance never ran
        self.assertIs(error.root_state, root)
        self.assertEqual(error.search_kwargs, kwargs)
        self.assertEqual(error.counters, {'search': {'leaf_batches': 3}, 'kernel': {}})
        tail = error.export_tail
        self.assertEqual((tail['nodes_total'], tail['events_total'], len(tail['nodes_tail']), len(tail['events_tail'])),
                         (200, 300, 64, 64))
        self.assertEqual(tail['nodes_tail'][-1]['id'], 199)
        self.assertEqual(tail['root']['id'], 0)
        self.assertEqual(tail['other_keys'], ['extra'])
        self.assertEqual((error.which, error.value, error.limit, error.sims, error.advances), ('advances', 6, 5, 1, 6))
        summary = error.search_budget
        json.dumps(summary)                                       # attempt-row safe
        self.assertEqual(summary['source'], 'orchestrator')
        self.assertEqual(summary['search_kwargs']['hot_regions'], repr({'gondor'}))
        self.assertFalse(runner.is_native_refusal(error))

    def test_wall_and_memory_breaches_message_and_export_policy(self):
        orch = self.orch
        ctl = FakeController()
        guard = self._guard({'max_seconds': 250.0}, sims=8, clock=Clock(100.0), memory=lambda: 0)
        with self.assertRaises(orch.SearchBudgetExceeded) as caught:
            for _ in range(10):
                guard.before_advance(ctl)
        self.assertEqual(str(caught.exception),
                         'native search budget exceeded: wall_seconds=300.0 (limit 250.0) sims=8 advances=3')
        self.assertIsNotNone(caught.exception.export_tail.get('nodes_total'))   # small tree: exported
        # memory: +1 GB per sample, one sample per advance (clock step 1.0 >= cadence)
        ctl = FakeController()
        guard = self._guard({'max_private_gb': 2.5}, sims=8, clock=Clock(1.0), memory=Memory(1.0))
        with self.assertRaises(orch.SearchBudgetExceeded) as caught:
            for _ in range(10):
                guard.before_advance(ctl)
        error = caught.exception
        self.assertEqual(str(error), 'native search budget exceeded: private_gb=3.0 (limit 2.5) sims=8 advances=3')
        self.assertEqual(error.private_growth_bytes, 3 * 1024 ** 3)
        self.assertIn('skipped', error.export_tail)               # auto: never copy a runaway tree
        self.assertEqual(ctl.export_calls, 0)
        self.assertEqual(error.counters, {'search': {'leaf_batches': 3}, 'kernel': {}})
        # 'always' exports even on a memory breach; 'never' never does; export errors are recorded
        guard = self._guard({'max_private_gb': 2.5, 'export': 'always'}, clock=Clock(1.0), memory=Memory(1.0))
        with self.assertRaises(orch.SearchBudgetExceeded) as caught:
            for _ in range(10):
                guard.before_advance(ctl)
        self.assertEqual(caught.exception.export_tail['nodes_total'], 200)
        ctl = FakeController(fail_export=True)
        guard = self._guard({'max_advances': 1, 'export': 'always'}, clock=Clock(0.001), memory=lambda: 0)
        with self.assertRaises(orch.SearchBudgetExceeded) as caught:
            for _ in range(10):
                guard.before_advance(ctl)
        self.assertIsNone(caught.exception.export_tail)
        self.assertTrue(any(e.startswith('export:') for e in caught.exception.capture_errors))
        guard = self._guard({'max_advances': 1, 'export': 'never'}, clock=Clock(0.001), memory=lambda: 0)
        with self.assertRaises(orch.SearchBudgetExceeded) as caught:
            for _ in range(10):
                guard.before_advance(ctl)
        self.assertIn('disabled', caught.exception.export_tail['skipped'])

    def test_disabled_limits_only_count(self):
        ctl = FakeController()
        calls = []
        guard = self._guard({}, clock=Clock(1000.0), memory=lambda: calls.append(1) or 0)
        self.assertFalse(guard.enabled)
        for _ in range(10000):
            guard.before_advance(ctl)
        self.assertEqual(guard.advances, 10000)
        self.assertEqual(calls, [])                               # psutil never consulted when the check is off
        self.assertEqual(guard.summary()['private_growth_gb'], None)

    def test_export_tail_is_bounded_and_tolerant(self):
        orch = self.orch
        self.assertEqual(orch.export_tail({'nodes': [], 'events': []})['root'], None)
        self.assertIn('error', orch.export_tail(None))
        tail = orch.export_tail({'nodes': list(range(10)), 'events': list(range(3))}, tail=4)
        self.assertEqual((tail['nodes_tail'], tail['events_tail']), ([6, 7, 8, 9], [0, 1, 2]))

    def test_guard_is_wired_before_every_advance_in_the_orchestrator_source(self):
        source = ORCHESTRATOR.read_text(encoding='utf-8')
        loop = source[source.index('                while True:\n                    guard.before_advance(ctl)'):]
        self.assertIn('status = ctl.advance()', loop[:200])
        self.assertIn('guard = SearchBudgetGuard.from_env(', source)
        self.assertLess(source.index('guard = SearchBudgetGuard.from_env('),
                        source.index('with random_stream.NativeRandom(random.getstate()) as rng'))


class RunnerBudgetSettingsTests(unittest.TestCase):
    def test_settings_defaults_and_validation(self):
        with clean_budget_env():
            settings = runner.search_budget_settings()
            self.assertEqual((settings['max_seconds'], settings['max_private_gb'], settings['max_advances_per_sim'],
                              settings['watchdog_interrupt']), (300.0, 4.0, 20.0, True))
            self.assertIn('watchdog_interrupt=1', runner.search_budget_text(settings))
        with clean_budget_env(WOTR_SEARCH_MAX_SECONDS='0', WOTR_SEARCH_MAX_PRIVATE_GB='0',
                              WOTR_SEARCH_WATCHDOG_INTERRUPT='0'):
            settings = runner.search_budget_settings()
            self.assertFalse(settings['watchdog_interrupt'])
            self.assertIsNone(runner.SearchBudgetWatchdog.from_settings(settings, '/x', 0))   # no thread at all
        for bad in ({'WOTR_SEARCH_MAX_SECONDS': '-5'}, {'WOTR_SEARCH_WATCHDOG_INTERRUPT': 'yes'},
                    {'WOTR_SEARCH_MAX_PRIVATE_GB': 'inf'}):
            with clean_budget_env(**bad):
                with self.assertRaises(ValueError):
                    runner.search_budget_settings()
        import run_td_v33loop as driver
        for name in runner.SEARCH_BUDGET_ENV_NAMES:
            self.assertIn(name, driver.JOURNAL_ENVIRONMENT_EXCLUDED)
        text = driver.search_budget_policy({})
        self.assertIn('max_seconds=300 max_private_gb=4.0 max_advances=20 x sims watchdog_interrupt=1', text)
        self.assertIn('max_seconds=7', driver.search_budget_policy({'WOTR_SEARCH_MAX_SECONDS': '7'}))

    def test_budget_exceptions_are_not_native_refusals(self):
        error = runner.SearchWatchdogBreach('search budget exceeded (watchdog): wall_seconds=301.2 (limit 300)')
        self.assertFalse(runner.is_native_refusal(error))


def _budget_episode(ai, state, search_seconds=None):
    """Fake episode: the first episode searches once (the fake search decides how long), the
    second completes normally - the pattern of the graceful-failure tests."""
    calls = []

    def episode(fp, sh, config):
        calls.append(config)
        if len(calls) == 1:
            state._declare_phase = True
            try:
                ai.mcts_search(state, 'sh', fp, sh, num_simulations=config['mcts_sims'])
            finally:
                state._declare_phase = False
        return {'game_stats': {'winner': 'fp', 'literal_fixture': True}}
    return episode, calls


class WatchdogWorkerTests(unittest.TestCase):
    def setUp(self):
        self.stack = patch.object(runner, 'SEARCH_WATCHDOG_POLL_SECONDS', 0.05)
        self.stack.start()
        self.addCleanup(self.stack.stop)

    def _rows(self, ai, temp, **env):
        with clean_budget_env(**env):
            return _run_worker(runner, ai, temp, 2)

    def test_wall_breach_interrupts_the_python_search_and_the_worker_continues(self):
        state = FakeState()
        entered = []

        def search(state_, side, fp, sh, num_simulations=100, **kwargs):
            entered.append(time.monotonic())
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:          # a Python loop that never converges
                time.sleep(0.002)
            raise AssertionError('the watchdog did not interrupt the search within 20 s')
        ai = _fake_ai(None)
        ai._run_single_episode, calls = _budget_episode(ai, state)
        ai.mcts_search = search
        with tempfile.TemporaryDirectory(prefix='wotr_watchdog_') as temp:
            rows = self._rows(ai, temp, WOTR_SEARCH_MAX_SECONDS='0.3', WOTR_SEARCH_MAX_PRIVATE_GB='0')
            self.assertEqual([row[0] for row in rows], ['p', 'p', 'e', 'p', 'g', 'p'])
            error = rows[2][3]
            self.assertEqual((error['status'], error['episode'], error['error_type']), ('error', 0, 'SearchWatchdogBreach'))
            self.assertIn('search budget exceeded (watchdog): wall_seconds=', error['error'])
            budget = error['search_budget']
            self.assertEqual((budget['source'], budget['which'], budget['limit'], budget['interrupted']),
                             ('watchdog', 'wall_seconds', 0.3, True))
            self.assertGreaterEqual(budget['value'], 0.3)
            self.assertLess(time.monotonic() - entered[0], 10.0)   # the search ended at the breach, not at 20 s
            failure = error['failure']
            self.assertEqual(failure['budget'], budget)
            self.assertEqual((failure['decision_index'], failure['side'], failure['turn']), (0, 'sh', 3))
            self.assertTrue(failure['phase_flags_at_search']['_declare_phase'])
            fixture = pickle.loads(Path(failure['fixture_path']).read_bytes())
            self.assertEqual(fixture['search_budget']['which'], 'wall_seconds')
            self.assertTrue(pickle.loads(fixture['state'])._declare_phase)
            watchdog_fixture = Path(budget['fixture_path'])
            self.assertEqual(watchdog_fixture.name, 'w7_g0_watchdog_d0.pkl')
            payload = pickle.loads(watchdog_fixture.read_bytes())
            self.assertEqual((payload['schema'], payload['decision'], payload['episode'], payload['side']),
                             (runner.WATCHDOG_FIXTURE_SCHEMA, 0, 0, 'sh'))
            restored = pickle.loads(payload['state'])          # shallow copy, board detached, flags as live
            self.assertTrue(restored._declare_phase)
            self.assertIsNone(restored.board)
            self.assertIsNotNone(state.board)                   # the live object was never mutated
            game = rows[4][2]
            self.assertEqual((game['status'], game['episode']), ('completed', 1))
            self.assertEqual(len(calls), 2)
            ledger = runner._failure_ledger_row(error, 'label', 'phase')
            self.assertEqual(ledger['budget']['which'], 'wall_seconds')
            json.dumps(ledger)

    def test_memory_breach_without_interrupt_marks_the_search_and_discards_its_result(self):
        state = FakeState()

        def search(state_, side, fp, sh, num_simulations=100, **kwargs):
            time.sleep(0.6)
            return ('an action that must never be played', {'visits': 1})
        ai = _fake_ai(None)
        ai._run_single_episode, calls = _budget_episode(ai, state)
        ai.mcts_search = search
        memory = Memory(0.75)                                    # +0.75 GB per sample, limit 2 GB
        original = runner.SearchBudgetWatchdog.from_settings
        with tempfile.TemporaryDirectory(prefix='wotr_watchdog_mem_') as temp, \
                patch.object(runner.SearchBudgetWatchdog, 'from_settings',
                             classmethod(lambda cls, settings, failures_dir, wid, **kw:
                                         original(settings, failures_dir, wid, memory=memory, **kw))):
            rows = self._rows(ai, temp, WOTR_SEARCH_MAX_SECONDS='0', WOTR_SEARCH_MAX_PRIVATE_GB='2',
                              WOTR_SEARCH_WATCHDOG_INTERRUPT='0')
            self.assertEqual([row[0] for row in rows], ['p', 'p', 'e', 'p', 'g', 'p'])
            error = rows[2][3]
            budget = error['search_budget']
            self.assertEqual((budget['source'], budget['which'], budget['limit'], budget['interrupted'],
                              budget['result_discarded']), ('watchdog', 'private_gb', 2.0, False, True))
            self.assertGreater(budget['value'], 2.0)
            self.assertIn('result discarded', error['error'])
            self.assertEqual(error['error_type'], 'SearchWatchdogBreach')
            self.assertTrue(Path(budget['fixture_path']).is_file())
            self.assertEqual(rows[4][2]['episode'], 1)

    def test_default_limits_never_fire_and_rows_match_disabled_limits(self):
        def make_ai():
            def episode(fp, sh, config):
                ai.mcts_search(FakeState(), 'fp', fp, sh, num_simulations=config['mcts_sims'])
                return {'game_stats': {'winner': 'sh', 'turns': 9, 'config_seed': config.get('rng_episode_seed')}}
            ai = _fake_ai(episode)
            ai.mcts_search = lambda *a, **k: ('a', {})
            return ai
        outputs = []
        for env in ({}, {'WOTR_SEARCH_MAX_SECONDS': '0', 'WOTR_SEARCH_MAX_PRIVATE_GB': '0'}):
            with tempfile.TemporaryDirectory(prefix='wotr_watchdog_parity_') as temp:
                rows = self._rows(make_ai(), temp, **env)
            cleaned = []
            for row in rows:
                record = dict(row[2])
                if row[0] == 'g':
                    record.pop('wall_s', None)
                    record['trace_path'] = os.path.basename(record['trace_path'])   # temp dirs differ
                cleaned.append((row[0], row[1], json.dumps(record, sort_keys=True)))
            outputs.append(cleaned)
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual([r[0] for r in outputs[0]], ['p', 'p', 'g', 'p', 'p', 'g', 'p'])
        self.assertNotIn('search_budget', json.loads(outputs[0][2][2]))
        self.assertFalse(any(t.name.startswith('wotr-search-watchdog') for t in threading.enumerate()))

    def test_orchestrator_budget_error_is_recorded_with_counters_and_export_tail(self):
        orch = load_orchestrator()
        ctl = FakeController()
        state = FakeState()

        def native_mcts_search(ai_module, state_, side, fp, sh, **kwargs):
            guard = orch.SearchBudgetGuard(ai_module, {'max_seconds': 0, 'max_private_gb': 0, 'max_advances': 3,
                                                       'export': 'auto'}, kwargs['num_simulations'],
                                           root_state=state_, search_kwargs=dict(side=side, **kwargs),
                                           clock=Clock(0.001), memory=lambda: 0)
            while True:                                  # the orchestrator's loop shape; never converges
                guard.before_advance(ctl)
                ctl.advance()
        drops = []
        stamp = {'search_route': 'native', 'controller_binary_sha256': 'c' * 64, 'reference_sha256': 'r' * 64}
        orchestrator = types.SimpleNamespace(native_mcts_search=native_mcts_search,
                                             drop_cached_child=lambda ai_module: drops.append(1))
        ai = _fake_ai(None)
        ai._run_single_episode, calls = _budget_episode(ai, state)
        ai.mcts_search = lambda *a, **k: None            # replaced by the native route wrapper
        import queue
        sink = queue.Queue()
        with tempfile.TemporaryDirectory(prefix='wotr_native_budget_') as temp, \
                patch.dict(sys.modules, __import__('test_loop_resilience_served')._worker_modules(ai)), \
                patch.object(runner, 'native_search_setup', lambda ai_module: (orchestrator, dict(stamp))), \
                clean_budget_env(WOTR_SP_FP_PROFILES='ring', WOTR_SP_SH_PROFILES='military', WOTR_COMBAT_FORECAST_EVAL='0',
                                 WOTR_NATIVE_SEARCH='1', WOTR_SEARCH_MAX_SECONDS='0', WOTR_SEARCH_MAX_PRIVATE_GB='0'):
            runner._worker(3, object(), 2, 4, os.path.join(temp, 'traces'), 123, 'base', sink)
            rows = []
            while not sink.empty():
                rows.append(sink.get_nowait())
            errors = [row for row in rows if row[0] == 'e']
            games = [row for row in rows if row[0] == 'g']
            self.assertEqual((len(errors), len(games)), (1, 1))
            error = errors[0][3]
            self.assertEqual((error['status'], error['episode'], error['search_route'], error['native_refusal']),
                             ('error', 0, 'native', False))
            self.assertIn('native search budget exceeded: advances=4 (limit 3) sims=4 advances=4', error['error'])
            budget = error['search_budget']
            self.assertEqual((budget['source'], budget['which'], budget['value'], budget['limit'], budget['sims'],
                              budget['advances']), ('orchestrator', 'advances', 4, 3, 4, 4))
            self.assertEqual(error['failure']['budget'], budget)
            fixture = pickle.loads(Path(error['failure']['fixture_path']).read_bytes())
            self.assertEqual(fixture['search_budget']['counters'], {'search': {'leaf_batches': 3}, 'kernel': {}})
            self.assertEqual(fixture['search_budget']['export_tail']['nodes_total'], 200)
            self.assertEqual(fixture['search_budget']['search_kwargs']['num_simulations'], 4)
            self.assertTrue(pickle.loads(fixture['state'])._declare_phase)   # root state at the failing search
            self.assertEqual(games[0][2]['episode'], 1)
            self.assertEqual(ctl.calls, 3)
        self.assertEqual(len(drops), 3)

    def test_loop_ledger_row_carries_the_budget(self):
        import run_td_v33loop as driver
        with tempfile.TemporaryDirectory(prefix='wotr_loop_budget_') as temp:
            loop = os.path.join(temp, 'loop')
            bdir = os.path.join(loop, 'batch_009')
            os.makedirs(bdir)
            lines = []
            with patch.object(driver, 'LOOP', loop), patch.object(driver, 'log_line', lines.append), \
                    patch.dict(os.environ, {}, clear=False):
                os.environ.pop('WOTR_LOOP_MIN_BATCH_GAMES', None)
                row = {'worker_id': 3, 'episode': 5, 'status': 'error', 'error': "SearchBudgetExceeded('...')",
                       'error_type': 'SearchBudgetExceeded', 'search_route': 'native',
                       'failure': {'fixture_path': '/x/failures/w3_g5_failure_d1698.pkl', 'decision_index': 1698,
                                   'budget': {'source': 'orchestrator', 'which': 'private_gb', 'value': 4.1,
                                              'limit': 4.0, 'sims': 1000, 'advances': 2931}}}
                driver._record_batch_shortfall(9, bdir, {'phase_id': 'p' * 32}, [row], 24, 25, 0, False)
            ledger = [json.loads(line) for line in Path(loop, driver.FAILURES_LEDGER).read_text().splitlines()]
            self.assertEqual(ledger[0]['budget']['which'], 'private_gb')
            self.assertTrue(any('[SEARCH BUDGET orchestrator: private_gb=4.1 limit 4.0]' in line for line in lines))


if __name__ == '__main__':
    unittest.main()
