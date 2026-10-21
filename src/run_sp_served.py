# -*- coding: utf-8 -*-
"""Self-play through the batched-GPU inference server (SCALE, session 34)
for BIG-net architectures (wide / xformer) whose CPU forward is too slow for
local per-worker inference (xformer CPU fwd B=64 = 2.2 s -> server-only).

Same game protocol as src/run_td_v33sp.py (the standard self-play runner):
ai._run_single_episode with max_turns 40, Shadow VP 10, the runner's sims,
random FP/SH profile per game, tree reuse, cross-turn horizon 1, traces per
game, games jsonl + window report - but the workers hold RemoteNet handles
(infer_server.attach_client) and ONE infer_server process on CUDA serves both
nets (the bench_scale.py 'server-*' pattern).  No frozen anchor, no roster.

Usage (all outputs under --out-dir; nothing in data/ is touched):
  python src/run_sp_served.py --arch xformer --fp-net wotr_fp_corpus_v42sxtd.pt
      --sh-net wotr_sh_corpus_v42sxtd.pt --games 500 --workers 12 --sims 1000
      --out-dir data/v42sx_selfplay_500

Each invocation requires a fresh output directory. Independent scheduled
tasks must pass unique --shard-id values within one output generation;
each gets out-dir/shard_NNNNNN and a disjoint worker seed interval. The
caller must use a fresh generation root for a new run and arrange GPU/CPU
affinity externally. Flux tasks must explicitly pass --shard-id.

GRACEFUL PER-GAME FAILURE (USER 2026-09-11: "don't crash fully on 1 break,
just handle it gracefully and we can note it in the log for later root cause
detection"): a game whose episode raises is recorded as a status=error
attempt (error text, traceback tail, decision index / turn / side of the
failing search) with a FAILURE FIXTURE pickled under <out-dir>/failures/
(the root GameState + turn-start phase flags AS THEY WERE AT THE FAILING
SEARCH, harness-revision-5 style, plus ai's own search_failure diagnostic),
one line in <out-dir>/failures.jsonl, and the worker CONTINUES with its next
assigned episode.  The parent stops the batch only for worker-level failures
(init, crash, reporting gap) or when the number of game failures exceeds
WOTR_SP_MAX_GAME_FAILURES (default = the batch size, i.e. never for game
errors).  The exit code is 0 when every assigned episode was reported.

NATIVE SEARCH ROUTE (WOTR_NATIVE_SEARCH=1; default off = the Python search,
byte-for-byte the previous behaviour): the worker's ai.mcts_search is routed
through the native controller (native_search_core/inference/orchestrator.py)
after a loud pin check against the current immutable source pin (override for
tests: WOTR_NATIVE_SEARCH_REFERENCE) - the worker engine must BE the pinned
runtime; the candidate controller DLL (WOTR_NATIVE_SEARCH_BOX, default the
resident controller/compile_check build) is the first native library loaded
in the process; the required capability groups are checked by symbol.  Every
attempt / game record is stamped with the route, pin, controller build and
orchestrator identities.  A native-route error is a per-game failure like any
other (fixture + continue); a native SETUP failure is a worker-init failure.

PER-SEARCH BUDGET (package Y2, 2026-09-11; incident supervisor_r3_native batch 4
episode (3,5): ONE native search never finished, 42 GB private bytes, the
progress heartbeat stayed alive so the 1800-s stall watchdog never fired):
  * native route: the orchestrator checks wall time / private-bytes growth /
    advance count BETWEEN controller advances (WOTR_SEARCH_MAX_SECONDS=300,
    WOTR_SEARCH_MAX_PRIVATE_GB=4.0 growth, WOTR_SEARCH_MAX_ADVANCES=20 x sims;
    0 = off) and raises SearchBudgetExceeded - a per-game failure recorded
    here with its counters / export tail (fixture + ledger), batch continues.
  * both routes: SearchBudgetWatchdog, a runner-side daemon thread armed by the
    failure-capture wrapper around every ai.mcts_search call.  On the same
    wall / memory thresholds it (1) pickles the root state + decision index to
    <out-dir>/failures/w<k>_g<j>_watchdog_d<n>.pkl, (2) MARKS the search so its
    result, if it ever returns, is discarded and the game recorded as
    status=error, and (3) with WOTR_SEARCH_WATCHDOG_INTERRUPT=1 (default)
    delivers SearchWatchdogBreach into the worker's main thread through
    PyThreadState_SetAsyncExc, which ends a Python-route search at its next
    bytecode boundary (ai.py is not edited; the exception travels through
    ai's own _search_failure_scope like any search error).  What the thread
    CANNOT do: interrupt a single native advance() / any C call that never
    returns (the exception is delivered only when the main thread re-enters
    the Python eval loop), or free the memory of a search that is still
    running - there it only writes the fixture and logs; the 1800-s stall
    watchdog (no progress) or the OS remain the backstop.  With the
    interrupt disabled the thread only records + marks.
"""
import argparse
import contextlib
import copy
import functools
import hashlib
import json
import math
import os
import pickle
import random
import sys
import threading
import time
import traceback
import uuid

from served_lifecycle import cleanup as _cleanup_children

_CLEANUP_JOIN_SECONDS = 5.0
_CLEANUP_KILL_SECONDS = 5.0

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
os.environ.setdefault('PYTHONHASHSEED', '0')
os.environ.setdefault('WOTR_TURN_HORIZON', '1')
os.environ.setdefault('WOTR_ACT_GEN', 'v41')
os.environ.setdefault('WOTR_VALUE_HEAD_ACT', 'scaled2')

# Explicit shards receive disjoint worker seed intervals, even when their
# worker counts differ. NumPy's legacy seed API requires unsigned 32-bit seeds.
SHARD_SEED_STRIDE = 65536

# ---- graceful per-game failure + native route (package Y, 2026-09-11) ----
NATIVE_SEARCH_ENV = 'WOTR_NATIVE_SEARCH'
NATIVE_SEARCH_BOX_ENV = 'WOTR_NATIVE_SEARCH_BOX'
NATIVE_SEARCH_REFERENCE_ENV = 'WOTR_NATIVE_SEARCH_REFERENCE'
NATIVE_SEARCH_CORE_ENV = 'WOTR_NATIVE_SEARCH_CORE'
MAX_GAME_FAILURES_ENV = 'WOTR_SP_MAX_GAME_FAILURES'
# The optional milestone-2 export groups production needs (root noise, tree
# reuse, observer roots, combat requests); runner_hook_design.md section 2.
NATIVE_REQUIRED_CAPABILITIES = ('root_priors_exchange', 'tree_reuse', 'observer_events', 'combat_dist')
# Turn-start flags the frozen episode loop's finally blocks clear on the SAME
# state object while the exception propagates (harness revision 5).
TURN_START_FLAGS = ('_declare_phase', '_eye_allocation_phase', '_fellowship_tail_phase')
FAILURE_FIXTURE_SCHEMA = 'served_game_failure_fixture_v1'
# ---- per-search budget (package Y2, 2026-09-11) ----
SEARCH_MAX_SECONDS_ENV = 'WOTR_SEARCH_MAX_SECONDS'            # wall seconds per search, 0 = off
SEARCH_MAX_PRIVATE_GB_ENV = 'WOTR_SEARCH_MAX_PRIVATE_GB'      # private-bytes GROWTH per search (GB), 0 = off
SEARCH_MAX_ADVANCES_ENV = 'WOTR_SEARCH_MAX_ADVANCES'          # native: advances per simulation (factor), 0 = off
SEARCH_WATCHDOG_INTERRUPT_ENV = 'WOTR_SEARCH_WATCHDOG_INTERRUPT'  # 1 (default) = async-raise into the search thread
SEARCH_BUDGET_DEFAULTS = {'max_seconds': 300.0, 'max_private_gb': 4.0, 'max_advances_per_sim': 20.0,
                          'watchdog_interrupt': True}
SEARCH_WATCHDOG_POLL_SECONDS = 1.0
WATCHDOG_FIXTURE_SCHEMA = 'served_search_watchdog_fixture_v1'
SEARCH_BUDGET_ENV_NAMES = (SEARCH_MAX_SECONDS_ENV, SEARCH_MAX_PRIVATE_GB_ENV, SEARCH_MAX_ADVANCES_ENV,
                           SEARCH_WATCHDOG_INTERRUPT_ENV, 'WOTR_SEARCH_BUDGET_EXPORT')
GB = float(1024 ** 3)


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def native_search_enabled():
    return os.environ.get(NATIVE_SEARCH_ENV, '0') == '1'


def max_game_failures(games):
    """Per-batch game-failure budget: WOTR_SP_MAX_GAME_FAILURES, default = the batch size."""
    raw = os.environ.get(MAX_GAME_FAILURES_ENV)
    if raw is None or not raw.strip():
        return int(games)
    value = int(raw)
    if value < 0:
        raise ValueError('%s must be a nonnegative integer' % MAX_GAME_FAILURES_ENV)
    return value


def _env_nonnegative(name, default, kind):
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = kind(raw)
    except ValueError:
        raise ValueError('%s must be a nonnegative number (0 disables the check); got %r' % (name, raw))
    if value < 0 or (isinstance(value, float) and not math.isfinite(value)):
        raise ValueError('%s must be a nonnegative number (0 disables the check); got %r' % (name, raw))
    return value


def search_budget_settings():
    """The per-search budget in force (env, defaults above; 0 = that check disabled).  The same
    env names drive the orchestrator's guard (native route) and the runner's watchdog thread."""
    raw = (os.environ.get(SEARCH_WATCHDOG_INTERRUPT_ENV) or '1').strip()
    if raw not in ('0', '1'):
        raise ValueError('%s must be 0 or 1; got %r' % (SEARCH_WATCHDOG_INTERRUPT_ENV, raw))
    return {'max_seconds': _env_nonnegative(SEARCH_MAX_SECONDS_ENV, SEARCH_BUDGET_DEFAULTS['max_seconds'], float),
            'max_private_gb': _env_nonnegative(SEARCH_MAX_PRIVATE_GB_ENV, SEARCH_BUDGET_DEFAULTS['max_private_gb'], float),
            'max_advances_per_sim': _env_nonnegative(SEARCH_MAX_ADVANCES_ENV,
                                                     SEARCH_BUDGET_DEFAULTS['max_advances_per_sim'], float),
            'watchdog_interrupt': raw == '1',
            'watchdog_poll_seconds': SEARCH_WATCHDOG_POLL_SECONDS}


def search_budget_text(settings):
    return ('max_seconds=%s max_private_gb=%s max_advances=%s x sims watchdog_interrupt=%d'
            % (settings['max_seconds'], settings['max_private_gb'], settings['max_advances_per_sim'],
               int(settings['watchdog_interrupt'])))


def process_private_bytes():
    """Private bytes of this process (psutil; Windows `private` = commit charge, POSIX RSS)."""
    import psutil
    info = psutil.Process().memory_info()
    return int(getattr(info, 'private', None) or info.rss)


def native_search_core_dir():
    return os.path.abspath(os.environ.get(NATIVE_SEARCH_CORE_ENV)
                           or os.path.join(ROOT, 'native_backend', 'experiments', 'native_search_core'))


def native_search_reference_path():
    """The tested current-source pin; an explicit environment pin remains supported."""
    return os.path.abspath(os.environ.get(NATIVE_SEARCH_REFERENCE_ENV)
                           or os.path.join(ROOT, 'native_backend', 'reference_sep16_own_current_v3_r2.json'))


def native_search_box_dir(orchestrator=None):
    box = os.environ.get(NATIVE_SEARCH_BOX_ENV)
    return os.path.abspath(box) if box else os.path.join(
        native_search_core_dir(), 'own_value_controller_sep13', 'compile_check_sep16r1')


def _native_own_adapter():
    import importlib.util
    path = os.path.join(native_search_core_dir(), 'own_value_controller_sep13',
                        'served_integration', 'adapter.py')
    name = 'wotr_production_own_served_adapter'
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    elif os.path.normcase(os.path.abspath(module.__file__)) != os.path.normcase(os.path.abspath(path)):
        raise RuntimeError('Native own-model adapter path changed in this process')
    os.environ[module.ENV_PIN] = native_search_reference_path()
    os.environ[module.ENV_BOX] = native_search_box_dir()
    return module


def native_search_source_identity():
    """Actual own-model adapter identity, shared with the loop journal."""
    return _native_own_adapter().identity()


def verify_native_reference(ai_module, reference_path):
    """Loud pin check (runner_hook_design.md section 2.1): the worker's engine must BE the runtime
    pinned by the reference file - ai.__file__ is <runtime>/src/ai.py and every pinned source /
    data file hashes as pinned.  Never applies the reference environment (the loop's env is
    authoritative) and never imports a second engine."""
    with open(reference_path, encoding='utf-8') as handle:
        reference = json.load(handle)
    runtime = os.path.abspath(reference['runtime'])
    source = os.path.join(runtime, 'src')
    actual_ai = os.path.normcase(os.path.abspath(ai_module.__file__))
    pinned_ai = os.path.normcase(os.path.join(source, 'ai.py'))
    if actual_ai != pinned_ai:
        raise RuntimeError('native search pin check: worker engine %s is not the pinned runtime %s (pin file %s)'
                           % (ai_module.__file__, pinned_ai, reference_path))
    for name, expected in sorted((reference.get('source_sha256') or {}).items()):
        path = os.path.join(source, name)
        if not os.path.isfile(path) or _file_sha256(path) != expected:
            raise RuntimeError('native search pin check: reference source drift: %s (pin file %s)' % (name, reference_path))
    for name, expected in sorted((reference.get('data_sha256') or {}).items()):
        path = os.path.join(runtime, 'data', name)
        if not os.path.isfile(path) or _file_sha256(path) != expected:
            raise RuntimeError('native search pin check: reference data drift: %s (pin file %s)' % (name, reference_path))
    return {'reference_path': reference_path, 'reference_sha256': _file_sha256(reference_path),
            'reference_runtime': runtime, 'reference_ai_sha256': (reference.get('source_sha256') or {}).get('ai.py'),
            'pinned_files': len(reference.get('source_sha256') or {})}


def native_search_setup(ai_module):
    """Verify the current source and load the approved root-only controller."""
    return _native_own_adapter().setup(ai_module)


@contextlib.contextmanager
def native_search_route(ai_module, orchestrator):
    """Use the existing served own-value route: never pass the opponent net."""
    with _native_own_adapter().route(ai_module, orchestrator):
        yield ai_module.mcts_search


def is_native_refusal(error):
    """oracle/test_parity.is_refusal: controller refusals are 'native: unavailable', never a
    crash report - stamped on the attempt so root-cause work separates the two."""
    name = type(error).__name__
    text = str(error)
    return (isinstance(error, NotImplementedError) or 'Unavailable' in name or 'Unsupported' in name
            or 'not available' in text or 'refuses' in text or 'refused' in text
            or 'unproved execution scope' in text or text.startswith('unported:'))


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in (sorted(value, key=repr) if isinstance(value, (set, frozenset)) else value)]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, 'name') and hasattr(value, 'value'):        # Enum members (Side)
        return str(value.name)
    return repr(value)


class SearchWatchdogBreach(RuntimeError):
    """Raised for a search the runner-side SearchBudgetWatchdog found over budget: delivered
    asynchronously into the searching thread (PyThreadState_SetAsyncExc instantiates the class
    with no args IN that thread - the message and `search_budget` registered for the thread by
    the watchdog are picked up here, so ai's own failure diagnostic sees the full text) or raised
    by the capture wrapper when a marked search returns.  Carries `search_budget` (dict) like the
    orchestrator's SearchBudgetExceeded.  Not a native refusal."""
    _pending = {}                 # thread ident -> (message, breach record), set before the async raise
    _pending_lock = threading.Lock()

    def __init__(self, *args):
        if not args:
            with self._pending_lock:
                pending = self._pending.pop(threading.get_ident(), None)
            if pending is not None:
                args = (pending[0],)
                self.search_budget = pending[1]
        super().__init__(*args)

    @classmethod
    def register(cls, thread_ident, message, record):
        with cls._pending_lock:
            cls._pending[int(thread_ident)] = (message, record)

    @classmethod
    def unregister(cls, thread_ident):
        with cls._pending_lock:
            cls._pending.pop(int(thread_ident), None)


def _async_raise(thread_ident, exc_type):
    """Queue `exc_type` in the thread `thread_ident` (CPython C API, documented for exactly this
    use).  Delivered at that thread's next bytecode boundary - never inside a C call."""
    import ctypes
    api = ctypes.pythonapi
    api.PyThreadState_SetAsyncExc.restype = ctypes.c_int
    modified = api.PyThreadState_SetAsyncExc(ctypes.c_ulong(int(thread_ident)), ctypes.py_object(exc_type))
    if modified > 1:                                     # ambiguous thread id: revert, never guess
        api.PyThreadState_SetAsyncExc(ctypes.c_ulong(int(thread_ident)), None)
        raise SystemError('PyThreadState_SetAsyncExc modified %d thread states' % modified)
    return modified == 1


def _shallow_state_pickle(state):
    """Pickle a board-less SHALLOW copy of the root state from the watchdog thread: the live
    object is never mutated (the search thread may still read it).  (bytes, error)."""
    try:
        clone = copy.copy(state)
        try:
            clone.board = None
        except Exception:
            pass
        return pickle.dumps(clone, protocol=4), None
    except Exception as error:
        return None, '%s: %s' % (type(error).__name__, error)


class SearchBudgetWatchdog(threading.Thread):
    """Runner-side per-search budget thread (both routes; see the module docstring).

    arm()/disarm() bracket every ai.mcts_search call (the capture wrapper does it).  While armed,
    each poll compares the wall time since arm and the process' private-bytes growth since arm
    with the limits; the first breach (once per search) writes the watchdog fixture, records
    `breach` (read by disarm) and, when `interrupt` is on, async-raises SearchWatchdogBreach into
    the ARMING thread under the same lock disarm() takes - so no exception is ever queued after
    the wrapper has disarmed.  Limits all zero -> `from_settings` returns None (no thread).
    """

    def __init__(self, settings, failures_dir, worker_id, *, clock=time.monotonic, memory=process_private_bytes,
                 raise_async=_async_raise, poll_seconds=None):
        super().__init__(name='wotr-search-watchdog-w%s' % worker_id, daemon=True)
        self.settings = dict(settings)
        self.failures_dir = failures_dir
        self.worker_id = worker_id
        self.clock, self.memory, self.raise_async = clock, memory, raise_async
        self.poll_seconds = float(poll_seconds if poll_seconds is not None
                                  else settings.get('watchdog_poll_seconds') or SEARCH_WATCHDOG_POLL_SECONDS)
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.stopping = False
        self.armed = None          # {'state','side','kwargs','decision','episode','thread','started','private_start'}
        self.breach = None         # the breach record of the armed search (set once)
        self.episode = None
        self.breaches = 0
        self.errors = []

    @classmethod
    def from_settings(cls, settings, failures_dir, worker_id, **overrides):
        if not (settings.get('max_seconds') or settings.get('max_private_gb')):
            return None
        return cls(settings, failures_dir, worker_id, **overrides)

    # -- called from the searching thread -------------------------------------------------
    def arm(self, state, side, kwargs, decision_index, episode=None):
        with self.lock:
            private = None
            if self.settings.get('max_private_gb'):
                try:
                    private = self.memory()
                except Exception as error:
                    self.errors.append('memory baseline: %s: %s' % (type(error).__name__, error))
            self.armed = {'state': state, 'side': side, 'kwargs': dict(kwargs or {}), 'decision': decision_index,
                          'episode': self.episode if episode is None else episode,
                          'thread': threading.get_ident(), 'started': self.clock(), 'private_start': private}
            self.breach = None
        self.wake.set()

    def disarm(self):
        """Stop watching the current search; returns its breach record (or None)."""
        with self.lock:
            breach, self.armed, self.breach = self.breach, None, None
        return breach

    def stop(self):
        self.stopping = True
        self.wake.set()

    # -- thread body ----------------------------------------------------------------------
    def run(self):
        while not self.stopping:
            self.wake.wait(self.poll_seconds)
            self.wake.clear()
            if self.stopping:
                break
            try:
                self.check()
            except Exception as error:          # the watchdog must never take the worker down
                self.errors.append('check: %s: %s' % (type(error).__name__, error))

    def check(self):
        with self.lock:
            armed = self.armed
            if armed is None or self.breach is not None:
                return
            elapsed = self.clock() - armed['started']
            which = value = limit = None
            growth = None
            max_seconds = self.settings.get('max_seconds') or 0
            if max_seconds and elapsed > max_seconds:
                which, value, limit = 'wall_seconds', round(elapsed, 1), max_seconds
            max_gb = self.settings.get('max_private_gb') or 0
            if which is None and max_gb and armed['private_start'] is not None:
                growth = self.memory() - armed['private_start']
                if growth > max_gb * GB:
                    which, value, limit = 'private_gb', round(growth / GB, 3), max_gb
            if which is None:
                return
            record = {'source': 'watchdog', 'which': which, 'value': value, 'limit': limit,
                      'decision_index': armed['decision'], 'episode': armed['episode'],
                      'elapsed_seconds': round(elapsed, 3),
                      'private_growth_gb': None if growth is None else round(growth / GB, 4),
                      'interrupt': bool(self.settings.get('watchdog_interrupt')), 'interrupted': False,
                      'fixture_path': None, 'fixture_error': None, 'worker_id': self.worker_id}
            record['fixture_path'], record['fixture_error'] = self.write_fixture(armed, record)
            if record['interrupt']:
                message = ('search budget exceeded (watchdog): %s=%s (limit %s) decision=%s elapsed=%ss'
                           % (which, value, limit, armed['decision'], record['elapsed_seconds']))
                SearchWatchdogBreach.register(armed['thread'], message, record)
                try:
                    record['interrupted'] = bool(self.raise_async(armed['thread'], SearchWatchdogBreach))
                except Exception as error:
                    record['interrupt_error'] = '%s: %s' % (type(error).__name__, error)
                if not record['interrupted']:
                    SearchWatchdogBreach.unregister(armed['thread'])
            self.breach = record
            self.breaches += 1
            print('[search-watchdog] worker %s episode %s decision %s: %s=%s over limit %s after %.1fs -> %s%s'
                  % (self.worker_id, armed['episode'], armed['decision'], which, value, limit, elapsed,
                     'interrupting the search' if record['interrupted'] else 'marked (result will be discarded)',
                     '; fixture %s' % record['fixture_path'] if record['fixture_path'] else ''), flush=True)

    def write_fixture(self, armed, record):
        try:
            state = armed['state']
            state_pickle, state_error = _shallow_state_pickle(state)
            payload = {'schema': WATCHDOG_FIXTURE_SCHEMA, 'worker_id': self.worker_id, 'episode': armed['episode'],
                       'decision': armed['decision'], 'side': _jsonable(armed['side']),
                       'turn': getattr(state, 'turn', None),
                       'phase_flags_at_breach': {name: getattr(state, name, None) for name in TURN_START_FLAGS},
                       'kwargs': _jsonable(armed['kwargs']), 'budget': {k: v for k, v in record.items()
                                                                        if k not in ('fixture_path', 'fixture_error')},
                       'state': state_pickle, 'state_pickle_error': state_error,
                       'state_captured_at': 'watchdog_breach_live_search (shallow copy; the search was still running)',
                       'settings': dict(self.settings), 'written_at': time.time()}
            os.makedirs(self.failures_dir, exist_ok=True)
            name = 'w%s_g%s_watchdog_d%s.pkl' % (self.worker_id, armed['episode'],
                                                 'na' if armed['decision'] is None else armed['decision'])
            path = os.path.join(self.failures_dir, name)
            with open(path, 'wb') as handle:
                pickle.dump(payload, handle, protocol=4)
            return path, None
        except Exception as error:
            return None, '%s: %s' % (type(error).__name__, error)


def _budget_of(error):
    """The per-search budget record an exception carries (orchestrator guard or watchdog)."""
    budget = getattr(error, 'search_budget', None)
    return dict(budget) if isinstance(budget, dict) else None


def _budget_summary(budget):
    """JSON-able short form for attempt rows / ledgers (no counters or export)."""
    if not budget:
        return None
    keys = ('source', 'which', 'value', 'limit', 'sims', 'advances', 'elapsed_seconds', 'private_growth_gb',
            'interrupted', 'fixture_path', 'result_discarded')
    return _jsonable({k: budget.get(k) for k in keys if k in budget})


class SearchFailureCapture:
    """Observation-only wrapper around ai.mcts_search (installed like ProgressEmitter's): on
    success it returns the inner result untouched and draws no randomness; on an exception it
    pickles the root GameState (board detached) with the turn-start phase flags AS AT THE
    FAILING SEARCH plus side / turn / kwargs / decision index, then re-raises.  The frozen
    episode loop's finally blocks clear _declare_phase / _eye_allocation_phase on this same
    object before the worker sees the exception - which is why the capture lives here.

    With a SearchBudgetWatchdog the wrapper arms it before and disarms it after every search;
    a breach recorded while the search ran makes the wrapper raise SearchWatchdogBreach (the
    result of a marked search is DISCARDED, never played) and is attached to the error as
    `search_budget`."""

    def __init__(self, ai_module, watchdog=None):
        self.ai = ai_module
        self.watchdog = watchdog
        self.decisions = 0
        self.captured = None
        self.original = None
        self.installed = False

    def reset(self):
        self.decisions = 0
        self.captured = None

    def install(self):
        original = getattr(self.ai, 'mcts_search', None)
        if original is None:
            # A stubbed engine without a search (tests): nothing to observe; the
            # fixture records `state_captured_at: no_search_capture` in that case.
            return None
        self.installed = True
        capture = self
        watchdog = self.watchdog

        @functools.wraps(original)
        def captured_search(state, side, *args, **kwargs):
            index = capture.decisions
            capture.decisions += 1
            if watchdog is not None:
                watchdog.arm(state, side, kwargs, index)
            try:
                result = original(state, side, *args, **kwargs)
            except BaseException as error:
                breach = watchdog.disarm() if watchdog is not None else None
                if breach is not None and _budget_of(error) is None:
                    # The watchdog's own interrupt (no args) or another error during a marked
                    # search: carry the breach on the error; name the breach in the message.
                    error.search_budget = breach
                    if isinstance(error, SearchWatchdogBreach) and not error.args:
                        error.args = ('search budget exceeded (watchdog): %s=%s (limit %s) decision=%s elapsed=%ss'
                                      % (breach['which'], breach['value'], breach['limit'], breach['decision_index'],
                                         breach['elapsed_seconds']),)
                capture.captured = capture.capture(state, side, args, kwargs, error, index)
                raise
            breach = watchdog.disarm() if watchdog is not None else None
            if breach is not None:
                # Marked while it ran (interrupt off or not yet delivered): the result is discarded.
                breach = dict(breach, result_discarded=True)
                error = SearchWatchdogBreach('search budget exceeded (watchdog): %s=%s (limit %s) decision=%s '
                                             'elapsed=%ss; the search returned after the breach - result discarded'
                                             % (breach['which'], breach['value'], breach['limit'],
                                                breach['decision_index'], breach['elapsed_seconds']))
                error.search_budget = breach
                capture.captured = capture.capture(state, side, args, kwargs, error, index)
                raise error
            return result
        captured_search.__wotr_failure_capture__ = capture
        self.original = original
        self.ai.mcts_search = captured_search
        return original

    def capture(self, state, side, args, kwargs, error, index):
        options = {name: kwargs.get(name) for name in ('num_simulations', 'c_puct', 'temperature', 'add_noise',
                                                        'hot_regions', 'bonus_sims', 'determinize') if name in kwargs}
        if args:
            options['positional_args'] = len(args)
        out = {'decision_index': index, 'side': _jsonable(side), 'turn': getattr(state, 'turn', None),
               'phase_flags_at_search': {name: getattr(state, name, None) for name in TURN_START_FLAGS},
               'search_options': _jsonable(options), 'error': '%s: %s' % (type(error).__name__, error),
               'error_type': type(error).__name__, 'state_captured_at': 'failing_search',
               'state_pickle': None, 'state_capture_error': None}
        board = getattr(state, 'board', None)
        try:
            state.board = None                  # oracle fixtures store states board-less
            out['state_pickle'] = pickle.dumps(state, protocol=4)
        except Exception as capture_error:      # a pickling error is recorded, never raised
            out['state_capture_error'] = '%s: %s' % (type(capture_error).__name__, capture_error)
        finally:
            state.board = board
        return out


def failures_dir_for(trace_dir):
    """<out-dir>/failures beside <out-dir>/traces; a bare directory keeps its fixtures inside."""
    trace_dir = os.path.abspath(trace_dir)
    if os.path.basename(trace_dir) == 'traces':
        return os.path.join(os.path.dirname(trace_dir), 'failures')
    return os.path.join(trace_dir, 'failures')


def write_failure_fixture(failures_dir, wid, episode, captured, ai_module, attempt, error, trace_path):
    """Pickle <failures_dir>/w<wid>_g<episode>_failure_d<decision>.pkl and return its summary.
    Layout follows selfplay/native_episode.write_failure_fixture: payload['state'] = the pickled
    board-less GameState (as at the failing search), 'side', 'kwargs', 'error', 'phase_flags_at_search',
    'state_captured_at', plus ai's search_failure diagnostic packet (search-entry RNG, root/node
    snapshots) when the search scope captured one, and the attempt identity / trace path."""
    diagnostic = getattr(ai_module, '_SEARCH_FAILURE_DIAGNOSTIC', None)
    try:
        diagnostic = copy.deepcopy(diagnostic) if diagnostic is not None else None
        diagnostic_error = None
    except Exception as copy_error:
        diagnostic, diagnostic_error = None, '%s: %s' % (type(copy_error).__name__, copy_error)
    captured = dict(captured or {})
    decision = captured.get('decision_index')
    budget = _budget_of(error)
    budget_pickle_error = None
    if budget is not None:
        try:
            pickle.dumps(budget, protocol=4)
        except Exception as pickle_error:       # counters/export must never block the fixture
            budget_pickle_error = '%s: %s' % (type(pickle_error).__name__, pickle_error)
            budget = _jsonable(budget)
    payload = {'schema': FAILURE_FIXTURE_SCHEMA, 'worker_id': wid, 'episode': episode,
               'decision': decision, 'site': None,
               'side': captured.get('side'), 'turn': captured.get('turn'),
               'kwargs': captured.get('search_options'),
               'phase_flags_at_search': captured.get('phase_flags_at_search'),
               'state_captured_at': captured.get('state_captured_at') if captured else 'no_search_capture',
               'state': captured.get('state_pickle'), 'state_pickle_error': captured.get('state_capture_error'),
               'error': {'type': type(error).__name__, 'text': str(error), 'repr': repr(error),
                         'traceback': traceback.format_exc()[-6000:], 'refusal': is_native_refusal(error)},
               'search_failure': diagnostic, 'search_failure_copy_error': diagnostic_error,
               # Y2: the per-search budget record (orchestrator guard: counters + export tail +
               # kwargs; watchdog: breach + its own fixture path), None for ordinary failures.
               'search_budget': budget, 'search_budget_pickle_error': budget_pickle_error,
               'attempt': _jsonable({k: v for k, v in attempt.items() if k != 'native_search'}),
               'trace_path': trace_path, 'episode_config_note': 'see attempt (profiles, checkpoint_ids, rng_episode_*)',
               'search_route': attempt.get('search_route', 'python'), 'native_search': attempt.get('native_search'),
               'ai_path': getattr(ai_module, '__file__', None), 'written_at': time.time()}
    os.makedirs(failures_dir, exist_ok=True)
    name = 'w%d_g%d_failure_d%s.pkl' % (wid, episode, 'na' if decision is None else decision)
    path = os.path.join(failures_dir, name)
    with open(path, 'wb') as handle:
        pickle.dump(payload, handle, protocol=4)
    return {'fixture_path': path, 'fixture_sha256': _file_sha256(path), 'schema': FAILURE_FIXTURE_SCHEMA,
            'decision_index': decision, 'side': captured.get('side'), 'turn': captured.get('turn'),
            'phase_flags_at_search': captured.get('phase_flags_at_search'),
            'state_captured': captured.get('state_pickle') is not None,
            'search_failure_captured': diagnostic is not None,
            'search_failure_error': (diagnostic or {}).get('error') if isinstance(diagnostic, dict) else None,
            'budget': _budget_summary(budget)}


def _prepare_run(args):
    """Resolve caller paths and reserve a fresh output before engine imports."""
    from rng_streams import current_contract as rng_contract, LEGACY
    random_contract = rng_contract()
    if args.games < 1 or not 1 <= args.workers <= SHARD_SEED_STRIDE or args.sims < 1:
        raise ValueError('games and sims must be positive; workers must be in 1..65536')
    shard = getattr(args, 'shard_id', None)
    if 'FLUX_TASK_RANK' in os.environ and shard is None:
        raise ValueError('Flux tasks require an explicit unique --shard-id within this output generation')
    if shard is not None and shard < 0:
        raise ValueError('shard-id must be nonnegative')
    seed = args.seed0 + (shard or 0) * SHARD_SEED_STRIDE
    if args.seed0 < 0 or seed + args.workers - 1 >= 2 ** 32:
        raise ValueError('Requested shard worker seeds exceed the unsigned 32-bit seed range')
    paths = {side: os.path.abspath(os.path.join(ROOT, getattr(args, side + '_net')))
             for side in ('fp', 'sh')}
    for path in paths.values():
        if not os.path.isfile(path):
            raise FileNotFoundError('missing net: %s' % path)
    from served_completed_reuse import load as load_completed_reuse
    args._completed_reuse = load_completed_reuse(args, paths, seed)
    output = os.path.abspath(args.out_dir)
    if shard is not None:
        output = os.path.join(output, 'shard_%06d' % shard)
    # An existing directory is never a resume request. This also atomically
    # refuses concurrent tasks accidentally assigned the same shard.
    os.makedirs(output, exist_ok=False)
    manifest = {'status': 'prepared', 'shard_id': shard, 'seed0': args.seed0,
                'worker_seed_base': seed, 'worker_seed_stride': SHARD_SEED_STRIDE,
                'workers': args.workers, 'games': args.games, 'sims': args.sims,
                'arch': args.arch, 'output': output,
                'scheduler': {key: os.environ.get(key) for key in (
                    'FLUX_TASK_RANK', 'FLUX_JOB_ID', 'CUDA_VISIBLE_DEVICES',
                    'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES')},
                'checkpoints': {side: {'path': path, 'sha256': _file_sha256(path)}
                                for side, path in paths.items()}}
    if random_contract != LEGACY:
        manifest.update(rng_contract=random_contract, rng_run_seed=seed,
                        rng_episode_assignment='global_index = worker_id + local_episode * workers',
                        rng_episode_indices=[0, args.games - 1])
    if native_search_enabled():
        # Route identities are file hashes only; the DLL loads in the workers.
        manifest['search_route'] = 'native'
        manifest['native_search'] = native_search_source_identity()
    manifest['search_budget'] = search_budget_settings()       # Y2: limits in force (validated here, loudly)
    with open(os.path.join(output, 'launch.json'), 'x', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2)
    args.out_dir = output
    return paths, seed


def _server_entry(conns, net_paths, arch):
    # Preserve the scheduler's per-task GPU assignment. Only the spawned
    # game workers hide GPUs in their own environment.
    sys.path.insert(0, SRC)
    import infer_server
    infer_server.server_main(conns, net_paths, arch)


def _episode_randomness(run_seed, episode_index):
    """Return explicit isolated episode identity and independent profile RNG."""
    from rng_streams import current_contract, LEGACY, served_episode_seed, derive_seed
    if current_contract() == LEGACY:
        return {}, random
    episode_seed = served_episode_seed(run_seed, episode_index)
    config = {'rng_episode_seed': episode_seed,
              'rng_episode_identity': {'run_seed': run_seed, 'episode_index': episode_index}}
    return config, random.Random(derive_seed(episode_seed, 'profiles', bits=64))


def _worker(wid, conn, n_games, sims, trace_dir, seed0, arch, out_q, worker_stride=1, completed_episodes=()):
    trace_dir = os.path.abspath(trace_dir)
    failures_dir = failures_dir_for(trace_dir)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'        # workers: CPU search only
    os.environ['WOTR_NET_ARCH'] = arch
    os.environ['WOTR_TREE_REUSE'] = '1'
    os.environ.setdefault('WOTR_SEARCH_DEFER_GC', '1')
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    sys.path.insert(0, SRC)
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import ai
    if os.environ.get('WOTR_COMBAT_FORECAST_EVAL') == '1':
        from combat_forecast_eval import install
        install(ai)
    ai.DEVICE = torch.device('cpu')
    from served_progress import ProgressEmitter
    progress = ProgressEmitter(wid, out_q)
    progress.completed('worker_started', force=True)
    native = native_search_enabled()
    orchestrator, native_stamp = None, None
    try:
        import infer_server
        import ai_policies
        from policy_keys_backend import activate
        policy_backend = activate(ai_policies)
        if native:
            # Section 2 order: pin check, candidate DLL first, orchestrator, capabilities,
            # decorated native search built before ai.mcts_search is patched below.
            orchestrator, native_stamp = native_search_setup(ai)
        handles = infer_server.attach_client(conn, ['fp', 'sh'], require_metadata=True)
        fpn, shn = handles['fp'], handles['sh']
        checkpoint_ids = {key: handles[key].checkpoint_id for key in ('fp', 'sh')}
        fp_profs = [p for p in (os.environ.get('WOTR_SP_FP_PROFILES')
                                or ','.join(ai.FP_PROFILES)).split(',') if p]
        sh_profs = [p for p in (os.environ.get('WOTR_SP_SH_PROFILES')
                                or ','.join(ai.SH_PROFILES)).split(',') if p]
    except Exception as e:
        record = {'stage': 'worker_init', 'error': repr(e), 'traceback': traceback.format_exc()[-4000:]}
        if native:
            record.update(search_route='native', native_refusal=is_native_refusal(e),
                          native_search=native_stamp)
        out_q.put(('e', wid, 'worker init failed: %s' % repr(e)[:160], record))
        return
    random.seed(seed0 + wid)
    np.random.seed(seed0 + wid)
    os.makedirs(trace_dir, exist_ok=True)
    route = native_search_route(ai, orchestrator) if native else contextlib.nullcontext()
    # Y2: the runner-side budget watchdog (None when both its limits are 0); it is armed per
    # search by the capture wrapper, lives for the worker's whole loop and is a daemon thread.
    watchdog = SearchBudgetWatchdog.from_settings(search_budget_settings(), failures_dir, wid)
    with route:
        # Wrapper order (outermost first): progress -> failure capture (+ watchdog) -> route
        # (native | python).  Both wrappers pass the inner result through untouched and draw
        # no randomness; the watchdog thread reads the clock and psutil only.
        capture = SearchFailureCapture(ai, watchdog)
        capture.install()
        progress.install(ai)
        if watchdog is not None:
            watchdog.start()
        try:
            _worker_episodes(wid, n_games, sims, trace_dir, failures_dir, seed0, arch, out_q, worker_stride,
                             completed_episodes, ai, fpn, shn, checkpoint_ids, fp_profs, sh_profs, policy_backend,
                             native, native_stamp, orchestrator, progress, capture, watchdog)
        finally:
            if watchdog is not None:
                watchdog.stop()
                watchdog.join(timeout=5.0)


def _worker_episodes(wid, n_games, sims, trace_dir, failures_dir, seed0, arch, out_q, worker_stride,
                     completed_episodes, ai, fpn, shn, checkpoint_ids, fp_profs, sh_profs, policy_backend,
                     native, native_stamp, orchestrator, progress, capture, watchdog):
    """The worker's episode loop (split out of _worker so the watchdog thread is stopped in
    one place; behaviour unchanged)."""
    for k in range(n_games):
        if k in completed_episodes:
            # Only receipt-verified isolated episodes are skipped; the next
            # episode still derives its original global identity and RNG.
            continue
        # Existing balanced counts correspond to round-robin global episodes.
        # This identity is unchanged when the same run uses a new worker count.
        rng_config, profile_rng = _episode_randomness(seed0, wid + k * worker_stride)
        ai.PLAYER_PROFILES['fp'] = profile_rng.choice(fp_profs)
        ai.PLAYER_PROFILES['sh'] = profile_rng.choice(sh_profs)
        tp = os.path.join(trace_dir, 'w%d_g%d.trace.gz' % (wid, k))
        t0 = time.time()
        from reward_contract import current_contract
        attempt = {'record_type': 'game_attempt', 'worker_id': wid, 'episode': k,
                   'profiles': dict(ai.PLAYER_PROFILES), 'checkpoint_ids': dict(checkpoint_ids),
                   'reward_contract': current_contract(), 'trace_path': tp, 'frozen_side': None}
        if rng_config:
            from rng_streams import ISOLATED
            attempt.update(rng_contract=ISOLATED, **rng_config)
        if native:
            attempt.update(search_route='native', native_search=native_stamp)
        capture.reset()
        if watchdog is not None:
            watchdog.episode = k
        try:
            progress.completed('episode_started', force=True)
            traj = ai._run_single_episode(
                fpn, shn, {'max_turns': 40, 'mcts_sims': sims, 'shadow_vp': 10,
                           'trace_path': tp, 'frozen_side': None,
                           'checkpoint_ids': dict(checkpoint_ids), **rng_config})
            gs9 = dict(traj.get('game_stats') or {})
            gs9['wall_s'] = round(time.time() - t0, 1)
            gs9.setdefault('trace_path', tp)
            gs9.setdefault('profiles', {'fp': ai.PLAYER_PROFILES['fp'],
                                        'sh': ai.PLAYER_PROFILES['sh']})
            gs9['search_acceleration'] = {
                'policy_keys': policy_backend['backend'],
                'deferred_gc': os.environ.get('WOTR_SEARCH_DEFER_GC', '0') == '1',
                'cuda_graph_requested': os.environ.get('WOTR_INFER_CUDA_GRAPH', '0') == '1'}
            if native:
                gs9['search_route'] = 'native'
                gs9['search_acceleration']['native_search'] = native_stamp
            gs9['served_arch'] = arch
            gs9['checkpoint_ids'] = dict(checkpoint_ids)
            gs9.update(worker_id=wid, episode=k, status='completed',
                       reward_contract=current_contract())
            out_q.put(('g', wid, gs9))
            progress.completed('episode_completed', force=True)
        except Exception as e:
            # Graceful per-game failure: record + fixture, then CONTINUE with the
            # next assigned episode (the parent enforces the failure budget).
            attempt.update(status='error', stage='episode', error=repr(e),
                           error_type=type(e).__name__, wall_s=round(time.time() - t0, 1),
                           traceback=traceback.format_exc()[-4000:])
            if native:
                attempt['native_refusal'] = is_native_refusal(e)
            budget = _budget_of(e)
            if budget is not None:
                attempt['search_budget'] = _budget_summary(budget)
            try:
                attempt['failure'] = write_failure_fixture(failures_dir, wid, k, capture.captured, ai,
                                                           attempt, e, tp)
            except Exception as fixture_error:      # the fixture must never mask the game error
                attempt['failure'] = {'fixture_path': None,
                                      'fixture_error': '%s: %s' % (type(fixture_error).__name__, fixture_error),
                                      'decision_index': (capture.captured or {}).get('decision_index'),
                                      'side': (capture.captured or {}).get('side'),
                                      'turn': (capture.captured or {}).get('turn'),
                                      'budget': _budget_summary(budget)}
            out_q.put(('e', wid, repr(e)[:200], attempt))
            # No progress message for failed work (the stall deadline is renewed
            # only by successful searches/actions; the next episode_started does).
            if orchestrator is not None:
                # No native TreeSession may carry into the next episode.
                orchestrator.drop_cached_child(ai)
            continue


def _failure_ledger_row(attempt, label, phase_id):
    failure = attempt.get('failure') or {}
    return {'record_type': 'game_failure', 'batch': label, 'phase_id': phase_id,
            'worker_id': attempt.get('worker_id'), 'episode': attempt.get('episode'),
            'error': attempt.get('error'), 'error_type': attempt.get('error_type'),
            'decision_index': failure.get('decision_index'), 'side': failure.get('side'),
            'turn': failure.get('turn'), 'fixture_path': failure.get('fixture_path'),
            'fixture_sha256': failure.get('fixture_sha256'),
            'search_failure_error': failure.get('search_failure_error'),
            'trace_path': attempt.get('trace_path'), 'search_route': attempt.get('search_route', 'python'),
            'native_refusal': attempt.get('native_refusal'), 'wall_s': attempt.get('wall_s'),
            'rng_episode_identity': attempt.get('rng_episode_identity'),
            'budget': failure.get('budget') or attempt.get('search_budget'),
            'traceback_tail': (attempt.get('traceback') or '')[-1500:], 'recorded_at': time.time()}


def run(args):
    paths, seed0 = _prepare_run(args)
    import multiprocessing as mp
    import queue as queue_module
    from run_td_v33sp import (_append_attempt, _message_record,
                             _write_phase_summary, _publish_phase_summary)
    ctx = mp.get_context('spawn')
    out_q = ctx.Queue()
    phase_id = uuid.uuid4().hex
    attempts_path = os.path.join(args.out_dir, 'attempts.jsonl')
    phase_path = os.path.join(args.out_dir, 'phase.json')
    games_log = os.path.join(args.out_dir, 'games.jsonl')
    report_path = os.path.join(args.out_dir, 'report.md')
    trace_dir = os.path.join(args.out_dir, 'traces')
    failures_path = os.path.join(args.out_dir, 'failures.jsonl')
    failure_budget = max_game_failures(args.games)
    label = 'served-%s selfplay' % args.arch
    assignments = {wid: args.games // args.workers + (wid < args.games % args.workers)
                   for wid in range(min(args.workers, args.games))}
    stats, errors, attempts, failures = [], [], [], []
    worker_errors = []
    seen = set()
    from served_progress import ProgressWatch, stall_seconds
    progress_watch = ProgressWatch(stall_seconds())
    workers, pairs = [], []
    control_pair = None
    server = None
    cleanup_report = None
    parent_error = None
    post_cleanup_error = None
    _write_phase_summary(phase_path, phase_id, label, args.games, attempts, failures, attempts_path)
    if native_search_enabled():
        print('[%s] SEARCH ROUTE: native (WOTR_NATIVE_SEARCH=1; pin %s)'
              % (label, native_search_reference_path()), flush=True)
    print('[%s] SEARCH BUDGET: %s (0 = off; a breach = one status=error game, the batch continues)'
          % (label, search_budget_text(search_budget_settings())), flush=True)

    def handle(message):
        if isinstance(message, (tuple, list)) and len(message) == 3 and message[0] == 'p':
            progress_watch.accept(message[1], message[2])
            return
        if not isinstance(message, (tuple, list)) or len(message) < 3 or message[0] not in ('g', 'e'):
            raise ValueError('Malformed served worker message')
        wid = message[1]
        if isinstance(wid, bool) or wid not in assignments:
            raise ValueError('Unassigned served worker ID')
        attempt, failure = _message_record(message, phase_id, label)
        if attempt is not None:
            episode = attempt.get('episode')
            if (isinstance(episode, bool) or not isinstance(episode, int)
                    or not 0 <= episode < assignments[wid]):
                raise ValueError('Invalid served episode identity')
            identity = (wid, episode)
            if identity in seen:
                raise ValueError('Duplicate served episode result')
            seen.add(identity)
            attempts.append(attempt)
            _append_attempt(attempts_path, attempt)
        if failure is not None:
            failures.append(failure)
        _write_phase_summary(phase_path, phase_id, label, args.games, attempts, failures, attempts_path)
        if message[0] == 'g':
            stats.append(message[2])
            ai.append_game_log(games_log, message[2], label)
            return
        errors.append(message)
        print('[%s] worker %d ERROR: %s' % (label, wid, message[2]), flush=True)
        if attempt is None:
            # Worker-level failure (init, protocol): the worker cannot report its games.
            worker_errors.append(message)
            raise RuntimeError('Served worker failed; stopping the batch: %s' % message[2])
        # A game failure: ledger + loud line, the batch continues within its budget.
        _append_attempt(failures_path, _failure_ledger_row(attempt, label, phase_id))
        game_failures = sum(row.get('status') == 'error' for row in attempts)
        detail = attempt.get('failure') or {}
        budget = detail.get('budget') or attempt.get('search_budget')
        print('[%s] GAME FAILURE %d/%d budget (worker %d episode %s, decision %s turn %s side %s): %s -> fixture %s; '
              'batch continues%s' % (label, game_failures, failure_budget, wid, attempt.get('episode'),
                                     detail.get('decision_index'), detail.get('turn'), detail.get('side'),
                                     message[2], detail.get('fixture_path'),
                                     (' [SEARCH BUDGET %s: %s=%s limit %s]' % (budget.get('source'), budget.get('which'),
                                                                             budget.get('value'), budget.get('limit')))
                                     if budget else ''), flush=True)
        if game_failures > failure_budget:
            raise RuntimeError('Served batch exceeded %s=%d game failures; stopping the batch: %s'
                               % (MAX_GAME_FAILURES_ENV, failure_budget, message[2]))

    try:
        import ai
        from served_completed_reuse import materialize as materialize_completed
        imported = materialize_completed(args._completed_reuse, args.out_dir)
        for row in imported:
            handle(('g', row['worker_id'], row))
        # A separate, unused control pipe can always carry the small stop
        # message without writing behind an interrupted inference request.
        for _ in assignments:
            pairs.append(ctx.Pipe())
        control_pair = ctx.Pipe()
        server = ctx.Process(target=_server_entry,
                             args=([pair[0] for pair in pairs] + [control_pair[0]], paths, args.arch))
        server.start()
        for pair in pairs:
            pair[0].close()
        control_pair[0].close()
        time.sleep(3)
        if not server.is_alive():
            raise RuntimeError('Inference server failed during startup; no game workers were launched')
        for wid, count in assignments.items():
            worker_args = (wid, pairs[wid][1], count, args.sims, trace_dir,
                           seed0, args.arch, out_q, args.workers)
            if imported:
                worker_args += (tuple(r['episode'] for r in imported if r['worker_id'] == wid),)
            process = ctx.Process(target=_worker,
                                  args=worker_args)
            workers.append((wid, process))
            progress_watch.started(wid)
            process.start()
            pairs[wid][1].close()
        import ai
        while len(attempts) < args.games:
            active = [wid for wid, _ in workers
                if sum(row.get('worker_id') == wid for row in attempts) < assignments[wid]]
            progress_watch.check(active)
            progress_watch.publish(os.path.join(args.out_dir, 'progress.json'), phase_id, active)
            try:
                message = out_q.get(timeout=1)
            except queue_module.Empty:
                if not server.is_alive():
                    raise RuntimeError('Inference server exited before all game results arrived')
                for wid, process in workers:
                    if process.exitcode is not None and sum(row.get('worker_id') == wid for row in attempts) < assignments[wid]:
                        raise RuntimeError('Served worker %d exited before reporting its assigned games' % wid)
                if not any(process.is_alive() for _, process in workers):
                    break
                continue
            handle(message)
    except BaseException as exc:
        parent_error = {'stage': 'parent', 'error': repr(exc), 'error_type': type(exc).__name__}
        failures.append(parent_error)
        # Publish the failure before bounded child cleanup, so supervisors do
        # not mistake cleanup time for healthy progress. Preserve prior rows.
        failed = _write_phase_summary(phase_path, phase_id, label, args.games,
                                      attempts, failures, attempts_path)
        failed.update(parent_error=parent_error, stopping=True)
        _publish_phase_summary(phase_path, failed)
        raise
    finally:
        try:
            cleanup_report = _cleanup_children(
                workers, server, control_pair[1] if control_pair else None,
                join_seconds=_CLEANUP_JOIN_SECONDS, kill_seconds=_CLEANUP_KILL_SECONDS)
            # A worker may enqueue a duplicate just after the last expected
            # result. Once children are joined, drain remaining messages so
            # that extra results cannot be hidden by the terminal count.
            if parent_error is None and not cleanup_report['alive']:
                try:
                    while True:
                        try:
                            message = out_q.get_nowait()
                        except queue_module.Empty:
                            break
                        handle(message)
                except BaseException as exc:
                    post_cleanup_error = exc
                    protocol_error = {'stage': 'post_cleanup_protocol', 'error': repr(exc),
                                      'error_type': type(exc).__name__}
                    failures.append(protocol_error)
                    if parent_error is None:
                        parent_error = protocol_error
            for wid, count in assignments.items():
                observed = sum(row.get('worker_id') == wid for row in attempts)
                if observed < count:
                    failures.append({'record_type': 'worker_error', 'stage': 'reporting_gap',
                                     'worker_id': wid, 'assigned_games': count,
                                     'observed': observed, 'unreported': count - observed})
            summary = _write_phase_summary(phase_path, phase_id, label, args.games, attempts,
                                           failures, attempts_path, finished=True)
            summary.update(cleanup=cleanup_report, parent_error=parent_error,
                           unreported_episodes=[{'worker_id': wid, 'episode': episode}
                                                for wid, count in assignments.items()
                                                for episode in range(count) if (wid, episode) not in seen],
                           game_failures=sum(row.get('status') == 'error' for row in attempts),
                           max_game_failures=failure_budget, failures_ledger=failures_path,
                           search_route='native' if native_search_enabled() else 'python')
            _publish_phase_summary(phase_path, summary)
        finally:
            for pair in pairs + ([control_pair] if control_pair else []):
                for connection in pair:
                    connection.close()
            out_q.close()
        if cleanup_report is not None and cleanup_report['alive']:
            raise RuntimeError('Served cleanup left live child processes: %s' % cleanup_report['alive'])
    if post_cleanup_error is not None:
        raise post_cleanup_error
    if (cleanup_report['errors'] or cleanup_report['server_exitcode'] != 0
            or any(code != 0 for code in cleanup_report['worker_exitcodes'].values())):
        raise RuntimeError('Served children did not exit cleanly: %s' % cleanup_report)
    _require_batch_complete(attempts, worker_errors, args.games, failure_budget, label, failures_path)
    if stats:
        ai.write_td_window_report(stats, label, report_path)
    game_failures = sum(row.get('status') == 'error' for row in attempts)
    if game_failures:
        print('=== BATCH COMPLETE WITH %d GAME FAILURE(S): %d/%d games completed; fixtures + ledger under %s ==='
              % (game_failures, len(stats), args.games, os.path.dirname(failures_path)), flush=True)
    else:
        print('=== ALL GAMES COMPLETE ===', flush=True)


def _require_batch_complete(attempts, worker_errors, expected, failure_budget, label, failures_path):
    """A complete batch has every assigned episode REPORTED (completed or error), no worker-level
    error and at most `failure_budget` game failures.  Anything else exits 1 as before."""
    completed = sum(row.get('status') == 'completed' for row in attempts)
    game_failures = sum(row.get('status') == 'error' for row in attempts)
    if worker_errors or len(attempts) != expected or game_failures > failure_budget:
        print('=== %s INCOMPLETE: %d/%d games completed, %d game failure(s) (budget %d), %d worker error(s); '
              'ledger %s ===' % (label, completed, expected, game_failures, failure_budget,
                                 len(worker_errors), failures_path), flush=True)
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='xformer')
    ap.add_argument('--fp-net', required=True)
    ap.add_argument('--sh-net', required=True)
    ap.add_argument('--games', type=int, default=500)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--sims', type=int, default=1000)
    ap.add_argument('--seed0', type=int, default=43000)
    ap.add_argument('--shard-id', type=int, default=None,
                    help='Unique nonnegative task ID within this generation; creates a fresh shard directory and disjoint worker seed interval (required under Flux)')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--reuse-completed-manifest')
    ap.add_argument('--reuse-completed-sha256')
    ap.add_argument('--reuse-review')
    ap.add_argument('--reuse-review-sha256')
    run(ap.parse_args())


if __name__ == '__main__':
    main()
