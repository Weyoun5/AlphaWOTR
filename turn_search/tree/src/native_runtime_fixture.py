# -*- coding: ascii -*-
"""Test-only: the environment a native runtime test needs, selected by the test itself (R6-NATIVE-TEST-HARNESS).

The native runtime tests drive a built DLL through native_backend/snapshot.py and compare it with the Python engine.
They depend on two things the plain gate environment (claude_bugscan_2026-10-21/tools/run_gates.py sets only CUDA /
KMP / hash seed / bytecode / io encoding for src tests) does not provide:

  * A reference pin.  bridge.load_reference(), reached by every RNG-backed native transition through
    combat_context.require_native_context, verifies the pin named by WOTR_NATIVE_SEARCH_REFERENCE, else its own
    default native_backend/reference_sep16_own_current_v3_r2.json, which predates this campaign (159 drifted files),
    so the tests skipped on that drift even right after a rebuild and re-pin.  The pin every re-pin rewrites
    (round5/dep1/repin_live.py) is the live box pin native_backend/live_search_oct19/reference_live.json, also the
    default of wotr_bridge/live/native_search.load().  current_reference() names it unless the caller chose a pin.
  * The v52 action generation.  ai reads WOTR_ACT_GEN once, at import, and observation_tables refuses an engine
    imported without the complete v52 contract.  A process that imported ai under another generation cannot switch,
    so PinnedChild runs the class ONCE in a fresh interpreter under the pin's own environment and each test of the
    parent process reports its own outcome from that run (no test is counted twice).
  * Pristine engine callables (R9-PARITY-REST-FUNNEL-POLLUTION).  combat_context refuses every RNG-backed native
    transition while a Python override is active, and the first LiveServer of a process installs the live randomness
    funnels for good.  python_overrides_active() names them; a class that sees one runs in the same PinnedChild.

Every unmet precondition is a loud skip with its reason: a stale build or a drifted pin is the coordinator's rebuild /
re-pin, not a test failure, and it is never a silent pass.  Nothing here writes a file or loads the DLL.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / 'native_backend'
LIVE_PIN = BACKEND / 'live_search_oct19' / 'reference_live.json'
REFERENCE_ENV = 'WOTR_NATIVE_SEARCH_REFERENCE'
CHILD_MARKER = 'NATIVE_RUNTIME_PINNED_CHILD'
GATE_ENV = {'CUDA_VISIBLE_DEVICES': '-1', 'KMP_DUPLICATE_LIB_OK': 'TRUE', 'PYTHONHASHSEED': '0',
            'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONIOENCODING': 'utf-8'}
V52_FLAGS = ('_CARD_CONTEXT_ON', '_COMPANION_CONTEXT_ON', '_FELLOWSHIP_DIE_CONTEXT_ON', '_HUNT_PROTECTION_CONTEXT_ON',
             '_REVEAL_ROUTE_CONTEXT_ON', '_ROUTE_EVENT_CONTEXT_ON', '_BREAKING_CONTEXT_ON', '_COMBAT_ROUND_CONTEXT_ON')
_SENTINEL = 'NATIVE_RUNTIME_CHILD_RESULT '


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def current_reference():
    """The pin a native runtime test verifies against: WOTR_NATIVE_SEARCH_REFERENCE when the caller set it, else the
    live box pin (the one the re-pin recipe maintains)."""
    chosen = os.environ.get(REFERENCE_ENV)
    return Path(chosen).resolve() if chosen else LIVE_PIN


def reference_drift(reference=None):
    """None when every source the pin hashes is unchanged (bridge.load_reference's own source check), else the loud
    reason naming the pin and the first drifted files."""
    reference = Path(reference) if reference else current_reference()
    pin = json.loads(reference.read_text(encoding='utf-8'))
    source = Path(pin['runtime']) / 'src'
    # R7-NATIVE-GATE-SUITE-POST-BUILD: the identities bridge.reference_source_hashes certifies - the runtime projection
    # (runtime_source_sha256) when the pin records one, else every pinned src file - so an edited src TEST file no longer
    # reads as a drifted engine pin here either.
    hashes = pin.get('runtime_source_sha256')
    if hashes is None:
        hashes = pin['source_sha256']
    elif not hashes.get('ai.py') or any(pin['source_sha256'].get(name) != digest for name, digest in hashes.items()):
        return ('native reference pin %s: runtime projection disagrees with the pinned sources; re-pin required'
                % reference)
    drift = [name for name, digest in sorted(hashes.items())
             if not (source / name).is_file() or _sha(source / name) != digest]
    if drift:
        return ('native reference pin %s drifted (%d files, e.g. %s); re-pin required'
                % (reference, len(drift), drift[:3]))
    return None


def main_build_stale(backend=BACKEND):
    """None when the main-tree build manifest passed and its sources and binary are unchanged (bridge.library()'s own
    checks, without loading the DLL), else the loud reason."""
    backend = Path(backend)
    manifest = json.loads((backend / 'build' / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('passed') is not True:
        return 'no successful main-tree native build'
    stale = [name for name, digest in sorted(manifest['sources'].items())
             if not (backend / name).is_file() or _sha(backend / name) != digest]
    if stale:
        return ('main-tree native DLL is stale (%d sources changed since the build, e.g. %s); rebuild required'
                % (len(stale), stale[:3]))
    binary = ((backend / manifest['binary_relative']).resolve() if manifest.get('binary_relative')
              else Path(manifest['binary']).resolve())
    if not binary.is_file() or _sha(binary) != manifest['binary_sha256']:
        return 'main-tree native binary drift: %s' % binary
    return None


def has_v52_contract(ai):
    """True when this process imported ai with the complete v52 contract the native observation tables require."""
    return all(getattr(ai, flag, False) for flag in V52_FLAGS)


# R9-PARITY-REST-FUNNEL-POLLUTION: the live randomness funnels.  The first LiveServer (or src/test_live_driver.Rig) of
# a process installs them for good (wotr_bridge/live/randomness.py ClientRandomness.install: once per process, no
# uninstall, by design), and native_backend/combat_context refuses every RNG-backed native transition while one is
# active ('Native combat refuses active Python override: combat.roll_d6_batch').  A native runtime class in such a
# process (the one-process bridge suite) runs in a fresh PinnedChild, whose engine callables are pristine.  The
# surfaces are the engine-module funnels ClientRandomness.install replaces.
RANDOMNESS_OWNERS = ('live.randomness', 'wotr_bridge.live.randomness')
ENGINE_FUNNELS = (
    ('combat', ('roll_d6_batch', '_apply_pre_combat_for_side', '_apply_post_casualty_for_side', 'apply_onslaught_choice',
                'apply_words_of_power', 'apply_they_are_terrible', 'apply_dice_modifiers', 'apply_mighty_attack',
                'resolve_combat', 'resume_combat', '_emit_round_annotations')),
    ('game_logic', ('roll_d6_batch', 'pick_hunt_tile', 'pop_hunt_tile', 'draw_event_card', 'use_dwarven_ring',
                    'pick_random_companion', 'pick_random_card')),
    ('onslaught_choices', ('_advance',)),
    ('event_effects', ('play_event_card', 'effect_dead_men_of_dunharrow')),
)


def _callable_origin(function):
    if function is None:
        return 'missing'
    return '%s.%s' % (getattr(function, '__module__', None) or '?',
                      getattr(function, '__qualname__', None) or type(function).__name__)


def python_overrides_active(modules=None):
    """None when no live randomness owner (RANDOMNESS_OWNERS) is installed and every ENGINE_FUNNELS surface is its own
    module's plain function, else the loud reason naming each override.  It reads only modules already imported
    (`modules`, default sys.modules: a module never imported carries no override) and never imports, unwraps or
    patches anything (R9-PARITY-REST-FUNNEL-POLLUTION)."""
    modules = sys.modules if modules is None else modules
    found = ['%s._INSTALLED' % name for name in RANDOMNESS_OWNERS
             if modules.get(name) is not None and getattr(modules[name], '_INSTALLED', False)]
    for module_name, names in ENGINE_FUNNELS:
        module = modules.get(module_name)
        if module is None:
            continue
        for name in names:
            function = getattr(module, name, None)
            if not (type(function) is types.FunctionType and function.__globals__ is vars(module)):
                found.append('%s.%s is %s' % (module_name, name, _callable_origin(function)))
    if not found:
        return None
    return ('Python overrides active in this process (native combat refuses them; %d): %s'
            % (len(found), ', '.join(found)))


class ReferenceSelection:
    """WOTR_NATIVE_SEARCH_REFERENCE -> current_reference() for an in-process native class; exit() restores the
    variable exactly (unset stays unset).  Idempotent exit."""

    def __init__(self):
        self.reference = None
        self._saved = None
        self.active = False

    def enter(self):
        if self.active:
            raise RuntimeError('ReferenceSelection.enter() called twice without exit()')
        self._saved = os.environ.get(REFERENCE_ENV)
        self.reference = current_reference()
        os.environ[REFERENCE_ENV] = str(self.reference)
        self.active = True
        return self.reference

    def exit(self):
        if not self.active:
            return
        if self._saved is None:
            os.environ.pop(REFERENCE_ENV, None)
        else:
            os.environ[REFERENCE_ENV] = self._saved
        self.active = False


def pinned_child_env(reference=None):
    """The child's environment: this process's variables minus every WOTR_* (they carry this process's generation),
    plus the gate variables, the pin's own environment, the pin itself and the child marker."""
    reference = Path(reference) if reference else current_reference()
    pin = json.loads(reference.read_text(encoding='utf-8'))
    env = {k: v for k, v in os.environ.items() if not k.startswith('WOTR_')}
    env.update(GATE_ENV)
    env.update(pin['environment'])
    env[REFERENCE_ENV] = str(reference)
    env[CHILD_MARKER] = '1'
    return env


_CHILD_RUNNER = r'''
import json, sys, unittest
name, sentinel = sys.argv[1], sys.argv[2]


class Result(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.outcomes = {}

    def _key(self, test):
        test = getattr(test, 'test_case', test)      # a skip inside subTest() arrives as the _SubTest
        method = getattr(test, '_testMethodName', None)
        if method is None:
            return '__class__'      # setUpClass / tearDownClass / setUpModule (_ErrorHolder)
        return '%s.%s' % (type(test).__name__, method)

    def _put(self, test, status, text):
        key = self._key(test)
        old = self.outcomes.get(key)
        if old is None or old[0] == 'ok' or (old[0] == 'skip' and status != 'ok'):
            self.outcomes[key] = [status, text]
        elif status in ('fail', 'error'):
            old[1] += '\n' + text
            if old[0] == 'skip':
                old[0] = status

    def addSuccess(self, test):
        self._put(test, 'ok', '')

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._put(test, 'fail', self._exc_info_to_string(err, test))

    def addError(self, test, err):
        super().addError(test, err)
        self._put(test, 'error', self._exc_info_to_string(err, test))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._put(test, 'skip', reason)

    def addSubTest(self, test, subtest, err):
        if err is not None:
            status = 'fail' if issubclass(err[0], test.failureException) else 'error'
            self._put(test, status, '%s\n%s' % (subtest, self._exc_info_to_string(err, test)))
        super().addSubTest(test, subtest, err)


suite = unittest.defaultTestLoader.loadTestsFromName(name)
result = Result()
suite.run(result)
print(sentinel + json.dumps({'ran': result.testsRun, 'outcomes': result.outcomes}), flush=True)
'''


class PinnedChild:
    """One fresh interpreter that ran a TestCase class under the pinned environment, and its per-test outcomes.

    run() is called from the class's setUpClass when this process imported ai without the v52 contract (or, for
    R9-PARITY-REST-FUNNEL-POLLUTION, when python_overrides_active() names an override).  It raises a
    loud SkipTest when the main-tree build is stale instead of spawning anything, and never spawns from inside a
    pinned child.  A drifted pin is reported by the child per test (only the tests that reach bridge.load_reference
    skip).  report() is the class's run(): it records the child's outcome of that one test (ok / skip with the
    child's reason / failure with the child's traceback) without running the body in this process, which has the
    wrong generation.
    """

    def __init__(self, outcomes, ran, output):
        self.outcomes = outcomes
        self.ran = ran
        self.output = output

    @classmethod
    def run(cls, case_class, cwd, timeout=600):
        if os.environ.get(CHILD_MARKER):
            raise unittest.SkipTest('the pinned child imported ai without the v52 contract (WOTR_ACT_GEN=%r); '
                                    'the pin environment does not select v52' % os.environ.get('WOTR_ACT_GEN'))
        stale = main_build_stale()
        if stale:
            raise unittest.SkipTest(stale)
        module = Path(sys.modules[case_class.__module__].__file__).stem
        name = '%s.%s' % (module, case_class.__name__)
        try:
            proc = subprocess.run([sys.executable, '-c', _CHILD_RUNNER, name, _SENTINEL], cwd=str(cwd),
                                  env=pinned_child_env(), capture_output=True, text=True, encoding='utf-8',
                                  errors='replace', timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            return cls({}, 0, 'pinned child %s timed out after %ss:\n%s' % (name, timeout, exc.stdout or ''))
        output = (proc.stdout or '') + '\n' + (proc.stderr or '')
        lines = [line for line in (proc.stdout or '').splitlines() if line.startswith(_SENTINEL)]
        if proc.returncode != 0 or not lines:
            return cls({}, 0, 'pinned child %s failed (rc=%s):\n%s' % (name, proc.returncode, output[-6000:]))
        data = json.loads(lines[-1][len(_SENTINEL):])
        return cls(data['outcomes'], data['ran'], output[-6000:])

    def outcome(self, case):
        key = '%s.%s' % (type(case).__name__, case._testMethodName)
        if '__class__' in self.outcomes and self.outcomes['__class__'][0] != 'ok':
            return tuple(self.outcomes['__class__'])
        if key not in self.outcomes:
            return ('error', 'pinned child reported no outcome for %s (ran %d):\n%s' % (key, self.ran, self.output))
        return tuple(self.outcomes[key])

    def report(self, case, result):
        if result is None:
            result = case.defaultTestResult()
        result.startTest(case)
        try:
            status, text = self.outcome(case)
            if status == 'ok':
                result.addSuccess(case)
            elif status == 'skip':
                result.addSkip(case, 'pinned child: ' + text)
            else:
                try:
                    raise case.failureException('pinned child %s (%s):\n%s' % (status, current_reference(), text))
                except case.failureException:
                    result.addFailure(case, sys.exc_info())
        finally:
            result.stopTest(case)
        return result
