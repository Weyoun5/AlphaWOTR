# -*- coding: ascii -*-
"""Audit 2026-08-29 pins - SPC2 family (loop infra).

Pins (finding ids from SPLP_report / SPTR_report / SPCP_report):
  * SPLP-2/SPTR-1/SPCP-4: restart_loop.bat ENV block pins
    WOTR_ACT_GEN=v41, and run_td_v33loop.preflight_generation refuses
    a roster whose action_dim disagrees with the resolved generation
    (fake 92-dim checkpoint in a fake tree must be refused loudly).
  * SPLP-1: WOTR_NATIVE is NOT set by the ENV block (the Aug-14 pyd
    is a stale mirror); fingerprint and marker helpers retain their
    declared format. The separate native-build provenance tests verify
    that main refuses to restamp a stale transcription.
  * SPLP-4: ReplayBuffer.add is atomic (tmp + os.replace); load
    quarantines an unreadable b*.pkl.gz as .corrupt instead of dying.
  * SPLP-5: the kill sweep's root match requires a path separator
    after the escaped root - sibling trees never match.
  * SPLP-7: runner_stdout.log rotates to runner_stdout.prev.log
    before relaunch; a played-but-never-buffered batch gets a loud
    boot line (_unbuffered_last_batch); the next batch index derives
    from max existing + 1 (_next_batch_idx), never len(glob).
  * SPLP-6: play_batch carries a wall cap (BATCH_WALL_SEC) with a
    loud partial-batch line; run_arena_one reports n_played beside
    the parsed score.
  * SPTR-5/SPLP-3 (lead-sanctioned flip 2026-08-29): _resolve_ops
    defaults to the v2 adopted operators; the run-7 legacy v1 ops
    are reachable ONLY via explicit WOTR_OPS_V2=0; either path
    prints one loud line naming the chosen operators.
  * SPTR-6: the v1 delta op counts per-sample failures (first error
    kept) and RAISES when samples were given but zero trained,
    instead of reporting a 0.0 loss as success.
  * SPTR-8 driver half: _log_driver_stats logs-and-resets
    delta_trainer.DRIVER_STATS per batch (wired in train_nets_v2).

Pattern per test_clear_artifacts_keeplist.py / test_launch_head.py:
fake tree + env overrides (WOTR_LOOP_DIR) in a SUBPROCESS so nothing
touches data\\ or any production *.pt; bat behavior is pinned on the
bat TEXT the way td_dashboard._bat_config parses it.
"""
import hashlib
import importlib.util
import inspect
import os
import re
import shutil
import subprocess
import sys
import tempfile

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
RESTART = os.path.join(ROOT, 'scripts', 'restart_loop.bat')
CLEAR = os.path.join(ROOT, 'scripts', 'clear_artifacts.bat')
PYX = os.path.join(SRC, 'wotr_native.pyx')
PY = sys.executable
CREATE_NO_WINDOW = 0x08000000

BOUNDARY = r"[regex]::Escape('%ROOT%') + '(\\|/|$)'"


def _bat_text(path):
    with open(path, encoding='utf-8', errors='replace') as f:
        return f.read()


def _env_block(path):
    """Replica of td_dashboard._bat_config's ENV-block parse (the
    dashboard renders exactly this, so the pin and the GUI read the
    same contract)."""
    out, inside, saw_begin, saw_end = [], False, False, False
    with open(path, encoding='utf-8', errors='replace') as f:
        for ln in f:
            ls = ln.strip()
            if ls.startswith(':: ENV-BEGIN'):
                inside = True
                saw_begin = True
                continue
            if ls.startswith(':: ENV-END'):
                saw_end = True
                break
            if inside and ls.lower().startswith('set "'):
                out.append(ls[5:].rstrip('"'))
    return out, saw_begin, saw_end


def test_env_block_pins(fails):
    """MEASURES: ENV markers survive; WOTR_ACT_GEN=v41 pinned;
    WOTR_NATIVE cleared (never =1) with the fingerprint re-enable
    condition documented."""
    entries, saw_begin, saw_end = _env_block(RESTART)
    if not (saw_begin and saw_end):
        fails.append(('env_markers', (saw_begin, saw_end)))
        return
    if 'WOTR_ACT_GEN=v41' not in entries:
        fails.append(('env_act_gen_v41', entries))
    natives = [e for e in entries if e.startswith('WOTR_NATIVE=')]
    if natives != ['WOTR_NATIVE=']:
        fails.append(('env_native_cleared', natives))
    if 'MIRROR_FINGERPRINT' not in _bat_text(RESTART):
        fails.append(('env_native_reenable_condition_documented', ''))


def test_killsweep_boundary_in_bats(fails):
    """MEASURES: every root-scoped sweep/gate line in both bats builds
    the pattern with a path-separator boundary after the escaped
    root."""
    for tag, path in (('restart', RESTART), ('clear', CLEAR)):
        lines = [ln for ln in _bat_text(path).split('\n')
                 if "[regex]::Escape('%ROOT%')" in ln]
        if not lines:
            fails.append(('%s_sweep_line_missing' % tag, ''))
            continue
        for ln in lines:
            if BOUNDARY not in ln:
                fails.append(('%s_sweep_no_boundary' % tag, ln[:160]))


def test_killsweep_boundary_semantics(fails):
    """MEASURES: the boundary regex (python replica, IGNORECASE like
    PowerShell -match) matches this tree's command lines and NEVER a
    sibling tree's (SPLP-5 demo command lines)."""
    root = r'C:\Users\sarve\Downloads\WOTR_AI_training'
    rx = re.compile(re.escape(root) + r'(\\|/|$)', re.IGNORECASE)
    yes = [
        r'C:\py\python.exe -u '
        r'C:\Users\sarve\Downloads\WOTR_AI_training\src\run_td_v33sp.py',
        r'cmd /c ""C:\py\python.exe" -u "C:\Users\sarve\Downloads'
        r'\WOTR_AI_training\src\run_td_v33loop.py""',
        root,
    ]
    no = [
        r'C:\py\python.exe -u "C:\Users\sarve\Downloads'
        r'\WOTR_AI_training - Copy\src\run_td_v33sp.py"',
        r'C:\py\python.exe C:\Users\sarve\Downloads'
        r'\WOTR_AI_training_lab\src\run_td_v33loop.py',
    ]
    for cl in yes:
        if not rx.search(cl):
            fails.append(('boundary_should_match', cl[-70:]))
    for cl in no:
        if rx.search(cl):
            fails.append(('boundary_matched_sibling', cl[-70:]))


def test_runner_log_rotation(fails):
    """MEASURES: the bat rotates runner_stdout.log to
    runner_stdout.prev.log BEFORE the detached launch (SPLP-7: `>`
    truncated the previous crash's stdout), and the stale smoke-gate
    claim is gone from the final echo."""
    txt = _bat_text(RESTART)
    lines = txt.split('\n')
    rot = [i for i, ln in enumerate(lines)
           if 'runner_stdout.prev.log' in ln and 'move /y' in ln]
    launch = [i for i, ln in enumerate(lines)
              if ln.startswith('start "WOTR_LOOP"')]
    if not rot:
        fails.append(('rotation_line_missing', ''))
    elif not launch:
        fails.append(('launch_line_missing', ''))
    elif min(rot) > min(launch):
        fails.append(('rotation_after_launch', (rot, launch)))
    if 'smoke gate runs first' in txt:
        fails.append(('stale_smoke_gate_echo', ''))


_FAKE_MOD_SRC = (
    "def clone_state(s):\n    return s\n\n\n"
    "def encode_state(s, side):\n    return [0.0]\n\n\n"
    "def _state_sig(s):\n    return 1\n\n\n"
    "def encode_action(a):\n    return [1.0]\n\n\n"
    "def _action_cache_key(a):\n    return 'k'\n"
)


def _load_fake_mod(tmpdir):
    p = os.path.join(tmpdir, 'spc2_fake_mirror_mod.py')
    with open(p, 'w', encoding='ascii') as f:
        f.write(_FAKE_MOD_SRC)
    spec = importlib.util.spec_from_file_location(
        'spc2_fake_mirror_mod', p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mirror_fingerprint_contract(fails):
    """MEASURES: build_wotr_native.mirror_fingerprint = sha1 over the
    concatenated inspect.getsource of MIRROR_FUNCS in declared order
    (the ONE implementation both the build and the ai.py guard use);
    a missing function is a loud RuntimeError, never a silent skip."""
    import build_wotr_native as BWN
    if getattr(BWN, 'MIRROR_FUNCS', None) != (
            'clone_state', 'encode_state', '_state_sig',
            'encode_action', '_action_cache_key'):
        fails.append(('mirror_funcs_tuple',
                      getattr(BWN, 'MIRROR_FUNCS', None)))
        return
    tmpdir = tempfile.mkdtemp(prefix='wotr_spc2_fp_')
    try:
        mod = _load_fake_mod(tmpdir)
        got = BWN.mirror_fingerprint(mod)
        h = hashlib.sha1()
        for name in BWN.MIRROR_FUNCS:
            h.update(inspect.getsource(getattr(mod, name))
                     .encode('utf-8'))
        want = h.hexdigest()
        if got != want:
            fails.append(('fingerprint_algo', (got, want)))
        import types
        partial = types.SimpleNamespace(**{
            k: getattr(mod, k) for k in BWN.MIRROR_FUNCS[:-1]})
        try:
            BWN.mirror_fingerprint(partial)
            fails.append(('fingerprint_missing_func_silent', ''))
        except RuntimeError:
            pass
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_pyx_injection(fails):
    """MEASURES: inject_fingerprint bakes exactly one marker-
    delimited MIRROR_FINGERPRINT block into a pyx COPY, is idempotent
    on rebuild (replaces, never duplicates), preserves CRLF and the
    cython directive header."""
    import build_wotr_native as BWN
    if not hasattr(BWN, 'inject_fingerprint'):
        fails.append(('inject_fingerprint_missing', ''))
        return
    tmpdir = tempfile.mkdtemp(prefix='wotr_spc2_pyx_')
    try:
        cp = os.path.join(tmpdir, 'wotr_native.pyx')
        shutil.copy2(PYX, cp)
        with open(PYX, 'rb') as f:
            first = f.readline()
        BWN.inject_fingerprint(cp, 'a' * 40)
        BWN.inject_fingerprint(cp, 'b' * 40)
        with open(cp, 'rb') as f:
            data = f.read()
        txt = data.decode('ascii')
        fps = re.findall(r"MIRROR_FINGERPRINT = '([0-9a-f]{40})'", txt)
        if fps != ['b' * 40]:
            fails.append(('inject_not_idempotent', fps))
        if b'\r\nMIRROR_FINGERPRINT' not in data:
            fails.append(('inject_crlf_lost', data[-120:]))
        with open(cp, 'rb') as f:
            if f.readline() != first:
                fails.append(('inject_header_disturbed', ''))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_build_dir_guard(fails):
    """MEASURES: the rmtree target is refuse-if-unexpected (USER
    2026-08-22 no-destructive-patterns): only <src>\\build_native
    passes; anything else raises."""
    import build_wotr_native as BWN
    if not hasattr(BWN, '_check_build_dir'):
        fails.append(('build_dir_guard_missing', ''))
        return
    BWN._check_build_dir(os.path.join(SRC, 'build_native'))
    for bad in (os.path.join(SRC, 'build_other'),
                os.path.join(ROOT, 'build_native'),
                tempfile.gettempdir()):
        try:
            BWN._check_build_dir(bad)
            fails.append(('build_dir_guard_accepted', bad))
        except SystemExit:
            pass


# ---- fake-tree subprocess battery (everything that imports
# run_td_v33loop runs with WOTR_LOOP_DIR pointing at scratch, so no
# check can ever touch data\v33loop or a production *.pt) ----------

_SUB_SCRIPT = '''
import os, sys
os.environ['WOTR_REWARD_CONTRACT'] = 'legacy_profile_v1'
fake_loop = os.environ['WOTR_LOOP_DIR']
sys.path.insert(0, sys.argv[1])
res = []


def check(name, fn):
    try:
        fn()
        res.append('CHECK %s PASS' % name)
    except BaseException as e:
        res.append('CHECK %s FAIL %s: %s'
                   % (name, type(e).__name__, e))


import run_td_v33loop as L
KEY = L.NET_KEYS[0]
BUFFER_CONTEXT = {
    'reward_contract': 'legacy_profile_v1',
    'observation_contract': 'v44_fellowship_tail_v1',
    'ai_policy_contract': 'elite_and_bonus_draw_v1',
    'action_encoding_contract': 'casualty_operations_v2'}


class Boom(object):
    def __reduce__(self):
        raise RuntimeError('boom')


def c_buffer_atomic():
    broot = os.path.join(fake_loop, 'buffer_at')
    buf = L.ReplayBuffer(broot, [KEY])
    try:
        buf.add(2, {KEY: dict(BUFFER_CONTEXT, value=[Boom()], delta=[])})
        raise AssertionError('add did not raise')
    except RuntimeError as error:
        assert str(error) == 'boom', error
    final = os.path.join(broot, '%s_%s' % KEY, 'b002.pkl.gz')
    assert not os.path.exists(final), (
        'partial buffer member left behind: %s' % final)


def c_buffer_quarantine():
    broot = os.path.join(fake_loop, 'buffer_q')
    buf = L.ReplayBuffer(broot, [KEY])
    p = os.path.join(buf._dir(KEY), 'b001.pkl.gz')
    with open(p, 'wb') as f:
        f.write(b'not a gzip member')
    out = buf.load(KEY)
    assert out == dict(BUFFER_CONTEXT, value=[], delta=[]), out
    assert not os.path.exists(p), 'corrupt member still in the glob'
    assert os.path.exists(p + '.corrupt'), 'no .corrupt quarantine'


def c_buffer_roundtrip():
    broot = os.path.join(fake_loop, 'buffer_rt')
    buf = L.ReplayBuffer(broot, [KEY])
    buf.add(0, {KEY: dict(BUFFER_CONTEXT, value=[('a', 1)], delta=[{'x': 1}])})
    final = os.path.join(broot, '%s_%s' % KEY, 'b000.pkl.gz')
    assert os.path.exists(final), 'member not written'
    assert not os.path.exists(final + '.tmp'), 'tmp left behind'
    out = buf.load(KEY)
    assert out['value'] == [('a', 1)], out
    assert out['delta'] == [{'x': 1}], out


def c_unbuffered_batch():
    bdir = os.path.join(fake_loop, 'batch_000')
    os.makedirs(bdir, exist_ok=True)
    got = L._unbuffered_last_batch([bdir])
    assert got == 0, 'expected 0 (unbuffered), got %r' % (got,)
    bp = os.path.join(fake_loop, 'buffer', '%s_%s' % KEY)
    os.makedirs(bp, exist_ok=True)
    with open(os.path.join(bp, 'b000.pkl.gz'), 'wb') as f:
        f.write(b'x')
    got = L._unbuffered_last_batch([bdir])
    assert got is None, 'expected None (buffered), got %r' % (got,)


def _fake_ckpt(dim):
    import torch
    ck = {'model_state': {}, 'input_dim': 3742, 'action_dim': dim,
          'head_act': 'scaled2', 'side': 'fp'}
    torch.save(ck, os.path.join(fake_loop, 'live_fp_ring.pt'))


def c_preflight_green_93():
    _fake_ckpt(93)
    L.preflight_generation()


def c_preflight_red_92():
    _fake_ckpt(92)
    try:
        L.preflight_generation()
    except SystemExit as e:
        msg = str(e)
        assert 'WOTR_ACT_GEN' in msg, msg
        assert '93' in msg and '92' in msg, msg
        assert 'live_fp_ring.pt' in msg, msg
        assert 're-seed' in msg.lower(), msg
        return
    raise AssertionError('preflight ACCEPTED a 92-dim roster under '
                         'the v41 generation')


def c_v1_zero_sample_raises():
    import torch

    class FakeNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.policy_delta = torch.nn.Linear(2, 1)

    net = FakeNet()
    # every sample errors inside the per-sample try (enc None) -- the
    # op must RAISE (count + first error), never report 0.0 loss
    samples = [(None, {}), (None, {})]
    try:
        L._delta_update_op(net, samples, 1e-4)
    except RuntimeError as e:
        msg = str(e)
        assert '0' in msg and 'error' in msg.lower(), msg
        assert 'first' in msg.lower(), msg
        return
    raise AssertionError('v1 delta op trained 0/2 samples and did '
                         'not raise')


def c_driver_stats_line():
    import contextlib
    import inspect
    import io
    import delta_trainer
    st = getattr(delta_trainer, 'DRIVER_STATS', None)
    assert isinstance(st, dict) and st, 'no DRIVER_STATS contract'
    for k in st:
        st[k] = 0
    k0 = sorted(st)[0]
    st[k0] = 7
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        line = L._log_driver_stats()
    assert line and 'DRIVER_STATS' in line, repr(line)
    assert ('%s=7' % k0) in line, line
    assert all(v == 0 for v in st.values()), st
    assert 'DRIVER_STATS' in buf.getvalue(), 'not printed'
    assert '_log_driver_stats' in inspect.getsource(L.train_nets_v2), \
        'train_nets_v2 does not log DRIVER_STATS'


def c_v1_declare_flag_twin():
    import inspect
    import delta_trainer
    # SPB 1b2: the v1 collector's inline flag twin must carry
    # declare_decline under _declare_phase like the adopted
    # delta_trainer._flag_for -- and the adopted contract itself is
    # pinned here so the twin cannot silently diverge either way
    src = inspect.getsource(L._collect_search_samples)
    assert "'declare_decline'" in src, \
        'v1 collector skew-drops declare_decline rows (no flag)'
    for at, want in (('fellowship_declare', '_declare_phase'),
                     ('declare_decline', '_declare_phase'),
                     ('eye_allocate', '_eye_allocation_phase'),
                     ('army_move', None)):
        got = delta_trainer._flag_for(at)
        assert got == want, (at, got, want)


def c_next_batch_idx():
    base = os.path.join(fake_loop, 'idxcase')
    done = []
    for nm in ('batch_000', 'batch_003', 'batch_005_quarantined_x'):
        d = os.path.join(base, nm)
        os.makedirs(d, exist_ok=True)
        done.append(d)
    got = L._next_batch_idx(sorted(done))
    assert got == 6, 'expected 6 (max+1), got %r' % (got,)
    assert L._next_batch_idx([]) == 0, 'empty tree must start at 0'


def c_splp6_and_wiring():
    import inspect
    assert isinstance(L.BATCH_WALL_SEC, int) and \
        L.BATCH_WALL_SEC >= 3600, 'no sane BATCH_WALL_SEC'
    src = inspect.getsource(L.play_batch)
    assert 'BATCH_WALL_SEC' in src, 'play_batch has no wall cap'
    asrc = inspect.getsource(L.run_arena_one)
    assert 'n_played' in asrc, 'run_arena_one lacks n_played report'
    msrc = inspect.getsource(L.main_v2)
    assert '_next_batch_idx' in msrc, 'main_v2 not on _next_batch_idx'
    assert '_unbuffered_last_batch' in msrc, \
        'main_v2 lacks the unbuffered-batch loud line'
    assert 'preflight_generation' in msrc, \
        'main_v2 skips the generation preflight'
    ssrc = inspect.getsource(L._smoke_v2)
    assert 'preflight_generation' in ssrc, \
        '_smoke_v2 skips the generation preflight'


for nm, fn in (('buffer_atomic', c_buffer_atomic),
               ('buffer_quarantine', c_buffer_quarantine),
               ('buffer_roundtrip', c_buffer_roundtrip),
               ('unbuffered_batch', c_unbuffered_batch),
               ('v1_zero_sample_raises', c_v1_zero_sample_raises),
               ('driver_stats_line', c_driver_stats_line),
               ('v1_declare_flag_twin', c_v1_declare_flag_twin),
               ('next_batch_idx', c_next_batch_idx),
               ('splp6_and_wiring', c_splp6_and_wiring),
               ('preflight_green_93', c_preflight_green_93),
               ('preflight_red_92', c_preflight_red_92)):
    check(nm, fn)
print(chr(10).join(res))
'''

_SUB_CHECKS = ('buffer_atomic', 'buffer_quarantine',
               'buffer_roundtrip', 'unbuffered_batch',
               'v1_zero_sample_raises', 'driver_stats_line',
               'v1_declare_flag_twin',
               'next_batch_idx', 'splp6_and_wiring',
               'preflight_green_93', 'preflight_red_92')


# ---- ops-resolution pins (SPTR-5/SPLP-3 flip): each case needs its
# own interpreter because _resolve_ops reads the env once at call
# time and the chosen ops are module globals -------------------------

_OPS_SCRIPT = '''
import contextlib
import io
import os
import sys
sys.path.insert(0, sys.argv[1])
import run_td_v33loop as L
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    L._resolve_ops()
out = buf.getvalue()
import delta_trainer
import value_trainer_v2
v2 = (L.DELTA_COLLECTOR is delta_trainer.collect_for_driver
      and L.DELTA_UPDATE_OP is delta_trainer.delta_update_for_driver
      and L.VALUE_UPDATE_OP is value_trainer_v2.value_update_for_driver)
v1 = (L.DELTA_COLLECTOR is L._collect_search_samples
      and L.DELTA_UPDATE_OP is L._delta_update_op
      and L.VALUE_UPDATE_OP is L._value_update_op)
print('RESOLVED ' + ('v2' if v2 else ('v1' if v1 else 'MIXED')))
print('LINE ' + repr(out))
'''


def test_ops_resolution(fails):
    """MEASURES (two subprocesses, fake tree): an UNSET WOTR_OPS_V2
    resolves to the v2 adopted operators with a loud line; explicit
    WOTR_OPS_V2=0 resolves to the legacy v1 operators with a loud
    LEGACY notice.  Neither path is silent."""
    tmpdir = tempfile.mkdtemp(prefix='wotr_spc2_ops_')
    try:
        fake_loop = os.path.join(tmpdir, 'data', 'v33loop')
        os.makedirs(fake_loop, exist_ok=True)
        script = os.path.join(tmpdir, 'spc2_ops.py')
        with open(script, 'w', encoding='ascii') as f:
            f.write(_OPS_SCRIPT)
        base = dict(os.environ,
                    WOTR_LOOP_DIR=fake_loop,
                    WOTR_ACT_GEN='v41',
                    KMP_DUPLICATE_LIB_OK='TRUE',
                    PYTHONHASHSEED='0',
                    CUDA_VISIBLE_DEVICES='-1')
        base.pop('WOTR_OPS_V2', None)
        for case, envval, want_res, want_in_line in (
                ('unset_defaults_v2', None, 'RESOLVED v2', 'v2'),
                ('explicit0_legacy_v1', '0', 'RESOLVED v1',
                 'LEGACY')):
            env = dict(base)
            if envval is not None:
                env['WOTR_OPS_V2'] = envval
            r = subprocess.run([PY, '-u', script, SRC],
                               capture_output=True, text=True,
                               env=env, timeout=600, cwd=SRC,
                               creationflags=CREATE_NO_WINDOW)
            out = (r.stdout or '') + (r.stderr or '')
            if want_res not in out:
                fails.append(('ops_%s_resolved' % case,
                              out[-300:].strip()))
                continue
            line = next((ln for ln in out.split('\n')
                         if ln.startswith('LINE ')), 'LINE (none)')
            if 'OPS' not in line or want_in_line not in line:
                fails.append(('ops_%s_loud_line' % case,
                              line[:200]))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_loop_fake_tree(fails):
    """MEASURES (subprocess, fake tree): SPLP-4 buffer atomicity +
    quarantine; SPLP-7 unbuffered-batch detection; the generation
    preflight passes a 93-dim roster and refuses a 92-dim roster
    under v41 with the exact facts + re-seed requirement in the
    message."""
    tmpdir = tempfile.mkdtemp(prefix='wotr_spc2_tree_')
    try:
        fake_loop = os.path.join(tmpdir, 'data', 'v33loop')
        os.makedirs(fake_loop, exist_ok=True)
        script = os.path.join(tmpdir, 'spc2_sub.py')
        with open(script, 'w', encoding='ascii') as f:
            f.write(_SUB_SCRIPT)
        env = dict(os.environ,
                   WOTR_LOOP_DIR=fake_loop,
                   WOTR_ACT_GEN='v41',
                   KMP_DUPLICATE_LIB_OK='TRUE',
                   PYTHONHASHSEED='0',
                   CUDA_VISIBLE_DEVICES='-1')
        r = subprocess.run([PY, '-u', script, SRC],
                           capture_output=True, text=True, env=env,
                           timeout=600, cwd=SRC,
                           creationflags=CREATE_NO_WINDOW)
        out = (r.stdout or '') + (r.stderr or '')
        for name in _SUB_CHECKS:
            if ('CHECK %s PASS' % name) not in out:
                line = next((ln for ln in out.split('\n')
                             if ('CHECK %s ' % name) in ln),
                            '(no CHECK line; rc=%s tail=%r)'
                            % (r.returncode, out[-400:]))
                fails.append(('sub_%s' % name, line.strip()))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    tests = (test_env_block_pins,
             test_killsweep_boundary_in_bats,
             test_killsweep_boundary_semantics,
             test_runner_log_rotation,
             test_mirror_fingerprint_contract,
             test_pyx_injection,
             test_build_dir_guard,
             test_loop_fake_tree,
             test_ops_resolution)
    total, failed = 0, 0
    for t in tests:
        fails = []
        try:
            t(fails)
        except Exception as e:
            fails.append(('%s_raised' % t.__name__,
                          '%s: %s' % (type(e).__name__, e)))
        total += 1
        if fails:
            failed += 1
            for name, detail in fails:
                print('FAIL %s :: %s :: %r'
                      % (t.__name__, name, detail))
        else:
            print('PASS %s' % t.__name__)
    print('TOTAL=%d FAIL=%d' % (total, failed))
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
