# -*- coding: ascii -*-
"""Regression pins for the 2026-08-29 SPD fix wave: amendment 1.259,
the REAFFIRMED 30%% frozen-opponent anchor (disabled-loud by default).

Contract pinned here:
  * loop_config.json keys frozen_opponent_p / frozen_ckpt_fp /
    frozen_ckpt_sh (pair-validated; wrong-generation frozen nets are
    refused at BOOT by the SPC2 generation preflight).
  * run_td_v33sp selects the frozen side per episode from a dedicated
    worker-seeded RNG (_frozen_rng + _frozen_pick; the pin replays the
    exact seeded calls, never statistics).
  * The episode stamps 'frozen_side' into the trace meta, every trace
    row of the frozen side, the returned trajectory (retired
    traj['frozen_side'] contract) and game_stats.
  * EVERY collector (delta_trainer.collect_for_driver and the legacy
    run_td_v33loop._collect_search_samples twin) drops frozen-side
    rows, counted as DRIVER_STATS['frozen_dropped'] (loud in the
    per-batch DRIVER_STATS: line); the value path excludes the frozen
    side's sequences while the live side's rows still train.
  * Unset config -> ONE loud DISABLED line per batch, zero frozen
    episodes; set config -> frozen_episodes=N/M side_split telemetry.

Runs standalone: prints TOTAL=<n> FAIL=<m>, exit 1 on any failure.
"""
import contextlib
import glob
import gzip
import io
import json
import os
import pickle
import sys
import tempfile
import time
import traceback

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('WOTR_ACT_GEN', 'v41')
os.environ.setdefault('PYTHONHASHSEED', '0')

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)

SCRATCH = tempfile.mkdtemp(prefix='spd_pins_')
# the loop module binds LOOP from env at import: point it at a fake
# tree BEFORE the first import so no pin can touch data\v33loop
os.environ['WOTR_LOOP_DIR'] = os.path.join(SCRATCH, 'fake_loop')
os.makedirs(os.environ['WOTR_LOOP_DIR'], exist_ok=True)

_FROZEN_ENV_KEYS = ('WOTR_SP_FROZEN_P', 'WOTR_SP_FROZEN_FP',
                    'WOTR_SP_FROZEN_SH')

_DISABLED_LINE = ('FROZEN-OPPONENT ANCHOR DISABLED: no frozen '
                  'checkpoint configured (1.259; intended anchor = '
                  'the v42 BC baseline)')

_EP_CACHE = {}


def _set_frozen_env(p=None, fp=None, sh=None):
    old = {k: os.environ.get(k) for k in _FROZEN_ENV_KEYS}
    for k, v in zip(_FROZEN_ENV_KEYS, (p, fp, sh)):
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)
    return old


def _restore_env(old):
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


class _Q(object):
    def __init__(self):
        self.msgs = []

    def put(self, m):
        self.msgs.append(m)


class _FakeNet(object):
    def __init__(self, path):
        self.src_path = path

    def eval(self):
        return self


def _run_stub_worker(seed0, n_games, env_kw):
    """Run run_td_v33sp._worker with a stubbed loader + episode; returns
    the recorded (fp_src, sh_src, config, profiles) per game."""
    import ai
    import run_td_v33sp as sp
    calls = []

    def fake_load(path, tag):
        return _FakeNet(path)

    def fake_ep(fpn, shn, config):
        calls.append((getattr(fpn, 'src_path', '?'),
                      getattr(shn, 'src_path', '?'),
                      dict(config), dict(ai.PLAYER_PROFILES)))
        return {'game_stats': {'winner': 'fp', 'turns': 1,
                               'trace_path': config.get('trace_path'),
                               'frozen_side':
                                   config.get('frozen_side')}}

    old_load = sp._load_net
    old_ep = ai._run_single_episode
    old_env = _set_frozen_env(**env_kw)
    q = _Q()
    try:
        sp._load_net = fake_load
        ai._run_single_episode = fake_ep
        sp._worker(0, n_games, 1, os.path.join(SCRATCH, 'trc'),
                   seed0, 'live_fp.pt', 'live_sh.pt', q, net_dir=None)
    finally:
        sp._load_net = old_load
        ai._run_single_episode = old_ep
        _restore_env(old_env)
        import ai as _ai9
        _ai9.PLAYER_PROFILES['fp'] = None
        _ai9.PLAYER_PROFILES['sh'] = None
    errs = [m for m in q.msgs if m[0] != 'g']
    assert not errs, 'stub worker reported errors: %s' % errs[:2]
    assert len(calls) == n_games, \
        'stub worker played %d/%d games' % (len(calls), n_games)
    return calls


def test_spd1_selection_replays_worker_seeded_rng():
    """1.259 pin (a) selection: the per-episode frozen pick replays the
    dedicated worker-seeded RNG exactly; the frozen side's net is the
    frozen checkpoint's; unset env = zero frozen and an unperturbed
    profile stream."""
    import run_td_v33sp as sp
    assert hasattr(sp, '_frozen_pick') and hasattr(sp, '_frozen_rng'), \
        'SPD: run_td_v33sp has no _frozen_pick/_frozen_rng ' \
        '(1.259 selection missing)'
    frz_fp = os.path.join(SCRATCH, 'frozen_fp.pt')
    frz_sh = os.path.join(SCRATCH, 'frozen_sh.pt')

    # leg 1: p=0.30, seeded -- expected picks come from replaying the
    # SAME rng calls (never statistics)
    calls = _run_stub_worker(777, 4, dict(p='0.30', fp=frz_fp,
                                          sh=frz_sh))
    exp_rng = sp._frozen_rng(777, 0)
    expected = [sp._frozen_pick(exp_rng, 0.30) for _ in range(4)]
    got = [c[2].get('frozen_side') for c in calls]
    assert got == expected, \
        'SPD: frozen selection diverges from the seeded RNG replay: ' \
        'got %s want %s' % (got, expected)
    assert any(e is not None for e in expected) \
        and any(e is None for e in expected), \
        'pin seed must exercise both frozen and live episodes at ' \
        'p=0.30; got %s (adjust the pin seed)' % (expected,)
    for (fp_src, sh_src, cfg, _pr), e in zip(calls, expected):
        if e == 'fp':
            assert fp_src == frz_fp and sh_src == 'live_sh.pt', \
                'frozen fp episode played nets (%s, %s)' % (fp_src,
                                                            sh_src)
        elif e == 'sh':
            assert sh_src == frz_sh and fp_src == 'live_fp.pt', \
                'frozen sh episode played nets (%s, %s)' % (fp_src,
                                                            sh_src)
        else:
            assert fp_src == 'live_fp.pt' and sh_src == 'live_sh.pt', \
                'live episode played nets (%s, %s)' % (fp_src, sh_src)

    # leg 2: p=1.0 -- every episode frozen, side uniform from the rng
    calls2 = _run_stub_worker(778, 4, dict(p='1.0', fp=frz_fp,
                                           sh=frz_sh))
    exp_rng2 = sp._frozen_rng(778, 0)
    expected2 = [sp._frozen_pick(exp_rng2, 1.0) for _ in range(4)]
    got2 = [c[2].get('frozen_side') for c in calls2]
    assert got2 == expected2 and all(e in ('fp', 'sh')
                                     for e in got2), \
        'p=1.0 leg: got %s want %s' % (got2, expected2)

    # leg 3 (pin c half): unset env -> zero frozen episodes AND the
    # profile stream is identical to the enabled leg (dedicated rng)
    calls3 = _run_stub_worker(777, 4, dict())
    got3 = [c[2].get('frozen_side') for c in calls3]
    assert got3 == [None] * 4, \
        'unset config must run 100%% live; got %s' % (got3,)
    profs_enabled = [c[3] for c in calls]
    profs_disabled = [c[3] for c in calls3]
    assert profs_enabled == profs_disabled, \
        'SPD: enabling the anchor perturbed the profile RNG stream: ' \
        '%s vs %s' % (profs_enabled, profs_disabled)


def test_spd2_frozen_env_pair_validation():
    """A half-configured frozen pair must refuse loudly (RuntimeError:
    worker-catchable, reported via out_q), never run half-frozen."""
    import run_td_v33sp as sp
    assert hasattr(sp, '_frozen_env'), \
        'SPD: run_td_v33sp has no _frozen_env (1.259 config missing)'
    old = _set_frozen_env(fp=os.path.join(SCRATCH, 'only_fp.pt'))
    try:
        try:
            sp._frozen_env()
            raise AssertionError(
                'SPD: half-configured frozen pair was accepted')
        except RuntimeError:
            pass
    finally:
        _restore_env(old)
    old = _set_frozen_env(p='0.25', fp='a.pt', sh='b.pt')
    try:
        assert sp._frozen_env() == (0.25, 'a.pt', 'b.pt')
    finally:
        _restore_env(old)
    old = _set_frozen_env()
    try:
        assert sp._frozen_env() is None
    finally:
        _restore_env(old)


def _ensure_probe_episode():
    """One tiny REAL episode with frozen_side='fp' (cached).  Returns
    (resolved_trace_path, loaded_trace, traj)."""
    if 'tr' in _EP_CACHE:
        return _EP_CACHE['path'], _EP_CACHE['tr'], _EP_CACHE['traj']
    import random

    import numpy as np
    import torch

    import ai
    import game_trace
    random.seed(4242)
    np.random.seed(4242)
    torch.manual_seed(4242)
    fpn = ai.WOTRValueNet(ai.get_feature_dim(), side='fp').eval()
    shn = ai.WOTRValueNet(ai.get_feature_dim(), side='sh').eval()
    ai.PLAYER_PROFILES['fp'] = 'ring'
    ai.PLAYER_PROFILES['sh'] = 'military'
    tp = os.path.join(SCRATCH, 'spd_probe.trace.gz')
    try:
        traj = ai._run_single_episode(
            fpn, shn, {'max_turns': 3, 'mcts_sims': 8, 'shadow_vp': 10,
                       'trace_path': tp, 'frozen_side': 'fp'})
    finally:
        ai.PLAYER_PROFILES['fp'] = None
        ai.PLAYER_PROFILES['sh'] = None
    real = ai.resolve_trace_path(tp)
    assert real and os.path.exists(real), \
        'probe episode saved no trace (config %s)' % tp
    tr = game_trace.load_trace(real)
    _EP_CACHE.update({'path': real, 'tr': tr, 'traj': traj})
    return real, tr, traj


def _expected_qualifying_rows(tr, tag):
    """Replicates the collectors' precondition (search_stats present,
    side row, K>=2, a previous snapshot exists)."""
    want_side = 'FP' if tag == 'fp' else 'SH'
    exp = 0
    prev = None
    for s in tr['steps']:
        st = s.get('state')
        if st is None:
            continue
        ss = s.get('search_stats')
        if (ss is not None and s.get('side') == want_side
                and prev is not None and len(ss.get('p', ())) >= 2):
            exp += 1
        prev = st
    return exp


def test_spd3_episode_stamps_frozen_side():
    """1.259 pin (a) stamps: trace meta + every frozen-side trace row +
    traj (retired contract) + game_stats all carry frozen_side."""
    _real, tr, traj = _ensure_probe_episode()
    assert traj.get('frozen_side') == 'fp', \
        'SPD: traj carries no frozen_side (retired traj contract): %r' \
        % (traj.get('frozen_side'),)
    gs = traj.get('game_stats') or {}
    assert gs.get('frozen_side') == 'fp', \
        'SPD: game_stats carries no frozen_side: %r' \
        % (gs.get('frozen_side'),)
    assert (tr.get('meta') or {}).get('frozen_side') == 'fp', \
        'SPD: trace meta carries no frozen_side stamp: %r' \
        % ((tr.get('meta') or {}).get('frozen_side'),)
    fp_rows = [s for s in tr['steps'] if s.get('side') == 'FP']
    assert fp_rows, 'probe episode produced no FP rows'
    unstamped = [s.get('action_type') for s in fp_rows
                 if s.get('frozen_side') != 'fp']
    assert not unstamped, \
        'SPD: FP trace rows missing the frozen_side stamp: %s' \
        % unstamped[:6]
    sh_bad = [s.get('action_type') for s in tr['steps']
              if s.get('side') == 'SH' and s.get('frozen_side')]
    assert not sh_bad, \
        'live-side rows must NOT carry the stamp: %s' % sh_bad[:6]


def test_spd4_collectors_drop_frozen_rows():
    """1.259 pin (a) drop + (b) live rows: BOTH collectors drop every
    frozen-side row (counted as frozen_dropped, loud in the
    DRIVER_STATS: line) while the live side's rows still collect."""
    _real, tr, _traj = _ensure_probe_episode()
    import delta_trainer as dt
    import run_td_v33loop as rl
    assert 'frozen_dropped' in dt.DRIVER_STATS, \
        'SPD: DRIVER_STATS has no frozen_dropped counter'
    exp = _expected_qualifying_rows(tr, 'fp')
    assert exp >= 1, 'probe episode has no qualifying FP search rows'
    for k in dt.DRIVER_STATS:
        dt.DRIVER_STATS[k] = 0
    with contextlib.redirect_stdout(io.StringIO()):
        rows_fp = dt.collect_for_driver(tr, 'fp')
    assert len(rows_fp) == 0, \
        'SPD: %d frozen-side rows entered training via ' \
        'collect_for_driver' % len(rows_fp)
    assert dt.DRIVER_STATS['frozen_dropped'] == exp, \
        'SPD: frozen_dropped=%s, want %d (every frozen row counted)' \
        % (dt.DRIVER_STATS['frozen_dropped'], exp)
    with contextlib.redirect_stdout(io.StringIO()):
        rows_sh = dt.collect_for_driver(tr, 'sh')
    assert len(rows_sh) >= 1, \
        'SPD(b): live-side rows must still train (0 collected)'
    # legacy v1 twin
    dt.DRIVER_STATS['frozen_dropped'] = 0
    v1_fp = rl._collect_search_samples(tr, 'fp')
    assert v1_fp == [], \
        'SPD: v1 twin collected %d frozen-side rows' % len(v1_fp)
    assert dt.DRIVER_STATS['frozen_dropped'] == exp, \
        'SPD: v1 twin frozen_dropped=%s, want %d' \
        % (dt.DRIVER_STATS['frozen_dropped'], exp)
    v1_sh = rl._collect_search_samples(tr, 'sh')
    assert len(v1_sh) >= 1, 'v1 twin lost the live side'
    # the counter is loud in the per-batch DRIVER_STATS: line
    for k in dt.DRIVER_STATS:
        dt.DRIVER_STATS[k] = 0
    dt.DRIVER_STATS['frozen_dropped'] = 7
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        line = rl._log_driver_stats()
    assert line and 'frozen_dropped=7' in line, \
        'SPD: DRIVER_STATS line hides frozen_dropped: %r' % (line,)
    for k in dt.DRIVER_STATS:
        dt.DRIVER_STATS[k] = 0


def test_spd5_timeout_episode_is_not_training_data():
    """The real frozen-opponent probe is an intact timeout, not a decided
    game. Admission must exclude both sides until bootstrap-aware truncated
    training exists. Completed live/frozen cohort coverage is maintained in
    test_frozen_completed_admission.py, using a declared provenance overlay.
    """
    _real, tr, _traj = _ensure_probe_episode()
    import run_td_v33loop as rl
    import shutil
    assert tr['meta']['winner'] == 'timeout'
    assert tr['meta']['victory_type'] == 'timeout'
    assert tr['meta']['frozen_side'] == 'fp'
    bdir = os.path.join(SCRATCH, 'batch_spd5')
    os.makedirs(bdir, exist_ok=True)
    # Preserve the episode bytes and terminal provenance exactly.
    shutil.copyfile(_real, os.path.join(bdir, 'w0_g0_fp_ring.trace.gz'))
    import delta_trainer as dt
    for key in dt.DRIVER_STATS:
        dt.DRIVER_STATS[key] = 0
    with contextlib.redirect_stdout(io.StringIO()):
        rl._resolve_ops()
        out = rl.extract_batch_streams(bdir)
    with open(os.path.join(bdir, 'training_admission.json'), encoding='utf-8') as stream:
        report = json.load(stream)
    assert report['accepted'] == 0 and report['rejected'] == 1, report
    trace = report['traces'][0]
    assert trace['valid'] and not trace['accepted'], trace
    assert [error['code'] for error in trace['errors']] == ['truncated'], trace
    assert trace['decisions']['fp'] > 0 and trace['decisions']['sh'] > 0, trace
    for key, data in out.items():
        assert data['value'] == [] and data['delta'] == [], (key, data)
    assert dt.DRIVER_STATS['rows'] == 0, dt.DRIVER_STATS



def test_spd6_loop_config_keys_and_pair_validation():
    """loop_config.json accepts frozen_opponent_p / frozen_ckpt_fp /
    frozen_ckpt_sh; a half-configured pair refuses loudly."""
    import run_td_v33loop as rl
    for name in ('FROZEN_OPPONENT_P', 'FROZEN_CKPT_FP',
                 'FROZEN_CKPT_SH'):
        assert hasattr(rl, name), \
            'SPD: run_td_v33loop has no %s (1.259 config missing)' \
            % name
    cfgp = os.path.join(rl.LOOP, 'loop_config.json')
    os.makedirs(rl.LOOP, exist_ok=True)
    old = (rl.FROZEN_OPPONENT_P, rl.FROZEN_CKPT_FP, rl.FROZEN_CKPT_SH)
    try:
        with open(cfgp, 'w', encoding='utf-8') as f:
            json.dump({'frozen_opponent_p': 0.25,
                       'frozen_ckpt_fp': 'x_fp.pt',
                       'frozen_ckpt_sh': 'x_sh.pt'}, f)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rl._apply_loop_config()
        except SystemExit as e:
            raise AssertionError(
                'SPD: loop_config.json frozen keys refused: %s' % e)
        assert rl.FROZEN_OPPONENT_P == 0.25, rl.FROZEN_OPPONENT_P
        assert rl.FROZEN_CKPT_FP == 'x_fp.pt', rl.FROZEN_CKPT_FP
        assert rl.FROZEN_CKPT_SH == 'x_sh.pt', rl.FROZEN_CKPT_SH
        with open(cfgp, 'w', encoding='utf-8') as f:
            json.dump({'frozen_ckpt_fp': 'only_fp.pt'}, f)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rl._apply_loop_config()
            raise AssertionError(
                'SPD: half-configured frozen pair was accepted by '
                '_apply_loop_config')
        except SystemExit:
            pass
    finally:
        try:
            os.remove(cfgp)
        except OSError:
            pass
        (rl.FROZEN_OPPONENT_P, rl.FROZEN_CKPT_FP,
         rl.FROZEN_CKPT_SH) = old


def test_spd7_disabled_is_loud_once_per_batch():
    """1.259 pin (c): unset config -> the exact DISABLED line, emitted
    from run_batch_cycle (per batch), never silent."""
    import inspect

    import run_td_v33loop as rl
    assert hasattr(rl, '_log_frozen_telemetry'), \
        'SPD: run_td_v33loop has no _log_frozen_telemetry ' \
        '(1.259 disabled-loud missing)'
    old = (rl.FROZEN_OPPONENT_P, rl.FROZEN_CKPT_FP, rl.FROZEN_CKPT_SH)
    bdir = os.path.join(SCRATCH, 'batch_spd7')
    os.makedirs(bdir, exist_ok=True)
    try:
        rl.FROZEN_CKPT_FP = None
        rl.FROZEN_CKPT_SH = None
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rl._log_frozen_telemetry(bdir)
        printed = buf.getvalue()
        assert _DISABLED_LINE in printed, \
            'SPD: DISABLED notice missing/wrong; printed %r' \
            % printed[-300:]
        src9 = inspect.getsource(rl.run_batch_cycle)
        assert '_log_frozen_telemetry' in src9, \
            'SPD: run_batch_cycle never emits the per-batch 1.259 ' \
            'notice'
    finally:
        (rl.FROZEN_OPPONENT_P, rl.FROZEN_CKPT_FP,
         rl.FROZEN_CKPT_SH) = old


def test_spd8_enabled_telemetry_and_env_plumbing():
    """1.259 telemetry: configured anchor logs frozen_episodes=N/M
    side_split=fp:x sh:y from the batch's games.jsonl; play_batch
    plumbs the WOTR_SP_FROZEN_* env to the selfplay subprocess."""
    import inspect

    import run_td_v33loop as rl
    bdir = os.path.join(SCRATCH, 'batch_spd8')
    os.makedirs(bdir, exist_ok=True)
    with open(os.path.join(bdir, 'games.jsonl'), 'w',
              encoding='utf-8') as f:
        for fs in ('fp', 'fp', 'sh', None):
            f.write(json.dumps({'winner': 'fp', 'frozen_side': fs})
                    + '\n')
    old = (rl.FROZEN_OPPONENT_P, rl.FROZEN_CKPT_FP, rl.FROZEN_CKPT_SH)
    try:
        rl.FROZEN_OPPONENT_P = 0.30
        rl.FROZEN_CKPT_FP = 'a_fp.pt'
        rl.FROZEN_CKPT_SH = 'a_sh.pt'
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            line = rl._log_frozen_telemetry(bdir)
        assert line and 'frozen_episodes=3/4' in line \
            and 'side_split=fp:2 sh:1' in line, \
            'SPD: telemetry line wrong: %r' % (line,)
    finally:
        (rl.FROZEN_OPPONENT_P, rl.FROZEN_CKPT_FP,
         rl.FROZEN_CKPT_SH) = old
    src9 = inspect.getsource(rl.play_batch)
    for tok in _FROZEN_ENV_KEYS:
        assert tok in src9, \
            'SPD: play_batch does not plumb %s to the selfplay ' \
            'subprocess' % tok


def test_spd9_preflight_refuses_wrong_generation_frozen():
    """1.259 pin (d): the SPC2 generation preflight covers the frozen
    checkpoints too -- a wrong-generation (or missing/unstamped)
    frozen net is refused at BOOT, not per episode."""
    import torch

    import ai
    import run_td_v33loop as rl
    pdir = os.path.join(SCRATCH, 'preflight_spd9')
    os.makedirs(pdir, exist_ok=True)
    old_loop = rl.LOOP
    old = (rl.FROZEN_OPPONENT_P, rl.FROZEN_CKPT_FP, rl.FROZEN_CKPT_SH)
    bad = os.path.join(pdir, 'frozen_fp_92.pt')
    good_fp = os.path.join(pdir, 'frozen_fp_ok.pt')
    good_sh = os.path.join(pdir, 'frozen_sh_ok.pt')
    torch.save({'model_state': {}, 'input_dim': ai.get_feature_dim(),
                'action_dim': 92, 'side': 'fp'}, bad)
    for p9, sd9 in ((good_fp, 'fp'), (good_sh, 'sh')):
        torch.save({'model_state': {},
                    'input_dim': ai.get_feature_dim(),
                    'action_dim': int(ai.ACTION_ENCODING_DIM),
                    'side': sd9}, p9)
    try:
        rl.LOOP = pdir            # empty roster: roster loop skips
        rl.FROZEN_CKPT_FP = bad
        rl.FROZEN_CKPT_SH = good_sh
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rl.preflight_generation()
            raise AssertionError(
                'SPD(d): wrong-generation frozen checkpoint was NOT '
                'refused at boot')
        except SystemExit as e:
            msg = str(e)
            assert 'frozen' in msg.lower() and '92' in msg \
                and bad in msg, \
                'refusal names the wrong facts: %r' % msg
        # missing file refused too
        rl.FROZEN_CKPT_FP = os.path.join(pdir, 'nope.pt')
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rl.preflight_generation()
            raise AssertionError(
                'SPD(d): missing frozen checkpoint was NOT refused')
        except SystemExit:
            pass
        # green leg: a current-generation pair boots clean
        rl.FROZEN_CKPT_FP = good_fp
        rl.FROZEN_CKPT_SH = good_sh
        with contextlib.redirect_stdout(io.StringIO()):
            rl.preflight_generation()
    finally:
        rl.LOOP = old_loop
        (rl.FROZEN_OPPONENT_P, rl.FROZEN_CKPT_FP,
         rl.FROZEN_CKPT_SH) = old


def main():
    tests = [
        ('test_spd1_selection_replays_worker_seeded_rng',
         test_spd1_selection_replays_worker_seeded_rng),
        ('test_spd2_frozen_env_pair_validation',
         test_spd2_frozen_env_pair_validation),
        ('test_spd3_episode_stamps_frozen_side',
         test_spd3_episode_stamps_frozen_side),
        ('test_spd4_collectors_drop_frozen_rows',
         test_spd4_collectors_drop_frozen_rows),
        ('test_spd5_timeout_episode_is_not_training_data',
         test_spd5_timeout_episode_is_not_training_data),
        ('test_spd6_loop_config_keys_and_pair_validation',
         test_spd6_loop_config_keys_and_pair_validation),
        ('test_spd7_disabled_is_loud_once_per_batch',
         test_spd7_disabled_is_loud_once_per_batch),
        ('test_spd8_enabled_telemetry_and_env_plumbing',
         test_spd8_enabled_telemetry_and_env_plumbing),
        ('test_spd9_preflight_refuses_wrong_generation_frozen',
         test_spd9_preflight_refuses_wrong_generation_frozen),
    ]
    fails = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print('PASS %s (%.1fs)' % (name, time.time() - t0))
        except Exception as e:
            fails += 1
            traceback.print_exc()
            print('FAIL %s: %s' % (name, e))
    print('TOTAL=%d FAIL=%d' % (len(tests), fails))
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
