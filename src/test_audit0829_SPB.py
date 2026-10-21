#!/usr/bin/env python3
"""SPB regression pins (self-play TRAINING files, audit 2026-08-29).

Findings pinned (SPTR report + VERIFY_selfplay_tail confirmations):
  SPTR-3   stale STORED arms trained with positive weight
  SPTR-4   outcome-gated policy distillation (CURRENT_RULES sec 7 item 4;
           TD side only - the BC half is a USER ruling question)
  SPTR-2   value_td residual clip = the adopted operator's head-matched
           clip; fresh seed nets honor WOTR_VALUE_HEAD_ACT default scaled2
  SPTR-5/6 a batch that trains 0 samples must fail loudly (v2 op side)
  SPTR-8   delta-collector drop/error stats surfaced, not discarded
  SPTR-9   menu ORDER hashseed-independent at the collection site
  SPTR-10  _normalize_cands columns derived from ai (stale 81/82)
  SPTR-7   BC label layout round-trip stat (states not stored; the loud
           stat is the honest half - re-extraction is the real fix)
  SPTR-14  value_td head BN scope (1.224e) via ONE shared helper
  SPTR-17  corrupt npz counted separately from empty

Pattern: plain test_* functions, main() prints TOTAL/FAIL, exit 1 on
any failure.  ASCII only (cp1252 console).
"""
import copy
import hashlib
import os
import subprocess
import sys
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('WOTR_ACT_GEN', 'v41')

import numpy as np

import ai
import board
import game_logic
import game_state as gs
from game_state import Side

SRC = os.path.dirname(os.path.abspath(__file__))
PYEXE = sys.executable

_FIX = {}


def _fixture():
    """Shared state + FP menu + a fresh net (test_rules-style setup)."""
    if _FIX:
        return _FIX
    import random
    import torch
    b = board.load_board()
    np.random.seed(7)
    random.seed(7)
    st = gs.setup_initial_game(b)
    game_logic.initialize_decks(st)
    for s_ in (Side.FREE_PEOPLES, Side.SHADOW):
        for d_ in ('character', 'strategy'):
            try:
                game_logic.draw_event_card(st, s_, d_)
            except Exception:
                pass
    game_logic.roll_action_dice(st, pool=Side.FREE_PEOPLES)
    game_logic.roll_action_dice(st, pool=Side.SHADOW)
    legal = ai.get_legal_actions(st, Side.FREE_PEOPLES)
    ae = np.array([ai.encode_action(a, st, Side.FREE_PEOPLES)
                   for a in legal], dtype=np.float32)
    ri = np.array([ai._get_action_region_index(a) for a in legal],
                  dtype=np.int64)
    torch.manual_seed(0)
    net = ai.WOTRValueNet(ai.get_feature_dim(), side='fp')
    net.eval()
    _FIX.update(st=st, legal=legal, ae=ae, ri=ri, net=net)
    return _FIX


def _ss_clean(K=6):
    f = _fixture()
    n = np.array([120., 90., 60., 40., 30., 25.])[:K]
    return {'ae': f['ae'][:K].astype(np.float16),
            'ri': f['ri'][:K].astype(np.int16),
            'p': np.full(K, 1.0 / K, dtype=np.float32),
            'q': np.zeros(K, dtype=np.float32),
            'n': n.astype(np.float64), 'played': 0}


def _ss_fake(K=6):
    """Probe-F construction: last stored arm mutated so it matches
    NOTHING in the regenerated menu, and given the most visits."""
    ss = _ss_clean(K)
    ae = np.asarray(ss['ae'], dtype=np.float32).copy()
    fake = ae[0].copy()
    fake[15] = 0.5 - fake[15]
    fake[14] = 0.5 - fake[14]
    ae[K - 1] = fake
    ss = dict(ss)
    ss['ae'] = ae.astype(np.float16)
    ss['n'] = np.array([120., 90., 60., 40., 30., 300.])[:K]
    return ss


def _mk_trace(ss, winner='fp', victory='ring_destroyed'):
    f = _fixture()
    post = ai.clone_state(f['st'])
    return {'meta': {'winner': winner, 'victory_type': victory},
            'steps': [{'state': f['st'], 'kind': 'new_turn'},
                      {'state': post, 'kind': 'action', 'side': 'FP',
                       'action_type': 'army_move', 'search_stats': ss}]}


def _dparams(net):
    return {k: v.detach().clone() for k, v in net.named_parameters()
            if k.startswith('policy_delta')}


def _with_profiles(fn):
    ai.PLAYER_PROFILES['fp'] = 'ring'
    ai.PLAYER_PROFILES['sh'] = 'military'
    try:
        return fn()
    finally:
        ai.PLAYER_PROFILES['fp'] = None
        ai.PLAYER_PROFILES['sh'] = None


# ------------------------------------------------------------------
# SPTR-3: stale stored arms masked out of the CE target
# ------------------------------------------------------------------

def test_sptr3_stale_arm_masked():
    import torch
    import delta_trainer as dt
    f = _fixture()

    def body():
        tr = _mk_trace(_ss_fake())
        rows, stats = dt.collect_delta_samples(tr, 'fp')
        assert len(rows) == 1, 'row was not collected: %r' % (stats,)
        r = rows[0]
        # the fake arm really is off-menu (guards the construction)
        d = np.abs(f['ae'][None] - np.asarray(
            r['ae'], np.float32)[:, None]).max(-1).min(1)
        assert d[5] > dt._ATOL, 'fake arm accidentally matches the menu'
        assert stats.get('stale_arms') == 1, (
            'collector stats carry no stale_arms count: %r' % (stats,))
        assert 'sm' in r, 'row carries no stored-match mask (sm)'
        sm = np.asarray(r['sm'], dtype=bool)
        assert sm[:5].all() and not sm[5], 'wrong mask: %r' % (sm,)
        pi_k, keep, n_masked, why = dt._masked_target(r)
        assert why is None, 'row unexpectedly skipped: %r' % (why,)
        assert n_masked == 1
        assert 5 not in list(keep), 'stale arm kept in the CE softmax'
        assert abs(float(np.sum(pi_k)) - 1.0) < 1e-9, (
            'target mass not renormalized: sum=%r' % float(np.sum(pi_k)))
        net2 = copy.deepcopy(f['net'])
        r2 = dict(r)
        r2['ow'] = 1.0
        m = dt.train_policy_delta(
            net2, [r2], dt.DeltaTrainConfig(epochs=1, lr=1e-3,
                                            kl_beta=0.0))
        assert m.get('stale_arms_masked') == 1, (
            'trainer metrics carry no stale_arms_masked: %r' % (m,))
    _with_profiles(body)


# ------------------------------------------------------------------
# SPTR-4 / SPEP-9: outcome-gated policy distillation (TD side)
# ------------------------------------------------------------------

def test_sptr4_outcome_gate():
    import torch
    import delta_trainer as dt
    f = _fixture()

    def body():
        # LOSING side (fp lost, sh military win): axis outcome <= 0
        tr_loss = _mk_trace(_ss_clean(), winner='sh', victory='military')
        rows = dt.collect_for_driver(tr_loss, 'fp')
        assert rows, 'collector returned no rows'
        assert 'ow' in rows[0], (
            'rows carry no outcome weight ow (CURRENT_RULES 7.4)')
        assert rows[0]['ow'] == 0.0, (
            'losing-side ow=%r, want 0.0' % (rows[0]['ow'],))
        net2 = copy.deepcopy(f['net'])
        p0 = _dparams(net2)
        dt.train_policy_delta(
            net2, [dict(rows[0])],
            dt.DeltaTrainConfig(epochs=2, lr=1e-2, kl_beta=0.0, wd=0.0))
        p1 = _dparams(net2)
        changed = [k for k in p0 if not torch.equal(p0[k], p1[k])]
        assert not changed, (
            'losing-side rows moved policy_delta params: %s'
            % changed[:3])
        # WINNING side trains normally
        tr_win = _mk_trace(_ss_clean(), winner='fp',
                           victory='ring_destroyed')
        rows_w = dt.collect_for_driver(tr_win, 'fp')
        assert rows_w[0]['ow'] > 0.0, (
            'winning-side ow=%r, want > 0' % (rows_w[0]['ow'],))
        net3 = copy.deepcopy(f['net'])
        p0 = _dparams(net3)
        m = dt.train_policy_delta(
            net3, [dict(rows_w[0])],
            dt.DeltaTrainConfig(epochs=2, lr=1e-2, kl_beta=0.0, wd=0.0))
        p1 = _dparams(net3)
        assert any(not torch.equal(p0[k], p1[k]) for k in p0), (
            'winning-side row did not train: %r' % (m,))
    _with_profiles(body)


# ------------------------------------------------------------------
# SPTR-2: value_td clip shares the adopted operator's head-matched
# constant; fresh seed nets honor WOTR_VALUE_HEAD_ACT default scaled2
# ------------------------------------------------------------------

def test_sptr2_value_td_clip():
    import torch
    import train_corpus as tc
    import value_trainer_v2 as vt
    f = _fixture()
    net_t = copy.deepcopy(f['net'])
    net_t.value_head[-1] = torch.nn.Tanh()
    net_s = copy.deepcopy(f['net'])
    vt.swap_head_scaled2(net_s)
    assert hasattr(tc, '_td_target_clip'), (
        'value_td still hardcodes +-2.0 (no shared clip helper)')
    ct = tc._td_target_clip(net_t)
    cs = tc._td_target_clip(net_s)
    assert abs(ct - vt._head_clip(net_t)) < 1e-12 and \
        abs(ct - 0.98) < 1e-9, 'tanh clip %r != 0.98' % ct
    assert abs(cs - vt._head_clip(net_s)) < 1e-12 and \
        abs(cs - 1.96) < 1e-9, 'scaled2 clip %r != 1.96' % cs
    # synthetic residual target beyond +-1 is REACHABLE under scaled2
    assert float(np.clip(1.5, -cs, cs)) == 1.5
    assert float(np.clip(2.5, -cs, cs)) > 1.0


def test_sptr2_fresh_head_stamp():
    import train_corpus as tc
    assert hasattr(tc, '_fresh_head'), (
        'no fresh-head env application (seeds stamp whatever '
        'ai._value_head_act defaulted to)')
    env0 = os.environ.pop('WOTR_VALUE_HEAD_ACT', None)
    try:
        n1 = ai.WOTRValueNet(ai.get_feature_dim(), side='fp')
        tc._fresh_head(n1, 'pin')
        assert tc._head_act_of(n1) == 'scaled2', (
            'env-less fresh net stamps %r, want scaled2 (run-7 bug 2)'
            % tc._head_act_of(n1))
        os.environ['WOTR_VALUE_HEAD_ACT'] = 'tanh'
        n2 = ai.WOTRValueNet(ai.get_feature_dim(), side='fp')
        tc._fresh_head(n2, 'pin')
        assert tc._head_act_of(n2) == 'tanh', (
            'explicit legacy env not honored')
    finally:
        if env0 is None:
            os.environ.pop('WOTR_VALUE_HEAD_ACT', None)
        else:
            os.environ['WOTR_VALUE_HEAD_ACT'] = env0


# ------------------------------------------------------------------
# SPTR-5/6 (v2 side): zero-usable-sample batch fails loudly
# ------------------------------------------------------------------

def test_sptr5_zero_sample_batch_fails():
    import delta_trainer as dt
    f = _fixture()
    bad = {'enc': ai.encode_state(f['st'],
                                  Side.FREE_PEOPLES).astype(np.float32),
           'ae': np.asarray(f['ae'][:3]), 'ri': f['ri'][:3],
           'p': np.full(3, 1.0 / 3.0),
           'n': np.array([5., 4., 3.])}          # no 'q' -> prep error
    net2 = copy.deepcopy(f['net'])
    m = None
    try:
        m = dt.train_policy_delta(net2, [bad],
                                  dt.DeltaTrainConfig(epochs=1))
    except RuntimeError:
        return
    raise AssertionError(
        'zero-usable-sample batch reported success: %r' % (m,))


# ------------------------------------------------------------------
# SPTR-8: collector stats surfaced through the driver adapter
# ------------------------------------------------------------------

def test_sptr8_collector_stats_surfaced():
    import delta_trainer as dt
    assert hasattr(dt, 'DRIVER_STATS'), (
        'collect_for_driver discards its stats (SPTR-8)')

    def body():
        before = dict(dt.DRIVER_STATS)
        tr = _mk_trace(_ss_fake())
        rows = dt.collect_for_driver(tr, 'fp')
        assert rows
        assert dt.DRIVER_STATS['rows'] >= before.get('rows', 0) + 1
        assert dt.DRIVER_STATS['stale_arms'] >= \
            before.get('stale_arms', 0) + 1, (
            'stale_arms not accumulated: %r' % (dt.DRIVER_STATS,))
    _with_profiles(body)


# ------------------------------------------------------------------
# SPTR-9: menu ORDER at the collection site is hashseed-independent
# ------------------------------------------------------------------

_HASHSEED_SCRIPT = r'''
import os, sys, hashlib
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['WOTR_ACT_GEN'] = 'v41'
sys.path.insert(0, %(src)r)
import numpy as np
import random
import board as board_mod
import game_state as gs
import game_logic
import ai
from game_state import Side
b = board_mod.load_board()
np.random.seed(7); random.seed(7)
st = gs.setup_initial_game(b)
game_logic.initialize_decks(st)
for s_ in (Side.FREE_PEOPLES, Side.SHADOW):
    for d_ in ('character', 'strategy'):
        try:
            game_logic.draw_event_card(st, s_, d_)
        except Exception:
            pass
game_logic.roll_action_dice(st, pool=Side.FREE_PEOPLES)
game_logic.roll_action_dice(st, pool=Side.SHADOW)
import delta_trainer as dt
legal = ai.get_legal_actions(st, Side.FREE_PEOPLES)
enc = np.array([ai.encode_action(a, st, Side.FREE_PEOPLES)
                for a in legal], dtype=np.float32)
ri = np.array([ai._get_action_region_index(a) for a in legal],
              dtype=np.int64)
srt = sorted(range(len(enc)), key=lambda i: enc[i].tobytes())
K = 4
ss = {'ae': enc[srt[:K]], 'ri': ri[srt[:K]],
      'p': np.full(K, 0.25), 'q': np.zeros(K),
      'n': np.array([5., 4., 3., 2.]), 'played': 0}
post = ai.clone_state(st)
tr = {'meta': {'winner': 'fp', 'victory_type': 'ring_destroyed'},
      'steps': [{'state': st, 'kind': 'new_turn'},
                {'state': post, 'kind': 'action', 'side': 'FP',
                 'action_type': 'army_move', 'search_stats': ss}]}
ai.PLAYER_PROFILES['fp'] = 'ring'
rows, stats = dt.collect_delta_samples(tr, 'fp')
ai.PLAYER_PROFILES['fp'] = None
r = rows[0]
blob = np.round(np.concatenate([np.asarray(r['ae'], np.float32),
                                np.asarray(r['extra_ae'], np.float32)]),
                3).astype(np.float32).tobytes()
print('ORDERMD5', hashlib.md5(blob).hexdigest(), len(r['extra_ae']))
'''


def test_sptr9_hashseed_menu_order():
    script = _HASHSEED_SCRIPT % {'src': SRC}
    fd, path = tempfile.mkstemp(suffix='.py', prefix='spb_hashseed_')
    with os.fdopen(fd, 'w') as fh:
        fh.write(script)
    try:
        outs = []
        for seed in ('0', '12345'):
            env = dict(os.environ)
            env['PYTHONHASHSEED'] = seed
            cp = subprocess.run([PYEXE, path], capture_output=True,
                                text=True, timeout=600, env=env,
                                cwd=SRC)
            line = next((ln for ln in cp.stdout.splitlines()
                         if ln.startswith('ORDERMD5')), None)
            assert line, ('subprocess (seed %s) gave no ORDERMD5; '
                          'rc=%d stderr tail: %s'
                          % (seed, cp.returncode, cp.stderr[-400:]))
            outs.append(line.split()[1])
        assert outs[0] == outs[1], (
            'regenerated-menu order depends on PYTHONHASHSEED: '
            '%s vs %s' % (outs[0], outs[1]))
    finally:
        os.remove(path)


# ------------------------------------------------------------------
# SPA handoff item 1 (SPEP-8 second half): declare_decline trace rows
# are declare-phase rows — the collector must restore _declare_phase
# so the regenerated menu matches the stored declare arms and the
# absence sample TRAINS (not skew-drop / stale-mask fodder)
# ------------------------------------------------------------------

def test_spa1_declare_decline_rows_train():
    import delta_trainer as dt
    f = _fixture()
    assert dt._flag_for('declare_decline') == '_declare_phase', (
        'declare_decline not recognized as a declare-phase row')

    def body():
        st0 = f['st']
        # stored arms encoded exactly as the declare-root search did:
        # under the transient _declare_phase flag
        setattr(st0, '_declare_phase', True)
        try:
            dm = ai.get_legal_actions(st0, Side.FREE_PEOPLES)
            enc_d = np.array(
                [ai.encode_action(a, st0, Side.FREE_PEOPLES)
                 for a in dm], dtype=np.float32)
            ri_d = np.array([ai._get_action_region_index(a)
                             for a in dm], dtype=np.int64)
        finally:
            setattr(st0, '_declare_phase', False)
        K = len(dm)
        assert K >= 2, 'declare menu unexpectedly trivial: %d' % K
        played = next(i for i, a in enumerate(dm)
                      if a.action_type == 'pass')

        def ss():
            n = np.full(K, 10.0)
            n[played] = 30.0                 # the decline was picked
            return {'ae': enc_d.astype(np.float16),
                    'ri': ri_d.astype(np.int16),
                    'p': np.full(K, 1.0 / K, dtype=np.float32),
                    'q': np.zeros(K, dtype=np.float32),
                    'n': n, 'played': played}
        # 3-turn trace: one declare_decline row per turn start
        # (SPA handoff item 5 row shape: FP-sided, die_idx=-1,
        # search_stats attached, post-state == pre-state)
        steps = []
        prev = st0
        for _t in range(3):
            steps.append({'state': prev, 'kind': 'new_turn'})
            steps.append({'state': ai.clone_state(prev),
                          'kind': 'action', 'side': 'FP',
                          'action_type': 'declare_decline',
                          'die_idx': -1, 'search_stats': ss()})
            prev = ai.clone_state(st0)
        tr = {'meta': {'winner': 'fp',
                       'victory_type': 'ring_destroyed'},
              'steps': steps}
        rows, stats = dt.collect_delta_samples(tr, 'fp')
        assert stats['rows'] == 3 and stats['skew_dropped'] == 0, (
            'declare_decline rows not collected: %r' % (stats,))
        assert stats['stale_arms'] == 0, (
            'declare arms treated as stale (menu regenerated without '
            'the declare flag): %r' % (stats,))
        for r in rows:
            pi_k, keep, n_masked, why = dt._masked_target(r)
            assert why is None and n_masked == 0, (
                'declare row skipped (%r) / masked (%d)'
                % (why, n_masked))
            assert len(keep) == K
            assert abs(float(np.sum(pi_k)) - 1.0) < 1e-9, (
                'declare row target mass %r' % float(np.sum(pi_k)))
            assert float(pi_k[played]) > 0.5, (
                'decline arm lost its visit mass: %r' % (pi_k,))
        # the flagged state must be restored (no stuck flag)
        assert not getattr(st0, '_declare_phase', False)
        # side attribution: an SH collect sees none of these rows
        rows_sh, stats_sh = dt.collect_delta_samples(tr, 'sh')
        assert stats_sh['rows'] == 0 and not rows_sh, (
            'declare_decline rows leaked to the SH collector')
    _with_profiles(body)


# ------------------------------------------------------------------
# SPTR-10 / SPCP-7: _normalize_cands columns from ai (stale 81/82)
# ------------------------------------------------------------------

def test_sptr10_normalize_cands_columns():
    import train_corpus as tc
    A = ai.ACTION_ENCODING_DIM
    p_idx = ai._ACTION_TYPE_TO_IDX['pass']
    # (a) a non-skip pass with a stray die face: face must be zeroed
    #     and the card-option flag must SURVIVE
    row = np.zeros((1, A), dtype=np.float32)
    row[0, p_idx] = 1.0
    row[0, ai._ACT_DIEFACE_OFF] = 0.5
    row[0, ai._ACT_CARDOPT_OFF] = 0.7
    out = tc._normalize_cands(row.copy())
    assert out[0, ai._ACT_DIEFACE_OFF] == 0.0, (
        'v30 pass-face strip is DEAD (stale column constants)')
    assert out[0, ai._ACT_CARDOPT_OFF] == 0.7, (
        'card-option flag corrupted by the stale columns')
    # (b) same with a low face value (the pre-fix corruption path)
    row2 = np.zeros((1, A), dtype=np.float32)
    row2[0, p_idx] = 1.0
    row2[0, ai._ACT_DIEFACE_OFF] = 0.3
    row2[0, ai._ACT_CARDOPT_OFF] = 0.7
    out2 = tc._normalize_cands(row2.copy())
    assert out2[0, ai._ACT_DIEFACE_OFF] == 0.0
    assert out2[0, ai._ACT_CARDOPT_OFF] == 0.7, (
        'card-option flag zeroed (mistaken for the die face)')
    # (c) skips legitimately keep their face
    row3 = np.zeros((1, A), dtype=np.float32)
    row3[0, p_idx] = 1.0
    row3[0, ai._ACT_SKIP_OFF] = 1.0
    row3[0, ai._ACT_DIEFACE_OFF] = 0.5
    out3 = tc._normalize_cands(row3.copy())
    assert out3[0, ai._ACT_DIEFACE_OFF] == 0.5
    # (d) resolved columns match ai (single source)
    assert tc._act_cols() == (ai._ACT_DIEFACE_OFF, ai._ACT_SKIP_OFF), (
        'train_corpus columns %r do not track ai (%d, %d)'
        % (tc._act_cols(), ai._ACT_DIEFACE_OFF, ai._ACT_SKIP_OFF))


# ------------------------------------------------------------------
# SPTR-7 (honest half): BC label layout round-trip stat
# ------------------------------------------------------------------

def test_sptr7_label_roundtrip_stat():
    import train_corpus as tc
    f = _fixture()
    assert hasattr(tc, '_label_roundtrip_fail'), (
        'BC has no label layout round-trip stat (SPTR-7)')
    bad = [(i, tc._label_roundtrip_fail(r))
           for i, r in enumerate(f['ae'])
           if tc._label_roundtrip_fail(r) is not None]
    assert not bad, ('live-encoded actions flagged as failing: %r'
                     % bad[:5])
    r0 = f['ae'][0].copy()
    r0[:len(ai.ACTION_TYPE_CATEGORIES)] = 0.0
    assert tc._label_roundtrip_fail(r0), 'zero-category row not flagged'
    r1 = f['ae'][0].copy()
    other = (int(np.argmax(r1[:len(ai.ACTION_TYPE_CATEGORIES)])) + 1) \
        % len(ai.ACTION_TYPE_CATEGORIES)
    r1[other] = 1.0
    assert tc._label_roundtrip_fail(r1), 'two-category row not flagged'
    assert tc._label_roundtrip_fail(f['ae'][0][:-1]), (
        'wrong-width row not flagged')


# ------------------------------------------------------------------
# SPTR-14: value_td head BN scope (1.224e) via ONE shared helper
# ------------------------------------------------------------------

def test_sptr14_head_bn_scope():
    import torch
    import train_corpus as tc
    f = _fixture()
    assert hasattr(tc, '_head_train_bn_eval'), (
        'value_td trains head BN buffers (no 1.224e scope helper)')
    net2 = copy.deepcopy(f['net'])
    tc._head_train_bn_eval(net2.value_head)
    assert net2.value_head.training, 'head not in train mode'
    bns = [m for m in net2.value_head.modules()
           if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))]
    assert bns, 'fixture head has no BatchNorm - pin needs updating'
    assert all(not m.training for m in bns), (
        'head BN left in train mode (buffer drift, 1.224e)')


# ------------------------------------------------------------------
# SPTR-17: corrupt npz counted separately from empty
# ------------------------------------------------------------------

def test_sptr17_corrupt_npz_counted():
    import train_corpus as tc
    assert hasattr(tc, '_UNREADABLE'), (
        'corrupt npz silently conflated with empty (SPTR-17)')
    tmpd = tempfile.mkdtemp(prefix='spb_npz_')
    corrupt = os.path.join(tmpd, 'corrupt.npz')
    with open(corrupt, 'wb') as fh:
        fh.write(b'this is not an npz file')
    empty = os.path.join(tmpd, 'empty.npz')
    np.savez(empty, other=np.zeros(1))
    n0 = len(tc._UNREADABLE)
    assert tc.is_empty_npz(corrupt) is True
    assert len(tc._UNREADABLE) == n0 + 1, (
        'corrupt npz not recorded: %r' % (tc._UNREADABLE[-2:],))
    assert tc.is_empty_npz(empty) is True
    assert len(tc._UNREADABLE) == n0 + 1, (
        'EMPTY npz wrongly recorded as unreadable')
    os.remove(corrupt)
    os.remove(empty)
    os.rmdir(tmpd)


TESTS = [
    test_sptr3_stale_arm_masked,
    test_sptr4_outcome_gate,
    test_sptr2_value_td_clip,
    test_sptr2_fresh_head_stamp,
    test_sptr5_zero_sample_batch_fails,
    test_sptr8_collector_stats_surfaced,
    test_spa1_declare_decline_rows_train,
    test_sptr9_hashseed_menu_order,
    test_sptr10_normalize_cands_columns,
    test_sptr7_label_roundtrip_stat,
    test_sptr14_head_bn_scope,
    test_sptr17_corrupt_npz_counted,
]


def main():
    fails = 0
    for t in TESTS:
        try:
            t()
            print('PASS %s' % t.__name__, flush=True)
        except Exception:
            fails += 1
            print('FAIL %s' % t.__name__, flush=True)
            traceback.print_exc()
    print('TOTAL=%d FAIL=%d' % (len(TESTS), fails), flush=True)
    if fails:
        sys.exit(1)


if __name__ == '__main__':
    main()
