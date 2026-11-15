# -*- coding: utf-8 -*-
"""In-recipe seed self-certification: the DEPTH BATTERY.

USER order 2026-08-10 ("fix the bugs, no fallbacks") + SEARCH's
ablation verdict (comms\\25): the v34 recipe poisons DEPTH-2+ interior
values (sit/army continuation states priced ~+1.68 vs the push's
+1.14) at BOTH the value-td and train-joint rungs, while every
ONE-STEP ladder probe passed those rungs — one-step ordering is
PROVEN INSUFFICIENT as a seed cert.  The only instrument that catches
interior-value poison is live search-Q behavior on the user-specified
fixtures, i.e. exactly the smoke gate's fixture check.

This module packages that check as a callable battery over a
CALLER-SUPPLIED net pair so training stages can self-certify per
epoch/rung, BEFORE anything is installed:
  - bars are the smoke gate's own (launch_smoke_gate.check_fixtures),
    mirrored 1:1 by hand (NO pin test exists — keep the two judge
    blocks in sync manually on every bar change); _declare_violation
    is DEFINED here and imported by launch_smoke_gate (single
    implementation in the import-light leaf: importing the gate
    module would execute its env pins + run_td_v33loop chain inside
    the certifying training process);
  - a cheap one-step VALUE leg is kept as a REPORT-ONLY pre-screen
    (it catches the jt-class surface inversions early in an epoch
    trace) — it can never PASS a net, only annotate.

SEEDING (R4 instrument finding, comms\\35): np.random was NEVER
seeded here — PYTHONHASHSEED does not govern numpy streams, so every
battery run consumed a fresh RNG stream and thin-margin verdicts
(erebor's 0.0025 Q-gap) were RE-ROLLS, not measurements.  Every
fixture probe now seeds numpy + random DETERMINISTICALLY
(sha256(fixture|stream) -> 32-bit seed) at probe start, so verdicts
reproduce bit-for-bit run-over-run under a fixed engine.
WOTR_CERT_STREAMS=K (default 1) runs K independently-seeded streams
per fixture and takes the STRICT-MAJORITY verdict (ties FAIL; use
odd K).  Per-stream records + decision margins (top-2 Q/visit gaps)
ride each leg so below-resolution margins are VISIBLE in the record,
never silently coin-flipped.  PYTHONHASHSEED=0 remains the caller's
job (env, pre-interpreter); this module cannot set it.

Partner-net doctrine: certifying side X uses a caller-chosen partner
for side Y.  Inside a retrain, the same-era partner is the honest
choice (both poisoned -> cert fails -> correct outcome); for rung
LOCALIZATION, hold the partner fixed at a known-healthy net so the
failing rung is isolated.  A skipped or errored fixture is a FAIL,
never a silent pass.  ASCII only.
"""
import glob
import hashlib
import json
import os
import pickle
import random

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
FIXTURES_DIR = os.environ.get('WOTR_FIXTURES_DIR') or os.path.join(
    ROOT, 'data', 'behavior_fixtures')
DEFAULT_SIMS = int(os.environ.get('WOTR_JT_CERT_SIMS', '400') or 400)
DEFAULT_STREAMS = int(os.environ.get('WOTR_CERT_STREAMS', '1') or 1)

# The smoke-gate fixture roster (launch_smoke_gate.check_fixtures):
# 5 named + every declare_burn_*.
NAMED = ('siege_choice_erebor', 'mordor_push', 'erebor_seal',
         'erebor_recapture', 'wr_recapture')


def _fixture_names():
    burns = sorted(os.path.basename(p)[:-4] for p in glob.glob(
        os.path.join(FIXTURES_DIR, 'declare_burn_*.pkl')))
    return list(NAMED) + burns


def _declare_violation(ai, state, action):
    """(is_violation, detail) for a picked fellowship_declare: metric
    delta < 0 with no compensation (heal / gateway / activation).
    Canonical single implementation — launch_smoke_gate imports it."""
    from board import sanitize
    m0 = ai._ring_progress_metric(state)
    pre_corr = int(state.fellowship.corruption)
    pre_active = {n for n, v in state.political.active.items() if v}
    clone = ai.clone_state(state)
    clone._declare_phase = False
    ai._execute_action(clone, action)
    m1 = ai._ring_progress_metric(clone)
    delta = m1 - m0
    heal = int(clone.fellowship.corruption) < pre_corr
    reg = sanitize(str(action.params.get('region', '')))
    gateway = (reg in ('morannon', 'minas morgul')
               or bool(clone.fellowship.mordor_entered))
    activated = bool({n for n, v in clone.political.active.items()
                      if v} - pre_active)
    burning = delta < -1e-9 and not (heal or gateway or activated)
    detail = ('declare@%s metric %.1f->%.1f (delta %+.1f) heal=%d '
              'gateway=%d activation=%d'
              % (reg or '?', m0, m1, delta, heal, gateway, activated))
    return burning, detail


def _fixture_seed(name, stream=0):
    """Deterministic 32-bit seed per (fixture, stream): first 4 bytes
    of sha256('name|stream'), little-endian.  Independent of
    PYTHONHASHSEED and of probe order."""
    h = hashlib.sha256(
        ('%s|%d' % (name, int(stream))).encode('ascii')).digest()
    return int.from_bytes(h[:4], 'little')


def _majority(oks):
    """STRICT majority of stream verdicts; a tie is a FAIL (even K is
    discouraged — use odd)."""
    return sum(1 for o in oks if o) * 2 > len(oks)


def _margins(vi):
    """Decision-margin annotations from the root visit info: top-2 Q
    gap (the argmax-Q judge's deciding quantity) and top-2 visit gap
    (the pick judges' deciding quantity).  Report-only."""
    import numpy as np
    try:
        q = np.asarray(vi['q_values'], dtype=float)
        n = np.asarray(vi['visit_counts'], dtype=float)
        if len(q) >= 2:
            qs = np.sort(q)
            ns = np.sort(n)
            return {'q_top2_gap': round(float(qs[-1] - qs[-2]), 6),
                    'n_top2_gap': int(round(float(ns[-1] - ns[-2])))}
    except Exception:
        pass
    return {'q_top2_gap': None, 'n_top2_gap': None}


def _load_fixture(name):
    import board as board_mod
    with open(os.path.join(FIXTURES_DIR, name + '.pkl'), 'rb') as f:
        d = pickle.load(f)
    st, meta = d['state'], (d.get('meta') or {})
    if st.board is None:
        st.board = board_mod.load_board()
    return st, meta


def _judge(name, act, vi, st, side, ai):
    """The smoke gate's bar per fixture, mirrored 1:1 BY HAND from
    launch_smoke_gate.check_fixtures (no pin test exists — sync
    manually on every bar change).  Returns (ok, tracked_only,
    detail)."""
    import numpy as np
    picked = (act.action_type if act is not None else None)
    if name == 'siege_choice_erebor':
        return (picked == 'retreat_into_siege', False,
                'picked %s (expect retreat_into_siege)' % picked)
    if name == 'mordor_push':
        return (picked == 'fellowship_move', False,
                'picked %s (expect fellowship_move)' % picked)
    if name == 'erebor_seal':
        ok = (picked in ('army_move', 'army_attack')
              and 'erebor' in str(
                  (act.params or {}).get('to', '')).lower())
        return ok, False, 'picked %s (expect dale->erebor)' % picked
    if name == 'wr_recapture':
        return True, True, 'picked %s (tracked-only)' % picked
    if name == 'erebor_recapture':
        q9 = np.asarray(vi['q_values'], dtype=float)
        ae9 = np.asarray(vi['action_encs'], dtype=np.float32)
        accs = {np.asarray(ai.encode_action(x, st, side),
                           dtype=np.float32).tobytes(): x
                for x in ai.get_legal_actions(st, side)}
        bj = int(np.argmax(q9))
        ba = accs.get(ae9[bj].tobytes())
        breg = str((getattr(ba, 'params', None) or {})
                   .get('region', '')).lower()
        ok = (ba is not None and ba.action_type.startswith('muster')
              and 'iron hills' in breg)
        return ok, False, ('best-Q=%s@%s picked=%s'
                           % (getattr(ba, 'action_type', '?'),
                              breg, picked))
    # declare_burn_*: burning declares are the violation
    if picked == 'fellowship_declare':
        burning, detail = _declare_violation(ai, st, act)
        return (not burning), False, detail
    return True, False, 'picked %s (no declare)' % picked


def one_step_top(net, st, side_obj):
    """REPORT-ONLY pre-screen (proven insufficient as a gate: the
    depth-poisoned td/base rungs passed it).  Reads the CURRENT
    ai.PLAYER_PROFILES shaping — call it while the fixture's meta
    profiles are applied, or the annotation is computed under the
    wrong regime."""
    import ai
    best = (None, -1e18)
    n_fail = 0
    last_err = None
    acts = ai.get_legal_actions(st, side_obj)
    for a in acts:
        try:
            c = ai.clone_state(st)
            ai._execute_action(c, a)
            v = float(ai._mcts_eval_state(c, side_obj, net, net))
        except Exception as e:
            n_fail += 1
            last_err = e
            continue
        if v > best[1]:
            best = (a, v)
    if best[0] is None and acts and n_fail:
        # every candidate failed: loud sentinel, not a silent None
        return 'ERROR all %d actions failed (last: %r)' % (n_fail,
                                                           last_err)
    return getattr(best[0], 'action_type', None)


def _probe_once(name, fp_net, sh_net, sims, stream,
                prescreen=False):
    """One SEEDED probe of one fixture: fresh state load, profiles
    from fixture meta, numpy+random seeded from
    _fixture_seed(name, stream) immediately before the search.
    Returns (record, state, side).  When prescreen is set the
    report-only one_step_top annotation is computed HERE, while the
    fixture's meta profiles are still applied — computing it after
    the finally restore would price it under the CALLER's shaping
    regime, not the one the search verdict was judged under."""
    import numpy as np
    import ai
    from game_state import Side
    st, meta = _load_fixture(name)
    side = (Side.FREE_PEOPLES
            if str(meta.get('side', 'FP')) == 'FP'
            else Side.SHADOW)
    old = dict(ai.PLAYER_PROFILES)
    ai.PLAYER_PROFILES['fp'] = meta.get('fp_profile')
    ai.PLAYER_PROFILES['sh'] = meta.get('sh_profile')
    is_declare = name.startswith('declare_burn')
    if is_declare:
        st._declare_phase = True
    try:
        ai._TREE_CACHE['child'] = None
        ai._TREE_CACHE['sig'] = None
        seed = _fixture_seed(name, stream)
        np.random.seed(seed)
        random.seed(seed)
        act, vi = ai.mcts_action_with_policy(
            st, side, fp_net, sh_net, num_simulations=sims,
            temperature=0.0, add_noise=False)
        ok, tracked, detail = _judge(name, act, vi, st, side, ai)
        rec = {'ok': bool(ok), 'tracked': bool(tracked),
               'detail': detail, 'seed': int(seed),
               'picked': (act.action_type if act is not None
                          else None)}
        rec.update(_margins(vi))
        if prescreen:
            net9 = fp_net if side == Side.FREE_PEOPLES else sh_net
            try:
                rec['one_step_top'] = one_step_top(net9, st, side)
            except Exception as e:
                rec['one_step_top'] = 'ERROR %r' % e
        return rec, st, side
    finally:
        if is_declare:
            st._declare_phase = False
        ai.PLAYER_PROFILES.update(old)


def depth_battery(fp_net, sh_net, sims=None, value_prescreen=True,
                  streams=None):
    """Run the full fixture battery under live SEEDED search at
    `sims`, `streams` independently-seeded streams per fixture
    (default WOTR_CERT_STREAMS, else 1).  Returns {'pass',
    'n_gating', 'n_streams', 'legs': [...]}: pass = every non-tracked
    leg majority-ok; errors and missing fixtures FAIL, never skip.
    Leg records carry per-stream verdicts, seeds, and top-2 Q/visit
    margins."""
    if sims is None:
        sims = DEFAULT_SIMS
    if streams is None:
        streams = DEFAULT_STREAMS
    streams = max(1, int(streams))
    legs = []
    for name in _fixture_names():
        path = os.path.join(FIXTURES_DIR, name + '.pkl')
        if not os.path.exists(path):
            legs.append({'fixture': name, 'ok': False,
                         'tracked': False, 'detail': 'missing'})
            continue
        srecs = []
        for s in range(streams):
            _pre = (value_prescreen and s == 0
                    and not name.startswith('declare_burn'))
            try:
                rec, _st9, _side9 = _probe_once(name, fp_net, sh_net,
                                                sims, s,
                                                prescreen=_pre)
            except Exception as e:
                rec = {'ok': False, 'tracked': False,
                       'detail': 'ERROR %r' % e,
                       'seed': _fixture_seed(name, s),
                       'picked': None,
                       'q_top2_gap': None, 'n_top2_gap': None}
            srecs.append(rec)
        oks = [bool(r['ok']) for r in srecs]
        qg = [r['q_top2_gap'] for r in srecs
              if r.get('q_top2_gap') is not None]
        ng = [r['n_top2_gap'] for r in srecs
              if r.get('n_top2_gap') is not None]
        leg = {'fixture': name,
               'ok': _majority(oks),
               'tracked': all(bool(r.get('tracked'))
                              for r in srecs),
               'detail': (srecs[0]['detail'] if streams == 1 else
                          '%d/%d streams ok; s0: %s'
                          % (sum(oks), len(oks),
                             srecs[0]['detail'])),
               'n_streams': streams,
               'ok_stream0': oks[0],
               'q_top2_gap_med': (round(float(sorted(qg)[len(qg)
                                  // 2]), 6) if qg else None),
               'n_top2_gap_med': (int(sorted(ng)[len(ng) // 2])
                                  if ng else None),
               'streams': srecs}
        if 'one_step_top' in srecs[0]:
            leg['one_step_top'] = srecs[0]['one_step_top']
        legs.append(leg)
    gating = [l for l in legs if not l['tracked']]
    return {'pass': bool(gating) and all(l['ok'] for l in gating),
            'n_gating': len(gating), 'n_streams': streams,
            'legs': legs}


def cert_report(tag, net, partner_net, sims=None, streams=None):
    """Certify one side's candidate: candidate on its side, partner
    on the other.  partner_net REQUIRED (a depth battery without an
    opponent model is meaningless) — None returns a hard FAIL."""
    if partner_net is None:
        return {'pass': False, 'n_gating': 1, 'n_streams': 0,
                'legs': [{'fixture': '_partner', 'ok': False,
                          'tracked': False,
                          'detail': 'no partner net supplied'}]}
    fp_net = net if tag == 'fp' else partner_net
    sh_net = partner_net if tag == 'fp' else net
    return depth_battery(fp_net, sh_net, sims=sims, streams=streams)


def pick_last_passing(cert_by_epoch):
    """Index of the LAST epoch whose cert passed, else None."""
    best = None
    for i, rep in enumerate(cert_by_epoch):
        if rep and rep.get('pass'):
            best = i
    return best


def summarize(rep):
    return {'pass': rep['pass'],
            'fails': ['%s: %s' % (l['fixture'], l['detail'])
                      for l in rep['legs']
                      if not l['ok'] and not l['tracked']]}


if __name__ == '__main__':
    import sys
    import ai
    fp = ai.load_value_net(sys.argv[1], side='fp')
    sh = ai.load_value_net(sys.argv[2], side='sh')
    fp.eval()
    sh.eval()
    sims = int(sys.argv[3]) if len(sys.argv) > 3 else None
    streams = int(sys.argv[4]) if len(sys.argv) > 4 else None
    rep = depth_battery(fp, sh, sims=sims, streams=streams)
    print(json.dumps(summarize(rep), indent=1))
    for leg in rep['legs']:
        print(' %-22s %s%s  qgap=%s ngap=%s  %s'
              % (leg['fixture'], 'OK ' if leg['ok'] else 'FAIL',
                 ' (trk)' if leg['tracked'] else '',
                 leg.get('q_top2_gap_med'),
                 leg.get('n_top2_gap_med'),
                 leg.get('detail', '')))
