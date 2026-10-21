# -*- coding: utf-8 -*-
"""Behavioral SMOKE GATE for the v33 TD loop (2026-08-04; MANUAL
tool since 2026-08-11 — the loop no longer auto-invokes it, run it by
hand before launches).

WHY THIS EXISTS: component-level gates (prior calibration, holdout
agreement, fixtures replayed offline, unit suites) repeatedly passed
while LIVE PLAY was broken — latest the v32jt seeds whose cross-side
value heads disagree (v_fp(s) + v_sh(s) far from 0) and produced
garbage games; before that, selection-formula changes validated
offline against a different regime than live.  The missing check is
BEHAVIORAL: play actual games with the actual stack (same nets, same
episode entry ai._run_single_episode, same env pins as the loop) and
measure what comes out.

Checks (each prints value + threshold + PASS/FAIL/WARN):
  1. FIXTURE BEHAVIOR (live search, temp 0, ~400 sims):
     data/behavior_fixtures — siege_choice_erebor must pick
     retreat_into_siege (hard bar); erebor_recapture value-ORDERING
     hard bar (best evaluated Q = iron-hills muster); mordor_push and
     erebor_seal hard-run; wr_recapture tracked-only; declare_burn_*
     must NOT pick a ring-metric-burning declare (metric delta < 0
     with no heal / gateway / activation compensation;
     ai._ring_progress_metric is the judge).
  2. SMOKE GAMES (default 6 games, 300 sims, max_turns 40,
     deterministic per-game seeds, profile pins from
     run_td_v33loop.NET_KEYS): elven-ring usage, declare rate, FP
     siege choices, game length, winner split, played-vs-argmax rate,
     mean per-decision regret, idle separations, backward/Moria
     declares, army oscillation — thresholds calibrated from this
     week's healthy/diseased measurements — plus a RARITY TABLE
     (self-play rate vs corpus reference; >= 3x corpus auto-WARNs a
     signature even when under its FAIL bar).

Exit code 0 only if there are ZERO FAILs (WARNs allowed, listed).
JSON report: data/v33loop/smoke_gate_last.json (or --json PATH).

Usage:
  python launch_smoke_gate.py [--games 6] [--sims 300] [--json out]

ASCII only (cp1252 console).  CPU only; <= 6 workers (a training
agent may own the GPU/remaining cores).
"""
import argparse
import glob
import hashlib
import json
import os
import pickle
import sys
import time

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
# CPU only ('-1', never '': the '' form makes torch report cuda with
# 0 devices).  Hard-set: the gate must never touch the GPU.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# PYTHONHASHSEED: a no-op for THIS interpreter (only takes effect
# pre-start) but NOT for the spawn-context _game_worker children,
# which are fresh interpreters that honor the inherited value — and
# with auto-invocation retired nothing else exports it.  For a fully
# reproducible in-process fixture leg too, export it before launch.
os.environ.setdefault('PYTHONHASHSEED', '0')
# live-stack env pins, exactly what run_td_v33sp workers run under
os.environ.setdefault('WOTR_TURN_HORIZON', '1')
os.environ['WOTR_NET_ARCH'] = 'base'
os.environ['WOTR_TREE_REUSE'] = '1'

# profile pins come from the loop itself — single source of truth
from run_td_v33loop import LOOP, NET_KEYS, _nm  # noqa: E402

FIXTURE_DIR = os.path.join(ROOT, 'data', 'behavior_fixtures')
SMOKE_TRACE_DIR = os.path.join(LOOP, 'smoke_gate_traces')
REPORT_JSON = os.path.join(LOOP, 'smoke_gate_last.json')

# ---- thresholds ----
# RING BARS (USER order 2026-08-11) = the v37 extraction ledger's
# HONEST rates (fp 1.9010 / sh 1.1412 uses/game) x 1.68 (the fp band's
# original generosity): fp 3.2, sh 1.9.
ELVEN_FP_PER_GAME_THR = 3.2   # honest corpus fp 1.9010 x 1.68
ELVEN_SH_PER_GAME_THR = 1.9   # honest corpus sh 1.1412 x 1.68
DECLARES_PER_GAME_THR = 3.0   # human ~2.0; disease 2.9-3.6
FP_FIELD_RATE_THR = 0.25   # healthy ~8-15%; disease 45%
MEDIAN_TURNS_MIN = 7       # 9-turn all-blowouts = value-asymmetry sig
LONG_GAME_TURNS = 10       # at least one game must reach this
ARGMAX_RATE_LO = 0.55      # both extremes are known pathologies
ARGMAX_RATE_HI = 0.97
REGRET_MEAN_THR = 0.05
IDLE_SEP_PER_GAME_THR = 0.5
OSC_PER_GAME_THR = 1.0     # SH military ping-pong disease
MORIA_DECLARE_MAX = 1      # corpus 0.32% of games
# corpus reference rates for the rarity table; >= 3x -> auto-WARN
# (ring references = the honest v37 rates above).
CORPUS_REF = (('declares_per_game', 'declares_per_game', 2.0),
              ('fp_rings_per_game', 'elven_ring_fp_per_game', 1.9010),
              ('sh_rings_per_game', 'elven_ring_sh_per_game', 1.1412),
              ('moria_declare_game_rate', 'moria_declares', 0.0032),
              ('fp_field_battle_rate', 'fp_field_battle_rate', 0.125),
              # 858-game rendered measurement: 1.689/game, 70% of
              # games >=1, SH 1.32 (armies staging IS oscillatory)
              ('army_oscillations_per_game',
               'army_oscillations_per_game', 1.69))
RARITY_WARN_X = 3.0

MAX_WORKERS = 6            # hard cap - the training loop owns the rest


def _sha12(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:12]


def _load_live_net(side, prof):
    """ONE loader (HUNT-B F1): this duplicated the sp-worker loader,
    but the workers moved to ai.load_value_net on 2026-08-10 (CONF-4
    DARK fix) and this copy silently kept the old semantics — no
    head_act reconstruction (head from env luck) and no confidence
    provider registration, so the GATE certified under a different
    config than play.  The gate must load exactly like play does."""
    import ai
    net = ai.load_value_net(_nm('live', side, prof), side=side)
    net.eval()
    return net


# ------------------------------------------------------------------
# Check 1: fixture behavior (live search)
# ------------------------------------------------------------------

# Single implementation of the declare-burn judge.  It LIVES in
# jt_cert (an import-light leaf) and is imported here — not the other
# way around — because importing THIS module executes env pins and
# the run_td_v33loop chain, side effects a certifying training
# process must never inherit.
from jt_cert import _declare_violation  # noqa: E402


def check_fixtures(sims, results):
    import numpy as np
    import ai
    from game_state import Side
    print('[1] FIXTURE BEHAVIOR (live search, temp 0, %d sims)'
          % sims, flush=True)
    fp_profs = [p for s, p in NET_KEYS if s == 'fp']
    sh_profs = [p for s, p in NET_KEYS if s == 'sh']
    nets = {'fp': {}, 'sh': {}}
    for p in fp_profs:
        nets['fp'][p] = _load_live_net('fp', p)
    for p in sh_profs:
        nets['sh'][p] = _load_live_net('sh', p)
    prim_fp = nets['fp'][fp_profs[0]]
    prim_sh = nets['sh'][sh_profs[0]]
    names = (['siege_choice_erebor', 'mordor_push', 'erebor_seal',
              'erebor_recapture', 'wr_recapture']
             + sorted(os.path.basename(p)[:-4] for p in glob.glob(
                 os.path.join(FIXTURE_DIR, 'declare_burn_*.pkl'))))
    for name in names:
        path = os.path.join(FIXTURE_DIR, name + '.pkl')
        rname = 'fixture_' + name
        if not os.path.exists(path):
            print('    %s: MISSING -> FAIL' % name, flush=True)
            results.append({'name': rname, 'status': 'FAIL',
                            'error': 'fixture file missing'})
            continue
        try:
            with open(path, 'rb') as f:
                d = pickle.load(f)
            st, meta = d['state'], (d.get('meta') or {})
            if st.board is None:
                import board
                st.board = board.load_board()
            side = (Side.FREE_PEOPLES
                    if str(meta.get('side', 'FP')) == 'FP'
                    else Side.SHADOW)
            # shaping regime from the fixture meta (manifest contract);
            # nets = the live roster (meta profile if trained, else the
            # primary live pair — the stack that will actually play)
            ai.PLAYER_PROFILES['fp'] = meta.get('fp_profile')
            ai.PLAYER_PROFILES['sh'] = meta.get('sh_profile')
            fp_net = nets['fp'].get(meta.get('fp_profile'), prim_fp)
            sh_net = nets['sh'].get(meta.get('sh_profile'), prim_sh)
            ai._TREE_CACHE['child'] = None
            ai._TREE_CACHE['sig'] = None
            is_declare = name.startswith('declare_burn')
            if is_declare:
                st._declare_phase = True
            try:
                act, _vi = ai.mcts_action_with_policy(
                    st, side, fp_net, sh_net, num_simulations=sims,
                    temperature=0.0, add_noise=False)
            finally:
                if is_declare:
                    st._declare_phase = False
            picked = (act.action_type if act is not None else None)
            if name == 'siege_choice_erebor':
                ok = picked == 'retreat_into_siege'
                detail = 'picked %s (expect retreat_into_siege)' % picked
            elif name == 'mordor_push':
                ok = picked == 'fellowship_move'
                detail = 'picked %s (expect fellowship_move)' % picked
            elif name == 'erebor_seal':
                ok = (picked in ('army_move', 'army_attack')
                      and 'erebor' in str(
                          (act.params or {}).get('to', '')).lower())
                detail = 'picked %s (expect dale->erebor seal)' % picked
            elif name == 'wr_recapture':
                # tracked-only training meter: ring-convert survival
                # line (turn-end death behind 6 SH dice; see autopsy)
                ok = True
                detail = ('picked %s (tracking; winning line starts '
                          'elven_ring)' % picked)
            elif name == 'erebor_recapture':
                # hard bar = value ORDERING (currency-fix tripwire):
                # best-Q arm must be an iron-hills muster.  The PICK
                # flips only at >=4000 sims until training widens the
                # death/survival value gap — tracked, not required.
                q9 = np.asarray(_vi['q_values'], dtype=float)
                # unvisited arms carry the 0.0 Q sentinel, which would
                # outrank every genuinely evaluated arm in this
                # all-negative fixture — mask them out of the ordering
                vc9 = np.asarray(_vi.get('visit_counts',
                                         [1] * len(q9)), dtype=int)
                q9 = np.where(vc9 > 0, q9, -np.inf)
                ae9 = np.asarray(_vi['action_encs'], dtype=np.float32)
                accs = {np.asarray(ai.encode_action(x, st, side),
                                   dtype=np.float32).tobytes():
                        x for x in ai.get_legal_actions(st, side)}
                bj = int(np.argmax(q9))
                ba = accs.get(ae9[bj].tobytes())
                breg = str((getattr(ba, 'params', None) or {})
                           .get('region', '')).lower()
                ok = (ba is not None
                      and ba.action_type.startswith('muster')
                      and 'iron hills' in breg)
                detail = ('best-Q=%s@%s picked=%s (ordering%s; pick '
                          'needs >=4k sims)'
                          % (getattr(ba, 'action_type', '?'), breg,
                             picked, ' OK' if ok else ' BROKEN'))
            else:
                if picked == 'fellowship_declare':
                    burning, detail = _declare_violation(ai, st, act)
                    ok = not burning
                else:
                    ok, detail = True, 'picked %s (no declare)' % picked
            status = 'PASS' if ok else 'FAIL'
            print('    %s: %s -> %s' % (name, detail, status),
                  flush=True)
            results.append({'name': rname, 'status': status,
                            'picked': str(picked), 'detail': detail})
        except Exception as e:
            print('    %s: ERROR %r -> FAIL' % (name, e), flush=True)
            results.append({'name': rname, 'status': 'FAIL',
                            'error': repr(e)[:300]})
        finally:
            ai.PLAYER_PROFILES['fp'] = None
            ai.PLAYER_PROFILES['sh'] = None
    del nets, prim_fp, prim_sh


# ------------------------------------------------------------------
# Check 2: smoke games (parallel workers, spawn-safe)
# ------------------------------------------------------------------

def _game_worker(wid, jobs, sims, trace_dir, fp_profs, sh_profs,
                 seed0, out_q):
    """One smoke-game worker: the EXACT live-play stack
    (run_td_v33sp._worker semantics — same env pins, same nets, same
    episode entry, per-game deterministic seeds)."""
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['WOTR_NET_ARCH'] = 'base'
    os.environ['WOTR_TREE_REUSE'] = '1'
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    os.environ.setdefault('WOTR_TURN_HORIZON', '1')
    sys.path.insert(0, SRC)
    import random
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import ai
    ai.DEVICE = torch.device('cpu')
    try:
        fp_nets = {p: _load_live_net('fp', p) for p in fp_profs}
        sh_nets = {p: _load_live_net('sh', p) for p in sh_profs}
    except Exception as e:
        out_q.put(('e', wid, 'net load failed: %s' % repr(e)[:200]))
        return
    for gi in jobs:
        fp_p = fp_profs[gi % len(fp_profs)]
        sh_p = sh_profs[gi % len(sh_profs)]
        ai.PLAYER_PROFILES['fp'] = fp_p
        ai.PLAYER_PROFILES['sh'] = sh_p
        seed = seed0 + 1000 * gi
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        # %03d: resolve_trace_path globs '<stem>*.trace.gz' — an
        # unpadded g1 stem would also match g10+ at higher --games
        tp = os.path.join(trace_dir, 'smoke_g%03d.trace.gz' % gi)
        t0 = time.time()
        try:
            traj = ai._run_single_episode(
                fp_nets[fp_p], sh_nets[sh_p],
                {'max_turns': 40, 'mcts_sims': sims, 'shadow_vp': 10,
                 'trace_path': tp})
            g = dict(traj.get('game_stats') or {})
            g['game_idx'] = gi
            g['wall_s'] = round(time.time() - t0, 1)
            g['trace_path'] = ai.resolve_trace_path(tp) or tp
            g['profiles'] = {'fp': fp_p, 'sh': sh_p}
            out_q.put(('g', wid, g))
        except Exception as e:
            out_q.put(('e', wid, 'game %d: %s' % (gi, repr(e)[:250])))


def _analyze_trace(tp):
    """Behavioral counters from one smoke trace."""
    import numpy as np
    import ai
    import game_trace
    from board import sanitize
    tr = game_trace.load_trace(tp)
    steps = tr['steps']
    out = {'elven_ring_fp': 0, 'elven_ring_sh': 0, 'declares': 0,
           'siege_decisions': 0,
           'siege_field': 0, 'n_stats': 0, 'n_argmax': 0,
           'regrets': [], 'idle_sep': 0, 'back_declare': 0,
           'moria_declare': 0, 'oscillations': 0, 'osc_greedy': 0,
           'meta': {k: tr.get('meta', {}).get(k)
                    for k in ('winner', 'victory_type', 'turns')}}
    end_turn = int(tr.get('meta', {}).get('turns') or
                   max((s.get('turn') or 0) for s in steps))
    prev_state = None
    moves = {'FP': [], 'SH': []}
    attacks = []
    seps = []
    for i, s in enumerate(steps):
        if s.get('kind') == 'action':
            at = str(s.get('action_type'))
            p = s.get('params') or {}
            if at == 'elven_ring':
                out['elven_ring_sh' if s.get('side') == 'SH'
                    else 'elven_ring_fp'] += 1
            elif at == 'fellowship_declare':
                out['declares'] += 1
                reg = sanitize(str(p.get('region', '')))
                if reg == 'moria':
                    out['moria_declare'] += 1
                if prev_state is not None \
                        and s.get('state') is not None:
                    try:
                        pre, post = prev_state, s['state']
                        d9 = (ai._ring_progress_metric(post)
                              - ai._ring_progress_metric(pre))
                        comp = (post.fellowship.corruption
                                < pre.fellowship.corruption
                                or post.fellowship.mordor_entered
                                or reg in ('morannon', 'minas morgul')
                                or ({n for n, v in
                                     post.political.active.items()
                                     if v}
                                    - {n for n, v in
                                       pre.political.active.items()
                                       if v}))
                        # lost positions exempt (all arms terminal-bad:
                        # prior arbitrates between equally dead lines)
                        st9 = s.get('search_stats') or {}
                        # 'q' is a float32 ndarray: `or []` truthiness
                        # raises on >1 element (and the except below
                        # would swallow the whole back-declare count)
                        qa9 = st9.get('q')
                        q9 = ([float(v) for v in qa9]
                              if qa9 is not None and len(qa9) else [])
                        lost = bool(q9) and max(q9) <= -0.85
                        if d9 < -1e-9 and not comp and not lost:
                            out['back_declare'] += 1
                    except Exception:
                        pass
            elif at == 'companion_separate':
                ph = (getattr(prev_state, 'pending_hunt', None)
                      if prev_state is not None else None)
                if not (isinstance(ph, dict)
                        and ph.get('phase') == 'hobbit_place'):
                    names = (p.get('companions')
                             or ([p.get('companion')]
                                 if p.get('companion') else []))
                    seps.append((i, int(s.get('turn') or 0),
                                 [str(n) for n in names]))
            elif at == 'army_move':
                sd = s.get('side')
                fr = sanitize(str(p.get('from', '') or ''))
                to = sanitize(str(p.get('to', '') or ''))
                if sd in moves and fr and to and fr != to:
                    ss9 = s.get('search_stats')
                    am = True
                    if isinstance(ss9, dict) and 'n' in ss9 \
                            and 'played' in ss9:
                        am = int(ss9['played']) == int(np.argmax(
                            np.asarray(ss9['n'], dtype=np.int64)))
                    moves[sd].append((i, fr, to, am))
            elif at == 'army_attack':
                attacks.append(
                    (i, sanitize(str(p.get('from', '') or '')),
                     sanitize(str(p.get('to', '') or ''))))
            if at in ('retreat_into_siege', 'combat_no_retreat') \
                    and prev_state is not None:
                pc = getattr(prev_state, 'pending_combat', None)
                if isinstance(pc, dict) \
                        and pc.get('phase') == 'defender_siege_choice':
                    ds = pc.get('deciding_side')
                    if getattr(ds, 'name', str(ds)) == 'FREE_PEOPLES':
                        out['siege_decisions'] += 1
                        if at == 'combat_no_retreat':
                            out['siege_field'] += 1
            ss = s.get('search_stats')
            if isinstance(ss, dict) and 'n' in ss and 'played' in ss:
                out['n_stats'] += 1
                if int(ss['played']) == int(np.argmax(
                        np.asarray(ss['n'], dtype=np.int64))):
                    out['n_argmax'] += 1
            if s.get('regret') is not None:
                out['regrets'].append(float(s['regret']))
        if s.get('state') is not None:
            prev_state = s['state']
    for i, t_sep, names in seps:
        for nm in names:
            ref = None
            for s2 in steps[i + 1:]:
                if s2.get('kind') != 'action':
                    continue
                sp = str(s2.get('params') or {})
                at2 = str(s2.get('action_type'))
                if at2 == 'crown_aragorn':
                    sp += ' Strider Aragorn'
                elif at2 == 'transform_gandalf':
                    # separated companion is recorded by full name
                    sp += ' Gandalf the Grey'
                if nm and nm in sp:
                    ref = int(s2.get('turn') or 0)
                    break
            if ref is None:
                if end_turn - t_sep >= 2:
                    out['idle_sep'] += 1
            elif ref - t_sep > 6:
                out['idle_sep'] += 1
    for sd in ('FP', 'SH'):
        ms = moves[sd]
        for k, (i1, a, b, am1) in enumerate(ms):
            nxt = [m for m in ms[k + 1:] if m[1] == b][:2]
            for i2, _f, t2, am2 in nxt:
                if t2 == a and not any(
                        i1 < ia < i2 and (fa in (a, b) or ta in (a, b))
                        for ia, fa, ta in attacks):
                    out['oscillations'] += 1
                    # both legs greedy = the disease; a sampled leg is
                    # temperature exploration (measured: v33 pairs all
                    # had a sampled leg, v31jt greedy rate 0.17/g)
                    if am1 and am2:
                        out['osc_greedy'] += 1
                    break
    del tr
    return out


def _metric(results_list, name, value, thr_text, status, extra=None):
    row = {'name': name, 'value': value, 'threshold': thr_text,
           'status': status}
    if extra:
        row.update(extra)
    results_list.append(row)
    print('    %-28s = %-10s (%s) %s'
          % (name, value, thr_text, status), flush=True)


def check_smoke_games(n_games, sims, workers, results, reuse=False):
    import multiprocessing as mp
    import numpy as np
    print('[2] SMOKE GAMES: %d games, %d sims, %d workers, '
          'max_turns 40, pins fp=%s sh=%s'
          % (n_games, sims, workers,
             ','.join(p for s, p in NET_KEYS if s == 'fp'),
             ','.join(p for s, p in NET_KEYS if s == 'sh')),
          flush=True)
    fp_profs = [p for s, p in NET_KEYS if s == 'fp']
    sh_profs = [p for s, p in NET_KEYS if s == 'sh']
    games, errors = [], []
    existing = sorted(glob.glob(os.path.join(SMOKE_TRACE_DIR,
                                             'smoke_g*.trace.gz')))
    if reuse and len(existing) >= n_games:
        # deterministic per-game seeds -> a replay reproduces these
        # traces byte-for-byte; reuse re-scores without re-playing
        print('    REUSING %d existing traces (--reuse-traces; '
              'deterministic seeds)' % n_games, flush=True)
        import re as _re
        for tp in existing[:n_games]:
            m9 = _re.match(r'smoke_g(\d+)', os.path.basename(tp))
            gi = int(m9.group(1)) if m9 else 0
            games.append({'game_idx': gi, 'trace_path': tp,
                          'profiles':
                          {'fp': fp_profs[gi % len(fp_profs)],
                           'sh': sh_profs[gi % len(sh_profs)]}})
    elif reuse:
        # NEVER fall through to the fresh path here: it wipes the very
        # traces the caller asked to re-score before replaying
        sys.exit('--reuse-traces: have %d traces, need %d '
                 '(run without --reuse-traces to replay)'
                 % (len(existing), n_games))
    else:
        # fresh trace dir per gate run
        if os.path.isdir(SMOKE_TRACE_DIR):
            for f in glob.glob(os.path.join(SMOKE_TRACE_DIR,
                                            '*.trace.gz')):
                try:
                    os.remove(f)
                except OSError:
                    pass
        os.makedirs(SMOKE_TRACE_DIR, exist_ok=True)
        ctx = mp.get_context('spawn')
        out_q = ctx.Queue()
        procs = []
        for w in range(workers):
            jobs = [i for i in range(n_games) if i % workers == w]
            if not jobs:
                continue
            p = ctx.Process(target=_game_worker,
                            args=(w, jobs, sims, SMOKE_TRACE_DIR,
                                  fp_profs, sh_profs, 777000, out_q))
            p.start()
            procs.append(p)
        t0 = time.time()
        import queue as _qmod
        while len(games) + len(errors) < n_games:
            try:
                m = out_q.get(timeout=120)
            except _qmod.Empty:
                if not any(p.is_alive() for p in procs):
                    print('    all workers dead with %d/%d games — '
                          'aborting collection'
                          % (len(games), n_games), flush=True)
                    break
                continue
            if m[0] == 'g':
                games.append(m[2])
                print('    game %s: winner=%s turns=%s vtype=%s '
                      '(%.0fs, cum %.0f min)'
                      % (m[2].get('game_idx'), m[2].get('winner'),
                         m[2].get('turns'), m[2].get('victory_type'),
                         m[2].get('wall_s', 0),
                         (time.time() - t0) / 60), flush=True)
            else:
                errors.append(m[2])
                print('    worker %d ERROR: %s' % (m[1], m[2]),
                      flush=True)
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()

    # ---- per-trace behavioral counters ----
    agg = {'elven_ring_fp': 0, 'elven_ring_sh': 0, 'declares': 0,
           'siege_decisions': 0,
           'siege_field': 0, 'n_stats': 0, 'n_argmax': 0,
           'regrets': [], 'idle_sep': 0, 'back_declare': 0,
           'moria_declare': 0, 'oscillations': 0, 'osc_greedy': 0,
           'moria_games': 0}
    game_rows = []
    n_analyzed = 0
    for g in sorted(games, key=lambda r: r.get('game_idx', 0)):
        tp = g.get('trace_path')
        c = None
        if tp and os.path.exists(tp):
            try:
                c = _analyze_trace(tp)
            except Exception as e:
                g['analyze_error'] = repr(e)[:200]
        if c is not None:
            for k in ('winner', 'victory_type', 'turns'):
                if g.get(k) is None:
                    g[k] = c['meta'].get(k)
        row = {'game_idx': g.get('game_idx'),
               'winner': g.get('winner'),
               'victory_type': g.get('victory_type'),
               'turns': g.get('turns'),
               'profiles': g.get('profiles'),
               'wall_s': g.get('wall_s'),
               'trace': os.path.basename(str(g.get('trace_path')))}
        if g.get('analyze_error'):
            row['analyze_error'] = g['analyze_error']
        if c is not None:
            n_analyzed += 1
            for k in ('elven_ring_fp', 'elven_ring_sh', 'declares',
                      'siege_decisions',
                      'siege_field', 'n_stats', 'n_argmax',
                      'idle_sep', 'back_declare', 'moria_declare',
                      'oscillations', 'osc_greedy'):
                agg[k] += c[k]
            agg['regrets'].extend(c['regrets'])
            if c['moria_declare'] > 0:
                agg['moria_games'] += 1
            row.update({k: c[k] for k in
                        ('elven_ring_fp', 'elven_ring_sh', 'declares',
                         'siege_decisions',
                         'siege_field', 'idle_sep', 'back_declare',
                         'moria_declare', 'oscillations',
                         'osc_greedy')})
        game_rows.append(row)

    metrics = []
    rarity = []
    ng = len(games)
    _metric(metrics, 'games_completed', '%d/%d' % (ng, n_games),
            '== %d' % n_games,
            'PASS' if (ng == n_games and not errors) else 'FAIL',
            {'errors': errors[:6]})
    # behavioral rates divide by ANALYZED traces: an unanalyzable
    # trace contributing zero counters over an ng denominator would
    # silently deflate every rate
    _metric(metrics, 'traces_analyzed', '%d/%d' % (n_analyzed, ng),
            '== %d' % ng, 'PASS' if n_analyzed == ng else 'FAIL')
    na = float(n_analyzed or 1)
    if ng > 0:
        elv_fp = agg['elven_ring_fp'] / na
        _metric(metrics, 'elven_ring_fp_per_game', round(elv_fp, 2),
                '<= %.1f' % ELVEN_FP_PER_GAME_THR,
                'PASS' if elv_fp <= ELVEN_FP_PER_GAME_THR else 'FAIL')
        elv_sh = agg['elven_ring_sh'] / na
        _metric(metrics, 'elven_ring_sh_per_game', round(elv_sh, 2),
                '<= %.1f' % ELVEN_SH_PER_GAME_THR,
                'PASS' if elv_sh <= ELVEN_SH_PER_GAME_THR else 'FAIL')
        decl = agg['declares'] / na
        _metric(metrics, 'declares_per_game', round(decl, 2),
                '<= %.1f' % DECLARES_PER_GAME_THR,
                'PASS' if decl <= DECLARES_PER_GAME_THR else 'FAIL')
        if agg['siege_decisions'] > 0:
            fr = agg['siege_field'] / float(agg['siege_decisions'])
            _metric(metrics, 'fp_field_battle_rate',
                    '%.2f (%d/%d)' % (fr, agg['siege_field'],
                                      agg['siege_decisions']),
                    '<= %.2f' % FP_FIELD_RATE_THR,
                    'PASS' if fr <= FP_FIELD_RATE_THR else 'FAIL')
        else:
            _metric(metrics, 'fp_field_battle_rate', 'n/a (0 dec)',
                    '<= %.2f' % FP_FIELD_RATE_THR, 'PASS')
        turns = sorted(int(g.get('turns') or 0) for g in games)
        med = float(np.median(turns))
        _metric(metrics, 'median_turns', med,
                '>= %d' % MEDIAN_TURNS_MIN,
                'PASS' if med >= MEDIAN_TURNS_MIN else 'FAIL',
                {'turns': turns})
        _metric(metrics, 'max_turns_reached', max(turns),
                '>= %d (one game)' % LONG_GAME_TURNS,
                'PASS' if max(turns) >= LONG_GAME_TURNS else 'FAIL')
        fp_w = sum(1 for g in games
                   if str(g.get('winner')).lower() == 'fp')
        sh_w = sum(1 for g in games
                   if str(g.get('winner')).lower() == 'sh')
        decided = fp_w + sh_w
        split = 'fp %d / sh %d (of %d)' % (fp_w, sh_w, ng)
        # the fp_ring-vs-sh_military pin sweeps SH under EVERY seed
        # pair incl. known-good v31jt - a sweep alone is
        # config-structural; the DISASTER signature is sweep +
        # blowout-length games, which stays FAIL.  (No near-sweep WARN:
        # at N=6 a 5/6 split is noise, USER 2026-08-11.)
        if (max(fp_w, sh_w) == decided == ng and ng >= 6
                and med < 10):
            w_status = 'FAIL'
        else:
            w_status = 'PASS'
        _metric(metrics, 'winner_split', split,
                'no sweep w/ median<10 turns at N>=6', w_status)
        if agg['n_stats'] > 0:
            ar = agg['n_argmax'] / float(agg['n_stats'])
            _metric(metrics, 'played_eq_visit_argmax',
                    '%.3f (%d/%d)' % (ar, agg['n_argmax'],
                                      agg['n_stats']),
                    '[%.2f, %.2f]' % (ARGMAX_RATE_LO, ARGMAX_RATE_HI),
                    'PASS' if ARGMAX_RATE_LO <= ar <= ARGMAX_RATE_HI
                    else 'FAIL')
        else:
            _metric(metrics, 'played_eq_visit_argmax', 'n/a',
                    'search stats present', 'FAIL',
                    {'note': 'no search_stats in traces'})
        if agg['regrets']:
            rm = float(np.mean(agg['regrets']))
            _metric(metrics, 'mean_decision_regret', round(rm, 4),
                    '<= %.2f' % REGRET_MEAN_THR,
                    'PASS' if rm <= REGRET_MEAN_THR else 'FAIL',
                    {'n_decisions': len(agg['regrets'])})
        else:
            _metric(metrics, 'mean_decision_regret', 'n/a',
                    'regret present', 'FAIL',
                    {'note': 'no regret rows in traces'})
        idle = agg['idle_sep'] / na
        _metric(metrics, 'idle_separations_per_game', round(idle, 2),
                '<= %.1f' % IDLE_SEP_PER_GAME_THR,
                'PASS' if idle <= IDLE_SEP_PER_GAME_THR else 'FAIL')
        _metric(metrics, 'backward_declares', agg['back_declare'],
                '== 0',
                'PASS' if agg['back_declare'] == 0 else 'FAIL')
        _metric(metrics, 'moria_declares', agg['moria_declare'],
                '<= %d (whole set)' % MORIA_DECLARE_MAX,
                'PASS' if agg['moria_declare'] <= MORIA_DECLARE_MAX
                else 'FAIL')
        oscg = agg['osc_greedy'] / na
        _metric(metrics, 'greedy_oscillations_per_game',
                round(oscg, 2), '<= %.1f' % OSC_PER_GAME_THR,
                'PASS' if oscg <= OSC_PER_GAME_THR else 'FAIL')
        osc = agg['oscillations'] / na
        _metric(metrics, 'army_oscillations_per_game', round(osc, 2),
                'rarity table (incl. temperature exploration)', 'PASS')
        rates = {'declares_per_game': agg['declares'] / na,
                 'fp_rings_per_game':
                 agg['elven_ring_fp'] / na,
                 'sh_rings_per_game':
                 agg['elven_ring_sh'] / na,
                 'army_oscillations_per_game':
                 agg['oscillations'] / na,
                 'moria_declare_game_rate':
                 agg['moria_games'] / na,
                 'fp_field_battle_rate':
                 (agg['siege_field'] / float(agg['siege_decisions'])
                  if agg['siege_decisions'] else 0.0)}
        print('    RARITY TABLE (self-play vs corpus reference):',
              flush=True)
        print('      %-26s %9s %9s %8s' %
              ('signature', 'selfplay', 'corpus', 'ratio'),
              flush=True)
        esc = []
        for sig, mname, ref in CORPUS_REF:
            rate = rates[sig]
            ratio = rate / ref if ref > 0 else 0.0
            hot = ratio >= RARITY_WARN_X
            print('      %-26s %9.3f %9.3f %7.1fx%s'
                  % (sig, rate, ref, ratio,
                     '  >=3x-corpus' if hot else ''), flush=True)
            rarity.append({'signature': sig, 'selfplay': round(rate, 4),
                           'corpus': ref, 'ratio': round(ratio, 2),
                           'auto_warn': bool(hot)})
            if hot:
                for m in metrics:
                    if m['name'] == mname and m['status'] == 'PASS':
                        m['status'] = 'WARN'
                        m['rarity'] = '%.1fx corpus' % ratio
                        esc.append(mname)
        if esc:
            print('      auto-WARN (>=3x corpus): %s'
                  % ', '.join(esc), flush=True)
    worst = ('FAIL' if any(m['status'] == 'FAIL' for m in metrics)
             else 'WARN' if any(m['status'] == 'WARN'
                                for m in metrics) else 'PASS')
    results.append({'name': 'smoke_games', 'status': worst,
                    'metrics': metrics, 'games': game_rows,
                    'rarity_table': rarity,
                    'n_errors': len(errors)})


# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', type=int, default=6)
    ap.add_argument('--sims', type=int, default=300)
    ap.add_argument('--fixture-sims', type=int, default=400)
    ap.add_argument('--json', default=REPORT_JSON)
    ap.add_argument('--workers', type=int, default=MAX_WORKERS)
    ap.add_argument('--reuse-traces', action='store_true',
                    help='re-score existing smoke traces instead of '
                         're-playing (seeds are deterministic, so a '
                         'replay reproduces them; dev/re-verdict use)')
    a = ap.parse_args()
    n_games = max(1, a.games)
    workers = max(1, min(MAX_WORKERS, a.workers, n_games))

    import torch
    torch.set_num_threads(2)
    import ai
    ai.DEVICE = torch.device('cpu')

    t0 = time.time()
    print('=== WOTR PRE-LAUNCH SMOKE GATE ===', flush=True)
    nets_meta = {}
    for s, p in NET_KEYS:
        path = _nm('live', s, p)
        if not os.path.exists(path):
            print('missing live net: %s' % path, flush=True)
            sys.exit(2)
        nets_meta['%s_%s' % (s, p)] = {
            'path': path, 'sha256_12': _sha12(path)}
        print('net %s_%s: %s (sha %s)'
              % (s, p, os.path.basename(path),
                 nets_meta['%s_%s' % (s, p)]['sha256_12']),
              flush=True)

    results = []
    for fn, args in ((check_fixtures, (a.fixture_sims, results)),
                     (check_smoke_games,
                      (n_games, a.sims, workers, results,
                       a.reuse_traces))):
        try:
            fn(*args)
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({'name': fn.__name__, 'status': 'FAIL',
                            'error': repr(e)[:300]})

    def _walk_status(want):
        out = []
        for r in results:
            if r['status'] == want:
                out.append(r['name'])
            for m in r.get('metrics', []):
                if m['status'] == want:
                    out.append('smoke_games.%s' % m['name'])
        return out

    fails = [n for n in _walk_status('FAIL')
             if n != 'smoke_games']     # container mirrors its metrics
    warns = [n for n in _walk_status('WARN') if n != 'smoke_games']
    overall = 'FAIL' if fails else 'PASS'

    report = {'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
              'tool': 'launch_smoke_gate',
              'config': {'games': n_games, 'sims': a.sims,
                         'fixture_sims': a.fixture_sims,
                         'workers': workers,
                         'net_keys': [list(k) for k in NET_KEYS]},
              'nets': nets_meta,
              'checks': results,
              'failed_checks': fails,
              'warned_checks': warns,
              'overall': overall,
              'wall_min': round((time.time() - t0) / 60.0, 1)}
    try:
        with open(a.json, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=1)
        print('report: %s' % a.json, flush=True)
    except OSError as e:
        print('report write FAILED: %r' % e, flush=True)

    if warns:
        print('WARNs (%d): %s' % (len(warns), ', '.join(warns)),
              flush=True)
    if fails:
        print('FAILs (%d): %s' % (len(fails), ', '.join(fails)),
              flush=True)
    print('=== SMOKE GATE: %s (%.1f min) ==='
          % (overall, (time.time() - t0) / 60.0), flush=True)
    sys.exit(0 if overall == 'PASS' else 1)


if __name__ == '__main__':
    main()
