# -*- coding: utf-8 -*-
"""Per-batch behavioral METERS for the TD loop (TELEMETRY ONLY since
the loop-v2 gate removal, USER 2026-08-11: TRAIN-ALWAYS).

Stage A (post-play): census over the just-played batch's stored search
stats.  Stage B (post-update): probe-set comparison of the updated
nets against the seed reference.  Both produce verdict dicts the
driver LOGS - nothing quarantines, discards, reverts, or halts.
LoopState keeps the LR multiplier + history.

Pure logic (census_from_steps, stage_a_verdict, stage_b_compare) is
separated from IO so unit tests need no nets and no traces.  Traces
are keyed by FULL PATH (basenames repeat across batches).  ASCII only
(cp1252 console).
"""
import glob
import gzip
import json
import os
import pickle

import numpy as np

FSP_CAT = 6          # ae one-hot col: fellowship_move category

# Stage-A bands: (lo, hi) on RATES (scale-free for mini-runs); None =
# unbounded on that side.  Calibrated on healthy seed play (batch_001 /
# batch_004 census + smoke-gate bars) under fp=ring / sh=military.
# VALUE-UNIT METER (t1_q_mean) is head-calibration dependent: after
# any head/recalibration change these constants are WRONG (honest fp
# T1 ~ -1.4 tally currency, VALUE comms\07 sec 2) — the loop loads
# LOOP\stage_a_bands.json when present (re-derived from the new
# seeds' baseline play at adoption; load_bands_file()).
STAGE_A_BANDS = {
    'fsp_moves_per_game':   (9.0, 16.0),    # seed 11.2-12.8; disease 0.7-1.8
    'zero_fsp_rate':        (None, 0.05),   # seed 0/50; disease 14-30/50
    'fsp_prior_mass_mean':  (0.30, 0.45),   # seed 0.36; disease 0.027-0.06
    'fsp_visit_share_mean': (0.20, 0.40),   # seed 0.29; disease 0.03-0.07
    't1_q_mean':            (-0.35, 0.25),  # FP; seed -0.04..-0.12
    # SH T1 (VALUE comms\13: t1 bands must be PER-NET — honest values
    # differ by side); unbounded until stage_a_bands.json derives it
    't1_q_mean_sh':         (None, None),
    'declares_per_game':    (None, 3.0),    # smoke bar
    # DEFECT-1 lesson: bars must BRACKET healthy, never fence one
    # direction.  Floors: below 0.25 = logged census failure (usage
    # ~dead: the 0.17 mini-run and 0.0 production eras); [0.25, 0.7)
    # WARNs via the stage-A bands file.  Telemetry only (v2).
    # Ceilings (USER order 2026-08-11) = the v37 extraction ledger's
    # HONEST rates (fp 1.9010 / sh 1.1412 uses/game) x 1.68 (the fp
    # band's original generosity): fp 3.2, sh 1.9.
    'rings_fp_per_game':    (0.25, 3.2),
    'ring_coverage_fp':     (0.35, None),   # frac games >= 1 FP take
    # PRIMARY scale-free ring meter (VALUE corpus anchor 0.030
    # conditional): wide logged band; stage_a_bands.json may tighten
    # to [0.01, 0.08].  Telemetry only (v2).
    'take_rate_when_legal_fp': (0.005, 0.12),
    'rings_sh_per_game':    (None, 1.9),    # ceiling: 1.1412 x 1.68
    'argmax_rate':          (0.55, 0.97),   # smoke bars
}

# Stage-B bounds (battery v2, comms\03 frozen numbers)
STAGE_B_BOUNDS = {
    'b1_state_ratio_lo': 0.5,     # per-state fsp mass vs seed
    'b1_state_ratio_hi': 2.0,
    'b1_state_min_seed_mass': 0.02,
    'b1_action_shrink_max': 10.0,  # per-action prior shrink vs seed
    'b1_action_min_seed_prior': 0.01,
    'b1_aggregate_min': 0.6,      # mean fsp mass vs seed
    'b3_t1_swing_max': 0.15,      # per update
    'b3_t1_anchor_cap': 0.35,     # B3' escape: toward-anchor moves
                                  # allowed up to this hard cap
    'b3_t1_cum_warn': 0.30,       # vs seed (WARN, not FAIL)
    # B4 (VALUE one-table addendum, REGIME ruling = option b): the
    # bound is anchored drift — max |log prior ratio| vs the SEED
    # reference probe rows (no arena promotion in v2, so the anchor
    # never advances).  The RAW delta-head magnitude
    # (vs own base scorer) demotes to WARN + trend when an anchor is
    # supplied; it stays the B5 movement floor's numerator.
    'b4_delta_logit_max': 2.3,    # nats (~10x prior ratio)
    'b4_anchor_floor': 1e-3,      # actions must matter somewhere
    # ring-mass rise allowed up to max(ref*mult, ref+abs): an absolute
    # eps alone let a 3x relative rise pass at low seed mass (run-7
    # R0 retrain: 0.008 -> 0.0224 must FAIL)
    'ring_mass_mult': 1.5,
    'ring_mass_abs': 0.01,
    # paired per-update trust region (SEARCH comms\06 sec 4): KL(pre-
    # update prior || post-update prior) per probe state; healthy
    # band p50 .003-.009 / p95 .012-.032, damaging updates .196-1.49
    'kl_p50_max': 0.02,
    'kl_p95_max': 0.10,
}

def census_from_steps(steps_iter):
    """Behavioral counters for ONE game's steps (list of dicts).

    Returns raw per-game counters; aggregate with aggregate_census.
    FP-centric fellowship meters, both-sides rings/argmax."""
    g = {'fsp_played': 0, 'declares': 0, 'rings_fp': 0, 'rings_sh': 0,
         'n_stats': 0, 'n_argmax': 0, 'prior_mass': [],
         'visit_share': [], 't1_q': None,
         't1_q_sh': None,
         # DEFECT-1 leg-D: ring-legal proxy + 'fn' floor counters
         'ring_legal_fp': 0, 'fn_rows': 0, 'forced_visits': 0,
         'earned_on_forced': 0, 'n_dec_rows': 0,
         # CONF-4: per-arm support ('cf') meters; DARK-never-silent
         'conf_rows': 0, 'conf_dark_rows': 0, 'conf_means': [],
         'conf_uniform_novelty': 0}
    prev = None
    for s in steps_iter:
        st9 = s.get('state')
        if s.get('kind') != 'action':
            if st9 is not None:
                prev = st9
            continue
        at = str(s.get('action_type'))
        sd = s.get('side')
        if at == 'elven_ring':
            g['rings_sh' if sd == 'SH' else 'rings_fp'] += 1
        if sd in ('FP', 'SH'):
            g['n_dec_rows'] += 1
            # ring-legal proxy: acting FP while holding >= 1 ring
            # (pre-decision state); corpus take-rate anchor 0.030
            if sd == 'FP' and prev is not None and \
                    int(getattr(prev, 'fp_elven_rings', 0) or 0) > 0:
                g['ring_legal_fp'] += 1
            ss9f = s.get('search_stats')
            if isinstance(ss9f, dict) and 'fn' in ss9f:
                try:
                    fn = np.asarray(ss9f['fn'], dtype=np.float64)
                    nv = np.asarray(ss9f['n'], dtype=np.float64)
                    if fn.sum() > 0:
                        g['fn_rows'] += 1
                        g['forced_visits'] += int(fn.sum())
                        m9 = fn > 0
                        g['earned_on_forced'] += int(
                            np.maximum(nv[m9] - fn[m9], 0.0).sum())
                except Exception:
                    pass
            if isinstance(ss9f, dict):
                # CONF-4: 'cf' = per-arm support from the confidence
                # provider (SEARCH LAW 2).  Absent payload counts as
                # DARK — visible in the aggregate, never silent.
                if 'cf' in ss9f:
                    try:
                        cf = np.asarray(ss9f['cf'], dtype=np.float64)
                        if len(cf):
                            g['conf_rows'] += 1
                            g['conf_means'].append(float(cf.mean()))
                            if (float(cf.max() - cf.min()) < 0.05
                                    and float(cf.mean()) < 0.5):
                                g['conf_uniform_novelty'] += 1
                        else:
                            # empty payload = DARK, never silent
                            g['conf_dark_rows'] += 1
                    except Exception:
                        g['conf_dark_rows'] += 1
                else:
                    g['conf_dark_rows'] += 1
        if st9 is not None:
            prev = st9
        if sd != 'FP':
            ss2 = s.get('search_stats')
            if isinstance(ss2, dict) and 'n' in ss2 and 'played' in ss2:
                g['n_stats'] += 1
                if int(ss2['played']) == int(np.argmax(
                        np.asarray(ss2['n'], dtype=np.int64))):
                    g['n_argmax'] += 1
                if sd == 'SH' and g['t1_q_sh'] is None:
                    try:
                        g['t1_q_sh'] = float(np.asarray(
                            ss2['q'], dtype=np.float64)[
                                int(ss2['played'])])
                    except Exception:
                        pass
            continue
        if at in ('fellowship_move', 'mordor_move'):
            g['fsp_played'] += 1
        if at == 'fellowship_declare':
            g['declares'] += 1
        ss = s.get('search_stats')
        if not isinstance(ss, dict):
            continue
        try:
            ae = np.asarray(ss['ae'], dtype=np.float32)
            p = np.asarray(ss['p'], dtype=np.float64)
            n = np.asarray(ss['n'], dtype=np.float64)
            q = np.asarray(ss['q'], dtype=np.float64)
        except Exception:
            continue
        if 'played' in ss:
            g['n_stats'] += 1
            if int(ss['played']) == int(np.argmax(
                    np.asarray(n, dtype=np.int64))):
                g['n_argmax'] += 1
            pq = float(q[int(ss['played'])])
            if g['t1_q'] is None:
                g['t1_q'] = pq
        is_fsp = ae[:, FSP_CAT] > 0.5
        if is_fsp.any():
            g['prior_mass'].append(float(p[is_fsp].sum()))
            g['visit_share'].append(float(n[is_fsp].sum()
                                          / max(1.0, n.sum())))
    return g


def aggregate_census(per_game):
    """Aggregate per-game counters into the Stage-A meter dict."""
    n = max(1, len(per_game))

    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    return {
        'games': len(per_game),
        'fsp_moves_per_game': float(np.mean(
            [g['fsp_played'] for g in per_game])) if per_game else None,
        'zero_fsp_rate': ((sum(1 for g in per_game
                               if g['fsp_played'] == 0) / n)
                          if per_game else None),
        'fsp_prior_mass_mean': _mean(
            [_mean(g['prior_mass']) for g in per_game]),
        'fsp_visit_share_mean': _mean(
            [_mean(g['visit_share']) for g in per_game]),
        't1_q_mean': _mean([g['t1_q'] for g in per_game]),
        't1_q_mean_sh': _mean([g.get('t1_q_sh') for g in per_game]),
        'declares_per_game': float(np.mean(
            [g['declares'] for g in per_game])) if per_game else None,
        'rings_fp_per_game': float(np.mean(
            [g['rings_fp'] for g in per_game])) if per_game else None,
        'ring_coverage_fp': (sum(1 for g in per_game
                                 if g['rings_fp'] >= 1) / n)
        if per_game else None,
        'take_rate_when_legal_fp': (
            sum(g['rings_fp'] for g in per_game)
            / float(sum(g.get('ring_legal_fp', 0)
                        for g in per_game)))
        if sum(g.get('ring_legal_fp', 0) for g in per_game) > 0
        else None,
        # trend counters (not banded): floor health from 'fn'
        'free_cand_presence_rate': (
            sum(g.get('fn_rows', 0) for g in per_game)
            / float(max(1, sum(g.get('n_dec_rows', 0)
                               for g in per_game)))),
        'forced_visits_total': sum(g.get('forced_visits', 0)
                                   for g in per_game),
        'earned_on_forced_total': sum(g.get('earned_on_forced', 0)
                                      for g in per_game),
        # CONF-4 aggregates (not banded; driver logs the trend and
        # WARNs on >0.2 batch-over-batch conf_mean drop)
        'conf_mean': _mean([_mean(g.get('conf_means') or [])
                            for g in per_game]),
        'conf_uniform_novelty_rate': (
            sum(g.get('conf_uniform_novelty', 0) for g in per_game)
            / float(max(1, sum(g.get('conf_rows', 0)
                               for g in per_game)))),
        'conf_meter_state': (
            'LIVE' if sum(g.get('conf_rows', 0)
                          for g in per_game) > 0 else 'DARK'),
        'conf_dark_rows': sum(g.get('conf_dark_rows', 0)
                              for g in per_game),
        'rings_sh_per_game': float(np.mean(
            [g['rings_sh'] for g in per_game])) if per_game else None,
        'argmax_rate': (float(sum(g['n_argmax'] for g in per_game))
                        / max(1, sum(g['n_stats'] for g in per_game))),
    }


def batch_census(bdir):
    """IO wrapper: census over every trace in a batch dir (full-path
    keyed; per-game rows returned alongside the aggregate)."""
    rows = {}
    for tp in sorted(glob.glob(os.path.join(bdir, '*.trace.gz'))):
        try:
            with gzip.open(tp, 'rb') as f:
                tr = pickle.load(f)
            rows[tp] = census_from_steps(tr['steps'])
        except Exception as e:
            rows[tp] = {'error': repr(e)[:200]}
    good = [g for g in rows.values() if 'error' not in g]
    agg = aggregate_census(good)
    agg['read_errors'] = len(rows) - len(good)
    return agg, rows


def load_bands_file(path):
    """Bands override file: {'meter': [lo|null, hi|null], ...} —
    unknown meters are rejected loudly (a typo must not silently run
    different gates); missing meters keep their defaults."""
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    bands = dict(STAGE_A_BANDS)
    for k, v in raw.items():
        if k not in STAGE_A_BANDS:
            raise ValueError('stage_a_bands: unknown meter %r' % k)
        lo, hi = v
        bands[k] = (None if lo is None else float(lo),
                    None if hi is None else float(hi))
    return bands


def stage_a_verdict(census, bands=None):
    """(ok, failures) — failures = [(meter, value, (lo, hi)), ...].
    Meters with value None (no data) FAIL loudly: an absent meter must
    never silently pass (the win-rate-masked-everything lesson)."""
    bands = bands or STAGE_A_BANDS
    fails = []
    for meter, (lo, hi) in bands.items():
        v = census.get(meter)
        if v is None:
            fails.append((meter, None, (lo, hi)))
            continue
        if lo is not None and v < lo:
            fails.append((meter, v, (lo, hi)))
        elif hi is not None and v > hi:
            fails.append((meter, v, (lo, hi)))
    return (not fails), fails


def _rows_by_id(rows):
    return {r['id']: r for r in rows}


def paired_kl(pre_rows, post_rows):
    """Per-state KL(pre prior || post prior) over the shared action
    keys (menus regenerate identically for the same state)."""
    _check_probe_contracts(pre_rows, post_rows)
    post = _rows_by_id(post_rows)
    kls = []
    for r0 in pre_rows:
        r1 = post.get(r0['id'])
        if r1 is None:
            continue
        p0 = r0['delta_on']['priors']
        p1 = r1['delta_on']['priors']
        ks = [k for k in p0 if k in p1]
        if len(ks) < 2:
            continue
        a = np.asarray([max(p0[k], 1e-12) for k in ks])
        c = np.asarray([max(p1[k], 1e-12) for k in ks])
        a, c = a / a.sum(), c / c.sum()
        kls.append(float((a * np.log(a / c)).sum()))
    return kls


def anchored_logit_drift(anchor_rows, new_rows, floor=1e-3):
    """Max |log(p_new / p_anchor)| over actions that carry >= floor
    prior on either side (participation floor keeps 1e-5-noise from
    reading as drift; a real crush 0.5 -> 1e-9 still screams)."""
    _check_probe_contracts(anchor_rows, new_rows)
    anc = _rows_by_id(anchor_rows)
    worst = 0.0
    for r in new_rows:
        r0 = anc.get(r['id'])
        if r0 is None:
            continue
        p0 = r0['delta_on']['priors']
        p1 = r['delta_on']['priors']
        for k, v1 in p1.items():
            v0 = p0.get(k)
            if v0 is None or max(v0, v1) < floor:
                continue
            worst = max(worst, abs(float(np.log(
                max(v1, 1e-9) / max(v0, 1e-9)))))
    return worst


def stage_b_compare(ref_rows, new_rows, prev_t1=None, bounds=None,
                    pre_rows=None, anchor_t1=None, anchor_rows=None):
    """Battery-B comparison of probe rows (probe_metrics.py schema).

    ref_rows: SEED reference probe rows.  new_rows: post-update rows.
    prev_t1: {'fp': x, 'sh': y} T1 means from the PREVIOUS update's
    probe (None on the first update -> seed reference is the previous).
    pre_rows: PRE-update probe rows for the paired-KL trust region
    (None skips that check).
    anchor_t1: {'fp': x, 'sh': y} REALIZED-outcome calibration anchor
    (mean MC return over T1-comparable states of recent healthy
    batches).  B3': a swing > b3_t1_swing_max is allowed when it
    moves STRICTLY TOWARD the anchor and stays <= b3_t1_anchor_cap —
    legitimate calibration convergence must not read as thrash, but
    an unbounded jump is still an unstable operator.
    Returns verdict dict with ok / fails / warns / metrics."""
    _check_probe_contracts(ref_rows, new_rows, pre_rows or [], anchor_rows or [])
    b = dict(STAGE_B_BOUNDS)
    if bounds:
        b.update(bounds)
    ref = _rows_by_id(ref_rows)
    fails, warns = [], []
    ratios, ring_ref, ring_new = [], [], []
    t1 = {'fp': [], 'sh': []}
    t1_ref = {'fp': [], 'sh': []}
    max_delta = 0.0
    _n_matched9 = 0
    for r in new_rows:
        r0 = ref.get(r['id'])
        if r0 is None:
            continue
        _n_matched9 += 1
        max_delta = max(max_delta, float(r['max_abs_delta_logit']))
        if r['stratum'] == 't1_3':
            t1[r['side']].append(float(r['v_raw']))
            t1_ref[r['side']].append(float(r0['v_raw']))
        m0 = float(r0['delta_on']['fsp_mass'])
        m1 = float(r['delta_on']['fsp_mass'])
        if r['side'] == 'fp' and m0 >= b['b1_state_min_seed_mass']:
            ratio = m1 / m0
            ratios.append(ratio)
            if not (b['b1_state_ratio_lo'] <= ratio
                    <= b['b1_state_ratio_hi']):
                fails.append(('b1_state', r['id'], round(ratio, 4)))
        ring_ref.append(float(r0['delta_on']['ring_mass']))
        ring_new.append(float(r['delta_on']['ring_mass']))
        p0 = r0['delta_on']['priors']
        p1 = r['delta_on']['priors']
        for k, v0 in p0.items():
            if v0 < b['b1_action_min_seed_prior']:
                continue
            v1 = p1.get(k)
            if v1 is None:
                continue
            if v1 * b['b1_action_shrink_max'] < v0:
                fails.append(('b1_action', r['id'], k,
                              round(v0, 4), round(v1, 6)))
    if ratios and float(np.mean(ratios)) < b['b1_aggregate_min']:
        fails.append(('b1_aggregate', round(float(np.mean(ratios)), 4)))
    drift = None
    if anchor_rows is not None:
        drift = anchored_logit_drift(anchor_rows, new_rows,
                                     b['b4_anchor_floor'])
        if drift > b['b4_delta_logit_max']:
            fails.append(('b4_anchored_drift', round(drift, 3)))
        if max_delta > b['b4_delta_logit_max']:
            warns.append(('b4_raw_delta_logit', round(max_delta, 3)))
    elif max_delta > b['b4_delta_logit_max']:
        fails.append(('b4_delta_logit', round(max_delta, 3)))
    if ring_ref:
        rr = float(np.mean(ring_ref))
        rn = float(np.mean(ring_new))
        if rn > max(rr * b['ring_mass_mult'], rr + b['ring_mass_abs']):
            fails.append(('ring_mass_increase', round(rr, 4),
                          round(rn, 4)))
    metrics = {'max_abs_delta_logit': max_delta,
               'b4_anchored_drift': drift,
               'n_rows': _n_matched9,
               'b1_mean_ratio': (float(np.mean(ratios))
                                 if ratios else None)}
    if _n_matched9 == 0:
        # zero row overlap = a dead probe run / regenerated probe set —
        # a vacuous pass must never read as a real one
        fails.append(('b0_no_overlap', len(ref_rows), len(new_rows)))
    if pre_rows is not None:
        kls = paired_kl(pre_rows, new_rows)
        if kls:
            k50 = float(np.percentile(kls, 50))
            k95 = float(np.percentile(kls, 95))
            metrics['kl_p50'] = k50
            metrics['kl_p95'] = k95
            if k50 > b['kl_p50_max']:
                fails.append(('kl_p50', round(k50, 4)))
            if k95 > b['kl_p95_max']:
                fails.append(('kl_p95', round(k95, 4)))
    for side in ('fp', 'sh'):
        if not t1[side]:
            continue
        cur = float(np.mean(t1[side]))
        seed = float(np.mean(t1_ref[side]))
        pv = prev_t1.get(side) if prev_t1 else None
        prev = seed if pv is None else float(pv)
        metrics['t1_%s' % side] = cur
        swing = abs(cur - prev)
        if swing > b['b3_t1_swing_max']:
            av = anchor_t1.get(side) if anchor_t1 else None
            toward = (av is not None
                      and abs(cur - float(av)) < abs(prev - float(av))
                      and swing <= b['b3_t1_anchor_cap'])
            if toward:
                warns.append(('b3_t1_toward_anchor_%s' % side,
                              round(prev, 4), round(cur, 4),
                              round(float(av), 4)))
            else:
                fails.append(('b3_t1_swing_%s' % side, round(prev, 4),
                              round(cur, 4)))
        if abs(cur - seed) > b['b3_t1_cum_warn']:
            warns.append(('b3_t1_cumulative_%s' % side,
                          round(seed, 4), round(cur, 4)))
    return {'ok': not fails, 'fails': fails, 'warns': warns,
            'metrics': metrics}


PROBE_ACTION_CONTRACT = 'complete_action_identity_v1'


def _probe_identity_value(value):
    """JSON-safe, lossless identity for supported action parameter types."""
    from enum import Enum
    if isinstance(value, Enum):
        cls = type(value)
        return ['enum', cls.__module__, cls.__qualname__, value.name]
    if value is None or type(value) in (bool, int, float, str):
        return [type(value).__name__, value]
    if isinstance(value, dict):
        pairs = [(_probe_identity_value(k), _probe_identity_value(v))
                 for k, v in value.items()]
        return ['dict', sorted(pairs, key=lambda pair: json.dumps(pair[0]))]
    if isinstance(value, (tuple, list, set, frozenset)):
        items = [_probe_identity_value(v) for v in value]
        if isinstance(value, (set, frozenset)):
            items.sort(key=json.dumps)
        return [type(value).__name__, items]
    raise TypeError('Unsupported probe action parameter: %s' % type(value).__name__)


def action_key(a):
    """Preserve the physical die and every parameter in diagnostic priors."""
    return json.dumps([PROBE_ACTION_CONTRACT, a.action_type, a.die_idx,
                       _probe_identity_value(a.params or {})],
                      ensure_ascii=True, allow_nan=False, separators=(',', ':'))


def _check_probe_contracts(*groups):
    versions = {row.get('action_identity_contract')
                for group in groups for row in group}
    if len(versions) > 1 or (versions and not versions <= {None, PROBE_ACTION_CONTRACT}):
        raise ValueError('Incompatible probe action identities; regenerate the reference '
                         'from its original model and probe set in a fresh run directory')


def validate_probe_reference(rows):
    """Refuse incompatible persisted diagnostics before starting a batch."""
    if not rows or any(row.get('action_identity_contract') != PROBE_ACTION_CONTRACT
                       for row in rows):
        raise ValueError('Incompatible or empty probe reference; regenerate the reference '
                         'from its original model and probe set in a fresh run directory')


def probe_record(rec, net):
    """One probe-set record -> metrics row (priors over the REGENERATED
    legal menu with delta on/off, raw value, fsp/ring mass, max |delta
    logit|).  Lazy torch/ai imports keep the module test-light."""
    import numpy as _np
    import torch
    import ai
    from game_state import Side
    tag = rec['side']
    side = Side.FREE_PEOPLES if tag == 'fp' else Side.SHADOW
    st = rec['pre_state']
    flag = rec.get('phase_flag')
    try:
        if flag:
            setattr(st, flag, True)
        actions = ai.get_legal_actions(st, side)
        if len(actions) < 2:
            return None
        vec = ai.encode_state(st, side)
        phi = float(ai._profile_shaping(st))
    finally:
        if flag:
            try:
                setattr(st, flag, False)
            except Exception:
                pass
    keys = [action_key(a) for a in actions]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate complete action identities in probe menu')
    from training_device import model_device
    device = model_device(net)
    x = torch.from_numpy(_np.array([vec], dtype=_np.float32)).to(device)
    with torch.no_grad():
        h, ne = net.forward_backbone(x)
        v = float(net.value_head(h).squeeze(-1).item())
        cm = {}
        ae = _np.array([ai._encode_action_cached(a, st, side, cm)
                        for a in actions], dtype=_np.float32)
        ri = _np.array([ai._region_index_cached(a) for a in actions],
                       dtype=_np.int64)
        out = {}
        logits_by = {}
        for wd in (False, True):
            logits = net.action_scores(h, ne, torch.from_numpy(ae).to(device),
                                       torch.from_numpy(ri).to(device),
                                       with_delta=wd)
            logits_by[wd] = logits.detach().cpu().numpy().astype(_np.float64)
            pri = torch.softmax(logits, dim=0).detach().cpu().numpy() \
                .astype(_np.float64)
            fsp = float(sum(p for a, p in zip(actions, pri)
                            if a.action_type in ('fellowship_move',
                                                 'mordor_move')))
            ring = float(sum(p for a, p in zip(actions, pri)
                             if a.action_type == 'elven_ring'))
            out['delta_on' if wd else 'delta_off'] = {
                'fsp_mass': fsp, 'ring_mass': ring,
                'priors': dict(zip(keys, map(float, pri)))}
    dmax = float(_np.max(_np.abs(logits_by[True] - logits_by[False])))
    return {'action_identity_contract': PROBE_ACTION_CONTRACT,
            'id': rec['id'], 'stratum': rec['stratum'],
            'side': tag, 'turn': rec['turn'], 'n_legal': len(actions),
            'v_raw': v, 'phi': phi, 'max_abs_delta_logit': dmax,
            **out}


def probe_rows(fp_net_path, sh_net_path, probeset_path,
               profiles=('ring', 'military')):
    """Full probe run over a frozen probe set: rows for
    stage_b_compare.  Board is reattached (probe sets store stripped
    states)."""
    import ai
    with gzip.open(probeset_path, 'rb') as f:
        recs = pickle.load(f)
    import board as board_mod
    _b = board_mod.load_board()
    for rec in recs:
        rec['pre_state'].board = _b
    nets = {'fp': ai.load_value_net(fp_net_path, side='fp'),
            'sh': ai.load_value_net(sh_net_path, side='sh')}
    for n in nets.values():
        n.eval()
    ai.PLAYER_PROFILES['fp'] = profiles[0]
    ai.PLAYER_PROFILES['sh'] = profiles[1]
    rows = []
    _n_err9 = 0
    _last_err9 = None
    try:
        for rec in recs:
            try:
                row = probe_record(rec, nets[rec['side']])
            except Exception as _pe9:
                row = None
                _n_err9 += 1
                _last_err9 = _pe9
            if row is not None:
                rows.append(row)
    finally:
        ai.PLAYER_PROFILES['fp'] = None
        ai.PLAYER_PROFILES['sh'] = None
    if _n_err9:
        print('[probe] %d/%d records failed (last: %r)'
              % (_n_err9, len(recs), _last_err9))
    if recs and not rows:
        # a systematically dead probe run must be LOUD, not an
        # empty-rows vacuous pass downstream
        raise RuntimeError(
            'probe run produced 0 rows from %d records (last error: '
            '%r)' % (len(recs), _last_err9))
    return rows


class LoopState(object):
    """File-backed per-net loop state: LR multiplier + history (loop
    v2, USER 2026-08-11: TRAIN-ALWAYS - the old fail/halt/revert
    transition methods were DELETED 2026-08-24; only lr(), batch_ok()
    and the history list are read.  a_fails/b_fails keys persist in
    old state files and are ignored)."""

    def __init__(self, path, net_names):
        self.path = path
        self.d = {n: {'lr_mult': 1.0, 'a_fails': 0, 'b_fails': 0,
                      'history': []} for n in net_names}
        if path and os.path.exists(path):
            try:
                with open(path, encoding='utf-8') as f:
                    saved = json.load(f)
                for n in net_names:
                    if n in saved:
                        self.d[n].update(saved[n])
            except (OSError, ValueError):
                pass

    def save(self):
        if not self.path:
            return
        tmp = self.path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self.d, f, indent=1)
        os.replace(tmp, self.path)

    def _log(self, net, event):
        self.d[net]['history'].append(event)
        self.d[net]['history'] = self.d[net]['history'][-200:]

    def lr(self, net, base_lr):
        return base_lr * float(self.d[net]['lr_mult'])

    def batch_ok(self, net):
        self.d[net]['a_fails'] = 0
        self.d[net]['b_fails'] = 0
        self._log(net, ['BATCH_OK'])
        self.save()
