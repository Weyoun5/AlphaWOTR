# -*- coding: utf-8 -*-
"""Distill recorded root-search targets with full-menu cross-entropy and KL.

Versioned rows preserve actual search targets, eligibility and decision roles.
Training defaults to the policy delta and appended observation columns.
Explicit full_network_v1 also trains the shared encoder and original scorer;
head_only_legacy preserves the historical frozen-trunk operator.
The pre-update distribution anchors KL while current logits use current weights.
"""
import os
from dataclasses import dataclass

import numpy as np

_ATOL = 2e-3          # stored-arm match tolerance (float16 encodings)


@dataclass
class DeltaTrainConfig:
    bs: int = 64
    lr: float = 1e-4
    epochs: int = 4
    wd: float = 1e-4
    kl_beta: float = 0.5
    match_min: float = 0.5     # drop rows below this stored-arm match
    min_elig: int = 2          # visit_tf: supported targets; sigma: Q-eligible arms
    shuffle_seed: int = 0
    max_extra: int = 2048      # cap regenerated extras per row (RAM)


def _flag_for(action_type):
    # 'declare_decline' = the searched pass at a declare root (SPEP-8
    # absence sample, 1.230/session-34 doctrine): the search ran under
    # _declare_phase, so the regenerated menu must too — without the
    # flag every decline row is skew-dropped or stale-masked
    return ('_declare_phase'
            if action_type in ('fellowship_declare', 'declare_decline')
            else '_eye_allocation_phase'
            if action_type == 'eye_allocate' else None)


def frozen_side_of(tr):
    """1.259 (frozen-opponent anchor): the trace's frozen side
    ('fp'/'sh') or None.  ONE predicate for every collector - the
    frozen side's decisions are NEVER trained on.  The episode loop
    stamps the trace meta AND every frozen-side row ('frozen_side');
    the meta is the authority, the per-row twin covers partial
    traces."""
    fs = (tr.get('meta') or {}).get('frozen_side')
    return fs if fs in ('fp', 'sh') else None


def collect_delta_samples(tr, tag, cfg=None):
    """(rows, stats) from one loaded trace for side `tag` ('fp'/'sh').

    Row: {enc f32[D], ae f32[K,A], ri i64[K], p/q/n f64[K], tf f64[K]?,
    played int, sm bool[K], extra_ae f32[E,A], extra_ri i64[E]} —
    stored arms + the regenerated remainder of the legal menu.  Rows
    whose stored arms do not match the regenerated menu (turn-boundary
    snapshot skew) are DROPPED and counted in stats['skew_dropped'].
    SPTR-3: 'sm' marks which stored arms exist in the regenerated
    legal menu — arms with sm=False are masked out of the CE target
    (stats['stale_arms'] counts them).  The regenerated menu is
    byte-sorted so its order (and the max_extra subset) cannot depend
    on PYTHONHASHSEED (SPTR-9 at the collection site)."""
    import ai
    from game_state import Side
    from trace_decisions import (forced_policy_reason, trace_ai_policy_contract,
                                 trace_observation_contract)
    policy_contract = trace_ai_policy_contract(tr)
    trace_observation_contract(tr)
    from trace_decisions import trace_action_encoding_contract
    from reward_contract import uses_predecision, PREDECISION
    action_contract = trace_action_encoding_contract(tr, required=(uses_predecision() or
        (tr.get('meta') or {}).get('reward_contract') in PREDECISION))
    from selector_contract import (training_identity, validate_identity,
                                   validate_training_row, copy_evidence)
    selector_expected = training_identity()
    if selector_expected or 'selector_contract' in (tr.get('meta') or {}):
        validate_identity(tr.get('meta'), expected=selector_expected or None)
    if cfg is None:
        cfg = DeltaTrainConfig()
    side = Side.FREE_PEOPLES if tag == 'fp' else Side.SHADOW
    rows = []
    stats = {'rows': 0, 'skew_dropped': 0, 'error': 0, 'stale_arms': 0,
             'frozen_dropped': 0, 'forced_dropped': 0}
    fs9 = frozen_side_of(tr)
    prev = None
    for s in tr['steps']:
        st = s.get('state')
        if st is None:
            continue
        if forced_policy_reason(s) is not None:
            if s.get('side', '').lower() == tag:
                stats['forced_dropped'] += 1
            prev = st
            continue
        ss = s.get('search_stats')
        sd = s.get('side')
        if ss is not None and sd in ('FP', 'SH') and sd.lower() == tag:
            validate_training_row(ss)
        if (ss is not None and sd in ('FP', 'SH')
                and sd.lower() == tag and prev is not None
                and len(ss.get('p', ())) >= 2):
            # 1.259: frozen-opponent rows are never trained on -
            # counted drop, loud in the DRIVER_STATS: line
            if fs9 == tag or str(s.get('frozen_side') or '') == tag:
                stats['frozen_dropped'] += 1
                prev = st
                continue
            flag = _flag_for(str(s.get('action_type')))
            # bind the flagged object ONCE: the except path reassigns
            # `prev` before `continue` fires the finally, which then
            # cleared the flag on the WRONG state and left it stuck
            # True on the real pre-state
            context = s.get('decision_context') or {}
            decision_state = context.get('state') or prev
            _flagged9 = decision_state
            old_flag = getattr(_flagged9, flag, False) if flag else None
            try:
                if flag:
                    setattr(_flagged9, flag, True)
                enc = (np.asarray(context['encoding'], dtype=np.float32)
                       if 'encoding' in context
                       else ai.encode_state(decision_state, side))
                legal = ai.get_legal_actions(decision_state, side)
                if policy_contract is not None:
                    from ai_policies import project_ai_actions
                    projected = project_ai_actions(decision_state, side, legal)
                    if projected.forced_reason is not None:
                        raise ValueError('fixed policy transition lacks its forced annotation')
                    legal = projected.actions
                legal_ae = np.array(
                    [ai.encode_action(a, decision_state, side) for a in legal],
                    dtype=np.float32) if legal else \
                    np.zeros((0, ai.ACTION_ENCODING_DIM),
                             dtype=np.float32)
                legal_ri = np.array(
                    [ai._get_action_region_index(a) for a in legal],
                    dtype=np.int64)
                if len(legal):
                    # SPTR-9: deterministic menu order at the
                    # collection site — a PYTHONHASHSEED change must
                    # not reorder targets or the max_extra subset
                    _ord9 = sorted(
                        range(len(legal)),
                        key=lambda i9: (legal_ae[i9].tobytes(),
                                        int(legal_ri[i9])))
                    legal_ae = legal_ae[_ord9]
                    legal_ri = legal_ri[_ord9]
            except Exception:
                stats['error'] += 1
                prev = st
                continue
            finally:
                if flag:
                    try:
                        setattr(_flagged9, flag, old_flag)
                    except Exception:
                        pass
            ae = np.asarray(ss['ae'], dtype=np.float32)
            K = ae.shape[0]
            matched = np.zeros(len(legal), dtype=bool)
            sm = np.zeros(K, dtype=bool)
            n_match = 0
            if len(legal):
                d = np.abs(legal_ae[None, :, :]
                           - ae[:, None, :]).max(-1)
                best = d.argmin(axis=1)
                ok = d[np.arange(K), best] <= _ATOL
                matched[best[ok]] = True
                sm = ok
                n_match = int(ok.sum())
            if n_match < cfg.match_min * K:
                stats['skew_dropped'] += 1
                prev = st
                continue
            # SPTR-3: stored arms NOT in the regenerated legal menu
            # must never train with positive weight — mask recorded
            # here, applied in _masked_target
            stats['stale_arms'] += int(K - n_match)
            extra = ~matched
            extra_ae = legal_ae[extra][:cfg.max_extra]
            extra_ri = legal_ri[extra][:cfg.max_extra]
            row = {'enc': enc.astype(np.float32), 'ae': ae, 'sm': sm,
                   'ri': np.asarray(ss['ri'], dtype=np.int64),
                   'p': np.asarray(ss['p'], dtype=np.float64),
                   'q': np.asarray(ss['q'], dtype=np.float64),
                   'n': np.asarray(ss['n'], dtype=np.float64),
                   'played': int(ss['played']),
                   'extra_ae': extra_ae, 'extra_ri': extra_ri}
            if 'tf' in ss:
                row['tf'] = np.asarray(ss['tf'], dtype=np.float64)
            if 'pv' in ss:
                row['pv'] = np.asarray(ss['pv'], dtype=np.int8)
            if 'pvv' in ss:
                row['pvv'] = np.asarray(ss['pvv'], dtype=np.float64)
            if 'fn' in ss:
                row['fn'] = np.asarray(ss['fn'], dtype=np.float64)
            if 'cf' in ss:
                row['cf'] = np.asarray(ss['cf'], dtype=np.float64)
            if 'ph' in ss:
                row['ph'] = np.asarray(ss['ph'], dtype=np.float64)
            if action_contract is not None:
                row['action_encoding_contract'] = action_contract
            if 'selector_contract' in ss:
                row.update(copy_evidence(ss))
            rows.append(row)
            stats['rows'] += 1
        prev = st
    return rows, stats


def _row_target(row):
    """Use the live target; explicit legacy rows retain historical reconstruction.

    Rounded diagnostic Q/terminal arrays cannot reproduce every live threshold
    decision. Versioned rows carry the target and eligibility before rounding.
    """
    from selector_contract import validate_training_row
    retained = validate_training_row(row)
    if retained is not None:
        return retained
    import ai
    cs = {'p': row['p'], 'q': row['q'], 'n': row['n']}
    if 'tf' in row:
        cs['tf'] = row['tf']
    if 'pv' in row:
        cs['pv'] = row['pv']
    if 'pvv' in row:
        cs['pvv'] = row['pvv']
    if 'fn' in row:
        cs['fn'] = row['fn']
    if 'cf' in row:
        cs['cf'] = row['cf']
    if 'ph' in row:
        cs['ph'] = row['ph']
    _i, pi, dg = ai.select_root_action(cs, 0.0)
    return pi, dg['elig']


def _visit_target_evidence(row):
    """Validate visit/proof evidence before the selector can turn it into pi.

    This is target admission, not a new selector or target formula. In
    particular, zero/fully-forced visits cannot manufacture evidence through
    the selector's uniform or raw-count play fallback.
    """
    import ai
    if row.get('policy_forced') or row.get('forced_policy_reason') is not None:
        raise ValueError('A fixed/forced policy transition cannot teach a policy target')
    try:
        n = np.asarray(row['n'], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Invalid policy visit evidence') from exc
    if n.ndim != 1 or len(n) == 0:
        raise ValueError('Policy visits must be a nonempty vector')
    k = len(n)

    def vector(name, default=None):
        try:
            value = np.asarray(row[name] if name in row else default,
                               dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError('Invalid policy evidence: ' + name) from exc
        if value.shape != (k,) or not np.isfinite(value).all():
            raise ValueError('Nonfinite or misaligned policy evidence: ' + name)
        return value

    n = vector('n')
    fn = vector('fn', np.zeros(k))
    if (np.any(n < 0) or np.any(fn < 0) or np.any(fn > n)
            or np.any(n != np.floor(n)) or np.any(fn != np.floor(fn))):
        raise ValueError('Policy visits/forced visits must be consistent nonnegative counts')
    p = vector('p')
    with np.errstate(over='ignore'):
        mass = float(p.sum())
    if np.any(p < 0) or not np.isfinite(mass) or mass <= 0:
        raise ValueError('Invalid policy prior evidence')
    vector('q')
    for name in ('tf', 'cf'):
        if name in row:
            value = vector(name)
            if np.any(value < 0) or np.any(value > 1):
                raise ValueError('Policy evidence outside [0,1]: ' + name)
    if 'ph' in row:
        vector('ph')
    pv = vector('pv', np.zeros(k))
    if not np.isin(pv, [-1, 0, 1]).all():
        raise ValueError('Invalid policy proof marks')
    pvv = vector('pvv') if 'pvv' in row else [None] * k
    # Apply exactly the selector's objective-bound proof semantics. An
    # off-profile board win does not prove maximum attainable utility.
    proof = np.asarray([
        {'W': 1, 'L': -1}.get(ai._objective_proof_mark(
            {1: 'W', -1: 'L'}.get(int(mark)), value), 0)
        for mark, value in zip(pv, pvv)], dtype=np.int8)
    sm = np.asarray(row.get('sm', np.ones(k, dtype=bool)))
    if sm.shape != (k,) or sm.dtype.kind != 'b':
        raise ValueError('Policy stored-action match mask must be an aligned boolean vector')
    extra = row.get('extra_ae')
    if extra is not None:
        extra = np.asarray(extra)
        if extra.ndim != 2 or not np.isfinite(extra).all():
            raise ValueError('Malformed regenerated policy alternatives')
    alternatives = int(sm.sum()) + (len(extra) if extra is not None else 0)
    return n, fn, proof, sm, alternatives


def _masked_target(row, min_elig=2):
    """SPTR-3 stale-arm masking on top of THE selector target: stored
    arms absent from the regenerated legal menu (row['sm'] False) get
    target 0 AND are excluded from the CE softmax; the remaining
    target mass is renormalized.  Rows without 'sm' (pre-fix buffer
    contents) are treated as fully matched.

    visit_tf requires supported target actions, independently of the Q
    override mask. A matched maximum-utility proof may teach a one-hot
    against legal alternatives. The all-proven-loss target is retained for
    compatibility; equal loss proofs do not establish a quality preference.
    sigma retains its existing Q-eligibility gate and target semantics.

    Returns (pi_kept, keep_idx, n_masked, why): why is None on
    success; ('elig' | 'zero_mass', ...) marks a skipped row."""
    import ai
    # Validate first, including ambient-versus-recorded experiment identity.
    # The target mode belongs to the recorded selector, never a new default.
    pi, elig = _row_target(row)
    visit_mode = (row['selector_config']['pi_mode'] if 'selector_contract' in row
                  else ai.SELECT_CFG.pi_mode) == 'visit_tf'
    evidence = _visit_target_evidence(row) if visit_mode else None
    pi = np.asarray(pi, dtype=np.float64)
    elig = np.asarray(elig).astype(bool)
    sm = row.get('sm')
    sm = (np.ones(len(pi), dtype=bool) if sm is None
          else np.asarray(sm, dtype=bool))
    n_masked = int((~sm).sum())
    pi = np.where(sm, pi, 0.0)
    if visit_mode:
        n, fn, proof, _sm, alternatives = evidence
        if (pi.shape != n.shape or not np.isfinite(pi).all()
                or np.any(pi < 0)):
            raise ValueError('Malformed selector policy target')
        if alternatives < 2:
            return None, None, n_masked, 'elig'
        winning_proof = np.any((proof == 1) & sm & (pi > 0))
        all_lost = np.all(proof == -1) and int(sm.sum()) >= min_elig
        if not winning_proof and not all_lost:
            # pi already discounts forced visits using the selector's played-
            # arm convention. Keep that target exactly; require real earned
            # evidence so its all-forced/raw fallback is never distilled.
            if (not np.any((n - fn)[sm] > 0)
                    or int(((pi > 0) & sm).sum()) < min_elig):
                return None, None, n_masked, 'elig'
    elif int((elig & sm).sum()) < min_elig:
        return None, None, n_masked, 'elig'
    mass = float(pi.sum())
    if mass <= 0.0:
        return None, None, n_masked, 'zero_mass'
    keep = np.flatnonzero(sm)
    return pi[keep] / mass, keep, n_masked, None


def train_policy_delta(net, rows, cfg=None):
    """Full-support CE and fixed-reference KL under the explicit weight scope.

    All modules remain in eval mode; full-network autograd still trains every
    parameter reached by policy loss, with stable singleton-row normalization.
    """
    import torch
    from selector_contract import validate_training_row
    from trace_decisions import validate_action_encoding_contract
    from reward_contract import uses_predecision, PREDECISION
    for row in rows:
        validate_action_encoding_contract(row, required=(uses_predecision() or
            row.get('selector_reward_contract') in PREDECISION))
        validate_training_row(row)
    if cfg is None:
        cfg = DeltaTrainConfig()
    rng = np.random.default_rng(cfg.shuffle_seed)
    for p in net.parameters():
        p.requires_grad = False
    dpars = [p for k, p in net.named_parameters()
             if k.startswith('policy_delta')]
    for p in dpars:
        p.requires_grad = True
    from observation_training import ObservationTrainingScope
    with ObservationTrainingScope(net) as scope:
        from optimizer_continuity import optimizer_for, updated
        opt = optimizer_for(net, 'policy', scope.parameter_groups(dpars, cfg.wd), scope, cfg.lr)
        net.eval()
        from training_device import model_device
        device = model_device(net)

        # Cache initial features/logits for legacy mode and the fixed KL anchor:
        # PRE-UPDATE reference probs (KL anchor), targets.  SPTR-15: the
        # per-arm `combined` matrix (~(K+E) x scorer-width f32) is NOT
        # cached — only (h, ne, base, pre_logp, tgt) — and is rebuilt per
        # visit from the row's own encodings.
        prep = []
        n_stale = n_skip_elig = n_skip_mass = n_ow0 = 0
        prep_err = 0
        first_err = None

        def _cat(a, b):
            return np.concatenate([a, b]) if len(b) else a

        def _combined_of(s):
            ae = torch.from_numpy(np.ascontiguousarray(
                _cat(s['ae_k'], s['x_ae']), dtype=np.float32)).to(device)
            ri = torch.from_numpy(np.ascontiguousarray(
                _cat(s['ri_k'], s['x_ri']), dtype=np.int64)).to(device)
            return net._action_combined(s['h'], s['ne'], ae, ri)

        def _current_logits(s):
            if scope.recompute_features:
                s['h'], s['ne'] = net._process_input(s['x'])
                combined = _combined_of(s)
                base = net.spatial_action_scorer(combined).squeeze(-1)
            else:
                combined = _combined_of(s)
                base = s['base']
            return base + net.policy_delta(combined).squeeze(-1), base

        with torch.no_grad():
            for r in rows:
                try:
                    pi_k, keep, n_masked, why = _masked_target(
                        r, cfg.min_elig)
                except Exception as e:
                    prep_err += 1
                    if first_err is None:
                        first_err = repr(e)
                    continue
                n_stale += n_masked
                if why == 'elig':
                    n_skip_elig += 1
                    continue
                if why == 'zero_mass':
                    n_skip_mass += 1
                    continue
                # Corrected contracts distill both sides with ow=1, including
                # losses. Legacy collectors use max(0, side-axis outcome).
                # Rows without 'ow' retain the historical weight of 1.
                ow = max(0.0, float(r.get('ow', 1.0)))
                if ow == 0.0:
                    n_ow0 += 1
                s = {'ae_k': np.asarray(r['ae'], np.float32)[keep],
                     'ri_k': np.asarray(r['ri'], np.int64)[keep],
                     'x_ae': r['extra_ae'], 'x_ri': r['extra_ri'],
                     'ow': ow}
                x = torch.from_numpy(r['enc'][None, :]).to(device)
                s['x'] = x
                s['h'], s['ne'] = net._process_input(x)
                combined = _combined_of(s)
                s['base'] = net.spatial_action_scorer(
                    combined).squeeze(-1)
                pre = s['base'] + net.policy_delta(combined).squeeze(-1)
                s['pre_logp'] = torch.log_softmax(pre, dim=0)
                tgt = torch.zeros_like(s['base'])
                tgt[:len(keep)] = torch.from_numpy(
                    pi_k.astype(np.float32)).to(device)
                s['tgt'] = tgt
                prep.append(s)
        if rows and not prep:
            # SPTR-5/6 loudness: a batch that trains 0 samples must
            # never report success
            raise RuntimeError(
                'train_policy_delta: 0 of %d rows usable '
                '(prep_errors=%d first=%s skipped_elig=%d '
                'skipped_zero_mass=%d)'
                % (len(rows), prep_err, first_err, n_skip_elig,
                   n_skip_mass))
        losses = []
        for _ep in range(cfg.epochs):
            order = rng.permutation(len(prep))
            tot = cnt = 0.0
            opt.zero_grad()
            pend = 0
            for j in order:
                s = prep[int(j)]
                logits, current_base = _current_logits(s)
                logp = torch.log_softmax(logits, dim=0)
                loss = s['ow'] * (-(s['tgt'] * logp).sum())
                if cfg.kl_beta > 0.0:
                    # KL anchor stays UNWEIGHTED: losing games still
                    # cannot drift the prior (SPTR-4 minimal fix)
                    pb = torch.exp(s['pre_logp'])
                    loss = loss + cfg.kl_beta * (
                        pb * (s['pre_logp'] - logp)).sum()
                (loss / cfg.bs).backward()
                pend += 1
                tot += float(loss.item())
                cnt += 1
                if pend >= cfg.bs:
                    opt.step()
                    updated(net, 'policy', opt)
                    scope.record_step('policy')
                    from trace_decisions import record_warm_start_update
                    record_warm_start_update(net, 'policy')
                    opt.zero_grad()
                    pend = 0
            if pend:
                opt.step()
                updated(net, 'policy', opt)
                scope.record_step('policy')
                from trace_decisions import record_warm_start_update
                record_warm_start_update(net, 'policy')
                opt.zero_grad()
            losses.append(round(tot / max(1.0, cnt), 4))
        # post-update movement metrics (trust-region verdict inputs)
        kls, dmax = [], 0.0
        with torch.no_grad():
            for s in prep:
                logits, current_base = _current_logits(s)
                logp = torch.log_softmax(logits, dim=0)
                pb = torch.exp(s['pre_logp'])
                kls.append(float((pb * (s['pre_logp'] - logp)).sum()))
                dmax = max(dmax, float(
                    (logits - current_base).abs().max()))
        net.eval()
        return {'ep_losses': losses, 'rows_used': len(prep),
                'rows_in': len(rows),
                'stale_arms_masked': n_stale,
                'rows_skipped_elig': n_skip_elig,
                'rows_skipped_zero_mass': n_skip_mass,
                'ow_zero_rows': n_ow0,
                'prep_errors': prep_err,
                'prep_first_error': first_err,
                'kl_p50': round(float(np.percentile(kls, 50)), 5)
                if kls else None,
                'kl_p95': round(float(np.percentile(kls, 95)), 5)
                if kls else None,
                'max_abs_delta_logit': round(dmax, 4)}


# ------------------------------------------------------------------
# v33loop driver adapters (run_td_v33loop v2 contracts; the adopted
# package flips DELTA_COLLECTOR / _delta_update_op to these)
# ------------------------------------------------------------------

# SPTR-8: collector stats accumulate here (the DELTA_COLLECTOR
# contract returns rows only); the driver can log/reset per batch.
DRIVER_STATS = {'rows': 0, 'skew_dropped': 0, 'error': 0,
                'stale_arms': 0, 'frozen_dropped': 0, 'forced_dropped': 0}


def _axis_outcome_weight(tr, tag):
    """Win/loss search distillation uses both sides of every decided game.

    The legacy contract uses max(0, side's axis outcome) (CURRENT_RULES sec 7
    item 4, outcome-gated policy distillation).  Terminal from THE
    single implementation (ai._terminal_for) using the trace's own
    meta; profile from ai.PLAYER_PROFILES, which the driver sets
    around the collector call exactly as it does for the value stream.
    An undecided/timeout trace (winner not fp/sh) weighs 0.0 — visits
    from a game with no outcome are never distilled."""
    import ai
    meta = tr.get('meta') or {}
    winner = meta.get('winner')
    if winner not in ('fp', 'sh'):
        return 0.0
    from reward_contract import uses_predecision, trace_contract
    if uses_predecision(trace_contract(tr)):
        # Search targets remain useful on the losing side. The value target
        # carries the negative outcome; multiplying CE by it is incorrect.
        return 1.0
    victory = str(meta.get('victory_type') or '')
    final_state = None
    for s in reversed(tr['steps']):
        if s.get('state') is not None:
            final_state = s['state']
            break
    term = float(ai._terminal_for(tag, ai.PLAYER_PROFILES.get(tag),
                                  victory, winner == tag, final_state))
    return max(0.0, term)


def collect_for_driver(tr, tag):
    """DELTA_COLLECTOR contract: (trace, tag) -> list (extend()-able
    into the replay buffer).  Full-support prep rows, skew-dropped.
    Each row is stamped with ow=1 for decided corrected-contract games;
    legacy rows use max(0, axis outcome) (SPTR-4).
    drop/error/stale counters accumulate in DRIVER_STATS and print
    loudly whenever anything was dropped or masked (SPTR-8)."""
    rows, stats = collect_delta_samples(tr, tag)
    ow = _axis_outcome_weight(tr, tag)
    for r in rows:
        r['ow'] = ow
    for k in DRIVER_STATS:
        DRIVER_STATS[k] += int(stats.get(k, 0))
    if stats['skew_dropped'] or stats['error'] or stats['stale_arms'] \
            or stats['frozen_dropped'] or stats['forced_dropped']:
        print('delta_collect[%s]: rows=%d skew_dropped=%d error=%d '
              'stale_arms=%d frozen_dropped=%d forced_dropped=%d ow=%.3f'
              % (tag, stats['rows'], stats['skew_dropped'],
                 stats['error'], stats['stale_arms'],
                 stats['frozen_dropped'], stats['forced_dropped'], ow), flush=True)
    return rows


def delta_update_for_driver(net, samples, lr, *, epochs=None, wd=None):
    """_delta_update_op contract: (net, samples, lr) -> metrics.
    samples = prep-format rows from collect_for_driver (possibly
    buffer-mixed and subsampled by the driver)."""
    if epochs is not None and (type(epochs) is not int or epochs <= 0):
        raise ValueError('Policy epochs must be a positive integer')
    if wd is not None:
        import math
        import numbers
        if (isinstance(wd, bool) or not isinstance(wd, numbers.Real)
                or not math.isfinite(wd) or wd < 0):
            raise ValueError('Policy weight decay must be finite and nonnegative')
    cfg = DeltaTrainConfig(lr=lr)
    if epochs is not None:
        cfg.epochs = epochs
    if wd is not None:
        cfg.wd = float(wd)
    return train_policy_delta(net, samples, cfg)
