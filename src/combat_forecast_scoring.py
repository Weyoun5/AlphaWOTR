"""Evaluation-only categorical forecast scores; never changes game decisions.

`complete` is the caller's assertion that omitted categories have zero model
probability. Set it False for incomplete/sample-discovered support. `samples`
records an empirical sample count; it does not establish complete support.
All scores concern the supplied forecast, not proof of its calibration.
"""
from __future__ import annotations

import math
from numbers import Real


def score_forecast(probabilities, observed, *, complete=True, top_k=5, samples=None):
    """Return JSON-safe scores and deterministic top-k positive-probability rows.

    Categorical labels must be strings. Values are never renormalized. Partial
    distributions expose supplied mass and missing mass, but do not receive
    full-distribution Brier/log scores. Empirical zero is not an exact zero.
    """
    if not isinstance(complete, bool) or not isinstance(observed, str):
        raise ValueError('complete must be bool and observed must be a string')
    if type(top_k) is not int or top_k < 1:
        raise ValueError('top_k must be a positive integer')
    if samples is not None and (type(samples) is not int or samples < 1):
        raise ValueError('samples must be a positive integer or None')
    values = {}
    for key, value in probabilities.items():
        if not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError('Expected string categories and real probabilities')
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Probabilities must be finite and in [0,1]')
        values[key] = value
    total = math.fsum(values.values())
    if total > 1 + 1e-9 or (complete and not math.isclose(total, 1, rel_tol=0, abs_tol=1e-9)):
        raise ValueError('Invalid probability mass; complete distributions must sum to one')
    ranked = sorted(((key, p) for key, p in values.items() if p > 0), key=lambda item: (-item[1], item[0]))
    top = ranked[:top_k]
    top_mass = math.fsum(p for _, p in top)
    supplied = values.get(observed)
    known_positive = supplied is not None and supplied > 0
    if known_positive:
        status = 'modeled'
        actual_probability = supplied
    elif complete:
        status = 'estimated_zero' if samples is not None else 'true_zero'
        actual_probability = 0.0
    else:
        status = 'unmodeled'
        actual_probability = None
    brier = None
    log_loss = None
    infinite = False
    if complete:
        # Multiclass Brier: sum over every outcome, including an observed
        # category absent from this declared-complete mapping.
        brier = math.fsum((p - (key == observed)) ** 2 for key, p in values.items())
        if observed not in values:
            brier += 1
        if actual_probability > 0:
            log_loss = -math.log(actual_probability)
        else:
            infinite = True  # JSON has no standard Infinity numeric literal.
    return {
        'complete': complete, 'sample_count': samples,
        'probability_basis': 'empirical' if samples is not None else 'supplied_model',
        'observed': observed, 'observed_status': status,
        'actual_probability': actual_probability, 'supplied_observed_probability': supplied,
        'top_k': top_k, 'top': [{'outcome': key, 'probability': p} for key, p in top],
        'ranking_scope': 'full_distribution' if complete else 'known_support_only',
        'top_k_probability': top_mass, 'remainder_probability': max(0.0, 1 - top_mass),
        'supplied_probability_mass': total, 'unlisted_probability_mass': max(0.0, 1 - total),
        'top_k_hit': observed in {key for key, _ in top},
        'brier': brier, 'log_loss': log_loss, 'log_loss_infinite': infinite,
    }


def aggregate_scores(records):
    """Compare observed top-k coverage with forecast mass, without mixing scope.

    Calibration bins are descriptive event frequencies, not a confidence claim.
    Incomplete support and empirical forecasts are reported in separate groups.
    Log-loss means preserve any infinite score instead of dropping zero events.
    """
    rows = list(records)
    grouped = {}
    for row in rows:
        key = ('complete' if row['complete'] else 'partial') + ':' + row['probability_basis'] + ':top' + str(row['top_k'])
        grouped.setdefault(key, []).append(row)
    results = {}
    for key, group in sorted(grouped.items()):
        count = len(group)
        predicted = math.fsum(row['top_k_probability'] for row in group) / count
        hit = sum(row['top_k_hit'] for row in group) / count
        bins = []
        for index in range(10):
            selected = [row for row in group if min(9, int(row['top_k_probability'] * 10)) == index]
            if selected:
                bins.append({'lower': index / 10, 'upper': (index + 1) / 10,
                             'count': len(selected),
                             'mean_predicted_coverage': math.fsum(row['top_k_probability'] for row in selected) / len(selected),
                             'observed_coverage': sum(row['top_k_hit'] for row in selected) / len(selected)})
        infinite = any(row['log_loss_infinite'] for row in group)
        full = group[0]['complete']
        results[key] = {
            'count': count, 'top_k_hits': sum(row['top_k_hit'] for row in group),
            'mean_predicted_coverage': predicted, 'observed_coverage': hit,
            'coverage_gap_observed_minus_predicted': hit - predicted,
            'calibration_bins': bins,
            'unmodeled_observations': sum(row['observed_status'] == 'unmodeled' for row in group),
            'mean_brier': math.fsum(row['brier'] for row in group) / count if full else None,
            'mean_log_loss': math.fsum(row['log_loss'] for row in group) / count if full and not infinite else None,
            'mean_log_loss_infinite': infinite,
        }
    return {'count': len(rows), 'groups': results}
