"""Retained root-search targets; independent of observation and reward versions.

The diagnostic Q/terminal arrays are lossy. Never regenerate a versioned
target from them or interpret it with a different ambient selector setting.
"""
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import numbers

import numpy as np

CONTRACT = 'root_search_target_v1'
_FIELDS = {
    'min_visits': int, 'term_frac': float, 'term_margin': float,
    'c_visit': float, 'c_scale': float, 'completed_q_min_n': int,
    'pi_mode': str, 'w_tf': float, 'conf_margin': float, 'conf_play': bool,
    's_low': float, 's_high': float, 'conf_play_margin': float,
    'temp_earned': bool,
}
IDENTITY_FIELDS = ('selector_contract', 'selector_config', 'selector_config_sha256')
EVIDENCE_FIELDS = IDENTITY_FIELDS + (
    'pi_sel', 'selector_eligible', 'selector_temperature', 'selector_reward_contract')


def normalized_config(cfg):
    if is_dataclass(cfg) and not isinstance(cfg, type):
        cfg = asdict(cfg)
    if not isinstance(cfg, Mapping) or set(cfg) != set(_FIELDS):
        raise ValueError('Selector configuration must contain every known field exactly once')
    out = {}
    for key, kind in _FIELDS.items():
        value = cfg[key]
        if kind in (bool, str):
            if type(value) is not kind:
                raise ValueError('Selector configuration has invalid type: ' + key)
        else:
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real)
                    or not math.isfinite(value) or value < 0
                    or (kind is int and value != int(value))):
                raise ValueError('Selector configuration has invalid numeric value: ' + key)
            value = kind(value)
        out[key] = value
    if out['pi_mode'] not in ('visit_tf', 'sigma'):
        raise ValueError('Selector configuration has unknown target mode')
    if any(out[key] > 1 for key in ('term_frac', 'w_tf', 's_low', 's_high')):
        raise ValueError('Selector probability/support setting outside [0,1]')
    if out['s_low'] > out['s_high']:
        raise ValueError('Selector support thresholds are reversed')
    return out


def identity(cfg):
    cfg = normalized_config(cfg)
    canonical = json.dumps(cfg, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return {'selector_contract': CONTRACT, 'selector_config': cfg,
            'selector_config_sha256': hashlib.sha256(canonical.encode('utf-8')).hexdigest()}


def current_identity():
    import ai
    return identity(ai.SELECT_CFG)


def validate_identity(data, *, expected=None):
    if not isinstance(data, Mapping) or data.get('selector_contract') != CONTRACT:
        raise ValueError('Selector contract missing or unsupported')
    actual = identity(data.get('selector_config'))
    if data.get('selector_config_sha256') != actual['selector_config_sha256']:
        raise ValueError('Selector configuration hash mismatch')
    if expected is not None:
        wanted = validate_identity(expected)
        if actual != wanted:
            raise ValueError('Selector configuration mismatch; use a fresh experiment')
    return actual


def validate_evidence(data, *, expected=None, reward_contract=None):
    """Return checked target and original eligibility without reconstructing either."""
    validate_identity(data, expected=expected)
    try:
        n = np.asarray(data['n'])
        k = len(n)
        pi = np.asarray(data['pi_sel'])
        eligible = np.asarray(data['selector_eligible'])
        temp = data['selector_temperature']
        played = data['played']
        from reward_contract import LEGACY, PREDECISION
        reward = data['selector_reward_contract']
        if reward not in (*PREDECISION, LEGACY):
            raise ValueError('unknown reward contract')
        if reward_contract is not None and reward != reward_contract:
            raise ValueError('reward contract mismatch')
        if (n.shape != (k,) or n.dtype.kind not in 'fiu' or not np.isfinite(n).all()
                or np.any(n < 0) or np.any(n != np.floor(n))):
            raise ValueError('visits must be aligned finite nonnegative counts')
        if (k < 1 or pi.shape != (k,) or pi.dtype.kind not in 'fiu'
                or not np.isfinite(pi).all() or np.any(pi < 0)
                or not np.isclose(float(pi.astype(np.float64).sum()), 1., rtol=0., atol=1e-5)):
            raise ValueError('target must be an aligned finite probability vector')
        if eligible.shape != (k,) or eligible.dtype.kind != 'b' or not eligible.any():
            raise ValueError('original eligibility must be an aligned boolean vector')
        if (isinstance(temp, (bool, np.bool_)) or not isinstance(temp, numbers.Real)
                or not math.isfinite(temp) or temp < 0):
            raise ValueError('temperature must be finite and nonnegative')
        if (isinstance(played, (bool, np.bool_)) or not isinstance(played, numbers.Integral)
                or not 0 <= played < k):
            raise ValueError('sampled played index outside original action menu')
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Selector evidence invalid: ' + str(exc)) from exc
    return pi.copy(), eligible.copy()


def make_evidence(cfg, temperature, pi_sel, *, eligible, reward_contract):
    """Producer: preserve live target and eligibility before diagnostic rounding."""
    result = dict(identity(cfg), pi_sel=np.asarray(pi_sel, dtype=np.float32).copy(),
                  selector_eligible=np.asarray(eligible).copy(),
                  selector_temperature=temperature, selector_reward_contract=reward_contract)
    # The actual sampled index is attached by mcts_search; checking index zero
    # here validates the vectors without inventing a retained action choice.
    validate_evidence(dict(result, n=np.zeros(len(result['pi_sel'])), played=0))
    return result


def copy_evidence(data):
    """Copy a complete producer/row payload; missing fields never mean legacy."""
    # visit_info uses visit_counts, while serialized and collected rows use n.
    view = dict(data)
    if 'n' not in view and 'visit_counts' in view:
        view['n'] = view['visit_counts']
    validate_evidence(view)
    return {key: (value.copy() if isinstance(value, (dict, np.ndarray)) else value)
            for key in EVIDENCE_FIELDS for value in (data[key],)}


def training_identity():
    """Selector provenance is mandatory for corrected self-play, not BC data."""
    from reward_contract import uses_predecision
    return current_identity() if uses_predecision() else {}


def validate_training_row(row):
    from reward_contract import current_contract, uses_predecision
    if row.get('policy_forced') or row.get('forced_policy_reason') is not None:
        raise ValueError('Selector target cannot belong to a fixed/forced policy transition')
    explicit = any(key in row for key in EVIDENCE_FIELDS)
    if explicit or uses_predecision():
        return validate_evidence(row, expected=current_identity(),
                                 reward_contract=current_contract())
    return None
