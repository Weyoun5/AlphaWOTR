"""Structural admission for recorded self-play, independent of playing quality.

validate_trace consumes a loaded trace and returns JSON-compatible diagnostics.
It writes nothing, changes no state, and never interprets a loss as bad data.
It is not a proof that every historical rule/decision was represented correctly.
"""
from collections.abc import Mapping
from enum import Enum
import math
import numbers

import numpy as np

VALIDATOR_VERSION = 1
# Source failures are rejectable unless reviewed as ordinary legal play.
# Future execution/recovery codes must not silently become training data.
WARNING_SOURCE_CODES = frozenset({'card_play_no_legal_choice'})


def _nonfinite(value, path, seen=None):
    """Find nonfinite numeric leaves; do not traverse the shared Board object."""
    if isinstance(value, (str, bytes, bool, Enum, type)) or value is None:
        return None
    if isinstance(value, numbers.Real):
        return path if not math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        if value.dtype.kind in 'fc':
            return path if not np.isfinite(value).all() else None
        if value.dtype.kind != 'O':
            return None
        value = value.flat
    if seen is None:
        seen = set()
    if id(value) in seen:
        return None
    seen.add(id(value))
    if isinstance(value, Mapping):
        entries = value.items()
    elif isinstance(value, (list, tuple, set, np.flatiter)):
        entries = enumerate(value)
    elif hasattr(value, '__dict__'):
        entries = vars(value).items()
    else:
        return None
    for key, item in entries:
        if key == 'board':
            continue
        bad = _nonfinite(item, '%s.%s' % (path, key), seen)
        if bad:
            return bad
    return None


def validate_trace(tr, *, require_decided=True, audit_rules=True,
                   decision_contract=None, observation_contract=None,
                   ai_policy_contract=None, action_encoding_contract=None, selector_contract=None,
                   selector_config=None, selector_config_sha256=None):
    """Return {accepted, valid, errors, warnings, decisions, terminal, version}.

    `valid` describes data integrity. `accepted` additionally requires a
    decided game by default; an intact timeout is valid but excluded with the
    code `truncated`. No thresholds for win rate, regret, reward, length,
    profile, or tactical quality are applied. Legacy traces without source
    failure instrumentation can pass observed checks with a warning.
    Supply decision_contract to require that explicit pre-decision context
    schema on every player decision before admitting it to a new learner.
    Fresh collectors additionally require explicit observation and AI menu
    contracts; default diagnostics may still inspect untagged historical data.
    """
    out = {'accepted': False, 'valid': False,
           'validator_version': VALIDATOR_VERSION,
           'errors': [], 'warnings': [], 'decisions': {'fp': 0, 'sh': 0},
           'forced_transitions': {'fp': 0, 'sh': 0},
           'terminal': {'winner': None, 'victory_type': None}}

    def flag(code, detail, step=None, warning=False):
        out['warnings' if warning else 'errors'].append(
            {'code': code, 'step': step, 'detail': str(detail)})

    if not isinstance(tr, Mapping):
        flag('trace_schema', 'trace must be a mapping')
        return out
    meta = tr.get('meta')
    steps = tr.get('steps')
    if not isinstance(meta, Mapping):
        flag('metadata_missing', 'trace metadata must be a mapping')
        meta = {}
    from trace_decisions import (trace_ai_policy_contract, trace_observation_contract,
                                 trace_action_encoding_contract)
    policy_contract = None
    try:
        policy_contract = trace_ai_policy_contract({'meta': meta})
    except ValueError as exc:
        flag('ai_policy_contract', exc)
    try:
        trace_observation_contract({'meta': meta})
    except ValueError as exc:
        flag('observation_contract', exc)
    from reward_contract import PREDECISION
    try:
        trace_action_encoding_contract({'meta': meta}, required=(
            decision_contract in PREDECISION or action_encoding_contract is not None))
    except ValueError as exc:
        flag('action_encoding_contract', exc)
    for field, expected in (('action_encoding_contract', action_encoding_contract),
                            ('observation_contract', observation_contract),
                            ('ai_policy_contract', ai_policy_contract)):
        if expected is not None and meta.get(field) != expected:
            flag(field + '_mismatch', 'fresh learner requires %s=%s' % (field, expected))
    if not isinstance(steps, (list, tuple)) or not steps:
        flag('steps_missing', 'nonempty snapshot sequence required')
        return out
    if not isinstance(steps[0], Mapping) or steps[0].get('kind') != 'initial':
        flag('initial_missing', 'first row must be the initial snapshot', 0)
    winner, victory = meta.get('winner'), meta.get('victory_type')
    if decision_contract is not None and meta.get('reward_contract') != decision_contract:
        flag('reward_contract_mismatch',
             'trace metadata must identify the requested reward contract')
    from reward_contract import PROFILE, PREDECISION, validate_profile
    from selector_contract import current_identity, validate_identity, validate_evidence
    selector_required = decision_contract in PREDECISION or selector_contract is not None
    selector_meta = None
    selector_expected = None
    if selector_contract is not None:
        selector_expected = dict(selector_contract=selector_contract,
                                 selector_config=selector_config,
                                 selector_config_sha256=selector_config_sha256)
    elif selector_required:
        selector_expected = current_identity()
    if selector_required or 'selector_contract' in meta:
        try:
            selector_meta = validate_identity(meta, expected=selector_expected)
        except ValueError as exc:
            flag('selector_contract', exc)
    if decision_contract == PROFILE:
        for side in ('fp', 'sh'):
            try:
                validate_profile(side, (meta.get('profiles') or {}).get(side))
            except (ValueError, AttributeError) as exc:
                flag('profile_objective_missing', exc)
    out['terminal'] = {'winner': winner, 'victory_type': victory}
    if winner not in ('fp', 'sh', 'timeout'):
        flag('termination_missing', 'winner must be fp, sh, or explicit timeout')
    elif winner == 'timeout':
        if victory != 'timeout':
            flag('termination_mismatch', 'timeout winner requires timeout victory_type')
        if require_decided:
            flag('truncated', 'valid truncations require bootstrap-aware training')
    elif victory not in ('military', 'ring_destroyed', 'ring_corruption'):
        flag('termination_missing', 'decided game requires an identified victory_type')
    bad = _nonfinite(meta, 'meta')
    if bad:
        flag('nonfinite', bad)
    if not meta.get('source_integrity_version'):
        flag('source_integrity_unrecorded',
             'legacy trace cannot prove recovered execution errors were recorded', warning=True)
    source_errors = meta.get('source_errors', [])
    if not isinstance(source_errors, (list, tuple)):
        flag('source_errors_schema', 'source_errors must be a sequence')
        source_errors = []
    for item in source_errors:
        if not isinstance(item, Mapping):
            flag('source_errors_schema', 'source error record must be a mapping')
            continue
        code = str(item.get('code') or 'source_error_unknown')
        flag('source_' + code, item.get('detail') or item.get('ctx') or code,
             item.get('step'), warning=code in WARNING_SOURCE_CODES)
    last_turn = None
    final = None
    from trace_decisions import forced_policy_reason
    for i, row in enumerate(steps):
        if not isinstance(row, Mapping):
            flag('row_schema', 'row must be a mapping', i)
            continue
        forced_reason = None
        try:
            forced_reason = forced_policy_reason(row)
        except ValueError as exc:
            flag('forced_policy_schema', exc, i)
        if forced_reason is not None and (
                row.get('search_stats') is not None or row.get('regret') is not None):
            flag('forced_policy_target', 'fixed action carries search targets or regret', i)
        state = row.get('state')
        if state is None:
            flag('snapshot_missing', 'every row requires a state snapshot', i)
            continue
        final = state
        turn = row.get('turn', getattr(state, 'turn', None))
        if not isinstance(turn, numbers.Integral) or isinstance(turn, bool) or turn < 1:
            flag('turn_invalid', 'turn must be a positive integer', i)
        else:
            if last_turn is not None and turn < last_turn:
                flag('turn_reversed', '%s follows %s' % (turn, last_turn), i)
            if getattr(state, 'turn', turn) != turn:
                flag('turn_mismatch', 'row turn disagrees with snapshot', i)
            last_turn = turn
        side = row.get('side')
        atype = row.get('action_type')
        if side not in (None, 'FP', 'SH'):
            flag('side_invalid', 'side must be FP, SH, or None', i)
        if side in ('FP', 'SH'):
            out['forced_transitions' if forced_reason is not None else 'decisions'][side.lower()] += 1
            if row.get('kind') != 'action' or not isinstance(atype, str) or not atype:
                flag('decision_schema', 'player decision needs action kind and action_type', i)
            if not isinstance(row.get('params', {}), Mapping):
                flag('decision_schema', 'action params must be a mapping', i)
            context = row.get('decision_context')
            if context is not None or decision_contract is not None:
                try:
                    if not isinstance(context, Mapping):
                        raise ValueError('pre-decision context missing')
                    if decision_contract is not None and context.get('contract') != decision_contract:
                        raise ValueError('decision context contract mismatch')
                    if decision_contract is not None and getattr(context.get('state'), 'fellowship', None) is None:
                        raise ValueError('exact pre-decision state snapshot missing')
                    enc = np.asarray(context['encoding'])
                    if enc.ndim != 1 or not len(enc) or enc.dtype.kind not in 'iuf':
                        raise ValueError('context encoding must be a nonempty numeric vector')
                    if not isinstance(context['reward_stats'], Mapping):
                        raise ValueError('context reward_stats must be a mapping')
                    if not isinstance(context['potential'], numbers.Real):
                        raise ValueError('context potential must be numeric')
                    if context.get('contract') in PREDECISION and context['potential'] != 0:
                        raise ValueError('corrected contract requires zero potential')
                except (KeyError, TypeError, ValueError) as exc:
                    flag('decision_context_schema', exc, i)
            if forced_reason is not None and policy_contract is not None:
                try:
                    # Only versioned AI rows assert this convention. Historical
                    # human choices are never reclassified using today's menu.
                    import ai
                    from game_state import Side
                    from ai_policies import project_ai_actions, action_policy_key
                    if not isinstance(context, Mapping) or context.get('state') is None:
                        raise ValueError('forced transition requires its exact pre-state')
                    before = ai.clone_state(context['state'])
                    actor = Side.FREE_PEOPLES if side == 'FP' else Side.SHADOW
                    projection = project_ai_actions(
                        before, actor, ai.get_legal_actions(before, actor))
                    if projection.forced_reason != forced_reason or len(projection.actions) != 1:
                        raise ValueError('recorded reason does not describe a fully fixed menu')
                    actual_action = ai.Action(row.get('die_idx'), atype, dict(row.get('params', {})))
                    if action_policy_key(actual_action) != action_policy_key(projection.actions[0]):
                        raise ValueError('recorded action differs from the fixed policy action')
                except Exception as exc:
                    flag('forced_policy_semantics', '%s: %s' % (type(exc).__name__, exc), i)
        if atype == 'exec_failure':
            flag('execution_recovered', row.get('params') or 'recorded execution failure', i)
        bad = _nonfinite(row, 'steps[%d]' % i)
        if bad:
            flag('nonfinite', bad, i)
        ss = row.get('search_stats')
        if ss is None:
            if selector_required and side in ('FP', 'SH') and forced_reason is None:
                flag('selector_evidence_missing', 'learned decision requires retained search target', i)
            continue
        if not isinstance(ss, Mapping):
            flag('search_schema', 'search_stats must be a mapping', i)
            continue
        if selector_required or selector_meta is not None or 'selector_contract' in ss:
            try:
                validate_evidence(ss, expected=selector_meta or selector_expected,
                                  reward_contract=decision_contract or meta.get('reward_contract'))
            except ValueError as exc:
                flag('selector_evidence', exc, i)
        try:
            ae = np.asarray(ss['ae'])
            if ae.ndim != 2 or not len(ae):
                raise ValueError('ae must have shape (actions, features)')
            k = len(ae)
            for name in ('p', 'q', 'n', 'ri'):
                arr = np.asarray(ss[name])
                if arr.shape != (k,) or arr.dtype.kind not in 'iuf':
                    raise ValueError('%s must be a numeric vector of length %d' % (name, k))
            if ae.dtype.kind not in 'iuf':
                raise ValueError('ae must be numeric')
            p, n = np.asarray(ss['p']), np.asarray(ss['n'])
            if np.any(p < 0) or not np.isclose(p.sum(), 1.0, atol=1e-4):
                raise ValueError('priors must be a probability distribution')
            if np.any(n < 0):
                raise ValueError('visit counts must be nonnegative')
            played = ss['played']
            if not isinstance(played, numbers.Integral) or not 0 <= played < k:
                raise ValueError('played index outside stored actions')
            for name in ('tf', 'pv', 'pvv', 'fn', 'cf', 'ph', 'v'):
                if name in ss and np.asarray(ss[name]).shape != (k,):
                    raise ValueError('%s must have length %d' % (name, k))
        except (KeyError, TypeError, ValueError) as exc:
            flag('search_schema', exc, i)
    if not (sum(out['decisions'].values()) + sum(out['forced_transitions'].values())):
        flag('decisions_missing', 'trace has no recorded player decisions')
    # Rule checks require complete snapshots, otherwise report structural
    # errors above rather than masking them with an opaque audit exception.
    structural = {'row_schema', 'snapshot_missing', 'initial_missing'}
    if audit_rules and not any(e['code'] in structural for e in out['errors']):
        try:
            from trace_audit import audit_loaded_trace
            for detail in audit_loaded_trace(tr):
                flag('rule_violation', detail)
        except Exception as exc:
            flag('rule_check_failed', '%s: %s' % (type(exc).__name__, exc))
    if final is not None and winner in ('fp', 'sh', 'timeout'):
        try:
            import game_logic
            actual = game_logic.check_all_victory_conditions(final)
            actual_winner = {'free_peoples': 'fp', 'shadow': 'sh'}.get(actual['victory'])
            if winner == 'timeout':
                if actual_winner is not None:
                    flag('termination_mismatch', 'timeout snapshot already has a victory')
            elif (actual_winner != winner or actual.get('victory_type') != victory):
                flag('termination_mismatch',
                     'metadata %s/%s; final state %s/%s' % (
                         winner, victory, actual_winner, actual.get('victory_type')))
            if victory == 'military':
                pending = [name for name in (
                    'pending_combat', 'pending_hunt', 'pending_card_option',
                    'pending_card_followup', 'pending_discard',
                    'pending_army_move_die', 'pending_muster_die',
                    'pending_guide_change', '_gandalf_draw_pending')
                    if getattr(final, name, None) is not None]
                if pending:
                    flag('unfinished_terminal_decision', ', '.join(pending))
                # Military victory is checked only after all usable dice and
                # obligations are exhausted. A resource cap is not a turn end.
                import ai
                blockers = ai._turn_end_blockers(final)
                if blockers:
                    flag('unfinished_terminal_turn', ', '.join(blockers))
        except Exception as exc:
            flag('terminal_check_failed', '%s: %s' % (type(exc).__name__, exc))
    out['valid'] = not any(e['code'] != 'truncated' for e in out['errors'])
    out['accepted'] = not out['errors']
    return out
