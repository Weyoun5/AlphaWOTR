"""Explicit learned-versus-fixed policy roles in recorded action history.

Never infer an old row's role from today's rules or action name. A historical
human take/decline can be learned data even when a current AI policy fixes it.
"""

from action_schema import ACTION_ENCODING_CONTRACT, validate_active_width


FORCED_POLICY_REASONS = frozenset({
    'bonus_draw_always_take', 'recruit_elite_if_available',
    'combat_victim_priority', 'mighty_payer_lowest_leadership'})


def current_learning_context():
    """Semantic versions for newly encoded AI learning data, independent of shape."""
    import ai
    validate_active_width(ai.ACTION_ENCODING_DIM)
    from ai_policies import POLICY_VERSION
    return {'observation_contract': ai.OBSERVATION_CONTRACT,
            'ai_policy_contract': POLICY_VERSION,
            'action_encoding_contract': ACTION_ENCODING_CONTRACT}


def validate_action_encoding_contract(metadata, *, required=False):
    """Validate action semantics independently of tensor dimensions.

    Missing metadata is only readable as an explicitly historical input;
    callers admitting fresh learning or corrected checkpoints require it.
    Never infer a version from dimensions or relabel old arrays.
    """
    if 'action_encoding_contract' not in metadata:
        if required:
            raise ValueError('Missing action_encoding_contract; fresh learning requires re-extraction')
        return None
    version = metadata['action_encoding_contract']
    if version != ACTION_ENCODING_CONTRACT:
        raise ValueError('Unsupported action_encoding_contract: %r' % version)
    return version


def trace_action_encoding_contract(trace, *, required=False):
    return validate_action_encoding_contract(trace.get('meta') or {}, required=required)


def checkpoint_learning_lineage(net):
    """Preserve weight initialization provenance without claiming learned skill."""
    import copy
    scope = getattr(net, 'training_scope', None)
    extra = {}
    if scope is not None:
        from observation_training import validate_training_scope
        extra['training_scope'] = copy.deepcopy(validate_training_scope(scope, net=net))
    lineage = getattr(net, 'warm_start', None)
    if lineage is None:
        return extra
    if not isinstance(lineage, dict):
        raise ValueError('warm_start provenance must be a mapping')
    status = getattr(net, 'training_status', None) or 'warm_start_not_retrained'
    counts = getattr(net, 'warm_start_updates', None) or {}
    if (status not in ('warm_start_not_retrained', 'updated_after_warm_start') or
            not isinstance(counts, dict) or any(key not in ('value', 'policy') or
                type(value) is not int or value < 0 for key, value in counts.items())):
        raise ValueError('Invalid warm-start update provenance')
    if bool(sum(counts.values())) != (status == 'updated_after_warm_start'):
        raise ValueError('Warm-start training status and optimizer counts disagree')
    return {**extra, 'warm_start': copy.deepcopy(lineage), 'training_status': status,
            'warm_start_updates': dict(counts)}


def record_warm_start_update(net, component):
    """Record one completed optimizer step, never an attempted or skipped batch."""
    if getattr(net, 'warm_start', None) is None:
        return
    if component not in ('value', 'policy'):
        raise ValueError('Unknown warm-start optimizer component')
    counts = checkpoint_learning_lineage(net)['warm_start_updates']
    counts[component] = counts.get(component, 0) + 1
    net.warm_start_updates = counts
    net.training_status = 'updated_after_warm_start'


def trace_observation_contract(trace):
    """Validate explicit observation metadata without relabeling old vectors."""
    meta = trace.get('meta') or {}
    if 'observation_contract' not in meta:
        return None
    version = meta['observation_contract']
    expected = current_learning_context()['observation_contract']
    if version != expected:
        raise ValueError('Unsupported observation contract: %r' % version)
    return version


def trace_ai_policy_contract(trace):
    """Validate an explicit menu version; absent metadata stays historical.

    A versioned trace must regenerate the same projected menu used in play.
    Guessing the meaning of an unknown version could silently change targets.
    """
    from ai_policies import POLICY_VERSION
    meta = trace.get('meta') or {}
    if 'ai_policy_contract' not in meta:
        return None
    version = meta['ai_policy_contract']
    if version != POLICY_VERSION:
        raise ValueError('Unsupported AI policy contract: %r' % version)
    return version


def forced_policy_reason(row):
    """Return the recorded fixed-policy reason, or None for an ordinary row.

    Malformed annotations are refused so consumers cannot silently learn a
    supposedly fixed action or hide a genuine decision behind a truthy value.
    """
    flag = row.get('policy_forced', False)
    reason = row.get('forced_policy_reason')
    if type(flag) is not bool:
        raise ValueError('policy_forced must be boolean')
    if not flag:
        if reason is not None:
            raise ValueError('forced_policy_reason requires policy_forced=true')
        return None
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError('forced policy action requires a nonempty reason')
    if reason not in FORCED_POLICY_REASONS:
        raise ValueError('unknown forced policy reason: %r' % reason)
    if row.get('kind') != 'action' or row.get('side') not in ('FP', 'SH'):
        raise ValueError('forced policy metadata requires a player action row')
    return reason


def is_learning_decision(row):
    reason = forced_policy_reason(row)
    return (row.get('kind') == 'action' and row.get('side') in ('FP', 'SH')
            and reason is None)
