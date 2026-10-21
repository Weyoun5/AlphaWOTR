"""Bounded recovery of failed live continuations at an engine boundary.

Automatic retry preserves the failed step and its randomness journal.
Operator Skip uses the separate board-based recovery path instead.
"""


def _key(server):
    from .skip_recovery import capture
    return capture(server.driver)['key']


def _fingerprint(server):
    import hashlib
    import pickle
    try:
        engine = hashlib.sha256(pickle.dumps(server.driver.es)).hexdigest()
        return (engine, int(server.driver.n_commits_total),
                int(getattr(server.driver, 'undo_epoch', 0)),
                pickle.dumps(getattr(server.driver, 'echo_expect', {})))
    except Exception:
        return None


def _incident_state(value, _active=None):
    """Value identity across checkpoint serialization (only used on failure).

    Pickle bytes include dict/set iteration and object sharing. Restoring the
    same position can change those bytes and must not renew its retry budget.
    """
    from enum import Enum
    import random
    if isinstance(value, Enum):
        return (type(value).__name__, value.name)
    if value is None or isinstance(value, (str, int, float, bool, bytes)):
        return value
    if isinstance(value, random.Random):
        return ('Random', value.getstate())
    active = {} if _active is None else _active
    if id(value) in active:
        return ('cycle', active[id(value)])
    active[id(value)] = type(value).__name__
    def child(item):
        return _incident_state(item, active)
    try:
        if isinstance(value, dict):
            return ('dict', sorted([(child(k), child(v)) for k, v in value.items()], key=repr))
        if isinstance(value, (set, frozenset)):
            return ('set', sorted(map(child, value), key=repr))
        if isinstance(value, (list, tuple)):
            return (type(value).__name__, list(map(child, value)))
        if hasattr(value, '__dict__'):
            return (type(value).__name__, child(vars(value)))
        slots = getattr(value, '__slots__', ())
        if slots:
            return (type(value).__name__, [(k, child(getattr(value, k)))
                                          for k in slots if hasattr(value, k)])
        raise TypeError('Unsupported recovery state: ' + type(value).__name__)
    finally:
        active.pop(id(value))


def begin_step(server):
    """One proof per live AI step; never called inside search/simulations."""
    server._cycle_step_origin = _fingerprint(server)
    server._cycle_step_ref = server._open_step


def retained_retry(server):
    failure = getattr(server, '_cycle_failure', None)
    return bool(failure and failure.get('safe_retry')
                and failure.get('origin') is not None
                and server._open_step is not None
                and server._open_step[0] == failure.get('step')
                and failure['origin'] == _fingerprint(server))


def retry_skip(server, target):
    # Retry and Skip are distinct operator controls. A failed AI-owned step
    # must reach the same board-based Skip path as a failed human interaction.
    return False


def retry_started(server):
    """Remove only our own hold; retain the original cause and attempt budget."""
    failure = getattr(server, '_cycle_failure', None)
    if failure:
        server.driver.alarms[:] = [row for row in server.driver.alarms
                                  if row != failure.get('hold')]
        server.notes.append('retrying retained AI step after: ' + failure['cause'])
    server._cycle_err_sig = None
    server._cycle_err_n = 0


def manual_retry(server):
    failure = getattr(server, '_cycle_failure', None)
    if failure:
        # An explicit operator retry is separate from the one automatic retry.
        # Keep the automatic budget spent even if a harmless poll succeeds.
        failure['manual_retries'] = failure.get('manual_retries', 0) + 1
    server._reset_requested = True


def settled(server):
    """A completed interaction retires the active hold; a poll never does."""
    failure = getattr(server, '_cycle_failure', None)
    if failure and failure['key'] != _key(server):
        server._last_cycle_failure = failure
        server._cycle_failure = None
        server.driver._live_cycle_failure = False
        if server.rand.sticky_halt == failure.get('hold'):
            server.rand.sticky_halt = None


def rewind_completed(server):
    """Called only after the paired native/engine restore handshake completes."""
    failure = getattr(server, '_cycle_failure', None)
    if failure:
        server._last_cycle_failure = failure
        server.driver.alarms[:] = [row for row in server.driver.alarms
                                  if row != failure.get('hold')]
        if server.rand.sticky_halt == failure.get('hold'):
            server.rand.sticky_halt = None
    server._cycle_failure = None
    server._cycle_step_origin = None
    server.driver._live_cycle_failure = False
    server._reset_requested = False


def failed(server, error):
    """Return emitted recovery lines, or None for an unrelated fail-closed gate."""
    from .driver import DriverHalt, NativeDieFaceMismatch
    from . import chat
    failure = getattr(server, '_cycle_failure', None)
    sticky = getattr(server.rand, 'sticky_halt', None)
    if (failure and sticky == failure.get('hold') and isinstance(error, DriverHalt)
            and str(error).startswith(failure['hold'])):
        # A held status poll is not a new engine failure. In particular never
        # make `DriverHalt: COMBAT HELD (... COMBAT HELD ...)` a fresh cause.
        raise DriverHalt(failure['hold']) from None
    if isinstance(error, DriverHalt) and not isinstance(error, NativeDieFaceMismatch):
        return None
    if sticky and (failure is None or sticky != failure.get('hold')):
        # The randomness funnel also raises RuntimeError for integrity faults
        # (including journal/deck mismatch). They require explicit recovery.
        return None
    key = _key(server)
    origin = getattr(server, '_cycle_step_origin', None)
    if (server._open_step is None
            or server._open_step is not getattr(server, '_cycle_step_ref', None)
            or isinstance(error, NativeDieFaceMismatch)):
        origin = None
    safe_retry = (not isinstance(error, NativeDieFaceMismatch)
                  and origin is not None and server._open_step is not None
                  and server._open_step is getattr(server, '_cycle_step_ref', None)
                  and origin == _fingerprint(server))
    if failure is None or failure['key'] != key:
        cause = '%s: %s' % (type(error).__name__, ' '.join(str(error).split())[:600])
        history = getattr(server, '_cycle_failure_history', None)
        if history is None:
            history = server._cycle_failure_history = {}
        # Routing failures have no AI step origin. Their allowance belongs to
        # this engine position, not to every later occurrence of the same text
        # for the rest of the game. This work runs only on an actual failure.
        import hashlib
        position = hashlib.sha256(repr(_incident_state(server.driver.es)).encode()).hexdigest()
        identity = (cause, position)
        failure = history.get(identity) or {
            'key': key, 'cause': cause, 'automatic_retry': False,
            'rewind_attempted': False, 'manual_retries': 0,
            'origin': origin, 'safe_retry': safe_retry,
            'step': server._open_step[0] if server._open_step is not None else None,
        }
        failure['key'] = key
        history[identity] = server._cycle_failure = failure
        server.driver._live_cycle_failure = True
        server.driver.findings.append(('live_error_original', cause))
    else:
        current = '%s: %s' % (type(error).__name__, str(error))
        if current != failure['cause']:
            server.driver.findings.append(('live_recovery_error', current))
    cause = failure['cause']
    combat = server.driver.combat_recovery_open()
    message = ('%s: %s. Committed decisions, costs and rolls retained. '
               'Reset or retry retries retained work; rewind restores a verified checkpoint.'
               % ('COMBAT HELD' if combat else 'AI STEP HELD', cause))
    failure['hold'] = message
    server.rand.sticky_halt = message
    if message not in server.driver.alarms:
        server.driver.alarms.append(message)
    if not failure['automatic_retry'] and safe_retry:
        failure['automatic_retry'] = True
        server._reset_requested = True
        server.notes.append('automatic retained-state retry scheduled: ' + cause)
        return [server._say('prompt', chat._register(
            'AI step stopped: %s. Retaining committed results and retrying once.' % cause))]
    if not failure['rewind_attempted']:
        failure['rewind_attempted'] = True
        server.notes.append('retained retry failed or state changed; requesting verified rewind: ' + cause)
        from . import turn_rewind
        try:
            return list(turn_rewind.recover(server))
        except Exception as recovery_error:
            server.driver.findings.append(('battle_rewind_error', str(recovery_error)))
            server.notes.append('turn rewind failed; checkpoint and original error retained: '
                                + str(recovery_error))
    raise DriverHalt(message) from error
