"""Recover silent internal AI stalls; ordinary human/Java waits are excluded."""
import time

STALL_SECONDS = 45.0


def poll(server, output):
    from . import turn_rewind
    driver = server.driver
    rewind = turn_rewind._state(server)
    step = server._open_step
    if (output or step is None or rewind.phase != 'idle' or rewind.review
            or server.rand.pending() is not None or server.rand.sticky_halt
            or driver.game_over is not None or driver.echo_pending()
            or driver.human_physical_work_pending() or driver._human_window_open()
            or driver.deferred is not None or driver.chat_deferred is not None
            or driver.pending_buf is not None or getattr(server, '_manual_hold', False)):
        server._idle_recovery_watch = None
        return output
    import ai
    pending = ai.describe_pending(driver.es)
    if pending is not None and pending['side'] is server.human_side:
        server._idle_recovery_watch = None
        return output
    key = (int(driver.es.turn), driver.phase, step[0], driver.n_commits_total,
           getattr(driver, 'undo_epoch', 0))
    now = time.monotonic()
    watch = getattr(server, '_idle_recovery_watch', None)
    if watch is None or watch[0] != key:
        server._idle_recovery_watch = (key, now)
        return output
    if now - watch[1] < STALL_SECONDS:
        return output
    # Do not rewind an incomplete/older export or a human's unfinished piece
    # movement. The checkpoint already owns its matching hidden state.
    from .turn_bookkeeping import _fresh_board
    dump = _fresh_board(server)
    if dump is None:
        return output
    driver.view.refresh_dump()
    rows, staging = driver.census_rows()
    if rows or staging:
        return output
    checkpoint = rewind.latest_id
    if checkpoint is None:
        return output
    attempts = getattr(server, '_idle_recovery_attempts', None)
    if attempts is None:
        attempts = server._idle_recovery_attempts = set()
    # Re-capturing an unchanged board gives it a new checkpoint ID. That must
    # not grant another automatic retry of the same stalled continuation.
    try:
        board_hash = turn_rewind._proof(server)['board_sha256']
    except (OSError, ValueError):
        return output
    identity = (key[0], key[1], key[2], key[3], board_hash)
    repeated = identity in attempts
    attempts.add(identity)
    server._idle_recovery_watch = None
    server.notes.append('silent AI stall: %s; restoring matched board checkpoint%s' %
                        (step[0], ' for operator review after repeated stall' if repeated else ''))
    return turn_rewind.request(server, checkpoint_id=checkpoint,
                               session=str(driver.session_nonce), review=repeated)
