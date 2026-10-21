"""Bounded cleanup for direct served-runner child processes."""
import time


def _join_until(processes, seconds):
    deadline = time.monotonic() + seconds
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))


def cleanup(workers, server, control, *, join_seconds=5.0, kill_seconds=5.0):
    """Join workers, send stop on a dedicated pipe, then bound any fallback.

    The control pipe carries no inference payload and receives exactly one
    small stop message. No worker shares its writer. This avoids writing a
    stop behind a large request on a stalled worker's pipe.
    """
    if join_seconds < 0 or kill_seconds <= 0:
        raise ValueError('Invalid served cleanup timeouts')
    report = {'worker_terminated': [], 'worker_killed': [],
              'stop_sent': False, 'server_terminated': False,
              'server_killed': False, 'errors': []}
    started = [(wid, process) for wid, process in workers if process.pid is not None]
    _join_until([process for _, process in started], join_seconds)
    for wid, process in started:
        if process.is_alive():
            try:
                process.terminate()
                report['worker_terminated'].append(wid)
            except OSError as exc:
                report['errors'].append('worker %s terminate: %r' % (wid, exc))
    _join_until([process for _, process in started], kill_seconds)
    for wid, process in started:
        if process.is_alive():
            try:
                process.kill()
                report['worker_killed'].append(wid)
            except OSError as exc:
                report['errors'].append('worker %s kill: %r' % (wid, exc))
    _join_until([process for _, process in started], kill_seconds)
    if server is not None and server.pid is not None:
        if server.is_alive() and control is not None:
            try:
                control.send(('stop',))
                report['stop_sent'] = True
            except (EOFError, OSError) as exc:
                report['errors'].append('server stop: %r' % exc)
        server.join(timeout=join_seconds)
        if server.is_alive():
            try:
                server.terminate()
                report['server_terminated'] = True
            except OSError as exc:
                report['errors'].append('server terminate: %r' % exc)
        server.join(timeout=kill_seconds)
        if server.is_alive():
            try:
                server.kill()
                report['server_killed'] = True
            except OSError as exc:
                report['errors'].append('server kill: %r' % exc)
        server.join(timeout=kill_seconds)
    report['worker_exitcodes'] = {wid: process.exitcode for wid, process in started}
    report['server_exitcode'] = server.exitcode if server is not None else None
    report['alive'] = ['worker_%s' % wid for wid, process in started if process.is_alive()]
    if server is not None and server.pid is not None and server.is_alive():
        report['alive'].append('server')
    return report
