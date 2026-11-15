# -*- coding: utf-8 -*-
"""Served-runner lifetime evidence for pre-ready recovery (R2-SL-28).

USER standing directive: "broadly speaking, we need to prevent all softlocks".
A served attempt left in pre-ready stage 'launching' (runner killed from
outside, loop and runner killed together, runner dead before its first
phase.json) used to refuse every later loop start, with no automated exit.
run_sp_served.run() now leaves three facts in its fresh output directory and
PreReadyRecovery reads them.  Nothing in this module signals or kills a
process by PID.

  runner.lock        an exclusive, non-inheritable OS lock the runner holds for
                     its whole life.  The OS drops it when the runner exits or
                     is killed, so "lock acquirable" proves the runner is gone.
  runner_guard.json  pid, platform, start time, and whether every served child
                     sits in a Windows job object with
                     JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE.  Only the runner holds
                     the job handle (never inherited), so the kernel kills those
                     children when the runner ends, however it ends.
  spawning.json      written and fsynced immediately before the runner starts
                     its first served child.  If the runner has exited and this
                     file is absent, no served child was ever started.

launch.json declares this protocol (PROTOCOL), so an older runner's output,
which has no declaration, is never certified by the absence of a marker.
"""
import json
import os
import time

SCHEMA = 'served_runner_lifetime_guard_v1'
LOCK_NAME = 'runner.lock'
GUARD_NAME = 'runner_guard.json'
SPAWN_MARKER = 'spawning.json'
JOB_KIND = 'kill_on_job_close'
PROTOCOL = {'schema': SCHEMA, 'lock': LOCK_NAME, 'guard': GUARD_NAME, 'spawn_marker': SPAWN_MARKER}

_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000


def _lock(handle):
    """Exclusive, non-blocking lock of byte 0; OSError while another handle holds it."""
    handle.seek(0)
    if os.name == 'nt':
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(handle):
    handle.seek(0)
    if os.name == 'nt':
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def probe_lock(path):
    """'absent', 'held' (its owner is alive) or 'released' (its owner has exited).

    Any failure to take the lock counts as 'held': the caller then refuses."""
    try:
        handle = open(path, 'r+b')
    except FileNotFoundError:
        return 'absent'
    with handle:
        try:
            _lock(handle)
        except OSError:
            return 'held'
        _unlock(handle)
    return 'released'


def _loud(text):
    print('[runner-lifetime-guard] ' + text, flush=True)


class _WindowsJob:
    """An unnamed job object with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE."""

    def __init__(self):
        import ctypes
        from ctypes import wintypes

        class Basic(ctypes.Structure):
            _fields_ = [('process_time', ctypes.c_longlong), ('job_time', ctypes.c_longlong),
                        ('flags', wintypes.DWORD), ('min_ws', ctypes.c_size_t), ('max_ws', ctypes.c_size_t),
                        ('process_limit', wintypes.DWORD), ('affinity', ctypes.c_size_t),
                        ('priority', wintypes.DWORD), ('scheduling', wintypes.DWORD)]

        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in
                        ('read_ops', 'write_ops', 'other_ops', 'read_bytes', 'write_bytes', 'other_bytes')]

        class Extended(ctypes.Structure):
            _fields_ = [('basic', Basic), ('io', IO), ('process_memory', ctypes.c_size_t),
                        ('job_memory', ctypes.c_size_t), ('peak_process', ctypes.c_size_t),
                        ('peak_job', ctypes.c_size_t)]

        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel.SetInformationJobObject.restype = wintypes.BOOL
        kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel.IsProcessInJob.restype = wintypes.BOOL
        kernel.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
        kernel.CloseHandle.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self._ctypes, self._wintypes, self._kernel = ctypes, wintypes, kernel
        # Default security attributes: the handle is never inherited by children.
        self.handle = kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise OSError('CreateJobObjectW failed: winerror %d' % ctypes.get_last_error())
        limits = Extended()
        limits.basic.flags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(self.handle, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                                              ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.get_last_error()
            self.close()
            raise OSError('SetInformationJobObject(KILL_ON_JOB_CLOSE) failed: winerror %d' % error)

    def assign(self, process_handle):
        handle = self._wintypes.HANDLE(int(process_handle))
        if not self._kernel.AssignProcessToJobObject(self.handle, handle):
            raise OSError('AssignProcessToJobObject failed: winerror %d' % self._ctypes.get_last_error())
        inside = self._wintypes.BOOL()
        if not self._kernel.IsProcessInJob(handle, self.handle, self._ctypes.byref(inside)) or not inside.value:
            raise OSError('IsProcessInJob does not confirm the job assignment')

    def close(self):
        if self.handle:
            handle, self.handle = self.handle, None
            self._kernel.CloseHandle(handle)


class RunnerLifetimeGuard:
    """Held by run_sp_served.run() from just after its output is reserved until it returns.

    Order inside the runner: guard created (lock + job + runner_guard.json),
    spawning() immediately before the first served child starts, adopt() right
    after each child starts, close() after the children were joined and the
    final phase.json was published."""

    def __init__(self, output):
        self.output = os.path.abspath(output)
        self._handle = None
        self._job = None
        self._spawn_marked = False
        handle = open(os.path.join(self.output, LOCK_NAME), 'xb')     # a fresh output; never adopt a lock
        try:
            _lock(handle)
        except BaseException:
            handle.close()
            raise
        self._handle = handle
        try:
            if os.name == 'nt':
                try:
                    self._job = _WindowsJob()
                    job = {'kind': JOB_KIND, 'active': True}
                except OSError as exc:
                    job = {'kind': JOB_KIND, 'active': False, 'reason': 'job setup failed: %s' % exc}
                    _loud('KILL-ON-CLOSE JOB UNAVAILABLE (%s): an outside kill of this runner will need the '
                          'operator escape run_td_v33loop.py --abandon-pre-ready' % exc)
            else:
                job = {'kind': None, 'active': False,
                       'reason': 'no kernel job on %s; an interrupted spawned attempt needs the operator escape'
                                 % os.name}
            self.state = {'schema': SCHEMA, 'runner_pid': os.getpid(), 'platform': os.name,
                          'started_at': time.time(), 'job': job}
            self._publish()
        except BaseException:
            self.close()
            raise

    def _publish(self):
        path = os.path.join(self.output, GUARD_NAME)
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as stream:
            json.dump(self.state, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)

    def spawning(self):
        """Durably record, BEFORE the first served child exists, that children may exist."""
        with open(os.path.join(self.output, SPAWN_MARKER), 'x', encoding='utf-8') as stream:
            json.dump({'schema': SCHEMA, 'runner_pid': os.getpid(), 'at': time.time()}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        self._spawn_marked = True

    def adopt(self, process):
        """Put a just-started multiprocessing child in the kill-on-close job (Windows)."""
        if not self._spawn_marked:
            raise RuntimeError('Served child started before the spawn marker was written')
        if self._job is None or not self.state['job'].get('active'):
            return
        try:
            # multiprocessing on Windows: Process.sentinel is the child's process handle.
            self._job.assign(process.sentinel)
        except (OSError, ValueError) as exc:
            self.state['job'] = dict(self.state['job'], active=False,
                                     reason='child pid %s is not in the job: %s' % (process.pid, exc))
            self._publish()
            _loud('CHILD OUTSIDE THE KILL-ON-CLOSE JOB (pid %s: %s): an outside kill of this runner will need '
                  'the operator escape run_td_v33loop.py --abandon-pre-ready' % (process.pid, exc))

    def close(self):
        """Children are joined by now; closing the job kills any the runner could not join."""
        try:
            if self._job is not None:
                job, self._job = self._job, None
                job.close()
        finally:
            if self._handle is not None:
                handle, self._handle = self._handle, None
                try:
                    _unlock(handle)
                finally:
                    handle.close()
