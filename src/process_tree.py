"""Own and terminate a subprocess tree without targeting the caller's group.

Windows uses the native taskkill executable directly, with no shell. POSIX
children start new sessions so a batch's workers and server share an owned
process group. Descendants that deliberately escape that group are outside
this contract; the training workers do not create independent sessions.
"""
import os
import signal
import subprocess
import time
import copy


class ProcessTreeError(RuntimeError):
    pass


class ProcessTree:
    """A tree created by this module; callers cannot supply an arbitrary PID."""

    def __init__(self, args, *, cwd=None, env=None):
        self.platform = os.name
        options = {'cwd': cwd, 'env': env}
        if self.platform == 'nt':
            options['creationflags'] = (subprocess.CREATE_NEW_PROCESS_GROUP
                                        | subprocess.CREATE_NO_WINDOW)
        elif self.platform == 'posix':
            options['start_new_session'] = True
        else:
            raise ProcessTreeError('Unsupported process-tree platform: %s' % self.platform)
        self.process = subprocess.Popen(args, **options)
        self.pid = self.process.pid
        self._termination_receipt = None

    @property
    def termination_receipt(self):
        """A copy of the receipt created only by successful owned-tree cleanup."""
        return copy.deepcopy(self._termination_receipt)

    def wait(self, timeout=None):
        return self.process.wait(timeout=timeout)

    def poll(self):
        return self.process.poll()

    def terminate(self, *, grace_seconds=2.0, kill_timeout=10.0):
        """Bound cleanup; raise rather than certify cleanup after a failure."""
        if grace_seconds < 0 or kill_timeout <= 0:
            raise ValueError('Invalid process-tree cleanup timeouts')
        if self.pid != self.process.pid or self.pid <= 0:
            raise ProcessTreeError('Process-tree ownership changed')
        if self.platform == 'nt':
            receipt = self._terminate_windows(kill_timeout)
        elif self.platform == 'posix':
            receipt = self._terminate_posix(grace_seconds, kill_timeout)
        else:
            raise ProcessTreeError('Unsupported process-tree platform: %s' % self.platform)
        self._termination_receipt = copy.deepcopy(receipt)
        return receipt

    def _terminate_windows(self, timeout):
        # Keep the root alive until taskkill can enumerate its descendants.
        # A completed root is not evidence that detached descendants ended.
        if self.process.poll() is not None:
            raise ProcessTreeError('Tree root already exited; Windows descendant ownership is unavailable')
        try:
            result = subprocess.run(
                ['taskkill', '/PID', str(self.pid), '/T', '/F'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=timeout, check=False,
                creationflags=subprocess.CREATE_NO_WINDOW)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ProcessTreeError('Windows process-tree termination failed: %s' % exc) from exc
        if result.returncode != 0:
            diagnostic = (result.stdout + result.stderr).decode('utf-8', 'replace')[-1000:]
            raise ProcessTreeError('Windows taskkill failed (%d): %s' % (result.returncode, diagnostic))
        try:
            code = self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise ProcessTreeError('Windows tree root did not exit after taskkill') from exc
        return {'platform': 'nt', 'pid': self.pid, 'method': 'taskkill_tree',
                'root_returncode': code}

    def _terminate_posix(self, grace_seconds, kill_timeout):
        if self.pid == os.getpgrp():
            raise ProcessTreeError('Refusing to signal the caller process group')
        # The new-session launch owns pgid == root pid, even if the root exits
        # while a grandchild is still running. Do not infer group completion
        # solely from the root's wait status.
        def exists():
            try:
                os.killpg(self.pid, 0)
                return True
            except ProcessLookupError:
                return False

        def send(sig):
            try:
                os.killpg(self.pid, sig)
                return True
            except ProcessLookupError:
                return False

        sent_term = send(signal.SIGTERM)
        deadline = time.monotonic() + grace_seconds
        while exists() and time.monotonic() < deadline:
            self.process.poll()
            time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
        sent_kill = send(signal.SIGKILL) if exists() else False
        kill_deadline = time.monotonic() + kill_timeout
        try:
            code = self.process.wait(timeout=kill_timeout)
        except subprocess.TimeoutExpired as exc:
            raise ProcessTreeError('POSIX tree root did not exit after group termination') from exc
        # wait(root) cannot prove that a worker or server also exited. Refuse
        # archiving while any process still occupies our group, even if only
        # an unreaped descendant remains on a host with unusual init behavior.
        while exists() and time.monotonic() < kill_deadline:
            time.sleep(min(0.02, max(0.0, kill_deadline - time.monotonic())))
        if exists():
            raise ProcessTreeError('POSIX process group still exists after bounded termination')
        return {'platform': 'posix', 'pid': self.pid, 'method': 'owned_process_group',
                'sent_sigterm': sent_term, 'sent_sigkill': sent_kill,
                'root_returncode': code}
