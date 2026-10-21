"""Exclusive process ownership of the persistent Java-compatible alive byte."""
import os
import threading

_guard = threading.RLock()
_owners = {}


def claim(path, offset):
    key = os.path.normcase(os.path.realpath(path))
    with _guard:
        if key in _owners:
            raise OSError('This process already has a server for this client directory')
        handle = os.fdopen(os.open(path, os.O_CREAT | os.O_RDWR, 0o600), 'r+b', buffering=0)
        try:
            handle.seek(offset)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.lockf(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB, 1, offset)
            # A losing contender must never truncate or replace the owner's pid.
            handle.seek(0)
            handle.write(str(os.getpid()).encode('ascii'))
            handle.truncate()
            handle.flush()
        except BaseException:
            handle.close()
            raise
        _owners[key] = handle
        return handle


def release(path, handle):
    key = os.path.normcase(os.path.realpath(path))
    with _guard:
        if _owners.get(key) is not handle:
            return
        try:
            # Closing releases the OS lock, including after a process crash.
            handle.close()
        finally:
            del _owners[key]
        # Keep the inode/path: unlinking it lets another opener lock a distinct
        # file while an earlier contender still holds a handle to the old one.
