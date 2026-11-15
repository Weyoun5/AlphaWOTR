"""Exclusive process ownership of the persistent Java-compatible alive byte.

THE one proof that a server runs for a client directory, for every asker (a second server,
the launcher, the Java client): a live process holds an exclusive OS lock on one byte of the
directory's alive file, and the file starts with that process's pid.  The OS frees the byte
the instant its holder dies, and only the holder writes its pid, so the proof needs no
process list and no command line, and a reused pid cannot pass it."""
import os
import threading
import time

_guard = threading.RLock()
_owners = {}


def _lock(handle, offset, take=True):
    handle.seek(offset)
    if os.name == 'nt':
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK if take else msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.lockf(handle.fileno(), (fcntl.LOCK_EX | fcntl.LOCK_NB) if take else fcntl.LOCK_UN,
                    1, offset)


def holder(path, offset):
    """The pid of the live process that holds the alive byte of `path`, or None when none
    does (the pid in the file is then a dead owner's and means nothing)."""
    def pid():
        handle.seek(0)
        text = handle.read(32).decode('ascii', 'replace').strip()
        return int(text) if text.isdigit() else None
    try:
        handle = open(path, 'r+b', buffering=0)
    except OSError:
        return None
    with handle:
        try:
            _lock(handle, offset)
        except OSError:
            # held.  A new owner writes its pid right after it took the byte: the pid counts
            # only when it reads the same twice, a moment apart
            first = pid()
            time.sleep(0.05)
            return first if first == pid() else None
        _lock(handle, offset, take=False)
        return None


def claim(path, offset):
    key = os.path.normcase(os.path.realpath(path))
    with _guard:
        if key in _owners:
            raise OSError('This process already has a server for this client directory')
        handle = os.fdopen(os.open(path, os.O_CREAT | os.O_RDWR, 0o600), 'r+b', buffering=0)
        try:
            _lock(handle, offset)
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
