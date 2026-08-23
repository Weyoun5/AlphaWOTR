# -*- coding: ascii -*-
"""pid_owner.py - THE ONE ANSWERER for 'does this pid belong to us?'

P1 family (2026-08-16, two live failures in one afternoon):
  * a stale ai_server.lock held pid 19612, which Windows had REUSED
    for Discord.exe - the pid-exists check read that as 'live server'
    and refused the USER's only healthy server start;
  * ai_panel.pid carries the same class: a reused pid makes the panel
    silently refuse to open.

DOCTRINE (handoff sharp edge, verbatim): "Liveness is a heartbeat ...
never a pid: pids are reused."  Where a heartbeat is not available,
ownership must be PROVEN from the process command line, and a failed
probe must fall back CONSERVATIVELY (caller's choice), never silently.

Every pid-based ownership/refusal check in the bridge routes through
owner_check() - do not add a second implementation.

----------------------------------------------------------------------
USER-FACING FIX 2026-08-18, reported verbatim: "a bunch of powershell
windows saying [error 2147942632 (0x800700e8) ...] keep getting
launched during your work".

MEASURED MECHANISM (not inferred - reproduced 2026-08-18 with a
pythonw parent and a reporting child calling GetConsoleWindow):

    parent (pythonw.exe)         console hwnd 0      no console
    child, plain spawn           console hwnd 3868876, VISIBLE
    child, CREATE_NO_WINDOW      console hwnd 0      no window

'-NoProfile' and '-NonInteractive' have NOTHING to do with the window.
On Windows a console-subsystem child gets its OWN console unless the
parent passes CREATE_NO_WINDOW - and because the panel and the server
run under pythonw.exe (which has no console to inherit), EVERY console
child they spawn opened a NEW VISIBLE WINDOW on the USER's desktop.
Launched from a terminal the same code shows nothing, because the
child inherits the terminal's console.  That is why it survived
review.

THREE THINGS WERE WRONG AND ALL THREE ARE FIXED HERE:

1. IT SPAWNED A VISIBLE CONSOLE WINDOW.  The probe shelled out to
   powershell.  THERE IS NOW NO SUBPROCESS ON ANY PATH IN THIS FILE -
   not a suppressed one, none at all.  A suppressor is a promise that
   the next editor has to keep; no spawn is a property of the code.
   Ownership is read IN-PROCESS: psutil when it is installed (it is
   NOT in requirements.txt, so it cannot be assumed), otherwise a
   ctypes call to NtQueryInformationProcess for the command line.
   When neither can answer, the answer is 'cannot prove' and the
   caller applies its own conservative fallback.

2. IT SPAWNED ONE POWERSHELL PER PROBE.  A guard that costs a process
   spawn per call gets called in a loop by someone eventually - and it
   already was.  The answer is now cached per pid.  A pid's command
   line cannot change while that pid lives, but a pid CAN be recycled,
   so the cache entry is keyed by the process CREATION TIME as well:
   a recycled pid presents a different creation time and re-probes.
   (Same idea BOOT-1 used on the java side: identity from start time,
   not from a command line.)

3. IT SURFACED THE RACE AS AN ERROR.  0x800700e8 is ERROR_NO_DATA: the
   pipe closing because the target exited between enumeration and
   query.  A pid dying mid-probe is the ORDINARY case, not an incident.
   A process that is simply gone is a real answer ('' - not our
   process); a probe that cannot see the process at all returns
   'cannot prove ownership', which every caller already treats as
   do-not-adopt, and says so to the LOG STREAM, never to a window.

Gate: WOTR_AI_training_lab/streak/test_no_console_windows.py drives
this file from a real pythonw parent and fails if any spawn happens or
if any console window appears.  Its sensitivity control spawns an
unsuppressed console child and REQUIRES a window, so a clean result
here is never a detector that stopped working.
"""
import sys

# a pid's command line cannot change while that pid lives; the entry
# is (create_time_or_None, answer) so a RECYCLED pid re-probes
_CACHE = {}

_IS_WIN = sys.platform.startswith('win')


def _log(msg):
    """To the log stream (the panel and the server redirect stdout to
    their proc logs), never to a window."""
    try:
        print('    [pid_owner] %s' % msg)
    except Exception:
        pass


# ---------------------------------------------------------------- win32
# In-process readers.  Every one of these is a library call; none of
# them creates a process, and therefore none of them can create a
# window.

_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_PROCESS_QUERY_INFORMATION = 0x0400
_PROCESS_VM_READ = 0x0010
_STATUS_INFO_LENGTH_MISMATCH = 0xC0000004
_ProcessBasicInformation = 0
_ProcessCommandLineInformation = 60


def _open_process(access, pid):
    import ctypes
    h = ctypes.windll.kernel32.OpenProcess(access, False, int(pid))
    return h or None


def _ctypes_command_line(pid):
    """The command line of `pid` with no subprocess and no psutil.

    Returns the command line, '' when no such process exists, or None
    when we could not prove anything."""
    if not _IS_WIN:
        return None
    import ctypes
    from ctypes import wintypes

    class _UNICODE_STRING(ctypes.Structure):
        _fields_ = [('Length', wintypes.USHORT),
                    ('MaximumLength', wintypes.USHORT),
                    ('Buffer', ctypes.c_void_p)]

    ntdll = ctypes.windll.ntdll
    h = _open_process(_PROCESS_QUERY_LIMITED_INFORMATION, pid)
    if not h:
        # ERROR_INVALID_PARAMETER (87) means there is no such process -
        # a real answer.  Anything else (access denied, ...) means we
        # cannot prove ownership.
        err = ctypes.windll.kernel32.GetLastError()
        return '' if err == 87 else None
    try:
        # Windows 8.1+: ask for the command line directly.  Needs only
        # QUERY_LIMITED_INFORMATION and no memory reads.
        need = wintypes.ULONG(0)
        st = ntdll.NtQueryInformationProcess(
            ctypes.c_void_p(h), _ProcessCommandLineInformation,
            None, 0, ctypes.byref(need))
        if (st & 0xFFFFFFFF) == _STATUS_INFO_LENGTH_MISMATCH \
                and need.value:
            buf = ctypes.create_string_buffer(need.value)
            st = ntdll.NtQueryInformationProcess(
                ctypes.c_void_p(h), _ProcessCommandLineInformation,
                buf, need.value, ctypes.byref(need))
            if (st & 0xFFFFFFFF) == 0:
                us = ctypes.cast(
                    buf, ctypes.POINTER(_UNICODE_STRING)).contents
                if us.Buffer and us.Length:
                    return ctypes.wstring_at(us.Buffer, us.Length // 2)
                return ''
        return _peb_command_line(pid)
    except Exception as e:
        _log('ctypes probe failed for pid %s (%r) - treating as '
             'unknown' % (pid, e))
        return None
    finally:
        try:
            ctypes.windll.kernel32.CloseHandle(h)
        except Exception:
            pass


def _peb_command_line(pid):
    """Older-Windows fallback: walk the PEB.  Needs VM_READ."""
    import ctypes
    from ctypes import wintypes

    class _UNICODE_STRING(ctypes.Structure):
        _fields_ = [('Length', wintypes.USHORT),
                    ('MaximumLength', wintypes.USHORT),
                    ('Buffer', ctypes.c_void_p)]

    class _PBI(ctypes.Structure):
        _fields_ = [('Reserved1', ctypes.c_void_p),
                    ('PebBaseAddress', ctypes.c_void_p),
                    ('Reserved2', ctypes.c_void_p * 2),
                    ('UniqueProcessId', ctypes.c_void_p),
                    ('Reserved3', ctypes.c_void_p)]

    is64 = ctypes.sizeof(ctypes.c_void_p) == 8
    off_params = 0x20 if is64 else 0x10
    off_cmdline = 0x70 if is64 else 0x40
    h = _open_process(_PROCESS_QUERY_INFORMATION | _PROCESS_VM_READ,
                      pid)
    if not h:
        return None
    k32 = ctypes.windll.kernel32
    try:
        pbi = _PBI()
        got = wintypes.ULONG(0)
        st = ctypes.windll.ntdll.NtQueryInformationProcess(
            ctypes.c_void_p(h), _ProcessBasicInformation,
            ctypes.byref(pbi), ctypes.sizeof(pbi), ctypes.byref(got))
        if (st & 0xFFFFFFFF) != 0 or not pbi.PebBaseAddress:
            return None

        def _read(addr, size):
            b = ctypes.create_string_buffer(size)
            n = ctypes.c_size_t(0)
            ok = k32.ReadProcessMemory(
                ctypes.c_void_p(h), ctypes.c_void_p(addr), b, size,
                ctypes.byref(n))
            if not ok or n.value != size:
                return None
            return b

        b = _read(pbi.PebBaseAddress + off_params,
                  ctypes.sizeof(ctypes.c_void_p))
        if b is None:
            return None
        params = int.from_bytes(b.raw, 'little')
        if not params:
            return None
        b = _read(params + off_cmdline, ctypes.sizeof(_UNICODE_STRING))
        if b is None:
            return None
        us = ctypes.cast(b, ctypes.POINTER(_UNICODE_STRING)).contents
        if not us.Buffer or not us.Length:
            return ''
        b = _read(us.Buffer, us.Length)
        if b is None:
            return None
        return b.raw.decode('utf-16-le', 'replace')
    except Exception as e:
        _log('peb probe failed for pid %s (%r) - treating as unknown'
             % (pid, e))
        return None
    finally:
        try:
            k32.CloseHandle(h)
        except Exception:
            pass


def _create_time(pid):
    """Process creation time (an opaque comparable), or None.  Used to
    make the per-pid cache immune to pid recycling.  No subprocess."""
    try:
        import psutil
        try:
            return psutil.Process(int(pid)).create_time()
        except Exception:
            return None
    except ImportError:
        pass
    if not _IS_WIN:
        return None
    try:
        import ctypes
        from ctypes import wintypes
        h = _open_process(_PROCESS_QUERY_LIMITED_INFORMATION, pid)
        if not h:
            return None
        try:
            c, e, k, u = (wintypes.FILETIME(), wintypes.FILETIME(),
                          wintypes.FILETIME(), wintypes.FILETIME())
            ok = ctypes.windll.kernel32.GetProcessTimes(
                ctypes.c_void_p(h), ctypes.byref(c), ctypes.byref(e),
                ctypes.byref(k), ctypes.byref(u))
            if not ok:
                return None
            return (c.dwHighDateTime << 32) | c.dwLowDateTime
        finally:
            ctypes.windll.kernel32.CloseHandle(h)
    except Exception:
        return None


def _probe(pid):
    """One uncached lookup.  psutil first (it answers without touching
    ctypes at all), then the in-process win32 route."""
    try:
        import psutil
    except ImportError:
        return _ctypes_command_line(pid)
    try:
        return ' '.join(psutil.Process(pid).cmdline() or ())
    except psutil.NoSuchProcess:
        return ''                       # no such process: a real answer
    except psutil.AccessDenied:
        # a protected process; the win32 route will not do better, but
        # try it rather than guess
        return _ctypes_command_line(pid)
    except Exception as e:
        _log('psutil probe failed for pid %s (%r) - falling back'
             % (pid, e))
        return _ctypes_command_line(pid)


def pid_command_line(pid):
    """The command line of the process holding `pid`, or '' when no
    such process exists, or None when the probe itself could not prove
    anything (caller must fall back conservatively).

    Never spawns a process, so it can never open a window."""
    try:
        key = int(pid)
    except (TypeError, ValueError):
        return None
    ct = _create_time(key)
    hit = _CACHE.get(key)
    if hit is not None and hit[0] == ct:
        return hit[1]
    out = _probe(key)
    # only a real answer is cached: None ('cannot prove') is a
    # transient verdict and must be re-probed next time
    if out is not None:
        _CACHE[key] = (ct, out)
    return out


def owner_check(pid, needle):
    """(probe_ok, is_owner): is_owner is True only when the process on
    `pid` exists AND its command line contains `needle`.  probe_ok
    False means the answer is unknown - the caller must apply its own
    conservative fallback (and say so out loud)."""
    cl = pid_command_line(pid)
    if cl is None:
        _log('cannot prove ownership of pid %s - caller falls back'
             % (pid,))
        return False, False
    return True, (needle in cl)


def pid_alive_fallback(pid):
    """The OLD pid-exists check, exposed ONLY as the conservative
    fallback for a failed probe.  Never call this as the primary
    answer - a reused pid passes it."""
    try:
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if h:
            ctypes.windll.kernel32.CloseHandle(h)
            return True
    except Exception:
        pass
    return False
