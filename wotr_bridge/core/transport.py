"""File-bridge transport: the request/response serve loop + panel surface.

LIFTED from wotr_bridge/ai_move_server.py (read-only reference; nothing
is imported from it):
  * serve()                 :18324-18552  atomic request/response
        handshake, single-writer lock, heartbeat, pause discipline
  * atomic_write()          :18317-18321  temp + os.replace (no partial
        reads, ever)
  * parse_request()         :265-275      the request-file grammar
        (line 1 = side header, rest = log lines)
  * _write_status()         :18177-18253  panel status JSON with the
        BOOT-1 ownership discipline (foreign-pid session keys popped;
        pid/client_dir/ts stamped UNCONDITIONALLY, outside any guarded
        block)
  * _read_panel_control()   :18259-18314  panel control JSON (paused
        honored here; everything else surfaced to the driver)
  * the exclusive alive-byte lock (server_ownership) is the ONE proof
    that a server runs for this client dir: it refuses a second
    server, and the launcher and the Java client ask the same byte.
    Liveness is never a bare pid (Windows reuses pids) nor a command
    line.

THIS MODULE KNOWS NOTHING ABOUT THE GAME.  It moves request text in and
response lines out.  Response lines may mix plain inject lines with
`!CALL` directives (PROTOCOL.md, Directive protocol): AiOpp executes
them IN ORDER on the EDT - to the transport they are all just lines.

HANDSHAKE SEMANTICS (measured on the old loop, kept exactly):
  * the client writes REQ; we consume it ATOMICALLY (os.replace to a
    .processing name FIRST, then read, then delete).  The rename is the
    dedup: one request file is handled exactly once even when the
    delete fails (the old read-then-remove flow double-processed a
    request under a Windows file lock - the engine advanced two actions
    and the second response overwrote the first).
  * deletion is the handshake: a request REWRITTEN by the client (a
    fresh manual click, byte-identical content or not) is a NEW request
    and gets a fresh response - the dedup is per consumed FILE, never
    per content.
  * responses are written with atomic_write (sibling temp + rename):
    the response file either does not exist or is complete.
  * an empty reply is a real reply (empty file = no-op to AiOpp).

DELIBERATE DEVIATIONS from the old serve() (each noted):
  * decide/parrot/reference logic is the handler's business - serve()
    takes handler(request) -> lines or (lines, how_string).
  * the panel-process spawn (:18395-18440) stays with the entry point
    (live_server.py) - spawning a UI is not transport.
  * the AUTO pause-keepalive chat line (:18494-18508) was game-flavored
    (side + nick + chat grammar); it is now the `paused_reply` hook.
    The pause semantics themselves (no handler call, empty atomic
    reply, status 'paused') are unchanged.
  * refusal-to-start raises BridgeLockError (RuntimeError) instead of
    SystemExit so a driver embedding the bridge can report it.
  * the Java 8 Stop AI marker ends this process after its current
    request, releasing ownership before another server may start.
  * R3-PANEL-UNPAUSE-CHAT: the driver-owned `gate_escape` hook runs
    ahead of the pause gate (the chat backup for a closed panel's PAUSE /
    MANUAL EDIT); an answered escape is that cycle's whole reply.
"""
import json
import os
import time
import threading
from contextlib import contextmanager
from collections import namedtuple

from .server_ownership import claim, release
# L1104-F13 (OVERVIEW D2): the status text is never cut mid-word ([:140] cut
# the reveal list "... fords of bruinen, go"); fit() keeps it whole

_STATUS_ERROR_LIMIT = 240


def _status_fit(text, limit=600):
    """The published status text, whole words, at most `limit` characters."""
    text = ' '.join(str(text).split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(' ', 1)[0].rstrip(' ,;:') + ' ...'

# Default file names - the AiOpp contract.  All parameterized on
# FileBridge; nothing below hardcodes a client directory.
REQ_NAME = 'ai_bridge_request.log'
RESP_NAME = 'ai_bridge_move.txt'
STATUS_NAME = 'ai_bridge_status.json'
CONTROL_NAME = 'ai_panel_control.json'
PANEL_REQUEST_NAME = 'ai_panel_request.json'     # the in-client panel's one request slot
STOP_NAME = 'ai_server.stop'
# LIVENESS LOCK (2026-08-23, Java 8 support): a second file the server
# keeps OPEN with an exclusive OS byte lock for its whole lifetime.  The
# OS releases it the instant the process dies (crash included), so any
# reader can prove "a live server is writing for this client dir" with
# one non-blocking lock attempt - no pid API, no child process.  The
# client's Java 8 path (no ProcessHandle) proves server ownership this
# way; Java 9+ keeps its ProcessHandle proof unchanged.
ALIVE_NAME = 'ai_server.alive'
# The lock byte sits FAR past the pid text: msvcrt locks are MANDATORY
# on Windows, so a lock on byte 0 blocked the client's READ of the pid
# itself (tester 2026-08-24: 'The process cannot access the file
# because another process has locked a portion of the file') and the
# proof refused every LIVE server.  At 1 MiB no reader of the pid (an
# 8 KiB FileReader buffer) can touch the locked byte; locking past EOF
# is legal on Windows.  AiOpp.ownerHoldsAliveLock probes THIS offset -
# the two constants must stay equal.
ALIVE_LOCK_OFFSET = 1 << 20
# how long a step on a file the client holds is tried again before it is left to the next poll
HELD_TRIES, HELD_PAUSE_S = 20, 0.01

SIDE_HEADERS = ('SHADOW', 'FREE_PEOPLES', 'AUTO',
                'LIVE_SHADOW', 'LIVE_FREE_PEOPLES')

# The status keys that belong to a SESSION, not to the file: a status
# file written by a DIFFERENT pid is a dead session's - merging its
# session state would carry a stale 'turn' into ours (BOOT-1).
_SESSION_KEYS = ('turn', 'acting', 'phase', 'note', 'started', 'recovery',
                 'manual_edit', 'sims', 'thinking', 'request_id', 'request_session',
                 'request_started', 'request_overdue')

# R2-SL-24: a handler that has published no progress - its start, or the
# search's per-decision 'AI choosing ...' status - for this long is marked
# request_overdue=true in the status file.  One decision is ~11 s at the
# panel's 1000-sim maximum, so 90 s of silence is far outside a normal
# search.  A Python handler cannot be interrupted safely and the panel's
# control file is read only between handlers, so the heartbeat can only
# MARK the stall: AiOpp's idle Status line and the AI panel then say so and
# name the lever (AI -> Stop AI).  A boolean, never a counter.
THINKING_OVERDUE_S = 90.0

Request = namedtuple('Request', ('side', 'log_lines', 'raw'))


class BridgeLockError(RuntimeError):
    """A live server holds this client dir's alive byte - a second
    writer would contaminate the live game."""


def shared(step):
    """One step on a file both processes use (an open, a replace) -> its result.  Windows
    refuses it with PermissionError while the client - or a scanner - holds the file for a
    moment: it is tried again briefly; the last refusal is raised as it is, and what the
    step was for is done by the next poll (no caller drops anything on it)."""
    for left in reversed(range(HELD_TRIES)):
        try:
            return step()
        except PermissionError:
            if not left:
                raise
            time.sleep(HELD_PAUSE_S)


def read_shared(path):
    """A file of the exchange as text, read whole in ONE open: the shortest hold the server
    can have on it.  None: missing, or still held - the next poll reads it.  (Windows
    refuses the client's atomic replace of its board dump while ANY handle is open on the
    file, also one that shares 'delete', so a rare 'dumpBoard failed' on the client cannot
    be closed from this side; the client keeps its last complete dump and writes the next
    one with its next request.)"""
    def whole():
        with open(path, 'rb') as stream:
            return stream.read()
    try:
        return shared(whole).decode('utf-8', 'replace')
    except OSError:
        return None


def atomic_write(path, text):
    """Write-temp-then-rename: a reader can never observe a partial
    file (lifted :18317-18321, byte-identical discipline: trailing
    newline always present, os.replace for the atomic swap)."""
    tmp = path + '.tmp'

    def whole():
        with open(tmp, 'w', encoding='utf-8') as f:
            f.write(text if text.endswith('\n') else text + '\n')
    shared(whole)
    shared(lambda: os.replace(tmp, path))


def remove_shared(path, must=False):
    """Take a file of the exchange away: a marker or a handshake file of a run before, a request
    that was read.  Held for the moment: tried again briefly (shared).  Still held after that:
    raised when the file MUST go (must), else it stays - the next claim or write replaces it."""
    try:
        shared(lambda: os.remove(path))
    except FileNotFoundError:
        pass
    except OSError:
        if must:
            raise


class Slot(object):
    """A file the client writes for the server to take, one at a time: its request, the
    panel's request.  take() -> the file's text, or None: nothing is there, or the file is
    held for the moment.  The file is renamed away first - the rename is the dedup: a file
    renamed away is handled once, and the slot is free for the client's next one - and
    then read.  A claimed file that cannot be read yet is read by the next take(), before
    any newer file: no request is dropped, none is answered twice.  Only this object's own
    rename is a claim; a claimed file left by another run is never adopted."""

    def __init__(self, path, mark):
        self.path, self.aside, self.claimed = path, path + mark, False

    def take(self):
        if not self.claimed:
            if not os.path.exists(self.path):
                return None
            try:
                shared(lambda: os.replace(self.path, self.aside))
            except OSError:
                return None
            self.claimed = True
        text = read_shared(self.aside)
        self.claimed = text is None and os.path.exists(self.aside)
        return text


def parse_request(text):
    """The request-file grammar (lifted :265-275): line 1 = side
    header, rest = log lines.  Returns (side, lines-after-header) or
    (None, all-lines) when the header is missing/foreign."""
    lines = text.splitlines()
    if not lines:
        return None, []
    side = lines[0].strip()
    if side not in SIDE_HEADERS:
        return None, lines
    return side, lines[1:]


class FileBridge(object):
    """The file-bridge transport for ONE client directory.

    serve(handler) runs the request/response loop; handler(request)
    returns the response lines (a list of str), or (lines, how) where
    `how` is a short status string surfaced to the panel."""

    def __init__(self, client_dir,
                 req_name=REQ_NAME, resp_name=RESP_NAME,
                 status_name=STATUS_NAME, control_name=CONTROL_NAME,
                 poll=0.2,
                 status_extra=None, control_cb=None, paused_reply=None,
                 gate_escape=None):
        self.client_dir = client_dir
        # the two slots the client fills: its request, and the in-client panel's request
        self.request = Slot(os.path.join(client_dir, req_name), '.processing')
        self.panel = Slot(os.path.join(client_dir, PANEL_REQUEST_NAME), '.taken')
        self.resp = os.path.join(client_dir, resp_name)
        self.status_path = os.path.join(client_dir, status_name)
        self.control_path = os.path.join(client_dir, control_name)
        self.stop_path = os.path.join(client_dir, STOP_NAME)
        self.poll = poll
        # hooks (all optional; transport works stand-alone)
        self.status_extra = status_extra    # () -> dict merged in status
        self.control_cb = control_cb        # (control_dict) -> None
        self.paused_reply = paused_reply    # (side, ticks) -> lines
        # R3-PANEL-UNPAUSE-CHAT: (request, control_path) -> lines, called
        # ahead of the pause gate (see _gate_escape)
        self.gate_escape = gate_escape
        # state
        self._alive_fh = None               # the open handle that holds the alive byte
        self.paused = False
        self.control = {}
        self._pause_ticks = 0
        self._stop = False
        self._status_lock = threading.RLock()
        self._active_request = None
        self._heartbeat_interval = 2.0
        # R2-SL-24: when the active handler started, and when it last
        # published progress (any status write carrying fields; the beat
        # thread's bare refresh is not progress).  Guarded by _status_lock.
        self._request_started_at = None
        self._request_progress_at = None
        # Once the handler has run, its response belongs to that exact
        # request.  A transient file lock must not discard it or rerun
        # the handler.  Only one response can be awaiting publication.
        self._pending_response = None

    # ------------------------------------------------------------ lock
    def acquire_lock(self):
        """Become THE server of this client dir: write our pid into
        ALIVE_NAME and keep the handle open with an exclusive OS lock on
        the byte at ALIVE_LOCK_OFFSET for the rest of the process.
        Returns 'own' when this bridge already holds it, else 'fresh';
        raises BridgeLockError when another live process holds the byte
        (continuing would permit two engine writers)."""
        if self._alive_fh is not None:
            return 'own'
        try:
            self._alive_fh = claim(os.path.join(self.client_dir, ALIVE_NAME), ALIVE_LOCK_OFFSET)
        except Exception as _ae:
            raise BridgeLockError('Client directory already owned or unavailable: %s' % _ae) from _ae
        return 'fresh'

    def release_lock(self):
        """Give the alive byte back (the OS does it too when we die)."""
        if self._alive_fh is None:
            return
        release(os.path.join(self.client_dir, ALIVE_NAME), self._alive_fh)
        self._alive_fh = None

    # ----------------------------------------------------------- panel
    def write_status(self, **kw):
        with self._status_lock:
            self._write_status_locked(**kw)

    def _write_status_locked(self, _enrich=True, **kw):
        """Panel status JSON (lifted :18177-18253).  Ownership BEFORE
        the merge: a foreign-pid file's session keys are popped so a
        dead session's 'turn' never leaks into ours; identity keys
        (pid, client_dir, ts) are stamped UNCONDITIONALLY and OUTSIDE
        the guarded enrichment block, so a re-entering client can
        always prove who owns the directory."""
        try:
            try:
                cur = json.loads(read_shared(self.status_path) or '{}')
            except Exception:
                cur = {}
            _pid = os.getpid()
            if cur.get('pid') not in (None, _pid):
                for _k in _SESSION_KEYS:
                    cur.pop(_k, None)
            cur.update(kw)
            try:
                if _enrich and self.status_extra is not None:
                    extra = self.status_extra() or {}
                    for k, v in extra.items():
                        if v is not None:
                            cur[k] = v
            except Exception:
                pass
            # UNCONDITIONAL, and OUTSIDE the block above (BOOT-1): the
            # enrichment hook may be in a bad state exactly when a
            # re-entering client needs to know who owns the directory.
            cur['pid'] = _pid
            try:
                cur['client_dir'] = os.path.abspath(self.client_dir)
            except Exception:
                cur['client_dir'] = str(self.client_dir)
            cur['ts'] = time.time()
            # AiOpp.serverThinking reads this exact top-level boolean, not
            # state='thinking'. Only the currently executing request may own
            # it. Publish before nested diagnostic/menu fields for its simple
            # first-occurrence JSON reader. A heartbeat never recomputes menus.
            cur['thinking'] = (self._active_request is not None
                               and cur.get('state') == 'thinking')
            if self._active_request is not None:
                _, request_id, session = self._active_request
                cur['request_id'] = request_id
                cur['request_session'] = session
                # R2-SL-24: a write carrying fields is the handler's own
                # progress; the beat thread's bare refresh (no fields) is
                # not.  request_started stays fixed for the whole handler.
                if kw:
                    self._request_progress_at = cur['ts']
                cur['request_started'] = self._request_started_at
                cur['request_overdue'] = bool(
                    cur['thinking'] and self._request_progress_at is not None
                    and cur['ts'] - self._request_progress_at >= THINKING_OVERDUE_S)
            else:
                cur.pop('request_started', None)
                cur['request_overdue'] = False
            # IDENTITY KEYS ARE EMITTED FIRST - load-bearing: the
            # client reads this file with AiOpp.jsonField(), a naive
            # scanner that returns the FIRST occurrence of "pid"
            # anywhere in the text, and the human_menu publication
            # nests a "pid" on every DIE PIECE; stamping identity first
            # keeps the owner proof correct (a wrong pid makes the
            # client spawn a second server).  test_status_identity_first
            # pins it (it replays the client's exact scan).
            ident = ('pid', 'client_dir', 'ts', 'thinking', 'request_id', 'request_session',
                     'request_started', 'request_overdue')
            ordered = dict((k, cur[k]) for k in ident if k in cur)
            for k, v in cur.items():
                if k not in ident:
                    ordered[k] = v
            atomic_write(self.status_path, json.dumps(ordered))
        except Exception:
            pass

    @contextmanager
    def _request_heartbeat(self, request):
        """Keep native wait liveness fresh for this handler only.

        No game callbacks run on the heartbeat thread. Status updates share
        one lock/atomic writer and preserve existing diagnostics. Completion,
        exceptions and interrupted handlers retire the token before a later
        request can start; an old heartbeat cannot renew a new request.
        """
        from .client_protocol import split_seq_header
        header, _ = split_seq_header(request.log_lines)
        header = header or {}
        token = (object(), header.get('req'), header.get('session'))
        done = threading.Event()
        with self._status_lock:
            if self._active_request is not None:
                raise RuntimeError('Concurrent handlers cannot own one FileBridge')
            # R2-SL-24: stamped once per handler; the beat thread keeps it
            self._request_started_at = self._request_progress_at = time.time()
            self._active_request = token
            self._write_status_locked(state='thinking', requesting=str(request.side))

        def beat():
            while not done.wait(self._heartbeat_interval):
                with self._status_lock:
                    if self._active_request is not token:
                        return
                    # Preserve state/error/note fields, and do not invoke the
                    # engine's status_extra hook from this background thread.
                    # R2-SL-24: this bare refresh is what marks a handler
                    # with no progress request_overdue (it is not progress).
                    self._write_status_locked(_enrich=False)

        worker = threading.Thread(target=beat, name='FileBridge-request-heartbeat', daemon=True)
        try:
            worker.start()
            yield
        finally:
            done.set()
            with self._status_lock:
                if self._active_request is token:
                    self._active_request = None
                    self._write_status_locked(_enrich=False)
            if worker.ident is not None:
                worker.join()

    def read_control(self):
        """Panel control JSON (lifted :18259-18314).  PAUSE is honored
        HERE (it gates the serve loop); every other key (sims,
        start_game, resync, manual_edit epochs) is game/driver state
        and is surfaced through control_cb untouched."""
        try:
            text = read_shared(self.control_path)
            if text is not None:
                c = json.loads(text)
                self.control = c
                self.paused = bool(c.get('paused', False))
                # CLEAR ERROR (USER 2026-08-20 panel button).  Handled
                # HERE because the transport owns the sticky error
                # state: a failed cycle latches state='error' so the
                # heartbeat can no longer repaint it 'idle', and a
                # latch with no dismiss would follow the session for
                # ever.  Epoch-stamped: one press clears once, and a
                # NEW failure re-latches immediately.
                ce = float(c.get('clear_error', 0) or 0)
                if ce > getattr(self, '_clear_err_epoch', 0):
                    self._clear_err_epoch = ce
                    if getattr(self, '_err_note', ''):
                        self._err_note = ''
                        self.write_status(connected=True, state='idle',
                                          waiting_on='', note='')
                        print('panel: error cleared')
                try:
                    if self.control_cb is not None:
                        self.control_cb(c)
                except Exception as _ce:
                    print('! control callback error: %r' % (_ce,))
        except Exception:
            pass
        return self.control

    def _gate_escape(self, request):
        """R3-PANEL-UNPAUSE-CHAT: the chat backup for the panel's holds.

        PAUSE (honored below) and the driver's MANUAL EDIT hold are written
        only by the AI panel; a closed panel left them on for good.  The
        driver-owned hook runs ahead of the pause gate on every request and
        may rewrite the control FILE (a flag cleared only in memory would be
        re-read every cycle); its answer lines become this cycle's whole
        reply, after the control file is re-read, so the NEXT cycle plays.
        A hook failure is printed and the cycle goes on as before: the
        escape must never block play."""
        if self.gate_escape is None:
            return []
        try:
            lines = list(self.gate_escape(request, self.control_path) or [])
        except Exception as _ge:
            print('! gate_escape error: %r' % (_ge,))
            return []
        if lines:
            self.read_control()
        return lines

    def _panel_poll(self):
        """L1104-U04 (UI_SPEC 2.3): the in-client panel's request slot.  The
        optional `panel_cb` attribute (server.build sets it) takes what the
        slot `self.panel` holds between handlers on this thread and
        returns True when it answered one (it republishes the view itself).
        A callback failure is printed and the loop goes on: the panel must
        never stall the game."""
        cb = getattr(self, 'panel_cb', None)
        if cb is None:
            return False
        try:
            return bool(cb())
        except Exception as _pe:
            print('! panel callback error: %r' % (_pe,))
            return False

    # ----------------------------------------------------------- serve
    def stop(self):
        """Ask a running serve() to return after the current cycle."""
        self._stop = True

    def _client_requested_stop(self):
        """Java 8 cannot use ProcessHandle; it writes the proven owner's PID."""
        return (read_shared(self.stop_path) or '').strip() == str(os.getpid())

    def _flush_response(self):
        text, status = self._pending_response
        atomic_write(self.resp, text)
        self._pending_response = None
        return status

    def _write_response(self, lines, header, status=None):
        """Retain the complete wire reply until its atomic publication.

        The envelope is transport metadata, stripped by AiOpp before
        injection.  Legacy requests without #SEQ retain plain replies.
        Capture identity from the consumed request, never the next REQ
        file or a newer client session that arrived during the handler.
        """
        from .client_protocol import resp_echo_line
        body = list(lines or ())
        if header is not None and 'req' in header:
            echo = resp_echo_line(header['req'])
            if 'session' in header:
                echo += ' session=%d' % header['session']
            body.insert(0, echo)
        self._pending_response = ('\n'.join(body), dict(status or {}))
        return self._flush_response()

    def serve(self, handler, max_cycles=None):
        """Request/response with deletion as the handshake (lifted
        :18324-18552): the client writes REQ; we consume+delete it,
        the handler decides, we write all its lines to RESP; the
        client reads+deletes RESP and injects.  Deletion means each
        manual click gets a fresh response even when the log is
        unchanged (retry after a rejected move)."""
        self.acquire_lock()
        # Retire a previous owner's marker before advertising this new
        # server. A reused OS PID must not make an old Stop kill a fresh run.
        remove_shared(self.stop_path, must=True)
        # clear any stale handshake files from a previous run (skip
        # under a bounded self-test, which pre-writes its own request)
        if max_cycles is None:
            for p in (self.request.path, self.resp):
                remove_shared(p)
        self._err_note = ''
        print('watching %s  (poll %.2fs)' % (self.request.path, self.poll))
        print('replies -> %s' % self.resp)
        self.write_status(connected=True, state='idle')

        cycles = 0
        hb = 0.0
        while (max_cycles is None or cycles < max_cycles) \
                and not self._stop:
            if self._client_requested_stop():
                self._stop = True
                self.write_status(_enrich=False, connected=False,
                                  state='stopped', waiting_on='', note='')
                self.release_lock()
                return
            cycles += 1
            response_header = None
            response_published = False
            request_consumed = False
            try:
                if self._pending_response is not None:
                    # Do not consume another request (or apply its
                    # controls) ahead of a completed, unpublished reply.
                    status = self._flush_response()
                    response_published = True
                    if status.get('state') != 'error':
                        self._err_note = ''
                    self.write_status(**status)
                    continue
                # the request is taken out of its slot whole (Slot: renamed away first,
                # then read).  The rename IS the dedup: a request renamed away can never
                # be processed twice; one that is held for the moment stays the client's
                # outstanding request and is taken by the next cycle.
                text = self.request.take()
                if text is None:
                    # heartbeat: refresh the panel's status ts + apply
                    # the control file even while no requests arrive
                    now = time.time()
                    if now - hb > 2.0:
                        hb = now
                        self.read_control()
                        # a STICKY error: the heartbeat must never
                        # repaint state='idle' over an error a failed
                        # cycle wrote (the panel would show a calm idle
                        # server that is in fact halted)
                        if self._err_note:
                            self.write_status(
                                connected=True, state='error',
                                waiting_on=_status_fit(self._err_note, _STATUS_ERROR_LIMIT))
                        else:
                            self.write_status(connected=True,
                                              state='idle')
                    self._panel_poll()      # L1104-U04: in-client panel requests (UI_SPEC 2.3)
                    time.sleep(self.poll)
                    continue
                # (the read retired the claim: a failed deletion cannot replay a
                # request whose handler may already have committed)
                remove_shared(self.request.aside)
                if not text.strip():
                    time.sleep(self.poll)
                    continue
                from .client_protocol import split_seq_header
                # Wire identity has a fixed position after the side
                # line, including a request rejected for an invalid side.
                response_header, _ = split_seq_header(text.splitlines()[1:])
                side, log_lines = parse_request(text)
                request_consumed = True
                if side is None:
                    # the handshake still completes: an EMPTY reply is
                    # a no-op to AiOpp, while a missing reply makes
                    # requestMove wait its full 30 s
                    print('! request without side header - ignored')
                    self._write_response([], response_header)
                    response_published = True
                    continue
                self.read_control()
                self._panel_poll()          # L1104-U04: once right before each handler
                # R3-PANEL-UNPAUSE-CHAT: the chat escape is heard AHEAD of
                # the pause gate (the handler never runs while paused); an
                # answered escape is this cycle's whole reply.
                escape = self._gate_escape(Request(side, log_lines, text))
                if escape:
                    if self.paused:
                        reply_status = dict(state='paused', note='PAUSED by user')
                    elif self._err_note:
                        # a sticky error is never repainted 'idle'
                        reply_status = dict(state='error',
                                            waiting_on=_status_fit(self._err_note, _STATUS_ERROR_LIMIT))
                    else:
                        reply_status = dict(state='idle')
                    self._write_response(escape, response_header, reply_status)
                    response_published = True
                    self.write_status(**reply_status)
                    print('[%s] chat escape answered%s | %d line(s)'
                          % (side, ' - still PAUSED' if self.paused else '',
                             len(escape)))
                    for ln in escape:
                        print('    ' + ln[:118])
                    continue
                if self.paused:
                    # PAUSED: no decisions - decision-boundary idle.
                    # The reply hook (driver-owned) may emit a
                    # keepalive; transport itself replies empty so the
                    # handshake stays alive and the brain is untouched.
                    self._pause_ticks += 1
                    keep = []
                    try:
                        if self.paused_reply is not None:
                            keep = list(self.paused_reply(
                                side, self._pause_ticks) or [])
                    except Exception as _pe:
                        print('! paused_reply error: %r' % (_pe,))
                    reply_status = dict(state='paused', note='PAUSED by user')
                    self._write_response(keep, response_header, reply_status)
                    response_published = True
                    self.write_status(**reply_status)
                    print('[%s] PAUSED (tick %d)'
                          % (side, self._pause_ticks))
                    continue
                self._pause_ticks = 0
                t0 = time.time()
                with self._request_heartbeat(Request(side, log_lines, text)):
                    out = handler(Request(side, log_lines, text))
                self._err_note = ''      # a good cycle clears it
                if isinstance(out, tuple):
                    lines, how = out
                else:
                    lines, how = out, ''
                lines = list(lines or [])
                # waiting_on: the state machine's own words, ALWAYS
                # surfaced (USER 2026-08-16: "it should say what it's
                # waiting on")
                reply_status = dict(
                    state='idle', waiting_on=_status_fit(how),
                    note=(_status_fit(how)
                          if 'waiting for your' in str(how) else ''))
                self._write_response(lines, response_header, reply_status)
                response_published = True
                self.write_status(**reply_status)
                print('[%s] %4d log lines -> %s (%.2fs) | %d line(s)'
                      % (side, len(log_lines), how or '-',
                         time.time() - t0, len(lines)))
                for ln in lines:
                    print('    ' + ln[:118])
            except KeyboardInterrupt:
                print('\nstopped')
                return
            except Exception as e:    # keep serving, report loudly
                # LOUD TO THE HUMAN, not just to the console (manual
                # play 2026-08-20): the pre-handler write_status left
                # the panel reading 'thinking', so a handler halt was
                # an invisible freeze on the player's screen - exactly
                # the 'AI got stuck and the panel never said what it
                # was waiting on' class.  The panel is the only channel
                # a diagnostic may use (chat is whitelist-by-
                # construction and must never carry server errors).
                if self._err_note != str(e):      # (an OSError's text names its file)
                    print('! server cycle error: %s: %s' % (type(e).__name__, e))
                self._err_note = str(e)
                try:
                    # A handler error completes this identified request
                    # with a no-op.  A publication error retains the
                    # handler's output; a later panel/status error must
                    # never overwrite an already published response.
                    if (request_consumed and self._pending_response is None
                            and not response_published):
                        self._write_response([], response_header, dict(
                            state='error', waiting_on=_status_fit(e, _STATUS_ERROR_LIMIT),
                            note=str(e)[:400]))
                except Exception:
                    pass
                try:
                    self.write_status(state='error',
                                      waiting_on=_status_fit(e, _STATUS_ERROR_LIMIT),
                                      note=str(e)[:400])
                except Exception:
                    pass
                time.sleep(self.poll)
