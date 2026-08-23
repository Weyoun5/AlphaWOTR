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
  * pid-owner lock          :18340-18378  liveness is NEVER a bare pid
        (client fact 18: Windows reuses pids - the Discord.exe
        adoption, twice); ownership proven by COMMAND LINE via
        wotr_bridge/pid_owner.owner_check, THE one answerer -
        conservative pid-alive fallback only when the probe itself
        fails.

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
    SystemExit so a driver embedding the bridge can report it; the
    refusal condition and the conservative fallback are byte-for-byte
    the old rule.
  * release_lock() is NEW (tests need a clean teardown); production
    code never needs it - stale locks are handled by the liveness
    proof, exactly as before.
"""
import json
import os
import sys
import time
from collections import namedtuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_BRIDGE = os.path.abspath(os.path.join(_HERE, '..'))
if _BRIDGE not in sys.path:
    sys.path.insert(0, _BRIDGE)

# THE one answerer for 'does this pid belong to us?' (its own module
# doctrine: do not add a second implementation)
from pid_owner import owner_check, pid_alive_fallback

# Default file names - the AiOpp contract.  All parameterized on
# FileBridge; nothing below hardcodes a client directory.
REQ_NAME = 'ai_bridge_request.log'
RESP_NAME = 'ai_bridge_move.txt'
STATUS_NAME = 'ai_bridge_status.json'
CONTROL_NAME = 'ai_panel_control.json'
LOCK_NAME = 'ai_server.lock'

SIDE_HEADERS = ('SHADOW', 'FREE_PEOPLES', 'AUTO',
                'LIVE_SHADOW', 'LIVE_FREE_PEOPLES')

# The status keys that belong to a SESSION, not to the file: a status
# file written by a DIFFERENT pid is a dead session's - merging its
# session state would carry a stale 'turn' into ours (BOOT-1).
_SESSION_KEYS = ('turn', 'acting', 'phase', 'note', 'started',
                 'manual_edit', 'sims')

Request = namedtuple('Request', ('side', 'log_lines', 'raw'))


class BridgeLockError(RuntimeError):
    """A live server with a PROVEN matching command line owns this
    client dir - a second writer would contaminate the live game."""


def atomic_write(path, text):
    """Write-temp-then-rename: a reader can never observe a partial
    file (lifted :18317-18321, byte-identical discipline: trailing
    newline always present, os.replace for the atomic swap)."""
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(text if text.endswith('\n') else text + '\n')
    os.replace(tmp, path)


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
    `how` is a short status string surfaced to the panel.

    owner_needle: the substring that PROVES a pid on the lock file is a
    live bridge server (matched against the process command line, never
    against bare pid existence - client fact 18)."""

    def __init__(self, client_dir, owner_needle,
                 req_name=REQ_NAME, resp_name=RESP_NAME,
                 status_name=STATUS_NAME, control_name=CONTROL_NAME,
                 lock_name=LOCK_NAME, poll=0.2,
                 status_extra=None, control_cb=None, paused_reply=None):
        self.client_dir = client_dir
        self.owner_needle = owner_needle
        self.req = os.path.join(client_dir, req_name)
        self.resp = os.path.join(client_dir, resp_name)
        self.status_path = os.path.join(client_dir, status_name)
        self.control_path = os.path.join(client_dir, control_name)
        self.lock_path = os.path.join(client_dir, lock_name)
        self.poll = poll
        # hooks (all optional; transport works stand-alone)
        self.status_extra = status_extra    # () -> dict merged in status
        self.control_cb = control_cb        # (control_dict) -> None
        self.paused_reply = paused_reply    # (side, ticks) -> lines
        # state
        self.paused = False
        self.control = {}
        self._pause_ticks = 0
        self._stop = False

    # ------------------------------------------------------------ lock
    def acquire_lock(self):
        """Single-writer lock (lifted :18340-18378, semantics kept
        exactly).  Returns 'own' / 'fresh' / 'takeover'; raises
        BridgeLockError when a foreign pid PROVEN (or conservatively
        assumed, on probe failure) to be a live server owns the dir."""
        how = 'fresh'
        try:
            if os.path.exists(self.lock_path):
                _op = int(open(self.lock_path).read().strip() or 0)
                if _op and _op == os.getpid():
                    how = 'own'
                elif _op:
                    _probe_ok, _owner_is_server = owner_check(
                        _op, self.owner_needle)
                    if not _probe_ok:
                        # probe failed: keep the OLD conservative
                        # refusal (pid alive -> assume live server) -
                        # never weaker than before on the error path
                        _owner_is_server = pid_alive_fallback(_op)
                        print('! lock probe failed - conservative '
                              'pid-alive fallback says %s'
                              % _owner_is_server)
                    if _owner_is_server:
                        raise BridgeLockError(
                            'REFUSED: client dir %s is owned by live '
                            'server pid %d - a second writer would '
                            'contaminate the live game'
                            % (self.client_dir, _op))
                    print('! stale %s: pid %d is not a live server '
                          '(dead or reused) - taking over'
                          % (os.path.basename(self.lock_path), _op))
                    how = 'takeover'
            with open(self.lock_path, 'w') as _lf:
                _lf.write(str(os.getpid()))
        except BridgeLockError:
            raise
        except Exception as _le:
            print('! server lock check skipped: %r' % (_le,))
        return how

    def release_lock(self):
        """Remove the lock ONLY if it names our pid (test teardown
        helper - production relies on the liveness proof, as always)."""
        try:
            if os.path.exists(self.lock_path):
                if open(self.lock_path).read().strip() == \
                        str(os.getpid()):
                    os.remove(self.lock_path)
        except Exception:
            pass

    # ----------------------------------------------------------- panel
    def write_status(self, **kw):
        """Panel status JSON (lifted :18177-18253).  Ownership BEFORE
        the merge: a foreign-pid file's session keys are popped so a
        dead session's 'turn' never leaks into ours; identity keys
        (pid, client_dir, ts) are stamped UNCONDITIONALLY and OUTSIDE
        the guarded enrichment block, so a re-entering client can
        always prove who owns the directory."""
        try:
            cur = {}
            if os.path.exists(self.status_path):
                try:
                    cur = json.load(open(self.status_path,
                                         encoding='utf-8'))
                except Exception:
                    cur = {}
            _pid = os.getpid()
            if cur.get('pid') not in (None, _pid):
                for _k in _SESSION_KEYS:
                    cur.pop(_k, None)
            cur.update(kw)
            try:
                if self.status_extra is not None:
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
            # IDENTITY KEYS ARE EMITTED FIRST - and this is load-
            # bearing, not cosmetic.  The client reads this file with
            # AiOpp.jsonField(), a naive scanner that returns the
            # FIRST occurrence of "pid" anywhere in the text.  The
            # Phase-3 human_menu publication nests a "pid" on every
            # DIE PIECE, so the client was reading a die id ("210") as
            # the server pid, failing the owner proof, and spawning a
            # SECOND server - the RETIRED ai_move_server.py monolith -
            # alongside this one: two brains writing one
            # ai_bridge_move.txt for a live human game.  Reproduced
            # 2026-08-20 on the manual-play launcher (ai_boot.log:
            # 'NOT adopting pid 210 ... spawning our own').  Until the
            # jar's parser is fixed, the fix is ours: stamp identity
            # first so the first "pid" in the file IS the server's.
            # test_status_identity_first pins it (it replays the
            # client's exact scan).
            ident = ('pid', 'client_dir', 'ts')
            ordered = dict((k, cur[k]) for k in ident if k in cur)
            for k, v in cur.items():
                if k not in ident:
                    ordered[k] = v
            atomic_write(self.status_path, json.dumps(ordered))
        except Exception:
            pass

    def read_control(self):
        """Panel control JSON (lifted :18259-18314).  PAUSE is honored
        HERE (it gates the serve loop); every other key (sims,
        start_game, resync, manual_edit epochs) is game/driver state
        and is surfaced through control_cb untouched."""
        try:
            if os.path.exists(self.control_path):
                c = json.load(open(self.control_path,
                                   encoding='utf-8'))
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

    # ----------------------------------------------------------- serve
    def stop(self):
        """Ask a running serve() to return after the current cycle."""
        self._stop = True

    def serve(self, handler, max_cycles=None):
        """Request/response with deletion as the handshake (lifted
        :18324-18552): the client writes REQ; we consume+delete it,
        the handler decides, we write all its lines to RESP; the
        client reads+deletes RESP and injects.  Deletion means each
        manual click gets a fresh response even when the log is
        unchanged (retry after a rejected move)."""
        self.acquire_lock()
        # clear any stale handshake files from a previous run (skip
        # under a bounded self-test, which pre-writes its own request)
        if max_cycles is None:
            for p in (self.req, self.resp):
                try:
                    os.remove(p)
                except OSError:
                    pass
        self._err_note = ''
        print('watching %s  (poll %.2fs)' % (self.req, self.poll))
        print('replies -> %s' % self.resp)
        self.write_status(connected=True, state='idle')

        cycles = 0
        hb = 0.0
        while (max_cycles is None or cycles < max_cycles) \
                and not self._stop:
            cycles += 1
            try:
                if not os.path.exists(self.req):
                    # heartbeat: refresh the panel's status ts + apply
                    # the control file even while no requests arrive
                    now = time.time()
                    if now - hb > 2.0:
                        hb = now
                        self.read_control()
                        # a STICKY error: the 2s heartbeat used to
                        # repaint state='idle' over the error a failed
                        # cycle had just written, so the panel showed a
                        # calm idle server that was in fact halted.
                        if self._err_note:
                            self.write_status(
                                connected=True, state='error',
                                waiting_on=self._err_note[:140])
                        else:
                            self.write_status(connected=True,
                                              state='idle')
                    time.sleep(self.poll)
                    continue
                # consume ATOMICALLY (rename first, then read).  The
                # rename IS the dedup: a request that cannot be renamed
                # away is left for the next cycle; a request renamed
                # away can never be processed twice.
                proc = self.req + '.processing'
                try:
                    os.replace(self.req, proc)
                except OSError:
                    time.sleep(self.poll)
                    continue
                with open(proc, 'r', encoding='utf-8',
                          errors='replace') as f:
                    text = f.read()
                try:
                    os.remove(proc)
                except OSError:
                    pass
                if not text.strip():
                    time.sleep(self.poll)
                    continue
                side, log_lines = parse_request(text)
                if side is None:
                    # the handshake still completes: an EMPTY reply is
                    # a no-op to AiOpp, while a missing reply makes
                    # requestMove wait its full 30 s
                    print('! request without side header - ignored')
                    atomic_write(self.resp, '')
                    continue
                self.read_control()
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
                    atomic_write(self.resp,
                                 '\n'.join(keep) if keep else '')
                    self.write_status(state='paused',
                                      note='PAUSED by user')
                    print('[%s] PAUSED (tick %d)'
                          % (side, self._pause_ticks))
                    continue
                self._pause_ticks = 0
                self.write_status(state='thinking',
                                  requesting=str(side))
                t0 = time.time()
                out = handler(Request(side, log_lines, text))
                self._err_note = ''      # a good cycle clears it
                if isinstance(out, tuple):
                    lines, how = out
                else:
                    lines, how = out, ''
                lines = list(lines or [])
                atomic_write(self.resp,
                             '\n'.join(lines) if lines else '')
                # waiting_on: the state machine's own words, ALWAYS
                # surfaced (USER 2026-08-16: "it should say what it's
                # waiting on")
                self.write_status(
                    state='idle', waiting_on=str(how)[:140],
                    note=(str(how)[:140]
                          if 'waiting for your' in str(how) else ''))
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
                print('! server cycle error: %r' % (e,))
                self._err_note = str(e)
                try:
                    # complete the handshake (empty reply = no-op to
                    # AiOpp; no reply = a 30 s wait per failed cycle
                    # before the client re-requests)
                    atomic_write(self.resp, '')
                except Exception:
                    pass
                try:
                    self.write_status(state='error',
                                      waiting_on=str(e)[:140],
                                      note=str(e)[:400])
                except Exception:
                    pass
                time.sleep(self.poll)
