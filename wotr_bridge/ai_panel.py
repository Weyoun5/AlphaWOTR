# -*- coding: utf-8 -*-
"""Live-play control panel (USER 2026-07-29; PAUSE/SIMS wave 2026-08-14).

Small always-on-top window beside the VASSAL client:
  * 'AI connected' popup (once) + persistent connection light
  * big AI THINKING banner while the server computes a move
  * PAUSED banner + PAUSE / RESUME buttons (USER 2026-08-14: no client
    recompile needed — the server honors a 'paused' flag at decision
    boundaries and idles with a chat keepalive)
  * whose turn it is per the RULES ENGINE (turn number + acting side)
  * MCTS simulations slider 30..1000 — written to the control file,
    applied by the live server from its NEXT decision (the server log
    stamps 'sims=N' per decision so the effect is provable)

Talks to the live server (live/live_server.py via live/transport.py)
through two small json files in the client dir:
  ai_bridge_status.json   (server -> panel)
  ai_panel_control.json   (panel -> server)

Single-instance: writes ai_panel.pid; a second launch exits quietly if
that pid is a live panel (the java client spawns the panel via
-Dai.python).

Usage: python ai_panel.py [client_dir]
"""
import json
import os
import sys
import time
import tkinter as tk
from tkinter import messagebox

CLIENT_DIR = (sys.argv[1] if len(sys.argv) > 1
              else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'play_client'))
STATUS = os.path.join(CLIENT_DIR, 'ai_bridge_status.json')
CONTROL = os.path.join(CLIENT_DIR, 'ai_panel_control.json')
PIDFILE = os.path.join(CLIENT_DIR, 'ai_panel.pid')
PROCLOG = os.path.join(CLIENT_DIR, 'ai_panel_proc.log')


def _plog(msg):
    """THE PANEL MAY NEVER EXIT SILENTLY (USER live report 2026-08-16:
    'the ai panel isn't there' - ai_panel_proc.log was EMPTY, so a
    dead panel and a never-started panel were indistinguishable).
    Every startup, verdict, refusal and crash goes through here:
    appended to ai_panel_proc.log AND printed (the java spawner
    redirects stdout there too; the server spawner now does the
    same)."""
    line = '[%s pid=%d] %s' % (time.strftime('%Y-%m-%d %H:%M:%S'),
                               os.getpid(), msg)
    try:
        with open(PROCLOG, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except OSError:
        pass
    # BOTH spawners (java launchPanel, server serve()) redirect this
    # process's stdout INTO ai_panel_proc.log, so an unconditional
    # print wrote every line twice and made "did the panel start
    # twice?" — a session-state question — unanswerable from the log.
    # Print only when stdout is a real console (manual runs).
    try:
        if sys.stdout is not None and sys.stdout.isatty():
            print(line, flush=True)
    except Exception:
        pass


_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
try:
    from pid_owner import owner_check, pid_alive_fallback
except ImportError:                      # pragma: no cover
    owner_check = None
    pid_alive_fallback = None


def single_instance_or_exit():
    """PROBE -> LOG THE VERDICT -> ACT.  Every branch writes its
    reason to the proc log BEFORE acting; only a PROVEN live panel
    refuses (pid_owner.py, the one answerer - a reused ai_panel.pid
    must not silently kill the panel the way the reused
    ai_server.lock killed the server, P1 2026-08-16).  The 2026-08-16
    live incident: all three refusal branches were bare sys.exit(0)
    and the proc log stayed empty - a silent exit in the required
    control surface."""
    try:
        if os.path.exists(PIDFILE):
            old = open(PIDFILE).read().strip()
            if old.isdigit():
                refuse = False
                if owner_check is None:
                    # pid_owner (the one answerer) is missing: refuse
                    # conservatively rather than guess liveness here
                    msg = ('REFUSED (conservative): pid_owner '
                           'unavailable, cannot prove pid %s is not '
                           'a panel - delete ai_panel.pid to force a '
                           'start' % old)
                    refuse = True
                else:
                    ok, is_panel = owner_check(old, 'ai_panel')
                    if ok and is_panel:
                        msg = ('REFUSED: a live panel already owns '
                               'pid %s (proven from its command '
                               'line) - this launch exits' % old)
                        refuse = True
                    elif ok:
                        msg = ('stale ai_panel.pid %s: %s - taking '
                               'over'
                               % (old, 'pid recycled by a non-panel '
                                  'process' if pid_alive_fallback(old)
                                  else 'pid is dead'))
                    elif pid_alive_fallback(old):
                        msg = ('REFUSED (conservative): ownership '
                               'probe FAILED and pid %s is alive - '
                               'cannot prove it is not a panel; '
                               'delete ai_panel.pid to force a '
                               'start' % old)
                        refuse = True
                    else:
                        msg = ('stale ai_panel.pid %s: probe failed '
                               'but the pid is dead - taking over'
                               % old)
                _plog(msg)
                if refuse:
                    sys.exit(0)
        with open(PIDFILE, 'w') as f:
            f.write(str(os.getpid()))
    except OSError as e:
        _plog('pidfile handling failed (%r) - continuing WITHOUT the '
              'single-instance guard' % (e,))


def _write_control(**kw):
    """Merge-write the control file (never clobber the other keys)."""
    cur = {}
    try:
        if os.path.exists(CONTROL):
            with open(CONTROL, encoding='utf-8') as stream:
                cur = json.load(stream)
    except (ValueError, OSError):
        cur = {}
    cur.update(kw)
    try:
        with open(CONTROL, 'w', encoding='utf-8') as f:
            json.dump(cur, f)
        return True
    except OSError:
        return False


def clear_control_keys(*keys):
    """Remove one-shot lever keys from the control file (merge-write of
    the rest).  A one-shot epoch left in the file would replay on the
    next server start (its watermarks begin at 0)."""
    cur = {}
    try:
        if os.path.exists(CONTROL):
            with open(CONTROL, encoding='utf-8') as stream:
                cur = json.load(stream)
    except (ValueError, OSError):
        cur = {}
    hit = False
    for k in keys:
        if k in cur:
            del cur[k]
            hit = True
    if not hit:
        return
    try:
        with open(CONTROL, 'w', encoding='utf-8') as f:
            json.dump(cur, f)
    except OSError:
        pass


# one-shot levers: written with an epoch, consumed once by the server
# (its control poll runs every ~2s), expired from the file after this
ONE_SHOT_KEYS = ('whose_turn_epoch', 'clear_error', 'skip_pending',
                 'reset_resync', 'resync')
ONE_SHOT_TTL_MS = 15000


def read_control():
    """Current control-file dict ({} when absent/unreadable)."""
    try:
        if os.path.exists(CONTROL):
            with open(CONTROL, encoding='utf-8') as f:
                return json.load(f)
    except (ValueError, OSError):
        pass
    return {}


def read_manual_edit():
    """Is manual-edit mode currently ON per the control file?  Read at
    panel boot so a panel restart mid-edit shows the true mode (the
    bridge keeps honoring the flag either way)."""
    return bool(read_control().get('manual_edit'))


def write_manual_edit(on):
    """MANUAL EDIT (USER ruling 2026-08-16, verbatim: 'the human gets
    a button to allow him to do manual edits and then force resync').
    ON: gating suspends bridge-side - the human may move anything, no
    refusals, no ingestion.  OFF (finish): the same write requests the
    FORCED RESYNC (the existing resync control key) so the engine
    adopts the edited board and gated play resumes."""
    if on:
        _write_control(manual_edit=True, manual_edit_ts=time.time())
    else:
        _write_control(manual_edit=False, resync=time.time())


def turn_text(st):
    """'Turn N - X to act' from the status, or None when it carries no
    turn.  The live server nests turn / to_act / alternation under
    human_menu; top-level turn / acting is accepted as well."""
    hm = st.get('human_menu') if isinstance(st.get('human_menu'),
                                            dict) else {}
    turn = st.get('turn', hm.get('turn'))
    if turn is None:
        return None
    acting = (st.get('acting') or hm.get('to_act')
              or hm.get('alternation') or '?')
    return 'Turn %s - %s to act' % (turn, acting)


def note_text(st, last_status_note, local_note, recovery_requested_at=0):
    """The note line: a status note that CHANGED since the last poll
    wins; otherwise a note the panel set itself (button feedback)
    stays; the server's panel notes (status 'notes' tail) are shown
    when nothing else claims the line."""
    cur = str(st.get('note') or '')
    if cur and cur != (last_status_note or ''):
        return cur
    recovery = st.get('recovery') or {}
    if (recovery.get('epoch', 0) >= recovery_requested_at and recovery.get('summary')
            and (recovery_requested_at or not local_note)):
        return str(recovery['summary'])
    if local_note:
        return local_note
    if cur:
        return cur
    notes = st.get('notes')
    if isinstance(notes, (list, tuple)) and notes:
        return ' | '.join(str(n) for n in notes[-3:])
    return ''


# every live server run opens with these console lines (transport
# serve() / engine_play load_nets); only lines after the LAST banner
# belong to the current run
_RUN_BANNERS = ('watching ', 'nets loaded', 'RNG seed')


def server_death_line(client_dir):
    """The last error line of the CURRENT server run, or None.  Reads
    the launcher's redirect (ai_server_console.log) first, then the
    older ai_server.log name."""
    for name in ('ai_server_console.log', 'ai_server.log'):
        p = os.path.join(client_dir, name)
        if not os.path.exists(p):
            continue
        try:
            with open(p, 'rb') as f:
                f.seek(0, os.SEEK_END)
                sz = f.tell()
                f.seek(max(0, sz - 6000))
                tail = f.read().decode('utf-8', 'replace')
        except OSError:
            return None
        cut = max(tail.rfind(b) for b in _RUN_BANNERS)
        if cut >= 0:
            tail = tail[cut:]
        for ln in reversed([x.strip() for x in
                            tail.splitlines() if x.strip()]):
            if ('REFUSED' in ln or 'Error' in ln or 'HALT' in ln
                    or 'Traceback' in ln or 'server cycle error' in ln):
                return ln[:220]
        return None
    return None


MIRROR_LOG = os.path.join(CLIENT_DIR, 'ai_mirror_proc.log')


class Panel:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('WOTR AI')
        # position only: the HEIGHT is fitted to the content once the
        # widgets exist (USER 2026-08-22)
        self.root.geometry('+40+40')
        self.root.attributes('-topmost', True)
        self.root.resizable(False, True)

        self.conn = tk.Label(self.root, text='waiting for AI server...',
                             fg='#888', font=('Segoe UI', 10))
        self.conn.pack(pady=(8, 2))

        self.turn = tk.Label(self.root, text='-', font=('Segoe UI', 11))
        self.turn.pack()

        # in-battle prompts ('waiting for your <x> decision') — the
        # options themselves appear in the CLIENT chat
        self.note = tk.Label(self.root, text='', fg='#c60',
                             font=('Segoe UI', 9), wraplength=300,
                             justify=tk.CENTER)
        self.note.pack()

        # ALWAYS-ON expectation line (USER 2026-08-16: "it should say
        # what it's waiting on") - the server's own last state-machine
        # words, verbatim.  When the AI is stuck, THIS is the bug
        # report: read it aloud.
        self.waiting = tk.Label(self.root, text='waiting on: -',
                                fg='#579', font=('Segoe UI', 9),
                                wraplength=300, justify=tk.CENTER)
        self.waiting.pack()

        self.think = tk.Label(self.root, text='', fg='white',
                              bg=self.root['bg'],
                              font=('Segoe UI', 14, 'bold'))
        self.think.pack(pady=6, fill=tk.X)

        # ---- WHOSE TURN (USER 2026-08-20): manual alternation
        # override.  The engine owns the turn rule; these only correct
        # the driver's 'who last acted' bookkeeping when the two have
        # drifted, so a game never has to be abandoned over a stuck
        # alternation.  Each press stamps an epoch the server acts on
        # once.
        trow = tk.Frame(self.root)
        trow.pack(fill=tk.X, padx=10, pady=(0, 2))
        self.myturn_btn = tk.Button(trow, text='MY TURN',
                                    font=('Segoe UI', 9, 'bold'),
                                    command=self.my_turn)
        self.myturn_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.yourturn_btn = tk.Button(trow, text='YOUR TURN (AI)',
                                      font=('Segoe UI', 9, 'bold'),
                                      command=self.your_turn)
        self.yourturn_btn.pack(side=tk.LEFT, fill=tk.X, expand=True,
                               padx=(4, 0))

        # ---- CLEAR ERROR (USER 2026-08-20): the server's error state
        # is STICKY on purpose; sticky needs a dismiss
        crow = tk.Frame(self.root)
        crow.pack(fill=tk.X, padx=10, pady=(0, 2))
        self.clearerr_btn = tk.Button(crow, text='CLEAR ERROR + RESET',
                                      font=('Segoe UI', 9),
                                      command=self.full_reset)
        self.clearerr_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)
        # ---- SKIP DECISION (USER 2026-08-20): answer an owed decision
        # with its DECLINE option so a stuck pending cannot end the
        # game.  The turn buttons cannot do this - they only move whose
        # plain action is next - and a decision that has no decline is
        # refused with a reason rather than guessed at.
        self.skip_btn = tk.Button(crow, text='SKIP DECISION',
                                  font=('Segoe UI', 9),
                                  command=self.skip_decision)
        self.skip_btn.pack(side=tk.LEFT, fill=tk.X, expand=True,
                           padx=(4, 0))

        # Recover the visible board while preserving unfinished decisions
        # and the side that owns the next action.
        rr = tk.Frame(self.root)
        rr.pack(fill=tk.X, padx=10, pady=(0, 4))
        self.reset_btn = tk.Button(rr, text='RESYNC BOARD  (retain turn '
                                             'and unfinished decisions)',
                                   font=('Segoe UI', 10, 'bold'),
                                   fg='white', bg='#a4560a',
                                   wraplength=300,
                                   command=self.full_reset)
        self.reset_btn.pack(fill=tk.X)

        # The compact resync button requests the same board recovery.
        rrow = tk.Frame(self.root)
        rrow.pack(fill=tk.X, padx=10, pady=(0, 2))
        self.resync_btn = tk.Button(rrow, text='RESYNC (reset AI)',
                                    font=('Segoe UI', 9),
                                    command=self.full_reset)
        self.resync_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.mirror_btn = tk.Button(rrow, text='ENGINE MIRROR',
                                    font=('Segoe UI', 9),
                                    command=self.open_mirror)
        self.mirror_btn.pack(side=tk.LEFT, fill=tk.X, expand=True,
                             padx=(4, 0))

        rewind_row = tk.Frame(self.root)
        rewind_row.pack(fill=tk.X, padx=10, pady=(0, 4))
        self.rewind_turn_btn = tk.Button(rewind_row, text='Rewind turn',
            font=('Segoe UI', 9, 'bold'), command=self.rewind_turn, state=tk.DISABLED)
        self.rewind_turn_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.rewind_manual_btn = tk.Button(rewind_row, text='Rewind (manual)',
            font=('Segoe UI', 9, 'bold'), command=self.open_timeline, state=tk.DISABLED)
        self.rewind_manual_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        self._timeline_window = None
        self._timeline_poll_at = 0
        self._last_rewind_epoch = 0

        # ---- MANUAL EDIT (USER ruling 2026-08-16): the one sanctioned
        # bypass of the legality gate.  Toggle ON = gating suspended,
        # edit the board freely; toggle OFF = forced resync (engine
        # adopts the edited board), gated play resumes.  Boot state
        # restored from the control file so a panel restart mid-edit
        # still says so.
        mrow = tk.Frame(self.root)
        mrow.pack(fill=tk.X, padx=10, pady=(0, 2))
        self.manual_edit = read_manual_edit()
        self.medit_btn = tk.Button(mrow,
                                   font=('Segoe UI', 10, 'bold'),
                                   command=self.toggle_manual_edit)
        self.medit_btn.pack(fill=tk.X)
        self._style_manual_btn()

        # ---- ENFORCE LEGAL MOVES FOR THE HUMAN (beta) - USER
        # 2026-08-22.  Unchecked (default): the AI never countermands
        # the human's moves and takes the board as given after each
        # action.  Checked: the legality gate (refusals with a
        # client-native undo + notice).  Persisted in the control file;
        # the server reads it live.
        erow = tk.Frame(self.root)
        erow.pack(fill=tk.X, padx=10, pady=(2, 2))
        self.enforce_var = tk.BooleanVar(
            value=bool(read_control().get('enforce_human', False)))
        # the panel OWNS this default: written at boot so the server
        # (absent key = ON) follows the checkbox from the first poll
        _write_control(enforce_human=bool(self.enforce_var.get()))
        self.enforce_chk = tk.Checkbutton(
            erow, text='Enforce legal moves for HUMAN player (beta)',
            variable=self.enforce_var, font=('Segoe UI', 9),
            command=self.toggle_enforce, anchor='w')
        self.enforce_chk.pack(fill=tk.X)

        # ---- pause / resume (USER 2026-08-14) ----
        prow = tk.Frame(self.root)
        prow.pack(fill=tk.X, padx=10, pady=(0, 4))
        self.paused = False
        self.pause_btn = tk.Button(prow, text='PAUSE',
                                   font=('Segoe UI', 10, 'bold'),
                                   command=self.toggle_pause)
        self.pause_btn.pack(fill=tk.X)

        row = tk.Frame(self.root)
        row.pack(fill=tk.X, padx=10)
        tk.Label(row, text='MCTS sims').pack(side=tk.LEFT)
        # slider value (left) is what YOU set; server echo (right) is
        # what the live brain is actually using — they converge within
        # a heartbeat (~2s) once the server applies the control file
        self.sims_lbl = tk.Label(row, text='300')
        self.sims_lbl.pack(side=tk.RIGHT)
        self.srv_lbl = tk.Label(row, text='(server: ?)', fg='#888',
                                font=('Segoe UI', 8))
        self.srv_lbl.pack(side=tk.RIGHT, padx=(0, 6))
        # from_=10: the server honors any value >= 1
        self.slider = tk.Scale(self.root, from_=10, to=1000,
                               orient=tk.HORIZONTAL, resolution=10,
                               showvalue=False,
                               command=lambda v: self.sims_lbl.config(
                                   text=str(int(float(v)))))
        self.slider.set(300)
        self.slider.pack(fill=tk.X, padx=10)
        self.slider.bind('<ButtonRelease-1>', self.apply_sims)
        tk.Label(self.root, text='applies from the next AI decision',
                 fg='#888', font=('Segoe UI', 8)).pack(pady=(0, 2))

        # ---- SEARCH WIDTH (USER 2026-08-22: 'cut those caps but add
        # extra sliders with those set as default').  Two search
        # settings: how many children an MCTS node
        # expands (by prior) and how many distinct outcomes a chance
        # node tracks.  The right end of each slider = NO CAP (written
        # as 0).  Boot value: the control file's, else the default.
        ctl0 = read_control()
        self.children_slider, self.children_lbl = self._width_slider(
            'MCTS children / node', 10, 510, 10, 40,
            ctl0.get('max_children'), self.apply_children)
        self.chance_slider, self.chance_lbl = self._width_slider(
            'chance outcomes / node', 4, 100, 4, 24,
            ctl0.get('chance_k'), self.apply_chance)

        self._fit_to_content(initial=True)

        self._connected_seen = False
        self._blink = False
        self._last_status_note = ''
        self._local_note = ''
        # THIS PANEL'S SESSION STARTS NOW (A1, see poll()): anything
        # the status file was stamped with before this instant belongs
        # to a game that is over.
        self._boot_ts = time.time()
        self.root.after(300, self.poll)

    def _fit_to_content(self, initial=False):
        """Keep the bottom controls visible when status text gains wrapped lines."""
        self.root.update_idletasks()
        requested = self.root.winfo_reqheight()
        # Leave room for window decoration and the desktop taskbar. The
        # initial spare space accommodates ordinary multi-line waiting text.
        top = 40 if initial else max(0, self.root.winfo_y())
        available = max(1, self.root.winfo_screenheight() - top - 80)
        if initial:
            self._panel_width = max(320, self.root.winfo_reqwidth())
        current = 0 if initial else self.root.winfo_height()
        height = min(available, max(current, requested + 48))
        self.root.minsize(320, min(requested, available))
        if initial or height != current:
            self.root.geometry('%dx%d%s' % (
                self._panel_width, height, '+40+40' if initial else ''))

    def _style_manual_btn(self):
        if self.manual_edit:
            self.medit_btn.config(text='EDITING - CLICK TO FINISH',
                                  fg='white', bg='#c22')
        else:
            self.medit_btn.config(text='MANUAL EDIT',
                                  fg='black', bg='#da0')

    def toggle_manual_edit(self):
        self.manual_edit = not self.manual_edit
        write_manual_edit(self.manual_edit)
        self._style_manual_btn()
        _plog('manual edit %s'
              % ('ON (gating suspended)' if self.manual_edit
                 else 'FINISHED (forced resync requested)'))
        if self.manual_edit:
            self.waiting.config(
                text='waiting on: MANUAL EDIT - make your board '
                     'edits, then click FINISH', fg='#c22')
        else:
            self._say('manual edit finished - forced resync requested')

    def toggle_pause(self):
        if getattr(self, '_rewind_review', False):
            _write_control(paused=False, resume_rewind=time.time())
            self._rewind_review = False
            self.paused = False
            self.pause_btn.config(text='PAUSE')
            return
        self.paused = not self.paused
        _write_control(paused=self.paused)
        self.pause_btn.config(text='RESUME' if self.paused else 'PAUSE')

    def _say(self, text):
        """Panel-originated note: shown until the server's status note
        changes (poll() never clobbers it with an unchanged note)."""
        self._local_note = text
        self.note.config(text=text, fg='#c60', font=('Segoe UI', 9))
        self._fit_to_content()

    def _one_shot(self, text, **kw):
        """Write a one-shot lever, show the feedback until the status
        note changes, and expire the lever from the control file."""
        if 'reset_resync' not in kw:
            self._recovery_requested_at = 0
        _write_control(**kw)
        self._say(text)
        keys = tuple(k for k in kw if k in ONE_SHOT_KEYS)
        if keys:
            self.root.after(ONE_SHOT_TTL_MS,
                            lambda: clear_control_keys(*keys))

    def my_turn(self):
        """Tell the server the HUMAN acts next."""
        self._one_shot('turn override sent: YOUR (human) turn',
                       whose_turn='human', whose_turn_epoch=time.time())

    def your_turn(self):
        """Tell the server the AI acts next."""
        self._one_shot("turn override sent: the AI's turn",
                       whose_turn='ai', whose_turn_epoch=time.time())

    def full_reset(self):
        """Request recovery; the server verifies the board before clearing errors."""
        now = time.time()
        self._recovery_requested_at = now
        self._one_shot('RESYNC requested - checking the client board; '
                       'turn order, unfinished decisions and known rolls will be retained.',
                       reset_resync=now)

    def _load_timeline(self):
        from panel_timeline import load_timeline, read_json
        return load_timeline(CLIENT_DIR, read_json(STATUS), boot_ts=self._boot_ts)

    def _request_rewind(self, checkpoint, session):
        from panel_timeline import TimelineUnavailable
        model = self._load_timeline()
        epoch = max(time.time(), self._last_rewind_epoch + 0.000001)
        request = model.request(checkpoint, session, epoch)
        if not _write_control(**request):
            raise TimelineUnavailable('Could not write the rewind request. Check the client folder and try again.')
        self._last_rewind_epoch = epoch
        self._recovery_requested_at = 0
        target = model.turn_start if checkpoint is None else checkpoint
        self._say('Rewind requested: %s. Waiting for the client and AI to confirm.' %
                  model.checkpoint_label(target, detailed=True))
        # Keep this session-bound request while a long AI search completes.
        # The backend consumes its epoch once; a newer click replaces it.

    def rewind_turn(self):
        from panel_timeline import TimelineUnavailable
        try:
            model = self._load_timeline()
            # Explicit null clears a prior manual target in the merge-written
            # control file; the backend resolves this session's turn start.
            self._request_rewind(None, model.session)
        except (TimelineUnavailable, OSError) as error:
            self._say(str(error))

    def open_timeline(self):
        from panel_timeline import TimelineWindow
        if self._timeline_window is not None and self._timeline_window.exists():
            self._timeline_window.show()
        else:
            self._timeline_window = TimelineWindow(self.root, self._load_timeline, self._request_rewind)

    def _refresh_rewind_controls(self, status, fresh):
        state = status.get('turn_rewind') or {}
        if not isinstance(state, dict):
            state = {}
        supported = bool(fresh and state.get('supported') and state.get('session'))
        self._rewind_review = bool(supported and state.get('review'))
        receipt = (state.get('session'), state.get('request_epoch'), state.get('restored_checkpoint_id'))
        if self._rewind_review and receipt != getattr(self, '_rewind_receipt', None):
            self._rewind_receipt = receipt
            self._rewind_review_note = state.get('note') or 'Rewind complete. Press RESUME when ready.'
            self._say(self._rewind_review_note)
        elif (supported and not self._rewind_review
              and getattr(self, '_rewind_review_note', None) == self._local_note):
            # Chat Resume/Skip can end review without touching panel controls.
            # Retire only this panel's paused-rewind notice, not newer feedback.
            self._rewind_review_note = None
            self._say('Play has resumed from the restored position.')
        self.pause_btn.configure(text='RESUME' if self._rewind_review or self.paused else 'PAUSE')
        self.rewind_manual_btn.configure(state=tk.NORMAL if supported else tk.DISABLED)
        self.rewind_turn_btn.configure(state=tk.NORMAL if supported and state.get('turn_available') else tk.DISABLED)
        if (self._timeline_window is not None and self._timeline_window.exists()
                and time.time() - self._timeline_poll_at >= 1):
            self._timeline_poll_at = time.time()
            self._timeline_window.refresh()

    def toggle_enforce(self):
        on = bool(self.enforce_var.get())
        _write_control(enforce_human=on)
        self._say('legality enforcement for your moves: %s'
                  % ('ON (beta) - illegal moves are undone with a '
                     'notice' if on else
                     'OFF - your moves are taken as given'))

    def skip_decision(self):
        self._one_shot('skip requested - decline if possible, otherwise abandon '
                       'the blocked interaction and use the visible board.',
                       skip_pending=time.time())

    def open_mirror(self):
        """ENGINE MIRROR (USER 2026-08-16): read-only WotrGUI on the
        engine snapshot, divergences painted.  Own process; never
        writes into the session."""
        try:
            import subprocess
            viewer = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), 'live_view.py')
            # NO CONSOLE WINDOW: CREATE_NO_WINDOW, and the viewer's
            # stdout+stderr go to ai_mirror_proc.log (the panel runs
            # under pythonw, whose children have no console, so
            # without the redirect a viewer crash would be silent).
            log = open(MIRROR_LOG, 'ab')
            subprocess.Popen([sys.executable.replace(
                'pythonw.exe', 'python.exe'), viewer, CLIENT_DIR],
                cwd=CLIENT_DIR, stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            self._say('engine mirror opening...')
        except Exception as e:
            self._say('mirror launch failed: %r' % (e,))

    def apply_sims(self, _evt=None):
        _write_control(sims=int(self.slider.get()))

    # ---- search-width sliders ------------------------------------
    def _width_slider(self, title, lo, hi, step, default, boot, apply):
        """One labelled slider whose TOP value means 'no cap' (0 in
        the control file).  Returns (slider, value label)."""
        row = tk.Frame(self.root)
        row.pack(fill=tk.X, padx=10, pady=(4, 0))
        tk.Label(row, text=title).pack(side=tk.LEFT)
        lbl = tk.Label(row, text='')
        lbl.pack(side=tk.RIGHT)
        tk.Label(row, text='(right end = no cap)', fg='#888',
                 font=('Segoe UI', 8)).pack(side=tk.RIGHT, padx=(0, 6))

        def _fmt(v):
            v = int(float(v))
            return 'no cap' if v >= hi else str(v)
        sl = tk.Scale(self.root, from_=lo, to=hi, orient=tk.HORIZONTAL,
                      resolution=step, showvalue=False,
                      command=lambda v: lbl.config(text=_fmt(v)))
        try:
            bv = int(boot) if boot is not None else None
        except (TypeError, ValueError):
            bv = None
        if bv is None:
            sl.set(default)
        elif bv <= 0:
            sl.set(hi)                      # 0 = no cap = the top
        else:
            sl.set(min(hi, max(lo, bv)))
        lbl.config(text=_fmt(sl.get()))
        sl.pack(fill=tk.X, padx=10)
        sl.bind('<ButtonRelease-1>', apply)
        return sl, lbl

    @staticmethod
    def _cap_value(slider):
        v = int(slider.get())
        return 0 if v >= int(slider.cget('to')) else v

    def apply_children(self, _evt=None):
        v = self._cap_value(self.children_slider)
        _write_control(max_children=v)
        self._say('MCTS children per node: %s'
                  % ('no cap' if v == 0 else v))

    def apply_chance(self, _evt=None):
        v = self._cap_value(self.chance_slider)
        _write_control(chance_k=v)
        self._say('chance outcomes per node: %s'
                  % ('no cap' if v == 0 else v))

    def poll(self):
        st = {}
        try:
            if os.path.exists(STATUS):
                with open(STATUS, encoding='utf-8') as f:
                    st = json.load(f)
        except (ValueError, OSError):
            st = {}
        # A PREVIOUS SESSION'S STATUS IS NOT OUR STATUS (A1, USER
        # 2026-08-18): ai_bridge_status.json outlives the session and
        # the java client launches this panel BEFORE it spawns the
        # server, so a status stamped before this panel booted is not
        # ours.  A panel restarted mid-game loses nothing - the live
        # server's next heartbeat (<=2s) is stamped after our boot and
        # is adopted in full.
        if st and float(st.get('ts', 0) or 0) < self._boot_ts:
            st = {}
        # server heartbeats every ~2s now; 12s of silence = really gone
        fresh = st and (time.time() - float(st.get('ts', 0))) < 12
        self._refresh_rewind_controls(st, fresh)
        if st.get('connected') and not self._connected_seen:
            self._connected_seen = True
            self.conn.config(text='AI connected', fg='#2a2')
            # no modal popup here (USER 2026-08-14: the 'AI client
            # connected' interrupt could block unattended games) —
            # the persistent green light above carries the state
        if self._connected_seen:
            self.conn.config(
                text='AI connected' if fresh else 'AI server quiet...',
                fg='#2a2' if fresh else '#a80')
        # not fresh (or never connected): surface the server's own
        # death reason instead of an eternal gray 'waiting...' — the
        # numpy import death and the stale-lock refusal both become
        # readable on screen (throttled to ~2s).
        # NOT DURING THE BOOT WINDOW (A1): before the first heartbeat
        # the newest error in the log is the PREVIOUS run's - claim a
        # death only once this session's server was seen, or once the
        # whole boot budget passed with no heartbeat
        _booting = (not self._connected_seen
                    and (time.time() - self._boot_ts) < 45)
        if not fresh and not _booting:
            self._dead_tick = getattr(self, '_dead_tick', 0) + 1
            if self._dead_tick % 7 == 1:
                why = server_death_line(CLIENT_DIR)
                if why:
                    self.conn.config(text='AI SERVER DOWN', fg='#c22')
                    self.waiting.config(
                        text='server error: %s' % why, fg='#c22')
        elif _booting:
            self.waiting.config(
                text='waiting on: AI server starting (engine + nets)',
                fg='#579')
        tt = turn_text(st)
        if tt is not None:
            self.turn.config(text=tt)
        state = str(st.get('state'))
        # banner priority: MANUAL EDIT outranks everything (gating is
        # suspended - the loudest possible state), then PAUSED, then
        # the thinking blink.  Trust the bridge's echo when present so
        # panel and server agree; the local flag covers the gap before
        # the first echo.
        if st.get('manual_edit') or self.manual_edit:
            self.think.config(text='MANUAL EDIT MODE - gating '
                                   'suspended', bg='#c22')
        elif state == 'paused' or self.paused:
            self.think.config(text='PAUSED', bg='#c80')
        elif state == 'thinking':
            self._blink = not self._blink
            self.think.config(text='AI THINKING...',
                              bg='#c33' if self._blink else '#a11')
        else:
            self.think.config(text='', bg=self.root['bg'])
        sims = st.get('sims')
        if sims:
            # echo only — never touch the slider's own label
            self.srv_lbl.config(text='(server: %s)' % sims)
        # refused gestures are unmissable (USER ruling 2026-08-16:
        # illegal human moves are refused with the reason) - the
        # bridge's refusal notes start with 'illegal:'
        _cur = str(st.get('note') or '')
        _requested = getattr(self, '_recovery_requested_at', 0)
        _note = note_text(st, self._last_status_note, self._local_note, _requested)
        _recovery = st.get('recovery') or {}
        if (_requested and _recovery.get('summary')
                and _recovery.get('epoch', 0) >= _requested):
            self._local_note = ''
        if _cur and _cur != self._last_status_note:
            self._local_note = ''        # the server spoke: feedback done
        self._last_status_note = _cur
        if _note.startswith('illegal:'):
            self.note.config(text=_note, fg='#c22',
                             font=('Segoe UI', 9, 'bold'))
        else:
            self.note.config(text=_note, fg='#c60',
                             font=('Segoe UI', 9))
        if fresh:
            w = st.get('waiting_on')
            if state == 'thinking':
                w = 'thinking (%s sims)' % (sims or '?')
            if w:
                self.waiting.config(text=(w if str(w).lower().startswith('waiting on')
                                          else 'waiting on: %s' % w),
                                    fg='#579')
        self._fit_to_content()
        self.root.after(300, self.poll)

    def run(self):
        self.root.mainloop()


def main():
    """The panel may never exit silently: the start line is the FIRST
    statement, every refusal logs its reason (single_instance), and a
    crash leaves its traceback in ai_panel_proc.log before
    propagating (so the spawner's rc is diagnosable from the log)."""
    _plog('panel starting pid=%d py=%s client_dir=%s'
          % (os.getpid(), sys.version.split()[0], CLIENT_DIR))
    try:
        single_instance_or_exit()
        Panel().run()
        _plog('panel closed (mainloop ended)')
    except SystemExit as e:
        _plog('panel exiting code=%s' % (e.code,))
        raise
    except BaseException:
        import traceback
        _plog('panel CRASHED:\n' + traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
