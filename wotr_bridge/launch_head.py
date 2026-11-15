# -*- coding: ascii -*-
"""launch_head.py - THE launch facts of the one head (Play_vs_AI.bat).

ONE IMPLEMENTATION PER QUANTITY.  Every fact about how a human-vs-AI
(or AI vs AI watch) session is launched lives here and nowhere else:

  * HEAD_ENV        the environment every mode sets (CPU pins, the net
                    generation switch);
  * SERVER_TOKENS   the AI server command line (python -m core.server
                    ..., cwd = wotr_bridge);
  * JAVA_TOKENS     the client command line (java with the AiFeed agent
                    and the -D flags, cwd = the client dir);
  * WATCH_*         the AI vs AI (watch) mode: side AUTO (alias WATCH),
                    its feed side and extra client flags;
  * FP_NETS / SH_NETS / NET_TAGS  the nets an AI seat can play and their
                    file tags (the bat's NET MENU asks; NET_TAG = default);
  * LEFTOVER_FILES  the coordination files a fresh session must not
                    inherit; ROTATE_FILES the diagnostics kept as .prev;
  * the text of Play_vs_AI.bat itself (render_bat), generated from the
    facts above.

CONSUMERS
  * Play_vs_AI.bat IS render_bat(): `python launch_head.py --write-bat`
    regenerates it; `--check` (and src\\test_launch_head.py) fail when
    the file on disk differs from the render, so the bat and this
    module cannot drift.
  * WOTR_AI_training_lab\\streak\\boot_e2e_harness.py builds its java
    and server command lines from java_cmd() / server_cmd(), so a
    harness leg launches the client exactly as the USER's bat does.

BAT LAYOUT INVARIANT: the play path comes first and ends with the java
line followed by `goto cleanup`; the `:usage` section sits between
that jump and the `:cleanup` label (the engine GUI is the separate
generated Play_Engine_Only.bat; console mode was removed 2026-08-23).

RE-ENTRY GUARD (R4-BAT-REENTRY-GUARD): cmd.exe re-reads a running batch
file by byte offset; after the foreground java exits it goes on at the
offset where ITS java line ended, even in a file regenerated meanwhile.
Re-entry is made harmless by the JAVAGUARD variable, not by byte
offsets (rounds 1-3 moved `goto cleanup` 11249 -> 13235 bytes, so an
instance of the installed file resumed the regenerated one in the
in-game steps and ran on into the java line: a second client).  The top
of the bat sets JAVAGUARD=armed; the server start and the java line are
each preceded by `if not "!JAVAGUARD!"=="armed" goto cleanup`, and the
java line also by `set "JAVAGUARD=spent"`.  A resumed instance spent the
guard on its own java line, or (a file older than the guard) never armed
it, so the first check it reaches sends it to the cleanup: no second
client, no second server.  It may first echo a few lines, and a resume
in the heartbeat wait finds the old server's heartbeat.  Not covered: a
resume offset inside the guard / java lines themselves (about 260
bytes), and one past them when the play path shrank by more than PAD
(it can end in `:usage`, exiting without the cleanup).  So BAT-REGEN
still writes the file only while no Play_vs_AI.bat instance is running
(src\\test_round4_R4_BAT_REENTRY_GUARD.py checks the known offsets).

Stdlib only: the bat is regenerated and checked without torch.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRIDGE_DIR = os.path.join(ROOT, 'wotr_bridge')
PLAY_CLIENT = os.path.join(BRIDGE_DIR, 'play_client')
BAT_PATH = os.path.join(ROOT, 'Play_vs_AI.bat')

PY = r'C:\ProgramData\spyder-6\envs\spyder-runtime\python.exe'
PYW = r'C:\ProgramData\spyder-6\envs\spyder-runtime\pythonw.exe'
if os.name != 'nt':     # (not Windows: no development runtime to fall back on - the Python that runs this)
    PY = PYW = sys.executable

# the deployed nets: league_main's four final mains (cycle 200, 2M learning
# games each; USER 2026-11-01 "give alphawotr the option to load any of the
# 4", "the option should be given in the bat").  Each AI seat plays one of its
# side's two nets, chosen in the bat (NET MENU below) or by its arguments.
# The server opens wotr_<seat>_<tag>.pt at the project root, so there is one
# file pair per combination, tag NET_TAG_FORMAT % (fp key, sh key); a seat
# only opens its own file.  Encoding generation NET_GEN (HEAD_ENV
# WOTR_ACT_GEN; the loader refuses a checkpoint of any other generation).
# NET_TAG = the default combination (both WIN nets): server_cmd / java_cmd,
# the engine-only bat, and the bat when the side comes from an argument with
# no net argument (harness legs).  The D6 pair (d6_r1sp_r2sp) and the v52
# corpus_live pair stay on disk.
# VNETINSTALL (2026-11-03): the two WIN nets search with a separately trained
# value net and c_puct 1.5 (live/own_model.py SEARCH_VALUE_NETS); MILITARY and
# CORRUPTION keep their own value head and c_puct 4 (the value nets learnt the
# win objective).  The menu text says which ("c1.5" / "c4" = c_puct).  A text
# of the bat may change its length freely: the file is written with CR LF line
# ends (write_bat), and only in a file with bare LF does cmd's label search
# depend on byte offsets (the same-length rule of the paragraphs below is the
# history of that fault, no longer a rule).
# LF050 (2026-11-03, USER "install the current nets here"): WIN on each side
# is now the full-network league_full net at cycle 50 (500K learning games,
# EVERY layer trained; FP 776d5d5e, SH 2eb37451), the default; MILITARY and
# CORRUPTION stay league_main's c200 nets.  New tags lf050_fp*_sh* (one
# file pair per combination as before); the lg200_* pairs stay on disk but
# leave the menu: a third menu entry would change the bat's line layout
# (see above), so the c200 WIN nets come back only by the LF050 rollback.
# The tag prefix and the WIN texts keep their exact lengths, so every line
# of both bats keeps its byte offset (lfinstall_1103 make_stage.py guard).
# LF201 (2026-11-04, USER "install the 2M nets"): WIN on each side is now
# the same league_full net trained on to cycle 201 (2,000,000 learning
# games; FP 17dcfcc2, SH d5f7a332), the default; MILITARY and CORRUPTION
# are unchanged.  New tags lf201_fp*_sh*; the lf050_* and lg200_* pairs
# stay on disk (out of the menu; back only by the LF201 / LF050 rollback).
# Same-length tag prefix and WIN texts (lfinstall_1104 make_stage.py guard).
FP_NETS = (('win', 'FP-WIN', 'search-trained (c008); own judge c1.5'),
           ('mil', 'FP-MILITARY', 'military win, full network 2M games; own head c4'))
SH_NETS = (('win', 'SH-WIN', 'search-trained (c008); own judge c1.5'),
           ('cor', 'SH-CORRUPTION', 'corruption, full network 2M games; own head c4'))
NET_TAG_FORMAT = 'mc200_fp%s_sh%s'
NET_TAGS = tuple(NET_TAG_FORMAT % (f[0], s[0]) for f in FP_NETS for s in SH_NETS)
NET_TAG = NET_TAG_FORMAT % (FP_NETS[0][0], SH_NETS[0][0])
NET_GEN = 'v53'
SERVER_SIMS = 300
JAVA_PACE = 800
JAR_NAME = 'WOTR-Gandalf-AI.jar'
FEED_JAR = 'AiFeed.jar'
FEED_NAME = 'ai_feed.txt'
HEARTBEAT_FILE = 'ai_bridge_status.json'
SERVER_CONSOLE_LOG = 'ai_server_console.log'
HEARTBEAT_WAIT_S = 240
# the server holds its folder's alive byte within a few seconds of its start (before the
# nets load); from this many seconds on, "nobody holds it" ends the wait: the server is gone
SERVER_GRACE_S = 30

# the environment every mode sets ('-1', never an empty string: torch
# reports cuda with 0 devices on '')
HEAD_ENV = (
    ('CUDA_VISIBLE_DEVICES', '-1'),
    ('KMP_DUPLICATE_LIB_OK', 'TRUE'),
    ('KMP_BLOCKTIME', '0'),
    ('OMP_WAIT_POLICY', 'PASSIVE'),
    ('PYTHONHASHSEED', '0'),
    ('WOTR_ACT_GEN', NET_GEN),
    ('WOTR_REWARD_CONTRACT', 'predecision_profile_v1'),
    # Legacy wotr_native.pyd mirrors in ai.py; unrelated to the C++ search DLL.
    # Keep the live Python driver and select native search independently below.
    ('WOTR_NATIVE', '0'),
    ('WOTR_LIVE_SEARCH_BACKEND', 'native'),
    ('WOTR_LIVE_INFERENCE_THREADS', '1'),
    ('WOTR_LIVE_INFERENCE_CACHE', '1'),
    ('WOTR_LIVE_POLICY_THREADS', '8'),
    # Free-action floor (ai.py task-14 ring-starvation fixes; read at ai
    # import by the Python search AND the native live box's build_config):
    # the top-3-prior free arms (elven/dwarven ring, use_token) keep a
    # root slot and get >= 12 forced visits each.  Off, a ~1e-4-prior
    # ring arm got zero visits in live play (fixwave INV-RING-2).
    ('WOTR_FREE_CANDS', '1'),
    ('WOTR_FREE_FORCE', '12'),
)

# coordination files a new session must not inherit (a survivor would
# be adopted as THIS session's heartbeat / control / board)
LEFTOVER_FILES = (
    'ai_bridge_status.json', 'ai_bridge_request.log',
    'ai_bridge_move.txt', 'ai_panel_control.json', 'ai_feed.txt',
    'ai_feed.txt.meta.jsonl', 'ai_client_board.txt',
    # bridge2: core.server writes no ai_live_state.pkl / ai_live_status.json /
    # ai_census.jsonl; they stay listed so that a file left by the old
    # server is not inherited.  The taken panel request is the core's.
    'ai_live_state.pkl', 'ai_live_status.json', 'ai_census.jsonl',
    'ai_panel_request.json.taken',
    'ai_server.lock', 'ai_server.alive', 'ai_panel.pid',
    # X1-05: the client appends one ack per injected line forever and the
    # server re-reads the whole journal every tick; acks bind to THIS
    # session's operations only (live/ack_identity.py)
    'ai_inject_ack.jsonl',
    # X1-04: the Java 8 Stop AI marker (AiOpp.stopAdoptedServer writes it,
    # live/transport.py STOP_NAME reads it) belongs to ONE session; the
    # stopped server never deletes it (serve() retires it at startup)
    'ai_server.stop',
    # L1104-U04 (UI_SPEC 5.1): the in-client panel's view / request slot /
    # step acks belong to ONE session (live/l1104_panel.py)
    'ai_panel_view.json', 'ai_panel_request.json', 'ai_panel_ack.jsonl',
)
# diagnostics of the previous session: kept once as <name>.prev
ROTATE_FILES = ('ai_boot.log', 'ai_server_console.log',
                'ai_panel_proc.log', 'ai_server.log',
                # bridge2: the core's step and comparison log of one session
                'core_log.jsonl')

# the AI server: cwd = wotr_bridge (the package is `core`, never
# `wotr_bridge.core`).  bridge2: wotr_bridge\core\server.py replaced
# live\live_server.py - same cwd, same arguments
SERVER_TOKENS = ('{py}', '-u', '-m', 'core.server',
                 '--client-dir', '{client}', '--side', '{side}',
                 '--sims', '{sims}', '--net-tag', '{net}')

# the client: cwd = the client dir (resources load from the cwd).
# -Dai.feed.aiside is what blinds the AI to the human's hand; the
# client reads it and asks no side question.
# the client mods + AiFeed agent are compiled --release 8 (2026-08-23:
# testers on Java 8 / 17 died on the Java-21 class files; the ownership
# guard's Java-9 ProcessHandle is behind reflection with a file-lock
# proof fallback, so the original client's own Java 8 is enough)
JAVA_MIN = 8

JAVA_TOKENS = ('java',
               '-javaagent:' + FEED_JAR + '={feed}',
               '-Dai.feed.aiside={side}',
               '-Dai.python={pyw}',
               '-Dai.bridgedir={bridge}',
               '-Dai.net={net}',
               '-Dai.pace=%d' % JAVA_PACE,
               '-jar', JAR_NAME)

# AI vs AI (watch), R3-AI-VS-AI-LAUNCHER (USER X1-07 2026-10-23: "AI vs AI
# watch should work properly."): _side() maps AUTO and its alias WATCH to
# WATCH_SIDE.  The server then runs with --side AUTO (live_server
# AUTO_SIDE_ARGS: the AI plays both seats), and the client gets
# -Dai.feed.aiside=none followed by WATCH_JAVA_OPTS:
#   -Dai.feed.blinding=off  AiFeed blinds no hand, so with aiside=none it
#       attests '#BLIND aiside=none hands=none', the one attestation an AI
#       vs AI server accepts (live_server.AUTO_FEED_ATTESTATION).
#       -Dai.feed.aiside=none alone keeps BOTH hands blinded (hands=FP,SA),
#       and that server refuses it.  No hand belongs to a human here: the
#       server decodes both hands anyway, and each seat's search sees only
#       its own hand (1.261);
#   -Dai.side=AUTO  the launcher fixed the mode (AiOpp.fixedSide() reads
#       ai.side first; R3-AI-VS-AI-CLIENT makes it SA / FP / AUTO);
#   -Dai.autoplay=1  (R4-AUTO-STARTUP-DEFER) AiBoot starts AI vs AI by itself
#       on the FIRST game window it sees - the game the client opens (the
#       last game type played).  autoPlay now waits for the native game
#       initialization the way Play vs AI does (AiStartup.deferAuto beside
#       AiStartup.defer, CR-D9 / CR-L3-05), so it never requests against the
#       unfinished startup log.  The session binds to that game: File > New /
#       Load stops the watch and needs a fresh launch (J2-07), which the bat's
#       watch steps say.  Before R4-AUTO-STARTUP-DEFER the flag was withheld
#       and the watch started from the menu (AI -> AI vs AI (watch)), which
#       is still the route when the automatic start gave up.
WATCH_SIDE = 'AUTO'
WATCH_ALIASES = ('AUTO', 'WATCH')
WATCH_FEED_SIDE = 'none'
WATCH_JAVA_OPTS = ('-Dai.feed.blinding=off', '-Dai.side=' + WATCH_SIDE,
                   '-Dai.autoplay=1')
# the java token the watch options follow (java_cmd and the bat line)
_FEED_SIDE_TOKEN = '-Dai.feed.aiside={side}'

# tokens whose substituted value is a path (quoted in the bat)
_PATH_TOKENS = ('{py}', '{client}', '{pyw}', '{bridge}', '{feed}')


def head_env(base=None):
    """The launch environment: `base` (default os.environ) + HEAD_ENV."""
    env = dict(os.environ if base is None else base)
    for k, v in HEAD_ENV:
        env[k] = v
    return env


def server_cmd(client_dir, ai_side, sims=SERVER_SIMS, py=PY, extra=(),
               net_tag=NET_TAG):
    """argv of the AI server for `client_dir`; run it with
    cwd=BRIDGE_DIR and env=head_env().  `extra` appends e.g.
    ('--seed', '42') or ('--scenario', dir); `net_tag` is one of
    NET_TAGS (default: both WIN nets)."""
    ai_side = _side(ai_side)
    out = [t.format(py=py, client=client_dir, side=ai_side, sims=sims,
                    net=_net(net_tag))
           for t in SERVER_TOKENS]
    return out + [str(x) for x in extra]


def java_cmd(client_dir, ai_side, pythonw=PYW, bridge_dir=BRIDGE_DIR,
             extra_java_opts=(), net_tag=NET_TAG):
    """argv of the client for `client_dir`; run it with cwd=client_dir.
    `extra_java_opts` (harness agents / -D flags) go before -jar.
    AUTO (watch): -Dai.feed.aiside=none + WATCH_JAVA_OPTS right after it."""
    ai_side = _side(ai_side)
    watch = ai_side == WATCH_SIDE
    toks = [t.format(client=client_dir, feed=os.path.join(client_dir, FEED_NAME),
                     side=WATCH_FEED_SIDE if watch else ai_side,
                     pyw=pythonw, bridge=bridge_dir, net=_net(net_tag))
            for t in JAVA_TOKENS]
    if watch:
        i = JAVA_TOKENS.index(_FEED_SIDE_TOKEN) + 1
        toks[i:i] = list(WATCH_JAVA_OPTS)
    i = toks.index('-jar')
    return toks[:i] + [str(x) for x in extra_java_opts] + toks[i:]


def _side(ai_side):
    """'SA' / 'FP' (the AI's seat) or WATCH_SIDE for AUTO / WATCH (AI vs
    AI); any case.  Anything else is refused."""
    s = str(ai_side).upper()
    if s in WATCH_ALIASES:
        return WATCH_SIDE
    if s not in ('SA', 'FP'):
        raise ValueError('ai_side must be SA, FP or AUTO (alias WATCH), '
                         'got %r' % (ai_side,))
    return s


def _net(net_tag):
    """One of NET_TAGS; anything else is refused."""
    if net_tag not in NET_TAGS:
        raise ValueError('net_tag must be one of %s, got %r'
                         % (', '.join(NET_TAGS), net_tag))
    return net_tag


# ---------------------------------------------------------------------
# the bat: the same tokens, cmd-quoted, with the bat's variables
def _bat_tokens(tokens, **subs):
    out = []
    for t in tokens:
        if any(k in t for k in _PATH_TOKENS):
            if t.startswith('{'):
                out.append('"' + t.format(**subs) + '"')
            else:
                name, val = t.split('=', 1)
                out.append(name + '="' + val.format(**subs) + '"')
        else:
            out.append(t.format(**subs))
    return out


def bat_java_line():
    """java_cmd() in the bat's variables: FEEDSIDE is SA / FP / none and
    WATCHOPTS is WATCH_JAVA_OPTS in watch mode, undefined (expands to
    nothing) otherwise; the bat's :sidechosen block sets both."""
    toks = _bat_tokens(JAVA_TOKENS, client='!PLAY!', feed='!PLAY!\\' + FEED_NAME, side='%FEEDSIDE%',
                       pyw='!PYW!', bridge='!BRIDGE!', net='%NETTAG%')
    i = JAVA_TOKENS.index(_FEED_SIDE_TOKEN) + 1
    toks[i:i] = ['%WATCHOPTS%']
    return ' '.join(toks)


def bat_server_line():
    inner = ' '.join(_bat_tokens(SERVER_TOKENS, py='!PY!',
                                 client='!PLAY!', side='%AISIDE%',
                                 sims=SERVER_SIMS, net='%NETTAG%'))
    return ('start "AlphaWOTR AI server" /min cmd /c "' + inner
            + ' 1>"!PLAY!\\' + SERVER_CONSOLE_LOG + '" 2>&1"')


def bat_env_lines():
    return '\n'.join('set "%s=%s"' % kv for kv in HEAD_ENV)


# the bat's one way to stop what a session of !PLAY! left running (stop_ours below)
BAT_STOP_LINE = '"!PY!" "!BRIDGE!\\launch_head.py" --stop-ours "!PLAY!" >nul 2>&1'


def server_pid(client_dir):
    """The pid of the AI server that runs for `client_dir`, or None when none does: the
    process that holds the folder's alive byte (core.server_ownership, the one proof).  The
    server takes the byte before it loads its nets, so "none" a short while after a start
    means the server is gone."""
    from core import server_ownership, transport
    return server_ownership.holder(os.path.join(client_dir, transport.ALIVE_NAME),
                                   transport.ALIVE_LOCK_OFFSET)


def stop_ours(client_dir):
    """Stop the AI server of `client_dir` (server_pid) and its old panel window, and nothing
    else.  The panel: the pid in its own pid file, when its command line shows ai_panel.py
    started for this folder (pid_owner, the panel's own answerer).  A program that merely
    names the folder is never stopped.  Returns the pids."""
    import signal
    import pid_owner
    ours = [server_pid(client_dir)]
    try:
        with open(os.path.join(client_dir, 'ai_panel.pid')) as f:
            panel = int(f.read().strip() or 0)
    except (OSError, ValueError):
        panel = 0
    line = (pid_owner.pid_command_line(panel) or '').replace('"', '') if panel else ''
    folder = os.path.normcase(os.path.abspath(client_dir))
    if 'ai_panel.py ' in line and os.path.normcase(os.path.abspath(
            line.split('ai_panel.py ', 1)[1].strip())) == folder:
        ours.append(panel)
    stopped = []
    for pid in ours:
        if pid and pid != os.getpid():
            try:
                os.kill(pid, signal.SIGTERM)
                stopped.append(pid)
            except OSError:         # it ended by itself between the proof and now
                pass
    return stopped


# NET MENU: the bat block between the side choice and the fresh-session
# sweep.  It sets FPNET / SHNET (the FP_NETS / SH_NETS keys), NETTAG (the
# file tag the server and the client get) and prints the AI's net names.
def _bat_net_prompt(var, title, nets):
    keys = ''.join(str(i + 1) for i in range(len(nets)))
    out = ['echo    Which %s AI?' % title, 'echo.']
    out += ['echo      [%d]  %-15s %s' % (i + 1, name, what)
            for i, (_, name, what) in enumerate(nets)]
    out += ['echo.',
            'choice /c %s /n /m "    Press %s or %s: "'
            % (keys, ', '.join(keys[:-1]), keys[-1])]
    ladder = ' else '.join('if errorlevel %d (set "%s=%s")'
                           % (i + 1, var, nets[i][0])
                           for i in range(len(nets) - 1, 0, -1))
    out += [ladder + ' else (set "%s=%s")' % (var, nets[0][0]), 'echo.']
    return out


def _bat_net_args(var, arg, ok, nets):
    out = ['if not defined %s set "%s=1"' % (arg, ok)]
    for key, _, _ in nets:
        out += ['if /I "!%s!"=="%s" set "%s=%s"' % (arg, key, var, key),
                'if /I "!%s!"=="%s" set "%s=1"' % (arg, key, ok)]
    return out


def _bat_net_names(var, name_var, nets):
    out = ['set "%s=%s"' % (name_var, nets[0][1])]
    out += ['if /I "!%s!"=="%s" set "%s=%s"' % (var, key, name_var, name)
            for key, name, _ in nets[1:]]
    return out


def bat_net_menu():
    w = WATCH_SIDE
    lines = [
        ':: Which nets?  Each AI seat plays one of its side\'s final league',
        ':: nets (launch_head FP_NETS / SH_NETS).  Asked only when the side was',
        ':: asked; arguments skip it:  Play_vs_AI.bat SA COR  /  FP MIL  /',
        ':: %s MIL COR (Free Peoples net first).  A side argument with no net' % w,
        ':: argument takes the WIN nets, so a harness leg never meets a prompt.',
        'set "FPNET=%s"' % FP_NETS[0][0],
        'set "SHNET=%s"' % SH_NETS[0][0],
        'set "FPARG="',
        'set "SHARG="',
        'if /I "%AISIDE%"=="FP" set "FPARG=%~2"',
        'if /I "%AISIDE%"=="SA" set "SHARG=%~2"',
        'if /I "%%AISIDE%%"=="%s" set "FPARG=%%~2"' % w,
        'if /I "%%AISIDE%%"=="%s" set "SHARG=%%~3"' % w,
        'if defined FPARG goto netargs',
        'if defined SHARG goto netargs',
        'if not defined ASKNETS goto netschosen',
        'if /I "%AISIDE%"=="SA" goto asksh']
    lines += _bat_net_prompt('FPNET', 'Free Peoples', FP_NETS)
    lines += ['if /I "%AISIDE%"=="FP" goto netschosen', ':asksh']
    lines += _bat_net_prompt('SHNET', 'Shadow', SH_NETS)
    lines += ['goto netschosen', ':netargs', 'set "FPOK="', 'set "SHOK="']
    lines += _bat_net_args('FPNET', 'FPARG', 'FPOK', FP_NETS)
    lines += _bat_net_args('SHNET', 'SHARG', 'SHOK', SH_NETS)
    lines += [
        'if not defined FPOK goto netbad',
        'if not defined SHOK goto netbad',
        'goto netschosen',
        ':netbad',
        'echo   [X] Unknown net in "!FPARG! !SHARG!".  Free Peoples nets: %s;'
        % ' or '.join(k.upper() for k, _, _ in FP_NETS),
        'echo       Shadow nets: %s.  Example: Play_vs_AI.bat %s MIL COR'
        % (' or '.join(k.upper() for k, _, _ in SH_NETS), w),
        'echo.',
        'pause',
        'exit /b 1',
        ':netschosen',
        'set "NETTAG=%s"' % (NET_TAG_FORMAT % ('!FPNET!', '!SHNET!'))]
    lines += _bat_net_names('FPNET', 'FPNAME', FP_NETS)
    lines += _bat_net_names('SHNET', 'SHNAME', SH_NETS)
    lines += [
        'if /I "%AISIDE%"=="FP" set "AINAMES=!FPNAME!"',
        'if /I "%AISIDE%"=="SA" set "AINAMES=!SHNAME!"',
        'if /I "%%AISIDE%%"=="%s" set "AINAMES=!FPNAME! vs !SHNAME!"' % w,
        'echo    The AI plays !AINAMES! (net tag !NETTAG!).',
        'echo.']
    return '\n'.join(lines) + '\n'


# the comment line between the java line and `goto cleanup` (see the
# BAT LAYOUT INVARIANT above).  Since R4-BAT-REENTRY-GUARD it places
# nothing: the JAVAGUARD checks keep a resumed instance off the java
# line.  Its length is kept as a landing strip (a resume inside it falls
# through to the jump).  cmd runs a partial line from the resume byte on,
# even mid-word (a resume inside "guard:" runs `rd: ...`, which removes
# empty directories), so the strip is dashes, and no explanatory comment
# goes into the bat's resume zone (the play path after the probe).
PAD = (':: the game window closed - fall through to the cleanup below '
       + '-' * 136)

_PROBE_IMPORTS = 'import torch, numpy, shapely, PIL, matplotlib, psutil, tkinter'

_PROBE = r''':: ---- which Python: the BUNDLED runtime (python\python.exe, shipped in
::      the release zip), then ALPHAWOTR_PYTHON, then python.exe on PATH,
::      then the developer runtime; require all play dependencies, not only torch
set "PY="
if exist "!HERE!python\python.exe" (
    "!HERE!python\python.exe" -c "{PROBE_IMPORTS}" >nul 2>&1
    if not errorlevel 1 set "PY=!HERE!python\python.exe"
)
if not defined PY if defined ALPHAWOTR_PYTHON (
    "!ALPHAWOTR_PYTHON!" -c "{PROBE_IMPORTS}" >nul 2>&1
    if not errorlevel 1 set "PY=!ALPHAWOTR_PYTHON!"
)
if not defined PY (
    set "PY_PROBE="
    for /f "delims=" %%p in ('where python.exe 2^>nul') do (
        if not defined PY_PROBE set "PY_PROBE=%%p"
    )
    if defined PY_PROBE (
        "!PY_PROBE!" -c "{PROBE_IMPORTS}" >nul 2>&1
        if not errorlevel 1 set "PY=!PY_PROBE!"
    )
)
if not defined PY (
    if exist "{PY}" (
        "{PY}" -c "{PROBE_IMPORTS}" >nul 2>&1
        if not errorlevel 1 set "PY={PY}"
    )
)
if not defined PY (
    echo   [X] No Python with all play dependencies found.
    if exist "!HERE!python\python.exe" (
        echo       The bundled runtime python\python.exe exists but cannot import
        echo       all play dependencies. Its error follows:
        echo       ----------------------------------------------------------
        "!HERE!python\python.exe" -c "{PROBE_IMPORTS}" 2>&1 | findstr /V /C:"^$"
        echo       ----------------------------------------------------------
        echo       Usual causes: the zip was not fully unpacked ^(use Explorer or
        echo       7-Zip, to a SHORT path like C:\AlphaWOTR^), or a DLL load error
        echo       ^(try: install the Microsoft Visual C++ Redistributable x64^).
    ) else (
        echo       No python\ runtime in this folder ^(it ships in the release
        echo       zip^).  Install Python 3.10+ and run
        echo       pip install -r requirements.txt   ^(or set ALPHAWOTR_PYTHON^)
    )
    echo       Required: torch, numpy, shapely, Pillow, matplotlib, psutil, tkinter.
    echo       PIL is provided by the Pillow package in requirements.txt.
    echo.
    pause
    exit /b 1
)
set "PYW=!PY:python.exe=pythonw.exe!"
if not exist "!PYW!" set "PYW=!PY!"
echo   Python: "!PY!"
'''

_BAT = r'''@echo off
setlocal EnableExtensions DisableDelayedExpansion
:: the folder of this file is read while an exclamation mark is an ordinary
:: character (a folder name may hold one); afterwards every path is a !NAME!
set "HERE=%~dp0"
setlocal EnableDelayedExpansion
:: =====================================================================
:: Play_vs_AI.bat -- THE ONE HEAD of AlphaWOTR.  GENERATED from
:: wotr_bridge\launch_head.py (--write-bat); edit the module, never
:: this file (src\test_launch_head.py pins the two together).
::   Play_vs_AI.bat [SA|FP] [net]   play on the real client (SA = the AI
::                                  is Shadow, FP = the AI is Free Peoples)
::   Play_vs_AI.bat AUTO [fp] [sh]  watch AI vs AI on the real client (the
::                                  AI plays both sides; alias WATCH)
::   Play_Engine_Only.bat           the engine's own board, no Java (UNTESTED)
::   nets: Free Peoples {FP_KEYS_BAR}, Shadow {SH_KEYS_BAR} (default WIN)
:: Client wotr_bridge\play_client; AI server FIRST (heartbeat), then
:: java adopts it; nets {NET_TAG_PATTERN} (WOTR_ACT_GEN={NET_GEN}).
:: =====================================================================
title AlphaWOTR - Play vs AI
:: re-entry guard (launch_head.py RE-ENTRY GUARD): only a fresh run is
:: armed; the client launch spends it, so a running instance that resumes
:: a regenerated copy of this file goes to the cleanup instead
set "JAVAGUARD=armed"

set "LAUNCHLOG=!HERE!launcher.log"
echo [start %DATE% %TIME%] > "!LAUNCHLOG!"
set "PLAY=!HERE!wotr_bridge\play_client"
set "BRIDGE=!HERE!wotr_bridge"
{PROBE}
:: certified-path env (CPU pinned; '-1', never an empty string) and the
:: generation switch WOTR_ACT_GEN={NET_GEN} (the loader refuses other generations)
{ENV_LINES}

if /I "%~1"=="help" goto usage
if "%~1"=="/?" goto usage

:: ---- play: the real client against the live stack -------------------
echo.
echo   ================================================================
echo    AlphaWOTR  --  human vs AI
echo   ================================================================
echo.

:: ---- which Java: JAVA_HOME first (the standard override), then PATH.
::      The client and its add-ons are Java 8 class files, so any
::      working java runs them; the version check below only WARNS.
echo [java-check] >> "!LAUNCHLOG!"
if defined JAVA_HOME if exist "!JAVA_HOME!\bin\java.exe" set "PATH=!JAVA_HOME!\bin;%PATH%"
where java >nul 2>&1
if errorlevel 1 (
    echo   [X] Java was not found on PATH.
    echo       Install Java ^(Temurin 17 or 21 from https://adoptium.net^),
    echo       or set JAVA_HOME to an installed JDK/JRE folder.
    echo.
    echo [no-java] >> "!LAUNCHLOG!"
    pause
    exit /b 1
)
set "JV="
for /f tokens^=3 %%v in ('java -version 2^>^&1 ^| findstr /i "version"') do if not defined JV set "JV=%%~v"
echo   Java %JV%
echo [java-ok %JV%] >> "!LAUNCHLOG!"
if not exist "!PLAY!\{JAR_NAME}" (
    echo   [X] The play client is missing: !PLAY!
    echo.
    pause
    exit /b 1
)

:: Which side do YOU want?  Asked ONCE, before java starts: it sets the
:: server's --side AND -Dai.feed.aiside (the flag that hides your hand
:: from the AI); the client reads that flag and does not ask again; the
:: server refuses a session where the two disagree.  WATCH = AI vs AI:
:: the AI plays both seats (server --side AUTO; the client gets
:: -Dai.feed.aiside=none plus WATCHOPTS = launch_head WATCH_JAVA_OPTS).
:: An argument skips the prompt:  Play_vs_AI.bat SA   (AI plays Shadow)
::                                Play_vs_AI.bat FP   (AI plays Free Peoples)
::                                Play_vs_AI.bat AUTO (watch; alias WATCH)
set "ASKNETS="
set "AISIDE=%~1"
if /I "%AISIDE%"=="SA" goto sidechosen
if /I "%AISIDE%"=="FP" goto sidechosen
if /I "%AISIDE%"=="AUTO" goto sidechosen
if /I "%AISIDE%"=="WATCH" goto sidechosen
set "ASKNETS=1"
echo    Which side do you want to play?
echo.
echo      [1]  FREE PEOPLES   (the AI plays Shadow)
echo      [2]  SHADOW         (the AI plays Free Peoples)
echo      [3]  WATCH AI vs AI (the AI plays both sides)
echo.
choice /c 123 /n /m "    Press 1, 2 or 3: "
if errorlevel 3 (set "AISIDE={WATCH_SIDE}") else if errorlevel 2 (set "AISIDE=FP") else (set "AISIDE=SA")
:sidechosen
set "WATCHOPTS="
if /I "%AISIDE%"=="WATCH" set "AISIDE={WATCH_SIDE}"
if /I "%AISIDE%"=="{WATCH_SIDE}" (
    set "AISIDE={WATCH_SIDE}"
    set "FEEDSIDE={WATCH_FEED_SIDE}"
    set "WATCHOPTS={WATCH_JAVA_OPTS}"
    set "YOUSAY=You watch: the AI plays BOTH sides."
) else if /I "%AISIDE%"=="FP" (
    set "AISIDE=FP"
    set "FEEDSIDE=FP"
    set "YOUSAY=You play SHADOW."
    set "AIBUTTON=AI plays Free Peoples"
) else (
    set "AISIDE=SA"
    set "FEEDSIDE=SA"
    set "YOUSAY=You play FREE PEOPLES."
    set "AIBUTTON=AI plays Shadow"
)
echo.
echo    %YOUSAY%
echo.

{NET_MENU}
:: Fresh session.  An orphaned server or panel from a previous run
:: would be ADOPTED by the new client - previous brain and all - so
:: stop them and clear the coordination files first.  Only the server
:: and panel of THIS client dir are stopped (launch_head.py --stop-ours:
:: the process that holds this folder's alive byte, and its own panel):
:: never another program that merely names the folder.
echo [step1] >> "!LAUNCHLOG!"
echo   [1/3] Clearing any files left over from a previous session...
:: a game client still running on this play_client would be handed a
:: second server (or keep the old brain) - refuse until it is closed
powershell -NoProfile -Command "$p = $env:PLAY + '\'; $j = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' -and $_.CommandLine -like '*{JAR_NAME}*' -and $_.CommandLine.IndexOf($p, [StringComparison]::OrdinalIgnoreCase) -ge 0 }; if ($j) { exit 3 } else { exit 0 }" >nul 2>&1
if errorlevel 3 (
    echo.
    echo   [X] A game client from a previous session is still open on
    echo       !PLAY!
    echo       Close it, then run Play_vs_AI.bat again.
    echo.
    pause
    exit /b 1
)
{STOP_LINE}

cd /d "!PLAY!"
if not exist "logs\auto" mkdir "logs\auto"
if not exist "logs\archive" mkdir "logs\archive"
move /Y "logs\auto\*.log" "logs\archive\" >nul 2>&1
:: every leftover must actually go: a coordination file that survives
:: (locked by a process the sweep missed) would be adopted as THIS
:: session's heartbeat, so a failed delete aborts loudly
set "LEFTOVER="
for %%f in ({LEFTOVER_FILES}) do (
    if exist "%%f" del /F /Q "%%f" >nul 2>&1
    if exist "%%f" set "LEFTOVER=%%f"
)
if defined LEFTOVER (
    echo.
    echo   [X] Could not delete the leftover file %LEFTOVER%
    echo       in !PLAY!: another process still holds it.
    echo       Close any previous AI server, panel or client, then try again.
    echo.
    pause
    exit /b 1
)
for %%f in ({ROTATE_FILES}) do (
    if exist "%%f" move /Y "%%f" "%%f.prev" >nul 2>&1
)

:: ---------------------------------------------------------------------
:: 1. the AI server, FIRST.  The client adopts it through the
::    {HEARTBEAT_FILE} heartbeat ("spawn SKIPPED" in ai_boot.log).
:: ---------------------------------------------------------------------
echo [step2] >> "!LAUNCHLOG!"
echo   [2/3] Starting the AI server (AI side %AISIDE%, nets %NETTAG%)...
echo         The first start loads the nets; this can take a minute.
cd /d "!BRIDGE!"
if not "!JAVAGUARD!"=="armed" goto cleanup
{SERVER_LINE}

set /a TRIES=0
:waitbeat
if exist "!PLAY!\{HEARTBEAT_FILE}" goto beatok
set /a TRIES+=1
if %TRIES% GEQ {SERVER_GRACE_S} (
    "!PY!" "!BRIDGE!\launch_head.py" --server-runs "!PLAY!" >nul 2>&1
    if errorlevel 1 (
        echo.
        echo   [X] The AI server stopped before it was ready. Its last lines follow:
        echo.
        powershell -NoProfile -Command "Get-Content -Tail 20 '!PLAY!\{SERVER_CONSOLE_LOG}'"
        echo.
        pause
        goto cleanup
    )
)
if %TRIES% GEQ {HEARTBEAT_WAIT_S} (
    echo.
    echo   [X] No server heartbeat arrived within {HEARTBEAT_WAIT_S} seconds.
    echo       See: !PLAY!\{SERVER_CONSOLE_LOG}
    echo.
    pause
    goto cleanup
)
:: ping, not timeout: 'timeout' aborts outright when stdin is
:: redirected (a harness drives this file with a piped stdin)
ping -n 2 127.0.0.1 >nul 2>&1
goto waitbeat
:beatok
echo         The AI server is running.

:: ---------------------------------------------------------------------
:: 2. the client, in the FOREGROUND: closing the game window comes
::    back here and tears the server down.  -Dai.feed.aiside is
::    MANDATORY - it is what blinds the AI to your hand.  The client
::    spawns the AI status panel itself (-Dai.python / -Dai.bridgedir).
:: ---------------------------------------------------------------------
echo [step3] >> "!LAUNCHLOG!"
echo   [3/3] Launching the game client...
echo.
echo   ----------------------------------------------------------------
echo    IN THE GAME:
echo.
if "%AISIDE%"=="{WATCH_SIDE}" goto watchsteps
echo      1. Start a new game as usual (2nd edition, both sides).
echo      2. Menu  AI  ^>  "Play vs AI..."
echo      3. The side is already set (%AIBUTTON%): read
echo         the setup dialog and click Start Game.
echo.
echo    While playing: each turn the AI draws your cards and rolls your
echo    action dice for you, and it ends the turn. You roll your combat
echo    and Hunt dice when it asks. Answer each decision on the board by
echo    moving your pieces, or click the option in the AI panel, or type
echo    a word: "pass", "fp ok", "done", "nc", or an attack such as
echo    "dale from erebor". "back" puts the board back as the game has
echo    it, "undo" takes back your last decision, "help" lists the words.
goto stepsdone
:watchsteps
echo      1. No action is needed: the AI begins playing BOTH sides
echo         automatically, cards and dice included, on the game the
echo         client opens (the last game type played) once it has loaded.
echo      2. Keep that game: File ^> New or Load stops the watch, and
echo         watching again then requires a new Play_vs_AI.bat.
echo         If the watch did not start, use Menu  AI  ^>  "AI vs AI (watch)".
echo.
echo    While it plays, do not move pieces, draw cards or roll dice
echo    yourself: the AI does all of it. Menu  AI  ^>  "Stop AI"
echo    ends the watch.
:stepsdone
echo    Sound: menu  AI  ^>  "Mute sounds".
echo   ----------------------------------------------------------------
echo.
cd /d "!PLAY!"
if not "!JAVAGUARD!"=="armed" goto cleanup
set "JAVAGUARD=spent"
{JAVA_LINE}
{PAD}
goto cleanup

:usage
echo.
echo   Play_vs_AI.bat [SA^|FP] [net]   play on the real client
echo   Play_vs_AI.bat AUTO [fp] [sh]  watch AI vs AI on the real client, alias WATCH
echo   Play_Engine_Only.bat           engine board, no Java (UNTESTED)
echo   nets: Free Peoples {FP_KEYS_BAR_ESC}, Shadow {SH_KEYS_BAR_ESC} (default WIN)
echo   e.g.  Play_vs_AI.bat SA COR      Play_vs_AI.bat AUTO MIL COR
echo.
exit /b 0

:cleanup
echo.
echo   The client has closed: stopping the AI server and panel...
{STOP_LINE}
echo   Done.
ping -n 4 127.0.0.1 >nul 2>&1
endlocal
'''


_ENGINE_BAT = r'''@echo off
setlocal EnableExtensions DisableDelayedExpansion
:: the folder of this file is read while an exclamation mark is an ordinary
:: character (a folder name may hold one); afterwards every path is a !NAME!
set "HERE=%~dp0"
setlocal EnableDelayedExpansion
:: =====================================================================
:: Play_Engine_Only.bat -- the engine's OWN board GUI, no Java client.
:: GENERATED from wotr_bridge\launch_head.py (--write-bat); edit the
:: module, never this file (src\test_launch_head.py pins them).
::   Play_Engine_Only.bat [fp|sh|watch] [--sims N] [--seed N]
::   fp / sh = your side vs the AI; watch = AI vs AI.
:: UNTESTED: only the real-client path (Play_vs_AI.bat) has been
:: played and certified; this board has had no live testing at all.
:: =====================================================================
title AlphaWOTR - engine only (UNTESTED)

set "PLAY=!HERE!wotr_bridge\play_client"
set "BRIDGE=!HERE!wotr_bridge"

{PROBE}
:: certified-path env (CPU pinned; '-1', never an empty string) and the
:: generation switch WOTR_ACT_GEN={NET_GEN} (the loader refuses other generations)
{ENV_LINES}

echo.
echo   ================================================================
echo    AlphaWOTR - ENGINE-ONLY BOARD  (no Java client)
echo    THIS MODE IS UNTESTED.  The real-client path is the tested one.
echo   ================================================================
echo.
set "SIDE=%~1"
if "%SIDE%"=="" set "SIDE=fp"
if /I "%SIDE%"=="fp" goto guiok
if /I "%SIDE%"=="sh" goto guiok
if /I "%SIDE%"=="watch" goto guiok
echo   [X] The side must be fp, sh or watch. The value given was "%SIDE%".
pause
exit /b 1
:guiok
set "REST="
shift
:guirest
if "%~1"=="" goto guirun
set "REST=%REST% %1"
shift
goto guirest
:guirun
cd /d "!HERE!"
"!PY!" "!HERE!src\gui.py" --live %SIDE% --net {NET_TAG}%REST%
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" pause
endlocal & exit /b %RC%
'''

ENGINE_BAT_PATH = os.path.join(ROOT, 'Play_Engine_Only.bat')


def _render(template):
    txt = template
    for key, val in (('{PROBE}', _PROBE), ('{PROBE_IMPORTS}', _PROBE_IMPORTS),
                     ('{PY}', PY), ('{PYW}', PYW),
                     ('{JAVA_MIN}', str(JAVA_MIN)),
                     ('{ENV_LINES}', bat_env_lines()),
                     ('{NET_MENU}', bat_net_menu()),
                     ('{FP_KEYS_BAR}', '|'.join(k.upper() for k, _, _ in FP_NETS)),
                     ('{SH_KEYS_BAR}', '|'.join(k.upper() for k, _, _ in SH_NETS)),
                     ('{FP_KEYS_BAR_ESC}', '^|'.join(k.upper() for k, _, _ in FP_NETS)),
                     ('{SH_KEYS_BAR_ESC}', '^|'.join(k.upper() for k, _, _ in SH_NETS)),
                     ('{NET_TAG_PATTERN}', NET_TAG_FORMAT % (
                         '[%s]' % '/'.join(k for k, _, _ in FP_NETS),
                         '[%s]' % '/'.join(k for k, _, _ in SH_NETS))),
                     ('{NET_TAG}', NET_TAG), ('{NET_GEN}', NET_GEN),
                     ('{JAR_NAME}', JAR_NAME),
                     ('{LEFTOVER_FILES}', ' '.join(LEFTOVER_FILES)),
                     ('{ROTATE_FILES}', ' '.join(ROTATE_FILES)),
                     ('{HEARTBEAT_FILE}', HEARTBEAT_FILE),
                     ('{SERVER_CONSOLE_LOG}', SERVER_CONSOLE_LOG),
                     ('{HEARTBEAT_WAIT_S}', str(HEARTBEAT_WAIT_S)),
                     ('{SERVER_GRACE_S}', str(SERVER_GRACE_S)),
                     ('{SERVER_LINE}', bat_server_line()),
                     ('{STOP_LINE}', BAT_STOP_LINE),
                     ('{JAVA_LINE}', bat_java_line()),
                     ('{WATCH_SIDE}', WATCH_SIDE),
                     ('{WATCH_FEED_SIDE}', WATCH_FEED_SIDE),
                     ('{WATCH_JAVA_OPTS}', ' '.join(WATCH_JAVA_OPTS)),
                     ('{PAD}', PAD)):
        txt = txt.replace(key, val)
    bad = [c for c in txt if ord(c) > 127 or (ord(c) < 32 and c != '\n')]
    if bad:
        raise ValueError('render: non-ascii or CONTROL character in the '
                         'bat: %r' % bad[:5])
    _validate_paren_safety(txt)
    return txt.replace('\r\n', '\n')


def _validate_paren_safety(txt):
    """A ')' in the install folder (Windows' 'name (1)' duplicate
    downloads) terminates a parse-time %-expansion inside any ( ) block
    and ABORTS the batch (field failure 2026-08-23: every released bat
    died right after the banner in such folders).  Path variables must
    be delayed-expansion (!X!) everywhere; this refuses a render where
    a %-form or %~dp0 appears inside a parenthesized block."""
    import re
    pat = re.compile(r'%(HERE|PLAY|BRIDGE|LAUNCHLOG|PY|PYW|PY_PROBE|'
                     r'ALPHAWOTR_PYTHON|JAVA_HOME|JPATH)%|%~dp0')
    depth = 0
    for n, ln in enumerate(txt.split('\n'), 1):
        if depth > 0 and pat.search(ln):
            raise ValueError('render: parse-time path expansion inside a '
                             'parenthesized block (breaks in "(1)" '
                             'folders) at bat line %d: %s'
                             % (n, ln.strip()[:80]))
        depth += ln.count('(') - ln.count('^(')
        depth -= ln.count(')') - ln.count('^)')
        if depth < 0:
            depth = 0


def render_engine_bat():
    """The full text of Play_Engine_Only.bat (ascii, '\\n' between lines)."""
    return _render(_ENGINE_BAT)


def render_bat():
    """The full text of Play_vs_AI.bat (ascii, '\\n' between lines)."""
    return _render(_BAT)


def check_bat(path=BAT_PATH, renderer=None):
    """[] when the file on disk is exactly its renderer's text;
    otherwise a list of human-readable drift lines (first differing line
    quoted).  Default: Play_vs_AI.bat; check_engine_bat() for the
    engine-only bat."""
    want = (renderer or render_bat)().split('\n')
    try:
        with open(path, 'rb') as f:
            raw = f.read().decode('ascii', 'replace')
    except OSError as e:
        return ['cannot read %s: %r' % (path, e)]
    have = raw.replace('\r\n', '\n').split('\n')
    out = []
    if raw.count('\n') != raw.count('\r\n'):
        out.append('line ends are not CR LF: cmd misses labels in such a file')
    for i, (a, b) in enumerate(zip(want, have)):
        if a != b:
            out.append('line %d differs:\n  module: %s\n  file  : %s'
                       % (i + 1, a, b))
            break
    if not out and len(want) != len(have):
        out.append('length differs: module %d lines, file %d lines'
                   % (len(want), len(have)))
    return out


def check_engine_bat(path=ENGINE_BAT_PATH):
    return check_bat(path, render_engine_bat)


def write_bat(path=BAT_PATH, renderer=None):
    """Atomic whole-file write (the USER may be about to double-click
    it): render to <path>.tmp, then replace.  Written with CR LF line
    ends: in a file with bare LF cmd does not find a label whose line
    lies across a 512-byte block counted from the goto, so every edit
    of the text could break a jump."""
    data = (renderer or render_bat)().replace('\n', '\r\n').encode('ascii')
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        f.write(data)
    os.replace(tmp, path)
    return path


def write_engine_bat(path=ENGINE_BAT_PATH):
    return write_bat(path, render_engine_bat)


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    if a == ['--write-bat']:
        p = write_bat()
        print('wrote %s (%d bytes)' % (p, os.path.getsize(p)))
        p2 = write_engine_bat()
        print('wrote %s (%d bytes)' % (p2, os.path.getsize(p2)))
        return 0
    if a == ['--check']:
        rc = 0
        for name, fn in (('Play_vs_AI.bat', check_bat),
                         ('Play_Engine_Only.bat', check_engine_bat)):
            drift = fn()
            if drift:
                print('DRIFT: %s differs from launch_head:' % name)
                for d in drift:
                    print('  ' + d)
                rc = 1
        if rc == 0:
            print('OK: Play_vs_AI.bat == launch_head.render_bat(); '
                  'Play_Engine_Only.bat == render_engine_bat()')
        return rc
    if a[:1] == ['--stop-ours'] and len(a) == 2:
        print('stopped: %s' % (stop_ours(a[1]) or 'nothing'))
        return 0
    if a[:1] == ['--server-runs'] and len(a) == 2:
        return 0 if server_pid(a[1]) else 1
    if a[:1] == ['--print-java']:
        print(' '.join(java_cmd(a[1] if len(a) > 1 else PLAY_CLIENT,
                                a[2] if len(a) > 2 else 'SA')))
        return 0
    if a[:1] == ['--print-server']:
        print(' '.join(server_cmd(a[1] if len(a) > 1 else PLAY_CLIENT,
                                  a[2] if len(a) > 2 else 'SA')))
        return 0
    print('usage: launch_head.py --write-bat | --check | --stop-ours client_dir | '
          '--server-runs client_dir | '
          '--print-java [client_dir] [SA|FP|AUTO] | '
          '--print-server [client_dir] [SA|FP|AUTO]')
    return 2


if __name__ == '__main__':
    sys.exit(main())
