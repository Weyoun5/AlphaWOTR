# -*- coding: ascii -*-
"""launch_head.py - THE launch facts of the one head (Play_vs_AI.bat).

ONE IMPLEMENTATION PER QUANTITY.  Every fact about how a human-vs-AI
session is launched lives here and nowhere else:

  * HEAD_ENV        the environment every mode sets (CPU pins, the net
                    generation switch);
  * SERVER_TOKENS   the AI server command line (python -m live.live_server
                    ..., cwd = wotr_bridge);
  * JAVA_TOKENS     the client command line (java with the AiFeed agent
                    and the -D flags, cwd = the client dir);
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
line followed by `goto cleanup`; the gui/console sections sit between
that jump and the `:cleanup` label.  cmd.exe resumes a running batch
file at a byte offset after the foreground java exits, so a regenerated
file must keep a line boundary followed by `goto cleanup` at the offset
where the previous file's java line ended (PAD below is sized for
that); the jump rescans labels and lands on the cleanup wherever it is.

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

NET_TAG = 'corpus_live'
SERVER_SIMS = 300
JAVA_PACE = 800
JAR_NAME = 'WOTR-Gandalf-AI.jar'
FEED_JAR = 'AiFeed.jar'
FEED_NAME = 'ai_feed.txt'
HEARTBEAT_FILE = 'ai_bridge_status.json'
SERVER_CONSOLE_LOG = 'ai_server_console.log'
HEARTBEAT_WAIT_S = 240

# the environment every mode sets ('-1', never an empty string: torch
# reports cuda with 0 devices on '')
HEAD_ENV = (
    ('CUDA_VISIBLE_DEVICES', '-1'),
    ('KMP_DUPLICATE_LIB_OK', 'TRUE'),
    ('PYTHONHASHSEED', '0'),
    ('WOTR_ACT_GEN', 'v41'),
)

# coordination files a new session must not inherit (a survivor would
# be adopted as THIS session's heartbeat / control / board)
LEFTOVER_FILES = (
    'ai_bridge_status.json', 'ai_bridge_request.log',
    'ai_bridge_move.txt', 'ai_panel_control.json', 'ai_feed.txt',
    'ai_feed.txt.meta.jsonl', 'ai_client_board.txt',
    'ai_live_state.pkl', 'ai_live_status.json', 'ai_census.jsonl',
    'ai_server.lock', 'ai_server.alive', 'ai_panel.pid',
)
# diagnostics of the previous session: kept once as <name>.prev
ROTATE_FILES = ('ai_boot.log', 'ai_server_console.log',
                'ai_panel_proc.log', 'ai_server.log')

# the AI server: cwd = wotr_bridge (the package is `live`, never
# `wotr_bridge.live` - see src\test_live_prodimports.py)
SERVER_TOKENS = ('{py}', '-u', '-m', 'live.live_server',
                 '--client-dir', '{client}', '--side', '{side}',
                 '--sims', '{sims}', '--net-tag', NET_TAG)

# the client: cwd = the client dir (resources load from the cwd).
# -Dai.feed.aiside is what blinds the AI to the human's hand; the
# client reads it and asks no side question.
# the client mods + AiFeed agent are compiled --release 8 (2026-08-23:
# testers on Java 8 / 17 died on the Java-21 class files; the ownership
# guard's Java-9 ProcessHandle is behind reflection with a file-lock
# proof fallback, so the original client's own Java 8 is enough)
JAVA_MIN = 8

JAVA_TOKENS = ('java',
               '-javaagent:' + FEED_JAR + '={client}\\' + FEED_NAME,
               '-Dai.feed.aiside={side}',
               '-Dai.python={pyw}',
               '-Dai.bridgedir={bridge}',
               '-Dai.net=' + NET_TAG,
               '-Dai.pace=%d' % JAVA_PACE,
               '-jar', JAR_NAME)

# tokens whose substituted value is a path (quoted in the bat)
_PATH_TOKENS = ('{py}', '{client}', '{pyw}', '{bridge}')


def head_env(base=None):
    """The launch environment: `base` (default os.environ) + HEAD_ENV."""
    env = dict(os.environ if base is None else base)
    for k, v in HEAD_ENV:
        env[k] = v
    return env


def server_cmd(client_dir, ai_side, sims=SERVER_SIMS, py=PY, extra=()):
    """argv of the AI server for `client_dir`; run it with
    cwd=BRIDGE_DIR and env=head_env().  `extra` appends e.g.
    ('--seed', '42') or ('--scenario', dir)."""
    ai_side = _side(ai_side)
    out = [t.format(py=py, client=client_dir, side=ai_side, sims=sims)
           for t in SERVER_TOKENS]
    return out + [str(x) for x in extra]


def java_cmd(client_dir, ai_side, pythonw=PYW, bridge_dir=BRIDGE_DIR,
             extra_java_opts=()):
    """argv of the client for `client_dir`; run it with cwd=client_dir.
    `extra_java_opts` (harness agents / -D flags) go before -jar."""
    ai_side = _side(ai_side)
    toks = [t.format(client=client_dir, side=ai_side, pyw=pythonw,
                     bridge=bridge_dir) for t in JAVA_TOKENS]
    i = toks.index('-jar')
    return toks[:i] + [str(x) for x in extra_java_opts] + toks[i:]


def _side(ai_side):
    s = str(ai_side).upper()
    if s not in ('SA', 'FP'):
        raise ValueError('ai_side must be SA or FP, got %r' % (ai_side,))
    return s


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
    return ' '.join(_bat_tokens(JAVA_TOKENS, client='%PLAY%',
                                side='%AISIDE%', pyw='%PYW%',
                                bridge='%BRIDGE%'))


def bat_server_line():
    inner = ' '.join(_bat_tokens(SERVER_TOKENS, py='%PY%',
                                 client='%PLAY%', side='%AISIDE%',
                                 sims=SERVER_SIMS))
    return ('start "AlphaWOTR AI server" /min cmd /c "' + inner
            + ' 1>"%PLAY%\\' + SERVER_CONSOLE_LOG + '" 2>&1"')


def bat_env_lines():
    return '\n'.join('set "%s=%s"' % kv for kv in HEAD_ENV)


# the comment line between the java line and `goto cleanup` (see the
# BAT LAYOUT INVARIANT above); its length places the jump
PAD = (':: the game window closed - fall through to the cleanup below '
       '(this line is sized so that the jump sits at the byte where a '
       'running instance of the previous file resumes; launch_head.py '
       'BAT LAYOUT)')

_PROBE = r''':: ---- which Python: the BUNDLED runtime (python\python.exe, shipped in
::      the release zip), then ALPHAWOTR_PYTHON, then python.exe on PATH,
::      then the developer runtime; the first that imports torch+numpy wins
set "PY="
if exist "%~dp0python\python.exe" (
    "%~dp0python\python.exe" -c "import torch, numpy" >nul 2>&1
    if not errorlevel 1 set "PY=%~dp0python\python.exe"
)
if not defined PY if defined ALPHAWOTR_PYTHON (
    "%ALPHAWOTR_PYTHON%" -c "import torch, numpy" >nul 2>&1
    if not errorlevel 1 set "PY=%ALPHAWOTR_PYTHON%"
)
if not defined PY (
    set "PY_PROBE="
    for /f "delims=" %%p in ('where python.exe 2^>nul') do (
        if not defined PY_PROBE set "PY_PROBE=%%p"
    )
    if defined PY_PROBE (
        "%PY_PROBE%" -c "import torch, numpy" >nul 2>&1
        if not errorlevel 1 set "PY=%PY_PROBE%"
    )
)
if not defined PY (
    if exist "{PY}" (
        "{PY}" -c "import torch, numpy" >nul 2>&1
        if not errorlevel 1 set "PY={PY}"
    )
)
if not defined PY (
    echo   [X] No working Python found.
    if exist "%~dp0python\python.exe" (
        echo       The bundled runtime python\python.exe exists but cannot import
        echo       torch+numpy.  Its error follows; send it with a bug report:
        echo       ----------------------------------------------------------
        "%~dp0python\python.exe" -c "import torch, numpy" 2>&1 | findstr /V /C:"^$"
        echo       ----------------------------------------------------------
        echo       Usual causes: the zip was not fully unpacked ^(use Explorer or
        echo       7-Zip, to a SHORT path like C:\AlphaWOTR^), or a DLL load error
        echo       ^(try: install the Microsoft Visual C++ Redistributable x64^).
    ) else (
        echo       No python\ runtime in this folder ^(it ships in the release
        echo       zip^).  Install Python 3.10+ and run
        echo       pip install -r requirements.txt   ^(or set ALPHAWOTR_PYTHON^)
    )
    echo.
    pause
    exit /b 1
)
set "PYW=%PY:python.exe=pythonw.exe%"
if not exist "%PYW%" set "PYW=%PY%"
'''

_BAT = r'''@echo off
setlocal EnableExtensions
:: =====================================================================
:: Play_vs_AI.bat -- THE ONE HEAD of AlphaWOTR.  GENERATED from
:: wotr_bridge\launch_head.py (--write-bat); edit the module, never
:: this file (src\test_launch_head.py pins the two together).
::   Play_vs_AI.bat [SA|FP]      play on the real client (SA = the AI
::                               is Shadow, FP = the AI is Free Peoples)
::   Play_Engine_Only.bat        the engine's own board, no Java (UNTESTED)
:: Client wotr_bridge\play_client; AI server FIRST (heartbeat), then
:: java adopts it; nets corpus_live = the v41 pair (WOTR_ACT_GEN=v41).
:: =====================================================================
title AlphaWOTR - Play vs AI

set "HERE=%~dp0"
set "PLAY=%HERE%wotr_bridge\play_client"
set "BRIDGE=%HERE%wotr_bridge"
{PROBE}
:: certified-path env (CPU pinned; '-1', never an empty string) and the
:: v41 generation switch (the loader refuses the v41 nets without it)
{ENV_LINES}

if /I "%~1"=="help" goto usage
if "%~1"=="/?" goto usage

:: ---- play: the real client against the live stack -------------------
echo.
echo   ================================================================
echo    AlphaWOTR  --  human vs AI
echo   ================================================================
echo.

:: ---- which Java: JAVA_HOME first (the standard override), then PATH;
::      the client mods are Java {JAVA_MIN} class files - an older runtime
::      dies with "class file versions up to 52.0" inside -javaagent
if defined JAVA_HOME if exist "%JAVA_HOME%in\java.exe" set "PATH=%JAVA_HOME%in;%PATH%"
where java >nul 2>&1
if errorlevel 1 (
    echo   [X] Java was not found on PATH.
    echo       Install Java {JAVA_MIN} or newer ^(Temurin 17 or 21 from https://adoptium.net^),
    echo       or set JAVA_HOME to an installed JDK/JRE folder.
    echo.
    pause
    exit /b 1
)
set "JV="
for /f tokens^=3 %%v in ('java -version 2^>^&1 ^| findstr /i "version"') do if not defined JV set "JV=%%~v"
set "JMAJ="
for /f "tokens=1,2 delims=." %%a in ("%JV%") do (set "JMAJ=%%a" & set "JMIN=%%b")
if "%JMAJ%"=="1" set "JMAJ=%JMIN%"
set /a JMAJN=%JMAJ% 2>nul
if not defined JMAJN set "JMAJN=0"
set "JPATH="
for /f "delims=" %%p in ('where java') do if not defined JPATH set "JPATH=%%p"
if %JMAJN% LSS {JAVA_MIN} (
    echo   [X] Java %JV% found at %JPATH%
    echo       AlphaWOTR needs Java {JAVA_MIN} or newer.  Install Temurin 17 or 21 from
    echo       https://adoptium.net - and if an old Java stays first on PATH,
    echo       set JAVA_HOME to the new one ^(this launcher uses JAVA_HOME first^).
    echo.
    pause
    exit /b 1
)
echo   Java %JV%
if not exist "%PLAY%\{JAR_NAME}" (
    echo   [X] Play client missing: %PLAY%
    echo.
    pause
    exit /b 1
)

:: Which side do YOU want?  Asked ONCE, before java starts: it sets the
:: server's --side AND -Dai.feed.aiside (the flag that hides your hand
:: from the AI); the client reads that flag and does not ask again; the
:: server refuses a session where the two disagree.
:: An argument skips the prompt:  Play_vs_AI.bat SA   (AI plays Shadow)
::                                Play_vs_AI.bat FP   (AI plays Free Peoples)
set "AISIDE=%~1"
if /I "%AISIDE%"=="SA" goto sidechosen
if /I "%AISIDE%"=="FP" goto sidechosen
echo    Which side do you want to play?
echo.
echo      [1]  FREE PEOPLES   (the AI plays Shadow)
echo      [2]  SHADOW         (the AI plays Free Peoples)
echo.
choice /c 12 /n /m "    Press 1 or 2: "
if errorlevel 2 (set "AISIDE=FP") else (set "AISIDE=SA")
:sidechosen
if /I "%AISIDE%"=="FP" (
    set "YOUPLAY=SHADOW"
    set "AIBUTTON=AI plays Free Peoples"
) else (
    set "AISIDE=SA"
    set "YOUPLAY=FREE PEOPLES"
    set "AIBUTTON=AI plays Shadow"
)
echo.
echo    You play %YOUPLAY%.
echo.

:: Fresh session.  An orphaned server or panel from a previous run
:: would be ADOPTED by the new client - previous brain and all - so
:: kill them and clear the coordination files first.  Every sweep is
:: scoped to THIS client dir (%PLAY%): a harness leg on a scratch copy
:: is never in range, and vice versa.
echo   [1/3] clearing any leftovers from a previous session...
:: a game client still running on this play_client would be handed a
:: second server (or keep the old brain) - refuse until it is closed
powershell -NoProfile -Command "$j = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' -and $_.CommandLine -like '*{JAR_NAME}*' -and $_.CommandLine -like ('*' + $env:PLAY + '*') }; if ($j) { exit 3 } else { exit 0 }" >nul 2>&1
if errorlevel 3 (
    echo.
    echo   [X] a game client from a previous session is still open on
    echo       %PLAY%
    echo       Close it, then run Play_vs_AI.bat again.
    echo.
    pause
    exit /b 1
)
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like ('*' + $env:PLAY + '*') -and ($_.Name -match 'python') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" >nul 2>&1

cd /d "%PLAY%"
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
    echo   [X] could not delete the leftover file %LEFTOVER%
    echo       in %PLAY% - something still holds it.
    echo       Close any previous AI server / panel / client and retry.
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
echo   [2/3] starting the AI server (AI side %AISIDE%, nets {NET_TAG})...
echo         first start loads the nets - this can take a minute.
cd /d "%BRIDGE%"
{SERVER_LINE}

set /a TRIES=0
:waitbeat
if exist "%PLAY%\{HEARTBEAT_FILE}" goto beatok
findstr /C:"Traceback" "%PLAY%\{SERVER_CONSOLE_LOG}" >nul 2>&1
if not errorlevel 1 (
    echo.
    echo   [X] the AI server crashed on startup.  Last lines:
    echo.
    powershell -NoProfile -Command "Get-Content -Tail 20 '%PLAY%\{SERVER_CONSOLE_LOG}'"
    echo.
    pause
    goto cleanup
)
set /a TRIES+=1
if %TRIES% GEQ {HEARTBEAT_WAIT_S} (
    echo.
    echo   [X] no server heartbeat after {HEARTBEAT_WAIT_S}s.
    echo       See: %PLAY%\{SERVER_CONSOLE_LOG}
    echo.
    pause
    goto cleanup
)
:: ping, not timeout: 'timeout' aborts outright when stdin is
:: redirected (a harness drives this file with a piped stdin)
ping -n 2 127.0.0.1 >nul 2>&1
goto waitbeat
:beatok
echo         server up.

:: ---------------------------------------------------------------------
:: 2. the client, in the FOREGROUND: closing the game window comes
::    back here and tears the server down.  -Dai.feed.aiside is
::    MANDATORY - it is what blinds the AI to your hand.  The client
::    spawns the AI status panel itself (-Dai.python / -Dai.bridgedir).
:: ---------------------------------------------------------------------
echo   [3/3] launching the game client...
echo.
echo   ----------------------------------------------------------------
echo    IN THE GAME:
echo.
echo      1. Start a new game as usual (2nd edition, both sides).
echo      2. Menu  AI  ^>  "Play vs AI..."
echo      3. The side is already set (%AIBUTTON%) - just read
echo         the setup dialog and click Start Game.
echo.
echo    While playing: drag your own pieces, draw your own cards, roll
echo    your own dice - then KILL the spent die in Current Action to
echo    finish each action.  Type 'pass' to pass, 'resync' if the board
echo    ever looks wrong.  Move the Turn Marker at the end of a turn.
echo    Chat words (attacks, combat cards): see README.txt.
echo    Sound: menu  AI  ^>  "Mute sounds".
echo   ----------------------------------------------------------------
echo.
cd /d "%PLAY%"
{JAVA_LINE}
{PAD}
goto cleanup

:usage
echo.
echo   Play_vs_AI.bat [SA^|FP]      play on the real client
echo   Play_Engine_Only.bat        engine board, no Java (UNTESTED)
echo.
exit /b 0

:cleanup
echo.
echo   client closed - stopping the AI server and panel...
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like ('*' + $env:PLAY + '*') -and ($_.Name -match 'python') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" >nul 2>&1
echo   done.
ping -n 4 127.0.0.1 >nul 2>&1
endlocal
'''


_ENGINE_BAT = r'''@echo off
setlocal EnableExtensions
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

set "HERE=%~dp0"
set "PLAY=%HERE%wotr_bridge\play_client"
set "BRIDGE=%HERE%wotr_bridge"

{PROBE}
:: certified-path env (CPU pinned; '-1', never an empty string) and the
:: v41 generation switch (the loader refuses the v41 nets without it)
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
echo   [X] side must be fp, sh or watch - got "%SIDE%"
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
cd /d "%HERE%"
"%PY%" "%HERE%src\gui.py" --live %SIDE% --net {NET_TAG}%REST%
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" pause
endlocal & exit /b %RC%
'''

ENGINE_BAT_PATH = os.path.join(ROOT, 'Play_Engine_Only.bat')


def _render(template):
    txt = template
    for key, val in (('{PROBE}', _PROBE), ('{PY}', PY), ('{PYW}', PYW),
                     ('{JAVA_MIN}', str(JAVA_MIN)),
                     ('{ENV_LINES}', bat_env_lines()),
                     ('{NET_TAG}', NET_TAG), ('{JAR_NAME}', JAR_NAME),
                     ('{LEFTOVER_FILES}', ' '.join(LEFTOVER_FILES)),
                     ('{ROTATE_FILES}', ' '.join(ROTATE_FILES)),
                     ('{HEARTBEAT_FILE}', HEARTBEAT_FILE),
                     ('{SERVER_CONSOLE_LOG}', SERVER_CONSOLE_LOG),
                     ('{HEARTBEAT_WAIT_S}', str(HEARTBEAT_WAIT_S)),
                     ('{SERVER_LINE}', bat_server_line()),
                     ('{JAVA_LINE}', bat_java_line()),
                     ('{PAD}', PAD)):
        txt = txt.replace(key, val)
    if any(ord(c) > 127 for c in txt):
        raise ValueError('render: non-ascii character in the bat')
    return txt.replace('\r\n', '\n')


def render_engine_bat():
    """The full text of Play_Engine_Only.bat (LF line endings, ascii)."""
    return _render(_ENGINE_BAT)


def render_bat():
    """The full text of Play_vs_AI.bat (LF line endings, ascii)."""
    return _render(_BAT)


def goto_cleanup_offset(text=None):
    """Byte offset of the `goto cleanup` line that follows the java
    line (the resume point a running instance must find)."""
    txt = render_bat() if text is None else text
    i = txt.index(bat_java_line() + '\n')
    j = txt.index('\ngoto cleanup\n', i) + 1
    return len(txt[:j].encode('ascii'))


def check_bat(path=BAT_PATH, renderer=None):
    """[] when the file on disk is exactly its renderer's text;
    otherwise a list of human-readable drift lines (first differing line
    quoted).  Default: Play_vs_AI.bat; check_engine_bat() for the
    engine-only bat."""
    want = (renderer or render_bat)().split('\n')
    try:
        with open(path, 'rb') as f:
            have = f.read().decode('ascii', 'replace').replace(
                '\r\n', '\n').split('\n')
    except OSError as e:
        return ['cannot read %s: %r' % (path, e)]
    out = []
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
    it): render to <path>.tmp, then replace."""
    data = (renderer or render_bat)().encode('ascii')
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
        print('wrote %s (%d bytes; goto cleanup at byte %d)'
              % (p, os.path.getsize(p), goto_cleanup_offset()))
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
    if a[:1] == ['--print-java']:
        print(' '.join(java_cmd(a[1] if len(a) > 1 else PLAY_CLIENT,
                                a[2] if len(a) > 2 else 'SA')))
        return 0
    if a[:1] == ['--print-server']:
        print(' '.join(server_cmd(a[1] if len(a) > 1 else PLAY_CLIENT,
                                  a[2] if len(a) > 2 else 'SA')))
        return 0
    print('usage: launch_head.py --write-bat | --check | '
          '--print-java [client_dir] [SA|FP] | '
          '--print-server [client_dir] [SA|FP]')
    return 2


if __name__ == '__main__':
    sys.exit(main())
