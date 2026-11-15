@echo off
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
::   nets: Free Peoples WIN|MIL, Shadow WIN|COR (default WIN)
:: Client wotr_bridge\play_client; AI server FIRST (heartbeat), then
:: java adopts it; nets mc200_fp[win/mil]_sh[win/cor] (WOTR_ACT_GEN=v53).
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
:: ---- which Python: the BUNDLED runtime (python\python.exe, shipped in
::      the release zip), then ALPHAWOTR_PYTHON, then python.exe on PATH,
::      then the developer runtime; require all play dependencies, not only torch
set "PY="
if exist "!HERE!python\python.exe" (
    "!HERE!python\python.exe" -c "import torch, numpy, shapely, PIL, matplotlib, psutil, tkinter" >nul 2>&1
    if not errorlevel 1 set "PY=!HERE!python\python.exe"
)
if not defined PY if defined ALPHAWOTR_PYTHON (
    "!ALPHAWOTR_PYTHON!" -c "import torch, numpy, shapely, PIL, matplotlib, psutil, tkinter" >nul 2>&1
    if not errorlevel 1 set "PY=!ALPHAWOTR_PYTHON!"
)
if not defined PY (
    set "PY_PROBE="
    for /f "delims=" %%p in ('where python.exe 2^>nul') do (
        if not defined PY_PROBE set "PY_PROBE=%%p"
    )
    if defined PY_PROBE (
        "!PY_PROBE!" -c "import torch, numpy, shapely, PIL, matplotlib, psutil, tkinter" >nul 2>&1
        if not errorlevel 1 set "PY=!PY_PROBE!"
    )
)
if not defined PY (
    if exist "C:\ProgramData\spyder-6\envs\spyder-runtime\python.exe" (
        "C:\ProgramData\spyder-6\envs\spyder-runtime\python.exe" -c "import torch, numpy, shapely, PIL, matplotlib, psutil, tkinter" >nul 2>&1
        if not errorlevel 1 set "PY=C:\ProgramData\spyder-6\envs\spyder-runtime\python.exe"
    )
)
if not defined PY (
    echo   [X] No Python with all play dependencies found.
    if exist "!HERE!python\python.exe" (
        echo       The bundled runtime python\python.exe exists but cannot import
        echo       all play dependencies. Its error follows:
        echo       ----------------------------------------------------------
        "!HERE!python\python.exe" -c "import torch, numpy, shapely, PIL, matplotlib, psutil, tkinter" 2>&1 | findstr /V /C:"^$"
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

:: certified-path env (CPU pinned; '-1', never an empty string) and the
:: generation switch WOTR_ACT_GEN=v53 (the loader refuses other generations)
set "CUDA_VISIBLE_DEVICES=-1"
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "KMP_BLOCKTIME=0"
set "OMP_WAIT_POLICY=PASSIVE"
set "PYTHONHASHSEED=0"
set "WOTR_ACT_GEN=v53"
set "WOTR_REWARD_CONTRACT=predecision_profile_v1"
set "WOTR_NATIVE=0"
set "WOTR_LIVE_SEARCH_BACKEND=native"
set "WOTR_LIVE_INFERENCE_THREADS=1"
set "WOTR_LIVE_INFERENCE_CACHE=1"
set "WOTR_LIVE_POLICY_THREADS=8"
set "WOTR_FREE_CANDS=1"
set "WOTR_FREE_FORCE=12"

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
if not exist "!PLAY!\WOTR-Gandalf-AI.jar" (
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
if errorlevel 3 (set "AISIDE=AUTO") else if errorlevel 2 (set "AISIDE=FP") else (set "AISIDE=SA")
:sidechosen
set "WATCHOPTS="
if /I "%AISIDE%"=="WATCH" set "AISIDE=AUTO"
if /I "%AISIDE%"=="AUTO" (
    set "AISIDE=AUTO"
    set "FEEDSIDE=none"
    set "WATCHOPTS=-Dai.feed.blinding=off -Dai.side=AUTO -Dai.autoplay=1"
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

:: Which nets?  Each AI seat plays one of its side's final league
:: nets (launch_head FP_NETS / SH_NETS).  Asked only when the side was
:: asked; arguments skip it:  Play_vs_AI.bat SA COR  /  FP MIL  /
:: AUTO MIL COR (Free Peoples net first).  A side argument with no net
:: argument takes the WIN nets, so a harness leg never meets a prompt.
set "FPNET=win"
set "SHNET=win"
set "FPARG="
set "SHARG="
if /I "%AISIDE%"=="FP" set "FPARG=%~2"
if /I "%AISIDE%"=="SA" set "SHARG=%~2"
if /I "%AISIDE%"=="AUTO" set "FPARG=%~2"
if /I "%AISIDE%"=="AUTO" set "SHARG=%~3"
if defined FPARG goto netargs
if defined SHARG goto netargs
if not defined ASKNETS goto netschosen
if /I "%AISIDE%"=="SA" goto asksh
echo    Which Free Peoples AI?
echo.
echo      [1]  FP-WIN          search-trained (c008); own judge c1.5
echo      [2]  FP-MILITARY     military win, full network 2M games; own head c4
echo.
choice /c 12 /n /m "    Press 1 or 2: "
if errorlevel 2 (set "FPNET=mil") else (set "FPNET=win")
echo.
if /I "%AISIDE%"=="FP" goto netschosen
:asksh
echo    Which Shadow AI?
echo.
echo      [1]  SH-WIN          search-trained (c008); own judge c1.5
echo      [2]  SH-CORRUPTION   corruption, full network 2M games; own head c4
echo.
choice /c 12 /n /m "    Press 1 or 2: "
if errorlevel 2 (set "SHNET=cor") else (set "SHNET=win")
echo.
goto netschosen
:netargs
set "FPOK="
set "SHOK="
if not defined FPARG set "FPOK=1"
if /I "!FPARG!"=="win" set "FPNET=win"
if /I "!FPARG!"=="win" set "FPOK=1"
if /I "!FPARG!"=="mil" set "FPNET=mil"
if /I "!FPARG!"=="mil" set "FPOK=1"
if not defined SHARG set "SHOK=1"
if /I "!SHARG!"=="win" set "SHNET=win"
if /I "!SHARG!"=="win" set "SHOK=1"
if /I "!SHARG!"=="cor" set "SHNET=cor"
if /I "!SHARG!"=="cor" set "SHOK=1"
if not defined FPOK goto netbad
if not defined SHOK goto netbad
goto netschosen
:netbad
echo   [X] Unknown net in "!FPARG! !SHARG!".  Free Peoples nets: WIN or MIL;
echo       Shadow nets: WIN or COR.  Example: Play_vs_AI.bat AUTO MIL COR
echo.
pause
exit /b 1
:netschosen
set "NETTAG=mc200_fp!FPNET!_sh!SHNET!"
set "FPNAME=FP-WIN"
if /I "!FPNET!"=="mil" set "FPNAME=FP-MILITARY"
set "SHNAME=SH-WIN"
if /I "!SHNET!"=="cor" set "SHNAME=SH-CORRUPTION"
if /I "%AISIDE%"=="FP" set "AINAMES=!FPNAME!"
if /I "%AISIDE%"=="SA" set "AINAMES=!SHNAME!"
if /I "%AISIDE%"=="AUTO" set "AINAMES=!FPNAME! vs !SHNAME!"
echo    The AI plays !AINAMES! (net tag !NETTAG!).
echo.

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
powershell -NoProfile -Command "$p = $env:PLAY + '\'; $j = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' -and $_.CommandLine -like '*WOTR-Gandalf-AI.jar*' -and $_.CommandLine.IndexOf($p, [StringComparison]::OrdinalIgnoreCase) -ge 0 }; if ($j) { exit 3 } else { exit 0 }" >nul 2>&1
if errorlevel 3 (
    echo.
    echo   [X] A game client from a previous session is still open on
    echo       !PLAY!
    echo       Close it, then run Play_vs_AI.bat again.
    echo.
    pause
    exit /b 1
)
"!PY!" "!BRIDGE!\launch_head.py" --stop-ours "!PLAY!" >nul 2>&1

cd /d "!PLAY!"
if not exist "logs\auto" mkdir "logs\auto"
if not exist "logs\archive" mkdir "logs\archive"
move /Y "logs\auto\*.log" "logs\archive\" >nul 2>&1
:: every leftover must actually go: a coordination file that survives
:: (locked by a process the sweep missed) would be adopted as THIS
:: session's heartbeat, so a failed delete aborts loudly
set "LEFTOVER="
for %%f in (ai_bridge_status.json ai_bridge_request.log ai_bridge_move.txt ai_panel_control.json ai_feed.txt ai_feed.txt.meta.jsonl ai_client_board.txt ai_live_state.pkl ai_live_status.json ai_census.jsonl ai_panel_request.json.taken ai_server.lock ai_server.alive ai_panel.pid ai_inject_ack.jsonl ai_server.stop ai_panel_view.json ai_panel_request.json ai_panel_ack.jsonl) do (
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
for %%f in (ai_boot.log ai_server_console.log ai_panel_proc.log ai_server.log core_log.jsonl) do (
    if exist "%%f" move /Y "%%f" "%%f.prev" >nul 2>&1
)

:: ---------------------------------------------------------------------
:: 1. the AI server, FIRST.  The client adopts it through the
::    ai_bridge_status.json heartbeat ("spawn SKIPPED" in ai_boot.log).
:: ---------------------------------------------------------------------
echo [step2] >> "!LAUNCHLOG!"
echo   [2/3] Starting the AI server (AI side %AISIDE%, nets %NETTAG%)...
echo         The first start loads the nets; this can take a minute.
cd /d "!BRIDGE!"
if not "!JAVAGUARD!"=="armed" goto cleanup
start "AlphaWOTR AI server" /min cmd /c ""!PY!" -u -m core.server --client-dir "!PLAY!" --side %AISIDE% --sims 300 --net-tag %NETTAG% 1>"!PLAY!\ai_server_console.log" 2>&1"

set /a TRIES=0
:waitbeat
if exist "!PLAY!\ai_bridge_status.json" goto beatok
set /a TRIES+=1
if %TRIES% GEQ 30 (
    "!PY!" "!BRIDGE!\launch_head.py" --server-runs "!PLAY!" >nul 2>&1
    if errorlevel 1 (
        echo.
        echo   [X] The AI server stopped before it was ready. Its last lines follow:
        echo.
        powershell -NoProfile -Command "Get-Content -Tail 20 '!PLAY!\ai_server_console.log'"
        echo.
        pause
        goto cleanup
    )
)
if %TRIES% GEQ 240 (
    echo.
    echo   [X] No server heartbeat arrived within 240 seconds.
    echo       See: !PLAY!\ai_server_console.log
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
if "%AISIDE%"=="AUTO" goto watchsteps
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
java -javaagent:AiFeed.jar="!PLAY!\ai_feed.txt" -Dai.feed.aiside=%FEEDSIDE% %WATCHOPTS% -Dai.python="!PYW!" -Dai.bridgedir="!BRIDGE!" -Dai.net=%NETTAG% -Dai.pace=800 -jar WOTR-Gandalf-AI.jar
:: the game window closed - fall through to the cleanup below ----------------------------------------------------------------------------------------------------------------------------------------
goto cleanup

:usage
echo.
echo   Play_vs_AI.bat [SA^|FP] [net]   play on the real client
echo   Play_vs_AI.bat AUTO [fp] [sh]  watch AI vs AI on the real client, alias WATCH
echo   Play_Engine_Only.bat           engine board, no Java (UNTESTED)
echo   nets: Free Peoples WIN^|MIL, Shadow WIN^|COR (default WIN)
echo   e.g.  Play_vs_AI.bat SA COR      Play_vs_AI.bat AUTO MIL COR
echo.
exit /b 0

:cleanup
echo.
echo   The client has closed: stopping the AI server and panel...
"!PY!" "!BRIDGE!\launch_head.py" --stop-ours "!PLAY!" >nul 2>&1
echo   Done.
ping -n 4 127.0.0.1 >nul 2>&1
endlocal
