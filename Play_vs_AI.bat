@echo off
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
:: ---- which Python: the BUNDLED runtime (python\python.exe, shipped in
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
    if exist "C:\ProgramData\spyder-6\envs\spyder-runtime\python.exe" (
        "C:\ProgramData\spyder-6\envs\spyder-runtime\python.exe" -c "import torch, numpy" >nul 2>&1
        if not errorlevel 1 set "PY=C:\ProgramData\spyder-6\envs\spyder-runtime\python.exe"
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

:: certified-path env (CPU pinned; '-1', never an empty string) and the
:: v41 generation switch (the loader refuses the v41 nets without it)
set "CUDA_VISIBLE_DEVICES=-1"
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "PYTHONHASHSEED=0"
set "WOTR_ACT_GEN=v41"

if /I "%~1"=="help" goto usage
if "%~1"=="/?" goto usage

:: ---- play: the real client against the live stack -------------------
echo.
echo   ================================================================
echo    AlphaWOTR  --  human vs AI
echo   ================================================================
echo.

:: ---- which Java: JAVA_HOME first (the standard override), then PATH;
::      the client mods are Java 8 class files - an older runtime
::      dies with "class file versions up to 52.0" inside -javaagent
if defined JAVA_HOME if exist "%JAVA_HOME%in\java.exe" set "PATH=%JAVA_HOME%in;%PATH%"
where java >nul 2>&1
if errorlevel 1 (
    echo   [X] Java was not found on PATH.
    echo       Install Java 8 or newer ^(Temurin 17 or 21 from https://adoptium.net^),
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
if %JMAJN% LSS 8 (
    echo   [X] Java %JV% found at %JPATH%
    echo       AlphaWOTR needs Java 8 or newer.  Install Temurin 17 or 21 from
    echo       https://adoptium.net - and if an old Java stays first on PATH,
    echo       set JAVA_HOME to the new one ^(this launcher uses JAVA_HOME first^).
    echo.
    pause
    exit /b 1
)
echo   Java %JV%
if not exist "%PLAY%\WOTR-Gandalf-AI.jar" (
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
powershell -NoProfile -Command "$j = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' -and $_.CommandLine -like '*WOTR-Gandalf-AI.jar*' -and $_.CommandLine -like ('*' + $env:PLAY + '*') }; if ($j) { exit 3 } else { exit 0 }" >nul 2>&1
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
for %%f in (ai_bridge_status.json ai_bridge_request.log ai_bridge_move.txt ai_panel_control.json ai_feed.txt ai_feed.txt.meta.jsonl ai_client_board.txt ai_live_state.pkl ai_live_status.json ai_census.jsonl ai_server.lock ai_server.alive ai_panel.pid) do (
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
for %%f in (ai_boot.log ai_server_console.log ai_panel_proc.log ai_server.log) do (
    if exist "%%f" move /Y "%%f" "%%f.prev" >nul 2>&1
)

:: ---------------------------------------------------------------------
:: 1. the AI server, FIRST.  The client adopts it through the
::    ai_bridge_status.json heartbeat ("spawn SKIPPED" in ai_boot.log).
:: ---------------------------------------------------------------------
echo   [2/3] starting the AI server (AI side %AISIDE%, nets corpus_live)...
echo         first start loads the nets - this can take a minute.
cd /d "%BRIDGE%"
start "AlphaWOTR AI server" /min cmd /c ""%PY%" -u -m live.live_server --client-dir "%PLAY%" --side %AISIDE% --sims 300 --net-tag corpus_live 1>"%PLAY%\ai_server_console.log" 2>&1"

set /a TRIES=0
:waitbeat
if exist "%PLAY%\ai_bridge_status.json" goto beatok
findstr /C:"Traceback" "%PLAY%\ai_server_console.log" >nul 2>&1
if not errorlevel 1 (
    echo.
    echo   [X] the AI server crashed on startup.  Last lines:
    echo.
    powershell -NoProfile -Command "Get-Content -Tail 20 '%PLAY%\ai_server_console.log'"
    echo.
    pause
    goto cleanup
)
set /a TRIES+=1
if %TRIES% GEQ 240 (
    echo.
    echo   [X] no server heartbeat after 240s.
    echo       See: %PLAY%\ai_server_console.log
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
java -javaagent:AiFeed.jar="%PLAY%\ai_feed.txt" -Dai.feed.aiside=%AISIDE% -Dai.python="%PYW%" -Dai.bridgedir="%BRIDGE%" -Dai.net=corpus_live -Dai.pace=800 -jar WOTR-Gandalf-AI.jar
:: the game window closed - fall through to the cleanup below (this line is sized so that the jump sits at the byte where a running instance of the previous file resumes; launch_head.py BAT LAYOUT)
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
