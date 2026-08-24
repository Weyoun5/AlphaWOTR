@echo off
setlocal EnableExtensions EnableDelayedExpansion
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
set "PLAY=!HERE!wotr_bridge\play_client"
set "BRIDGE=!HERE!wotr_bridge"

:: ---- which Python: the BUNDLED runtime (python\python.exe, shipped in
::      the release zip), then ALPHAWOTR_PYTHON, then python.exe on PATH,
::      then the developer runtime; the first that imports torch+numpy wins
set "PY="
if exist "!HERE!python\python.exe" (
    "!HERE!python\python.exe" -c "import torch, numpy" >nul 2>&1
    if not errorlevel 1 set "PY=!HERE!python\python.exe"
)
if not defined PY if defined ALPHAWOTR_PYTHON (
    "!ALPHAWOTR_PYTHON!" -c "import torch, numpy" >nul 2>&1
    if not errorlevel 1 set "PY=!ALPHAWOTR_PYTHON!"
)
if not defined PY (
    set "PY_PROBE="
    for /f "delims=" %%p in ('where python.exe 2^>nul') do (
        if not defined PY_PROBE set "PY_PROBE=%%p"
    )
    if defined PY_PROBE (
        "!PY_PROBE!" -c "import torch, numpy" >nul 2>&1
        if not errorlevel 1 set "PY=!PY_PROBE!"
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
    if exist "!HERE!python\python.exe" (
        echo       The bundled runtime python\python.exe exists but cannot import
        echo       torch+numpy.  Its error follows; send it with a bug report:
        echo       ----------------------------------------------------------
        "!HERE!python\python.exe" -c "import torch, numpy" 2>&1 | findstr /V /C:"^$"
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
set "PYW=!PY:python.exe=pythonw.exe!"
if not exist "!PYW!" set "PYW=!PY!"

:: certified-path env (CPU pinned; '-1', never an empty string) and the
:: v41 generation switch (the loader refuses the v41 nets without it)
set "CUDA_VISIBLE_DEVICES=-1"
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "PYTHONHASHSEED=0"
set "WOTR_ACT_GEN=v41"

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
cd /d "!HERE!"
"!PY!" "!HERE!src\gui.py" --live %SIDE% --net corpus_live%REST%
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" pause
endlocal & exit /b %RC%
