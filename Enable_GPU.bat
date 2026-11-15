@echo off
rem Optional: the GPU build of torch for the game's own Python (NVIDIA only).
rem   Enable_GPU.bat            install it (about 2 GB to download)
rem   Enable_GPU.bat disable    back to the CPU build
rem   Enable_GPU.bat status     what is installed, and whether it sees a GPU
rem What it does and when it refuses: wotr_bridge\gpu_torch.py
setlocal
set "HERE=%~dp0"
set "ACT=%*"
if "%~1"=="" set "ACT=enable"
if not exist "%HERE%python\python.exe" (
    echo   This uses the release zip's own python\ folder, which is not here.
    echo   With your own Python:  python wotr_bridge\gpu_torch.py enable
    pause
    exit /b 1
)
"%HERE%python\python.exe" -B "%HERE%wotr_bridge\gpu_torch.py" %ACT%
echo.
pause
