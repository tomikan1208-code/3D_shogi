@echo off
setlocal

cd /d "%~dp0.."
set "PYTHON=%CD%\.venv_gpu311\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=%CD%\.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
  echo [ERROR] Python not found.
  pause
  exit /b 1
)

echo [INFO] Running separated GPU env benchmark...
"%PYTHON%" gpu_env_rebuild\run_benchmark.py --device cuda --force-cuda --batch-size 256 --max-plies 512 --seconds 20 --sampling-mode fast --switch-after-steps 12
if errorlevel 1 (
  echo [ERROR] CUDA run failed. Check CUDA-enabled torch environment.
)

pause
