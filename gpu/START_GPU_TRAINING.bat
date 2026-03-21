@echo off
setlocal

cd /d "%~dp0.."
set "GPU_VENV=%CD%\.venv_gpu311"
set "PYTHON=%GPU_VENV%\Scripts\python.exe"

if not exist "%PYTHON%" (
  echo [ERROR] GPU venv not found: %GPU_VENV%
  echo [INFO] Run gpu\INSTALL_GPU_PY311_ENV.bat first.
  pause
  exit /b 1
)

echo [INFO] Checking CUDA availability...
"%PYTHON%" gpu\gpu_precheck.py
if errorlevel 1 (
  echo [ERROR] CUDA is not available. Please install CUDA-enabled PyTorch.
  pause
  exit /b 1
)

set "SHOGI_DEVICE=cuda"
set "SHOGI_FORCE_CUDA=1"
if "%SHOGI_N_ENVS%"=="" set "SHOGI_N_ENVS=6"
if "%SHOGI_N_STEPS%"=="" set "SHOGI_N_STEPS=256"
if "%SHOGI_OPENING_RANDOM_PLIES%"=="" set "SHOGI_OPENING_RANDOM_PLIES=2"

echo [INFO] Starting GPU training...
"%PYTHON%" ai_training\train.py

echo [INFO] Training process ended.
pause
