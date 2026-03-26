@echo off
setlocal

cd /d %~dp0

if exist ".venv\Scripts\activate.bat" (
  call ".venv\Scripts\activate.bat"
) else (
  echo [WARN] .venv\Scripts\activate.bat not found. Running with system Python.
)

if exist ".\ai_training\build\lib.win-amd64-cpython-313\shogi3d_core.cp313-win_amd64.pyd" (
  copy /Y ".\ai_training\build\lib.win-amd64-cpython-313\shogi3d_core.cp313-win_amd64.pyd" ".\ai_training\shogi3d_core.cp313-win_amd64.pyd" >nul 2>nul
  if errorlevel 1 (
    echo [WARN] Could not update shogi3d_core.cp313-win_amd64.pyd ^(file may be in use^)
  ) else (
    echo [INFO] Updated shogi3d_core.cp313-win_amd64.pyd from build output
  )
)

if "%SHOGI_MODEL_PATH%"=="" set SHOGI_MODEL_PATH=./ai_training/models/shogi3d_ppo_latest.zip
if "%SHOGI_EVAL_GAMES%"=="" set SHOGI_EVAL_GAMES=100
if "%SHOGI_OPENING_RANDOM_PLIES%"=="" set SHOGI_OPENING_RANDOM_PLIES=2
if "%SHOGI_EVAL_WORKERS%"=="" set SHOGI_EVAL_WORKERS=1
if "%SHOGI_EVAL_FAST%"=="" set SHOGI_EVAL_FAST=1
if /I "%SHOGI_EVAL_FAST%"=="1" (
  set "SHOGI_USE_CPP_KINGCHECK=1"
  set "SHOGI_USE_CPP_LEGAL_MOVES=1"
  set "SHOGI_CPP_STRICT_FILTER=0"
  set "SHOGI_DETERMINISTIC_MOVE_ORDER=1"
  set "SHOGI_FAST_DROP_GENERATION=1"
  set "SHOGI_USE_LEGAL_MOVES_CACHE=1"
  set "SHOGI_FIXED_ACTION_SPACE=1"
  if "%SHOGI_EVAL_DEVICE%"=="" set SHOGI_EVAL_DEVICE=cuda
) else (
  if "%SHOGI_EVAL_DEVICE%"=="" set SHOGI_EVAL_DEVICE=cpu
)
if "%SHOGI_EVAL_LIVE_INTERVAL_SEC%"=="" set SHOGI_EVAL_LIVE_INTERVAL_SEC=0.25
if "%SHOGI_EVAL_LOG_INTERVAL_SEC%"=="" set SHOGI_EVAL_LOG_INTERVAL_SEC=0
set PYTHONUNBUFFERED=1
if "%OMP_NUM_THREADS%"=="" set OMP_NUM_THREADS=1
if /I "%SHOGI_EVAL_DEVICE%"=="cpu" set CUDA_VISIBLE_DEVICES=-1
for %%I in ("%SHOGI_MODEL_PATH%") do set "SHOGI_MODEL_NAME=%%~nxI"

echo Model: %SHOGI_MODEL_PATH%
echo Model name: %SHOGI_MODEL_NAME%
echo Games: %SHOGI_EVAL_GAMES%
echo Opening random plies: %SHOGI_OPENING_RANDOM_PLIES%
echo Workers: %SHOGI_EVAL_WORKERS%
echo Device: %SHOGI_EVAL_DEVICE%
echo Fast mode: %SHOGI_EVAL_FAST%
echo Live interval sec: %SHOGI_EVAL_LIVE_INTERVAL_SEC%
echo Periodic log sec: %SHOGI_EVAL_LOG_INTERVAL_SEC%
echo.

python -u ai_training\eval_vs_random.py --model "%SHOGI_MODEL_PATH%" --games %SHOGI_EVAL_GAMES% --opening-random-plies %SHOGI_OPENING_RANDOM_PLIES% --workers %SHOGI_EVAL_WORKERS% --device %SHOGI_EVAL_DEVICE%

echo.
echo Evaluation finished. Press any key to close.
pause >nul
