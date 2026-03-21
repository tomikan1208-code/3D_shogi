@echo off
setlocal

cd /d %~dp0

if exist ".venv\Scripts\activate.bat" (
  call ".venv\Scripts\activate.bat"
) else (
  echo [WARN] .venv\Scripts\activate.bat not found. Running with system Python.
)

set "SHOGI_USE_CPP_KINGCHECK=1"
set "SHOGI_USE_CPP_LEGAL_MOVES=1"
set "SHOGI_CPP_STRICT_FILTER=0"
set "SHOGI_DETERMINISTIC_MOVE_ORDER=0"
set "SHOGI_FAST_DROP_GENERATION=1"
set "SHOGI_USE_LEGAL_MOVES_CACHE=1"
set "SHOGI_FIXED_ACTION_SPACE=1"

if "%SHOGI_MODEL_PATH%"=="" set SHOGI_MODEL_PATH=./ai_training/models/shogi3d_ppo_latest.zip
if "%SHOGI_EVAL_GAMES%"=="" set SHOGI_EVAL_GAMES=100
if "%SHOGI_OPENING_RANDOM_PLIES%"=="" set SHOGI_OPENING_RANDOM_PLIES=2
if "%SHOGI_EVAL_WORKERS%"=="" set SHOGI_EVAL_WORKERS=1
if "%SHOGI_EVAL_DEVICE%"=="" set SHOGI_EVAL_DEVICE=auto
set PYTHONUNBUFFERED=1
if "%OMP_NUM_THREADS%"=="" set OMP_NUM_THREADS=1

echo Model: %SHOGI_MODEL_PATH%
echo Games: %SHOGI_EVAL_GAMES%
echo Opening random plies: %SHOGI_OPENING_RANDOM_PLIES%
echo Workers: %SHOGI_EVAL_WORKERS%
echo Device: %SHOGI_EVAL_DEVICE%
echo.

python -u ai_training\eval_vs_random.py --model "%SHOGI_MODEL_PATH%" --games %SHOGI_EVAL_GAMES% --opening-random-plies %SHOGI_OPENING_RANDOM_PLIES% --workers %SHOGI_EVAL_WORKERS% --device %SHOGI_EVAL_DEVICE%

echo.
echo Evaluation finished. Press any key to close.
pause >nul
