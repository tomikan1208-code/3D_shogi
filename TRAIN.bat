@echo off
title 3D Shogi Training
setlocal

set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo [ERROR] Python virtual environment not found: %PY%
    pause
    exit /b 1
)

echo [1] Starting training with %PY%
echo [1.1] Launcher version: train-bat-v4
pushd "%ROOT%ai_training"
if exist ".\build\lib.win-amd64-cpython-313\shogi3d_core.cp313-win_amd64.pyd" (
    copy /Y ".\build\lib.win-amd64-cpython-313\shogi3d_core.cp313-win_amd64.pyd" ".\shogi3d_core.cp313-win_amd64.pyd" >nul 2>nul
    if errorlevel 1 (
        echo [WARN] Could not update shogi3d_core.cp313-win_amd64.pyd ^(file may be in use^)
    ) else (
        echo [INFO] Updated shogi3d_core.cp313-win_amd64.pyd from build output
    )
)
set "SHOGI_DEVICE=cuda"
set "SHOGI_FORCE_CUDA=1"
set "SHOGI_USE_CPP_KINGCHECK=1"
set "SHOGI_USE_CPP_LEGAL_MOVES=1"
set "SHOGI_USE_CPP_PINNED=1"
set "SHOGI_CPP_STRICT_FILTER=0"
set "SHOGI_CPP_FILTER_LEVEL=1"
set "SHOGI_DETERMINISTIC_MOVE_ORDER=1"
set "SHOGI_FIXED_ACTION_SPACE=1"
set "SHOGI_FAST_DROP_GENERATION=1"
set "SHOGI_USE_LEGAL_MOVES_CACHE=1"
set "SHOGI_USE_UINT8_OBS=1"
set "SHOGI_TACTICAL_REWARD=1"
set "SHOGI_FORCE_DUMMYVEC=0"
set "SHOGI_N_STEPS=10000"
set "SHOGI_N_EPOCHS=4"
set "SHOGI_BATCH_SIZE=256"
set "SHOGI_MAX_MASK_GB=8.0"
echo [2] SHOGI_DEVICE=%SHOGI_DEVICE%  SHOGI_FORCE_CUDA=%SHOGI_FORCE_CUDA%
echo [3] CPP_LEGAL=%SHOGI_USE_CPP_LEGAL_MOVES%  CPP_KINGCHECK=%SHOGI_USE_CPP_KINGCHECK%  CPP_PIN=%SHOGI_USE_CPP_PINNED%  STRICT=%SHOGI_CPP_STRICT_FILTER%  FILTER_LEVEL=%SHOGI_CPP_FILTER_LEVEL%
echo [4] SHOGI_N_ENVS=%SHOGI_N_ENVS%  SHOGI_N_STEPS=%SHOGI_N_STEPS%  SHOGI_N_EPOCHS=%SHOGI_N_EPOCHS%
echo [4.1] SHOGI_FORCE_DUMMYVEC=%SHOGI_FORCE_DUMMYVEC%
echo [5] SHOGI_FIXED_ACTION_SPACE=%SHOGI_FIXED_ACTION_SPACE%  SHOGI_DETERMINISTIC_MOVE_ORDER=%SHOGI_DETERMINISTIC_MOVE_ORDER%
echo [6] SHOGI_MAX_MASK_GB=%SHOGI_MAX_MASK_GB%  SHOGI_USE_UINT8_OBS=%SHOGI_USE_UINT8_OBS%
echo [7] SHOGI_BATCH_SIZE=%SHOGI_BATCH_SIZE%  SHOGI_TACTICAL_REWARD=%SHOGI_TACTICAL_REWARD%
"%PY%" train.py
set "RC=%ERRORLEVEL%"
popd

echo.
echo Training finished with exit code %RC%
pause
exit /b %RC%
