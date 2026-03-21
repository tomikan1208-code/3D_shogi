@echo off
setlocal
cd /d %~dp0

echo [AI] Launcher cwd: %cd%

if exist ".venv\Scripts\activate.bat" (
	call ".venv\Scripts\activate.bat"
) else (
	echo [WARN] .venv\Scripts\activate.bat not found. Running with system Python.
)

rem High-speed defaults
set SHOGI_USE_CPP_KINGCHECK=1
set SHOGI_USE_CPP_LEGAL_MOVES=1
set SHOGI_CPP_STRICT_FILTER=0
set SHOGI_DETERMINISTIC_MOVE_ORDER=0
set SHOGI_FAST_DROP_GENERATION=1
set SHOGI_USE_LEGAL_MOVES_CACHE=1
if "%SHOGI_FIXED_ACTION_SPACE%"=="" set SHOGI_FIXED_ACTION_SPACE=1
set PYTHONUNBUFFERED=1

echo [AI] SHOGI_FIXED_ACTION_SPACE=%SHOGI_FIXED_ACTION_SPACE%
echo [AI] Starting ai_server.py ...

python -u ai_training/ai_server.py

