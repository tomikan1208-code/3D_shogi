@echo off
title 3D Shogi Server
setlocal

echo [1] Checking port 3000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":3000 " ^| findstr "LISTENING"') do (
    echo [1] Killing PID %%a ...
    start "Kill3000_%%a" /B cmd /c "taskkill /PID %%a /F >nul 2>&1"
)
echo [1] Checking port 5000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5000 " ^| findstr "LISTENING"') do (
    echo [1] Killing PID %%a ...
    start "Kill5000_%%a" /B cmd /c "taskkill /PID %%a /F >nul 2>&1"
)
timeout /t 1 /nobreak >nul
echo [1] Done.

echo [2] Starting server...
start "ShogiServer" /D "%~dp0" cmd /c "node server.js > server.log 2>&1"

echo [3] Starting AI Server...
start "ShogiAI" /D "%~dp0" cmd /c "call AI.bat"

if "%SHOGI_START_NGROK%"=="1" (
    echo [3.5] Starting ngrok tunnel...
    start "ShogiNgrok" /D "%~dp0" cmd /c "call n.bat"
)

timeout /t 10 /nobreak >nul

echo [4] Opening browser...
start "" "http://localhost:3000/watch.html"
start "" "http://localhost:3000"

echo.
echo Server started at http://localhost:3000
echo NPC Watch mode: http://localhost:3000/watch.html
echo Log: %~dp0server.log
echo.
pause
