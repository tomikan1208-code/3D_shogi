@echo off
setlocal

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
if errorlevel 1 (
  echo [ERROR] nvidia-smi failed. NVIDIA driver may be missing.
)

pause
