@echo off
setlocal

cd /d "%~dp0.."

where py >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python launcher (py) not found.
  echo Install Python 3.11 from https://www.python.org/downloads/release/python-3119/
  pause
  exit /b 1
)

py -3.11 -V >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python 3.11 is not installed.
  echo Install Python 3.11, then run this file again.
  pause
  exit /b 1
)

echo [INFO] Creating .venv_gpu311...
py -3.11 -m venv .venv_gpu311
if errorlevel 1 (
  echo [ERROR] Failed to create .venv_gpu311
  pause
  exit /b 1
)

set "PYTHON=%CD%\.venv_gpu311\Scripts\python.exe"

echo [INFO] Upgrading pip...
"%PYTHON%" -m pip install --upgrade pip
if errorlevel 1 goto :fail

echo [INFO] Installing CUDA PyTorch (cu121)...
"%PYTHON%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 goto :fail

echo [INFO] Installing project dependencies...
"%PYTHON%" -m pip install -r ai_training\requirements_colab.txt
if errorlevel 1 goto :fail

echo [INFO] Verifying CUDA...
"%PYTHON%" -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('count', torch.cuda.device_count()); print('name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
if errorlevel 1 goto :fail

echo [OK] GPU environment is ready: .venv_gpu311
pause
exit /b 0

:fail
echo [ERROR] Setup failed.
pause
exit /b 1
