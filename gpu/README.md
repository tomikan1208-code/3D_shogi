# GPU Folder

## Files

- START_GPU_TRAINING.bat
  - Double-click to start training on CUDA using `.venv_gpu311`.
  - Fails fast if CUDA is not available.
- INSTALL_GPU_PY311_ENV.bat
  - Double-click to create `.venv_gpu311` with Python 3.11 and CUDA PyTorch.
  - Requires Python 3.11 installed on Windows.
- CHECK_NVIDIA_GPU.bat
  - Double-click to check NVIDIA GPU and driver via `nvidia-smi`.
- gpu_precheck.py
  - Internal CUDA check used by launcher.

## Notes

- Launcher uses GPU venv: `.venv_gpu311/Scripts/python.exe`
- It sets:
  - `SHOGI_DEVICE=cuda`
  - `SHOGI_FORCE_CUDA=1`
  - default `SHOGI_N_ENVS=6` (if not already set)

## Why a separate venv?

- Current workspace venv uses Python 3.13.
- CUDA PyTorch wheels are not available for this setup, so GPU training requires Python 3.11 environment.
