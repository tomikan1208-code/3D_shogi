@echo off
REM sb3_contrib MaskablePPO action_masks dtypeをuint8にパッチ適用
REM venv再構築やpip install後にこのバッチを実行してください

setlocal
set SB3_BUFFERS=%~dp0.venv\Lib\site-packages\sb3_contrib\common\maskable\buffers.py

if not exist "%SB3_BUFFERS%" (
    echo [ERROR] sb3_contrib buffers.py が見つかりません: %SB3_BUFFERS%
    exit /b 1
)

REM 既存のaction_masks dtype行をuint8に書き換え
powershell -Command "(Get-Content '%SB3_BUFFERS%') -replace 'np.zeros\(\(self.buffer_size, self.n_envs, self.n_actions\), dtype=np.float32\)', 'np.zeros((self.buffer_size, self.n_envs, self.n_actions), dtype=np.uint8)' | Set-Content '%SB3_BUFFERS%'"

REM 2重適用防止のため、float32→uint8が既に適用済みかチェック
findstr /C:"dtype=np.uint8" "%SB3_BUFFERS%" >nul
if %errorlevel%==0 (
    echo [OK] sb3_contrib buffers.py action_masks dtype=np.uint8 にパッチ済み
    exit /b 0
) else (
    echo [ERROR] パッチ適用に失敗しました。手動でご確認ください。
    exit /b 1
)
