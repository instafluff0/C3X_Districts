@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\play_replay.ps1" %*
set "C3X_REPLAY_RESULT=%errorlevel%"
echo.
pause
exit /b %C3X_REPLAY_RESULT%
