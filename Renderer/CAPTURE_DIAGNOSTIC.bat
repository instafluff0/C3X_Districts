@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\capture_game.ps1" -ShortDiagnostic -Renderer64 %*
set "C3X_CAPTURE_RESULT=%errorlevel%"
echo.
pause
exit /b %C3X_CAPTURE_RESULT%
