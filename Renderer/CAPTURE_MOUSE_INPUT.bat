@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\capture_map_failure.ps1" -InputTrace %*
if errorlevel 1 echo Mouse-input capture did not start. Check the message above.
pause
exit /b %errorlevel%
