@echo off
rem Bounded logs/FPS only; no input replay or window-image recording.
call "%~dp0CAPTURE_GAME.bat" -NoReplayRecording %*
exit /b %errorlevel%
