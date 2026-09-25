@echo off
setlocal
set "C3X_SANDBOX_REPLAY_CLIP=1"
set "C3X_SANDBOX_DAY_NIGHT=1"
set "C3X_SANDBOX_CAPTURE_SEQUENCE="
call "%~dp0run_client.bat"
exit /b %errorlevel%
