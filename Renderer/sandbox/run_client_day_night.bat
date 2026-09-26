@echo off
setlocal
set "C3X_SANDBOX_REPLAY_CLIP=1"
set "C3X_SANDBOX_DAY_NIGHT=1"
if not defined C3X_SANDBOX_CAPTURE_SEQUENCE set "C3X_SANDBOX_CAPTURE_SEQUENCE=..\sandbox\out\day-night-coast-light"
call "%~dp0run_client.bat"
exit /b %errorlevel%
