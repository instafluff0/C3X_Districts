@echo off
setlocal
set "C3X_SANDBOX_REPLAY_CLIP=1"
set "C3X_SANDBOX_CAPTURE_SEQUENCE=..\sandbox\out\town-clip"
call "%~dp0run_client.bat"
exit /b %errorlevel%
