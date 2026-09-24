@echo off
setlocal
set "C3X_SANDBOX_CAPTURE_SEQUENCE=..\sandbox\out\motion"
call "%~dp0run_client.bat"
exit /b %errorlevel%
