@echo off
setlocal
set "C3X_SANDBOX_CAPTURE=..\sandbox\out\client-unit-frame.bmp"
set "C3X_SANDBOX_CAPTURE_MOVED=..\sandbox\out\client-unit-moved.bmp"
set "C3X_SANDBOX_CAPTURE_SCROLL=..\sandbox\out\client-scroll.bmp"
set "C3X_SANDBOX_CAPTURE_JUMP=..\sandbox\out\client-jump.bmp"
set "C3X_SANDBOX_CAPTURE_RETURN=..\sandbox\out\client-return.bmp"
set "C3X_SANDBOX_CAPTURE_WRAP=..\sandbox\out\client-wrap.bmp"
call "%~dp0run_client.bat"
exit /b %errorlevel%
