@echo off
setlocal
REM Evaluate the caller-driven handoff with the current injected camera hook.
REM RUN.bat compiles/injects this checkout; no INSTALL.bat step is required.
set "C3X_RENDERER_NATIVE_ASYNC=1"
call "%~dp0TEST_IN_GAME.bat" %*
exit /b %errorlevel%
