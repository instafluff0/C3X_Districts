@echo off
setlocal
REM Launch the installed Windows build with bounded renderer diagnostics.
REM Default 2 records every frame; pass 1 for sampled timings.
set "C3X_TEST_GAME_DIR=%C3X_RENDERER_CIV3_CONQUESTS%"
if not defined C3X_TEST_GAME_DIR set "C3X_TEST_GAME_DIR=%ProgramFiles(x86)%\GOG Galaxy\Games\Civilization III Complete\Conquests"
if not exist "%C3X_TEST_GAME_DIR%\Civ3Conquests.exe" (
  echo Civ III was not found. Set C3X_RENDERER_CIV3_CONQUESTS to its Conquests folder. 1>&2
  pause
  exit /b 1
)
tasklist /fi "imagename eq Civ3Conquests.exe" /nh 2>nul | find /i "Civ3Conquests.exe" >nul
if not errorlevel 1 (
  echo Civ III is already running. Exit it before starting this traced test session.
  pause
  exit /b 1
)
set "C3X_TEST_LOG_DIR=%C3X_TEST_GAME_DIR%\C3X_Districts\Renderer\verification"
if not exist "%C3X_TEST_LOG_DIR%" mkdir "%C3X_TEST_LOG_DIR%"
set "C3X_RENDERER_TRACE=2"
if "%~1"=="1" set "C3X_RENDERER_TRACE=1"
set "C3X_RENDERER_TRACE_FILE=%C3X_TEST_LOG_DIR%\in_game_trace.log"
echo Starting Civ III with renderer diagnostics. Trace: %C3X_RENDERER_TRACE_FILE%
start "" /d "%C3X_TEST_GAME_DIR%" "%C3X_TEST_GAME_DIR%\Civ3Conquests.exe"
exit /b %errorlevel%
