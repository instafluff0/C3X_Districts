@echo off
setlocal
REM Run the verified injection path against installed Civ III with diagnostics.
REM RUN.bat keeps the injected API and renderer DLL in sync without reinstalling.
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
set "C3X_TEST_MOD_DIR=%C3X_TEST_GAME_DIR%\C3X_Districts"
if not exist "%C3X_TEST_MOD_DIR%\RUN.bat" (
  echo The installed C3X_Districts checkout link was not found. 1>&2
  pause
  exit /b 1
)
if not exist "%C3X_TEST_MOD_DIR%\Renderer\bin\C3XRenderer.dll" exit /b 1
set "C3X_RENDERER_VISUAL_PROFILE=pickup-r1"
set "C3X_RENDERER_TRACE=2"
if "%~1"=="1" set "C3X_RENDERER_TRACE=1"
set "C3X_RENDERER_TRACE_FILE="
if "%~1"=="--check" (
  echo PASS pickup_test_launcher: installed checkout, renderer DLL, pickup-r1, trace, and RUN.bat route ready; game not launched.
  exit /b 0
)
echo Starting Civ III with pickup-r1 renderer diagnostics through OutputDebugStringA.
start "" /d "%C3X_TEST_MOD_DIR%" "%ComSpec%" /d /c call RUN.bat
exit /b %errorlevel%
