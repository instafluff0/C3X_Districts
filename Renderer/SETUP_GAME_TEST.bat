@echo off
setlocal EnableExtensions

rem Run from the installed Conquests\C3X_Districts checkout in Windows.
rem Builds and stages the matching x86 bridge, x64 renderer, and helper, then
rem installs the injected game executable. Art packs are supplied separately.
pushd "%~dp0.."
if errorlevel 1 (
  echo Could not open the C3X checkout. 1>&2
  exit /b 1
)

if not exist "..\Civ3Conquests.exe" (
  echo Civ3Conquests.exe was not found beside this C3X_Districts folder. 1>&2
  echo Run this script through the checkout inside the Windows game's Conquests folder. 1>&2
  goto fail
)
if not exist "Renderer\packs\TerrainNormalized\manifest.json" (
  echo Renderer packs are missing. Copy the local Renderer\packs folder before testing. 1>&2
  goto fail
)
if not exist "Renderer\packs\UnitAnimationRuntime" (
  echo The copied Renderer packs are missing UnitAnimationRuntime. 1>&2
  goto fail
)
tasklist /fi "imagename eq Civ3Conquests.exe" /nh 2>nul | find /i "Civ3Conquests.exe" >nul
if not errorlevel 1 (
  echo Exit Civ III before rebuilding or installing C3X. 1>&2
  goto fail
)
if /i "%~1"=="--check" (
  if not defined C3X_VS_PATH if not exist "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" (
    echo Visual Studio C++ build tools were not found. 1>&2
    goto fail
  )
  if exist "custom.c3x_config.ini" (
    findstr /r /i /c:"^[ ]*enable_custom_rendering[ ]*=[ ]*true" "custom.c3x_config.ini" >nul
    if errorlevel 1 (
      echo Set enable_custom_rendering = true in custom.c3x_config.ini before testing. 1>&2
      goto fail
    )
  ) else echo The setup run will create custom.c3x_config.ini with the renderer enabled.
  echo Game location, copied packs, and build-tool discovery are ready.
  echo Run Renderer\SETUP_GAME_TEST.bat without --check to build and install.
  popd
  exit /b 0
)

if not exist "custom.c3x_config.ini" (
  >"custom.c3x_config.ini" (
    echo enable_custom_rendering = true
    echo enable_custom_rendering_reflections = true
    echo enable_custom_rendering_waves = true
    echo enable_custom_rendering_cache = true
    echo enable_custom_rendering_zoom = true
  )
  if not exist "custom.c3x_config.ini" (
    echo Could not create the local renderer configuration. 1>&2
    goto fail
  )
  echo Created custom.c3x_config.ini with the full renderer enabled.
)
findstr /r /i /c:"^[ ]*enable_custom_rendering[ ]*=[ ]*true" "custom.c3x_config.ini" >nul
if errorlevel 1 (
  echo Set enable_custom_rendering = true in custom.c3x_config.ini before testing. 1>&2
  goto fail
)

echo Verifying the injected C code...
call TEST_INJECTED_CODE_COMPILE.bat
if errorlevel 1 (
  echo Injected-code compilation failed; the game was not installed. 1>&2
  goto fail
)

echo Building and staging the matching Renderer64 binaries...
call Renderer\native\BUILD_RENDERER64.bat
if errorlevel 1 (
  echo Renderer64 build or startup verification failed; the game was not installed. 1>&2
  goto fail
)

echo Installing C3X into Civ3Conquests.exe...
call INSTALL.bat
if errorlevel 1 (
  echo C3X installation failed. 1>&2
  goto fail
)

echo Renderer64 build and C3X installation finished. Launch Civ III normally to test.
popd
exit /b 0

:fail
popd
exit /b 1
