@echo off
setlocal EnableDelayedExpansion

set "SOURCE_ROOT=%~1"
set "LIVE_ROOT=%~2"

if not exist "!SOURCE_ROOT!\injected_code.c" (
  echo Shared C3X source root is unavailable: !SOURCE_ROOT! 1>&2
  exit /b 1
)
if not exist "!LIVE_ROOT!\INSTALL.bat" (
  echo Live C3X test root is unavailable: !LIVE_ROOT! 1>&2
  exit /b 1
)
if exist "!LIVE_ROOT!\temp.exe" (
  del /q "!LIVE_ROOT!\temp.exe" >nul 2>nul
  if exist "!LIVE_ROOT!\temp.exe" (
    echo The live C3X installer temp.exe is still in use. Close its dialog or process before synchronizing. 1>&2
    exit /b 1
  )
)

for %%F in (C3X.h Civ3Conquests.h common.c injected_code.c ep.c civ_prog_objects.csv trade_net_addresses.txt INSTALL.bat default.c3x_config.ini) do (
  copy /y "!SOURCE_ROOT!\%%F" "!LIVE_ROOT!\%%F" >nul
  if errorlevel 1 exit /b 1
)

if not exist "!LIVE_ROOT!\Renderer\bin" mkdir "!LIVE_ROOT!\Renderer\bin"
copy /y "!SOURCE_ROOT!\Renderer\bin\C3XRenderer.dll" "!LIVE_ROOT!\Renderer\bin\C3XRenderer.dll" >nul
if errorlevel 1 (
  echo Could not update the live renderer DLL. Exit Civ III before deploying. 1>&2
  exit /b 1
)
if not exist "!LIVE_ROOT!\Renderer\native" mkdir "!LIVE_ROOT!\Renderer\native"
copy /y "!SOURCE_ROOT!\Renderer\native\c3x_renderer_api.h" "!LIVE_ROOT!\Renderer\native\c3x_renderer_api.h" >nul
if errorlevel 1 exit /b 1

copy /y "!SOURCE_ROOT!\Renderer\default.custom_rendering.txt" "!LIVE_ROOT!\Renderer\default.custom_rendering.txt" >nul
if errorlevel 1 exit /b 1
if not exist "!LIVE_ROOT!\Renderer\handoffs" mkdir "!LIVE_ROOT!\Renderer\handoffs"
for %%F in (L9_terrain.json L10_dunes.json L11_marsh.json) do (
  copy /y "!SOURCE_ROOT!\Renderer\handoffs\%%F" "!LIVE_ROOT!\Renderer\handoffs\%%F" >nul
  if errorlevel 1 exit /b 1
)

for %%D in (TerrainNormalized VegetationNormalized DecalsNormalized) do (
  xcopy /e /i /y "!SOURCE_ROOT!\Renderer\packs\%%D" "!LIVE_ROOT!\Renderer\packs\%%D" >nul
  if errorlevel 1 exit /b 1
)

if not exist "!LIVE_ROOT!\custom.c3x_config.ini" if exist "!SOURCE_ROOT!\custom.c3x_config.ini" (
  copy /y "!SOURCE_ROOT!\custom.c3x_config.ini" "!LIVE_ROOT!\custom.c3x_config.ini" >nul
  if errorlevel 1 exit /b 1
)

for %%F in (C3X.h injected_code.c Renderer\bin\C3XRenderer.dll Renderer\native\c3x_renderer_api.h Renderer\default.custom_rendering.txt Renderer\handoffs\L9_terrain.json Renderer\handoffs\L10_dunes.json Renderer\handoffs\L11_marsh.json Renderer\packs\TerrainNormalized\manifest.json Renderer\packs\VegetationNormalized\vegetation_runtime.bin Renderer\packs\DecalsNormalized\manifest.json) do (
  fc /b "!SOURCE_ROOT!\%%F" "!LIVE_ROOT!\%%F" >nul
  if errorlevel 1 (
    echo Live file does not match the shared source: %%F 1>&2
    exit /b 1
  )
)

echo PASS live_c3x_sync: C3X_Districts matches the shared renderer and injected sources. Run INSTALL.bat interactively before testing.
exit /b 0
