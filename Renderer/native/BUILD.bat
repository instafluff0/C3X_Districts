@echo off
setlocal
pushd "%~dp0"

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
  echo Visual Studio Installer's vswhere.exe was not found. 1>&2
  exit /b 1
)

for /f "usebackq tokens=*" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "C3X_VS_PATH=%%I"
if not defined C3X_VS_PATH (
  echo A Visual Studio installation with the x86 C++ toolchain was not found. 1>&2
  exit /b 1
)

call "%C3X_VS_PATH%\VC\Auxiliary\Build\vcvars32.bat" >nul
if errorlevel 1 exit /b 1

if not exist "..\bin" mkdir "..\bin"
if not exist "build" mkdir "build"
if not exist "build\candidate" mkdir "build\candidate"

cl /nologo /std:c++17 /EHsc /O2 /W4 /WX /LD c3x_renderer.cpp terrain_scene_runtime.cpp environment_runtime.cpp terrain_definition_runtime.cpp scene_export.cpp frame_scheduler.cpp /Fo:build\ /Fe:build\candidate\C3XRenderer.dll /link /DEF:c3x_renderer.def /IMPLIB:build\candidate\C3XRenderer.lib d3d11.lib d3dcompiler.lib dxgi.lib gdi32.lib msimg32.lib user32.lib bcrypt.lib
if errorlevel 1 exit /b 1

cl /nologo /std:c++17 /EHsc /O2 /W4 /WX native_smoke.cpp environment_runtime.cpp /Fo:build\ /Fe:build\native_smoke.exe /link gdi32.lib
if errorlevel 1 exit /b 1

cl /nologo /std:c++17 /EHsc /O2 /W4 /WX biq_preview.cpp /Fo:build\ /Fe:build\biq_preview.exe /link gdi32.lib
if errorlevel 1 exit /b 1

build\native_smoke.exe "build\candidate\C3XRenderer.dll"
set "C3X_BUILD_RESULT=%errorlevel%"
if not "%C3X_BUILD_RESULT%"=="0" (
  popd
  exit /b %C3X_BUILD_RESULT%
)

if /i "%~1"=="portable" goto approved_terrain_done

set "C3X_APPROVED_PAYLOAD=1"
if not exist "..\packs\TerrainNormalized\manifest.json" set "C3X_APPROVED_PAYLOAD=0"
if not exist "..\packs\VegetationNormalized\vegetation_runtime.bin" set "C3X_APPROVED_PAYLOAD=0"
if not exist "..\packs\DecalsNormalized\manifest.json" set "C3X_APPROVED_PAYLOAD=0"
if not exist "..\packs\TerrainElementsNormalized\manifest.json" set "C3X_APPROVED_PAYLOAD=0"
if not exist "..\packs\ShoreNormalized\shore_runtime.bin" set "C3X_APPROVED_PAYLOAD=0"
if not exist "..\packs\RouteStylesNormalized\manifest.json" set "C3X_APPROVED_PAYLOAD=0"
if not exist "..\packs\RouteDoodadsNormalized\bridge_runtime.bin" set "C3X_APPROVED_PAYLOAD=0"
if not exist "..\packs\ResourceNormalized\resource_runtime.bin" set "C3X_APPROVED_PAYLOAD=0"
if not exist "..\packs\CityComponentsNormalized\city_runtime.bin" set "C3X_APPROVED_PAYLOAD=0"
if not exist "..\packs\CityAdjunctsNormalized\wall_runtime.bin" set "C3X_APPROVED_PAYLOAD=0"
if not exist "..\packs\ImprovementsNormalized\mine_runtime.bin" set "C3X_APPROVED_PAYLOAD=0"
if not exist "..\packs\ImprovementsNormalized\farm_runtime.bin" set "C3X_APPROVED_PAYLOAD=0"
if "%C3X_APPROVED_PAYLOAD%"=="1" (
  build\native_smoke.exe "build\candidate\C3XRenderer.dll" --definitions ..\.. ..\..\Renderer\default.custom_rendering.txt
  if errorlevel 1 (
    popd
    exit /b 1
  )
) else (
  echo SKIP approved_terrain_integration: local normalized L9-L19 payloads are unavailable.
)

:approved_terrain_done

copy /y "build\candidate\C3XRenderer.dll" "..\bin\C3XRenderer.dll" >nul
if errorlevel 1 (
  echo Live C3XRenderer.dll is in use; Renderer\bin still contains a stale build. Exit Civ III and rerun this workflow before INSTALL.bat. 1>&2
  popd
  exit /b 1
)
popd
exit /b %C3X_BUILD_RESULT%
