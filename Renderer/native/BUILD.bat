@echo off
setlocal
rem Historical regression fixtures explicitly select their frozen renderer.
set "C3X_RENDERER_VISUAL_PROFILE=frozen"
set "C3X_RENDERER_TRACE=0"
pushd "%~dp0"

if defined C3X_VS_PATH goto compiler_ready
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" exit /b 1
rem Match the verified benchmark toolchain discovery, including preview installs.
set "C3X_BUILD_VS_RECORD=%TEMP%\c3x-renderer-build-vs-path.txt"
"%VSWHERE%" -latest -prerelease -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath >"%C3X_BUILD_VS_RECORD%"
set /p C3X_VS_PATH=<"%C3X_BUILD_VS_RECORD%"
if not defined C3X_VS_PATH "%VSWHERE%" -all -products * -property installationPath >"%C3X_BUILD_VS_RECORD%"
if not defined C3X_VS_PATH set /p C3X_VS_PATH=<"%C3X_BUILD_VS_RECORD%"
del "%C3X_BUILD_VS_RECORD%" >nul 2>nul
:compiler_ready
if not defined C3X_VS_PATH exit /b 1
if not exist "%C3X_VS_PATH%\VC\Auxiliary\Build\vcvars32.bat" exit /b 1

call "%C3X_VS_PATH%\VC\Auxiliary\Build\vcvars32.bat" >nul
if errorlevel 1 exit /b 1

if not exist "..\bin" mkdir "..\bin"
if not exist "build" mkdir "build"
if not exist "build\candidate" mkdir "build\candidate"

if /i "%~1"=="linear-backup" (
  if not exist "build\linear-backup" mkdir "build\linear-backup"
  cl /nologo /std:c++17 /EHsc /O2 /W4 /WX test_linear_backup.cpp /Fo:build\linear-backup\ /Fe:build\linear-backup\test.exe /link /LARGEADDRESSAWARE d3d11.lib d3dcompiler.lib
  if errorlevel 1 exit /b 1
  build\linear-backup\test.exe
  if errorlevel 1 exit /b 1
  exit /b 0
)

if /i "%~1"=="composition-replay" (
  if not exist "build\composition-replay" mkdir "build\composition-replay"
  cl /nologo /std:c++17 /EHsc /O2 /W4 /WX replay_composition.cpp /Fo:build\composition-replay\ /Fe:build\composition-replay\replay_composition.exe /link /LARGEADDRESSAWARE
  if errorlevel 1 exit /b 1
  exit /b 0
)

if /i "%~1"=="tactical" (
  if not exist "..\lab\out" mkdir "..\lab\out"
  if not exist "..\lab\out\tactical-overlays" mkdir "..\lab\out\tactical-overlays"
  if not exist "build\tactical" mkdir "build\tactical"
  cl /nologo /std:c++17 /EHsc /O2 /W4 /WX /DC3X_TACTICAL_STANDALONE test_tactical_overlay.cpp /Fo:build\tactical\ /Fe:build\tactical\test.exe /link d3d11.lib d3dcompiler.lib gdi32.lib user32.lib
  if errorlevel 1 exit /b 1
  build\tactical\test.exe
  if errorlevel 1 exit /b 1
  exit /b 0
)

if /i "%~1"=="native-text" (
  if not exist "build\gpu-composition" mkdir "build\gpu-composition"
  cl /nologo /std:c++17 /EHsc /O2 /W4 /WX test_native_text.cpp /Fo:build\gpu-composition\ /Fe:build\gpu-composition\test_native_text.exe /link d3d11.lib d3dcompiler.lib gdi32.lib user32.lib
  if errorlevel 1 exit /b 1
  build\gpu-composition\test_native_text.exe
  if errorlevel 1 exit /b 1
  exit /b 0
)

if /i "%~1"=="gpu-frame" (
  ..\..\tcc\tcc.exe -m32 -run test_gpu_frame_api.c
  if errorlevel 1 exit /b 1
  if not exist "build\gpu-composition" mkdir "build\gpu-composition"
  cl /nologo /std:c++17 /EHsc /O2 /W4 /WX /wd4100 /wd4191 /DC3X_GPU_NATIVE_CONTRACT biq_preview.cpp test_native_worker.cpp /Fo:build\gpu-composition\ /Fe:build\gpu-composition\test_gpu_frame.exe /link /MAP:build\gpu-composition\test_gpu_frame.map /LARGEADDRESSAWARE gdi32.lib msimg32.lib user32.lib bcrypt.lib
  if errorlevel 1 exit /b 1
  exit /b 0
)

if /i "%~1"=="native-lifetimes" (
  if not exist "build\gpu-composition" mkdir "build\gpu-composition"
  cl /nologo /std:c++17 /EHsc /O2 /W4 /WX /wd4100 /wd4191 test_native_lifetimes.cpp /Fo:build\gpu-composition\ /Fe:build\gpu-composition\test_native_lifetimes.exe /link gdi32.lib user32.lib bcrypt.lib
  if errorlevel 1 exit /b 1
  exit /b 0
)

if /i "%~1"=="native-image-adapter" (
  if not exist "build\gpu-composition" mkdir "build\gpu-composition"
  cl /nologo /std:c++17 /EHsc /O2 /W4 /WX /wd4100 /wd4191 test_native_image_adapter.cpp /Fo:build\gpu-composition\ /Fe:build\gpu-composition\test_native_image_adapter.exe /link d3d11.lib d3dcompiler.lib gdi32.lib user32.lib bcrypt.lib
  if errorlevel 1 exit /b 1
  exit /b 0
)

if /i "%~1"=="gpu-image-operations" (
  if not exist "build\gpu-composition" mkdir "build\gpu-composition"
  cl /nologo /std:c++17 /EHsc /O2 /W4 /WX /wd4191 test_gpu_image_compositor.cpp /Fo:build\gpu-composition\ /Fe:build\gpu-composition\test_gpu_image_compositor.exe /link d3d11.lib d3dcompiler.lib gdi32.lib user32.lib bcrypt.lib
  if errorlevel 1 exit /b 1
  exit /b 0
)

if /i "%~1"=="native-observation" (
  if not exist "build\gpu-composition" mkdir "build\gpu-composition"
  cl /nologo /std:c++17 /EHsc /O2 /W4 /WX /wd4100 /wd4191 test_native_observation.cpp /Fo:build\gpu-composition\ /Fe:build\gpu-composition\test_native_observation.exe /link gdi32.lib user32.lib bcrypt.lib
  if errorlevel 1 exit /b 1
  exit /b 0
)

if /i "%~1"=="jgl-image-operations" (
  if not exist "build\gpu-composition" mkdir "build\gpu-composition"
  cl /nologo /std:c++17 /EHsc /O2 /W4 /WX test_jgl_image_operations.cpp /Fo:build\gpu-composition\ /Fe:build\gpu-composition\test_jgl_image_operations.exe /link gdi32.lib user32.lib
  if errorlevel 1 exit /b 1
  exit /b 0
)

if /i "%~1"=="gpu-composition" (
  if not exist "build\gpu-composition" mkdir "build\gpu-composition"
  cl /nologo /std:c++17 /EHsc /O2 /W4 /WX benchmark_gpu_composition.cpp /Fo:build\gpu-composition\ /Fe:build\gpu-composition\benchmark_gpu_composition.exe /link /LARGEADDRESSAWARE d3d11.lib d3dcompiler.lib dxgi.lib gdi32.lib msimg32.lib user32.lib
  if errorlevel 1 exit /b 1
  exit /b 0
)

if /i "%~1"=="city-fidelity" (
  cl /nologo /std:c++17 /EHsc /O2 /W4 /WX city_fidelity\contract.cpp /Fo:build\city_contract.obj /Fe:build\city_contract.exe
  if errorlevel 1 exit /b 1
  build\city_contract.exe ..\packs\CityCompositionRuntime\city.bin
  if errorlevel 1 exit /b 1
  cl /nologo /std:c++17 /EHsc /O2 /W4 /WX city_fidelity\d3d_contract.cpp /Fo:build\city_d3d_contract.obj /Fe:build\city_d3d_contract.exe /link d3d11.lib d3dcompiler.lib
  if errorlevel 1 exit /b 1
  build\city_d3d_contract.exe
  if errorlevel 1 exit /b 1
  exit /b 0
)

if /i "%~1"=="unit-bridge" (
  if not exist "build\unit_bridge_capture.h" exit /b 1
  cl /nologo /std:c++17 /EHsc /O2 /W4 /WX /wd4100 /wd4191 test_unit_bridge.cpp /Fo:build\ /Fe:build\test_unit_bridge.exe
  if errorlevel 1 exit /b 1
  build\test_unit_bridge.exe
  if errorlevel 1 exit /b 1
  exit /b 0
)

cl /nologo /std:c++17 /EHsc /O2 /W4 /WX /LD c3x_renderer.cpp terrain_scene_runtime.cpp environment_runtime.cpp terrain_definition_runtime.cpp scene_export.cpp frame_scheduler.cpp /Fo:build\ /Fe:build\candidate\C3XRenderer.dll /link /DEF:c3x_renderer.def /IMPLIB:build\candidate\C3XRenderer.lib d3d11.lib d3dcompiler.lib dxgi.lib gdi32.lib msimg32.lib user32.lib bcrypt.lib
if errorlevel 1 exit /b 1

cl /nologo /std:c++17 /EHsc /O2 /W4 /WX test_asset_content_hash.cpp /Fo:build\ /Fe:build\test_asset_content_hash.exe
if errorlevel 1 exit /b 1
build\test_asset_content_hash.exe
if errorlevel 1 exit /b 1

cl /nologo /std:c++17 /EHsc /O2 /W4 /WX native_smoke.cpp environment_runtime.cpp /Fo:build\ /Fe:build\native_smoke.exe /link gdi32.lib
if errorlevel 1 exit /b 1

cl /nologo /std:c++17 /EHsc /O2 /W4 /WX biq_preview.cpp /Fo:build\ /Fe:build\biq_preview.exe /link /LARGEADDRESSAWARE gdi32.lib
if errorlevel 1 exit /b 1

cl /nologo /std:c++17 /EHsc /O2 /W4 /WX test_animation_runtime.cpp /Fo:build\ /Fe:build\test_animation_runtime.exe
if errorlevel 1 exit /b 1
build\test_animation_runtime.exe
if errorlevel 1 exit /b 1

rem The portable unit-bridge test refreshes this header from actual injected C.
if exist "build\unit_bridge_capture.h" (
  cl /nologo /std:c++17 /EHsc /O2 /W4 /WX /wd4100 /wd4191 test_unit_bridge.cpp /Fo:build\ /Fe:build\test_unit_bridge.exe
  if errorlevel 1 exit /b 1
  build\test_unit_bridge.exe
  if errorlevel 1 exit /b 1
)

if /i "%~1"=="candidate-compile" exit /b 0

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
if not exist "..\packs\ShoreNormalized\cliff_runtime.bin" set "C3X_APPROVED_PAYLOAD=0"
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

if /i "%~2"=="no-stage" (
  popd
  exit /b %C3X_BUILD_RESULT%
)

if /i "%~1"=="candidate-only" (
  popd
  exit /b %C3X_BUILD_RESULT%
)

copy /y "build\candidate\C3XRenderer.dll" "..\bin\C3XRenderer.dll" >nul
if errorlevel 1 (
  echo Live C3XRenderer.dll is in use; Renderer\bin still contains a stale build. Exit Civ III and rerun this workflow before INSTALL.bat. 1>&2
  popd
  exit /b 1
)
popd
exit /b %C3X_BUILD_RESULT%
