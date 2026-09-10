@echo off
setlocal
rem Isolated build/output: never overwrites the game DLL or another Lab candidate.
if /i not "%~1"=="baseline" if /i not "%~1"=="candidate" exit /b 2
pushd "%~dp0"
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "C3X_VS_RECORD=%TEMP%\c3x-renderer-vs-path.txt"
"%VSWHERE%" -latest -prerelease -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath >"%C3X_VS_RECORD%"
set /p C3X_VS_PATH=<"%C3X_VS_RECORD%"
rem Preview Visual Studio channels can have the C++ tools before vswhere's
rem workload catalogue recognizes their component id. Verify vcvars directly.
if not defined C3X_VS_PATH "%VSWHERE%" -all -products * -property installationPath >"%C3X_VS_RECORD%"
if not defined C3X_VS_PATH set /p C3X_VS_PATH=<"%C3X_VS_RECORD%"
del "%C3X_VS_RECORD%" >nul 2>nul
if "%C3X_RENDERER_BUILD_DEBUG%"=="1" echo C3X VS path: %C3X_VS_PATH%
if not defined C3X_VS_PATH exit /b 1
if not exist "%C3X_VS_PATH%\VC\Auxiliary\Build\vcvars32.bat" exit /b 1
call "%C3X_VS_PATH%\VC\Auxiliary\Build\vcvars32.bat" >nul
if "%C3X_RENDERER_BUILD_DEBUG%"=="1" echo vcvars result: %errorlevel%
if errorlevel 1 exit /b 1
if not defined C3X_ZOOM_OUT set "C3X_ZOOM_OUT=build\zoom-%~1"
if not exist "%C3X_ZOOM_OUT%" mkdir "%C3X_ZOOM_OUT%"
set "C3X_ZOOM_CACHE_FLAGS=/DC3X_RENDERER_BENCHMARK_ORACLE"
if not defined C3X_ZOOM_GPU_CACHE_MIB set "C3X_ZOOM_GPU_CACHE_MIB=384"
if not "%C3X_ZOOM_GPU_CACHE_MIB%"=="384" if not "%C3X_ZOOM_GPU_CACHE_MIB%"=="768" exit /b 2
if "%C3X_ZOOM_LARGE_CACHE%"=="1" set "C3X_ZOOM_CACHE_FLAGS=%C3X_ZOOM_CACHE_FLAGS% /DC3X_RENDERER_BENCHMARK_LARGE_CACHE /DC3X_RENDERER_BENCHMARK_GPU_CACHE_MIB=%C3X_ZOOM_GPU_CACHE_MIB%"
if /i not "%~2"=="reuse" if /i not "%~2"=="preview-only" (
  cl /nologo /std:c++17 /EHsc /O2 /W4 /WX /LD %C3X_ZOOM_CACHE_FLAGS% c3x_renderer.cpp terrain_scene_runtime.cpp environment_runtime.cpp terrain_definition_runtime.cpp scene_export.cpp frame_scheduler.cpp /Fo:%C3X_ZOOM_OUT%\ /Fe:%C3X_ZOOM_OUT%\C3XRenderer.dll /link /DEF:c3x_renderer.def /MAP:%C3X_ZOOM_OUT%\C3XRenderer.map /IMPLIB:%C3X_ZOOM_OUT%\C3XRenderer.lib d3d11.lib d3dcompiler.lib dxgi.lib gdi32.lib msimg32.lib user32.lib bcrypt.lib
  if errorlevel 1 exit /b 1
)
rem ep.c enables large-address awareness on the installed game. Match that
rem 32-bit address-space model so the witness does not fail artificially early.
cl /nologo /std:c++17 /EHsc /O2 /W4 /WX biq_preview.cpp /Fo:%C3X_ZOOM_OUT%\ /Fe:%C3X_ZOOM_OUT%\biq_preview.exe /link /LARGEADDRESSAWARE gdi32.lib
if errorlevel 1 exit /b 1
if /i "%~2"=="build-only" exit /b 0
if /i "%~2"=="preview-only" exit /b 0
set "C3X_RENDERER_VISUAL_PROFILE="
set "C3X_RENDERER_TRACE=2"
set "C3X_RENDERER_TRACE_FILE=%C3X_ZOOM_OUT%\renderer.log"
set "C3X_RENDERER_PREVIEW_ZOOM=1"
set "C3X_RENDERER_PREVIEW_NAVIGATION="
if /i "%C3X_CAMERA_SCENARIO%"=="navigation" (
  set "C3X_RENDERER_PREVIEW_ZOOM="
  set "C3X_RENDERER_PREVIEW_NAVIGATION=1"
)
set "C3X_RENDERER_PREVIEW_ANIMATION=1"
if not defined C3X_ZOOM_ROOT set "C3X_ZOOM_ROOT=..\.."
set "C3X_RENDERER_PREVIEW_CUSTOM_DEFINITIONS=%C3X_ZOOM_ROOT%\Renderer\custom.custom_rendering.txt"
if not defined C3X_ZOOM_WIDTH set "C3X_ZOOM_WIDTH=640"
if not defined C3X_ZOOM_HEIGHT set "C3X_ZOOM_HEIGHT=480"
%C3X_ZOOM_OUT%\biq_preview.exe %C3X_ZOOM_OUT%\C3XRenderer.dll "%C3X_ZOOM_ROOT%" "%C3X_ZOOM_ROOT%\Renderer\default.custom_rendering.txt" ..\lab\.local\verification\world.csv %C3X_ZOOM_OUT%\zoom.bmp %C3X_ZOOM_WIDTH% %C3X_ZOOM_HEIGHT% 75 39 128 12 >%C3X_ZOOM_OUT%\benchmark.log 2>&1
set "C3X_ZOOM_RESULT=%errorlevel%"
type %C3X_ZOOM_OUT%\benchmark.log
exit /b %C3X_ZOOM_RESULT%
