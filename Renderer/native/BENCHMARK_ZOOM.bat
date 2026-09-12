@echo off
setlocal
echo BUILD_TIMING phase=begin clock=%TIME%
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
rem Reuse requires identical compiler/SDK identity in addition to source/flags.
>"%C3X_ZOOM_OUT%\toolchain.txt" echo %VCToolsVersion% %WindowsSDKVersion%
if not defined C3X_ZOOM_BUILD_UNITS set "C3X_ZOOM_BUILD_UNITS=all"
if exist "%C3X_ZOOM_OUT%\previous-toolchain.txt" (
  fc /b "%C3X_ZOOM_OUT%\previous-toolchain.txt" "%C3X_ZOOM_OUT%\toolchain.txt" >nul
  if errorlevel 1 set "C3X_ZOOM_BUILD_UNITS=all"
) else (
  set "C3X_ZOOM_BUILD_UNITS=all"
)
echo BUILD_TIMING phase=compiler_begin clock=%TIME%
if /i not "%~2"=="reuse" if /i not "%~2"=="preview-only" (
  for %%U in (c3x_renderer terrain_scene_runtime environment_runtime terrain_definition_runtime scene_export frame_scheduler) do (
    call :compile_unit %%U dll
    if errorlevel 1 exit /b 1
  )
  cl /nologo /LD "%C3X_ZOOM_OUT%\c3x_renderer.obj" "%C3X_ZOOM_OUT%\terrain_scene_runtime.obj" "%C3X_ZOOM_OUT%\environment_runtime.obj" "%C3X_ZOOM_OUT%\terrain_definition_runtime.obj" "%C3X_ZOOM_OUT%\scene_export.obj" "%C3X_ZOOM_OUT%\frame_scheduler.obj" /Fe:%C3X_ZOOM_OUT%\C3XRenderer.dll /link /DEF:c3x_renderer.def /MAP:%C3X_ZOOM_OUT%\C3XRenderer.map /IMPLIB:%C3X_ZOOM_OUT%\C3XRenderer.lib d3d11.lib d3dcompiler.lib dxgi.lib gdi32.lib msimg32.lib user32.lib bcrypt.lib
  if errorlevel 1 exit /b 1
)
echo BUILD_TIMING phase=dll_done clock=%TIME%
rem Match ep.c's x86 large-address-aware executable without staging it.
call :compile_unit biq_preview preview
if errorlevel 1 exit /b 1
cl /nologo "%C3X_ZOOM_OUT%\biq_preview.obj" /Fe:%C3X_ZOOM_OUT%\biq_preview.exe /link /LARGEADDRESSAWARE gdi32.lib
if errorlevel 1 exit /b 1
echo BUILD_TIMING phase=preview_done clock=%TIME%
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

:compile_unit
set "C3X_COMPILE_UNIT="
if "%C3X_ZOOM_BUILD_UNITS%"=="all" set "C3X_COMPILE_UNIT=1"
for %%U in (%C3X_ZOOM_BUILD_UNITS%) do if "%%U"=="%~1" set "C3X_COMPILE_UNIT=1"
if not exist "%C3X_ZOOM_OUT%\%~1.obj" set "C3X_COMPILE_UNIT=1"
if not defined C3X_COMPILE_UNIT exit /b 0
echo BUILD_UNIT unit=%~1
set "C3X_COMPILE_FLAGS="
if "%~2"=="dll" set "C3X_COMPILE_FLAGS=/LD %C3X_ZOOM_CACHE_FLAGS%"
cl /nologo /std:c++17 /EHsc /O2 /W4 /WX /c %C3X_COMPILE_FLAGS% %~1.cpp /Fo:%C3X_ZOOM_OUT%\%~1.obj
exit /b %errorlevel%
