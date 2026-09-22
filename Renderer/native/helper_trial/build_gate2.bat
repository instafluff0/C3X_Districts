@echo off
setlocal
pushd "%~dp0.."
if errorlevel 1 exit /b 1

if defined C3X_VS_PATH goto compiler_ready
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto fail
set "VS_RECORD=%TEMP%\c3x-helper-gate2-vs-path.txt"
"%VSWHERE%" -latest -prerelease -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath >"%VS_RECORD%"
set /p C3X_VS_PATH=<"%VS_RECORD%"
del "%VS_RECORD%" >nul 2>nul
:compiler_ready
if not defined C3X_VS_PATH goto fail
if not exist "%C3X_VS_PATH%\VC\Auxiliary\Build\vcvars64.bat" goto fail

call "%C3X_VS_PATH%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 goto fail
if not exist "build\helper_trial\gate2" mkdir "build\helper_trial\gate2"
if not exist "build\helper_trial\gate2\obj64" mkdir "build\helper_trial\gate2\obj64"
if /i "%~1"=="scene-only" goto scene_tools
cl /nologo /std:c++17 /EHsc /O2 /W4 /WX /bigobj /DC3X_HELPER_TRIAL /LD c3x_renderer.cpp terrain_scene_runtime.cpp environment_runtime.cpp terrain_definition_runtime.cpp scene_export.cpp frame_scheduler.cpp /Fo:build\helper_trial\gate2\obj64\ /Fe:build\helper_trial\gate2\C3XRenderer_x64.dll /link /DEF:c3x_renderer.def /IMPLIB:build\helper_trial\gate2\C3XRenderer_x64.lib d3d11.lib d3dcompiler.lib dxgi.lib gdi32.lib msimg32.lib user32.lib bcrypt.lib
if errorlevel 1 goto fail
cl /nologo /std:c++17 /EHsc /O2 /W4 /WX /wd4191 replay_inputs.cpp /Fo:build\helper_trial\gate2\obj64\ /Fe:build\helper_trial\gate2\replay_inputs_x64.exe /link user32.lib psapi.lib
if errorlevel 1 goto fail
:scene_tools
cl /nologo /std:c++17 /EHsc /O2 /W4 /WX /wd4191 helper_trial\scene_workload.cpp /Fo:build\helper_trial\gate2\obj64\scene_workload.obj /Fe:build\helper_trial\gate2\scene_workload_x64.exe /link psapi.lib d3d11.lib dxgi.lib
if errorlevel 1 goto fail

call "%C3X_VS_PATH%\VC\Auxiliary\Build\vcvars32.bat" >nul
if errorlevel 1 goto fail
if not exist "build\helper_trial\gate2\obj32" mkdir "build\helper_trial\gate2\obj32"
cl /nologo /std:c++17 /EHsc /O2 /W4 /WX /wd4191 helper_trial\scene_workload.cpp /Fo:build\helper_trial\gate2\obj32\scene_workload.obj /Fe:build\helper_trial\gate2\scene_workload_x86.exe /link /LARGEADDRESSAWARE psapi.lib d3d11.lib dxgi.lib
if errorlevel 1 goto fail
popd
exit /b 0

:fail
popd
exit /b 1
