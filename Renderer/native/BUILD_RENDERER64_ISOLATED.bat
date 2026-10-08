@echo off
setlocal
rem Build the matching bridge/x64 renderer/helper trio into a private output
rem directory with the same flags as BUILD_RENDERER64.bat, so concurrent Lab
rem candidate builds cannot lock or replace these objects. Optional "stage"
rem copies the trio into Renderer\bin\renderer64 with the same guards.
pushd "%~dp0"
if errorlevel 1 exit /b 1
set "OUT=build\isolated-trio"
if defined C3X_VS_PATH goto compiler_ready
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto fail
"%VSWHERE%" -latest -prerelease -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath >"%TEMP%\c3x-isolated-vs-path.txt"
set /p C3X_VS_PATH=<"%TEMP%\c3x-isolated-vs-path.txt"
del "%TEMP%\c3x-isolated-vs-path.txt" >nul 2>nul
:compiler_ready
if not defined C3X_VS_PATH goto fail
if not exist "%OUT%\x86" mkdir "%OUT%\x86"
if not exist "%OUT%\x64" mkdir "%OUT%\x64"
call "%C3X_VS_PATH%\VC\Auxiliary\Build\vcvars32.bat" >nul
if errorlevel 1 goto fail
cl /nologo /std:c++17 /EHsc /O2 /W4 /WX /bigobj /LD c3x_renderer.cpp terrain_scene_runtime.cpp environment_runtime.cpp terrain_definition_runtime.cpp scene_export.cpp frame_scheduler.cpp /Fo:%OUT%\x86\ /Fe:%OUT%\C3XRenderer.dll /link /DEF:c3x_renderer.def /MAP:%OUT%\x86\C3XRenderer.map /IMPLIB:%OUT%\x86\C3XRenderer.lib d3d11.lib d3dcompiler.lib dxgi.lib dcomp.lib gdi32.lib msimg32.lib user32.lib bcrypt.lib
if errorlevel 1 goto fail
cl /nologo /std:c++17 /EHsc /O2 /W4 /WX renderer64_startup_probe.cpp /Fo:%OUT%\x86\renderer64_startup_probe.obj /Fe:%OUT%\renderer64_startup_probe.exe
if errorlevel 1 goto fail
call "%C3X_VS_PATH%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 goto fail
cl /nologo /std:c++17 /EHsc /O2 /W4 /WX /bigobj /DC3X_HELPER_TRIAL /DC3X_RENDERER64_FRESH /LD ..\sandbox\resident_scene.cpp terrain_scene_runtime.cpp environment_runtime.cpp terrain_definition_runtime.cpp scene_export.cpp frame_scheduler.cpp /Fo:%OUT%\x64\ /Fe:%OUT%\C3XRenderer_x64.dll /link /DEF:c3x_renderer.def /IMPLIB:%OUT%\x64\C3XRenderer_x64.lib d3d11.lib d3dcompiler.lib dxgi.lib dcomp.lib gdi32.lib msimg32.lib user32.lib bcrypt.lib ole32.lib windowscodecs.lib
if errorlevel 1 goto fail
cl /nologo /std:c++17 /EHsc /O2 /W4 /WX /wd4191 helper_trial\scene_workload.cpp /Fo:%OUT%\x64\scene_workload.obj /Fe:%OUT%\C3XRendererHelper64.exe /link psapi.lib d3d11.lib dxgi.lib user32.lib
if errorlevel 1 goto fail
if /i not "%~1"=="stage" goto done
tasklist /fi "imagename eq Civ3Conquests.exe" /nh 2>nul | find /i "Civ3Conquests.exe" >nul
if not errorlevel 1 (
  echo Exit Civ III before staging Renderer64. 1>&2
  goto fail
)
if not exist "..\bin\renderer64" mkdir "..\bin\renderer64"
copy /y "%OUT%\C3XRenderer.dll" "..\bin\renderer64\C3XRenderer.dll" >nul
if errorlevel 1 goto fail
copy /y "%OUT%\C3XRenderer_x64.dll" "..\bin\renderer64\C3XRenderer_x64.dll" >nul
if errorlevel 1 goto fail
copy /y "%OUT%\C3XRendererHelper64.exe" "..\bin\renderer64\C3XRendererHelper64.exe" >nul
if errorlevel 1 goto fail
pushd ..\..
"%~dp0%OUT%\renderer64_startup_probe.exe" . "Renderer\bin\renderer64\C3XRenderer.dll"
set "C3X_STARTUP_RESULT=%errorlevel%"
popd
if not "%C3X_STARTUP_RESULT%"=="0" goto fail
echo Isolated Renderer64 trio staged together.
:done
popd
exit /b 0
:fail
popd
exit /b 1
