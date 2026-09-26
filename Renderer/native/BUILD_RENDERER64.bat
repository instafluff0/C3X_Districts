@echo off
setlocal
pushd "%~dp0"
if errorlevel 1 exit /b 1

rem Build the 32-bit game bridge and both 64-bit companions from this checkout.
call BUILD.bat candidate-compile no-stage
if errorlevel 1 goto fail

if defined C3X_VS_PATH goto compiler_ready
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto fail
"%VSWHERE%" -latest -prerelease -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath >"%TEMP%\c3x-renderer64-vs-path.txt"
set /p C3X_VS_PATH=<"%TEMP%\c3x-renderer64-vs-path.txt"
del "%TEMP%\c3x-renderer64-vs-path.txt" >nul 2>nul
:compiler_ready
if not defined C3X_VS_PATH (
  echo Visual Studio compiler not found. 1>&2
  goto fail
)
call "%C3X_VS_PATH%\VC\Auxiliary\Build\vcvars32.bat" >nul
if errorlevel 1 goto fail
cl /nologo /std:c++17 /EHsc /O2 /W4 /WX renderer64_startup_probe.cpp /Fo:build\renderer64_startup_probe.obj /Fe:build\renderer64_startup_probe.exe
if errorlevel 1 goto fail
call "%C3X_VS_PATH%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 goto fail
if not exist "build\renderer64\obj" mkdir "build\renderer64\obj"
cl /nologo /std:c++17 /EHsc /O2 /W4 /WX /bigobj /DC3X_HELPER_TRIAL /DC3X_RENDERER64_FRESH /LD ..\sandbox\resident_scene.cpp terrain_scene_runtime.cpp environment_runtime.cpp terrain_definition_runtime.cpp scene_export.cpp frame_scheduler.cpp /Fo:build\renderer64\obj\ /Fe:build\renderer64\C3XRenderer_x64.dll /link /DEF:c3x_renderer.def /IMPLIB:build\renderer64\C3XRenderer_x64.lib d3d11.lib d3dcompiler.lib dxgi.lib dcomp.lib gdi32.lib msimg32.lib user32.lib bcrypt.lib ole32.lib windowscodecs.lib
if errorlevel 1 goto fail
cl /nologo /std:c++17 /EHsc /O2 /W4 /WX /wd4191 helper_trial\scene_workload.cpp /Fo:build\renderer64\obj\scene_workload.obj /Fe:build\renderer64\C3XRendererHelper64.exe /link psapi.lib d3d11.lib dxgi.lib
if errorlevel 1 goto fail

if /i "%~1"=="no-stage" goto done
tasklist /fi "imagename eq Civ3Conquests.exe" /nh 2>nul | find /i "Civ3Conquests.exe" >nul
if not errorlevel 1 (
  echo Exit Civ III before staging Renderer64. 1>&2
  goto fail
)
if not exist "..\bin\renderer64" mkdir "..\bin\renderer64"
copy /y "build\candidate\C3XRenderer.dll" "..\bin\renderer64\C3XRenderer.dll" >nul
if errorlevel 1 goto fail
copy /y "build\renderer64\C3XRenderer_x64.dll" "..\bin\renderer64\C3XRenderer_x64.dll" >nul
if errorlevel 1 goto fail
copy /y "build\renderer64\C3XRendererHelper64.exe" "..\bin\renderer64\C3XRendererHelper64.exe" >nul
if errorlevel 1 goto fail
pushd ..\..
"%~dp0build\renderer64_startup_probe.exe" . "Renderer\bin\renderer64\C3XRenderer.dll"
set "C3X_STARTUP_RESULT=%errorlevel%"
popd
if not "%C3X_STARTUP_RESULT%"=="0" goto fail
echo Renderer64 bridge, renderer DLL and helper staged together.
:done
popd
exit /b 0
:fail
popd
exit /b 1
