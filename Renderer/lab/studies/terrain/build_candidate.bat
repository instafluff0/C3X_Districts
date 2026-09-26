@echo off
setlocal
pushd "%~dp0\..\..\..\native"
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" exit /b 2
set "C3X_TERRAIN_VS_RECORD=%TEMP%\c3x-terrain-vs-path.txt"
"%VSWHERE%" -latest -prerelease -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath >"%C3X_TERRAIN_VS_RECORD%"
set /p C3X_TERRAIN_VS=<"%C3X_TERRAIN_VS_RECORD%"
del "%C3X_TERRAIN_VS_RECORD%" >nul 2>nul
if not defined C3X_TERRAIN_VS exit /b 2
call "%C3X_TERRAIN_VS%\VC\Auxiliary\Build\vcvars32.bat" >nul
if errorlevel 1 exit /b 2
if not exist "build\grass_plains_obj" mkdir "build\grass_plains_obj"
if not exist "build\grass_plains_candidate" mkdir "build\grass_plains_candidate"
cl /nologo /std:c++17 /EHsc /O2 /W4 /WX /LD c3x_renderer.cpp terrain_scene_runtime.cpp environment_runtime.cpp terrain_definition_runtime.cpp scene_export.cpp frame_scheduler.cpp /Fo:build\grass_plains_obj\ /Fe:build\grass_plains_candidate\C3XRenderer.dll /link /DEF:c3x_renderer.def /IMPLIB:build\grass_plains_candidate\C3XRenderer.lib d3d11.lib d3dcompiler.lib dxgi.lib dcomp.lib gdi32.lib msimg32.lib user32.lib bcrypt.lib
exit /b %errorlevel%
