@echo off
setlocal
pushd "%~dp0"
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" exit /b 1
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -prerelease -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "C3X_SANDBOX_VS=%%i"
if not defined C3X_SANDBOX_VS exit /b 1
call "%C3X_SANDBOX_VS%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
if not exist out mkdir out
if not exist out\obj mkdir out\obj
cl /nologo /std:c++17 /EHsc /O2 /W4 /WX /bigobj /DC3X_HELPER_TRIAL /LD resident_scene.cpp ..\native\terrain_scene_runtime.cpp ..\native\environment_runtime.cpp ..\native\terrain_definition_runtime.cpp ..\native\scene_export.cpp ..\native\frame_scheduler.cpp /Fo:out\obj\ /Fe:out\C3XReference_x64.dll /link /DEF:..\native\c3x_renderer.def /IMPLIB:out\C3XReference_x64.lib d3d11.lib d3dcompiler.lib dxgi.lib dcomp.lib gdi32.lib msimg32.lib user32.lib bcrypt.lib
if errorlevel 1 exit /b 1
cl /nologo /std:c++17 /EHsc /O2 /W4 /WX /I..\native reference_x64.cpp /Fo:out\reference_x64.obj /Fe:out\reference_x64.exe /link gdi32.lib user32.lib
if errorlevel 1 exit /b 1
cl /nologo /std:c++17 /EHsc /O2 /W4 /WX /DC3X_SANDBOX_CLIENT /I..\native reference_x64.cpp client_x64.cpp /Fo:out\obj\ /Fe:out\client_x64.exe /link gdi32.lib user32.lib
if errorlevel 1 exit /b 1
call "%C3X_SANDBOX_VS%\VC\Auxiliary\Build\vcvars32.bat" >nul
if errorlevel 1 exit /b 1
cl /nologo /std:c++17 /EHsc /O2 /W4 /WX synthetic_host_x86.cpp /Fo:out\synthetic_host_x86.obj /Fe:out\synthetic_host_x86.exe
exit /b %errorlevel%
