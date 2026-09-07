@echo off
setlocal
set "C3X_PROBE_VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq tokens=*" %%I in (`"%C3X_PROBE_VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "C3X_PROBE_VS=%%I"
if not defined C3X_PROBE_VS exit /b 1
call "%C3X_PROBE_VS%\VC\Auxiliary\Build\vcvars32.bat" >nul
if errorlevel 1 exit /b 1
cl /nologo /std:c++17 /EHsc /O2 bake_source_occlusion.cpp /Fo:..\audits\beauty\out\ground-shader-source\bake-ao.obj /Fe:..\audits\beauty\out\ground-shader-source\bake-ao.exe /link d3d11.lib
if errorlevel 1 exit /b 1
pushd ..\audits\beauty\out\ground-shader-source
bake-ao.exe
exit /b %errorlevel%
