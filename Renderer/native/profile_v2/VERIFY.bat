@echo off
setlocal
pushd "%~dp0"
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq tokens=*" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "C3X_VS_PATH=%%I"
if not defined C3X_VS_PATH exit /b 1
call "%C3X_VS_PATH%\VC\Auxiliary\Build\vcvars32.bat" >nul
if errorlevel 1 exit /b 1
if not exist "..\build" mkdir "..\build"
cl /nologo /std:c++17 /EHsc /O2 /W4 /WX verify_d3d11.cpp /Fo:..\build\verify_profile_v2.obj /Fe:..\build\verify_profile_v2.exe /link d3d11.lib d3dcompiler.lib
if errorlevel 1 exit /b 1
..\build\verify_profile_v2.exe
set "C3X_VERIFY_RESULT=%errorlevel%"
popd
exit /b %C3X_VERIFY_RESULT%
