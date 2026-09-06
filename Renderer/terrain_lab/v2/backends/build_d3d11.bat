@echo off
setlocal
pushd "%~dp0"
set "Q0_VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq tokens=*" %%I in (`"%Q0_VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "Q0_VS=%%I"
if not defined Q0_VS exit /b 1
call "%Q0_VS%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
if not exist "build" mkdir "build"
cl /nologo /std:c++17 /EHsc /O2 d3d11.cpp /Fo:build\ /Fe:build\d3d11.exe /link d3d11.lib d3dcompiler.lib bcrypt.lib
set "Q0_RESULT=%errorlevel%"
popd
exit /b %Q0_RESULT%
