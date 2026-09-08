@echo off
setlocal
pushd "%~dp0"
call :build
set "C3X_LAB_RESULT=%errorlevel%"
popd
exit /b %C3X_LAB_RESULT%

:build
set "C3X_LAB_VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%C3X_LAB_VSWHERE%" exit /b 1
for /f "usebackq tokens=*" %%I in (`"%C3X_LAB_VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "C3X_LAB_VS=%%I"
if not defined C3X_LAB_VS exit /b 1
call "%C3X_LAB_VS%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
if not exist "..\.cache" mkdir "..\.cache"
cl /nologo /std:c++17 /EHsc /O2 d3d11.cpp /Fo:..\.cache\backend-d3d11.obj /Fe:..\.cache\d3d11.exe /link d3d11.lib d3dcompiler.lib bcrypt.lib
exit /b %errorlevel%
