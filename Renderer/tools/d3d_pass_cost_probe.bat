@echo off
setlocal
rem Builds d3d_pass_cost_probe.cpp into %TEMP%\c3x-pass-probe and runs it.
pushd "%~dp0"
if errorlevel 1 exit /b 1
set "OUT=%TEMP%\c3x-pass-probe"
if not exist "%OUT%" mkdir "%OUT%"
if defined C3X_VS_PATH goto compiler_ready
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto fail
"%VSWHERE%" -latest -prerelease -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath >"%OUT%\vs-path.txt"
set /p C3X_VS_PATH=<"%OUT%\vs-path.txt"
:compiler_ready
call "%C3X_VS_PATH%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 goto fail
cl /nologo /std:c++17 /EHsc /O2 /W4 d3d_pass_cost_probe.cpp /Fo:"%OUT%\\" /Fe:"%OUT%\d3d_pass_cost_probe.exe" /link d3d11.lib d3dcompiler.lib >"%OUT%\build.log"
if errorlevel 1 (type "%OUT%\build.log" & goto fail)
"%OUT%\d3d_pass_cost_probe.exe"
set "RESULT=%errorlevel%"
popd
rmdir /s /q "%OUT%" >nul 2>nul
exit /b %RESULT%
:fail
popd
rmdir /s /q "%OUT%" >nul 2>nul
exit /b 1
