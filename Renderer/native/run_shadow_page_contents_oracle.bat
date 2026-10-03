@echo off
setlocal
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "C3X_SHADOW_VS_RECORD=%TEMP%\c3x-shadow-oracle-vs.txt"
"%VSWHERE%" -latest -prerelease -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath >"%C3X_SHADOW_VS_RECORD%"
set /p C3X_SHADOW_VS=<"%C3X_SHADOW_VS_RECORD%"
del "%C3X_SHADOW_VS_RECORD%" >nul 2>nul
if not defined C3X_SHADOW_VS exit /b 2
call "%C3X_SHADOW_VS%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 2
pushd "%~dp0..\.."
if not exist "Renderer\native\build" mkdir "Renderer\native\build"
cl /nologo /std:c++17 /EHsc /O1 /W3 /I . Renderer\native\test_shadow_preparation.cpp /Fo:Renderer\native\build\shadow_preparation.obj /Fe:Renderer\native\build\shadow_preparation.exe
if errorlevel 1 exit /b 1
Renderer\native\build\shadow_preparation.exe
if errorlevel 1 exit /b 1
cl /nologo /std:c++17 /EHsc /O1 /W3 /I . Renderer\native\test_shadow_page_contents.cpp /Fo:Renderer\native\build\shadow_page_contents.obj /Fe:Renderer\native\build\shadow_page_contents.exe
if errorlevel 1 exit /b 1
Renderer\native\build\shadow_page_contents.exe
exit /b %errorlevel%
