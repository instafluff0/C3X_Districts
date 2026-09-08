@echo off
setlocal
pushd "%~dp0"
rem Compatibility for retained source probes; the current Lab owns this backend.
call "..\..\..\lab\backends\build_d3d11.bat"
if errorlevel 1 exit /b 1
if not exist "build" mkdir "build"
copy /y "..\..\..\lab\.cache\d3d11.exe" "build\d3d11.exe" >nul
set "C3X_LAB_RESULT=%errorlevel%"
popd
exit /b %C3X_LAB_RESULT%
