@echo off
setlocal

set "TOOLS_DIR=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%TOOLS_DIR%CONVERT_RESOURCE_ANIMATIONS.ps1"
exit /b %errorlevel%
