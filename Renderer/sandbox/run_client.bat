@echo off
setlocal
pushd "%~dp0..\native"
set "C3X_RENDERER_VISUAL_PROFILE="
if not defined C3X_SANDBOX_WHOLE_WORLD set "C3X_SANDBOX_WHOLE_WORLD=1"
set "C3X_RENDERER_TRACE=0"
set "C3X_RENDERER_PREVIEW_CUSTOM_DEFINITIONS=..\..\Renderer\custom.custom_rendering.txt"
set "C3X_RENDERER_SHARED_SCENE_SURFACE=1"
set "C3X_RENDERER_WATER_MOTION=1"
set "C3X_RENDERER_WAVES=1"
set "C3X_RENDERER_PREVIEW_UNITS=1"
if not defined C3X_SANDBOX_UNITS set "C3X_SANDBOX_UNITS=1"
..\sandbox\out\client_x64.exe ..\sandbox\out\C3XReference_x64.dll ..\.. ..\default.custom_rendering.txt ..\sandbox\out\test-biq.csv ..\sandbox\out\client-start.bmp 2240 1260 17 49 128 12
exit /b %errorlevel%
