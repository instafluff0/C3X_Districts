@echo off
setlocal
set "RENDERER=%~dp0..\..\..\"
set "TOOLS_DIR=%RENDERER%tools\asset_compiler\"
set "SOURCE=\\Mac\Home\Library\Application Support\Steam\steamapps\common\Sid Meier's Civilization VI\Civ6.app\Contents\Assets\Base\Platforms\Windows\BLPs\SHARED_DATA"
if not "%C3X_CIV6_UNIT_SOURCE%"=="" set "SOURCE=%C3X_CIV6_UNIT_SOURCE%"
set "PACK=%RENDERER%packs\UnitSettlerCarrierLab"
set "CONVERTER=%RENDERER%preview\out\animation_tools\export_civ6_animation.exe"
set "CIVNEXUS=%RENDERER%third_party\CivNexus6\bin\Release\CivNexus6.exe"

call "%TOOLS_DIR%BUILD_CIV6_ANIMATION_CONVERTER.bat"
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" settler idle ANIMATION_Settler_Backpack_IdleA
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" settler move ANIMATION_Settler_Backpack_Run
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" settler fidget ANIMATION_Settler_Backpack_FidgetA
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" settler fortify ANIMATION_Settler_Backpack_Run_Stop
exit /b %errorlevel%
