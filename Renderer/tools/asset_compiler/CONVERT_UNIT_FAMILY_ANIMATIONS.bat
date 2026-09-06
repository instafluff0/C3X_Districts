@echo off
setlocal

set "TOOLS_DIR=%~dp0"
set "SOURCE=Z:\Library\Application Support\Steam\steamapps\common\Sid Meier's Civilization VI\Civ6.app\Contents\Assets\Base\Platforms\Windows\BLPs\SHARED_DATA"
if not exist "%SOURCE%" set "SOURCE=\\Mac\Home\Library\Application Support\Steam\steamapps\common\Sid Meier's Civilization VI\Civ6.app\Contents\Assets\Base\Platforms\Windows\BLPs\SHARED_DATA"
set "PACK=%TOOLS_DIR%..\..\packs\UnitFamilyLab"
set "CONVERTER=%TOOLS_DIR%..\..\preview\out\animation_tools\export_civ6_animation.exe"
set "CIVNEXUS=%TOOLS_DIR%..\..\third_party\CivNexus6\bin\Release\CivNexus6.exe"

call "%TOOLS_DIR%BUILD_CIV6_ANIMATION_CONVERTER.bat"
if errorlevel 1 exit /b %errorlevel%

call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" archer idle ANIMATION_Archer_StandingIdleA
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" archer move ANIMATION_UnitMedium_Run_BowAndArrowA
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" archer attack ANIMATION_Archer_FireArrowStraightA
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" archer death ANIMATION_UnitMedium_ProjectileDeathFlyBackA
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" archer fidget ANIMATION_Archer_StandingIdleC
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" archer fortify ANIMATION_Archer_BraceA
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" archer defend ANIMATION_Crossbowman_ReactA
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" archer victory ANIMATION_Crossbowman_VictoryA
if errorlevel 1 exit /b %errorlevel%

call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" swordsman idle ANIMATION_Swordsman_IdleB
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" swordsman move ANIMATION_UnitMedium_Run_SwordAndShieldA
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" swordsman attack ANIMATION_Swordsman_AttackA
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" swordsman death ANIMATION_Warrior_DeathMeleeA
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" swordsman fidget ANIMATION_Swordsman_IdleD
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" swordsman fortify ANIMATION_Warrior_BraceA
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" swordsman defend ANIMATION_Swordsman_ReactA
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" swordsman victory ANIMATION_Hoplite_VictoryA
if errorlevel 1 exit /b %errorlevel%

call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" infantry idle ANIMATION_Infantry_IdleA
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" infantry move ANIMATION_Infantry_StrafeFwdA
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" infantry attack ANIMATION_Infantry_GunBurstSights
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" infantry death ANIMATION_Infantry_DeathA
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" infantry fidget ANIMATION_Infantry_IdleD
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" infantry fortify ANIMATION_Infantry_BraceA
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" infantry defend ANIMATION_Infantry_ReactB
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" infantry victory ANIMATION_Infantry_VictoryA
if errorlevel 1 exit /b %errorlevel%

call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" fighter idle ANIMATION_Fighter_IdleGround
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" fighter move ANIMATION_Fighter_IdleA
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" fighter attack ANIMATION_Fighter_AttackA
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" fighter death ANIMATION_Fighter_Death
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" fighter takeoff ANIMATION_Fighter_Takeoff
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" fighter landing ANIMATION_Fighter_Landing
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" fighter turn_left ANIMATION_Fighter_TurnLeft
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" fighter turn_right ANIMATION_Fighter_TurnRight
if errorlevel 1 exit /b %errorlevel%

call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" galley idle ANIMATION_Galley_IdleA
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" galley move ANIMATION_Galley_RunA
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" galley attack ANIMATION_Galley_AttackA
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" galley death ANIMATION_Galley_DeathA
if errorlevel 1 exit /b %errorlevel%
call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" galley defend ANIMATION_Galley_RunStopA
if errorlevel 1 exit /b %errorlevel%

echo Converted 37 unique unit-family animation clips for 44 logical action bindings.
exit /b 0
