@echo off
setlocal

if "%~3"=="" (
  echo usage: CONVERT_UNIT_FAMILY_ANIMATION_ONE unit action source-clip 1>&2
  exit /b 2
)
if "%SOURCE%"=="" exit /b 2
if "%PACK%"=="" exit /b 2
if "%CONVERTER%"=="" exit /b 2
if "%CIVNEXUS%"=="" exit /b 2

set "OUTPUT=%PACK%\animations\unit\%~1\%~2.c3anim"
if not exist "%SOURCE%\%~3" (
  echo Missing unit animation source: "%SOURCE%\%~3" 1>&2
  exit /b 1
)
if not exist "%PACK%\animations\unit\%~1" mkdir "%PACK%\animations\unit\%~1"
"%CONVERTER%" "%CIVNEXUS%" "%SOURCE%\%~3" "%OUTPUT%" 0.01
exit /b %errorlevel%
