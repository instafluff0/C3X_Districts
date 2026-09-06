@echo off
setlocal

set "TOOLS_DIR=%~dp0"
set "SOURCE_SETS=%TOOLS_DIR%compound_unit_source_sets.json"
set "PACK=%TOOLS_DIR%..\..\packs\CompoundUnitLab"
set "REPORT=%TOOLS_DIR%..\..\preview\out\units\compound_unit_build.json"

set "PYTHON_COMMAND="
where py >nul 2>nul && set "PYTHON_COMMAND=py"
if not defined PYTHON_COMMAND where python >nul 2>nul && set "PYTHON_COMMAND=python"
if not defined PYTHON_COMMAND where python3 >nul 2>nul && set "PYTHON_COMMAND=python3"
if not defined PYTHON_COMMAND (
  echo Python 3 was not found on PATH. 1>&2
  exit /b 1
)

%PYTHON_COMMAND% "%TOOLS_DIR%compound_unit_asset_importer.py" --source-sets "%SOURCE_SETS%" --pack "%PACK%" --report "%REPORT%"
if errorlevel 1 exit /b %errorlevel%

call "%TOOLS_DIR%BUILD_CIV6_ANIMATION_CONVERTER.bat"
if errorlevel 1 exit /b %errorlevel%

powershell -NoProfile -ExecutionPolicy Bypass -File "%TOOLS_DIR%CONVERT_COMPOUND_UNIT_ANIMATIONS.ps1" -SourceSets "%SOURCE_SETS%" -Pack "%PACK%"
if errorlevel 1 exit /b %errorlevel%

%PYTHON_COMMAND% "%TOOLS_DIR%compound_unit_asset_importer.py" --source-sets "%SOURCE_SETS%" --pack "%PACK%" --report "%REPORT%" --require-animations
exit /b %errorlevel%
