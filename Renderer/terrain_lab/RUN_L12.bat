@echo off
setlocal
pushd "%~dp0"

if exist "local_paths.bat" call local_paths.bat

call PREPARE_L12_BIQ_VIEWPORT.bat
if errorlevel 1 goto failed
call BUILD.bat
if errorlevel 1 goto failed

if not defined C3X_LAB_PACK set "C3X_LAB_PACK=..\packs\TerrainNormalized"
if not defined C3X_LAB_VEGETATION set "C3X_LAB_VEGETATION=..\packs\VegetationNormalized"
if not defined C3X_LAB_DECALS set "C3X_LAB_DECALS=..\packs\DecalsNormalized"
if not defined C3X_LAB_TERRAIN_ELEMENTS set "C3X_LAB_TERRAIN_ELEMENTS=..\packs\TerrainElementsNormalized"
set "C3X_LAB_OUTPUT=..\preview\out\terrain_lab"
set "C3X_LAB_BIQ_VIEW=%C3X_LAB_OUTPUT%\test_biq_l12_volcano_192.csv"
if not exist "%C3X_LAB_OUTPUT%" mkdir "%C3X_LAB_OUTPUT%"

build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\terrain_beauty_l12_volcano.bmp" beauty_volcano 0.26 4.0 1.0 72 0.085 "%C3X_LAB_VEGETATION%" "%C3X_LAB_DECALS%" "%C3X_LAB_BIQ_VIEW%" "%C3X_LAB_TERRAIN_ELEMENTS%"
if errorlevel 1 goto failed
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\terrain_beauty_l12_volcano_no_volcano.bmp" beauty_volcano_no_volcano 0.26 4.0 1.0 72 0.085 "%C3X_LAB_VEGETATION%" "%C3X_LAB_DECALS%" "%C3X_LAB_BIQ_VIEW%" "%C3X_LAB_TERRAIN_ELEMENTS%"
if errorlevel 1 goto failed
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\terrain_beauty_l12_volcano_only.bmp" beauty_volcano_only 0.26 4.0 1.0 72 0.085 "%C3X_LAB_VEGETATION%" "%C3X_LAB_DECALS%" "%C3X_LAB_BIQ_VIEW%" "%C3X_LAB_TERRAIN_ELEMENTS%"
if errorlevel 1 goto failed
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\terrain_beauty_l12_volcano_thumbnail.bmp" beauty_volcano_thumbnail 0.26 4.0 1.0 72 0.085 "%C3X_LAB_VEGETATION%" "%C3X_LAB_DECALS%" "%C3X_LAB_BIQ_VIEW%" "%C3X_LAB_TERRAIN_ELEMENTS%"
set "C3X_LAB_RESULT=%errorlevel%"
popd
exit /b %C3X_LAB_RESULT%

:failed
popd
exit /b 1
