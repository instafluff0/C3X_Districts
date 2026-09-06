@echo off
setlocal
pushd "%~dp0"

if exist "local_paths.bat" call local_paths.bat

call PREPARE_BIQ_VIEWPORT.bat
if errorlevel 1 goto failed
call BUILD.bat
if errorlevel 1 goto failed

if not defined C3X_LAB_PACK set "C3X_LAB_PACK=..\packs\TerrainNormalized"
if not defined C3X_LAB_VEGETATION set "C3X_LAB_VEGETATION=..\packs\VegetationNormalized"
if not defined C3X_LAB_DECALS set "C3X_LAB_DECALS=..\packs\DecalsNormalized"
set "C3X_LAB_OUTPUT=..\preview\out\terrain_lab"
set "C3X_LAB_BIQ_VIEW=%C3X_LAB_OUTPUT%\test_biq_l11_marsh_96.csv"
if not exist "%C3X_LAB_OUTPUT%" mkdir "%C3X_LAB_OUTPUT%"

build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\terrain_beauty_l11_marsh.bmp" beauty_marsh 0.26 4.0 1.0 72 0.085 "%C3X_LAB_VEGETATION%" "%C3X_LAB_DECALS%" "%C3X_LAB_BIQ_VIEW%"
if errorlevel 1 goto failed
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\terrain_beauty_l11_marsh_no_marsh.bmp" beauty_marsh_no_marsh 0.26 4.0 1.0 72 0.085 "%C3X_LAB_VEGETATION%" "%C3X_LAB_DECALS%" "%C3X_LAB_BIQ_VIEW%"
if errorlevel 1 goto failed
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\terrain_beauty_l11_marsh_only.bmp" beauty_marsh_only 0.26 4.0 1.0 72 0.085 "%C3X_LAB_VEGETATION%" "%C3X_LAB_DECALS%" "%C3X_LAB_BIQ_VIEW%"
if errorlevel 1 goto failed
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\terrain_beauty_l11_marsh_thumbnail.bmp" beauty_marsh_thumbnail 0.26 4.0 1.0 72 0.085 "%C3X_LAB_VEGETATION%" "%C3X_LAB_DECALS%" "%C3X_LAB_BIQ_VIEW%"
set "C3X_LAB_RESULT=%errorlevel%"
popd
exit /b %C3X_LAB_RESULT%

:failed
popd
exit /b 1
