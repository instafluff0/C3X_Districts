@echo off
setlocal
pushd "%~dp0"

if exist "local_paths.bat" call local_paths.bat

call PREPARE_BIQ_VIEWPORT.bat
if errorlevel 1 (
  popd
  exit /b 1
)

call BUILD.bat
if errorlevel 1 (
  popd
  exit /b 1
)

if not defined C3X_LAB_PACK set "C3X_LAB_PACK=..\packs\TerrainNormalized"
if not defined C3X_LAB_VEGETATION set "C3X_LAB_VEGETATION=..\packs\VegetationNormalized"
if not defined C3X_LAB_DECALS set "C3X_LAB_DECALS=..\packs\DecalsNormalized"
set "C3X_LAB_OUTPUT=..\preview\out\terrain_lab"
set "C3X_LAB_BIQ_VIEW=%C3X_LAB_OUTPUT%\test_biq_l11_marsh_96.csv"
if not exist "%C3X_LAB_OUTPUT%" mkdir "%C3X_LAB_OUTPUT%"

build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\grass_albedo.bmp" albedo 0.26 4.0 1.0 72 0.085
if errorlevel 1 (
  popd
  exit /b 1
)
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\grass_material.bmp" material 0.26 4.0 1.0 72 0.085
if errorlevel 1 (
  popd
  exit /b 1
)
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\grass_relief.bmp" relief 0.26 4.0 1.0 72 0.085
if errorlevel 1 (
  popd
  exit /b 1
)
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\grass_relief_shadow.bmp" shadow 0.26 4.0 1.0 72 0.085
if errorlevel 1 (
  popd
  exit /b 1
)
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\grass_authored_hill_standard.bmp" hill 0.26 4.0 1.0 42 0.085
if errorlevel 1 (
  popd
  exit /b 1
)
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\grass_authored_mountain_standard_01.bmp" mountain 0.26 4.0 1.0 118 0.085
if errorlevel 1 (
  popd
  exit /b 1
)
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\coast_straight_beach.bmp" coast_beach 0.26 4.0 1.0 72 0.085
if errorlevel 1 (
  popd
  exit /b 1
)
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\terrain_beauty_l9_3.bmp" beauty 0.26 4.0 1.0 72 0.085
if errorlevel 1 (
  popd
  exit /b 1
)
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\terrain_beauty_l9_3_no_relief.bmp" beauty_no_relief 0.26 4.0 1.0 72 0.085
if errorlevel 1 (
  popd
  exit /b 1
)
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\terrain_beauty_l9_3_no_water.bmp" beauty_no_water 0.26 4.0 1.0 72 0.085
if errorlevel 1 (
  popd
  exit /b 1
)
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\terrain_beauty_l9_3_thumbnail.bmp" beauty_thumbnail 0.26 4.0 1.0 72 0.085
if errorlevel 1 (
  popd
  exit /b 1
)
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\terrain_beauty_l9_4.bmp" beauty_vegetation 0.26 4.0 1.0 72 0.085 "%C3X_LAB_VEGETATION%"
if errorlevel 1 (
  popd
  exit /b 1
)
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\terrain_beauty_l9_4_vegetation_only.bmp" beauty_vegetation_only 0.26 4.0 1.0 72 0.085 "%C3X_LAB_VEGETATION%"
if errorlevel 1 (
  popd
  exit /b 1
)
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\terrain_beauty_l9_4_no_vegetation.bmp" beauty_no_vegetation 0.26 4.0 1.0 72 0.085 "%C3X_LAB_VEGETATION%"
if errorlevel 1 (
  popd
  exit /b 1
)
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\terrain_beauty_l9_4_thumbnail.bmp" beauty_vegetation_thumbnail 0.26 4.0 1.0 72 0.085 "%C3X_LAB_VEGETATION%"
if errorlevel 1 (
  popd
  exit /b 1
)
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\terrain_beauty_l9_5.bmp" beauty_shore 0.26 4.0 1.0 72 0.085 "%C3X_LAB_VEGETATION%"
if errorlevel 1 (
  popd
  exit /b 1
)
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\terrain_beauty_l9_5_no_vegetation.bmp" beauty_shore_no_vegetation 0.26 4.0 1.0 72 0.085 "%C3X_LAB_VEGETATION%"
if errorlevel 1 (
  popd
  exit /b 1
)
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\terrain_beauty_l9_5_no_water.bmp" beauty_shore_no_water 0.26 4.0 1.0 72 0.085 "%C3X_LAB_VEGETATION%"
if errorlevel 1 (
  popd
  exit /b 1
)
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\terrain_beauty_l9_5_no_surf.bmp" beauty_shore_no_surf 0.26 4.0 1.0 72 0.085 "%C3X_LAB_VEGETATION%"
if errorlevel 1 (
  popd
  exit /b 1
)
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\terrain_beauty_l9_5_thumbnail.bmp" beauty_shore_thumbnail 0.26 4.0 1.0 72 0.085 "%C3X_LAB_VEGETATION%"
if errorlevel 1 (
  popd
  exit /b 1
)
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\terrain_beauty_l9_promotion.bmp" beauty_promotion 0.26 4.0 1.0 72 0.085 "%C3X_LAB_VEGETATION%"
if errorlevel 1 (
  popd
  exit /b 1
)
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\terrain_beauty_l9_promotion_no_vegetation.bmp" beauty_promotion_no_vegetation 0.26 4.0 1.0 72 0.085 "%C3X_LAB_VEGETATION%"
if errorlevel 1 (
  popd
  exit /b 1
)
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\terrain_beauty_l9_promotion_no_water.bmp" beauty_promotion_no_water 0.26 4.0 1.0 72 0.085 "%C3X_LAB_VEGETATION%"
if errorlevel 1 (
  popd
  exit /b 1
)
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\terrain_beauty_l9_promotion_no_surf.bmp" beauty_promotion_no_surf 0.26 4.0 1.0 72 0.085 "%C3X_LAB_VEGETATION%"
if errorlevel 1 (
  popd
  exit /b 1
)
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\terrain_beauty_l9_promotion_thumbnail.bmp" beauty_promotion_thumbnail 0.26 4.0 1.0 72 0.085 "%C3X_LAB_VEGETATION%"
if errorlevel 1 (
  popd
  exit /b 1
)
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\terrain_beauty_l10_dunes.bmp" beauty_dunes 0.26 4.0 1.0 72 0.085 "%C3X_LAB_VEGETATION%" "%C3X_LAB_DECALS%"
if errorlevel 1 (
  popd
  exit /b 1
)
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\terrain_beauty_l10_dunes_no_dunes.bmp" beauty_dunes_no_dunes 0.26 4.0 1.0 72 0.085 "%C3X_LAB_VEGETATION%" "%C3X_LAB_DECALS%"
if errorlevel 1 (
  popd
  exit /b 1
)
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\terrain_beauty_l10_dunes_only.bmp" beauty_dunes_only 0.26 4.0 1.0 72 0.085 "%C3X_LAB_VEGETATION%" "%C3X_LAB_DECALS%"
if errorlevel 1 (
  popd
  exit /b 1
)
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\terrain_beauty_l10_dunes_thumbnail.bmp" beauty_dunes_thumbnail 0.26 4.0 1.0 72 0.085 "%C3X_LAB_VEGETATION%" "%C3X_LAB_DECALS%"
if errorlevel 1 (
  popd
  exit /b 1
)
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\terrain_beauty_l11_marsh.bmp" beauty_marsh 0.26 4.0 1.0 72 0.085 "%C3X_LAB_VEGETATION%" "%C3X_LAB_DECALS%" "%C3X_LAB_BIQ_VIEW%"
if errorlevel 1 (
  popd
  exit /b 1
)
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\terrain_beauty_l11_marsh_no_marsh.bmp" beauty_marsh_no_marsh 0.26 4.0 1.0 72 0.085 "%C3X_LAB_VEGETATION%" "%C3X_LAB_DECALS%" "%C3X_LAB_BIQ_VIEW%"
if errorlevel 1 (
  popd
  exit /b 1
)
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\terrain_beauty_l11_marsh_only.bmp" beauty_marsh_only 0.26 4.0 1.0 72 0.085 "%C3X_LAB_VEGETATION%" "%C3X_LAB_DECALS%" "%C3X_LAB_BIQ_VIEW%"
if errorlevel 1 (
  popd
  exit /b 1
)
build\terrain_lab.exe "%C3X_LAB_PACK%" "terrain_lab.hlsl" "%C3X_LAB_OUTPUT%\terrain_beauty_l11_marsh_thumbnail.bmp" beauty_marsh_thumbnail 0.26 4.0 1.0 72 0.085 "%C3X_LAB_VEGETATION%" "%C3X_LAB_DECALS%" "%C3X_LAB_BIQ_VIEW%"
set "C3X_LAB_RUN_RESULT=%errorlevel%"
popd
exit /b %C3X_LAB_RUN_RESULT%
