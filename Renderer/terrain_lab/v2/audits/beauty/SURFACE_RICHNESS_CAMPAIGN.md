# Terrain surface richness campaign

The user explicitly adopted the proposed next goal: terrain surface richness
at matched gameplay scale. Improve material bindings, texture scale and
filtering, then grass/soil/rock variation, surface relief and small shadows
together. Near-reference richness is the target; no Civ VI equivalence or
human approval is inferred. All milestone gates remain unchanged.

## Preserved inputs and visual acceptance

Use the fixed coastal, inland and wilderness 100-tile test.biq regions,
unchanged cameras, geometry, placement and output sizes. `river-corridor-r3`
is the latest complete four-region candidate; `shadow-receiver-r1` remains
the broader preserved 32-frame checkpoint. River pool r4 is an unfinished
diagnostic and is not the selected starting point. The integration preparation
archive is immutable. No units, cities, improvements or native work in scope.

Compare native gameplay pixels first, with noon/midnight at both fixed zooms.
Use canonical `sea_and_shore.png` and `mountain.png`, explicitly distinguishing
reference closeup scale from gameplay scale. Closeups diagnose defects only.
Select an additional previously untuned region before inspecting its result;
freshcanopy/freshshadow and earlier holdouts are now regression witnesses.

Repeat: render, inspect beside reference and previous best, identify the three
largest gaps, implement bounded fixes, compose, reject regressions. Require
changed pixels with a visual explanation. Engineering tests support but do
not constitute acceptance. Investigate geometry/material/lighting causes if
surface tuning does not improve a defect.

## Initial diagnosis

1. Ground lacks intermediate grass/soil/stone structure and convincing fine
   relief; its color mottling often reads as a flat sheet.
2. Mountains have large silhouettes but insufficient granular relief and
   coherent detail across steep faces. Their triplanar color path and height
   path do not currently share one material projection.
3. Smooth blends lack the smaller patches, exposed earth and rocky breaks
   visible at material boundaries in the reference.

The source grassland color and height textures are 4096 square with 11 mips.
The existing main material normal uses an unnormalized difference across
one LOD0 texel, scaled by 4, even after filtering to a much coarser gameplay
footprint. That makes response depend on source image resolution and weakens
the visible filtered relief. Supplemental detail uses the same height at
3x/8x, conservative amplitudes, and fades off raised bodies and slopes.

`surface-richness-r1` isolates a footprint-aware source-height surface gradient
transformed through actual UV/world derivatives. It uses existing geometry,
source art, lighting and shadow packets. Its amplitude is a Lab interpretation,
not a recovered Civ VI engine parameter. The first replay is diagnostic only;
the full combined matrix, holdout, stable crop/zoom checks and further material
layering remain required.

## Source completeness requirement

The user explicitly requires ALL applicable ground textures and correct layering
to be checked. The initial [ground-layer audit](GROUND_LAYER_FINDINGS.md) found
missing high-ground channels, incomplete mountain material channels, and full-
atlas repetition in place of individual grass decals. Resolve source bindings
and atlas transforms before further amplitude tuning. The first gradient
diagnostic is rejected and the preserved candidate remains unchanged.
