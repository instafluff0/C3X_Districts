# Rivers

Lab-only until the user accepts it. Game builds keep the current rivers:
the draw-state and delta changes sit behind `#ifndef C3X_RENDERER64_FRESH`, and
the game reads its own pinned shader pack.

## Changes

- **Banks never cover objects.** The river layer (ground kind 9) wrote a depth
  pulled about 27 units toward the camera across its whole bank band. Its
  ground basis weighted height far less than the natural terrain (0.35 against
  1.73 depth units per unit of relief), so it needed that bias. Bridges, farms,
  routes and object shadows drawn after it failed the depth test there; trees
  drawn before it were painted over. Now the generated hydrology shaders define
  `C3X_RIVER_NATURAL_DEPTH`: river vertices sort on the natural basis with a
  0.004 x H bias, and `submit_geometry` draws the river with the routes'
  test-only decal depth state. Tests: `Renderer/native/test_river_depth.py`.
- **Ocean optics.** The river material uses the in-game sea's response:
  Fresnel-weighted sky, the mirrored scene (lifted by the river's height above
  the sea plane, faded on raised reaches) and the open-sea sun and moon glare.
  The glare uses a calm normal so it reads as a smooth sheen.
- **Stream lines.** Thin pale lines parallel to the banks, broken into dashes
  that drift downstream along the drainage tangent; a broad world field picks
  occasional livelier reaches.
- **Natural banks.** The waterline wanders between the ground grid's samples.
  A translucent dark layer darkens the land beside the water instead of
  painting a soil ribbon, and authored river gravel collects along it.
  Shallows keep the channel color, so no pale rim outlines the water.
- **Delta.** At a mouth, two narrower distributaries leave the last reach and
  fan out to the sea beside the main outlet (`river_corridor.h`).
  Tests: `Renderer/native/test_river_delta.py`.

## Lab

```sh
python3 Renderer/lab/studies/rivers/study.py render after
python3 Renderer/lab/studies/rivers/study.py render after --hour 18
python3 Renderer/lab/studies/rivers/study.py motion after
$C3X_RENDERER_PYTHON Renderer/lab/studies/rivers/study.py sheet before after
$C3X_RENDERER_PYTHON Renderer/lab/studies/rivers/study.py gif after
```

`--dll` pins a candidate copy. `--shader-root NAME` renders with a complete
private shader tree under `Renderer/lab/out/river-study/NAME`, so tuning never
changes the checkout's generated shaders that other sessions render with.
The case lives in `cases.py` (category "infrastructure"): a meandering river
through farmland with bridged road and railroad, hills and a mine on the far
bank, forest on both banks, resources and a coast.

The rivers category's own water-motion lifecycle witness currently fails on
the unchanged checkout too (only the first of 48 frames changes); the study
keeps those stills and records the failure.

## Promotion (after acceptance)

- Remove the two `#ifndef C3X_RENDERER64_FRESH` gates (`c3x_renderer.cpp`
  `submit_geometry`, `river_corridor.h`).
- `sandbox/fresh_pipeline.h`: draw `geometry_river` with the decal depth state
  in `draw_layer` (as routes); its depth-only overlay prepass then writes
  nothing and can be skipped.
- Overlay the river hunks (translated_depth define and branch, river branch of
  `q3_water_material`) into the pinned `Renderer64ResidentRuntime` pack with a
  dated backup and receipt, as the route and resource overlays do.
