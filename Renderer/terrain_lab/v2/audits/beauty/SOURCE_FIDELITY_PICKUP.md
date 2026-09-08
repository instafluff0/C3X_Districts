# Consolidated source-fidelity natural-scene pickup

## Result: accepted Mac Lab composition

`source-fidelity-r11/inland` is the current 100-tile natural-scene appearance
target. It renders on macOS Metal with the exact accepted terrain, mountain and
forest providers, shared opacity-aware directional shadows, and retained river
and water geometry. It replaces the rejected r4 diagnostic for Integration
pickup; it is not itself a Civ III or Windows integration result.

- [Current native-size frame](out/source-fidelity-r11/inland/h12-z1-pan00.png)
- [Lossless r4-to-r11 comparison](out/source-fidelity-r11/inland/comparison.png)

Cities are deliberately absent and no city source, selection or appearance was
changed. Integration must use authoritative building footprints to enforce the
confirmed `ClipBuildings=true` forest rule before emitting trees.

## What is actually composed

- `fixtures/beauty/source-fidelity-r2/inland/terrain.module.json` invokes
  `systems/relief/beauty_terrain.cpp` and its source-material shader directly.
  It is no longer the `frozen_l21` approximation.
- The terrain mesh uses the Civ V Environment Skin grass material stack and
  independently seeded standard-hill relief. Each hill receives varied source
  height sampling and a deterministic weighted 3/2/2 family of irregular rock
  footprints. The footprint/material combination remains a labeled Lab
  inference, not a recovered Firaxis shader equation.
- `mountain.module.json` invokes `beauty_mountain.cpp` directly. Every mountain
  uses one of five authored 256 x 256 macro height/blend variants and the full
  2K base/top/snow color, height and specular stacks with triplanar response.
- `forest.module.json` invokes the accepted BeautyStudies object path: 22 source
  bodies, 25 ArtDef recipes, total `Count` weight 180, uniform source scale,
  packed normals, material address modes and `Generic_OPAC` cutouts.
- Forest geometry now registers as caster + receiver + alpha cutout in the Q6
  shared shadow field. The r4 circular contact-disk substitute is removed;
  visible shadows are projected from the actual opacity-tested tree triangles.
- `hydrology.module.json` redraws the retained river/water layer after replacement
  relief while explicitly suppressing its old hill and mountain classifications.
  This restores the river network without bringing back low-detail plateaus.
- Final output remains scene-linear with 4x MSAA, 16x anisotropy, 2x render scale,
  -1 mip bias and one linear reconstruction pass.

The Warrior remains a separate accepted witness because this natural-scene
fixture does not render units. Integration must still carry its packed normals,
per-material repeat/clamp modes and uniform XYZ scale.

## Reproduction

```sh
python3 Renderer/terrain_lab/v2/app/runner.py compose \
  --fixture Renderer/terrain_lab/v2/fixtures/beauty/source-fidelity-r2/inland/fixture.json \
  --candidate source-fidelity-r11-inland \
  --output Renderer/terrain_lab/v2/audits/beauty/out/source-fidelity-r11/inland \
  --hours 12
```

The retained noon zoom-1 BMP SHA-256 is
`b646fe09eb6ff7dc2653a5cc25a6cdce2fe1fe543cc65dbc1dae0eb8ce8be98c`.
The PNG SHA-256 is
`95a631ef1d18adce83328c1ba02e588c2f7e45f626ff02da776f5832337a5f6d`.
The lossless comparison SHA-256 is
`00a783a992bf1c0bcbb4fd81de5dee23a130a5a286f73933786203e197f53e08`.
A separate r12 replay reproduced both r11 raw zoom hashes exactly.

## Honest boundaries

This proves that the selected natural systems coexist at the 100-tile Lab
scale. It does not prove production caches, Civ III capture, Direct3D parity,
native overlays, cities, units, resources, roads, fog, labels or UI. Exact
Firaxis forest scatter, LEAN evaluation, ambient SH, temporal processing and
the two tundra snow-hill decal families remain unresolved. Source river/forest
clearance must be driven by authoritative game geometry during Integration.
