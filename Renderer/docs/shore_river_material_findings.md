# Shore and river material channels

The September 2026 surface audit found a material-channel omission, not a missing
gravel texture. The beach and river base-color **alpha** channels carry distinct
fine grain and pale pebble patterns. Their RGB channels are much smoother. Both
materials bind the same nearly flat `TER_Coast_H` displacement texture; increasing
its normal strength does not recover the missing pattern.

## Confirmed source and runtime evidence

- Installed `TerrainStyle.artdef` selects `ART_DEF_TERRAIN_MATERIAL_BEACH` for
  beaches and `ART_DEF_TERRAIN_MATERIAL_RIVER` for the standard river material.
- Resolving the typed material pointers and re-extracting base color, height and
  specular reproduces all six normalized Base channels byte for byte. Repeating
  this against the active Environment Skin package also reproduces all six active
  channels exactly. No resizing or lost mip payload was found: these channels
  are 512 × 512 with eight supplied mip levels.
- The active skin supplies `CivV_TER_Coast_B` and `CivV_TER_River_B`, with paler RGB
  than the Base materials. It retains detailed alpha patterns. This is a local
  asset selection, not a runtime source-format dependency.
- Active beach alpha spans 0–184 with standard deviation 18.13; river alpha spans
  0–181 with standard deviation 22.94. Their shared height spans only 97–116 with
  standard deviation 1.71. These are decoded 8-bit channel measurements, not
  inferred physical heights.
- The previous shader sampled only `.rgb` from those base textures, mixed the
  river material heavily into beach sand and dark soil, and used displacement
  height for grain. It therefore suppressed the authored fine pattern twice.
- The separate four-cell river clutter atlas is already present and bound with
  its paired relief texture. It supplies larger gravel patches; it is not the
  dense grain field. The beach-blanket texture is a prop atlas, not beach ground.

## Current reconstruction

`scene_material_v1.hlsl` preserves base-color alpha as material detail. Its
difference from a coarser local sample drives bounded albedo contrast and modest
surface normals; the river bank also uses it to interrupt outer coverage.
The river material now supplies most of the bank color. Wet-bank darkening is
restrained enough to retain the flecks. Beach detail continues into the shallow
wet margin, alongside the existing gravel and submerged-rock patches.

The grass-to-beach material fade uses the grass base-color detail channel while
keeping fully covered land and water endpoints fixed. Channel topology, water
width, rock geometry and map ownership do not change.

The source channel contents and bindings are confirmed. Using alpha as a local
contrast/normal/coverage field is a **C3X reconstruction**, not a recovered Civ VI
shader equation or a claim that its alpha is ordinary surface transparency.

## Repeatable check and review

Run `Renderer/tools/asset_compiler/probe_water_material_channels.py` with
`--package` pointing to the installed terrain material package, `--pack` to its
normalized pack and `--output` to an ignored directory under `Renderer/lab/out/`.
It re-extracts only the six relevant channels outside the pack, checks exact
matches and reports channel statistics without recording machine paths.

Review artifacts are under `Renderer/lab/out/shorelines/surface-transition/`:
`asset-audit/`, `skin-vs-base-materials.png`, `grain-comparison.png`, and the
`grain-after` / `grain-after-close` native captures. The comparison uses the same
DLL, scene, packs, camera, noon light and zoom on both sides. All six final native
captures completed without fallback. The channel correction leaves the open-water
control pixel-identical to the previous pass. All 121 focused checks pass, including
shader-adapter propagation and bounded, neutral flat-channel contrast.

The user accepted these previews and requested production promotion on
2026-09-08. The staged DLL already matches the approved preview DLL; the change
uses the current runtime shader files. Production-path checks are recorded under
`surface-transition/production/`. Fixed references remain unchanged. The existing
terrain edit-reuse replay limitation documented in the Lab README is separate
from this material correction; no live-game pass is claimed.
