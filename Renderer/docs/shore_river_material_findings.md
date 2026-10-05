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

## Consistent lowland blending (current Lab candidate)

The replacement ground previously began its fade at `beach_width + .06` and
finished only `.16` tile units later. The underlay's beach material already
finished fading at that starting distance, leaving a narrow, regular handoff
whose screen width varies with coast orientation. The new ground ramp begins at
`max(.025, beach_width - .10)` and retains the old `beach_width + .22` inland
endpoint. The optical edge stays sand-only, and relief still begins after the
fade. Desert retains its separate narrow sand-family edge; rocky cliff coverage
retains its existing early opaque ramp.

Two additional inconsistencies are corrected: source edge detail no longer uses
grass alpha for every biome, and plains/dune decals no longer carry tile-center
coast coverage across their whole footprint. Per-vertex shore queries preserve
ordinary dependency observation and caching. The same generic shared providers
feed native and retained scene builds; these remain C3X material choices.

The `biomes` / `biomes-turned` Lab fixtures cover all three lowland families on
both map axes. Matched current-checkout controls and candidate captures are under
`Renderer/lab/out/shorelines/blending/`. The user asked to retain the gentler
shoulder while improving the actual sand/land material boundary.

Layer-isolation renders confirmed that the underlay returned to grass before
the replacement ground finished fading. Texture changes to the upper layer
therefore often exposed another grass surface, leaving the smoother underlay
boundary visible. `q3_beach_coverage` now retains sand through the middle of the
ground fade and ends at `beach_width + .26`. Desert and rocky exclusions remain.
The ground's detail perturbation increases from `1.8` to `8`, permitting small
sand openings and land patches in the overlap. Multiplication by both coverage
and uncovered fraction preserves the solid shoulder and sand-only endpoints.
The rejected exponential-weight experiment exposed dark holes where almost
opaque ground had already started rising; its endpoint regression is retained.

The focused suite runs 154 tests with one optional skip. Executable coverage,
overlap, ramp and decal checks cover the handoff. `blending/material-edge/`
contains matched controls, shader snapshots, native completion receipts and the
material review renders. The terrain-edit witness rebuilds 127 tiles, reuses
260, and matches the cold render exactly (zero changed pixels, zero error).
All reviewed native captures complete with zero fallback.
These isolate the material change with a saved DLL;
they do not certify concurrent renderer/injected changes or live gameplay.
Visual acceptance of this material follow-up is pending; no fixed reference
has been replaced.

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
