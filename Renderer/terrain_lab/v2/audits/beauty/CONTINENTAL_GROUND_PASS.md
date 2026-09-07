# Continental ground and source material baking

The all-applicable-texture audit remains incomplete. These are bounded source
findings and combined diagnostics, not visual approval or a promoted package.

## Recovered ground geometry

`systems/terrain/prepare_continental_ground.py` extracts grass, plains and snow
continental height fields from Base and the selected terrain overlay. The two
source trees have identical parameters and corresponding payloads for these
three elements. Source grass/plains/snow height scales are 14/10/12; base
heights are 24/16/24. Both source LODs and hashes are in the fixture provenance.

The active BIQ geometry previously starts ordinary ground at 2.5 projected
pixels, then adds raised terrain and river constraints. The analytic rolling
ground function elsewhere is not the active BIQ continental-ground path.
The optional `TerrainHooksV1.ground_height` adds continuous source fields while
preserving water and the shore collar. The diagnostic uses 64-square filtered
fields, cubic sampling, six periods across the map and one projected pixel per
authored height unit. These scale/placement choices are Lab hypotheses; source
engine placement and units are not recovered. Snow is inventoried, not applied.

The first composition changed river routing and is rejected. The r2 caller
excludes the provider while preparing rivers, then clamps actual ground to
the existing corridors. An explicit river XY parity probe remains outstanding.
The field uses authoritative map context beyond the capture halo. The actual
source crop/wrap probe compares 1,764 common-domain points per case, including
horizontal wrap; maximum permitted error is 0.00001 projected pixels. It does
not claim correctness for calls beyond the documented capture halo.

`continental-material-r1` binds grass/plains high color, height and specular
with a shared diagnostic height mask. Its four inland noon/midnight/two-zoom
frames are complete. Broad pale patches, especially across the central flat
grass, fail to reproduce the canonical ground's small surface structure. Reject
this mask and appearance. Neither this nor the subtler r2 ground displaces the
preserved best. The full provider-disabled composition has four exact BMP and
packet matches to r3.

## Source shader evidence changes the next approach

`systems/terrain/inspect_ground_shaders.py` finds 932 valid DXBC containers in
the installed DX11 shader package; the expanded terrain/material inspection
selects 117. This is an inventory, not proof that every selected variant runs.
Raw containers and disassemblies remain ignored local derived assets.

Microsoft's disassembler produced 37 initial variants. The later VM connection
stopped; inspection reported the VM suspended, and both start/resume attempts
failed. The bounded local decoder then validated 3,029 supported instructions
against recovered Microsoft output and decoded seven small bake/resolve
variants. Unsupported operand forms fail, and unsupported reference containers
are explicitly recorded. See `GROUND_BAKE_DECODE_EVIDENCE.json`.

Confirmed instruction behavior:

- Bake variant at package offset `0x41c5fe` samples source color, squares its
  alpha, multiplies by an interpolated contribution, and uses that common
  weight for RGB, FOW RGB and source specular output.
- Height variant `0x4203b6` uses the same alpha-squared contribution, remaps
  height through a supplied minimum/maximum, emits four times weighted height,
  and writes the weight to a separate target.
- Resolve variants `0x421d66`, `0x422efe`, `0x424096` and `0x42524e` load paired
  inputs and divide accumulated color or scalar channels by a scalar input.
  This supports normalized weighted baking, not directly darkening RGB by alpha.
- Terrain final-shading variant `0x2c9038` blends four neighboring cached color,
  two-component normal and specular/AO textures. It transforms signed normal XY
  through the geometry basis. SpecAO's second channel multiplies sampled AO;
  shadow and opacity-shadow inputs are separate. Final base-color sampling uses
  RGB, so the original alpha's role occurs earlier in baking.

The exact selected variants, render-target connections, blend state, vertex
contribution generation, cache resolution, height conversion scale and AO
generation remain unproven. Height-to-normal compute variants at `0x426326`
and `0x426f26` are extracted, not yet reconstructed. Reflection also exposes
fuzz in broader material shaders; it does not establish terrain fuzz usage.

The four ground BaseColor alpha channels are nonconstant. Grass base/high have
the same alpha statistics (6–205, mean 87.1206); plains base/high likewise
(31–225, mean 104.9090). Alpha does not equal the corresponding height or
constant-62 specular channel. Do not reinterpret it as opacity or AO.

## Source-weight diagnostic and visual judgment

`qa/source_blend_pass.py` adds a guarded alpha-squared normalized weight to the
Lab's existing terrain contributions, reused by color, height and specular.
Mapping those contributions to the source engine is an explicit hypothesis.
The four inland frames reuse exactly the same complete packets as r3.

At noon/zoom1, 12,340 pixels change, confined to `[402,66,798,212]`: the upper
sand-to-grass boundary retreats slightly and picks up texture variation. The
standard central crop is unchanged. This is a limited transition correction,
not the missing whole-scene grit. No promotion or wider improvement claim is
made. See `out/source-blend-r1/inland/transition-comparison.png` at native scale.

Direct inspection against canonical `sea_and_shore.png` and `mountain.png`
still exposes three larger gaps: fine surface relief remains weak on ordinary
land, grass/soil transitions lack distinct small patches, and shallow ground
and rock joins have too little local shading structure. Canonical zoom differs;
its enlarged reference detail is diagnostic, not a matched pixel-density claim.

Next: reconstruct the cached height-to-normal and AO stages, trace the layer
contribution graph and source scale, then compose those corrections with the
recovered patch atlas. Avoid another global bump multiplier or pale high mask.
The source-weight branch needs the remaining fixed regions and a newly selected
holdout before it can be retained; the campaign's freshground is now a regression
witness. Preserve the frozen Integration pickup and all milestone gates.

`qa/verify_continental_pass.py` records 16 candidate frames and eight exact
disabled controls. The ground/crop tests and Lab workflow are supporting
engineering checks, not visual acceptance.
