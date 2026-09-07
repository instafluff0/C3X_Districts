# Implementation map and verification order

Paths below are repository-relative. Search the named symbols rather than
depending on line numbers; production caching work is active in the checkout.

## 1. Preserve the existing boundary

Production lives in `Renderer/native/c3x_renderer.cpp`,
`integrated_terrain.hlsl`, `terrain_rendering.hlsl`, `terrain_scene_runtime.*`
and the native environment/definition/cache helpers. `injected_code.c` captures
authoritative Civ III state and inserts output through the established
`Map_Renderer_m19_Draw_Tile_by_XY_and_Flags` / `Map_Renderer_m71_Draw_Tiles`
lifecycle. Keep that off-screen path. No second presenter, native-terrain replay,
new gameplay ownership or live CSV/file parsing belongs in this port.

The current production indexed tile-mesh, pixel-block, viewport and idle-worker
caches must survive. Do not copy `frozen_scene.cpp` over native code. It is the
Lab behavior reference; its executable, global test state, CSV adapter, fixed
camera and environment variables are not a production architecture.

**Concrete binding collision:** production `integrated_terrain.hlsl` already
uses **b1** for `C3XViewportSettings`; Lab `frame_shadow_v1.hlsl` uses **b1** for
`Q6SharedShadow`. Allocate a distinct native slot or explicitly combine the
layout and update every binding. Pasting the Lab declaration unchanged would
interpret camera constants as shadow constants. The native input layout also
lacks the Lab world/hydrology/relief-material extensions. Add them deliberately
to `Vertex`, D3D input descriptors and both shader stages as needed.

## 2. Port the terrain query as one source of truth

| Lab source / symbol | Native destination / constraint |
| --- | --- |
| `terrain_lab/v2/systems/terrain/scene_adapter.h` and its included material/field helpers | Replace/adapt material-weight and UV queries inside native terrain construction; keep world-stable seeds |
| `systems/hydrology/field.h`, `scene_adapter.h`; `initialize_articulated`, `shore_sample`, `signed_shore_distance`, `coast_segment` | Adapt native `SceneTopology`/shore queries to immutable captured tiles; no runtime CSV |
| `shared/frozen_scene.cpp`: `biq_tile_height`, `biq_world_height`, `biq_coastal_relief_envelope`, `make_biq_vertex` | Native terrain sampling and `sample_relief_chain`/ground-layer builders inside the frame construction path |
| `systems/relief/continuous_normal.h`: `continuous_normal`, `to_local_uv` | Derive normals from the same continuous height query used by terrain and grounding |

The Lab hydrology kernel uses a **tile-center lattice, land-positive distance**.
Its frozen adapter accepts tile-corner coordinates, subtracts .5, changes sign
to water-positive and returns `clamp(distance/.65,-1,1)` for the historical
shore-distance consumer. `shore_sample` returns true distance, beach width,
rockiness and depth through a separate channel. Do not confuse normalized
coverage with a distance or apply the sign conversion twice.

The retained shoreline profile is **2**, not the earlier varied profile 1.
Canonical source coordinates drive its deterministic variation and wrap policy.
The Lab adapter sets horizontal wrap for this dataset; production must use the
captured world's actual wrap flags. Preserve tile-center land/water identity,
native connections, crop translations and shared edges. Required halo size is
the closure of actual consumers, not simply the visible fixture size.

Material origins such as `Q3_MATERIAL_ORIGIN_X/Y` are fixture adapters to global
coordinates. Never bake coastal/inland fixture constants into native code.
Use authoritative global tile coordinates and canonical wrap representatives.
Likewise origin [104,380] and half-width 64 are replay cameras, not substitutes
for `tile_to_screen_coords`/captured native anchors.

## 3. Transfer source geometry and material coverage together

**Hills/cliffs:** retain the selected direct hill source and its module
calibration. `systems/relief/coast_rocks.h` places six source cliff bodies with
uniform transforms, preserved UVs and placement version 4. The terrain shoulder
is in `biq_tile_height` (selected-source Height 5/12, provisional .5 placement
calibration, 112-pixel authoring basis and bounded envelope). Translate that
authoring basis exactly once to the native geometry basis. Preserve the flat
water collar. The terrain shoulder and rock placement form one change.

`systems/relief/selected_coast_source.py` and `systems/relief/prepare_coast_rocks.py` are
offline builders. The selected skin overrides all four cliff channels even
though its source mesh vertices/indices match the Base pack. The runtime
bundle is `fixtures/beauty/source-coast-rocks-v2/coastal_rocks.bin`; source and
compiled hashes are in the manifest. Expected views are base/LEAN0/LEAN1/gloss
DXGI `[72,83,80,71]`; gloss is a linear view of unchanged compressed payload.
Preserve complete channels and fail visibly on missing required assets through
the existing custom-on failure path. Do not fall back to the old brown Base
cliff material or leave a partially bound source body.

**Mountains:** `biq_mountain_sample` retains source variant/rigid orientation;
source footprint is .50 connected or .68 isolated, divided by **1.30**. The
same scale multiplies source displacement. `biq_chain_relief_sample` includes
diagonal owner candidates for the broad-relief path. The bounded support is
`1-smoothstep((max(abs(local-.5))-.52)/.23)`, ending at radius .75: at most .25
tile beyond the owning cell edge, before the neighboring center. Respect the
existing coastal/native-water envelope and river carving. Do not enlarge only
height or only the shader footprint.

**Volcano:** retained aligned source footprint is **.62 / 1.60 = .3875**,
aspect 1, with the same local-v orientation in height and every material channel.
Both source-body XY and Z use **1.60**. The material can cross onto neighboring
land without changing that tile's gameplay terrain identity. Optional generic
`relief_material` float4 contains source-owner **UV.xy, coverage.z, state.w**.
It is Lab attribute index 18 / shader TEXCOORD16 after world/hydrology data;
choose an explicit native layout rather than assuming native indices match.
Use source-owner UV across both incident receiving tiles, never `frac(world)`
at the tile boundary. This fixes the earlier bare green cut through the body.
The witness's coordinate-derived active/dormant state is a diagnostic convention;
do not replace authoritative gameplay state with it or claim eruption support.

In native code, inspect `sample_mountain`, `sample_volcano`,
`sample_relief_chain`, `sample_normalized_field` and `measure_field_limits`.
Verify source sample values against the Lab before tuning dimensions: the
presence of similarly named samplers does not prove equal normalization,
min/max treatment, UV transforms or displacement. Preserve source asset hashes.

Vegetation retains source mesh, XY, yaw and uniform scale; its grounding Z
must be resampled from the changed surface. Cities/routes/resources need their
own later combined clearance regression; an empty object fixture does not prove
they remain unobstructed by the larger footprint.

## 4. Transfer shading and frame composition, not just parameters

The fixture's `combined.hlsl` is the authoritative define list.
`systems/lighting/prepare_linear_scene.py` generates
`shaders/lighting/generated/scene_linear_v1.hlsl`. Edit/adapt the generator or
its explicit modules; do not rely on unrecorded edits to generated output.
`shaders/relief/combined_material.hlsl` supplies source rock projection;
`shaders/hydrology/scene_material_v1.hlsl` supplies shore and static water
response. The selected water views are `[11,35,11,35]` for large/small normal
and moment channels. The C3X response is an adaptation, not recovered LEAN
equations or water animation.

The Lab uses scene-linear premultiplied output, shared sun/moon environment,
reconstruction before exposure/tone mapping/transfer, and attached source
casters. Compare the native render-target/blend/transfer chain before bringing
over the color functions. A direct source sample must not undergo sRGB decode
twice; a linear material channel must not use an sRGB view. Preserve the current
four-sample / anisotropy-8 / mip-bias-0 candidate settings for initial parity.

**World shadows:** `systems/lighting/scene_shadow.cpp`, `shadow_field_v1.h`,
`alpha_coverage_v1.h` and `shaders/lighting/{frame_shadow_v1,scene_shadow_v1,
shadow_visibility_v1}.hlsl` define the retained behavior. The Lab CPU builder
rasterizes actual opaque/cutout geometry; water/bed/decals do not cast. The
common field includes receiver/caster extent, alpha UV coverage and a translated
world origin. The retained R16 depth is replicated across RGBA16; the rejected
caster-plane experiment is not part of the format. Field span is bounded from
geometry, and resolution grows 1024→2048→4096 to target 6/1024 world-unit texels.

`Q6_TEXEL_RECEIVER_OFFSET=1` uses normal offset
`min(span/resolution,6/1024)` with receiver-plane derivatives from unshifted
geometry, the original 3x3 PCF and physical depth bias `.00060/span`.
Contact bounds `.0039..024` remain in normalized world units. These constants
cannot be pasted into pixel-space native positions. `omit_replaced_shadow_surface`
removes only the redundant base caster where the replacement exists; do not
omit it before the complete source caster pass is present.

The whole-field CPU implementation is a fidelity reference, not a claim of
acceptable live frame cost. A native GPU/cached implementation may replace its
execution strategy only while preserving verified caster/receiver behavior.
Do not rebuild a viewport-sized field separately for every cached tile.

## 5. Update dependency and damage accounting

- Geometry keys include visual-profile revision, pack/channel revisions,
  consumed base/real terrain and halo topology, source variant seeds, source
  scales, shoreline profile, and the grounding surface. Camera translation must
  not reseed or rebuild unchanged world geometry.
- Neighboring relief now affects a receiving land tile, its vegetation Z,
  material coverage, bounds and shadow. Track all consumed owner candidates and
  their topology dependencies. A one-cell dirty rectangle is insufficient.
- Coastal segments/source cliff pieces may span tiles. Expand damage using
  actual projected geometry bounds, including shadows and filtering, at both
  zooms. Preserve clipping and offscreen halo casters.
- Shared environment changes invalidate lighting/shadow/pixel results without
  needlessly rebuilding stable source meshes. Static scenes request no animation.
- Profile/shader/binding revisions invalidate incompatible mesh layouts and
  cached pixel blocks. A changed shadow field/origin cannot reuse old shaded
  blocks merely because tile IDs are unchanged.
- Keep worker inputs immutable and publication atomic. Preparation must not
  mutate live topology, shared source resources or published ownership arrays.
  Preserve existing cache budgets; record costs separately from visual parity.

Use the current `dependencies`/`anchor_dependencies` checks, frame signatures,
content revisions and actual bound calculations. This package intentionally
does not freeze native files or overwrite the evolving API/cache implementation.

## 6. Ordered implementation and validation checkpoints

1. Validate the package, inspect current native state and establish a refresh
   candidate under the existing gates. Keep production's current result intact.
2. Port/query-test terrain coordinates, source sampling and shoreline topology.
   Run `tests/relief/test_source_mapping.py`, `test_continuous_normal.py`,
   `test_combined_desert.py`, and `tests/hydrology/test_varied_coast.py` using
   unittest discovery. Add native tests that exercise the actual ported functions.
3. Port complete geometry/material/input-layout changes together. Rebuild the
   retained scenes on D3D11 at matched native output sizes; compare silhouettes,
   relief-footprint/material boundaries, normals and ground contact. Preserve
   all three original benchmarks plus longcoast and a holdout.
4. Port shared linear lighting/water/world shadows. Resolve b1 explicitly.
   Run lighting/platform contracts and compare both noon and midnight. Then
   cover sunrise/sunset, seasons, source alpha and emissive/material exclusions
   required by the existing gates; do not infer them from the retained matrix.
5. Compare cold and warm native renders, worker preparation, both zooms,
   adjacent and multi-tile scrolls, wrap, terrain edits, environment changes,
   pack/profile reload, viewport resize and device reset. Pixel-block results
   must match a fresh render; log time/memory and source bounds independently.
6. Run `python3 Renderer/tools/renderer_dev.py integration` for native changes,
   and the approved `TEST_INJECTED_CODE_COMPILE.bat` workflow if injected code
   or C3X.h changes. Run `full` at the required shared-contract/closure gate.
   Update `docs/civ3_patch_dependency_ledger.md` for the actual implementation
   step: currently no new symbol is proven necessary, audit candidates and
   required user actions for this preparation are empty.
7. At the documented strategic checkpoint, obtain the batched in-game evidence
   for authoritative anchors, retained HUD/fog/labels, correct ownership,
   config-off behavior and custom-on visible failure. Preserve the existing
   visual approval and 192-tile/four-phase convergence requirements. Request
   a concrete review; never infer approval from build or pixel-diff success.

### Reproduce the retained Lab matrix

```sh
python3 Renderer/terrain_lab/v2/qa/shadow_receiver_pass.py --region all --output-root Renderer/terrain_lab/v2/audits/beauty/out/integration-reference-replay-r1
python3 Renderer/terrain_lab/v2/qa/shadow_receiver_pass.py --region freshshadow --baseline --output-root Renderer/terrain_lab/v2/audits/beauty/out/integration-baseline-replay-r1
python3 Renderer/terrain_lab/v2/qa/inspect_shadow_receiver.py
python3 Renderer/tools/renderer_dev.py lab
```

Use fresh output directories; completed reports are protected. The inspector
checks the retained canonical paths, not arbitrary replay paths. The package
manifest pins all 32 reference BMPs so an Integration comparator can consume
them without guessing file names. Licensed packs remain local. The selected
source locator is `C3X_CIV6_ENVIRONMENT_SKIN`; normalized runtime data remains
source-agnostic. Missing local art is an extraction prerequisite, not permission
to substitute a different skin or incomplete material.
