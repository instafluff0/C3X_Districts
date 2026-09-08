# Natural geometry and material adapter

The current `city-fidelity` production profile builds on this natural adapter
with the current environment and city response. The accepted appearance is
recorded in the category catalog, not an old experiment label. This directory
adapts the selected terrain, mountain and forest providers to authoritative
Civ III anchors, immutable tile buffers and world-paged shadows. It does not
replace the capture path or automatically select newer source experiments.

## Selected paths

- `beauty_terrain.cpp` numerical hill/seed/relief kernels and
  `beauty_terrain.hlsl` source grassland, plains, tundra and authored rock response.
- `beauty_mountain.cpp` five complete macro height/blend variants and
  `beauty_mountain.hlsl` full source base/top/snow color, height and specular.
- BeautyStudies path of `beauty_objects.cpp` / `beauty_objects.hlsl`: all 22
  tree bodies, 25 recipes, count weight 180, authored opacity and packed normals.
- The exact r13 `hydrology.module.json` selection: source-fidelity-r2 base,
  shadow-receiver-r1 defines and selected `scene_linear.hlsl` closure. This
  selection uses static optics; later natural-water/reflection experiments are
  not selected by r13 and are not enabled here. River corridor and bank-rock
  placement use the selected source kernels around authoritative world data.
- Retained production city, resource and unit feature shaders and runtime packs.

`prepare.py` builds a generic local payload from the selected source channels.
Its DDS payloads are unchanged, not resized or transcoded. Current preparation
keeps independent runtime copies rather than editable-source hard links.
`provenance.json` records the consumed inputs. `prepare_hydrology.py` applies
the existing native binding adapter to the selected hydrology closure. Neither
script modifies Lab source or the live unit pack.

## Explicit native adapters

Shared CPU providers live under `Renderer/lab/shared/natural`: pack decoding and
lighting frames, river pages, relief queries, terrain grids, deterministic
placement/dunes, and relief/forest mesh emission. Native retains world/coast
invalidation observations, city exclusion collection, caches and GPU ownership.
The portable mesh adapters and native compiler include the same statement bodies.
Keep the native bodies at their existing call site: introducing a separate x86
mesh-function boundary was observed to change final channel rounding despite
numerically identical standalone tests. D3D image comparisons protect that
boundary; do not silently accept or replace a reference after a refactor.

`light_frame.h` supplies one normalized Q6 frame from EnvironmentState. All three
natural providers use exactly its ShadowL for face lighting and cast projection.
Source coordinates are converted to the canonical world basis once, before both
visible and shadow submission. Native anchors supply translation. The old
mountain x-origin mismatch is not restored.

Mountain casters apply the same interpolated footprint cutoff as the visible
shader. Previously the invisible fringe between raw blend .015 and the visible
smoothstep cutoff still cast, producing the reported broad black rings. Forest
casters sample their authored opacity with the source address mode. No blob
shadow geometry is emitted.

Live terrain uses the retained continuous authoritative material field in place
of the isolated Lab scalar biome selector. Independent grass/plains/desert/tundra
weights preserve their source interiors without hard per-tile diamonds or an
artificial plains band between grass and tundra. Desert's three unchanged source
channels and retained dune geometry participate in that blend. This is an
explicit correction requested after the candidate boundary regression; it does
not claim the isolated Lab originally implemented live map blending.

The replacement ground and hill decals use the authoritative shoreline distance
and beach width for continuous coverage. Its clipped caster uses the same
coverage. Land portions of coastal water cells are emitted too; tile ownership
does not truncate the geometric contour. The retained beach/water shader stays
visible underneath. Existing marsh-specific ground and decals remain on their
original path.

`coast_join.h` brings replacement relief to the flat retained beach before the
material fade. Height and slope join continuously; normals, tree roots and
casters use that same height query. The inland source height is unchanged. This
removes the raised dune rims exposed by the earlier alpha-only shoreline fade.
An executable contract checks variable beach widths, flat fade coverage,
monotonic height, endpoint slopes and exact inland preservation.

Natural geometry remains in the existing 76-byte packed vertex buffers. The
added terrain weights reuse existing fields. Texture sampling is 16x anisotropic
with -1 mip bias. Natural composition uses 4x MSAA at 2x render scale, in aligned
128-native-pixel blocks, with one equal-area scene-linear reconstruction before
existing exposure and transfer. The scratch target is at most 3,670,016 bytes.
Retained feature samplers and city/unit material equations are unchanged.

Existing tile/viewport/pixel-block budgets, prefetch cancellation, ownership,
world revision invalidation and wrapping remain active. River queries use at
most sixteen 8x8 pages with four-cell support halos. Tree exclusion uses the
current production building meshes and authoritative coast/river fields before
emission. Full body footprint bounds are tested, not just tree centers.

## Reproduction

```sh
python3 Renderer/renderer.py prepare
python3 Renderer/renderer.py lab grassland
python3 Renderer/renderer.py compare grassland
python3 Renderer/renderer.py test grassland
python3 Renderer/renderer.py integration grassland
```

Verification uses the Windows off-screen DLL and never starts Civ III or runs
INSTALL.bat. Current category results live under `Renderer/lab/out/`.
Runtime logging uses the established OutputDebugStringA path; file traces are
explicitly enabled only by these headless verification commands.

Explicit `source-fidelity-r13`, `pickup-r1` and `frozen` select earlier diagnostic paths;
they do not supersede current production appearance. See `Renderer/lab/README.md`.
