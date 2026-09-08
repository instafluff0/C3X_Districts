# Native source-fidelity r13 adapter

The selected authority is `source-fidelity-r13/inland`, with the isolated
terrain, mountain, forest and Warrior witnesses providing detailed contracts.
This directory adapts the selected providers to the existing authoritative Civ III
anchors, immutable tile buffers and paged Q6 shadows. It does not replace the
capture path or import newer city experiments.

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
Its DDS files are unchanged and hardlinked when supported, not resized or
transcoded. `provenance.json` records exact hashes. `prepare_hydrology.py` applies
the existing native binding adapter to the selected hydrology closure. Neither
script modifies Lab source or the live unit pack.

## Explicit native adapters

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
python3 Renderer/native/source_fidelity/prepare.py
python3 Renderer/native/source_fidelity/prepare_hydrology.py
python3 -m unittest Renderer.native.source_fidelity.test_contract -v
python3 Renderer/native/source_fidelity/verify.py first
python3 Renderer/native/source_fidelity/verify.py control
python3 Renderer/native/source_fidelity/verify.py matrix
python3 Renderer/native/source_fidelity/verify.py replay
python3 Renderer/native/source_fidelity/verify.py edits
python3 Renderer/native/source_fidelity/verify.py minimap
python3 Renderer/tools/renderer_dev.py integration
```

Verification uses the Windows off-screen DLL and never starts Civ III or runs
INSTALL.bat. Results are recorded in `Renderer/verification/source_fidelity/`.
Runtime logging uses the established OutputDebugStringA path; file traces are
explicitly enabled only by these headless verification commands.

The normal profile selects this adapter; `pickup-r1` and `frozen` remain explicit
historical regression selections. Production staging and remaining limitations
are recorded in the verification checkpoint, not inferred from compilation.
