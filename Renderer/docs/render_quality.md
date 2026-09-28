# Live scene quality

## Scope

Improve detail at normal and close zoom while preserving the asynchronous scene,
native camera anchors, fixed-size native HUD, city-view constraints and GPU-only
map delivery. Fixed Lab references are unchanged.

## Current standalone candidate

The current graphics study adds real low relief to grassland and plains, a
shared restrained HDR display curve, and seven main-map zoom endpoints through
3×. It is an isolated evaluation candidate; the earlier staged/live measurements
below describe the previous baseline. Fixed Lab reference images are unchanged.

### Gentle ground relief

`lab/shared/natural/low_relief.h` consumes optional, generic authored height
fields from `NaturalFidelityRuntime/low-relief.bin`. The offline compiler selects
the imported continental grassland/plains fields referenced by `StandardFlat`.
It preserves every 2048×2048 R8 source sample. Runtime contains no source-game
asset names. Missing optional data gives the original flat surface.

The selected study uses height amplitude 64 in the existing natural-surface
units and a nominal repeat span of 96 native tile-coordinate steps. These are
C3X calibration choices, not confirmed Civ VI engine settings. A 28-unit control
was too subtle. The authored values occupy only part of the normalized range;
64 does not mean every tile acquires a 64-pixel hill. Integral repeat counts
close world seams. Biome and river-neighborhood weights taper continuously;
water and river beds keep their datum, with a smooth coastal approach.

The shared query supplies foreground/prepared terrain, normals, object/city
placement and border ground meshes. Unit bodies and selection centers sample
the same low ground from authoritative captured anchors, including sub-tile
travel; reflections mirror the vertical offset. Unit projected shadows share
the raised anchor, but their existing planar projection is not yet a full
terrain-conforming shadow implementation. The existing terrain grid is reused:
this pass adds two bounded CPU height fields (8 MiB), not more triangles or GPU
readback. Reload/reset releases the optional fields with the natural assets.

### Display and material response

The common map output mixes the existing neutral transfer with a restrained
[Narkowicz ACES-like fit](https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/).
The default mixture is 0.5, controlled by `C3X_RENDERER_SCENE_FILMIC`; zero is
exact legacy neutral output. This is not the complete ACES pipeline. The change
shares the existing output pass and applies equally to terrain, units, cities
and other scene geometry, before native HUD. The existing CAS amount remains
0.35. No intermediate target or extra resolve is added.

A zero-height shading control isolated the grass's dark fine speckles to
excessive material-height response. The shared ground shader now uses one
quarter of that micro-height strength, preserving authored color, geometric
normals and broad material variation. Day, dusk and night captures retain
warm local facade lighting and emissive detail.

MSAA 2×/4× were compared on the actual shared scene. Neither is selected as the
default: 4× improved silhouette coverage but increased the median in a mixed
zoom/wrap workload from 16.70 to 31.40 ms. Native-resolution single-sample output
remains selected. Alpha-to-coverage consequently remains unimplemented here.
The private sprite box/Lanczos path is not the direct shared-unit path.

Validation passes the current grassland suite (137 tests, one missing-source
skip), extracted zoom/input and grounding contracts, actual GPU projection and
HDR tests, and the complete retained-composition oracle. Extending HUD bounds
to the new maximum fixed an exact-pixel clipping failure above 1.5×. Injected
compilation and the matching bridge/DLL/helper `BUILD_RENDERER64.bat no-stage`
build pass. Grassland and plains close-ups and a night view were inspected;
these checks do not substitute for a live extended-zoom run.

### Reproduction and bounds

`tools/scene_quality_study.py` creates a complete small world, loads the actual
Renderer64 DLL in its standalone Windows client, and records the precise DLL,
shader and pack hashes. `--zoom 3` selects close detail; `--zoom-peak 3
--start-ms 29500 --frames 90` exercises a 1×–3×–1× sweep. `--land plains` changes
the ground family. `--hour 0` checks night. Do not launch Civ III for this study.
Warm cases finish in about 6 seconds; a changed shader closure can require a
roughly one-minute first compilation.

For this candidate, pass `--dll
Renderer/native/build/renderer64/C3XRenderer_x64.dll`, `--client
Renderer/native/build/scene-quality/client_x64.exe` and `--shaders
Renderer/native/build/scene-quality/candidate-shaders`. The tool defaults point
at the earlier staged binaries/shaders. The shader directory contains the
canonical generated terrain and water shaders; receipts record their closure.

`--hdr` explicitly captures the standalone FP16 surfaces. The tool compresses
them losslessly and verifies the bytes before removing raw data. The separate
Mac analyzer reconstructs the selected GPU display curve within one 8-bit
channel value, then compares alternate curves with the existing CAS filter.
This is Mac image analysis, not a Metal scene renderer or a game CPU fallback.
Raw `frame.png` is the standalone output before CAS; `mixed-050.png` is the
verified analysis with the production 0.35 CAS treatment. No screenshot is
resized to manufacture a closer view.

At 1280×800, one sample, water/waves/reflections enabled, 87 measured frames
after three warmups in each 90-frame fixture:

| Fixture | Median frame ms | p95 frame ms |
| --- | ---: | ---: |
| 1×–3×–1×, neutral transfer, flat ground | 16.65 | 18.57 |
| 1×–3×–1×, selected transfer, flat ground | 16.70 | 18.61 |
| 1×–3×–1×, selected transfer, rolling ground | 16.66 | 18.73 |
| Settled 3×, selected transfer, flat ground | 16.70 | 18.36 |

These small, vsync-limited standalone runs are not live-game FPS qualification
or evidence of Civ VI visual parity. Geometry/lighting calibration, unit shadow
conformance and aliasing remain visible areas for further improvement.

Keep generated evidence under `native/build/scene-quality`, below 1 GiB.
Preserve the selected images, compact receipts and compressed HDR inputs;
remove compiler intermediates and redundant derived variants when finished.

## Visual target

The user's Civ VI reference sets a substantially higher bar than a sharpened
version of the previous game image: distinct roof tiles and masonry, readable
small silhouettes, clear lit/shaded surfaces, and terrain detail that follows
its shape. Compare at matching displayed object sizes and preserve the source
pixel count. The reference demonstrates the desired result; it does not prove
which proprietary filtering or lighting algorithms produced it. Later Lab
systems remain in their existing scope and must retain their detail through
this shared scene pipeline.

## Confirmed starting point

- The live Renderer64 path uses `sandbox/fresh_pipeline.h` and `direct_units.h`.
  It draws retained unit geometry into the shared scene. Its default is native
  resolution with one sample; `C3X_SANDBOX_MSAA_2X` enables two samples.
- The selected packs' `sample_scale=4` and private MSAA4 targets belong to the
  separate unit sprite path. They do **not** describe the live shared scene.
  The earlier unit sharpness audit and isolated sprite comparisons cannot serve
  as a live-game baseline.
- Custom zoom previously bilinearly enlarged the completed canonical map in
  `gpu_view_transform.h`. At 1.5x, one original pixel covered 2.25 display pixels.
  A sharper reconstruction filter alone cannot restore missing geometric detail.
- The shared unit path already preserves source tangents, normals, complete
  texture/mip dimensions, 16x anisotropy and the selected material model.
  The second LEAN texture's variance constants remain unresolved; they must not
  be guessed or interpreted as an ordinary normal map.

## Shared geometry projection

`SceneProjection` applies the displayed zoom in the geometry raster viewport.
Terrain, objects, unit bodies, selection rings, reflection geometry and final
visibility use the same center and scale. The original mesh coordinates,
animation palettes and native gameplay camera remain unchanged.

The retained compositor has an explicit projected-scene sample. World overlay
recipes compose over that sample before native HUD placement and fixed UI.
Canonical native save/restore images remain separate. Each displayed world
samples the scene once; it does not draw a second canonical view on that tick.
The existing image transform remains the compatibility path for immutable
native images and recipes without a projected scene provider.

The normal-zoom terrain scroll cache is unchanged. A change to the geometry
projection invalidates raster caches, never asset or mesh residency. The
measured quality controls below separate settled presentation from camera and
zoom changes; they do not establish uniform frame pacing.

## 0 A.D. source comparison

Reviewed revision `159d3390b233d0347d40f2ba2f156d937f6921fe`:

- [Renderer.cpp](https://gitea.wildfiregames.com/0ad/0ad/src/commit/159d3390b233d0347d40f2ba2f156d937f6921fe/source/renderer/Renderer.cpp):
  scene drawing, postprocessing and output scaling precede GUI composition.
- [PostprocManager.cpp](https://gitea.wildfiregames.com/0ad/0ad/src/commit/159d3390b233d0347d40f2ba2f156d937f6921fe/source/renderer/PostprocManager.cpp):
  separate antialiasing, sharpening and output-scaling choices. Ordinary CAS is
  skipped during upscaling; the FSR path uses its own reconstruction and RCAS.
- [cas.fs](https://gitea.wildfiregames.com/0ad/0ad/src/commit/159d3390b233d0347d40f2ba2f156d937f6921fe/binaries/data/mods/mod/shaders/glsl/cas.fs):
  a small spatial contrast-adaptive filter, not additional geometric detail.

This supports separating projection, sampling, reconstruction and GUI. It does
not establish Civ VI's proprietary lighting or material implementation.

## Evidence

`test_scene_projection.py` checks guarded projection at even/odd viewport sizes
and native/reflection raster scales. Its D3D test proves new one-pixel scene
detail survives zoom, overlay order and fixed HUD composition in 555/565 modes,
with one projected sample and no canonical sample per frame. Test readback is
confined to the oracle; the runtime path has no map readback.

The existing retained-composition and visibility GPU oracles also pass.
`test_scene_detail.py` checks disabled identity, increased interior contrast,
unchanged flat regions/extrema/alpha and zero extra import allocation/upload.
`test_skin_shadow.py` checks conservative weighted-pose bounds, GPU palette
skinning, maximum-height composition, cutout semantics and below-ground clipping.

## Sampling and detail choices

The selected path is native scene resolution, one sample, and modest CAS amount
0.35. The AMD no-scaling filter is adapted from the cited 0 A.D. implementation,
with its MIT notice retained. It shares the existing GPU image-import pass;
there is no new full-screen target or copy. It runs after display transfer and
visibility but before native HUD, text and general UI. All opaque scene elements
use the same rule, without unit-name or asset-family dispatch. Alpha boundaries
are left alone. `C3X_RENDERER_SCENE_SHARPNESS=0` gives exact bypass; values up to
1 are diagnostic controls. `C3X_RENDERER_SCENE_SAMPLES=2` selects optional MSAA2.

The shared unit material receives a 512-square pose-local GPU height map instead
of the previous empty texture. Every material part contributes with its own
cutout flag. Color-owner masks do not become cutouts. The pass uses the same
palette, blended pose and facing as the body and reflection; the existing
material shader applies its 3x3 self-shadow test. One 1 MiB texture is reused
across units. Per-joint bind bounds determine a conservative light-space fit;
CPU code never skins or rasterizes the unit's vertices. Asset textures, meshes,
poses, silhouettes and proportions are preserved.

The native private-sprite box filter is outside this live path, so replacing it
with Lanczos or comparing its 4x/2x supersampling would not fix game zoom. A
negative mip bias is not enabled: full mip chains and anisotropy are already
present, and extra distant texture shimmer is not a sharpness improvement.
MSAA2 did not justify its transition cost in this control. Coverage AA depends
on multisampling and is therefore not enabled with the selected single sample.
No unsupported second-LEAN interpretation or global lighting retune is applied.

## Live controls

Same disposable initial save, 2240x1260 window, ten wheel inputs, one-Hz window
sampling, detailed renderer trace disabled. Values are successful renderer
presentations per second, not physical scanout. Windows capture timestamps
anchor each interval. These controls precede the unit self-shadow change.

| Capture | Scene sampling / CAS | Normal view, 22–29 s | Zoom sequence, 30–44 s | Settled 1.25x, 46–54 s |
| --- | --- | ---: | ---: | ---: |
| `20260928-092829` | 1 / 0 | 52.14 | 35.19 | 49.60 |
| `20260928-093108` | 2 / 0 | 51.75 | 29.83 | 50.83 |
| `20260928-093331` | 1 / 0.35 | 51.02 | 35.34 | 51.38 |

Each completed all ten inputs without a native renderer failure or early exit;
the original save remained unchanged. Matched 1.5x crops exposed an omitted zoom argument on the main unit pass:
terrain was zoomed while the body and selection retained their normal size.
The call is corrected, and the unit draw now requires an explicit zoom argument
for both normal and reflection passes. The actual resident vertex shader's
footprint growth is covered by a GPU test. JPEG samples establish
appearance and placement, not lossless pixel differences. Exact GPU tests carry
the detail-preservation and HUD pixel claims. These short runs support selecting
modest CAS over MSAA2; they are not a statistical performance study or Civ VI
visual parity claim.


## Earlier integrated baseline

The final candidate is qualified by async fixture
`b71c8cee78144743bd2aac3ac7c7de4f`, with 59.88 FPS warm, 59.96 FPS during
24 pose/facing changes, 32 camera adoptions, 117 displayed frames while the host
pauses, 73 native publications while the renderer pauses, p95 submission
1.002 ms and zero CPU map readbacks. Those are fixture measurements. A bounded
window capture begins after the timed animation checks and shows the same shared
path with a Warrior, buildings, forests, rocky terrain and animated water.

Live zoom capture `20260928-094902` completes all ten wheel inputs without
renderer errors or save changes. With unit projection corrected, the same
normal/changing/settled intervals measure 51.07 / 35.12 / 49.51 presentations
per second. At matched 1.5x zoom, the old and new unit and selection footprints
agree; the new scene preserves more texture and silhouette detail.
The small optional comparison is
`native/build/quality-comparison/close-zoom-comparison.png` (native-sized crops
from the original sampled JPEGs). Animation and water phases differ between
runs, so it is not a pixel-difference oracle. The larger inspection sheet uses
nearest-neighbor enlargement and is not additional rendered resolution.

The matching bridge, x64 DLL and helper are staged for evaluation. No injection
hook changed for quality work; no patch-table address or injected compilation
was needed. Fixed Lab reference images and licensed source assets are unchanged.
All quality changes are shared implementations, with no unit-name special case.

Final live checks on the corrected live path:

- HUD capture `20260928-095027`: three map-text events, both native city zoom
  widths, two map moves and advisor open/close. Reviewed samples show a single
  city label at its updated position and intact native panels.
- Scroll capture `20260928-095347`: all 32 camera steps; 44.59 presentations/sec
  during seconds 20–55 and 50.55 during seconds 60–74, with one-Hz sampling and
  detailed trace disabled. This is one short run, not a stable performance gain.
- City capture `20260928-095508`: both 64/128-pixel native zoom widths report
  the same `1120,566` center. Wheel and edge-scroll input leave that center
  unchanged. The existing cold city-view preparation delay remains visible.

All three runs completed without native renderer errors or early exit and left
the original save unchanged. Pixel regressions cover HUD placement, partial UI
publication, retained-image lifetime, fog/cutout coverage and projected bodies;
sampled JPEGs provide complementary appearance checks.
These live checks used candidate `6efbe93cc0ce4d35b1724522403fbf09`. The final
build adds the same geometry projection to the mock sandbox unit entry point;
the checked live unit entry point and quality settings are unchanged.

The sampled-window fixture writes stdout and stderr separately. Its recorder
now checks both, including the deliberate rejected-presentation error. Runtime
build provenance compares compiled source closures separately from runtime art
inputs; both remain hashed. Diagnostic capture `9306f7d1a87a4e5eaf69d79459c9283f`
completed the native checks but its original receipt failed because stderr was
omitted. Its original receipt is preserved; the final qualification above uses
the corrected recorder and matches the build's source closure.

## Remaining quality work

The immediate result is less detail lost during close zoom, modest scene-wide
contrast recovery and self-shadow depth for every live unit. This does not
establish Civ VI visual parity or complete the category-specific Lab work.

The next visual comparisons should cover cloth, metal, foliage, stone and
buildings at matching displayed sizes, using the shared live scene rather than
the old private unit-sprite studio. Apply confirmed material/lighting corrections
to their common consumers. Expected gains are more readable folds, highlights,
surface relief and contact between objects and ground. Unresolved source
channels still require evidence before implementation. New terrain detail and
later Lab systems remain separate changes under their existing scope.

Maintain before/after camera, object size, lighting and animation phase for
future comparisons. Also measure settled, scrolling and changing-zoom intervals
separately: the current live zoom sequence is about 35 FPS, below the sandbox
target, even though settled views are near 50 FPS.
