# Zoom performance verification

## Evaluation handoff

Optimization experiments are paused for the requested in-game evaluation. The
unfinished mutable ground-point scratch-cache change was withdrawn; no partial
implementation remains. The evaluation DLL is
`native/build/zoom-evaluation-960/C3XRenderer.dll`, SHA-256
`1a2de66d8dddc3e9698c883495945470c68536b124a8b7d08a652bb6f55590b3`.
It uses the experimental 768 MiB GPU geometry, 192 MiB natural CPU mesh and
160 MiB resource-backdrop caps, plus the unchanged 32 MiB viewport tier. These
are cache budgets, not reservations or a cap on all process memory. Ordinary
build defaults remain unchanged; rebuilding normally loses these higher caps.

Final checks on this exact DLL:

- 56 focused indexing/cache, native bridge, custom zoom, render-core and resource
  animation regressions passed.
- Native ABI, scheduling, fallback and RGB555/RGB565 blit smoke checks passed.
- Current-asset 960x640 zoom-return, six resource-animation poses, scrolling and
  removal passed; posed frames performed no terrain builds or uploads.
- Current-asset 960x640 two-cycle zoom passed with exact repeat pixels and no
  device recoveries. Five warm changes took 47.792–82.111 ms, median 67.648 ms,
  with zero tile builds; this short handoff check is not a new p95 measurement.
  Sampled free virtual address space stayed above 1.60 billion bytes in the
  standalone witness, which does not include Civ III's own memory use.
- The 1280x720 stress run passed all five repeat images without fallback or
  device recoveries. Repeat zooms still take 3.0–4.8 seconds because the working
  set exceeds the geometry cap. Cold views remain slow at both sizes.

Logs are under `native/build/zoom-evaluation-960` and
`native/build/camera-ground-scratch-control`. The latter name denotes the
pre-scratch-change control: that abandoned change is not in this DLL. Its
comparison against the older `camera-direct-grid` DLL differs visually because
the compiled scene implementation has changed between snapshots; it is not a
valid before/after performance or image-parity claim. Current-view repeat,
scroll and removal checks above supply the handoff's cache-correctness evidence.

For game testing, enable `enable_custom_rendering_zoom = true` in the mod's
`custom.c3x_config.ini` (currently commented out). Use `Z` to cycle levels.
Native city-label alignment still needs the existing
`Main_Screen_Form_tile_to_screen_coords` CSV row changed from `define` to
`inlead`, with its signature and addresses untouched; see the
[patch dependency ledger](civ3_patch_dependency_ledger.md). No new patch symbol
is required by the performance work. The agent has not changed either file.
Run `INSTALL.bat` yourself after the desired configuration/patch-table edits.
No live-game pass or smooth-transition claim is made.

## Witness

The offscreen witness in `native/biq_preview.cpp` recaptures the same full-world
fixture at 128, 112, 96, 80 and 64 pixels, then repeats the cycle. It retains the
same resource sites and animation time so cached-image parity has a deterministic
reference. Every step checks ownership and fallback; repeated steps use the
existing pixel-error thresholds. The independent comparison tool checks the
baseline and candidate images at all five sizes and records both DLL hashes.

From `Renderer/native` in the Windows VM:

```bat
call BENCHMARK_ZOOM.bat baseline
rem Make the performance changes, then:
call BENCHMARK_ZOOM.bat candidate
```

Use `reuse` as the second argument to preserve an already compiled DLL, or
`build-only` to compile without running. `preview-only` refreshes just the witness
executable without changing the tested DLL or running it. Builds use separate `build/zoom-baseline`
and `build/zoom-candidate` directories and never stage or install. The fixture
requires the existing local `lab/.local/verification/world.csv` and asset packs.
`C3X_ZOOM_WIDTH` and `C3X_ZOOM_HEIGHT` default to 640 and 480. `C3X_ZOOM_ROOT`
optionally selects a snapshot containing `Renderer/native` shader providers,
the production and custom definitions, and `Renderer/packs`. Keep shader and
asset bytes fixed between runs; a concurrent Lab edit invalidates pixel comparison.

From the project root:

```sh
python3 Renderer/native/compare_zoom_benchmark.py
python3 -m unittest Renderer.native.test_zoom_mesh_cache
```

The comparison receipt is `native/build/zoom-candidate/comparison.json`.
Initial asset/shader loading is outside the timed zoom steps. The first 128-pixel
step is an unchanged-current-view control and is excluded from the median.
Geometry timing includes construction and upload; readback timing includes pending
GPU execution. Do not interpret a replay as a live-game or smooth-transition pass.

The earlier 960×640 baseline spent about 20–25 seconds on individual 112/96/80
steps, mostly in geometry construction and upload, and was stopped after the
64-pixel step stalled. Its partial output is preserved separately as
`native/build/zoom-baseline-wide`. Those observations are diagnostic, not a
completed five-level comparison.

## Measured performance pass

Windows VM, current production profile, fixed shader inputs, 640×480, full-world
100×100 fixture with animated resources; two cycles, 263–635 visible tiles:

- Median of the nine actual zoom changes: **12,567.849 ms → 4,445.532 ms**,
  approximately **2.8× faster** (65% less delay).
- Individual speedups: 1.84×–3.53×. These are single-run observations, not stable
  latency percentiles. The widest first-use step included a 5.2-second GPU-submit
  outlier; CPU geometry/upload remains the dominant repeat-zoom cost.
- All five baseline/candidate image comparisons passed. No pixel exceeded the
  existing two-channel-value tolerance; aggregate absolute channel error was
  0, 5, 1, 4 and 0 across the five images. Repeated candidate views were exactly
  equal to their first-cycle images. No terrain fallback occurred.
- CPU cache allocation remains 128 MiB combined: 96 MiB natural mesh data and
  32 MiB viewport bitmaps. The indexed GPU tile cap remains 192 MiB; this is not
  a cap on total renderer/device memory.

The exact DLL hashes and per-step timings are in the disposable comparison
receipt. The candidate is isolated under `native/build/zoom-candidate`, not
installed or staged. No native hook/address changes are needed for this pass.
This establishes reusable world meshes; it does **not** meet instant or smooth
zoom latency. The next architectural work is camera-independent GPU storage,
remaining ground/route geometry reuse, and immediate retained-image presentation.

Verification also passed 195 dependency-selected/full integration test cases,
plus the native production animation witness: zoom-away/return parity, six poses
with zero terrain builds/uploads, and exact cold-render parity after scrolling
and resource removal. The standalone vertex-index witness compares production
indexing against the old implementation and exercises reprojection across all
five levels and multiple target heights. These are automated offscreen checks;
the exact candidate also passed native ABI, scheduling, fallback, export and
RGB555/RGB565 blit smoke tests. No live-game performance claim is made.

## Interaction targets (ongoing)

These are engineering targets, not measured achievements. Primary workload:
960×640 production rendering with animated map objects; 640×480 is a fast
diagnostic and 1280×720 a stress case. Preserve current image quality, ownership,
fog/visibility, latest-request correctness and bounded memory.

| Interaction | First correct visual response, p95 | Final-quality image, p95 |
| --- | ---: | ---: |
| Warm zoom change | 50 ms | 150 ms |
| Nearby minimap move / cached revisit | 100 ms | 150 ms |
| Previously uncached distant minimap jump | 100 ms | 500 ms |
| Animation / continuous interaction frame | 33.4 ms | 33.4 ms |

An immediate response must represent the requested camera and respect visibility;
returning an unchanged old view does not count. Final quality must pass cold-image
parity. The current synchronous renderer cannot distinguish these milestones:
its blocking render duration counts against both. Eventually collect at least
30 actual changes per workload for meaningful p95 gates. The initial two-cycle
smoke sequence reports individual timings and medians, not a statistical p95 pass.
Initial shader/asset startup is reported separately from warm interaction.

For the minimap replay, set `C3X_CAMERA_SCENARIO=navigation` before calling the
benchmark. It recaptures records at six destinations (overlap, distant regions,
and wrapping) twice, logging `NAV` timings including replay capture, geometry,
draw and readback. `C3X_ZOOM_OUT` selects an isolated result directory, permitting
camera baselines to remain intact while new candidates are built. This exercises
the renderer side of minimap movement, not native mouse handling/capture overhead.

Set `C3X_RENDERER_PREVIEW_CYCLES=6` for 30 minimap revisits, or `7` for 30 zoom
revisits. The supported range is 2–10 cycles. Every repeat still checks image
parity. The comparison script accepts `--scenario navigation`, `--baseline DIR`
and `--candidate DIR`; it separates first-use/revisit distributions and emits a
nearest-rank p95 only for groups with at least 30 samples. Current logs also
expose device-recovery counts so successful pixels cannot hide reset stalls.

## Follow-up findings and current measurements

The regular-grid regression found that 4,096 distinct float vertices all started
in one of 8,192 hash slots. Adding a final avalanche mix removed that pathological
probing without changing vertex equality or triangle order. The 640×480 diagnostic
then measured 2,739.275 ms median zoom, versus 4,445.532 ms in the preceding pass;
revisit median was 1,941.741 ms. All five images matched exactly. Minimap gains
were concentrated in expensive destinations, not universal.

The initial primary-resolution run exposed a real 192 MiB geometry-cache failure
at 64-pixel tiles. Lossless R16 indices (R32 for large chunks) allow all 1,045
foreground/companion tiles to fit; the cap was not raised. At the initial
128-pixel view, tracked geometry fell from 124,597,432 to 113,240,686 bytes.
R16/R32 source-shadow rasterization, including mixed formats and alpha cutouts,
passed full pixel comparison. The four zoom images available from the failing
R32 run also matched the R16 candidate exactly.

Subsequent retries were driver device removals, not byte-accounting failures.
The preview executables lacked the large-address-aware flag that `ep.c` already
sets on Civ III. Benchmark and ordinary preview builds now match that memory
model. This corrects the test harness; it is not an additional in-game speedup.
The matching 960×640 candidate completed both zoom cycles with exact repeated
image parity and no reset/retry errors. Actual zoom changes still took about
2.8–6.7 seconds, so the zoom targets remain unmet.

Matched large-address-aware 960×640 minimap runs now have six cycles, including
30 revisits. The compact-index candidate measured:

- Five first-use moves: median 1,151.358 ms, maximum 3,308.133 ms. No first-use
  p95 claim is made from five samples.
- Thirty revisits: median 3.964 ms, p95 317.483 ms, maximum 6,016.227 ms.
- All six images and all repeats matched exactly; device recoveries were zero.

The large revisit outlier rebuilt evicted terrain geometry for animated-resource
compositing even though the immutable viewport bitmap remained cached. A bounded
eviction preference now retains recently used animated-view meshes ahead of
bitmap-only static views. This is not an additional pin: active-frame protection
and the byte cap remain unchanged, and the preference expires after 32 geometry
generations. The matched follow-up measured **92.032 ms p95, 92.462 ms maximum**
across 30 revisits, with no geometry builds on any revisit. All six images and
all repeats matched exactly, with zero device recoveries. First-use moves still
took up to 3,390.935 ms. This meets the cached-revisit target on this fixture,
not the cold minimap or overall interaction goal.

The same candidate passed both zoom cycles with exact image parity against the
previous candidate, and the production animation witness passed zoom-return,
six changing poses with zero terrain builds/uploads, scroll and removal parity.
Zoom remained 2.8–6.7 seconds. Executable indexing, projection, eviction and
native bridge contracts passed 39 cases.

Receipts and logs: `native/build/camera-zoom` (hash fix),
`native/build/camera-laa-960` (complete wide zoom), and
`native/build/camera-nav-laa-{baseline,candidate}` (matched navigation).
The eviction candidate is isolated under `native/build/camera-animation-priority`
and its zoom/animation witness under `native/build/camera-priority-zoom`. Neither
has been staged or installed. The overall interaction goal remains open.

## Cache-budget experiment

Set `C3X_ZOOM_LARGE_CACHE=1` in the isolated benchmark build to double the CPU
natural mesh tier to 192 MiB / 2,048 entries and use 384 MiB / 4,096 GPU tile
entries. `C3X_ZOOM_GPU_CACHE_MIB=768` selects the larger GPU experiment with
8,192 entries. Normal builds are unaffected by these benchmark-only definitions.
The subsequent backdrop-reuse pass also uses a 160 MiB resource-backdrop tier in
this experiment (32 MiB in normal builds). Viewport images, source shadows, unit
caches and the 64 MiB prefetch sub-budget remain unchanged. These numbers do not
represent total process or GPU memory.

The witness logs `CAMERA memory` after each interaction, outside its timer:
available virtual address space, the largest free region, and total user address
space. The comparison receipt reports minima over those samples; these are not
peak-allocation measurements and omit the memory occupied by a real Civ III game.
Keep LAA enabled, and require pixel parity and zero device recoveries as well as
timings. A larger cache cannot remove cold construction work or guarantee that a
large live game will have enough address space.

Matched 960×640 runs, current geometry and the same frozen shader/asset inputs:

| GPU / natural CPU cap | Median of five repeat zooms | Lowest sampled free VA |
| --- | ---: | ---: |
| 192 / 96 MiB | 4,307.351 ms | 2,153,504,768 bytes |
| 384 / 192 MiB | 2,718.726 ms | 1,884,659,712 bytes |
| 768 / 192 MiB | 124.228 ms | 1,635,328,000 bytes |

All five images and repeated views matched exactly; recoveries were zero. At
768 MiB all five projections fit (785,094,244 tracked bytes), so repeat zooms
built zero tiles instead of rebuilding every tile. The independent seven-cycle
run retained that zero-build behavior across 30 revisits: median 136.998 ms,
p95 252.512 ms, maximum 323.404 ms. This is a large gain but still misses the
50 ms response / 150 ms completion targets. First-use changes still take seconds.
The self-comparison receipt for that longer run records its distribution and
parity, not a second independent speedup comparison.

The 1280×720 stress test exposed two separate defects. The 192 MiB control cannot
fit its active 96-pixel frame and returns a cache-budget failure. The first
768 MiB attempt encountered an access violation in a tree lookup. Inspection
found that the natural compiler retained a reference into the river-page vector
while nested height queries could move/evict those pages. It now retains an
immutable owned page for that tile's compilation; the LRU still contains at most
16 pages and the extra active reference is released when the tile finishes.
The executable regression forces 32 intervening page queries, reset and release;
312 source-parity samples and AddressSanitizer/UndefinedBehaviorSanitizer pass.

With that fix, both 1280×720 zoom cycles pass exact repeat parity with zero device
recoveries and at least 1,349,046,272 sampled free VA bytes (largest free region
1,256,718,336 bytes). But the five projections no longer fit in 768 MiB: repeat
zooms rebuild geometry and take 3.1–5.3 seconds. Larger caches alone are not a
resolution-independent solution. The next work remains projection-independent
geometry reuse, reduced cold construction, and faster animated-view composition.

The exact lifetime-fix DLL also passed the 960×640 fixture against the preceding
768 MiB candidate with zero changed pixels at every zoom and zero device
recoveries. Its five revisits took 77.707–141.747 ms (median 97.032 ms), with zero
geometry builds. That short run is not a replacement for the longer percentile
measurement above. The updated river lifetime, zoom-mesh and bridge suite passes
41 cases; the standalone lifetime test also passes under both sanitizers.

Evidence directories: `native/build/camera-cache-control`, `camera-large-cache`,
`camera-cache-768`, `camera-cache-768-repeat`, `camera-cache-control-stress`,
`camera-cache-768-lifetime`, and `camera-cache-768-fixed-zoom`. Failed attempts are
retained as diagnostics; do not
count their exit logs as successful frames. Candidate DLLs are isolated, not
staged, installed, or a live-game memory-safety certification. No Civ III patch
symbols or addresses changed; no patch-table action is required.

## Indexed ground and cross-view resource backgrounds

Ground passes now hand their unique corners and exact indices directly to the
GPU uploader. The old path expanded each grid into triangle vertices and then
hashed it back into an indexed mesh. The mixed object-shadow pass still uses
ordinary indexing. An executable witness checks the entire expanded triangle
stream at six grid densities and six layers, scratch reuse, and cancellation.
The 1280×720 comparison in `native/build/camera-direct-grid` matches every image
and repeat exactly, with identical GPU byte counts. Timing gains are modest and
not universal; this alone does not remove the seconds-long cache-miss cost.

Animated resources now reuse the static scene-linear MSAA color/depth patches
behind them across exact camera revisits. Each block is keyed by the complete
static frame signature and rectangle; that signature includes camera, scene,
environment, wrap, ownership, content and device generation. Poses and their
shadows are drawn fresh. The byte-bounded LRU protects the current view against
its own scan; uncached blocks render normally and optional cache allocation
failure does not discard the valid background. The cache is cleared on reset.
The byte cap is 32 MiB normally and 160 MiB in the experimental large-cache tier.

At 960×640 with 768 MiB GPU geometry / 192 MiB natural CPU data, the five zooms
use 41 retained backdrop blocks (145,600,512 bytes), with zero misses on every
repeat. The matched seven-cycle witness measures 30 repeat zooms:

- Median: **136.998 → 55.156 ms**.
- p95: **252.512 → 88.861 ms**; maximum **92.368 ms**.
- All five images and all repeated views match exactly; zero geometry builds
  and device recoveries on revisits. Sampled free VA remains at least
  1,602,125,824 bytes; this is not a live-game memory certification.

The completion-time target passes on this warm fixture; the 50 ms first-response
target remains unmet. Cold changes still take seconds. The same DLL passes
zoom-return parity, six animation poses with zero terrain builds/uploads, and
exact scrolling/removal parity. The focused indexing, cache-budget, native bridge
and animation suite passes 44 tests. Evidence: `native/build/camera-backdrop`,
`camera-backdrop-repeat` and `camera-backdrop-nav`. These are isolated candidates;
no staging, installation, native patch additions or live-game pass is implied.

The matched current-geometry minimap control is
`native/build/camera-backdrop-nav-control` (the older animation-priority fixture
contains different cliff geometry and is not a valid image baseline here).
Across 30 revisits, p95 is **92.763 → 47.861 ms**, maximum **48.782 ms**;
median stays near 4 ms because four destinations have no animated resources.
All six images and all repeats match exactly, with zero device recoveries. Five
first-use moves take 610.774–3,149.261 ms, still outside the cold-move target.

`python3 Renderer/renderer.py integration resources` also passes with the normal
32 MiB backdrop / 192 MiB geometry budgets and current production shader inputs:
192 dependency-selected tests, native build checks, changing animation poses,
zoom-return, scrolling and removal parity. Its disposable receipt is
`lab/out/integration/resources.json`. This checks the normal build separately;
the low-budget build is not claimed to achieve the experimental warm timings.
