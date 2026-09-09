# Zoom performance verification

## Cold-view query/index pass

This pass removes redundant CPU work without raising cache limits or changing
the terrain samples, triangle order, materials or renderer ownership:

- Natural terrain and mountain grids now emit indexed vertices directly. A
  mountain patch retains 4,225 corners and 24,576 indices instead of expanding
  24,576 full vertices and hashing them back into the same mesh. First-reference
  order, coast clipping and append offsets are preserved; unused corners do not
  enlarge bounds. The shared Lab adapter still supports triangle-list output.
- The underlying relief query now knows when direct hill/mountain sources and
  analytic dunes are zero because separate natural meshes own those surfaces.
  Its flat certificate still observes the complete topology support. Away from
  volcanoes, coastal queries retain the exact coast rim and river attenuation
  without evaluating the discarded hill/dune/material expressions. Volcanoes,
  missing topology and the non-fidelity provider keep the existing path.

The isolated high-memory DLL is
`native/build/camera-cold-query-zoom/C3XRenderer.dll`, SHA-256
`cbcd840e6550e559fae6cd1db002429f3541a71f3cc95a8aaf0d7ecd1198c85d`.
The matching control uses this exact DLL and frozen runtime inputs with
`C3X_RENDERER_GRID_INDEX_CONTROL=1` and
`C3X_RENDERER_RELIEF_QUERY_CONTROL=1` in a fresh process. These are diagnostic
controls, not user configuration options. Both processes retain the prior
world-mesh sharing and full-size viewport caches.

At **2240x1192**, all five independent zoom images and all 30 repeated images
are byte-identical. First-use zoom changes have median **6,577.333 -> 5,722.332
ms** (about 13% lower), maximum **8,974.436 -> 7,775.887 ms**. Four first-use
changes are not enough to report p95. Warm revisits measured median **87.797
ms**, p95 **108.641 ms**, maximum **119.212 ms**, with zero builds/uploads,
fallback or recovery. This is a cold-build optimization, not evidence of a new
warm-cache speedup. Minimum free VA was 1,522,552,832 bytes; largest free region
at the minimum sample was 1,415,073,792 bytes. Geometry residency is unchanged
at 685,584,054 bytes. Receipt: `native/build/camera-cold-query-zoom/comparison.json`.

The separate grid-only A/B also reproduced all five images exactly:
`native/build/camera-grid-index-candidate/comparison.json`. Its timings were
noisy, including a slower first 96-pixel view; do not present it as a uniform
speedup. The combined same-DLL comparison above is the current measurement.

Cold views still take seconds and do **not** meet the targets below. Further
work must address remaining terrain/coast queries, new natural geometry and
upload rather than claiming cache hits represent first-use latency. This DLL
is not staged or installed; `bin/C3XRenderer.dll` remains `1a2de66d...`.

## Current full-size optimization

Active acceptance dimensions are **2240x1192**, for both the centered five-level
zoom cycle and minimap navigation. The previous evaluation DLL below remains
staged; the new candidates have not been staged, installed or launched in Civ III.

Latest evaluation build: `native/build/world-cache-evaluation/C3XRenderer.dll`,
SHA-256 `d90fdad0a1865d2138007d3614b0b9c31d16e9d0025b18e3814f9a9f9b647c29`.
It was rebuilt from current code and checked against current production shaders,
not the frozen A/B shader snapshot. All 30 full-size zoom revisits passed exactly,
with zero mesh builds/uploads, fallback or device recovery: median **78.457 ms**,
p95 **96.007 ms**, maximum **109.585 ms**. First-use changes still took
3.87–7.81 seconds. Sampled free VA stayed above 1,522,757,632 bytes. The native
ABI/scheduling/fallback and RGB555/RGB565 blit smoke tests also passed on this exact
DLL. Its `comparison.json` is a self-comparison for distribution/repeat parity,
not a new independent speedup claim. The staged DLL remains the previous
evaluation (`1a2de66d...`); running INSTALL alone will not select this new DLL.

The current implementation addresses three separate costs:

- Natural terrain/tree GPU meshes are owned once in world coordinates. Each
  camera entry holds a weak key and owner version; the vertex shader applies the
  selected projection. Active draws pin both owners inside the existing geometry
  budget. Eviction or a rebuilt owner's version rejects old draw identities.
- Retained animated viewports record weak draw identities, checked as a complete
  set before restore. The complete frame signature still validates scene, camera,
  visibility/ownership, environment, topology and asset/device revisions. Poses
  are recomposed, never frozen into the terrain bitmap. Pixel-block prefetch
  borrows both camera and shared natural layers, or skips an incomplete set.
- Draw lists now grow amortized instead of reserving exactly one additional
  tile at a time, which repeatedly copied the entire preceding layer.

The isolated high-memory candidate uses 768 MiB geometry, 192 MiB CPU natural
meshes, **128 MiB retained viewports** and **288 MiB animation backdrops**.
World owners and camera entries share the geometry byte budget; its entry limit
is 16,384. Normal-build byte budgets remain unchanged. These are separate cache
ceilings, not reservations and not a total-process memory guarantee.

An independent A/B used the exact same DLL and frozen runtime inputs with
`C3X_RENDERER_SHARED_NATURAL_CONTROL=1` only in the control process. All five
images passed existing parity thresholds (0–3 pixels exceeding two channel
levels per image); all repeated images were exact. Sharing alone reduced warm
revisits from 4.90–9.18 seconds to 0.92–1.77 seconds, with zero mesh builds on
revisits and about 407 million additional free VA bytes at the minimum sample.
Receipts: `native/build/camera-shared-natural-ab-control` and
`native/build/camera-shared-natural/comparison.json`. This diagnostic switch
selects a fresh-process control, not a runtime user setting.

Retaining the full-size images then removed the rerasterization cost. Thirty
revisits with amortized draw lists measured median 131.756 ms, p95 202.089 ms,
maximum 266.257 ms, with no mesh builds, fallback or recovery and exact repeat
pixels (`native/build/camera-shared-natural-linear`). The subsequent identity
restore measured 91.590 ms median / 118.705 ms p95 over 30 warm zoom changes.
The cross-build image comparison against the preceding candidate was rejected:
even its initial cold images differed. It is not an optimization-parity receipt;
a same-DLL control supplies the final comparison instead.

The final matched full-size zoom run uses DLL SHA-256
`188ace56f04c16c6b7d88e44211a9f3655c5a99db1e6313d26dc64ce45968be1`
and identical frozen shaders/assets in both processes. Across 30 warm changes:
median **6,830.628 -> 82.025 ms**, p95 **8,982.259 -> 106.448 ms**, maximum
**9,435.036 -> 111.056 ms**. Every shared-cache revisit builds/uploads zero tiles;
GPU residency stays at 685,840,246 bytes. All five independent images pass
(0–3 pixels over the existing channel threshold); all repeat images are exact,
with zero fallback or device recovery. Minimum free VA is 1,520,443,392 bytes,
largest free region 1,460,088,832 bytes. Receipt:
`native/build/camera-world-cache-zoom/comparison.json`, paired with
`native/build/camera-world-cache-control`. The 150 ms final-quality warm-zoom
target passes; the separate 50 ms response target does not. First-use changes
still take 4.48–8.55 seconds.

The six-destination **2240x1192 minimap** witness now passes all 30 repeat images
exactly, with no fallback or recovery: median **30.680 ms**, p95 **66.046 ms**,
maximum **76.811 ms**. This meets both cached-move targets. Minimum sampled free
VA was 1,639,870,464 bytes, largest free region 1,556,746,240 bytes. Receipt:
`native/build/camera-world-cache-navigation/comparison.json` (self-comparison,
distribution and repeat parity only). Cold first moves still took 341–7,275 ms,
including 341 ms for the nearby overlap move. Distant views spend seconds in
terrain/natural geometry generation and upload. They do not meet the cold target.

Neither these results nor the cache ceilings establish live-game latency, memory
safety or smooth transitions. Targets below remain unchanged.

Verification of the current normal-budget code passed 210 selected regressions;
98 focused workbench/cache/zoom checks also passed after the test-harness changes.
All six production behavior replays passed using private copies of both the DLL
and Lab executable: normal/reduced scrolling, world wrapping, resource animation
with zoom return/scroll/removal, and daytime/nighttime unit matrices. Receipt:
`native/build/world-cache-integration/verified/results.json`, DLL SHA-256
`54a870c155d51be0998dc869a6c3826a294e702ae5f986bc823f0f546594fd6d`.
The earlier shared-output integration attempt encountered file locks, mismatched
completion records and crashed verifier processes; it is not a successful receipt.
The wrap failure did not reproduce with both binaries isolated. Failed evidence
is preserved; only this task's two confirmed crashed verifier processes were
stopped, not Civ III or other tasks' processes.

`native_render(..., candidate=..., preview=...)` can now select a private preview
without rebuilding the shared executable. Offscreen preview exceptions log code,
module/offset and stack addresses and exit instead of leaving a hidden error
dialog. A controlled Windows exception verified this diagnostic path. No such
exception handler is installed in Civ III. The current-code guard also preserves
the original CPU projection for non-native diamond aspect ratios.

Remaining work is cold geometry generation/upload, the tighter 50 ms response
goal, and combined zoom/navigation working-set stress. Separate repeated-camera
benchmarks are not a claim that arbitrary mixed camera paths remain resident.

## Previous evaluation handoff

The prior requested in-game evaluation used the following snapshot. The
unfinished mutable ground-point scratch-cache change was withdrawn; no partial
implementation remains. The evaluation DLL is
`native/build/zoom-evaluation-960/C3XRenderer.dll`, SHA-256
`1a2de66d8dddc3e9698c883495945470c68536b124a8b7d08a652bb6f55590b3`.
It uses the experimental 768 MiB GPU geometry, 192 MiB natural CPU mesh and
160 MiB resource-backdrop caps, plus the unchanged 32 MiB viewport tier. These
are cache budgets, not reservations or a cap on all process memory. Ordinary
build defaults remain unchanged; rebuilding normally loses these higher caps.
The exact DLL is staged in `bin/C3XRenderer.dll`; hashes match. The previous
staged DLL is preserved as `native/build/zoom-evaluation-960/previous-staged.dll`.
Neither installation nor game launch was performed.

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
fixture in the current `Z` cycle: 128, 96, 64, 192 and 160 pixels, then repeats.
This is the centered range (normal plus two farther and two closer levels).
Historical measurements below used the old 128, 112, 96, 80, 64 ladder; the
comparison tool recognizes both sequences but never compares mismatched ladders.
It retains the
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

## Centered zoom range and actual-size check

The injected `Z` control now selects 50%, 75%, 100%, 125% and 150%, with
normal in the middle. Its render-time synchronization accepts both close-ups.
The existing staged evaluation DLL already accepts these numeric projections;
its bytes were not changed. Applying the new controls requires rerunning
`INSTALL.bat`; the automated compile check did not install or launch the game.

Verification passed the approved injected compile/injection smoke test and 49
focused tests, including execution of the actual key-handler/native-sync C,
inverse picking, world-mesh reprojection, unit body/HUD offsets and native-unit
suppression at every level. All seven 960x640 replay cycles passed, including
both close-ups, exact repeat pixels and zero device recoveries. Across 30 warm
changes, median was 46.650 ms and p95 90.019 ms. The receipt under
`native/build/camera-centered-zoom` is a self-comparison recording distribution
and parity, not an independent speedup measurement or comparison of different
zoom ranges.

The latest captured game viewport is 2240x1192, not 960x640. An additional
full-world replay at those dimensions passed two complete cycles with exact
repeat pixels and zero fallback/recoveries, including 160/192-pixel close-ups.
However, its five repeat changes took **5.744–14.170 seconds** and each rebuilt
all GPU tiles for that projection. The 768 MiB geometry cache remains full;
five raw full-view BGRA bitmaps alone also exceed the separate 32 MiB viewport
budget. This is direct evidence that small-view warm-cache results do not prove
responsiveness at the user's game size. The minimum sampled free VA was
1,154,306,048 bytes in the standalone process, not a live-game safety guarantee.
Logs and images: `native/build/camera-centered-live-size`.

Use **2240x1192** for the next interactive acceptance measurements (zoom and
minimap). Keep 960x640 as a compact regression fixture and retain the existing
latency targets below. The actual-size replay is a fully visible synthetic
world, not the user's exact fog/capture workload. Projection-independent GPU
storage, scalable retained images and reduced cold-build work remain necessary;
this range change does not claim to fix the remaining performance issue.

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

These are engineering targets, not measured achievements. The original primary
workload was 960×640 with animated map objects, with 640×480 fast diagnostics
and a 1280×720 stress case. The actual-size check above supersedes the primary
acceptance dimensions with 2240×1192; prior measurements retain their original
dimensions and must not be relabeled. Preserve current image quality, ownership,
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
