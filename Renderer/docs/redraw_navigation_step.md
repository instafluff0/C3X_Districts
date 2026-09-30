# Redraw and navigation performance step

This bounded follow-up to the city-light index adds an authoritative standalone
camera witness, opt-in submitted-work accounting, and one exact reduction:
single-sample HDR color targets are sampled directly after their render targets
are unbound. The two duplicate resolved textures and their copies are unnecessary.
No visibility policy, geometry, material, lighting equation, water/reflection
coverage, shadow-caster selection, or injected patch changes in this step.

## Camera witness

`Renderer/sandbox/camera_witness.h` copies authoritative tile occurrences from
the developed-scene fixture. It changes their anchors and wrapped coordinates,
retains a generous capture halo, and calls the existing camera begin, readiness,
and GPU adoption APIs. Each adopted ticket, identity, complete frame metadata,
and tile array must match the submitted capture. The production preparer owns
geometry placement. Separate legacy offsets remain disabled in FRESH.

The sequence covers origin, diagonal pan, changed occurrence strip, horizontal
world wrap, jump, settled 1.25× zoom, and pan at 1.25×. Logs distinguish submission,
preparation wait, adoption, first prepared render/Present call, and subsequent
steady samples. First-view latency includes preparation; it is not physical
scanout latency. Preparation `geometry_ms` and `draw_ms` are wall-clock spans of
the existing preparer, not validated GPU timestamp durations. Explicit untimed
captures save the ordinary tone-mapped color and exact packed scene D24/S8 depth.

This witness uses production preparation/adoption with a standalone fixture.
It does not qualify Civ III capture, native map/HUD allocation, suppression,
compositing, or fullscreen gameplay. The older in-place zoom trace is retained;
its unused camera offsets do not qualify navigation.

## Submitted-work accounting

`C3X_SANDBOX_PASS_COUNTS=1` enables bounded per-frame counters for the instrumented
FRESH passes and layers. Rows report tested/admitted records and vegetation
instances, actual draw calls, submitted index references/triangles (including
instancing), instrumented stream/constant-buffer upload bytes, clears/fullscreen
target footprints, actual texture-copy footprints, and selected cache events.
Vegetation alpha-depth and color draws both count; the old draw-call field now
includes these previously omitted submissions. Main and reflection vegetation
remain separate from shadow casters. Encapsulated native-provider updates and
inactive overlay systems are outside this accounting closure.

These are CPU submission facts. Index references are not unique vertex-shader
invocations; footprints are not fragment invocations, measured memory bandwidth,
or GPU time. Counts are copied and printed outside the client timing spans.
Frame-call wall time still includes instrumented submission overhead and Present
backpressure. No forced-completion timer is presented as a GPU duration.

## Single-sample HDR alias

Only FRESH's resolved static-cache and moving-scene color targets use aliases at
one sample. Both pointer slots retain their own COM reference, preserving ordinary
reset, resize, and swap ownership. `LinearTarget::bytes()` counts shared storage
once. Multisample targets retain their original separate resolve path. The
startup diagnostic `C3X_SANDBOX_RESOLVE_COPY_REFERENCE=1` retains the complete
unoptimized copy implementation for differential comparison.

At the 2240×1260 scene target, both guarded HDR targets are 2248×1268. A zoom
redraw removes 5,700,928 copied pixels, or 45,607,424 bytes (about 43.5 MiB), and the
same duplicate texture allocation. A stationary cached scene removes the moving
target's one copy (21.75 MiB/frame). Draw lists, submitted triangles, and scene
depth are unchanged. GPU differential tests exercise single-sample HDR pixel
equality over resize and swap/reset. The two-sample check proves separate
allocation only; it does not compare resolved multisample pixels.

## Evaluation protocol

Protected evidence is in `Renderer/.cache/redraw-navigation-step/`; preserve it
alongside the accepted city-light-index evidence. It contains starting sources,
the accepted indexed tuple/shaders, the diagnostic copy control, frozen candidate
source closure and binaries, complete input fingerprints, raw logs, receipts,
and untimed color/depth captures. No installed tuple or reference image is replaced.

`Renderer/tools/measure_redraw_navigation.py` uses one common x64 client and
identical scene, pack and shader bytes. The copy-reference and alias arms use
the same candidate DLL. Repeated pairs reverse execution order. The retained
windowed case is recorded separately from a controlled full-guest-area case
using the same standalone HWND with a borderless style. Each invocation re-queries
desktop resolution/refresh, client dimensions, scene target, DPI awareness, and
actual swapchain dimensions/settings. Borderless Present remains standalone
windowed DXGI presentation, not Civ III's presenter.

## Results

The guest remained 2240×1260 at 60 Hz. The original windowed client area was
1104×591; both modes used a 2240×1260 BGRA swapchain, two buffers, flip sequential,
stretch scaling and `Present(1,0)`, without exclusive fullscreen. Effective window
DPI awareness was 2 for the original window and 1 for the borderless control,
both at 96 DPI. The full-area case retains the active-zoom floor.

The table pools two reversed-order pairs per case, excluding the three separately
logged transition samples in each 90-frame trace. Times are frame-call milliseconds.
All raw samples, complete/ warmed trace totals, p99, preparation, first-view spans,
identities and work counts remain in receipts and `measurement-summary.json`.

| Full-area case | Arm | Mean | p50 | p95 | Worst | Calls over 16.67 ms |
|---|---|---:|---:|---:|---:|---:|
| Noon zoom | Copy reference | 144.46 | 131.90 | 257.36 | 1495.27 | 174/174 |
| Noon zoom | Alias | 132.15 | 132.08 | 165.94 | 408.20 | 174/174 |
| Night zoom | Copy reference | 143.00 | 143.03 | 241.60 | 491.06 | 174/174 |
| Night zoom | Alias | 141.23 | 140.64 | 268.78 | 446.94 | 174/174 |
| Noon stationary | Copy reference | 18.73 | 16.54 | 33.59 | 220.73 | 52/174 |
| Noon stationary | Alias | 17.06 | 16.61 | 32.93 | 34.22 | 45/174 |
| Night stationary | Copy reference | 18.15 | 16.57 | 17.22 | 256.78 | 49/174 |
| Night stationary | Alias | 18.48 | 16.58 | 23.54 | 274.05 | 44/174 |

The noon mean difference is dominated by stalls in one reference run; its second
mean was 133.31 ms. Windowed noon means were 133.11/130.49 ms, with nearly equal
medians. The accepted indexed DLL also measured 131.33 ms noon and 141.83 ms night
with the common windowed client. These results demonstrate exact work/storage
reduction and a small night-zoom change, not a consistent large frame-time gain.
The deadline counts are measured call durations, not observed physical dropped
frames. Active zoom remains far outside a 60 FPS budget.

Corrected navigation uses a view capture with a halo, then settles at each point;
it is a different workload from continuous in-place zoom. Each steady mean below
pools 54 warmed samples per arm. The first-view ranges include all four paired
runs and begin after the initial fixture load/prewarm, which is logged separately.

| View | Copy steady mean | Alias steady mean | First prepared view, seconds |
|---|---:|---:|---:|
| Origin | 30.52 | 27.08 | 14.21–14.94 |
| Diagonal | 27.97 | 28.34 | 1.90–2.15 |
| Changed strip | 27.01 | 27.10 | 2.11–2.43 |
| Wrap | 35.34 | 31.01 | 17.86–21.36 |
| Jump | 26.19 | 26.98 | 5.45–6.53 |
| Settled 1.25× | 26.17 | 26.67 | 2.33–2.59 |
| Pan at 1.25× | 27.84 | 26.66 | 1.68–1.80 |

Warm diagonal/zoom-pan submissions add 174 and remove 44 occurrences, and report
118,043,968 geometry upload bytes. The changed strip replaces 259 occurrences;
wrap changes the captured occurrence set and reports 775,049,290 upload bytes.
These spans expose expensive preparation, without proving which subset of its
geometry work is avoidable. All seven anchor/depth views differ as expected;
the diagonal depth overlap follows the submitted 96/48-pixel shift.

## Correctness and attribution limits

The same-frame GPU proof in `Renderer/tools/verify_scene_copy_parity.py` extends
current sources in a private DLL/client. It snapshots the exact completed static
and moving HDR targets, runs the existing bloom/display path through copied views,
and compares it with direct views. All 14 noon/night camera/zoom images and packed
depths match exactly. This proof is untimed and does not replace the timing binary.
The smaller GPU contract also verifies exact single-sample raw HDR values,
resize, swap/reset, and separate two-sample allocation.

Independent runs exhibit existing sparse variability: 55–143 changed color pixels
in the paired daytime views, comparable reference/reference and alias/alias noise,
and one strip repeat differs at one depth pixel by three D24 units. Independent
cold wrap/jump/zoom-pan endpoints have exact depth; their color differences are
72/68/965 pixels, respectively. These are retained findings, not a claim of
pixel-exact navigation across process/history changes or a new reference approval.

A warmed noon zoom submits about 5,896 draws and 10.16 million triangles. Main
scene, reflected scene, reflected terrain materials and water contribute about
2,487/1,363/718/1,260 draws. Vegetation contributes 128 actual alpha/color draws and
1.57 million triangles across main/reflection; every admitted chunk instance is
submitted. The copy-reference and alias arms keep identical draw/triangle work.
Layer counts are a stronger attribution fact than CPU stage timers: most total
frame time accumulates in Present/backpressure, and causal controls vary with
driver state. Removing all vegetation, reflections or the water pass still leaves
roughly 120–134 ms in the controls. No material gain from conservative vegetation
culling has been established, so its selection policy is preserved.

Next investigation should validate asynchronous device/queue timing and measure
prepared geometry builds/reuse before selecting a larger change. Terrain submission
batching and world-content preparation independent of view adoption are concrete
candidates; neither is implemented or promised here. They must preserve source
units, clipping, reflection coverage, caster ownership and the geometry-epoch guard.

Dependency-selected forest tests pass (126 checks, one optional skip). Focused
camera, HDR ownership, projection, reflection, light-index, skin-shadow and mesh
cache contracts are exercised separately. Three stale mesh-cache fixture schemas
were repaired to keep current ownership/grid tests executable. The old disposable
shader receipt was repaired only after all 62 freshly generated outputs matched
current files exactly; no generated shader or asset changed. Parallels transport
failures and process observations are retained separately from renderer results.
No injected source, patch-table entry or user action is required for this step.

The seven navigation timings above use the original stress capture, which marked
a margin of twelve full tile widths/heights as renderable. Native capture uses
twelve tile coordinates and distinct appearance/topology roles. Those waits are
stress observations, not native gameplay timings or proof of redundant uploads.
The ownership follow-up uses a corrected capture model and fresh controls.
