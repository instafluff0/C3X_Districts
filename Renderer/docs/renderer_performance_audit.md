# Renderer performance audit — 2026-09-29

**Subsequent source review:** see [performance engineering review](performance_engineering_review.md)
for current-source reruns, busy-city/night and many-unit scaling analysis,
priorities, and a correction to this report's standalone scroll/jump harness
scope. Some diagnostic/flush fixes and transition logging described below as
future work are now present. The binary identities below still define these
historical measurements; the follow-up does not reinterpret them as a newer build.

**Current quality policy:** the user subsequently permitted imperceptible
differences and brief detail reductions during transitions, while retaining
full-quality settled views and targeting sustained 60 FPS. See the performance
and perceptual quality target in the linked review; it supersedes blanket
restrictions on temporary detail reductions below.

## Conclusion and quality requirement

The renderer does not currently meet near-60-FPS navigation or idle performance consistently. There are two distinct problems: the full-quality scene path is much too expensive during dense-scene zoom, and Civ III integration adds composition, command-service and presentation costs even when very little world geometry is visible. Improving only one of these will leave the other unresolved.

**Graphics quality is a requirement.** The recommendations preserve real geometry projection during zoom, authored meshes and materials, source normals, current lighting, water, waves, reflections, shadows, animation, fog and native overlays. Lowering resolution, scaling a finished image instead of projecting geometry, dropping effects, weakening visibility rules or delaying authoritative UI updates would not satisfy this audit's target. Existing reference images were not replaced.

The strongest opportunities, in order of expected impact, are:

1. Make moving/zooming scene work proportional to the contributing geometry and pixels. Separate persistent world content from camera occurrence lists; build pass-specific, spatially selected draw batches. Preserve static pixel reuse for cases where it is valid.
2. Reduce animated-map dependencies and full-surface transfers in the retained native compositor while preserving exact Civ III operation order, aliases, pixel formats and fixed-size UI.
3. Coordinate presentation readiness and reliable command service with one bounded frame opportunity. A fast publication call and a high presentation counter do not establish a responsive camera.
4. Keep expensive first-use preparation out of interactive camera/zoom paths, with explicit residency and memory budgets.
5. Remove routine diagnostic formatting/output from ordinary rendering, then optimize measured secondary costs such as temporary allocations and repeated unit preparation.

These are engineering recommendations, not implemented speedups or a new milestone ladder. No production source, staged binary or injected patch was changed by this audit; pre-existing working-tree changes were preserved. No new patch-table capability is necessary for the first recommendations.

## Evidence and reproducibility

The current checkout was reviewed directly, including its uncommitted renderer work. An isolated optimized x64 DLL and standalone client were built from the audited source, without staging or installation. The recorded C/C++/HLSL source hashes still matched immediately after the measurements. The source fingerprint and per-file hashes are in the local audit receipts. Git HEAD alone is insufficient to identify this working-tree build.

A later final check found concurrent edits to `native/c3x_renderer.cpp`, `native/renderer64_startup_probe.cpp` and `lab/shared/natural/test_data.cpp`. The visible changes include pack configuration/reload and bootstrap/test handling. They were preserved and reviewed as later work; the measured binary identities below remain the authority for these timings. In particular, a changed natural-pack selection can change the workload and needs its own matched measurement. This report does not qualify those later edits.

| Evidence | Identity and scope |
| --- | --- |
| Current-source standalone build | x64 DLL SHA-256 `50b27f58e1268af07e11f5ebec56a73c13a909d67b53358636f43d3ef8afc8f9`; client `399b157d24198aca2e35a38ed6596a968e12324ae11a6e5442fae2e56c093d4b` |
| Source fingerprint | Sorted `path:sha256` source inventory digest `af8790b2ee5ad3cff24dc3651529e554afcc55e71ad44f1c4384133612be6832` |
| Installed/staged live renderer | x64 DLL `4cf3a021647506f03e67aec3d40777b3ea746514f7ee6f1881c5e8b410eaa7cf`; bridge `60110f1454b01c99159e1b065b17883289404ad5ecec2479a31974b6706bf0ca`; helper `54227cdc18b5f70053038d48b7ec42c14e6a4980855de5f2fcd33e05d276619f` |
| Original sandbox | Preserved `sandbox/out/matched-sandbox.log`, `matched-integrated-core.log`, sandbox README and source at `7088721a`; historical evidence, not a new execution of an unidentified old binary |
| 0 A.D. | Existing local source revision `0ed48b3a1fb1b4b718a78869fa497185af55e086`; source architecture review, no 0 A.D. performance benchmark or claim about current upstream HEAD |
| New local receipts | `Renderer/native/build/performance-audit-20260929/`: build/run results, source hashes, six standalone logs, live cadence/result/input receipts and six sampled JPEGs |

Native D3D11 measurements ran serially on the Windows 11 Parallels VM, at **2240 × 1260**, without simultaneous compilation or another renderer workload. Water, waves, shadows and reflections remained enabled. No image capture occurred inside standalone timed frames. The standalone scene uses the existing 100 × 100 BIQ-derived fixture, center `(24,56)`, tile width 128, noon, one scene sample and the existing full-detail asset/shader packs. Its source scene digest is `d778d526d51e36e52af3166e77ff24435aa605da768dbef0d1d38a572ab42283`.

The existing `CAPTURE_DIAGNOSTIC.bat` workflow is useful for reproducing native input and visual defects. Its recording/readback work is unsuitable as an unqualified frame-rate baseline. This audit instead used the existing standalone client and bounded scripted game tests with detailed profiling off. Live tests still attach DebugView and sample the window at 1 Hz, so their observer cost is not zero.

## New measurements

### Full-quality standalone scene

These are wall times for `draw + Present` calls, not measured physical scanout intervals. The replay clock advances one source step per frame at a simulated 30 Hz, while calls execute as fast as presentation permits. Zoom traces continuously change projection; they are not held-zoom tests.

| Workload | Timed frames | Median ms | p95 ms | Worst ms |
| --- | ---: | ---: | ---: | ---: |
| Idle at 1× | 177 | 16.67 | 34.11 | 246.66 |
| Scroll at 1× | 117 | 17.60 | 30.60 | 43.19 |
| Jump/return | 57 | 16.65 | 35.06 | 152.90 |
| Zoom 1×–1.25×, run A | 87 | 133.27 | 150.81 | 520.17 |
| Zoom 1×–3× | 87 | 83.34 | 128.85 | 296.35 |
| Zoom 1×–1.25×, repeat B | 87 | 133.21 | 347.03 | 417.36 |

The modest zoom is approximately **eight 16.7 ms budgets per median call**. The repeat confirms a severe problem while also showing large tail variability. The 3× trace is faster than the modest trace; cost does not simply grow with zoom magnitude. Changing visible object count, overdraw, invalidation and GPU queue pressure must be investigated together.

**Important harness limitation:** `client_x64.cpp` discards the first three frames of each replay arm. In the jump arm this discards the initial transition itself. The jump median therefore cannot establish first-jump responsiveness. The return still produces a 152.9 ms tail. Future transition measurements must include frame zero and distinguish the input trigger, preparation, adoption and first correct presentation.

The measured CPU/driver spans explain part of the work, but do not constitute GPU pass timing:

| Span | Idle median / p95 ms | Scroll median / p95 ms | Zoom 1.25× A median / p95 ms |
| --- | ---: | ---: | ---: |
| Draw call span | 7.45 / 9.03 | 16.68 / 27.35 | 21.86 / 44.27 |
| Present call span | 9.22 / 26.70 | 0.38 / 7.07 | 111.04 / 130.82 |
| Selection and shadows | 2.12 / 3.22 | 2.83 / 6.00 | 2.55 / 3.75 |
| Reflection | 0.23 / 0.37 | 7.29 / 12.05 | 6.52 / 9.41 |
| Static cache work | 0.00 / 0.00 | 0.01 / 0.04 | 8.95 / 15.00 |
| Water and waves | 4.76 / 5.62 | 5.30 / 9.83 | 3.37 / 6.70 |
| Synthetic animated units | 0.30 / 0.45 | 0.39 / 0.78 | 0.30 / 0.46 |

Do not add pass p95 values: the quantiles occur on different frames. Small CPU spans do not imply cheap GPU work. Most zoom wall time is charged to `Present`, where prior GPU work, queue throttling, synchronization, refresh pacing and VM behavior can surface. The data proves a costly complete path; it does not prove that the swapchain API itself consumes 111 ms of useful work.

The standalone presenter normally calls `Present(1,0)`. Live direct composition calls `Present(0,0)` after a nonblocking DXGI latency-permit check. The old sandbox, current standalone and live counters therefore do not have identical pacing semantics. Compare complete normal paths, and use matched presentation controls when attributing the remaining idle gap.

An existing eight-frame completion-probe experiment, on a different candidate, reported roughly 59 ms in reflection and long mirrored vegetation-layer waits. That experiment serializes execution with completion checks and uses unreliable VM completion behavior. It is useful evidence for investigating reflected foliage; it is **not** calibrated GPU timing or a normal-FPS result. Earlier effects-off controls also failed to cure the dense zoom problem. They are diagnostic evidence only.

Preparation is separately expensive: six new arms spent **17.7–20.9 seconds** in scene/reference preparation, uploading about **872 MiB**, then **2.3–4.8 seconds** priming scene and swapchain. These are cold full-fixture costs, not per-idle-frame work or measured live first-load durations.

### Real Civ III, staged renderer

The two new bounded tests used disposable copies of the existing 4000 BC autosave. They ran the installed triad, not the isolated DLL above. Both completed without recorded native renderer failure or early game exit; the scroll test recorded all 32 steps and zoom recorded all ten mouse commands. The original save was unchanged, and the harness executed its configuration/cursor cleanup. The test-generated tracked scene export was restored afterward.

| Live interval | Actual sampled interval, seconds after script start | Successful presentation submissions / second |
| --- | --- | ---: |
| Conservative moving-camera portion | 28.36–45.52 | 28.04 |
| After scroll settled | 46.53–54.60 | 51.92 |
| Longer settled tail | 55.61–73.80 | 51.76 |
| Before zoom commands | 22.02–28.98 | 53.57 |
| Repeated zoom commands | 30.04–42.98 | 35.10 |
| Settled after zoom | 44.01–53.99 | 54.11 |

These are QPC-normalized differences of the helper's successful-presentation counter. They measure submissions, not physical scanout, unique visible frames, frame-time p95 or input-to-correct-camera latency. The moving interval is conservative and omits some earlier scroll work. Some requests hit map limits or arrive while preparation is pending; command count is not distance traveled.

The sampled output shows terrain, animated unit, water, fog and native interface. It also shows a **sparse, mostly unexplored map**, not the dense scene above. Intermediate minimap samples are incomplete; the samples are not a blanket visual-correctness pass. Even this sparse scene fails to sustain near 60 submissions per second. Sparse live integration and dense standalone rendering are separate qualification cases, not interchangeable benchmarks.

During the scroll run, helper private bytes were about 2.63 GB initially and 2.62 GB at the end; game private bytes were about 247–249 MB. That short run does not exhibit unbounded memory growth. It does not establish a safe maximum world size or long-run lifetime behavior.

## What the active renderer actually does

`native/BUILD_RENDERER64.bat` builds the production x64 DLL through `sandbox/resident_scene.cpp` with `C3X_RENDERER64_FRESH`. That translation unit includes the shared production preparer and the fresh scene pipeline. “Sandbox” in the path does not mean this code is inactive. Conversely, many older CPU raster and regional-cache functions in `c3x_renderer.cpp` are compiled but bypassed by the fresh path. Their existence is not evidence that they run every frame.

The normal path is:

```text
Civ III draw traversal and authoritative anchors/state
    → copied visible capture and world observations
    → x86 asynchronous publication / ordered native image commands
    → helper request service / single renderer-worker transaction
    → persistent world meshes and camera occurrence selection
    → static scene, shadows, reflection, water/waves, units, HDR/bloom
    → GPU map fog and retained native composition
    → existing Civ III surface presentation boundary
```

The asynchronous fresh path explicitly rejects CPU map pixels. Full-map readback is not the ordinary idle/scroll/zoom mechanism. World topology auditing is conditional, not a complete world scan on every ambient frame. Source shaders are cached; there is no demonstrated per-frame shader recompilation. Vegetation is already hardware-instanced, and terrain already has shared mesh/region batches. These useful properties should be preserved.

## Bottlenecks and concrete recommendations

### 1. Zoom turns the cheap retained scene into a repeatedly redrawn scene

**Confirmed behavior:** `sandbox/fresh_pipeline.h` invalidates static, reflection and reflected-terrain state on every changed `projection_zoom`; it also invalidates them for camera motion while zoom is not 1×. `display_zoom` is now 1, so geometry is genuinely projected. Static-region strip reuse works well at unchanged 1× projection, but continuous zoom cannot reuse those old pixels correctly. The new static-cache span rises from approximately zero at idle to 8.95 ms median in the modest zoom trace, before accounting for GPU completion.

**Recommendation:** preserve geometry-based zoom. Make rerasterization cheap by retaining world mesh/material ownership independently of the view, selecting only contributors for the current projection, and keeping compatible batches ready. Reuse GPU scene targets and constant/instance streams. Do not try to retain incorrectly projected color/depth, and do not revert to stretching the finished image. Rendering real geometry at each new zoom is necessary; rebuilding unchanged geometry, scanning irrelevant objects and duplicating pass preparation are not.

**Additional invalidation concern:** `FreshPipeline::scene_revision()` combines content signature with `tile_geometry_epoch`. The preparer increments the epoch when it replaces the camera geometry record generation, including cases where the retained world cache may supply existing meshes. That makes a safe lifetime signal also act as a global content-invalidity signal: resident occurrence vectors, static state and shadow references can all reset. The epoch exists for a real pointer-lifetime reason. Replacing it with a content hash alone would risk stale references.

Split persistent world-content revision, immutable mesh lifetime, view occurrence revision and pixel-cache validity. Camera recentering should update occurrence transforms/selections; real topology/material changes should invalidate affected world chunks. Old retained views must keep their owned inputs or frozen samples until retirement. Validate zero unnecessary mesh builds/uploads on resident pans, jumps and zoom; preserve the recent retained-view lifetime fix.

Source: `sandbox/fresh_pipeline.h` around `scene_revision`, `capture` and `draw` (approximately lines 997, 1021, 1981); `native/c3x_renderer.cpp` geometry-generation branch around line 7447.

### 2. Moving reflection and foliage are high-value GPU candidates

**Confirmed behavior:** reflection is reused at idle, but its cache key includes camera X/Y, scene revision and shadow builds. Moving the camera redraws reflected terrain and objects. The new scroll trace charges 7.29 ms median / 12.05 ms p95 to the reflection CPU/driver span alone. Main static pixels can remain mostly cached while reflected objects are rendered again.

The renderer already uses a 0.375-scale reflection target and now computes water bounds with distortion/filter guards. Keep those quality settings and guards. A scissor saves fragment work; it does not automatically avoid CPU selection, instance uploads or vertex processing for objects submitted outside the useful reflection area.

`capture()` scans resident records separately for main and reflection lists and stores wrapped occurrences at ±world span. Draw-time inverse-projection culling then rejects some records. `draw_vegetation_instances()` groups shared meshes, but admission is by record bounds and emits all instances in the admitted record. It also performs an alpha-tested depth draw followed by the shaded draw. This can reduce expensive fragment shading, but doubles vertex and opacity-test work; the best choice depends on actual overdraw and driver behavior. Existing exploratory variants did not establish a stable complete-path win.

**Recommendation:** use the authoritative Civ III basis to spatially index immutable world chunks and select current main, reflected-water and shadow contributors independently. Select wrapped occurrences as transforms rather than duplicating the entire selected record set where practical. Cull vegetation instances conservatively against the real projected region; preserve tree tops, reflection displacement, shore guards and cross-tile overlap. Prepare shared-mesh batches once per relevant visibility revision. Measure selected records, instances, submitted triangles and pixel coverage before changing the foliage prepass.

Keep every reflection contributor needed by the existing visual contract. Skip a reflection pass only when there is provably no visible water/contribution or its complete inputs are unchanged. Periodically updating a stale reflection, reducing its resolution or hiding expensive foliage would require a separate visual tradeoff and is not the proposed fix.

Source: `sandbox/fresh_pipeline.h` `capture`, `source_bounds`, `reflected_water_bounds`, `draw_vegetation_instances`, reflection branch around line 2034; existing completion-probe receipt under `native/build/scene-quality/performance-completion-b/`.

### 3. Native retained composition adds work the direct sandbox did not have

**Confirmed behavior:** the production adapter tone-maps the scene, applies map fog, and samples it into a retained graph of native operations. A changing animated map revision can propagate through dependent copy, quantize, expand, projected-selection and overlay nodes. `RetainedComposition::draw()` collects dependencies, evaluates them, assembles the front, displays it, then copies the complete display to a retained buffer and flushes. The full display copy runs on each changed front.

This graph is required to preserve native operation order, aliased reads, masks, 555/565 canonical pixels and fixed-resolution UI. It already reuses node outputs and has shortcuts for complete partitions and direct read-only inputs; it does not allocate every texture from scratch on every frame. The recent history fix also prevents old map callbacks from keeping old worlds live. Neither “remove the graph” nor “allocate one more large cache” is an adequate answer.

**Recommendation:** retain native semantics but compile the current dependency graph into a smaller execution plan. Cache map-independent native layers and immutable glyph/sprite data. Reevaluate only rectangles/operations that depend on the changing map or changed native inputs. Compact overwritten history after proving alias and lifetime safety. Batch compatible operations without moving them across masking, read-before-write or presentation boundaries.

Audit and count full-surface GPU sweeps. Fuse compatible format/projection/sharpening steps when their numerical and ordering contracts permit it. For the direct surface route, investigate making the retained display-copy buffer current on demand for explicit diagnostic readback or native handoff, rather than copying it every ambient frame. It **is** used by `trial_surface_pixels`; deleting the copy unconditionally would break that witness. Validate menu transitions and fallback before changing its ownership.

At 2240 × 1260, one BGRA image is 11.29 MB and one FP16 RGBA image is 22.58 MB. A full read-and-write sweep at 60 Hz moves at least 1.35 GB/s for BGRA or 2.71 GB/s for FP16, excluding depth, texture samples and other passes. These are arithmetic traffic estimates, not measured hardware bandwidth. Several extra sweeps can matter on the VM even when CPU call spans are tiny.

Source: `native/retained_composition.h` `evaluate`, `assemble`, `assemble_projected`, `draw` (around line 713); `native/gpu_composition_session.h`; `sandbox/resident_scene.cpp`; `native/c3x_renderer.cpp` `trial_surface_pixels` and `trial_visual_shared`.

### 4. Publication, preparation, adoption and presentation are separate deadlines

**Confirmed behavior:** the x86 client copies and queues input without waiting for normal posts. The transport thread executes ordered requests through a single request/response channel. Camera observations have a replace key, while native image/lifetime/action commands remain reliable and ordered. Each RPC wakes the helper and awaits its response. A stream of small operations can consume many service turns and contend with ambient rendering even when game-thread publication is very fast.

The independent cadence uses a 16,667 μs period and 2 ms minimum pause. It tries the renderer transaction gates; BUSY retries soon. The direct worker polls the DXGI frame-latency signal without waiting, with maximum frame latency one. A not-ready surface returns PENDING, which does not request the same early retry as BUSY. Timer opportunities and presentation readiness are therefore separate. Scheduler phase and command-service pressure are plausible contributors to the sparse scene's 52–54 submissions/s; this audit has not isolated their individual shares.

**Recommendation:** preserve one immediate-context owner and ordered authoritative commands. Batch native image operations into a frame/transfer packet to reduce RPC overhead, with explicit barriers for aliases, lifetime changes and final transfer. Coalesce only replaceable observations; do not discard unit actions, edits or UI operations. Give current camera work bounded service slices and avoid obsolete work monopolizing the gate. Coordinate cadence wakeup with presentation readiness and newly committed input, rather than adding more polling threads or a second presenter.

Measure producer copy time, queue age/count/bytes, service time, camera preparation, ordered adoption, visual lock denial, not-ready opportunities and successful presents on the same clock. The small asynchronous fixture already reports approximately 31–47 ms camera-ready steps with negligible uploads, while old-front animation continues at high cadence. That is evidence that a 60-FPS fixture counter can conceal multi-refresh camera latency.

Source: `sandbox/async_publication.h`, `sandbox/async_scene_client.h`, `native/helper_trial/scene_client.h`, `scene_workload.cpp`, `native/visual_cadence.h`, `native/presentation_permit.h`, worker direct-surface branch; preserved `native/build/startup-simplification/freeze-fixed-async/test.log`.

### 5. Camera capture and first-use preparation remain navigation risks

The injected path traverses native drawing with authoritative anchors, captures visible tiles and a halo, and expands async capture to a coherent view. It correctly avoids treating unexplored terrain as authorized detail. Resident caches reduce mesh construction, but complete visible observation copying, occurrence assembly and native traversal still have cost. A small native dirty clip does not necessarily produce a small custom capture.

Keep authoritative capture, fog and picking aligned. Separate changes in world content from visible anchors and presentation metadata; use the existing dirty publication journal for affected chunks. Maintain useful coarse spatial selection rather than making every new viewport reconstruct all resident references. Prewarm meshes, shader variants and unit actions outside interaction deadlines when they can be admitted within budget.

World capture completeness is not the same as GPU residency. Existing page capture is bounded and should stay so. Introduce explicit coverage/residency accounting before claiming that every possible jump is prepared. Large worlds need a measured hot working set, nearby prefetch and bounded background compilation rather than unlimited whole-world allocation.

The documented first reduced native city zoom has previously taken around seven seconds. The new ordinary scroll/zoom tests do not retest that route. Treat native tile-width changes, city-centered zoom, minimap jumps, selected-unit centering and manual drag as distinct routes; their capture/adoption behavior differs. The initial jump excluded by the standalone harness is still unqualified.

Source: `injected_code.c` visible capture/halo, world-page and `patch_Main_Screen_Form_m71_Draw_Tiles` paths; `native/native_composition_owner.h`; `native/render_core/scene_publication.h`; current `docs/retained_renderer_plan.md` evidence.

### 6. Unit preparation can be shared across passes, but is not the first target

Meshes, authored animation palettes and shader skinning are already resident. The synthetic standalone units cost about 0.3–0.4 ms of CPU/driver span, far below the zoom wall-time problem. The live unit path is different and must be measured with representative density.

`direct_units.h::draw_real()` prepares pose transitions, action data, shadow fit and self-shadow rendering for reflected and main draws. Cache preparation once per unit/pose/light/input revision and reuse it across passes where compatible; do not share a single scratch shadow texture across distinct units without preserving each unit's result. Batch material bindings and palette updates when semantics permit. GPU compute skinning is a possible later density optimization, not a justified response to the present sparse-unit measurements.

### 7. Memory and transient allocation need budgets, not larger limits

The dense scene reports approximately 1.17 GB of geometry and 504 MB of materials before all targets and composition state. The standalone device reports about 2.16 GB local usage against a reported 6.87 GB local budget. These are virtual adapter reports, not proof of physical VRAM residency or available host memory.

Retained image and composition pools have separate budgets. The asynchronous x86 publication limit is 128 MiB / 8,192 packets: useful safety bounds, but a backed-up queue of that size is far beyond an interactive latency budget and consumes scarce 32-bit address space. Track peak bytes and oldest packet age, not just accepted/completed counts. Raising limits can postpone failure while worsening latency.

Frame selection, rigid-instance construction, vegetation batches, city-light selection and graph-version vectors use temporary containers. Reuse capacities or a bounded frame arena for short-lived selection/batching data. Preserve immutable node/world lifetime ownership; avoid a pool that accidentally keeps retired worlds alive. Profile heap time before making container replacement a major project.

Unused legacy material-preparation methods exist, but several are not called by the active main pass. Do not attribute their potential targets or old CPU raster costs to current frames without proving execution.

### 8. Normal-path diagnostics are measurable traffic, with unmeasured timing cost

The new 75-second scroll run emitted **8,744 log lines**, including 2,461 `native-operation`, 2,194 `native-map-transaction` and 830 `native-ui-present` records. The latter routines format and call `OutputDebugStringA` without requiring detailed renderer profiling. Attaching the collector can increase their cost.

Gate normal-success operation/transaction logs behind the existing diagnostic setting, or aggregate them into bounded counters. Retain immediate failures and a small bounded recent-event buffer for fault diagnosis. Benchmark with and without an attached collector after this change. This is an inexpensive cleanup candidate, but the current data does not establish how many milliseconds it saves; it cannot explain away the dense standalone zoom failure.

Source: `native/native_composition_owner.h::trace_map` and final native transfer; `native/c3x_renderer.cpp` exported native-image operation wrapper, around line 14785.

### 9. Shader and state optimization should follow actual pass work

The active pipeline binds substantial shared material tables and issues state/constant updates across terrain, rigid features, vegetation, reflection and water variants. Draw-parameter and instance streams already reduce some per-record calls. Several selection/flush paths still build temporary vectors and rebind unchanged resources. Shadow, reflection and alpha depth/color passes multiply geometry processing even when a mesh is shared.

After spatial selection and invalidation are corrected, record draws, state changes, constant/instance upload bytes, texture sampling pressure and overdraw by pass. Cache compatible state bindings and retain pass batches; specialize shaders for invariant pass/material controls while keeping the same output math and texture detail. Anisotropy and negative mip bias affect bandwidth, but weakening them is a quality tradeoff rather than a default optimization. The tiny CPU HDR/bloom span also does not prove its fullscreen GPU work is free. Choose shader or prepass changes from measured complete-path improvements and image/depth comparisons.

## Original sandbox comparison

The original sandbox contains valuable architecture already present in production: persistent meshes and materials, hardware-instanced vegetation, retained static color/depth, wider camera regions filled by strips, cached unchanged reflections, a dedicated water path, source animation palettes and compact postprocessing. Keep those mechanisms.

| Preserved sandbox observation | Current observation / implication |
| --- | --- |
| Normal run 16 ms median / 31 ms p95; flat-color 16 / 32 ms | The VM/presentation route itself has substantial tail behavior. Historical median near 60 does not establish uniform 60 Hz. |
| 1,819 steady samples; scroll 16 / 31 ms, jump 16 / 31 ms | Strong useful baseline, but historical worst jump/return/wrap frames were 109/172/141 ms. First-correct-camera latency was not established. |
| Zoom about 1×–1.2×; old code clamps completed-image scale to 1.35× | Current zoom projects geometry at up to 3×. Reverting image scaling would sacrifice the requested sharp geometry/detail behavior. |
| Matched log geometry 817,419,848 bytes | Current fixture 1,169,024,343 bytes: about 43% more. The views are not exactly matched, so this is workload growth, not an isolated regression ratio. |
| Materials 503,676,540 bytes | Same reported material bytes in the current fixture. Texture-pack size alone does not explain the difference. |
| 43 layers; 12,768 shadow casters / 20,013 instances | Current 53 layers; 17,169 casters / 36,060 instances. Approximately 80% more reported shadow instances. |
| Preparation 8.5 seconds, upload 652,444,068 bytes | Current cold preparation roughly 18–21 seconds, upload 914,356,054 bytes. Definitions and scene coverage differ; do not call this a precisely matched 2× slowdown. |
| One resident build, five full static draws, many reflection reuses | This is the behavior to preserve for valid static/camera cases. Continuous geometry zoom necessarily needs different pixel work. |
| Direct scene output, without Civ III integration obligations | Current live output adds authoritative capture, IPC, fog, native operation replay, canonical formats and existing-surface handoff. |

The preserved `matched-integrated-core.log` also stayed near 16/31 ms when the fresh core was initially adapted. Integration is not inherently doomed to be slow. Later workload, geometry zoom and native retained composition need separate attribution. The old sandbox water shader also had visual differences from the then-production water material; its entire output is not automatically a current quality substitute.

## 0 A.D. source review and C3X applicability

The review used the actual installed source at the pinned revision above, rather than relying on the earlier review document alone. It covered terrain render data/batching, model preparation and grouping, main/reflection/shadow submissions, water bounds, skinning, unit visibility, buffer allocation, texture lifetime, frame lifecycle and final output. Source findings are distinct from proposed C3X adaptations. 0 A.D. was not benchmarked on this VM, so its architecture is evidence for mechanisms, not a numerical promise.

### Persistent terrain and local invalidation

`TerrainRenderer::Submit()` attaches `CPatchRData` once and invokes its update before adding it to a visible pass list. `CPatchRData::Update()` rebuilds geometry only when update flags are nonzero. `CTerrain::MakeDirty(i0,j0,i1,j1,flags)` marks the affected patch range. Camera motion changes submissions, not terrain ownership. A dirty patch still rebuilds several internal products together; the code explicitly leaves finer within-patch updates as a TODO.

C3X should use the same separation between persistent world content and current view selection. It need not copy the square terrain grid: Civ III's authoritative tile basis, relief bounds, wrapping and imported generic packs determine its chunk layout. [Terrain submission](https://gitea.wildfiregames.com/0ad/0ad/src/commit/0ed48b3a1fb1b4b718a78869fa497185af55e086/source/renderer/TerrainRenderer.cpp#L182), [patch updates and batching allocator](https://gitea.wildfiregames.com/0ad/0ad/src/commit/0ed48b3a1fb1b4b718a78869fa497185af55e086/source/renderer/PatchRData.cpp#L826), [local dirty ranges](https://gitea.wildfiregames.com/0ad/0ad/src/commit/0ed48b3a1fb1b4b718a78869fa497185af55e086/source/graphics/Terrain.cpp#L756).

### Pass-specific visibility before drawing

`SceneRenderer::EnumerateSceneObjects()` submits the main frustum, each shadow caster frustum and water-scissor-constrained reflection/refraction frusta separately. Reflection rendering checks empty water bounds and constrains rasterization with a scissor. This saves CPU selection and vertex work as well as pixels. C3X has pass lists and scissors already; the improvement is making their selection tighter and spatial rather than repeatedly scanning broad resident occurrence lists. Preserve conservative reflection/distortion and shadow caster extents. [Pass enumeration](https://gitea.wildfiregames.com/0ad/0ad/src/commit/0ed48b3a1fb1b4b718a78869fa497185af55e086/source/renderer/SceneRenderer.cpp#L1152), [reflection bounds](https://gitea.wildfiregames.com/0ad/0ad/src/commit/0ed48b3a1fb1b4b718a78869fa497185af55e086/source/renderer/SceneRenderer.cpp#L631).

### Shared definitions and compatible material batches

`ModelRenderer::Render()` groups compatible opaque submissions by technique, model definition, texture and uniforms. Transparent models preserve distance order, batching only where that order allows. Terrain builds compatible effect/texture/buffer batches. The patch batching code uses a short-lived arena to limit allocation churn.

C3X should prebuild compatible opaque/cutout batches and reuse frame storage. It must preserve transparent water/overlay ordering and native read-before-write operations. “Sort every command by shader” would break native semantics. Also, 0 A.D.'s `InstancingModelRenderer` at this revision shares mesh definitions but calls `DrawIndexedInRange` per model; its name is not proof of automatic hardware instancing. C3X vegetation already uses `DrawIndexedInstanced`. [Model grouping](https://gitea.wildfiregames.com/0ad/0ad/src/commit/0ed48b3a1fb1b4b718a78869fa497185af55e086/source/renderer/ModelRenderer.cpp#L298), [actual unskinned draw](https://gitea.wildfiregames.com/0ad/0ad/src/commit/0ed48b3a1fb1b4b718a78869fa497185af55e086/source/renderer/InstancingModelRenderer.cpp#L264).

### Update an animated object once, consume it in several passes

The scene renderer deduplicates dirty skinned-model updates across cull groups, then prepares/uploads the unique set. GPU compute skinning uses resident source/palette/output buffers and explicit barriers; it can amortize skinning across later draws. C3X can first share pose/light/shadow preparation between main and reflection while retaining its existing vertex skinning. Compute skinning becomes justified only if dense-unit measurements show repeated vertex skinning is a major cost. [Unique preparation](https://gitea.wildfiregames.com/0ad/0ad/src/commit/0ed48b3a1fb1b4b718a78869fa497185af55e086/source/renderer/SceneRenderer.cpp#L189), [GPU skinning implementation](https://gitea.wildfiregames.com/0ad/0ad/src/commit/0ed48b3a1fb1b4b718a78869fa497185af55e086/source/renderer/GPUSkinnedModelRenderer.cpp#L367).

### Shadow work follows receivers and relevant casters

0 A.D. constrains its shadow setup to camera/receiver/caster bounds and supports cascades. C3X already caches compatible shadow inputs and groups caster instances. Improve conservative receiver/caster selection and share prepared data; do not port a perspective multi-cascade system wholesale into an orthographic, pixel-anchored Civ III viewport. [Shadow frame setup](https://gitea.wildfiregames.com/0ad/0ad/src/commit/0ed48b3a1fb1b4b718a78869fa497185af55e086/source/renderer/ShadowMap.cpp#L251).

### Frame storage is temporary; assets outlive the frame

`EndFrame()` clears submission lists while retained render data survives. Buffer handles release allocations through the vertex-buffer manager. Texture caching at this revision still has an explicit TODO for expiring unused textures; it is not a complete bounded-cache policy to copy. C3X should retain immutable asset data, clear/reuse frame selection storage and account explicitly for world, composition and queue budgets. [End of frame](https://gitea.wildfiregames.com/0ad/0ad/src/commit/0ed48b3a1fb1b4b718a78869fa497185af55e086/source/renderer/SceneRenderer.cpp#L999), [buffer handles](https://gitea.wildfiregames.com/0ad/0ad/src/commit/0ed48b3a1fb1b4b718a78869fa497185af55e086/source/renderer/VertexBufferManager.cpp#L102), [texture cache](https://gitea.wildfiregames.com/0ad/0ad/src/commit/0ed48b3a1fb1b4b718a78869fa497185af55e086/source/graphics/TextureManager.cpp#L908).

### Useful limits of the comparison

The unit renderer still scans units and notes a spatial-index TODO; interpolation also contains an offscreen-animation TODO with audio implications. These are not solved capabilities to attribute to 0 A.D. C3X must preserve native actions and authoritative visibility even when animation sampling is culled. [Unit submission and interpolation](https://gitea.wildfiregames.com/0ad/0ad/src/commit/0ed48b3a1fb1b4b718a78869fa497185af55e086/source/simulation2/components/CCmpUnitRenderer.cpp#L353).

0 A.D. owns its engine frame, postprocessing and output, then renders overlays/GUI. Its screenshot readback is an explicit operation. C3X must continue inserting into the existing Civ III boundary, retaining native UI/fog/selection authority; a new competing HWND or presenter is not an acceptable transfer of its architecture. [Engine output and screenshots](https://gitea.wildfiregames.com/0ad/0ad/src/commit/0ed48b3a1fb1b4b718a78869fa497185af55e086/source/renderer/Renderer.cpp#L650).

## How to verify improvements without sacrificing graphics

Use the existing category dispatcher, fixtures and bounded game scenarios. Keep one exact full-quality control: viewport, scene, pack/shader hashes, native tile width, current zoom, camera route, time, unit density, scene samples and presentation mode. Run matched arms serially; use short repeats to separate a real improvement from VM tails. Effects-off arms may locate costs but do not qualify a result.

The measurements needed to close the current gaps are:

| Question | Required evidence |
| --- | --- |
| Does a resident camera change rebuild unchanged world data? | Per-transition content/occurrence revisions, mesh builds, upload bytes, selected records/instances; include first transition frame. |
| Where is dense zoom GPU time spent? | Validated nonblocking GPU timestamps when available, or bounded causal pass probes plus normal-run wall times. Unknown is not zero; serialized completion probes remain separate. |
| Why is sparse live idle around 54 submissions/s? | Frame-opportunity timestamps, transaction denial, presentation-signal readiness, command-service occupancy, composition work and collector-on/off controls. |
| Is the displayed camera current and coherent? | Input → capture → prepare → ordered adopt → first correct map/overlays/picking; distinguish old-front animation from new view. |
| Are native composition copies necessary? | Per-frame node visits/reexecutions, dirty pixels, full-surface copies, target allocations and ownership/fallback pixel tests. |
| Does the working set remain safe? | Peak world/image/graph/queue memory, retirement after many cameras, x86 free/largest region, representative world size and density. |

The target is near 60 unique coherent frames per second, with frame work around **16.7 ms** across idle, continuous zoom, manual scroll and programmatic jumps. Report p95, worst and budget misses, not just average FPS. Preserve the existing camera-to-display goal below 33 ms p95 as a separately measured latency target; faster old-camera water does not satisfy it. Keep cold load and genuinely cold destinations separate from prepared revisits, and report their stalls honestly.

Correctness verification must retain current scene-capture, ownership, invalidation, wrapping, scrolling, native compositing, animation, visibility and config-off delegation tests. Add focused checks only for new semantics: dirty world versus view revision, spatial-bound conservatism, ordered batched commands or on-demand retained display copies. Use existing render comparisons for thin shorelines, reflected vegetation, mountain/tree overlap, depth ties, water normals, unit pose continuity, fog edges and fixed-size text/UI. Visual parity is part of qualification, and reference replacement still requires explicit user acceptance.

Long sustained gameplay qualification remains pending in the existing renderer notes; an earlier cancelled stress run is not a pass or proof that its prior freeze persists after fixes. The present short runs establish useful baselines and completed bounded operations, not final performance qualification.

## Storage and audit side effects

The audit removed approximately **57 MiB** of its generated build/image files, plus disposable saves and duplicate guest receipts. It retained about **3.6 MiB** of local evidence, including six selected live images. Audit-only binaries, object files, linker outputs, launch wrappers and both guest capture directories were removed after useful hashes and receipts were saved; no audit game or scheduled task remained. No source art, generic packs, ignored import inputs, prior expensive findings or pre-existing diagnostics were deleted. The staged triad remained unchanged, no gameplay was saved, and the generated tracked scene export was restored.

No deferred wonder/District ownership, new lighting aesthetic, source-specific runtime format or patch-table entry is proposed. The first work should strengthen the existing isolated renderer and integration boundary while maintaining graphics quality.
