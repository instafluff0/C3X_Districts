# Renderer64 performance review — October 4, 2026

Goal: near-60 FPS on the Windows VM for idle, zoom, map jumps, scrolling and
ordinary play on busy late-game maps, without lowering visual quality. This
note records the major problems found, the evidence, the industry-standard
remedy and what was changed. It supersedes the "where time goes" table in
[the October 3 overhaul](performance_overhaul_20261003.md); that note's
switches and contracts still apply.

## Method

Earlier live numbers were taken with `-ProfileRenderer` (trace level 2, a
DebugView collector and per-record `OutputDebugString`), which distorts both
FPS and the game-to-renderer transport. The baseline below is production-like:
`run_scripted_game_test.ps1 -Scenario zoom-out -MeasureCadence` (trace 0) on
the busy `input-1498AD.SAV`. Presentation rates are successful helper
presentations per second (`cadence.json`), not physical scanout. Per-phase
costs come from the level-2 trace of the same scenario
(`.cache/navigation-quality-20261004/final-busy-waves`).

| Busy 1498 AD save, trace 0 | Presentations/s |
| --- | --- |
| 1× settled | 18 |
| 0.5× settled | 11 |
| 3× settled | 51 |
| First map ready after load | 89 s |
| Helper private memory | 7.3 GB |

Settled per-frame scene cost (level-2 trace, `fresh-scene-phases`):

| Zoom | Scene total | CPU unit pose prep | Water pass (draws) | Units | Reflection |
| --- | --- | --- | --- | --- | --- |
| 3× | 5.1 ms | 0.7 ms | 1.8 ms (399) | 1.6 ms | 0.1 ms |
| 1× | 28.7 ms | 10.4 ms | 9.8 ms (2,645) | 5.4 ms | 2.3 ms |
| 0.5× | 44.9 ms | 14.5 ms | 19.2 ms (6,249) | 6.1 ms | 4.3 ms |

Retained composition then replays the native HUD over the new map each frame
(~3,860 recorded operations, ~270 copies, ~25 M copied pixels), adding several
milliseconds before `Present`.

## Results so far

Same scenario with `-ProfileRenderer` (level-2 tracing lowers absolute rates,
so compare rows with each other). Run-to-run noise is about ±3/s per segment.
Each cell is the mean presentations per second, with the worst single second
in parentheses. Captures are kept in `.cache/perf-review-20261004/`.

| Busy 1498 AD save | Before (run20) | After (run38) |
| --- | --- | --- |
| Zoom-out notches 1× → 0.5× | 12 (2) | 12–15 (5–6) |
| 0.5× edge scroll | 12 (1) | 24 (11–16) |
| Zoom back in | 14 (6) | 24–27 (12–16) |
| 1× idle (33 s tail, median) | 34 | 33–40 |
| Static work per 0.5× scroll step | 250–670 ms ×2 frames | 25–35 ms |
| Static work per zoom notch | 230–270 ms | 25–35 ms |
| Camera job per 0.5× scroll step | ~1.3 s | 0.35–0.85 s |

On the light saves (the 4000 BC settler save and the 3700 BC 60×60 navigation
witness), every zoom runs at 46–49 presentations/s, and map jumps and edge
scrolling at 35–50/s. Camera jobs there take a median of 74 ms (p90 126 ms).
That ~48/s ceiling, with only ~6 ms of renderer CPU per frame, is the fixed
cost of the VM's presentation path (see finding 5).

## Findings and remedies

### 1. Game facts convoyed behind every frame (critical)

**Evidence.** Each copied game fact (unit observation, animation cursor, state,
move, spawn) was one synchronous IPC to the helper, and the helper applied it
under the renderer's `call_mutex`. The ambient frame holds that gate for its
whole render (30–70 ms on this map). With ~700 facts/s, the game-side transport
thread was ~90% busy, a ~600-record backlog persisted, and every fact — including
unit moves and selection — reached the renderer **~850 ms late**. Because the
backlog stayed above 512 records, the helper's queue-pressure guard limited
ambient presentation to one frame per 250 ms (~90 holds/s observed).

**Practice.** Game-thread state ingestion must never wait on rendering. Engines
buffer simulation facts in a short-lock inbox (double-buffered or command-queue
game state) and apply them at the start of the next frame.

**Change.** `RendererWorker` now ingests unit facts through a copied inbox when
the call gate is busy. Every gate holder (`submit_locked`, camera begin, the next
uncontended fact) drains it in arrival order before its own work, so causal
order relative to cameras, image batches and frames is unchanged. The worker
drains it opportunistically with a bounded 1 ms retry, never blocking on the
gate. Late rejections are counted instead of faulting the transport.
Per-observation `fresh-unit-accepted` tracing moved to level 2.

### 2. Every unit's ground height recomputed every frame

**Evidence.** `unit_pose_ms` (10 ms at 1×, 15 ms at 0.5×) is dominated by
`unit_low_ground`: per unit per frame it built a fresh surface query, searched
the coast index, evaluated shore noise and borrowed the shared 16-page river
corridor cache — which thrashes once the view spans more than 16 river pages at
wide zoom. Frozen idle units paid the full cost.

**Practice.** Cache pure functions of static world state. Ground height under a
unit depends only on its world position and the terrain revision.

**Change.** `SandboxDirectUnits::low_ground` keeps a world-anchored height cache
(1/1024-tile quantization, cleared on topology/content/field changes). Per-part
key vectors reuse scratch storage and `FrameSampleCache` compares a 64-bit hash
before whole keys.

### 3. Civ III's UI thread spent a third of its time on a CPU hit-test model

**Evidence.** On the game thread, `C3X_NATIVE_IMAGE_DRAW` calls took a median
13.8 ms, about 10 times per second. Instrumentation showed the canvas diff and
upload were cheap (~0.5 ms per 2 s); the cost was the CPU input-coverage model
(`native_hit_scene.h`) that mirrors every native UI command for form hit
testing: ~5,500 commands/s and **~730 ms of every 2 s** of the UI thread, with
single fullscreen transfers splitting into 665 regional nodes (up to 33 ms).
Hit queries themselves are rare (pointer-driven). A stalled message pump delays
input, edge scrolling and Civ III's own unit animation ticks.

**Practice.** Keep the main/UI thread free of bookkeeping that is only needed
on demand: move it to a worker with an ordering fence, or evaluate lazily.

**Change.** `WorkerClient` now feeds an ordered `HitWorker` thread that applies
the identical model; a hit query waits for every earlier command before
sampling. Large transfers are also recorded as one deferred node whose regional
history materializes only where a later draw or query reaches. The canvas
mirror is updated in place for changed rows only, and a rate-limited
`native-source-refresh` summary measures the remaining upload cost.

**Correction found in live testing.** The worker's queue was unbounded. Civ
III's end-of-load UI burst queued seconds of model work, and the first hit
query (the hover action that accompanies a wheel zoom) waited for all of it:
the game thread went silent for ~7 s after the map appeared, and later zooms
took 1–3 s instead of ~0.15 s. The queue now applies backpressure: past 256
operations or 64 MiB of uploads the producer waits until it halves, so a
query never waits behind more than a few milliseconds of work and a burst
costs what inline processing did.

### 3b. One synchronous IPC per game fact

**Evidence.** Even without lock contention, each unit fact was its own
cross-process round trip, ~0.5 ms in the VM. Once the UI thread was freed it
produced facts faster, and the transport saturated again (median 3,749 queued
records, up to 6 s latency); the image queue's capacity wait then blocked the
game thread inside native presents (median 12.5 ms).

**Practice.** Command batching: coalesce small messages into one packet.

**Change.** `AsyncSceneClient` joins consecutive unit facts into one ordered
batch at the queue tail (so ordering against cameras/images/presents is
unchanged), encodes them on the transport thread at send time, and the helper
applies them in order (`Kind::unit` subtype 3), returning per-fact codes that
keep each fact's original strict/superseded acceptance.

### 4. Water submitted as one draw per tile chunk

**Evidence.** Water-dependent records are one `DrawIndexed` per tile-layer chunk,
each in its own per-tile buffer, so nothing merges. Cost is ~3 µs per draw at
every zoom, and draw count scales with visible tiles (399 → 2,645 → 6,249).

**Practice.** Draw-call batching: pack geometry into shared pages with
per-record placement data and issue one draw per contiguous run.

**Change.** `render_core/pulled_mesh_pages.h` keeps GPU-side copies of each
chunk's vertices and indices in raw buffers plus one 64-byte record per
occurrence (translation, natural projection, projection kind), keyed by content
identity and placement — both camera independent, so scrolling reuses pages
and only appends edges; heavy fragmentation repacks in current order. A
generated `VSIntegratedPulled` (built at runtime from the selected hydrology
source) binary-searches its record by `SV_VertexID`, loads the original vertex
bytes and runs the unchanged `VSIntegrated` body with per-record terms. Primitive
order, pixel shaders and per-occurrence constants are unchanged; fog-frozen
water splits runs on its water sample. Applies to ground, bed, water, river,
shadow and route records (168-byte integrated vertices) and, through a second
generated `VSIntegratedFeaturePulled`, to feature, mine, farm, site, wall and
cliff records (48-byte packed feature vertices). Cities, rigid sources,
animated/resource records, reflections, any refusal and
`C3X_RENDERER_PULLED_SUBMISSION=0` keep the per-record path.

**Correction found in live testing.** The first pulled build assumed
`SV_VertexID` includes `Draw`'s `StartVertexLocation`. It does not: a probe
(`Draw(4,10)` writing `SV_VertexID`) returns 0–3 on both the Parallels adapter
and WARP. Every range that did not start at the beginning of a page therefore
resolved the wrong record and underflowed its local index into out-of-range raw
loads. On screen this showed as black ocean with seabed diamonds at wide zoom,
and it coincided with a VM-wide device removal (`0x887a0005`) after ~60 s.
D3D11 bounds-checks raw buffer reads, but a translation layer over Metal need
not, so the same bug can fault the host GPU. The shader now draws
`Draw(count,0)` with an explicit base in its constant buffer and clamps every
index and vertex load to its record; no inconsistency can read outside a page.

### 4b. Overlay records starved of batching at wide zoom

**Evidence.** With water batched, the 1× per-frame water pass was mostly
overlays near rivers and coasts that must be redrawn over animated water: 550
routes, 816 mines and 640 farms per frame, one draw each (they use either the
excluded route layer or the packed feature layout). Rigid overlays use ordered
packets, which duplicate each occurrence's mesh in draw order. Their owner was
capped at 64 MiB and refuses, without eviction, once full; one busy 1× view
already needs ~64 MiB, so at 0.5× ~1,560 records per frame were refused into
single draws. This is a direct cause of slow 0.5× scrolling.

**Practice.** Size retained GPU caches for the working set of the widest
supported view, not a nominal default; a capacity refusal must not silently
change the submission algorithm.

**Change.** The ordered packet owner is now 512 MiB / 65,536 entries (still
charged against the world-geometry reserve), and non-rigid routes and
packed-feature overlays use the pulled path above.

### 4c. Static overlays redrawn over animated water every frame

**Evidence.** Even with batching, the per-frame water pass at 1× resubmitted
~1,800 overlay records (~1.2 M triangles) that are static but water-dependent:
roads, mines, farms, features, cliffs and their shadows on tiles touching a
river or coast. On a river-heavy late-game map that is most of the land.
Water-pass CPU stayed at 12–25 ms (1×–0.5×) and the GPU re-rasterized all of
it each frame.

**Practice.** Cache static content as a retained layer and re-render only
what animates (layer/impostor caching). Premultiplied "over" is associative,
so drawing static overlays once into a cleared layer and compositing that
layer over live water is mathematically identical to drawing them after the
water every frame. The only cross-term is depth, which is static here: water
vertices do not animate and its only clip uses static hydrology data.

**Change.** Each static raster slot owns a parallel overlay layer
(`OverlaySlot`). Whenever `write_slot` renders a static strip it also renders
that strip's water-dependent records into the layer, tested against the
strip's static depth and the water depth (drawn with color writes disabled).
`reset_slot`, `recenter` (scroll past the guard band) and `repair_front`
(local world edits) clear, move or repair the layer in exactly the same
pixels, and the layer is usable only while its revision equals its slot's.
Static dependency proofs now include water-dependent records, so a new road
by a river repairs or invalidates the slot like any other content edit. Each
frame draws only water and rivers live, composites the layer at the static
restore's pixel offset and depth shift (`OverlayComposite`, one full-screen
draw), then the waves. Previews, bootstraps, MSAA and any animated overlay
record fall back to the previous per-frame path;
`C3X_RENDERER_OVERLAY_CACHE=0` disables it. Live frames match the per-frame
path visually (river crossings, bridges, improvements, coasts at 1× and 0.5×).

Trace-0 busy save, settled presents/s: 1× 18 → 23 (batching) → 31 (overlay
layer), with 37–45/s idle right after load; 0.875× 12 → 31.

**Correction found in live testing.** Bootstrap images (jump/zoom previews)
reuse the strip writer with their own state and index 0. The first version
therefore retired slot 0's overlay layer on every bootstrap, and 1× stayed on
the per-frame path after any zoom-out. Only real static slots now touch
overlay layers (`test_overlay_slot_ownership.py`).

### 4d. Scrolling below 1× fell back to full progressive redraws

**Evidence.** Scrolling at 0.5× spent ~36 ms per moving frame in static
raster work, with previews, bootstraps and refinement restarts but zero
recenter copies. Passing a retained raster's guard band should copy the
overlapping pixels into the lane's other slot; `recenter` refused any
sub-pixel shift, and at a ladder zoom k/8 almost every camera step is
fractional (an odd step at 0.5× is half a raster pixel). Each refusal
discarded the raster and re-rendered the whole view progressively behind a
blurred preview.

**Practice.** Keep retained rasters on a fixed lattice and absorb the
remainder in the display transform (the static display path already does
this for every retained slot).

**Change.** `recenter` anchors the new slot at the nearest camera step whose
raster shift is whole (lattice 2 world pixels at 0.5×, 4 at 0.75×, 8 at
0.625×/0.875×). The copy, its overlay layer and the remaining display shift
stay exact. Animating (non-ladder) zooms still refuse.
`test_static_recenter_lattice.py` executes the production function.

### 4e. Refinement budget could not shrink on slow frames

**Evidence.** The per-frame static refinement budget adapted to the frame
interval but ignored any frame over 100 ms, so heavy zoom-out refinement on a
busy map kept its large budget and held frames at 2–10/s for seconds.

**Practice.** Time-slice background work against a measured cost target.

**Change.** The budget now follows the static work measured in the previous
frame toward an 8 ms target (`C3X_RENDERER_STATIC_BUDGET_MS`), with a lower
floor. (A 4 ms target kept frames cheap but left previews on screen for many
seconds after each zoom step; the hidden 1× lane still refines at a quarter.)

### 4f. Level of detail below 1×

With the user's direction that detail may drop below 1×, units contribute no
reflections below 0.8× (no reflection preparation, draws or per-frame mirror
copy) and render no self-shadow maps there (512² per changed pose, ~23 per
frame on the busy save). A unit there is a few dozen pixels and its mirror
image a few pixels; ground shadows and the static mirror of terrain, cities
and features are unchanged, and 0.875× keeps full unit detail. Previously the
contribution plan disabled culling below 1×, so every visible unit was
prepared and drawn three times (body, ground shadow, reflection).

### 4g. Frames blocked on the GPU while holding the renderer gate

**Evidence.** With CPU submission reduced, ambient frames spent a median
19 ms inside the final display draw: the translated DXGI latency permit
grants a frame before the GPU has drained, so the CPU blocked there while
holding the renderer gate. Every image batch, fact batch and zoom command
waited behind it; zoom input latency rose to 1–7 s once the game thread
published faster (finding 3).

**Practice.** CPU/GPU fences: never block on the GPU inside a critical
section; keep a bounded number of frames in flight.

**Change.** A 1×1 staging copy follows each delivered ambient frame. The
next ambient frame starts only when the copy from two frames back has
completed (non-blocking map); otherwise it returns BUSY and the gate stays
free for transport work. Startup's required presents are not gated.

### 4h. Every camera step discarded every retained raster (critical: scroll)

**Evidence.** Edge scrolling at 0.5× advanced one 180-px camera step per
1–2 s. Each step ran a 1× camera job and then a 0.5× frame, and both found
their retained slot invalid, so they bootstrapped and refined the whole view:
~0.4 s for the camera service, ~0.83 s for the 1× job frame and ~0.93 s for the
next 0.5× frame, with almost no frames displayed in between. Added
`static-compose` entry/repair diagnostics and validation counters traced a
chain of five independent causes:

1. *Visibility-mask churn.* The native draw's `visibility_mask` argument is
   0, 1 or 15 depending on whether a tile was captured on screen, in the
   halo or in a world page. Nothing reads it, but each change bumped the
   tile's visibility revision (~2,000 tiles per step) and its world input,
   which also queued world regions for re-preparation.
2. *Instance-placement overflow.* At 0.5× on the busy save, body and shadow
   placements exceeded the 32 MiB joint cap. The retry retired every optional
   shadow page, forcing a 175-page, ~12k-draw atlas rebuild (~400 ms) on each
   step.
3. *Residency mistaken for removal.* Membership validation required every
   recorded contributor to be re-observed. Contributors whose tiles had left
   the resident set looked like removals.
4. *Capture-order churn.* Membership records followed the capture index, and
   the capture lists the view before its halo. A camera step moved tiles
   between the two lists, reversing the recorded order of unchanged
   contributors; any reversal rejected the whole raster.
5. *All-or-nothing repair.* More than 24 dirty rectangles collapsed into one
   bounding box (always above the 45% limit), and a successful repair then
   re-registered the entire slot (150–300 ms).

**Practice.** Cache invalidation must follow the world, not the camera. Use
canonical, camera-independent ordering and repair proportional to the
changed area (as tile renderers do).

**Changes.**
- `ScenePublication` normalizes the mask, and `CapturedScene` ignores it for
  revisions.
- The joint placement budget is 256 MiB.
- Membership ignores contributors that left residency (proofs and visibility
  are still exact), and records are ordered by screen anchor, which equals
  Civ III's on-screen draw order.
- Repair marks reordered draws instead of rejecting, merges dirty areas on a
  128-px cell grid, and forgets and re-registers only the repaired regions.
- Static rasters no longer register per-input watches. Each new proof
  expanded ~3,000 keys (per-tile river inputs), so registering a view cost
  150–650 ms. Proofs and visibility are still checked exactly; an unrelated
  world change now costs one complete revalidation (~5 ms at 0.5×) instead of
  nothing.

Tests: `test_static_dependency_reuse`, `test_raster_reorder_repair`,
`test_static_refinement_consistency`.

**Result (busy save, 0.5× edge scroll).** Static work per camera step fell
from 250–670 ms on both the 1× job frame and the 0.5× frame to 25–35 ms
(a local repair of 6–13% of the slot). A recenter fell from ~270 ms to ~34 ms.
Remaining step latency is the camera job's world preparation and the mirror
redraw.

### 4i. Each zoom-out notch drew the whole view synchronously

**Evidence.** Every wheel notch below 1× stalled ~230–270 ms in the
bootstrap. Timing showed the draw itself was 16–60 ms; the rest was exact
dependency registration for an image displayed for only a few frames.

**Changes.**
- Bootstraps carry a coarse validity stamp: any world, residency or asset
  change redraws them.
- A zoom-out bootstrap draws only the ring around the raster the preview
  already shows: the previous zoom, or the current 1× raster when the lane
  is still refining. Its hole is proven to lie behind that raster for every
  intermediate zoom (`test_zoom_ring_bootstrap`).
- A settled zoomed-out view may be previewed from a current raster at up to
  twice its resolution (minified, never magnified) while the destination
  refines, as map tiles are (`test_static_scene_transition`).

**Result.** Static work per notch fell from 230–270 ms to 25–35 ms; the
largest remaining bootstrap (a direct jump from 1× to 0.5×) costs ~80 ms.

### 4j. Remaining hitches, and experiments that did not help

Remaining costs, from the latest busy traces:
- **Camera-step shadow pages.** The shadow pages rebuild on 1× camera-job
  frames (`city_shadow_ms` 58–230 ms). Each caster draw re-issued its page
  constants, shaders, layout and buffers. Redundant state is now skipped (the
  constants change only for wrapped copies), but issuing hundreds of draws
  still dominates. Batching casters (vertex pulling with per-caster offsets in
  a buffer) is the structural fix.
- **Mirror redraws.** The mirror is redrawn on every zoom step and camera
  step (30–115 ms). A progressive rebuild was tried: show the shifted old
  mirror and draw the new one in four bands over several frames. One band
  cost about as much as the full mirror (~103 ms versus ~114 ms), so it was
  removed. Cheaper reflected submission (batching or LOD of reflected
  features below 1×) is the lever.
- **Preview-frame water.** In preview frames during zoom refinement, every
  water-dependent overlay is drawn live (35–67 ms), because the retained
  overlay layer is composited only for whole-pixel shifts, not the preview's
  affine resample.
- **World preparation on 0.5× camera jobs.** These prepare world regions
  for 350–850 ms. In this VM the geometry budget is capped by ~3 GB of free
  RAM, so the cache evicts on every step and rebuilds tiles. The repair path
  now absorbs those rebuilds locally.

### Visual defects found during this review (pre-existing)

The first two reproduce in the pre-change baseline capture at the same moments.

- **Fog missing below 1×.** The fog pass draws one feathered quad per captured
  tile and scales anchors about the view center for the current zoom, but
  `VisibilityCoverage::capture` culled anchors at the canonical 1× viewport.
  Zoomed out (and mid-zoom), the outer ring of explored-but-unseen and
  unexplored ocean stayed fully lit while a block of fogged tiles inside the
  1× area looked isolated. Coverage now spans the widest outward zoom
  (`test_visibility_zoom_extent.py`).
- **Dashed horizontal lines mid-zoom.** Zoom previews resample retained
  rasters and live water depth-tests against reconstructed preview depth.
  The depth slope read neighbors only clamped to the texture, so at a
  coverage edge it used cleared or stale texels; one row moved in front of the
  water, which failed its depth test and showed dashed seabed lines with
  tile-shaped teeth. Neighbor reads are now clamped to each source's drawn
  texels.
- **Every zoomed-out frame failed while city sites were shown.**
  `GpuCitySiteOverlay::draw` rejected zooms below 1×, a floor that predates
  the outward zoom levels. On a save with a settler (city-site suggestions)
  the map callback failed on every frame below 1×, and the zoom-out scenario
  stopped with "failed a zoom-out camera preparation". The pass already
  scales anchors about the view center, so it now accepts the full
  `SceneProjection` range. The busy save showed no sites, which hid the bug.
- **Wide fog coverage hardened.** The outer ring added for zoomed-out fog
  now skips a conflicting duplicate anchor instead of rejecting the whole
  capture (which fails the frame). The 1× viewport keeps its exact contract
  (`test_visibility_zoom_extent`).

### 5. HUD replayed in full over every animated frame

**Evidence.** When the map animates, the world-view node changes revision, so the
HUD batch re-executes with two full-screen base copies and every map fragment is
re-assembled.

**Practice.** Composite static UI as a retained premultiplied layer over the
animated scene; touch only damaged regions.

**Measurements.** `compose_prepare_ms` includes the fresh scene render (the
map sample's prepare callback calls `c3x_renderer64_render_fresh`). The HUD
replay itself (`evaluate`, `assemble`, `display`) costs ~3–5 ms of CPU per
frame. Its GPU side is fixed: ~320 copies and ~22 M copied pixels per frame,
even on the light save, where the whole renderer CPU frame is ~6 ms yet
presentation stays at ~48/s. Frame-fence denials are zero and the swap chain
already allows two queued frames, so that ceiling is GPU and presentation
work in the VM, largely these copies.

The biggest copies are the HUD node's two full-screen base planes, the owned
`select_world` assembly, and the front's map fragments. Two remedies are
mapped in the code:
- *Option A:* make the HUD batch node sparse, writing only its ink envelope.
- *Option B:* keep a display-only "above-map" layer and composite it with
  the map plane in the display pass.

Both must keep the HUD's command order and the map-dependent HUD pixels
(palette tints, keyed shadows), which cannot be cached as premultiplied
"over".

**Status.** Pending; it touches the Game Integration composition graph.

### 6. Moving units held still during every camera transaction

**Evidence.** Each camera job paused unit travel until ordered adoption. Edge
scrolling and Civ III's own recentering issue back-to-back camera jobs, so a
moving unit advanced only between adoptions — slow motion during scrolling and
a visible stop when the map follows a unit. On the busy map a camera
transaction itself took 0.65–1.0 s when it prepared new tiles.

**Practice.** Simulation/presentation time must not stall on render latency;
fix the latency, and at most hide a bounded amount of unseen motion.

**Change.** The hold now starts 125 ms after a camera job begins
(`UnitInstances` samples `min(now, hold)`), so ordinary scroll/recenter
transactions never slow travel, while an unusually long preparation still
cannot skip a large unseen distance.

### 7. Loading and first launch

**Evidence.** On the busy save the first map appears ~89 s after launch:
~38 s of asset loading before the world seed and ~39 s preparing world
geometry. After shader sources change, the shader cache recompiles at load —
`PSIntegrated` alone took 41 s with `D3DCOMPILE_OPTIMIZATION_LEVEL3`; that run's
first camera poll then failed (`first-map-ready result=0`), while the identical
binaries with a warm cache loaded normally.

**Practice.** Ship precompiled shader bytecode (compile at pack-preparation
time, never at first launch) and stream world preparation behind a playable
view.

**Status.** Not changed in this pass; recorded for a dedicated loading task.

### Environment notes

- GPU timestamp queries never complete under the Parallels D3D11 translation
  (every `fresh-gpu-phases` sample is invalid), and event queries report
  completion at submission. `C3X_RENDERER_PROFILE=1` enables a readback
  timeline (`gpu-event-timeline` records): each phase mark copies a 1×1
  texture to staging and collection maps them in order, which cannot complete
  early. It serializes CPU/GPU and is for profiling only.
- D3D11 `SV_VertexID` excludes `Draw`'s `StartVertexLocation` (verified on the
  Parallels adapter and WARP).
- Windows applied the Fault Tolerant Heap shim to `Civ3Conquests.exe` in this VM
  (logged at process start) after earlier development crashes. It slows heap
  operations in the game process, so game-thread costs here are pessimistic.
  Resetting it is a system setting for the owner of the VM.
- Profiled runs (`-ProfileRenderer`) attach DebugView and trace every
  publication; use `-MeasureCadence` alone for frame-rate comparisons.

