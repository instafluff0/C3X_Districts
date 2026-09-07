# Integration Cache And Renderer Worker Evidence

## Current maintenance: bounded idle preparation (2026-09-06)

Production API 12 extends the fixed-scroll cache with optional mesh preparation
between foreground frames. The complete appearance of a tile is still keyed by
world content and consumed dependencies; scrolling changes its native anchor.
No Lab v2 asset, shader or promotion is consumed. The global checkpoint stays
LQ0 and this performance goal remains **in progress**.

### Capture, scheduling and resource ownership

- The existing native projection callable must agree with every captured logical
  anchor before extending the capture. A twelve-coordinate topology halo contains
  a four-coordinate inner ring of complete appearance snapshots, including
  cities, improvements and resources. API flag `PREFETCH` accompanies
  `TOPOLOGY_HALO`, never `RENDER`. These records neither claim native ownership
  nor change visible mesh density. Unsupported projections, ambiguous small
  wrapped maps, capture bounds and allocation failures retain the original path.
- Capture remains on Civ III's thread. One existing D3D worker owns a private,
  immutable snapshot for idle work, separate from the next foreground job buffer.
  It prepares one tile at a time, checks for foreground requests during ground
  construction and vertex packing, and yields for 2 ms between jobs. Direction
  biases the order of the nearest ring; at most 384 candidates belong to the
  latest snapshot. There is no speculative backlog. Unit/UI redraws with unchanged
  render content do not restart the queue. Incomplete work is never presented.
- Unused prefetched meshes have a 64 MiB sub-budget within the existing 192 MiB
  total GPU-buffer / 2,048-entry bounds. Foreground use adopts an entry; older
  speculative entries can be evicted without churning the current ring. Optional
  allocation/capacity failure stops preparation and reports unavailable work.
  Active geometry remains pinned. The 32-view / 128 MiB bitmap budget is unchanged;
  source assets, targets and bounded CPU metadata are additional working memory.
- Mesh-only preparation never writes the published pixel vector, ownership or
  fallback arrays, current-frame geometry references, bitmap LRU, or GDI surfaces.
  GPU-buffer ownership is exception-safe; reset cancels and joins the worker.
  Foreground calls serialize with reset and native blits. A short trace mutex
  serializes UI/worker diagnostics.
- Scene topology is built once per immutable snapshot and reused across its tile
  jobs. Both signature and record-buffer identity guard the cached tile pointers.
  River graph construction uses a coordinate index. River mesh keys and distance
  sampling use only conservatively local nodes, sorted by raw coordinate. The
  window covers the 24-pixel maximum approved shader response plus a full tile
  diagonal and safety pixel, preserving interpolation as well as point samples.
  Existing wrapped-coordinate behavior remains unchanged.

API counters expose current pending/unavailable preparation, cumulative built
and interrupted tiles/time, and unused-prefetch bytes. Timestamped traces retain
invalidation reasons, phase timings, worker waits, memory, injected capture and
whole-map/blit times. Cumulative preparation time is excluded from per-frame
latency percentiles by the analyzer.

### Executable evidence

The six-view fixed-jump fixture renders 400 tiles with 400 companion records and
1,008 halo records. It moves four native coordinates per step (256 horizontal or
128 vertical pixels), includes rivers and a city with walls, a mine and resource,
and returns through exact cached views. Every prepared jump must build/upload
zero meshes and reuse all 400. The harness waits for each neighborhood to finish;
these measurements do **not** represent continuous scrolling without idle time.

While preparation is pending, the harness changes unit direction, repeatedly
checks the original published pixel/ownership pointers, and compares a real GDI
blit byte-for-byte. It requires no unavailable work, enforces both memory bounds,
and limits foreground p95 to 16.7 ms. A city-size mutation must rebuild and change
pixels with city ownership preserved. Reset during active preparation must join
safely. Portable C++ tests compare full-world and locally culled river-distance
interpolation at tile widths 64, 128 and 256, as well as disjoint scroll damage,
removal, mutation, overhangs and ambiguous-translation rejection.

The final full Windows run passed
(`Renderer/verification/prewarm_full.json`, ignored; timings retained here):

| Workload | Measured time |
|---|---:|
| Historical-scale cold, 400 tiles / 800 records | 1351.197 ms |
| Unit selection, 64 samples | p95 1.083 ms |
| Unprepared fixed jumps, 5 samples | p50 271.402 / p95 741.861 ms |
| Exact cached return, 6 samples | p95 3.138 ms |
| Prepared fixed jumps, 5 samples | 33.011–57.594 ms; p95 57.594 ms |
| Foreground redraw during preparation, 471 samples | p95 2.238 / max 5.065 ms |

All five prepared jumps reused 400 meshes with zero construction or upload. The
initial ring prepared 272 meshes; the sequence built 623 optional meshes with 51
interruptions. Peak unused-prefetch storage was 66,459,288 bytes, below 64 MiB.
The final accumulated unprepared jump/cold witness had 178 pixels with a channel
error above 2/255 and aggregate channel error 6,525, within unchanged parity
bounds. The full workflow passed portable/source gates, native and licensed
replays, both zoom previews, `TEST_INJECTED_CODE_COMPILE.bat`, and live-link checks.
The candidate DLL is built into the linked checkout; this is not evidence that
`INSTALL.bat` was run or that the running game's executable uses API 12.

### Remaining work and limits

Prepared scrolling still spends tens of milliseconds drawing and reading back
newly exposed strips. Transfers now copy only disjoint damaged rectangles from
GPU to staging. Prepared jumps spend 4.907–5.373 ms validating/assembling geometry,
3.115–5.001 ms submitting draws and 22.415–44.840 ms in readback (including pending
GPU execution). Partial transfer alone did not materially reduce latency in this
VM run. Further cached-pixel preparation must preserve depth,
overhangs, authoritative dependencies and published pixel lifetimes.

Rapid consecutive jumps, large recentering and environment changes may outrun
idle work and remain synchronous cold cases. Actual injected capture/whole-map
latency and native projection-guard hit rate remain unmeasured. Earlier VM runs
intermittently exceeded incremental/cold parity bounds, then passed on retry;
the cause is unproven. Failure-only raw BGRA witnesses are now retained under
`native/build/` for investigation. Keep the existing strict thresholds; do not
attribute failures to driver noise without evidence. No CSV entry or user action
is required for this increment.

## Prior verified increment: fixed tile jumps (2026-09-06)

The primary scroll workload now changes complete rows/columns in Civ III's
staggered coordinate grid. It uses 400 rendered tiles, 400 companion calls,
and an eight-coordinate authoritative topology halo. Four-coordinate jumps
translate anchors by 256 pixels horizontally or 128 vertically. These are
representative several-tile jumps requested by the user, not a claim that every
native scroll action has that one displacement. The decompiled existing
`Main_Screen_Form::FUN_004de2a0` chooses quarter/half/full tile dimensions
according to the native scroll-speed setting; keyboard and edge-scroll callers
use those fixed increments. Unit-driven camera recentering may jump farther.
Small-pixel timings below are secondary regression measurements only.

Changes in this increment:

- Track the compiled mesh version and actual bounds behind the most recently
  rendered bitmap. Added, removed and changed footprints mark disjoint damage
  rectangles; unchanged overlap is moved in place. The planner tolerates changed
  capture order and tile sets, rejects inconsistent/duplicate occurrences, and
  never double-blends overlapping damage. An LRU image without matching footprint
  metadata cannot donate pixels. Environment, zoom, wrap and reset guards remain.
- Snapshot an eight-coordinate native topology halo before compositing. The
  existing `Main_Screen_Form_tile_to_screen_coords` callable supplies its anchors;
  every captured logical tile must agree with the same offset first. Other map
  forms, ambiguous small wrapped maps, bounds and allocation failures keep the
  original capture. This is a bounded, lightweight terrain/river/route snapshot;
  it performs no off-thread game reads and adds no renderable halo geometry.
- Neighbor dependencies now hash only consumed topology fields. City/resource
  and unit metadata cannot invalidate unrelated neighboring terrain. Mesh density
  excludes halo-only records, preserving the pre-existing zoom/density choices.
- Cache repeatedly queried integer neighborhoods in small per-tile scratch
  arrays. Skip hill texture sampling when support is zero. Stop shadow rays above
  the proven production height bound (104 relief + 3 blend + 18.6 dune < 128),
  or when their final occlusion is saturated. Neither cutoff lowers shadow quality.
- Use local pixel depth with a power-of-two range and explicit subpixel alignment
  to reduce camera-dependent depth/edge roundoff. Bitmap translation requires
  even X/Y offsets to retain shader derivative-quad phase; odd movements reuse
  meshes and rasterize the image. Native fixed steps satisfy this condition.

The Windows replay verifies six distinct views and their reverse return trip,
plus cold reconstruction of the final shifted image. Exact bitmap returns are
byte-identical. Incremental/cold comparisons classify <=2/255 channel changes as
quantization noise, require larger changes on less than 0.1% of pixels, and cap
aggregate channel error to 0.04 per pixel. These are bounded raster differences,
not a claim of byte-identical shading after translation. A portable executable
also verifies damage coverage, disjoint rectangles, overhang/removal/mutation,
all four directions, diagonal motion, and rejected ambiguous translations.

Latest full Windows run (`Renderer/verification/tile_jump_full.json`, ignored):

| Workload | Measured time |
|---|---:|
| Historical-scale cold, 400 tiles / 800 records | 1455.690 ms |
| Single changed-boundary case | 49.283 ms |
| Unit selection, 64 samples | p95 2.095 ms |
| First exposure across several tiles, 5 samples | p50 231.004 / p95 365.848 ms |
| Reverse return through cached views, 6 samples | p95 2.617 ms |

A horizontal new strip builds exactly 80 tiles, reuses 320, and redraws
344,488 of 1,336,232 pixels (about 26%). A vertical strip builds 40, reuses
360, and redraws about 14%. The final accumulated jump/cold comparison has
176 pixels with a channel difference >2 and aggregate channel error 6,190.
Full verification passed the source-independent suite, completed portable gates,
Windows native and licensed production replays, both BIQ previews, the approved
injected compile and live-link check. The dedicated portable damage executable
also passes. VM measurements can vary; first exposure is explicitly not within
the interaction budget. Existing production art and all Lab handoffs remain
unchanged. The runtime/native shader adapter changed only caching, sampling work
elimination, and numerically stable coordinate/depth evaluation.

At this prior checkpoint, newly exposed meshes still built
synchronously. The planned follow-up was bounded idle prewarming from an immutable,
authoritative capture, with no stale presentation, no speculative backlog and
no mutation of a bitmap while Civ III blits it. The lightweight halo is topology
only; object-bearing prewarming needs a complete authoritative appearance
snapshot. River surfaces still depend on all captured river nodes and need a
separate locality audit. Actual injected capture/whole-map latency and native
projection-guard hit rate remain unmeasured; replay latency is not game-wide parity.
No new Lab promotion or CSV entry is required.

## Prior verified increment: indexed tiles and bitmap strips (2026-09-06)

The user resumed production performance work independently of Lab v2. The
global LQ0 checkpoint and subsequent system integration gates do not advance.

The current implementation replaces the earlier expanded viewport/region and
96 MiB world-sample caches described below. It stores immutable indexed meshes
per tile, with exact vertex equality preserving material and normal seams.
Only neighbor coordinates actually consulted during mesh generation enter the
dependency list; missing neighbors are dependencies too. Content, environment,
zoom, target size, world wrapping, device generation, and geometry density
remain invalidators. River surfaces retain the approved global river-node
distance dependency. River rocks also validate relative authoritative anchors.
Multiple neighborhood versions remain available for revisits.

All GPU mesh buffers, including the active frame's shared references, count
against 192 MiB and 2,048 tile entries. No expanded viewport-sized CPU mesh or
duplicate compiled GPU region remains. Scratch geometry is per tile/layer.
The 32-view bitmap LRU remains bounded to 128 MiB and now counts capture and
ownership metadata along with pixels. Source assets, D3D targets, and bounded
CPU metadata are additional working memory, not included in the GPU-buffer cap.

Meshes use local pixel coordinates; current Civ III anchors are applied at
draw time, with depth clamped after translation. Tiny explicit depth separation
preserves underlay/land/bed/water order and removes coplanar depth flicker.
The production adapter now initializes the previously missing tundra weight.
No Lab v2 shader or pack refresh is consumed.

For an exactly matching uniformly translated scene, existing pixels move in
place and only the disjoint exposed strips are rasterized and copied back.
Chunk bounds cull draws outside those strips. There is no second full-size
CPU scratch bitmap. Environment/content/ownership/wrap/resize changes reject
this fast path. Static unit selectors and partial UI clips keep the exact
bitmap without geometry, uploads, or raster work.

The single existing worker continues to own D3D and mutable renderer caches.
Its immutable tile snapshot now reuses vector capacity, without a temporary
duplicate. The caller waits for its exact sequence; the renderer has no
speculative queue or stale-frame presentation. This is multithreaded ownership,
not a claim that synchronous cold-frame costs are hidden from the game.

API 11 exposes per-frame mesh builds/reuse/evictions, GPU bytes, upload bytes,
geometry/draw-submission/readback timings, and reused/drawn pixel counts.
The native smoke now checks incremental-versus-cold output, four scroll
directions, 64 unit-selection redraws, memory bounds, and latency percentiles.
Cold-versus-shifted raster comparison separately bounds one-channel rounding
from GPU derivative-quad changes, larger edge differences, and total error;
exact bitmap cache hits must remain byte-identical to their primed snapshot.

The latest expanded Windows integration run passed native fixtures, both BIQ
previews, `TEST_INJECTED_CODE_COMPILE.bat`, and the live shared-checkout check:

| 400 tiles / 800 records | Measured time |
|---|---:|
| Cold render | 6455.141 ms |
| Single small scroll | 7.280 ms |
| Boundary change | 412.239 ms |
| Unit selection, 64 samples | p50 0.951 / p95 1.082 / max 1.107 ms |
| Small scrolling, 32 samples | p50 7.007 / p95 8.913 / max 9.530 ms |

Boundary work built 20 meshes and reused 380, uploaded 4,865,760 bytes, and
retained 100,258,020 GPU buffer bytes. The earlier boundary measurement below
was 1948.219 ms. Cold cost has increased from that historical 5314.920 ms run;
these runs are not a controlled same-build hardware comparison. Cold build and
first-exposure latency remain open work. The goal of no noticeable production
slowdown is **not complete**, and these replay timings do not prove total
in-game frame latency.

The VM's desktop drive letters changed: `Y:` no longer identifies the checkout.
For this session the verified override is
`C3X_RENDERER_WINDOWS_ROOT=C:\Mac\Home\fun\Civilization III Complete\Conquests\C3X_Districts`.
That directory is the VM's existing link to the shared home folder. The
`C3X_Shared_Verify` directory symlink was retargeted from stale `Y:` to the same
checkout via UNC. The workflow now accepts UNC roots using `pushd` as well.

### Diagnostics

Set `C3X_RENDERER_TRACE=1` before launching Civ III for invalidation summaries
and sampled activity, or `2` for every frame/phase. Default `0` avoids DLL trace
formatting and I/O. Messages contain monotonic QPC timestamps, milliseconds,
thread, sequence, cache decision, reason flags, memory/work counters, and phase
times. Readback time includes pending GPU execution, not just the memory copy.
Injected capture/composite/slow-map summaries use the same QPC clock and report
capture, synchronous render wait, blit, and whole-map cost. No tile loop logs.

An optional `C3X_RENDERER_TRACE_FILE` path writes a buffered trace, capped at
8 MiB per process. Choose a dedicated diagnostic file: it is overwritten on
DLL load. Debugger output remains available after the file cap. The native
smoke enables a bounded `Renderer/native/build/cache-trace.log` automatically.

Summarize a captured log with:

```text
python3 Renderer/tools/analyze_renderer_trace.py path/to/trace.log
```

The analyzer separates internal render times from worker wait and injected
capture/blit times; it does not double-count composite counters or treat
historical maxima as individual frame timings. Summary-mode percentiles
describe only the logged samples.

### Historical follow-up at the prior increment

Investigate a stable authoritative topology halo and bounded prewarming for
newly exposed terrain. The current live capture has no explicit topology-halo
records, so removing a captured tile can correctly invalidate neighboring
samples; the boundary fixture intentionally exercises this. An existing
`Main_Screen_Form_tile_to_screen_coords` callable is available for auditing
native halo anchors; this is not a new patch request. Also extend bitmap reuse
to changed visible sets using dependency-validated damage regions, and measure
actual injected capture/whole-map latency before claiming game-wide parity.
Keep worker ownership, native overlays, no stale presentation, and hard memory
bounds through that follow-up. The full regression workflow passed for this
increment: 358 source-independent unit tests, completed portable gates, native
smoke, expanded production replay, both BIQ previews, injected compilation, and
the live-checkout check. Reports are ignored machine outputs at
`Renderer/verification/tile_cache_iteration.json` and `tile_cache_full.json`.

## Historical sample-cache increment

This maintenance increment preserves the approved L9-L19 production rendering
contract while removing screen anchors from reusable world-tile work.

Automated source contracts in `Renderer/native/test_native_bridge_contract.py`
prove:

- a 32-entry exact viewport LRU bounded to 128 MiB;
- canonical world-coordinate semantic records whose content keys exclude screen
  anchors and retained Civ III overlays;
- anchor-independent surface, relief, normal, and environment-specific shadow
  samples bounded to a 96 MiB LRU;
- two bounded compiled GPU regions, with 192 MiB total and a 96 MiB per-entry
  ceiling so a full expanded live viewport is never duplicated in the 32-bit
  process;
- a deep-copied frame/tile payload consumed by one renderer worker, exact
  sequence completion, worker-owned D3D/reset, UI-thread final blit, and worker
  shutdown before `FreeLibrary`.

`Renderer/native/native_smoke.cpp` proves exact recent-viewport hits, bounded
eviction, uniform anchor translation, a small compiled-region revisit, reset
equivalence, both game zooms, clipping, horizontal wrapping, retained overlays,
zero fallback, and a changed visible tile set that cannot hit the exact viewport
or whole-viewport geometry cache.

The 2026-09-06 Windows integration replay reported:

- 400 rendered tiles / 800 captured records cold: 5314.920 ms;
- uniform three-pixel/two-pixel camera translation: 26.318 ms;
- one logical tile plus companion crossing the visible boundary: 1948.219 ms;
- zero fallback and deterministic approved-scene completion.

The tile-boundary result is a measured intermediate improvement. It still
rebuilds and uploads expanded viewport geometry; future indexed/chunked GPU
buffers and staging-readback work remain valid optimizations. No stale custom
frame and no native terrain fallback is permitted while those are pending.

No new `civ_prog_objects.csv` symbol is required. The only injected lifecycle
change calls the existing renderer reset export before unloading the DLL.
