## Objective and scope

The active execution order is maintained in
[the benchmark workflow](benchmark_workflow.md), with current status in
[the retained-renderer plan](retained_renderer_plan.md) and operating rules in
[the autonomous execution contract](autonomous_renderer_execution.md). This
document supplies architecture and acceptance criteria; its earlier staged
experiment ordering is not a command to restart completed work.

Make scrolling, zooming and arbitrary map jumps feel immediate while preserving
current visual quality and Civ III's simulation, camera, visibility, interaction
and presentation ownership. Retain reusable scene content, make prepared views
cheap to draw, prepare missing regions efficiently, and remove rendering waits
from ordinary game-thread redraws.

This is an implementation handoff requested by the user. No implementation,
benchmark, staging or game launch was performed while writing this plan. The
steps below organize this task only; they do not restore the retired milestone
or release workflow. Current code and category workflows remain authoritative.

Limit changes to currently owned renderer systems. Natural wonders, constructed
wonders and Districts remain deferred to their documented M9/M10/M11 contracts.
Preserve the existing separate map and unit planes. Do not add a presenter,
overlay window, gameplay simulation, source-specific runtime or new visual style.

## Read first and inspect the current checkout

- Project `AGENTS.md`, [Renderer README](../README.md), [Lab README](../lab/README.md)
  and [category catalog](../lab/catalog.json). Read the affected category recipes.
- [Workstreams](renderer_workstreams.md), [visual fidelity playbook](visual_fidelity_playbook.md),
  [environment contract](environment_lighting_and_ambient_effects.md),
  [shared shadows](shared_shadow_contract.md).
- [Current continuation](navigation_continuation.md), [custom zoom](custom_rendering_zoom.md),
  [performance receipts](zoom_performance.md),
  [render-loop boundaries](civ3_render_loop_viability.md),
  [frame pacing](runtime_animation_and_frame_pacing.md),
  [patch ledger](civ3_patch_dependency_ledger.md).

Inspect Git status before editing and preserve concurrent work. Some narrative
documents describe earlier implementations: verify claims against current code.
In particular, do not implement natural world-space GPU sharing or a camera
queue from scratch; both already have implementations worth extending.

Relevant starting points, with paths relative to the repository root:

| Area | Current location |
| --- | --- |
| GPU ownership, geometry assembly, draw submission, worker, publication and blit | `Renderer/native/c3x_renderer.cpp`: `CachedTileGeometry`, `draw_cached_geometry`, `submit_geometry`, `PublishedMapFrame`, `RendererWorker`, `MapBlitter` |
| Terrain compilation and common queries | `Renderer/native/source_fidelity/geometry.h`, `Renderer/lab/shared/natural/`, `Renderer/native/render_core/` |
| Existing camera ABI | `Renderer/native/c3x_renderer_api.h`, `Renderer/native/c3x_renderer.def`: camera begin/poll/cancel exports |
| Native capture/compositing | `injected_code.c`: `composite_custom_renderer_frame`, map draw patches; renderer state in `C3X.h` |
| Benchmarks | `Renderer/native/biq_preview.cpp`, `BENCHMARK_ZOOM.bat`, `compare_zoom_benchmark.py`, `analyze_resident_navigation.py` |
| Executable contracts | `Renderer/native/test_zoom_mesh_cache.py`, `test_frame_publication.py`, `test_scroll_damage.py`, `test_custom_zoom.py`, `test_native_bridge_contract.py`, `test_unit_bridge.py` |

The completed resident/cold comparison recorded in `zoom_performance.md` is
motivation, not a fresh baseline: at 2240x1192, fourteen new resident views built
zero tiles and uploaded zero geometry bytes, yet took 235.560 ms median and
745.483 ms maximum. Fresh geometry for the same views took 8,527.645 ms median.
The experiment used a 768 MiB GPU geometry tier and disabled waves. It excluded
initial definition loading and did not measure native input-to-display latency.
Small retained/cold pixel differences remain unexplained. Its 120.354 ms median
readback interval includes pending GPU work; it is not measured transfer cost.

## Success criteria

Use 2240x1192 as the primary viewport, supported zoom levels 128/160/192, and
smaller diagnostic sizes. Include the widest visible working set, dense relief/forest/
cities, water, current animated resources, waves and native-directed units.
Record map dimensions, assets, build flags, hardware/VM and memory tier.

These are engineering targets, not achieved results or promises:

| Workload | Target |
| --- | --- |
| Continuous camera movement over prepared content | 30 delivered correct frames/second initially; p95 frame intervals at most 33.4 ms, p99 at most 50 ms; 60 FPS is a later stretch goal |
| Warm pan or zoom input | First camera-correct visual response within 50 ms p95; final current-quality result within 100 ms p95 |
| Arbitrary previously unvisited or evicted destination after initial map preparation | Complete current-quality result within 100 ms p95; measure worst cases separately |
| Intermediate cold-region development checkpoint | Useful current-camera image within 100 ms and final quality within 500 ms p95; explicitly insufficient for the final instant-jump objective |
| Ordinary game-thread renderer submission/poll bookkeeping | At most 2 ms p95; no waiting for geometry, GPU work or disk reads |
| Correctness and memory | No stale view/visibility publication, unbounded queues, memory leaks, cache-budget frame failures, or silent visual degradation |

Measure input-to-visible-result inside Civ III separately from standalone request
completion. Include capture, native redraw scheduling and final compositing;
do not report worker completion as presentation. If display timing cannot be
observed directly, label the endpoint and its limitation precisely.

Use at least 100 actual changes for discrete-workload distributions and at least
1,000 presented frames for continuous-workload p99 measurements. Report medians,
p95, p99, maxima, missed intervals, cancellations and time to the final requested
view after input stops. One-time startup/map preparation is a separate reported
cost. Test both uninterrupted movement and bursts with reversals and jumps.

## Architecture to converge toward

```text
Civ III game/UI thread
  authoritative tile/object capture + camera/visibility changes
               |
  compact retained scene state + immutable request identity
               |
  replaceable pending request -> preparation/render worker
               |                   retained meshes/materials/chunks
               |                   bounded uploads and rendering
               |                   GPU completion/readback
               v
  completed image + coverage + identity + ownership, published together
               |
  ordinary Civ III redraw requested on completion
               |
  existing map-compositing boundary -> native overlays/unit plane/UI
```

Separate semantic world identity, prepared-resource identity, and presentation
identity. Canonical wrapped tiles own content; visible occurrences own anchors.
Camera movement selects/reprojects resources without recompiling unchanged
content. Civ III's supplied basis and anchors remain authoritative.

The persistent world is a compact description plus bounded detailed resources.
It does not require every detailed mesh or view bitmap to remain resident.
Do not assume retained topology contains complete appearance or visibility data.

## Step 1: Establish trustworthy timing, memory and integration evidence

1. Reproduce the resident sweep and zoom/navigation witnesses on current code in
   isolated output directories. Freeze and hash the DLL, fixture, packs, shaders,
   environment clock, flags and benchmark executable for matched comparisons.
   Keep startup-cold, geometry-cold, disk-cache-warm and fully resident cases distinct.
2. Add phase counters for capture/diff, snapshot copying, dependency checks,
   geometry construction, asset reads/decompression, uploads, culling/batching,
   CPU submission, GPU passes, readback, CPU conversion and native blit.
   Use delayed D3D11 timestamp/disjoint queries to distinguish GPU rendering from
   transfer and synchronization; profiling must not force a wait every frame.
3. Extend the resident witness beyond its fourteen fixed views. Prove real new
   cameras over resident content with zero static-geometry builds/uploads.
   Track pixel-cache reuse separately. Resolve the existing exact-parity failure
   or isolate and explain its cause before using the comparison as a passing gate.
4. Inventory all retained and peak allocations. Include CPU capacities, temporary
   compilation data, snapshots, map topology, publications, decoded textures,
   GDI surfaces, GPU buffers, shadows, reflections and driver-related observations.
   Sample free virtual address space and largest free region around allocations
   and resets, not just after interactions. Track deferred GPU destruction too.
5. Audit the native async presentation contract early: how a completion requests
   a full enough redraw, how native overlays/picking remain aligned when no new
   image is ready, and which existing hook capabilities support the path. This
   is a source audit first; do not guess executable addresses or install hooks.

Deliverable: reproducible phase/memory reports and a concrete next bottleneck.
Maintain a short current-results section in `zoom_performance.md`; use Git and
disposable receipts for history. Continue independent work if one witness is
blocked, while retaining that witness as an unmet requirement.

## Step 2: Retain scene content and GPU resources across views

1. Introduce or extract compact canonical scene records and dependency revisions
   from existing state. Keep tile callbacks as the authoritative initial capture
   and validation mechanism. Add updates through already available hooks only
   where useful; do not require a complete new event-hook system up front.
2. Separate camera/clip/target changes from semantic edits. Audit cache keys for
   unnecessary global invalidation, including world revision use. Replace broad
   keys with local dependencies only when executable edit tests prove safety.
   Keep conservative validation when complete change notification is unavailable.
3. Extend existing natural world-space GPU ownership to remaining currently owned
   static components that still rebuild for projection changes. Keep camera
   transforms and wrapped occurrence offsets in small parameter/instance buffers.
   Separate moving objects and poses from static terrain retention.
4. Organize spatial chunks and material batches using measured sizes. Tiles remain
   invalidation identities; chunk size must balance draw count, edit amplification,
   culling and memory. Share compatible repeated meshes through instances while
   preserving source normals, cutouts, transforms and deformation requirements.
5. Cache conservative bounds for geometry, shadow casters and reflection
   contributors. Cross-boundary overhangs and both wrap axes must remain correct.

Gate: a resident pan/zoom sweep has zero unchanged static mesh construction or
geometry uploads; small parameter updates are reported separately. Local edits
rebuild only proven dependent resources. Inspect bytes per visible tile/object,
draw counts and preparation time rather than assuming chunking is a speedup.

## Step 3: Make the prepared scene cheap to draw

1. Profile the current 128-pixel block recursion, receiver collection, material
   submission, shadow page work and reflection recursion. Existing request-scoped
   caster-list reuse is already present; do not count it as a new optimization.
2. Share scene preparation and valid static world-shadow data across regions and
   frames. Shadow validity includes caster/material/alpha state, light state,
   coverage and device generation. Newly needed pages still require rendering.
   Preserve current unit/local-shadow ownership and filter quality.
3. Compare the existing blocks, larger bounded regions and full-viewport passes
   under identical quality and content. A scene-wide pass schedule may still use
   bounded scratch targets. Cache reflection inputs where valid; camera-dependent
   reflected images, water animation and changed lighting cannot be reused blindly.
4. Batch compatible opaque/cutout work, preserve transparent ordering and the
   existing scene-linear resolve, glow, antialiasing and final color transfer.
   Preserve useful scroll/dirty-region reuse when it actually wins. Static views
   should remain idle; continuous full-screen drawing is not required.
5. Prototype a bounded staging/readback ring only after distinguishing GPU cost
   from transfer. Poll readiness without blocking normal UI work. A ring may
   overlap work but must not accumulate old-camera frames or conceal latency.

Gate: first reach under 100 ms for genuine prepared new views, then the 33.4 ms
continuous target. Preserve quality and measure CPU and GPU improvements separately.
If transfer plus native composition alone exceeds the frame budget, identify
that constraint before expanding the renderer redesign. Graphics interoperability
is a separate investigation, not an assumed capability of this plan.

## Step 4: Bound memory and make missing regions fast

1. Establish one explicit budget policy covering resource categories and peaks.
   Existing normal ceilings are 96 MiB CPU natural data, 32 MiB viewport cache,
   192 MiB GPU tile geometry and 32 MiB resource backdrops. These are separate
   categories, not total process usage. Existing high-memory experiments use
   different ceilings; always identify the tier, and do not pass by silently
   growing it. Reallocate from redundant caches as retention improves.
2. Share immutable resources across requests. Keep one active and one replaceable
   pending camera snapshot. Bound front/back publications and transient replacement
   copies. The current publication cap is 32 MiB per owner; audit every concurrent
   owner and scratch allocation rather than assuming total use is just two frames.
3. Choose the production envelope using measured game use and allocation spikes.
   A provisional live-test reserve is at least 512 MiB free process address space
   or twice the measured largest transient spike, whichever is greater. Also verify
   a sufficiently large contiguous region for the largest planned allocation.
   Revisit this proposed reserve using real maps; it is not a Windows guarantee.
   Account GPU residency separately from CPU virtual address space.
4. Reduce cold preparation at the source: shared mesh instances, reusable indexed
   terrain templates, exact dependency-scoped samples, retained spatial queries,
   fewer temporary copies and precompiled generic asset data. Preserve the current
   geometry and material result. Do not substitute lower detail to claim a pass.
5. Preload common assets during explicit initialization. Prepare map regions in
   bounded background work using captured immutable inputs, prioritizing the current
   view and surrounding region. A distant foreground request supersedes prefetch.
   Keep completed valid chunks when their requesting camera is cancelled.
6. For arbitrary jumps, evaluate compact whole-map preparation and versioned
   disk-backed compiled regions. Neighbor prefetch alone cannot cover random travel.
   Key derived data by compiler/pack versions and all semantic dependencies; reject
   corrupt/stale data and retain a bounded rebuild path. Keep derived local assets
   local. Capture complete region inputs safely on the game thread before workers
   consume them; unknown appearance or visibility is not permission to draw.
7. Exercise forced eviction and pressure below the preferred tier. Evict unpinned
   optional data first. The active view must fit without relying on allocation
   failure or device reset. If it cannot, reduce representation/scratch cost or
   document an unmet supported envelope; preserve custom-on failure semantics.

Gate: warm, unvisited and evicted destinations satisfy separately reported latency
and memory targets. Include terrain edits to disk-cached regions, revisits after
edits, save/load, map changes, viewer changes and repeated reset/recovery cycles.
Report map-preparation time and disk footprint; moving seconds of work to startup
is a tradeoff to measure, not a free speedup.

## Step 5: Complete asynchronous rendering in standalone/replay

1. Extend the existing camera begin/poll/cancel API and worker. Requests reference
   immutable renderer-owned data, never mutable Civ III objects. Keep D3D context
   ownership serialized and game/UI surface access on the game thread.
2. Publish image, coverage, camera transform, visible occurrences, ownership,
   viewer/visibility epoch, semantic/environment revisions and device generation
   atomically. Ownership arrays must match the publication's captured identities
   and ordering, not a newer callback array that happens to have the same length.
3. Replace pending cameras immediately. Check cancellation between bounded work
   units and before further GPU submission. Already submitted GPU work cannot be
   recalled: retire its resources safely and reject obsolete completions. Ensure
   continuous input does not starve all useful completions or final refinement.
4. Fix interaction with the existing synchronous unit path: unit takeover currently
   cancels camera work. Define scheduling/resumption so active units do not starve
   the requested map. Retain Civ III's unit action director and dynamic plane.
5. Cover publication allocation failure, poll/blit lifetime, reset, shutdown,
   reconfiguration and visibility changes during work. The front image remains
   immutable until the UI copy is finished. Latest-request correctness includes
   lifecycle changes, not just an increasing camera ticket.

Gate: race/lifetime tests plus mixed navigation, zoom and active-unit replays show
bounded memory, prompt submission, eventual latest-view completion, no queue
growth and no stale or hidden content. Repeated screenshots and first-preview
times cannot substitute for completed new-camera rendering.

## Step 6: Integrate completion with Civ III's native drawing

1. Bind the optional camera exports through a versioned/validated bridge while
   retaining the known synchronous compatibility path for older DLLs and controlled
   evaluation. Keep renderer implementation in `Renderer/`; change only `C3X.h`
   and `injected_code.c` for injected integration.
2. Make tile callbacks capture/validate content and placement. Submit a request
   only when scene, camera or time requires it. At the existing map boundary,
   copy a completed compatible publication; never wait there for rendering.
   Polling-triggered redraws must not resubmit identical work indefinitely.
3. Implement the pending-view presentation policy before enabling this path.
   Prefer a useful renderer-produced current-camera view assembled from retained
   resources. A reprojected older bitmap is provisional and valid only for covered,
   visibility-safe pixels. The existing flat terrain preview is unaccepted and
   lacks detailed ownership; it is not an automatic solution. Never silently
   expose native terrain to fill missing content in custom-on mode.
4. Prove one coherent displayed transform across terrain, native unit/HUD planes,
   labels, highlights and inverse mouse picking. If retaining an older complete
   display temporarily, native overlays and interaction must retain that same
   display identity too. Do not claim nonblocking integration until the no-ready-
   image case works; attaching an ID alone does not coordinate native drawing.
5. Completion must request ordinary native redraw work even when no tile changes.
   Use a coalesced, lifetime-safe notification and a proven UI-thread invalidation
   path. The worker never renders into native surfaces or calls native game logic.
   No reentrant map rendering, manual message pumping or busy polling.
6. Preserve the existing approximately 66 ms native timer and simulation cadence.
   To reach 30/60 FPS, investigate a safe higher-frequency presentation invalidation
   path; do not assume the timer can deliver it. Preserve input guards, modal/minimized
   behavior and pause-filtered animation time. Completion notifications alone also
   need a schedule for visible continuous animation when the camera is stationary.
7. Verify hook capabilities in the CSV and ledger. Existing starting symbols include
   map m71/m19, the key handler, tile-to-screen coordinates, unit-body boundaries and
   the timer. `Animator_update` is callable, not entry-patchable. Preserve the separate
   outstanding city-HUD capability request. Record concrete new requirements with
   exact symbol, capability, signature, supported-build addresses, reason, fallback
   and `required_user_action`; never edit the CSV or invent addresses.

Gate: replay/injected tests establish camera/visibility/ownership correctness,
normal native redraw completion, config-off behavior and absence of render waits.
A subsequent live checkpoint must measure the actual supported game build; a
standalone worker benchmark is insufficient evidence for this step.

## Step 7: Verify the complete experience and hand off for game evaluation

Run dependency-selected category tests throughout, using
`python3 Renderer/renderer.py test CATEGORY` and category integration. Shared
lighting/shadow/transition changes select their affected consumers. Use focused
standalone diagnostics early and the full integration sweep at the final strategic
checkpoint. Preserve executable tests for capture, edits, ownership, partial clips,
scroll/wrap, zoom/picking, compositing, animation, reset and config-off behavior.

Final workloads must include:

- Hundreds of distinct pans, all five zooms, both wrap seams and duplicate visible
  occurrences; repeat after forced cache eviction and at the widest view.
- Random distant destinations without convenient prefetch warmup, plus separate
  fresh-map and fully prepared-map runs; dense and sparse map regions.
- Concurrent unit movement/actions, resource/wave animation and camera reversals;
  no unit-driven map starvation or stale selection/labels.
- Local terrain/city/resource edits, visibility loss/reveal and viewer changes,
  hour/season changes, save/load, config-off, reset and allocation pressure.
- Sustained idle and interactive runs that establish bounded memory and no leak,
  peak overlap of active/pending/publication work, and correct recovery.

Compare cached and independently recomputed results at identical clocks and
inputs. Exact cache-path parity and native ownership are correctness checks.
Fixed reference differences are visual review evidence, not automatic integration
failures. Any intentional material visual change needs comparison and user
acceptance; never loosen tests or replace references to conceal a regression.

Native builds/replay run on the Windows VM through the existing dispatcher/shared
checkout mechanism. When injected files change, run
`TEST_INJECTED_CODE_COMPILE.bat` through `C3X_Shared_Verify`. Documentation-only
and standalone-tool work do not require injected compilation. Do not hard-code
machine paths or drive letters; use the existing configurable environment values.

Follow current Lab staging/visual-acceptance rules. Candidate builds and this plan
do not authorize staging, `INSTALL.bat` or launching Civ III. Exhaust automated
and valid existing evidence before requesting one batched strategic live checkpoint.
If the user is not ready, record that checkpoint as pending and continue independent
work. Do not repeatedly request screenshots or claim live performance without evidence.

Deliver a short report of achieved versus pending targets, exact tested artifact
identity, supported workload/memory envelope, remaining risks, and any concrete
patch-table action. Update affected current contracts to match the final code.
Remove superseded experimental paths only after replacement behavior is covered;
preserve source findings and necessary ignored asset inputs.

## First assignment for the implementing agent

Start with Step 1 and the native presentation feasibility audit. Produce current
phase and memory evidence, then implement the highest-impact prepared-view change
from Steps 2-3. Continue through the remaining steps without treating an early
cache hit, queue implementation or coarse preview as completion. The finished
outcome is measured responsive navigation in Civ III within a stated memory
envelope, with current quality and ownership preserved.
