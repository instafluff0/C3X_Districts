# Renderer validation and measurement

This document defines evidence, not a tooling phase or execution queue. Use the
[roadmap](retained_renderer_plan.md) for current status and milestone selection.
Reuse existing harnesses and controls; add diagnostics only for a named missing
measurement/correctness question that can change the implementation decision.

## Measure the complete interaction

Keep separate: setup/build/load, capture, caller/queue waits, worker CPU work,
GPU execution when measurable, finishing/composition, publication and display.
CPU spans may contain waits and overlap GPU work; do not add overlapping totals.
Unknown is unmeasured, not zero. Staging Map duration is not pure GPU/copy time.
Desktop completion is not physical scanout. Worker completion, native caller wait
and actual fresh-frame cadence are different endpoints.

Use identical assets, quality, inputs, clocks, viewport and normal memory limits.
The normal workload always enables shoreline waves, water motion and reflections.
Effects-off runs are explicit diagnostic controls, never performance acceptance.
Earlier reflections-off receipts remain historical and do not establish this baseline.
Distinguish process-cold, assets-loaded/content-cold, prepared-resident and revisits.
Report preparation time, reset coverage, retained content and completed-image reuse
explicitly. Broad CPU/GPU scene preparation and final pixel prefetch have different
costs; account for both, useful work surviving cancellation and publication age.

Compare preserved control and candidate with equivalent state, serial GPU work and
matched/alternated runs. Close Civ III before standalone timing and verify that
its process and prior benchmark processes have exited. No benchmark may overlap
compilation, live play or another GPU test; exclude overlapped performance
results explicitly while retaining their correctness evidence.
Report sample counts and variation; close results are inconclusive. Do not infer
p99 from short runs. Detailed traces should be buffered outside timed work.
Before claiming acceptance, verify full source/binary/assets identities before and
after the run; quick metadata-only receipts remain provisional.

## Async recovery witness

`record_gpu_frame --native-recovery` adds cancellation, config-off camera handoff,
reset while pending, and reset after readiness but before fresh-capture validation
to the real JGL/GPU navigation fixture. Each retirement rejects stale COMMIT;
a fresh authoritative request recreates and commits, and reset preserves the
prior displayed pixels. Use this with `--atomic-camera-views` for the independent
cold-pixel/coverage matrix. Keep the normal throughput run separate so recovery
setup is not misreported as steady-state cost.

The executable hook/owner tests inject copy, allocation, poll and readback/display
handoff failures, plus viewer changes, retirement/address reuse and held-worker
pack/definition reload. Failed ownership barriers must block DLL release and
native drawing. These controlled failures do not emulate physical device removal
with irretrievable GPU-only native surfaces; that remains a terminal ownership
barrier, not a proven successful recovery. Live-game acceptance is separate.

## Meaningful checkpoints

| Checkpoint | Evidence |
| --- | --- |
| Concrete design | Existing owners, replacement path, expected eliminated work, ownership/visual contracts and unresolved attribution |
| Working representative path | Independent color/depth/ownership parity, selected inputs reaching real submissions, bounded lifetimes and local invalidation |
| Complete comparison | Stationary animation, dense scrolling and local content change; whole-request/display costs and work actually removed |
| Sustained/integration | Affected zoom, wrap, distant/evicted content, effects, independent unit actions, cancellation and UI/picking; live evidence at a strategic checkpoint |

Use tiny boundary/alias/depth fixtures when those contracts change. Revisit equality
proves repeatability, not correctness; compare with an independent full redraw or
established native oracle. Preserve exact output for unchanged-appearance changes.
New appearance differences follow Lab approval; full detail remains the policy.
Existing accepted precision/text differences retain their documented limits.

Expand runs when an implemented capability, failure or affected contract warrants
it, not after every edit. Use category-selected tests and existing replay/native
fixtures. A visual-frame timer test is not live-game acceptance. Batch manual
checkpoint requests and reuse evidence; do not repeatedly ask for screenshots.

Track combined resident/transient use, allocation failures, evictions, queued work,
total free VA and sampled largest contiguous VA. Preserve at least 512 MiB sampled
contiguous headroom in the 32-bit process and disclose transient sampling limits.
Capacity increases require working-set evidence, not benchmark-only allowances.

## Arbitrary-destination navigation acceptance (M3.8/M4)

Extend the existing navigation and native GPU receipts. Cover each real camera
trigger: scrolling, minimap, zoom/reframing, newly selected unit centering, action
following and other native programmatic moves. Minimap and automatic centering
currently use the exact native centering path rather than the deferred idle-pan
fast path; both require direct evidence. Use real input/native captures
for trigger-to-display acceptance; replay/JGL fixtures can establish rendering and
ownership facts but must label excluded game work. Current historical distant
CPU render/capture timings cannot stand in for the native GPU path.

Use fixed, recorded seeds and 100 distributed legal destinations per supported
map/viewport/quality case, with at least 50 distinct destinations where possible.
Include far first visits, dense cities/infrastructure, coasts/rivers, map edges
and wraps, mixed visibility, all supported zooms and repeated revisits. Also test
rapid supersession and native selection/action centering. On smaller worlds,
report distinct destination count rather than padding it with duplicates. Repeat
matched runs for percentile/tail claims; 100 samples do not establish a robust p99.

Run and report these populations separately:

| Initial state | What it establishes |
| --- | --- |
| Normal initialization, then first visits | Actual user experience, without pre-visiting the test destinations; report remaining preparation coverage |
| Whole-world preparation complete, dependencies GPU resident | Camera-only construction invariants and view/render/publication cost |
| CPU prepared, GPU content deliberately evicted | Honest residency/upload cost; distinguish re-upload from recompilation |
| Authoritative local/remote edit, reveal/viewer change or reload | Correct invalidation scope, fresh capture, cancellation and rebuild cost |

Whole-world preparation policy must be independent of the measured route and
must not render/cache every destination screenshot. Record time to first usable
view, time to complete declared world preparation, background CPU/GPU consumption,
coverage by authority/compiled/resident state and combined resource peaks. If the
normal readiness claim covers only part of the world, report that limitation;
selecting random points only from the warm subset cannot pass the anywhere goal.
Test the normal supported map-size/asset-density envelope and memory pressure,
not just a small fully resident fixture. Keep water effects and full detail on.

For each input/native-camera-event and request identity record these endpoints
and counters:

- Input receipt or native camera decision, authoritative capture/diff duration
  and submission completion. Automatic selection/action moves start at the native
  decision to move the camera; do not count intentional pre-move gameplay delay
  as renderer latency or start the clock after expensive capture has finished.
- World compilation by category, bytes constructed and dependency changes.
- World GPU allocations, uploads and eviction/re-upload bytes, separately from
  ordinary camera/instance/animation parameter updates and render-target reuse.
- Foreground preparation joins, lock/queue waits, readbacks and CPU waits for GPU
  completion; separate concurrent worker spans from the actual critical path.
- View/pass selection, draw submission, finishing, native composition, ready time,
  ready-to-adopt/message-pump delay and first correct coherent displayed frame.
- Total/free/contiguous VA, GPU/native/history/queue storage and transient peaks;
  record sampling limits and retain the 512 MiB contiguous-headroom floor.

Counters missing from current instrumentation are unmeasured, not zero. Normal
production timing remains separate from costly profiling. `--profile` currently
includes repeated address-space walks and resident-buffer enumeration inside
requests; its latency is diagnostic, not ordinary production performance. Preserve source/binary,
assets, clocks, memory limits and quality in comparisons. On this VM, rejected or
unstable GPU timestamps cannot establish shader cost: use validated timing or
bounded causal ablations and report uncertainty. `record_gpu_frame` exposes the
existing `--completion-probe` and `--half-pixels` oracle controls explicitly;
receipts mark them diagnostic-only and reject a production DLL without those
controls. The latter changes coverage and cannot establish visual acceptance.
On this VM an EVENT query can report readiness before a one-pixel readback stops
waiting; neither query readiness nor short CPU submission proves completed pixels.

Resident, unchanged destinations require zero static-world compilation,
static-geometry allocation/upload, foreground content joins and map readback.
GPU command dependencies still exist, and normal dynamic parameter uploads are
allowed. Explicit native CPU/UI ownership barriers are measured separately and
must not be silently counted as a camera-only pass. Validate coverage/pixels
against independent cold renders and exercise live overlays, fog, picking,
selection, action centering, cancellation and config-off after every cutover.

The target is <33 ms p95 from input or native camera decision to first correct
coherent display for unchanged initialized-world navigation on nominal 100 × 100
Standard maps (5,000 actual tiles), with viewport, density and target hardware
declared. Huge maps (12,800 actual tiles) retain separate capacity/latency results;
they need not meet 33 ms for the Standard-map performance win. The later objective
is <16.7 ms. Report mean, median, p95, maximum,
sample count and each >100 ms stall separately for each camera-trigger class.
An aggregate dominated by fast manual pans cannot pass slow selected-unit jumps.
Also report end-to-end results for normal first visits including every readiness
miss: a fast conditional resident result cannot replace the user-facing metric.
Desktop completion is not physical scanout. A short enqueue, old front still
animating, or correct destination with stale native overlays/picking is not a pass.
M3.8 closes structural readiness/cutover gaps; remaining rendering-budget misses
must retain an explicit owner and unpassed status through M4.

## Existing harnesses

Paths below are relative to the repository root. Use `--help` for case-specific
options; these are entry points, not commands to run automatically on continuation.

| Entry point | Purpose |
| --- | --- |
| `python3 Renderer/renderer.py test CATEGORY` / `integration CATEGORY` | Dependency-selected production category and integration checks |
| `python3 -m Renderer.native.record_gpu_frame --scene SCENE --dll DLL --width 1120 --height 1192 --benchmark` | Connected native CPU/GPU whole-request comparison, display and ownership checks; records 384 timed samples across routes/workloads |
| Same command with `--visual-only` instead of `--benchmark` | Independent visual frames, actual timer transport and retained native UI, without the foreground comparison |
| `python3 -m Renderer.native.record_renderer_build` | Isolated x86 build provenance and dependency-aware reuse |
| `python3 -m Renderer.native.record_navigation_evidence` / `analyze_navigation_run` | Deterministic navigation, repeated cases, reset/warmup accounting and matched endpoint analysis |
| `python3 Renderer/renderer.py diagnose-navigation` | Existing route/object/pixel causal ablations, only when relevant to an unresolved question |

The connected GPU fixture requires the existing captured scene and audited local
JGL binary; preserved inputs live under `Renderer/native/build/gpu-composition/`.
A `--benchmark` receipt's CPU/GPU arms compare routes within the supplied DLL;
comparing old/new implementations additionally requires separate preserved DLL
runs with equivalent settings. Capture is outside that harness's timed requests;
its receipt states this limit. Do not label the result complete input-to-display.
`--dense-scene` enables the existing world-fixed city/infrastructure/resource
stress fixture; match it in both arms when measuring missing-object construction.

`--world-readiness-only` runs bounded full-world capture and canonical preparation
before 100 seeded distributed destinations, followed by six independent cold-image
oracles. Preparation must finish the declared authority sequence and every region;
a timeout or unavailable region fails the coverage claim. It saves each oracle
image for comparison with the established expanded-geometry path. This fixture
measures actual HWND/desktop completion but excludes Civ III input/capture/overlays.
`--world-geometry-mib` is an isolated capacity control, not a new production default.
`--rigid-sources 0` selects the expanded control in benchmark builds only. Report
shared-source allocation, dynamic instance streams and restored static geometry
separately: compiled backing reuse does not imply GPU residency.

The ordinary fixture's few central objects can stay resident throughout scrolling. Use
a coastal scene with the default `--waves 1 --reflections 1 --water-motion 1`
to exercise resident water and shoreline animation. Effects-off arms measure
added effect cost, not a visual-equivalent speedup. `scene-reflection` reports
mirror-cell builds/reuse, resolved-image bytes and the existing cache budget.
`scene-waves` reports visible/frozen occurrences, geometry/upload bytes and cell
build/reuse counts. Correlate it with `shared-scene-surface` and frame QPC bounds
to prove static submissions and ribbon uploads disappear on time-only updates.
The independent `VISUAL_SAMPLE` intervals also carry QPC bounds and are retained
in benchmark receipts. Report them separately: stationary native requests can
reuse a held map and do not measure independent map animation cadence. The
independent fixture has one selected unit, regardless of `--unit-count`.
`--object-workers 0` runs the identical GPU-ready object compiler synchronously;
keep source/DLL identity, budgets and instrumentation matched when using it as a
scheduling control. Also measure a flat dense-city scene and verify nonzero legal
city part counts: the mixed-terrain fixture can contain only fallback buildings.
`--dense-city-case 0,3,1,1` selects the supported modern medium capital case at
every dense city site, exercising source ground and paving as well as rigid
bodies. It changes only the explicit stress fixture; match it in both arms.
The `object-preparation` trace reports worker/join time, queued memory, rejected or
evicted jobs, recovery and adopted GPU bytes. Its worker time overlaps terrain,
ground and foreground work; do not add it to whole-request time.
The ordinary world path reports `world-preparation`: selected/consumed jobs,
rejection/eviction/recovery, lane count, ready CPU+GPU+proof peak, GPU upload bytes,
worker span and foreground join. Per-component ground/terrain/object/upload spans
are summed producer wall times; they overlap across lanes and include driver waits.
The world join occurs before the tile-local `mesh-phases` timers, so that trace
alone no longer includes every preparation wait. Correlate both with whole-request
QPC boundaries. The former ground/object queue traces report zero scheduled jobs
on this path; they still describe reduced/non-world and explicit serial controls.
The existing total worker allowance is unchanged; the combined ready budget is
64 MiB by default rather than adding separate ground/object queues to that budget.
The world pool now persists across frame leases. `ready_reused` counts validated
completed results surviving an earlier lease; `ready_bytes` reports the remaining
owned reservoir after readers join and borrowed callbacks retire. Worker and
consumption counters remain per-request deltas; reused compilation does not add
its historical producer times to the current frame. `queue_peak_bytes` is the
pool's high-water mark, reset when its budget changes. Compare it with the same
64 MiB cap and sampled process address space, not with another independent cache
allowance. Reset and asset/configuration changes retire the pool's content.

`record_gpu_frame --atomic-camera-views` requires the production atomic GPU view
export and runs 16 transitions: clip/animation metadata, honest held samples,
occurrence order, zoom, extent, panning, wrapped coordinates, visibility and scene
scope epochs, topology revision, lighting/season, configuration and reset. Each
adopted result must carry the exact copied description; retired tickets cannot
alias a recreated worker. Independent cold CPU renders verify pixels and
replacement coverage. This is a correctness workload, separate from the complete
native frame timing comparison. Keep all water effects enabled.

`record_gpu_frame --camera-requests` additionally exercises the production GPU
camera exports: 64 superseding exact requests, duplicate tickets, freed caller
inputs, current-front reads during assembly, stable adoption, independent pixel
parity, cancellation, synchronous joining and reset retirement. It records begin
latency separately from whole-frame timings and restores a clean session before
the native workload. Pending polls must leave output untouched. Pair this with
the held-render worker test; short observed timings alone cannot prove absence
of a render wait. An OK poll still imports the completed map on the GPU owner,
and the M3.6 live caller retains exact barriers for directed native work,
programmatic centering and projection changes.

`FRAME_SAMPLE` QPC boundaries locate the corresponding aggregate renderer trace.
`geometry_ms` includes content assembly, uploads and possible driver waits;
`draw_ms` excludes later shared-scene composition and is not total GPU time.
Selected-pass `selection_ms` includes caster/receiver preparation. Use these spans
for bounded attribution alongside whole-request and desktop-completion results.

Navigation session resets include `process_cold`, `assets_loaded` and
`prepared_resident`; inspect reset receipts rather than assuming all caches are
cold. Constructor-only settings require another process. Never unload active
workers or borrowed publications. No harness stages, installs or launches Civ III.
Windows D3D/injected verification uses the documented VM/shared-checkout workflow.
After transport failure, confirm child completion before retrying; do not kill
unrelated processes. Separate correctness, performance, capacity and environment
failures rather than treating a failed dispatch as a renderer result.

## GPU timestamp validity on the current VM

The September 13 Parallels probe found mutable completed D3D11 timestamps;
retrieval order could reverse event order. Query owners reject mutable, disjoint,
error, incomplete and nonmonotonic results. Older `valid=1` records without that
guard are not calibrated GPU timing. Preserve wall-time evidence independently.

Use bounded workload ablations when timestamps cannot answer a causal question.
Omitted route/body/shadow/finish work or reduced pixel coverage produces diagnostic
pixels, not an acceptable renderer. Differences depend on the other work present;
they are not additive pure GPU costs. Counters establish eliminated work, while
whole-request and actual display evidence establish user benefit.

## Preserve the investment

Keep source, licensed packs, references, staged/rollback DLLs, input manifests and
representative evidence. [Storage policy](storage_retention.md) governs cleanup;
ignored files are not automatically disposable. [Checkpoint evidence](history/retained_renderer_checkpoints_20260915.md)
links current controls and earlier findings. Do not repeat closed mechanisms
without a materially different implementation and reason:

- More prepared coverage can lose through eviction/refill or address-space limits.
- Smaller readback atlases gave exact pixels without a material latency gain.
- Full-view/enlarged-block projection shortcuts broke guarded depth/projection.
- Transparent wave overlays lacked required scene/depth context.
- An index removed 98% of bounds tests but regressed total frame time.
- Raising map sampling globally to 30 Hz regressed warm request cost.
- Empty-pass filtering and parameter streaming alone did not reliably improve
  whole scrolling requests. Compatible layer grouping removed real submission work,
  but exposed content construction/upload as the larger remaining cost.
- Expanding terrain workers into the outer topology halo, including a demand-first
  revision, increased cache pressure/scrolling cost. That expansion was removed;
  revisit only with a different reuse/working-set mechanism.

These findings constrain mechanisms, not all future batching or retained rendering.
Detailed old tooling proposals remain recoverable from Git at `c1360ea9`; they
are no longer a prerequisite list.

## Result record

Extend existing receipts with source/binary/assets and fixture identity; initial
state/preparation/reset coverage; latency endpoints/sample counts/variation;
publication age, cancellation and useful prepared work; independent correctness,
resource bounds and missing coverage; capability gained and measured effect.
Update the roadmap's short status, not a new diary. Retain/reject/inconclusive
classifications must distinguish structural progress from demonstrated speedup.

## Tactical overlay evidence

`record_gpu_frame --tactical --visual-only` exercises selected markers, copied
native JGL route lines/turn text, grid on/off and exact erase/cancellation through
an admitted native map. It records actual display previews and independent
`tactical-execute` traces. `TACTICAL_VISUAL_SAMPLE` includes retained map/unit/UI
replay, tactical drawing and presentation; desktop completion is separate.

`--tactical --benchmark` adds grid, selection and route primitives inside every
GPU whole-request interval (before units/UI). Both the 8- and 32-unit workloads
contain one selection highlight and one route; only unit-body count changes.
Compare against the same DLL with
`--tactical` absent to measure added feature cost. The CPU benchmark arm keeps its
existing output, so a tactical-enabled CPU/GPU ratio is not an equivalent-image
speedup. Tactical packet preparation is included; authoritative gameplay route
calculation remains outside this captured-input fixture.

## Milestone 2 combined animation acceptance

`record_gpu_frame --visual-units 1|8|16|32 --visual-unit-case selected|work|mixed`
sets the actual retained independent-frame workload. There is exactly one selected
idle unit; `work` adds authored worker loops, while `mixed` adds frozen idle and
captured native-action cases. This count is separate from `--unit-count`, which
controls the complete native-demand benchmark. Use `--visual-frames 120` for
percentile evidence. `--tactical` adds route/marker/grid lifecycle witnesses.

Use production runs without `--profile` for cadence. Detailed profiling performs
whole-process address-space walks at several boundaries; the measured overhead
can be tens of milliseconds per frame. Keep separate profiled memory/attribution
runs and do not call removal of diagnostic overhead a renderer speedup. The
native fixture's explicit memory samples still report address-space headroom.

`python3 -m Renderer.native.analyze_visual_frames <receipt-directory>` joins each
visual request to its exact trace interval. Missing stages are unproved, not zero.
It records static geometry/draw/caster work, mirror builds, water/wave uploads,
material selection reuse and complete-receipt status. Historical DLL comparisons
record their build receipt independently of the current harness source closure.
Failed or incomplete fixture runs cannot establish acceptance.

The shoreline lifecycle keeps water motion and reflections on during normal
playback, zoom and scrolling. Its wave-off control compares the same time with
only foam disabled; water still animates. A separate both-motion-off control
proves the stopped image. Do not infer that disabling one effect stops another.

`--native-camera-requests` routes the native composition fixture through the
production begin/poll/validate/commit exports. Pending polls must leave output
untouched and cannot commit. `NATIVE_CAMERA_SAMPLE` separates enqueue, maximum
poll duration and completion time; the test driver alone waits between polls.
The driver services real messages and waits for the registered completion hint,
with a bounded retry timeout. It does not spin or raise system timer resolution.
The ordinary whole-request/desktop measurements still include that completion
wait, native units, UI and transfer. This prevents reporting a short enqueue as
an end-to-end speedup. Match all effects/assets and keep the synchronous run as
a control. These are real DLL/JGL transactions, not a live Animator/picking test;
the guarded injected camera cutover is described under M3.6 below; the complete
live input-to-display acceptance remains M3.8.

`record_gpu_frame --native-navigation` exercises the M3.6 DLL navigation owner
through real JGL admission, pending polling, ready camera advancement, fresh
capture validation and native PREPARE/COMMIT. It retains the existing composition,
CPU compatibility and final-transfer oracles. The driver services completion
hints; the injected bridge uses native Animator opportunities. These timings
exclude live native capture and do not prove live input-to-display latency.
Extracted hook tests separately verify unchanged pending camera/picking inputs,
native early returns, action barriers, wrapped camera bounds and immediate native
unit-selection centering. The production owner test changes anchors, visibility,
topology and viewer scope between readiness and commit and rejects every stale
prepared result. Both tests remain in the normal integration suite.
