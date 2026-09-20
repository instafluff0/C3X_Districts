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
matched/alternated runs. No benchmark may overlap compilation or another GPU test.
Report sample counts and variation; close results are inconclusive. Do not infer
p99 from short runs. Detailed traces should be buffered outside timed work.
Before claiming acceptance, verify full source/binary/assets identities before and
after the run; quick metadata-only receipts remain provisional.

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
