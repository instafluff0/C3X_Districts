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
