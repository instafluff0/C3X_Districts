# Retained renderer program

Current status and preserved evidence for the retained renderer. Planned work
is not evidence that implementation or performance targets pass.

## Current status — move from local raster experiments to pass consolidation

Tooling is complete. Static-pass handoff and bounded animation postprocessing
were correct but failed the whole-sequence usefulness criterion; both renderer
candidates have been removed. The latest profiling-off dense baseline averages
195.805 / 198.330 ms per request. The below-100 ms navigation gate and native
presentation remain unmet.

[The architecture](renderer_architecture.md) defines the destination;
[the execution contract](autonomous_renderer_execution.md) governs continuation;
[the benchmark workflow](benchmark_workflow.md) specifies validation. This status
is the single active task record.

### Prioritization correction

The user explicitly requested established RTS rendering approaches to guide
priorities rather than a chain of nearby experiments. Apply the destination's
persistent world content/local revisions, spatial selection with batching and
instancing, explicit static/dynamic passes and bounded GPU ownership as the
basis for implementation selection. Measurements choose among these mechanisms
and validate complete changes; a large phase does not justify successively
trimming every nearby operation. Preserve useful existing content reuse rather
than adding more completed-image cache policy. Keep readback/publication at the
Civ III bridge, with its exact-camera/overlay constraints.

The concrete code issue now selected is repeated general-purpose pass setup:
`compose_resource_animations` calls `submit_geometry` for each animated region.
That path rebuilds/binds a terrain resource table, visits natural-terrain/forest
pipeline bindings even when those layers are empty, and handles general terrain,
reflection, lighting and finishing responsibilities for a body/shadow workload.
This is a reason to consolidate the pass, not proof of a predicted speedup.

**Next task:** complete the animated-resource pass with bounded GPU region batches.
Preparation/binding consolidation alone did not reduce whole-request latency.
Batch compatible body/shadow submissions and reconstruction across independently
guarded region surfaces under the existing renderer/context owner. Preserve each
region's 136-pixel projection, MSAA4 samples, depth, draw order and exact output;
this is not a larger-projection or reduced-pixel shortcut. Charge any transient
batch surfaces against the existing backdrop allowance and preserve fallback and
reset lifetimes. Reuse the current candidate's pass selection and shared shadow
preparation as scaffolding, keeping it opt-in until the complete pass is validated.
Do not promote preparation-only organization as a speedup or tune its metadata.

Use the boundary/full-redraw fixture first, followed by matched profiling-off dense
navigation and affected animation/effect checks. Evaluate complete request latency,
setup and headroom. Retention requires exact observable output and a material whole-
sequence benefit (at least 10% and 20 ms outside repeat variation). No new live
rendering ownership, staged binary or visual acceptance is implied.

### Prepared resource pass — revise to batch GPU execution

The candidate now prepares ordered, region-specific body/shadow draw lists and
submits them through an explicit GPU pass using the existing shader/depth/guard
contract. Consecutive regions share exact source-shadow page demand in batches
within the existing 32-page atlas. A static backdrop miss invalidates the active
batch before animation resumes. The synchronous pass borrows pinned pose buffers;
its retained draw/page descriptions have a conservative 4 MiB allowance and fall
back to ordinary submission when exceeded. Wave frames retain ordinary submission
until their scene/depth pass contract is explicitly covered. No cache budget,
rendering ownership, staged binary or asset changes are involved.

`prepared-resource-batched-build-20260912` compiled successfully.
`prepared-resource-batched-boundary-20260912` passed all six exact comparisons
against independent ordinary-path full redraws, including camera/guard movement,
advancing clock, removal and reappearance; inputs/sources/binaries were unchanged.
`test resources` passed 132 tests with one existing skip. These establish the
initial correctness checkpoint, not dense navigation performance.

`prepared-resource-reversal-candidate2-20260912/comparison.json` closes the
preparation-only candidate: two profiling-off 14-request controls averaged
193.713 / 194.173 ms; candidates 198.170 / 193.230 ms. Paired savings were
-4.457 / 0.943 ms, below the 20 ms criterion. All four runs had exact saved
images/revisits, unchanged inputs/sources/binaries and no observed competing GPU
or compiler process. Host-boundary contiguous headroom was at least 1,691.730 MiB;
internal transient/GPU-query coverage remains unmeasured. The first four-column
candidate was exact but suffered a 3,841.913 ms readback wait (4,058.441 ms whole
request); the matched control was approximately 336 ms. That stall did not recur
in either reversal, but its cause is unestablished and it is not discarded as a
known environment fault. No useful performance pass or advancement gate is claimed.

The remaining dominant animation cost is GPU completion of regional rendering and
reconstruction, with per-request pose/selection work also present. Current code
reduces setup calls but still executes the same serial region passes. The next
revision groups actual GPU work; the preparation candidate remains temporary
scaffolding under the same opt-in control, not a retained production optimization.
Focused region/publication/analysis checks passed 29 tests; the Pillow-dependent
check was rerun successfully using the bundled Python after system Python lacked
Pillow.

### Session tooling evidence (deliverable 2 complete)

Per-translation-unit include hashing and compiler/SDK stamps now reuse intact
objects. `session-incremental-seed-retry-20260912` compiled seven units in 37.345 s
wrapper time; `session-incremental-warm-20260912` compiled none in 3.108 s. Changing
only the preview compiled that unit alone. Recipe, dependency and object changes
reject reuse.

`session-four-verified-20260912/comparison.json` records four assets-loaded cases:
39.943 s wrapper total, 9.986 s per case versus 14.285 s for the matched fresh
one-shot. All initial/final images exactly match that independent fresh process;
full inputs, sources and binaries verified unchanged. This is about 30% less
iteration wait in this short comparison, not a renderer speedup or tail estimate.
Resets preserve budgets and clear mutable content/publication while retaining
assets/device. `session-prepared-verified-20260912/comparison.json` independently
matches the same fresh images after explicit warmup (5.056 s through its final
check), retaining 84.174 MB geometry and 30.628 MB ground content with no evictions.
Its single 146.326 ms small-view transition is not a navigation performance pass.

An earlier four-case run failed at teardown and entered Windows Error Reporting;
it remains invalid evidence. A subsequent watchdog exit-code reporting fault was
fixed and verified against success, nonzero exit and owned-process timeout cases.
The accepted runs above completed teardown under the bounded watchdog. The initial
transient teardown fault was not reproduced or localized; preserve that limitation
without reopening completed session validation as an unbounded investigation.

### Current evidence and limits

Ignored evidence under `Renderer/native/build/`:

- Bounded animation reconstruction is **rejected and removed**.
  `bounded-resource-boundary-20260912` passed six exact independent full-redraw
  checks; focused regressions passed (16), resource category tests passed (132,
  one existing skip), and navigation analysis/causal checks passed (16).
  Four-column time was 354.472 ms baseline versus 339.654 ms candidate, exact
  saved images but only 14.818 ms saved. Deciding profiling-off reversal repeats
  in `bounded-resource-reversal-candidate2-20260912/comparison.json` were
  195.805 / 198.330 ms baseline versus 189.115 / 190.932 ms candidate: only
  6.690 / 7.398 ms saved, below the 20 ms threshold. All four runs had identical
  saved endpoint pixels, exact revisits, unchanged inputs/sources/binaries and
  no observed GPU/build conflicts. Host-boundary contiguous headroom was at least
  1,689.332 MiB; internal transient/GPU-query coverage was unmeasured with profiling
  off. Reduced postprocessing area was not a sufficient architectural improvement.
  Advancing-clock expansion was not earned. The existing analyzer now supports
  explicitly requested profiling-off cases with constructor/host-only coverage
  labels and rejects accidental mixing with profiled cases.


- Static-pass sharing is **rejected and removed**. The preserved
  `shared-static-build-20260912` / `shared-static-boundary-20260912` passed six
  exact independent full-redraw comparisons (pixels, replacement flags and
  animation counts). Four-column paired repeats were 392.812 / 405.994 ms baseline
  versus 329.058 / 332.316 ms candidate, a 16.2–18.1% saving. However,
  `shared-static-reversal-candidate2-20260912/comparison.json` records the deciding
  14-request sequence: baseline 227.392 / 249.731 ms, candidate 244.933 / 246.393 ms.
  Paired savings were -17.541 / 3.338 ms, below the 23.856 ms usefulness threshold.
  All four sequence runs matched saved endpoint pixels, exact revisits, unchanged
  inputs/sources/binaries and zero observed GPU/build conflicts; minimum sampled
  contiguous headroom was 1,694.574 MiB. No budget was increased. Handoffs were
  often unused (114 of 180 initial captures); reduced duplicate submission did
  not yield a whole-sequence win. Advancing-clock dense expansion was not earned.
  The earlier failed baseline dispatch produced no child and remains invalid;
  its completed retry is `shared-static-four-base2-retry-20260912`.
- `retained-production-cost-20260912/production-cost.json`: one profiling-off
  14-request sequence against the cleaned renderer averages 193.617 ms. Phase
  means are geometry 34.906, submission 6.646, static readback 27.236 and animation
  composition 114.019 ms. Revisit geometry is 11.474–12.106 ms; revisit animation
  is 68.844–80.381 ms. All saved images match the profiled baseline and revisits
  are exact. This isolates production work; turning off diagnostic memory scans
  is not a renderer improvement. It is a single scoped cost check, not a repeated
  performance gate, independent full-redraw oracle or live-game measurement.
  Profiling-off omits per-frame address-space/GPU-query coverage; the separate
  profiled runs above preserve headroom evidence for the same budgets/workload.
- `retained-boundary-clean-20260912`: cleaned renderer build reuses the original
  DLL translation unit and rebuilds the retained preview fixture. Region/cache/
  causal tests pass (30); resource category tests previously passed (132, one
  existing skip). `retained-boundary-clean-check-20260912` passes all six exact
  reset/full-redraw comparisons with unchanged inputs/sources/binaries. The
  temporary handoff retirement test was removed with its
  rejected owner, preserving the original budget/LRU behavior test. No injected
  edit, staging, installation, game launch or reference replacement.

- `dense-route-current-20260912/results.json`: completed automatic batch, two
  repetitions in alternating arm order, 2240×1192, tile width 128, fixed clock,
  waves/reflections off, normal retained budgets. The 100×100 world supplies
  1,921 captured occurrences with 5 cities, 349 roads, 80 rails, 344 improvements
  and 150 resources. All 20 cases verified unchanged inputs/sources/binaries and
  no observed competing GPU/build process; minimum sampled contiguous headroom
  1,681.852 MiB. Repeated endpoints are exact; prepared-content endpoints match
  fresh-content rendering. Intermediate independent full-redraw parity is still
  unmeasured, and omitted-pixel arms cannot establish production correctness.

  | Arm | Four-column moves (ms) | Reversal mean per request (ms) | Decision |
  | --- | --- | --- | --- |
  | Full | 391.963 / 402.839 | 251.355 / 249.466 | Target unmet |
  | Route draws omitted | 394.056 / 396.714 | 243.751 / 250.359 | Reject primary target |
  | Route surfaces omitted | 405.555 / 406.747 | 245.846 / 250.916 | Reject primary target |
  | Prepared content | 345.472 / 309.586 | 224.613 / 227.262 | Useful four-column effect; reversal inconclusive at 10% threshold |
  | Half geometry pixels | 341.074 / 347.185 | 223.917 / 224.520 | Useful four-column diagnostic; reversal inconclusive at 10% threshold |

  Baseline reversal phase means: geometry 62.799 ms, submission 20.311 ms,
  readback 27.963 ms, animation composition 114.886 ms. Existing traces and source
  inspection show animated-region backdrop misses resubmit static geometry after
  main-map rendering. That observation led to the now-rejected sharing candidate above; the
  diagnostic batch itself established neither correctness nor speedup.


- `dense-route-quiet-20260912`: two uncontested, fully verified four-column cases
  completed on the 2240×1192 dense fixture, tile width 128, waves/reflections off,
  existing retained defaults and normal budgets. The 100×100 world supplied 1,921
  captured occurrences: 5 cities, 349 roads, 80 rails, 344 improvements and 150
  resources. Full rendering took 398.012 ms; route surface draws omitted took
  395.704 ms. Their pixels differ as intended. These are single samples, not a
  matched-repeat decision: route drawing showed no large effect in this pair.
  Full-render phases were geometry 114.525, submission 31.905, readback 71.509
  and animation composition 150.163 ms; sampled contiguous headroom exceeded
  1.8 GiB. Initial setup remains separately recorded. Runtime shader changes and
  another `native_preview` interrupted the third arm; it was rejected, and no
  performance conclusion is drawn from that arm or the incomplete matrix.
- `dense-route-diag2-20260912` contains the earlier overlapping batch, explicitly
  invalid for causal timing. The overlap watchdog was then added and correctly
  rejected `dense-route-serial-20260912` before rendering. It monitors known Lab
  renderer/compiler processes once per second, records conflicts and terminates
  only its own child. All diagnostic children are terminal. An initial startup
  exit in `dense-route-diagnostic-20260912` was not reproduced; the preview now
  reports the exact setup precondition on any recurrence.
- `dense-diag-setup-20260912`: isolated x86 build passed, with benchmark-only route
  surface and half-pixel controls. The category dispatcher exposes the fixed
  batch and the existing analyzer owns its comparisons. Focused causal/endpoint/
  build/damage tests passed (19); fixture/trace/worker checks passed (8).
  `test infrastructure` passed 123 tests with one existing skip. No injected edit,
  staged binary, reference replacement, installation or game launch.
- `endpoint-short-20260912`: 640×480, width 128, waves off, fixed-clock 14-offset
  reversal. All 14 requests and exact revisits passed, with zero fallback/recovery.
  Capture median 0.371 ms; request-to-checked-result median 164.303 ms, range
  31.628–858.733 ms. Initial render 5.550 s; before/after input verification 7.010 s;
  wrapper total 18.389 s. Revisit equality is repeatability, not full-redraw parity.
- `endpoint-four-column-20260912/analysis.json`: completed final endpoint layout,
  same small viewport and one four-column move. 759 captured occurrences, 728.304 ms
  request-to-checked-result: capture 0.447, worker rendering 725.935, queue 0.053,
  snapshot/drain 0.773, publication/preparation 0.905 ms. Within worker rendering,
  geometry 607.059, submission 14.240, readback 57.834, animation composition 39.297,
  unaccounted remainder 7.505 ms. The final delayed GPU sample is unmeasured; no
  blocking query was added. Native capture/presentation and independent pixel
  parity remain outside this tooling check. Full runtime input/binary verification
  passed; both builds produced the same initial image hash.
- `endpoint-accounting-complete-20260912/build-evidence.json`: isolated x86 `/O2`
  build passed. Compiler setup 2.110 s, DLL compile/link 22.620 s, preview 4.860 s;
  outer compile/dispatch 30.462 s. No staging, install, game launch or injected edit.
- Endpoint ordering/missing-query/invalid-span analysis, bounded trace behavior,
  dense fixture identity and delayed GPU telemetry tests pass (15 tests). Category
  dispatcher `test infrastructure`: 123 tests passed, one existing skip. The
  system Python 3.9 cannot run the telemetry test's newer tempfile option; the
  bundled workspace Python ran it successfully without changing that test.

**Dominant measured cost:** animated-resource composition (114.019 ms mean in
the profiling-off sequence), including GPU work awaited by the final readback.
Newly exposed content still adds geometry preparation and static rendering.
Diagnostic address-space scans inflated the earlier geometry/submission numbers;
do not treat their removal as architectural progress. Route-specific tuning,
unfiltered static-pass sharing and bounded animation reconstruction are closed.
The next change consolidates animated GPU pass preparation/submission; actual
benefit remains to be demonstrated on the full workload.

The selected rendering migration remains retained route/object composition:
translate the valid static front, compose exposed strips/dirty bounds and keep
ambient/unit invalidation independent. Extend persistent content and local
dependencies where the diagnostics justify them. This is a step toward the
architecture, not its permanent definition. If evidence rejects the selected
causal target, replace this record's next task with one supported alternative;
do not start a competing sequence or the architecture's entire mechanism list.

| Capability / target | Evidence carried forward | Remaining limit |
| --- | --- | --- |
| Stationary retained front | Opt-in same-view ambient publication and independent cached-pose CPU composition are implemented; prior standalone minute-long 24-unit run reports 1.720 ms p95 render entry and 7.42 Hz fresh maps. | Still captures full native view; this is not live presentation or fast navigation. |
| Useful preparation | Existing `prefetch-guard2-replay-20260910/benchmark.log` records 176 prepared tiles before its first jump and exact return parity. | First three jumps were 707.534, 828.214 and 785.799 ms; short diagnostic, not a sustained gate. |
| Dense resident navigation | Retained overlap and bounded working sets already exist; preserve them. | Below-100 ms gate unmet; route/object composition and synchronous completion remain the selected causal target. |
| Regional geometry / zoom | Some structural sharing exists. | General regional batching shared across 128/160/192 is unfinished. |
| Native asynchronous presentation | Versioned publication and stale-ticket rejection exist in the DLL. | Native coordination, current-camera presentation and input-to-visible performance remain unverified. |
| Iteration tooling | Timing, incremental compilation, persistent sessions and the fixed dense causal batch are complete. | Additional tooling requires a specific missing correctness/measurement question for the selected rendering change. |

The preparation receipt is under `Renderer/native/build/`; older stationary
measurements are preserved in the linked experiment archive. These are scoped inherited
results, not a newly matched baseline/candidate comparison or current-binary
certification. No universal percentage-complete or overall speedup is claimed.

Preserve the staged DLL, rollback, licensed assets, source findings and existing
uncommitted renderer/build changes. Do not promote experimental options or alter
cache budgets, visual ownership or references during tooling work. Record future
progress here as capability, workload, evidence, unmet target and one next action.

## Preserved evidence

The [experiment archive](history/retained_experiments_20260910.md) retains the
original protocols, measurements and failed hypotheses. Read only the evidence
needed for the current mechanism; its embedded next-step instructions are historical.
