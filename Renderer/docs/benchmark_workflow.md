# Renderer benchmark workflow

## Status and authority

This document defines measurement, bounded tooling deliverables and validation
for the renderer. The user requested groundwork, not another test campaign.
This change defines the work; it does not implement a
persistent runner, correct timers, run benchmarks, or establish new performance.

Read this with [the architecture](renderer_architecture.md) for the destination
and [the execution contract](autonomous_renderer_execution.md) for bounded decisions.
[The retained plan](retained_renderer_plan.md) alone records current status and the
single next task; the deliverable order below supports that task sequence.
Older handoffs and the retained plan's experiment history preserve evidence;
their embedded "next" instructions are superseded. AGENTS.md and the Lab's
ownership, category verification and visual-acceptance rules still apply.

The tooling phase starts with timing and correctness accounting in the existing
harness and ends at the three validated deliverables below. Do not begin by
rerunning an oracle, broad baseline, cold matrix, idle soak or failed rendering experiment.
When implementation resumes, validate each tooling increment only as needed;
the groundwork request itself does not call for those runs.

## Implemented harness entry points

The retained plan records which deliverables and evidence are complete. The
existing owners expose these bounded commands (all outputs below are ignored):

```sh
python3 -m Renderer.native.record_renderer_build --out Renderer/native/build/example-build --tier normal --reuse-from Renderer/native/build/previous-build
python3 -m Renderer.native.record_navigation_evidence --binaries Renderer/native/build/example-build --out Renderer/native/build/example-cases --tier normal --scenario scroll --case-repeats 4 --case-reset assets_loaded --waves 0 --width 640 --height 480 --profile
python3 -m Renderer.native.analyze_navigation_run Renderer/native/build/example-cases --case-reference Renderer/native/build/fresh-one-shot --out Renderer/native/build/example-cases/comparison.json
python3 Renderer/renderer.py diagnose-navigation --binaries Renderer/native/build/example-build --out Renderer/native/build/example-dense-diagnostic
```

Omit `--case-repeats` for the existing one-shot interface. Add `--scroll-sequence`
for the fixed 14-offset reversal. Persistent cases currently share one immutable
constructor configuration and synchronous scroll workload; a changed configuration
uses another process. `process_cold` permits one case in a fresh process;
`assets_loaded` clears scene content while retaining assets/device;
`prepared_resident` performs the same untimed sequence before clearing completed
images and measuring. Reset receipts expose actual retained content and unchanged
budgets. `--case-time-limit` bounds the session; an external watchdog covers a
stuck child/teardown and only terminates that invocation's process tree.

`--instrumentation timing` disables detailed traces; diagnostic mode buffers them
until teardown. Missing internal/GPU coverage is explicit. `--verification quick`
uses provisional metadata checks after initial hashing and cannot pass acceptance.
Acceptance performs full before/after input verification. The session checks
file metadata between cases and stops on changes. Build reuse checks transitive
local includes, recipe/tier, intact objects and the compiler/SDK stamp. Neither
command stages the DLL or launches Civ III.

`diagnose-navigation` freezes the deliverable-3 world fixture, viewport, clocks,
normal retained budgets, two camera workloads and five arms. It alternates arm
order over two repetitions and uses the existing endpoint analyzer for causal
decisions. Road/rail surface controls preserve bridges and other objects. The
prepared arm retains all scene content, including route buffers; it does not
establish route-only attribution. The reduced-pixel arm halves geometry scissor
coverage with identical captured geometry and candidate selection; shadow pages,
finishing and readback remain ordinary. Pixel ablations cannot pass production
correctness. Saved receipts separate first exposure, revisits, setup and warmup.
The opt-in `--exclusive-gpu` watchdog rejects observed competing Lab renderer or
compiler processes before/during each case, sampling every second and terminating
only its own child on a conflict. A rejected/conflicted invocation is preserved,
never used as performance evidence. Batch completion remains recorded solely in
the retained plan; implementing this entry point does not itself close the task.

## Preserve the investment

Keep the current renderer, source findings, runtime packs, optional reference
images, staged DLL and rollback artifacts. Preserve concurrent uncommitted work.
Do not delete ignored assets or evidence as part of this transition.

Retain these mechanisms and extend their existing owners:

| Existing component | Role in the new workflow |
| --- | --- |
| `native/biq_preview.cpp` | Production-DLL host, immutable capture fixtures, short navigation and ambient workloads; add session execution here, not a second simulator. |
| `native/record_navigation_evidence.py` | Input/binary provenance, VM dispatch and evidence receipts; add explicit quick and acceptance modes. |
| `native/record_renderer_build.py`, `native/BENCHMARK_ZOOM.bat` | Isolated x86 candidates; make compilation incremental without changing release flags or staging policy. |
| `native/analyze_navigation_run.py` | Matched comparisons, timing summaries and coverage; extend instead of adding a competing analyzer. |
| `native/render_core/frame_telemetry.h` | Delayed nonblocking GPU queries and address-space observations; retain validity checks and expose missing measurements. |
| `native/busy_session_plan.h`, `native/retained_replay_preview.h` | Existing deterministic requests and paced input semantics; retain for later sustained validation. |
| Existing publication, pose, retention and capture tests | Regressions for current ownership, cancellation, action phase and lifecycle contracts. |

Keep the immutable stationary front, cached-pose CPU composition, camera queue,
stale-ticket rejection, retained geometry/backdrops and useful terrain preparation.
These are building blocks, not proof that dense navigation or native presentation
passes. Keep the two-tile prefetch control as an opt-in diagnostic; do not promote
it or increase its memory allowance to satisfy a benchmark.

Carry forward these closed hypotheses from the retained plan's experiment record:

| Prior experiment | Finding that must inform the next decision |
| --- | --- |
| Larger bounded preparation oracle | Working-set eviction/refill and address-space failures can erase preparation gains; preserve cost/capacity reporting. |
| Smaller animation readback atlas | Exact pixels alone did not produce a material latency win; staging area is not established as the dominant wait. |
| Full-view / enlarged-block projection shortcuts | Guarded block depth/projection contracts were not preserved; prove those contracts in the tiny fixture first. |
| Transparent wave overlay | Wave shading requires scene/depth context; simple CPU alpha composition is not an equivalent implementation. |
| Draw-candidate index with 98% fewer bounds tests | Frame time regressed; bounds-test reduction alone is not a useful performance target. |

These reject specific implementations/hypotheses, not all batching or retained
rendering. Reopen one only with a changed mechanism and a stated reason; do not
repeat it because an older note still calls it the next experiment.

## Tooling deliverables and rendering handoff

Implement the unfinished tooling deliverables in order, recording completion in
the retained plan. These are bounded prerequisites for the selected rendering
capability, not a recurring baseline campaign.

### 1. Make measurements explain their endpoints

Instrument wrapper elapsed time for source/input verification, compile/link,
dispatch, process/device startup, asset/shader load, preparation, playback,
comparison and evidence output. Report total user wait as well as render latency.
Do not guess which setup phase accounts for minutes of waiting.

Correct the short scroll timers: currently `capture_view()` precedes the start
timestamp, so `capture_ms` there is not capture duration. Report separately:

- Request-to-capture-complete and request-to-correct-result wall time.
- Caller entry/return and lock/queue waits; worker CPU phase intervals.
- GPU execution/copy intervals from valid delayed queries, associated by request.
- Blocking readback wait and CPU bitmap composition/publication.
- For paced work, accepted input, queue delay, fresh publications and image age.

CPU intervals may contain waits and GPU work overlaps CPU work. Do not add
overlapping totals or label staging `Map` duration as pure copy/GPU time.
Missing phase coverage is `unmeasured`, not zero. Quantify the unexplained portion
of an elapsed endpoint using nonoverlapping CPU spans; flag incomplete accounting
before recommending another micro-optimization. Buffer detailed traces and flush
outside timed work; separate diagnostic and low-overhead timing modes.

**Deliverable:** one structured report from the existing harness that separates
setup from playback and correctly accounts for capture. During implementation,
check timestamp ordering and one existing short workload; do not run a full matrix.

### 2. Amortize setup without contaminating results

Add a bounded request loop to `biq_preview.cpp`. One process loads a verified
asset set and device, then runs multiple cases against the real production DLL.
Keep the one-shot interface for reproduction and acceptance. A small versioned
request/response format should include case ID, configuration ID, immutable
request-sequence digest, reset policy, warmup policy, repeat count and time limit.
No fixture or command may silently inherit a previous case's environment settings.

Distinguish three initial states: process-cold, renderer-cache-cold with assets
loaded, and explicitly prepared resident. Reset renderer caches/publications and
restore identical preparation for each matched arm. Record what reset actually
clears; device/shader/driver caches must not be implied cold. Use separate fresh
processes where configuration is constructor-only or reliable reset is unavailable.
Never dynamically unload a DLL with active workers or borrowed GPU publications.

Make build inputs dependency-aware so an unchanged preview or DLL translation
unit is not rebuilt. Preserve x86/LAA behavior, compiler flags, source closure
and candidate identities. No timed GPU work may overlap compilation or another
GPU benchmark. End a session when its inputs or binary change.

Verify assets once against an immutable session input manifest, use cheap change
detection between quick cases, and label those receipts provisional. Changed
inputs invalidate the session. Acceptance retains full before/after verification;
metadata checks alone must never certify immutable content. Avoid copying packs
or writing every frame image. Save bounded mismatch artifacts and summary data;
perform comparisons outside measured intervals and respect the evidence reserve.

**Deliverable:** repeated short cases share startup, with visible reset/warmup
accounting and reduced measured setup cost. A fresh one-shot run must reproduce
the same case's output. Persistence must not change cache budgets or draw policy.

### 3. Add one automatic route/object diagnostic batch

Freeze one workload manifest using the existing dense world fixture: 2240x1192,
width 128, fixed camera/clock, explicit city/route/improvement/resource contents,
waves and reflections off, recorded cache limits and captured tile counts.
Use the existing four-column move and the existing 14-offset reversal sequence.
Separate cold exposed strips, prepared exposed strips and revisits. Retaining
the front/overlap and structural data is intended; replaying an entire finished
destination image must not hide exposed-strip composition in the navigation gate.

Run these causal controls serially in one batch, with equivalent initial state:

| Arm | Question |
| --- | --- |
| Full production rendering | What is the actual transition cost? |
| Route draws omitted, preparation retained | How much work is downstream of route submission? |
| Route preparation and draws omitted | Is route preparation substantial? |
| Explicitly prepared route buffers, ordinary drawing | Can structural reuse remove the measured preparation? |
| Reduced pixel-work diagnostic, same captured geometry | Does the downstream cost respond to pixel workload? |

Implement only controls required to answer an unresolved question; reuse existing
controls when their semantics match. Changes that alter pixels are diagnostic
ablations and cannot pass the production correctness gate. Their differences
are not additive layer costs because rendering dependencies interact.

Alternate baseline/candidate order and restore equivalent state. Start with a
few matched repetitions to reject large regressions; extend only for close results.
Declare the expected phase reduction and a minimum useful whole-transition benefit
before running. If even eliminating that phase cannot materially advance the
100 ms target, choose a different architectural target. Reject an optimization
when correctness fails or the targeted phase does not improve. Small timing
differences within repeat variation are inconclusive, not wins.

**Deliverable:** one invocation returns baseline/arms, variation, causal phase
changes, correctness classifications and retain/reject/inconclusive decisions.
Do not keep running the batch once it cannot change the next decision.

### Tooling completion boundary

The tooling phase is complete when the endpoint report accounts for setup/capture,
repeated session cases show reset/warmup accounting and measured setup reduction
with one-shot output reproduction, and one invocation produces the diagnostic
batch's evidence and decisions. Use each deliverable's specified validation;
full gameplay performance is not a prerequisite for finishing tooling.

Mark these deliverables complete in the retained plan and replace its single next
task with one bounded scene/rendering implementation selected by the evidence.
The current migration step is the retained route/object path below. If the batch
rejects its causal premise, record the finding and select one evidence-backed
component instead. Do not repeat completed tooling or expand the harness without
a named missing measurement/correctness check that can change the implementation
decision. The architecture document adds no separate implementation queue.

### 4. Prove one retained route/object implementation

Before another full-scene projection experiment, add a tiny production-renderer
fixture: adjacent guarded blocks, an object crossing the boundary, a shadow,
an animated resource, and a camera translation. Compare the optimized path with
an independent full redraw using identical inputs and time. Revisit equality
alone is repeatability, not an independent correctness oracle.

On the first mismatch, save bounded color/depth/ownership diagnostics and a
cropped difference image. Classify projection, depth, sampling, contributor or
invalidation faults. Restore the validated contract rather than trying unrelated
full-target/transparent composition shortcuts. Exact pixels are required for
an optimization claiming unchanged output; a deliberate appearance change follows
the separate Lab review process, not a weakened performance comparison.

The current migration step is one coherent path: translate the immutable static
front, compose only exposed strips and dirty route/object bounds, keep ambient/unit updates independent,
then publish a correct current-camera result. Use the diagnostic report to decide
where preparation or regional mesh batching belongs. Do not add an LRU tier,
speculative disk format or second presenter as a substitute for this path.
Advance persistent route/object content, local dependency tracking and reusable
draw preparation through existing owners where they eliminate measured work.
This step does not make the current cache layout or raster blocks the permanent
architecture, nor require every preferred mechanism to be implemented at once.

**Deliverable:** tiny independent parity, then the fixed short dense sequence.
Only a useful, correct short result advances to sustained validation.

## Validation depth and stopping rules

These are test scopes, not a new release ladder. Use the category dispatcher for
affected formal checks; integrate the batch entry point with that workflow rather
than establishing a separate release gate.

Test observable contracts: pixels, depth, ownership, visibility, captured anchors,
action timing, cancellation/recovery and memory limits. Structural tests may evolve
when their implementation assumptions change, with equivalent behavioral coverage
and an explanation of the replaced assumption. Do not retain obsolete internals
solely to satisfy structural assertions, or weaken correctness for a speedup.
For the selected capability, verify bounded effects of view movement, local edits
and dynamic updates; do not build an unrelated architecture test matrix.

| Scope | Trigger and bounded purpose |
| --- | --- |
| Tiny contract | Relevant edit; catch a boundary/depth/ownership defect before loading a dense world. |
| Short diagnostic | An unresolved causal hypothesis; seconds/tens of seconds after setup is a tooling target, not a claimed runtime. |
| Short exact sequence | A promising implementation; preserve pans/reversals/strips and separate initial residency. |
| Sustained dense no-water | Short result is useful and correct; expand to 24 then 64 independently acting units and >=100 camera changes for navigation tails. |
| Full busy/lifecycle | Dense navigation passes; add effects, zooms, distant/evicted destinations, edits, visibility, wrap, cancellation, reset and memory pressure. |
| Native presentation | A coherent standalone feature passes; strategic input-to-visible, overlay/picking and unit/UI checks with required user authorization. |

Keep 100 ms then 33 ms as resident navigation targets; caller bookkeeping has a
separate 2 ms p95 target. Native 30 Hz requires actual presented-frame evidence,
normally 1,000 frames. Report sample count and variation; sparse samples do not
support p99 claims. Preparation time and warm-up are always reported separately.
Preserve at least 512 MiB sampled largest contiguous VA, recording transient
sampling limits, total free VA, allocation failure, eviction and pending work.

Classify failures: rendering correctness, performance, capacity, measurement
invalidity or environment/transport. Fix a harness/path/locking failure once at
its source; it is not a renderer performance rejection. After a transport timeout,
confirm the child has exited before retrying. One slow case must not silently
trigger a larger matrix. Honor per-batch time limits, stop safely, and report
incomplete coverage. Do not kill the game or unrelated VM processes.

## Compact result contract

Extend existing JSON receipts with these fields rather than another status diary:

- Identity: source/binary/assets, fixture/config/request digest, VM/adapter context,
  instrumentation mode, session/case ID and evidence paths (local details ignored).
- Initial state: cold/resident policy, preparation time, retained working set,
  reset coverage and whether completed destination images were excluded.
- Latency: caller, capture, worker phases, GPU validity, readback wait, composition,
  total correct-result time, sample count, median/tails when supported and variation.
- Continuity: publication count/age, pending/backlog, cancelled work and useful
  prepared content that survived supersession; unit action/phase coverage.
- Correctness/capacity: independent comparison, ownership, fallback/recovery,
  memory observations, missing measurements and incomplete scenarios.
- Decision: targeted benefit, observed whole-workload benefit, retain/reject/
  inconclusive, architectural component/owner, capability gained, current dominant
  cost and exactly one next unresolved question. Before another experiment, state
  whether its outcomes can change the implementation decision; stop if they cannot.

Update the short current status in the retained plan. A failed experiment may
close a hypothesis but does not pass a gameplay target. A new benchmark runner
improves iteration capability, not rendering performance. No new measured claims
are made by this groundwork.
