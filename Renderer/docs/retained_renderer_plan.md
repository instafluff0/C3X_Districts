# Retained renderer program

Current status and preserved evidence for the retained renderer. Planned work
is not evidence that implementation or performance targets pass.

## Current status — timing and session tooling complete; dense-map diagnosis next

Implementation resumed on 2026-09-12. The first bounded tooling deliverable is
complete: the existing preview/evidence/analyzer now separates initial preparation
from requests, measures capture before rendering, records disjoint caller/worker
endpoints, associates delayed GPU queries by renderer sequence and flags missing
coverage. Diagnostic traces are bounded and buffered until teardown; low-overhead
mode omits detailed traces. Build receipts distinguish compiler setup, DLL and
preview compilation from wrapper time. No rendering policy or cache budget changed.

[The architecture](renderer_architecture.md) defines the destination;
[the execution contract](autonomous_renderer_execution.md) governs continuation;
[the benchmark workflow](benchmark_workflow.md) specifies measurement, tooling
deliverables and validation. This status is the single active task record.

**Next task:** finish the fixed dense-map route/object diagnostic batch
(workflow deliverable 3), through the existing preview/evidence owners and category
dispatcher. Use the existing world fixture at 2240×1192, tile width 128, fixed
camera/clock, dense cities/routes/improvements/resources, waves/reflections off,
the four-column move and 14-offset reversal. Record capture counts and unchanged
budgets; distinguish cold exposure, prepared exposure and revisits. Compare the
specified causal controls with matched serial repetitions and equivalent state.
The unresolved decision is whether persistent route/object preparation or reduced
regional submission/composition removes the dominant whole-transition cost.
Treat at least 10% and 20 ms whole-transition reduction, outside repeat variation,
as useful initial evidence; diagnostics that omit pixels cannot pass correctness.
The batch entry point and controls are implemented; finish matched validation in
one stable-input interval with exclusive VM rendering. Concurrent Lab shader
edits and native previews interrupted the attempted batch, so restart its fixed
matrix against one current input identity once that work is paused or complete.
Do not retry while the competing work remains active. Stop when the batch selects or rejects that causal target, then replace this single
next task with one bounded retained-scene implementation. Do not extend tooling
without a missing measurement that could change that decision.

**Tooling phase: deliverables 1–2 complete; deliverable 3 unfinished.** The recent
640×480 runs validate measurement and setup only. Realistic dense navigation below
100 ms, independent full-redraw parity for the migration, and live presentation
remain unmet. The next workload is a standalone populated world fixture; it does
not establish saved-game capture or live Civ III performance.

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

**Dominant measured costs:** the first uncontested dense four-column sample is
dominated by animation composition and geometry preparation. Omitting route draws
showed only a 2.308 ms difference in one pair; no architectural choice is justified
until the matched preparation/pixel controls finish. Repeated setup has been
reduced. These findings neither pass the 100 ms navigation target nor certify
the subsequently edited runtime shaders. Remaining process-spawn/driver setup and
unretired GPU queries are explicitly unmeasured, not zero.

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
| Iteration tooling | Timing endpoints, buffered diagnostics, incremental compilation and persistent sessions validated on short production-DLL workloads. | The fixed dense diagnostic batch remains unfinished. |

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
