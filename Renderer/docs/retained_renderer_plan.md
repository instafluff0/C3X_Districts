# Retained renderer program

Current status and preserved evidence for the retained renderer. Planned work
is not evidence that implementation or performance targets pass.

## Current status — timing accounting complete; amortized setup next

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

**Next task:** implement amortized session setup in the existing `biq_preview.cpp`
and build/evidence owners (workflow deliverable 2). Repeated short cases must share
verified assets/device while resetting equivalent renderer state and preparation;
record reset/warmup coverage, quick versus acceptance verification, and incremental
compilation. Stop this task when repeated cases show measured setup reduction and
an independent fresh one-shot reproduces their output. Do not expand the workload.

**Tooling phase: deliverable 1 complete; deliverables 2–3 unfinished.** Complete the
session prerequisite and automatic route/object diagnostic batch in order, then
replace the next task with the evidence-selected retained scene implementation.
Additional harness work requires a named missing measurement that changes that
implementation decision.

### Session tooling progress (deliverable 2 remains open)

Per-translation-unit include hashing and compiler/SDK stamps now reuse intact
objects. `session-incremental-seed-retry-20260912` compiled seven units in 37.345 s
wrapper time; `session-incremental-warm-20260912` compiled none in 3.108 s. Changing
only the preview compiled that unit alone. Recipe, tier, dependency and object
changes reject reuse. A long VM command failure was corrected with a short batch
transport after confirming that no compiler/linker remained running.

A bounded same-configuration process loop retains assets/device, distinguishes
process-cold, assets-loaded and explicitly warmed resident policies, and uses a
new benchmark-only reset that preserves cache budgets. `session-two-case-20260912`
passed two assets-loaded cases with identical initial/final pixels and full input
verification. Its 30.655 s wrapper time did not establish worthwhile amortization.
The four-case follow-up generated all images but failed during/after teardown and
entered Windows Error Reporting; it was dumped and only that owned process/tree
was terminated after exceeding its limit. That run is invalid performance evidence.
The wrapper now retains invalid receipts and bounds the actual child process,
including teardown; a checkpointed preview is being validated to locate the fault.
Do not mark session tooling complete or use its failed timing as a rendering win.

### Current evidence and limits

Ignored evidence under `Renderer/native/build/`:

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

**Dominant measured costs:** small-view first-exposure geometry and cold setup;
input hashing and compilation also dominate iteration wait. These are scoped
measurement findings, not dense-navigation speedup claims. The next capability
eliminates repeated setup so the fixed dense diagnostic can resolve route/object
work without another baseline campaign. Remaining process-spawn/driver setup and
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
| Iteration tooling | Timing endpoints, buffered diagnostics and wrapper/build accounting validated on short production-DLL workloads. | Persistent sessions and automatic diagnostic decisions remain unimplemented. |

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
