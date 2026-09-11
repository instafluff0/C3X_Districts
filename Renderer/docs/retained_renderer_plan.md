# Retained renderer program

Current status and preserved evidence for the retained renderer. Planned work
is not evidence that implementation or performance targets pass.

## Current status — benchmark workflow transition

The user requested groundwork for faster, systematic development while preserving
progress, and explicitly did not request restarting tests. No new benchmark,
compile, installation, staging or gameplay measurement accompanies this update.

The active engineering sequence is [the benchmark workflow](benchmark_workflow.md).
[The execution contract](autonomous_renderer_execution.md) governs continuation.
The next implementation is timing/correctness accounting in the existing harness,
then amortized session setup and one automatic route/object diagnostic batch.
Only after those bounded tooling steps should the retained route/object rendering
change resume. Do not rerun the baseline/oracle program to begin this transition.

| Capability / target | Evidence carried forward | Remaining limit |
| --- | --- | --- |
| Stationary retained front | Opt-in same-view ambient publication and independent cached-pose CPU composition are implemented; prior standalone minute-long 24-unit run reports 1.720 ms p95 render entry and 7.42 Hz fresh maps. | Still captures full native view; this is not live presentation or fast navigation. |
| Useful preparation | Existing `prefetch-guard2-replay-20260910/benchmark.log` records 176 prepared tiles before its first jump and exact return parity. | First three jumps were 707.534, 828.214 and 785.799 ms; short diagnostic, not a sustained gate. |
| Dense resident navigation | Retained overlap and bounded working sets already exist; preserve them. | Below-100 ms gate unmet; route/object composition and synchronous completion remain the selected causal target. |
| Regional geometry / zoom | Some structural sharing exists. | General regional batching shared across 128/160/192 is unfinished. |
| Native asynchronous presentation | Versioned publication and stale-ticket rejection exist in the DLL. | Native coordination, current-camera presentation and input-to-visible performance remain unverified. |
| Iteration tooling | Existing preview, evidence runner, analyzer and delayed GPU telemetry are reusable. | Persistent sessions, revised timing endpoints and automatic diagnostic decisions are specified, not implemented. |

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
