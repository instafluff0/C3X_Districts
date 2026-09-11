# Retained renderer program

Current status and preserved evidence for the retained renderer. Planned work
is not evidence that implementation or performance targets pass.

## Current status — bounded tooling before scene/rendering implementation

The user requested groundwork for faster, systematic development while preserving
progress, and explicitly did not request restarting tests. No new benchmark,
compile, installation, staging or gameplay measurement accompanies this update.

[The architecture](renderer_architecture.md) defines the destination;
[the execution contract](autonomous_renderer_execution.md) governs continuation;
[the benchmark workflow](benchmark_workflow.md) specifies measurement, tooling
deliverables and validation. This status is the single active task record.

**Next task:** implement timing/correctness accounting in the existing harness
(workflow deliverable 1). It must separate setup/playback and correct capture
endpoints, validated by timestamp ordering and one existing short workload when
implementation is authorized. This documentation update does not run that workload.
Do not rerun the baseline/oracle program to begin the transition.

**Tooling phase: specified, not complete.** After the next task, finish amortized
session setup and the automatic route/object diagnostic batch in the workflow's
order. When all three reusable deliverables meet their specified validation,
record tooling complete here and replace the next task with the selected bounded
scene/rendering capability. Do not leave "improve the harness" as an indefinite
task. Additional tooling requires a named missing measurement/correctness check
that can change the implementation decision.

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
