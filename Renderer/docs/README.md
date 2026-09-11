# Renderer documentation

## Start here

- [Benchmark workflow](benchmark_workflow.md): one active engineering sequence.
- [Current capability status](retained_renderer_plan.md): achieved and unmet targets.
- [Execution contract](autonomous_renderer_execution.md): scope, evidence and stopping rules.
- [Lab workbench](../lab/README.md): current category rendering and verification.
- [Storage retention](storage_retention.md): preview-first maintenance and restoration.

## Durable contracts and findings

Read these when relevant to a change, not as competing task queues:

- [Workstream ownership](renderer_workstreams.md), [visible scene](visible_scene_contract.md),
  [configuration](renderer_config_spec.md) and [patch dependency ledger](civ3_patch_dependency_ledger.md).
- [Visual fidelity playbook](visual_fidelity_playbook.md), [source-art findings](source_art_findings.md)
  and [cliff findings](coastal_cliff_findings.md), including their preserved local inputs.
- [Shared environment](environment_lighting_and_ambient_effects.md) and
  [Civ VI lighting evidence](civ6_lighting_findings.md).
- [Native presentation constraints](native_async_presentation_audit.md) and
  [navigation architecture](navigation_implementation_plan.md); implementation status
  comes from current code and the capability scorecard.
- [Natural wonders](natural_wonder_rendering.md) and
  [wonders/Districts](wonder_and_district_rendering.md): preserved deferred contracts.

The remaining category/import findings retain their existing filenames and links.
They are references, not mandatory reading before every implementation step.

## Historical evidence

- [Retained renderer experiment archive](history/retained_experiments_20260910.md).
- [Navigation continuation archive](history/navigation_continuation_20260910.md).
- [Navigation activation and rollback handoff](navigation_handoff.md).
- [Actual game findings](live_usage_findings_20260909.md) and
  [busy workload](busy_navigation_session.md).

These preserve successes, rejected hypotheses and measurement limits. Historical
binary identities and instructions do not certify a current candidate or select
the next task. Do not reproduce an old workload merely to begin a new task.
