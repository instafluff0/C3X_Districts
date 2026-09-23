# Renderer documentation

## Read first

1. [Current roadmap and status](retained_renderer_plan.md): the live work queue and unaccepted responsibilities.
2. [Renderer64 scene and motion](renderer64_scene_and_motion.md): the target process boundary, state delivery, unit-motion UX and playable cutover gate.
3. [Validation](benchmark_workflow.md): use at the integrated gate or for a focused contract, not as a separate implementation campaign.

[Deep architecture](renderer_architecture.md) explains retained world/GPU
contracts; its older presenter and timing assumptions are superseded by the
Renderer64 cutover. Read only the relevant subsystem when working on it.

The [64-bit migration plan](helper64_migration_plan.md) records gate evidence;
[Gate 2 results](helper64_gate2_results.md) records its gameplay limits. The
[cross-process surface trial](direct_surface_trial.md) preserves its graphics
proof and native-UI layering requirement.

[Execution rules](autonomous_renderer_execution.md) are a short operating reference.
The [Lab workbench](../lab/README.md) owns category commands and visual approval.
No historical document supplies an additional queue or mandatory baseline campaign.

## Read when relevant

| Area | References |
| --- | --- |
| Complete workload recording | [Ten-minute input/replay contract](recorded_renderer_workload.md) |
| Implemented visual frames | [Clock, retained composition and lifecycle](visual_frame_ownership.md) |
| Native integration | [Workstreams](renderer_workstreams.md), [visible scene](visible_scene_contract.md), [configuration](renderer_config_spec.md), [patch ledger](civ3_patch_dependency_ledger.md) |
| Visual/source contracts | [Fidelity playbook](visual_fidelity_playbook.md), [source findings](source_art_findings.md), [cliffs](coastal_cliff_findings.md) |
| Lighting/effects | [Environment contract](environment_lighting_and_ambient_effects.md), [Civ VI evidence](civ6_lighting_findings.md) |
| Deferred scope | [Natural wonders](natural_wonder_rendering.md), [wonders/Districts](wonder_and_district_rendering.md) |
| Assets and generated evidence | [Storage retention](storage_retention.md) |

Category/import notes retain their existing links. They preserve specific contracts
and findings, not alternate architecture or implementation status.

## Historical evidence — consult selectively

- [September 15 checkpoint](history/retained_renderer_checkpoints_20260915.md): tested binary, comparisons, scope and preserved controls.
- [Earlier retained checkpoints](history/retained_renderer_checkpoints_20260914.md), [completed output work](history/retained_output_completed_20260913.md), [experiment archive](history/retained_experiments_20260910.md).
- [Native camera audit](native_async_presentation_audit.md): failed handoff mechanisms, ABI/ordering evidence and earlier corrections.
- [Earlier navigation design](navigation_implementation_plan.md), [activation/rollback handoff](navigation_handoff.md), [continuation archive](history/navigation_continuation_20260910.md).
- [Game findings](live_usage_findings_20260909.md), [busy workload](busy_navigation_session.md).

Historical “current”, “next” and staging statements apply to their recorded
versions, not today's checkout. The former long roadmap remains in Git history.
Preserve expensive findings and ignored inputs; documentation cleanup is not
permission to delete assets or controls.

## Keeping this usable

Architecture owns principles; roadmap owns status and sequence; validation owns
evidence rules. Update each in place rather than copying these responsibilities
into handoffs. Put detailed timings in receipts and selected historical summaries.
Remove obsolete instructions from active guides; retain only the evidence or
contract that can still affect a decision. Do not append a new status diary.

The [tactical overlay contract](tactical_overlay_contract.md) documents selected
markers, copied native route/turn draws and the native-setting-driven grid.
