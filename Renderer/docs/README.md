# Renderer documentation

## Active reading path

1. [Architecture](renderer_architecture.md): ownership, GPU-ready scene design and firm contracts.
2. [Roadmap and current status](retained_renderer_plan.md): four agreed milestones, actual gaps and the next discussion.
3. [Validation](benchmark_workflow.md): existing harnesses, measurement endpoints and meaningful checkpoints.

[Execution rules](autonomous_renderer_execution.md) are a short operating reference.
The [Lab workbench](../lab/README.md) owns category commands and visual approval.
No historical document supplies an additional queue or mandatory baseline campaign.

## Read when relevant

| Area | References |
| --- | --- |
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

Historical “current”, “next” and staging statements apply to their recorded versions,
not today's checkout. The pre-synthesis active guides are recoverable from Git at
`c1360ea9`. Preserve expensive findings and ignored inputs; documentation cleanup
is not permission to delete assets or controls.

## Keeping this usable

Architecture owns principles; roadmap owns status and sequence; validation owns
evidence rules. Update each in place rather than copying these responsibilities
into handoffs. Put detailed timings in receipts and selected historical summaries.
Remove obsolete instructions from active guides; retain only the evidence or
contract that can still affect a decision. Do not append a new status diary.
