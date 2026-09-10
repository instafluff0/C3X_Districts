# Autonomous renderer execution contract

This is the operating contract for an agent told to continue renderer
optimization autonomously. It supplements `AGENTS.md`; it does not weaken the
ownership, licensing, visual-acceptance or staging rules there.

## Default authority

The current checkout, current production source and newest evidence are the
authority. Git history and handoff documents provide context, not an excuse to
restore retired approaches or repeat superseded experiments. Preserve existing
user changes and inspect `git status` before editing.

An agent may choose and implement renderer, test-harness, telemetry and
documentation changes, build isolated candidates, and run focused or broad
standalone verification without pausing for approval. It must pause only for
actions that change external or user-facing state, including staging, install,
launching Civ III, changing injected hooks or patch tables, accepting visual
appearance, replacing references, or materially expanding the requested scope.

## First move after “go”

1. Read `Renderer/README.md`, `Renderer/lab/README.md`,
   `Renderer/native/render_core/README.md`, this contract, the latest navigation
   continuation, and the current retained-renderer plan.
2. Check the worktree, identify the newest valid evidence, and reproduce one
   narrow current-code measurement before designing a broad solution.
3. State the intended in-game benefit, the measured bottleneck, the experiment's
   pass/fail gate, its memory limit, and its rollback or removal path.
4. Implement the smallest architectural change that can test that hypothesis;
   do not create a parallel simulator or a cache tier without a causal target.
5. Record exact inputs, binary identity, workload coverage, parity/ownership
   results, memory observations and the next decision in the current plan.

Do not ask the user to choose between ordinary implementation steps. Ask only
when the next action needs one of the external approvals listed above or when
the evidence shows that the requested goal itself is materially ambiguous.

## Convergence toward the real game workload

Short experiments are diagnostic gates, not the destination. Every optimization
must move through this ladder and retain the previous gates as regressions:

1. **Focused cause gate:** isolate one layer, such as static terrain,
   city/improvement composition, unit poses or a wave region. Require exact
   output and a causal phase measurement.
2. **Resident dense-map gate:** run a sustained no-water scene with dense
   cities, infrastructure, resources and relief while the camera scrolls,
   reverses and returns. Include 24 and 64 visible units with independent idle,
   worker and combat timelines. No completed-map reuse is allowed to hide work.
3. **Full busy-session gate:** add waves, reflections, animated resources,
   zoom changes, distant jumps, local scrolling after jumps, reversals, combat,
   interruption, selection and visibility changes. Preserve queued discrete input
   and report missed/coalesced continuous input rather than silently dropping it.
4. **Pressure and lifecycle gate:** repeat after cold start, forced eviction,
   local edits, visibility changes, wrap seams, save/load or scenario reset,
   device recovery and memory pressure. Track preparation, backlog, cancellations,
   largest contiguous address space and deferred resource release.
5. **Native presentation gate:** measure actual Civ III input-to-visible-result,
   presented frame intervals, overlay/picking alignment and unit/UI ownership.
   Standalone completion time cannot substitute for this gate.

Passing a focused gate only unlocks the next rung. It never establishes the
busy-game result. After each meaningful optimization, run at least one broader
convergence workload and state whether the change improves the full workload,
only a subcase, or merely removes a diagnostic cost.

## Active priority order

The project is optimizing for a busy Civ III experience: dense map objects,
rapid scroll/reversal, zoom, distant jumps, independent unit movement and
combat, animation, visibility changes and bounded memory. Optimize in this
order unless fresh measurements clearly change it:

1. **Resident interactive navigation.** Build the retained static-front path:
   translate/copy covered pixels, render only newly exposed strips, and redraw
   only dirty city/improvement/resource bounds. Keep static terrain, animated
   overlays and units as independent ownership/invalidation layers. First gate
   a dense no-water resident scroll below 100 ms, then below 33 ms, with exact
   pixels and ownership, zero fallback/recovery and no completed-map reuse.
2. **Presentation responsiveness.** Complete the latest-exact asynchronous
   publication path at the existing Civ III map boundary. Keep submit/poll
   bookkeeping below 2 ms p95, never display an old-camera image beneath current
   overlays or picking, and preserve native unit/UI ownership. A responsive
   render entry is not the same as a fast camera result.
3. **Cold and distant preparation.** Prepare camera-independent structure and
   complete appearance regions incrementally. Use versioned compact regional
   storage only when capture, visibility and invalidation authority is complete.
   Measure preparation time, disk footprint, first-use latency and eviction;
   topology alone does not authorize cities, resources or visibility-dependent
   appearance.
4. **Dynamic unit/action workload.** Preserve Civ III's action director and
   independent phases. Exercise movement, reversals, attack, fortify, idle,
   selection, interruption and visibility changes at realistic density without
   synchronizing units merely to improve cache hits or allowing unit work to
   starve the map.
5. **Water and reflections.** Return to wave/reflection throughput after the
   no-water static/object path meets its gate. Their depth-aware ownership must
   remain exact; a transparent shortcut or unrelated full-view approximation is
   not a valid optimization.

## Anti-rabbit-hole rules

- A high cache-hit rate with slow frames means the cache is not the bottleneck.
  Stop tuning keys, LRU policy, bounds, query proofs or cache capacity when
  misses are rare and CPU composition, submission, readback or presentation is
  dominant.
- Do not increase a memory tier to make a benchmark pass unless the working-set
  benefit, address-space headroom and eviction behavior are measured. Civ III is
  a 32-bit process; a larger cache can convert latency into instability.
- Do not call a cache-hit, stationary, waves-off, unit-warm or standalone
  completion result a navigation or gameplay pass. Keep resident scroll,
  distant jump, first-use, repeated-use, animation, combat and native
  input-to-display measurements separate.
- Do not run a large matrix before a short gate identifies a useful direction.
  A failed hypothesis is valuable; a repeated matrix that cannot change the
  next decision is not progress.
- Do not hide latency with stale-camera terrain, incomplete ownership,
  topology-only appearance, dropped actions, synchronized animation or silent
  native fallback.
- Prefer removal of a failed experiment over preserving another opt-in branch.
  Keep a diagnostic only when it is reproducible, bounded and still informs a
  live decision.

## Evidence gates for every meaningful change

Each experiment must answer:

- What user-visible stall or failure does this target?
- Which phase is expected to improve, and by how much?
- What exact workload proves or rejects it?
- Are pixels, anchors, ownership, visibility, wrapping, action phase and
  current-camera identity exact?
- What happens under edits, cancellation, eviction, reset and allocation
  pressure?
- Does it improve the actual game path, or only an isolated helper?
- Does it improve the next broader rung of the convergence ladder, or merely
  make the focused fixture faster?

Use exact independent comparisons for correctness. Report median, p95, p99,
maximum, sample count, backlog or dropped work, preparation separately from
timed playback, and CPU/GPU/readback endpoints precisely. For continuous work,
prefer 1,000 presented frames; for discrete navigation, use at least 100 actual
camera changes when a distribution is claimed. If native presentation is not
measured, say so explicitly.

## Big-picture progress reporting

Every user-facing progress update and final report must include a compact
project-level status, even when the work is a narrow code change:

- **Goal:** which part of realistic Civ III play this work targets;
- **Measured change:** the before/after result and workload coverage;
- **Current dominant bottleneck:** what still prevents busy-map viability;
- **Convergence position:** which ladder rung passed, which remains unmet, and
  whether the worst-case workload was actually exercised;
- **Next action:** the single highest-value step and its stop condition.

Do not report cache hits, unit warm-up, a waves-off subcase or a standalone
worker completion as unqualified progress toward the full-game goal. State the
qualification in the first sentence of the update.

## Completion standard

An autonomous continuation is complete only when it either reaches a measured
gate or leaves a reproducible, evidence-backed next step. The final report must
separate achieved targets, unmet targets, unsupported workloads, memory limits,
environment blockers and external actions still requiring user approval. Never
describe a standalone candidate as live-game ready merely because its cache
tests or image comparisons pass.
