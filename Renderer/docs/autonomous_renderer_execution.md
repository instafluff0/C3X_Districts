# Autonomous renderer execution contract

This contract governs renderer optimization continuation. AGENTS.md, the user's
current scope and the Lab ownership/visual-acceptance rules remain authoritative.
The active work sequence is [the benchmark workflow](benchmark_workflow.md);
current achieved/unmet capabilities are at the top of
[the retained plan](retained_renderer_plan.md). Historical handoffs preserve
context, not additional active task queues.

## Current authority and scope

The user requested groundwork for faster, systematic iteration, explicitly without
restarting tests. The groundwork is documentation only: it neither implements the
new runner nor grants a performance pass. Do not interpret reading this contract
as an instruction to start a benchmark, build or baseline reproduction.

When the user resumes implementation, proceed with the next unfinished tooling
step in the workflow. Routine Renderer edits, isolated builds and focused checks
within that request do not need repeated permission. Existing user authorization
persists; do not infer an approval barrier from this document. Staging, install,
game launch, injected integration and visual acceptance remain subject to the
actual user request and repository rules, not historical handoff authorization.

Preserve the current worktree, staged DLL, rollback, source findings, ignored packs
and reference images. Workers receive immutable C3X-owned input and never native
pointers/canvases. Keep Civ III's authoritative camera, capture, visibility,
actions, native overlays and picking. Config-off retains the native path;
custom-on map failure must not silently replay native terrain. Preserve the
separate unit fallback. Wonders and Districts remain deferred.

## First move on implementation continuation

1. Read the Renderer/Lab entry guides, relevant catalog/category, this contract,
   the workflow and the retained plan's short current status. Read deeper history
   only for the mechanism or evidence actually needed.
2. Inspect the worktree and the existing owners of the next deliverable. State
   what is already implemented, the unresolved question and the bounded change.
3. Begin timing/correctness accounting, then session setup, then the automatic
   route/object diagnostic batch, then the retained route/object implementation.
   Update the current status as each deliverable completes; do not repeat it.
4. Validate the changed deliverable at the smallest relevant scope. Do not make
   full baseline reproduction the entry fee for tooling, documentation or an
   unrelated contract fix. Do not create a second renderer or simulator.
5. Record identity, endpoints, initial state, coverage and the next decision in
   existing receipts. Update current status in place instead of appending another
   competing "next gate" to the experiment history.

## What counts as progress

Separate three outcomes: tooling capability, validated rendering behavior and
measured gameplay performance. Faster setup is useful tooling progress; a small
exact fixture is correctness progress; neither establishes dense navigation.
A rejected hypothesis is learning, not a passed performance gate.

Keep the primary rendering target: a correct dense no-water resident scroll,
using translated overlap plus newly exposed strips and dirty route/object bounds.
Require below 100 ms first, then 33 ms; preserve unit/action and ambient layers
independently. Reusing the current front is intended. Replaying whole completed
destination images must not hide work in this navigation gate.

After a useful short exact result, expand to sustained 24/64-unit dense navigation,
then effects/zoom/distant/lifecycle workloads, then a strategic native presentation
checkpoint. Keep >=512 MiB sampled contiguous address-space headroom and report
transient limits. A broader run is earned by a result or required by an affected
contract; do not run it automatically after every harness edit or failed probe.
Use the category dispatcher for required affected verification; record environment
failures without claiming a category pass.

The later architecture goals remain latest-exact native presentation, compact
complete-appearance regional preparation/batching shared across 128/160/192,
and full dynamic/effect throughput. Native caller bookkeeping has a separate
2 ms p95 target. Native 30 Hz requires actual presentation evidence, not worker
completion counts. Do not redirect current work to water or cache expansion
without causal evidence and an explicit update of the single active workflow.

## Experiment discipline

- State the targeted elapsed phase, maximum plausible whole-transition benefit,
  correctness oracle, initial residency and stop condition before an experiment.
- Use fixed inputs, equivalent resets/preparation and serial GPU workloads.
  Reuse the existing harness and controls; batch independent diagnostic cases.
- Separate setup, capture, caller waits, worker work, GPU execution/readback,
  publication age and total correct-result latency. Unknown is not zero.
- Revisit equality proves repeatability. Require independent full-redraw output,
  ownership, visibility, anchor and phase checks for an optimization's correctness.
- Use a tiny boundary/depth fixture before another full-scene projection change.
  Preserve exact output for unchanged-appearance optimizations. Do not weaken
  correctness checks to turn an incorrect shortcut into a performance pass.
- Record preparation costs and actual useful work surviving cancellation. Do not
  synchronize unit phases, drop discrete actions or hide stale-camera content.
- Stop key/LRU/index/capacity tuning when its maximum benefit cannot materially
  affect the current latency target. Capacity increases need measured working-set,
  eviction and address-space evidence, not a benchmark-only pass.
- Reject wrong or slower variants; keep a diagnostic branch only while it answers
  a live question. Close hypotheses with evidence so later agents do not rerun them.
- Use matched repeated comparisons. Report variation and sample counts; close
  results are inconclusive. Reserve long-tail claims for adequate sustained samples.
- Separate renderer, measurement, capacity and environment failures. Fix a recurring
  harness failure at its source. Check child completion before a transport retry.
- Keep quick receipts provisional; full input verification and affected checks
  remain necessary for acceptance. Never stage a benchmark binary by implication.

## Progress report and completion

Every meaningful update states the user-visible goal, changed capability or measured
before/after with workload, dominant remaining cost, strongest validated scope and
one next action with a stop condition. During groundwork, say no measurements ran.
Do not describe cache-hit, sparse, stationary, waves-off or standalone results as
an unqualified gameplay improvement.

A tooling task completes at its specified reusable deliverable, with validation
appropriate to the user's scope and unimplemented work explicit. An optimization
completes at a measured gate or a reproducible rejection with one evidence-backed
next question. Keep native integration, memory limits, unavailable evidence and
external actions explicit. Never require the user to approve ordinary next steps
already authorized by the session.
