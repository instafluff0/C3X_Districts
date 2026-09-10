# Retained renderer: plan and Spark handoff

Planning brief for GPT-5.3 Codex Spark. This is a proposed sequence of bounded
experiments, not evidence that the implementation or performance targets pass.

## Summary

The supplied analysis argues that further viewport-cache tuning has diminishing
returns. Cold terrain construction, first-use unit poses, repeated scene assembly,
CPU draw submission and synchronous GPU readback remain expensive. Its reported
live measurements include 36.6 ms median unchanged-camera rendering, approximately
119 ms between callbacks, 51 ms cold poses versus 0.5 ms hits, and 12.2 seconds for
initial terrain geometry. These describe historical workloads, not a fresh baseline.

The proposed architecture preserves Civ III's existing draw sequence and makes
C3X retained internally. Civ III still owns state, tile traversal, authoritative
anchors, camera, visibility, actions and native overlays. C3X reuses prepared
structure, poses and exact current-view pixels behind those callbacks.

The later clarification in the analysis governs the plan: initially preserve
capture and compare its complete records. Do not skip native traversal, publish
old-camera terrain beneath current overlays, or introduce another presenter.
Atomic displayed-camera ownership and asynchronous camera refinement are deferred.

## Overall goal

Reduce renderer-induced stalls during stationary play, scrolling, zoom and distant
jumps by retaining reusable world structure and prepared animation, while preserving
current-camera correctness and existing Civ III ownership. Establish benefits with
standalone production-renderer replay before changing the injected bridge.

## Plan

| Step | Work | Evidence required before proceeding |
| --- | --- | --- |
| 1. Baseline and perfect preparation | Extend the existing busy-session harness with explicit baseline and oracle modes. Prepare all required structural inputs and exact poses before oracle timing. | Identical deterministic requests, independently verified pixels, preparation cost, coverage, phase timings and memory. Identify the remaining bottleneck. |
| 2. Exact same-view reuse | After authoritative capture, recognize an identical map view and return its retained bitmap. Keep unit and map animation dependencies separate. | Repeated unchanged map requests perform zero geometry construction, GPU submission and readback. Mutation witnesses invalidate the correct content. |
| 3. Preparation that survives cancellation | Use bounded immutable jobs keyed by content. Prepare nearby structure and likely next unit/resource frames; retain valid completed entries after camera supersession. | Causal scheduling uses only information available so far. Cancellation preserves reusable work; unexpected actions remain immediately correct. Report misses, useful commits, latency and memory. |
| 4. World-space regional batching | Prototype one terrain subset in regional material/layer buffers shared across widths 128/160/192. Consider 16×16 or 32×32 regions as experiments. | Exact cold-render parity, fewer draw calls, lower submission time, measured compilation and memory costs. Expand only if benefit justifies complexity. |
| 5. Readback experiments | Compare immediate Map, staging rings with nonblocking polling, GDI-compatible BGRA surfaces and dirty transfers in the standalone native harness. | Measure total completion and wait time. Discard unhelpful approaches. Delayed animation results must retain the correct camera and scene identity. |
| 6. Narrow integration checkpoint | Integrate only proven changes at the existing map composition boundary. | Preserve overlay/picking alignment, clipping, wrapping, action ownership, redraw and fallback behavior. Live cadence requires a separately authorized game check. |

Start with step 1. If ideally prepared rendering remains slow, prioritize the
measured submission/readback bottleneck before building a broad preparation
scheduler. A failed performance hypothesis is a useful result when the experiment
is valid; it is not permission to hide missing coverage or change visual fidelity.

## Ready-to-paste first task for GPT-5.3 Codex Spark

Implement and evaluate **step 1 only** of
`Renderer/docs/retained_renderer_spark_plan.md`.

**Goal:** produce a reproducible baseline-versus-perfect-preparation experiment
that determines how much of busy-map latency can be removed by preparing terrain
structure and unit poses before presentation. Finish with a measured recommendation
for the next bounded experiment. Do not implement the entire architecture now.

Read AGENTS.md, Renderer/README.md, Renderer/lab/README.md, the relevant entries in
Renderer/lab/catalog.json, and Renderer/docs/renderer_workstreams.md. Then inspect
Renderer/docs/busy_navigation_session.md,
Renderer/docs/live_usage_findings_20260909.md,
Renderer/docs/native_async_presentation_audit.md and the current navigation handoff.
The current checkout is authoritative; verify existing functionality before adding
anything. Historical handoffs do not independently authorize staging or game use.

1. Inspect the busy-session implementation, renderer preparation/cache APIs and
   existing telemetry. Extend the existing harness rather than building a second
   simulation. Keep code, tests, output and notes under Renderer/.
2. Define explicit baseline and perfect-preparation modes. Baseline starts cold,
   without hidden warm-up. Oracle may inspect the complete deterministic request
   sequence and prepare all required structural content and poses before timing.
   Keep those preparation costs and memory in the report. Do not obtain an oracle
   speedup by replaying cached final viewport images or skipping required draws.
3. Verify actual preparation coverage and timed misses. If existing APIs cannot
   prepare a category, report that limitation and call the result partial. If the
   full working set does not fit the safe budget, report eviction and incomplete
   coverage rather than silently expanding memory or calling it perfect.
4. Replay identical ordered immutable tile/anchor/unit requests for deterministic
   comparisons. Preserve timestamps, visibility and actions. Separately run the
   same authored wall-clock input schedule to measure queued/coalesced work;
   completion-dependent streams may differ and must not be called identical.
5. Exercise cold startup, idle animation, scroll/reversal, widths 128/160/192,
   distant jumps and return travel, first at 24 and then 64 units per region.
   Report actual visible counts and incomplete phases. Use waves/reflections and
   the existing workload settings consistently across modes.
6. Report preparation separately from timed playback. Split capture/assembly,
   geometry, submission, readback wait, unit-pose hits/misses and total completion
   where instrumentation supports them. Include sample counts, median/p95/max,
   memory/cache pressure, total free VA and largest contiguous free region.
   Do not describe the entire readback interval as pure GPU execution time.
7. Compare representative results against independently cold-rendered identical
   inputs after timed playback, avoiding warm-up contamination. Require exact
   pixels and semantic ownership/anchor/phase checks. Preserve existing meaningful
   invalidation, scrolling, wrapping, compositing and action tests.
8. Run focused tests and affected current-code category checks through
   Renderer/renderer.py. Native D3D verification uses the Windows 11 VM and an
   isolated candidate. Run timed GPU workloads serially and keep evidence writes
   outside measured render intervals. Record exact inputs and binary identities.
9. Produce a concise results note with reproduction commands, coverage, comparison
   table, correctness results, limitations and one recommended next step. Update
   this plan's status briefly without replacing historical source evidence.

**Constraints:** Preserve native capture and ordering. Workers receive immutable
C3X-owned inputs and never use live game pointers, canvases or native functions.
Topology authorizes structural preparation only; appearance/visibility requires
complete authoritative capture. Maintain bounded memory, targeting at least
512 MiB contiguous free VA while reporting transient sampling limitations. Preserve
generic runtime asset formats, licensed local assets and existing failure behavior.
Do not edit injected_code.c, C3X.h, ep.c, patch tables or reference headers for this
first experiment. Do not stage a DLL, install, launch Civ III, replace reference
images or begin deferred wonders/Districts work. Do not request manual screenshots
for this standalone experiment. Scan touched files for personal or sensitive data.

**Definition of done:** executable comparison modes; meaningful focused checks;
reproducible native results with honest preparation/workload coverage; exact
independent comparisons; and a supported next-step recommendation. Prepared
distant-view geometry below 100 ms is a hypothesis to evaluate, not a promised
result. Standalone throughput is not native presented FPS. If native verification
is unavailable, finish independent implementation and tests, and explicitly leave
the native measurement requirement pending.
