# Retained renderer program

Implementation and validation plan for bounded retained-renderer experiments.
Planned work is not evidence that implementation or performance targets pass.

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

## Step 1 experiment protocol

Evaluate **step 1 only** before expanding the architecture.

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

## Step 1 measured continuation — 2026-09-10

Commit `d9898d35` was rebuilt in
`Renderer/native/build/retained-oracle-20260910-v6` with MSVC x86
`/std:c++17 /EHsc /O2 /W4 /WX`, the standalone oracle define and an LAA preview.
The first oracle-v3 replay completed, but its evidence was rejected because four
derived city shader objects were created on first use. The identical stable-input
rerun is `oracle-replay-24-oracle-v4`; its matched cold rerun is
`oracle-replay-24-baseline-v3`, and their controlled report is
`oracle-replay-24-pair-v3-analysis.json`.

The 200-request logical streams were identical and all 11 representative images
matched byte-for-byte across modes. Each run covered eight phases, all seven
discrete events and widths 128/160/192 with 24 mixed-action units per zone, dense
objects, waves and reflections. The oracle cleared completed viewport and
publication state before timing (`final_map_cache=cleared`); it did not reuse a
completed map. Both runs reported zero fallback and zero device recovery.

| Measure | Cold baseline | Bounded oracle |
| --- | ---: | ---: |
| Completion median / p95 / max, ms | 263.438 / 1,078.159 / 17,214.213 | 533.616 / 1,195.532 / 9,745.875 |
| Map p95, ms | 575.248 | 601.454 |
| Unit plane p95, ms | 902.067 | 979.354 |
| Geometry builds / uploads | 6,168 / 667,525,252 B | 4,701 / 401,919,462 B |
| Timed geometry evictions | 0 | 7,404 |

Oracle preparation examined all 200 requests and took 163,043 ms. It admitted
6,168 geometry entries, retained 536,745,514 geometry bytes and 134,113,936 pose
bytes after the benchmark-only 512/128 MiB caps, plus 20,512,944 ground,
31,504,896 wave, 35,753,056 pose-payload and 134,217,728 shadow bytes. Preparation
was partial: timed replay still built geometry, and the bounded trace observed
145 pose misses among its final 552 recorded unit requests (the trace covered only
24/200 timed views, so it is not a complete miss count). The analyzer inferred at
least 1,265 preparation pose evictions. The oracle's p95 ratio was 0.902x versus
baseline: preparation made this bounded configuration slower, not faster.

The sampled contiguous-VA requirement failed. Largest free contiguous VA declined
from 981,139,456 bytes after preparation to 527,536,128 bytes (503.10 MiB), below
the 512 MiB floor, while sampled total free VA reached 686,759,936 bytes. Therefore
the 64-unit replay/session matrix was not run: increasing density after this stop
condition would repeat the prior bounded refill failure and risk the uncapped
oracle's post-trim crash. Preserve both earlier findings: the uncapped full oracle
retained excessive memory and crashed after trim; the first bounded oracle began
above the floor but later refilled caches and crossed it. V6 reduces retained state
but does not eliminate that refill/thrash mode.

Focused retained-oracle and busy-session contracts passed 4/4. Renderer-only
`integration animation` and `integration shadows` were both attempted after a
fresh current candidate build, but the shared Windows regression harness failed
before category replay (animation: 219 tests, 1 failure/15 errors/16 skipped;
shadows: 244 tests, 2 failures/33 errors/16 skipped). Failures were dominated by
Unix-only `fcntl`, hard-coded GCC `c++`, Windows executable/file locks and
platform-dependent generated-provenance paths; no category Integration pass is
claimed.

**Selected next experiment:** implement a benchmark-only sliding working-set
oracle that prepares and pins only the current and immediately next logical
phase/zoom, releases expired geometry and poses at phase boundaries, and reports
admission usefulness and eviction cause. Require zero timed completed-map reuse,
exact paired pixels, zero fallback/recovery, at least 512 MiB contiguous VA at
every sample, and separately zero timed geometry and pose misses for the prepared
window. This is the shortest test of whether targeted background preparation can
deliver the retained architecture's benefit inside Civ III's fixed 32-bit address
space. If one current+next window still cannot meet those conditions, move directly
to shared world-space regional batching/compact compiled storage before building
a causal scheduler; the approximately 56 ms readback p95 is secondary to the
observed 979 ms unit-plane tail and cache thrash.

## Short-gate refinement — 2026-09-10

Iteration was reduced to a 32-request replay (four samples per phase) at
1024x576. The V6 eight-unit oracle retained 468,387,436 geometry bytes and
69,913,020 pose bytes, completed all structural preparation, recorded 236/236
pose hits, zero fallback/recovery and a 1,620,377,600-byte minimum sampled
contiguous region. Its matched baseline had identical requests and all 11 images
matched byte-for-byte. Completion p95 improved from 5,757.522 to 588.549 ms
(9.78x), establishing that retained preparation is effective when the working
set fits. A preview-only fix now excludes wholly offscreen topology-halo unit
sprites; it removed a false 9+-actor fixture failure without changing renderer
output.

Civ III's native action cursor is now authoritative for every unit pose. The
renderer no longer advances asset-marked ambient unit clips from its own clock:
unselected ordinary units remain frozen, while worker, selected and directed
combat actions still advance when Civ III advances their cursor. At 2240x1192,
eight measured frames with 24 frozen units and waves fell from 419.568–463.999 ms
to 123.431–137.653 ms. Unit composition fell from 285.190–315.634 ms to
10.956–13.221 ms; the remaining map work was 111.863–124.281 ms. With waves off,
the same idle fixture measured 80.142–86.058 ms, including 68.691–72.059 ms of
map work, so waves account for roughly 45–55 ms but are not the only continuous
map cost.

The short mixed-action capacity boundary was 18 units at the former 128 MiB pose
limit. At 24 units it retained only 421 of 509 required poses and completion p95
was approximately 1,005 ms. Raising the benchmark-only pose limit to 256 MiB
retained all 509 poses in 160,905,760 bytes, reduced the same p95 to 573.187 ms,
kept timed unit work to 4.798–21.703 ms, and preserved at least 1,525,063,680
bytes of sampled contiguous VA. Both runs had zero timed geometry builds/uploads,
zero fallback/recovery and no completed-map reuse; their 32 timed output hashes
were unchanged. Focused unit-pose and retained-oracle contracts pass 5/5, and the
candidate builds cleanly with MSVC x86 `/O2 /W4 /WX` in
`retained-oracle-20260910-v9-native-cursor-pose256`.

The new realistic-idle fixture uses 21 frozen units, one native-cursor selected
idle loop, one worker task loop and one directed combat action. All eight measured
frames changed with zero fallback/recovery. First-use unit work was
17.580–144.468 ms while unseen active poses were admitted. After a complete
16-frame warmup, every traced pose was a cache hit; seven of eight unit totals
were 13.492–28.699 ms, with one 72.175 ms GDI-flush/scheduling outlier. Map work
remained dominant and variable at 169.902–274.283 ms in that warmed run.

**Selected next experiment:** prototype regional animation compositing. Retain
the static full map and update only wave regions first; repeat for the remaining
62 visible animated resources/effects. Also prepare and pin the few active unit
cycles when Civ III changes their action, leaving frozen poses resident. Gate
each short step on exact paired pixels, native cursor ownership, zero
fallback/recovery and >=512 MiB contiguous VA. Target <=16.7 ms for unchanged
idle frames and report active-region and GDI-flush costs separately. Keep the
all-mixed replay as the worst-case ceiling; do not return to the full serial
matrix until these short gates pass.

Two existing controls were rejected as the next idle solution. On the same
eight-frame frozen-unit fixture, world-aligned backdrop retention measured
125.200–152.986 ms and water-coverage culling measured 136.562–160.570 ms versus
the 123.431–137.653 ms control. Neither reduces stationary continuous-animation
cost; regional animated-layer update/compositing is therefore the selected
implementation experiment.

## Idle animation compositor experiments — 2026-09-10

A four-frame phase trace at 2240x1192 isolated the remaining stationary-map
cost. Each frame had 62 visible resource animations, 57 wave chunks and 105
dirty 128x128 regions (1,583,872 pixels). All 105 static scene-linear backdrop
lookups hit; backdrop submission cost only 0.075–0.083 ms. Pose preparation cost
9.4–10.2 ms, the 105 animation submissions cost 25.7–26.9 ms, and the blocking
GPU-to-CPU map waited 68.4–82.5 ms. Total map work was 111.5–126.9 ms. This
confirms that static retention is working and that repeated small animation
passes plus the synchronous readback dependency are now the idle bottleneck.

The first bounded implementation packed the 105 finished blocks into a
1408x1280 staging atlas instead of routing them through the 2240x1192 staging
layout. Its initial image and all three timed animation images matched the
control byte-for-byte, with zero fallback/recovery and no terrain builds or
uploads. It did not improve latency: the control map measured 142.1–145.5 ms
and the atlas measured 138.5–143.5 ms in the clean final witness, with
67.4–73.9 ms still spent waiting. The wait is therefore dominated by preceding
GPU rendering, not staging texture area. The exact atlas remains useful
infrastructure, but it is not itself a performance win.

A deliberately short full-view batching probe reduced animation submission
from 26–33 ms to 2–6 ms, demonstrating the expected order-of-magnitude batching
benefit. It was rejected for correctness: 46,221–47,133 of 2,670,080 pixels
(about 1.7 percent) differed from the guarded-block reference, with channel
errors up to 216, and an attempted guarded-finish variant froze the published
animation. The cause is architectural: current retained color/depth and city
glow are defined in independent guarded block projections, so their depth
values cannot be pasted into one full-view depth space. Failed full-view code
was removed rather than retained behind a switch.

**Selected next experiment:** build a small true guarded animation atlas, first
for a fixed subset of wave blocks. Each 136x136 atlas cell will keep its own
local projection, 4-pixel city-glow guard, scene-linear MSAA color and depth.
Attach an atlas-cell transform to each wave draw and render all selected cells
in one submission, then use the already-proven packed readback. Gate the first
four-cell version on exact pixels, zero fallback/recovery and at least 512 MiB
contiguous VA; expand geometrically (4, 16, then all visible wave cells) only
while those gates pass. Once wave batching is exact, reuse the same path for
resource bodies and overlap readback with the next 15 Hz animation tick through
a bounded staging ring. Frozen units remain cached; only native-directed worker,
selection and combat cursors prepare new poses. This preserves the existing
Civ III bitmap boundary while adopting the batching and pipelining used by
modern renderers.

The first four-block shortcut reused the existing 264x264 city-fidelity region
target. It completed with zero fallback/recovery, no timed geometry build or
upload, and approximately 1.82 GiB minimum contiguous VA, but failed the pixel
gate. Differences were confined to the selected region: 1,658–1,816 pixels per
frame, maximum channel error 88–91 and mean absolute channel error about 0.015.
It reduced 105 regions to 102 but did not improve time (map work
140.4–150.6 ms; animation submission 25.9–32.1 ms). A 264-pixel regional
projection is therefore not equivalent to four independent 136-pixel guarded
projections. The shortcut was removed. The next implementation must preserve
each cell's 136-pixel inverse size and depth space, using an explicit per-cell
transform/viewport in the wave shader before attempting larger batches.

## Asynchronous ambient-publication gate — 2026-09-10

The smallest stationary-camera pipeline gate now uses the renderer's existing
versioned camera worker and atomic pixel/ownership publication. The consumer
retains the last exact same-camera bitmap while the next 15 Hz ambient tick is
pending; it accepts only the matching completed ticket. Three 2240x1192 ambient
ticks returned control in 0.502–8.396 ms and completed exact off-thread frames
in 80.493–108.958 ms. Each new frame differed from the retained bitmap and
matched its independently synchronous reference byte-for-byte, proving that the
test did not publish a completed-map hit. Static geometry was reused as intended,
with zero timed geometry builds/uploads, zero fallback and zero recovery.

A deliberately in-flight request was then superseded by a translated camera.
The newer capture returned control in 0.518 ms, the old ticket reported
`SUPERSEDED`, and the new ticket eventually matched its synchronous pixel and
ownership reference exactly. All four exact publications passed. The minimum
sampled contiguous free VA was 1,904.4 MiB, comfortably
above the 512 MiB floor. This is a latency-architecture pass, not a completed
native integration or 15 Hz throughput pass: the UI can remain responsive by
presenting a recent exact frame, but the worker still needs 80–109 ms to produce
each stationary ambient update and 763 ms for the cold translated view.

The strict `/O2 /W4 /WX` x86 build passed in the rolling
`animation-guarded-atlas-current` scratch directory. Focused publication,
retention and animation/shadow C++ tests emitted no assertion failure, but Windows
reported eight cleanup errors because their temporary `contract.exe` files were
still locked. The renderer-only category
integration command was also attempted and stopped before compilation by the
generated-source guard for `Renderer/native/city_fidelity/shader-provenance.json`;
no category Integration pass is claimed.

The realistic unit coexistence gate first exposed a scheduler mistake rather
than a pose-rendering cost. All measured poses were warm cache hits taking about
0.2–1.6 ms internally, but the first unit call cancelled the in-flight ambient
map and waited for its D3D cancellation boundary. Eight units therefore consumed
78.8–106.6 ms on the presentation thread. Cached posed bodies are already
independent CPU pixel publications, so they now composite directly without
taking over the map worker; cache misses retain the serialized D3D path.

With that split, the identical eight-unit mix (five frozen, one selected, one
worker and one directed combat unit) took 3.883–4.302 ms for the complete unit
plane, with a 0.903 ms maximum individual call. No stationary camera request was
cancelled. The single promised 24-unit repeat—21 frozen and the same three active
units—took 10.556–12.793 ms total, with a 1.210 ms maximum call. Its three ambient
requests returned in 0.606–4.966 ms and completed exact pixels off-thread in
83.229–106.085 ms. All four map publications matched independent references,
reported no completed-map publication reuse, fallback or recovery, rejected the
forced stale camera, and preserved at least 1,824.0 MiB contiguous VA.

**Selected next experiment:** adopt the proven latest-exact asynchronous
publication policy at the existing Civ III map boundary, without changing scene
ownership: poll and publish only matching camera/visibility epochs, keep units/UI
on the current exact front, and allow cold pose misses to use the existing safe
serialized path. First verify a stationary view with a worker plus selected unit,
then one scroll supersession and one combat action. Only after that small live-
boundary gate should the remaining serial matrix run. Background production is
now the primary architecture; guarded wave batching is a later throughput and
animation-smoothness improvement, not a prerequisite for responsive idle UI.
