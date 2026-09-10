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

## Synchronous-boundary ambient gate — 2026-09-10

The DLL now has an opt-in compatibility policy behind the existing synchronous
render ABI; the injected caller and scene ownership are unchanged. A render call
may return the immutable last exact bitmap only when the current frame, ordered
tile payload and topology payload match byte-for-byte after excluding the
presentation clock, native-unit animation count and dirty rectangle. Any camera,
visibility, ownership, topology, environment or scene change cancels/takes over
the background work and uses the existing exact synchronous path. Ambient work
is single-flight and drops intermediate clock ticks rather than building a
backlog. Default behavior is unchanged.

At 2240x1192 with retained world caching and the bounded 256 MiB pose option, the
eight-unit gate (five frozen, selected, worker and combat) passed three stationary
ticks at 3.376–3.984 ms for the complete unit plane and 1.025 ms maximum per unit.
The existing synchronous map entry returned in 0.520–0.877 ms while retaining
the prior exact image. Fresh ambient images differed from that front and matched
independent synchronous pixel and ownership references exactly after 78–125 ms.
The in-flight translated-camera takeover returned only its new exact reference
after 784.048 ms. Fallback and device recovery stayed zero and minimum contiguous
free VA was 1,781.1 MiB.

The single crowded repeat with 24 units (21 frozen plus the same three active)
also passed: 9.813–10.021 ms for all units, 1.184 ms maximum per unit,
0.545–1.051 ms maximum stationary render-entry time, and exact ambient completion
in 94–125 ms. Its changed camera was exact after 721.890 ms, with zero fallback,
zero recovery and 1,836.0 MiB minimum contiguous free VA. An intentionally
minimal-cache diagnostic was not accepted: its 8 MiB pose cache retained only
about 26 full-size poses, causing active pose misses and 409–502 ms unit planes;
the retained-world profile reduced the separately isolated map completion from
about 296–312 ms to the measured range above. This confirms that bounded memory
retention is part of the viable architecture, not an optional micro-optimization.

The rolling x86 candidate again builds with `/O2 /W4 /WX`. Fifteen focused
publication, animation-retention, unit-animation, pose-cache and shadow tests
emitted no assertion failure; ten were subsequently reported as errors only
because Windows kept their completed temporary `contract.exe` files locked, and
three optional tests were skipped. The renderer-only animation integration entry
point was retried with its required Python packages and stopped before compilation
at the pre-existing generated-source guard for
`Renderer/native/city_fidelity/shader-provenance.json`; it was not altered and no
formal category Integration pass is claimed.

**Selected next experiment:** a short 15 Hz cadence soak through this same
synchronous ABI: 30 stationary ticks with 24 realistic units, accepting only
monotonic exact completed ticks, measuring dropped/coalesced ticks and UI-call
p95, then one visibility mutation and one scroll. Gate on zero wrong-view or
wrong-ownership publication, zero fallback/recovery, unit-plane p95 below 16 ms,
render-entry p95 below 2 ms and at least 512 MiB contiguous VA. If it passes,
make the bounded retained/pose policy a documented renderer configuration and
move to a user-authorized live-game evaluation; guarded wave batching remains
the next throughput optimization rather than a presentation blocker.

## Realistic busy-idle cadence and navigation — 2026-09-10

The cadence witness progressed incrementally from 2 seconds to 10 seconds and
then 60 seconds at the actual 15 Hz caller rate. The 2240x1192 dense scene used
frequent cities, roads/rails, mines/irrigation, animated resources, waves and
reflections plus 24 units: 21 frozen ordinary units, one selected, one working
and one in combat. The full minute completed all 900 caller ticks and delivered
445 fresh atomic map publications (7.42 Hz). Synchronous render-entry p95 was
1.720 ms and the complete 24-unit-plane p95 was 12.074 ms. Rare scheduling
outliers reached 22.476 ms and 36.615 ms respectively, without an accumulating
slowdown. Available virtual memory changed by +0.6 MiB, the minimum contiguous
free region was 1,744.4 MiB, and fallback and device recovery remained zero.
The preceding exact-reference checks again matched all four pixel/ownership
publications and rejected the old camera. This is a standalone boundary pass,
not yet a live Civ III presentation pass.

Navigation is not yet viable for normal play. In the same dense retained-world
profile, a four-tile overlapping move took 1,261.780 ms. First visits to distant
map areas took 4,073.237–8,949.875 ms while constructing 490–940 visible tile
records. Exact warm revisits still took 1,355.235–2,665.543 ms even with
1,042–1,057 of 1,057 geometries reused. Fourteen smaller retained scroll steps
took 745.284–1,814.641 ms; they reused about 2.24–2.51 million of 2.67 million
pixels but continued to pay substantial draw/readback and unreported dense
object/composition time. All warm revisit pixels matched their first-visit
references, every step had zero fallback/recovery, and contiguous VA remained
well above 512 MiB. Correctness and residency therefore pass, but navigation
latency fails by roughly one to two orders of magnitude.

**Selected next experiment:** one exact 64-pixel retained scroll at 2240x1192,
run as a short serial category ablation: static terrain; cities/improvements;
resources; then waves/reflections. Record total, geometry, draw, readback,
resource/city composition and pixel-reuse time for each without writing frame
sets. This will identify the first layer that pushes a resident scroll over the
interactive budget. Optimize that layer using a translated immutable front plus
newly exposed strips and bounded asynchronous refinement; do not attempt another
large navigation matrix until a single resident scroll is below 100 ms, then
below 33 ms. Distant jumps can initially use an exact low-detail publication
only if ownership and current-camera coverage are proven; otherwise they remain
blocking until neighboring/world-region preparation makes them fast.

## Incremental resident-scroll ablation — 2026-09-10

The first targeted scroll used a four-map-column move (approximately two or
more visible isometric tile widths) from the same 2240x1192 starting view. Each
case rendered one exact current-camera result with retained world caching,
recorded no frame-set images, and kept the same ownership/fallback/recovery
checks. With waves and reflections explicitly disabled, terrain-only took
207.379 ms total (201.793 ms CPU, 85.009 ms geometry, 16.065 ms draw,
89.655 ms readback, 2,364,928 reused pixels). Animated resources raised this
to 303.643 ms (297.555 ms CPU). Waves/reflections raised it to 660.582 ms
(653.380 ms CPU). The synthetic city/improvement case took 1,081.191 ms
(1,075.952 ms CPU, 578.285 ms geometry, 213.992 ms draw, 272.383 ms
readback, 2,037,248 reused pixels). Every case returned zero fallback and zero
recovery with a largest free region above 2,098 MiB.

The trace explains the blocker: resident terrain is already about 202 ms of
CPU-side frame/composition work; resources add about 96 ms, waves about 355
ms, and cities about 874 ms. The broad navigation matrix remains byte-exact
on warm revisits, but its 1.35–2.67 second warm latency is consistent with
this ablation and is not acceptable for normal wheel/key movement. Animation
content is not the first target; the static CPU path must be shortened first,
then city composition.

**Selected next experiment:** prototype a bounded translated-front plus
newly-exposed-strip publication for this two-column move, preserving exact
current-camera ownership while the strip renders asynchronously. Measure the
existing CPU/composition counter separately from the translated copy and strip
render. Re-run the four categories only after terrain-only latency is materially
lower, then isolate city composition as the second stage. Keep ambient
refinement single-flight behind the retained front; do not hide a camera change
with an old view or relax ownership checks.

## Realistic scroll sequence witness — 2026-09-10

The next short witness exercised the retained path as an actual input stream:
one-column, two-column, four-column and eight-column moves, reversal, return to
the origin, then the same pattern in the opposite direction. It ran at
2240x1192 with the dense world fixture, retained world regions and no frame-set
reuse. All eight unique cameras and every warm revisit were byte-exact, passed
ownership, and reported zero fallback and zero device recovery. The largest
contiguous free region stayed above 1.69 GiB (well above the 512 MiB floor).

The first visits built 23–94 newly visible geometry records as the jump grew.
Warm returns built zero records and submitted essentially no draw/readback work,
with 2.36–2.59 million pixels reused. They still measured 145.6–770.6 ms of
CPU-side work, so overlap/raster reuse is real but not yet an interactive
scrolling solution. The dominant remaining cost is full CPU composition and
object/material preparation repeated for a camera whose retained pixels already
cover most of the viewport.

**Selected next quick gate:** instrument and isolate that CPU work, then run a
benchmark-only translated-front plus newly exposed-strip composition that copies
the retained overlap and prepares only the strip. Require exact current-camera
pixels/ownership and measure copy, strip preparation, and composition separately.
Do not add a larger cache or another long navigation matrix until this resident
path is below 100 ms (then 33 ms); nearby world-region preparation remains the
right tier for the next uncached scroll once the resident path is fast.

The profiling rerun identifies the first concrete target. On warm returns with
zero geometry builds and zero draw/readback work, animation composition still
reported roughly 102–646 ms per frame. Its subphases included backdrop-submit
about 0–324 ms, animated-submit about 29–197 ms, and GPU readback waits about
61–498 ms; the visible animation set was about 60–64 resources/waves. This is
why a resident bitmap can be correct yet still feel slow: the animated overlay
pipeline is being recomposed even when the static overlap is already retained.
The next gate should therefore split static-front publication from an
independently dirty animated overlay (continuous worker actions stay live;
frozen units do not force redraw), then verify that a camera move with only a
small animated dirty region meets the 33 ms budget.

The static control confirms the direction: with waves/resources/objects off,
warm returns in the same sequence were 27.9–30.0 ms with no CPU, geometry, draw,
or readback work, and remained byte-exact. First visits were 128–276 ms as the
new strip grew. One reverse cold step missed the retained raster and spiked to
1,332 ms, so the nearby-region path still needs a bounded directional halo for
reversals. This is now a focused overlay problem, not evidence that the basic
translated front is infeasible.

## Animated-overlay translation proof — 2026-09-10

The harness now also compares each new frame's overlap with a pixel-translated
copy of the immediately preceding frame. The static control produced zero
overlap mismatches for every move and reversal. The dense mixed scene produced
752 and 725 mismatched pixels on the first one- and two-column moves, then only
2–12 mismatches on the remaining moves and returns. These differences are
small but real; translating the entire animated layer would therefore be an
incorrect shortcut.

The implementation contract is consequently: translate the immutable static
front; invalidate animated/object bounds (with a conservative guard); redraw
the newly exposed strip and those dirty overlay regions; publish only after the
current-camera ownership and exact-pixel checks pass. Continuous worker/action
animation remains eligible for its own dirty region, while frozen units do not
force a full-map redraw. This keeps the modern retained-frame model without
masking camera changes behind an old image.

## Realistic worker-idle control — 2026-09-10

The standalone idle gate used the normal 2240x1192 map with ten ambient
resources and eight units for 30 pose steps after a ten-frame warmup. Six units
remained idle; one was selected, one directed, and one was working, matching the
game's action ownership rather than animating every unit continuously. It passed
all 30 frames with zero fallback/recovery and a minimum contiguous free region
of 1.93 GiB. After warmup, unit drawing was 3.4–6.4 ms, while the map/ambient
plane remained 78–94 ms per step. The worker-specific unit cost is therefore
already small; the ambient map compositor is the current idle-frame blocker.

The existing animation readback-atlas and dependency-backed backdrop switches
were also exercised on the mixed scroll sequence. Both preserved exact pixels
and ownership but did not materially reduce the 100–700 ms CPU range; the cost
is pose/overlay submission rather than simply packing a large readback.

The worker-realistic asynchronous boundary then delivered 74 fresh publications
in each 10-second run (7.40 Hz), with exact stationary/camera-change checks,
zero fallback/recovery, and 8-unit pose sets at 3.4–5.5 ms. Render-entry p95
was 2.28 ms without the atlas and 2.01 ms with it, narrowly missing the
current 2.0 ms standalone threshold. This is functionally responsive but not a
timing pass yet; the remaining margin is scheduling/overlay work, not unit pose
generation.

The next implementation should preserve this separation: action-director unit
poses update independently, while the map front and ambient overlays use their
own dirty cadence. A one-minute pass at the real 15 Hz caller rate should be
the acceptance gate after that compositor split, with p95 map-plus-unit work
under 33 ms and no growth in memory or pending work.

The one-minute worker-realistic async run completed 900 caller ticks and 445
fresh ambient publications (7.42 Hz). Exact stationary and camera-change
publications held, fallback/recovery stayed zero, contiguous VA stayed above
1.75 GiB, and unit-set p95 was 7.84 ms. The strict async render-entry target
still missed at 2.299 ms p95 (47.278 ms maximum), so this is a functional
responsiveness pass but not yet a timing pass. The next optimization should
reduce worker lock/scheduling and ambient publication variance before adding
more cache tiers; the current UI-facing unit path is not the limiting cost.

The resource-only control (same ten animated resources and eight realistic
units, waves disabled) delivered 149/150 fresh publications in ten seconds
(14.90 Hz), with exact boundary/camera checks, zero fallback/recovery, and unit
set p95 4.74 ms. Render-entry p95 was 4.91 ms, so the strict caller threshold
still needs tuning, but the cadence result isolates the throughput loss: the
full mixed scene's 7.42 Hz is primarily the wave/reflection layer. The next
implementation gate is therefore a retained wave/reflection overlay with
bounded dirty regions and asynchronous refinement; resources and worker poses
should remain independent layers rather than being folded into that cache.

The existing bounded block-clip/post and animation-atlas switches were then
combined with the worker-realistic mixed run. They preserved exact output and
ownership but remained at 7.40 Hz (10-second sample), so the wave/reflection
throughput problem is not solved by readback packing or post-guard clipping.

Finally, disabling only reflections while leaving waves enabled also remained
at 7.40 Hz (10-second sample), with exact boundary/camera checks and zero
fallback/recovery. Waves alone therefore account for the cadence loss. The
selected implementation experiment is to retain immutable wave cells but
decouple their block raster/readback from the main ambient publication, so the
static/resource/unit layers can continue at the caller cadence while wave
refinement publishes independently.

## Current continuation checkpoint — 2026-09-10

The current candidate was rechecked with the realistic retained-world sequence:
all 14 offsets (including reversals and returns) remained exact, with zero
fallback/recovery and contiguous free VA above 1.8 GiB. A 10-second
resource-only ambient control delivered 149/150 fresh publications (14.90 Hz),
with zero fallback/recovery and a 1,949 MiB minimum contiguous region; its
2.203 ms render-entry p95 is a narrow caller-threshold miss, not a cadence or
correctness failure. The full mixed case remains wave-limited at about 7.4 Hz.

The renderer-only animation and shadow integration commands were retried with
the bundled Python runtime, but the Windows harness still fails before replay
because it invokes the space-containing repository path without quoting. No
integration pass is claimed and no source or generated provenance file was
changed.

**Next quick gate:** implement an opt-in independent wave overlay publication
using the retained immutable cells, then compare 10-second and one-minute
mixed runs. The gate is exact current-camera pixels/ownership, zero fallback or
recovery, no completed-map reuse, >=512 MiB contiguous VA throughout, and
caller render p95 below 2 ms with the resource/unit layers still at their
existing cadence. Do not add another cache tier until this overlay gate passes.

The first transparent-wave probe was rejected: it kept the same ~7 Hz mixed
cadence and diverged materially from the reference because wave shading depends
on the scene/depth backdrop. The probe was removed; the default renderer path
is unchanged. The follow-on must therefore retain depth-aware wave composition
and move only its publication/readback boundary, not simply alpha-blend waves
over the finished CPU bitmap.

A second full-target probe was also removed after an x86 D3D11 access violation
in the experimental direct-target/resolve path. It produced no valid frame and
is not evidence against the retained architecture. The next implementation must
reuse the existing `LinearTarget` ownership and resolve contract end-to-end;
raw render-target insertion into `submit_geometry` is not an acceptable shortcut.

A third opt-in probe skipped source-shadow preparation for wave-only frames. It
preserved exact boundary output but regressed fresh mixed publications to 4.5 Hz
and raised camera completion to about 895 ms. It was removed. Shadow preparation
therefore remains on the critical path until wave draws can be batched while
borrowing the existing retained depth/shadow state.

## No-water core focus — 2026-09-10

Water reflections and waves are now deliberately excluded from the active
optimization target. The retained sequence was rerun with both disabled while
keeping the dense terrain/city/improvement fixture, retained world regions and
the exact ownership checks. All eight unique cameras and every revisit were
byte-exact, with zero fallback/recovery; the largest contiguous free region was
about 1.80 GiB. First visits for one-, two-, four- and eight-column moves took
355–641 ms, while warm returns took 67–135 ms. The front is correct and
resident, but repeated CPU/object composition still makes ordinary scrolling
too slow.

The realistic no-water async soak makes the cost split unambiguous. With eight
units (one selected, one directed, one working, five frozen), the small-object
control delivered 147/150 publications in ten seconds (14.70 Hz), with exact
camera/ownership publication, zero fallback/recovery and a 1.92 GiB minimum
contiguous region. The dense city/improvement fixture delivered 78/150
(7.80 Hz) under the same conditions. Unit drawing remained 3.5–6.4 ms; the
lost cadence is static object composition, not idle unit animation or water.

A two-cycle distant navigation matrix also passed exact screenshot parity and
zero fallback/recovery. Cold destinations built 470–940 tiles and took
3.1–6.0 s; warm revisits built no geometry yet still took 0.47–1.17 s. The
largest contiguous region remained about 1.48 GiB, above the 512 MiB safety
floor. This confirms that a larger cache alone is not the answer: the renderer
needs a retained static front with dirty object bounds and a newly-exposed
strip, plus directional nearby preparation for uncached jumps.

The one-minute dense no-water soak completed all 900 caller ticks with 488
exact publications (8.13 Hz), render-entry p95 1.980 ms, zero fallback/recovery,
and a minimum contiguous free region of about 1.77 GiB. It therefore passes the
current stability gate and shows no memory creep or correctness failure under a
busy idle screen. It does not yet sustain the full 15 Hz publication cadence;
that gap is the city/improvement composition budget identified above.

**Selected next experiment:** add a benchmark-only static-object composition
split for the no-water path. Translate/copy the immutable terrain front, redraw
only the newly exposed strip, and independently invalidate city/improvement/
resource bounds. Measure object compilation, strip composition, and publication
separately; require exact pixels/ownership, no completed-map reuse, zero
fallback/recovery, and >=512 MiB contiguous VA. Keep action-driven unit poses
on their own dirty cadence so frozen units do not trigger a map redraw. Do not
add another cache tier or resume water-effect optimization until this object
path is below the 33 ms interactive budget.
