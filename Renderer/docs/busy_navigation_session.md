# Busy-map navigation session

User-requested continuation after staging the verified September 9 navigation
improvements. This extends the existing native preview and camera queue. It is
not a new presenter, simulation engine or permission to draw uncaptured objects.

## Workload

Run one continuous renderer instance with waves and reflections enabled at
2240×1192. Populate fixed world positions with cities, connected roads and
railroads, farms, mines, camps, supported resources and units from several
families. Units keep stable identities as they enter and leave the view. Report
actual visible counts after every camera change; a requested count alone is not
coverage. Exercise 24 units first, then 64 where sufficient visible sites exist.

The sequence starts with a cold view and no hidden pose warm-up:

1. Idle for ten seconds while waves, resources and independently phased units
   animate. Include first-use costs in the startup report.
2. Scroll continuously for ten seconds, reverse, and return through visited
   content. Continue unit movement, direction changes and ambient animation.
3. Cycle 128 → 160 → 192 → 160 → 128 while the map remains busy. Include first-use
   zoom costs separately from later returns.
4. Jump to several distant minimap destinations, pause long enough to observe
   completion, and scroll locally at each destination. Keep unseen, overlapping
   and previously visited destinations distinguishable.
5. Return to the initial area and idle again. Include attack, fortify, fidget and
   interruption/held-endpoint transitions throughout the session.

Do not reset caches between these phases. Do not keep every unit permanently
screen anchored or make all units change action together. Preserve native action
cursors and identity-based ambient offsets. A view with fewer eligible units
must report that fact rather than silently claiming the requested density.

## Timing and correctness

Maintain two explicitly separate modes: a deterministic script for repeatable
pixel/ownership comparisons, and wall-clock playback for responsiveness. The
latter advances clocks by actual elapsed time, records intended and dispatched
input times, and reports skipped/coalesced input and delayed completions. A slow
render must not slow the authored world clock or silently remove missed updates.

Report cold startup, initial idle, scrolling, first/repeated zooms, distant jumps,
return travel and final idle separately. Include total completion, capture,
geometry/assembly, map/GPU wait, unit-body drawing, pose misses, allocation/cache
pressure and address-space samples. Preserve detailed event records without
writing a full BMP on every measured frame; use bounded representative images
and output hashes, keeping evidence writes outside measured intervals.

Check replacement ownership against each current capture, unit bounds/clipping,
stable identities, action transitions and exact current-camera results. Sample
independent redraws after the timed session so correctness verification does not
warm or evict the measured session's caches. Include returning to an edited
region in a separate invalidation witness.

These are standalone completed frames. Actual input-to-display and the plan's
1,000 native-presented-frame/30 FPS target remain a separate integration check.
The native bridge is synchronous and its existing approximately 66 ms animation
timer does not establish 30 Hz. This test must expose those limits rather than
label an unpaced render loop as live-game performance.

## Implementation and reproduction

Implemented in `native/busy_session_plan.h`, `native/busy_session_preview.h` and
`biq_preview.cpp`, extending the existing preview. A 60-second independent wall
clock drives one synchronous renderer. Discrete zoom/minimap actions at seconds
20/22/24/26/28/40/50 remain queued in order; continuous scrolling coalesces while
rendering blocks. Pending discrete work can drain after 60 seconds, bounded at
180 seconds. The final output reports missing phases honestly. This models a
scripted input queue; it does not install a new game camera or input handler.

The scene has stable world actors at the home region and two distant regions,
with 24 or 64 actors per region. The visible count can fall while scrolling.
Seven unit families use stable IDs and offset mixed action timelines, native
body requests, movement offsets and direction changes. Cities, road/rail networks,
farms, mines, camps and supported resources are tied to world positions. Clipped
captured occurrences, rather than screen-fixed substitutes, determine visibility.

Representative composed images use a 96 MiB pixel budget. Snapshot copying and
logging are outside the reported render call but advance the wall clock and are
reported as evidence overhead. After timing, each saved snapshot is independently
rendered from cold state and compared exactly. This deterministic snapshot replay
is the correctness companion to wall-clock timing, not a hidden pre-warm step.
Existing separate edit/removal and action-matrix witnesses cover broader
invalidation and unit-action semantics; this preview is not a combat simulation.

The corrected preview is
`Renderer/native/build/busy-session-queued-preview-installed-20260909`.
It was built with normal MSVC `/O2 /W4 /WX` and `/LARGEADDRESSAWARE` against the
unchanged staged DLL `c2b3a2e8cf5ebfc8c32c185ea411cb68062a2683fbacead9660ad592d18b8af9`.
The receipt records a compiler-path override because the Visual Studio update
was pending reboot. The ordinary discovery attempts failed before compilation;
no game, installer, or VM restart was launched.

From the repository root, with Python 3.12 and a fresh output directory:

```sh
python -m Renderer.native.record_navigation_evidence \
  --binaries Renderer/native/build/busy-session-queued-preview-installed-20260909 \
  --out Renderer/native/build/busy-session-new-24 --scenario session \
  --width 2240 --height 1192 --tile-width 128 --waves 1 \
  --idle-units 24 --unit-actions mixed --dense-scene \
  --world-grid --world-regions --raster-control --region-size 128 \
  --region-metadata-mib 96 --region-receiver-shadows --tight-natural-bounds \
  --region-input-ring 4 --production-defaults
python -m Renderer.native.analyze_navigation_run \
  Renderer/native/build/busy-session-new-24 \
  --out Renderer/native/build/busy-session-new-24-analysis.json
```

Repeat serially with 64 units and a new directory. The immutable `inputs.json`
in each measured run contains the exact complete argument/environment tuple.
Do not build, clean storage or run another GPU workload during timed playback.

## Evidence

The first `busy-session-production-{24,64}` runs used a latest-state camera model
that could skip discrete zoom actions during long renders. Both missed width192;
the 24-unit run also missed local scrolling after the first jump. Each preserved
all five independently checked images, but neither covered the full schedule.
Retain them as diagnostics, not the completed fixture's three-zoom proof.
They exposed cold unit poses and region construction as the dominant stalls.
The corrected queued-input runs and workload table below supersede that model.

Preserve [the original targets](navigation_implementation_plan.md) and
[native presentation constraints](native_async_presentation_audit.md). Pixel
correctness and scripted phase coverage are separate from performance acceptance.

## Corrected queued-input results — September 9

Both runs use the unchanged production DLL with waves/reflections on at 2240×1192.
Each completed all seven discrete actions and visited widths 128/160/192. All six
independent snapshots per run matched exactly; no reported fallback or device recovery.
Inputs and binaries remained unchanged. The 24-unit run visited all eight phases.
The 64-unit run missed phase 4 (local scrolling after jump 1) because rendering
blocked through that interval. Its schedule coverage remains **incomplete**; queued
clicks were retained, but continuous scroll samples coalesced. Do not call it a
full all-phase pass. These are single-session observations, not 100-sample tails
per workload or native presentation evidence.

| Measure | 24 units per zone | 64 units per zone |
|---|---:|---:|
| Initial map render, ms | 16,493.94 | 15,725.46 |
| Completed updates | 98.00 | 51.00 |
| Timed wall duration, ms | 60,111.39 | 60,092.26 |
| Completed updates/second (not native FPS) | 1.63 | 0.85 |
| Median completion, ms | 280.74 | 539.12 |
| p95 completion, ms | 1,692.03 | 4,881.12 |
| Maximum completion, ms | 8,793.33 | 8,658.79 |
| Undispatched continuous input slots | 1,703.00 | 1,750.00 |
| Sampled minimum free VA, MiB | 749.75 | 710.45 |
| Sampled largest free region minimum, MiB | 698.31 | 641.05 |

Initial map rendering precedes the 60-second script; DLL load/configuration precedes
and is excluded from that initial-render measurement. Snapshot pixel storage was
61.11 MiB per run within the 96 MiB pixel cap (metadata additional). Address-space
samples are not live peaks or proof of the largest-transient headroom requirement.

| Workload | 24 units median / p95 ms (N) | 64 units median / p95 ms (N) |
|---|---:|---:|
| idle | 791.4 / 1,148.8 (15) | 1,908.1 / 2,387.6 (6) |
| scroll | 680.8 / 1,326.4 (14) | 906.6 / 2,411.1 (9) |
| zoom | 2,333.0 / 4,779.8 (4) | 3,673.9 / 5,911.9 (4) |
| jump1_idle | 8,793.3 / 8,793.3 (1) | 8,658.8 / 8,658.8 (1) |
| jump1_scroll | 1,692.0 / 1,692.0 (1) | Not dispatched |
| jump2_idle | 3,203.0 / 3,203.0 (1) | 4,432.6 / 4,432.6 (1) |
| jump2_scroll | 264.3 / 933.8 (17) | 535.7 / 1,166.7 (5) |
| return_idle | 176.4 / 431.6 (45) | 296.1 / 1,096.8 (25) |

The zoom row mixes first-use and returning zooms. The event table separates them
and includes the time waiting in the scripted queue. It is **not** measured native
input-to-display latency. First-use unit poses remain cold; camera changes do not
reset the caches or stop the world animation clock.

| Discrete action | 24 units request→completion ms | 64 units request→completion ms |
|---|---:|---:|
| First160 | 5,053.5 | 6,162.2 |
| First192 | 6,592.8 | 9,043.6 |
| Return160 | 5,747.9 | 9,515.3 |
| Return128 | 4,318.2 | 8,241.1 |
| Distant jump1 | 11,111.6 | 14,900.1 |
| Distant jump2 | 4,012.1 | 7,337.1 |
| Return home | 1,061.7 | 1,458.1 |

Per-category maxima across timed views (the maxima need not share one view):

| Category | 24-unit run | 64-unit run |
|---|---:|---:|
| visible | 721 | 721 |
| cities | 3 | 3 |
| roads | 219 | 219 |
| railroads | 62 | 62 |
| farms | 146 | 146 |
| mines | 64 | 64 |
| camps | 7 | 7 |
| resources | 61 | 61 |

The 24-unit run drew 16–24 unit occurrences per completed view. The 64-unit run drew 37–64 unit occurrences per completed view.

## Interpretation against the plan

Nearby scrolling and zoom exceed the ≤100 ms warm-final target in these busy
sessions; cold distant views still take seconds. No global map preparation was
performed, so the ≤100 ms prepared-unseen-region target remains untested. Correct
first native response ≤50 ms, submit/poll ≤2 ms, physical frame intervals and
1,000-frame native presentation remain unmeasured here. The return-idle improvement
shows cache value, but neither density approaches the native 30 FPS target.

Cold pose preparation and terrain structure are the next priorities because they
dominate the measured stalls. Native async completion is still needed to avoid
blocking the game thread; a queue alone cannot make cold rendering fast. Preserve
exact current-camera ownership and independent unit/action semantics.

Evidence: `Renderer/native/build/busy-session-queued-production-{24,64}` and their
`-analysis.json` companions. Timed trace views aligned completely in both runs;
post-session cold replays are excluded from timings and object counts.
Fifteen focused fixture/analyzer tests pass, including delayed discrete order,
missing coverage, excluded animation-clock timestamps, and compressed evidence.
The production runtime was not modified for the final fixture; its earlier 249-test
integration result remains scoped to that staged runtime. No injected compile was
needed for these preview/tool/document changes.

## Structural preparation after measurement

Global basic caches are a concrete next direction for cold jumps. Compact map
topology and coastline structures already exist, as do bounded retained ground
samples and world meshes. The missing step is preparing structural content before
the camera first needs it and retaining it beyond GPU eviction.

Prioritize camera-independent ground/relief inputs and reusable grid/index
templates, then versioned compiled structural regions with bounded disk storage.
Share them across 128/160/192 rather than storing three copies of identical world
data. Preserve exact projected vertices/materials and dependency observations;
the recent small per-tile query caches did not establish a speed benefit.

Topology-only inputs can support pure terrain structure, not arbitrary cities,
resources, forest exclusions, ownership or visibility. Complete immutable regional
appearance must be captured before preparing those consumers. Pack/compiler
identity, map changes, local edits, wrapping and corruption must invalidate or
reject stored data. Report preparation time, disk footprint, pressure/eviction,
and first-use/revisit latency separately. No global compiled-region cache is
implemented or claimed complete yet.
