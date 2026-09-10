# Navigation optimization continuation — September 9, 2026

Scope: widths **128/160/192**, current visual quality, native camera/capture,
visibility, overlays, picking and presentation ownership. The original
[plan](navigation_implementation_plan.md) is unchanged. Timings here are
standalone completed rendering (including capture for navigation),
**not live-game presentation**.

## Current production activation

The user subsequently requested putting the best verified approaches in production
before resuming experiments. `Renderer/bin/C3XRenderer.dll` is now the normal-tier
`navigation-usage-build-20260909` DLL, SHA-256
`c2b3a2e8cf5ebfc8c32c185ea411cb68062a2683fbacead9660ad592d18b8af9`.
The existing cache switch now supplies the previously measured wave/backdrop,
exact dependency, receiver-index and larger-memory defaults. Explicit standalone
controls still override them; waves/reflection preferences are unchanged.

Before staging, current core integration passed 247 tests with one skip, terrain
edit/resource playback and day/night unit-action checks. The small default-policy
change then passed 11 focused option/pose/analyzer tests and a normal MSVC build.
Same-build explicit/default pairs at 128/160/192 matched all 21 saved images;
each passed exact zoom return, changing animation, scroll/cold and removal/cold.
These six-frame temporal checks are correctness evidence, not performance tails.
Compiled source closure, runtime inputs and DLL hashes were rechecked before
the atomic replacement. The installed GOG executable is Large Address Aware;
live peak/contiguous-memory verification remains pending.

The logging follow-up uses the existing `OutputDebugStringA` path; see
[live usage logging](live_usage_logging.md). Current integration passed 249 tests
with one skip and the terrain/resource/day-night unit witnesses. Native log
verification matched 40 calls, 480 unit draws and 24 identities without missing
or invalid pairs. All seven width-128 animated images matched the preceding
production DLL exactly. This is logging verification, not a busy-session pass.
The performance-only DLL is also preserved as `previous-performance-C3XRenderer.dll`
(SHA-256 `ef1d63ff1c652c01fbc1804dbcc5c8576d6dbfa078e6663395dbcf5907ff209a`).

The receipt and rollback are under `Renderer/lab/out/navigation/promotion/`.
Restore `previous-C3XRenderer.dll` to `Renderer/bin/C3XRenderer.dll` to return to
the accepted mountain DLL, SHA-256
`ccd8e9f76d06e0c66c6cdc5f1d400713772c72c113d6ad8d114dcf59ec18e82f`;
`previous-run_navigation_evaluation.cmd` preserves the prior launcher as well.
Restart the game to load either DLL. No installer or game launch occurred.
The cache switch is enabled locally; waves and reflections are currently disabled
in the user's local configuration. Verification explicitly enabled both.

The next test is a [continuous busy-map session](busy_navigation_session.md),
including cold startup, idle animations, scrolling, all three zooms, distant jumps
and return travel without cache resets. Isolated warm tests do not establish it.

## Reproduction and evidence

The destination checkout started at `f2696829d9bc549ae01597cf42d25f3ddab736de`.
An external commit, `d94b46f2` (`mountains looking better`), subsequently included
the accepted mountain work and the first dependency-cache changes. It was
preserved. The later external commit `79307ece` (`updated`) includes the
queue/memory changes, region-aware backdrops, dense/mixed witnesses and individual
unit phases. It was preserved as well. Subsequent continuation changes
remain in the working tree; individual build receipts identify the actual
compiled source closure.
Historical handoff build directories were unavailable, so reproduction used a
fresh MSVC x86 `/O2 /W4 /WX` build of this checkout. The Windows 11 Pro ARM VM
has approximately 12 GiB RAM, four logical processors and Parallels Display
Adapter WDDM driver `20.18.2700.58628`; the historical 8 GiB configuration is not
the current machine. Evidence is under `Renderer/native/build/`.

At 2240×1192, width 128, waves disabled, ring four, the reproduced candidate
(`navigation-destination-clean-128`) measured 60.335 ms median / 125.548 ms p95
over 100 nearby changes. All 100 images matched the same-build independently
drawn regions (`navigation-destination-independent-128`), whose median/p95 were
887.678/1001.098 ms. The reproduced sweep built 33 static tiles and uploaded
6,970,854 bytes, so it does **not** reproduce the historical zero-build envelope.

Each verified run retains input/binary identities, completion markers and
post-run image hashes. `analyze_navigation_run.py` rejects modified inputs,
incomplete sweeps, mismatched cameras/quality and changed images. Paired
comparisons require identical binaries and runtime inputs, allowing only the
explicit dependency/cache controls. Timings exclude evidence hashing and BMP
writes. GPU workloads run sequentially. Clocks are fixed; changing animation is
a separate correctness witness.

## Experiments and their purpose

1. **Measure repeated preparation.** The phase-only run
   `navigation-phase-clean-128` found median contributor scanning of 19.311 ms
   and shadow-dependency work of 28.531 ms. These repeated on already retained
   geometry, making them the highest-value first target for smoother nearby pans.
2. **Retain exact shadow dependency proofs.** Extend `PreparedCasters` with
   exact ordered input comparison and bounded receiver proofs. Versions, GPU
   allocation identities, bounds, offsets, material bindings and light basis
   remain dependencies. The shadow-only cache/control pair matched all 100
   images; median fell from 57.253 to 36.963 ms. Tail improvement was smaller,
   pointing to remaining assembly and GPU waits.
3. **Index contributors conservatively.** A bounded spatial index selects
   candidates in original draw order; the original exact intersection and key
   serialization still decide reuse. This reduces repeated full-scene scans.
   The first index-control timing was invalid because its fallback allocated a
   full candidate vector per region. That harness defect was corrected to the
   original direct loop; its timing is superseded, though pixel equality held.
4. **Retain tile-center shoreline queries.** Repeated assembly now validates
   and replays the exact world/coast observations of center samples. It primes
   the existing per-tile query cache. Edits, wrapping and reset preserve the
   original query behavior. This targets capture-boundary spikes, rather than
   only frames with no assembly work.
5. **Preserve identical camera work.** Extend the existing one-active/one-pending
   queue so identical requests keep their ticket and ready result. Complete
   input bytes and epochs are compared, excluding caller addresses. Repeated
   completion redraws can consume useful work instead of cancelling it. This
   addresses a concrete prerequisite for native asynchronous completion; the
   injected bridge remains synchronous. See the [native audit](native_async_presentation_audit.md).

The first four steps preserve geometry and pixel contracts. Optional admission
failure falls back to independent computation. New CPU caps are 32 MiB for
receiver proofs, 8 MiB for their retained caster descriptors, 16 MiB for the
contributor index and 16 MiB for center queries. Existing prepared bounds/page
selection caps remain 8 MiB each. Descriptor replacement and query scratch can
overlap these owners; these caps are not a process or driver-residency bound.
The current width-128 sweep used about 0.69 MiB of receiver proofs, 0.63 MiB of
index storage and 3.43 MiB of center samples. No new queue snapshot owner is added.

## Current-shader measurements

A separate user-authorized mountain task accepted and staged a shader update
after the initial experiments. Measurements below use that accepted appearance.
The first run regenerated four compiled mountain shader files and was rejected
as stable-input evidence. The subsequent cache/control pair had unchanged
inputs and matched **all 100 images exactly**.

| Workload | Median ms | p95 ms | Evidence directory |
| --- | ---: | ---: | --- |
| Nearby 128, waves off, all three new dependency caches disabled | 66.754 | 118.388 | `navigation-current-dependency-control-128` |
| Nearby 128, waves off, caches enabled | 27.401 | 87.631 | `navigation-current-clean-128d` |
| Final verification, nearby 128, waves off | 26.081 | 81.024 | `navigation-verified-pan-128b` |
| Nearby 160, waves off | 22.860 | 63.727 | `navigation-verified-pan-160` |
| Nearby 192, waves off | 20.192 | 49.280 | `navigation-verified-pan-192` |
| Nearby 128, waves on | 434.556 | 520.449 | `navigation-final-waves-128` |
| Nearby 160, waves on | 365.733 | 423.181 | `navigation-final-waves-160` |
| Nearby 192, waves on | 351.342 | 402.835 | `navigation-final-waves-192` |
| Warm 128/192/160 changes, waves on, 102 changes | 522.234 | 671.188 | `navigation-final-zoom` |
| Warm zoom changes, waves off, larger cache, 102 changes | 54.524 | 69.384 | `navigation-verified-zoom-no-waves` |
| Distant destinations without map preparation, waves on, 100 jumps | 649.794 | 2867.215 | `navigation-final-distant` |

The two first-use zoom changes measured 1693.620 and 2378.444 ms; two samples
are not a p95 workload claim. All 102 warm zoom images matched their first-use
image exactly. Distant maximum was 6172.308 ms; all 100 completed with zero
fallback and no device recovery. These regions mixed unseen, overlapping and
evicted content. A unique camera destination is not proof that every visible
tile was unseen.

The final width-128 waves-off verification matched all 100 same-build independent
region renders (`navigation-verified-independent-128`). The waves-off zoom
witness's first-use changes took 1446.054 and 2112.001 ms; all subsequent
102 changes matched exactly and completed below 72 ms. These first-use samples
remain separate from the warm distribution.

The warm zoom trace recorded 7,735 animation-background misses and no hits over
105 renders, despite retained static meshes. Their combined multisample
color/depth footprint exceeds the existing 128 MiB cap. An opt-in
`C3X_RENDERER_THREE_ZOOM_MEMORY=1` experiment uses a 64 MiB viewport cache
(or the larger compile-time tier) and an 832 MiB backdrop cap. It extends the
existing owners and eviction logic; it preserves resolution, MSAA and depth.
Changing the cap clears optional cached owners before admitting under the new
bound. Defaults remain unchanged. This is being tested for repeated zooms,
with exact image parity and the plan's address-space reserve as stop conditions;
it is not a live-game memory envelope or whole-map storage implementation.

Its first 102 warm changes measured 171.973 ms median / 229.685 ms p95, with
7,514 backdrop hits and 221 first-use misses over all 105 renders. Minimum
sampled free virtual address space was 1,693,528,064 bytes, with a minimum
1,610,350,592-byte contiguous free region. These are sampled process address
space, not a peak-memory or GPU-residency measurement. The same-build lower-cap
control (`navigation-memory-zoom-control`) measured 518.315 ms median /
667.520 ms p95 for its 102 warm changes. All first-use images matched across
runs, and every repeated image in both runs matched its first-use image exactly.
The larger-cap improvement is therefore a controlled result, although it still
misses the 100 ms final-completion target.

The waves-off pair builds the same 33 static tiles. Its geometry p95 improves
from 41.117 to 28.308 ms and median static draw/submission from 49.315 to
10.709 ms. This isolates a useful CPU improvement, without claiming a
controlled comparison to the old shader build. Wave uploads and animation
background work are separate from the static build counters.

The waves-off nearby result reaches the 100 ms final-completion p95 threshold
for this specific standalone sweep. It does not reach the 50 ms response
threshold, certify all three widths with animation, establish distant prepared
destinations, or supply 1,000 presented frames. Native presentation remains
unmeasured. The distant witness explicitly starts without whole-map preparation;
its result must not be labelled a prepared-map pass.

## Waves, crowded scenes and stationary animation

The follow-up tests the existing world-anchored backdrops and retained wave
cells with the larger backdrop cap. At width 128, their combined cache/control
pair measured **72.762/414.675 ms** versus **404.644/472.701 ms** median/p95.
All 100 images matched exactly. Wave uploads fell from 1,010,930,688 bytes to
zero; 4,401 backdrop hits left 752 misses. Captured-set changes still discarded
many unaffected backgrounds, explaining why the long stalls persisted.

`C3X_RENDERER_BACKDROP_DEPENDENCIES=1` now extends the existing backdrop owner
with the completed-region dependency key. It uses the same guarded projection,
ordered static contributors, light/reflection and receiver-shadow dependencies.
It retains both unresolved scene-linear MSAA color **and depth**. A bitmap-only
region hit remains prohibited for animation backgrounds. Metadata is charged to
the existing backdrop cap, and a rejected proof falls back to the original
whole-capture identity. Unchanged-view lookup stays cheap. No animation pose
enters a static backdrop key.

| Waves-on nearby pan | Candidate median/p95 ms | Same-build independent median/p95 ms |
| --- | ---: | ---: |
| 128 | 78.379 / 141.727 | 405.983 / 470.925 |
| 160 | 74.926 / 122.686 | 386.045 / 444.768 |
| 192 | 72.499 / 105.015 | 333.283 / 408.168 |

Each pair matched all 100 images exactly. Evidence is
`navigation-backdrop-dependency-{pan,control}-WIDTH` (128 control has suffix
`128b`; its earlier dispatch failed without a renderer process). The separate
same-build width-128 dependency-disabled ablation measured 73.639/413.938 ms
and also matched exactly: the new proof reduces tail stalls while adding some
typical validation cost. All three waves-on tails still miss the 100 ms target.
These pan results do not update the zoom or distant-jump measurements above.

The new `--scenario idle` witness keeps the camera stationary and advances
authored poses at 15 Hz, **unpaced**. It reports completed rendering, not native
frame delivery. At 2240×1192 and width 128, 100 samples with the existing
14 animated-resource-body fixture measured **34.396/40.349 ms** without waves
and **87.572/89.296 ms** with waves. Both used ten warmup renders and built or
uploaded no static terrain during measurement. This establishes that animated
idle itself needs work; static-image reuse does not prove animated throughput.

The synthetic `--dense-scene` places world-fixed supported cities,
infrastructure, camps and resources while preserving captured draw eligibility.
The separate `--idle-units` pass uses native unit-body APIs on a GDI canvas,
with its copy and unit time reported separately. The width-128 fixture contains
24 units, 3 cities, 219 road tiles, 146 farms, 64 mines, 6 camps and 42 resource
tiles (62 animated bodies), plus 57 visible wave chunks. Required replacement
flags are checked, so omitting objects cannot count as a fast pass. This is not
a live game capture or proof of every density/culture/action combination.

Its first ten-frame smoke test exposed roughly 400 ms in the unit pass alone.
The existing 8 MiB pose cache repeatedly evicted completed frames. The optional
`C3X_RENDERER_UNIT_POSE_MEMORY=1` raises that same cache to **256 MiB / 4,096
entries**. Shrinking evicts old entries; optional admission failure preserves
the currently completed body. Native projection, materials, shadows, action
cursors and pixels are unchanged by this memory option. Pixel capacity is
accounted; cache-vector metadata and the temporary/current output owners are
additional bounded storage, not included in the pixel cap.

The user also requires independently phased ambient units. Their source-time
phase now derives from native unit identity, so camera changes and callback
order cannot restart a loop or synchronize neighboring units. Movement, combat
and explicitly native-directed fidget clips retain their native cursors.
The exact pose-cache key includes the resulting pose frame. The queued-action
test now compares against the new identity's own idle frame rather than
requiring different identities to share an ambient pose.

With these individual phases, the matched 100-frame dense idle pair measured:

| Dense width-128 idle, waves on | Total median/p95 ms | Unit pass median/p95 ms |
| --- | ---: | ---: |
| Original 8 MiB pose cache | 542.144 / 555.948 | 407.729 / 419.304 |
| Optional 256 MiB pose cache | 131.277 / 136.868 | 10.541 / 10.991 |

All 100 completed images matched exactly (`navigation-dense-unit-{memory,control}-128`).
Both runs used **80 declared warmup renders**. Their accumulated warmup render
times were 26.932 seconds candidate and 43.568 seconds control, excluding the
initial map render. This is experimental preparation, not an implemented native
loading phase or proof of cheap first-use units. Candidate pose pixels reached
145,784,364 bytes; sampled free VA remained at least 1,814,020,096 bytes with
1,749,483,520 bytes contiguous. Other widths, repeated zooms, larger unit counts
and live-game memory still need separate evidence. The map portion remains the
dominant warm cost, at 120.443/125.606 ms median/p95.

`--unit-actions mixed` adds separate scripted move, attack, return, fortify and
idle timelines, including direction reversals and native body-anchor changes.
Noncombatants use their supported fidget action in place of attack. At width
128 with waves and 24 units, 100 mixed updates measured **143.037/180.314 ms**
median/p95 with the larger pose cache, versus **785.725/838.258 ms** with the
same build and original cache. All 100 images matched exactly. The unit pass
alone measured 10.204/48.909 ms versus 651.156/708.323 ms; remaining occasional
pose misses still matter. Measured calls include 969 movement, 361 attack,
475 fortify and 476 idle draws; remaining noncombatant draws use fidget.
The 80-frame warmup accumulated 30.321 seconds candidate / 63.180 seconds control,
excluding the initial map render. Candidate sampled free VA was at least
1,756,643,328 bytes, with 1,657,798,656 bytes contiguous. Evidence is
`navigation-dense-mixed-unit-{memory,control}-128`. This is a stationary map with
moving bodies, not native simulation ownership or concurrent navigation/
presentation coverage. Other zooms and first-use behavior remain outstanding.

The current rendered unit witnesses also prove three equal-facing warriors at
one clock have distinct ambient pixels, and that repeated identity/time produces
exact pixels. Their native action/compositing checks execute 582 action draws.
The first integration attempt correctly failed because its verifier still
required the obsolete count of 564, despite passing native runs. The verifier
now requires 582 and the new phase witness explicitly. The complete current-code
animation integration rerun passed **221 tests with one skip**, resource playback
and day/night unit action/compositing witnesses. Its receipt is
`Renderer/lab/out/integration/animation.json`, DLL SHA-256
`6ab1a685c703b313a0a7f204b09bfbd6ec6cd107d952bc7ce0d255f58a5ddea9`.
Both unit runs preserved terrain and matched the cold post-draw map exactly.
No injected compilation, staging or game launch occurred. The navigation recorder now uses a disposable
batch file, matching the category dispatcher's short Parallels command, rather
than transporting its entire environment in one long command argument.

### Animated composition receiver selection

The phase-instrumented build `navigation-animation-phase-build-20260909`
(DLL SHA-256 `2eb4ba22c435d79c4abb4e6642b2121b0647f26d9b76aee2eff9eb77513395d9`)
measured the dense width-128 idle scene with waves, 24 individually phased units,
80 warm-up frames and 100 measured frames. Median resource-pose preparation was
about 8.4 ms, animated-block submission 36.8 ms and final readback wait 70.0 ms.
These are CPU submission/Map-wait intervals, **not GPU timestamp durations**;
GPU execution overlaps submission. Background-copy submission was 0.083 ms and
CPU image assembly about 1.4 ms. No static geometry was built or uploaded.
The bounded trace contains only 97 of these measured frames. The analyzer now
matches idle trace records to measured animation clocks and reports partial
coverage, rather than filling a last-100 summary with warm-up records. Primary
benchmark timings and exact saved-image comparisons still cover all 100 frames.

The highest-value bounded CPU experiment reuses the existing conservative
static contributor index when selecting each animation block's shadow receivers.
`--composition-receiver-index` enables it; it remains optional. The exact original
intersection, receiver order and shadow-page preparation remain authoritative.
Posed/foreign buffers, missing or stale indexes, multiple rectangles, oversized
queries and allocation failure use the complete original scan. This adds no
retained owner or memory budget. Production-code tests compare actual collection
against complete scans over reflected/translated rectangles and fallback cases.

Build `navigation-receiver-index-build-20260909` (DLL SHA-256
`fa081580878391f5b926af1d263e7f05e7dd8221438547443393f2d7a3b0ac2a`) produced
**100 exact images** in `navigation-dense-receiver-comparison-128.json`:

| Dense idle, width 128 | Median / p95 ms |
| --- | --- |
| Same-build complete receiver scan | 128.155 / 139.476 |
| Indexed exact receiver selection | 118.240 / 120.413 |
| Animated-block CPU submission, scan (97 trace-aligned frames) | 36.063 / 37.013 |
| Animated-block CPU submission, indexed (97 trace-aligned frames) | 27.048 / 27.634 |

The final wait remains about 67 ms median, so this does not meet the 33.4 ms
frame-interval target. Warm-up is excluded and native presentation remains
unmeasured. The focused region/analyzer suite passed 17 tests. Broader dense
160/192 runs are separate capacity measurements, not implied by this pair.

### Supported-width unit-pose capacity

With 24 independently phased idle units, the same indexed build measured
99.358/101.571 ms at width 160, but **225.661/229.477 ms at width 192**.
The 192 pose cache reached 268,098,224 bytes and missed on 300 of 2,400 measured
unit draws; the larger images no longer fit in its 256 MiB budget. The map itself
took only 68.070/71.714 ms. Width 160 retained 227,772,336 bytes and had no unit
pose misses. Fixture contents vary with viewport coverage: at 160 there are
3 cities, 178 road tiles, 114 farms, 55 mines, 6 camps and 34 resource tiles;
at 192 there are 3 cities, 125 roads, 77 farms, 41 mines, 4 camps and 22 resources.

The next experiment raises the optional pose-pixel budget to **512 MiB**, retaining
the 4,096-entry limit. This is a measured capacity correction, not a quality
change. `--unit-pose-memory --unit-pose-memory-mib 512` selects it; default behavior
remains unchanged. Tier reductions evict owners before cache lookup. The approved
larger bound has its own unit test and does not reserve the entire budget eagerly.

Build `navigation-unit-512-build-20260909` (DLL SHA-256
`1f99f1b62e7c31ade3cdbade71b47de99a8e1dd1b754cc2a0a6729ae6c5ae5b7`) produced
100 exact width-192 images in `navigation-dense-unit-512-comparison-192.json`:

| Dense idle, waves on, width 192 | Total median / p95 ms | Unit plane median / p95 ms |
| --- | --- | --- |
| Same-build 256 MiB pose cache | 234.743 / 246.611 | 159.150 / 169.751 |
| 512 MiB pose cache | 85.577 / 88.322 | 19.049 / 19.990 |

The larger tier retains 328,002,224 pixel-capacity bytes (about 313 MiB) and hits
all 2,400 measured unit draws. Sampled free address space is at least
1,692,647,424 bytes, with a 1,626,066,944-byte contiguous region. These are sampled
standalone values, not peak/live-game memory proof. The 80-frame warm-up took
26,734.614 ms, excluding initial map rendering; first-use preparation remains
unfinished. Three simultaneous zoom working sets and mixed actions require
separate capacity evidence.

The same 512 MiB build also measured 100 dense idle frames with waves disabled
at each supported width (`navigation-dense-unit-512-no-wave-{128,160,192}`).
All 2,400 unit draws hit retained poses at each width, all 99 adjacent images
changed, and static terrain builds/uploads and recovery counts remained zero.

| Dense idle, 24 units, waves off | Total median / p95 ms | Map median / p95 ms | Unit plane median / p95 ms |
| --- | --- | --- | --- |
| 128 | 81.796 / 87.132 | 71.293 / 76.497 | 10.111 / 10.486 |
| 160 | 71.807 / 78.419 | 57.228 / 63.635 | 14.194 / 15.169 |
| 192 | 65.143 / 71.458 | 45.537 / 51.749 | 19.304 / 20.041 |

These are waves-off **crowded** scenes; the older 34/40 ms waves-off idle witness
had fewer animated resources and no unit plane. The three crowded warm-ups took
22,681.653 / 23,525.346 / 25,126.813 ms over 80 frames, excluding initial map
rendering. The width-192 run had one 116.821 ms outlier. All three remain above
the 33.4 ms frame target, despite fitting the warm 100 ms p95 completion endpoint.
The final focused workflow/region/analysis/unit/fixture suite passed 69 tests.

`navigation-dense-mixed-unit-512-wave-192` keeps that crowded map animated while
24 units move, reverse, attack, fortify, fidget and idle with independent phases.
Its 100 completed frames measured **90.506/143.658 ms**, split into map
69.070/74.490 ms and unit plane 19.729/69.951 ms. The action counts are 969 moving,
361 attacking, 475 fortifying, 476 idling and 119 noncombatant fidgets. There are
2,371 pose hits and 29 misses; retained pixel capacity reaches 435,058,352 bytes.
The cache-capacity trace never decreases, so these misses are first encounters,
not the width-192 idle eviction cycle. The 80-frame warm-up took 31,731.895 ms
excluding initial map rendering. This identifies preparation of first-use poses
as the next unit-specific latency target. It is a complete scripted-action
capacity witness, not an independent-image pair or concurrent camera/presentation
pass. Earlier same-build unit-cache pairs and native action/phase tests retain
their separate correctness scope.

The final current-code `renderer.py integration animation --renderer-only`
passed 221 tests with one skip, resource playback and both day/night unit replays
(582 directed-action draws and 288 body-matrix draws per unit replay). Independent
ambient phases, exact repeats, clipping/color-key behavior, retained terrain and
post-unit cold terrain parity passed. The first attempt had a VM file-access
dispatch failure for the day replay; it exited, and a complete retry passed.
Logs are `navigation-animation-final-integration.log` and
`navigation-animation-final-integration-retry.log`. The final integration receipt
is `Renderer/lab/out/integration/animation.json`, implementation identity
`76489df1da0928add1a738ead5a5d542b417c0f1eaedab631bfd265279b8ed09`, DLL SHA-256
`24876436ba52eb01ab722e13265f5bb94d24d68be7a5abeb0acd299d40a030b2`.
No injected sources changed, so injected compilation was not needed. No staging,
installation or live-game launch occurred. Default category replays and opt-in
dense performance runs retain their distinct scopes.

## Cold and distant-region continuation

The latest warm-path improvements do not resolve cold navigation.
`navigation-current-distant-128`, using the unit-512 build with retained waves,
dependency-aware backdrops and indexed receivers, measured **472.243/2812.934 ms**
over 100 distant destinations. Geometry preparation measured 117.548/2031.977 ms;
16,789 tiles were built and 1,041,671,130 bytes uploaded. Maximum completion was
6077.113 ms. All destinations completed without fallback or recovery. This is
still **unprepared** travel; unique cameras do not imply all their tiles are unseen.

The existing background queue prepares at most 512 nearby PREFETCH records from
the current captured appearance snapshot. It does not provide arbitrary-map
preparation. Retained ground samples and world meshes remain the extension points;
topology alone still cannot supply complete object appearance or visibility.

`navigation-cold-phase-distant-128` is a ten-destination diagnostic using build
`navigation-cold-phase-build-20260909` (DLL SHA-256
`f474e5bb35aa4ea7ef697ed659f49883a661049b3851043cacda64103a657c76`). The historical
`cliffs_ms` field also includes all natural geometry. New profile-only
`natural-mesh-phases` records separate base ground, surface decals, relief,
vegetation floor, cities and forests. Base-ground and relief generation dominate
this fixture; city and forest emission are much smaller. These profile records
include the initial view separately from the ten jumps and are not a p95 pass.

An experiment reused the mountain halo's exact interior samples during vertex
emission. Production tests preserved vertex bytes and unique dependency
observations. The same-build native comparison preserved **all 100 images**, but
enabled completion was 498.566/2911.667 ms versus 460.215/2821.553 ms disabled.
It did not establish a navigation benefit and was **removed, not promoted**.
`navigation-mountain-samples-comparison-128.json` retains the comparison; build
`navigation-mountain-samples-build-20260909` retains its binaries, source-base
commit and `experiment.patch` for reconstruction. Its DLL SHA-256 is
`1f0e4702711cf60a0fcf1451eb5ce5bf9c1cb6f7f3149e93136d26e816f02986`.

A second experiment cached exact material-weight queries within each immutable
tile compilation, using at most 512 KiB of table storage. Portable tests preserved
values and dependency observations through edits and wrapping; all 100 native
distant images matched. Enabled completion was **469.244/2790.818 ms**, versus
**460.355/2826.376 ms** disabled. The small, mixed timing change does not establish
a useful improvement. This experiment was also removed. Its exact comparison is
`navigation-material-samples-comparison-128.json`; binaries, source-base commit
and reconstruction patch remain in `navigation-material-samples-build-20260909`
(DLL SHA-256
`b8abcbe3031d1e82142031059c1acbdb896574faea076442890d4730b1620833`). The analyzer
continues recognizing these archived controls. Neither experiment reduces the
remaining need for bounded preparation before travel and exact invalidation.

## Activation and cleanup

The final portable contract suite passed 86 tests, including bounded cache
shrink, dependency edits/wrapping, publication supersession, unit takeover and a
deliberately stalled completion copy. The waves-on animation pairs at
128/160/192 matched all 21 saved images exactly against same-build independent
rendering. Each pair also passed exact zoom return, scroll/cold and
removal/cold checks. These six-frame temporal witnesses certify correctness,
not a 100-sample or continuous-frame performance target. A contact sheet is
under `Renderer/lab/out/navigation/verification-contact-sheet.png`; its objects
are deliberate fixture placements, not an in-game capture.

The final queue-only follow-up is `navigation-completion-build-20260909`, DLL
SHA-256 `3a92e8a1add7ed209f71957cac64104d9b186fccde0989aecfbd16122983265f`.
Its synchronous width-128 pan rerun measured 27.040/87.562 ms median/p95; all
100 outputs matched its stressed asynchronous queue exactly. Submission,
per-request maximum polling and duplicate-submission p95 were
0.505/0.430/0.276 ms. Publication copying and obsolete-owner cleanup now occur
outside the queue lock, with a final ticket/cancellation check before the swap.
See the native audit for workload and memory limits. The native bridge still
does not bind this API or schedule completion redraws.

`python3 Renderer/renderer.py integration shadows --renderer-only` passed its
current-code checks: 244 tests passed and one was skipped, followed by terrain
edit, resource playback and day/night unit-action witnesses. Terrain edit used
127 built / 260 reused tiles and matched cold pixels exactly. The integration
candidate DLL hash is
`17e041e4d50a403a34ebbfb6757fb688be1be1ece5a76644d54c074bab4bb631`;
its independent receipt is `Renderer/lab/out/integration/shadows.json`.
No injected compilation was requested because injected sources were unchanged.

Before the later user-authorized production activation above, this continuation
had not staged a navigation DLL, run `INSTALL.bat`, or launched Civ III.
The separate mountain task had staged DLL SHA-256
`ccd8e9f76d06e0c66c6cdc5f1d400713772c72c113d6ad8d114dcf59ec18e82f`
from `f2696829` native code plus the accepted shaders; it excludes this
continuation's native C++ changes. Its rollback is
`Renderer/lab/out/mountains/promotion/previous-C3XRenderer.dll`, SHA-256
`c9b627cc0ec05518bf9424a3c7f4e0931764399af2dfd042358fc136ece3e7bd`.
The promotion receipts describe its verification limits. Historical handoff
hashes are not evidence of the currently staged binary. A live scene-file edit
arrived externally and was preserved.

The explicitly disposable `dirty-block-clip-20260908/fixture` snapshot contained
31,793 files exactly duplicated by current repository-relative inputs. After
size/SHA-256 comparison, 6,479,415,488 logical bytes of duplicates were removed;
27 differing files, its world input and notes remain. The fixture's
`duplicate-cleanup.json` maps every removed copy to its identical surviving
input for reconstruction. Required packs, ignored source studies and visual
findings were preserved. Logical deletion is not a guarantee of immediate APFS
physical-space recovery.

After verification, 2,299 completed benchmark images were checked against their
receipts. Of these, 1,435 byte-identical copies (15,145,506,482 logical bytes)
were replaced atomically with verified APFS copy-on-write clones. All filenames,
image bytes and receipts remain available; later edits to one file do not alter
the others. Free disk space increased from approximately 13.9 to 28.0 GiB.
`Renderer/native/build/navigation-evidence-storage.json` records every shared
copy. Rejected/incomplete runs and unique evidence were left intact.

The later refresh verified 3,677 completed images and shared 2,183 identical
copies while preserving every payload and filename. Free space rose from
approximately 12.9 to 20.3 GiB. This is an additional physical-space observation,
not a second claim that the older logical duplicate bytes were newly deleted.

A subsequent refresh checked 4,384 completed images and shared 2,588 identical
copies. Free space rose from 14,015,528,960 to 18,356,551,680 bytes (about 4.0 GiB
recovered). Later runs consume additional space; this records that cleanup
observation rather than the current free-space total.

The final refresh checked 4,990 completed images and shared 2,791 identical
copies. It recovered a further 2,162,221,056 bytes (about 2.0 GiB), leaving
16,038,711,296 bytes free at completion. All paths and receipt-verified image
payloads remain intact; required asset packs and unique evidence were preserved.

## Remaining targets and next work

The user explicitly requires dense mixed scenes at all three supported widths:
many native-directed units, cities, roads/railroads, farms/mines, barbarian camps,
supported resources and waves. Category correctness witnesses do not establish
combined-scene capacity. Add separate stationary changing-clock measurements
with waves off/on and increasing visible unit/resource counts, followed by mixed
pan/zoom/action runs. Report object counts, warm-up, static rebuild/upload work,
animation and unit-plane cost, frame interval tails and memory. Keep the existing
1,000 actually presented-frame requirement separate from standalone completion
timings. The existing approximately 66 ms native timer supplies about 15 callback
opportunities per second for timer-driven idle redraw; rendering cost and native
scheduling can reduce delivery further. Six-frame animation parity witnesses
are correctness evidence, not an idle throughput pass. Retain current category
limitations and native ownership; this request does not expand deferred wonder
or Districts rendering scope.

| Plan target | Current evidence |
| --- | --- |
| Correct response ≤50 ms p95 | Only the width-192 waves-off standalone pan sweep reaches this endpoint; native first response is unmeasured. |
| Warm final completion ≤100 ms p95 | All three waves-off pan sweeps and larger-cache waves-off warm zooms pass this standalone endpoint. Waves-on pans and zooms fail. |
| 30 delivered FPS; 1,000 presented frames | Not demonstrated; no live presentation measurement exists. |
| Unseen/evicted destination after map preparation ≤100 ms p95 | Whole-map preparation/storage is not implemented. Unprepared distant jumps remain slow. |
| Submission/polling ≤2 ms p95 | The 100-request standalone queue call measurements pass; the synchronous injected game-thread path does not inherit this result. |

The next high-value work is to reduce the remaining animated map submission/GPU
completion cost and prepare first-use unit poses. Dense idle now has waves-off/on
measurements at all three supported widths. Mixed actions at 160, larger visible
unit counts, simultaneous three-zoom pose working sets and concurrent navigation
still need capacity evidence. A large warm cache is insufficient. Reduce/retain
cold region compilation using complete immutable appearance inputs as well.
Compact topology alone does not provide appearance or draw
eligibility. Whole-map preparation and versioned compiled-region storage need
separate startup, disk-footprint, edit/invalidation and pressure evidence.
Native async presentation still requires a current-camera compatible image and
a proven UI-thread completion redraw; copying an older bitmap beneath current
overlays/picking is not an acceptable shortcut. These are outstanding work,
not grounds to reinterpret the old 60/138 ms resident result as a live-game pass.
