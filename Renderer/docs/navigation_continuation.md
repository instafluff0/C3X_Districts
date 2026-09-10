# Navigation optimization continuation — September 9, 2026

Scope: widths **128/160/192**, current visual quality, native camera/capture,
visibility, overlays, picking and presentation ownership. The original
[plan](navigation_implementation_plan.md) is unchanged. All timings here are
standalone capture plus completed rendering, **not live-game presentation**.

## Reproduction and evidence

The destination checkout started at `f2696829d9bc549ae01597cf42d25f3ddab736de`.
An external commit, `d94b46f2` (`mountains looking better`), subsequently included
the accepted mountain work and the first dependency-cache changes. It was
preserved. Later queue/memory/witness changes remain in the working tree;
individual build receipts identify the actual compiled source closure.
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
Noncombatants use their supported fidget action in place of attack. This
fixture is being verified; it does not claim native simulation ownership or
concurrent navigation/presentation coverage.

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

This continuation has not staged a navigation DLL, run `INSTALL.bat`, or
launched Civ III. The separate mountain task staged DLL SHA-256
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

The next high-value work is to extend the existing world-anchored animation
background/wave retention into the measured pan workload, with independent
controls, then reduce/retain cold region compilation using complete immutable
appearance inputs. Compact topology alone does not provide appearance or draw
eligibility. Whole-map preparation and versioned compiled-region storage need
separate startup, disk-footprint, edit/invalidation and pressure evidence.
Native async presentation still requires a current-camera compatible image and
a proven UI-thread completion redraw; copying an older bitmap beneath current
overlays/picking is not an acceptable shortcut. These are outstanding work,
not grounds to reinterpret the old 60/138 ms resident result as a live-game pass.
