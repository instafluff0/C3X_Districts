# Zoom performance verification

**Animation correctness correction:** a live report revealed that the e25
candidate's bitmap region hits could leave stale linear color/depth in the
resource-animation backdrop path. The new staged DLL is `71e969c7…`; it requires
actual linear backdrop restoration and retains main-map caching. Seven animation
images match independent rendering exactly; the old DLL reproduced the reported
block corruption. Historical static timings below do not validate that old
animation path. See [the current handoff](navigation_handoff.md) for receipts
and the corrected output contract.

**Handoff staging:** the user requested production staging on September 9.
The exact `navigation-region-inputs-20260909` DLL (`e25e139c…`) is now in
`Renderer/bin/` for evaluation. No installation, game launch or visual acceptance
is implied. See [the handoff](navigation_handoff.md) for full hashes, activation,
rollback, compact evidence, measured targets and next work. Statements below
that a historical experiment was not staged describe that earlier checkpoint.

## Current results — September 9, 2026

**Current supported zoom envelope:** the user approved restricting custom zoom
to the three closest levels, 128/160/192, on September 9, 2026. The injected `Z`
control and native-state synchronization now exclude 64/96. Earlier five-level
and widest-64 findings below remain diagnostic history, not new production gates.
This reduces working-set and verification scope; it does not excuse the remaining
full-view latency at 128 or the cold-destination/native-presentation requirements.

The navigation implementation is in progress. No candidate from this work has
been staged and Civ III has not been launched. The native boundary is still
synchronous; [the source audit](native_async_presentation_audit.md) records the
completion/redraw, displayed-transform and unit-scheduling requirements.

**Architecture assessment:** the latest clean four-tile current-input-ring sweep
(`region-input-ring4-clean-100`) has 35 misses rather than the preceding 295,
60.130 ms median / 137.580 ms p95 / 176.845 ms p99 / 226.245 ms maximum.
All 100 images exactly match the prior two-tile-ring images (different DLL).
This is a large reduction in unnecessary raster work, but routine CPU dependency
work and periodic geometry assembly remain too expensive. Median CPU submission
is 37.363 ms; geometry assembly p95 is 36.695 ms despite zero static builds/uploads.
The next high-value targets are retaining prepared scene/dependency calculations
across unchanged captures and incremental handling of capture-set changes. The
100 ms p95, continuous 30 FPS and native presentation targets remain unmet.

The preceding receiver-scoped shadow dependency
experiment completes 100 prepared new views in 56.978 ms median / 197.096 ms p95,
versus 938.716 / 1021.405 ms independently rendered with the same DLL and inputs.
All 100 images match exactly. The 100 ms p95 and 30 FPS targets remain unmet.
Compared with the preceding broad-page-key run, misses fall from 593 to 323 and
p95 from 286.502 to 197.096 ms, but median rises from 41.541 to 56.978 ms. That
cross-build comparison suggests a tail/CPU tradeoff, not a universal speedup.

Completed world-region caching demonstrates a
large, exact-image improvement for newly assembled prepared views. In the latest
matched 100-view pair, median completion falls from 924.136 to 41.541 ms and p95
from 1026.033 to 286.502 ms. All 100 images match independently rendered regions
byte for byte; both paths build/upload zero static geometry. Legacy translated
bitmap reuse and viewport-cache hits are disabled in this witness. This is useful
evidence for prepared scrolling and revisits, but p95 still fails 100 ms, and
fast pan/zoom anywhere is not established. The measured workload is 128 zoom,
waves off, standalone completion; it does not certify other zooms, unseen or
evicted destinations, animation, native presentation, or 30 FPS.

The latest cost attribution rules out reflection-only work as the main solution.
With the entire reflected-scene pass disabled, 100 complete prepared views still
take 511.645 ms median, 565.262 ms p95, 635.980 ms p99 and 684.696 ms maximum.
The matched current-quality run is 633.708 / 689.040 / 707.643 / 787.392 ms.
This is a **diagnostic pass omission**, not a quality-preserving speedup: images
intentionally lack reflections and cannot pass the resident-quality analyzer.
Both runs use the same frozen `cb5324c0...` DLL and runtime inputs, no waves,
no pixel reuse, and 100 full rasters with zero static builds/uploads. Receipt:
`reflection-ablation-100/comparison.json`. The changed tool exposes the existing
reflection-control switch, labels ablation receipts and rejects those receipts
from quality gates; the executable receipt-rejection test passes.

The implemented opt-in world-region cache owns completed guarded 136-by-136 GPU
images for the 128-pixel world grid. Full value keys cover guarded ordered draw
contributors, off-region shadow casters, reflections, nearby city lights,
lighting/material/content revisions and current captured eligibility. Dynamic
textures are excluded. The cache requires prepared geometry and includes its
allocation versions: it cannot bypass geometry compilation after mesh eviction.
It owns no native surfaces and adds no presenter. Cancellation is checked between
regions, including cache hits. Defaults remain unchanged pending broader checks.

The latest run records 17,929 hits and 593 misses, reusing 261,992,576 of
267,008,000 output pixels across the sweep. Retained GPU images peak at
112,307,712 bytes and accounted key/entry metadata at 44,495,520 bytes, within
explicit 256 MiB GPU / 96 MiB metadata / 4,096-entry limits. There are zero
evictions and rejected admissions. An earlier 32 MiB metadata run had 613 misses
and eviction pressure; raising the limit removed eviction but barely changed
misses. **Memory pressure is therefore not the principal explanation for the
remaining misses.** The next high-value experiment must distinguish previously
unprepared world regions from changed draw/shadow/capture dependencies. That
determines whether advance preparation or dependency work will remove scrolling
hitches. Do not weaken validity keys or keep raising budgets without that evidence.

The first complete dependency trace (`world-region-dependencies-100`) now resolves
that question: all 593 misses are previously observed world-region occurrences
with changed component fingerprints. None are first-observed regions or unchanged
fingerprints. Main shadows change in 517 misses, reflected shadows in 542, main
draw contributors in 250 and reflected draw contributors in 274 (overlapping
counts). Global context and city lights remain stable. This supports investigating
capture-dependent draw/shadow invalidation before expanding neighborhood prefetch.
Fingerprints are diagnostics only; exact value keys still govern cache reuse.
The completed pixel diagnostic (`world-region-pixels-complete-100`) repeats those
same counts. Of 593 misses, 273 have earlier fully covered region images available
inside this sweep; all 273 have identical currently visible pixels. The other 320
lack that comparison coverage and remain unclassified. Guard pixels are not
compared. This is evidence of avoidable invalidation in this workload, not a
general proof that changed casters/contributors can be ignored. The next targeted
implementation investigation is narrowing draw/shadow dependencies to contributors
that can affect guarded region pixels, while preserving off-screen caster reach,
reflections and current visibility eligibility. Simply prefetching more neighboring
regions or increasing memory does not address the demonstrated cause.

The diagnostic DLL is
`c0af26ce0bec0f553dc241761cd95718e040f1527d4b5abfebd2b53e4bd5b7ce`;
its `/W4 /WX` build and runtime/binary integrity receipts pass. All 100 complete
images also match the prior independent-region images exactly (different DLL,
unchanged runtime content). Four region ownership/dependency/analysis tests and
the resident evidence rejection test pass. Trace-heavy timings are excluded from
performance claims and rejected by the performance analyzer. The earlier
`world-region-pixels-100` trace hit the default 8 MiB log cap and was correctly
rejected as incomplete; the explicit diagnostic mode now permits a bounded 32 MiB
trace. No native presentation, visual acceptance or staging is implied.

The receiver-scoped matched pair is `receiver-shadows-100` versus
`receiver-shadows-independent-100`, DLL
`c067345a3c2b471ac4317e8e2a9261d394fb78e2ea3270864515215567257d13`.
Both completion and runtime/binary integrity receipts pass. Cached p99/max are
313.221/344.486 ms versus 1094.383/1139.843 ms independently rendered. The cache
records 18,199 hits / 323 misses, 265,617,248 reused pixels, zero rejected
admissions/evictions, and peaks of 85,673,472 GPU image bytes / 32,532,744 metadata
bytes. Median CPU submission is 35.149 ms and GPU-completion/readback 5.614 ms.
The actual shadow atlas and shader are unchanged; only completed-region validity
uses the subset of page casters reaching receiver bounds, with a conservative
four-shadow-texel margin. Required pages and material/version/offset dependencies
remain represented. The option stays off by default pending broader edits,
visibility, reflection and supported-zoom validation. The next implementation
must address remaining misses and avoid repeatedly calculating identical
dependency subsets; raising budgets is not supported by this run's zero evictions.

The follow-up `receiver-shadow-dependencies-100` diagnostic confirms 323 changed-
dependency misses: main draw fingerprints change in 231, reflected draw in 254,
main shadows in 240 and reflected shadows in 264 (overlapping counts). All 76
misses with an earlier fully covered region image have identical visible pixels;
247 lack comparison coverage. This directs the next experiment toward culling
extrema of actual natural-mesh vertices, since projected 3D boxes include empty
space and inflate both draw and receiver dependencies. Diagnostic timings are
excluded from performance results.

The opt-in actual-vertex projected-bound follow-up (`tight-natural-bounds-100`)
removes only 28 further misses: 295 misses / 18,227 hits, median 55.196 ms,
p95 185.345 ms, p99 275.989 ms, maximum 286.076 ms. All 100 images match the
preceding independent-region renders, but this is a different-DLL reference, not
a new matched independent timing pair. Build and runtime/binary integrity pass;
five region tests pass, including extrema against the actual GroundProjection
at supported zooms, negative coordinates and sloped meshes. Fixed-size extrema
replace projected 3D-box corners only in the experimental culling path; geometry
and shaders remain unchanged. There are zero rejected admissions. This is a
modest follow-up, not a route to the remaining nearly twofold p95 improvement.
Stop further bounding-box refinements as the primary strategy; investigate stable
region scene inputs/preparation across capture changes and repeated dependency
calculation. Native presentation, arbitrary cold/evicted jumps and other zooms
remain unmet. No candidate has been staged.

The projected-bound candidate DLL SHA-256 is
`e1cbd382a23aa2395c8e0104e1a2df3007c64bdbddd649491d297ff16c99ced0`.

The clean ring-four candidate DLL is
`e25e139c85265de233e9e7d0da5057466f2c321ed8677f184e5c038a36d5c406`.
Runtime/binary integrity and completion receipts pass. Six region tests cover
the current support selection, including rejection of topology-only inputs.
It reuses 1,457 prepared tiles per view versus 1,057 in the earlier ring-two
witness. Region images peak at 62,368,512 bytes, metadata at 22,834,772 bytes,
with no eviction or rejected admission. These figures exclude retained meshes,
scratch, driver allocations and native game state. The first `region-input-ring4-100`
run overlapped storage investigation and is timing-diagnostic only; the fresh
`-clean-100` run above is the uncontended measurement.

The installed `Civ3Conquests.exe` PE header is x86 and large-address-aware;
`Civ3Conquests-Unmodded.exe` is x86 without that flag. This read-only check removes
one uncertainty about process addressability, but does not establish live-game
headroom or allocation-spike safety. Neither executable was modified or launched.

Navigation evidence accumulated about 40 GiB of mostly BMP files. User-directed
cleanup scripts preserve reports and recent reference runs; old deleted images
must be regenerated for renewed pixel inspection. Filesystem compression was
stopped when free space declined and is not used by the runner. New runs now
check estimated image output plus an eight-GiB free-space reserve before creating
an output directory. This check covers the filesystem visible to the Python host;
remote native hosts still require their own disk-space verification.

The preceding matched pair is `world-regions-metadata96-100` versus
`world-regions-current-independent-100`; the former contains `comparison.json`.
Both freeze DLL
`163acb0037d9a47181934c9133f3e36632a7e1b8491f5be2649de057088af928`
and executable
`9554855651b964d13f41cef319abe6781d6d639fd0fd91ecad48913640660347`,
with stable runtime inputs and successful completion. Cached p99/max are
526.850/585.588 ms versus 1172.173/1270.996 ms independently rendered. Median CPU
submission falls from 176.690 to 21.403 ms; readback/GPU-completion interval falls
from 716.155 to 5.928 ms. The latter remains a combined interval, not isolated
transfer timing. After-frame free address space stays above 1,831,845,888 bytes,
largest contiguous region 1,704,984,576 bytes; these do not prove allocation-spike,
deferred-driver or live-game memory bounds.

The candidate builds with `/W4 /WX` and unchanged native/shared source closure.
The preceding focused region/telemetry/animation/publication/core suite passes
18 tests; the latest metadata/cancellation candidate passes all five region and
telemetry tests. Executable tests cover cache ownership, pressure, queued-resource
lifetime and dependency changes (off-region casters, alpha state, reflections,
lights, draw order, content and visible geometry). Broader visibility, wrap,
eviction/reset, all-supported-zoom and native integration checks remain required.
Cold map preparation and native completion/pending-view integration remain
separate required work.

Fresh isolated measurements use the exported 100-by-100 `test.biq` map (5,000
tiles), 2240-by-1192 output, fixed noon/season/presentation clock and the current
local packs/shaders. The host is an 8 GiB Windows ARM Parallels VM on Apple
Silicon, four logical processors, Parallels WDDM driver 20.18.2641.57516. Native
witnesses use optimized MSVC x86 builds and large-address-aware executables.
`native/record_navigation_evidence.py` freezes the DLL/executable and records
runtime input hashes, switches, logs, images and exit receipts in a new folder.
Receipts are disposable under `native/build/navigation-evidence-20260909/`.

| Workload / receipt folder | Samples | Median / p95 / p99 (ms) | Finding |
| --- | ---: | --- | --- |
| Original 192 MiB GPU tier, `normal-resident-run` | 0 | unavailable | First view exceeds the geometry cap; retry also fails. |
| Original high tier, `high-profile-resident` | 14 | 313.383 / 1187.936 / 1187.936 | Profiled; zero unchanged static builds/uploads. Too few samples for the plan's distribution gate. |
| Independent high-tier cold, `high-profile-cold` | 14 | 8054.173 / 11957.117 / 11957.117 | Geometry dominates; reset and definition loading excluded. |
| Cached caster bounds + block clipping, `block-clip-resident` | 14 | 167.241 / 935.982 / 935.982 | Unprofiled, waves off; exact against the earlier retained images. |
| Waves enabled, `modern-100-resident` | 100 | 946.377 / 1511.149 / 1577.561 | 50 full rasters; 32 MiB backdrop cap at build time. Static counters exclude wave buffers. |
| Experimental world-anchored raster grid, `world-grid-100-retained` | 100 | 67.434 / 141.960 / 176.037 | Waves off; only one full raster, zero static builds/uploads; 21 requests exceed 100 ms. All 100 images match independent same-DLL cold images byte for byte. |
| Camera queue before bitmap transaction fix, `view-100-retained` | 100 | 965.784 / 1215.764 / 1244.192 | Deliberately superseded requests discarded still-valid raster reuse; 100 full rasters. |
| Camera queue after bitmap transaction fix, `transaction-view-100-retained` | 100 | 120.114 / 162.939 / 207.811 | Same workload; one full raster, zero static builds/uploads. All 100 images exactly match independent cold results; 113 publication identity checks pass. |
| Whole prepared views, `full-view-128-100` | 100 | 875.448 / 945.686 / 968.047 | Pixel reuse disabled deliberately; 100 full rasters, zero static builds/uploads. Current full rendering remains far above the 100 ms target. |
| Larger-region whole prepared views, `full-view-512-100-complete` | 100 | 697.394 / 774.149 / 781.632 | Same DLL/clock/content, zero static builds/uploads. CPU submission median falls to 67.389 ms, but the readback/GPU-completion interval remains 608.784 ms. |
| Same-build full-view water-coverage control, `water-control-full-view-128-100` | 100 | 948.532 / 1265.777 / 1365.058 | All pixels rendered, waves off, coverage culling disabled. |
| Empty-water/reflection rejection, `water-coverage-full-view-128-100` | 100 | 808.053 / 864.015 / 884.902 | Same DLL/executable/input hashes, all pixels rendered; all 100 images byte-identical to control. Still far above target. |
| World-grid pans with waves, `world-grid-waves-100` | 100 | 528.288 / 769.447 / 1087.881 | One full terrain raster, zero static builds/uploads; wave geometry and backdrop work remain expensive. |
| World-aligned backdrops redrawn independently, `world-backdrops-100-fresh` | 100 | 657.049 / 1064.858 / 1448.105 | Same current DLL/grid/clock; all 5,153 static animation regions redrawn. |
| Retained world-aligned backdrops, `world-backdrops-100-retained` | 100 | 432.654 / 859.108 / 1302.410 | 3,026 hits and 2,127 misses; all 100 images byte-identical to the independent-backdrop control. |
| Independent coast-cell wave rebuilding, `world-wave-cells-100-fresh` | 100 | 310.100 / 608.876 / 687.331 | Same current DLL/projection/backdrop settings; 87,300 cell queries rebuilt and 1,010,930,688 wave-buffer bytes uploaded. |
| Retained coast-cell waves, `world-wave-cells-100-retained` | 100 | 258.975 / 587.893 / 666.435 | Zero wave/static mesh builds or uploads; 87,300 cell hits; all 100 images exactly match the wave-rebuild control. |
| Independent per-region caster preparation, `composition-casters-100-control` | 100 | 215.118 / 509.632 / 566.567 | Same DLL and wave/backdrop retention; 38–127 caster preparations per frame. |
| Shared composition caster preparation, `composition-casters-100-shared` | 100 | 194.264 / 461.097 / 512.428 | Two preparations per frame; all 100 images exactly match control; zero static/wave builds or uploads. |
| Same-build whole-view control, `whole-view-128-matched-100` | 100 | 869.901 / 976.769 / 1064.910 | 100 complete prepared views; waves and pixel reuse off; no test/build work overlapped timing. |
| Bounded rectangular strips, `whole-view-strips-uncontended-100` | 100 | 653.641 / 708.204 / 738.716 | 2240-by-256 regions; same DLL/content; zero static builds/uploads. Small pixel differences remain, and the 100 ms gate fails. |
| Complete strip post-processing, `bounded-post-control-100` | 100 | 660.675 / 719.876 / 749.117 | Same current DLL/shader inputs, full prepared views with waves/pixel reuse off. |
| Guarded strip post-processing, `bounded-post-stable-100` | 100 | 633.708 / 689.040 / 707.643 | All 100 images exactly match control; glow dispatch work falls 57.8%, but p95 improves only 4.3%. |

All times above end at standalone render/capture completion, **not native
presentation**. The current experiments do not meet 30 FPS or the warm p95
100 ms target. The fixture does not establish animated-resource or native-unit
coverage. The 1,000-presented-frame gate, all-three-supported-zoom distributions, arbitrary
cold/evicted destinations and live game-thread bookkeeping remain unmet.

The composition comparison shares one bounded immutable caster preparation
across static backdrop and animated-region submissions. It extends the existing
recursive submission sharing; it does not introduce a persistent shadow cache.
Both runs use DLL `9cae17c50683a3ecf8e2f534ac2109188af6dcafff9fdd2d3b6603888f7dd723`
and matching frozen runtime inputs. Median improves by 9.7% and p95 by 9.5%;
maximum is 1321.001 ms shared versus 1238.962 ms control, so worst-case latency
does not improve. The independent control rebuilds caster preparation over the
same retained static base and backdrops, not a wholly cold scene. The focused
suite at this checkpoint has 52 passing tests and 17 compiler-path skips.

The rectangular strip experiment lowers whole-view p95 by 27.5%, but it is
still 7.1 times the 100 ms target. CPU submission median falls from 166.262 to
47.888 ms (71.2%), while the GPU-completion/readback interval falls only from
669.597 to 588.372 ms (12.1%). Total maxima are 1083.621 ms control and
782.960 ms strips. This is evidence that repeated submission is expensive,
and equally that reducing it alone is insufficient. The next full-view work
must reduce GPU pass/resolve/reconstruction cost, including unused scratch
area, rather than count another small-pan cache improvement as that solution.
The VM's delayed GPU timestamps remain unusable for precise pass attribution.

The strip pair uses DLL
`7b76d8357940868fd2b03e39cbf7aba4d3378e6d556877c7c53b87112f91cfb5`,
with matching frozen inputs and a `/W4 /WX` build whose native/shared source
closure was unchanged. Every image differs from the 128-pixel control: at most
434 of 2,670,080 pixels and 5/255 in a channel. A focused/context comparison is
saved beside `comparison.json`; this is not visual acceptance. The default
remains 128-pixel regions. The earlier `whole-view-strips-100` run is diagnostic
only because test compilation overlapped it; it is excluded from these timings.

After-frame address-space samples in the uncontended strip sweep retain at
least 1,914,060,800 bytes free, with a largest free region of 1,783,627,776 bytes.
These are standalone after-frame samples, not allocation-spike or live-game
reserve verification. Rectangular scratch preserves MSAA4, 2x reconstruction
and pixel-sized guards, with separate 128-pixel animation scratch. The focused
current suite has 53 passing tests and 17 compiler-path skips, including actual
C++ clipping checks for independent strip axes; full integration and all-zoom
stress remain pending.

The bounded-post pair shows that unused glow/conversion work is a secondary
cost, not the missing large speedup. Across 100 complete views, compute lanes
fall from 664,688,640 to 280,592,256. Median CPU submission is almost unchanged
(48.474 versus 47.297 ms), and GPU-completion/readback median falls from
587.947 to 564.660 ms. Maximum total time is worse in this pair (765.728 versus
787.392 ms); do not claim improved worst-case latency. The next rendering
investigation must address scene/reflection/resolve work rather than continue
optimizing glow as though it dominates. Native presentation remains unmeasured.

Both valid runs use DLL
`cb5324c0acf5403261e9ed2aded9725175406b5bdb24be525ac57208cf19a696`
and matching stable runtime inputs. All 100 control images also exactly match
the previous shader's strip outputs. `bounded-post-100` is excluded because
its first execution refreshed `hdr_glow.hlsl.CSPost.cso`; only the fresh stable
pair above passed input-integrity checks. The shared shader source and native
adapter preserve the original workgroups by using an eight-pixel-aligned
dispatch origin. Scratch dimensions, MSAA resolve and filter quality are
unchanged; accumulated/non-strip paths retain full reconstruction. The feature
is opt-in. The current focused suite executes 60 passing tests with 17
compiler-path skips, including six city pickup checks and actual C++ dispatch
coverage/coordinate tests. The build passes `/W4 /WX` with unchanged native/shared
build inputs.

The bitmap transaction fix preserves a completed CPU image when cancellation
occurs before the next readback changes it. Interrupted geometry assembly still
invalidates independently. Identity, ordered captured occurrences and ownership
now publish atomically through optional versioned camera-view exports. Unit
takeover pauses/resumes the requested map instead of unconditionally losing it.
These exports are not bound by the injected bridge. Queue begin p95 was 1.175 ms
over 113 calls in the transaction witness; this is not total game-thread
capture/submission/poll/composition cost. Its 120 ms total includes the witness's
deliberate obsolete request and 10 ms pause. The before/after images agree with
the previously frozen independent cold DLL, not a new same-DLL cold sweep.

For the whole prepared-view control, median capture was 1.793 ms, geometry
assembly 2.306 ms, CPU draw submission 174.927 ms and readback interval 665.600 ms.
The last interval includes GPU completion. These measurements establish that
removing cold mesh construction alone cannot meet the arbitrary-jump objective.
The next decision points are larger-region/full-view cost, widest-zoom memory,
animated-water cost and missing-region preparation, measured separately.

The 512-pixel experiment reduces whole-view p95 by 18%, insufficient to close
the nearly eightfold remaining gap. Compared with 128-pixel regions, all 100
images differ: at most 430 pixels per 2,670,080-pixel image, with maximum channel
difference 5/255. Keep this path experimental; it is not an exact-parity win or
accepted appearance change. The earlier `full-view-512-100` directory was
interrupted before its completion marker and must not be treated as a complete
distribution receipt. Only the `-complete` run above passed execution/input
integrity checks. Larger blocks alone are not a viable complete solution.

The water-coverage experiment omits bed/water geometry only when every actual
uploaded hydrology sample has positive land distance beyond a conservative
margin. Unknown/nonfinite samples retain their passes. Regions with no drawable
water skip their unused reflected scene. Two executable/contract tests pass,
and the 100-frame paired full-view witness preserves exact output. Median CPU
submission falls from 200.394 to 140.778 ms in that pair. The control also shows
greater variation in unchanged capture/geometry work; do not extrapolate this
single pair's percentage to other hardware or the complete experience. It is
useful work elimination, not evidence that full-view rendering meets the goal.
The switch remains opt-in. Its DLL SHA-256 is
`9d6d9388f20e763a04c91110b16d722fbdbea42529755368f5cf2a68b3608fad`;
the isolated build passed `/W4 /WX` with unchanged native/shared source inputs.
Telemetry and water-coverage tests are now included in the renderer workbench's
integration test selection.
The focused run at the water-coverage checkpoint executes 65 tests: 48 pass and
17 skip for unavailable compiler paths; there are no failures or errors in
this set (`navigation-current-native-tests.log`). This includes the actual
MSVC publication/query/coverage/render-core witnesses. The broader integration
limitations recorded below remain outstanding.

The latest waves witness uses the 128 MiB backdrop tier and current raster
transaction fix, with water-coverage rejection disabled. Its trace reaches the
last frame: 30 wave ribbons, 38 backdrop regions, zero backdrop hits, 38 misses,
131,395,584 backdrop bytes and 5,944,320 newly uploaded wave-geometry bytes.
The dynamic composition alone takes 310.264 ms on that frame. Static mesh
counters do not include those wave buffers. Both `prepare_wave_chunks` and
backdrop lookup still use the complete frame signature, so camera changes
invalidate otherwise reusable data in that baseline. Increasing backdrop
capacity alone cannot fix those camera-invalidated entries. The waves sweep
has no independent whole-scene cold comparison yet.

The next experiment removes camera anchors from backdrop identity and places
regions on the world raster grid. It retains conservative whole-capture scene
and ownership identity, target/zoom, light, wrap, content/device generation and
the topology revision even when a request carries no new topology payload.
Negative or overhanging region origins remain intact for caching and rendering;
only the final native-size output copy is clipped. Two actual MSVC tests cover
the grid's complete/non-overlapping output coverage, all-five-zoom translation
keys and conservative static-identity changes. The frozen build is
`navigation-world-backdrops-topology-20260909`, DLL SHA-256
`04d03081c6d9c59d921ae84add93f95cdc41f3b5d66d4d9646968aa5cbe2bf7c`;
`/W4 /WX` compilation and unchanged-source build receipts pass.
The current focused native test run passes 50 executed tests, with 17 compiler
path skips and no failures/errors (67 total, `navigation-backdrop-native-tests.log`).

The paired 100-pan witness above uses identical DLL/executable/input hashes,
waves and region placement. Reuse avoids 59% of static animation-region
rebuilds, lowers median completion by 34% and p95 by 19% in this pair, and
preserves every output byte. Backdrop textures peak at 131,395,584 bytes within
the unchanged 128 MiB cap. This control independently redraws each animation
backdrop over retained terrain; it is not an independent geometry-cold scene.
The new grid also differs from the previous screen-aligned animation grid, so
do not treat the older waves run as a matched timing or appearance control.
Whole-capture changes and capacity still cause misses; wave ribbons still use
complete-frame identity and rebuild/upload during pans. This is an opt-in
intermediate improvement, not a warm-latency, arbitrary-jump or native pass.

The wave-cell experiment retains immutable cell-local buffers and cached empty
cells across views, with a 32 MiB GPU-buffer cap and 16,384 metadata-entry cap.
The complete current captured set pins its existing cells before admission or
eviction; cached off-screen cells alone never make a visible occurrence. Raw
wrapped coordinates preserve source world/shadow placement. Whole topology
revision, dimensions/wrap, target/zoom, content and device generations remain
conservative invalidation boundaries. Light and animation time do not change
wave geometry. RAII owners release partially allocated cells on failure, while
active views retain their own references. The old 16 MiB path now explicitly
fails overflow rather than silently omitting ribbons.

The matched 100-pan test eliminates 1.01 GB of repeated uploads and all 87,300
cell rebuild queries, while preserving every output byte against independent
cell rebuilding. The pool peaks at 33,486,336 GPU-buffer bytes and 3,464 entries.
Median completion improves 16%, but p95 improves only 3% and remains 588 ms:
wave preparation is no longer the main obstacle in this prepared workload.
Full-view draw/submission cost and remaining backdrop misses still require work.
These are isolated comparisons; do not combine percentage changes across runs.

Cell-local arithmetic differs from the previous projection in at most 72 pixels
per image over the 100-view comparison, all by one channel level. Keep the path
experimental pending broader visual/zoom/edit/pressure coverage. The reference
rebuilds wave cells over retained terrain/backdrops; it is not a whole-scene cold
or native-presentation test. DLL SHA-256:
`74e720c988e59cb18030284d71a82c6b416b66cfd9c4f2cecde5c106d6cc3a13`,
from `navigation-world-wave-cells-20260909`; `/W4 /WX` and source-closure receipts
pass. The latest focused native suite runs 69 tests: 52 pass, 17 skip, with no
failures/errors (`navigation-wave-retention-native-tests.log`).

The previous retained/cold pixel discrepancy is isolated to bitmap translation:
`raster-control-resident` disables translated overlap/block reuse and matches all
14 independent cold images exactly with retained geometry. This excludes the
retained meshes as the cause in that witness. The world-grid experiment keeps
raster-region coordinates attached to authoritative world anchors; it remains
off by default pending exact comparison and visual/context review. Larger
256-pixel regions did not improve total latency and changed a few pixels; they
also remain opt-in. No shading, detail or reference acceptance gate was relaxed.

Delayed timestamp/disjoint queries use eight fixed slots and non-flushing polls.
This VM returns implausibly tiny GPU timestamp intervals despite valid status,
so they cannot support GPU speedup claims here. A separate 100-copy D3D11
readback-floor witness measured 2240-by-1192 copy/Map/CPU-copy median 2.751 ms,
p95 10.550 ms, p99 11.018 ms. It excludes scene rendering and native blitting;
the much larger renderer Map interval includes outstanding GPU work.

With the user's explicit higher-memory authorization, the candidate default is
768 MiB GPU geometry, 96 MiB CPU natural data, 32 MiB viewport cache and 128 MiB
resource backdrops. These are separate caps, not a process budget. The earlier
high tier used 192/128/288 MiB for the latter three categories. Sampled free VA
was above 1.83 GB in the profiled high-tier retained sweep; this is not a live
game reserve or peak guarantee. Allocation-point VA/capacity telemetry and
separate wave-buffer accounting are now available; the complete simultaneous
CPU/GPU/publication/driver inventory and pressure envelope are still required.

The profiled widest-zoom witness `widest-64-memory` now completes warmup and
100 distinct pans at tile width 64, up to 3,443 captured tiles, without fallback,
static mesh builds or geometry uploads in the resident sweep. Across 474
allocation/phase/reset memory samples, retained GPU geometry peaks at
734,584,218 bytes (700.6 MiB), combined natural/ground CPU caches at 16,806,416
bytes and viewport cache at 26,404,640 bytes. Minimum sampled free address
space is 1,610,506,240 bytes; the smallest largest-free-region observation is
1,531,904,000 bytes. This supports the 768 MiB geometry tier for this standalone
map/zoom witness. It is not an inventory of all GPU allocations, a live Civ III
reserve, a leak test or an all-five-zoom pressure pass. Waves are disabled.
Profiled completion median/p95/p99 is 230.375/330.471/406.363 ms; frequent memory
sampling overhead is included. No independent cold pixel comparison at this
zoom has yet been completed, so this is a memory/residency result, not a visual
parity or latency pass.

The tested bitmap-transaction DLL SHA-256 is
`ec94dc08ab391df4dc3385589de0f58dd610a35d0d3913e4b3f8f3445d8be988`;
the frozen build receipt records unchanged native/shared source hashes and
successful MSVC `/W4 /WX` compilation. Focused telemetry, publication, resident
witness and bridge tests ran 61 cases: 44 executed successfully and 17 skipped.
Two additional render-core tests passed with actual MSVC numerical witnesses.
The broader shadow integration test set does not pass in this Windows setup
(217 cases, one failure, 18 errors, 23 skips); the remaining output includes
Unix-only compiler/locking assumptions, symlink permissions and path handling.
Do not interpret focused success as a full integration or live-game pass.

Before the bitmap-only cancellation change, the current shared-shadow lab run
rendered all 16 focused/context, time and zoom combinations without fallback.
Four matched noon/midnight focused/context images were byte-identical to the
frozen pre-change renderer. Optional fixed-reference comparison could not run
because its BMPs are absent. No appearance acceptance, staging or live pass is
claimed by this evidence.

## Exact height reuse and new-view witness

The current candidate adds bounded exact-coordinate height/support reuse within
one tile compilation. All terrain-height consumers in the native natural-mesh,
cliff-placement and site-placement path use the same sampler. Its table clears
before each tile, preserving the owner-specific dependency scope; it is not a
persistent cache of world heights. Capacity is at most 16,384 slots (393,216
bytes observed). Exhausted admission computes normally, without quantization,
interpolation, changed mesh density or changed visibility. The same-DLL control
is `C3X_RENDERER_HEIGHT_CACHE_CONTROL=1` (reuse disabled).

The extracted production sampler passes tests for exact values and support,
nearby unequal coordinates, source/dependency scope changes, disabled control,
bounded admission and exhaustion. Together with shared natural-query/relief
checks, 58 focused tests pass. DLL compilation succeeded; SHA-256:
`95971dd834166317f6c3559f61665113b3fbc9f2500112acf6bdfddf82160e1f`.
The 2240x1192 control finished all six destinations and revisits. The enabled
run completed only four destinations; all four saved BMPs are byte-identical
to control. Initial compilation recorded 468,223 reused height results, with
2,356,134 misses. This is evidence of avoided queries, not a speedup claim or
completed native verification. Cold geometry still takes seconds.

Native verification was interrupted on September 8: Windows Defender reported
`Trojan:Win32/Bearfoos.B!ml`, threat ID `2147731849`, against
`native/build/retained-height-samples/biq_preview.exe` and removed that file.
The enabled replay ended without its completion marker, and Windows confirmed
no remaining preview process. This is not established as a false positive or
as a renderer crash. No protection setting was changed and no retry was run
after identifying the detection until the user explicitly allowed it and
requested another attempt. Native execution has now resumed; see the paired
results below. Logs remain in `native/build/retained-height-control/` and
`native/build/retained-height-candidate/`; the prepared retry folder was not run.

An opt-in `C3X_RENDERER_PREVIEW_RESIDENT_SWEEP` witness has been added to
`biq_preview.cpp`. After the ordinary navigation warmup it requests fourteen
new camera positions between the two loaded vertical views, reporting actual
tile builds, reuse, upload bytes, and phase timings. It does not assume those
views are cheap or silently count old screenshots as responses. The updated
witness has now been compiled and run on Windows after the user's allow action.
No DLL staging, INSTALL, or game launch was performed by this work.

The witness now also supports `C3X_RENDERER_PREVIEW_RESIDENT_COLD=1`, which
resets renderer state and reloads definitions before each intermediate view to
produce independent reference pixels. Both modes record viewport/camera identity,
completed image checksums, builds/uploads and render-plus-capture latency. Image
writing and checksumming are outside the measured interval. Cold reset/definition
loading is also outside that interval; cold timings are not launch-time results.
Run each invocation in a fresh isolated output directory with the same DLL,
fixture, assets, environment and presentation clock. The read-only verifier is:

```sh
python3 -m Renderer.native.analyze_resident_navigation \
  --reference Renderer/native/build/resident-cold \
  --candidate Renderer/native/build/resident-retained --max-ms 100
```

Those directory names are illustrative; actual completed outputs are below. The
verifier requires complete warmup/revisit logs, all fourteen new cameras,
successful exit receipts, same-DLL identity and matching image receipts. It
checks exact cold/retained pixels, zero retained builds/uploads, and every
retained render-plus-capture time against the target. It rejects malformed,
partial or stale-image evidence and distinguishes incorrect pixels from invalid
receipts. Its synthetic tests pass; they do not establish native performance.
This narrow resident-content test cannot certify arbitrary-map navigation,
cold-region response, or Civ III input-to-display latency.

### Allowed retry: completed native resident/cold comparison

After the user allowed the detected program and requested a retry, Windows
reported the threat-specific action entry and both benchmark processes completed
with exit zero. No Defender setting was changed by this work. A default-budget
build compiled but was not run; the measured candidate uses the previous
high-memory tier (768 MiB GPU geometry budget). Build and output root:
`native/build/resident-review-high-20260908/`, with independent `retained/` and
`cold/` directories. Both use DLL SHA-256
`708f9bc7da20dc4580a9890209b9e0793b12b8c40c16f999e7c69f425848ff0e`.
The benchmark executable SHA-256 is
`ed063f4a138ebb75a0f4c4aa3975aa31300ec3eb476250dcfeddf778e78e6c34`.
Renderer/benchmark source, fixture, definitions and runtime shader-tree hashes
were unchanged across the pair. Waves are disabled in both processes; this does
not qualify wave animation. The 57 cache/publication/bridge tests and two shared
natural-query/relief tests pass locally.

At 2240x1192, all fourteen new retained views reused 1,057 tiles, built zero
tiles and uploaded zero geometry bytes. Retained capture-plus-render latency:
minimum 201.966 ms, median 235.560 ms, maximum 745.483 ms (first new view).
Median geometry preparation is 31.404 ms, draw submission 60.483 ms and readback
wait 120.354 ms. Phase medians do not sum to the total median; readback includes
pending GPU work and is not a pure transfer-time measurement. Later views reuse
roughly 2.5 million pixels while requesting about 161,280–174,592 new pixels,
but still issue 72–82 shadow passes across render blocks/reflections. No claim
that all this work can be removed follows from the pass count alone.

Fresh references build all 1,057 tiles per view and take 7,769.508–9,592.342 ms
(median 8,527.645 ms), dominated by geometry preparation. Reset/definition loading
is excluded, as noted above. Both ordinary six-destination/revisit workloads
also complete with zero pixel errors on revisits, fallback or device recovery.
Sampled available virtual address space stays above 1,690,443,776 bytes across
the pair, with a largest free region above 1,597,050,880 bytes. These are sampled
observations, not hard minimum guarantees for the game or other maps.

The stricter independent comparison **fails**: only the first retained image is
byte-identical to its cold reference. Other frames differ in 4–180 of 2,670,080
pixels; four frames each have one pixel with a maximum channel error of four,
and all remaining differences are at most two. These differences meet the older
benchmark's tolerance, but the new exact test was not relaxed. The diagnostic
report is `comparison.json`: residency passes, exact parity and the 100 ms
latency target fail. Small numerical differences still need explanation; this
is not an accepted visual change or completion of the performance goal.

Next work should target block/reflection submission and GPU completion for
resident scrolling, alongside the separate cold tile-preparation cost. Raising
caps or adding more screenshot-cache hits does not address the measured warm
bottleneck. No staging, INSTALL or Civ III launch was performed.

## Retained draw-request implementation

The active objective is near-instant navigation anywhere, including new views,
not only repeat bitmap-cache hits. Civ III's callbacks and supplied anchors
remain authoritative. This work is not complete and has not been staged.

First implementation: `submit_geometry` now collects static shadow-caster
descriptors once per outer submission. Its block and reflection passes borrow
that immutable list while retaining their own receiver selection. Existing GPU
geometry owners remain pinned; the descriptor list expires before the call
returns, so later content edits, unit/resource work, resets and evictions cannot
reuse stale descriptors. This changes neither shaders nor raster quality. It is
a submission optimization, not a claim that all tile geometry is camera-neutral.

The actual extracted collector passes executable tests for caster selection,
ground/animated exclusions, order, material bindings, buffer/version identity,
both wrap axes, empty scenes and subsequent content edits. All 55 focused
cache/publication/bridge tests pass; the worker mock now implements the current
ambient-count accessor used by the concurrently updated renderer.

Isolated Windows control/candidate DLLs:
`native/build/retained-submit-ab-control/` and
`native/build/retained-submit-ab-candidate/`. SHA-256 respectively:
`2f253586f6b3390fa3c35a243fe61c37d0f1f24e9796f9febdd152d29b4819f9` and
`973bb38cf270510efd3ab1d94c3b7f9b938a986a5cb992c5609c4f2ef8366053`.
Both compile with the existing high-memory tier. The full 2240x1192 navigation
workload has six destinations and six revisits, all pixel-exact across candidates
and within each run, with no fallback or device recovery. The candidate builds
75 caster lists for 1,992 shadow passes; the control rebuilt a list in every
pass. Runtime shader-tree hashes remained unchanged during the comparison.
This is a small correctness/performance experiment, not broad navigation proof.

The first control was affected by colder inputs: cold-view median 5,646.428 ms
versus candidate 4,709.154 ms is not attributable entirely to this change.
Candidate revisits ranged 9.114–95.147 ms but often reuse complete bitmaps; those
numbers do not prove cheap new camera views. A fresh-process, warm-input control
is retained separately at `native/build/retained-submit-ab-control-warm/`.
It completed with cold-view median 4,850.396 ms versus candidate 4,709.154 ms;
the candidate was slower at the final wrapped destination and on several
bitmap revisits. One control/candidate/control sequence is insufficient to
claim a significant overall speedup. CPU submission and GPU completion overlap,
so lower draw-submission time alone is not a GPU-speedup measurement. These
results direct the next work toward geometry preparation and genuine new-view
reuse, not an assertion that descriptor caching solved the bottleneck.
The obsolete frozen-input attempt failed before drawing because it lacked
`PSCoastalWave`; its failure log remains under `native/build/retained-submit-control/`.
Current-input runs disable waves in both processes to isolate this work from
concurrent wave development. They do not validate wave rendering.

Remaining: remove screen-dependent reconstruction from other currently owned
tile components, reduce block/page submission repetition, add many distinct
camera positions over resident content with rebuild/upload assertions, and
address genuinely uncached regions. Cold geometry preparation still dominates
multi-second travel. Broad zoom/edit/wrap/eviction and live bridge timing must
be verified before declaring the requested outcome achieved.

## Test handoff

Experiments are paused for the user's in-game evaluation. The verified
high-memory DLL is staged; see **Current install staging** below. The game still
uses synchronous rendering: the optional camera queue and coarse previews are
not connected to injected hooks or enabled for this test. Their standalone
first-image timings below are **not in-game performance claims**. Cold views
still take seconds. Test the existing five Z zoom levels and minimap movement;
this handoff does not claim smooth transitions or a completed performance goal.

## Current-camera terrain preview experiment

`C3X_RENDERER_CAMERA_PREVIEW=1` enables a CPU terrain-only provisional image in
the optional camera queue. It uses the new request's captured anchors, render
flags and normalized base-color textures, never an unchanged prior-camera
bitmap. `PREVIEW` is distinct from `OK`; the caller must keep polling for the
full scene. Only terrain replacement bits are set. Objects, overlays and all
detailed geometry remain outside preview ownership. The injected game bridge
does not consume this result yet; the preview is disabled and not visually accepted.

The first Windows candidate (`camera-terrain-preview`, `6eec1285...`) produced
first images at 2240x1192 with these request-to-first-image times:

- Minimap: 37 requests, median 33.494 ms, p95 55.391 ms, maximum 85.056 ms.
- Zoom: 36 requests, median 35.577 ms, p95 65.872 ms, maximum 82.380 ms.

These include the initial request. Five navigation requests completed before a
preview was observed; their full completion times count as first image, not
missing/zero samples. All five independent zoom images, six navigation images
and 30 repeats per scenario were exact at final completion against the prior
queue candidate. The receipts are under `native/build/camera-terrain-preview/`.
Cold final detail still takes seconds, so first-image latency is not substituted
for the final-completion target.

Visual inspection rejected raw water base textures as a useful preview: they
are seabed textures and looked sandy. The follow-up adds an approximate optical
depth per captured water family using the renderer's existing absorption/tint
response, and shades each material once instead of once per tile. It uses the
shared captured environment; it is not an exact production-shader approximation
or a claim about source-game optics. The corrected view remains a flat,
terrain-only draft with hard tile transitions—not accepted visual quality.

Follow-up DLL SHA-256:
`6a57b54f4877224e16d966e8bd4a50fb668aa8a06f94d9791d19a31131f20899`.
Compilation succeeded; the Windows guard paused verification while Civ III
was running. After the game closed, smoke with the switch on/off and both
full-size camera workloads completed on this exact DLL. First-image median /
p95 / maximum is **34.714 / 38.655 / 38.669 ms** for zoom (36 requests) and
**34.594 / 42.307 / 48.661 ms** for navigation (37 requests). The measured CPU
preview routine itself has median 23.722 ms for zoom and 23.600 ms for navigation;
polling and cancelled prior work account for additional first-image latency.

All five independent zoom images, six navigation images and 30 repeats per
scenario are exact at final completion against the earlier no-preview queue.
Receipts: `native/build/camera-terrain-preview-optics/zoom/comparison.json` and
`native/build/camera-terrain-preview-optics/navigation/comparison.json`.
Warm full-completion p95 is **161.293 ms** for zoom and **137.134 ms** for
navigation; cold medians are **2,794.500 ms** and **4,907.174 ms**, respectively.
Thus the coarse image improves first response, but does not satisfy the detailed
cold-completion target, sustained-input smoothness or game integration.
No fallback/device recovery occurred. Sampled free VA stayed above
1,594,478,592 bytes, with a largest free region above 1,503,002,624 bytes.
Mixed zoom/navigation plus active units is not yet a covered stress workload.
All benchmark/build processes finished; the user's game was left alone.
The 54 focused checks and 131 selected grassland checks pass locally.

`native/camera_preview_cpu.py` executes the actual production preview routine
against an explicit generic pack and BIQ fixture without a VM. The corrected
2240x1192 local preview took 7.467 ms; this is local CPU evidence, not Windows
latency or a full-renderer baseline. Its pixels match the Windows preview at
this exact navigation destination; that single comparison is not a general
cross-platform renderer qualification. Image/receipt:
`native/build/camera-terrain-preview-optics/preview-current.png` and `.json`.
Executable tests cover anchors, zoom, terrain-only ownership, halo exclusion,
water-family selection, shared time, malformed DDS, bounded output and extreme
off-screen coordinates. Final-image quality, preview usefulness during rapid
input, native redraw integration and cold-detail speed remain unfinished.
The experiment was subsequently staged with preview disabled for the user's
evaluation; see the install handoff below. No INSTALL or game launch was run.

## Cancellable camera requests (standalone experiment)

The optional DLL camera begin/poll/cancel exports now run on the existing D3D
worker. There is one immutable active snapshot and one replaceable pending
snapshot, with a monotonic ticket fencing completion. Begin copies both tile
records and authoritative world topology. Older tickets never return success;
poll only publishes the matching complete image and its ownership arrays.
No game hook uses these exports yet, and the synchronous ABI remains intact.

The caller owns the front publication; the worker can only fill the separate
back publication. Poll swaps them under the call/state gates. Published frames
retain the 32 MiB-per-owner cap; the back is cleared before another capture, so
front plus candidate is bounded to 64 MiB. Input counts retain the existing ABI
limits (8,192 tile records and at most 2,097,152 packed topology cells per
snapshot); replacing a pending capture temporarily holds its old and new
arrays alongside the active snapshot. This is bounded storage, not an
unlimited camera history or a total-process memory guarantee.

Synchronous render, configuration, units and reset cancel/drain camera work
before mutating shared renderer state. Intentional cancellation invalidates
partial viewport/draw assemblies without resetting the device or deleting the
individually validated tile cache. Unit takeover currently supersedes the
camera request; automatic resumable refinement remains future work. The
caller must explicitly request a new frame after such a takeover.

The first candidate (`camera-queue`, `9ca86f5e...`) accepted 37 minimap requests
in 0.440 ms median / 0.854 ms maximum, rejecting all 37 obsolete tickets.
All five zoom images were exact against the same DLL's synchronous path;
navigation passed the existing pixel tolerance, with aggregate channel error
4 in one image and zero in the others. All 30 repeats per scenario were exact.
However, the supersession workload exposed uncancelled GPU submission: warm
zoom p95 was 1,005.179 ms and minimap p95 was 187.242 ms. Those include deliberately
starting and replacing an earlier scene, unlike the synchronous control, so
they are not a clean asynchronous-overhead measurement or a speedup claim.

The follow-up forwards the cancellation flag into the existing interruptible
draw/shadow path and checks it before submission/readback. Candidate:
`native/build/camera-queue-draw-cancel/C3XRenderer.dll`, SHA-256
`38751820606b5ef654e39c27131ac989beaffcaa1617d148428738c0f794c72b`.
Native smoke passes with publication on and off. The actual worker runs in a
portable threaded test with deliberately blocked rendering, 30 replacements,
copied input mutation, immutable full-size front pixels, unit takeover and
reset/join. All 53 focused tests and 131 selected grassland tests pass on the
final source. All verification processes finished; staging is unchanged.
The full-size follow-up uses the same
deliberate supersession workload as the first queue, including 30 warm revisits
per scenario:

| Queued workload | First queue p95 / max | Interruptible draw p95 / max |
| --- | --- | --- |
| Five-level zoom | 1,005.179 / 1,489.096 ms | 132.305 / 142.673 ms |
| Minimap movement | 187.242 / 816.987 ms | 101.130 / 104.467 ms |

Latest request acceptance maxima were 0.945 ms (zoom, 36 requests) and 0.576 ms
(navigation, 37 requests); every obsolete request was rejected. Cold-change
medians remain 2,592.786 ms for zoom and 4,839.132 ms for navigation; maxima were
5,342.687 and 5,517.835 ms. The initial asset-loading request took about 12.6 s
and is reported separately, not hidden inside the cold-change median. There
are too few cold samples to establish p95.

The five independent zoom images are exact against the first candidate's
synchronous control. Navigation passes the same tolerance as above (aggregate
channel error 4 at one destination, otherwise exact); all 30 repeated images
per scenario are exact. This comparison crosses the draw-cancellation code
change and is not described as a same-DLL A/B. Receipts:
`native/build/camera-queue-draw-cancel/zoom/comparison.json` and
`native/build/camera-queue-draw-cancel/navigation/comparison.json`.
Geometry residency remains 685,584,054 bytes for zoom and 611,114,938 bytes for
navigation; no fallback or device recovery occurred. Minimum sampled free VA
and largest free-region sizes are recorded in those receipts. Mixed zoom plus
distant-navigation stress and sustained unit animation are not covered here.

This removes a synchronous request/ownership obstacle, not the user-visible
stall yet. Request acceptance is **not map response**: a correct new-camera
preview, scheduled final redraw and sustained input/animation coordination
remain necessary. Cold geometry work is still multi-second; GPU readback and
some asset/mesh phases still lack bounded cooperative cancellation. The staged
evaluation DLL remains `91b5e315...`; no INSTALL, game launch, injected source
change or new Civ III patch-table entry was involved.

## Current install staging

At the user's explicit wrap-up/staging request, the verified
`camera-terrain-preview-optics` evaluation DLL was copied to
`Renderer/bin/C3XRenderer.dll`. Candidate and staged SHA-256 hashes match:
`6a57b54f4877224e16d966e8bd4a50fb668aa8a06f94d9791d19a31131f20899`.
The previous staged DLL (`91b5e315...`) is preserved at
`native/build/install-backup-iCT0xD/C3XRenderer.dll`; the older `dd7b9d5b...`
backup remains at `native/build/install-backup-siha3z/C3XRenderer.dll`.
Civ III was confirmed closed before staging. All 54 focused tests passed again;
the exact candidate already passed Windows native smoke with preview on/off,
131 selected grassland tests, and full-size zoom/navigation final-image checks.

This retains the high-memory cache build. Both
`C3X_RENDERER_CAMERA_PUBLICATION` and `C3X_RENDERER_CAMERA_PREVIEW` default off;
no environment switch was enabled. The asynchronous camera API is experimental
and not called by the injected game bridge. Cold views remain multi-second.
Runtime assets, configuration, references and injected code were not changed by
staging. INSTALL and game launch remain for the user; neither was run here.
Earlier statements that staging was untouched describe their individual passes.

## UI-owned blit lifetime

`MapBlitter` now owns the GDI bitmap/DC and destination rounding tables, separately
from RendererState. Its calls and cleanup run on the calling thread; worker
shutdown joins before releasing these UI resources. It consumes the completed
output and captured rounding phase, not mutable tile/camera data. The shared
trace sequence is atomic for eventual concurrent render/blit diagnostics.
Correction to the earlier investigation: ordinary `RendererState::reset` did
not destroy its GDI surface; its destructor did. This refactor removes that
ownership coupling without claiming a previously unobserved device-reset race.

Candidate SHA-256:
`91b5e31540bcb49fd01425e0cadc0d9c9e6c659a19013e1f0389b98d26bb5ece`.
The opt-in publication path passes all six full-size navigation images and
30 repeats exactly against the prior flat-shore candidate. This is cross-build
functional parity, not an isolated performance A/B: publication is enabled only
in the new run. Warm navigation measured **37.296 ms median / 74.213 ms p95 /
76.941 ms maximum**; cold median remains **4,805.152 ms**. Receipt:
`native/build/camera-ui-blitter/comparison.json`. There is no new latency win
claimed here; responsive/background rendering is still not enabled.

The executable UI-resource test runs the actual blitter with instrumented GDI
calls: all calls stay on its owner thread, resizing releases old handles,
double reset is harmless, clipping preserves untouched pixels, and DC/bitmap/
selection/BitBlt failure paths recover without leaking or deleting a selected
bitmap. Native RGB555/RGB565 smoke passes with publication on and off. All 131
selected grassland tests and 51 focused bridge/cache/publication tests pass.

The first native smoke attempt crashed, not merely timed out. The process was
confirmed stopped and Windows recorded an access violation in DLL memcpy.
The fixture had retained its primary output across two smaller material-boundary
renders, then blitted that expired pointer using the old dimensions. Previously
reused vector storage could conceal the lifetime error; owned publication exposed
it. The fixture now reacquires the primary frame before blitting. Borrowed output
lifetimes are documented in the ABI header (no layout/version change), and the
corrected executable passes against the unchanged DLL in both modes. Preserve
`native/build/camera-ui-blitter/smoke/stale-output-failed-trace.log` as failed
evidence; the old smoke passes do not prove that stale-pointer use was valid.

Staging is untouched. No INSTALL, game launch, injected compilation or new
patch-table entry was needed. All test processes are finished. Remaining work
is a bounded/cancellable foreground refinement path, unit-request coordination,
and a current-camera preview with a scheduled final redraw—not returning an
unchanged old map while a new view builds.

## Immutable publication groundwork (opt-in, still synchronous)

`C3X_RENDERER_CAMERA_PUBLICATION=1` enables an experimental completed-frame
owner in `RendererWorker`. It copies pixels, fallback indices and replacement
flags together, and captures the blit rounding phase from the completed camera.
Later mutation of render scratch no longer changes that publication. Identical
scene/time hits reuse it without another image copy. Each publication is capped
at 32 MiB including ownership and allocated vector capacities; a transactional
commit can temporarily hold two. At 2240x1192 one image is 10,680,320 bytes.
Allocation/size rejection preserves the ordinary synchronous exact result;
it does not return a stale frame or silently skip ownership validation.

This is a prerequisite for progressive presentation, **not asynchronous
rendering yet**. It is off by default and not staged. The same-DLL 2240x1192 A/B
reproduces all five independent images and 30 revisits exactly. Warm median is
**83.254 -> 86.393 ms**, p95 **99.854 -> 105.835 ms**; this added copy has a cost,
not a speedup. Candidate SHA-256:
`bd0b999b9296544b7a8db0fb1af60bb5d6381eefc6707f1fbcc77493631d6957`.
Receipt: `native/build/camera-publication/zoom/comparison.json`. Minimum sampled
free VA is 1,595,432,960 bytes, largest free region 1,510,281,216 bytes. Geometry
residency is unchanged; no fallback or recovery occurred.

Native smoke passes with the switch both on and off, including ABI, scheduling,
fallback, RGB555/RGB565 gradients, exact clipping and unchanged source pixels.
The 131 selected grassland regressions and 50 focused bridge/cache/publication
checks pass. The executable publication test runs the actual production owner,
mutates source buffers, injects allocation failure at each copy step, rejects
oversized/invalid inputs, tests aliased capture and verifies release on clear.
Integration now selects that test. No injected code or patch-table change was
made; no INSTALL or game launch ran. All processes for this pass finished.

At this pass GDI resources still belonged to RendererState (its destructor,
not ordinary device reset, released them). The UI-blitter pass above separates
that lifetime. Remaining asynchronous hazards are explicit: unit drawing shares the
worker's D3D context; foreground render currently has no cooperative deadline;
and a provisional image must depict the new camera, never an unchanged old
view. Bounded/cancellable refinement and a
validated current-camera preview/final-redraw path are still needed. Do not
enable background rendering merely because the output arrays are now separable.

## Flat-height shoreline certificate

The shared natural height query now evaluates its two height sources before
the shoreline response. When authored height is exactly 2.5 and pickup relief
is exactly zero, both coastal formulas return 2.5 regardless of shoreline.
That case skips the redundant shore query; non-flat samples retain the existing
formula. Source observations are still recorded, including absent neighboring
hills, so introducing relief invalidates the certificate. No resolution,
geometry ownership, rendering budget or cache ceiling changes.

Candidate: `native/build/camera-flat-shore/C3XRenderer.dll`, SHA-256
`e2ef305378b91dec60e3d569f0407f5e4ef1ded799d81c4e8f34640fe884078d`.
Its same-DLL minimap control sets only `C3X_RENDERER_FLAT_SHORE_CONTROL=1`.
At **2240x1192**, all six independent images and 30 repeated images are exact,
with no fallback or recovery. First-use median **4,949.151 -> 4,796.570 ms** and
maximum **5,858.602 -> 5,656.593 ms** are modest improvements, not a solution to
multi-second cold rendering. Warm moves measured **29.723 ms median / 70.183 ms
p95 / 71.642 ms maximum**, versus control p95 68.517 ms; no warm improvement is
claimed. Five cold samples do not establish p95. Receipt:
`native/build/camera-flat-shore/navigation/comparison.json`.

Geometry residency remains 611,114,938 bytes. Minimum sampled free VA was
1,724,035,072 bytes and minimum largest free region was 1,642,860,544 bytes.
The source test compares 26,010 exact height/support results across hills,
coasts, wrapping, tiny/zero/negative source displacements and distant samples;
8,952 redundant shore calls are avoided. The zero-height test still observes
all nine hill-source cells and detects a new hill without a coast query.
All 131 selected grassland tests pass. The explicit production terrain-edit
witness rebuilt 127 tiles, reused 260 and matched the cold image exactly
(`native/build/camera-flat-shore/terrain-edit/witness.txt`). No full zoom or
seven-case replay result is attributed to this DLL. Staging is unchanged; no
INSTALL or game launch ran, and all processes started for this pass finished.

Cold interaction remains synchronously blocked: `RendererWorker::render`
submits the exact captured scene, `submit_locked` waits for its exact sequence,
and only then can the injected bridge validate ownership and blit. Moving work
onto that existing worker does not make input responsive. Any progressive or
asynchronous path must preserve immutable publications, current-camera pixels,
ownership, cancellation, unit-draw serialization and a scheduled final redraw;
merely returning an old frame or shortening the wait would violate the current
contract. This is the next architectural constraint to investigate alongside
remaining cold geometry/upload cost, not a claim that asynchronous presentation
has been implemented.

## Exact nested-grid reuse

The next isolated candidate reuses a retained finer ground grid when every
requested coarse coordinate is already present exactly. It verifies integral
stride and float-division equality, selects those vertices directly, and keeps
the requested triangle stream. There is no interpolation, mesh-resolution
change, extra cache tier or expanded ownership. Invalidation still validates
the finer grid's full dependency set, a conservative superset for decimation.

Candidate: `native/build/camera-nested-ground-grid/C3XRenderer.dll`, SHA-256
`09a53c028d0c5ba19daf7e5cc5fd409cedeb33160afb8418ee472825101a9d60`.
The paired control uses the exact same DLL with
`C3X_RENDERER_NESTED_GRID_CONTROL=1`; all earlier ground retention remains on.
At **2240x1192**, all five independent zoom images and all 30 revisits match
exactly, with no fallback or recovery. Receipt:
`native/build/camera-nested-ground-grid/zoom/comparison.json`.

The widest first-use change measured **7,023.546 -> 5,951.272 ms** (about 15%
lower); its ground phase measured **1,911.401 -> 969.453 ms**. Overall first-use
median barely changed (**2,873.254 -> 2,820.725 ms**), as the other three changes
already use matching grid sizes. Warm zoom measured **78.446 ms median /
98.196 ms p95 / 101.626 ms maximum**; this does not establish a warm speedup.
Retained CPU grids finished at **112,710,888 bytes**, versus **136,421,756** in
the control, because redundant coarser copies are not admitted. GPU residency
is unchanged at 685,584,054 bytes. Minimum sampled free VA was 1,604,648,960
bytes; minimum largest free region was 1,514,606,592 bytes. The limits remain
unchanged and these samples are not a total-process memory guarantee.

The 12 focused executable tests now exercise actual fine/coarse selection and
indexing for 8/12/16/24/32 divisions, all five zooms and all retained layer types.
They compare complete vertex bytes, reject non-nested/invalid grids, exercise
the diagnostic control, and preserve cancellation and dependency tests.
All 131 selected grassland regressions and the isolated native ABI/scheduling/
fallback/RGB555/RGB565 smoke passed. The seven behavior replays below belong to
the preceding candidate; they were not rerun or attributed to this DLL.

No minimap speedup is claimed for this change: constant-zoom moves normally
request the same grid size. Existing full-size navigation traces identify
roughly 0.9--1.75 seconds in ground and 2.3--3.0 seconds in the combined
natural/cliff phase on distant first moves. Further cold-view work must address
new geometry/query work and uploads, not just exact cache revisits. The staged
DLL remains untouched pending the existing user choice; INSTALL and game launch
were not run. All benchmark and smoke processes are terminal.

## Underlying ground-grid retention

The current isolated candidate retains sampled ground grids across zoom changes.
It keeps exact raw relief heights and normal differences, then reconstructs
projection and zoom-dependent river junction distance. Full vertex-byte tests
cover all five zoom levels. Cache identity uses the raw wrapped occurrence,
not just the canonical gameplay tile: world coordinates and UVs must not alias
across a minimap seam. Semantic, coast and world-topology dependencies are
validated before reuse, and cancelled grids are not published.

This reuses the existing CPU natural-mesh tier (192 MiB experimental, 96 MiB
normal) while world GPU sharing is active; the two CPU caches are not retained
together. It adds no budget and does not change renderer ownership or hooks.
Allocation reserves precede moving retained grids, so an allocation failure
cannot leave a partially moved entry available for later reuse.

The matched DLL is `native/build/camera-ground-grid-wrap-safe/C3XRenderer.dll`,
SHA-256 `b1e9e40dc66d08e0b9aa27211ffcc0c92803cb70e68500b062340f3df8cfa8e5`.
Both processes use this exact DLL and frozen runtime inputs; the control alone
sets `C3X_RENDERER_GROUND_GRID_CONTROL=1`. At **2240x1192**, all five independent
zoom images, six navigation images and 30 revisits per scenario are byte-identical.
Both `zoom/comparison.json` and `navigation/comparison.json` under that build
directory pass, with no fallback or device recovery.

First-use zoom changes measured **5,228.201 -> 2,656.077 ms median**; maximum
**6,823.032 -> 6,525.392 ms**. Close views improve substantially, but the widest
view remains slow. Warm zoom measured **74.571 ms median / 93.837 ms p95 /
95.093 ms maximum**. Minimum sampled free VA was 1,572,741,120 bytes; smallest
sampled largest free region was 1,496,977,408 bytes. Geometry residency remained
685,584,054 bytes.

Minimap first-use latency was essentially unchanged: **4,914.913 -> 4,924.060 ms
median**, maximum **5,816.306 -> 5,827.818 ms**. Warm navigation measured
**30.700 ms median / 65.021 ms p95 / 68.211 ms maximum**. Minimum sampled free VA
was 1,724,153,856 bytes; minimum largest free region was 1,628,700,672 bytes.
Four cold zooms and five cold moves do not establish cold p95. This pass does
not meet instant first-response, cold-view or smooth-transition targets.

Twelve focused executable cache tests pass, including exact projection, raw
wrapped occurrence identity, dependency invalidation and bounded admission.
The selected grassland dispatcher suite passed 131 tests. An earlier centered
comparison passed before the raw-occurrence correction, but is not the current
candidate. The intermediate navigation comparison against a different build
was rejected (even initial images differed); it is not optimization-parity proof.
Only the matched four-run sweep above supplies this pass's parity and timings.

The same high-memory DLL passed native ABI/scheduling/fallback and RGB555/RGB565
smoke, plus seven current-production behavior replays: normal/reduced scrolling,
world wrapping, resource playback/removal, day/night unit actions/compositing
and explicit terrain-edit invalidation. The edit rebuilt 127 tiles, reused 260,
and matched the cold image exactly. Receipt: `verified/results.json` under the
candidate directory. The resource-focused dispatcher selected six cases, so the
seventh terrain witness was run explicitly rather than assumed covered.
The isolated smoke folder initially omitted its shader inputs; that run failed
initialization, and the preserved `smoke/missing-shader-failed.log` is not a pass.
With the required shader files present and the intended frozen synthetic profile,
smoke passed with pixel hash 7513777451713106803. No production code change was
needed for that test-setup correction.

The staged DLL changed independently after the previous handoff. This experiment
has not overwritten that newer staged build. The staging record below describes
the earlier handoff, not a guarantee of the current binary's identity.
At this handoff the staged SHA-256 is
`dd7b9d5baef6f17fe8475969a8feae87e2d873233e9ec9f294c8e263f7763435`.
Replacing it with the verified performance candidate is awaiting the user's
choice because it would overwrite another task's staging. All test processes
have finished. No INSTALL, game launch, injected compilation or reference-image
replacement was performed for this DLL-only pass.

## Cold-view query/index pass

This pass removes redundant CPU work without raising cache limits or changing
the terrain samples, triangle order, materials or renderer ownership:

- Natural terrain and mountain grids now emit indexed vertices directly. A
  mountain patch retains 4,225 corners and 24,576 indices instead of expanding
  24,576 full vertices and hashing them back into the same mesh. First-reference
  order, coast clipping and append offsets are preserved; unused corners do not
  enlarge bounds. The shared Lab adapter still supports triangle-list output.
- The underlying relief query now knows when direct hill/mountain sources and
  analytic dunes are zero because separate natural meshes own those surfaces.
  Its flat certificate still observes the complete topology support. Away from
  volcanoes, coastal queries retain the exact coast rim and river attenuation
  without evaluating the discarded hill/dune/material expressions. Volcano
  neighborhoods and the non-fidelity provider keep the existing path; missing
  topology still prevents the flat certificate.

The isolated high-memory DLL is
`native/build/camera-cold-query-zoom/C3XRenderer.dll`, SHA-256
`cbcd840e6550e559fae6cd1db002429f3541a71f3cc95a8aaf0d7ecd1198c85d`.
The matching control uses this exact DLL and frozen runtime inputs with
`C3X_RENDERER_GRID_INDEX_CONTROL=1` and
`C3X_RENDERER_RELIEF_QUERY_CONTROL=1` in a fresh process. These are diagnostic
controls, not user configuration options. Both processes retain the prior
world-mesh sharing and full-size viewport caches.

At **2240x1192**, all five independent zoom images and all 30 repeated images
are byte-identical. First-use zoom changes have median **6,577.333 -> 5,722.332
ms** (about 13% lower), maximum **8,974.436 -> 7,775.887 ms**. Four first-use
changes are not enough to report p95. Warm revisits measured median **87.797
ms**, p95 **108.641 ms**, maximum **119.212 ms**, with zero builds/uploads,
fallback or recovery. This is a cold-build optimization, not evidence of a new
warm-cache speedup. Minimum free VA was 1,522,552,832 bytes; the minimum sampled
largest free region was 1,415,073,792 bytes. Geometry residency is unchanged
at 685,584,054 bytes. Receipt: `native/build/camera-cold-query-zoom/comparison.json`.

The matching **2240x1192 minimap** A/B also reproduces all six independent
images and all 30 revisits exactly, without fallback or recovery. First-use
median is **5,846.188 -> 5,667.444 ms**, maximum **8,515.228 -> 6,285.094 ms**;
the nearby overlap move is slightly slower (**331.764 -> 348.365 ms**).
These five first-use moves do not establish a cold p95 or a uniform speedup.
Warm revisits measured **31.365 ms median / 122.566 ms p95 / 133.225 ms maximum**.
That meets the 150 ms completion target, but not the stricter 100 ms response
target in this run. The prior 66 ms p95 is not a guarantee for every run.
Minimum free VA was 1,641,324,544 bytes; minimum sampled largest free region
was 1,558,183,936 bytes. Geometry residency remained 611,114,938 bytes. Receipt:
`native/build/camera-cold-query-navigation/comparison.json`.

The category dispatcher passed 131 selected grassland regressions; 99 focused
workbench/cache/zoom checks also passed. Executable geometry tests expand direct
indices and compare every vertex byte, including holes, append offsets and
cancellation. The source-aware query test compares 28,566 points against the
unoptimized evaluator, including coast/river rims, missing cells and volcanoes,
and checks all 25 flat-certificate topology observations. Both high-memory and
normal-budget DLLs passed native ABI/scheduling/fallback/RGB555/RGB565 smoke.
All six isolated current-production replays passed with the normal-budget DLL:
normal/reduced scrolling, world wrapping, resource playback/removal and day/night
unit actions/compositing. Receipt:
`native/build/camera-cold-query-verification/verified/results.json`, DLL SHA-256
`94b7bb43dbf567de87bf3e88aeb9865f483d532fd5035339959f46bb7fc923b0`.
The bounded VM transport wait expired during the first replay; its task-owned
process continued and subsequently produced the matching successful completion
receipt. No replay was killed or represented as passed while still running.

The separate grid-only A/B also reproduced all five images exactly:
`native/build/camera-grid-index-candidate/comparison.json`. Its timings were
noisy, including a slower first 96-pixel view; do not present it as a uniform
speedup. The combined same-DLL comparison above is the current measurement.

Cold views still take seconds and do **not** meet the targets below. Further
work must address remaining terrain/coast queries, new natural geometry and
upload rather than claiming cache hits represent first-use latency.

At the user's explicit request, the high-memory `cbcd840e...` evaluation DLL was
staged at `bin/C3XRenderer.dll`; candidate/staged hashes matched at handoff. The previous
`1a2de66d...` DLL is preserved at
`native/build/camera-cold-query-zoom/previous-staged.dll`. Custom rendering and
custom zoom are already enabled in the local configuration. No patch-table
entry is needed for this performance pass. INSTALL and game launch were **not**
run; the user can run their usual Windows `INSTALL.bat` and test the five-level
Z cycle. Automated replay processes have finished, so this task leaves no
benchmark workload competing with the game. This is evaluation staging, not
visual acceptance, reference replacement or a claim of instant cold views.

## Earlier full-size cache optimization

Active acceptance dimensions are **2240x1192**, for both the centered five-level
zoom cycle and minimap navigation. The following records the earlier cache pass;
the cold-query evaluation staged above supersedes its unstaged candidates.

Latest evaluation build: `native/build/world-cache-evaluation/C3XRenderer.dll`,
SHA-256 `d90fdad0a1865d2138007d3614b0b9c31d16e9d0025b18e3814f9a9f9b647c29`.
It was rebuilt from current code and checked against current production shaders,
not the frozen A/B shader snapshot. All 30 full-size zoom revisits passed exactly,
with zero mesh builds/uploads, fallback or device recovery: median **78.457 ms**,
p95 **96.007 ms**, maximum **109.585 ms**. First-use changes still took
3.87–7.81 seconds. Sampled free VA stayed above 1,522,757,632 bytes. The native
ABI/scheduling/fallback and RGB555/RGB565 blit smoke tests also passed on this exact
DLL. Its `comparison.json` is a self-comparison for distribution/repeat parity,
not a new independent speedup claim. At the end of that pass, the staged DLL was
still the previous evaluation (`1a2de66d...`).

The current implementation addresses three separate costs:

- Natural terrain/tree GPU meshes are owned once in world coordinates. Each
  camera entry holds a weak key and owner version; the vertex shader applies the
  selected projection. Active draws pin both owners inside the existing geometry
  budget. Eviction or a rebuilt owner's version rejects old draw identities.
- Retained animated viewports record weak draw identities, checked as a complete
  set before restore. The complete frame signature still validates scene, camera,
  visibility/ownership, environment, topology and asset/device revisions. Poses
  are recomposed, never frozen into the terrain bitmap. Pixel-block prefetch
  borrows both camera and shared natural layers, or skips an incomplete set.
- Draw lists now grow amortized instead of reserving exactly one additional
  tile at a time, which repeatedly copied the entire preceding layer.

The isolated high-memory candidate uses 768 MiB geometry, 192 MiB CPU natural
meshes, **128 MiB retained viewports** and **288 MiB animation backdrops**.
World owners and camera entries share the geometry byte budget; its entry limit
is 16,384. Normal-build byte budgets remain unchanged. These are separate cache
ceilings, not reservations and not a total-process memory guarantee.

An independent A/B used the exact same DLL and frozen runtime inputs with
`C3X_RENDERER_SHARED_NATURAL_CONTROL=1` only in the control process. All five
images passed existing parity thresholds (0–3 pixels exceeding two channel
levels per image); all repeated images were exact. Sharing alone reduced warm
revisits from 4.90–9.18 seconds to 0.92–1.77 seconds, with zero mesh builds on
revisits and about 407 million additional free VA bytes at the minimum sample.
Receipts: `native/build/camera-shared-natural-ab-control` and
`native/build/camera-shared-natural/comparison.json`. This diagnostic switch
selects a fresh-process control, not a runtime user setting.

Retaining the full-size images then removed the rerasterization cost. Thirty
revisits with amortized draw lists measured median 131.756 ms, p95 202.089 ms,
maximum 266.257 ms, with no mesh builds, fallback or recovery and exact repeat
pixels (`native/build/camera-shared-natural-linear`). The subsequent identity
restore measured 91.590 ms median / 118.705 ms p95 over 30 warm zoom changes.
The cross-build image comparison against the preceding candidate was rejected:
even its initial cold images differed. It is not an optimization-parity receipt;
a same-DLL control supplies the final comparison instead.

The final matched full-size zoom run uses DLL SHA-256
`188ace56f04c16c6b7d88e44211a9f3655c5a99db1e6313d26dc64ce45968be1`
and identical frozen shaders/assets in both processes. Across 30 warm changes:
median **6,830.628 -> 82.025 ms**, p95 **8,982.259 -> 106.448 ms**, maximum
**9,435.036 -> 111.056 ms**. Every shared-cache revisit builds/uploads zero tiles;
GPU residency stays at 685,840,246 bytes. All five independent images pass
(0–3 pixels over the existing channel threshold); all repeat images are exact,
with zero fallback or device recovery. Minimum free VA is 1,520,443,392 bytes,
largest free region 1,460,088,832 bytes. Receipt:
`native/build/camera-world-cache-zoom/comparison.json`, paired with
`native/build/camera-world-cache-control`. The 150 ms final-quality warm-zoom
target passes; the separate 50 ms response target does not. First-use changes
still take 4.48–8.55 seconds.

The six-destination **2240x1192 minimap** witness now passes all 30 repeat images
exactly, with no fallback or recovery: median **30.680 ms**, p95 **66.046 ms**,
maximum **76.811 ms**. This meets both cached-move targets. Minimum sampled free
VA was 1,639,870,464 bytes, largest free region 1,556,746,240 bytes. Receipt:
`native/build/camera-world-cache-navigation/comparison.json` (self-comparison,
distribution and repeat parity only). Cold first moves still took 341–7,275 ms,
including 341 ms for the nearby overlap move. Distant views spend seconds in
terrain/natural geometry generation and upload. They do not meet the cold target.

Neither these results nor the cache ceilings establish live-game latency, memory
safety or smooth transitions. Targets below remain unchanged.

Verification of the current normal-budget code passed 210 selected regressions;
98 focused workbench/cache/zoom checks also passed after the test-harness changes.
All six production behavior replays passed using private copies of both the DLL
and Lab executable: normal/reduced scrolling, world wrapping, resource animation
with zoom return/scroll/removal, and daytime/nighttime unit matrices. Receipt:
`native/build/world-cache-integration/verified/results.json`, DLL SHA-256
`54a870c155d51be0998dc869a6c3826a294e702ae5f986bc823f0f546594fd6d`.
The earlier shared-output integration attempt encountered file locks, mismatched
completion records and crashed verifier processes; it is not a successful receipt.
The wrap failure did not reproduce with both binaries isolated. Failed evidence
is preserved; only this task's two confirmed crashed verifier processes were
stopped, not Civ III or other tasks' processes.

`native_render(..., candidate=..., preview=...)` can now select a private preview
without rebuilding the shared executable. Offscreen preview exceptions log code,
module/offset and stack addresses and exit instead of leaving a hidden error
dialog. A controlled Windows exception verified this diagnostic path. No such
exception handler is installed in Civ III. The current-code guard also preserves
the original CPU projection for non-native diamond aspect ratios.

Remaining work is cold geometry generation/upload, the tighter 50 ms response
goal, and combined zoom/navigation working-set stress. Separate repeated-camera
benchmarks are not a claim that arbitrary mixed camera paths remain resident.

## Previous evaluation handoff

The prior requested in-game evaluation used the following snapshot. The
unfinished mutable ground-point scratch-cache change was withdrawn; no partial
implementation remains. The evaluation DLL is
`native/build/zoom-evaluation-960/C3XRenderer.dll`, SHA-256
`1a2de66d8dddc3e9698c883495945470c68536b124a8b7d08a652bb6f55590b3`.
It uses the experimental 768 MiB GPU geometry, 192 MiB natural CPU mesh and
160 MiB resource-backdrop caps, plus the unchanged 32 MiB viewport tier. These
are cache budgets, not reservations or a cap on all process memory. Ordinary
build defaults remain unchanged; rebuilding normally loses these higher caps.
That DLL was staged at the previous handoff and is now the rollback for the
cold-query evaluation above. Its own earlier rollback is preserved as
`native/build/zoom-evaluation-960/previous-staged.dll`. Neither installation nor
game launch was performed by that handoff.

Final checks on this exact DLL:

- 56 focused indexing/cache, native bridge, custom zoom, render-core and resource
  animation regressions passed.
- Native ABI, scheduling, fallback and RGB555/RGB565 blit smoke checks passed.
- Current-asset 960x640 zoom-return, six resource-animation poses, scrolling and
  removal passed; posed frames performed no terrain builds or uploads.
- Current-asset 960x640 two-cycle zoom passed with exact repeat pixels and no
  device recoveries. Five warm changes took 47.792–82.111 ms, median 67.648 ms,
  with zero tile builds; this short handoff check is not a new p95 measurement.
  Sampled free virtual address space stayed above 1.60 billion bytes in the
  standalone witness, which does not include Civ III's own memory use.
- The 1280x720 stress run passed all five repeat images without fallback or
  device recoveries. Repeat zooms still take 3.0–4.8 seconds because the working
  set exceeds the geometry cap. Cold views remain slow at both sizes.

Logs are under `native/build/zoom-evaluation-960` and
`native/build/camera-ground-scratch-control`. The latter name denotes the
pre-scratch-change control: that abandoned change is not in this DLL. Its
comparison against the older `camera-direct-grid` DLL differs visually because
the compiled scene implementation has changed between snapshots; it is not a
valid before/after performance or image-parity claim. Current-view repeat,
scroll and removal checks above supply the handoff's cache-correctness evidence.

For game testing, enable `enable_custom_rendering_zoom = true` in the mod's
`custom.c3x_config.ini` (currently commented out). Use `Z` to cycle levels.
Native city-label alignment still needs the existing
`Main_Screen_Form_tile_to_screen_coords` CSV row changed from `define` to
`inlead`, with its signature and addresses untouched; see the
[patch dependency ledger](civ3_patch_dependency_ledger.md). No new patch symbol
is required by the performance work. The agent has not changed either file.
Run `INSTALL.bat` yourself after the desired configuration/patch-table edits.
No live-game pass or smooth-transition claim is made.

## Witness

The offscreen witness in `native/biq_preview.cpp` recaptures the same full-world
fixture in the current `Z` cycle: 128, 96, 64, 192 and 160 pixels, then repeats.
This is the centered range (normal plus two farther and two closer levels).
Historical measurements below used the old 128, 112, 96, 80, 64 ladder; the
comparison tool recognizes both sequences but never compares mismatched ladders.
It retains the
same resource sites and animation time so cached-image parity has a deterministic
reference. Every step checks ownership and fallback; repeated steps use the
existing pixel-error thresholds. The independent comparison tool checks the
baseline and candidate images at all five sizes and records both DLL hashes.

From `Renderer/native` in the Windows VM:

```bat
call BENCHMARK_ZOOM.bat baseline
rem Make the performance changes, then:
call BENCHMARK_ZOOM.bat candidate
```

Use `reuse` as the second argument to preserve an already compiled DLL, or
`build-only` to compile without running. `preview-only` refreshes just the witness
executable without changing the tested DLL or running it. Builds use separate `build/zoom-baseline`
and `build/zoom-candidate` directories and never stage or install. The fixture
requires the existing local `lab/.local/verification/world.csv` and asset packs.
`C3X_ZOOM_WIDTH` and `C3X_ZOOM_HEIGHT` default to 640 and 480. `C3X_ZOOM_ROOT`
optionally selects a snapshot containing `Renderer/native` shader providers,
the production and custom definitions, and `Renderer/packs`. Keep shader and
asset bytes fixed between runs; a concurrent Lab edit invalidates pixel comparison.

From the project root:

```sh
python3 Renderer/native/compare_zoom_benchmark.py
python3 -m unittest Renderer.native.test_zoom_mesh_cache
```

The comparison receipt is `native/build/zoom-candidate/comparison.json`.
Initial asset/shader loading is outside the timed zoom steps. The first 128-pixel
step is an unchanged-current-view control and is excluded from the median.
Geometry timing includes construction and upload; readback timing includes pending
GPU execution. Do not interpret a replay as a live-game or smooth-transition pass.

## Centered zoom range and actual-size check

The injected `Z` control now selects 50%, 75%, 100%, 125% and 150%, with
normal in the middle. Its render-time synchronization accepts both close-ups.
The existing staged evaluation DLL already accepts these numeric projections;
its bytes were not changed. Applying the new controls requires rerunning
`INSTALL.bat`; the automated compile check did not install or launch the game.

Verification passed the approved injected compile/injection smoke test and 49
focused tests, including execution of the actual key-handler/native-sync C,
inverse picking, world-mesh reprojection, unit body/HUD offsets and native-unit
suppression at every level. All seven 960x640 replay cycles passed, including
both close-ups, exact repeat pixels and zero device recoveries. Across 30 warm
changes, median was 46.650 ms and p95 90.019 ms. The receipt under
`native/build/camera-centered-zoom` is a self-comparison recording distribution
and parity, not an independent speedup measurement or comparison of different
zoom ranges.

The latest captured game viewport is 2240x1192, not 960x640. An additional
full-world replay at those dimensions passed two complete cycles with exact
repeat pixels and zero fallback/recoveries, including 160/192-pixel close-ups.
However, its five repeat changes took **5.744–14.170 seconds** and each rebuilt
all GPU tiles for that projection. The 768 MiB geometry cache remains full;
five raw full-view BGRA bitmaps alone also exceed the separate 32 MiB viewport
budget. This is direct evidence that small-view warm-cache results do not prove
responsiveness at the user's game size. The minimum sampled free VA was
1,154,306,048 bytes in the standalone process, not a live-game safety guarantee.
Logs and images: `native/build/camera-centered-live-size`.

Use **2240x1192** for the next interactive acceptance measurements (zoom and
minimap). Keep 960x640 as a compact regression fixture and retain the existing
latency targets below. The actual-size replay is a fully visible synthetic
world, not the user's exact fog/capture workload. Projection-independent GPU
storage, scalable retained images and reduced cold-build work remain necessary;
this range change does not claim to fix the remaining performance issue.

The earlier 960×640 baseline spent about 20–25 seconds on individual 112/96/80
steps, mostly in geometry construction and upload, and was stopped after the
64-pixel step stalled. Its partial output is preserved separately as
`native/build/zoom-baseline-wide`. Those observations are diagnostic, not a
completed five-level comparison.

## Measured performance pass

Windows VM, current production profile, fixed shader inputs, 640×480, full-world
100×100 fixture with animated resources; two cycles, 263–635 visible tiles:

- Median of the nine actual zoom changes: **12,567.849 ms → 4,445.532 ms**,
  approximately **2.8× faster** (65% less delay).
- Individual speedups: 1.84×–3.53×. These are single-run observations, not stable
  latency percentiles. The widest first-use step included a 5.2-second GPU-submit
  outlier; CPU geometry/upload remains the dominant repeat-zoom cost.
- All five baseline/candidate image comparisons passed. No pixel exceeded the
  existing two-channel-value tolerance; aggregate absolute channel error was
  0, 5, 1, 4 and 0 across the five images. Repeated candidate views were exactly
  equal to their first-cycle images. No terrain fallback occurred.
- CPU cache allocation remains 128 MiB combined: 96 MiB natural mesh data and
  32 MiB viewport bitmaps. The indexed GPU tile cap remains 192 MiB; this is not
  a cap on total renderer/device memory.

The exact DLL hashes and per-step timings are in the disposable comparison
receipt. The candidate is isolated under `native/build/zoom-candidate`, not
installed or staged. No native hook/address changes are needed for this pass.
This establishes reusable world meshes; it does **not** meet instant or smooth
zoom latency. The next architectural work is camera-independent GPU storage,
remaining ground/route geometry reuse, and immediate retained-image presentation.

Verification also passed 195 dependency-selected/full integration test cases,
plus the native production animation witness: zoom-away/return parity, six poses
with zero terrain builds/uploads, and exact cold-render parity after scrolling
and resource removal. The standalone vertex-index witness compares production
indexing against the old implementation and exercises reprojection across all
five levels and multiple target heights. These are automated offscreen checks;
the exact candidate also passed native ABI, scheduling, fallback, export and
RGB555/RGB565 blit smoke tests. No live-game performance claim is made.

## Interaction targets (ongoing)

These are engineering targets, not measured achievements. The original primary
workload was 960×640 with animated map objects, with 640×480 fast diagnostics
and a 1280×720 stress case. The actual-size check above supersedes the primary
acceptance dimensions with 2240×1192; prior measurements retain their original
dimensions and must not be relabeled. Preserve current image quality, ownership,
fog/visibility, latest-request correctness and bounded memory.

| Interaction | First correct visual response, p95 | Final-quality image, p95 |
| --- | ---: | ---: |
| Warm zoom change | 50 ms | 150 ms |
| Nearby minimap move / cached revisit | 100 ms | 150 ms |
| Previously uncached distant minimap jump | 100 ms | 500 ms |
| Animation / continuous interaction frame | 33.4 ms | 33.4 ms |

An immediate response must represent the requested camera and respect visibility;
returning an unchanged old view does not count. Final quality must pass cold-image
parity. The current synchronous renderer cannot distinguish these milestones:
its blocking render duration counts against both. Eventually collect at least
30 actual changes per workload for meaningful p95 gates. The initial two-cycle
smoke sequence reports individual timings and medians, not a statistical p95 pass.
Initial shader/asset startup is reported separately from warm interaction.

For the minimap replay, set `C3X_CAMERA_SCENARIO=navigation` before calling the
benchmark. It recaptures records at six destinations (overlap, distant regions,
and wrapping) twice, logging `NAV` timings including replay capture, geometry,
draw and readback. `C3X_ZOOM_OUT` selects an isolated result directory, permitting
camera baselines to remain intact while new candidates are built. This exercises
the renderer side of minimap movement, not native mouse handling/capture overhead.

Set `C3X_RENDERER_PREVIEW_CYCLES=6` for 30 minimap revisits, or `7` for 30 zoom
revisits. The supported range is 2–10 cycles. Every repeat still checks image
parity. The comparison script accepts `--scenario navigation`, `--baseline DIR`
and `--candidate DIR`; it separates first-use/revisit distributions and emits a
nearest-rank p95 only for groups with at least 30 samples. Current logs also
expose device-recovery counts so successful pixels cannot hide reset stalls.

## Follow-up findings and current measurements

The regular-grid regression found that 4,096 distinct float vertices all started
in one of 8,192 hash slots. Adding a final avalanche mix removed that pathological
probing without changing vertex equality or triangle order. The 640×480 diagnostic
then measured 2,739.275 ms median zoom, versus 4,445.532 ms in the preceding pass;
revisit median was 1,941.741 ms. All five images matched exactly. Minimap gains
were concentrated in expensive destinations, not universal.

The initial primary-resolution run exposed a real 192 MiB geometry-cache failure
at 64-pixel tiles. Lossless R16 indices (R32 for large chunks) allow all 1,045
foreground/companion tiles to fit; the cap was not raised. At the initial
128-pixel view, tracked geometry fell from 124,597,432 to 113,240,686 bytes.
R16/R32 source-shadow rasterization, including mixed formats and alpha cutouts,
passed full pixel comparison. The four zoom images available from the failing
R32 run also matched the R16 candidate exactly.

Subsequent retries were driver device removals, not byte-accounting failures.
The preview executables lacked the large-address-aware flag that `ep.c` already
sets on Civ III. Benchmark and ordinary preview builds now match that memory
model. This corrects the test harness; it is not an additional in-game speedup.
The matching 960×640 candidate completed both zoom cycles with exact repeated
image parity and no reset/retry errors. Actual zoom changes still took about
2.8–6.7 seconds, so the zoom targets remain unmet.

Matched large-address-aware 960×640 minimap runs now have six cycles, including
30 revisits. The compact-index candidate measured:

- Five first-use moves: median 1,151.358 ms, maximum 3,308.133 ms. No first-use
  p95 claim is made from five samples.
- Thirty revisits: median 3.964 ms, p95 317.483 ms, maximum 6,016.227 ms.
- All six images and all repeats matched exactly; device recoveries were zero.

The large revisit outlier rebuilt evicted terrain geometry for animated-resource
compositing even though the immutable viewport bitmap remained cached. A bounded
eviction preference now retains recently used animated-view meshes ahead of
bitmap-only static views. This is not an additional pin: active-frame protection
and the byte cap remain unchanged, and the preference expires after 32 geometry
generations. The matched follow-up measured **92.032 ms p95, 92.462 ms maximum**
across 30 revisits, with no geometry builds on any revisit. All six images and
all repeats matched exactly, with zero device recoveries. First-use moves still
took up to 3,390.935 ms. This meets the cached-revisit target on this fixture,
not the cold minimap or overall interaction goal.

The same candidate passed both zoom cycles with exact image parity against the
previous candidate, and the production animation witness passed zoom-return,
six changing poses with zero terrain builds/uploads, scroll and removal parity.
Zoom remained 2.8–6.7 seconds. Executable indexing, projection, eviction and
native bridge contracts passed 39 cases.

Receipts and logs: `native/build/camera-zoom` (hash fix),
`native/build/camera-laa-960` (complete wide zoom), and
`native/build/camera-nav-laa-{baseline,candidate}` (matched navigation).
The eviction candidate is isolated under `native/build/camera-animation-priority`
and its zoom/animation witness under `native/build/camera-priority-zoom`. Neither
has been staged or installed. The overall interaction goal remains open.

## Cache-budget experiment

Set `C3X_ZOOM_LARGE_CACHE=1` in the isolated benchmark build to double the CPU
natural mesh tier to 192 MiB / 2,048 entries and use 384 MiB / 4,096 GPU tile
entries. `C3X_ZOOM_GPU_CACHE_MIB=768` selects the larger GPU experiment with
8,192 entries. Normal builds are unaffected by these benchmark-only definitions.
The subsequent backdrop-reuse pass also uses a 160 MiB resource-backdrop tier in
this experiment (32 MiB in normal builds). Viewport images, source shadows, unit
caches and the 64 MiB prefetch sub-budget remain unchanged. These numbers do not
represent total process or GPU memory.

The witness logs `CAMERA memory` after each interaction, outside its timer:
available virtual address space, the largest free region, and total user address
space. The comparison receipt reports minima over those samples; these are not
peak-allocation measurements and omit the memory occupied by a real Civ III game.
Keep LAA enabled, and require pixel parity and zero device recoveries as well as
timings. A larger cache cannot remove cold construction work or guarantee that a
large live game will have enough address space.

Matched 960×640 runs, current geometry and the same frozen shader/asset inputs:

| GPU / natural CPU cap | Median of five repeat zooms | Lowest sampled free VA |
| --- | ---: | ---: |
| 192 / 96 MiB | 4,307.351 ms | 2,153,504,768 bytes |
| 384 / 192 MiB | 2,718.726 ms | 1,884,659,712 bytes |
| 768 / 192 MiB | 124.228 ms | 1,635,328,000 bytes |

All five images and repeated views matched exactly; recoveries were zero. At
768 MiB all five projections fit (785,094,244 tracked bytes), so repeat zooms
built zero tiles instead of rebuilding every tile. The independent seven-cycle
run retained that zero-build behavior across 30 revisits: median 136.998 ms,
p95 252.512 ms, maximum 323.404 ms. This is a large gain but still misses the
50 ms response / 150 ms completion targets. First-use changes still take seconds.
The self-comparison receipt for that longer run records its distribution and
parity, not a second independent speedup comparison.

The 1280×720 stress test exposed two separate defects. The 192 MiB control cannot
fit its active 96-pixel frame and returns a cache-budget failure. The first
768 MiB attempt encountered an access violation in a tree lookup. Inspection
found that the natural compiler retained a reference into the river-page vector
while nested height queries could move/evict those pages. It now retains an
immutable owned page for that tile's compilation; the LRU still contains at most
16 pages and the extra active reference is released when the tile finishes.
The executable regression forces 32 intervening page queries, reset and release;
312 source-parity samples and AddressSanitizer/UndefinedBehaviorSanitizer pass.

With that fix, both 1280×720 zoom cycles pass exact repeat parity with zero device
recoveries and at least 1,349,046,272 sampled free VA bytes (largest free region
1,256,718,336 bytes). But the five projections no longer fit in 768 MiB: repeat
zooms rebuild geometry and take 3.1–5.3 seconds. Larger caches alone are not a
resolution-independent solution. The next work remains projection-independent
geometry reuse, reduced cold construction, and faster animated-view composition.

The exact lifetime-fix DLL also passed the 960×640 fixture against the preceding
768 MiB candidate with zero changed pixels at every zoom and zero device
recoveries. Its five revisits took 77.707–141.747 ms (median 97.032 ms), with zero
geometry builds. That short run is not a replacement for the longer percentile
measurement above. The updated river lifetime, zoom-mesh and bridge suite passes
41 cases; the standalone lifetime test also passes under both sanitizers.

Evidence directories: `native/build/camera-cache-control`, `camera-large-cache`,
`camera-cache-768`, `camera-cache-768-repeat`, `camera-cache-control-stress`,
`camera-cache-768-lifetime`, and `camera-cache-768-fixed-zoom`. Failed attempts are
retained as diagnostics; do not
count their exit logs as successful frames. Candidate DLLs are isolated, not
staged, installed, or a live-game memory-safety certification. No Civ III patch
symbols or addresses changed; no patch-table action is required.

## Indexed ground and cross-view resource backgrounds

Ground passes now hand their unique corners and exact indices directly to the
GPU uploader. The old path expanded each grid into triangle vertices and then
hashed it back into an indexed mesh. The mixed object-shadow pass still uses
ordinary indexing. An executable witness checks the entire expanded triangle
stream at six grid densities and six layers, scratch reuse, and cancellation.
The 1280×720 comparison in `native/build/camera-direct-grid` matches every image
and repeat exactly, with identical GPU byte counts. Timing gains are modest and
not universal; this alone does not remove the seconds-long cache-miss cost.

Animated resources now reuse the static scene-linear MSAA color/depth patches
behind them across exact camera revisits. Each block is keyed by the complete
static frame signature and rectangle; that signature includes camera, scene,
environment, wrap, ownership, content and device generation. Poses and their
shadows are drawn fresh. The byte-bounded LRU protects the current view against
its own scan; uncached blocks render normally and optional cache allocation
failure does not discard the valid background. The cache is cleared on reset.
The byte cap is 32 MiB normally and 160 MiB in the experimental large-cache tier.

At 960×640 with 768 MiB GPU geometry / 192 MiB natural CPU data, the five zooms
use 41 retained backdrop blocks (145,600,512 bytes), with zero misses on every
repeat. The matched seven-cycle witness measures 30 repeat zooms:

- Median: **136.998 → 55.156 ms**.
- p95: **252.512 → 88.861 ms**; maximum **92.368 ms**.
- All five images and all repeated views match exactly; zero geometry builds
  and device recoveries on revisits. Sampled free VA remains at least
  1,602,125,824 bytes; this is not a live-game memory certification.

The completion-time target passes on this warm fixture; the 50 ms first-response
target remains unmet. Cold changes still take seconds. The same DLL passes
zoom-return parity, six animation poses with zero terrain builds/uploads, and
exact scrolling/removal parity. The focused indexing, cache-budget, native bridge
and animation suite passes 44 tests. Evidence: `native/build/camera-backdrop`,
`camera-backdrop-repeat` and `camera-backdrop-nav`. These are isolated candidates;
no staging, installation, native patch additions or live-game pass is implied.

The matched current-geometry minimap control is
`native/build/camera-backdrop-nav-control` (the older animation-priority fixture
contains different cliff geometry and is not a valid image baseline here).
Across 30 revisits, p95 is **92.763 → 47.861 ms**, maximum **48.782 ms**;
median stays near 4 ms because four destinations have no animated resources.
All six images and all repeats match exactly, with zero device recoveries. Five
first-use moves take 610.774–3,149.261 ms, still outside the cold-move target.

`python3 Renderer/renderer.py integration resources` also passes with the normal
32 MiB backdrop / 192 MiB geometry budgets and current production shader inputs:
192 dependency-selected tests, native build checks, changing animation poses,
zoom-return, scrolling and removal parity. Its disposable receipt is
`lab/out/integration/resources.json`. This checks the normal build separately;
the low-budget build is not claimed to achieve the experimental warm timings.

## Current experiment decisions

| Workload | Established result | Highest-value remaining question |
| --- | --- | --- |
| Unchanged camera/scene | Completed bitmap reuse is cheap; not proof of animated performance. | Preserve idle reuse while integrating completion. |
| Small prepared pan | Correct pixel reuse helps; waves-on p95 remains hundreds of milliseconds. | Which substantial static work still repeats during movement? |
| Complete prepared destination | Geometry retention removes compilation; full render p95 is still about 0.7 seconds. | Can most scene/shadow/reflection work be retained or eliminated? |
| Unprepared/evicted destination | Geometry preparation remains measured in seconds. | Compact map preparation and bounded compiled-region reuse remain necessary. |
| Supported zoom transitions | 128/160/192 selection is implemented, but timing distributions are unproven. | Measure genuine changes, not cached repeats, at the three supported levels. |
| Native interaction/display | The game bridge remains synchronous. | Safe current-camera pending output and completion-driven native redraw remain prerequisites. |

For each next experiment, state the affected workload, measured reason for its
priority, expected in-game benefit, and success/stop criterion before running it.
A small isolated saving is not evidence for fast navigation everywhere. The
bounded-glow result closes glow as a major optimization target: 58% less dispatch
work saved only about 4% total completion time. Inspect retained shadow-page build
counters before increasing their budget or redesigning their lifetime.

The `bounded-post-widest-diagnostic` run failed at navigation step 2 (tile width
64): `source-shadow-failed` reported exceeded page budget or interruption. This
synchronous run has no superseding camera requests; the 32-page capacity check
is the relevant remaining constraint. Inputs/binaries stayed unchanged. The
experimental strip path therefore does not establish a supported widest-view
solution. Under the user's subsequent three-level decision, 64 is a pressure
probe rather than a production requirement. Scratch ledger values at failure
were 132,937,728 bytes main linear and 137,453,568 bytes reflected linear, plus
other resources. No silent quality reduction or native terrain replay was used.

Inspection of the existing 100-frame shadow counters rules out atlas capacity
as the typical full-view bottleneck at supported width 128. The bounded-post
control and candidate each build 130 pages total (7,502 source draws), with
median zero page builds and zero source draws per frame. The 128-pixel-region
control builds 146 pages total and also has median zero. Increasing atlas size
is therefore not the next median-latency optimization. Separate main-scene and
reflected-scene cost attribution is the next major decision gate; any pass-off
measurement must be labeled diagnostic and cannot count as correct-quality
performance.
