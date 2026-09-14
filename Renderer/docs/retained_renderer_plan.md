# Retained world → view → submission implementation

## Active implementation: useful preparation across scene owners

The user approved prioritizing work by usefulness before demand, across scene
elements rather than terrain alone. Preserve the staged native-handoff DLL as
control; current evidence starts at `native/build/useful-preparation-20260914/`.

The supplied live trace changes the priority: 962 unit pixel misses consume
40.637 seconds of logged unit caller time; 4,902 hits consume 2.539 seconds.
The cache grows monotonically to 962 entries (about 354 MiB), so this is cold
pose diversity, not eviction churn. Direction, native cursor and owner color
create legitimate distinct outputs. The 429 map boundaries consume 34.964 seconds,
including cold start; these are separate distributions, not additive frame costs.
The trace has no native-handoff/camera-begin records and cannot prove that the
opt-in handoff was active. Its 1119×1192 view and sparse roads also differ from the
dense 2240×1192 synthetic workload. Existing ambient work completes 383 uncancelled
future frames and consumes 258; speculative work is useful but not free.

The bounded stage diagnostic identifies CPU pose/shadow construction as the main
unit miss cost: median 30.068 ms of a 45.753 ms miss, versus 6.489 ms waiting for
GPU completion. It uses CPU wall-clock boundaries, not Parallels GPU timestamps.

`UnitPoseCompiler` now compiles the existing exact pose, tangent frames, projected
vertices, clipping proof and shadow field into immutable CPU content. The existing
`ContentPreparation` pool supplies two unit helpers by default (four terrain helpers
remain separate); `C3X_RENDERER_UNIT_PREPARATION=0/1/2/4` is the bounded control.
Native observations offer the next frame of the currently known action and
facing. Repeated fixed cursors schedule nothing new; observed advances outrank
unused predictions. The demand path adopts ready content, joins matching active
work, or compiles unstarted work itself. Owner color is absent from this CPU key;
unit/action, direction, native cursor/frame count, projection, hour and season
remain dependencies. Final pixels keep their complete color-inclusive key.

Animation jobs hold immutable leases in the existing 96 MiB payload owner.
Demanded payload pressure revokes optional leases before admission fails. Pending
jobs are capped at 32, ready content at 128 MiB, retained consumed content at 32 MiB,
and each helper's estimated result/scratch at 24 MiB. A conservative virtual-memory
check drops optional content before endangering the preserved 512 MiB reserve.
Reset joins readers before releasing assets. Helpers touch no D3D context, native
object, canvas or mutable pixel publication; existing full-detail GPU submissions,
finishing and caller-owned GDI composition consume the prepared content.

Near-term finite ambient preparation now precedes optional static guard expansion
on the existing GPU worker. No extra GPU unit predictions or redraw callbacks were
added. This extends preparation across scene owners without a parallel renderer.

Build1 exposed a full reservoir of unused stationary predictions: only four of
53 unit misses consumed preparation, with no unit-time improvement. Preserve it
as a rejected scheduling mechanism. Build2 prioritizes observed advances and
prevents repeated frozen predictions. Its 20-frame diagnostic matches every
control image exactly; whole map-plus-eight-unit mean falls 51.747 → 23.715 ms.
This is preliminary diagnostic evidence. The first production mixed-action run
showed only a 3% mean improvement: 522 helper builds, 38 consumed and 476 evicted.
That two-frame window/front-insertion mechanism is rejected for busy native sets.
The final scheduler keeps one predicted pose per observation, preserves FIFO order
among advancing units, and lets those finite jobs use the full 128 MiB ready budget;
unused fixed-unit results are evicted before advancing results. This prevents new
future work displacing the older predictions needed by the rest of the current draw.
The production comparison below is the release measurement. CPU work increases when predictions are unused. Report that
cost separately from work removed from the caller and from exact content reuse.
No new staging, installation or game launch is part of this implementation run.

### Production checkpoint: full-detail CPU preparation

Candidate `Renderer/native/build/candidate/C3XRenderer.dll`, SHA-256
`ead40b16387ad67573adbd89e4baf0fba3760255d8710427253eda051314c12e`,
replaces the inline demand-only CPU compiler in the existing unit rendering owner.
The staged control remains unchanged. All artifacts, including rejected windows,
are under `native/build/useful-preparation-20260914/`.

Trace-disabled paired production runs use identical verified preview binaries,
10,321 runtime inputs, full detail, waves/reflections off, and a 1119×1192 view.
Whole requests include map rendering, GDI copy and all eight native unit draws;
initial setup and ten warmup frames are outside the measured windows.

| Workload | Samples | Control → candidate mean | Median | Maximum |
| --- | --- | --- | --- | --- |
| Paced idle, selected/directed/working subset | 30 | 39.395 → 30.734 ms (22% lower) | 18.227 → 20.677 ms | 139.575 → 86.372 ms |
| Unpaced mixed native actions | 40 | 279.527 → 147.644 ms (47% lower) | 283.801 → 137.520 ms | 331.890 → 217.686 ms |

Every paired image is exact. These small synthetic windows establish whole-request
improvement, not native FPS or reliable tail percentiles. Warm idle calls do not
consistently improve. Instrumented idle was more favorable (40.001 → 17.051 ms);
use the trace-disabled numbers above, not that result, for the performance claim.
The same-binary helper-disabled diagnostic is 42.070 → 17.051 ms, also exact.

Work attribution: the final busy diagnostic consumes 350 prepared results across
380 pixel misses; only 30 compile on demand. Helpers finish 358 builds, with zero
ready evictions/rejections and one cancelled task. Thus 388 completed CPU builds
serve 380 requested misses, with eight future results unused at shutdown. The
rejected window performed 864 completed builds (522 helper + 342 foreground) for
the same misses. The final path mainly moves necessary work before demand;
color-independent retained content additionally avoids repeated CPU construction.
Summed helper compilation wall time is 8.741 s; foreground pose stage totals
0.739 s including warmup. Those are not CPU-cycle or GPU-time measurements.

Dense 2240×1192 scrolling remains effectively unchanged: 14-step median
75.295 → 78.121 ms, maximum 186.282 → 157.870 ms, exact images. No scrolling
speedup is claimed. Four terrain/topology and six city/forest edits pass independent
full-redraw comparisons; these instrumented barriers remain correctness evidence.
The map path is unchanged by the final unit-window correction. Busy samples retain
at least 1754 MiB free virtual address space and a 1682 MiB largest free region;
peak ready unit content is 64.6 MiB within its 128 MiB cap. These are sampled
process values, not a universal memory peak guarantee.

Validation covers the production build, CPU queue/lease/cancellation/validity tests,
existing native camera publication tests, the six native replay groups and a final
unit day/night rerun after the scheduling correction. The initial full test sweep
had one stale worker-test stub compile failure; that stub was updated and its
whole test module passed. No injected source changed, so no injected compile was
needed. Evidence tools now allow an explicitly changed renderer DLL while still
requiring identical verified harness/assets; timing-only runs explicitly mark
trace-path evidence unavailable rather than inventing counts. Their rejection
and missing-trace tests pass.

Remaining costs: cold/unseen poses still compile, every pixel miss still performs
GPU submission/finishing/readback, and cold/local map barriers remain synchronous.
The supplied game log does not demonstrate native async-camera activation or
input-to-display cadence. The native strategic interaction checkpoint remains
pending; the renderer still never requests a redraw. No new visual difference,
installation, staging, game launch, Git mutation or deferred-category work occurred.

### User evaluation staging

After the implementation checkpoint, the user requested testing through
`INSTALL.bat` and authorized a Mac commit followed by a Windows push. The exact
validated candidate above is now staged at `Renderer/bin/C3XRenderer.dll`; the
previous DLL remains preserved as `native/build/useful-preparation-20260914/control.dll`.
Civ III was confirmed closed before staging. No installation or game launch was
performed. `INSTALL.bat` updates injection but does not build or stage the DLL.
The new unit CPU preparation is enabled by default with normal game launch.
At the user's request, the native caller-driven async handoff is also enabled by
default when its existing hook and exports are available. Normal `INSTALL.bat`
and game launch require no environment settings or alternate launchers. The
explicit `C3X_RENDERER_NATIVE_ASYNC=0` diagnostic opt-out remains available.
This default-selection change does not establish the pending live cadence and
interaction checkpoint. Missing capabilities still retain the exact path.
The default-selection change passed `TEST_INJECTED_CODE_COMPILE.bat` on the
Windows GOG verification link, plus nine native identity/publication checks
including default activation, diagnostic disable and missing-capability fallback.
The newly changed live capture is preserved outside the implementation commit.

## Implemented checkpoint: caller-driven native handoff

The user approved the native asynchronous handoff after accepting and staging
the four-helper neighborhood path below. Preserve that DLL/source as the control
under `Renderer/native/build/native-handoff-20260913/`.

The implementation now owns one requested view and one displayed native view around the existing
map boundary. A pending camera retains a full-detail displayed view; its fresh
capture must validate content, visibility, occurrence order and epochs before
old pixels can be used. The user-authorized native `move_camera` inlead separates
relative scroll intent from the displayed camera fields, so native traversal,
overlays, direct picking and unit culling share the displayed bounds even between
calls. Original native movement still owns wrap/clamp. Zoom/resize and invalid
publications require exact rendering. No partial terrain preview is accepted.

Copied authoritative requests are submitted on native calls. An active request
survives until completion; the newest demand is selected when its slot becomes available.
This bounds backlog without repeatedly cancelling work during sustained movement.
The DLL owns immutable pixels and the GPU; completion never requests a redraw.
Caller blocking, full render completion, displayed age and request latency are
measured separately. Validation covers the actual bridge and worker, production
replay and the approved injected compile. Live cadence/interaction remains a strategic checkpoint
requiring game-launch authorization; no installation or launch is authorized here.

### User evaluation staging

The user subsequently requested: “Please allow me to test it.” The exact validated
production candidate is now staged for evaluation at `Renderer/bin/C3XRenderer.dll`,
SHA-256 `d92d104627d4edf6246fbf53dc042b6c67abbb1419edbe7843b9f0452fa6ed3c`.
`Renderer/TEST_NATIVE_ASYNC.bat` originally enabled `C3X_RENDERER_NATIVE_ASYNC=1` for its
session and delegates to the existing game-test launcher and `RUN.bat`. That route
injects the current camera hook when the user starts the game; no installation is
needed. The prior DLL and staging receipt are under
`native-handoff-20260913/evaluation-staging/`. This supersedes the unstaged status
in the implementation checkpoint below. Live acceptance remains pending.

### Implementation checkpoint and measured limits

`native-handoff-build3` implements the native displayed/requested camera owner,
optional exact presentation lease, existing queue integration and ambient producer
transfer. The GOG `move_camera` inlead is `0x004DF700`; user-requested non-GOG
addresses remain `0x0`. The patch ledger records the GOG-only limitation. Mode is
enabled by default for user evaluation, requires complete world visibility
capture, and preserves exact fallback when the display proof fails. No native
completion notification exists. Full detail and existing visual contracts remain.

Same-binary paired replay, 2240×1192, accepted retained city profile with
waves/reflections off, four default CPU helpers and existing work ahead:

| Workload | Synchronous whole request median / maximum | Async whole request median / maximum | Async individual-call median / maximum |
| --- | --- | --- | --- |
| Dense 14-step scrolling/reversal, 3 s preparation | 79.207 / 150.130 ms | 94.441 / 189.906 ms | 1.720 / 3.119 ms |
| 30 stationary ticks, 67 ms pace after 10 warmup ticks | 0.554 / 71.077 ms | 2.024 / 70.762 ms | 1.960 / 2.543 ms |

The individual-call column is the distribution of each request's largest call;
whole requests include completion and checked output. These are small samples,
not reliable tail percentiles. All paired images are exact. Scrolling made 70
simulated calls, holding a freshly validated full-detail view on 56. The fixture
polls every 16 ms and waits for each requested result; it does not measure native
cadence or simulate every accumulated native input. The compiled bridge test
separately covers sustained latest-intent accumulation and bounded submission.

This is a measured reduction in caller blocking, **not an overall rendering
speedup**. Polling adds publication latency; recapturing the held view also costs
work. Stationary ready-frame calls were already fast. Prepared animation still
runs: candidate completed 34 future frames and consumed 32 across warmup/measured
work, with one cancelled/failed result, zero foreground joins and zero compiled
geometry/uploads in that producer. Control completed 28 and consumed 27, with one
join. Different timing can change speculative work; counts alone are not a win.
Sampled candidate contiguous free address space stayed above 1440 MiB (not a peak
ownership measurement), above the preserved 512 MiB floor.

The initial build2 handoff restarted compatible ambient work: idle median
47.245 ms versus control 0.524 ms, zero consumed future frames and 23 cancellations.
That approach is rejected and preserved. Build3 transfers the producer/result
through the same queue and retains the finite horizon. An actual-worker test holds
the active producer, verifies nonwaiting native calls, one render of the requested
bucket, full-detail pixels, no cancellation and visibility rejection.

Local validity: both arms pass four terrain/topology edits and six city/forest
appearance edits against independent full redraws. These deliberately instrumented
edit fixtures are correctness/attribution evidence; the general navigation analyzer
rejects them as performance samples. Visible terrain changes still require exact
barrier rendering (candidate 750/738 ms, control 727/822 ms in this fixture).
One first distant edit took 1051 ms versus 68 ms: its trace places 976 ms in
completion wait after an existing background static-margin submission; geometry
was 23 ms and no camera poll was involved. GPU-only time is unavailable on
Parallels. This exposes the remaining cost of already-submitted speculative GPU
work at exact barriers; the bridge does not guarantee short calls for cold start,
visibility/content changes, zoom/resize or native unit takeover. Do not hide that
cost behind caller-latency numbers or reopen the closed timestamp experiments.

The full current-code integration check passes: 270 tests (one skipped), the
approved GOG compile/injection smoke, and all six production replay groups
(scrolling, reduced zoom, wrapping, resource playback, units by day and night).
The separately versioned presentation exports are present in the production DLL.
A separate public-API confirmation uses that unstaged production DLL plus the
verified current preview executable: all dense scrolling images, six independent
boundary redraw checks and three warmed zoom widths are exact. Production
scrolling medians are 80.404 ms synchronous whole request, 98.680 ms async whole
request, and 1.672 ms maximum individual call per request (3.822 ms largest call).
There were 67 simulated calls and 53 held views. The extra publication/poll delay
remains; no native FPS claim follows. Production build and preview provenance are
stored separately in `native-handoff-production-binaries/`, without substituting
a benchmark DLL for the production candidate. See `production-scroll-comparison.json`,
`production-replays.log` and `integration-receipt.json` in the evidence directory.

Evidence is under `Renderer/native/build/native-handoff-20260913/`:
`fixed-scroll-comparison.json`, `fixed-idle-comparison.json`,
`fixed-content-checks.json`, `local-change-spike.json`, focused test and build logs.
Isolated replay inputs/binaries/images are in `native-handoff-fixed-*` sibling
build directories. The accepted staged DLL and starting source archive remain
preserved; the candidate has not been staged, installed or launched.

The first five architectural responsibilities remain with the world/content,
selection, compatible submission and retained surface owners assessed below.
The sixth now has implemented native display/request coordination and queue
consumption, with **live acceptance still pending**. The next integration step is
an authorized GOG game checkpoint for scrolling, picking/overlays, unit takeover,
zoom/recenter and lifecycle/visibility changes. Any subsequent throughput work
should address measured producer/submission and speculative GPU admission costs,
not another isolated output helper.

## Accepted implementation: prepared world neighborhood

The user authorized completing preparation through GPU-ready surrounding scenery
on the resized VM (8 logical CPUs, 15,025,766,400 physical-memory bytes). This
supersedes the preceding recommendation to do general native async handoff next.
The previous full-detail DLL and source are preserved under
`Renderer/native/build/world-preparation-20260913/`. Recompare on this hardware;
the earlier four-core timings do not establish a worker-count choice here.

Connected design:

1. Extend the existing CPU compiler pool to six helpers, with configurable bounded
   result storage and demand priority. Consume ready results through the existing
   complete tile compiler/upload owner. Prepare complete surrounding terrain,
   cities, vegetation and infrastructure; parallelize expensive pure compilation
   and keep GPU work under its existing owner. Do not add a second scene framework.
2. Batch neighborhood adoption under one immutable captured-world lease instead
   of repeatedly entering the entire frame setup for single-tile preparation.
   Keep content across view changes; local proofs and GPU residency govern reuse.
   Prioritize all-side proximity, then motion direction, within captured appearance.
3. Separate a persistent static color/depth margin from the viewport-sized dynamic
   and finishing surface. Select static pass inputs into the margin during idle
   opportunities. Foreground demand draws missing visible coverage first; idle
   work fills surrounding coverage. Use existing materials, shared depth, source
   shadow selection and compatible submissions, without recursive regional scenes.
4. Retain exact native pixels/coverage/occurrence publication. Unknown appearance,
   visibility, changed gameplay and native unit actions cannot be predicted into
   display ownership. General native async camera handoff remains deferred.

Validate ownership/cancellation and independent cold redraws, then compare cold
construction, stationary animation, idle-to-scroll, reversals, continuous scrolling
and local changes. Record preparation duration, coverage, useful adoption, waste,
GPU submission/completion and checked result latency separately. Preparation does
not itself imply throughput improvement. Preserve full detail, closed resolve
findings, unreliable Parallels GPU timestamps and the 512 MiB contiguous address
headroom floor. Size the static margin from a joint target budget; extra VM RAM
does not enlarge the 32-bit process address space.

### Implemented candidate and measured checkpoint

The user accepted the displayed full-detail comparison: “Looks great. I want to
use it and have it be as fast as possible. I accept.” The connected path is now
enabled by default for eligible shared-scene views. Set
`C3X_RENDERER_WORLD_PREPARATION=0` to reproduce the existing control. Four CPU
helpers are the measured default choice; `C3X_RENDERER_CPU_PREPARATION=0|1|2|4|6` permits reproduction. Six
helpers did not outperform four on the resized VM. This is full-detail scenery
preparation, unrelated to the game's worker units.

- The existing CPU compiler admits up to 8192 captured jobs, prioritizes demand,
  and uses a 64 MiB ready reservoir for this path (16 MiB for the control).
  Configuration supports bounded reservoirs through 128 MiB; active compilation,
  private scratch and thread stacks remain additional owned storage.
- The existing GPU owner adopts complete neighboring tiles in groups of four
  under one captured-input lease, while pure terrain helpers continue compiling.
  Cities, vegetation bodies and infrastructure use their existing complete
  compilers/uploads. They have not acquired separate parallel compilers.
  Proximity in every direction precedes motion preference. The existing 64 MiB
  prefetched-geometry limit and residency/eviction owners remain authoritative.
- Selected static passes fill a persistent circular color/depth margin. A dirty
  cell owns coverage only; it owns no miniature scene, mesh or render target.
  Foreground requests fill missing visible samples; bounded background batches
  fill the remaining margin without finishing or readback. Viewport-sized
  animation, hardware resolve, incremental finishing and output ownership remain.
- Joint targets are capped at 1408 MiB versus the control's 1152 MiB. The margin
  is at most 256 native pixels per side, reduced to 192 at 2240×1192. Lowest
  sampled contiguous free address space across candidate witnesses was about
  1253 MiB, above the existing 512 MiB floor. This is sampled headroom, not a
  proof of peak memory or permission to enlarge a 32-bit process indefinitely.

Final same-binary comparisons are in
`Renderer/native/build/world-preparation-20260913/{idle-scroll,continuous,idle,edit,workers}-comparison.json`.
The binary/source control is `world-preparation-build5`; verified runs have
unchanged sources, inputs and binaries. Dense scrolling uses 28 checked requests
per arm (two 14-move sequences), not a hundred-sample tail-latency claim.

| Whole-request measurement | Two-helper control | Four-helper candidate |
| --- | ---: | ---: |
| Dense scroll after equal 3000 ms idle opportunity, median / p95 | 109.67 / 184.40 ms | 68.51 / 145.31 ms |
| Continuous dense scroll, median / p95 | 94.54 / 161.12 ms | 79.20 / 156.03 ms |
| Stationary prepared animation, median / p95, 30 demands after warmup | 0.539 / 1.139 ms | 0.499 / 0.685 ms |
| Two visible local edits, checked request latency | 784.23 / 729.60 ms | 648.11 / 669.61 ms |

The scrolling medians improve about 38% and 16% in these final pairs. Earlier
same-input controls ranged from 94.2–109.7 ms after idle and 94.5–105.7 ms without
it; do not present one pair as a precise hardware-independent percentage. Cold
construction remains roughly 7–8 seconds with no consistent improvement. Six
helpers measured 69.91 ms scrolling median and 177.30 ms p95 after idle, versus
four's 68.51/145.31 ms. More helpers are not automatically better.

Work actually removed from demand in the idle-scroll pair: tile builds drop
from 752 to 564; selected static inputs submitted during the 28 requests drop
from 12,336 to 4,833. The first nearby moves submit no static geometry. The
candidate also submits 8.75 million background native pixels across the traced
warmup/cases/horizons, including unused work; this is not free. Background CPU
submission totals 529 ms over 72 batches. Those durations neither measure GPU
completion nor add to request latency. Stationary delivery was already prepared
by the preceding implementation: its sub-millisecond wait does not mean the
underlying animated GPU frame became sub-millisecond. Composition/finishing and
completion remain the main warm-request cost; continuous scrolling still has
foreground compilation, and cold load/upload remains expensive.

### Correctness and review status

The production checkpoint passes 269 tests (one existing skip), all six native
replay groups, and 582 lifecycle / 288 body draws in each day/night unit witness.
The focused new preparation/coverage/accounting checks also pass. The prior opt-in
DLL (`21106fc708013903806e8d042a6e856d878f9af6d96bbfd16b04fff39b6b3bac`)
also passes the public-API dense replay with preparation enabled and produces
exactly the benchmark candidate's pixels. It is isolated under
`Renderer/native/build/world-preparation-production/`. The first
whole-suite attempt lacked VM process permissions for three Windows-only tests
and exposed one outdated support-ring test harness declaration; the declaration
was updated and the complete approved VM-enabled rerun passed.

After acceptance, the default-enabled production build again passed 269 tests
(one existing skip) and all six native replay groups. Public-API dense and
boundary replays with both preparation controls unset verified four helpers,
background coverage preparation, exact accepted dense pixels and six passing
boundary checks. The exact tested DLL was staged to `Renderer/bin/C3XRenderer.dll`:
`d1f31de5919a0879c873dc2e1ab0a7b689c012783629c3086a0fa5db0136fa8f`.
Receipts and the previous staged DLL are preserved under
`Renderer/native/build/world-preparation-20260913/`; see
`accepted-default-staging.json`. The user's live capture remains unchanged.
No installation, game launch or Git mutation was performed.

The final candidate passes all six boundary checks, four topology changes and
six fixed-topology city/vegetation content changes against independent cold
renders. Dense repeat cameras are exact, with no fallback or device recovery.
Four/six-helper images are identical. Existing output coverage, native overlays,
picking, visibility and caller ownership remain authoritative. No injected
source or patch-table change is needed.

**Visual acceptance received.** Against the full-detail control, the dense
initial/result image changes 6122 of 2,670,080 pixels (0.23%), maximum channel
change 62. See the preserved `visual-comparison.png` and `visual-difference.json`
in the evidence directory. A bounded static-only diagnostic still changes 6201
pixels, so animated composition does not explain it. Preserving the original
viewport projection arithmetic did not change a single candidate pixel; that
extra machinery was removed. The remaining difference belongs to static surface
addressing/submission, with finer attribution unresolved. Its visual acceptance
comes from the user’s review, not from full asset detail or warm/cold consistency.
The user accepted this comparison at the strategic visual checkpoint and
therefore authorized normal staging under `Renderer/README.md`. No references
were replaced. The measured four-helper path is now the shipping default;
verification exercises unset environment controls rather than relying on an
opt-in flag. Installation and game launch remain outside this authorization.

Other preserved findings: the first connected build redirected explicit MSAA
attachments through the legacy target helper; explicit scene passes now retain
caller-supplied targets. A bounded dirty-batch limit initially returned unmerged
scan rows; joining them reduced background submission overhead. Build/replay
failures and both diagnostics remain in the evidence directory. Parallels GPU
timestamps/event-query limitations remain closed findings, not speed evidence.

### Responsibilities after this change

| Architectural responsibility | Actual ownership and remaining scope |
| --- | --- |
| Persistent world/instances and reusable content | Existing `CapturedScene`, `ResidentContent`, shared assets and local instance handles remain the sole owners. The new producer populates those owners ahead of selection; GPU eviction does not erase world identity. Known captured surroundings can be prepared without guessing camera direction. Unknown future appearance/gameplay is not invented. |
| Local content validity | Existing world/coast/river/appearance proofs govern compilation and adoption. Static coverage uses complete footprint changes plus environment, viewport, depth and device context. Failure dirties the target; changed content must pass current proofs before reuse. Tested responsibilities are complete for the owned map categories. |
| World → view selection | Construction publishes handles; current occurrences assemble the pass inputs and the spatial index selects real submissions. A captured view plus bounded margin defines selection, separately from world identity. Not every world mesh must stay on the GPU. |
| Compatible pass submission | Existing material ordering, shared mesh instances and 32-page shadow batches are reused. Explicit scene targets are now caller-owned; foreground and background use the same static submission path. Separate city/body parallel compilers are not claimed. |
| GPU color/depth and output | Persistent static margin plus viewport-sized dynamic/finishing surfaces is implemented for the bounded city profile with waves/reflections off. Other profiles retain existing execution. The displayed static shading differences have been accepted. |
| Caller-driven asynchronous integration | Renderer workers, cancellation, copied snapshots and exact eligible idle publications exist. The opt-in bridge above now connects native camera begin/poll/publication and displayed fields; live cadence/interaction acceptance remains an explicit integration responsibility. Civ III still calls; the renderer does not notify it or request redraws. |

Deferred wonders/Districts, native unit-action ownership and source-asset
contracts are unchanged. No Civ III install/launch or Git operation is part of
the accepted preparation checkpoint. The native handoff above is its subsequent
implementation; remaining live validation is distinct from measured producer or
submission work. Do not restart an output-helper queue.

## Previous implementation: bounded concurrent preparation

The default full-detail path now has **two CPU helpers**, plus the existing render
thread and single GPU owner. Helpers compile ground, surface details, relief and
vegetation floors from resident shared assets and captured world inputs. Each has
private river/query/layout scratch. Results carry world, coast and river proofs;
adoption validates them before entering the existing compiled-world owner and
selected pass submission. The render thread compiles unstarted demanded work
itself and joins already-running work, avoiding duplicate builders.

CPU preparation is bounded to the captured render/prefetch inputs (at most 8192
jobs), a 16 MiB completed-content reservoir and two river pages per worker. Active
results stop at an 8 MiB buffer threshold; capacity growth can cross it before the
next cancellation check. Byte-based backpressure leaves workers useful while the
GPU owner consumes earlier content. Newly selected demand gets priority over
unneeded speculative results; valid ready demand is protected, and stale CPU
results are rejected before they can suppress replacement jobs. Scratch, active results, thread stacks and
existing GPU owners are additional memory, not part of the reservoir claim.
Configuration, world mutation, reset and benchmark reset join CPU readers before
changing their borrowed inputs. No game pointer or D3D context reaches a helper. Preparation selection and view
assembly share one complete per-frame validity receipt, keyed by geometry epoch
and occurrence anchor. GPU residency is still checked on every use; the 24-byte
receipt is charged to the compiled-content budget. Only stable eligible views
acquire an isolated front for speculative GPU work; moving-camera synchronous
results keep their existing borrowed-output lifetime.

After successive authoritative requests confirm a stationary view, the GPU owner
also prepares at most two future quantized resource-animation frames. Each uses
the existing full-detail scene/pass/finishing path and owns immutable pixels.
Consumption requires exact camera, ordered appearance/topology, native identity,
environment and time-bucket agreement. Foreground changes cancel the horizon;
camera movement does not start it. Snapshot plus speculative publications share
a 32 MiB cap. There is no wall-clock game loop, game-state prediction or redraw
callback. Caller consumption alone advances the finite horizon.

`C3X_RENDERER_CPU_PREPARATION=0|1|2|4` selects the control or helper count; unset
means two. `C3X_RENDERER_PREPARE_AHEAD=0` disables future frames; unset enables
them only for the eligible retained city profile with waves/reflections disabled.
The current custom game configuration already selects that profile. Full terrain
density, meshes, shaders, native overlays, visibility and picking remain intact.

This is scene preparation, unrelated to Civ III worker units. Cities, tree bodies,
infrastructure and units retain their existing content/pose owners; they have not
all acquired new parallel compilers. General caller-driven native asynchronous
camera handoff is still an integration responsibility. A camera miss still waits
for its requested result. No injected changes or new patch symbols are needed.

Preserved control/source and all new evidence are under
`Renderer/native/build/ahead-preparation-20260913/`; the original staged full-detail
DLL is `starting.dll`. The first shallow queue made cold rendering slower even
with four helpers. Replacing tile-count backpressure with byte-based admission
and allowing the foreground to take unstarted tasks is a materially different
scheduler: matched corrected runs reduced the initial dense request from
12.170 s to 7.567 s with two helpers, versus 7.730 s with four. Dense scrolling
remained about 90–100 ms. Four helpers offered no useful gain on the four-core VM.
The corrected scaling run retained more than 1.5 GB of contiguous address headroom. No GPU timestamp
claim is made; the known Parallels timestamps remain unreliable.

The shallow queue and the first combined `ahead-final-*` comparison are retained
as rejected controls. The combined path initially slowed visible edits because
speculative content blocked new demand. Priority admission corrected that. A
bounded endpoint diagnostic then attributed about 5.7 ms of moving-camera cost to
an isolated output copy performed even when no future rendering was scheduled.
Stable-view ownership removed that cost. Per-frame validity receipts avoid checking
the same content twice, but neither correction establishes a scrolling win.

### Final matched work-ahead comparison

`ahead-ownership-build` is the source-matched benchmark build. Runs named
`ahead-ownership-{dense,idle,edit}-{control,candidate}` use identical binaries,
assets, definitions and authoritative requests. The control explicitly disables
both new strategies; the candidate enables two CPU helpers and exact work ahead.
The city profile uses full detail, waves/reflections off, at 2240 × 1192 pixels.

| Complete result | Control | Two helpers + work ahead | Interpretation |
| --- | ---: | ---: | --- |
| Dense initial view, assets loaded and scene reset; two cases | 11.777, 11.933 s | 7.141, 7.048 s | About 40% less latency; serial surface compilation overlaps across workers. |
| Dense scroll, 28 requests; median / p95 | 87.822 / 154.080 ms | 94.997 / 165.754 ms | Slower; mean 97.095 → 100.434 ms. No scrolling-speedup claim. |
| Two visible topology edits; mean | 1372.933 ms | 746.723 ms | About 46% less latency; each checked against an independent cold render. |
| Two distant topology edits; mean | 52.346 ms | 56.355 ms | Slightly slower; no local-edit gain claimed for this case. |
| Stationary resource animation, 30 requests at 67 ms cadence; median / p95 | 34.302 / 39.800 ms | 0.590 / 0.854 ms | Preparation moved ahead of demand; not a throughput gain. |

A same-compiler idle control with helpers enabled and future frames disabled took
31.786 ms median; all 30 images match the prepared publications exactly. The
candidate actually performed 40 future renders, cancelled one and consumed 39,
including warmup: 1532.076 ms of traced work, median 36.707 ms per preparation.
Two consumptions joined active work. Future frames built/uploaded no geometry.
The 67 ms schedule uses absolute deadlines; saved bitmaps can delay demand. In
the same-compiler pair, median deadline-to-completion was 45.082 → 8.567 ms,
including those delays. This is standalone scheduling evidence, not native FPS.
The earlier unpaced witness also passed exact parity, but bitmap writes outside
API calls provide preparation time; its lower waits cannot be called higher FPS.

CPU helpers preserve vertex counts, detail, pass selection and GPU work. They
remove compilation from the serial consumer when a proven result is available;
the first dense construction adopted 813 helper results. Jobs already running are
joined, and demanded unstarted jobs can be compiled by the consumer. The final
repeated dense run records 2815 adoptions including bootstrap/cold cases and
scrolling. These are reuse counts, not additional speedup evidence. The GPU owner
still performs requested rendering and readback. Dense scrolling remains dominated
by completion waits (roughly 44 ms median in the control); more CPU workers cannot
remove that cost. Do not expand helper count or another output-helper experiment
on the strength of cache-hit counts.

The maximum completed CPU reservoir in this pair was 12,722,492 bytes of its
16 MiB cap; exact future-frame ownership peaked at 24,278,372 bytes of its 32 MiB
cap. The final dense candidate retained at least 1,520,304,128 bytes (1.42 GiB)
of contiguous free address space. Across the eight final workloads the minimum
was 1,368,051,712 bytes (1.27 GiB), above the 512 MiB floor. Private scratch, active
builds and thread stacks are additional to the preparation caps. Four helpers
remain an explicit measured option, not the default: their corrected cold result
was slightly slower than two and used more aggregate compiler time.

The CPU compiler changes 13 of 2,670,080 pixels by one channel level in the dense
control comparison. This is a new recorded precision difference, not an approved
visual change or a detail cut. Worker counts agree with one another; full-detail
controls, assets and fixed references are preserved. Inspect the
[control image](../native/build/ahead-ownership-dense-control/zoom.bmp.case0.bmp)
and [candidate image](../native/build/ahead-ownership-dense-candidate/zoom.bmp.case0.bmp);
the archived difference receipt records the exact count and maximum channel delta.

The full checkpoint before the final scheduler/ownership corrections passed 268
tests (one existing skip), scrolling, reduced zoom, wrapping, resource
animation/scroll/removal and day/night unit bodies/lifecycle/terrain parity.
The final native content fixture passes four topology changes and six city/forest
appearance changes against independent cold renders; the latter hold topology
revision fixed. The old unit witness incorrectly expected identity-driven idle phases; it also
failed against `starting.dll`. It now supplies distinct native cursors and checks
that identity/time cannot change a fixed cursor. Production unit behavior is
unchanged. Final corrections pass 48 focused executable tests and the production
full checkpoint: **269 tests (one existing skip), plus all six native replay
groups** for scrolling, reduced zoom, wrapping, resources and day/night units.
Each unit replay passes 582 lifecycle draws, 288 body draws and exact post-unit
terrain parity. The production receipt is preserved as
`ahead-preparation-20260913/production-integration.json` beneath the build directory.

### Evaluation delivery

The verified production candidate is staged at `Renderer/bin/C3XRenderer.dll`
under the user's earlier request to finalize changes for Civ III testing. It is
an **evaluation build**, not visual acceptance or a live-game pass. Candidate,
archived production DLL and staged DLL have SHA-256
`0bdf99e7e405a34a1e49cd5899d84ab95e1fce35e2ceb51803fe1873eb262f65`.
The rollback is `Renderer/native/build/ahead-preparation-20260913/starting.dll`,
SHA-256 `ae7a7306adbe756af8c4b9cd73375ea242b616570ec95bd423d6b8143e22b70e`.
`delivery.json`, `production-integration.json`, `ownership-acceptance-summary.json`
and matched comparison files in that same evidence folder record provenance.
The user's live scene remains unchanged. No installation, game launch, Git mutation
by the agent or native asynchronous camera integration is included in this work.

The next larger responsibility is the caller-driven native asynchronous camera
handoff below: prepare authoritative snapshots and adopt complete results on later
Civ III calls while keeping displayed pixels, overlays, visibility and picking
consistent. This targets native blocking and presentation ownership; the measured
GPU/completion cost remains a throughput limit. Additional worker count or new
speculation strategies need useful-work and complete-result evidence.


## Completed foundation: shared geometry and terrain patches

The user selected **full detail** for Civ III evaluation. The shared-mesh checkpoint
staged the original terrain density (`C3X_RENDERER_PATCH_PIXELS=0` by default), with
shared connectivity and tree instancing enabled in the eligible retained profile.
Reduced-detail measurements remain preserved diagnostics, not the delivery mode.
There is no temporary low-detail image or automatic refinement in this build.

The existing CPU-phase diagnostic places the largest geometry cost in terrain:
2068 ms in natural ground, 3203 ms in relief, and only 40 ms in tree expansion on
the preserved dense initial scene. Therefore a forest-only instancing claim would
miss the dominant cost. The connected implementation has two responsibilities:

- Terrain patches separate shared canonical connectivity from locally valid
  surface values. CPU layouts and GPU index buffers are reused by all compatible
  patches; color and shadow passes consume the same content. An explicit
  screen-space detail policy selects a common power-of-two mountain lattice for
  the view and its collar neighbors. Zero preserves existing density; positive
  settings are visual candidates requiring review, not equal-quality speedups.
  The pixel setting describes nominal projected horizontal grid pitch, not a
  proven image-error bound on sloped relief.
- Immutable tree placements reference one resident body mesh per generic asset.
  The existing world owners retain placement and exclusion/height dependencies;
  view assembly supplies authoritative projection; selected color inputs form
  ordered instance batches, and shadow casters consume the same placement data.
  Source assets, density, placement, clipping and materials remain unchanged.

The existing mesh budgets remain in force. Shared terrain indices are capped at
4 MiB and charged in geometry admission; instance source meshes are capped at
32 MiB; each pass stream is capped at 1 MiB. Compatible batches append into unused
stream storage and discard only at wrap, avoiding a full stream rename per batch.
Device/reset/eviction and publication continue through existing owners. Reflection/legacy profiles retain their existing
geometry execution. Native asynchronous handoff remains outside this work.

Preserve the current uncommitted source and verified binary control; test shared
layout/placement ownership and both GPU passes, then compare stationary, dense
scrolling and local edits through complete checked results. Distinguish unchanged
quality effects from detail-policy tradeoffs. No Git mutation, installation or game
launch. Evidence: `Renderer/native/build/mesh-instances-20260913/`.

Current verification passes: 259 production integration tests (one skipped),
resource animation/scroll/removal and common-depth checks, 21 focused ownership
checks, and final boundary/zoom/content replays. The latter include four topology
changes and six city/forest appearance changes checked against independent cold
renders. Sources, binaries and runtime inputs remain unchanged within each run.
A stale extracted test fixture now includes the shared instance type. The initial
suite attempt used an older Python without Pillow and lacked VM access; its errors
are preserved and superseded by the successful bundled-runtime integration run.

A first complete comparison did not establish a scrolling win. Its initial
per-batch stream discarded 1 MiB for every small submission; the connected
correction uses bounded append/no-overwrite storage and discards on wrap. The
D3D content replay now submits 50 color and 167 shadow batches with one
initial discard per stream, then none on the following request. Earlier runs,
shader-cache warmups, transport failures and the one dense run overlapping CPU
tests remain preserved; final acceptance timings run without builds or tests.


### Historical shared-mesh comparison and delivery

The preserved starting source/binary and final `shared-instances-stream-build`
use the same art, definitions, world data and request sequences. Sixteen enumerated
implementation inputs differ: the new instance shader and its generated adapters,
compiled shader caches/provenance, and the shader-cache ignore file. Every run
verifies its own unchanged inputs, source closure and binaries. The normal city
profile has waves/reflections off. Dense scrolling uses two 14-request sequences;
stationary animation uses 30 requests after ten warmups; edit means contain two
distant and two visible changes, each compared with an independent cold redraw.
Initial views start with assets loaded and renderer caches cleared.

| Whole request | Starting control | Full detail | 3-pixel detail candidate |
| --- | ---: | ---: | ---: |
| Initial dense view | 13033.80 ms | 12354.74 ms | 8611.38 ms |
| Dense scrolling | 99.05 ms | 99.76 ms | 95.35 ms |
| Stationary animation | 27.56 ms | 26.32 ms | 25.23 ms |
| Distant topology change | 70.95 ms | 68.11 ms | 59.17 ms |
| Visible topology change | 1555.52 ms | 1403.20 ms | 945.94 ms |

Full detail improves initial construction by 5.2% and visible edits by 9.8%; dense
scrolling is effectively unchanged. The detail candidate improves initial work by
33.9%, visible edits by 39.2% and scrolling by 3.7%. Stationary static work was
already skipped; its timing variation does not establish an instancing benefit.
These are standalone checked-result latencies, not native presented FPS.

Initial geometry uploads fall from 317762132 to 278106174 bytes at full detail,
or 197937814 bytes with the detail option. New shared asset meshes total 112616
bytes; initial color/shadow instance streams add 179904/264256 bytes, reported
separately from geometry uploads. Dense requests still build 26.86 entries and
upload 421114 geometry bytes, adding about 7090/20622 instance bytes. They mostly
reuse terrain already constructed. Shadow source draws fall from 284.86 to
170.43 per request without changing the 3.14 rebuilt pages. The append correction
eliminates discards during the measured dense playback. Smaller uploads and fewer
submissions have not eliminated its approximately 50 ms completion wait.
Parallels GPU timestamps remain unsuitable for attributing that wait to a specific
GPU stage. No new output-helper campaign or native handoff was substituted.

Renderer target accounting stays at 1165363200 bytes, distinct from actual process
residency. Shared mesh/index/stream caps stay bounded. The minimum reported
contiguous free region across full-detail workloads is 1351.74 MiB; the detail
candidate's minimum is 1432.34 MiB. Existing cache and target budgets were not enlarged.

Across all final saved images, full detail differs from the starting renderer in
at most 23 of 2670080 pixels, with maximum channel difference 5/255. This is new
numerical variation in equivalent geometry/shader calculations, not byte-exact parity or an
already accepted visual change. The detail option changes up to 9.8% of pixels,
including relief silhouettes and shading; it is not an equal-quality speedup.
Both preserve same-mode cold-render parity and returned ownership checks. Review
`mesh-instances-20260913/final-terrain-comparison.png` and
`final-terrain-detail-crop.png` beneath the native build evidence directory.
No fixed references were replaced; `C3X_RENDERER_PATCH_PIXELS=0` remains default.
`--patch-pixels 3` selects the tested detail candidate in the existing harness;
`--legacy-tree-meshes` isolates the old tree geometry mechanism when needed.

The verified production DLL is staged **for evaluation**, under the earlier user
authorization, at `Renderer/bin/C3XRenderer.dll`, SHA-256
`ae7a7306adbe756af8c4b9cd73375ea242b616570ec95bd423d6b8143e22b70e`.
Its passing production receipt, prior DLL and exact replacement are preserved in
`mesh-instances-20260913/evaluation-staging/`. No installation, game launch or Git
mutation was performed. Native caller-driven asynchronous handoff and live cadence
remain explicit integration responsibilities. Reflection/legacy execution and other
object categories retain their existing paths. The user selected full detail; the
reduced terrain option remains disabled for delivery.

Final comparison data/scripts and the starting source snapshot are preserved in
`Renderer/native/build/mesh-instances-20260913/`. Final candidate runs are
`Renderer/native/build/shared-instances-final-{dense,stationary,edit}-{full,detail3}`;
control runs live in that snapshot. Rehydrate `control-snapshot` at
`Renderer/validation/mesh_instance_control` before reproduction, because runtime
input discovery excludes directories beneath `build`. The new shared path replaces
baked trees in its eligible profile; there is no separate renderer framework.


### Remaining native integration recommendation

Complete the native caller-driven asynchronous preparation/publication handoff,
including the displayed-view contract described in
[native asynchronous presentation](native_async_presentation_audit.md). The retained
world and selected passes supply full-detail work to the existing bounded worker
queue. Separate requested state from the actually displayed view so pixels, native
overlays, visibility and inverse picking always agree. A later Civ III rendering
call may adopt a compatible complete publication; completion must never notify
Civ III or request a redraw. Pending input should coalesce without starving useful
full-detail completions. This requires more than binding the existing begin/poll
exports: the no-ready-image case must be correct before enabling async camera mode.

Validate stationary animation, sustained navigation, zoom, visibility loss, local
edits and unit takeover through that complete integration. Measure native caller
blocking, input-to-correct-presentation latency, displayed-frame age and complete
render-job time separately. This targets responsiveness and finishes the remaining
end-to-end ownership responsibility; it does not claim to remove the roughly 50 ms
pending-GPU-work wait or make a full-detail job intrinsically faster. Preserve the
current full-detail control, closed resolve findings and GPU-timestamp limitations.
This is a recommendation, not a new native implementation started in this handoff.

## Objective and control

The user's connected architecture instruction supersedes the former experiment
queue. Complete ownership and behavior through the existing renderer; GPU mesh
residency remains bounded. Preserve native ownership, asset/visual contracts and
deferred scope. Native asynchronous handoff remains a separate integration gap.

Incremental output is finished. At the start, candidate source verification passed and
the staged DLL matched its receipt (`b5cd1b08…c1f4d`). Its final dense comparison
was 119.40 → 103.28 ms; stationary work showed no improvement. Exact finishing,
edit and boundary witnesses passed. Evidence, rejected resolve approaches and
staging are preserved in [the output record](history/retained_output_completed_20260913.md).
Full hardware resolve and output handling are frozen for this implementation.
Parallels GPU timestamps and event-query completion are unreliable attribution.

## Six responsibilities: code assessment

Completion below concerns ownership and behavior for the admitted map plane, not
permanent GPU residency, an uncaptured game simulation, or every optional backend.

| Goal | Starting gap | Implemented responsibility and limits |
| --- | --- | --- |
| Persistent world/instances and reusable content | The 8192-entry appearance cache forgot identities; world records also held camera observations. | `CapturedScene` now retains canonical render appearance/revisions/bindings separately from a reusable observation pool. Camera departure and GPU eviction preserve identity. `ResidentContent` and existing mesh owners remain the only GPU lifetime authority. Never-captured appearance stays unknown. Metadata admission is bounded and fails explicitly. |
| Local revisions and complete content validity | Appearance checking was split; outer hits bypassed forest/city exclusion validity; river queries did not propagate complete dependencies. | The instance fast path and ordinary cache lookup share validity. Own appearance, neighboring city presence/appearance, semantic neighbors, coast nodes, world samples, local river nodes and river query cells govern reuse. All three compiled tiers carry river proofs. Native population/labels/selectors and unit state do not invalidate static content. Broad natural/ground topology keys are replaced by these proofs. Asset/device/world-basis contexts still invalidate globally. |
| Spatial selection separate from world construction | Compilation appended active draw lists; circular spans scanned every layer. | Compilation publishes protected handles. A separate assembly consumes current authoritative occurrences; the view index selects actual static pass inputs for damage spans, with exact intersection and ordered deduplication. Current capture remains the admission set. The index is view-scoped; world identity and compiled validity outlive it. Dynamic pose lists retain their small scan. |
| Compatible submissions through explicit passes | Ordered layers/page limits existed, but selected inputs were incomplete. | Selected static/dynamic inputs feed existing ordered material submissions and bounded 32-page receiver batches. Main/caster/resource ownership and common depth are preserved. This completes representative pass ownership; selected tree placements now feed hardware-instanced color and shadow submissions. Reflection and other object categories retain their existing execution. |
| GPU reuse, finishing and readback | Incremental output already finished. | Complete for the tested city profile, waves/reflections off and bounded extents. Circular color/depth, sparse restore, incremental finishing, hardware resolve and consolidated readback remain unchanged. Other profiles retain existing output execution. |
| Caller-driven asynchronous preparation/publication | DLL preparation/publication and exact queued joins existed. | Bounded CPU preparation and exact future ambient publications run ahead of demand inside the DLL. The opt-in GOG native handoff now owns requested/displayed views, the authorized camera hook and exact presentation leases. General input/overlay/picking acceptance remains pending; the supplied live trace does not establish that mode was active. Exact content/visibility/zoom barriers can still wait. CPU unit pose preparation extends shared content to directed native units without changing animation or adding renderer notifications/redraws. |

## Ownership and invalidation details

The world topology retains authoritative whole-map terrain. Canonical appearance
records retain only render-relevant state observed through the capture contract;
full records take precedence over duplicate lightweight halos. The observation
pool is capped at 8192 occurrences. Persistent record admission and conservative
metadata accounting are capped at 128 MiB, rather than evicting logical identity
on camera pressure. Explicit renderer/world reset restarts that ownership.
Compiled meshes remain evictable under the existing geometry/CPU budgets.

`NaturalWorld` now owns a dependency chain from copied world inputs through river
pages to exact ordered query-cell values and compiled consumers. Missing cells
are observations too. Source proofs survive page eviction, so validating unchanged
meshes does not reconstruct fields. On an actual source change, the exact queried
cell values determine consumer validity. Page construction still uses the original
expressions and the globally invalidated 16-page cache. This differs from the
closed page-cache experiment: it establishes **compiled mesh** validity, not merely
page reuse. Source inputs are capped at 4096/page, cell metadata at 512 KiB/page,
and compiled observation sets at 256 cells. Retained proof payloads are charged
conservatively to existing compiled-cache budgets, including shared data.

The first broad proof propagated every page input directly to every consumer.
It passed redraw parity but increased the small visible-edit cost to about 3.1 s.
Exact cell proofs restored the same workload to about 1.29 s. That intermediate
source and failed control setup receipts remain in the evidence directory.

## Connected design

1. Separate persistent canonical appearance/identity from current observations.
   Retain identities across camera departure and GPU eviction; bounded admission
   fails explicitly instead of silently forgetting the world. Only current full
   captures authorize appearance, selection or replacement.
2. Carry complete appearance and actual observed dependencies through every
   compiled tier. Forest exclusions observe city appearance, including absence;
   river fields connect construction and height-callback inputs to exact query-cell
   proofs retained by consumers.
   Remove broad mesh keys only with these proofs. Preserve the globally invalidated
   small river-page cache: this completes compiled-content validity rather than
   reopening the rejected page-cache experiment.
3. Compilation publishes protected content handles. Separate view assembly consumes
   current authoritative occurrences. Reuse the contributor index to select actual
   ordered pass inputs for circular damage, rather than only raster keys. Preserve
   exact intersection, layer/occurrence order, bindings and shadow-page limits,
   with bounded fallback scans.
4. Preserve starting source/build and a same-binary control. Verify portable
   ownership/validity/selection contracts and independent replay correctness, then
   whole-request stationary, dense-scroll and local-edit comparisons. Report
   eliminated work and initial construction separately. Automated tests do not
   imply staging, installation, launch or visual acceptance.

## Previous world/view checkpoints

Implementation: independent topology edit and six city/forest appearance edit
checks pass; the latter hold topology revision fixed and exercise growth, removal,
style changes and reversals. All preserve exact pixels, replacement flags and
animation ownership against cold redraws.

Final-source portable ownership, dependency, selection and measurement checks pass
(34 tests). The shared river witness preserves all 312 original-expression samples,
wrapped lookup, revision invalidation and the 16-page LRU. Native final-source
replays pass six retained-boundary cases, zoom and ambient transitions, four
topology edits and all six fixed-topology appearance edits. Each replay receipt
confirms unchanged sources, binaries and runtime inputs. No new visual difference
is expected or requested for approval; earlier accepted visual contracts remain.

Production verification passes via `python3 Renderer/renderer.py integration
resources --renderer-only`: 257 tests run, one skipped, plus config-off/playback,
scroll/removal parity and native-depth checks. Three extracted C++ fixtures were
updated to include the new dependency owner and preserve the correct extraction
boundary; the ground-cache witness now checks both local validity and the old
global-revision diagnostic control. Parallels transport failures are preserved
in the earlier logs and superseded by the successful final run. No injected code
changed, so injected compilation was not required.

Under the earlier authorization to add the renderer for Civ III evaluation, the
verified production candidate is staged at `Renderer/bin/C3XRenderer.dll`, SHA-256
`22e87ecc770005ab3aa84074ab6877072b2b0755352301cb21a7a846370ed96b`.
The previous DLL, new DLL, source/build receipt and passing integration receipt
are preserved in `world-view-20260913/evaluation-staging`. No installation or Civ III
launch was performed. Native live acceptance remains unmeasured.

### Previous world/view whole-request comparison

The preserved starting binary (`incremental-output-delivery-build`) and final
`world-view-validated-build` use identical runtime dependency hashes and matched
normal-tier city profiles, waves/reflections off. Dense scrolling comprises two
14-request runs; stationary animation measures 30 requests after 10 warmups. Local
edits contain two distant and two visible changes, each checked against a separate
cold redraw. Initial construction is excluded and reported separately. Edit runs
use a 180-second limit; the earlier timed-out control is retained and excluded.

| Workload | Control | Current | Observed change |
| --- | ---: | ---: | ---: |
| Dense scrolling | 109.30 ms | 103.96 ms | 4.9% faster |
| Stationary animation | 28.14 ms | 26.77 ms | 4.9% lower; no new eliminated rendering work |
| Distant topology change | 65.07 ms | 63.96 ms | 1.7% faster |
| Visible topology change | 1459.74 ms | 1495.37 ms | 2.4% slower |

Dense preparation falls from 36.18 to 31.62 ms, with 1030.29 ready instances per
request. Both arms build 26.86 entries and upload 421114 bytes per request; resolve,
finishing and readback work are unchanged. A separate same-binary attribution pair
(`ready-dense-control` / `ready-dense`) measured 120.03 → 114.73 ms and the same
build/upload counts: the old submission scan inspected 13979.71 candidates versus
921.93 indexed static candidates plus 507.43 dynamic candidates. Selected inputs
and batches remained identical. That pair isolates the new mechanism; it is not
the preserved-source headline control.

Stationary timing varies despite identical output work, so do not attribute its
entire observed improvement to this architecture. GPU completion remains about
50.69 ms of the current dense request, versus 52.60 ms in the control. Those are
CPU-observed waits, not reliable GPU-stage timestamps. Initial dense preparation
still takes 12.90–13.73 seconds (control 12.59–12.86). Visible edits still compile
81 entries, reuse 976, and spend 1247.63 ms in geometry (control 1195.81). The local
validity correction does not make the existing terrain compiler cheap. This is a
modest scrolling improvement, not a large universal speedup or a native FPS claim.

All common saved comparison images are exact: 5 dense, 32 stationary and 3 edit
images. Renderer-target allocation accounting is unchanged at 1165363200 bytes;
this is not a process-residency measurement. Dense world metadata averages
2.40 MiB. The smallest reported contiguous free region is 1532 MiB for candidate
dense scrolling and 1443 MiB for its local edits. Existing geometry/output caps and
the new metadata/proof caps continue to bound ownership in the 32-bit process.

### Evidence and reproduction

`Renderer/native/build/world-view-20260913/` preserves the starting commit/source,
intermediate broad-proof implementation, comparison script/JSON, logs, and rejected
setup receipts. Complete runs are `Renderer/native/build/world-view-final-{dense,
stationary,edit}-{control,candidate}/`; each contains input/binary/source receipts,
request endpoints, ownership checks and trace data. `C3X_RENDERER_RETAINED_WORLD=0`
remains a same-binary diagnostic control; it is not the historical source binary.

The isolated original-source snapshot is preserved under the evidence directory
as `control-snapshot`. To rerun it, copy it to
`Renderer/validation/control_snapshot_20260913` because runtime discovery excludes
paths beneath `build`. Its local runtime inputs are verified hard links/copies,
not redistributable assets. Its measurement harness also hashes the 111 textures
actually referenced by the city material table outside the declared pack folders;
the original dependency inventory omitted those files. Both final arms include
the same complete 10315-file runtime set. No assets were discarded.

### Remaining implementation responsibility

Native caller-driven asynchronous handoff remains explicit: capture/prepare on
Civ III calls, publish only matching completed output on subsequent calls, and
retain exact visibility, overlay, picking and config-off behavior. No unsolicited
renderer callback or redraw mechanism is appropriate. Shared tree instancing now
completes a representative asset/placement/pass path, while broader category and
reflection instancing remains outside the measured profile.
Future performance work should address the measured geometry construction and GPU
completion costs; this change does not reopen output-helper experiments or expand
native ownership, wonders, or Districts scope.
