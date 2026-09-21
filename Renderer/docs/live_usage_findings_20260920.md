# September 20 live capture: slow gameplay

User reports very slow gameplay. The completed capture used staged production
DLL SHA-256 `3b32a1b1fad9f91a9c004638b620aa02b639fe25236d4675a09d35cb0a8a8754`,
Standard 100×100 / 5,000 actual tiles, 1119×1192 output and 128×64 tile basis.
Waves and reflections were on; expensive profiling was off. Both collectors
finished and the game exited with code zero. This does not establish that every
requested manual action was performed or certify the earlier crash sequence.

Local evidence: `native/build/live-captures/20260920-214002-92e441/` contains
`renderer.log`, `frames.csv`, `session.json` and derived `analysis.json` with
source hashes. Preserve the capture. The existing trace analyzer can regenerate
the DLL analysis; frame statistics use positive `msBetweenDisplayChange` values
from non-dropped rows, separately for each swap chain, and nearest-rank percentiles.

## Measured behavior

- Both device records report hardware D3D11 through the Parallels display adapter,
  not WARP. This identifies the guest driver, not physical host GPU utilization.
- The gameplay swap chain has 1,579 submissions and 1,551 non-dropped display
  intervals: mean **74.65 ms**, median **33.38 ms**, p95 **450.00 ms**, maximum
  **1,400.00 ms**. That is **13.40 displayed updates/sec** over the measured
  intervals, with **66 gaps above 500 ms**. A separate startup swap chain has
  47 displayed intervals. These include native UI/modal updates, not exclusively
  unique animated map frames; they are not input-to-display measurements.
- Only **1 of 311** native map composites used `gpu_map=1`; the following
  **310** used the CPU bitmap insertion path. There are no logged
  `native-resident-present` events. GPU presentation alone does not establish
  resident map composition.
- **128 of 130** logged shared-scene passes read back pixels. Their completion
  waits total **30.91 seconds**, with median **273.20 ms**, p95 **419.37 ms**,
  maximum **545.22 ms**. Those waits include outstanding GPU work; they do not
  isolate transfer bandwidth or pure shader cost. The outer `frame` readback
  field misses this inner scene wait and must not be used to dismiss it.
- Initial native map construction/composition took **5,463 ms**. The subsequent
  310 native map calls have median **11.40 ms**, p95 **569.19 ms**, maximum
  **1,098.15 ms**. Cheap unchanged calls must not conceal the expensive updates.
- All **724** logged unit draws used direct GPU composition. Units are not
  universally falling back to CPU drawing when the map falls back.
- All **1,456** prepared-area records were marked cancelled, including 93 that
  completed their local preparation before cancellation. Their reported
  preparation durations sum to **44.49 seconds** of worker activity; this is
  not an additive UI-thread stall total. The work is not evidence of successful
  prepared-view adoption.
- The final sampled CPU-screen counter reports 1,560 presentations and **3.59 GB**
  of cumulative uploads. This includes legitimate UI work and is not all avoidable
  map traffic. Never sum cumulative `native-screen` counters across log records.

## Ownership repair and evidence

The second capture, `native/build/live-captures/20260920-215506-ed1f15/`, used
diagnostic DLL `88b2d64d58248cd66650ff7b24362be147ef644027096b225ff302e2bf8c82c9`.
Only **1 of 51** map composites remained resident. The first revocation is an
unscoped public DC access (`operation=8`, `context=0`) at 13.146928 seconds,
immediately after the first GPU composite at 13.133585 seconds and before
map-complete/unit drawing. The stack walker returned only a renderer frame;
it does not independently identify the game caller.

Source audit found an unconditional matching operation in the native map tail:
`FUN_004e48d0` initializes `OpenGLRenderer` before testing whether any colonies
need outlines. That initializer acquires the map's public DC even with no lines.
The executable lifetime test now exercises this exact acquire through the actual
hooked JGL vtable and reproduces permanent loss of eligibility. EXE initialization
and GDI+ endpoints are explicit stand-ins in the extracted-hook harness; this is
source plus native-operation evidence, not a recovered live stack or a new live
performance measurement.

The fix routes already-resident targets through the existing DLL overlay pass.
It preserves endpoints, color, opacity, width, GL stippling, GDI+ dash spacing
and clipping. The injected bridge records only the current native target and
style; worker inputs are copied values. CPU UI and real CPU access retain their
barriers and original backend, including a safe mid-scope fallback. No new patch
table entries or executable hooks are needed; installation must update both DLL
and injected bridge. See the [patch ledger](civ3_patch_dependency_ledger.md#native-map-outline-ownership-repair).

`native/build/live-outline-regression/receipt.json` passes the production DLL
with real JGL hooks at 640×480: **144 of 144** empty native line initializations
preserve the resident map and untouched CPU storage. Four displayed GPU strokes
pass independent color/alpha/width/dash/clip checks; real DC escape still invokes
the CPU fallback. Units, native text, fog, selection/path/grid, exact displayed
composition, scene retirement and independent visual frames pass with all water
effects enabled. The lifetime receipt is
`native/build/gpu-composition/266f8e09de8d4e2eb8bb348aba98db8a/receipt.json`.

The full 1119×1192 run, `native/build/live-outline-full-resolution-verified/`,
passes the same pixel/ownership checks plus native camera cancellation, config-off,
pending/ready reset and mid-line configuration-off. Its 100 independent resident
visual frames, on the coast fixture with one selected unit and all water effects
on, average **24.736 ms** per request (p95 **33.138 ms**, max **96.774 ms**).
Request through desktop completion averages **35.029 ms** (p95 **50.047 ms**,
max **99.629 ms**). These include the first frame; the measured workload is
recorded in `visual_summary.json`. They are neither live gameplay FPS nor a
before/after speedup measurement. A preliminary full-resolution run failed a
fixture assumption that a screen stayed owned after reset; the corrected test
uses the freshly committed map for its config-off scope.

The final bridge also resets width/stipple for a fresh OpenGL context while
retaining GDI+ pen state. Its focused native/default/failure-barrier receipt is
`native/build/gpu-composition/b178599e620645e1b50b9824db8a3134/receipt.json`.
The approved final injected smoke test passes in
`native/build/live-outline-injected-compile.json`. Staging is recorded in
`native/build/live-outline-stage.json`; DLL SHA-256 is
`71b3b064413a5d35035e7bc6e40a81ef8c2dd644a2656cd10d9145d7e6baa827`.
Re-run `INSTALL.bat` before the next capture. No installer or game was launched.

## Installed bridge result and final-screen repair

The third capture, `native/build/live-captures/20260920-222421-1a5c0e/`, uses
the installed outline repair (`71b3b064…a827`). Both collectors finished and the
game exited normally. All **114/114** map composites use `gpu_map=1`, there are
zero ownership-revocation records, and all 63 shared-scene records have zero
readbacks. The first repair is active. However, there are still **zero resident
final-screen presentations**. The gameplay swap chain's 577 displayed intervals
average **80.50 ms** (median **33.62**, p95 **383.34**, max **1,216.64**), or
**12.42 updates/sec**, with 17 gaps above 500 ms. This shorter, different usage
sequence is not a matched before/after speed comparison. Derived `analysis.json`
records the source hashes and separates startup/UI presentation.

The last cumulative CPU-screen sample reports **600 transfers / 1.37 GB uploaded**.
At startup, surface 1 receives a public bits escape on its very first CPU menu
presentation, before the map exists. Source and native replay identify the caller:
our own `ScreenSnapshot::capture` invoked the public getter to make a private
copy. This permanently excluded the eventual screen from GPU destination
admission. Resident map-to-screen copies therefore fell back to CPU ownership.

The DLL snapshot now copies JGL's audited logical pixel member directly on the
caller thread, without public escape evidence or a retained pointer. The owner
explicitly returns any GPU-owned source to CPU ownership before this fallback;
real game access and failed barriers retain their previous behavior. No injected
code or patch-table change is required. The negative-control replay in
`native/build/live-screen-negative-control/` fails the new startup-admission
assertion with the previously staged DLL, reproducing the live sequence.

The corrected 1119×1192 replay in `native/build/live-screen-full-resolution/`
passes all 24 startup copies without a public pixel escape, followed by resident
map/screen composition with unchanged CPU map storage and exact displayed pixels.
Units/text, outlines, fog, selection/path/grid, camera cancellation/reset and
config-off recovery pass. Its 100 selected-unit coast frames with all water
effects on average **26.45 ms** per visual request (p95 **34.47 ms**) and
**36.73 ms** through desktop completion (p95 **50.13 ms**), including the first
sample. These controlled fixture timings are not a live speedup measurement.

The user then requested fullscreen validation. The VM reports **2240×1260** with
no game process running; the prior capture was 1119×1192. GPU admission had a
1192-pixel height ceiling. The DLL and replay bounds now allow 2240×1260 through
scene, image, overlay and presentation paths, including matching prepared-view
padding limits. The first fullscreen replay rejected the taller working area in
the HDR and damage-region bounds; those now admit the matching 2248×1268 area
including its border. Existing memory budgets remain unchanged. The five
scene-bound and prepared-view CPU tests pass, including complete coverage at the
new lower edge and rejection beyond the bound. Live fullscreen mode still needs its own capture;
desktop dimensions alone do not prove a game's selected mode.

The final fullscreen replay, `native/build/live-screen-fullscreen-final/receipt.json`,
passes at **2240×1260**, including native final display, startup admission, units,
outlines, fog/tactical overlays and camera/config-off recovery. Its 100 independent
coast frames with one selected unit and all water effects on measure **31.29 ms
mean / 42.34 ms p95 / 128.09 ms max** per request and **42.10 / 53.38 / 146.19 ms**
through desktop completion, including the first sample. These are fixture timings,
not gameplay FPS or input-to-display acceptance. The sampled minimum contiguous
free virtual-address region was **858 MiB**; this is not a whole-game memory proof.

The preliminary fullscreen replay in `live-screen-fullscreen-verified/` failed
because two independent outline checks retained their full-screen temporary
targets simultaneously. The fixture now restores/releases its completed saved
image before creating the CPU-escape test target, preserving every pixel and
fallback assertion without raising the production budget. The separate
`live-screen-fullscreen-image-operations.json` proves full-size unit composition
with exact pixels and 24,576 scratch bytes under its existing 64 MiB limit.

Staging: `native/build/live-screen-stage.json`; DLL SHA-256
`f17edf16895f7e839ef48a38eadffd0da4222cd746b0d98380dc1444dfe9d375`.
The previous DLL is preserved. The installed outline bridge is already active in
the latest capture, so **restart through `CAPTURE_GAME.bat`; no new `INSTALL.bat`
run is needed**. No game or installer was launched by the agent.

Remaining work includes speculative preparation: **766 of 768** attempts were
cancelled, consuming **8.83 seconds** of worker activity (not additive UI stall
time). Verify final-screen residency and independent visual presentation in the
next installed-game capture before attributing their remaining cost. A resident
replay is not proof that every live UI/native sequence stays resident or reaches
the Standard-map latency target.

The stream also reports Windows' fault-tolerant heap shim at startup. Its cost
and capture overhead are unmeasured; neither explains away the observed route
loss. No system compatibility settings were changed. No further manual capture
is requested until the current evidence and automated reproduction are exhausted.
M3.8 and the <33 ms p95 target remain unpassed. No live FPS improvement is claimed
for the final-screen repair before another installed-game measurement.


## Fullscreen follow-up: native startup transfer

The next capture, `native/build/live-captures/20260920-225200-11158a/`, confirms
**2240×1260** in the running game and the staged `f17edf16…` DLL. All **98/98**
map composites use GPU composition with zero shared-scene readbacks. However,
there are still **zero resident final presentations and zero independent visual
frames**. The sampled CPU-screen counter reaches 480 transfers / **2.18 GB**
uploaded. The gameplay presentation chain has 438 positive non-dropped display
intervals: **95.85 ms mean / 233.34 ms p95 / 1433.35 ms max**, roughly **10.43
updates/sec**. The capture's `analysis.json` preserves counts and definitions.
There are no prepared-area events at this resolution; the earlier 766/768
cancellation observation must not be attributed to this session.

The private pixel-getter defect is gone, but eligibility is already lost before
final map-to-screen adoption. The existing stack logger records only losses after
map demand, so no revocation event does not establish a clean startup lifetime.
The native regression now exercises **12 original, configuration-off Graphsy
transfers followed by 12 configured CPU-screen transfers on the same image**,
then resident map/screen composition. The previously staged implementation fails
its first native startup admission assertion, reproducing an additional defect:
original Graphsy's temporary DC borrow was incorrectly classified as a permanent
game-code escape. This is a reproduced cause of lost eligibility, not yet proof
that no other live startup path can lose it.

The tiny existing Graphsy wrapper scopes this audited native operation, while the
DLL permits only that private DC lease. CPU materialization barriers and public
pixel/DC escapes remain intact. No quality setting, water effect or memory budget
changes. This repair includes injected code and therefore requires `INSTALL.bat`
once; replacing the DLL alone cannot fix the startup wrapper.

Validation: `native/build/live-native-startup-fullscreen/receipt.json` passes the
full **2240×1260** production replay, including all 24 startup transfers, unchanged
CPU map/screen contents during resident composition, exact displayed RGB, units,
text/outlines, fog/tactical controls, camera/reset and config-off recovery.
`visual-analysis.json` covers all 100 selected-unit coast frames with water effects
on: request **35.31 / 43.22 / 325.15 ms** mean/p95/max, desktop completion **45.05 /
52.83 / 333.37 ms**, including startup. All intervals prove zero static-world
builds/uploads, static-scene draws, reflection builds and water/wave geometry
uploads. These controlled timings do not establish a live speedup. The sampled
minimum contiguous free virtual-address region is about **724.5 MiB**, not a
whole-game memory guarantee.

The native lifetime regression passes in
`native/build/gpu-composition/1609c233f4a94062806653d201b82fd7/receipt.json`,
including explicit private-DC versus pixel/bits/unscoped-DC rejection.
`native/build/live-native-startup-injected-compile.json` records the passing
approved injection smoke test. The staged DLL SHA-256 is
`6c73829c77e602e847d0a18dc1cd7d59262601d4d3b7f23240bc47c5f45471e4`;
`native/build/live-native-startup-stage.json` preserves its validation and rollback.
The user must run `INSTALL.bat` once, then `Renderer/CAPTURE_GAME.bat`; no game or
installer was launched by the agent. Confirm resident final display in that live
run before claiming the startup paths are fully covered or M3.8 acceptance met.


## Fullscreen follow-up: retained memory amplification

Capture `native/build/live-captures/20260920-230709-cb9049/` confirms the previous
startup fix: the sampled resident-present counter reaches **640**. However,
the first two independent visual requests fail with **retained composition
texture budget**, retaining roughly 118–119 MiB across 1,812–1,813 nodes. The next
GPU map publication rejects with result 2; native fallback then requests a DC and
revokes map eligibility. Only **2/293 map composites** remain on the resident
map path. A resident final transfer alone therefore did not prove resident map
rendering or independent animation.

The gameplay chain records 1,002 positive non-dropped display intervals:
**82.53 ms mean / 216.67 ms p95 / 4316.67 ms max**, about **12.12 updates/sec**.
These mixed-play intervals are not an A/B speedup or input-latency measurement.
The local `analysis.json` preserves counts and the distinction between observed
failure and inferred capacity cause.

Two automated fullscreen controls reproduce concrete capacity defects:

- Retaining 17 unchanged full-screen copies reaches 124,185,600 bytes and fails
  the 128 MiB recipe limit. Same-coordinate/same-format copies now share immutable
  patches, preserving source versions and later partial writes. The exact-pixel
  control passes with **11,289,600 bytes / one node**, including subsequent source
  replacement. Shifted, converted and procedural operations keep their original
  replay path. No retained-history or scratch budget is raised.
- The old 96 MiB live-image ceiling rejects a new immutable map while the old map
  and seven fullscreen native canvases remain alive. The live-image cap is now
  **128 MiB (+32 MiB maximum)**, covering publication overlap. Its regression also
  verifies hard-limit rejection, unchanged ticket on failure, release/retry and
  retirement back to one map. The live trace does not record enough allocation
  detail to prove this was its particular publication rejection; the new
  `map-publication-rejected` record reports dimensions, resident bytes and cap.

A visual replay exception now discards failed history immediately while retaining
the last completed display. Fresh authoritative map publication can rebuild the
recipe; it no longer keeps failed output allocations until later native writes.
This continuation changes only the DLL, not the injected bridge or patch table.
Full visual quality, waves and reflections remain enabled.

Validation `native/build/retained-memory-regression.json` records the passing
capacity/oracle controls and failing old-code logs. The full native replay in
`native/build/live-retained-memory-fullscreen/receipt.json` passes at 2240×1260,
including 24 extra map/screen save/restore copies before independent animation,
exact GPU/native display pixels, fog/tactical overlays, camera/reset and config-off.
Its 100 selected-unit coast frames, with all water effects on, retain **34,689,824
bytes / 15 nodes**, versus the previous fixture's **68,558,624 / 26**. This is a
measured representation saving, not a live-game FPS claim. Requests measure
**33.21 / 41.00 / 104.56 ms** mean/p95/max; desktop completion **43.10 / 52.36 /
112.17 ms**, including the first sample. All 100 intervals still prove zero
static-world builds/uploads, static-scene redraws, reflection builds and water/wave
uploads. The sampled largest free address region never falls below **830.81 MiB**;
that is a fixture observation, not whole-game memory acceptance.

The fullscreen eight-unit mixed capacity check in
`native/build/live-retained-memory-mixed/receipt.json` also passes: one selected
idle, three work loops, two frozen idles, two native actions, 30 independent frames,
**39,463,080 retained bytes / 24 nodes**. Request mean/p95 **36.97 / 49.24 ms**;
desktop mean/p95 **43.33 / 67.71 ms**. It is a shorter stress check, not 100-sample
performance acceptance. All sampled static-work elimination checks pass.

Staged DLL SHA-256:
`d5a0fe93fddaa99c134e2387b9632aa4b94b9f9bb2196bb88bb24705c1b08bfb`.
`native/build/retained-memory-stage.json` records current-source validation and
rollback. The injected source hash is unchanged from the last installed bridge;
**restart through `Renderer/CAPTURE_GAME.bat`; no reinstall is needed**. No game
or installer was launched. Next live acceptance must show successful independent
visual frames and continued resident **map** composition after publication, not
just the resident final-screen counter. The Standard <33 ms p95 goal remains open.


## Fullscreen freeze: allocation pressure and removed device

Capture `20260920-232812-ce8053` used the preceding `d5a0fe93…` evaluation
DLL. Resident map delivery survived through frame 11 and independent visuals
advanced 13 times. At 37.12 seconds the process reports repeated **bad allocation**;
at 37.19 seconds Direct3D removes the device. GPU upload and native display
handoffs subsequently fail against the terminal session until exit. This is
allocation failure followed by loss of authoritative GPU pixels, not evidence
of a mouse handler deadlock. The installed executable is already x86
large-address-aware. There is no crash dump or whole-process memory sample in
this capture, so its exact exhausted resource is not established.

The scene alone allocated **1,467,084,288 bytes** at 2240×1260. Removing the
superseded off-screen guard and its padded color/depth storage reduces that to
**1,231,400,448 bytes**, saving **235,683,840 bytes / 224.77 MiB**. The existing
visible circular scene and sparse static backup now serve every view. Exposed
scroll damage, content invalidation, the four-pixel finishing margin, full
sampling quality and all water effects remain. The retired guard utility and
its implementation-only test are removed; scrolling/filter/invalidation tests
remain. Replay also releases temporary image handles on sampling/display
exceptions, keeps node accounting correct when allocation throws, and checks
replacement output capacity before allocating its texture.

The standalone native harness can now reserve address space in 64 MiB chunks
with `--reserve-address-mib`; this reserves **no physical RAM** and deliberately
models a co-resident process footprint without claiming to reproduce Civ III's
actual allocation layout. With a 1 GiB reservation, the old DLL fails to allocate
the full-screen native display oracle. The initial corrected DLL passes the
complete same-resolution mixed-unit/native UI workload with 100 independent
frames (`native/build/freeze-fixed-pressure/receipt.json`). Its control image
changes four of 2,822,400 pixels by at most one channel level. Scene allocation
savings are measured, not an FPS claim. Minimum sampled available virtual space
is only **93,876,224 bytes** under this deliberately severe fixture; this is a
successful stress check, not generous whole-game headroom.

That 100-frame check measures request **34.72 / 49.68 / 86.22 ms** mean/p95/max,
with desktop completion **42.76 / 62.27 / 95.13 ms**. All measured intervals have
zero static-world builds/uploads, static-scene redraws, reflection builds and
water/wave uploads. The final candidate adds pre-allocation rejection and failure
telemetry; its first 100-frame run completes the animation samples but fails the
fixture's foreground-window prerequisite for the subsequent timer check. Keep
that failed receipt; it is not end-to-end acceptance or a device-loss recurrence.

Lightweight once-per-second `process-memory` samples now report whole-process
available virtual, physical and pagefile memory without the profiling-only
VirtualQuery/buffer walk. Worker and visual failure records include the removed
device HRESULT before cleanup. Native composition still refuses stale CPU pixels
when the GPU authority is lost. **General device-removal recovery is not added by
this repair.** Its scope is reducing avoidable memory pressure and preserving
bounded cleanup on recoverable recipe errors. Live stability, whole-game memory
headroom and the Standard <33 ms p95 navigation goal remain open.


Final candidate `34ae28b5…` passes the complete 30-frame 1 GiB pressure retry in
`native/build/freeze-final-pressure-retry/receipt.json`, including real timer
transport, native display pixels, fog/tactical controls, camera cancellation,
reset and config-off. All 30 static-work elimination proofs pass. Request
mean/p95 is 34.93/45.30 ms; desktop completion 42.20/51.68 ms. The earlier final
run's foreground prerequisite failure remains preserved separately. This shorter
retry is a correctness/capacity check, not a replacement 100-sample timing campaign.
The focused contract suite passes 11 tests; the local executable byte audit is
skipped because that Mac-side executable is absent. Injected code is unchanged.

The exact candidate also passes all 40 fullscreen fine-pan, two-axis, zoom and
cancellation coverage cases without missing strips
(`native/build/freeze-scroll-fullscreen/receipt.json`). It is staged as
`34ae28b52a1c03c9914d0ccea6e5601cd1d6eefc3ec6e3c7aa867e9a4a878c37`;
`native/build/freeze-memory-stage.json` preserves rollback and runtime identity.
Restart through `Renderer/CAPTURE_GAME.bat`; no reinstall is needed. Live-game
stability remains pending. No installer or game was launched.


## Smoother capture with frozen effects and another failure

Capture `20260920-234930-a86f69` used `34ae28b5…`. Twenty native map
composites retained `gpu_map=1`; 400 independent visual frames succeeded. Yet
only twenty scene-water/shared-scene draws occurred, corresponding to native
map requests. All twenty map sample factories admitted copied dynamic inputs.
The captured front therefore needs further tracing: this evidence does not
establish whether a native compatibility operation severed its map dependency
or the callback could not advance. No public ownership revocation was logged.
Do not describe unit-only frames as successful whole-scene animation.

Available virtual space fell from 344,834,048 bytes at 39.8 seconds to
118,583,296 near 82.8 seconds, while available physical/pagefile memory remained
several GiB. Allocation failures preceded a retained texture-budget failure at
82.61 seconds (`device_reason=0`), followed by device removal and worker failure
(`0x887a0020`). This supports aggregate x86 address-space pressure; individual
cache ceilings do not bound the process. General device-loss recovery is still
unfinished.

The worker now checks available VA at most every 250 ms and enters pressure
mode below 768 MiB, leaving only above 1 GiB. Reusable reflection pages fall
from a 256 to 64 MiB ceiling; old unit poses from 192 to 48 MiB. Optional tile,
world and pose preparation pauses. Foreground draws, authoritative GPU pixels,
resolution, geometry detail and all water effects remain unchanged. These are
reproducible caches, not visibility or gameplay authority.

Native Animator can draw fog after movement without rebuilding the map. The
previous custom hook suppressed that draw without copying the changed visibility.
It now compares native visibility with rendered tile records and latches the
existing redraw request. The original timer receives its audited dirty bit on
the next callback, after Animator has finished clearing the movement flag.
The extracted regression verifies reveal, conceal, unchanged visibility and
config-off; the timer regression verifies forwarding even with a retained front.
Actual in-game reveal acceptance remains pending.

Candidate `41c438da…` passes the complete 2240×1260 native workload with 1 GiB
reserved VA, 100 mixed-unit frames and all water effects on:
`native/build/live-followup-pressure/receipt.json`. Minimum sampled free VA is
206.81 MiB. Focused tests: sixteen pass, one local executable audit skipped;
approved injected compilation also passes. Timing is not a controlled speedup
comparison; another small GPU contract ran during part of this validation.
Bounded CPU-barrier stacks and periodic reachable-map-source/sample counts are
included to diagnose the remaining live-only animation failure.

The exact candidate is staged with rollback/provenance in
`native/build/live-followup-stage.json`. **INSTALL.bat is required** for the fog
and timer bridge changes. No installer or game was launched. Live animation,
movement visibility and stability are not yet accepted.

## Follow-up: normal exit, animated map dependency missing

Capture `20260921-002051-c7effa` uses `41c438da…`, exits with code zero after
approximately 103 seconds, and retains GPU map delivery for all 36 composites.
The user reports smoother play but barely moving water/resources. All six
periodic/error visual records show `map_samples=0 map_sources=0`, including
successful visual frame 128. There are 36 scene-water draws, corresponding to
native map work, and 31 successful dynamic-map capture admissions. No
`visual-map-unavailable` event occurs. The displayed composition has lost its
sampled map dependency; reducing an animation interval will not repair it.

The first map-to-CPU barrier follows a failed native lifetime admission, between
map insertion and native map completion. Sixteen bounded barrier stacks all stop
at the same DLL frame; they do not identify the responsible native operation.
The DLL diagnostics now record adapter operation, surface slot and copy-admission
reason/rectangles directly. This replaces ineffective guest stack walking without
changing admission, CPU barriers or drawing. No injected change is needed.
Reproduce the identified operation through the existing real-JGL harness before
changing ownership rules; native pointer escape protection remains mandatory.

Minimum sampled available VA is **263.82 MiB**, with no logged worker/device
failure. Two retained replay failures remain: texture budget at 17.15 seconds
and retired unit selection at 31.19 seconds. Both report a healthy D3D device;
neither is a live stability pass. The gameplay swap chain has 1,183 non-dropped
positive display intervals, mean **68.50 ms**, p95 **99.98 ms**, maximum
**1,266.65 ms** (14.60 displayed updates/sec). These mix gameplay/UI and frozen
map frames; they are not unique animated map FPS or input latency. Preserve
`native/build/live-captures/20260921-002051-c7effa/followup-analysis.json`.

Direct game automation is unnecessary for this investigation. The user runs
the existing capture launcher; bounded operation logs identify the seam for
standalone reproduction. No installer, debugger attachment or game launch was
performed by the agent.


Diagnostic candidate `a9702ec4…` passes the full native fixture at 2240×1260,
30 mixed-unit visual frames, all water effects on and 1 GiB extra VA reservation
(`native/build/live-animation-diagnostic-check/receipt.json`). It is staged;
`native/build/live-animation-diagnostic-stage.json` retains provenance/rollback.
No injected source changed. Restart with the existing capture launcher; no new
installation is required. This is diagnostic delivery, not an animation fix.

## Allocation failure and two reproduced ownership defects

Capture `20260921-003619-8a3069` uses `a9702ec4…` and exits with code 1.
Allocation failures start at 103.27 seconds. At 103.31 seconds the retained
texture-budget failure reports 116,826,112 bytes (111.41 MiB) available VA and
`device_reason=0`; several GiB of physical/pagefile memory remain available.
Worker failure follows. Process-headroom trimming was already active at
19.68 seconds, so lowering those cache ceilings alone did not solve this case.
`followup-analysis.json` beside the capture preserves the bounded evidence.

The first full-screen map COPY fails destination lifetime admission before
native map completion. Successful periodic visual records still have zero map
samples/sources. Source inspection identifies a missed startup path:
`FUN_004e2b00` initializes the line renderer on the main-screen canvas before
any GPU map exists. Its public DC acquisition invalidates that canvas's lifetime
evidence. Deferring initialization only for an already-owned map was too late.
Custom-on initialization now records the target; only an actual CPU stroke
initializes its native backend. Config-off remains immediate, and actual CPU
access still revokes GPU ownership. The lifetime fixture fails on the previous
wrapper and passes with this change (`startup-line-negative.log` and
`startup-line-positive.log`). Approved injected compilation passes.

Later 32×32 cursor-copy admission also fails. Disposable CPU source mirrors can
fill the adapter's fixed slots and force a full-map readback. The adapter now
trims least-recently-used unowned, clean source mirrors between operations,
before borrowing internal pointers. GPU-owned images are never evicted this
way; immutable retained versions preserve recorded copies. The 32-slot/64 MiB
ceiling is unchanged. A 48-source churn regression fails on the previous adapter
and passes with zero readbacks after this correction; all existing native pixel
parity/lifecycle checks pass (`native-source-churn-positive.log`).

The first sustained 2240×1260 fixture completes all 1,200 direct visual samples
with mixed units, visibility, all water effects and an additional 1 GiB reserved
VA. Periodic frame 1,152 retains one map source and 1,152 map samples, 24 nodes
and 37.64 MiB retained history. Minimum sampled available VA is 221.82 MiB;
there are no allocation/device failures. Request/desktop means are 38.98/48.06 ms,
not a controlled speedup comparison. The whole receipt is nevertheless **failed**:
the subsequent timer test could not acquire foreground focus after the long loop.
The harness now services non-timer window messages during sustained sampling;
timer delivery remains separately asserted. Preserve
`native/build/startup-ownership-sustained/sustained-analysis.json` and the original
failed receipt. Live allocation stability and animation acceptance remain open.

The final candidate repeats all 1,200 direct samples with minimum sampled VA
209.76 MiB, request/desktop means 39.17/48.20 ms and the same bounded history.
That receipt also fails the subsequent foreground precondition; message pumping
alone did not resolve it. The fixture now waits up to 500 ms for asynchronous
activation, preserving the foreground assertion and logging a failure's window
identities. `native/build/startup-ownership-complete/receipt.json` then **passes**
the complete 30-frame workload, including timer-delivered frames, native line
OpenGL/GDI+ fallbacks, UI, visibility, tactical overlays, camera/reset and
config-off. Inputs are unchanged. This separates sustained animation evidence
from the complete shorter regression; neither substitutes for live acceptance.

Candidate `4d413b5a…` is staged with matching injected-source identity and rollback
in `native/build/startup-ownership-stage.json`. **Re-run INSTALL.bat** before the
next capture. The source-mirror cache stays in the DLL; only the existing native
line initialization/stroke bridge changes. No patch-table entry is needed.
The installer and Civ III were not launched by the agent.

## Base defaults precede renderer configuration

Capture `20260921-012844-ec2871` uses `4d413b5a…`, exits normally after about
64 seconds, and contains 18 native map composites and exactly 18 water draws.
Periodic visual records still report zero map samples/sources. Four retained
texture-budget failures occur with a healthy device; minimum sampled available
VA is 304.41 MiB. After the initial loading interval, 707 non-dropped positive
display intervals have mean/p95/max 57.80/83.36/1,033.34 ms. This is a mixture
of UI and frozen-map updates, not animated map FPS. Preserve the capture's
`followup-analysis.json`.

Read-only inspection of the installed executable verifies the previous hook is
present. GOG startup calls `FUN_004e2b00` at `0x004C8E09`, then the registered
line initializer at `0x004E2B7F`. This runs before `patch_load_scenario` loads
configuration files; program initialization installed only base defaults with
`enable_custom_rendering=false`. The previous fixture incorrectly enabled
rendering before this call, so its passing result missed the real startup state.

The hook now uses the existing loaded-config list to recognize base defaults
alone. It defers unused initialization before configuration as well as when
custom-on. A real CPU stroke still initializes/draws through the original backend
and ownership barrier; configured-off initialization remains immediate. No new
state, symbol or CSV change. The corrected pre-configuration regression fails
on the previous code (`native/build/preconfiguration-line-negative.log`) and
passes after repair, including actual CPU strokes before configuration
(`preconfiguration-line-final.log`). Approved injected compilation passes.

`native/build/preconfiguration-complete/receipt.json` passes the complete native
fixture at 2240×1260 with all water effects, mixed units, visibility, tactical
overlays, native UI, timer, camera/reset and config-off. Its 100 direct frames
produce 100 map samples, 800 unit samples and 400 pose changes without native
redraws. History is 37.64 MiB / 24 nodes; minimum sampled VA is 216.25 MiB with
1 GiB additionally reserved. Request mean/p95 is 41.57/57.16 ms; desktop is
51.14/68.90 ms. No speedup comparison or live acceptance is claimed.

The capture also shows ordinary sprite fallbacks during mouse holds. The DLL
now logs at most eight source-layout/scale records for these full-screen
ownership barriers, without source pixels or altered fallback behavior. That
remaining path is not fixed yet; do not hide it by claiming timer-only success.
Candidate `0649cfe6…` is staged with matching source/rollback in
`native/build/preconfiguration-stage.json`. **INSTALL.bat is required again**
because the startup decision changed. No installer or game was launched.

## Broader native UI audit after unchanged live behavior

Capture `20260921-020053-d1fc60` uses `0649cfe6…`, exits normally, and still
has exactly 37 water draws for 37 native map composites. Periodic visual records
have no map samples/sources; 19 retained texture-budget failures occur. Startup
copy admission improved, but a full-screen sprite-blend fallback at 19.805 s is
followed by an unscoped DC revocation at 19.957 s and the first retained-budget
failure at 20.273 s. The stack still does not identify the native caller. The
capture's `followup-analysis.json` preserves log identity and this sequence.

Eight mouse-hold diagnostics identify ordinary sprites with the verified native
vtable, null pixels, zero dimensions and zero bit depth. JGL slot 17 (RVA 0x8180)
asks its pure bits getter (0x98a0) before touching the destination, and returns 7
for this case. The adapter now passes these calls through without a readback or
ownership change, preserving the actual native error. The audit also covers
empty keyed/mask/shadow/lookup/opacity calls and the format rejection in all
three alpha programs. The mask returns 16, unlike the ordinary sprite's 7;
regressions compare the real native results rather than treating every no-op as
success. Unknown or genuinely pixel-touching programs keep their CPU barriers.

The proactive full-resolution canvas test reproduces another admission failure:
six 2240×1260 CPU mirrors require 64.6 MiB, above the 64 MiB source-cache budget.
These mirrors are stale once the GPU owns the canvases. They are now released
on takeover; an actual CPU fallback allocates one bounded temporary readback.
The source-cache budget is unchanged. `native-hud-negative/receipt.json` fails
the six-canvas admission assertion before the correction. This is a reproduced
capacity defect, **not proof of the first live blend's rejection reason**; that
capture lacks blend-input metadata. Bounded blend-layout diagnostics now cover
that missing evidence.

The renderer's own CPU unit fallback also used an unscoped DC lease. Its calls
are synchronous, and neither DC escapes to native callers. The existing sprite
scope now covers acquisition, drawing and release, including denied destination
or background DCs. This preserves future admission while retaining the real
CPU barrier. Actual extracted bridge tests check scope restoration on every
exit; genuine external/foreign access must still revoke eligibility. No new
hook, patch-table entry or injected state is needed.

Native replay now hashes and consumes six panel/button pairs from the locally
installed Civ III PCX files, through JGL's actual sprite allocator and palettes.
The binary fixture remains ignored; no game artwork enters source control.
The small native oracle checks exact pixels and returns at all clipped edges.
The complete fullscreen replay composes these panels over the animated map,
exercises empty hover draws and repeated save/restore, then runs independent
ambient and mixed-unit frames, timer delivery, selection/path/grid, native
camera/reset and config-off recovery. A preliminary expanded fixture leaked its
own six probe canvases by installing its cleanup callback too late; the callback
now precedes the probe. That was a test setup error, not a production finding.


Expanded verification: `native/build/native-hud-display-check/receipt.json`
passes at 2240×1260 with 100 direct mixed frames and the full recovery/config-off
sequence. Before mixed units are introduced, eight ambient-only frames change
31,392 interior desktop pixels; this witnesses displayed water/resource motion
rather than relying only on successful counters. The 1,200-frame
`native-hud-sustained` run passes independent animation and timer transport:
1,200 map samples, 9,600 unit samples, 4,800 pose changes, 40,904,712 retained
bytes and 36 nodes. Mean request/desktop durations are 39.269/48.859 ms with
1 GiB extra VA reserved. Its **later fresh-capture recovery assertion fails**;
the overall receipt remains failed. Do not claim full soak acceptance. An earlier
long run failed timer delivery; restricting the fixture to withhold only the
renderer callback avoids suppressing unrelated timers, but this alone does not
establish the earlier failure cause (this run saw zero auxiliary timers).

Actual lifetime/private-DC tests pass in
`gpu-composition/140eae6dc3d74ffd9598f5aacb477278/receipt.json`; the approved
injected compile passes in `native-hud-injected.json`. Extracted bridge and
lifecycle/cadence/camera contracts pass (45 tests, one skipped).

`native/build/native-hud-stage.json` records candidate `97593ec9…`, source and
rollback identities. Under the explicit game-launch authorization, INSTALL
showed success and the game reached its main menu. The installer later printed
an access-denied line while cleaning up its temporary executable; the success
dialog had already been observed. The standard capture launcher failed to start
its elevated FPS helper (`live-captures/20260921-024313-e53810`); a bounded
300-second debug-only launch followed (`agent-game-observation/session.json`).
Automated ordinary clicks, Return and keypad Enter did not navigate the menu.
The user then took over testing; there is no gameplay/FPS acceptance from this
attempt. Parallels was temporarily changed from full-screen to windowed to expose
its current display; do not compare that launch with the 2240×1260 replay timings.


## Ambient recovery after a frozen native snapshot

The next user report is still no water/resource animation, plus pauses during
interturn. The debug-only agent launch produced an empty `renderer.log`; its
collector was bounded to 300 seconds. No new frames or failure metadata can be
inferred from that file. Read-only installed-executable inspection confirms the
`C3X_Shared_Verify` path, which links to the current shared checkout. The capture
launcher now also sets a bounded 64 MiB renderer-owned file and continues with
logs if optional FPS startup fails. Windows PowerShell parsing passes; no game
was launched during this investigation.

A separate production D3D regression reproduces a recovery defect: after CPU
upload replaces an animated map with a static snapshot, `Session::visual_ready`
previously still returned true. The native timer then withheld compatibility
recovery. An animated unit could keep that misleading state alive indefinitely.
`ambient-recovery-negative.log` fails specifically on the frozen-map readiness
assertion. The DLL now propagates an animated-map dependency separately from
ordinary animation and requires it when the published map expects ambient
motion. Restoring map writes restores readiness; static maps remain eligible.
`ambient-recovery-positive.log` passes this case, the unit-only variant, restored
map copies, static-map readiness and 126 existing exact GPU composition oracles.
This proves a recovery bug, not the initiating failure in the unlogged user run.
The extracted native timer/UI/camera checks also pass (seven passed, one skipped).

Interturn pauses have a separate architectural explanation: the Win32 visual
timer and final DXGI Present still run on the game UI thread. A blocked native
message pump can stop visible animation despite worker-owned rendering. Explicit
interturn continuity is now in the M3.8 responsibility table and visual-frame
contract; it has not been implemented or certified by the current recovery fix.


`native/build/ambient-recovery-fullscreen/receipt.json` passes with unchanged
inputs at 2240×1260, all water effects, mixed units and 1 GiB extra VA reserved.
Ambient-only desktop changes: 31,526 pixels. The 1,200-frame soak records 1,200
map samples, 9,600 unit samples and 4,794 pose changes, retaining 40,904,712 bytes
and 36 nodes. Mean request/desktop costs: 41.726/51.121 ms. Timer transport,
actual HUD, tactical overlays, fog/reveal, config-off and all four native
recovery cases pass. Each recovery result and native lifetime is now logged.
The preceding long-run recovery failure did not recur; this pass does not
establish its cause or certify complete recovery under arbitrary conditions.

Candidate `7a46796b…` is staged with source and rollback in
`native/build/ambient-recovery-stage.json`; no new injected installation is
needed. This is a verified recovery correction, not a claim that the unlogged
live initiating failure or interturn presentation has been solved.

## Independent delivery while the native window thread is blocked

The DLL now owns a bounded cadence thread; WM_TIMER is no longer the visual
transport. Rendering stays on the existing D3D worker under the existing native
transaction gate. A DirectComposition swap chain replaces the HWND-bound chain
on the same game window, so ambient Present does not depend on that window
thread dispatching messages. It uses only copied authorized scene inputs.
Directed movement/combat still follows native cursors. Native capture, camera
adoption, modal policy, fog and image ownership remain unchanged. There are no
new injected hooks or patch-table changes. Windows 8+ is now required for this
presentation path; Windows 11 remains the verified target.

The small presenter probe produced 30 visible changes with its window thread
blocked and restored GDI afterward. The complete 2240×1260 renderer then
produced 41–42 frames and visible map changes during two-second UI stalls with
no native draws. The first complete fixture failed its GDI handoff oracle:
the oracle had moved most of the window off-screen, then expected off-screen
GDI writes to survive later exposure without any WM_PAINT handling. It now
exposes the complete full-screen window before handoff; partial native writes,
config-off, and exact display recovery pass. Production retains ordinary native
repainting for later exposure; no window-procedure shim or repaint loop was added.

The next run exposed test-mode contention: the automatic callback briefly took
the transaction gate even while manual timing was selected. The fixture-only
manual control now returns before taking that gate. Production always enables
automatic delivery. Extracted contracts prove pending Present retry without a
new render revision, no false completed-frame count, native-camera priority,
foreign-caller rejection, and re-enabling native recovery on draw/Present errors.


Final acceptance: `native/build/independent-visual-acceptance/receipt.json` passes
with unchanged inputs, all water effects, 2240×1260 and an extra 1 GiB VA reserve.
The blocked-UI witness records 40 accepted frames, 30 map samples and 26,992
changed desktop pixels with no native draws. Resource-only motion changes
31,157 pixels. The 1,200-frame soak records 1,200 map samples, 9,600 unit samples,
4,800 pose changes, 40,904,712 retained bytes and 36 nodes; mean request/desktop
cost is 39.314/47.063 ms. Actual HUD, fog/reveal, tactical motion/cancellation,
config-off, exact GDI handoff and all four cancellation/reset recovery cases
pass. The capture helper now restores the fully exposed window after every
scan, including before reset handoff; no pixel comparison was removed.

The tested `7f1fe9df…` DLL is staged with rollback and source hashes in
`native/build/independent-stage.json`. The installed bridge is unchanged and
Civ III was confirmed stopped before staging. No game launch or live acceptance
is claimed. The earlier intermittent fresh-capture failure remains unattributed;
the explicit off-screen GDI oracle failure diagnosed here is a different case.


## First-use HUD blend breaks the live animated map dependency

Capture `native/build/live-captures/20260921-033959-708375` confirms the staged
`7f1fe9df…` DLL, hardware GPU, and enabled water effects, but **zero independent
visual frames**. At 15.345 s, operation 108 reads back the owned screen as a HUD
blend background. There is no owned-destination blend rejection: the destination
is initially CPU-owned, and the adapter never attempts to admit it. The CPU
blend loses the animated map dependency. Later public DC escapes revoke native
image lifetimes, causing further map-copy fallbacks. Their precise native caller
is not established by this capture; they must not be exempted from ownership
checks on assumption.

At 22.811 s, publication also fails with 128,719,512 live image bytes against
a 134,217,728-byte cap. This explains a separate fallback and cannot be repaired
by animation timer changes. GPU presents alone do not prove animated content.

The adapter now admits an eligible fresh blend destination from its owned
background using the existing lifetime/lease checks, just as native copies do.
The hard live-image ceiling is 256 MiB for the actual paired fullscreen family
and old/new map overlap; retained history and replay limits remain unchanged.
The present trace now reports animated-map readiness. No injected code, native
function address or timer behavior changes.

Negative controls: `native/build/live-hud-admission-negative` fails the new
first-use HUD regression against the previously staged implementation;
`native/build/live-capacity-negative.log` fails allocation of eight fullscreen
packed/full-color native pairs at the old cap. Both failures occur before the
corresponding fix. Production-DLL tests additionally require a subsequent native
outline to stay GPU-owned without a public DC. Genuine CPU escapes retain their
fallback and cannot be re-admitted.


Positive proof: `native/build/live-capacity-positive.log` and
`native/build/live-hud-acceptance/receipt.json` pass. At 2240×1260 with water
motion/reflections/shore waves, eight mixed units and 1 GiB extra VA reservation,
the new first-use HUD path retains its background with zero readbacks. The
production DLL keeps its subsequent outline GPU-owned with no native DC.
Ambient-only desktop motion changes 31,644 pixels; a blocked UI thread admits
37 frames / 28 map samples / 36,097 changed pixels in at least two seconds.
The sustained run advances all 1,200 frames and map samples, 9,600 unit samples
and 4,796 poses, retaining 63,806,136 bytes / 39 nodes. Mean direct request /
desktop costs are 40.897 / 50.362 ms. This is not a matched speedup or gameplay
FPS claim. Exact native composition, fog/reveal, tactical controls, config-off
and all four native recovery cases pass with unchanged inputs.

Candidate `57a39bc4…` is staged with rollback in `native/build/live-hud-stage.json`.
The game was stopped and its executable matched the earlier audited bridge;
no injected source changes or new installation were needed. The user-owned live
scene was not modified. Live stability, attribution of any remaining public DC
escape, and navigation performance remain open; no game was launched.


## Retained-history capacity and replay cost after HUD admission

Capture `native/build/live-captures/20260921-040059-757bf2` loads `57a39bc4…`.
The user reports improved operation but low FPS. Independent delivery starts,
then three retained-history texture-budget errors occur; the last logged visual
failure has 46 delivered frames / six map samples, and later presents report
`visual_ready=0`. There are no live-image map-publication rejections. One public
BITS lifetime escape follows the first failed replay; it remains fail-closed.

For non-dropped PresentMon display changes with `14 < TimeInSeconds < 89`,
859 samples average 86.03 ms (11.62 display changes/s), median 49.99 ms and
p95 250.59 ms. These include native pauses and interaction; they are not isolated
GPU rendering timings. `native/build/retained-budget-live-diagnosis.json` records
the selection and counts. At the first visual failure, the process still reports
818,991,104 available VA bytes (~781 MiB) and no device removal.

The retained-history cap is now 256 MiB, matching the bounded live-image family;
replay scratch remains 128 MiB. The eight-fullscreen-pair regression fails at
11 source textures / 124,185,600 bytes under the old cap, and passes at
192,243,200 retained bytes (~183 MiB), stable through eight animated replacements.
The hard-limit rejection test remains; increasing capacity does not remove
lifetime, visibility, stale-output or allocation-failure checks.

Replay now reuses up to 32 MiB / 32 private textures within its existing cap.
Idle scratch is evicted before denying a necessary allocation. Published output
and borrowed input textures cannot enter the pool; handles, format, revision and
zero-initialization semantics are tested. The base of a fully covered disjoint
picture is copied once rather than separately for every surviving fragment.
Sparse/zero coverage and aliased input unions retain the original exact path.

The matched 2240×1260 dense replay (eight retained native pairs, 1,000 UI writes,
eight changing frames, explicit correctness readback) averaged 206.123 ms before
reuse, 175.083 ms with pooling, and 94.000 / 84.609 ms with base-copy compaction.
The final run has zero hot texture allocations and 8,008 reuses, exact pixels,
stable retained bytes, and passing reset/budget/aliasing/partial-publication
checks. Logs: `native/build/retained-dense-{allocation-baseline,reuse,compaction,final}.log`.
This focused result is not a live-game FPS claim. Full workload validation and
staging are recorded below. No injected changes or new hooks.


`native/build/retained-budget-reuse-acceptance/receipt.json` passes with unchanged
inputs: 2240×1260, all water effects, eight mixed units, 1 GiB extra VA reserved,
1,200 frames/map samples, 9,600 unit samples and 4,796 pose changes. Retained
history stays at 63,806,136 bytes / 39 nodes. Mean request / desktop times are
41.341 / 52.667 ms versus the preceding 40.897 / 50.362 ms; this ordinary fixture
does not establish a general speedup. Blocked-UI delivery yields 39 frames /
28 map samples / 36,307 changed pixels; ambient-only motion changes 31,451 pixels.
Exact HUD/native composition, fog/reveal, tactical controls and all four native
recovery cases pass.

The separate `native/build/retained-dense-pressure.log` also reserves 1 GiB VA,
keeps the full 192,243,200-byte retained family, and passes 32 fresh native map
publications / 32,000 UI writes with exact pixels and stable memory. Its timed
dense replay averages 94.632 ms. Candidate `7b8b82f9…` is staged with source,
validation and rollback hashes in `native/build/retained-budget-reuse-stage.json`.
The game was stopped; the installed executable matches the audited bridge.
No game launch, injected changes or reinstall. Live sustained readiness and
complete-frame latency remain the next checkpoint.

## Latest live limit and reproducible composition inputs

Capture `20260921-042106-4a9c31` loads `7b8b82f9…` and exits with code 1.
It has no retained-history capacity failures or map-publication capacity
rejections. At 150.647 s D3D11 removes the device; the subsequent worker failure
reports `device_reason=0x887a0020`, available VA 125,906,944 bytes (~120 MiB),
then repeated unusable-session errors. Three native ownership revocations are
also recorded. This establishes severe process address-space pressure and a
driver-internal failure, not that memory pressure caused device removal or the
exact cause of game exit. The prior isolated pressure test did not predict it.

The user authorized a bounded recorder and automated replay before another live
test. `composition_recording.h` records native GPU image inputs and stable resource
identities, native ownership decisions and independent visual readiness. External
map/pose results are exact snapshots; direct unit writes capture only their
affected rectangles. The strict standalone reader executes production image
commands and the existing native presenter. Scope, bounded-stop and corruption
rules are in [validation](benchmark_workflow.md#recorded-native-composition).
It does not yet re-execute native admission decisions, world compilation or the
retained animation graph. Do not turn its scoped pass into gameplay acceptance.

`native/build/composition-recording-display-check/receipt.json` passes the
2240×1260 production DLL fixture with all water effects, eight mixed units,
fog/tactical controls, independent frames and native recovery. Its ~257 MiB
recording spans 68.34 seconds and replays 3,617 commands, 2,520 exact pixel
checkpoints and 107 display boundaries. Native/visual observations are retained,
not silently interpreted as replayed gameplay. The old 128 MiB live-image budget
fails replay of the admitted fullscreen family, while the normal budget passes.
555/565, aliased copies, readonly source rejection, partial external patches,
corrupt/truncated/oversized records and missing-footer classification are tested.

An earlier full fixture in `native/build/composition-recording-final/` fails
desktop capture 2 after restoring a saved screen; its recorded GPU composition
still replays 2,493 exact pixel checks. The subsequent fixture adds a source-image
oracle before that desktop exposure and passes both source and desktop checks.
The intermittent desktop mismatch is not explained or declared fixed by that
passing repeat. Preserve both receipts. The final timer-only recorder refinement
starts the three-minute limit at first composition rather than DLL startup, so
loading menus do not consume the gameplay recording window.

Final candidate `6e80a66d…` passes the exact production capture/replay again in
`native/build/composition-recording-ready-shutdown/`, including 2,520 pixel
checkpoints and 107 display boundaries. It avoids waiting for a terminated
journal writer during process detach; a missing footer remains an explicit
prefix. Per-invocation VM output/exit receipts distinguish two observed Parallels
transport failures from actual positive/negative test results. Staged with
rollback in `native/build/composition-recording-stage.json`; game stopped,
installed bridge unchanged, no INSTALL or launch. The requested live recording
is the next input, not a gameplay-performance acceptance test.


### Live flashing/freeze and missing-journal diagnosis

Capture `20260921-053809-a0e677` used the staged `6e80a66d…` candidate and exited
with code 1. The user observed late UI/minimap flashing followed by an apparent
freeze. Its preserved `analysis.json` records 290 allocation failures starting
at 132.405 s, device removal at 139.125 s (`0x887a0020`, 142.56 MiB available
process VA), 7,599 unusable-session errors and 264 failed display handoffs through
158.916 s. Scene targets peak at 1,174.35 MiB. This establishes memory pressure
and broken presentation, not the exact failing allocation or the cause of the
driver reset. An earlier `direct unit selection retired` failure at 32.405 s is
a separate unresolved lifecycle witness. Windows reports its fault-tolerant heap
shim at startup; fixtures must not silently assume an identical allocation path.
No global compatibility or heap settings were changed.

There is **no composition recording** in this capture, and no DLL runtime trace
file. The launcher's requested-recording metadata did not establish activation.
A harmless 32-bit native probe loading the same staged DLL produces both files
with ordinary launch; `WINXPSP2` compatibility loses both, while direct process
creation from an unelevated host fails with elevation-required error 740. This
reproduces the environment loss at the game's compatibility/elevation boundary.
A preliminary PowerShell probe failed because script execution was disabled;
that was not evidence of environment loss. The native probe is the confirming
experiment. The installed mod link's DLL hash matches the staged candidate.

`capture_game.ps1` now elevates the capture host before assigning diagnostic
variables, converts mapped-share paths to UNC across that boundary, and uses
explicit `UseShellExecute=false` for the game. The same host covers FPS collection.
Missing replay files now set `result=recording-missing` and return failure while
preserving other logs. `test_capture_launch.py` exercises the real launcher with
a harmless stand-in named for the game, loads the staged production DLL, verifies
its journal and runtime log, and checks that a deliberately missing journal is
rejected. Test receipts explicitly say `game_launched=false`; these are launcher
checks, not gameplay runs. No renderer binary/injected changes or reinstall.

The actual gameplay traffic remains unavailable for replay. One replacement
live recording is required through the corrected launcher; preserve this failed
session as evidence rather than claiming its synthetic counterpart fixed the
flashing, device loss or freeze.


### Recorded live composition and CPU-source validation optimization

The replacement capture `20260921-130328-65f6bd` contains a 535,554,808-byte
journal, stopping cleanly at its configured byte limit after 80.788 seconds.
Strict D3D replay verifies the captured prefix: 471,209 events, 29,341 commands,
116 exact pixel checkpoints, 1,931 external snapshots and 524 display boundaries.
Peak compositor storage is 140,270,064 bytes; CPU submission totals 329.281 ms.
That total is neither frame time nor game FPS. Native admission observations,
retained animation sampling, scene production and the game message pump are not
re-executed. `replay-initial.json` preserves the exact result and invocation.

The live log shows two public native-access revocations; independent map
animation readiness disappears by 24.848 s. Subsequent fullscreen map-to-canvas
copies repeatedly fail lifetime admission and materialize the source for native
use. The latest session exits normally and reports no allocation/device failures;
it does not resolve those failures in the preceding session. The first untracked
DC caller remains unidentified. Do not weaken lifetime evidence to admit it.

A separate avoidable cost is in `Adapter::refresh`: every CPU-source check
allocated and widened the full image before comparing it. It now compares every
visible native 16-bit word against a packed baseline, respecting row stride and
flushing GDI first. Only changed content allocates an expanded GPU upload. Private
leases release on all exits, including allocation failure, before worker dispatch;
cache revisions advance only after successful upload. CPU-source baseline storage
is halved. Public-pointer lifetime exclusions and exact fallback remain intact.

The real JGL adapter fixture compares the old and new complete copy paths,
including final GPU readback completion (recording off, 96 copies per case):

| Source | Old total ms | New total ms | Uploads |
|---|---:|---:|---:|
| 2240×1260 unchanged | 1134.417 | 111.354 | 0 |
| 2240×1260, edit every 12 copies | 1166.304 | 214.813 | 8 |
| 2239×1260 padded rows, unchanged | 1122.082 | 103.597 | 0 |
| 2239×1260 padded rows, periodic edits | 1147.479 | 188.909 | 8 |

All pixels match, including writes through escaped pointers to the final row;
unchanged checks produce zero expanded-upload bytes. Existing sprite, text,
source-churn, palette, stretch, ownership, reinit and config-off tests pass.
Receipts: `native/build/live-cpu-source-{baseline,optimized,comparison}.json`.
This is a measured 10.2× improvement for unchanged fullscreen copying and 5.4×
for the periodic-edit case, not a live-game FPS improvement. The next architectural
responsibility is the independent-animation dependency lost through the native
canvas lifetime barrier, followed by complete workload/memory validation.


The exact optimized DLL `ad03fe4077a52e7320a121bd4af2106b98040afb94fcae1fc28b7c4f13578bba`
passes `native/build/live-cpu-source-production/receipt.json` at 2240×1260 with
all water effects, eight mixed units, visibility/tactical, native UI, independent
frames, recovery and config-off. Staged with matching source/binary proof and
rollback in `native/build/live-cpu-source-stage.json`; no injected change,
INSTALL or game launch. Overall gameplay performance remains unaccepted.


### Replay-backed ambient continuity and address-space regression

The same saved prefix now re-executes 184,198 production lifetime decisions with
zero mismatches, in addition to the 116 exact GPU pixel checks. Its version-2
journal lacks caller thread IDs, so the result explicitly reports the legacy
owner-thread assumption; new version-3 journals include them. Reconstructed
native display images and their contact sheet are in the capture's
`replayed-frames/`. This makes the recorded view inspectable but does not recreate
scene compilation, ambient sampling or Civ III's CPU/memory/scheduling workload.

The strict ambient outcome check fails this capture: readiness is lost at journal
23.394 s and never recovers during the remaining 57.394 seconds. Native image 29
first receives line-target/stroke operations while eligible but not yet owned;
falling back initializes the native line DC, permanently revoking that canvas.
The production stroke path now admits an eligible destination on its first valid
stroke, using the existing adapter. Public CPU/DC escapes still fall back.
`replay-lifetimes.json` and `replay-ambient-negative.json` preserve these results.

A separate real-DLL regression reproduces a new native action retiring the
currently displayed unit selection before the next native screen transfer. That
must freeze the old published pose until replacement, not discard the surrounding
ambient graph. Water/resources continue from their immutable visible records;
native action time and anchor remain authoritative. Popup/Advisor/button setup
no longer pauses the ambient policy, and a visible unfocused window remains
eligible. Injected changes remove policy calls from existing hooks; no new patch
symbols, native state or drawing ownership. The approved injected smoke passes.

The failing-before fixtures are `native/build/action-continuity-before/` and
`native/build/action-continuity-line-before/`. Fullscreen tests now exercise 12
uncommitted action replacements, 24 native movement transitions with no explicit
visual calls, a blocked UI thread, lost focus, and the recorded first-stroke/DC
sequence. Existing visibility/reveal, UI pixel parity, recovery and config-off
checks remain part of the same production DLL/JGL workload.

Reserving 1 GiB of co-resident address space exposed another failure: the
unprofiled whole-view-backup candidate failed fresh-capture recovery after reset
(`ambient-continuity-pressure-unprofiled`). Splitting the same full-view backup
into single-sample planes failed earlier with `bad_alloc`; that experiment is
rejected and preserved only in ignored build evidence. The replacement stores
original-format color/depth samples in small backup tiles under animated damage.
An independent GPU oracle compares every color/depth sample across partial
updates, clears, boundaries, resize and retirement; its negative control differs.
The tiled pressure run passes all recovery cases. Its scene working/backup target
storage is 1,008,317,952–1,025,846,784 bytes versus 1,231,400,448 previously;
the largest individual backup allocation falls from about 348 MiB to 2 MiB.
These are target-storage estimates, not the entire process footprint. A 1 GiB
reservation tests reduced VA capacity, not exact game heap fragmentation.


Final receipts: `native/build/ambient-continuity-reuse/` and
`native/build/ambient-continuity-reuse-pressure/` both pass with unchanged inputs
and complete trace coverage. All 120 timed frames reuse the static backup, with
zero world builds/uploads, static draws, reflection builds or wave uploads.
Normal fullscreen request mean/p95 is 31.26/35.16 ms; desktop mean/p95 is
38.26/52.88 ms. With the 1 GiB reservation, desktop mean/p95 is 47.66/53.18 ms.
The earlier tiled-but-recapturing pressure run was 54.63/68.68 ms. This removes
unnecessary work and repairs observed capacity failure; it does not establish
33 ms performance or a live-game FPS win. Pressure also eliminates the ordinary
frozen-body cache reuse in this fixture, an unresolved residency cost.

The normal run delivers 38 autonomous ambient frames and 30 map samples during
24 native movement steps, with no explicit visual calls; all 12 unpublished
unit-action transitions retain ambient readiness. A visible unfocused window
continues sampling. Both runs pass first-stroke/native DC behavior, visibility,
UI composition, navigation and all four reset/recovery cases.

Staged evaluation DLL:
`80c3bc111fb38433e5012aff8220605017e4681c6d040744a55adb16a11a99aa`.
`native/build/ambient-continuity-stage.json` binds the passing receipts, compiled
inputs, staged hash and preserved rollback. Civ III was stopped; no INSTALL or
game launch. The injected wrapper removals require a new `INSTALL.bat` run.
Full captured scene/ambient producer replay and live performance remain open;
no further manual capture is requested for these targeted fixes.
