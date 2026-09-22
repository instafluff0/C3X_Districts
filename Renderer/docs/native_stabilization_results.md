# Native stabilization evidence

Preserved evidence from the September stabilization work. This is historical
context, not a second task queue: build/staging identities and next-step language
below describe their individual checkpoints. The [roadmap](retained_renderer_plan.md)
holds the current handoff. In particular, the shoreline reset mismatch is now
fixed; the captured removed-device freeze remains open.

### Shared-scene delivery target retirement

Production GPU shared-scene rendering uses `region_glow` and `gpu_map_texture`.
The legacy `render_texture`/depth pair had no draw consumer on that path, and
GPU delivery did not consume `readback_texture` or `pixels`. They are no longer
created there. CPU shared-scene delivery lazily creates staging/bitmap storage;
a later GPU demand releases it without invalidating the retained scene. Legacy
profiles retain their required targets. The executable production-method test
covers repeated GPU demand, CPU transition/reuse, failed CPU staging admission,
resizing and the surviving alternate-profile consumer. At 2240×1260 the retired
allocations total 43.07 MiB (32.30 MiB logical GPU + 10.77 MiB CPU), without a
sampling, format, effect, detail or capacity-cap change.

The preceding allocation-only candidate fails two tight-pressure runs differently:
`retained-map-owner-pressure-1216` at an independent tactical opportunity (its
buffered file did not survive early fixture exit), and
`retained-map-owner-pressure-1216-trace` at fresh native recovery after reset.
The prior DLL passes `wave-pressure-1216-trace`; this variability prevents a
claim that the latest sampler caused the failures or that one pass fixes them.
`pressure-failure-context-1216` then reproduces reset recovery failure and records
`device_reason=0x887a0020`, total free VA 816,828,416 bytes and largest free block
68,136,960 bytes. Main-scene attachments report 684,111,360 bytes. These match the
live device-removal code, not necessarily its cause. No exception loop or stale
CPU fallback was accepted. `pressure-reflection-context-1216` passes; therefore
an exact failing reflection HRESULT has not yet been observed. The added
failure-only diagnostics remain to identify it if it recurs.

Candidate `native/build/shared-scene-delivery/C3XRenderer.dll` SHA-256
`241ca76b1446abc6b68f38656bbc49e031f4968697577a010da319b01711cc9a` is not staged. The strict replay passes 9,406 calls / 336 frames and CPU
restore. Thirteen focused tests pass, including existing worker ownership/camera
and view-retirement contracts. `shared-scene-delivery-pressure-1216` passes the
same connected fullscreen cases with all effects and eight mixed units; all 15
fixed images match the preceding 1 GiB candidate, while wall-clock tactical images
remain excluded from equality. Sampled minimum VA is 155.75 MiB; the 30-frame
request/desktop means are 31.69/41.65 ms. This is not a live FPS measurement or
stable failure-rate claim. The normal 1 GiB run also passes, with 15 fixed images
exact, no GPU-failure records, sampled minimum VA 321.63 MiB (prior candidate:
291.49 MiB), and request/desktop means 30.21/41.73 ms. All generated BMPs are
preserved losslessly compressed. Neither pressure run certifies a live fix.

The unresolved capacity audit includes the large required MSAA/resolve surface,
other writer-only targets, and precise native ownership after device removal.
Do not enlarge caches or change visual quality to relabel this as solved.

### Retained ambient output ownership

The sampler supplies either unchanged output, an immutable packed source, or a
working BGRA region to import immediately. The retained node owns one reusable
packed output and UAV, retaining the existing conversion shader's exact pixels.
The import target leaves the temporary compositor's accounting and enters the
existing retained-output accounting; neither budget increases. Initial native
map versions remain immutable. `snapshot_bgra`, its session wrapper, the initial
closure snapshot, and per-clock texture creation are removed. Direct unit
operations and immutable source sampling retain their contracts.

GPU tests preserve all 126 pixel oracles, 120 clock frames, 32 native map
publications and 32,000 UI writes. Eight native-only ambient frames allocate one
target, unchanged clocks do not reimport, and original CPU ownership reads remain
exact. Sixteen simultaneous 2240×1260 sampled versions update three times with
16 target allocations and release every retained/scratch byte on reset. This
preserves capacity that the first draft would have charged to the temporary budget.
Twelve production worker/ownership/view-retirement contracts pass. No injected
files changed, so no injected compile was required.

Candidate `native/build/retained-map-owner/C3XRenderer.dll` has SHA-256
`8b9885f9d1dde285486db6da323941aecd2b9dfbea34cbabc2cc0fd17d3ab0a7`. It is not staged. Strict replay passes 9,406 calls,
336 displayed fingerprints and CPU restoration against the corrected corpus.
Fullscreen with eight mixed units, all water effects and 1 GiB reserved VA passes
native-only/automatic animation, fog/reveal, tactical overlays, native cameras,
reset/config-off and CPU fallback. All 15 fixed images match; 12 wall-clock
motion images are not equality oracles. The generated BMPs were compressed
losslessly, saving 260.46 MiB. Inputs and failed receipts were retained.

The fullscreen trace records 125 samples / one target allocation. Sampled minimum
free VA is 291.49 MiB versus 273.95 MiB in the preceding candidate run; this is
not a reproduction of the game heap or evidence of removed-device recovery.
At 1120×630, the alternating baseline/candidate/candidate/baseline campaign
(`retained-map-owner-service/4-baseline` through `7-baseline`) completes all calls:

| Metric | Baseline runs | Candidate runs |
|---|---:|---:|
| Unpaced replay envelope | 57.059–57.179 s | 57.122–57.145 s |
| Ambient opportunity p95 (444/run) | 32.920–34.082 ms | 31.797–33.255 ms |
| GPU map mean (14/run) | 54.309–54.976 ms | 51.589–53.116 ms |
| Native presentation p95 (60/run) | 39.030–43.021 ms | 21.044–32.984 ms |

This is unpaced native service, not live FPS, camera-decision-to-display latency,
or a Standard-map workload. Ambient rows include unchanged-clock opportunities.
No overall speed gain is established; presentation maxima still vary and exceed
100 ms in one candidate run. Initial service attempts 2/3 omitted the explicit
candidate-comparison flag and were rejected before replay; their failed receipts
are preserved and excluded. The strict replay's forensic timing is excluded too.

### Shoreline reset mismatch fixed (candidate verified)

A cancelled/reassembled view could change the geometry translation from
`(-52, 28)` to `(0, 0)` while retaining the same complete camera signature.
Shoreline occurrences still carried the old placement. They now invalidate at
`clear_geometry_vertex_buffers` with the other view contributors; immutable
coast cells stay resident. No new cache, native hook, clock or visibility policy.

The deterministic GPU control keeps terrain contributors identical: the prior
DLL differs from an independent cold render at 6,616 pixels; the corrected DLL
matches exactly. All 279 coast cells are reused with zero wave builds/uploads.
Same-time reset, cancelled cameras, fog, unseen coverage and reveal pass. The
production view-retirement test now executes the actual invalidation method.

The earlier recording matches all 337 displayed fingerprints, then rejects the
corrected CPU restore because its recorded image contains the displaced waves.
That failed receipt is preserved. A fresh connected recording passes native UI,
unit/action/ambient continuity, tactical overlays, camera cancellation, reset and
configuration-off. Two strict replays each complete 9,406 calls and match all 336
displayed frames, including CPU restoration. This fixes the intermittent reset
mismatch; it does not establish a live freeze fix or performance gain.
The 2240×1260 connected fixture also passes with eight mixed units, all water
effects and 1 GiB reserved VA. All 15 fixed BMPs match the prior candidate;
sampled free VA reaches 273.95 MiB. Wall-clock tactical-motion images are not
cross-run equality oracles. Generated BMPs are retained losslessly compressed.

Candidate `native/build/wave-occurrence-retirement/` (`ebb37917…`) is not staged.
Evidence: `wave-placement-baseline/`, `wave-placement-fixed/`,
`wave-retirement-native/`, `wave-retirement-native-replay/` and
`wave-retirement-pressure/` under `native/build/`.
The focused GPU witness is `C3X_RENDERER_WAVE_VISIBILITY_TEST=1` in the existing
GPU-frame fixture. Intermediate selection-control comparisons changed contributors
and are not the final parity oracle. Temporary forensic instrumentation is removed.

**Next:** M3.8 remains open for the captured removed-device freeze, required
attachment/native ownership capacity and complete-path navigation attribution.
Continue through M3.12; M3.13 integrated acceptance remains outside this request.

### Mixed-unit scratch allocation churn removed (candidate verified)

The failure capture contains 442 direct-unit reports before device failure and
269 changes between 2,764,800-byte scout scratch and 78,643,200-byte worker
scratch. Their changed-size allocations total 10,990,080,000 bytes; this is
cumulative allocation traffic, not simultaneous residency or measured driver
memory. Immediate-context commands can outlive the released application handles.

Unit raster scratch now retains sufficient capacity across smaller requests,
within the existing 96 MiB bound. Combining wide/tall requests cannot exceed
that bound. Native viewport, clipping, sample scale, format, pose timing and
one-second idle retirement are unchanged. New allocation traces report capacity
separately from raster dimensions. No cache cap or image-quality reduction.

The GPU witness alternates the actual 240/1280-pixel extents for 24 draws: two
allocations, exact HDR RGB/depth samples against independent tightly sized
targets, and existing 98,304 packed/full-color comparisons have zero mismatches.
The isolated candidate replays all 9,433 recorded calls and matches all 337
displayed fingerprints. The 2240×1260 connected fixture with 1 GiB reserved VA,
eight mixed units, all water effects, native camera/recovery and fog passes;
all 15 fixed image outputs match the prior candidate. Twelve wall-clock tactical
animation images are not cross-run equality oracles. It records four scratch
allocations across reset/lifecycle exercise; sampled minimum free VA is 276.91
MiB versus the previous 256.23 MiB. This single comparison establishes no stable
FPS gain, live freeze fix or removed-device recovery.

Evidence: `native/build/unit-scratch-reuse/`, `unit-scratch-reuse-replay/`,
`unit-scratch-reuse-pressure/`. Generated BMPs are retained losslessly compressed.
The game-staged DLL remains `f2598dad…`; this new candidate is not staged or
capture-qualified. M3.8 remains open for the intermittent reset mismatch and
lost-device/native ownership decision; the substantial attachment floor also
remains a measured M3.12 capacity responsibility.

### Live failure captured after the ownership repair

Logs-only session `20260922-054649-e648f1` completed and preserved the exact
`f2598dad…` evaluation DLL, runtime/DebugView logs and 712 PresentMon rows.
The user reports mostly smooth motion with intermittent hitches, then a freeze.
This capture contains no input journal or window images; it establishes failure
timing and device/memory state, not the exact displayed pixels or a replay.

At sequence 142, available process VA falls to 98,209,792 bytes (93.66 MiB).
The next recorded GPU image failure reports device reason `0x887a0020`
(`DXGI_ERROR_DRIVER_INTERNAL_ERROR`, device removed). The last PresentMon
presentation precedes that report by about 46 ms. There are 522 subsequent
worker-failure memory records; the dead device does not recover even when free
VA later rises to about 558 MiB. This confirms a failed-device retry loop and
severe address-space pressure, but does not establish that pressure caused the
driver failure. Hardware rendering is confirmed on the Parallels D3D11 device.

At the first failure, tracked working attachments are 819.70 MiB and composition
plus publication is 166.86 MiB. The ordinary shared-scene target alone reports
652.42 MiB (11,401,856 resolve pixels for a 2,822,400-pixel viewport). These are
logical resource sizes, not a complete accounting of driver mappings or Civ III
memory. The attachment floor remains high after cache trimming; increasing cache
budgets would not address this failure. The main swap chain has 654 presents over
37.78 seconds, median interval 28.88 ms and p95 132.27 ms, including camera/action
work. These are submission intervals, not fresh-frame display acceptance.

Next stabilization responsibility: reduce the measured full-resolution attachment
overlap without changing fidelity, and handle a removed device through explicit
native ownership/reconstruction instead of repeatedly submitting to it. Healthy
device allocation-failure preservation remains necessary and must stay separate
from removed-device recovery. The intermittent reset replay mismatch remains open.
No additional manual recording is needed to establish these immediate targets.

Capture launcher cleanup now recognizes only its own orphaned CLI collector
(known executable, game filter, capture log path, absent parent and matching
process creation time). GUI/foreign/live-host collectors remain untouched.
Windows launcher ownership/parser/observer-stop tests pass without launching Civ III.

### Native failure ownership repair (evaluation staged)

The new live report adds an old “Hail Lincoln!” panel over a black map after unit
movement, with about 2.9 GB observed memory. This is consistent with stale GDI
content resurfacing, but neither an actual reopened modal nor memory exhaustion
is established from the screenshot. It is not proved to be the reset replay bug.

Source inspection found two concrete failure hazards: native GPU presentation
reset/detached the display before successful preservation, and a worker exception
could destroy the composition session while native canvases (and its caller)
still referenced it. A third boundary rejected a completed native transfer when
its optional new visual sample failed allocation. The candidate keeps ownership
until explicit drain, preserves the attached display after rejected presentation,
and uses the current completed native GPU canvas when visual sampling fails on
a healthy device. A fresh map publication restores animation history. Removed
device failures remain explicit failures, not claimed successful recovery.

The new deterministic allocation-failure GPU test fails on the prior path and
passes after repair: exact current display, exact CPU restoration and resumed
animation after fresh publication. All 126 existing GPU image oracles pass.
Portable tests execute the production presentation/worker failure branches;
12 selected ownership/camera/cadence tests pass, one native-byte audit skips.
The 2240×1260 connected fixture with eight mixed units, visibility, tactical,
navigation/reset/config-off and 1 GiB reserved VA passes. Sampled free VA reaches
256.2 MiB without worker/visual failure. The reservation is a pressure model,
not a reproduction of Civ III heap fragmentation or the reported live freeze.

Two replays reproduce all 337 displayed fingerprints exactly. One completes;
the other retains the intermittent reset failure at event 19709 / call 9296,
now identified as native map image 10 (1120×630) CPU restoration. This is not
resolved by the successful run. The exact repair DLL (`f2598dad…`) is staged
for evaluation; `native/build/native-failure-ownership/staging.json` records its
identity and the prior `898acda7…` rollback. No installer or game was launched.
`CAPTURE_FAILURE.bat -CheckOnly` passes for the existing logs-only collector.
The next live check can reproduce the reported movement/turn failure with bounded
logs and timing, without another input journal or window-image sequence.

Candidate/evidence: `native/build/native-failure-ownership/`,
`native-failure-ownership-pressure/` and `native-failure-ownership-replay/`.
No injected changes, patch symbols, memory-cap increase or visual-quality change.
The initiating live failure, intermittent reset mismatch, and M3.8 complete-path
latency attribution remain open; no claim of live fix or performance improvement.

### Ambient continuity and keyed tactical correction (evaluation staged)

Native screen transfers now sample the committed scene at the current visual
clock, so unit/UI updates cannot restore the original water/resource sample.
Camera preparation/adoption retains its existing scheduling guard. Visible
resources, water, selected idle units and working units animate; unselected idle units and fogged content
stay frozen. Slow visible unit frames no longer discard elapsed clip time.
Selection and route hooks supply the separate native map background when drawing
onto the keyed unit canvas. No new patch symbols, listener, timer or presenter.

Functional candidate: `native/build/continuous-presentation-final/C3XRenderer.dll`
(SHA-256 `0428f2a5…`, not staged). The current diagnostic build in
`continuous-presentation-diagnostic/` only adds image/extent/hash context to a
replay CPU-handoff error; it does not alter rendering or relax comparison. At the user’s
explicit request, that diagnostic build (`898acda7…`) was staged for
`INSTALL.bat` evaluation, with the prior DLL preserved for rollback in
`native/build/continuous-presentation-stage-rollback.dll`. The repair above now
supersedes it; neither evaluation build is input-capture-qualified.
Final connected fixture
`continuous-presentation-final-native` passes at 1120×630, including visible
marker rotation, native navigation/recovery and complete input recording.
The fullscreen `continuous-presentation-native-c` fixture passes: resource-only
motion, 8 native-only updates / 8 fresh map samples with cadence disabled,
24 movement updates / 33 map samples, blocked UI, fog/reveal, keyed selection and
route, cancellation, reset and configuration-off. The direct GPU regression checks
eight exact changing displays with no autonomous offers. Injected compile passes;
portable eligibility/source-pose tests preserve frozen unselected idle units.
Independent mixed-scene requests average 29.9 ms at 2240×1260 (41.1 ms including
desktop completion); this is fixture timing, not a live FPS or M3.8 acceptance.

The desktop-size recorded fixture reproduces the old invisible keyed marker and
shows the corrected marker/route in `continuous-presentation-recorded/`.
Its native assertions pass and capture closes; its original aggregate receipt is
failed because the collector still expected retired padded-map adoption below
fullscreen. The collector now requires the current exact bounded-map witness at
all sizes. That diagnostic receipt is preserved, not relabeled as a pass.
The first candidate also exposed intermittent camera-adoption divergence in a
repeated replay; the final candidate preserves the original camera guard.
All 337 final-candidate displayed fingerprints match across four runs. However,
two runs fail the strict native CPU ownership hash at event 19709 / call 9296,
a pending-camera reset after those presentations. Another final-candidate replay
and both diagnostic-build runs complete. This remains **unqualified**, not fixed
by successful retries; logs and the diagnostic summary are preserved under
`continuous-presentation-final-replay/`.

**Immediate priority:** the user reports that the evaluation build jumps to a map
corner and appears frozen after the first turn. This is a separate unresolved
live regression report, not a demonstrated instance of the replay reset mismatch.
Check turn transition, native selected-unit centering, requested/adopted/displayed
view identity and ownership progress before further performance changes. Existing
replays do not yet reproduce this symptom. The focused failure tests and pressure fixture above now pass, but do not
reproduce this live trigger; the next specific evidence is the logs-only failure
check, not a new input recording. The staged DLL is not short-capture-qualified. The reset mismatch also remains unresolved;
causation versus the prior renderer is unproved. Refresh qualification after the
fixes, then resume optimization. Preserve the user’s latest live scene export.
No game was launched by the agent or fixed references replaced; the user has
now tested the evaluation build. Generated BMP evidence is losslessly compressed.

The formal [M3 extension](#m3-extension-deliverables) below replaces the informal
ranked TODO list. M3.8 stabilizes and attributes; M3.9–3.12 simplify the warm path,
ambient scheduling, world readiness and ownership; M3.13 verifies them together.
M4 submission experiments remain conditional on measured benefit. The optional
[M3.H capacity trial](#m3h---conditional-64-bit-helper-trial) has an explicit decision
boundary and is not permission to silently replace the production renderer.

**Completed consolidation:** one world-region preparation order replaces the
neighborhood geometry queue; exact retained-scene views replace padded/alternate
camera images and their injected requests. Combined compilation now owns immutable
observations and shared topology, survives camera replacement and validates current
dependencies before adoption. Frame-local combined callbacks/joins are removed.
Native draw preludes share the following resource-operation handoff; adjacent
compatible rigid draws are instanced without reordering. Measured process VA
coordinates geometry growth and preparation capacity, with bounded immutable
inputs and compiler scratch reserves. No new patch symbols or rendering ownership;
all normal water effects remain enabled. Native CPU access and asset/device reset
barriers are preserved. See the linked results for the retirement ledger.

**Measured:** same-workload paired replay has cold maxima 7.64–7.73 →
6.93–7.25 s, map boundary p95 336–342 → 333–379 ms, and ambient p95
28–34 → 25–37 ms. The service envelope remains variable (34.0–35.0 →
29.5–35.0 s); there is no demonstrated stable overall FPS gain. Native fixture
handoffs fall 6,402 → 4,475 for the same 2,014 draw batches. All 518 recorded
frame fingerprints match; 45 selected executable contracts pass. The final
fullscreen native fixture passes animation, fog/reveal, native composition,
camera changes, reset and configuration-off. Memory improves modestly, but
contiguous free VA still falls below 512 MiB. These are service measurements,
not live FPS or acceptance of the killed recording's failed tail.

**Previous qualification:** the preceding `9e76737a…` build was qualified for
short diagnostic capture. The evaluation DLL staged above supersedes it for direct
`INSTALL.bat` testing; the capture launcher receipt still belongs to the old build.
The four-arm overhead campaign, fullscreen window witness, stop/recovery controls
and two identical 960-presentation replays pass. Run `INSTALL.bat` before testing
if the preceding camera-consolidation bridge changes have not been installed.
No game was launched. Ten-minute recording and live FPS remain unqualified.

**Next unfinished responsibility:** shorten the exact map boundary through
view/pass assembly, GPU execution and native composition on the existing recorded
workload. Cold content and foreground adoption remain costly. The Standard
<33 ms p95 camera goal and live heap/failed-tail behavior are not accepted.
No new manual recording is required before that investigation. M3.8 remains open;
this work does not start M4 or a separate-process renderer.

