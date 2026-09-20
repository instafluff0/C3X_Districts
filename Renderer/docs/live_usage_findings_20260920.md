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

Next: verify live residency after installing the corrected bridge, then measure
remaining camera and speculative-preparation costs. A resident replay is not
proof that every live UI/native sequence now stays resident or reaches the
Standard-map latency target.

The stream also reports Windows' fault-tolerant heap shim at startup. Its cost
and capture overhead are unmeasured; neither explains away the observed route
loss. No system compatibility settings were changed. No further manual capture
is requested until the current evidence and automated reproduction are exhausted.
M3.8 and the <33 ms p95 target remain unpassed. No post-fix live FPS improvement
is claimed before another installed-game measurement.
