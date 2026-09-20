# Actual-use renderer logging

## One-click game capture

The current native startup-transfer repair changes both the renderer DLL and the
existing injected Graphsy wrapper. **Run `INSTALL.bat` once before the next
capture**, even if the previous outline bridge is already installed. Replacing
only the DLL leaves the early native transfer that revokes screen eligibility.
This capture launcher does not install the bridge.

In the Windows VM, close Civ III and double-click
`Conquests\C3X_Districts\Renderer\CAPTURE_GAME.bat`. Approve the Windows PowerShell
permission prompt for the FPS collector. The game itself runs without elevation;
neither installation nor injection is performed by this launcher.

Load a usual save, idle with water/units visible for 20 seconds, scroll for 20
seconds, make several distant jumps and switch selected units, then open/close a
city and select a worker. Quit normally after about 2–3 minutes and leave the
launcher open until it reports that the capture is saved. No debugger, manual
export or upload is needed. Captures stop after at most 15 minutes.

The launcher saves `renderer.log`, PresentMon `frames.csv`, collector errors and
`session.json` under `Renderer/native/build/live-captures/<session>/`. Per-frame
files are written to the VM's local temporary directory first, then copied to the
shared checkout on exit. The session records the staged DLL hash and collector
completion, not account names or save filenames. A failed setup leaves diagnostic
files and does not terminate a running game. Closing the launcher prematurely can
prevent the final copy; local temporary capture files remain recoverable.

Portable, signed [Microsoft DebugView](https://learn.microsoft.com/en-us/sysinternals/downloads/debugview)
and [Intel PresentMon](https://github.com/GameTechDev/PresentMon) are prepared in
the ignored `Renderer/native/build/live-tools/` folder in this checkout. Other
checkouts need `DebugView/dbgviewcli64a.exe` (ARM64) or `dbgviewcli64.exe` (x64), and
the console `PresentMon.exe`. They are not redistributed with source. The launcher
verifies publisher signatures and targets only Civ III. Its elevated helper owns
a unique ETW session, explicitly stops it on game exit, and flushes CSV output;
it does not install a service or change group membership. `CAPTURE_GAME.bat
-CheckOnly` validates prerequisites without starting collectors or the game.

The renderer's `graphics-device` startup record identifies `driver=hardware` or
`driver=warp`, feature level, adapter description and vendor/device IDs. In
Parallels, hardware identifies the VM's graphics driver, not proof of the host
GPU's physical utilization. PresentMon's guest presentation intervals complement
renderer timings; neither is a physical input-to-monitor measurement. Logging
has some overhead. The launcher sets ordinary trace level 1 and disables expensive
`C3X_RENDERER_PROFILE` profiling only for its child processes. It preserves game
rendering controls and does not change persistent environment settings.

## Existing debug stream

The renderer uses the existing `OutputDebugStringA` stream. Ordinary gameplay
does not open a log file. Capture that stream with the existing Windows debug
capture workflow and retain its exported text. Start capture before launching
the game when possible so the session calibration and effective settings are
included. There are no player names, save paths or credentials in the new records.

The accepted UI-corruption fix preserves the existing JGL graphics owner. Its
temporary `ui-sprite-source`, `ui-sprite-palette`, `ui-native-draw`,
`ui-sprite-oracle` and `ui-diagnostic-route` instrumentation is removed from game
execution. Preserved diagnostic checkpoints retain the source/palette evidence.

The startup integration uses compact route evidence:

- `native-tracking event=reset`: start/end of a native DLL tracking lifetime.
- `native-lifetime`: sampled eligibility queries and cumulative accepted counts;
  eligibility is ownership evidence, not a speed measurement.
- `composite gpu_map=1`: this map was committed through the resident GPU seam.
- `native-resident-present`: completed final transfer from a resident image,
  without a CPU screen snapshot.
- `native-screen`: CPU snapshot compatibility transfers, including static UI or
  explicit fallback. Counters are cumulative; divide final time totals by calls,
  not percentiles of cumulative totals.

For unit-pass attribution, `unit-body` also reports cumulative `shadow_passes`
and `shadow_input_bytes` for GPU caster submission, plus `cpu_shadow_upload_bytes`
for the independent CPU route. Compare deltas within a route; mixed CPU/GPU tests
naturally accumulate both. Resident pose hits submit neither pass nor input.

Use ordinary `INSTALL.bat` and the existing debug stream; no environment settings.
The strategic game check covers first map, idle/work animation, scrolling, zoom,
selection/picking, Advisor/modal screens, save/load and config-off behavior. A GPU
presenter alone does not prove resident composition; inspect map and final routes
together. Keep whole-request measurements separate from counts of eliminated work.

The DLL adds these events without changing the injected bridge or renderer ABI:

- `usage-session`: process identity, QPC frequency and UTC Unix milliseconds.
  Every DLL event also has QPC, monotonic milliseconds, thread and render sequence.
- `usage-settings`: effective waves/reflections, retained-cache switches and
  selected viewport, backdrop and unit-pose limits.
- `usage-view`: a unique synchronous request ID, captured view basis, zoom,
  target/map dimensions, wrap/revision and animation clock, plus visible terrain,
  city, road/railroad, farm, mine, camp, resource and tile-unit counts.
- `usage-result`: matching request ID, success/failure, complete DLL call time
  including worker/queue waits, geometry builds/reuse/uploads, draw submission,
  readback wait and visible animation. It includes immutable-bitmap fast returns.

The existing injected `world-topology`, `composite` and `map-complete` records
provide native capture, renderer wait, blit and complete map-pass timing with QPC
timestamps. Existing `unit-body` events provide identity, native action/cursor,
direction, placement, pose-cache hit/miss, cache bytes and body/blit duration.
This separates unit work from the map's animated resources and waves.

Analyze an exported capture from the repository root:

```sh
python3 -m Renderer.tools.analyze_renderer_trace path/to/captured-debug-output.log
```

The report groups matched calls into initial, stationary, nearby camera changes,
zoom, distant camera changes, resize and world-change workloads. Camera movement
is inferred from the captured view basis: a distant change does not prove the
minimap was clicked, and stationary does not prove the user was inactive.
Native map timing remains separate from DLL latency. Tile-unit counts are not
the number of separately drawn unit bodies; unit activity is reported separately.
Incomplete starts/ends, invalid measurements and short sample counts are explicit.
UTC timestamps are never treated as latency samples.

These boundaries do not measure physical screen presentation or the interval
from the user's original input event. Do not claim a native 30 FPS or
input-to-display pass from these logs alone. Time gaps may be deliberate pauses,
focus changes or modal UI. The session report is diagnostic evidence for selecting
the next optimization, together with the continuous busy-map simulation.

`C3X_RENDERER_TRACE=0` disables DLL diagnostics. Levels 1 and 2 both retain the
new complete-call events; level 2 also retains the existing detailed stages.
Explicit standalone `C3X_RENDERER_TRACE_FILE` output remains capped at 8 MiB
(32 MiB in explicit region-diagnostic mode); debugger output continues after
that cap. Do not use a truncated standalone trace as a full usage session.
Logging has some cost, particularly with an attached collector; the new code
adds no GPU wait, per-tile I/O, asset loading or game-state mutation outside the
explicit temporary UI oracle described above.

The first user-supplied capture is analyzed in
[September 9 live usage findings](live_usage_findings_20260909.md). The analyzer
excludes `map-complete.animation_ms`, an animation clock, from latency statistics.
The one-click capture's first live result is recorded in
[September 20 live usage findings](live_usage_findings_20260920.md): hardware
presentation was active while almost all map insertions fell back to CPU bitmaps.
