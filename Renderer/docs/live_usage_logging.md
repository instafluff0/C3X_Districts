# Actual-use renderer logging

The renderer uses the existing `OutputDebugStringA` stream. Ordinary gameplay
does not open a log file. Capture that stream with the existing Windows debug
capture workflow and retain its exported text. Start capture before launching
the game when possible so the session calibration and effective settings are
included. There are no player names, save paths or credentials in the new records.

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
adds no GPU wait, per-tile I/O, asset loading or game-state mutation.

The first user-supplied capture is analyzed in
[September 9 live usage findings](live_usage_findings_20260909.md). The analyzer
excludes `map-complete.animation_ms`, an animation clock, from latency statistics.
