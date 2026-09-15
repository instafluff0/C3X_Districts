# Actual-use renderer logging

The renderer uses the existing `OutputDebugStringA` stream. Ordinary gameplay
does not open a log file. Capture that stream with the existing Windows debug
capture workflow and retain its exported text. Start capture before launching
the game when possible so the session calibration and effective settings are
included. There are no player names, save paths or credentials in the new records.

The owner-preservation evaluation retains the temporary diagnostics below. The
version-3 input had 83 post-draw samples / 128,505 checked pixels / zero mismatches;
its 8-bit intermediates led to the reproduced graphics-owner replacement defect.
Restart the game after installation so controls are recreated with native defaults.

The Advisor-corruption evaluation emits `ui-diagnostic-route version=3` at the
first final transfer through each composition state. `composition_active=0`
means CPU snapshot presentation; this must not be called a retained GPU map.
`ui-sprite-source route=cpu` now runs at the DLL dispatch even without a map owner.
It samples up to 96 ordinary/other and 96 keyed indexed sources, with fingerprints
for palette/content changes at reused addresses. `ui-sprite-palette` contains four
64-word chunks of each selected RGB555 palette for offline asset comparison.
Source hashes cover raw indexed rows; `raw_hash_valid=0` excludes compacted sources
from byte-hash comparison. These records contain no screen text or image addresses.

`ui-native-draw` samples up to 256 distinct ordinary raw indexed draw outcomes
immediately after the original sprite method returns. At native 1:1 scale it
compares nontransparent source indices against 8-bit destination indices or
selected 555/565 palette words against 16-bit destination pixels. `checked=0`
means no comparison was possible, not success. `mismatches` counts differing
pixels; `first`, `expected` and `actual` identify the first difference. Indexed
destinations also report `dib_colors` and `dib_palette_hash`. A later incorrect
outcome is not hidden by an earlier correct draw of the same source. This check
reads existing CPU storage and flushes pending GDI work; it does not draw or take
a new destination lease. Correlate source hashes with the preceding palette records.

`ui-sprite-oracle` still compares up to 96 eligible ordinary/keyed GPU programs
against actual JGL scratch drawing and full-color opaque output. It requires an
active composition owner, RGB555 destination and source at most 768×256. Absence
of oracle samples is **not** evidence of correct drawing. The September 15 game
capture exposed this coverage gap in the first diagnostic build. No original game
destination is drawn by the oracle. Temporary GPU samples add explicit readbacks;
do not benchmark this build. CPU diagnostic collection adds no GPU readback.
Use ordinary `INSTALL.bat` and the existing debug stream; no environment settings.
Remove diagnostics after the defect is resolved. `native-screen` counters are
cumulative; divide final callback totals by final calls, never report a percentile
of those totals as per-transfer latency.

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
