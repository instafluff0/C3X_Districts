# Actual game log — September 9

The user supplied 9,643 debugger-output lines covering approximately 100 seconds
of renderer events. Local evidence is preserved in
`Renderer/lab/out/navigation/live-usage-20260909/`: `debugger-output.txt`,
`analysis.json`, and `details.json`. Original SHA-256:
`2501c6a921868aabd3e479ad47c9827b857a66058c9b30b733e94336481a33a5`.
Raw debugger output remains ignored; do not commit its local paths.

## What this actually exercises

One 1138×1188 viewport, zoom 128 only, waves **off**, reflections **off**.
The log confirms the production cache profile: world regions, retained waves,
backdrops and receiver index enabled; 64 MiB viewport, 832 MiB backdrop and
512 MiB unit-pose limits. The loaded module is logged, but its exact DLL hash is
not embedded in this capture. Do not claim cryptographic loaded-binary identity.

All 109 request/result pairs match, with zero failed calls, invalid records,
reported tile fallbacks or device recoveries. Maximum captured counts were
743 visible tiles, three cities, three roads, 43 resources and 13 tile-unit
entries. There were no logged railroads, farms, mines or camps. Unit-body logs
contain 60 distinct IDs across the session, not 60 simultaneously visible units.
These counts do not establish dense-map coverage; the separate fixture does.

The capture ends after several thread exits with a debugger exception in
`KernelBase.dll`: code `0x0000087A`, parameters `0x887A0001, 0x00000053` (line
9,641). There is no subsequent successful program-exit record or call stack.
Zero failed render pairs does **not** establish a clean shutdown or rule out a
later failure. Preserve this tail for investigation; this log alone cannot tell
whether the exception was handled or fatal, or identify its root cause.

## Latency and likely feel

| Workload | Samples | Median | p95 / maximum | Interpretation |
|---|---:|---:|---:|---|
| Initial map DLL call | 1 | 12,872 ms | 12,872 ms | A substantial first-view stall |
| Unchanged camera DLL call | 99 | 36.6 ms | 369.9 / 682.8 ms | Usually cheap map reuse, with noticeable interruptions |
| Nearby camera change | 1 | 472.5 ms | 472.5 ms | One roughly half-second sample, insufficient pan distribution |
| Distant camera change | 7 | 2,498.7 ms | 7,519.0 ms | Multi-second freezes; minimap clicks are inferred, not logged input events |
| World revision change | 1 | 1,277.8 ms | 1,277.8 ms | Changed content can invalidate otherwise resident work |
| Native map pass | 109 | 52.9 ms | 1,566.5 / 12,896.5 ms | Includes capture/composite/native map work, not every unit update |

Stationary means the camera did not change; it does not prove the user was idle.
No zoom changes or waves-on samples are present. Do not extrapolate their live
latency from this log or compare this smaller viewport as a controlled speedup
against the 2240×1192 standalone fixtures.

The scheduler callback gap was 118.9 ms median / 153.0 ms p95 (96 intervals).
That is roughly eight callback opportunities per second in the typical segment,
not a measured physical display frame rate. Native timer work was 110.4 ms median
/ 127.2 ms p95; scheduler bookkeeping itself was only 0.169 / 0.263 ms. Map-pass
completion intervals were 123.7 ms median, with long travel/activity gaps.
One callback gap was 73.9 seconds; it spans intervening activity and cannot all
be attributed to a single renderer call. The 30 native-presented-FPS target and
1,000-frame requirement are not met or proven by these counters.

A 37 ms map render therefore does not mean smooth 27 FPS gameplay. Current idle
animation can feel visibly stepped, cold unit poses can hitch, and new regions
can freeze for seconds. These are observations under debugger capture; detailed
logging overhead is not separately measured.

## Where the time went

- Initial geometry took 12,205.6 ms of the 12,872.0 ms DLL call. Its mesh breakdown
  included 6,329.8 ms cliffs, 2,743.9 ms ground and 1,326.7 ms upload. The slowest
  distant call spent 7,018.8 of 7,519.0 ms in geometry. This makes preparation of
  reusable world structure the highest-value terrain experiment.
- Existing background prewarm reported 25 events: 24 interrupted, one ready,
  zero built entries and zero prefetched bytes. Diagnose why ordinary redraws
  prevent useful preparation before adding another global cache layer. The worker
  should retain useful bounded progress without publishing stale scene data.
- 3,883 unit-body calls: 3,051 pose hits and 832 misses. Hits cost 0.503 ms median /
  0.905 ms p95; misses cost 51.152 / 57.143 ms, maximum 138.309 ms. Misses consumed
  39.59 seconds of the 41.26 seconds summed unit-call durations (about 96%). This
  sum is work across the capture, not additional wall time to add to map totals.
  Profile pose generation/readback and bounded preparation of likely next poses;
  retain exact identity/action/direction/cursor keys and independent ambient phase.
- Unit action codes observed were 1 (2,748 calls), 2 (598), 7 (74), and 12 (463).
  This is not full combat/direction/interruption coverage. Use existing action
  witnesses and the mixed-action continuous fixture for the missing paths.
- Four `city-composition-fallback` events report `no-legal-immutable-composition`.
  Successful tile output does not erase these narrower composition fallbacks.
  Investigate the current city composition contract separately; do not expand
  deferred wonder or District ownership.
- The analyzer previously treated `map-complete.animation_ms` as a duration. The
  injected source confirms it is an animation clock. It is now excluded from
  latency statistics and covered by a regression test; the roughly 59-second
  values were not 59-second individual animation renders.

## Next experiments

First instrument/reduce cold unit-pose cost, then prepare camera-independent
terrain/relief and share compact geometry across 128/160/192. Both attack measured
multi-second stalls. Extend the existing retained caches and bounded camera queue;
keep native asynchronous completion a separate integration change. Never publish
an old-camera bitmap under current-camera overlays, visibility or picking.
Use [the busy session](busy_navigation_session.md) before/after with exact images,
and retain [the original targets](navigation_implementation_plan.md) and
[native presentation audit](native_async_presentation_audit.md).
