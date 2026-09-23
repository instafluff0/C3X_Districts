# Input recorder and playback handoff

The current protocol captures immutable renderer and native-owner inputs and
re-executes the production DLL. It can regenerate composed frames, inspect a
frame/time range, compare a changed DLL, and play visibly on Windows. It is not
yet qualified as a reproduction of live game responsiveness. The
[recording contract](recorded_renderer_workload.md) remains the acceptance gate.

## Current renderer evaluation

The four measured working-set/submission changes are implemented and staged as
DLL `54aac95f5e1174aaf3c9b8f034a16c3c786fad8258a4e336c8bcce75e14558d4`.
See [results, removed machinery and next responsibility](frame_working_set_results.md).
All 518 successful live-prefix frames match; paired replay execution improves
20.8%, ambient p95 falls from 49 to 25–27 ms, and minimum free VA rises from
268–320 to 581–643 MiB. Fullscreen native lifecycle/animation/fog tests also pass
with eight units and 1 GiB reserved. These are not live FPS or failure-recovery
qualification. Long native map-preparation boundaries remain unfinished.

Short diagnostic capture has now been requalified for this exact DLL and the
current tools. All four overhead arms pass; two fresh complete replays match all
962 frames across 17,226 calls. Requested-stop, window capture, actual Windows
preflight and installed-path identity pass. Evidence is under
`native/build/input-recording/frame-working-set-capture-{campaign,controls,observer,preflight}`.
The qualifier accepts both supported build-receipt formats while checking every
runtime source and rejecting mismatched DLLs. `CAPTURE_DIAGNOSTIC.bat` is ready;
no reinstall or new manual capture is required for continued diagnosis.

## Implemented and verified

- Native image/composition/navigation, reset and configuration transactions use
  owned external values during replay. Game pointers are never dereferenced.
  Recorder failure preserves the original native pixel lease. Destroy identities
  remain valid until both lifecycle notifications finish.
- Recorder baseline: `6c1faaab6275f403fdcd82f207117d8b5102fe33a14da8a5027adc6a0b0fc7f1`.
  Two 2240×1260 replays pass all output witnesses and match all 426 display
  fingerprints across 12,930 calls. Capture: 98.97 seconds, 515.56 MB and
  17.78 MB peak writer queue. Evidence:
  `native/build/input-recording/native-bridge-replay-20260921-g/receipt.json`.
- The final short suite passes frame/time seeking, explicit incomplete-prefix
  recovery and eight semantic negative controls. A removed native dependency
  separately rejects with exit 1; native-value tests check the precise missing
  field before any proxy dereference.
- Earlier protocol-8 storage endurance remains distinct evidence: a real
  600-second capture and two matching 10,200-frame replays. It is not relabeled
  as a protocol-10 ten-minute native-owner capture.
- Unpaced service measurements complete the same corpus normally and with
  1 GiB VA reserved. Minimum sampled free VA is 1,134/297 MiB respectively;
  the pressure run's largest free region reaches 194 MiB. These are capacity
  probes, not live FPS or the earlier game's approximately 120 MiB envelope.
  Evidence: `native/build/input-recording/native-bridge-performance-20260921-g/`
  and `native-bridge-pressure-20260921-g/`.
- `PLAY_REPLAY.bat` opens a session chooser and launches paced visible playback
  using the session's pinned DLL/tools. Direct playback matches all six short
  control frames; launcher preflight and actual playback pass on the shared VM
  path. The player drops no presentations and reports lag. Escape stops it.
- Capture launcher quoting, parser, owned-window sampling and automatic shutdown
  pass without launching Civ III. It freezes DLL/tool identities and records
  environment metadata. Input capture remains opt-in and admission-gated.
- A final capture-off/on native fixture pair passes, including a 30-second
  Standard-map interval with 1 GiB VA reserved. Idle p95 is 25.23/23.52 ms,
  action-phase p95 26.09/25.52 ms, and camera-phase p95 248.56/260.45 ms.
  Minimum sampled free VA is 254.85/251.06 MiB. The recorded arm closes at
  522.05 MB with a 17.78 MB peak queue. This single ordered pair shows no large
  added tail in these cases; faster recorded idle samples are not a speedup.
  Receipt: `native/build/input-recording/native-overhead-comparison-20260921-g.json`.
- Real-time comparison candidate:
  `9714a267e2f33cb03bec2be2d522e4c747719d9e471a12d9a29d866080868790`.
  [COMPARE_REPLAY.bat](realtime_replay_comparison.md) plays Before then After
  on the original input timeline, with independent production animation cadence.
  The same-DLL control completes 17,218 calls in 128.118/128.084 seconds with
  1 GiB reserved, generating 851/844 autonomous presentations. Visible window
  samples confirm changing water and unit poses. This is a workflow control,
  not a speedup measurement; window observation overlapped part of one arm.
  Evidence: `native/build/input-recording/realtime-compare-20260921-a/` and
  `realtime-window-20260921-a/`. The current production cadence remains 33 ms;
  no test-only faster scheduler or per-frame readback is added.
  Exact forensic playback remains a separate correctness check: this candidate
  passes all 12,930 calls and matches the recorder baseline's 426 fingerprints.
  Receipt: `native/build/input-recording/realtime-final-20260921-a/receipt.json`.

## Earlier recording foundation and capture gate

The previous short-session preparation completed with staged
DLL `46f0cfe973a73432d68f22662c585e8866c2d3c91cda93ea7da5b37d4844edea`;
The prior Windows preflight passed,
including signed tools and exact launcher/tool identities. Normal ten-minute
admission remains closed. See [short diagnostic capture](short_diagnostic_capture.md).

The fullscreen Standard fixture with all water effects and 1 GiB reserved passes
the off/on/on/off recorder-overhead gate: idle p95 23.707/24.159 ms, action
24.925/25.965 ms, camera 252.224/272.739 ms (off/on). Idle and open-transaction
stop tests, canvas ownership and window collector checks pass. Two fresh replays
of 17,191 calls match all 950 displayed fingerprints. Combined input/window
capture also passes; its observer cost is not calibrated as live-game overhead.
An earlier recorder-off run with 1,152 MiB reserved fails an ambient-pixel check
near 85 MiB free address space. Preserve that limit; the launcher requests an early
stop below 128 MiB sampled free space rather than claiming the pressure issue fixed.

Evidence is under `native/build/input-recording/short-capture-`: campaign
`campaign-20260921-b`, controls `controls-20260921-a`, combined observer
`observer-20260921-a`, and final preflight `preflight-20260921-b`. The readiness
receipt pins the build and rollback. Redundant historical images/intermediates
were removed; retain journals and measurements and regenerate only needed frames.

The first live session is captured and investigated; see
[measured bottlenecks and replay limits](live_bottleneck_findings.md). It preserves
119.662 seconds despite process termination. Two exact-DLL replays reproduce all
518 successful presentations identically, and the last regenerated frame visually
matches its nearby window sample. Both diverge at the live device-failure boundary:
the standalone process successfully renders the offer that failed in-game.
Replay also requires the game's working directory for recorded relative paths;
general launchers still need that contract fixed. Do not call the failure or live
FPS reproduced. No additional manual capture is currently needed.

Next: reduce the approximately 1.14 GiB scene attachment/backup footprint, address
the 64 MiB reflection cache versus approximately 125 MiB demanded pages, and
measure remaining synchronous native boundaries on this corpus. Pressure/failure
calibration and ten-minute/live-FPS qualification remain unfinished. Preserve the
failed live boundary as evidence rather than bypassing its replay check.

The service experiment serializes completed calls and preserves recorded native
consumption points. It cannot yet measure how the game would react to an earlier
camera completion, reproduce its heap fragmentation, or claim identical OS/driver
timing. Architecture comparisons must retain those limits until calibrated.
Real-time playback adds intervening ambient frames, but preserves these same
native consumption points. It does not yet prove that replay FPS predicts live FPS.
Standard navigation below 33 ms p95 and sustained live stability remain open.
No injected source or patch-table changes, INSTALL, or game launch were needed.

Candidate replays can now use `--compare-candidate --pixel-audit` to complete a
recording after a rendering-only pixel change. The mode still rejects changed
result, dimensions, ownership, camera identity and non-pixel witnesses; it
counts map and CPU-unit pixel-witness mismatches in the final summary. Strict
pixel matching remains the default. A pixel-audited replay is diagnostic, not
visual acceptance: compare exported candidate frames with the control before
promoting changed art.
