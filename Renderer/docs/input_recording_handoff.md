# Input recorder and playback handoff

The current protocol captures immutable renderer and native-owner inputs and
re-executes the production DLL. It can regenerate composed frames, inspect a
frame/time range, compare a changed DLL, and play visibly on Windows. It is not
yet qualified as a reproduction of live game responsiveness. The
[recording contract](recorded_renderer_workload.md) remains the acceptance gate.

## Implemented and verified

- Native image/composition/navigation, reset and configuration transactions use
  owned external values during replay. Game pointers are never dereferenced.
  Recorder failure preserves the original native pixel lease. Destroy identities
  remain valid until both lifecycle notifications finish.
- Final candidate: `6c1faaab6275f403fdcd82f207117d8b5102fe33a14da8a5027adc6a0b0fc7f1`.
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

## Next unfinished responsibility

Finish automated capture-overhead and live-pressure qualification, then use one
strategic live input corpus to compare regenerated frames with its sampled game
window evidence. Existing pixel-only live captures cannot supply missing scene
inputs retroactively. Do not request another long manual capture before the
automated acceptance gate passes.

The service experiment serializes completed calls and preserves recorded native
consumption points. It cannot yet measure how the game would react to an earlier
camera completion, reproduce its heap fragmentation, or claim identical OS/driver
timing. Architecture comparisons must retain those limits until calibrated.
Standard navigation below 33 ms p95 and sustained live stability remain open.
No injected source or patch-table changes, INSTALL, or game launch were needed.
