# Input coverage and remaining qualification

Protocol 10 declares **zero complete-coverage bits**. A captured function or a
successful short replay does not establish the complete gameplay contract in
[recorded renderer workloads](../../docs/recorded_renderer_workload.md).

| Boundary | Captured and executed | Unfinished proof or missing input |
| --- | --- | --- |
| Configuration and assets | Pack/definition/unit controls; observed file contents and optional misses; DLL SHA-256; environment changes; altered-configuration rejection | Compiled shader blobs and device/driver capabilities |
| Map production | Copied frame, ordered tiles and topology through CPU/GPU production render exports; full native fixture output witnesses and two exact repeated replays; visibility mutation rejection | Broader gameplay coverage |
| World acquisition | Callback registration and copied page results through the production page owner; fresh 5,000-tile fixture, 201 page returns and two exact 100-destination replays; changed range/scope/topology cleanly rejected; portable stale-publication rejection/retry | Broader native callback failure coverage |
| Units | GPU and 16/32-bit CPU paths, native canvas edits, action inputs, forget events; one bounded CPU canvas registry across serial caller threads | Broader action/lifecycle mutations; simultaneous CPU producers and section-backed/overlapping canvases explicitly stop capture |
| Camera | CPU/GPU begin/poll/cancel, preparation calls; adopted identity, sample time, ownership and occurrence witnesses | Production native navigation/request/poll/map transactions now re-execute; broader game caller/window scheduling remains unqualified |
| Composition | Actual composition owner and image adapter; operation arguments, native metadata, pixels, palettes/sprites, lookup/font/DC state; nested GPU implementation calls suppressed | Unsupported external formats stop capture; CPU fallback snapshots constrain comparisons to equivalent native pixel semantics |
| Native ownership | Lifecycle identity/caller role and actual owner retirement; destroy identity persists through both notifications; safe replay teardown | Broader native aliases and nonowner lifecycle scheduling |
| Ambient | Consumed logical clocks and exported queries, absolute QPC/UTC correlation, policy, offers, accepted outputs; retained display inspection | Automatic pending/backpressure decisions and window visibility/focus transitions |
| Presentation | Actual existing presenter with a replay HWND; retained native/ambient display source; external native/DirectComposition fixture window samples with verified clock alignment | Live-game window coverage, calibrated observer overhead, correlated Present/scanout outcomes; device-loss and window lifecycle controls |
| Reset/fallback | Root reset/configuration transactions; actual owner drains and native CPU output witnesses; CPU unit/native screen fallbacks | Malformed API requests rejected before recording begins, compatibility map blit |
| Storage | Bounded async writer, segment/metadata checksums, quota/slow-write/I/O tests, incomplete prefix rejection and explicit recovery; real ten-minute capture and two matching 10,200-frame replays | Opt-in launcher collects/pins journal, window samples and tools; parser/quoting/automatic stop verified without game launch; calibrated live overhead remains |

Current short controls reject missing clocks, assets, units and resets and altered
configuration, visibility, action and CPU input pixels. All eight reject with exit
code 1 and the expected diagnostic; a crash or unrelated failure does not pass.
The action change requests an uncaptured animation asset and is rejected at that
boundary; this does not establish coverage of every action cursor/lifecycle field.
The current short capture also alternates CPU unit callers between threads.
Frame selection
and time selection execute the complete prefix and agree with full replay. A torn
final record is rejected by default; explicit prefix replay retains
`complete:false` and `qualified:false`.

Full native replay has already exposed a production geometry-reuse defect: a
translated camera could retain off-screen contributors selected for the prior
viewport. Later cancellation could force a cold assembly with a different set.
The candidate now checks selection membership before reusing translated geometry.
The recorded example selected 1,511 contributors initially and 1,403 after moving;
this is a correctness finding, not a performance measurement. The repaired full
fixture and two replays passed: 13,128 calls and 519 composed presentation images
identical between replays, including the reset-cleanup output witnesses.

The Standard whole-world fixture also passed: 100 destination presentations match
between two replays, with six exact independent cold pixel comparisons in capture.

The real ten-minute 2240×1260 capture passed two exact-DLL replays: 92,130 calls,
50,611 sampled clocks and 10,200 identical display fingerprints. Compact witnesses
use 2.52 MB for both replays. The matching capture-off arm also passed; comparison
excludes a known initial compiler-interference interval from both arms. This is
one ordered fixture pair, not calibrated live-game overhead or a FPS result.

The combined native/window pressure fixture passes all native pixel oracles and
both process exit checks. Its 119 sampled window images align with the input clock
within 0.044 ms; two replays match all 970 display fingerprints. Minimum available
VA is 282.2 MiB, above the earlier live 120 MiB envelope. A virtual reservation
does not reproduce heap fragmentation, and sampled window images are not an
every-frame physical display oracle.

The current native-owner fixture passes all pixel/adoption/reset checks and two
exact 426-frame replays: 12,930 calls, 949 native-root calls and 515.56 MB. The
recorder's previously premature identity retirement is repaired. Replay rejects
missing external values before dereferencing proxy identities. Recording failures
preserve the original native pixel lease and release behavior. Native dependency
storage is bounded and retired with destroyed images.

A separate unpaced measurement mode executes real production work and excludes
frame-fingerprint/export work from the measured run. It preserves recorded camera
consumption points because external CPU snapshots are not available earlier;
actual worker readiness is awaited there. It is a service-cost/capacity probe,
not a new game-caller simulation or live responsiveness qualification. Native
setup reconstruction and journal I/O still compete for CPU, and replay memory is
included in process measurements. Never interpret these service samples as FPS.

Next responsibility: finish the automated overhead/pressure admission gate, then
collect one strategic live session and compare actual window symptoms with replay.
Calibrate the co-resident CPU/GPU/memory envelope and observer overhead before
claiming live timing equivalence. This checkpoint determines which architectural comparison
is valid; successful fixture replay alone does not prove the live slowdown or
freeze is reproduced. Complete-coverage bits intentionally remain zero.
