# First live input-capture bottleneck findings

The live recording confirms two problems: expensive, intermittent scene work
before failure, and GPU presentation failure under severe 32-bit address-space
pressure. These are the original baseline findings. The subsequent
[working-set implementation and comparison](frame_working_set_results.md) address
the first measured causes; live-FPS qualification remains open.

Corpus: `native/build/live-captures/20260922-004213-95ba3a/`. Pinned DLL SHA-256:
`46f0cfe973a73432d68f22662c585e8866c2d3c91cda93ea7da5b37d4844edea`.
The Standard world has 5,000 tiles; presentation is 2240 × 1260, with waves and
reflections enabled. The hardware D3D11 device is the Parallels display adapter.
The killed process left 119.662 seconds of verified journal, 308,603 completed
calls, 518 accepted renderer presentations, 516 PresentMon rows and 349 sampled
window images. Missing footer means incomplete session, not corrupt prefix.

## Measured live behavior

| Evidence | Measurement | Interpretation |
| --- | --- | --- |
| DXGI presentation intervals, entire capture | Median 35.55 ms; p95 392.87 ms | Includes startup/lifecycle; does not measure every native GDI update |
| Gameplay interval 30–71.8 s | 448 presents, approximately 10.7/s; median interval 32.29 ms, p95 357.34 ms | Short smooth stretches conceal large stalls |
| Final GPU presentation | 71.804 s | No later DXGI presents recorded before process termination |
| First worker failure | 71.883 s; 111.55 MiB available virtual; device reason `0x887a0020` | Strong evidence of pressure-associated device failure, not proof of the precise failed allocation |
| Subsequent recovery | Failures at 73.347 and 74.086 s; two texture-load failures | Recovery did not restore GPU presentation |
| Whole-process memory | Peak sampled private bytes 3,352 MiB; smallest sampled free region 17.98 MiB | Total free space alone understates fragmentation |
| Scene target accounting | Approximately 1,171 MiB, including 519 MiB static backup | Logical attachment estimate, not a direct measurement of driver VA mappings |
| Initial map request | 11.34 s renderer request; 1,511 compiled records; 6.49 s worker join | Initial content preparation remains expensive |
| Reflection rebuilds | 13 rebuilds, 2,546 cells, zero page reuses; 205–1,224 ms each | A large measured source of intermittent scene stalls |
| Scene composition | Median 13.32 ms, p95 509.03 ms | Explains why a favorable idle measurement missed gameplay tails |
| Native map-prepare boundaries | 21 calls; median 303.59 ms, maximum 11.54 s | Exact synchronous compatibility path still blocks callers |

Native calls and sampled window updates continue after the last GPU presentation.
Do not label this a proved whole-game deadlock. The window changes to a different
map scale after the failure; the final appearance is consistent with broken GPU
presentation/ownership recovery. No thread dump was captured, so the user's final
input unresponsiveness cannot be attributed to a specific blocked thread.

Present API time is small (p95 0.232 ms), while reported GPU-active p95 is
16.04 ms. These observations argue against Present itself explaining the long
tails, but VM GPU metrics are not a complete per-pass GPU profile. Call durations
include waits and overlap; never add nested call totals as independent CPU work.

## Architectural causes and prioritized experiments

1. **Reduce the resident scene working set before increasing caches.**
   `compose_scene_surface` retains a twice-width/twice-height HDR scene with
   four-sample color/depth and finishing targets. `LinearBackup` stores matching
   MSAA color/depth tiles underneath animated damage; in this capture those tiles
   occupy nearly a full viewport. The existing pressure controller only trims
   reflection and unit caches, releasing about 61 MiB at 32.8 s while these much
   larger attachments remain. Measure a bounded attachment/backup redesign with
   shared transient lifetimes and conservative visible damage. Preserve fog-edge,
   overlap and animation correctness; any quality change needs explicit comparison.
   Acceptance: no failed allocations/device recovery, no growing retained memory,
   and adequate total and contiguous headroom throughout this corpus.

2. **Stop reflection-page cache thrashing and overbroad receiver work.**
   At pressure, the cache limit becomes 64 MiB. A page costs 663,552 bytes, so it
   holds at most 101 pages; the common requested set has 198 pages. Sequential
   replacement cannot retain that working set. Logs show all pages rebuilt after
   invalidation and no reused pages; key invalidation may contribute as well.
   Receiver selection currently uses water/river geometry bounds before final fog
   composition. The visible game contains only a small explored area. Investigate
   conservative visibility-aware receiver selection, retained current-view atlas
   damage and finer invalidation rather than increasing memory limits blindly.
   Acceptance: unchanged relevant pages survive scene edits and rebuild work scales
   with affected visible receivers, with water/reflections still enabled.

3. **Remove remaining expensive exact native boundaries where contracts permit.**
   Native map preparation uses the synchronous GPU render path when an existing
   navigation result cannot be adopted. Native copies, sprites, units and presents
   also have long outliers. Separate renderer execution, native ownership waits
   and driver backpressure before rewriting their synchronization. The cold
   geometry build is separate from warm reflection/composition work: only 141
   additional geometry records were built across subsequent traced map frames.
   Acceptance: report trigger-specific p95 and coherent output latency, not only
   isolated idle-frame speed or quicker request submission.

A separate 64-bit renderer could relieve address-space competition, but would
not remove the attachment bandwidth, page thrashing, or excess submission work.
This capture justifies measuring both memory architecture and rendering work;
it does not justify assuming that process separation alone reaches 33 ms.

## Replay fidelity actually established

The first replay rejected configuration because the game recorded relative paths
such as `C3X_Districts`. Running from the installed Conquests working directory
resolves that mismatch without altering the DLL or journal. General replay
launchers still need a durable working-directory contract; fixture-only tests used
paths that did not expose this assumption.

Two fresh exact-DLL runs then regenerated **all 518 successful presentations with
identical fingerprints** and passed the recorded checks up to event 508145,
call 253274. Both reject there: the recorded ambient offer returned 0, while replay
returns 1. This is the failure interval; the standalone process has different
available address space and does not reproduce the live allocation/device failure.
Do not suppress this mismatch or certify the entire failed session as reproduced.

Only frame 518 was exported. It visually matches window sample 224, recorded
about 94 ms later, including map, fog and native HUD. JPEG window samples are not
an exact pixel oracle. This establishes repeatability plus one direct visual
comparison of the pre-failure output, not every-frame scanout equivalence or
replay/live timing equivalence. No replay timings are reported as gameplay FPS.

The next validation work is to preserve the working-directory requirement and
exercise measured pressure/failure behavior explicitly, while optimizing the
successful prefix independently. A fixed reservation models capacity, not the
game's allocation history or fragmentation. No further manual capture is needed
for this first investigation.

## Observer and storage limits

The 128 MiB early-stop guard did not trigger: the once-per-second observer's
lowest free-space sample was about 179 MiB; the renderer caught a lower transient
between those samples. This guard is not crash prevention. Observer address scans
had median 22.18 ms and p95 41.05 ms, on the observer process. Its cumulative CPU
time was about 7.5 seconds over the session. Capture overhead is real and not yet
calibrated against this exact live run. The journal stored about 720 MB of events.

Evidence: `diagnosis/live-summary.json`, both `replay-game-directory/` and
`replay-repeat/`, `replay-findings.json`, `window-contact.jpg`, and
`replay-window-comparison.jpg` beneath the corpus. Fingerprints replace full
image sequences; the sole exported BMP was converted losslessly to PNG.
Original inputs are reused in place. No game launch, renderer behavior change,
staging, injection, or new user recording was required.
