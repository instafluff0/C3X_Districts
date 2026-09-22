# Camera preparation consolidation

This M3.8 change retires duplicate preparation/image machinery inside the existing
renderer. It preserves full detail, all water effects, authoritative native input,
atomic publication, fog freezing, native centering and configuration-off behavior.
It does not complete the Standard-map 33 ms p95 acceptance goal.

## Surviving owners and removals

- `CapturedScene`, the world compiler/backing/residency owners and their existing
  bounded workers own content. `WorldPreparationSchedule` orders unfinished
  regions near current demand, retaining completed work across camera movement.
  Removed the separate neighborhood geometry queue, frame/topology copies,
  signature/cursor, and its worker branch. Region input is an immutable lease;
  appearance/lifetime/assets/device/projection changes re-arm it. Missing authority
  is retried after a new appearance revision; cancelled work stays pending.
- Exact camera requests use the retained scene, circular scene damage and ordinary
  publication at every supported extent. Removed `PreparedViewArea`, padded and
  alternate-zoom images, retained view family, prospective/refresh queues, crop
  adoption, partial-donor finishing damage and dependency collection. Removed the
  injected speculative zoom helper, post-composite requests and three fields.
  Old optional DLL exports explicitly decline; old bridges retain their exact
  render fallback. No patch-table changes or new worker/presenter.
- Existing native batching, read/write ownership and transaction barriers remain.
  They protect escaped CPU surfaces and coherent native UI composition. Optional
  image preparation no longer occupies these transactions. Frame-bound compiler
  leases and exact native map preparation are still meaningful remaining costs;
  this is not a claim that camera demand can never wait for content.
- Working-set cache admission now includes live composition allocations and
  simultaneous CPU/GPU publications, with room for the next publication. Required
  attachments retain their existing limits; optional caches receive the remainder.
  Geometry/assets keep their content budgets and process-VA pressure includes
  Civ III itself. Logical resource accounting is not measured physical VRAM and
  does not reproduce the live game's heap fragmentation.

CPU-output compatibility still has its regional raster and bounded ambient/unit
sample preparation. Those serve explicit CPU delivery and unsupported scene
extents, not a second production GPU camera-image route. Old optional ABI exports
remain only for older bridges and recorded calls. Retire them with that ABI,
not by breaking old recordings silently.

## Attribution and rejected optimization

The existing fullscreen recorded prefix executes no neighborhood/prepared-image
jobs. Their removal cannot explain its camera stalls or be credited with a
fullscreen speedup. Initial work compiles 1,511 records and about 229 MiB of
prepared GPU payload; the instrumented world join is roughly 4.45 s inside the
roughly 7.6 s native map boundary. Later requests can spend hundreds of milliseconds
in preparation/composition despite extensive content reuse. Worker totals overlap
caller totals and must not be added.

An initial candidate separated visibility from static surface identity. It matched
all 518 recorded frame fingerprints but failed the smaller-viewport independent
fog/reveal cold oracle. Frozen content participates in the retained surface, so
that relaxation was withdrawn. Full static invalidation remains conservative.
This is why successful-prefix parity alone does not certify general correctness.

## Evidence and remaining responsibility

Candidate DLL: `caf7d555dd8b2d2911f5698b17b95ed58c603e69c70f712e4f50958fc1b5c96a`.
Baseline DLL: `54aac95f5e1174aaf3c9b8f034a16c3c786fad8258a4e336c8bcce75e14558d4`.
Two fresh baseline runs followed by two candidate runs use the same incomplete
Standard 5,000-tile, 2240×1260 gameplay prefix, with no other GPU test running.
Each executes 253,272 calls, 569 actual presentations, 345 successful ambient
offers and 34 map boundaries. The separate forensic run preserves all 518
recorded frame fingerprints exactly; the killed tail remains unqualified.

| Measured quantity | Baseline runs | Candidate runs |
| --- | --- | --- |
| Replay service envelope | 34.41 / 35.50 s | 35.58 / 33.96 s |
| Map prepare median | 15.23 / 15.63 ms | 12.19 / 12.20 ms |
| Map prepare p95 | 352.59 / 354.03 ms | 332.69 / 321.95 ms |
| Cold maximum map prepare | 7.72 / 8.00 s | 7.93 / 7.70 s |
| Ambient offer p95 | 35.44 / 35.28 ms | 29.27 / 27.50 ms |
| Peak sampled private memory | 2,999 / 2,953 MiB | 2,982 / 2,923 MiB |
| Minimum sampled free VA | 540 / 545 MiB | 602 / 599 MiB |
| Minimum sampled largest free region | 400 / 419 MiB | 423 / 458 MiB |

The observed camera/ambient tails improve modestly; total workload time is
essentially unchanged. No overall FPS gain or cold-start improvement is claimed.
Contiguous headroom remains below the documented 512 MiB objective. Replay's
serial consumption and reconstructed inputs do not reproduce Civ III's heaps,
thread overlap or actual input-to-display latency. Native CPU access and cold
content preparation remain the next measurable responsibilities, not another
manual recording prerequisite or a justification for removing ownership waits.

Evidence: `native/build/camera-consolidation-validated/` (build receipt, comparison,
518-frame forensic replay, contracts and injected smoke),
`camera-consolidation-measure-{before,after}/receipt.json`, and
`camera-consolidation-reveal/` (corrected independent frozen/reveal oracle).
The 42 selected contract tests pass, including the Windows backing test rerun
with VM permission. Tests retain real cancellation, coherent ownership, native
zoom/centering, wrapping, world capacity, eviction/lifecycle and memory admission;
obsolete prepared-image tests are removed. The approved injected compilation
passes. Snapshot capacity tests include 5,000, 12,800 and 55,112 actual tiles;
that is input-record coverage, not a large-world GPU performance claim.

`native/build/camera-consolidation-native/receipt.json` passes at 2240×1260
with eight mixed units and 1 GiB reserved VA. This includes actual JGL wrappers,
CPU/native UI barriers, automatic movement and ambient continuity, independent
frozen/reveal oracles (zero reveal builds/uploads), exact GPU output/ownership,
navigation, pending/ready cancellation, reset/recreation and config-off. All
water effects are on. This is additional fixture coverage, not a paired live-FPS
measurement. Its 30-second idle/action/camera p95 is 21.08/24.20/275.18 ms;
minimum sampled free VA is 387 MiB and contiguous space 244 MiB, still below the
512 MiB headroom objective. The smaller 640×480 fixture covered the retired padded-view extent;
its independent reveal failure was corrected and rerun separately.

**Delivery:** the candidate is built and tested but not staged. The previously
qualified `54aac95f…` DLL and matching short-capture receipt remain installed in
`Renderer/bin/`; no game or installer was launched. A later staging checkpoint
must requalify short capture for this candidate before asking for live evidence,
and rerun INSTALL for the removed injected requests. Another recording on the
old staged DLL is not required to continue the remaining performance work.

Generated BMPs and bulky measurement JSONL are losslessly gzip-compressed and
verified by SHA-256; intermediate objects are removed. The original gameplay
inputs, candidate/baseline DLLs, receipts and final build objects are preserved.
`camera-consolidation-validated/storage.json` records the cleanup.
