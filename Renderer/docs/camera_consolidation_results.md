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

Final measurements and qualification are recorded below when complete.
