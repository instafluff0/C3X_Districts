# Cache and performance maintenance

This round preserves the approved `city-fidelity` baseline appearance. The
installed DLL and approved images remain unchanged; the candidate is tested
through the category dispatcher without launching Civ III.

Changes:

- Cache authored hill placements for the nine sample cells surrounding each
  tile. Finite-difference samples still evaluate the exact height function;
  integer neighborhood lookup and placement construction happen once per cell.
  Scratch has a fixed bound and dies with its dependency owner. Distant object
  samples retain the ordinary query path.
- Reuse the exact absence of relief/dunes within an integer sample cell when
  the point is beyond the cliff shoulder. The full contributing neighborhood
  remains observed for edit invalidation.
- Hash immutable asset payloads with the public-domain MurmurHash3 x86_128
  algorithm, then fold the digest into the existing 64-bit content revision.
  This avoids a 64-bit multiply for every source byte in the 32-bit process.
  This key is for runtime cache invalidation; provenance remains SHA-256.
  [Upstream algorithm and attribution](https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp).
- Add ordinary debug-output timings for device/terrain, natural assets, city
  assets, reflections and post-processing, including separate read/hash/upload
  costs. File logging remains opt-in for headless diagnosis.

Verification results are recorded in `performance-checkpoint.json`. Repeatable
raw runs live in ignored `Renderer/lab/out/cache-performance/final`; they compare
the approved and candidate DLLs with identical assets and alternate run order.
Timing variability is expected in the shared VM; report medians and exact pixel
comparisons separately. Startup measurements do not prove vanilla-speed play.

Existing production edit-reuse failure remains exposed: introducing the first
coast into the dry test world invalidates broad nearest-coast dependencies and
rebuilds all visible tiles. No dependency or pixel threshold is bypassed here.
Scrolling, wrapping, animation and approved-view comparisons remain required.

The rejected floating-point height-result cache was removed: too few coordinates
repeat to justify its overhead. A SHA-256 runtime-hash trial was also removed
after profiling its cost in the x86 VM. Neither trial is part of this candidate.
