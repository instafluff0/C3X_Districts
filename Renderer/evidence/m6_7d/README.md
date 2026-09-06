# M6.7d Coastline and Water Evidence

> **Visual verdict: failed.** This folder records a technically stable prototype,
> not an accepted art result. Direct comparison with the Civ VI reference shows
> repetitive scalloped coastlines, a narrow glowing rim, flat water, inadequate
> beach/cliff mass, and primitive triangular rock clutter.

## Implemented layers

- M6.7d1: one adjacency-derived scalar contour shared by land and water,
  deterministic world/wrap-stable displacement, and convex/neutral/concave
  topology classification.
- M6.7d2: source-independent authored beach, cliff, and cold-cliff materials
  with relief-aware beach width and cliff selection.
- M6.7d3: coast/sea/ocean depth color, restrained shallows, narrow
  contour-following foam and haze, and Fresnel response. The rejected periodic
  whitecap/glint experiment is not present in the accepted shader.
- M6.7d4: canonical shared-edge-seeded low-poly rock clusters on rugged
  shoreline profiles, with sparse flat-shore dressing.

## Technical diagnostics (not visual acceptance)

- The 32-bit native build and smoke render complete successfully.
- Eighteen focused asset-compiler/native contract tests complete successfully.
- The stable test.biq replay renders 605 visible tiles at 128px and 1,543 at
  64px with zero fallback. The intermediate 96px view renders 995 tiles with
  zero fallback after viewport-aware tessellation and bounded geometry uploads.
- Two independent 128px renders are byte-identical:
  070652076F8383FE1B98C3CCD226AC156E5818A97A461A87F7D6F47F05B78E25.
- Equivalent horizontal-wrap views at centers 0 and 100 differ by 8,580 bytes
  out of 5,760,000 color bytes; maximum channel delta is 1 and mean absolute
  delta is 0.00148958. This is sub-quantization floating-point variation, not a
  visible seam or a changed edge seed.
- samples/scenes/m6_7d4_coast_profiles.csv isolates standard, desert, hill,
  mountain, tundra, coast, sea, and ocean transitions.

## Visual outputs

- preview/out/m6_7d4_coast_profiles.bmp
- preview/out/m6_7d5_close_a.bmp
- preview/out/m6_7d5_far.bmp
- preview/out/m6_7d5_wrap_0.bmp
- preview/out/m6_7d5_wrap_100.bmp

## Required rework before acceptance

- Broad, naturally tapered low-coast beaches rather than a uniform luminous rim.
- Large, grounded cliff profiles with an irregular vertical silhouette.
- Coherent multi-scale water normals and localized surf rather than a flat
  cyan-to-blue gradient.
- Authored rock/cliff instances with contact shadows rather than triangle shards.
- Close/far side-by-side reference review demonstrating comparable visual
  hierarchy. Hashes and fallback counts cannot satisfy this criterion.

## Deferred integration checkpoint

The user explicitly deferred live-game checks while away from the main
computer. M6.7d5 remains active only for the eventual batched ownership,
overlay, and live-scroll confirmation. No new patch symbol or
civ_prog_objects.csv entry is required. The full verify_project.py suite was
intentionally not run; project policy schedules it once at M6.7g.
