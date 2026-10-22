# Spring on Renderer64 (Lab study)

Goal: make C3X's spring season (frame season 3) beautiful before anything
reaches Renderer64. The target is the concept
`Renderer/lab/out/seasons/scene-concepts/spring-vibrant.png` (an image edit of
the `test.biq` view at camera 20,75; the earlier seasons gallery pairs it with
spring). The earlier Mac Metal spring in `../seasons/` was a start: fresher turf
and faint blossom clusters on a diagnostic scene. This study renders through
the **current production renderer** instead, with the method of the autumn
study (`../autumn/`).

## How it works

- `study.py render LABEL --look LOOK [--views ...] [--trace]` renders the
  concept view (test.biq, camera 20,75, tile 128, noon) with the current
  candidate DLL on the Windows VM. Other views: `close` (tile 256), `evening`
  (18:00), `plains`, `village`/`village-night` (ancient cities, roads, farms,
  a hill mine), `salamanca*` (1498 AD save window). The preview accepts tiles
  up to 256 px (2x), so Renderer64's 3x zoom is checked with `LOOK-3x` looks
  (`at_3x` in `spring.py`): the `close` view with blooms and their spacing
  scaled 1.5x, which covers exactly the pixels a bloom covers at 3x. `--trace` turns on the
  renderer trace, the only place shader compile errors are reported.
- A look is a **private shader root**: a copy of the flattened production
  shaders with this study's patches (`spring.py`) applied, selected with
  `C3X_RENDERER_SHADER_SOURCE_ROOT`. Production shaders, packs and staged
  binaries are never touched. Every patch is anchored and fails loudly if
  production drifts.
- `spring_common.hlsl` holds the colour and flower work.
- `study.py measure A B` prints OKLab lightness/chroma/hue per concept region
  against the target; `study.py sheet A B --target [--crop l t r b]` builds
  labelled comparisons (both need Pillow; NumPy for `measure`).
- `summer` is the unpatched season-0 render of the same DLL.
- The harness is adapted from the autumn study's; the two share the VM and
  wait for each other's previews.

## Layers

| Layer | Where | Change |
| --- | --- | --- |
| Turf | relief, mountain-tile and coast shaders | greens h≈111→117 and richer (concept grass C≈.11–.12); broad world-space lush/sunny patches; straw keeps its hue so plains stay distinct; pale sand untouched |
| Painterly turf | relief shader | the source grass/plains blade pattern amplified against a mid mip (blade-scale only), lighter blades a little yellower |
| Meadow | relief shader, open ground | flower drifts (see below), **only on grassland, plains and hills** (user, 2026-10-21): grassland full density, plains lighter, hill turf partly, never exposed hill rock |
| Hills | relief shader | hill turf lighter, a little richer, pulled toward the concept's sage/yellow-green (h≈107) |
| Crowns | object shader | one rich mid-green family per tree (narrow variety); sunlit tops yellower, shaded sides bluer; ellipsoid crown light and soft leaf speckle from the autumn study; backlit glow |
| Pines | object shader | unchanged |
| Water | coast/water shader | deep water a cleaner, deeper blue (concept L .42 C .055 h 248); rivers at 60%; fades back at night |

### Flowers

Flowers are world-anchored marks composed into the ground albedo before
lighting, so they take the same sun, shadows, evening and night as the turf.

- **Drifts**: a stretched fbm field in world tiles (drifts lie along the screen
  horizontal, as the concept's do). A threshold sets coverage; the fringe thins
  raggedly; small clumps gather blooms inside a drift; rare lone blooms dot the
  open turf.
- **Palettes**: each drift takes one palette from the nearest jittered feature
  point (even shares): buttercup yellow, daisy white, clover pink, cornflower
  blue and lilac, each with a companion colour; a few blooms stray. The concept's
  flower pixels split roughly yellow 38%, white 30%, pink 12%, lilac 12%,
  blue 8%.
- **Blooms** (from v15, `SPRING_PETAL_MODE 2`): upright and camera-facing.
  Painted flat on the ground, the isometric view squashed each bloom 2:1 into
  a soft oval. The bloom grids (two, the second rotated 37°) live in
  screen-proportional units, where one unit is one pixel along either screen
  axis, so blooms are round and the edge is an exact fraction of a pixel
  (`SPRING_AA` .35). Scalloped petals (`SPRING_NOTCH`), each its own shade,
  lighter tips, a shaded rim once a bloom is a few pixels wide, a warm eye and
  a hard contact shadow away from the sun. A bloom is never drawn thinner than
  a pixel; below that its coverage carries its energy, so it does not sparkle.
- **Placement** (from v16): a bloom's centre may sit anywhere in its cell; each
  pixel looks at the 2x2 cells around its nearest cell corner, which hold
  every bloom that can reach it. Both grids are tilted off the screen axes,
  about 45% of blooms get a sibling beside them, and fine clump noise thins
  some patches, so blooms gather in little groups with gaps instead of rows.
  This costs up to eight cell evaluations per pixel; production should
  prepare the drift and palette fields rather than recompute them per cell.
- **Ground** (from v17, `SPRING_SLOPE`): blooms are part of the terrain
  surface, so they sit at its height, are hidden behind trees and hills and
  take cast shadows. Their screen-proportional grid also folds in height
  (0.41 tile widths of screen lift per world z, 0.58 grid units), so they stay
  round and evenly spaced on slopes; density is scaled by the ground area under
  each pixel, so blooms per tile stay constant on camera-facing and averted
  slopes. Petals are lit with the slope normal bent toward up
  (`SPRING_BLOOM_UP`), without the grass micro relief or sheen.
  `slope-probe-flat` / `slope-probe-height` draw an unjittered field of large
  blooms to show the difference on a hill.
- White petals are cream: neutral white picks up the blue skylight and reads
  lavender.

### Where flowers grow

Only grassland, plains and hills (user rule). In the relief pass that means the
grass/plains biome weights, scaled down on hill turf and to zero on exposed hill
rock; tundra and desert weights get none. Forest and jungle tiles keep their base
grass/plains ground under the forest-floor decal, which covers their blooms;
marsh fades the relief out; floodplains sit on desert; mountains and coasts use
other shaders without flowers. Looks before v9 also had tundra blooms and
woodland-edge flowers (`spring_woodland`, now off).

## Lab-only stand-ins that production must replace

- **Broadleaf detection** uses the leaf atlas's average colour (1×1 mip), as in
  the autumn study; production should carry a `seasonal_foliage` flag in pack
  metadata.
- **Season selection**: the private shader root *is* the spring variant.
  Production needs the frame season in a constant buffer.
- **Map wrap**: the meadow uses raw world coordinates, so drifts would seam at
  the map's wrap edge. Production must fold them into canonical wrapped
  coordinates (or make the drift period divide the map width).
- **Water**: in the game, open water is drawn by `PSWaterSurface`
  (`sandbox/water_surface.hlsl`), which tints by the Coast/Sea/Ocean family;
  the Lab preview's `q3_water_material` path has no such tint, so Lab water
  measurements mislead. The port applies the deep-water change there as a
  relative recolour (hue rotation, chroma and lightness scale after the family
  tint), not a pull to a fixed hue. Rivers still go through `q6_raw_main`.
- **Forest margins**: the concept gathers drifts at forest edges; the relief
  pass has no forest field, so placement here is biome-only. Production could
  pass an explicit terrain mask so flowers never depend on the floor decal
  covering forest interiors.

## Looks

Looks v1–v28 are in `spring.py`; the shared HLSL has moved since v1, so older
looks re-render close to, not exactly as, their recorded images.

- v1: turf, flowers, crowns, water. Flowers too solid and pastel; whole crowns
  varied yellow to teal.
- v2: lone blooms and diffuse drifts read as confetti; turf contrast against a
  very low mip made yellow blotches.
- v3: firm streaky drifts, blade-scale turf, one crown family. Close up the
  crowns read lime; whites read lavender.
- v4: deeper crowns, cream whites, lemon yellows, blooms clumped on tufts.
- v5: bigger, richer blooms, woodland edge flowers, stronger strokes; crowns
  measured too dark and blue, blue drifts too common.
- v6: lighter crowns match the concept; a hashed palette pick left the view
  46% pink.
- v7: low-discrepancy (R2) palette pick; drifts still pastel patches.
- v8 (not rendered) / v9: smaller, looser, mixed clusters; user rule applied
  (grassland, plains, hills only); strokes at 1.8 bleached the turf.
- v10: user preferred small, sharp blooms: 0.016–0.032 tiles, crisp edges
  (`SPRING_AA` .5), lighter tufts.
- v11: strokes back to v7 strength; yellow crowded out blue.
- v12: balanced shares, density .75 at the same bloom size. `-blossom`
  variants add white/pink bloom to 12% of broadleaf crowns.
- v13: user loved the blossoms (now in the look) and asked for crisper,
  less fluffy blooms at 3x. Notched petals, hard eye and contact shadow; found
  that the outline had always been a disc (`max(petals, shown)`), fixed.
- v14: still smudgy at 3x because blooms lay flat (7x5 px ovals) and the edge
  softness was sized from a footprint about twice a pixel: upright,
  screen-proportional blooms with exact pixel footprints.
- v15: shallower scallop, shaded petal rim, third-pixel edges. Blooms measure
  round on screen (width/height 1.00, v12 1.39).
- v16: user saw evenly spaced, grid-like blooms (v14 had cut jitter to .30 on
  a screen-aligned grid; nearest-neighbour directions peaked at 2.3x their
  mean). Free placement with neighbour search, tilted grids, siblings and finer
  clumps: the peak is 1.3x (random scatter), petal coverage unchanged.
- v17: user asked that flowers follow ground depth and angle: height-aware
  bloom grid, slope density, upright petal lighting.
- v18: aiming at the concept's crisper flowers: short dark stems, top-lit
  heads, coral, richer petals, tighter bunches.
- v19: bloom centres snapped to pixel centres (`spring_snap`; the upright grid
  is exactly screen-proportional), so small blooms fill whole pixels.
- v20: the concept's flowers read crisp because each is an isolated 1-2 px
  head with grass around it; v19's touched and merged. Smaller heads, siblings
  apart, gentler clumping, lower density over slightly larger drifts: 3.6%
  petal pixels (concept 3.8%, v19 4.5%).
  The concept's grass is softer at pixel scale than ours (fine contrast .053
  vs .070, broad strokes equal at .080) because the image model repainted it;
  the user chose to keep our grass as it is.
- v21–v25: user asked for the concept's more clumped patches. Measured on
  grass blocks of the concept view (scratch metric: share of bare blocks,
  share of petals in the densest fifth, petal blob sizes): concept 71% bare at
  12 px, blobs median 3 px / p90 16; v20 54%, 2 / 9. v21/v22: higher drift
  threshold, sharper and smaller drifts. v23: posies (`SPRING_BUNCH`, up to
  five touching heads on a golden-angle spiral, 3x3 cell search). v24: tighter
  posies, fewer patches; 71% bare, matching. v25: patches as long thin strips
  (`SPRING_DRIFT_STRETCH` 2.6), plains density .8.
- v26–v28: user: "clumps" meant the density and positioning of flowers, and
  v25's posies merged into fuzzy blobs. Back to separate dots, packed densely
  inside the strip patches. v26: tiny sparse dots, too faint. v27: exact
  pixel shapes (`SPRING_PIXEL_SIZES`: below 1.4 px a bloom is one solid pixel
  or a five-pixel plus with a warm centre; radii of .6-1.2 px had spilled a
  soft halo into neighbour pixels), finer grid, denser, stronger colour.
  v28: on plains most drifts take pink, blue or lilac palettes
  (`SPRING_STRAW_COLOR`), since white and yellow vanish against straw.

The current look is **v28** (blossoms included); the user confirmed its direction.
Against the concept view it measures turf L .69/C .11/h 116–118 (target .67–.69/.11–.12/116–118),
crowns L .52 h 120 (target .535/121), deep water L .425 C .056 h 246 (target .42/.055/248),
4.2% petal pixels on open grass (target 3.8%) and 64% bare 12 px grass blocks (target 71%).
At the 3x stand-in blooms measure round on screen (width/height 1.00).

The review page (summer, target and v15 across concept, close, 3x, village,
18:00 and night, with a 3x pixel loupe) is a private claude.ai artifact.
Outputs live under `Renderer/lab/out/spring/` (ignored, disposable); superseded
looks were deleted after review, since `spring.py` can re-render them.
