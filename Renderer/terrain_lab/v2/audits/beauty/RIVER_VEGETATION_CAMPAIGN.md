# River, forest and jungle visual campaign

User-directed goal established 2026-09-06. Active, not complete or accepted.
Baseline: `shadow-receiver-r1`. Preserve the separately frozen integration
preparation package and its source archive. New work must remain an opt-in,
versioned Lab candidate until composition and existing promotion gates pass.

## Required visible outcomes

- Forest and jungle stands retain their current source art, plausible scale,
  density and grounded appearance, with deterministic variation in source-piece
  order, orientation and placement. Adjacent tiles must not read as copies.
  Variation must not change with camera position, crop, traversal order or zoom.
- Rivers remain seamless across incident tiles, branches and junctions; true
  headwaters have small source-lake presentations, and outlets reach the actual
  optical sea contour without a land or beach plug.
- Banks use source-backed material/detail and read like the canonical river
  reference, with natural widths and transitions instead of a uniform outline.
- Meanders follow the valleys around hills and mountains. Do not merely drape
  water over high relief or carve arbitrary trenches through source peaks.
  Preserve authoritative Civ III river connectivity and terrain identities.
- Assess the combined result at actual gameplay size, with day/night and both
  matched zooms, the three fixed approximately 100-tile benchmarks, a relevant
  long river/coast witness and a previously untuned region. Preserve each best
  and reject regressions. Closeups diagnose; they do not replace gameplay review.

## Initial evidence and implementation traps

Canonical `Renderer/canonical/river.png` and `forest.png` were inspected directly
at the start. The river reference shows continuous curves, bank-width variation,
headwater water bodies and channels running beside relief. It is an appearance
target, not proof of source-engine geometry equations or flow metadata.

The retained scene's actual river geometry is in
`shared/frozen_scene.cpp`: `build_river_graph`, `biq_river_edge_distance`,
`biq_river_distance`, `biq_river_node_distance`, `biq_tile_height` and the river
patch construction. Its sine-based bend seed uses **local column/row** rather
than canonical source identity. Reciprocal owners agree within one crop, but
that alone does not establish crop-stable world curves. Valley lowering only
runs on a tile with river flags and retains a fraction of the raised surface;
this does not guarantee a route around an adjacent peak.

`systems/hydrology/field.h` already has a separately computed continuous river
graph with degree-two tangent treatment. However, `scene_adapter.h` labels its
`river_sample` as proposed and unbound, and
`shaders/hydrology/scene_material_v1.hlsl` explicitly consumes the frozen
analytic distance in `input.river_data.x`. Editing only the Q3 field would not
improve the currently displayed river. First connect a single authoritative
curve/corridor query to geometry, surface carving, source/bank material,
outlets, headwaters and vegetation clearance; protect coordinate conversions.

The source loader already binds river base/height/specular, river LEAN channels,
source-decal base/height, clutter-decal base/height and river-bank noise. Inspect
their actual sampled roles before adding decorative substitutes. The current
static-optics river branch uses a distance-based water/bank strip; source lake
appearance must be composed into that actual path, not an unused shader branch.

Vegetation `add_feature_group` uses a 6x6 forest / 7x7 jungle jittered grid.
Per-tile source-coordinate seeds already vary jitter, weighted selection,
rotation and source scale, so variation is not wholly absent. The first three
forest / six jungle selections are fixed named anchors in the first grid slots.
`named_feature_placement` returns null for anchors absent from the selected
broadleaf group, leaving the corresponding slots empty. A stable permutation
of layout slots and de-correlated source selection can remove repeated stand
structure while preserving the already-good source appearance. Measure counts
and inspect gaps before treating a density change as an improvement.

## Work sequence

1. Establish source-stable canopy ordering and natural stand variation as an
   isolated opt-in change. Compare forest and jungle together, including edges,
   shadows and river clearance. Keep source meshes, material channels and scale
   calibrations; do not manufacture new tree assets or randomize every frame.
2. Replace the displayed river's competing local curve calculations with one
   topology-preserving corridor query. Prove shared-edge, junction, crop and wrap
   agreement before judging river shape. Use the actual uncarved relief field
   when selecting the bounded meander, avoiding circular height/river queries.
3. Compose headwater lakes, continuous sea outlets and source-backed banks into
   that same field and material path. Distinguish real inland terminal nodes
   from clipped halo boundaries and water outlets. BIQ river flags alone do not
   prove directional flow; document any presentation-only direction inference.
4. Recompose vegetation beside the revised corridors and inspect the complete
   scene against canonical images and previous best. Expand to a fresh region,
   dawn/dusk and the remaining required coverage before any gate closure.

The setup itself changed no appearance. The first implementation loop now has
16 matched canopy frames and 16 combined river/canopy frames; see
[RIVER_VEGETATION_PASS_r2.md](RIVER_VEGETATION_PASS_r2.md). The goal remains active.
Human approval is never inferred from the agent's assessment.
