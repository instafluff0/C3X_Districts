# Q5 networks: candidate, not accepted

The local source-route candidate is ready for cross-owner convergence. Final
acceptance is blocked by incompatible composed owner modules and the missing
final shared surface/placement witness. Do not promote this proxy context into
Game Integration. Historical L14/L15/L21 and all v1 files remain unchanged.

## Primary contextual before/after

All paths below are relative to `Renderer/terrain_lab/v2/`.

- Before source UV correction: `audits/networks/out/Q1/Q5-networks/source-06/h12-z1-pan00.png`.
- After source UV correction, identical geometry/camera/materials:
  `audits/networks/out/Q1/Q5-networks/source-07/h12-z1-pan00.png`.
- Chosen source route/bridge geometry with fitted grade and source-covered rail
  width: `fixtures/networks/source-10/scene.bin`.
- Four-phase/two-zoom/scroll checkpoint:
  `audits/networks/out/Q1/Q5-networks/final-10/`.
- New scene-linear branch: `audits/networks/out/Q1/Q5-networks/linear-14/`.
  Q5 writes linear premultiplied color and independent validity; Q0 applies
  exposure, reconstruction and output transfer. It does not clear category depth.

The 640x384 contextual crop is pinned to the normal 128x64 Civ III tile basis,
with a 320x192 reduced output, three diagnostic settlement markers, nearby worked
plots, a rail corridor, road loop, branches and crossings. These markers, ground,
relief and water are **diagnostic_proxy**. They are not source cities or verified
terrain and do not satisfy gameplay beauty/clearance acceptance. The dense
43-edge all-direction fixture remains a diagnostic, not the beauty target.

## Direct visual iterations and corrections

1. The first packet returned clear-only frames: Q0 did not bind vertex constants.
   Q5 moved the pixel projection into its owned CPU packet adapter. No backend
   was copied or modified.
2. Inspected normal/reduced contextual images showed angular corners and a
   ladder-like wide rail. Degree-two quarter turns failed the tangent test.
   Corrected exact opposite tangents while preserving every node and edge.
3. Inspected authored atlas data directly (`rail-atlas.png`). The inherited
   formula sampled diagonally through rectangular pieces and used the short
   steel end-piece row. Corrected rectangle UV mapping and sampled both full
   steel strips over the sleeper/ballast rectangle. Road material slicing was
   corrected by the same observation. This is evidence from decoded pixels and
   normalized recipes, not a claim about Civ VI's original engine implementation.
4. Retained source textures, alpha, bridge mesh proportions, normals and UVs.
   Rail quad width is 12 world pixels; the source's alpha occupies less than the
   full quad. Normal zoom now shows a continuous paired-rail read and sleepers;
   reduced zoom preserves a fine rail corridor, with individual sleepers subtle.
5. Bank-grade measurement found 18 vertices below the diagnostic ground near a
   bridge approach. `fit_crossing_grade` now samples the span and full lateral
   width, fitting a deck without moving the hydrology anchor or river. All 9,660
   selected route vertices are now at least 0.34999 pixels above ground. Higher
   offsets belong to bridge decks/approaches; ordinary roads drape to the surface.
6. The new linear branch initially wrote no validity attachment, yielding black
   output despite a successful tool exit. Added required independent validity
   MRT, directly inspected the corrected image, then ran its phase/zoom check.

Observed remaining weaknesses: the proxy ground is smooth and repetitive; the
primitive settlement markers are not final art and some existing marker extents
intersect the new clearance witness. No composed source-tree/building clearance
is claimed. Source road height BC5 exists but is not bound pending channel
semantics. Shared final shadows and source-object composition remain required.

## Numeric evidence

`metrics.json`, `no_route_control.json`, and `repeat_receipts.json` are executable
measurement results, not visual approval.

- 19 tests, including all 256 eight-neighbor masks, exact shared endpoints,
  all degree-two tangent pairs, stable input ordering, legal/wrapped directions,
  stages/pillage/coexistence, two crossings on one gameplay edge, grade failure,
  complete-footprint/crown intersections, bridge bounds, and wrap aliases.
- Old short steel row had 58.203125% opaque longitudinal coverage; both corrected
  full-length source rail rows have 100%.
- UV correction changed 14,858 of 245,760 contextual pixels, mean absolute
  channel change 0.44742. This measures the change, not its desirability.
- No-route controls are byte-identical in all eight phase/zoom cases.
- 32 final diagnostic variants include four phases, two zooms, offsets (0,0),
  (8,4), (-8,-4), (0,0), and an independent repeat. All eight return-to-origin
  pairs are byte-identical. Original images and reports are preserved.
- Source state sheet and source-only isolation were rendered and inspected.
  State rows are ancient/medieval/industrial/modern/rail; columns normal/pillaged.

## Verified real terrain

Source `test.biq`: SHA-256
`a6a88d7fffcc567c3500bbd5aa947398dd48170d4f412aa1e518bb45ffe8453e`,
17,173 bytes, 100x100 raw coordinates, 5,000 tiles, horizontal wrap only.
Q0 registry and parser hashes are retained in each fixture and consumed-contract
record. Source CSV bytes and two-tile halos remain unchanged.

| Region | Raw origin | Result |
| --- | --- | --- |
| mixed | (14,42) | Q0 frozen source terrain + explicit augmentation rendered at all phases/zooms; 30 Q5 land edges including one-tile route halo, no crossings |
| mixed-holdout | (18,46) | Fixed holdout rendered; 58 halo-aware Q5 land edges and 7 exact Q3 crossing anchors |
| wrap | (94,50) | Source mostly water; 4 halo-aware land edges, no crossings |
| wrap-holdout | (98,54) | Source ocean, no legal route edges; absence recorded, not filled with fake land |

These are **reference renders using Q0's frozen provider**, not acceptance of
Q5 composed geometry. Direct inspection showed source forests/jungle obscuring
parts of the legacy route layer, illustrating the unresolved placement gate.
The holdout's river is at the visible crop edge; visible-only augmentation
produced zero crossings, while Q5's separately labeled halo augmentation finds
seven. Preserve these coordinates; do not replace the holdout to improve scores.
`q5-crossings.json` keeps exact hydrology IDs, route/water tangent distinction,
angle-adjusted span width, and deliberate null deck heights until final surface
conversion. The adapter explicitly receives registry horizontal wrap.

## Placement exchange

- Local final-curve capsule contract: `systems/networks/clearance.py`.
- Shared publisher: `systems/networks/exchange.py`.
- Published sidecar: `fixtures/networks/source-linear/corridors.json`.
- Schema: `c3x.lab_v2.corridors.v1`; coordinate space:
  `civ3_raw_delta_pixels_v1`; occupied width includes roads, rails, junctions,
  bridges, approaches; additional margin 4 pixels; stable geometry hashes and
  height bounds included. Shared validation passes.
- `clearance-overlay` directly displays yellow corridors and red/green full
  footprint test results. It is a diagnostic, not a draw-order solution.
- Synthetic envelope has `halo_complete=false`. It is not sufficient for final
  real-map placement. Q4/Q7 consume read-only and reserve source-instance
  footprints/crowns; Q3 publishes matching river/bank exclusions.

## Reproducible blocker and remaining gates

Run:

```sh
python3 Renderer/terrain_lab/v2/app/runner.py validate --fixture Renderer/terrain_lab/v2/fixtures/networks/compose-witness/fixture.json
```

It combines Q2 base, Q3 static hydrology and Q5 linear. At this audit revision it
fails with `mixed shader contracts require a versioned per-draw shader interface`.
Q0 has added per-draw support for linear modules; the available Q2/Q3 modules
still publish the legacy display branch. They need compatible owner revisions,
a common authoritative projection/depth convention and final terrain sampler.
Q5 must not edit those modules or merely relabel display-encoded shaders linear.
Q0's frozen surface query is available as baseline evidence; it does not supply
the final combined Q2/Q3/Q4 surface contract.

Pending: actual Q5 geometry over verified primary/holdout terrain; source-tree
and source-building corridor clearance; shared final lighting/shadows; combined
192-tile/four-phase suite, Windows parity, full project closure and approval.
The package is not closed, so global `ROADMAP.md`/`project_status.json` and I#
gates were not advanced. No injected C changed; no injected compile was run.

Storage briefly reached zero available space. Q5 reclaimed 119,837,552 bytes of
verified duplicate repeat BMPs, retaining originals, hashes and replay recipes.
No source assets, shared cache or other-owner outputs were removed. Large runs
paused; bounded linear checks resumed after space returned. Intermediate local
source images/meshes are ignored, not redistributed.

All changes are inside Q5's owned directories and own status file. Portable
text was scanned for personal home paths, email addresses and credentials.
No personal/sensitive data was added. An attempted coordinating-task message
was rejected by automatic approval review because destination trust was not
established; the interface findings remain in this owned audit instead.
