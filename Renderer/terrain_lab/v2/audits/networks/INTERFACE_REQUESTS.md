# Q5 interface witnesses

Q5 continues with explicit local proxies; these requests are convergence work.

- **Q0: vertex constants.** `before/report.json` and `curve-01/report.json`
  rendered clear-only images with a vertex-stage `Frame` b0. Metal binds the
  argument buffer only with `setFragmentBuffer` (backend line 351 when observed).
  Q5 `module.cpp` now projects CPU vertices, which avoids blocking local work.
  Bind reflected vertex arguments/resources before supporting vertex transforms.
- **Q0: composition.** `runner.fixture` rejects modules whose shader hashes
  differ. Q5's `module.json`, Q2's `base.module.json`, and Q3's
  `static.module.json` have different vertex/material layouts and shaders.
  Publish per-draw shader identity plus shared camera/depth units. A single
  copied shader is not a correct fix. Current semantic `interfaces_v1.h` is a
  proposal, not a callable composition surface.
- **Q0: real-map registry.** `shared/real_map/` was absent at first inspection;
  Q0 status records verified source/region registry pending. Need test.biq hash,
  raw coordinate/extent/wrap/halo metadata, replay adapter and named flat/relief/
  crossing regions plus a neighbor. Historical Ancient Treasures is insufficient.
- **Q2/Q3/Q4: final world surface.** Q2 `Surface.sample` returns base height 2.5
  in its own basis; Q3 `Field.sample` provides shore/river carving in tile units.
  Q5 needs the final composed world-pixel height and normals at arbitrary ribbon
  vertices, including relief and the region halo. Q5 accepts a caller callback;
  the owned analytic witness must be replaced, not added to the final ground.
- **Q3: crossing semantics.** `Field.crossings` tangent is the *water* tangent;
  Q5's authored bridge uses the route travel tangent. Preserve Q3 point/edge ID,
  convert the width to distance along route using the crossing angle, and provide
  bank/deck height from the final surface. Multiple crossings on one route edge
  require ordered spans; do not silently choose the first.
- **Q6/Q0: final color.** Q5 opaque diagnostic shader uses shared evaluator and
  explicit provisional display encoding. Adopt Q6 scene-linear premultiplied
  output and Q0 final transfer when composition supports them. Opaque Q5 produces
  no emissives, animation, bloom, or independent clock.

## Latest disposition

- Verified source acquisition is **resolved**. Four Q5-local real-map exports
  retain the registry source/terrain hashes. The owned Q3 adapter now passes
  registry `wrap.x` explicitly; it exports 30/58/4/0 land edges in the visible
  crop plus one-tile route halo and seven exact holdout crossing anchors.
- Q0 now supports per-draw shaders **for scene-linear modules**. Q5 publishes
  `systems/networks/linear.module.json` and successfully renders through that
  branch, including independent validity output. Q2 `base.module.json` and Q3
  `static.module.json` are still legacy display modules at this witness revision.
  The reproducible `fixtures/networks/compose-witness/fixture.json` combining
  those with Q5 linear still fails validation. Each owner must publish compatible
  modules and coordinate/projection/depth conventions; Q5 cannot change them.
- Q0's frozen `surface_query.py` is now available. It is useful baseline evidence,
  not the missing final composed Q2/Q3/Q4 terrain surface. Q5 keeps its callable
  height sampler boundary and exposes `fit_crossing_grade` for the final sampler.
  No derived final relief/river height or real Q5 bridge acceptance is claimed.
- Shared corridor plumbing is **adopted**. Q5's
  `fixtures/networks/source-linear/corridors.json` validates under
  `c3x.lab_v2.corridors.v1`; its fixture pins hash, owner and schema. Coordinate
  space `civ3_raw_delta_pixels_v1` agrees with Q3. The current envelope is a
  synthetic-context witness, `halo_complete=false`; it must not masquerade as a
  final real-map placement constraint. `exchange.py` is the reusable publisher.
- Actual source-tree/city composition and clearance remain Q4/Q7/Q8 convergence
  work. Q5 supplies occupied/clearance bounds, stable IDs and footprint queries;
  it does not edit or reposition their source instances.
- Storage incident: 119,837,552 bytes of Q5-owned exact repeat BMPs were reclaimed
  only after byte comparison with retained originals. `repeat_receipts.json`
  preserves their hashes and identities. No shared cache/source/other-owner
  output was deleted. Heavy promotion was not attempted during the incident.
