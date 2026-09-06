# Q3 static hydrology candidate interface v1

Provisional, source-independent implementation: `systems/hydrology/field.h`
(relative to Lab v2). This is an owned candidate proposal, not an edit to Q0's
shared contracts. It consumes exact tile-center lattice coordinates, base/real
terrain and physical river flags. Civ III hills are real terrain **5**;
coast/sea/ocean are base **11/12/13**. The CSV final header field is halo count,
not a wrap flag. Wrap policy is supplied separately; raw origin comes from the
actual `(0,0)` tile. A two-tile halo is mandatory; missing neighbors fail.

Q4 consumes `Field::sample(P)` and `Field::coast`. Shore distance is positive on
land, negative offshore, in tile units. `rocky` is a continuous 0..1 fraction
of water-facing hill ownership. Beach width tends to zero on fully rocky edges.
Every seam segment supplies endpoints at height zero and a rocky fraction.
Q4 owns raised rock geometry and hill normals; Q3's smooth raised bank is an
explicit proxy only. No rock body or cliff equivalence is claimed. Q4 should
join its face foot to that same zero contour without adding a sand underlay.
Mixed hill/lowland corners blend the class instead of selecting a water family.

River curves are canonical physical shared edges with deduplicated reciprocal
bits 2/32 and 8/128. IDs hash ordered raw-coordinate corner pairs with canonical
horizontal wrap. Curves use shared tangents at degree-two nodes and an asymmetric
bow that vanishes with zero derivative at both ends. `sample()` supplies signed
river corridor distance, width, and a carved base height. A relief consumer must
fade its additional relief height by `smooth(.025, .80, river_distance-width/2)`;
the wide slope avoids a raised proxy bank occluding the channel at Civ III pitch.
Do not infer flow direction or waterfalls from these undirected edges.

Q5 calls `Field::crossings(route_a, route_b)` for exact segment/polyline
intersections. Each returned anchor includes stable ID, hydrology edge ID,
position, river tangent, water width and bed height. Width is full world-lattice
width, not screen pixels. Q5 owns road/rail bridge geometry and its grade. The
export utility publishes the same curve samples and center-to-center crossing
witnesses as JSON; it creates no bridges.

The standalone material stack uses normalized sand, shallow-bed color/height,
and rock materials, with Q6 response v1 and the shared environment evaluator.
There is no phase/time-dependent water geometry, normal animation, surf texture,
wave ribbon, caustic or redraw request. Hour changes lighting only.

Pending composition interface: Q0 currently rejects different shader modules in
one fixture. Reproducer: combine `systems/hydrology/static.module.json` with
`systems/lighting/proxy.module.json`; the runner reports mixed shader contracts.
Request per-draw shader identity plus shared scene-linear attachments so Q2/Q4/Q5
packets can compose before Q6 display encoding. The isolated hydrology shader
currently encodes its complete local opaque composite once, solely for the
frozen BGRA8 runner. It must not be stacked after another owner's tone map.

Source fidelity limitation: terrain/relief/vegetation here are receiver proxies;
only hydrology assets are evaluated. Final beauty requires the actual owner
candidates and shared display pass, not approval of this proxy terrain.

Placement clearance extension (same additive proposal v1): `Field::exclusions`
returns the union of capsules swept along the **same rendered river polyline**.
Each capsule has `edge_id`, endpoints `a/b`, `water_radius = width/2 + .009`,
`bank_radius = width/2 + .043`, and `clearance_radius = bank_radius + margin`
(default margin `.04`). These include material antialias support and the full
bank tint support. Shared endpoints naturally union at branches and mouths.
Coordinates and radii are local tile-center world units; raw-coordinate lift and
wrap metadata come from the exact same CSV/registry. Exclusions are footprint
constraints regardless of component altitude; consumers conform height using
Q2/Q4's surface. They must not use a low river-height bound to permit a building
or canopy overhang to cover the waterway.

`Field::intersects_footprint(convex_polygon, margin)` tests actual transformed
footprint/overhang extents, segment crossings and capsule distances with a
`1e-9` tolerance. Polygon vertices must be a convex cyclic boundary. Numeric
witnesses include an outside origin whose extent intrudes, a clear polygon,
and a junction-covering polygon. JSON `exclusion_capsules` is exported alongside
river curves and exact crossing witnesses. Consumers retain city anchors and
river topology and reserve these envelopes before selecting source parts.
Q3 does not relocate city buildings or source trees. Mode 4 renders an explicit
red exclusion overlay; it is a diagnostic mask, never source art or a draw-order
workaround. Composed building/tree-footprint witnesses remain Q4/Q7/Q8 work.

Shared Q0 sidecars are now available via `systems/hydrology/publish_corridors.py`
and `fixtures/hydrology/rivers.corridors.json` / `real-mouth/corridors.json`.
They validate as `c3x.lab_v2.corridors.v1`. The exchanged coordinate convention is
`civ3_raw_delta_pixels_v1`: X=64*(column+row), Y=64*(column-row),
Z=64*local_height, anchored at source tile `(0,0)`. XY radii use 64*sqrt(2).
Horizontal wrapping is the single vector `[64*map_width, 0]`, not independent
wrapping of the original two lattice axes. The extra crossing witness field is
explicitly named `crossings_local_tile_lattice` and retains its stated local units.
Shared envelopes are conservative convex hulls of circumscribed 16-gon endpoint
caps. Tests prove no capsule undercoverage; the radial overcoverage is bounded
by `sec(pi/16)-1` (under 2%). Each sidecar pins terrain and actual river-geometry
hashes. Geometry identity contains no camera or time. The JSON shared sidecar and
C++ analytic capsule APIs represent the same river/bank occupancy.

The final mixed-edge correction samples rockiness continuously at the nearest
contour foot. `coast` segment midpoint classes are summaries; consumers may use
`shore_rockiness(P)` / `sample(P)` directly. Exported segment endpoint values
`rocky_a` and `rocky_b` permit continuous interpolation for source-cliff joins.

Q0 now publishes `HydrologyHooksV1` in `shared/scene_hooks.h`. The opt-in
`scene_adapter.h` initializes from the exact CSV, converts frozen corner lattice
coordinates to tile centers (-.5 on both axes), and returns water-positive
clamped signed distance normalized by .65. Its optimized nearest-contour path
exactly matches `Field::sample` without river/material work. The adapter is
configured for the verified horizontally wrapped dataset; a caller using another
map must supply that map's wrap policy before rebuilding the field. A null hook
retains frozen v1. Q2 consumes this through its owned thin header wrapper;
`COMPOSED_REGRESSION.json` proves the isolated-basin support fix with source art.
This compatibility callback does not replace the full material/class API above.
