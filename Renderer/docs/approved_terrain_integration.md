# Approved Terrain Integration Contract

M6.7 established the delivery boundary; the paired I9-I13A implementation is
generated and tested against the approved `handoffs/L9_terrain.json`,
`handoffs/L10_dunes.json`, `handoffs/L11_marsh.json`, and
`handoffs/L12_volcano.json`, `handoffs/L13_rivers.json`, and
`handoffs/L13A_lighting.json` records. The game DLL
does not parse handoff approval flags or hashes at runtime. It contains only the
draw paths deliberately integrated from those handoffs and reads generic
normalized packs plus the layered renderer definition. Lab art direction remains
authoritative and no source-game format or identity enters the DLL.

The game path executes a frozen production copy of the approved Lab `PSMain`
and `PSFeature` terrain functions. It does not include the live
`terrain_lab/terrain_lab.hlsl`, which has already advanced into L14. The game-owned
`native/integrated_terrain.hlsl` file is only a vertex/input adapter from
authoritative Civ III anchors and state into that shared surface contract.
That adapter transfers exact base and visible terrain identities, computed
geometry normals, material weights, authored relief height/blend, signed shore
distance, water depth, and authoritative active-effect state. It does not
average categorical terrain IDs or synthesize replacement values. Production
submits the same ordered viewport-wide underlay, land, bed, water, river, cast-
shadow, and feature passes as the approved Lab stack. Production has a semantic
constant buffer for texel scale plus the approved shared sun, moon, ambient,
exposure, shadow, water, emissive-policy, and hour state. Standalone fixture
selectors and milestone labels are not C++ runtime state. Roads remain
compile-time absent and no route geometry is submitted.

The L9-L11 mapping is frozen in `terrain/m6_7_handoff_fidelity.json`; the L12,
L13, and L13A deltas are frozen in `terrain/i12_handoff_fidelity.json`,
`terrain/i13_handoff_fidelity.json`, and `terrain/i13a_handoff_fidelity.json`.
A game viewport cannot be byte-compared to
the differently framed standalone beauty scene. Fidelity is therefore proven
by exact reference hashes, exact asset/channel identities and lab constants,
material pixel-difference witnesses for dunes and vegetation, and deterministic
viewport invariants. Projection, clipping, sampling, and the shared day/night
environment may adapt to authoritative Civ III state. Substitute geometry,
color grading, proxy vegetation, invented cliffs, and shoreline-rock dressing
are outside the tolerance.

The cache is intentionally correctness-first and bounded to one complete
terrain viewport. Its fingerprint includes target size, zoom basis, ordered
canonical identity plus screen occurrence anchors, renderer-owned tile state,
visibility, world size and wrap, environment, definition/pack content revision,
ownership, and device generation. The dirty clip is a compositing instruction,
not terrain identity. Dirty flags never authorize reuse. Roads, resources,
cities, units, selection state, and other retained later-plane selectors are
excluded from terrain cache identity. Authoritative `river_code`, hour, and
season are included because they now change renderer-owned geometry or lighting.

The live-scale path avoids repeated CPU work without weakening that identity.
Each terrain layer reuses one evaluated shared grid per tile, common surface and
material samples are memoized within the tile, river topology is evaluated only
for the river pass, and a bounded anchor-independent cast-shadow field is reused
only when its terrain, feature, visibility, wrap, zoom, environment, content,
and device fingerprint is unchanged. Screen-anchor changes still rebuild the
viewport geometry. On the Windows native 400-rendered-tile/800-record fixture,
the cold render fell from 57.790 seconds to 5.241 seconds; a three-pixel/two-pixel
camera move completed in 1.054 seconds while proving a scene invalidation and
shadow-field reuse. Canonical synthetic and approved-scene pixel hashes remained
unchanged.

Partial Civ III redraws preserve the destination map outside the authoritative
dirty rectangle. When custom rendering is enabled, the `m71` hook deliberately
asks Civ III for the complete visible tile traversal even if a unit or UI change
supplied a smaller traversal rectangle. This guarantees that a partial redraw
can neither become a partial terrain render nor poison the retained cache. The
renderer fills the complete off-screen target, while API v7 returns the exact
original image clip and the bridge performs `SRCCOPY` only for that rectangle.
Repeated or reordered subsets of the same static terrain are also accepted as
zero-tick cache hits. Native smoke proves unchanged full-frame pixels and exact
ownership for those subsets, then exhaustively checks every destination pixel
inside and outside a nontrivial partial clip.

Every rendered tile receives explicit ownership bits for validation and later
family-by-family integration. An active custom frame owns the complete `m19` map
plane: the bridge captures and composites at the same audited boundary but never
calls or replays the original multiplexed tile renderer. Production I13A requires
zero fallback tiles. A category without an integrated draw path is simply not
called. Failure to load any configured terrain, relief, dune, marsh, volcano,
clutter, or vegetation payload rejects the custom frame atomically and visibly.
An incomplete river payload atomically omits the entire river subsystem and does
not claim river ownership; the already integrated exclusive custom terrain plane
continues and native rivers are never replayed. Structural capture, renderer/device,
validation, or blit errors remain visible and emit diagnostics. Configuration
off remains fully vanilla. Unported roads, improvements, resources, and cities
are intentionally absent until their matching I# gate.

I11 ports the approved `GrassMarsh` base-color, height, and specular channels
plus the exact L11 `CLUTTER_MARSH` projected-decal composition. Marsh identity,
adjacency, anchors, wrap occurrences, zoom, environment, pack revision, and
ownership all participate in the existing frame signature and reset/rebuild path.

I12 consumes the user-approved L12 shared stack. Volcano geometry samples the
normalized ordinary-volcano height and blend fields with the handoff's rigid
orientation, bounded aspect/footprint fit, connected mountain/volcano shoulder,
and vertical calibration rules. Dormant base/height and active base/specular
channels remain authored; captured `Tile.Body.active_tile_effect` selects the
state and participates in the frame signature. I12 also ports the approved
36/49 forest/jungle canopy density at 0.42/0.40 scale, signed-shore relief
flattening, and source-backed projected land/water clutter. It does not generate
volcano effects or range connectors and does not restore native terrain.

I13 consumes Civ III's authoritative per-tile river mask and the captured
topology halo. The frozen Lab graph defines canonical physical edges, bends,
sources, junctions, coast mouths, connected relief valleys, and horizontal-wrap
identity. Production loads the approved base/height/specular/LEAN, source,
clutter, bank-noise, and five normalized river-rock assets; successful river
tiles receive their own replacement bit. Road masks remain excluded from this
cache and no road code is present.

I13A evaluates the existing shared environment from authoritative hour and
season each frame. The frozen shader applies coherent sun/moon/ambient response,
output exposure, water Fresnel/specular response, normal self-shading, and one
cast direction across raised terrain, volcanoes, vegetation, shore bodies, and
river clutter. Static scenes remain idle, and visible object lights stay absent
until L17/I17.

The production replay is part of both `integration` and `full`. It renders the
approved 192-tile L13 BIQ fixture and its topology-only halo through the built
32-bit DLL to ignored near-noon and far-sunset BMPs. The two images require zero
fallback and witness authoritative rivers, terrain/relief/features, both Civ III
zoom bases, and shared lighting before an interactive game install.
