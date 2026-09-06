# Approved Terrain Integration Contract

M6.7 established the delivery boundary; the paired I9-I18 implementation is
generated and tested against the approved L9 terrain through L18 mine handoff
records. The game DLL
does not parse handoff approval flags or hashes at runtime. It contains only the
draw paths deliberately integrated from those handoffs and reads generic
normalized packs plus the layered renderer definition. Lab art direction remains
authoritative and no source-game format or milestone identity enters the DLL.
The in-progress L19 farm/irrigation and tundra work is absent.

The game path executes a frozen production copy of the approved Lab `PSMain`
and `PSFeature` terrain functions. It does not include the live
`terrain_lab/terrain_lab.hlsl`, which has advanced into L19. The game-owned
`native/integrated_terrain.hlsl` file is only a vertex/input adapter from
authoritative Civ III anchors and state into that shared surface contract.
That adapter transfers exact base and visible terrain identities, computed
geometry normals, material weights, authored relief height/blend, signed shore
distance, water depth, and authoritative active-effect state. It does not
average categorical terrain IDs or synthesize replacement values. Production
submits the same ordered viewport-wide underlay, land, bed, water, river, cast-
shadow, feature, route, resource, city, wall, and mine passes as the approved
Lab stack. Production has a semantic
constant buffer for texel scale plus the approved shared sun, moon, ambient,
exposure, shadow, water, emissive-policy, and hour state. Standalone fixture
selectors and milestone labels are not C++ runtime state.

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

The cache is intentionally correctness-first. It retains up to 32 complete
terrain viewports in a 128 MiB exact-signature LRU. Its fingerprint includes target size,
zoom basis, ordered canonical identity plus screen occurrence anchors,
renderer-owned tile state, world size and wrap, environment, definition/pack
content revision, ownership, and device generation. The dirty clip is a
compositing instruction, not terrain identity. Dirty flags never authorize
reuse. Unit animation, selection, `SquareParts`, native overlay bits, fog
traversal, and exact city population are excluded because they belong to
retained Civ III planes or do not change the selected city body. Authoritative
routes, resource identity, city style/size/state, mines, `river_code`, hour, and
season are included because they change renderer-owned geometry or lighting. A
visible native animation continues requesting Civ III redraws while terrain
itself remains a zero-tick cache hit.

The live-scale path avoids repeated CPU work without weakening that identity.
Each terrain layer reuses one evaluated shared grid per tile, river topology is
evaluated only for the river pass, and canonical world tiles retain
anchor-independent surface, relief, normal, and cast-shadow samples in a 96 MiB
LRU. A tile sample key covers its renderer-owned state plus the complete local
topology neighborhood reached by relief and shadow evaluation, zoom, wrapping,
world dimensions, and content revision. Shadow keys additionally cover the
environment and sample density. Native overlay state and screen anchors are
excluded. Any tile mutation clears that tile's dependent samples; new scenario,
world-size, wrap, definition, or pack state clears or rejects the affected
entries. A second geometry signature deliberately
excludes occurrence anchors. When every captured occurrence has unchanged
renderer-owned content and moves by one uniform screen delta, the DLL retains
the complete generated layer geometry and applies matching XY and depth
translation in the production vertex adapter. The generated layers are uploaded
once into bounded immutable vertex-buffer chunks and their large CPU vertex
arrays are released; camera-only frames issue draws directly from those retained
buffers. The Windows 400-rendered-tile/800-record fixture measures 5.315 seconds
for a cold build and 0.026 seconds for a three-pixel/two-pixel camera move, while
still reporting authoritative scene invalidation and an incremented
geometry-cache hit. A device-reset cold rebuild bounds translation equivalence
to fewer than one differing raster-edge pixel per thousand and a maximum channel
delta of 128; the measured fixture is 292 of 393,216 pixels and 99. Exact
unchanged or recently visited matching viewports return at zero renderer ticks.
The current Windows 400-rendered-tile/800-record fixture measures 5.315 seconds
cold, 0.026 seconds for uniform anchor translation, and 1.948 seconds after one
logical tile plus its companion crosses the visible boundary. The boundary case
defeats both exact pixels and whole-viewport geometry, proving reuse comes from
the world-coordinate semantic/sample cache. Canonical synthetic and approved-
scene pixel hashes remain deterministic.

Generated GPU regions are separately bounded to two entries, 192 MiB total and
96 MiB per entry. A full live viewport larger than the per-entry limit is not
retained; this prevents the 32-bit process exhaustion reproduced by the native
stress fixture. Crossing a tile boundary still rebuilds and uploads the expanded
viewport geometry, so 1.948 seconds is an intermediate result, not the final
interactive target. Future indexed/chunked GPU geometry and staging-readback
work may reduce that remainder, but it may not reuse an incomplete topology edge
or silently present a stale viewport.

One lazily started renderer worker owns definition/pack mutation, renderer state,
D3D creation, rendering, recovery, and reset. It receives a deep copy of the
captured tile array. The capacity-one synchronous handoff accumulates no frame
backlog and returns only the exact submitted sequence; Civ III's UI thread still
captures authoritative state and performs the final serialized GDI blit while
the worker is idle. Reset joins the worker before the DLL is unloaded. This is
not a second presenter or game loop, and it never authorizes stale or native
terrain fallback.

Partial Civ III redraws preserve the destination map outside the authoritative
dirty rectangle. When custom rendering is enabled, the `m71` hook deliberately
asks Civ III for the complete visible tile traversal even if a unit or UI change
supplied a smaller traversal rectangle. This guarantees that a partial redraw
can neither become a partial terrain render nor poison the retained cache. The
renderer fills the complete off-screen target, while API v9 returns the exact
original image clip and the bridge performs `SRCCOPY` only for that rectangle.
The DLL retains its compatible DIB section and source DC between blits and copies
only the requested clip into that surface before `BitBlt`, avoiding per-frame
GDI object creation without changing destination pixels outside the clip.
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
An incomplete configured subsystem rejects the custom frame atomically rather
than mixing in native map-plane pixels. Structural capture, renderer/device,
validation, or blit errors remain visible and emit diagnostics. Configuration
off remains fully vanilla. Systems without an approved I# gate are simply not
compiled or called.

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

I14 and I15 consume Civ III road and railroad topology plus the visible
civilization's era as the approved route style. The renderer uses the frozen
exact-node curves, route layers, river bridge bodies, coexistence order, and
relief grounding. Each family has a separate ownership bit and every route
selector participates in the terrain signature.

I16 captures visibility-conditioned resource identity, class, and name. The
frozen source-backed resource bundle selects the approved stable composition
and grounding for land and aquatic bodies. Civilopedia, city-screen,
trade-network, advisor, and diplomacy icons remain native.

I17 captures city identity, owner, size band, culture group, visible era,
capital, and wall state from loaded Civ III objects. Production ports the
approved compact city and wall compositions, material slots, retained-label
clearance, and night emissive handling. Names, population text, production
status, borders, units, HUD, and UI remain native overlays.

I18 captures authoritative mine presence and uses the visible civilization era,
stable tile seed, terrain/resource context, and shared environment. Production
ports the approved preindustrial/industrial families, three variants,
terrain-following excavation decals, recursive component placements, one
compound shadow, and source-authored emissives. Mine state has its own ownership
bit and invalidates cached terrain exactly when the improvement changes.

The production replay is an explicit, separately reported Windows command in
both `integration` and `full`; the compile/synthetic build uses `BUILD.bat
portable` so Parallels cannot stop relaying before the licensed-payload replay.
The approved smoke requires zero fallback and exact ownership for the L9-L18
stack at both zooms, including clips, scrolling, wrapping, cache reuse/eviction,
state invalidation, and deterministic reset. The workflow also renders the
approved 192-tile L13 BIQ fixture and topology halo to ignored near-noon and
far-sunset BMPs before an interactive game install.
