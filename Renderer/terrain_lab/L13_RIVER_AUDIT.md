# L13 River Audit

## Authoritative topology

The BIQ `riverconnectioninfo` byte is preserved in the V2 lab CSV after masking
to the four known physical-edge bits: northeast `2`, southeast `8`, southwest
`32`, and northwest `128`. Northeast/southwest and southeast/northwest are
reciprocal descriptions of the same two edge families. The exporter and lab
canonicalize each reciprocal pair to one local lattice edge, so it is never
drawn or scored twice.

The curve endpoints are the exact shared corners of the adjacent Civ III
diamonds. Sixteen deterministic segments add a restrained two-frequency bend
between those fixed endpoints. The seed is the canonical local edge, not either
owner tile or raw wrapped BIQ X coordinate. Terrain, feature anchors, river
curves, and the river graph all use the same continuous `(column + row,
column - row)` lattice at a horizontal wrap.

The graph counts physical edges, connected components, longest chain, degree-1
sources, degree-1 coast mouths, and degree-3-or-greater junctions. Those metrics
drive deterministic fixture selection and local source/junction/mouth widening.

## Normalized source channels

L13 consumes generic normalized roles only:

- alternate-skin river-bed base color, height, and specular;
- inherited river LEAN normal/variance pair;
- river-source base/height decal;
- river-clutter base/height decal;
- R8 river-bank noise height field;
- five verified normalized river-rock mesh/material variants.

These inputs are art assets; the lab adds no generated smoke, mist, fire,
particles, or animation. The bank is a narrow low-opacity tint over the
neighboring terrain, while the water is subdued gray-blue to match
`Renderer/canonical/river.png`. Source material detail remains visible in the
bed and bank instead of producing a cream outline.

## Terrain conformance

The common terrain height field is lowered smoothly around the evaluated river
curve. The river pass then follows that same curve and writes over the shared
surface, so flat land, hills, mountains, forest/jungle overhang, marsh, and
coast do not use separate disconnected river sprites. Sparse source-backed
rocks are placed from canonical edge seeds. Relief depth remains authoritative,
preventing a later neighboring hill or mountain draw from cutting the channel.

The BIQ provides neither flow direction nor a waterfall bit. Although static
waterfall masks exist in the normalized water catalog, inventing waterfall
locations from elevation alone would not be authoritative. L13 therefore
records waterfalls as unreachable from this input contract and leaves the
static assets dormant. It also intentionally omits waterfall smoke/mist and all
animation, per the user's direction.

## Promotion fixtures

`PREPARE_L13_BIQ_VIEWPORT.bat` deterministically exports a 16x12 (192-cell)
horizontal-wrap viewport from the installed `Intro1 Ancient Treasures.biq`.
It requires desert, plains, grassland, floodplain, hills, mountains, forest,
jungle, marsh, volcano, coast, sea, ocean, and at least one river.

`PREPARE_L13_RIVER_TOPOLOGY_VIEWPORT.bat` exports a second 16x12
horizontal-wrap viewport from the installed `5 Mesoamerica.biq`, prioritizing
unique physical edges, longest connected chain, sources, mouths, and junctions.
This denser view makes river continuity through relief and vegetation easy to
inspect without weakening the all-terrain regression scene.

`RUN_L13.bat` requires the local `Civ5EnvironmentSkin` and
`Civ5EnvironmentVegetation` packs and emits complete, no-rivers, rivers-only,
thumbnail, and river-topology renders. L13 remains a standalone lab gate; it
does not modify injected code or transfer game ownership.

The all-terrain export is locked to raw origin `(34,26)`: 192 visible cells,
40 river-bearing cells, 28 unique physical edges, longest chain 10, seven
sources, one coast mouth, and horizontal wrap. Its CSV SHA-256 is
`1500c085e88425796fb150db4a96a904722d1f8b0cce1d614d46db58bc3c0b4b`.
The topology export is locked to raw origin `(39,53)`: 192 visible cells, 43
river-bearing cells, 33 unique physical edges, longest chain 19, four sources,
two coast mouths, five junctions, and horizontal wrap. Its CSV SHA-256 is
`a90295b59ef0d4004433b36eeb2b7b917efb4b4e0b4315470017d19806c2ca95`.

Two consecutive executions of the official `renderer_dev.py lab` path produced
byte-identical output. Candidate BMP SHA-256 values are:

- complete: `3613a0f3d5e171f929f0adfa36e15934a980257c36e683c8486b765f0f4c07f8`;
- no rivers: `8dd8ff650bf606197d40d62d0df59b0439527fbe2966c58b2ee18b2cc40fa666`;
- rivers only: `bd0a090ef07e022f52f4378783408fd726b82e3b80bf79266208a6faa3b3a09e`;
- thumbnail: `f2e5d2d3ce0d529120ba8040951df3709e08fd98f7e8df03696f4c00e3a4a4e0`;
- topology: `c6c6812ec7e17cbf49e95f7614e4b5acbaee65556b08f45371574230c4a47039`.

The 2026-09-05 official lab report passed 52 Python tests, 12 exporter tests,
the Windows build, and all five L13 renders. The user explicitly approved the
alternate-skin promotion on 2026-09-05. The frozen handoff is
`Renderer/handoffs/L13_rivers.json`; visual work advances to L13A while I13 is
left to the separate Game Integration workstream.
