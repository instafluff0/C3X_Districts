# Scene exchange v1

Status: shared opt-in data boundary; frozen packets retain their original meaning.
Every input is owned by its provider and read-only to consumers. Fixture `sidecars`
records pin path, SHA256, schema and owner. Their complete referenced resource
closure participates in geometry identity. No sidecar changes the frozen look by
itself; a consuming provider must explicitly declare its use.

`c3x.lab_v2.world_scene.v1` carries `coordinate_space`, `terrain_sha256`,
`region_id`, `meshes`, `materials`, and `instances`. Meshes have stable `id`,
`positions` (XYZ), `normals` (XYZ), `uv` (UV), and triangle `indices` in inline arrays. Texture resources may use content-hashed
external records; external mesh-array loading is not implemented. Instance records have
stable `id`, `mesh`, `material`, uniform `scale`, 3x3 proper `rotation`, and
XYZ `translation`. Do not deform source mesh axes or UVs. Material records carry
`id`, `alpha_mode` (opaque/cutout/translucent), `alpha_cutoff`, `caster`,
`receiver`, `baked_occlusion`, `emissive`, and content-hashed texture channel
records with explicit transfer semantics. Source evidence includes pack/asset
ID and hash plus `source_reuse`, `source_adaptation`, or `diagnostic_proxy`.
Missing cutout alpha cannot be treated as an opaque shadow caster.

`c3x.lab_v2.corridors.v1` carries `coordinate_space`, `terrain_sha256`,
`region_id`, `provider`, `revision`, `wrap_period` (XY; zero means no wrap), and
`envelopes`. Each envelope has stable `id`, `kind` (road/rail/river/bank/bridge/city/building/wall),
`polygon` (ordered XY ring, no repeated closing point), `height_range` (min/max),
`clearance` (additional world XY margin), and `source_geometry_sha256`.
Providers polygonize the actual rendered swept geometry, including bends,
junctions, banks, ballast, decks and approaches. Multiple polygons are unioned
by testing all members; shared stable IDs identify wrap aliases. Consumers
compare transformed source footprints and crown/overhang extents, never only
instance origins. `shared/scene_exchange.py` supplies an intersection witness
and validation. Camera scrolling cannot participate in placement identity.

The common coordinate convention is explicit per scene; all exchanged sidecars
must name the same convention and terrain revision. Recommended lattice space
uses X=column+u, Y=row+(1-v), Z upward in uniformly normalized tile units.
Frozen `surface_query.v1` reports source authoring height and projection
coefficients separately; an adapter must explicitly convert units and normals.
It is incorrect to infer world Z from camera depth or use pixel Z as tile units.
Q6 owns shadow evaluation; Q3/Q5 own actual corridor geometry; Q4/Q7 own placement.
