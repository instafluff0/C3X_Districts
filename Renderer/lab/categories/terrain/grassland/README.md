# Grassland

The current C3X appearance combines complete source grass color, height and
specular channels with source surface-detail shading. Optional disconnected
grass triangles are omitted because their straight edges remained visible.
The fine color sample is retained while its coarse repeated band is softened.
World-space sampling, continuous neighboring biome weights, the
shared production lighting and the coastline height join keep the result
continuous across tile boundaries.
The current implementation and render recipe are in `standard.json`.
Renderer64's coordinated Terrain/Mountains import is recorded in
`Renderer/docs/lab_material_integration.md`.
Terrain and mountain material/height inputs now come from the selected terrain
pack's `natural_runtime/` payload. Both local normalized packs are prepared;
default definitions select the Civ VI import and the local custom layer can
select the Civ V skin without rebuilding or replacing the other pack.

The production mesh uses `Renderer/lab/shared/natural/ground.h` and its shared
168-byte vertex layout. Shared `natural/queries.h`
provides wrapped neighborhood lookup, coast sampling, biome weights and the
combined height query, retaining native cache-invalidation observations.
Shared `natural/mesh.h` supplies projected surface patches, the exact production
rock decals, mountain surface and forest bodies/exclusions. `patterns.h` supplies
the remaining deterministic placement kernels.

The detail fixture is an uninterrupted grassland patch. The gameplay fixture includes
plains, a hill, forest and a mountain around the grassland. Both use the current
production DLL and its ordinary profile, including its existing sampling and
display transfer. The production relief queries retain their exact flat-ground
shortcut; unsupported fixture inputs are rejected. Synthetic fixture terrain is
explicitly diagnostic and is not represented as captured game state.

The `wetland-edge` fixture isolates marsh beside grassland and plains on both
map axes. Optional rolling-ground height now stays flat across each marsh or
floodplain tile and its material collar, then returns smoothly to the original
field within two tiles of the wetland edge. Previously, interpolation toward
zero at wetland centers left substantial height at the boundary and lifted the
upper grass/plains layer above the flat marsh receiver. The shared height query
keeps terrain and object grounding consistent; authored hills, mountains, river
carving and coastal joins retain their existing rules. The executable low-relief
regression covers both wetland types, dry biomes, corner blends and world wraps.

The current checkout remains authoritative. Fixed references are comparison aids;
replace them only when the user explicitly wants a new visual comparison point.
