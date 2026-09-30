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

The detail fixture is a flat grassland patch. The gameplay fixture includes
plains, a hill, forest and a mountain around the grassland. Both use the current
production DLL and its ordinary profile, including its existing sampling and
display transfer. The production relief queries retain their exact flat-ground
shortcut; unsupported fixture inputs are rejected. Synthetic fixture terrain is
explicitly diagnostic and is not represented as captured game state.

The current checkout remains authoritative. Fixed references are comparison aids;
replace them only when the user explicitly wants a new visual comparison point.
