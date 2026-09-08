# Grassland

Start with the current C3X appearance: complete source grass color, height and
specular channels, world-space sampling, continuous neighboring biome weights,
the shared production lighting, and the production coastline height join.
The current implementation and render recipe are in `standard.json`.

The production mesh uses `Renderer/lab/shared/natural/ground.h` and its shared
168-byte vertex layout. Both native and Mac CPU tooling can use this exact
projection, surface sampling and grid emission. Shared `natural/queries.h`
provides wrapped neighborhood lookup, coast sampling, biome weights and the
combined height query, retaining native cache-invalidation observations.
Shared `natural/mesh.h` supplies the exact production rock decals, mountain
surface and forest bodies/exclusions; `patterns.h` supplies deterministic
placement and the existing analytic dune field.
The current approved images are D3D11. The Mac detail preview now uses these
shared inputs and the production material shader. Run `lab grassland --backend
metal --case detail`, then `compare grassland --backend metal` through
`Renderer/renderer.py`. Its measured maximum channel difference from the approved
detail image is 1/255, with 97.08% of pixels exactly equal. This is a diagnostic
comparison, not a new approval or complete category parity claim.

The detail fixture is a flat grassland patch. The gameplay fixture includes
plains, a hill, forest and a mountain around the grassland. Both use the current
production DLL and its ordinary profile, including its existing sampling and
display transfer. The Mac path currently rejects the gameplay fixture: mixed
relief and vegetation geometry now render in an internal context diagnostic,
but its shadow pass is not yet connected. Both paths use the production relief
queries, including their exact flat-ground shortcut; unsupported fixture inputs
are rejected. Synthetic fixture terrain is explicitly diagnostic; it is not
represented as captured game state.

User approval of the current build establishes revision 1. Fresh renders of that
unchanged build may fill its reference slots. Subsequent visual changes require
explicit approval and a new revision.
