# Grassland

Start with the current C3X appearance: complete source grass color, height and
specular channels, world-space sampling, continuous neighboring biome weights,
the shared production lighting, and the production coastline height join.
The current implementation and render recipe are in `standard.json`.

The detail fixture is a flat grassland patch. The gameplay fixture includes
plains, a hill, forest and a mountain around the grassland. Both use the current
production DLL and its ordinary profile, including its existing sampling and
display transfer. Synthetic fixture terrain is explicitly diagnostic; it is not
represented as captured game state.

User approval of the current build establishes revision 1. Fresh renders of that
unchanged build may fill its reference slots. Subsequent visual changes require
explicit approval and a new revision.
