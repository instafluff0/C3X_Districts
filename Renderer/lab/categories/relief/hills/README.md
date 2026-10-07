# Hills

Standard Civ VI 512x512 hill relief, stable world-seeded rolling chains,
irregular rock decals and the production coastline join. The natural pack
builder selects the normalized field even when a local Hillier Hills import is
present. A different generic R8 field can still be tried in an isolated study
without creating a runtime dependency on its source game or mod.

Close-zoom ground uses 64 cells per hill patch and its neighbors. Flat interiors
retain 16 cells, with refined boundary triangles meeting the same 64-cell edge.
Rock decals share those receiver triangles. This removes coarse square facets
and cracks without changing authored hill heights or blurring their materials.

Native Civ III chooses forested or jungled hill art from the four diagonal
neighbors of a hill tile. The renderer now applies that selection to captured
terrain and places the resulting plants on the authored hill surface, checking
their base vertices against the slope. Original `test.biq` includes both cases;
the sandbox examples use hill tiles `(20,84)` and `(32,50)` respectively.
Forested hills use the same source forest bodies, base color, paired LEAN,
gloss, opacity, and slope-following leaf/dirt floor decals as ordinary forests.

Rivers run on tile edges, but hill bodies reach across them. Hill relief stays
under a 1.4 units/pixel bank rising from the drawn water's edge
(`SurfaceQueries::terrain_height`), so no hill covers or shades a river.
River distances are exact to 54 screen pixels (`river::Corridor` files
segments within .95 tiles; river-terrain detail keeps the .65-tile cells), so
the bank and low relief no longer cliff on cell lines. Accepted by the user on
2026-10-07; the study is `Renderer/lab/studies/hills/`, tests
`test_hill_river_banks.py`.

`standard.json` identifies the shared implementation, dependencies, fixture recipe
and focused regression tests. The current checkout is authoritative.
Reference captures reproduce production using small synthetic diagnostic scenes;
they are not claimed to be live Civ III captures. References are optional review
aids. See `Renderer/docs/visual_fidelity_playbook.md`.
