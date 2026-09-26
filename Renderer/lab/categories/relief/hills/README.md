# Hills

Authored high-resolution hill relief, stable world-seeded rolling chains,
irregular rock decals and the production coastline join. A locally imported
generic R8 height field may replace the normalized baseline without creating a
runtime dependency on its source game or mod.

Native Civ III chooses forested or jungled hill art from the four diagonal
neighbors of a hill tile. The renderer now applies that selection to captured
terrain and places the resulting plants on the authored hill surface, checking
their base vertices against the slope. Original `test.biq` includes both cases;
the sandbox examples use hill tiles `(20,84)` and `(32,50)` respectively.

`standard.json` identifies the shared implementation, dependencies, fixture recipe
and focused regression tests. The current checkout is authoritative.
Reference captures reproduce production using small synthetic diagnostic scenes;
they are not claimed to be live Civ III captures. References are optional review
aids. See `Renderer/docs/visual_fidelity_playbook.md`.
