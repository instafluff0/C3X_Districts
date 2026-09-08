# Mountains

Five source macro height/blend variants, complete rock and snow materials, matching silhouette and shadow coverage. Edge-adjacent mountains use their captured topology to broaden along the range axis and compose neighboring fields into one height surface. Each world patch owns a non-overlapping portion of that surface with identical shared-edge samples, while retaining the authored field resolution. Mountain collars inherit the continuous underlying grass, plains, tundra, or desert weights and the authoritative shoreline coverage, including matching source-shadow clipping.

`standard.json` identifies the shared implementation, dependencies, fixture recipe
and focused regression tests. The current checkout is authoritative.
Reference captures reproduce production using small synthetic diagnostic scenes;
they are not claimed to be live Civ III captures. References are optional review
aids. See `Renderer/docs/visual_fidelity_playbook.md`.
