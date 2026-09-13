# Forests

Current weighted source-tree recipes, opacity masks, normals, uniform source proportions,
building/water exclusions, and the recovered terrain-following dirt/leaf-litter floor decals.
The source zero-count decal pool follows the individual recipe `ShowDecal` flag and tree
centers; the unavailable engine scatter is reconstructed deterministically.
Decal footprints use a 0.28 Civ III scene conversion on a deterministic half
of eligible `ShowDecal` tree placements.

`standard.json` identifies the shared implementation, dependencies, fixture recipe
and focused regression tests. The current checkout is authoritative.
Reference captures reproduce production using small synthetic diagnostic scenes;
they are not claimed to be live Civ III captures. References are optional review
aids. See `Renderer/docs/visual_fidelity_playbook.md`.

The retained profile now stores immutable placements separately from shared tree
body meshes. Spatially selected color and shadow passes use compatible instance
batches; source recipes, exclusions, density and material shading are unchanged.
Reflection and legacy profiles retain the existing geometry execution. The current
comparison and validation are recorded in `Renderer/docs/retained_renderer_plan.md`.
