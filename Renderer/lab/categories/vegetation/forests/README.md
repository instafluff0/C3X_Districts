# Forests

Current weighted source-tree recipes, opacity masks, authored vertex normals,
paired source slope textures, uniform source proportions,
building/water exclusions, and the recovered terrain-following dirt/leaf-litter floor decals.
The source zero-count decal pool follows the individual recipe `ShowDecal` flag and tree
centers; the unavailable engine scatter is reconstructed deterministically.
Decal footprints use a 0.28 Civ III scene conversion on a deterministic half
of eligible `ShowDecal` tree placements.
The atlas and placement metadata are confirmed source data; the half-density choice,
scene conversion, canopy attenuation, floor geometry and exact scatter remain C3X reconstruction.
Tree base-color, paired LEAN textures and gloss retain their original DDS resolution
and mip chains. The paired textures now set the material flag consumed by the
object shader; their visual normal response remains an inferred C3X approximation
of Civ VI's unrecovered LEAN lighting.

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
