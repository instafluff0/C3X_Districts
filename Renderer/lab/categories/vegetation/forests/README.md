# Forests

Current weighted source-tree recipes, opacity masks, authored vertex normals,
paired source slope textures,
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
The Lab pack scales tree bodies to 50% of their source height while retaining
their horizontal footprint. It applies the corresponding inverse normal transform;
source meshes and texture payloads remain unchanged. This is a C3X visual choice,
not a recovered Civ VI scale.

Civ III's per-tile pine flag selects the source all-conifer forest recipe
(pines, pine clumps and shrubs) on the same bodies; pine forest on tundra uses
the source snow-covered pine set. Both sets come from the optional
`forest-varieties.bin` beside `natural.bin`; without it every forest stays
broadleaf. Raised canopy on hills follows the variety of its forest neighbours.

Canopy keeps clear of the routes, resources, mines, goody huts and barbarian
camps of its tile and the eight around it, using the same placements the object
compiler draws. A tree is also removed when the lower 30% of its crown would
cover them on screen (the narrow lane the user chose over a 60% cut). A resource
tile keeps a thinner stand (9-12 trees) spread between the resource and the
tile edge, after the source resource clutter sets that replace the full forest
with a few trees. Stationary resources (plants, rocks, decals) let that stand
close in to 70% of the opening and hide them with half the reach; animated
animals keep the full clearance because they move. Floor decals keep their core off the same ground and off
river channels and the shore; the clearing is part of the terrain compile key.
The hiding fraction, stand size and path widths are C3X choices. Cases `roads`, `resources`, `rivers-sites` and
`pines` (Lab study: `Renderer/lab/studies/vegetation/`) exercise them.

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
