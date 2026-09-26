# Infrastructure

Current roads, railroads, mines and farms; preserve category ownership and fallback.

The fixtures supply two connected runs and a cross-branch, plus a separate mine
and irrigated farm, with and without surrounding relief/vegetation. All four
custom ownership flags are checked through the production API. The existing
category reference shows gaps between route segments despite connected input
nodes; the candidate in `Renderer/lab/studies/roads/` addresses tile ownership
at those seams and is being evaluated over `test.biq`. It also tests all eight
neighbor directions, deterministic shape variation, four road eras, mountain
bypass paths, and river bridges with a deck between the authored side arches.
This category does not imply new support for pollution, craters, colonies or
other tile improvements that the current renderer does not replace.

`standard.json` identifies the shared implementation, dependencies, fixture recipe
and focused regression tests. The current checkout is authoritative.
Reference captures reproduce production using small synthetic diagnostic scenes;
they are not claimed to be live Civ III captures. References are optional review
aids. See `Renderer/docs/visual_fidelity_playbook.md`.

The farm study under `Renderer/lab/studies/farms/` renders deterministic irrigation
placements over the unchanged `test.biq` terrain. Its current candidate uses
source crop-atlas rows instead of shrinking the full atlas into each field,
samples field vertices against terrain relief, clips them at the shore and river
bank, and
varies palette, placement, and source tree/building composition by tile seed.
The atlas subregion choice and field layout are visual reconstruction decisions;
the source decal metadata confirms the materials and footprints, not this exact
Civ III tile arrangement. The compact runtime still omits the source crop height,
specular, and foliage opacity response. These examples have not been visually
accepted or staged for game use.
