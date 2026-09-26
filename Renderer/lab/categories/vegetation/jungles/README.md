# Jungles

Retained production jungle recipe and assets, grounded against the current terrain surface
with the recovered terrain-following jungle leaf-floor decals represented by eight placements
anchored to actual vegetation slots. The darker canopy response and selection density are
inferred reconstructions; source texture identity,
color space, footprint, height map, scale and variation remain confirmed package data.
Decal footprints use a 0.28 Civ III scene conversion.

The current Lab candidate routes all ten desktop jungle bodies and their 121-count
source recipe through the natural material renderer. It keeps the original DDS
base color, paired LEAN, and gloss channels at their source resolution and mip
chains. The LEAN lighting response is a C3X approximation; the source engine
equation remains unrecovered. Bodies are 50% of their original height with their
horizontal dimensions and UVs unchanged; normals receive the inverse transform.
This proportion is a visual experiment, not a recovered source scale. Placement
uses the forest's varied deterministic spiral and terrain grounding. All sixteen
ArtDef floor decals follow selected jungle centers. Both bodies and shadows use
the existing source geometry path.

`standard.json` identifies the shared implementation, dependencies, fixture recipe
and focused regression tests. The current checkout is authoritative.
Reference captures reproduce production using small synthetic diagnostic scenes;
they are not claimed to be live Civ III captures. References are optional review
aids. See `Renderer/docs/visual_fidelity_playbook.md`.
