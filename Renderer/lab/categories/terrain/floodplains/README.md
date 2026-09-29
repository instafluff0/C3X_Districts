# Floodplains

Current floodplain mapping, river relationship and retained production ground response.

`standard.json` identifies the shared implementation, dependencies, fixture recipe
and focused regression tests. The current checkout is authoritative.
Reference captures reproduce production using small synthetic diagnostic scenes;
they are not claimed to be live Civ III captures. References are optional review
aids. See `Renderer/docs/visual_fidelity_playbook.md`.

The current candidate keeps the plains ground material and lower relief scale,
then layers the selected grassland floodplain decals through the production
natural-surface mesh and D3D shader. The installed Expansion 2 ArtDefs define
separate four-decal plains and grassland
sets. Each has authored scale 1.15, variation 0.15, count 4, rotation, and
overlap permissions. The normalized local import retains the source triangles,
atlas UVs, base color/opacity, height, gloss and fog channels. The runtime
candidate uses color/opacity, height and gloss; fog remains preserved in the
normalized local import for the game's separate fog pass. Civ VI's
`FloodplainsMtl` binding is empty, so the decal sets dress existing ground;
the exact engine scatter and shading equations remain unknown. C3X's tile
selection, density, scale and 0.48 opacity response are inferred calibration.
Neighbor floodplain IDs fade patches across terrain boundaries through observed
scene queries. The source meshes, UVs and DDS bytes are unchanged. The game reads only the
generic `NaturalFidelityRuntime` pack, never the installed source art.

To rebuild the local ignored source pack before a natural asset preparation,
run `python3 Renderer/tools/asset_compiler/generic_decal_compiler.py --mapping
Renderer/lab/categories/terrain/floodplains/source_decals.json --pack
Renderer/packs/FloodplainNormalized --report
Renderer/lab/out/floodplains/source-runtime-build.json` from the project root.
The ignored pack is a required local build input and must be preserved until
it has been rebuilt from the installed art. Do not redistribute it.

Run `python3 Renderer/lab/categories/terrain/floodplains/source_preview.py`
from the project root to rebuild the ignored local source pack and the
`Renderer/lab/out/floodplains/source-preview/` studies. They include source
close-ups, a matched control, two densities on detail and gameplay context,
and difference images. The compositor uses the source meshes and channels;
its orthographic placement, density, shade approximation and object mask are
lab inferences. The current D3D witnesses are generated with
`python3 Renderer/renderer.py lab floodplains` under
`Renderer/lab/out/floodplains/`. The older source-preview images are review
studies, not production D3D renders or reference replacements. The candidate
has not been visually accepted or promoted to a staged game installation.
