# Floodplains

Current floodplain mapping, river relationship and retained production ground response.

`standard.json` identifies the shared implementation, dependencies, fixture recipe
and focused regression tests. The current checkout is authoritative.
Reference captures reproduce production using small synthetic diagnostic scenes;
they are not claimed to be live Civ III captures. References are optional review
aids. See `Renderer/docs/visual_fidelity_playbook.md`.

The current runtime gives flood plains the plains ground material and a lower
relief scale. It does not bind the dedicated Civ VI floodplain decals. The
installed Expansion 2 ArtDefs define separate four-decal plains and grassland
sets. Each has authored scale 1.15, variation 0.15, count 4, rotation, and
overlap permissions. The normalized local import retains the source triangles,
atlas UVs, base color/opacity, height, gloss and fog channels. Civ VI's
`FloodplainsMtl` binding is empty, so the decal sets dress existing ground;
the exact engine scatter and shading equations remain unknown.

Run `python3 Renderer/lab/categories/terrain/floodplains/source_preview.py`
from the project root to rebuild the ignored local source pack and the
`Renderer/lab/out/floodplains/source-preview/` studies. They include source
close-ups, a matched control, two densities on detail and gameplay context,
and difference images. The compositor uses the source meshes and channels;
its orthographic placement, density, shade approximation and object mask are
lab inferences. These images are review studies, not production D3D renders,
reference replacements, or a promoted floodplain implementation.
