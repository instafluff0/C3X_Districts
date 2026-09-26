# Grassland and plains on `test.biq`

Build the isolated candidate with `build_candidate.bat` on the Windows VM, then
run this from the project root:

```sh
python3 Renderer/lab/studies/terrain/test_biq.py --dll Renderer/native/build/grass_plains_candidate/C3XRenderer.dll --output Renderer/lab/out/grass-plains-test-biq/candidate
```

The script exports the unchanged BIQ to a preview scene and records input and
image hashes for two camera distances and noon/evening lighting. Candidate
screenshots stay under ignored `Renderer/lab/out/`; nothing is staged into the
game installation.

The normalized source pack contains 11 weighted grassland surface recipes and
8 plains recipes, with authored triangle shapes, UVs, and texture atlases. The
base material uses source color, height, and specular channels. A source R8
detail field adds continuous albedo grain and a restrained light-facing normal
response. Tile coordinates seed patch count, selection, position, rotation,
and scale, so neighboring views reproduce the same detail. The count range,
frequencies, and shader strengths are C3X Lab reconstruction choices, not a
confirmed Civ VI scatter or lighting equation. This follows
`Renderer/docs/visual_fidelity_playbook.md`: preserve the source material stack
and UV detail, use stable source composition, and evaluate form with the shared
scene light.

The desert boundary is a required visual check. A rejected material-mixing
trial created a green triangle along the ground mesh edge; that shader path was
removed. The current candidate retains the continuous base biome weights and
restores the original desert dune footprint, height, and normal. Grassland and
plains patches fade with their interpolated biome ownership. Compare the
`plains-close` view against the baseline at the desert edge before changing
biome mixing or decal coverage again.

This candidate improves material texture and light-facing detail. It does not
add terrain elevation or new cast shadows to flat grassland/plains. True
microrelief would require a shared height query, normals, object anchors, and
shadow casters to agree at tile and biome boundaries. The source assets are
used more fully than a base-color-only pass, but exact source scatter rules and
some source material responses are still unresolved. These views are Lab
examples, not accepted game integration evidence. The category dispatcher was
blocked by a stale shader-preparation cache receipt in this checkout; the
isolated native previews and current unit suites provide the evidence here.
