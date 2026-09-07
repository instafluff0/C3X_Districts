# Warrior source-fidelity pass

The isolated Warrior now uses the complete authored component recipe and fixes
the visible gray stretching across the figure's right shoulder, hand, and eye.
The accepted local comparison is `beauty-warrior-r4`; it is rendered by the Mac
Metal Lab and then passed through the installed Civ V environment-skin noon LUT.

## Confirmed source data

- `units/units.blp` supplies the selected body, head, armor, helmet, and weapon
  meshes, their skeleton or attachment binding, UV0, skin weights, materials,
  and textures.
- The 32-byte skinned vertex profile retains its authored octahedral normal at
  bytes 6–7. Across the selected body, head, and armor geometry, those normals
  have positive dot products with the recomputed geometric normals; the means
  are 0.9683, 0.9893, and 0.9873 respectively. The differences preserve artist
  smoothing rather than inventing new surface detail.
- The selected body, head, and armor material records explicitly require
  `repeat` addressing in both UV axes. Helmet and weapon require `clamp`.

## Root cause and correction

The earlier quick-study path discarded the packed normals and sampled every
non-foliage texture with a clamp sampler. UVs outside the unit atlas therefore
held the texture's edge texel across whole triangles, producing the apparent
metal bands over exposed skin and the eye. The unit importer now opts into the
authored packed normals, and beauty-bundle version 3 carries the source address
mode to the shader. The shader uses repeat or clamp per material.

The paired LEAN textures remain present in the material records but are not
treated as ordinary tangent-space normals in this pass. Their exact moment and
variance evaluation is still unresolved; enabling the earlier guessed decode
made the figure less source-faithful.

## Verification

- 37 focused importer, skinning, forest, bundle, and object-renderer tests pass.
- The tree regression output for r19 is byte-identical to the accepted r18
  Metal render before and after the Civ V LUT.
- No Windows, Civ III integration, or VM path is involved.
