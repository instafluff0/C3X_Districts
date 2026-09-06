# L9.4 Vegetation Composition Audit

## Input boundary

L9.4 reads `VegetationNormalized/vegetation_runtime.bin`, a generic build product emitted from the accepted L8 pack. The 172 KB bundle contains 22 normalized meshes, four deduplicated BC1 base-color atlases, and the base-forest, snow-forest, and jungle placement groups. It contains no Civ VI paths, package names, landmark names, or cooked-format dependencies.

The lab never reads `Clutter.artdef` or `environment/clutter.blp`. Those remain build-time inputs to `clutter_blp_extractor.py` only.

## Render path

Vegetation uses a separate Direct3D draw pass with:

- the normalized indexed triangles expanded into a compact triangle list;
- the corrected source UV0 and actual base-color atlases;
- area-weighted normals produced by the L8 adapter;
- one depth buffer for self-occlusion and overlap between feature bodies;
- deterministic Z rotation and ArtDef scale variation;
- restrained contact shadows drawn on the land layer; and
- one shared `0.82` scene-composition scale applied after the exact ArtDef scale.

The four-tile fixture draws four base-forest instances and eight jungle instances. Semantic anchors guarantee at least one individual pine, pine clump, shrub, large grass, two palms, and three plant forms; remaining instances use the normalized ArtDef count weights. Candidate positions are deterministic, stay landward of the beach, and retry away from the authored mountain's high footprint. Snow-forest assets remain validated by the L8 sheet but are intentionally absent from this grass/coast fixture.

## Rejected draft

The first native draft forced every base-forest and jungle variant into the two land tiles. Multi-tree clumps made its effective density much higher than its instance count, obscured the authored mountain, clipped the back canopy, and produced oversized shadow blocks. That draft is rejected. It did prove that all selected meshes, material indices, and depth-tested overlaps rendered through the new native path.

## Acceptance candidate

The final candidate produces:

- `terrain_beauty_l9_4.bmp`: complete terrain, water, and real vegetation;
- `terrain_beauty_l9_4_vegetation_only.bmp`: the identical feature draw and contact shadows without terrain or water;
- `terrain_beauty_l9_4_no_vegetation.bmp`: the terrain/water baseline without the feature pass; and
- `terrain_beauty_l9_4_thumbnail.bmp`: a 256x128 box-filtered reduction of the complete pass.

The no-vegetation output is byte-identical to `terrain_beauty_l9_3.bmp`, proving that L9.4 did not change the accepted terrain/water baseline. The complete candidate has no clipped canopy, keeps the authored mountain and shoreline visible, separates forest and jungle into readable zones, and retains those relationships at thumbnail size.

L9.4 is ready for visual sign-off. Production integration remains outside Terrain Lab and begins only after the L9 exit gate is accepted.
