# Civ V Environment Tree Source Pass

This pass pauses the combined beauty scene and isolates the forest. It uses the
Steam Workshop Environment Skin source as a local prototype input; no source
asset or source-specific runtime dependency is intended for redistribution.

## Confirmed source data

- `Clutter.artdef` selects 22 desktop forest meshes through 25 placement
  records. Their authored `Count` total is 180. Scale, scale variation,
  `RotateZ`, priority, overlap, decal, width, center-model, minimum-count, and
  low-end fields are preserved in the local generic pack.
- The forest definition also says `ClipBuildings=true`, `ClipRiver=true`, and
  `ClipCoastline=true`. Those are future scene-composition constraints; they
  are not exercised in this isolated tree fixture.
- Each leafy body is a complete authored mesh. The three `leafy_clump` bodies
  are already compound multi-tree meshes. There is no source record describing
  procedural construction of a crown from smaller parts.
- Every forest mesh uses static vertex profile `0x315CFCD9`, stride 24. Position
  is half-float XYZ at byte 0, an authored octahedral signed-byte normal is at
  byte 6, and half-float UV0 is at byte 8. The remaining bytes are zero in the
  inspected leafy buffers. Across the 22 bodies the authored normals have a
  weighted mean dot of 0.933 with recomputed geometry normals; the only negative
  cases are 33 of 217 vertices in one compound clump, consistent with deliberate
  foliage-facing rather than a reason to replace the source data.
- Leafy materials bind base color, gloss, two LEAN moment textures, and a
  separate `Generic_OPAC` texture at material offset `0x30`. The earlier study
  omitted `Generic_OPAC`; that omission exposed the low-poly canopy carrier
  polygons as wedges. Candidate r16 restores the source mask with a bounded
  0.5 single-sample cutoff and produces irregular leafy silhouettes and gaps.
- `DEFAULT_LIGHTING` supplies a noon sun color of 6.2/4.5/3.5, approximately
  50-degree zenith, and zero exposure offset. The skin also supplies a 64x64x64
  RGBA8 noon color-key volume. Candidate r18 preserves the sun chromatic ratio,
  applies that exact LUT offline after the Metal render, and carries the same
  source opacity mask into the geometry-derived soft-shadow witness.

## Still inferred

The source ArtDef stores scatter recipes, not final per-tile transforms. The
isolated patch therefore uses deterministic low-discrepancy positions while
preserving source variant weights, scale variation, and rotation semantics.
The precise engine scatter sequence, multisample opacity-coverage function,
LEAN BRDF evaluation, shadow-map filtering, ambient SH evaluation, and temporal
postprocessing remain engine behavior rather than recovered metadata.

## Current witness

The current isolated review image is
`out/beauty-trees-r18/h12-z1-pan00-civ5-lut.png`. It is rendered through the
macOS Metal backend at 1536x1024 with 4x MSAA, then transformed by the exact
source noon color cube. The full city/mountain/Warrior scene remains paused.
