# Coastal cliff attachment and materials

The September 2026 audit found errors in projection, normal orientation,
attachment, depth and terrain tessellation. The earlier conclusion that the
rocks were not stretched was incomplete: it checked imported mesh bytes and
instance scale, but missed the separate vertical projection. The user selected
the Civ V Environment Skin. The current revision is a preview awaiting visual
acceptance.

## Confirmed source evidence

- All four large and four small cliff bodies are present. Re-extraction from
  Base Civ VI and the installed Environment Skin yields identical source vertex
  and index buffers for all eight. Their four texture channels differ: base
  color, both LEAN channels, and gloss. The Base material is substantially browner
  in the matched native render.
- The imported vertices retain authored UVs and octahedral normals. Normalization
  uses one uniform scale. Runtime instance scaling is uniform before conversion
  into the shared relief coordinate system. The projection error below happened
  after import; changing texture sets could not fix it.
- `Clutter.artdef` gives `CLUTTER_CLIFF` `TerrainHeight=false` and `FixedHeight=0`.
  Large bodies extend below their source origin: rock 01 spans Z -3.34960938 to
  4.69140625. The old importer moved the lowest vertex to zero, exposing the lower
  skirt when attached near the waterline. Its exact intended submergence is an
  inference: fixed origin alone does not recover the terrain layer attachment
  equation. The small group uses terrain height;
  its bodies also extend below their origin.
- The large recipe supplies count 16 per variant, scale 1 and variation 0.10;
  the small recipe supplies count 12, scale 1 and variation 0.15. Both specify
  `RotateZ` and allow overlap. The runtime previously bypassed these imported
  selection weights and scales.
- `TerrainStyle.artdef` describes a generated cliff surface with large, foot and
  upper clutter, not a standalone wall mesh. Its cliff layer has Height 5,
  PositionOffset 1.1, HeightOffset 1.3 and roughness 0.5. Large-rock offsets are
  0 to 0.75, foot-rock offsets 12 to 22, and upper-rock offsets -9 to 0. These are
  source values; their final engine coordinate mapping is not recovered.

## Current candidate

- **Projection and depth:** cliffs previously used `112 × .82 = 91.84` vertical
  pixels at the canonical feature scale, while other imported features use 150.
  They therefore retained only 61.23% of the corresponding feature height.
  Conversion into relief coordinates now preserves that 150-pixel feature basis.
  `GroundProjection` supplies the same screen/depth mapping as natural terrain;
  the older independent depth formula is removed.
- **Normals:** positions rotated into positive map Y while normals reflected Y.
  `CliffTransform` now applies one consistent basis to position and normal, with
  inverse-transpose correction for the vertical coordinate conversion. The
  normal-map derivative basis uses that same world convention. Tests verify
  perpendicularity against transformed sloped faces at multiple rotations/scales.
- **Placement:** the landward inset falls from .19 to .08 tiles. Large-body scale
  is bounded to .42–.52 before imported variation, and minimum contour spacing
  falls from .30 to .20 tiles. The rim intersects the bodies lower down instead
  of burying their faces. Foot details vary from .19 to .44 tiles seaward, use .40 scale and a
  slightly raised waterline anchor. Smaller .27-scale upper details occur on
  one third of placements, varying from .18 to .46 tiles inland instead of
  forming a parallel row.
  Imported selection weights and variation remain active.
- **Terrain join:** the coastal floor is now a minimum, not an added ledge on
  top of the hills (28 relief units rather than an additive 46.67). The natural
  hills use the steep .04–.18-tile cliff envelope through rocky sections;
  ordinary beaches retain their gentle envelope. Both large rocks and upper
  details query the same final surface used by the visible hills, and upper
  details sample at their own actual placement coordinates. The narrow cap
  extends across water-owned cells where the visual contour encloses land.
  Rocky coverage is complete by .04 tiles, before the face rises; retaining
  beach alpha here had made elevated faces translucent. Rocky coast tiles use
  a 48-step ground grid; ordinary terrain retains 16.
- **Material:** source UVs and all four Civ V texture payloads are unchanged.
  Cliffs now use their requested clamp addressing and the natural-detail sampler
  (16× anisotropy, mip bias −1), instead of the coarser wrapping terrain sampler.
  The existing slope channels have unit strength rather than .35 attenuation.
  Generated steep faces use the selected cliff-body color atlas with the
  existing terrain height detail, projected through three planar views; grass
  remains on top. This replaces the separate brown terrain material at the join. The approved beach/river surface
  material is unchanged. A narrow wet-rock band darkens the cliff material and
  slightly increases gloss close to the shared water plane; dry faces retain
  source color. This response is a C3X material interpretation.

The eight-way mesh audit records only the earlier vertical-origin translation:
topology, XY, UVs, source normals and skin texture bytes remain unchanged. Other
shore assets retain their original normalization. The generated cap, placement
fit, offset mapping, slope interpretation and terrain face projection remain
**C3X reconstruction**, not recovered Civ VI scatter or shader equations.
The source brown cliff terrain material is distinct from the pale clutter
texture. Matching the generated face to that clutter is a C3X material choice,
not the recovered TerrainStyle binding. The white Dover material is not used.

## Verification and promotion

`Renderer/lab/out/shorelines/cliff-audit/` preserves the source and mesh audits.
`joined-skin/` contains the current native D3D11 views;
`joined-skin-comparison.png` and `joined-skin-verification.json` describe the
comparison and checks. All 144 focused checks pass. Three native captures
completed without fallback, and the lowland control is pixel-identical. The preceding
`corrected/` candidate records the user's improved-but-still-detached wall feedback.
The current checks cover non-additive hill height, full rocky coverage before
elevation, hill height retained at the cliff, and attachment to sloped ground.
Earlier directories are diagnostic iterations, not alternative skin choices.
The `fine`, `proportion`, `normal` and `world-normal` views isolate tessellation,
attachment/size, detail strength and normal orientation respectively. Sun-angle
and color-coded ownership diagnostics are also retained.

The preview uses a frozen C++ and shader input snapshot because other tasks are
changing renderer caching and shadows concurrently. The receipt identifies the
DLL, compiled inputs, shader set and corrected cliff pack. No production staging
or reference replacement is performed here. This task did not stage its cliff
candidate; production may change independently in other tasks.

On visual acceptance, promote the corrected cliff meshes and rebuilt bundle
with a compatible tested DLL. Do not pair this placement with the old rebased
cliff bundle. Preserve the approved shore/river materials and unrelated native
work. INSTALL and live-game launch remain separate actions.
