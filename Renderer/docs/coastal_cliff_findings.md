# Coastal cliff attachment and materials

The September 2026 hilly-coast audit found an attachment-height error and a
placement recipe bypass. It did not find nonuniform mesh stretching. The user
selected the Civ V Environment Skin and rejected the initial boulder-only
composition against the supplied Environment Skin reference. The current cliff
revision is a preview, not an accepted production change.

## Confirmed source evidence

- All four large and four small cliff bodies are present. Re-extraction from
  Base Civ VI and the installed Environment Skin yields identical source vertex
  and index buffers for all eight. Their four texture channels differ: base
  color, both LEAN channels, and gloss. The Base material is substantially browner
  in the matched native render.
- The imported vertices retain authored UVs and octahedral normals. Normalization
  uses one uniform scale, and runtime instances scale all three axes equally.
  The source rocks are already broad bodies; changing texture sets cannot change
  their proportions.
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

The importer preserves the vertical origin for cliff bodies. The eight-way audit
shows only a Z translation: topology, XY, UVs, normals and skin texture payloads
remain unchanged. Other shore assets retain their previous normalization.

The main mismatch was in the terrain adapters. `ReliefQuery` generated a rocky
coastal shoulder, but `SurfaceQueries::height` multiplied that already shaped
height by the ordinary beach envelope a second time. Rock placement used the
unflattened height. The natural terrain therefore receded below the rocks into a
beach instead of meeting them at a cliff rim.

The candidate preserves the rocky pickup height through the natural adapter,
removes the extra half-height and tile-edge attenuation from the generated cliff
shoulder, and tightens its seaward rise. Lowland beaches retain the original
formula. The natural terrain shader now binds the existing selected cliff color
and height channels and projects them onto steep faces using three planar views.
Pixel derivatives identify the actual tessellated face even when a narrow rise
falls between vertex-normal sample positions. Grass remains on the upper rim.
The accepted beach/river surface shader is unchanged by this revision.

Large rocks use the imported recipe and uniform scale/variation, with a bounded
size and a landward inset into the rim. Smaller bodies dress the foot and top.
The importer keeps their source origins, UVs, normals and mesh proportions. This
composition, face projection, local height fit and offset mapping remain **C3X
reconstruction**, not recovered Civ VI scatter or shader equations.

## Review and promotion

`Renderer/lab/out/shorelines/cliff-audit/` contains the source and mesh audits,
`placement-comparison.png`, `skin-comparison.png`, and `verification.json`.
The earlier `before`, `after` and `base` views record the rejected boulder study.
The current Civ V-only revision is `cliff/`; intermediate `rim`, `face` and
`joined` directories are diagnostic iterations, not alternative recommendations.
Focused validation includes the natural terrain/query tests, raised-rim behavior,
coastal material markers, source attachment preservation and deterministic rock
placement. In-game verification and visual acceptance remain pending.

The preview DLL was built from a frozen input snapshot because another task was
editing native sources concurrently. The receipt records that DLL and pack
hashes. `ShoreCliffStudy` contains the preferred corrected skin assets;
`cliff-audit/root/Renderer/packs/ShoreCliffBaseStudy` changes only their textures
for the comparison. Production still uses its previous DLL and pack.

On acceptance, promote the corrected cliff meshes and rebuilt cliff bundle
together with a compatible tested DLL. Do not pair the new placement with the
old rebased cliff bundle. Preserve the approved shore/river surface materials and
unrelated native work. Fixed references, INSTALL and live-game launch are separate
from this preview review.
