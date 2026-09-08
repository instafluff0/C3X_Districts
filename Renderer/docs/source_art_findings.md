# Source-art findings worth retaining

These are source observations and rendering lessons, not a promotion ledger.
The accepted C3X build remains the category baseline. The
[visual fidelity playbook](visual_fidelity_playbook.md) provides the general
method; this page preserves concrete findings from the retired source studies.
Source-derived art stays local and is not redistributable with C3X.

## Terrain, relief and vegetation

The selected standard hill macro field is authored 512×512 R8 relief. Mountain
macro height/footprint is 256×256, while its base/top/snow material channels are
2048×2048. Material height is surface detail, not a replacement for macro shape.
Missing channels, mismatched UV projection, insufficient sampling and compressed
lighting contrast caused major quality losses; output resolution alone did not
solve them. Keep color, height and specular in the same material coordinates.

The selected hill definition binds distinct grassland and elevated grass-hill
materials, with authored high-grass threshold 10. Its three grass-hill decal
entries have counts 3/2/2, rotation, scale 3 and 0.1 variation. Using their alpha
footprints with hill-top material is a C3X adaptation, not recovered Firaxis shader
math. Tundra has a separate snow-hill clutter layer above source height 7, with
two count-30 entries; identifying those entries does not mean they are imported.

Grassland, plains and desert expose normalized surface/decal placement groups in
addition to their base material channels. Their placement scales, variation,
counts, exact shared-buffer triangles, per-vertex atlas UVs, alpha footprints and
paired color/height channels are confirmed source data. C3X consumes those
records through a generic terrain-pack recipe and projects them onto the
continuous terrain surface. The deterministic world-cell selection, density,
sparse desert distribution and modest normal/AO response are explicit C3X
composition choices; the exact source scatter sequence and terrain-cache merge
equation remain unrecovered.

Grass and paving textures are atlases of parts. Exact source triangles/UVs are
required; repeating or stretching the whole sheet is not source reconstruction.
Installed shader inspection established alpha-squared weighting before terrain
cache resolution. That does not by itself recover source cache-normal/AO baking,
source merge order, thresholds or every layer's runtime selection.

The forest source has 22 complete bodies, 25 placement records and count weight
180. Three leafy-clump bodies are already compound meshes. Clutter metadata
requires building, river and coastline clipping. Static profile 0x315CFCD9 has
stride 24: half-float position at byte 0, octahedral signed-byte normal at byte 6,
and half-float UV0 at byte 8. Preserve authored normals rather than replacing
them with face averages. Generic_OPAC at material offset 0x30 supplies the leaf
mask; omitting it exposed carrier polygons as solid wedges.

Source scatter weights are confirmed; exact engine scatter sequence, opacity
coverage, LEAN evaluation, ambient SH and temporal filtering remain inferred.
Use actual opacity-tested source triangles for both visible bodies and casters.
All participating providers must agree on canonical coordinates and the same
face/cast light direction, including offscreen and wrap contributors.

## Units and cities

The inspected skinned-unit profile stores authored octahedral normals at bytes
6–7 of its 32-byte vertices. Warrior body/head/armor materials repeat in both UV
axes; helmet and weapon clamp. Clamping repeat UVs caused the apparent stretched
metal bands over skin and eyes. Paired LEAN textures are not ordinary normal maps;
an unverified decode made that study less faithful.

CityGenerators contains Generator and GroundingMaterials collections in addition
to block lists. Their spine, occupancy, area, era-order and weight parameters
are authored data, not the recovered placement algorithm. Keep Base/Expansion2
profiles distinct when content merge order is unknown. The user's single-era
city policy supersedes attempts to reproduce mixed-era distribution weights.

City paving uses selected source atlas parts, terrain-following projection and
independent material coordinates. It receives shadows but is not a rigid building
foundation. Tested AO-bearing bodies use a separate UV set; do not generalize
that observation to every source material. Tangent directions, reconstructed
normal Z and cooked dual-lobe roughness were established by source shader
inspection; exact broader environment/variance response is still limited.

Current central American palace placement uses an inferred +30-degree footprint
correction followed by quarter turns, preserving whole-body source proportions.
Ground, lights and shadows follow the same transform. The coastal constrained
site still cannot fit the surrounded seven-house recipe: preserve the existing
fallback rather than erase terrain or weaken clearance. The alternate inland
layout is retained as unapproved input, not a newer standard.

## Water and reflection

The retained natural-water interpretation combines large/small source slopes,
periodic calm regions and shared sun/moon/ambient response. Its interpretation of
the lighting control as base reflectance is an adaptation, not recovered source
Fresnel semantics. A holdout called freshwater was actually coast/sea/ocean;
fixture naming must not invent authoritative lake classification.

Planar reflection is a second view of the same scene about an explicit water
plane. Preserve original illumination, material coordinates, alpha coverage and
shadow coordinates; reflect position/depth. Exclude water, submerged geometry
and hidden supporting terrain. Resolve linear premultiplied color/coverage,
sample before tone mapping, and retain sky where coverage is zero. Include
objects visible only through reflection without granting them native ownership.
Prototype slot t121 was main/water-only, not a global texture reservation.
Multiple river elevations require a separate water-plane contract.

## Retained local studies

Ten selected source/context review images are under
`Renderer/lab/references/source-studies`. They are labeled source studies and do
not replace any category's approved D3D images. The alternative central-city
recipe/data, original BIQ map and extracted texture inputs are under
`Renderer/packs/RendererSourceStudies`; current production sources live in their
normal curated packs. Git preserves the retired experiments and their original
reproduction code. No milestone or historical package is needed for current work.
