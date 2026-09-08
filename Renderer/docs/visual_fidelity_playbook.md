# Visual fidelity playbook

## Purpose

This document records general methods for making source art retain its detail,
depth, and material character in a custom game renderer. It applies to terrain,
relief, vegetation, units, structures, improvements, resources, and other
rendered map objects.

The central lesson is that a high-definition appearance usually comes from
preserving and correctly interpreting source information. It does not primarily
come from sharpening the final image. A result that looks soft, stretched,
flat, or "smooshed" often indicates that useful information was discarded,
misread, or reduced earlier in the pipeline.

## 1. Investigate the source before recreating it

Before inventing geometry, textures, noise, shadows, or shader behavior,
inventory everything the source supplies:

- meshes and compound-object relationships;
- large-scale height fields and object footprints;
- base-color, height, normal, gloss/specular, ambient-occlusion, opacity, and
  blend-mask textures;
- vertex-normal and tangent encodings;
- UV coordinates and per-material repeat or clamp addressing;
- texture color spaces, formats, mip chains, and channel meanings;
- object scale, rotation, attachment, socket, and placement metadata;
- authored variation families, weights, counts, and exclusion rules;
- lighting, material, and shadow metadata.

Treat upstream assets and metadata as the first authority. Programmatic
reconstruction should fill a confirmed gap, not replace source data that has
not yet been examined.

## 2. Preserve macro form and micro detail separately

Large-scale geometry establishes silhouette, elevation, and readable shape.
Material textures add smaller surface information. These are complementary and
must not substitute for one another.

Examples:

- A mountain needs its authored macro height field and footprint to establish
  peaks, ridges, slopes, and silhouette. Rock color and normal textures cannot
  recover that shape after it has been flattened.
- A hill needs actual elevation plus an irregular footprint. Painting a dark
  oval on flat terrain does not produce the same result.
- A unit needs its complete component geometry and source proportions. A sharp
  diffuse texture cannot repair distorted limbs or independently squashed axes.
- A building needs its authored roof and facade geometry before material
  normals, windows, and surface roughness can read correctly.

Do not flatten macro geometry and then attempt to restore it with contrast,
embossing, sharpen filters, or synthetic high-frequency noise.

## 3. Carry the complete material stack

Do not reduce a rich source material to base color alone. Preserve every
understood, useful channel through import, normalized storage, runtime binding,
and shading.

Typical roles include:

- base color or albedo;
- macro and detail height;
- geometric or tangent-space normals;
- gloss, roughness, or specular response;
- ambient occlusion or cavity information;
- opacity or cutout coverage;
- material and biome blend masks;
- emissive channels;
- owner-color or variation masks.

Preserve channel semantics and color spaces. Color textures are commonly sRGB,
while height, normals, masks, opacity, and material scalars are linear data.
Incorrect color-space conversion can erase contrast or distort material
response even when the texture itself is correct.

If a channel's encoding is unresolved, retain and label it without guessing.
For example, a moment/variance texture must not be treated as an ordinary XYZ
normal map merely because it resembles one.

## 4. Preserve authored normals and UV behavior

Source-authored vertex normals often contain intentional smoothing that cannot
be recovered by recomputing one normal per triangle. Decode and preserve them
when their format is proven.

For animated or transformed objects:

1. apply the authoritative pose;
2. apply the model transform;
3. transform normals with the appropriate inverse-transpose transform;
4. renormalize the result;
5. convert positions and normals into the renderer's common world basis.

Normals are directions, not positions. Do not apply translation to them.

UV addressing is also material data. Preserve repeat or clamp behavior per
material or primitive. Do not force every texture through one global sampler.
Clamping a repeat-addressed atlas can hold an edge texel across an entire
triangle, creating apparent bands of metal, skin, foliage, or roof material in
the wrong place. If one source material is used by primitives requiring
different addressing, create explicit normalized material variants.

## 5. Avoid destructive fitting

Preserve source geometry, UVs, proportions, component attachments, and relative
transforms. For an intact object, prefer:

- translation;
- rotation;
- uniform XYZ scale.

Do not independently squash width, depth, or height merely to fit a legacy tile
or sprite box. Adjust the camera, footprint, spacing, composition, visibility
bounds, or renderer-owned region instead. If nonuniform scale is genuinely
required by authoritative animation data, normals must still use the correct
inverse-transpose transform.

## 6. Retain resolution until final reconstruction

Do not pre-shrink or precompress source textures, geometry, masks, or
intermediate renders to the final gameplay size. Once detail has been removed,
a later sharpen pass cannot reconstruct it reliably.

A high-quality path should generally provide:

- source-appropriate texture resolution and complete mip chains;
- sufficient geometric sampling for the available height information;
- anisotropic filtering for oblique terrain and surfaces;
- multisample antialiasing for geometry edges;
- a controlled render scale when the category benefits from it;
- an evidence-backed mip bias rather than a universal hard-coded value;
- one final reconstruction into the gameplay-sized image.

Judge these settings by category. Terrain viewed obliquely may benefit greatly
from anisotropy and a modest negative mip bias, while a different unit or UI
path may require another choice. More sharpening is not automatically more
detail; excessive bias can shimmer, alias, or reveal compression artifacts.

## 7. Use one coherent scene-wide environment

Terrain, hills, mountains, trees, units, buildings, and other objects should
consume one authoritative environment state for a frame.

The same normalized light direction must drive:

- face lighting;
- diffuse and specular evaluation;
- cast-shadow projection;
- any phase-dependent directional response.

Do not give each category a private sun direction or clock. Independent light
vectors produce contradictory face shading and shadows even if every shader
looks plausible in isolation.

All casters and receivers must also share one authoritative world-coordinate
basis. A shadow map can be mathematically correct yet visibly detached if an
object's displayed projection and submitted shadow geometry use different
origins, axis flips, height units, or scale factors.

## 8. Cast shadows from real source geometry

Use actual opaque or opacity-tested source triangles as shadow casters wherever
practical. Preserve source opacity masks and alpha cutoffs for foliage, fences,
thin structures, and other cutout materials.

Avoid generic circles, ellipses, or dark blobs as permanent substitutes. They
do not follow silhouette, pose, or lighting direction and can make otherwise
detailed art look pasted onto the ground. If such a fallback is temporarily
necessary, identify it explicitly and keep it out of accepted source-fidelity
evidence.

Face shading and cast shadows solve different visual problems. Face shading
describes form on the object; cast shadows anchor it in the scene. Both must be
readable, share a direction, and retain enough ambient fill that shaded faces do
not become featureless black regions.

## 9. Create variation through stable source composition

Obvious repetition is almost as damaging as missing detail. Prefer multiple
authored variants and deterministic composition over stamping one identical
feature on every tile.

Useful inputs include:

- stable per-tile or per-object seeds;
- multiple authored geometry and material families;
- weighted source variants;
- bounded rotation, uniform scale, and placement variation;
- different footprint combinations;
- source-defined density and count ranges.

Variation must be deterministic so caching, replay, scrolling, and comparison
remain stable. It should change arrangement without destroying the identity of
the source assets. Synthetic detail should be introduced only when the source
truly lacks an equivalent, and it must remain labeled as an inference.

## 10. Compose a whole visible scene

Isolated tiles are useful for diagnosis, but the final renderer should build a
coherent visible scene or viewport where possible. Cross-tile terrain,
mountains, trees, shadows, transitions, and objects need natural overlap and
shared depth.

Use authoritative geometry for exclusions and clearance. For example, forest
placement should respect building footprints, rivers, shorelines, roads, and
other occupied regions rather than relying on a hand-painted empty circle.
Exclusion geometry is a placement constraint, not replacement shadow geometry.

Do not allow a new high-detail layer to restore an older low-detail version of
another layer. When replacement relief is active, a retained river or overlay
pass must not silently redraw obsolete hills or mountains with it.

## 11. Distinguish source evidence from inference

Record whether each important behavior is:

- confirmed by an asset, material record, package field, or authored metadata;
- confirmed by an executable experiment;
- inferred for visual reconstruction;
- unresolved and deliberately disabled.

A plausible shader equation or scatter recipe does not become source-authentic
merely because it looks good. Preserve unresolved source channels for future
work rather than forcing them through a guessed interpretation.

Prefer a simple, truthful approximation over a complex but incorrect decode.

## 12. Validate visually and structurally

Every visual-system promotion should include:

1. an isolated close-up witness for geometry, material, UV, and normal review;
2. a representative gameplay-scale composition;
3. a matched control that changes only the feature being evaluated;
4. an image or difference view that makes the contribution visible;
5. a deterministic replay with identical output hashes;
6. regression checks for previously accepted systems;
7. explicit evidence at every supported gameplay zoom;
8. representative variation across material, terrain, unit, culture, era, or
   object families as applicable.

Tests and hashes prove reproducibility and contract stability; they do not by
themselves prove that an image looks good. Direct inspection remains necessary.
Conversely, an attractive screenshot does not prove correct metadata,
determinism, animation, exclusions, or integration behavior. Both forms of
evidence are required.

## 13. Port methods and contracts, not screenshots

The runtime should consume generic normalized meshes, materials, textures,
metadata, and scene contracts. Do not bake a Lab screenshot into the game or
introduce source-specific runtime branches.

During integration, preserve existing responsibilities that are outside the
visual replacement:

- authoritative game state and screen anchors;
- visibility and fog;
- overlays, labels, selection, and UI;
- animation timing and action state;
- caching, invalidation, scrolling, and dirty bounds;
- native fallback for unsupported content;
- removal and transition behavior.

Only the proven visual path should change. A newer experiment must not
implicitly replace a previously accepted system merely because its file or
revision number is later.

## Recommended investigation order

When an asset looks soft, distorted, or materially wrong, investigate in this
order:

1. Confirm the correct source asset and all composition metadata were selected.
2. Confirm geometry, component relationships, proportions, and macro height.
3. Confirm UVs and per-material repeat/clamp addressing.
4. Confirm vertex-normal encoding, tangent basis, and transformed normals.
5. Confirm texture formats, color spaces, channels, and mip chains.
6. Confirm all expected material channels are bound and sampled correctly.
7. Confirm every object uses the same world, height, lighting, and shadow basis.
8. Confirm opacity-tested geometry casts and receives appropriate shadows.
9. Confirm variation is source-backed, deterministic, and nonrepeating.
10. Confirm MSAA, anisotropy, render scale, mip bias, and final reconstruction.
11. Only then consider restrained artistic calibration or synthetic detail.

## Common failure patterns

| Symptom | Likely causes |
| --- | --- |
| Soft or smeared terrain | Early downsampling, missing mip detail, weak anisotropy, flattened height, incomplete material stack |
| Barely visible hills | Flat geometry, insufficient macro height, no footprint variation, rock material applied without relief |
| Flat mountains | Material textures used without macro height, lost normals, incomplete top/snow/rock blending |
| Repeating rocky patches | One stamped decal or seed reused for every hill |
| Metal or color stretched across skin | Repeat-addressed atlas sampled with clamp, damaged UVs, wrong material binding |
| Incorrect animated limbs | Missing mesh-local skin-palette remap, mismatched skeleton, incorrect pose binding |
| Faceted or plastic objects | Authored smoothing normals discarded, incorrect normal transform, guessed normal-map decode |
| Trees look like blobs | Procedural crown replacement, missing opacity mask, incorrect addressing, excessive ambient fill |
| Detached or contradictory shadows | Different caster/receiver coordinate bases, different light vectors, wrong height units, proxy blobs |
| Sharp but noisy or shimmering output | Excessive negative mip bias, oversharpening, missing temporal stability, insufficient antialiasing |
| New layer regresses an old one | Retained pass redraws obsolete geometry, revision selected by recency instead of explicit acceptance |

## Final rule

Before adding sharpening, noise, contrast, or procedural detail, prove that the
renderer has preserved the source silhouette, material channels, normals, UV
addressing, texture resolution, lighting basis, shadow geometry, and final
sampling path. Most apparent softness is lost or misinterpreted information
upstream, and no final-image filter can faithfully restore it.
