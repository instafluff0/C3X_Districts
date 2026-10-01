# Matching the selected autumn scene

The selected [autumn concept](../../out/seasons/scene-concepts/fall.png) is the
target. Both `refined.png` and `beauty.png` were rejected. The next experiment
must change how the crowns and ground read, rather than continue tuning the
same color multipliers. This is a proposed Lab implementation sequence, not a
new rendered result or a claim of visual acceptance.

## What the comparison actually shows

| Element | Target | Rejected candidate | Required intervention |
| --- | --- | --- | --- |
| Deciduous crowns | Rounded gold/amber volumes; bright tops and darker warm interiors | Shorter-looking, fragmented orange/yellow patches | Audit existing proportions; separate crown illumination from leaf detail; remap values as well as hue |
| Grassland | Bronze/olive turf with golden dry patches and textured edges | Fairly uniform gray/yellow olive grain | Stronger variation at turf-patch scale while preserving source fine texture |
| Plains | Pale honey/straw with a drier, lighter appearance | Too close to grassland's yellow cast | Independent calibration of color and pattern contrast |
| Hills and rock | Bright neutral stone, readable fissures, warm turf between exposures | Green/yellow casts reduce stone/vegetation separation | Protect stone and reveal existing relief channels |
| Water and shore | Deep blue, fine sparse glints, soft irregular foam fringe | Repeated rope-like waves, broad pale reflection, outlined shores | Correct the Mac witness against the actual Summer capture |

These are visual observations, not recovered upstream engine behavior. The
concept also changes some crown silhouettes and ground marks. Use their volume,
lighting and material character as targets; do not project the concept onto the
map or substitute generated trees.

## Confirmed inputs and a proportion issue

- The selected forest uses 22 cached Civ V bodies: five pine bodies, two shrubs,
  twelve individual leafy variants and three leafy clumps. Their selected
  channels are color, paired slope/LEAN textures, gloss and, for leafy bodies,
  opacity. These materials do **not** supply separate AO textures; do not assume
  an unused AO map is available to fix their lighting.
- Inspected leaf color atlases are 2048 square; grassland/plains albedo and
  grassland height are 4096 square. Additional giant textures are not the first
  solution.
- Source forest meshes retain authored smooth vertex normals and stacked crown
  surfaces. Seasonal masks must follow the current working UV addressing,
  including wrapped coordinates where present. Descriptor defaults previously
  caused solid crowns and must not be blindly reapplied.
- Existing forest-floor footprints carry authored color/height and follow tree
  centers. They are the first carrier for fallen-leaf treatment.
- `TREE_HEIGHT_SCALE = .50` in
  [the compiler](../../../native/source_fidelity/prepare.py) halves forest height
  without narrowing the footprint, and transforms vertex normals accordingly.
  [The forest category](../../categories/vegetation/forests/README.md) documents
  this as a C3X visual choice. The current Summer and autumn study inherit it.

Height compression is a concrete candidate explanation for the flattened crown
impression, not proof that one scale adjustment will reproduce the concept.
Cached packs and preserved seasonal inputs suffice for the first experiments;
no further source extraction is required.

## Ordered experiments

### 1. Test the existing proportions in a controlled witness

Keep the same `test.biq` crop, camera, shared sun direction and body placements.
Keep the real Summer capture beside the Mac Summer preview so diagnostic
water/shore differences remain visible.

Render representative individual leafy, clump, shrub and pine bodies. Compare
the existing .50 vertical conversion against .65 and .75, with source proportions
at 1.0 as a diagnostic bound. Read original cached body data and apply the
conversion in the study adapter, with corresponding inverse-transpose normals,
grounding and shadow bounds. Hold X/Y footprint, body selection, placements and
opacity fixed; inspect river/building exclusions and hill grounding. Do not
clone a terrain pack or invent a tree model.

This is a bounded same-mesh proportion experiment. If fuller proportions improve
the match, record a shared vegetation calibration candidate: trees must not grow
when seasons switch. Keep the existing-height control throughout subsequent
material experiments. No production forest change or Summer/Winter reference
replacement is part of this study.

### 2. Give each existing canopy volume and luminous gold

Replace increasingly strong RGB multipliers with a leaf-only value/color
transform. Build an explicit tissue mask from atlas regions and mesh/UV evidence,
using chroma as fallback rather than the sole classifier. Preserve bark, opacity
and evergreens. Separate slow source luminance variation from fine residual
detail; map the slow component through ochre-shadow, amber-mid and gold-highlight
ramps, then restore source residual detail. Use linear light and a smooth
highlight shoulder so bright leaves retain shading instead of clipping to flat
yellow.

Supply compact crown metadata with existing instances: crown center, radii,
local height and, for clumps, several crown lobes inferred from the original
mesh. Evaluate broad crown irradiance using the **same sun and sky** as the
scene, alongside retained mapped-normal response. This should establish a bright
upper/near-sun surface and shaded side across the whole crown while original
leaf normals supply granular detail. Keep original normals and an unchanged
irradiance control; do not replace normal maps with a sphere or bake one fixed
sun direction into the atlas.

Try restrained leaf transmission only where tissue and exposure permit it.
Shadowed illumination comes from the common environment; it must not become
emissive fill or brighten bark/deep occluded interiors. Blender can inspect or
bake metadata from existing meshes without generating trees. Shared UVs cannot
uniquely encode crown position, so store positions/lobes per body/instance or
vertex rather than solely in a UV mask.

Make coherent gold-dominant crowns, a smaller amber share and sparse russet,
with small internal variation and dark pine contrast. Judge **visible leaf area**
distribution: recipe probabilities alone do not establish scene color balance.

### 3. Rebuild ground treatment at three scales

Separate broad color, intermediate turf pattern and fine texture. Prior fine
pattern correlations established surviving grain while allowing a nearly
uniform intermediate appearance.

- Preserve fine source luminance, height, normal, cavity and gloss detail.
- Use source grass/plains color and height features to reveal dry turf patches:
  bronze/olive recesses and honey-gold exposed blades. Retain each biome's own
  pattern instead of spreading the plains material everywhere.
- Organize those patches with restrained larger world-space variation, eligible
  by biome weights, shore/river distance and existing vegetation coverage.

At this camera, inspect roughly 2-6, 8-30 and 40-100 pixel feature bands as
measurements, not screen-space stamping coordinates. Placement remains
world-defined, deterministic, wrap-safe and mip-filtered. The target's turf
strokes motivate the middle band; do not trace its looping AI marks or replace
gritty source texture with generic noise.

Calibrate grassland to bronze/olive/gold; plains to lighter honey/straw;
floodplain to moister muted olive; desert to warm pale sand; tundra to cooler
gray stone and sparse vegetation. Compare both colors and patterns side by side
in the five-biome scene. Remain within the selected restrained autumn palette,
rather than returning to the rejected rich-green variant.

### 4. Connect forest floor and exposed rock to that treatment

Start with actual source floor albedo/height. Selectively remap litter features
to gold/amber and reveal tiny relief through their existing height response.
Strength follows placed deciduous crowns: stronger underneath and near edges,
sparse beyond, absent in channels and on steep exposed stone. Preserve soft
footprint alpha and dirt gaps; avoid orange carpets and independent colored
disks under trees.

Retain rock geometry, normals, height and stone blend coverage on hills. Grade
vegetation separately from stone. If rock remains muddy, inspect intermediate
height/cavity contrast and filtering before increasing global sharpness.
Neutral sunlit facets and cooler shaded fissures should reveal the existing
relief beside golden turf. Foliage lighting must not warm every stone surface.

### 5. Restore warm/cool balance and correct the Mac witness

Use one coherent environment with restrained warm sunlight and cooler skylight,
then tune material response. Transmission, terrain highlights and cast shadows
share that environment. Evaluate any environment adjustment separately from the
leaf transform; avoid a whole-frame orange filter and protect night behavior.

Correct water/shore differences using cached slope textures and shared equations:
reduce repeated large-wave contrast, localize/thin glints, retain deep blue
outside them, and recover the actual Summer capture's soft irregular foam
fringe. Preserve river/coast fields and widths. These are preview/provider
corrections; production retains its shared motion and reflection providers.

### 6. Tune against the reference, then prove preservation

Use four fixed crops: crowns beside pines, grass/plains, rocky hills and shore.
Compare target, actual Summer, unchanged-height seasonal control and cumulative
candidate at identical display sizes. Add one intervention at a time, then
evaluate the whole `test.biq` scene; a successful swatch is insufficient.

Use semantic reference regions aligned by stable map landmarks to compare
palette area, highlight/mid/shadow values, and ground variation at several
spatial scales. Whole-image pixel error is unsuitable because the concept
changed details and some silhouettes. Metrics guide calibration; the actual
side-by-side scene establishes whether the result is closer. Texture correlation
and biome distance remain separate preservation/readability evidence, not beauty
scores.

The next cumulative scene should show rounded luminous gold crowns, dark pines,
bronze/olive turf with golden patches, lighter straw plains, neutral detailed
rock and a quieter blue coast. If a crown still reads as an orange patch, inspect
proportion and broad irradiance before adjusting another saturation slider.

Finish with five-biome, reduced-zoom, dusk/night and wrap witnesses; check
bark/evergreen protection, opacity, source integrity, relief, finite normals and
disabled-policy behavior. The unchanged-geometry material control must retain
Summer/Winter within the existing recorded tolerance. Proportion experiments
have separate Summer/Winter controls and are reported separately, never silently
invalidating that guarantee.

## Implementation and storage boundary

Remain in this study on the Mac. Extend `render.py`, `scene.mm`, `lab_geometry.h`,
`seasonal_policy.hlsl` and `fall_review.py` as needed, with a small offline helper
for source-derived masks/crown metadata. Use generic recipe/pack semantics and
no source-game branches in runtime shaders. Preserve the instance, shadow and
reflection contracts in `PLAYBOOK.md`.

Reuse cached textures in place. Use procedural ramps/small lookup tables instead
of three complete seasonal copies of 2048/4096 textures. Budget new masks,
metadata and optional floor detail below 12 MiB, bounded reviews below 20 MiB;
overwrite/remove temporary outputs after each run. Preserve protected source
inputs and the selected concept. Keep off the Windows VM and leave production
sources, staged DLLs and references untouched.

First priority: existing-tree proportions plus crown material and irradiance.
Second: intermediate ground pattern. These are the largest expected gains;
particles, more art extraction and color-only full-scene iterations should wait.
