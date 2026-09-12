# Mountain material diagnosis

## Current body-grain study (2026-09-12)

The user subsequently accepted the final `collar-triplanar` result and requested
production use. The shared mountain shader now contains that treatment with
ordinary captured volcano ownership replacing the study's fixed coordinates.
The frozen experiment below remains reproducible. `verify_body.py` compares the
current production candidate against its four accepted views; technical/staging
receipts and rollback inputs live in `lab/out/mountains/body-promotion/`.

`body_study.py` isolates added rock contrast using the frozen volcano Lab
candidate and shaders behind the user's comparison image. Its control reproduces
the earlier `skin-lava-shadow-probe` gameplay image exactly. Concurrent volcano
runtime work is excluded. This is a visual experiment, not current-code runtime
verification or staging. The frozen candidate differs from the turn-start
checkout in renderer/shadow compiled inputs; every comparison uses that same DLL.

The `balanced-rock` trial retains 28% of the extra grain/crevice contrast on the
body, but restores the old response near the snow line. The user's follow-up
requested the original rocky base and less upper grain. `slope-rock` therefore
preserves the response below 0.38 world units of rise and smoothly reduces added
contrast through 0.75 units, retaining 8% above that. It does not restore the
extra contrast at the upper snow transition. Full snow's color multiplier,
snow masks, source textures, broad/fine normals, geometry, lighting and terrain
coverage remain unchanged. These are C3X artistic controls, not source-engine
equations. Normal-only reduction did little to remove the grain; added color
and crevice contrast dominate it. Full removal looked too smooth in the first
body-only control.

With a Python containing Pillow, reproduce and arrange the actual D3D frames:

```sh
python3 Renderer/lab/studies/mountains/body_study.py --variants current balanced-rock slope-rock
python3 Renderer/lab/studies/mountains/body_study.py --category mountains --cases coastal detail --variants current balanced-rock slope-rock
python3 Renderer/lab/studies/mountains/body_study.py --variants current slope-rock --hour 8
python3 Renderer/lab/studies/mountains/body_study.py --variants current slope-rock --zoom 128
python3 Renderer/lab/studies/mountains/body_compare.py
```

All output and frozen provenance live in `lab/out/mountains/body-study/`.
The seed is the retained `lab/out/volcanoes/material-study/`; preserve it to
reproduce this exact comparison. Ordinary mountain cases strip the fixed volcano
probe; volcano-context cases preserve it. Linked asset files remain read-only.
The script never writes production shaders, stages a DLL or replaces references.

### Lower-slope texture projection

The user liked `slope-rock` but identified the vertically stretched base in the
morning view. `ground_material` samples all grass/plains/tundra/desert color,
height and specular channels in world XY, even as the unified mountain surface
rises steeply. The `collar-no-bump` control leaves the curtain visible, whereas
`collar-triplanar` substantially removes it. This confirms a material-projection
problem rather than a stretched mountain mesh.

`collar-triplanar` retains `slope-rock`, all existing terrain coverage and source
texture scales/rotations/offsets. It blends the 20 fine terrain/hill material
lookups toward three-axis projection on rising, steep ground portions. Flat
ground keeps its original XY samples and the broad terrain color field remains
world aligned. All color/height/specular channels use the same mapping; the
upper rock and snow path is unchanged. The diagnostic volcano footprint remains
protected. This is a Lab proposal, not a recovered source shader.

```sh
python3 Renderer/lab/studies/mountains/body_study.py --variants collar-no-bump collar-triplanar --hour 8
python3 Renderer/lab/studies/mountains/body_study.py --variants collar-triplanar --category mountains --cases coastal detail
python3 Renderer/lab/studies/mountains/body_study.py --variants collar-triplanar --zoom 128
python3 Renderer/lab/studies/mountains/body_compare.py
```

## Earlier snow and banding study

Lab-only comparison of the current Civ V Environment Skin mountains. Run with a
Python containing Pillow:

```sh
python3 Renderer/lab/studies/mountains/study.py
```

The script uses `Renderer/renderer.py` category scenes and the production D3D11
renderer through the Windows VM. It requires a matching candidate build, copies
that DLL, and adapts shared shader source in an isolated root under
`Renderer/lab/out/mountains/zebra-study/`. Production shaders, staged DLL, source
art, macro geometry, and fixed references are not modified. Linked pack files
are read-only inputs; never edit those files through the study root.

## Source evidence

The installed Civ V skin's Windows `TerrainMaterialSet_Base.blp` resolves the
three ordinary mountain materials as MTN_BASE, MTN_TOP and MTN_SNOW. Re-extracting
the nine color/height/specular channels gives byte-identical matches against
`Civ5EnvironmentSkin` and the hashed `NaturalFidelityRuntime` textures. All nine
are 2048×2048. Colors are BC3 sRGB; heights and specular are BC4 linear. The skin's
base and top height/specular pairs are identical. The top color adds conspicuous
white patches; it is not another plain gray rock material.

The skin's `ArtDefs/TerrainStyle.artdef`, collection `RidgelineMountain`, calls
MTN_TOP `MountainTopLowMtl` and MTN_SNOW `MountainTopHighMtl`. It specifies
MountainHeight 32, SnowLowHeight 24 and SnowHighHeight 26. The corresponding
normalized ratios are 0.75 and 0.8125. These names, values and texture bytes are
confirmed source evidence. The exact engine blend equation and slope response
are not recovered.

C3X currently weights the patchy-white MTN_TOP using final rise
`smoothstep(0.08, 0.62, mountain_rise)`, making it the dominant material far below
the summit. Its separate snow mask begins at normalized source height 0.79 and
also rejects steep faces. This leaves patchy white rock across broad faces and
little continuous peak snow. The retained category description that calls
MTN_TOP simply "upper rock" misses its snow content.

## Controlled variants

- **current:** unchanged shared mountain shader and current skin.
- **no-top:** only sets the patchy-white top weight to zero. Existing snow,
  normals, extra crevice/grain contrast, ground coverage and geometry remain.
  This diagnoses the material mask independently of texture-detail strength.
- **summit:** confines the top blend to source heights 0.60–0.78, with snow at
  0.76–0.90 and less restrictive slope gating. This is a C3X visual proposal,
  informed by source layer roles; it is not a recovered Civ VI shader.
- **snowcaps:** a fuller-cap alternative, using a top blend at 0.52–0.68,
  snow at 0.62–0.78, and a 0.02–0.25 slope gate. The geometry, source textures,
  normal strength, crevice/grain response and accepted ground blend are intact.
- **no-extra-contrast:** removes the added crevice and fine-color multipliers
  while preserving the current material masks, as a separate contrast control.

Generated render receipts retain image, shader and candidate hashes. Images are
synthetic category fixtures, not screenshots from a launched game. Nothing in
this study grants visual acceptance, staging, or reference replacement.

## Current comparison results

The no-top control removes the conspicuous white mottling while retaining the
existing rock detail; the source layer coverage is the primary demonstrated
cause. Both summit proposals preserve the gritty Civ V skin. The fuller-cap
option makes the white peak treatment more visible while retaining gray faces.
The proposals deliberately preserve the existing crevice/grain response; the
optional no-extra-contrast control is available but was not needed for this
finding and has not been rendered in this study.

Current, no-top and summit were rendered in detail, gameplay and coastal cases
at zoom 128. Current, summit and snowcaps were also rendered in detail and
coastal cases at zoom 224; snowcaps has the three zoom-128 cases as well. These
18 D3D frames report zero fallback tiles. The existing natural and source-fidelity
contract suites pass 31 tests. No visual promotion or game launch was performed.

Generate paired review sheets with:

```sh
python3 Renderer/lab/studies/mountains/compare.py --proposal snowcaps
python3 Renderer/lab/studies/mountains/compare.py --proposal snowcaps --zoom 224 --cases detail coastal
```

`--reuse-candidate` continues using the study's copied DLL when unrelated native
work is underway. Never substitute a different DLL halfway through a comparison.
The normal category preparation guard found pre-existing generated-file drift;
this study generated bindings in its private root rather than altering those
files. The candidate was compiled using `BUILD.bat candidate-compile`; no injected
C was changed or compiled. All experiments and comparison outputs remain in Lab.

## Remaining dark-band investigation

The first snow-layer finding explains white mottling, **not the remaining dark
horizontal lines**. The user's follow-up prompted these matched controls, all
using the previous `snowcaps` Lab proposal and the same isolated DLL:

- Removing added crevice/grain multipliers leaves the horizontal ledges visible.
- Removing cast/self shadows leaves them visible.
- Disabling the texture-driven normal perturbation largely removes them.
- Constant-gray material with geometric normals shows smooth macro surfaces,
  rather than the strong terracing seen with the original material response.
- Rotating side projections, correcting derivative blending alone, or increasing
  rock repetition by 3×/5× does not adequately remove the pattern.
- Reducing the broad height response while retaining the finer source-height
  contribution gives the strongest useful improvement without flattening the
  actual mountain geometry or removing the gritty source color/crevice detail.

The `micro-relief` proposal retains the prior snow masks, textures, geometry,
lighting and ground-coverage masks. Broad material-height derivatives use 0.04
of the previous amplitude; the existing finer 3.7× height sample retains its
0.12 contribution. Grain/color/crevice/specular inputs remain unchanged. The
`micro-relief-balanced` alternative uses broad amplitude 0.08 and fine amplitude
0.08. Neither changes the macro height field, tessellation, or silhouette.

`height_derivatives.hlsl` differentiates each projection before weighting it and
blends the resulting derivatives, avoiding artificial gradients from changing
projection/layer weights. This is a principled C3X shader proposal; the
`gradient` control shows that this correction alone is not the main visual fix.
These amplitude choices are visual calibrations, not recovered Firaxis values.

The main comparison is **previous snowcaps → micro-relief**, in detailed and
coastal zoom-224 frames and coastal/gameplay zoom-128 frames:

```sh
python3 Renderer/lab/studies/mountains/study.py --reuse-candidate --variants micro-relief --cases coastal detail --zoom 224
python3 Renderer/lab/studies/mountains/compare.py --control snowcaps --proposal micro-relief --cases coastal detail --zoom 224
```

Every new diagnostic/proposal shader compiled and rendered through D3D11 with
zero fallback tiles. During diagnosis, shader changes remained confined to the isolated study;
production code and fixed references were unchanged.
The previous snow-cap control and micro-relief coastal zoom-224 proposal were
rerendered and reproduced their prior BMP hashes exactly. The follow-up index
is `banding-review.json` (16 diagnostic/proposal frames); the repeat check is
`banding-repeat.json`, both under the study output directory.

## Production acceptance

On 2026-09-09 the user accepted fine rock relief and requested production use.
The shared mountain shader now contains the exact `micro-relief` equations
(with neutral production helper names). Normal `renderer.py lab mountains`
now renders that accepted source. Historical variants need the prior source:

```sh
python3 Renderer/lab/studies/mountains/study.py --reuse-candidate --baseline-shader Renderer/lab/out/mountains/promotion/prior-mountain.hlsl --variants snowcaps micro-relief --cases coastal --zoom 224
```

The retained prior source is byte-identical to the mountain shader in Git
`f2696829d9bc549ae01597cf42d25f3ddab736de`; it can be recovered from Git if the
ignored output is cleared. The study command itself still never stages a DLL.
Promotion uses an isolated copy of that committed baseline plus the accepted
shader, excluding the navigation task's uncommitted C++ changes. Its verification
receipt and rollback DLL are retained under `lab/out/mountains/promotion/`.
