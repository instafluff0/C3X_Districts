# Unit size and sampling study

Candidate only. The user requested that production remain unchanged pending
approval. `UnitAnimationFidelity`, `UnitAnimationRuntime`, staged DLLs and fixed
references are not study outputs. `UnitQualityStudy` is a separate local pack,
selected only by the three explicit Lab cases below.

```sh
python3 Renderer/renderer.py lab units --case sizing
python3 Renderer/renderer.py lab units --case sizing-gameplay
python3 Renderer/renderer.py lab units --case sizing-move
python3 Renderer/renderer.py test units
python3 Renderer/renderer.py integration units --renderer-only
```

The sheets are native 1200x880 captures at tile widths 128 and 192 (true 1x and
1.5x projection, not image enlargement). Columns are Warrior, Spearman, Pikeman,
Archer, Settler, Worker. Rows are current fit, anatomy fit, and anatomy fit with
2x material supersampling. All rows use the same pose, time, color and output
projection. Idle uses source time zero; movement uses native cursor 7/16.
The neutral and gameplay sheets differ only in background/scene context.

The current shared sampler remains 16x anisotropic with zero mip bias and 4x
MSAA. Optional per-unit binding `sample_scale: 2` doubles scratch width/height
and box-reconstructs scene-linear color and coverage before display transfer.
Omission retains scale 1. Neither native sprite dimensions nor cache limits
change. This does not claim equivalence to Civ VI's antialiasing.

## Sizing result

The old fit used the whole kit, including long weapons and buried inventory.
The explicit `sizing.json` anatomy sets exclude weapons, shields and headgear.
The first idle pose's vertical anatomy extent provides scale and ground; the
Warrior's existing scale anchors the comparison. Source member-size ratios
(1.22 for these melee figures, 1.10 for the other three) remain proportional.
This is an authored C3X calibration, not recovered Civ VI engine sizing.

| Unit | Old anatomy height | Candidate | Uniform scale multiplier |
| --- | ---: | ---: | ---: |
| Warrior | 44.5 px | 44.5 px | 1.000 |
| Spearman | 31.6 px | 44.5 px | 1.412 |
| Pikeman | 18.6 px | 44.5 px | 2.399 |
| Archer | 34.7 px | 40.2 px | 1.156 |
| Settler | 41.6 px | 40.2 px | 0.966 |
| Worker | 46.1 px | 40.2 px | 0.870 |

These are vertical anatomy extents projected using the renderer's Z scale, not
full screen bounding boxes. The Archer's old ground reference was the bottom of
its stowed bow, which raised the body. Anatomy-ground fitting removes that lift
and allows the existing ground clip to hide the stowed geometry.

The canvas is explicitly 320x320 at normal zoom for this study. It is larger
than the old 191x191 diagnostic. Sweeping every stored animation frame and all
eight directions, including triangle/ground intersections, requires symmetric
canvas sides up to 288 px for Spearman movement and 266 px for Pikeman idle/
fortify. Archer movement reaches 194 px. Thus this pack cannot simply replace
the production pack: integration must reconcile visible bounds and dirty regions
with the authoritative native Sprite dimensions first. Runtime clipping and
fallback guards remain intact. No new patch-table entry is requested.

Six captures (three cases at two zooms) completed with 18 draws each. The two
neutral sampling comparisons change 4,268 and 9,069 pixels respectively between
the anatomy rows, demonstrating an active sampling change at equal output size.
These counts are not a perceptual quality score. The dominant visible change is
the corrected sizing; supersampling is a subtler reconstruction improvement.

Verification: 114 category tests and 194 expanded tests passed. Native day and
night witnesses passed action timing, cached translation, repeated cursors,
RGB555/RGB565 clipping, magenta-underlay compositing, config-off and retained
terrain checks. The combined current-checkout receipt remains uncertified because
another task changed shared implementation inputs during verification. These
passes do not certify the larger study anatomy against native game dirty bounds.

Production isolation: this task issued no staging/install command, and the
production unit catalogs retain their original hashes. During verification the
separately authorized shadow/cliff deployment replaced the staged DLL with
`dd7b9d5baef6f17fe8475969a8feae87e2d873233e9ec9f294c8e263f7763435`.
Its preserved compiled-source snapshot includes the optional `sample_scale`
support added here. That support is dormant with the unchanged production unit
bindings; experimental sizes and 2x sampling remain selected only in the study
pack. This task did not overwrite or roll back the other task's deployment.

`prepare.py` independently copies unchanged payloads and textures, changes only
study aliases, fit/ground and sampling metadata, and writes detailed sizing and
all-action envelope evidence to the isolated pack's `study.json`. The initial
native fixture row-index error was corrected before the six successful renders.

## Source material investigation

```sh
python3 -m Renderer.lab.studies.units.source_probe --disassemble
```

The archive probe uses the Python standard library. `--archive PATH` overrides
the local installed archive. The optional disassembly step uses the Windows VM and the
generic offline `disassemble.cpp` tool. Bytecode, disassembly and hashes remain
local under `lab/out/units/source-shaders/`.

The installed DX11 archive contains 932 DXBC shaders; 28 bind the object-material
`g_LeanMap0`/`g_LeanMap1` pair. This is distinct from the 42 water variants using
uppercase `LEAN0`/`LEAN1`. Inspected object shader offset 1835904 confirms:

- `g_LeanMap0.rg` is remapped by `2*x-1`, with positive Z reconstructed as
  `sqrt(max(0,1-dot(xy,xy)))`, then combined with the supplied tangent basis.
- The second map's red channel is multiplied by `g_LeanInfo.z` and a clamped
  UV-derivative footprint, then added to two specular distribution parameters.
  It does not multiply the first map's normal strength as the old approximation
  did. The footprint uses UVs multiplied by `g_LeanInfo.xy`; the engine values
  supplied to that constant are not yet recovered.
- The inspected shader consumes three-channel `g_Roughness`, metalness,
  environment illumination and two distribution terms. C3X's single gloss
  channel and GGX approximation do not establish material equivalence.

This proves equations in an installed object shader family. Exact unit shader
permutation selection, specular preprocessing and engine constant values remain
unproven. The isolated comparison below implements the recovered first-map
normal and positive-alpha tint equations without claiming full shader parity.

Archive SHA-256: `9d511ac04639936ada55e36d3bc6224e5fc514b16299a2c0f683c6e0d079c469`.
Shader SHA-256: `e6115bacfa1e76b156106be0d89810b03445b3b4b797697887a8ba585765df4d`.
The repeatable probe records every selected shader and disassembly hash.

## Isolated material comparison

```sh
python3 -m Renderer.lab.studies.units.frame_probe
python3 -m Renderer.lab.studies.units.material_study --render
python3 -m unittest Renderer.lab.studies.units.test_material_study
```

These two material preparation tools require NumPy. They re-extract only the
Warrior into ignored Lab output; they do not overwrite the normalized art packs.
`material_study.py` copies the current C++/shader inputs into a private source
snapshot, applies guarded experimental patches there, builds a separate DLL and
preview, and renders using `UnitMaterialStudy`. No shared runtime source, build
candidate, staged DLL, reference or production unit binding is a write target.
Source-input hashes identify the snapshot; the standard fixture's inherited
“production”/“families” log text means it exercised the native body API, not that
this experimental DLL was promoted or that 18 distinct units were rendered.

The six sheets are under `lab/out/units/material-study/`: neutral, terrain-context
and movement at 128/192 tile widths. Each has six Warrior headings and three
rows, all with identical body size and 2x material sampling:

1. Current shading.
2. Current shading plus authored tangent-space surface normals.
3. Surface normals plus the installed tint-family equation.

`material-comparison-1to1.png` crops three headings from the 192 sheet without
resizing any rendered pixels. Displaying it at native size avoids the original
2x-image-enlargement problem. Normal detail is a smaller visible improvement at
map scale; the corrected blue cloth produces the most obvious material change.

### Recovered frame evidence

Eight object skinned vertex variants are now extracted by `source_probe.py`.
Inspected variant 2086936 decodes the position's packed normal, then independently
oct-decodes `TANGENT.xy` and `.zw`. It transforms and normalizes the two resulting
directions separately. The source pixel shader combines map X with the first,
map Y with the second and reconstructed positive Z with the mesh normal.

The five actual Warrior primitives contain 1,601 vertices. Their recovered
position/UV identity matches the normalized imports. Packed tangent pairs occupy
bytes 20–23 in the 32-byte skin profile and bytes 12–15 in the 24-byte rigid
profile. Offset identification is empirical package evidence, not a DXBC input
layout reflection claim. Mean agreement with UV derivatives is +0.900 to +0.957
for the first direction, and **-0.895 to -0.963** for the second. Thus simply
using a conventional positive-V derivative would invert the second direction.
The pair's mean absolute dot is below 0.005 on all five primitives. Some authored
normals are not exactly perpendicular to this pair; the experiment preserves
those source directions instead of rebuilding or orthogonalizing them.

A private diagnostic `C3XANM2` payload appends both authored directions to each
vertex. Position, normal, UV, influences, indices and palettes remain byte-for-
byte unchanged. Hair retains its existing 1.1 uniform component scale. Tangents
follow the same native animation phase and source linear skin matrices, then the
existing C3X object-lighting basis. The existing mesh-normal inverse transpose
and source linear transform differ by at most 0.000111 degrees on the sampled
Warrior idle/movement poses. The inspected VS uses three skin influences, while
some imported Warrior vertices have a fourth weight up to 0.07451. The diagnostic
preserves all four existing influences; it does not assume this VS permutation
is the exact live unit binding or change animation geometry to imitate it.

### Tint evidence

Installed object PS 2169080 begins with an unambiguous positive-alpha equation:
`albedo = base.rgb * lerp(1, tint.rgb, base.a)`. VS 2086936 supplies the instance
tint to the matching output semantic. Our current shader instead multiplies
all base RGB by the component tint and uses smoothed **inverse alpha** for
owner-color modulation, including an extra `.45 + color * 1.10` remap.

The third row uses the positive-alpha equation and substitutes Civ III's display
color for the tint of the Warrior's owner-colored component. This turns the
previously gray chest/cloth region blue while preserving its texture detail.
It is evidence of a concrete material interpretation mismatch, not a claim that
Civ III's palette value or C3X tone mapping equals Civ VI's full color pipeline.
The experiment retains current AO, gloss/GGX, lighting, exposure and shadows;
it does not guess the second LEAN map's missing specular constants.

All six native material sheets passed 18 body draws each. Four targeted tests
passed: native old/new payload and pose parity, corrupt payload rejection,
analytic tangent rotation/interpolation, source-frame identity/orientation,
equal size/sampling, unchanged baseline payloads and production artifact hashes.
The private new decoder and fixture are intentionally not a production API.

Remaining before production promotion: apply an approved material result through
the generic maintained pipeline, validate the broader roster, and reconcile the
larger Spearman/Pikeman animation bounds with native dirty regions. The second
LEAN/specular pipeline and exact source engine binding remain research limits;
none is silently represented as solved by these normal/tint comparisons.

## Six-unit extension and independent studio

`material_study.py --roster --render` now extends the private material experiment
to Warrior, Spearman, Pikeman, Archer, Settler and Worker. The new pack is
`UnitMaterialRosterStudy`; all three rows use the same corrected anatomy fit and
2x sampling. `source-roster-frame-pack` preserves 32 recovered components and
9,716 vertices. All source tangent pairs retain positive-U / negative-V
orientation; mean derivative agreement stays above 0.80 in magnitude. Tangent
pair orthogonality, distinct offset selection and source/payload identity are
validated without forcing authored normals perpendicular to the pair.

All six native roster sheets passed 18 draws. Four dedicated roster tests check
all generated payloads, native idle/move decoding across all six units, source
frames, and identical anatomy size/sampling across rows. A build-route error was
caught before rendering and corrected; two VM dispatch errors were retried only
after Windows confirmed the preview process was absent. These are still private
snapshots, not game-dirty-region certification.

The user's subsequent request for an unconstrained reference is implemented in
[the independent unit studio](STUDIO.md). That experiment is separate from the
DLL and uses the retained source-family dual-lobe specular evidence. It supplies
a high-resolution reference and final-reconstruction comparisons, rather than
assuming that the smaller material differences above establish visual parity.
