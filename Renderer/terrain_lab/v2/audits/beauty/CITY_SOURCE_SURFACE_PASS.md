# City material restoration r31–r33

The combined medieval candidate has slightly clearer masonry/timber edges and
roof highlights. This is a modest material gain, not the requested final city
quality. Modern windows and the lake reflection remain visible. Preserve r28,
r29/r30 and the capital composition candidates; none of the existing visual or
milestone gates advances. Single-era selection and deferred connecting roads
remain unchanged.

## Installed shader evidence changes the approach

The installed DX11 shader library contains inspectable rigid-model vertex and
pixel shaders. `qa/city_shader_material_probe.py` preserves two bounded local
shader containers, checks their stages and instruction witnesses, and writes
[source findings](CITY_SHADER_MATERIAL_SOURCE.json). Raw bytecode/disassembly
stays in ignored local output. The precise active city permutation and constant
bindings are still unproven; these findings concern the inspected shader family.

- The vertex shader decodes octahedral normals from the packed position payload
  and two octahedral tangent directions. For the tested static 24-byte profile,
  normal bytes are 6–7, tangent bytes 12–13 and bitangent bytes 14–15. The previous
  investigation's failure to find perfect orthogonality did not justify
  discarding the authored frame. The source shader normalizes the directions
  independently after the model transform.
- `normal_0` is sampled as signed XY and used with reconstructed positive Z.
  The earlier derivative/slope approximation was not the source diffuse-normal
  calculation. The optional adapter now transports the actual frame and applies
  this reconstruction.
- The cooked RGB `Generic_Gloss` texture feeds `g_Roughness`. Its first two
  channels control two specular lobes; the third supplies a broad component and
  participates in dielectric reflectance. Treating its red channel as a scalar
  gloss slider was not supported. The texture's stored sRGB format is preserved.
- `normal_1` supplies scalar variance, scaled by `g_LeanInfo` and UV screen
  derivatives. That scale remains unresolved, so this pass does not invent it.
- The source also samples a roughness-filtered environment cubearray and evaluates
  structured point/spot lights. The current Lab material lacks those contributions.

The existing `--surface-detail` experiment remains a separate unselected
approximation. New `--source-surface normal` applies the authored frame and normal
texture; `--source-surface lit` also applies a bounded adaptation of the source
direct specular calculation with the shared sun/moon and shadow visibility.
Its view vector comes from the preserved Civ III orthographic projection.
This partial adapter does not reproduce the full source BRDF, environment cube,
variance filtering or local-light transport.

The extended Lab wire (`0x3A514353`) carries five texture paths and 84-byte
vertices: the previous 60-byte attributes followed by two float3 directions.
Existing feature draws receive zero padding, with their original attributes
unchanged. Runtime packs are not rewritten. Frame maps are generic per-mesh
overrides with geometry fingerprints and finite-unit-vector validation.

## Combined comparisons and controls

r31 preserves r29's coastal city, paving, lighting, camera and two gameplay
sizes. The four disabled controls are **pixel-identical** to r29. Both controls
reuse the exact candidate packets and postprocess; they do not rebuild geometry.
At normal gameplay size:

| Isolated contribution | Noon pixels >2 levels | Midnight pixels >2 levels |
| --- | ---: | ---: |
| Authored frame + normal texture | 526 | 75 |
| Direct specular | 733 | 493 |
| Combined | 1,141 | 557 |

The changed pixels are on wall details, roof edges and corners. Their modest
scale explains why the scene still falls short of the canonical visual richness.
See `out/city-material-r2/source-surface-native.png` at native pixel size.

r32 repeats the existing modern city at coastal anchor (7,5), retaining the
single-era pool and all seven building placements. Its comparison to r18 also
includes the intervening source-normal/AO and dry-cell ground-clipping work;
it is a combined comparison, not an isolated specular experiment. Some modern
source normals oppose area-weighted geometric normals. They remain explicitly
recorded in the source audit rather than flipped or hidden. Source shader
decoding is stronger evidence than requiring agreement with recomputed normals.
See `out/city-material-r2/modern-source-surface-native.png`.

r33 repeats the previously fixed inland city/material witness at (7,4), with
no regional tuning. This extends the combined material regression check, not
the set of previously unseen terrain regions. Fixed wilderness city coverage
and the full culture/era/size matrix remain open.

Ten Windows comparisons pass (four medieval, four modern, two inland). Ten
focused decoder/ground/palace tests pass. `qa/city_source_surface_evidence.py`
rechecks sampled regions, building placements, controls, local pixel changes,
input identities and backend results. These support the visual evidence and
are not visual acceptance. [Evidence](CITY_SOURCE_SURFACE_r31_r33_EVIDENCE.json).

## Concrete missing material inputs

A broader read of the modern source material records found accepted source
texture slots that the current compound importer omits:

| Source material offset | Installed class | Current omission |
| --- | --- | --- |
| `0x2c` | `Generic_Metalness` | No metallic reflectance input |
| `0x30` | `Generic_OPAC` | No separate opacity mask/state |

The twelve modern source components contain 100 nonempty records across those
two slots (including material/state variants), 51 associated with components
present in r32. This is not a count of unique textures or visible draws.
`out/city-shader-source-r1/modern-extra-slots.json` preserves the exact roster.
Do not claim all city material maps are in use. Neither field was silently
enabled in this pass; opacity needs its actual blend/cutout state as well as the
texture, and metalness needs the corresponding reflection contribution.

The next concrete work is to normalize these missing channels with their source
state bindings, recover the variance constants, and supply coherent environment
reflection/local lighting. Broader urban ground coverage remains a separate
visible composition gap. These are independently solvable Lab tasks, not a
request for human approval or a new integration milestone.

Reproduction uses the r28 city command, adding a freshly generated frame map and
`--source-surface lit`:

```bash
python3 Renderer/terrain_lab/v2/qa/prepare_city_source_normals.py \
  --pool european/medieval --include-frame --output NEW_FRAME_NORMALS.json
```

Use `--source-normals NEW_FRAME_NORMALS.json` and a new city revision. The producer
requires explicit `--ao-uv 1` for the extended layout. Run
`qa/city_surface_controls.py --source COMBINED_RENDER_DIRECTORY --mode off`
and again with `--mode normal` for the isolated controls. Source inspection is
reproducible with `qa/city_shader_material_probe.py --disassemble` through the
approved Windows dispatcher. Preserve prior evidence.
