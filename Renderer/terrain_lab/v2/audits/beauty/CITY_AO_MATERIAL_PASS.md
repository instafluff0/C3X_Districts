# Source AO coordinates and city composition, r23-r28

Status: partial material improvement, not complete Civ VI appearance or city
acceptance. Single-era selection and all existing milestone gates remain intact.
The previous bests are preserved. No native code, pack promotion or human approval
is implied by this Lab pass.

## Material finding

The selected European medieval houses have dedicated BC4 ambient-occlusion maps.
The source diffuse and AO atlases have different layouts. Drawing the normalized
mesh UVs over the source AO image shows UV1 following small assigned islands;
UV0 triangles cross unrelated atlas regions. This supports UV1 as the AO
coordinate for these tested bodies. It does not establish every source shader's
coordinate convention or the source engine's full lighting equation.

[Source atlas coordinate comparison](out/city-material-r2/ao-atlas-uv-comparison.png)

The previous optional `--channels` path sampled AO with diffuse UV0. That legacy
diagnostic remains unselected. `--ao-uv 1` now carries a separate interpolated
float2 while retaining UV0 for diffuse and UV2 for the already-verified light
atlas. The source AO value feeds the shared ambient-visibility input; it is not
a blanket multiplier over the final image or over emissive light.

All 12 European medieval pool components have AO-bearing material parts. The
standalone modern American towers do not; some complete modern blocks do.
The ancient capital check changes only 58 noon / 50 night pixels with AO, so it
is not counted as another meaningful visual improvement. Apply the channel only
where declared and verify other families rather than assigning it universally.

## Visible result and composition

r24 retains all seven medieval building bodies, positions, scales, camera,
100-tile coastal terrain and light settings. Compared with AO disabled, it changes
2,890 noon / 2,023 midnight pixels at normal gameplay size. Eaves, wall recesses
and spaces between bodies have stronger shading separation. It does not solve
the brown palette or restore specular material response.

r26 composes the AO fix with source paving. r28 also enables the previously
implemented source repeat/clamp addressing. One block part has UVs from roughly
-1.08 to 2.92 and requests repeat sampling; clamping those coordinates was
incorrect. The addressing correction adds 228 noon / 201 night changed pixels
relative to r26. These changes are small but lie on the affected city surfaces.
The final result retains the source ground projection and dry-cell/shore clipping.

[Final matched gameplay-size comparison](out/city-material-r2/medieval-final-native.png)

The larger image changes are shading separation, not additional source geometry.
Normals remain the existing recomputed geometric normals. The source tangent
frame, LEAN pair and gloss are still not fully interpreted or applied. Therefore
**not all declared source material maps are in active use**. Source ground-height
response and broad urban ground coverage also remain open.

## Earlier era and region checks

r23 applies the prior civic-composition recipe to the preserved Mesoamerican
ancient capital with matched source bodies, size, scale and night settings. It
exposes more of the stepped palace face and keeps the house selection in one era.
The independent r27 AO test on that city is only a small-contribution diagnostic.

[Ancient capital composition](out/city-scene-r23/review/ancient-capital-native.png)

The fixed inland benchmark had not been used for city/material tuning. A
metadata-only survey selected anchor (7,4), the only sampled dry 3x3 envelope
with height range <=2.5, before inspecting city images. The same medieval AO and
ground recipe renders at noon and midnight there. No local visual tuning followed
the selection. This is now a regression witness, not an untuned region for future
passes. The remaining fixed wilderness benchmark and full geometry-clearance
coverage are still open.

[Inland city/material witness](out/city-material-r2/inland-native.png)

## Implementation and checks

The optional Lab wire version adds a float2 after the existing 52-byte vertex
payload (60-byte stride). Existing non-city feature draws receive zero padding
for the new attribute, preserving the single shader namespace. Main terrain
vertices are unchanged. A local flattened shader closure adds the matching
varying; shared terrain shader inputs are not mutated. The default legacy wire
path remains available. A production implementation should carry generic
per-channel coordinate selection in its material/vertex contract rather than
require this Lab-specific CLI.

The six disabled-control city draws retain byte-identical original 52-byte
vertex attributes and texture bindings. AO strength zero exactly matches r8 in
three frames; the fourth differs by 1/255 at one pixel. This also checks that the
new vertex transport does not alter the scene when its material contribution is
disabled. Diffuse and emissive coordinates stay unchanged.

Four ancient composition, four combined AO/paving, two inland, and four final
addressing/AO/paving frames pass standalone Windows D3D comparisons. A VM transport
failure interrupted the first AO batch after two completed daylight frames.
The renderer process was confirmed absent before resumption. The new `--resume`
option verifies saved input/output hashes and renders only missing frames; the
completed daylight results were not overwritten. Seven focused ground-geometry
and palace-importer tests pass. No full Lab or injected-code compile is claimed.

`qa/city_ao_evidence.py` rechecks byte preservation, matched inputs, disabled
controls, isolated material contributions, the holdout and all fourteen parity
results. [Evidence](CITY_AO_r28_EVIDENCE.json). These engineering checks support
the visible comparison; they do not grant visual acceptance.

Reproduce the final material composition with a new integer revision:

```bash
python3 Renderer/terrain_lab/v2/qa/city_scene_pass.py \
  --revision NEW_REVISION --pool european/medieval --size 1 --factor 1.5 \
  --expanded --authored-ground --emissive-gain 8 --emissive-uv 2 --glow \
  --ao-uv 1 --source-addressing --all-zooms \
  --compound-ground Renderer/terrain_lab/v2/fixtures/beauty/city-generator-source-r2/medieval-ground-parts.json
```

`--ao-strength 0` is the matched disabled control. Preserve prior renders.
Connecting roads remain deferred. Next work should address the remaining
material response and broad ground coverage, then carry the complete composition
through other cultures/eras and city sizes with matched day/night views.

After verification, regenerable linear GPU intermediates were discarded
(278.1 MiB reclaimed). BMP/PNG references, masks, shader snapshots,
source inputs and shared replay packets remain. The evidence recheck passes
after cleanup.
