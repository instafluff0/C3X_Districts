# City quality campaign

The user redirected the Lab from water to cities: improve their appearance at
different sizes, eras and cultures, including night lighting reflected in water.
The user also explicitly permits modest overlap into neighboring tiles,
especially for larger cities. Water-natural-r6, water-reflection-r5 and all
historical pickups remain preserved. No native implementation or milestone
promotion is authorized by this campaign alone.

## Required combined result

- All five Civ III culture groups, four eras and three population sizes must
  produce readable, distinct and plausible cities. Growth must retain a stable
  arrangement rather than reshuffle the whole settlement.
- Buildings retain source proportions, all applicable parts and correctly bound
  material channels. A source-backed focal building or block must improve the
  actual scene, rather than merely increase the component inventory.
- City footprint allowance may grow with population; preserve the city anchor
  and keep complete building bounds out of water, rivers, routes and neighbors.
- Day, sunset, midnight and sunrise use the common environment. Localized source
  window emission must remain readable at both gameplay zooms and contribute to
  the same water reflection pass as the actual buildings.
- Primary evidence uses the fixed approximately 100-tile test.biq scenes, matched
  cameras, unchanged terrain and explicitly labeled city augmentation. Include
  an untuned region, growth/clearance cases and daylight/night controls. Galleries
  support coverage; they do not replace contextual visual review.
- Existing capital, walls, retained label/UI, parity, cost and promotion gates
  remain applicable. Native C3X delivery is separate; wonders/Districts stay deferred.

## Current findings and preserved experiments

The previous city renderer's proof pack contains only four components per
culture/era pool. The importer takes the first four successful candidates in
alphabetical package/entry order. The currently mapped source pools contain
23–38 candidates each. This restricts architectural variety even though the
complete generator graph was already discovered.

The separate `CityStudyExpanded` local pack now contains twelve components in
each pool: 132 unique complete components, 304 geometry parts, 450 materials,
299 emissive material bindings and 137 textures. Its 123 sockets retain unresolved
resource status; they are not invented point lights. This is source intake,
not visual acceptance. Strategy:
`fixtures/beauty/city-scene-foundation/expanded-city-strategy.json`.
Source report: `out/city-source-expanded-r1/build.json`.

The active combined producer is `qa/city_scene_pass.py`. It appends complete
worked-state source parts to copies of the current coastal terrain packets,
rebuilds the common object/terrain shadow field, and renders the city through
the current GPU water reflection pass. No second shader namespace or presenter
is required. The initial sites are local coastal tiles [3,2] and [8,5]; both are
explicit Lab city placements, not city state decoded from test.biq.

The fixed coastal terrain remains 100 tiles, 1360x800 internal pixels, half-width
64, noon/midnight during bounded iteration. Zoom 2 uses the existing 680x400
reconstruction. The original terrain-only benchmarks remain unchanged.

- r0 is a failed input diagnostic; the first shore check used the wrong sign.
  The surface sampler is negative on land, unlike the optical water payload.
- r1 establishes the complete seven-body medieval baseline. Input/build failures
  were resolved before any image existed; completed images remain immutable.
- r2 increases uniform source scale by 1.5. The paired pixels show more readable
  roof shapes, facades and windows, but still repeat only four building variants.
  Four medieval day/night/zoom frames and a separate modern waterfront pair exist.
- r3 adds a whole source neighborhood plus varied buildings at the retained
  physical scale. The initial tight layout failed; the user-authorized cross-tile
  footprint permits the same anchor without shrinking the buildings. Its large
  exposed platform is a defect under investigation, not a selected final look.
- r4 tests grounding the source z=0 plane, keeping negative foundation geometry
  underground. The old layout subtracted every mesh's minimum Z: the medieval
  block extends to -0.0648 source units, so that operation raised a substantial
  skirt above terrain. This is a geometry/placement correction, not mesh editing.

The new buildability callback checks a frozen dense shoreline/height witness;
final transformed corners and centers are sampled again. These flat-site probes
have zero ground-height range. Complete route/river/vegetation and neighboring
city envelopes still need composition coverage. The initial expanded-layout
half-extents are .65/.80/.95 tiles for town/city/metropolis; these are bounded
Lab parameters, not approved final dimensions.

## Material investigation

Base and emissive maps are bound; optional AO is wired but not yet visually
selected. The source `normal_0` and `normal_1` are a LEAN pair, not an established
ordinary tangent normal map. Gloss interpretation remains unresolved. Do not
blindly bind BC5 as a standard normal to claim complete materials.

The current importer recomputes area-weighted normals from indexed source
triangles. Earlier Q7 text calling these authored vertex normals was inaccurate;
the packed source tangent frame remains undecoded. The baseline normalized only
UV0; r8's separate opt-in source study retains two additional coordinate sets.
The modern base and emissive atlases differ in dimensions/layout, so UV/material
usage needs checking before intensity alone is blamed. A separate UV set is a
hypothesis at baseline, resolved for the tested r8 bodies below. The original
lights used a 1.45 emissive gain; no additional lamp resources have been invented
or activated.

## r8 night-light checkpoint

The user explicitly rejected faint windows and absent glow, pointing to
`canonical/nightlights.jpg`. Simply raising emission exposed a deeper defect:
the old shader sampled the emissive atlas with the diffuse UVs. Medieval roofs
developed misplaced bright spots; modern windows became scattered speckles.
Those r6/r7 images are preserved diagnostics and are rejected as final appearance.

The cooked 24-byte static vertex profile contains additional half2 coordinates
at offsets 16 and 20; the 32-byte profile carries them at 24 and 28. The optional
`--auxiliary-uvs` city import retains these as generic `uv1` and `uv2` in the
separate `CityStudyAuxiliaryUV` pack. Default imports remain unchanged. Matched
medieval and modern renders establish **UV2 as the light-atlas coordinate for
these tested source bodies**: roof artifacts disappear and lights follow window
rows. UV1's precise role and the packed tangent frame still need verification.
Do not infer the normal map or all auxiliary channel roles from this finding.

`qa/city_scene_pass.py --emissive-uv 2 --emissive-gain 8 --glow` produces r8.
For this bounded diagnostic, a depth-tested additive emissive draw retains the
same source triangles but uses UV2; the diffuse draw keeps UV0. The extra draw
neither writes depth nor casts a duplicate shadow. Reflections consume both.
A production multi-UV vertex layout can combine these draws later; that change
is not a prerequisite for inspecting the corrected source mapping now.

The glow is an opt-in scene-linear GPU postprocess before the shared tone curve.
It thresholds HDR values above 1.0, filters two compact optical scales, and
preserves Q1 reconstruction, alpha and validity. The tiled separable version
keeps all threads alive through workgroup barriers for Metal and FXC/D3D11.
No bloom is applied to retained Civ III overlays or UI. This is lens glow;
it does not claim point-light illumination of neighboring ground or buildings.

Preserved selected views:

- `out/city-scene-r8/modern-glow-portable/`: noon/midnight, both gameplay zooms,
  unchanged 100-tile coastal terrain and modern city at [7,5].
- `out/city-scene-r8/european-medieval-s1/combined/`: corrected medieval windows,
  same four views and the preserved [3,2] city layout.
- `out/city-scene-r8/review/`: native before/after crops and enlarged UV
  diagnostics. Full source frames remain available beside the crops.
- `CITY_NIGHT_r8_EVIDENCE.json`: reproducible source-geometry preservation,
  glow/light/reflection controls and Windows comparison results.

The modern lights-off control removes warm pixels from a measured interior
lake patch separated from direct city glow. Glow-off controls demonstrate real
halos at both zooms. Daylight differs by at most one 8-bit level in two pixels,
consistent with the previously recorded isolated Metal replay rounding; it is
not claimed byte-identical. Four standalone D3D11 frames pass unchanged parity
thresholds. The initial FXC synchronization compile failure is preserved in the
empty `windows-modern` attempt; `windows-modern-portable` is the passing retry.
The two shader versions match exactly at night; one daylight pixel differs by
one level. No native C3X city implementation or visual approval is implied.

Verification: `qa/city_night_evidence.py` (Pillow/NumPy), 16 focused importer
tests including sparse auxiliary-UV remapping for both vertex profiles, and
`renderer_dev.py lab` (132 Python tests, 12 Node tests and campaign validation).

## Storage correction and bounded continuation

The user explicitly requested cleanup after the full city matrix consumed too
much storage. The batch is stopped. The city producer had bypassed the existing
`app/packet_store.py` resource deduplication, duplicating the complete terrain
payload in every city/hour/zoom packet. It now compacts final packets before
rendering and immediately deletes the disposable pre-shadow intermediate.

Cleanup converted 54 completed city packets to the existing directly replayable
shared-resource format, verified by expanding each to its original byte hash.
Images, source packs, fixture inputs and reports remain intact. Earlier packet
hashes describe the original serialization; `CITY_STORAGE_CLEANUP.json` maps
those hashes to their compact representations. A fresh Metal night render is
pixel-identical to the preserved r8 result; its temporary output was removed.
596 geometry-cache entries unreferenced by saved replay/evidence records were
removed. Referenced shared resources remain available. Renderer storage fell
from approximately 67.1 GiB to 50.1 GiB during this cleanup, excluding the earlier
removal of disposable city intermediates.

`qa/city_matrix_pass.py` now defaults to six cases, supports an explicit start
index, and uses shared packets directly. Both the matrix and single-city tools
stop below 8 GiB free space. r9 records the capacity failure; r10 records the
user-requested stop. Completed r10 images remain usable for culture/era review;
the 60-case matrix is incomplete and must not be reported as passed.

## Next work

### r11 growth and r13 capital follow-up

The colonial size-one placement failure exposed a counting error: a complete
source neighborhood counted as one house, followed by six more houses. The
opt-in `--weighted-growth` recipe counts its footprint in one to four house
equivalents. r11 colonial town/city/metropolis contain 1/4/8 complete components;
the previous component positions, scales, rotations and order survive growth.
The coastal placement checks pass without shrinking source buildings. Middle
Eastern and Mediterranean medieval pairs also render; full growth coverage
remains incomplete.

The user then requested actual palace buildings in capitals where identifiable.
Two source compounds were already located in the Gran Colombia/Maya package:
`DIS_CTY_RSAM_Palace` and `DIS_CTY_RCOL_Palace`. A small separate
`CityPalaceStudy` pack reimports only these two with auxiliary coordinates. All
source triangles, positions, normals and UV0 match the prior normalized pack;
no source body is edited. The default compound importer behavior is unchanged.
The colonial body has four unresolved operational component attachments, so
its complete-kit integration remains pending. Optional Mesoamerican flame
sockets and ground decal state are also unimplemented, explicitly retained as
limitations rather than invented effects.

`qa/city_scene_pass.py --capital` resolves the explicit style mapping in
`systems/objects/capital_styles.json`, reserves a center site and adds one palace
through the same material, shadow, glow and water-reflection passes. The
`--omit-capital` diagnostic keeps the same site and houses but omits palace draws.
The proof is an American ancient-style city; the shared medieval mapping is
authored but not yet visually tested. Unsupported styles fail the explicit Lab
probe; the production contract retains ordinary cities and the native capital
indicator when no palace art is mapped.

r12's 0.42-tile palace was too small beside the houses. r13 increases the whole
compound uniformly to a 0.60-tile span, exposing the colored stepped facade as a
readable centerpiece. Its source proportions remain intact; the size is a Lab
calibration, not a confirmed source-engine scale. The selected r13/control pair
preserves all seven surrounding houses exactly, as well as terrain, anchor,
camera and output sizes. At normal zoom the palace changes 926 noon and 895
midnight city pixels by more than two channel levels; reduced zoom changes 256
and 243 pixels. The changed pixels expose the stepped facade and its lighting;
counts alone are not visual acceptance.

Evidence: `CITY_CAPITAL_r13_EVIDENCE.json`, generated by
`qa/city_capital_evidence.py`, and
`out/city-scene-r13/review/capital-native.png`. All 16 capital/control frames
across r12/r13 retain shared-resource packets; the two new source models occupy
about 10.3 MiB. About 29 GiB remained free after these bounded captures.

Sixteen focused importer tests and four standalone Metal/D3D11 palace comparisons
pass (mean channel error 0.074–0.101/255, silhouette IoU 1.0). No game was launched.
The full Lab run reaches 132 Python tests
with one failure: the unchanged L19A tile-object input hash gate expects
`16e1acdb...` but the current `TileObjectsNormalized/tile_object_runtime.bin`
hash begins `1bf64e5b...`. This pass did not modify that pack or its test; the
gate remains open and no full-workflow pass is claimed. Capital visual review,
all culture/era/size coverage and production integration remain separate gates.

Extend the corrected source selection and light mapping across sizes/cultures,
including stable growth and additional city light atlases. Resolve UV1, source
tangent/LEAN handling, and source-backed local light resources so nearby streets
and walls receive plausible pools of light as in the canonical reference.
Add inland/untuned context, growth, clearance, four phases/two zooms and Windows
parity at meaningful combined checkpoints. No city quality or milestone approval
has been recorded.

## Metadata-led direction after r14–r16

The user pointed out structured building arrangement and ground textures in
Civ VI references and explicitly deferred roads. The [generator findings](CITY_GENERATOR_FINDINGS.md)
record the missing authored era mix, center ordering, population/filler/area
parameters and ground material roles. r14 surface detail is subtle; r15's
height-sorted metropolis cannot fit and is not a complete growth recipe. r16
recovers source paving triangles/UVs and terrain projection but exposes only a
narrow strip under the crowded buildings. Preserve these as diagnostics. The
next meaningful visual pass must use mixed-era city fabric and ground pieces
together, rather than continue isolated height-ordering or shading tweaks.

The user subsequently rejected r17's mixed-era look and selected one current era
per city. This preference supersedes the preceding mixed-era plan. Preserve
same-era source families; apply useful generator growth, placement and ground
metadata without historical-era mixing. The generator-profile CLI defaults to
that policy. Roads remain deferred.

## Single-era ground and broader palace intake

[r18/r19 combined findings](CITY_GROUND_AND_CAPITAL_PASS.md) preserve the selected
one-era policy. Modern and medieval source paving produces small visible base
improvements. The American palace from the new 47-root pack renders with its
night material and seven matched surrounding city pieces; its facade occlusion
and wider scattered composition remain defects. Six focused tests and eight
standalone Windows comparisons pass. No overall city-best replacement, full Lab
pass, fresh-region acceptance or milestone advancement is claimed.

## Capital composition and growth, r21/r22

[Combined findings](CITY_COMPOSITION_PASS.md): focal visibility scoring now uses
the authored ground plane, improving the modern palace facade contribution.
The 4/7/11 ordinary-building sequence and palace remain fixed as the city grows.
A city-untuned terrain region renders with the same recipe. r22 combines source
paving, with exact dry-cell and smoothed-shore clipping. Ten Windows comparisons
and seven focused tests pass. This is a provisional American modern composition
improvement; other cultures/eras, broad ground coverage, materials, local light
transport and all existing gates remain open. No full Lab pass is claimed.

## Source AO coordinate recovery and combined materials

[r23-r28 findings](CITY_AO_MATERIAL_PASS.md) establish source-atlas alignment and
visible shading improvement from UV1 AO for the tested medieval bodies. Diffuse
UV0 and emissive UV2 stay unchanged. The final r28 case combines that material
fix with source repeat addressing and ground pieces. AO-disabled controls retain
original geometry/bindings and match the baseline within one 1/255 pixel. The
ancient composition and a city-untuned inland region provide wider checks; the
ancient AO delta itself is too small to count as an improvement. Fourteen Windows
comparisons pass. Not all source maps are in use: tangent/LEAN/gloss and source
ground-height response remain unresolved, along with broad coverage and gates.

### Source-normal diagnostic r29/r30

The source static vertex normal bytes now have an opt-in, geometry-checked Lab
decoder. Matched coastal and inland renders show subtle roof/corner changes,
not a new accepted best; r28 remains the preceding material candidate. Six
Windows comparisons and ten focused tests pass. Tangent/LEAN/gloss interpretation,
urban ground coverage and local night lighting remain open. See
[CITY_SOURCE_NORMAL_PASS.md](CITY_SOURCE_NORMAL_PASS.md).

### Source shader material restoration r31–r33

Installed rigid-model shader inspection establishes octahedral tangent directions,
reconstructed normal-map Z and cooked dual-lobe roughness parameters. The opt-in
combined pass makes modest medieval material gains, preserves modern windows/
reflection, and passes the inland regression witness. Four disabled images match
r29 exactly; ten Windows comparisons pass. Source material slots 0x2c metalness
and 0x30 opacity are omitted by the current importer and are the next concrete
intake work, alongside variance/environment/local-light response. Full city
quality and all gates remain open. See
[CITY_SOURCE_SURFACE_PASS.md](CITY_SOURCE_SURFACE_PASS.md).

### Connected neighborhoods and rendered river constraints

The [connected-growth pass](CITY_CONNECTED_GROWTH_PASS.md) joins the Asian large
city's detached outer houses while keeping its original sixteen-body prefix.
It also finds and fixes missing river-bank exclusion for the small holdout;
r64/r68/r69 remain preserved overlap diagnostics. The medium river fit is still
unresolved. Ancient large growth and a previously city-untuned coastal region
render without local recipe changes. Twelve Windows comparisons pass; visible
acceptance remains provisional. Next broaden single-era palace landmarks and
source facade variation, retaining river/forest clearance, shared lighting and
all native/milestone gates.

### Broader single-era capital composition

[Palace composition findings](CITY_PALACE_COMPOSITION_PASS.md) connect the
47-root source library to the Asian medieval and ancient-brick house families.
Selected r93/r94/r92 retains the palace and exact 8/16/24-house prefixes,
with a matched r95 palace-off control. The small palace joins its neighborhood
instead of sitting across a gap. Source proportions, authored paving and
night windows survive composition. Ancient inland r77 and main coastal r98
add coverage; separate capital-untuned coast r99 does not fit and remains
explicit negative evidence. Twelve Windows comparisons and independent packet
checks pass, supporting the pixels without granting visual acceptance.

Continue connected usable-land reasoning for the unresolved coast/river
growth, broader single-era styles, facade/environment response and varied
open ground. The current canonical comparison still shows clear gaps in
window/facade detail and deliberate ground composition. No new reflection
quality claim or native gate advancement. Cleanup removes only 74 completed
new linear readbacks (614.3 MiB), preserving images, failures and replay inputs.

### City environment reflection and remaining material gaps

[Environment findings](CITY_ENVIRONMENT_PASS.md) preserve two provisional
modern material candidates with shared sky/ground reflection and existing
metalness maps. The same recipe works on inland-large and wilderness-medium
without moving geometry or changing source textures, paving or local lights.
The source-family roughness attenuation was missing from the first trial;
restoring it reduces washout, but the Asian roof result remains unselected.
The prior palace appearance is retained. Eight current Windows comparisons,
four independent material-bit inspections and an exact disabled control pass.

The American capital/lake fixture still uses an earlier city material layout
without bound metalness. Restore the complete house/palace material inputs
there next and verify its night reflections together. The analytic environment
is an authored approximation, not recovered cube/SH calibration or a global
city default. Broader single-era coverage, open-ground composition, unresolved
coastal/river growth and all milestone/manual gates remain part of the active
goal. Cleanup removes 30 new completed readbacks (249.0 MiB), retaining all
images, rejected trials and replay resources.


### Current capital composition preference and selected scenes

The user now requires the palace at the city core, surrounded by ordinary
buildings, with all footprints aligned to the city grid and quarter-turn choices.
This supersedes the foreground-palace preference. Keep one era per city.
[Current central-capital findings](CITY_CENTRAL_CAPITAL_PASS.md) select r111
inland and r112 freshcanopy provisionally. [Gameplay comparison](out/city-central-capital-r2/inland-native.png)
and [previously untuned region](out/city-central-capital-r2/holdout-native.png)
show the centered, aligned palace with corrected paving and local lighting.
Four Windows frames, twenty independent composition checks and 33 focused tests
pass; all milestone/manual gates remain open.

Supporting completed work: [American capital materials](CITY_CAPITAL_MATERIAL_PASS.md),
[aligned paving footprint](CITY_PALACE_GROUND_ALIGNMENT_PASS.md), and
[light reaching the paving border](CITY_PALACE_FACADE_ALIGNMENT_PASS.md).
The final recipe uses `--central-capital --orthogonal-buildings`, source-hull
paving and facade-plane capital lights. The coastal seven-house surrounded fit
remains unresolved after 25 legal-core attempts and alternate side assignments;
retain `out/city-palace-facade-alignment-r1/environment/render` there. Do not
claim that fallback meets the new central layout preference. Next resolve its
footprint/foundation constraint, broaden single-era styles and sizes, and improve
source environment/variance response and open-ground materials. Connecting roads
and native delivery remain deferred. Completed readback cleanup is recorded in
the four corresponding capital/material/ground/facade cleanup reports.
