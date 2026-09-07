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

Extend the corrected source selection and light mapping across sizes/cultures,
including stable growth and additional city light atlases. Resolve UV1, source
tangent/LEAN handling, and source-backed local light resources so nearby streets
and walls receive plausible pools of light as in the canonical reference.
Add inland/untuned context, growth, clearance, four phases/two zooms and Windows
parity at meaningful combined checkpoints. No city quality or milestone approval
has been recorded.
