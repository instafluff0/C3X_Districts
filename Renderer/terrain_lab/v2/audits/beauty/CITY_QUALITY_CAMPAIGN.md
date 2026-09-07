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
the packed source tangent frame remains undecoded. Only UV0 is normalized.
The modern base and emissive atlases differ in dimensions/layout, so UV/material
usage needs checking before intensity alone is blamed. A separate UV set is a
hypothesis, not a recovered source fact. Current lights use the existing 1.45
emissive gain; no additional lamp resources have been invented or activated.

## Next work

Inspect and verify source-origin grounding, then compare the expanded source
selection across sizes and cultures. Resolve city material coordinates and
source tangent/LEAN handling; calibrate localized night emission with paired
emission-disabled and reflection-disabled controls. Demonstrate reflected city
light on a visible waterfront, not merely a successful reflection draw call.
Add inland/untuned context, growth, clearance, four phases/two zooms and Windows
parity at meaningful combined checkpoints. No city quality or milestone approval
has been recorded.
