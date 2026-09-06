# Q2 terrain r09: isolated candidate and composed hook convergence

Status: independent base-material work implemented and visually reviewed;
**work-package acceptance remains pending**. No LQ/I gate or v1 handoff changed.
The scheduling, gameplay, source-art and placement-clearance updates are consumed.
Q2 places no trees, buildings or clutter; clearance-aware placement remains with
its declared owners. No original-art fallback has been selected.

## Primary contextual comparison

Open `out/Q1/Q2-terrain/context-before-after-z1.png` and
`context-before-after-z2.png`. Columns are the v1-equation diagnostic proxy,
new blend/detail-off, and r06 detail-on. Rows are noon and midnight. Each cell
is shown at actual output size, with no scaling or sharpening. The synthetic
8x8 countryside is cropped to 512x256 normal / 256x128 reduced pixels using
Civ III's exact 128x64 / 64x32 tile basis. It contains no invented settlement
or object placements and is not presented as captured gameplay. The baseline
column is an ablation in the same Q2 provider, not an immutable-L21 rerender.

Direct review: green grass, yellow-brown plains and pale desert remain legible
at reduced zoom. The wider continuous transition and periodic domain distortion
avoid visible individual tile diamonds or cracks. Detail-on adds modest local
surface response instead of a noisy high-contrast overlay. The isolated normal
view reveals source-height structure; it does not turn the lit ground into
sparkling sandpaper. Flat desert remains deliberately soft and is not a complete
dune scene. The tundra source is conspicuously gray-white and patchy; that is the
selected source texture, not a newly painted texture. Marsh is wet ground only;
its pools and clutter are external. Full gameplay composition remains required.

## Iterations and highest-impact corrections

1. r01 rendered a mostly black surface. The Metal argument parser populated
   only one array texture binding. Explicit t0..t14 bindings in r02 corrected
   this without editing Q0. Both images are preserved in separate output names.
2. r02 established visible continuous five-material blends. The loader was then
   corrected to select ROOT normalized height/specular roles, not the first
   nested elevated role with the same key. Base and elevated source channels
   must never be conflated.
3. r03 fixed the pixel basis and replaced the auto-fit floating overview as the
   primary comparison with a filled contextual countryside crop.
4. r04 increased subordinate SOURCE-height scales enough to make material
   structure observable, while limiting slope and albedo breakup. No noise
   texture or replacement terrain body was generated.
5. r05 adopted Q6's published hue-preserving shoulder and exact sRGB output
   response, consumed read-only. It does not claim the current backend already
   supplies a scene-linear composition attachment.
6. r06 rotated source-height derivatives from raw map axes into the lattice
   tangent basis. It completed detail/off/baseline, height/normal/roughness/
   albedo/weights, both zooms, four phases, and deterministic scrolling checks.

## Rendered and portable evidence

`metrics_v1.json` enumerates exact-size sheets and all source image paths.
All 32 pair sheets (4 phases x 2 zooms x 2 axes x 2 ownership orders) were
opened and directly inspected; each contains the 15 actual unordered base
material pairs. The contact sheets have no resampling or sharpening.
They preserve material identity, show no black cracks, and retain coherent
blending through the gray-white tundra and green marsh cases. Noon is brightest,
night cooler and darker; dusk and dawn remain distinguishable without losing
material identity. No isolated sparkle/speckle defect was observed in those
static final-size sheets or in the reviewed scroll witnesses.

The portable recipe covers all 14 effective vanilla terrain families: 105
unordered pairs including homogeneous controls, 1,260 axis/ownership/wrap cases,
60 three-way cases and 20 conservative base/real stress states. The latter are
not assertions of legal vanilla placement. The compiled actual Q2 kernel tests
13,860 seam samples: shared-weight delta 0, map-width alias delta 0, maximum
weight sum error 2.23e-16. The recorded v1-style raw-noise equation differs by
up to 0.390115 in the same alias stress test. Base geometry is one continuous
flat grid, height datum 2.5 with +Z geometric normal. These tests do not claim
Q3/Q4's composed heights/normals already match.

The distinct base-material matrix rendered 60 cases (15 pairs, two axes,
reversed ownership), each at four phases/two zooms with deterministic repeats.
In total the r06 checkpoint has 77 reports and 832 images; all 792 `check` images
have byte-identical rerenders. Forty isolation `compose` images are not
misreported as deterministic-repeat checks.

Detail-on versus detail-off: mean RGB absolute difference 0.7294 normal and
0.6620 reduced; 99th percentile 4/255 at each zoom; maximum 6/255. Both retain
macro identity. Aligned one-pixel scroll errors average below 0.009/255;
worst native difference is 3/255, reduced 1/255, consistent with raster/sample
rounding. Return-to-origin images are byte-identical at every phase/zoom.
Numerical values constrain change and stability; they do not define beauty.

## Actual test.biq evidence

Source: `test.biq`, 17,173 bytes, SHA-256
`a6a88d7fffcc567c3500bbd5aa947398dd48170d4f412aa1e518bb45ffe8453e`.
Q0 verified the 100x100, 5,000-tile dataset with horizontal wrap only. Every
owned export retains the exact source/region/parser/halo record and is checked
by the shared runner. No augmentation or source edit was made.

The initial mixed (14,42), fixed mixed holdout (18,46), and wrap (94,50)
regions ran first. They mostly reduce to grass in base-only isolation, so Q2
requested actual dry/cold/wet material regions instead of claiming coverage
from a convenient name. Q0 registered q2-dry (18,40), q2-cold (81,19), q2-wet
(83,39) and each fixed (+4,+4) neighboring holdout. All six ran 24 variants
(4 phases x 2 zooms x 3 scroll positions), passed exact source validation and
repeats, and their full-region sheets were opened and inspected.

Dry shows a broad pale desert patch blending through plains and grass, including
source flood plains mapped to their confirmed plains material. Cold preserves
the corner tundra witness instead of cropping it away. Wet preserves its darker
marsh ground and surrounding dry land. Their fixed holdouts show uninterrupted
base texture and stable mixtures. Full-region diamond views here are focused
coverage diagnostics, not the primary beauty target. Rivers, shore, vegetation
and relief are still absent from this base-only draw; an all-water region's
neutral underlying material is NOT a land/water visual acceptance result.
Volcano is absent from the actual BIQ and remains explicitly synthetic.

## Source and interface provenance

`source_materials_v1.json` records normalized material/texture hashes and measured
64x64 channel samples for grassland, plains, desert, tundra, marsh and flood
plains. Base color is hardware sRGB; height/specular are linear. The selected
alternate-skin pack is preserved. Classification is `source_adaptation`:
source material channels on a generated flat continuous base grid, with 1x/3x/8x
samples of the same source height. No generated dominant landform or invented
texture is included. Detail amplitude, height-correlated roughness and inverse
specular interpretation are C3X-authored, not claimed as decoded Firaxis logic.
Diagnostic weights and proxy geometry remain `diagnostic_proxy` evidence.

Runtime consumption remains generic normalized C3X material descriptors and
DDS packet storage. Source-specific conversion/discovery stays offline.
`INTERFACE_REQUESTS.md` records the callable field and outstanding requirements.
Q1 was sent the exact controls for filtering/detail comparison. Q0 received the
concrete surface provider request and texture-array witness. The immutable L21
handoff is referenced only; no existing native renderer or injected code changed.

## Current acceptance gates

- Q0 base weight/UV and Q3 signed-shore hooks are connected and directly rendered.
  The rejected pale rectangle is fixed. Q6 scene-linear material calls and Q1
  sampling recommendations are consumed and reviewed in the r14 checkpoints.
- Exact incident source-geometry normals fail in wet/dry regions. This is the
  decisive remaining Q4/Q0 blocker; see the final sections and seam JSON records.
- Full 14-family composed geometry, complete Q3 water material, Q4 source relief,
  Q7/Q8 gameplay benchmarks, 192-tile promotion and coordinated D3D11 parity are
  unpassed. This is a candidate convergence delivery, not milestone closure.

Exact replay commands are in `systems/terrain/README.md`. Portable/static tests,
`tests/terrain/verify_evidence.py`, source audit and metrics generation are
executable. Touched portable source/fixture/audit/status files are scanned for
personal paths and sensitive strings before delivery; generated render payloads
are ignored/local and no licensed pixel assets are committed.

## r08/r09 composed convergence

Q0 published optional versioned weights and source-UV hooks. The Q2 adapter is
consumed by `composed-complete.module.json`, using the complete scene path so
all five terrain material bindings, including tundra, are available. Empty
owned object augmentation files explicitly add zero objects. The real terrain,
rivers, forest and relief records are unaltered. Complete cold and wet source
scenes and the cold frozen baseline were rendered in bounded quick runs and
opened directly at 768x512. Cold retains the white-gray tundra corner, broad
plains/grass transitions, river and source mountain/forest shapes. Wet retains
source grass/plains forms and its coast cliff. The frozen canopy placement and
coast wall remain visually conspicuous; this is convergence diagnostic evidence,
not a claim of gameplay/beauty acceptance or Q3/Q4/Q8 closure. The old lighting
mode omitted tundra bindings, so it is not the all-material validation path.

The owned recapture test loads actual primary/held-out dry, cold and wet CSVs,
computes their lattice displacement from raw BIQ coordinates, and compares 771
samples on the shared boundary. Weight and UV errors are zero. This tests actual
neighboring capture inputs independently of the camera-only scroll test.

The isolated material formula was extracted to `material_response_v1.hlsl`
without changing arithmetic. All four r09 noon/native scroll frames exactly
match r06; the first was opened again. `scene_material_v1.hlsl` is a separate
composed supplemental detail adaptation: it preserves existing authored base,
raised-form and decal normal responses, adding only the 3x/8x source bands.
It contains neither lighting nor output encoding. Its scene render acceptance
is pending Q6's full linear shader hook.

A shared storage incident paused large convergence matrices. Q2 owns roughly
526 MiB of prior outputs, retained because reports reference the evidence; zero
bytes were deleted and no shared cache was touched. Only bounded quick witnesses
continued after available disk space recovered. Heavy runs require coordination.

## User-rejected straight pale rectangles: diagnosed and corrected

The user explicitly rejected the sharp straight pale terrain edges in the
composed evidence. Surface-kind and material-weight isolations retained the
original coverage masks and showed that the defect lies on land pass 1 while
Q2 material weights stay continuous. Q3 traced the cause to frozen isolated-coast
basin support: `max(negative_land, zero_outside_basin)` incorrectly makes a
rectangular inland area have shoreline distance zero. Its material turns beach
colored and its coastal relief envelope collapses, creating a vertical lip.
The exact source counterexample is documented read-only in
`audits/hydrology/FROZEN_BASIN_BUG.md`.

Q0 published `HydrologyHooksV1`; Q2's owned `hydrology_consumer.h` includes Q3's
provider without copying or editing it. `composed-hydro-on.module.json` now
combines Q2 weights/UV, Q3 signed shoreline and Q6 scene-linear material hooks
in the existing whole source scene. Before `wet-linear-on-r12` and after
`wet-hydro-on-r13` preserve the same real source, viewpoint, Q2 detail and
4x MSAA. Direct inspection confirms removal of the entire pale rectangle and
raised lip; continuous grass/marsh and the existing source pools are visible.
This corrects the field causing the defect, rather than disguising its edge.
The prior interpretation of that pale plate as a legitimate coast cliff was
wrong; r08 is rejected evidence, not an accepted coast treatment.

The r14 wet/cold/dry checkpoints run four phases at both zooms with repeats.
They retain macro terrain identity, remove the inland pale rectangles, and
preserve source relief, river, forest and tundra shapes. Exact-size per-phase
strips are directly inspected. Supplemental composed detail on/off at both
zooms has mean all-frame RGB difference 0.0983/0.0787, p99=1, max=4/3; the effect
is subordinate and does not repair macro geometry. No negative mip bias or
sharpening is used. Q1's selected anisotropy8 and 4x MSAA are retained.

## Remaining proven incident-normal failure

The exact pinned Q0 source builder was queried on both incident sides of every
internal edge of the three real regions, 408 sample pairs each. Current heights
and Q3 shore distances agree (cold height rounding below 2.4e-7), but wet has
145 normal mismatches above .001 (maximum component delta .421376169) and dry
has 153 (maximum .826748308). Cold's maximum .000953 passes this tolerance.
`wet_surface_seams.json`, `dry_surface_seams.json`, and `cold_surface_seams.json`
record exact edges and terrain families; their source query payloads pin the
builder, terrain, packs and fixture. The owned `composed_surface_audit.py`
reproduces them through Q0's exact surface-query workflow.

Frozen `make_biq_vertex` differentiates height with samples clamped within each
ownership tile. Equal incident heights do not imply equal derivatives. This
remaining shared raised-surface normal failure prevents universal transition
acceptance. Q4 received the exact witnesses and request for a continuous
height/normal provider; Q2 must not replace authored relief with its flat base
normal or silently edit shared geometry. The full 14-family composed matrix,
final gameplay benchmark and coordinated promotion/parity remain unpassed.

The fixed wet neighboring holdout also passes eight phase/zoom variants with
byte-identical repeats and direct final-size review. The r14 composed total is
32 checked images, plus two detail-off controls and eight scroll controls.
Aligned one-pixel scroll mean RGB errors are .00590 native/.00394 reduced;
maxima 15/4 occur in the full scene with object edges. Return-to-origin is exact
at both zooms. These values do not grant relief-normal or beauty acceptance.
`tests/terrain/acceptance_gate.py` verifies all 32 image hashes/repeats and fails
explicitly on the recorded geometry and outstanding full-scene criteria.

A subsequent identity audit found the original r14 on/off controls had identical
packet bytes and feature shader stages but different backend executable cache
keys. The on control was rerendered as r15 on the off-control backend. It matches
the r14 on image byte-for-byte at both zooms, preserving the reported visual
observations and differences. r15-on/r14-off packet bytes/content hashes and
backend identity match; non-material shader stages are identical after bijective
renaming of compiler-generated identifiers. Exact details are recorded under
`detail_control_identity` in `composed_metrics_v1.json`. The earlier backend
identity mismatch is not concealed or attributed to Q2 material response.

Q1 independently consumed the current Q2/Q3/Q6 candidate in its mixed actual-map
source scene with city, tank and route augmentations and its selected postprocess.
Q2 also opened both on/off images at both final sizes. Macro ground identity is
stable; no additional ringing or terrain sparkle is apparent. Q1 measures
whole-frame mean RGB differences .01424/.01203, p99=1, max=2 at both zooms.
This is contextual material validation, not scene beauty approval: floating
city components, steep route/relief overlaps and conspicuous object scale remain
visibly provisional and belong to the other scene owners.

Independent outputs are under `audits/sampling/out/Q1/Q1-sampling/` in
`c030q2-linear-on` and `c031q2-linear-off`; the former contains
`detail-on-off-metrics.json`. They are consumed read-only.

## Updated caster-shape requirement

The coordinator's new user-directed requirement is consumed: every cast shadow
must use the actual transformed/posed source caster geometry and authored cutout
alpha, through Q6 shared lighting and real receivers. Generic foliage ribbons,
ovals or footprint substitutes cannot pass. Current frozen/composed diagnostic
shadows are not promoted as accepted shadow evidence. Q2 adds no shadow hack or
shadow suppression. Q2 publishes continuous base weights/UV and flat base datum;
Q4 owns the actual raised terrain and vegetation geometry, while Q0's exact
surface query exposes their current receiver geometry. Its normal failures remain
explicit. Final acceptance additionally requires Q6's actual-silhouette shared
shadow composition; physical placement clearance remains a separate contract.
