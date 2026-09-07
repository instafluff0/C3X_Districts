# Local city facade lighting r1–r3

Status: provisional combined night-light improvement. The full city goal, native
delivery and all human/milestone gates remain open. One era per city is preserved.

## Visible result

Illuminated lower windows now cast short-range warm light onto nearby ground,
paving and facing buildings. The medieval city has clearer pools beneath its
windows; the modern city and American palace gain smaller, localized warmth
around their bases. These move toward the grounded architectural lighting in
`Renderer/canonical/nightlights.jpg`. They do not complete its broader civic
ground treatment, facade environment response or street-light placement.

The selected local candidates are under `out/city-facade-light-r3/`: `wilderness`,
`medieval`, `inland`, `freshshadow`, and `capital`. The comparison is
`out/city-facade-light-r3/selected-native-comparison.png`, at native gameplay
pixel size. Each case reuses its exact previous geometry, material bindings,
camera, terrain, sun/moon shadow packets and postprocess. Only the local-light
shader contribution changes. The wilderness case retains the fixed shadow grid
established in [CITY_GROWTH_PASS.md](CITY_GROWTH_PASS.md).

At normal gameplay size, night changes above 2/255 are:

| Case | Changed pixels | Visible contribution |
| --- | ---: | --- |
| Wilderness modern | 1,837 | Warm bases and facing facades |
| Coastal medieval | 3,275 | Window-side ground pools and paving |
| Inland large modern | 2,417 | Local warmth among eleven buildings |
| Freshshadow modern | 2,590 | Same lighting policy on the holdout |
| Modern American capital | 4,370 | Palace/city base illumination and nearby water response |

The two gameplay zooms are checked for medieval and capital cases. Daylight is
exactly preserved except one 1/255 pixel in each of the normal-zoom medieval,
freshshadow and capital images. No daylight pixel changes by more than 1/255.
Outside each local city/light region, differences are at most 1/255. The original
gain-zero control matches both wilderness images exactly. Earlier candidates
and the lower-gain r1 experiment remain preserved.

Freshshadow received this lighting recipe without local parameter tuning. Its
crowded skyline remains a known composition failure; adding lights does not
convert it into an accepted city layout. The capital case uses the already
mapped American palace; it does not imply broader palace selection coverage.

## Source evidence versus approximation

The expanded source city proof resolves the modern tower's `FX_Light_Blink008`
socket and the compound's `FX_Light_Blink002` socket to rooftop transforms.
These are named blinking-effect attachments, with resource binding still
unresolved. They are not evidence for facade/street lights, and are not enabled
or repurposed by this pass. The prepared `AnalyticLightsNormalized` library is
also not silently enabled.

This pass instead implements an **authored emissive-spill approximation**, as
permitted by `Renderer/docs/environment_lighting_and_ambient_effects.md`.
Offline tooling samples the actual displayed SRGB emissive textures at UV2,
using the same clamped atlas sampling as the separate emission pass. It samples
lower facade triangles, groups their light by cardinal facade direction, and
places bounded proxies on the outer facade planes. Source color and geometry
inform those records; intensity floors/caps, lower-height cutoff, quadrature,
proxy range and transport are explicit Lab choices. This is neither recovered
Civ VI light binding nor physically exact area-light transport.

The point-light-like approximation uses a facade-facing term, receiver normal,
smooth finite range, and the shared environment's continuous night/emissive
activation. All source bodies remain unchanged. City bounding boxes block
spill through neighboring buildings; the owner is excluded because its light
proxy is placed just outside its emitting facade. Exact mesh/terrain shadowing
for local lights is not implemented. Range is bounded to 0.32–0.55 tiles; the
probe caps 48 lights and 12 blockers. A union envelope rejects distant receivers
before the per-light work. It preserves the selected wilderness pixels exactly
relative to the unculled r2 shader.

This implementation injects generic light records into a frozen shader closure
for Lab replay. It adds no source-specific runtime branch and duplicates no city
or terrain pack. Native delivery still needs a generic light-list upload/culling
path tied to authoritative owner visibility, environment invalidation and fallback.
Static lights do not imply continuous redraw or a new presentation clock.

## Reflection and occlusion checks

Disabling the local building blockers changes 488 wilderness night pixels, with
a maximum 34/255 difference. This supports the local occlusion path; box blockers
remain an approximation rather than an exact visible-surface shadow proof.

The capital lake ROI [858,491,885,501] stays inside the same water surface.
Disabling object reflections while retaining local illumination changes 99
pixels there by more than 2/255, with a maximum 65/255 difference. The new spill
versus the preceding lit-city render changes 53 lake pixels, maximum 10/255.
That latter delta combines water illumination and object-reflection response;
it is not an isolated measurement of newly reflected spill. The existing warm
window reflection survives composition. No new multi-height water reflection
support is claimed.

## Reproduction and verification

Run `qa/city_facade_light_probe.py` using Python with Pillow and NumPy. It takes
an existing city augmentation, its rendered source snapshot, a new output
directory and a bounded gain. For example, from the repository root:

```sh
python3 Renderer/terrain_lab/v2/qa/city_facade_light_probe.py \
  --augmentation Renderer/terrain_lab/v2/fixtures/beauty/city-scene-r46/american-modern-s1-wilderness-at6-6/augmentation.json \
  --source-render Renderer/terrain_lab/v2/audits/beauty/out/city-growth-r1/r46-fixed-shadow-frame/render \
  --output Renderer/terrain_lab/v2/audits/beauty/out/city-facade-light-next/wilderness \
  --gain 4
```

Use `--gain 0`,
`--no-blockers`, or `--no-object-reflections` for explicitly labeled controls.
`--resume` permits preparation repairs only before any rendered result exists.
Pillow's missing BC1/2/3 DX10 SRGB aliases are handled in memory by decoding the
identical compressed blocks through UNORM aliases, then explicitly linearizing;
original textures are never rewritten. Non-SRGB emission inputs are rejected
by this bounded probe rather than silently decoded incorrectly.

`qa/city_facade_light_evidence.py` verifies hashes, exact packet reuse, light
owner planes and bounds, source texture identity, day/night/locality controls,
reflection/occlusion controls and Windows results. Its record is
[CITY_FACADE_LIGHT_r1_r3_EVIDENCE.json](CITY_FACADE_LIGHT_r1_r3_EVIDENCE.json).
Fourteen Windows comparisons pass across all five cases; twenty focused tests
pass, including independent SRGB/DDS/atlas checks. One VM dispatch failed before
rendering and was resumed only after its terminal error and empty output were
verified. These checks support the visual assessment, not visual approval.

The envelope optimization's wilderness comparison is pixel-identical. Individual
GPU timings are exploratory and were not an isolated native performance study;
no native cost target is claimed. No injected compilation or full-Lab completion
is claimed for this standalone shader/probe work.

Cleanup is recorded in `CITY_FACADE_LIGHT_CLEANUP.json`: disposable linear render
buffers are removed after evidence, while images, shader closures and prior
candidates remain. The next largest gaps remain coherent city arrangement and
urban ground, environment specular/material richness, and full culture/era/size
plus capital/terrain coverage. Roads stay deferred and no milestone is advanced.
