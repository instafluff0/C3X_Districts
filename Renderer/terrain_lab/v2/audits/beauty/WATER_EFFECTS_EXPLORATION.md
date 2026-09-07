# Water-effects exploration

The user pivoted from terrain grit to water effects, referencing
[Alex Tardif's Water Walkthrough](https://alextardif.com/Water.html) and the
canonical water images. This is an opt-in Lab exploration. The earlier land
experiments, fixed gameplay benchmarks, frozen pickup, and milestone gates
remain intact. No Integration promotion or Civ VI-quality approval is claimed.

## Current static-water checkpoint

**Follow-up:** after the user said “Water looks great,” they requested object
reflections. [The planar-reflection pass](WATER_OBJECT_REFLECTIONS.md) preserves
the r6 appearance and adds real above-water silhouettes on GPU. It is a separate
candidate with its own evidence and pending delivery gates.

The user's subsequent goal is to make water more natural, explicitly deferring
animation and coastal surf. **water-natural-r6** is the retained water candidate
for that narrower scope. It is opt-in through `Q3_NATURAL_WATER`; the earlier
water-effects-r2 prototype, complete preserved best, and frozen pickup remain.

At native gameplay size, the bay has softer reflected wave faces and broad
calm patches. Compared with water-effects-r2, fewer dark/light ridges continue
across the entire bay. The new source-selected coast shows the same interruption
of surface activity without obvious tile copies. Small inland pools remain
relatively calm; the river channel branch is unchanged. Midnight water stays
blue and subdued under the shared lighting. This is an incremental visual
improvement, not a claim that the overall scene matches Civ VI.

- [Earlier water prototype versus current water, native pixels](out/water-natural-r6/review/previous-prototype-comparison.png)
- [Preserved baseline versus current water, native pixels](out/water-natural-r6/review/native-water-comparison.png)
- [Canonical coast versus current water](out/water-natural-r6/review/canonical-water-comparison.png) — unscaled pixels, different source zooms.
- [Previously unseen 100-tile coast](out/water-natural-r6/review/freshwater-h12-z2-pan00.png)
- [Fixed coast at midnight](out/water-natural-r6/review/coastal-h00-z2-pan00.png)

The three largest remaining visible gaps are shallow-bed detail/softness,
excess directional uniformity in some open-water patches, and a shared sea/pool
response that still lacks authoritative water-body classification. The current
change addresses the surface response. It does not add scene refraction,
reflection of nearby geometry, or change the seabed/shore geometry.

### Diagnosis and rejected attempts

1. Natural r1 separated mean-plane volume lighting from normal-driven reflection,
   but erased too much visible detail. Rejected.
2. Natural r2 increased source slopes and exposed scratch-like narrow highlights.
   Rejected. R3 broadened the response but remained too faint.
3. The previous material multiplied a roughly .02 Fresnel value by the shared
   rig's .04 noon response: about .0008. Natural r4 instead interpreted the rig's
   small control as base reflectance, restoring visible sky response, but its
   broad striped highlights were excessive. Rejected.
4. Natural r5 reduced slope amplitude, mixed large/small/crossing source detail,
   and limited the sky gradient. R6 added broad periodic calm lanes. Retain r6
   as the next water candidate; keep every earlier expanded shader and render.

This is a deliberate Lab interpretation of the generic lighting control, not
recovered Civ VI Fresnel semantics or a physically calibrated sky. The sky lobe
uses shared ambient/sun/moon radiance. Three bound source LEAN0 textures supply
adapted slopes, including the previously unused secondary small-wave slot 35.
The spectral bed attenuation and optical-depth geometry are unchanged. Water
coverage includes the reflection term before the existing single premultiply.

### Matched evidence and limits

Twenty final frames cover coastal, inland, wilderness (100 tiles each), longcoast
(128), and freshwater (100), noon/midnight and both frozen output sizes. Complete
geometry/shadow packets are hash-identical to each region's baseline. All output
linear values are finite and validity masks match. Noon zoom1 changed pixels:
134,119 coastal; 7,885 inland; 61,799 wilderness; 217,842 longcoast; 283,217 holdout.
These numbers locate the affected image area; they do not establish quality.

The **freshwater** diagnostic name identifies a new water holdout, not a lake
classification. Source origin [52,30], extent [10,10], halo 6. It contains 24 coast,
39 sea and 6 ocean tiles. All 100 visible source tiles were outside earlier beauty
regions. The selector was frozen before viewing: maximize coast count then total
water among unseen crops with 30–75 water, at least 20 coast and at least 15 sea/ocean.

Four disabled compiled Metal shaders are byte-identical to the preceding disabled
water control. Three rendered controls are byte-identical; midnight zoom1 differs
in one channel of one pixel by 1/255. Its raw half-float output differs in 14 values,
maximum 0.0000611. An identical repeat also changes that one final channel, so
the discrepancy is repeatability at a rounding boundary, not evidence of an
enabled shader branch. It remains recorded; no existing parity gate is relaxed.

Four water-only coordinate tests shift both material axes by 50, corresponding
to the BIQ's raw-X wrap of 100. Maximum final-channel difference is 1/255 and
mean difference is below .00012 code values. This checks the new water field's
periodicity on GPU; it is not full shifted-crop/geometry/shadow parity.

The Lab workflow passes 132 Python and 12 Node tests. D3D parity, controlled cost,
full crop/edge checks and human visual acceptance remain pending. None of these
supporting results promote the candidate or complete a milestone.

Reproduce new captures with `qa/water_effects_pass.py --region REGION --natural 6`.
Controls use `--disabled` or `--coordinate-shift 50`; the driver refuses completed
destinations. `qa/verify_natural_water.py` produces the review sheets and
`WATER_NATURAL_r6_EVIDENCE.json`. To reproduce a historical revision after editing
the live shader, use `qa/replay_shader.py` with that output's preserved
`shaders/source.hlsl` and original baseline report, into a new output directory.

## Earlier animated-phase/surf exploration (preserved)

Tardif combines wave deformation, scrolling normals, fragmented highlights,
foam, depth softening, refraction and reflection. His reflection/refraction
stages require scene color and depth inputs. These are useful techniques to
evaluate individually, not a shader to drop unchanged into this renderer.

Canonical `sea_and_shore.png` shows fine directional open-water texture,
interrupted surf near the coast, and textured shallow bottoms. `river.png`
shows calmer, readable channels and clear confluences. The canonical images
have different zooms; actual gameplay output remains the acceptance scale.

The current Lab already has continuous shore distance, optical depth, source
water textures, a separately shaded bed, shared lighting and shadow receiving.
It lacks a continuously advancing water clock and opaque-scene color/depth
sampling in this material pass. It deliberately suppresses the old captured
foam geometry. These facts favor a first pass on the existing water surface.

`water_effects.hlsl` independently implements three directional surface-normal
waves, drifting source detail, wave-facing sky response, and broken shore foam
using the existing foam texture and continuous shore distance. Wavelengths
are compatible with the map's wrap period; the breakup hash is periodic too.
GPU crop/wrap parity is still pending. Geometry, shore shape, water depth,
object placement and the camera are unchanged.

Foam uses shared sun/moon/ambient illumination and correct coverage composition.
The water material is still premultiplied only once by the existing output
wrapper. No self-lit white foam or displaced tile edges are introduced.
Normal motion and foam phase currently come from a deterministic shader
constant. Two phases demonstrate motion, not a live runtime animation system.

## Visible result

`water-effects-r2` contains 20 matched frames: the three fixed 100-tile coastal,
inland and wilderness scenes and the 128-tile long coast, noon/midnight at
both zooms; the long coast also has a second phase at 1.5 seconds.
Every frame reuses its previous-best complete geometry/shadow packet.

In the native long-coast crop, the broad dark water now has visible directional
ripples, and the small island and near shore have interrupted pale surf. These
are concrete changes toward the surface activity in the canonical coast.
Noon zoom1 changes 189,868 pixels in the long coast, 115,717 in the fixed coast,
54,429 in the wilderness, and 5,327 in the inland scene. These counts locate
the effect; visual inspection, not counts, supports the limited improvement.

The night comparison retains coast readability without daylight-bright foam.
The inland river itself is unchanged. Surf currently also appears in enclosed
pools; this is a visible limitation, so the prototype is not a promoted best.
Open-water waves can still look too regular. Shallow-water softness remains.

- [Native gameplay crop, before/after](out/water-effects-r2/review/native-water-comparison.png)
- [Full long coast, actual zoom2](out/water-effects-r2/review/longcoast-h12-z2.png)
- [Fixed coast at midnight](out/water-effects-r2/review/coastal-h00-z2.png)
- [Wilderness comparison](out/water-effects-r2/review/wilderness-h12-z2.png)

## Earlier proposed sequence (superseded by static-water scope)

1. Add authoritative sea/lake/river classification and coast exposure so surf
   belongs on active shores; vary wave patches to reduce repetition.
2. Add a bounded animation clock and map-only redraw policy. Keep simulation
   time separate from hour-of-day lighting and freeze phase for capture/tests.
3. Add limited shallow-bed distortion and improved depth transition. True scene
   refraction needs an opaque-color/depth copy and rejection of samples above
   water; the current effect does not implement it.
4. Evaluate sparse highlight breakup and low-amplitude swell at gameplay size.
   Mesh displacement and screen-space reflections need independent visual/cost
   evidence before expanding the rendering pipeline.

For eventual animated-water promotion, add continuous-motion inspection,
GPU shifted-crop/wrap checks, controlled cost measurement, D3D parity and a
combined human visual checkpoint. No gate has been waived.

## Verification and reproduction

Run `qa/water_effects_pass.py --region longcoast` and the same command with
`--phase 1.5` for deterministic phase replays. The driver refuses to overwrite
completed diagnostics. `qa/verify_water_effects.py` verifies packet identities,
20 frame comparisons, four byte-identical disabled controls and four changing
phase pairs, and generates the review images. Evidence is in
`WATER_EFFECTS_r2_EVIDENCE.json`.

The Lab workflow initially found that an unrelated canonical screenshot had
been renamed to `unit_texture_and_civ_colors1.png`. Its dimensions and SHA-256
exactly match the existing contract; only the expected filename was corrected.
No reference image or expected pixel hash was changed.

## Preserved land-work checkpoint

Before the pivot, the source normal filter was verified against original DXBC
execution on a 1,024-pixel synthetic field with zero RGBA8 error. The original
source AO shader was executed offline for five ground materials; flat-ground
and million-pixel periodic-shift controls passed. `source-normal-cache-r1/r2`
store provenance; `cached-normal-r1/r2/inland` each has four composed frames.
The visible gains remain subtle and unaccepted. Source cache scale and
combined-layer bake ordering remain unproven. The attempted freshcache land
holdout found no region satisfying its constraints and registered nothing.
This work is preserved; water is now the user's active exploration.
