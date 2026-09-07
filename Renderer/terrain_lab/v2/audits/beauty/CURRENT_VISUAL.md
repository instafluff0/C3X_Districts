# Active focus: city quality

The user redirected the Lab to city sizes, eras and cultures, including night
lights reflected in water, and permits modest cross-tile city footprints.
[Current campaign and findings](CITY_QUALITY_CAMPAIGN.md). Complete source cities
now render inside the fixed 100-tile coastal water scene. A scale comparison
improves roof/facade readability; source variety, underground foundation placement
and material/night-light fidelity remain active. No city result is promoted.

# Preserved natural static water

The user responded “Water looks great” to r6 and requested object reflections.
[water-reflection-r5](WATER_OBJECT_REFLECTIONS.md) now renders real planar object
reflections in a GPU prepass, preserving the r6 water elsewhere. Twenty matched
frames include the fixed benchmarks and coastal holdout. Sixteen focused
Metal/D3D11 comparisons pass, with identical Windows repeat renders. Four shifted
camera probes preserve reflections when their captured objects move offscreen.
Native integration, provider halo/culling coverage and multi-height rivers remain
pending; no milestone gate is closed.
[Native rock comparison](out/water-reflection-r5/review/rocks-native.png).

The user pivoted to water effects after referencing Alex Tardif's walkthrough
and canonical water images. [Exploration and current evidence](WATER_EFFECTS_EXPLORATION.md).
The user then deferred animation and coastal surf. Current water candidate:
**water-natural-r6**, with softer sky reflections and broad calm patches in
20 matched frames across the three fixed benchmarks, long coast, and a wholly
unseen 100-tile coast. [Native previous-prototype comparison](out/water-natural-r6/review/previous-prototype-comparison.png).
Shallow-bed detail and broader parity/cost checks remain. Disabled shaders are
byte-identical; a one-channel 1/255 control repeatability discrepancy is recorded.
No Civ VI equivalence, human approval, milestone closure or pickup replacement.

# Preserved terrain surface richness

The user reoriented the active goal to terrain surface richness at matched
gameplay scale and explicitly requires complete source texture/layer auditing.
[Campaign](SURFACE_RICHNESS_CAMPAIGN.md) and [initial layer findings](GROUND_LAYER_FINDINGS.md).
The first gradient diagnostic is rejected; four disabled-branch control frames
are byte-identical to r3. No new visual best or approval is recorded.

[Recovered ground decal pass](GROUND_DECAL_PASS.md): exact triangle/UV data
for 19 variants, selected override color/height textures, and 20 matched
diagnostic frames including a new wholly unseen 100-tile region. Patch
variation is subtle and GPU cost is excessive; r4 is not promoted. Current
default-off controls remain byte-identical in four frames. Missing high-ground
roles and mountain channel/projection coherence are the next corrections.

[Mountain channel correction](ROCK_CHANNEL_PASS.md): sixteen combined r2
diagnostic frames have coherent projected height/specular and eight additional
snow/stripe channels. Gray rock faces show modestly clearer granular relief;
the overall best is not replaced pending normal composition, crop/wrap and
cost checks. All 72 source mountains have grass base terrain, so desert
material coverage is explicitly synthetic. Four default-off controls are
byte-identical. Continental ground geometry and flat/high/hill layering are
the next source investigation.

[Continental ground and source baking](CONTINENTAL_GROUND_PASS.md): source
continental height fields are recovered, but the first high-ground mask is
rejected for broad pale patches. Bytecode inspection confirms alpha-squared
weighted material baking and separate normalization. Four source-weighted
diagnostic frames change only an upper sand/grass transition; the missing
whole-scene grit remains. Sixteen candidate frames and eight exact controls
are recorded. Next, reconstruct cached height-to-normal/AO processing and the
layer contribution graph. No best, approval, or frozen pickup is replaced.

# Preserved river, forest and jungle campaign

The preceding campaign is preserved. Latest complete combined candidate:
**river-corridor-r3**, following **river-corridor-r2** and **canopy-variation-r1**. Sixteen matched frames
cover the three fixed gameplay regions and the new 100-tile forest/jungle
witness. This candidate has not replaced the complete preserved baseline below.

- [Inland headwater pool and forested channels, native gameplay size](out/river-corridor-r2/review/inland-h12-z1.png)
- [Inland rivers at night](out/river-corridor-r2/review/inland-h00-z1.png)
- [Fresh forest/jungle region and sea outlet](out/river-corridor-r2/review/freshcanopy-h12-z1.png)
- [Stable forest arrangement](out/canopy-variation-r1/review/coastal-h12-z1.png)

[First-loop findings](RIVER_VEGETATION_PASS_r2.md) record changed pixels, the
rejected offshore-pipe result, preserved source appearance, real-source crop
checks and the next visible gaps. Banks, irregular pool shape, bank-rock
placement and broader coverage remain active. No Civ VI-quality acceptance,
human approval or promotion is implied.

The [bank-rock follow-up](RIVER_BANK_ROCK_PASS_r3.md) moves existing source
rocks alongside the actual river and removes unsafe placements. It is a small
detail correction; [inland comparison](out/river-corridor-r3/review/inland-h12-z1.png).
The larger bank and pool-shape gaps remain open.

## Preserved terrain/lighting checkpoint — cleaner shadow receiving

2026-09-06. Sole lead, local Metal. Retained work in progress:
**shadow-receiver-r1**, including **combinedvolcano** as a separate synthetic
witness. This builds on the larger source bodies in `relief-size-r3`.

This is an incremental visual improvement, not Civ VI-level acceptance.
No human approval, milestone closure or Integration promotion is recorded.
Cities, units and improvements remain deferred. LQ0 remains ready/unaccepted.

- [Cleaner wilderness sand — native gameplay comparison](out/shadow-receiver-r1/review/wilderness-h12-z1-comparison.png)
- [Wilderness at night — full native zoom 2](out/shadow-receiver-r1/review/wilderness-h00-z2-comparison.png)
- [Inland mountains and forest shadows](out/shadow-receiver-r1/review/inland-h12-z1-comparison.png)
- [Fixed coast — full native zoom 2](out/shadow-receiver-r1/review/coastal-h12-z2-comparison.png)
- [Long coast — full native zoom 2](out/shadow-receiver-r1/review/longcoast-h12-z2-comparison.png)
- [New 100-tile desert/forest/mountain holdout](out/shadow-receiver-r1/review/freshshadow-h12-z2-comparison.png)
- [New holdout at night](out/shadow-receiver-r1/review/freshshadow-h00-z2-comparison.png)
- [Current combined volcano witness — explicitly synthetic](out/shadow-receiver-r1/review/combinedvolcano-h12-z1-comparison.png)
- [Full wilderness scene](out/shadow-receiver-r1/wilderness/h12-z1-pan00.png)
- [Full new holdout](out/shadow-receiver-r1/freshshadow/h12-z1-pan00.png)

Most thin dark mesh-edge lines across the wilderness sand are gone. Visible
forest and mountain cast shadows remain. The correction sizes the receiver
normal offset to its bounded shadow texel footprint and derives its plane from
unshifted geometry. It changes no terrain, source material, camera, vegetation
placement or shadow caster. The seven real regions and synthetic witness have
32 noon/midnight frames at both fixed zooms; all matched input packets are
byte-identical. Those invariants support comparison and do not grant acceptance.

Mountains still use 1.30 uniform source-body scale and volcanoes 1.60, with
bounded foothill overlap. The previous `relief-size-r3`, `coast-pass-rocks-r8`
and rejected attempts are preserved. Detailed changed-pixel locations,
diagnosis and reproduction are in [SHADOW_RECEIVER_PASS.md](SHADOW_RECEIVER_PASS.md).
[SHADOW_RECEIVER_r1_EVIDENCE.json](SHADOW_RECEIVER_r1_EVIDENCE.json) records
all matched outputs; [SHADOW_RECEIVER_DIAGNOSTICS.json](SHADOW_RECEIVER_DIAGNOSTICS.json)
records the rejected tests and numerical probe.

The three largest remaining gaps are:

1. Mountain/volcano projection, material detail and unproven source physical
   reconstruction. Direct inspection shows that the volcano height texture is
   incorrectly treated as a two-component normal. The first red-height normal
   reconstruction has too little visible benefit and is not selected.
2. Source dune reconstruction and residual facet artifacts. The inherited
   analytic dune body remains an unapproved proxy; cleaner shadows do not
   resolve that source-fidelity defect.
3. Soft shallow-water structure and abrupt cliff/grass joins.

The new [freshshadow benchmark](../../fixtures/beauty/shadow-receiver-foundation/freshshadow/BENCHMARKS.json)
was selected before viewing this candidate and received no local tuning. It is
now a regression witness, not an untuned witness for later acceptance.
Previous relief work is in [RELIEF_SIZE_PASS.md](RELIEF_SIZE_PASS.md); coastal
source work is in [COAST_SOURCE_JOIN_PASS.md](COAST_SOURCE_JOIN_PASS.md).

## Integration pickup

The user requested consolidation for implementation in C3X. The verified
[candidate preparation package](../../../../handoffs/candidates/lab_v2_terrain_lighting_r1/README.md)
contains the implementation map, pinned source snapshot, all retained frame
hashes, local asset inventory and explicit remaining gates. No additional
visual pass or native implementation was made during that preparation.
