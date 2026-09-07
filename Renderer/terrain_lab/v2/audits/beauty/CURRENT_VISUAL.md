# Active focus: city quality

The user redirected the Lab to city sizes, eras and cultures, including night
lights reflected in water, and permits modest cross-tile city footprints.
[Current campaign and findings](CITY_QUALITY_CAMPAIGN.md). Complete source cities
now render inside the fixed 100-tile coastal water scene. **City r8** retains the
source-origin foundation correction, recovers the separate source light-map UVs,
and adds stronger window emission with soft GPU HDR glow. Roof light artifacts
from earlier UV0 experiments are rejected. Modern reflected lights have matched
lights-off/glow-off controls and four passing Metal/D3D11 comparisons.
[Modern native-size comparison](out/city-scene-r8/review/modern-night-native.png),
[medieval native-size comparison](out/city-scene-r8/review/medieval-night-native.png).
Full culture/era/growth coverage, local ground-light transport, material normal
fidelity and clearance remain active. No city result is promoted.

The subsequent r11 growth probe counts complete neighborhoods by footprint;
colonial town/city/metropolis layouts preserve their existing building prefix.
**r13 capital comparison** adds the source Mesoamerican palace, with a reserved
center site and seven unchanged houses in the no-palace control. Both gameplay
zooms and noon/midnight render on the same 100-tile coast.
[Capital at gameplay size](out/city-scene-r13/review/capital-native.png).
Other cultures and the colonial palace's unresolved components remain open.

The user selected **one era per city**; the mixed-era r17 diagnostic is rejected.
The [single-era ground/palace pass](CITY_GROUND_AND_CAPITAL_PASS.md) adds source
paving under all selected buildings in modern/medieval r18, with a small visible
base-edge improvement and shoreline clipping. r19 connects the American palace
from the complete 47-root pack, but foreground towers obscure its facade and the
wider city is less coherent; it is not a new city best. Eight Windows comparisons
pass. Broader size/culture/fresh-region coverage remains open.

The [r21/r22 capital composition pass](CITY_COMPOSITION_PASS.md) improves the
American modern palace's visible facade, preserves building positions across
4/7/11-component growth, and composes source paving with the revised layout.
A region previously unused for city tuning also renders with unchanged source
scale and recipe. Ten Windows comparisons pass. This is a provisional modern
capital improvement; material richness, wider culture/era/region coverage and
all visual/integration gates remain open.

The [r28 material pass](CITY_AO_MATERIAL_PASS.md) restores the tested medieval
source AO atlas through UV1, preserving diffuse UV0 and light-map UV2. Combined
source paving and repeat addressing survive the change; eaves and wall recesses
read more clearly. An inland region new to city tuning also renders. Fourteen
Windows comparisons pass, including the earlier-era capital comparison. This is
partial material restoration: tangent/LEAN/gloss, broad ground coverage, other
families and full city acceptance remain open.

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

### Opacity and wilderness composition r34–r39

r34 is a partial modern material candidate: restored roof openings change 259
pixels at gameplay size in both day and night. r36 disabled controls exactly
match r32. Direct-only metalness r35 remains unselected. Wilderness r37 exposed
forest overlap; r39 clears the four-body stage at the same anchor, but the
seven-body stage still fails after 33 bounded alternatives. The site is now a
regression witness. Eight Windows comparisons pass, including the rejected r37
composition; this is supporting evidence, not visual approval. See
[CITY_EXTRA_MATERIAL_PASS.md](CITY_EXTRA_MATERIAL_PASS.md).

### City growth and fixed shadow frame r40–r49

The provisional wilderness candidate is r46 geometry rendered at
`out/city-growth-r1/r46-fixed-shadow-frame/render/`. It separates the skyline,
clears the unchanged forest and changes only the city/shadow/glow area compared
with r37. r47 preserves its four-body prefix; r48 fits eleven inland buildings.
Wilderness eleven-body growth and the crowded freshshadow holdout remain open.
The fixed shadow frame prevents distant terrain shadow resampling from being
miscounted as improvement. Ten Windows comparisons and seventeen focused tests
pass; no overall city acceptance or promotion. See
[CITY_GROWTH_PASS.md](CITY_GROWTH_PASS.md).

### Local city night lighting

`out/city-facade-light-r3/` contains provisional facade-spill candidates for
wilderness, medieval, inland, freshshadow and capital cities. Warm pools beneath
windows improve grounding without changing their layouts or source packets.
The technique is an authored emissive-spill approximation with city-box blockers.
Daylight remains unchanged apart from isolated 1/255 rounding, and the existing
capital lake reflection survives. Fourteen Windows comparisons pass; broader
city composition/material quality and all gates remain open. See
[CITY_FACADE_LIGHT_PASS.md](CITY_FACADE_LIGHT_PASS.md).

### Era grounding atlas r1

`out/city-ground-binding-r1/` composes era-specific paving on all five facade-light
candidates while preserving their source ground geometry and UVs. The inland
modern city shows the clearest small gain; pads remain mostly hidden beneath
buildings. Fourteen Windows comparisons pass. No-op packets/images and the
capital lake reflection ROI match exactly; two noon frames have one isolated
1/255 pixel outside the city. Full city ground and quality remain unfinished. See
[CITY_GROUND_BINDING_PASS.md](CITY_GROUND_BINDING_PASS.md).

### Connected modern settlement ground

Modern local candidates move to `out/city-settlement-ground-r2/`: inland,
wilderness, freshshadow, capital and small. They join existing bases with paved
ground while preserving original body/terrain geometry, materials, lighting and
shadow maps. Small/medium paving coordinates stay fixed; the lake reflection ROI
and disabled control are exact. The medieval underlay is rejected, retaining
`out/city-ground-binding-r1/medieval/render`. Twelve Windows comparisons pass.
Small-city towers appear too early; growth hierarchy and broader culture/era
coverage are the next larger visual work. See
[CITY_SETTLEMENT_GROUND_PASS.md](CITY_SETTLEMENT_GROUND_PASS.md).

The [growth hierarchy pass](CITY_GROWTH_HIERARCHY_PASS.md) replaces the small
modern stage's tallest tower with a lower source-body prefix. The combined
[gameplay sequence](out/city-growth-hierarchy-r1/inland-growth-native.png) keeps
source scale, earlier placements, paving and night lighting stable across sizes.
Freshshadow receives the same recipe without local tuning. Twelve Windows frames
pass; no general city acceptance follows. Daylight facade variety, broader
single-era culture/capital coverage and the large wilderness fit remain open.
The prior capital stays selected; the new wilderness reflection control is weak.

The [individual-house culture pass](CITY_CULTURE_DENSITY_PASS.md) improves the
Asian medieval and ancient-brick medium neighborhoods using sixteen houses at
unchanged source scale. The [matched Windows images](out/city-culture-r2/selected-native-comparison.png)
retain the same cameras, terrain, paving and light recipe. All twelve Windows
frames complete; only the four small baseline Metal comparisons pass so far.
Eight denser-frame Metal comparisons remain pending after prolonged compilation;
replace shader-embedded light data before repeating those trials. Large scattered
growth is unselected, and full culture/era/size/palace coverage remains open.

The [buffered-light pass](CITY_LIGHT_BUFFER_PASS.md) resolves the dense-city Metal
compilation problem. Eight matched Windows controls are exact, and twelve new
Metal/D3D comparisons pass. The [full-spill comparison](out/city-light-buffer-r1/full-light-native-comparison.png)
restores 53 proxies in both Asian medium scenes, improving local night ground and
facade readability without daylight or outside-city changes. These are the new
local night candidates; large scattered growth remains unselected. The next
visible focus is connected large-city placement and broader palette/palace
coverage. Existing capital lake reflections and all gates remain preserved.

The [connected-growth pass](CITY_CONNECTED_GROWTH_PASS.md) selects the r65 Asian
large neighborhood and r72 river-clear small layout provisionally.
[Matched large city](out/city-connected-growth-r1/large-full-light-native.png),
[river correction](out/city-connected-growth-r1/river-clearance-native.png).
The prior r64/r68/r69 river-site geometries overlap the rendered corridor and
are superseded for clearance; their fixed-geometry lighting evidence remains
valid. Ancient large r70 and previously city-untuned coastal r73 add coverage.
Twelve Windows comparisons pass. Medium/large river growth, architectural
landmarks, facade variety and all gates remain open; next connect broader
single-era palace styles with the new river exclusion.

The [palace composition pass](CITY_PALACE_COMPOSITION_PASS.md) selects the
East Asian single-era r93/r94/r92 capital growth sequence provisionally.
[Gameplay day/night growth](out/city-palace-composition-r2/capital-growth-native.png)
keeps the palace and every earlier house fixed; the detached small-stage gap
is closed. [Palace off/on](out/city-palace-composition-r2/palace-off-on-native.png)
isolates its courtyard, stepped silhouette and night windows. Ancient inland
r77 and [main coastal r98](out/city-palace-composition-r2/coastal-capital-native.png)
add style coverage. Twelve Windows comparisons pass. Separate coastal r99
capital fit remains unresolved; all gates stay open. Earlier ordinary-city
bests and the American capital/lake reflection witness remain preserved.

The [environment material trial](CITY_ENVIRONMENT_PASS.md) adds provisional
modern material candidates under `out/city-environment-r2/inland-large` and
`wilderness-medium`. [Matched modern day/night](out/city-environment-r2/selected-modern-native.png)
shows a modest cool-facade reflection gain at unchanged geometry and lighting.
The Asian roof trial is unselected; keep the preceding palace pass appearance.
Eight current Windows comparisons and the exact disabled control pass.
The old American capital/lake scene still needs complete material intake; its
previous reflection evidence remains preserved. Actual environment calibration,
facade detail, broader city coverage and all gates remain open.
