# Drawing beautiful seasons over the configured terrain

The practical approach is a **material policy over the existing summer scene**.
Keep the selected terrain's textures, relief, decals, forests, opacity and
placements. Tint them for autumn and spring; coat eligible surfaces for winter.
Apply the policy after source material composition and before lighting. This
avoids four copies of the world or four complete art packs.

Executable Mac Metal prototypes now render all three seasons, Summer, and an
autumn tint/hue-blend comparison with both `TerrainNormalized` and
`Civ5EnvironmentSkin`. They use `test.biq`, actual source materials, production
terrain/forest mesh statement bodies, river/coast queries, surface decals and
coastal cliffs. [Inspect the full scenes and comparisons](../../out/seasons/programmatic/index.html).

These are material feasibility results, **not production render-graph parity or
visual acceptance**. The harness substitutes diagnostic water, shoreline
composition, field textures and a shadow atlas; it omits buildings and units.
Consequently the selected concepts' surf, water reflections, canopy proportions,
soft compositing and some fine detail differ. Small black marks and triangular
shadow artifacts also occur in the diagnostic Summer. Integrating this policy
should preserve the current production providers for those systems. Do not port
the diagnostic adapters or change summer assets to imitate imagegen alterations.

## The executable recipe

`seasonal_policy.hlsl` is the generic implementation. `render.py` adapts the
current material shaders in temporary memory/files; `scene.mm` binds existing
packs read-only and renders a bounded viewport. `recipes.json` holds offline
pack calibration and atlas dimensions. There are no source-game names in the
shader ABI. No production code, reference images, injected code or game install
was changed; the Windows VM was not used.

| Season | Ground treatment | Foliage treatment | Preserved source information |
| --- | --- | --- | --- |
| Autumn | Luminance-preserving multiplicative RGB tint. Grass is warm olive/gold; plains is brighter honey-straw; floodplains retains a greener undertone. Desert and exposed stone stay intact. | A mostly gold, then amber, then sparse russet tint on deciduous foliage. Leaf eligibility protects wood; evergreen and tropical roles retain their autumn colors. | All source normals, height detail, AO, opacity, relief, source color variation and decal pattern. |
| Winter | Cool the exposed substrate, then blend granular snow with continuous biome, slope, stone and world-noise coverage. Retain source height grain and rock luminance variation beneath the coating. | Five authored snowy pine color/gloss overrides; retain original normals and opacity. Other eligible foliage receives an upward-facing frost layer, with much of its original mapped normal retained. | Exact macro geometry, rocky outlines, dune shapes, source grain, crevice/AO response, caster topology and source opacity. |
| Spring | A small green lift over summer grass; a stronger fresh-green tint on floodplains. Irregular clusters of pink, baby-blue, yellow and white flowers. Plains has fewer flowers; desert/stone excludes them. | A slight fresh-leaf tint on deciduous bodies. | Existing material pattern and normals. Flower tint replaces only sparse surface coverage; no replacement lawn texture. |

Autumn's default is tinting, as requested. Its core operation is:

```text
colored = source_linear_rgb * lerp(1, tint_rgb, strength)
result = colored * luminance(source) / max(luminance(colored), epsilon)
result *= small_brightness_adjustment
```

This retains local light/dark variation and original color differences. A hue
blend also preserves luminance, but pushes more pixels toward one common palette.
The gallery lets these approaches be compared directly. Neither is a screen-wide
color filter: sand, rock, plains, grass and tree bark have separate eligibility.
The greener Civ VI material needs more red-channel lift than the warmer Civ
V-style material. Those are offline pack parameters, not shader branches based
on a game's name. Unknown packs use a restrained default and can supply their own
calibration. Spring also uses source multiplication, with a separate floodplain
profile to prevent floodplains and plains from merging.

The [refined Civ V autumn study](FALL.md) carries one stable palette value per
original tree: gold-dominant crowns with amber and occasional russet. Retain that
value in generic instance appearance data in production, rather than sampling a
palette field across each crown. Existing forest-floor decals receive a broken
gold/amber litter tint with water-boundary eligibility. The ground is quieter
olive with distinct honey-straw plains. The target-led `--fall-beauty` candidate
now uses stronger leaf tint and brightness, normalized original-atlas leaf
eligibility, restrained wrapped leaf light and broader bronze/olive variation.
Its noon grassland/plains separation is 16.30; the smallest pair is 11.72.
These are material diagnostics, not beauty acceptance. It still falls short of
the selected concept's canopy light/shape and rich terrain treatment. The Lab
coast/validity/water corrections have a separate corrected Summer baseline and
must not replace production shared providers. The older table below describes
the preceding four-season gallery; current autumn/winter notes carry refinement
evidence and open visual gaps.

## Winter: snow without losing the summer landscape

The [current Civ V winter study](WINTER.md) adds exposure masks, authored snow
decals, local drift relief, stronger canopy shelter and retained summer texture
contrast, with unchanged tree meshes. Snow decals derive their bump response
from red height; their unresolved green channel is retained and unused. Its
separate review is the current winter material experiment; the original gallery
preserves the preceding four-season scenes.

The current authored trial coverage targets are grassland `.93`, plains `.86`,
desert `.73`, tundra `.985`, floodplains `.965`, varied continuously by two noise
scales. These are C3X recipe values, not recovered Civ VI engine settings. A
slope envelope reduces the coat on vertical faces; exposed stone scales the
remaining coverage by `.56`. This leaves recognizable rock faces and granular
outcrops beneath luminous snow rather than burying everything in white paint.

Biome identity uses several cues together:

| Biome | Snow / substrate direction | Structural cue |
| --- | --- | --- |
| Grassland | Soft celadon-blue snow | Fine grass/soil grain remains underneath; slightly uneven coverage. |
| Plains | Brighter pearl-white snow | Straw/earth texture and a lighter overall value. |
| Desert | Pale silver with a restrained warm undertone | More exposed substrate and existing dune/decal structure. |
| Floodplains | Cool teal-white snow | Fertile surface pattern and the actual river corridor remain visible. |
| Tundra | Deeper ice-blue snow | Strongest snow cover and existing coarse tundra/rock grain. |

Snow albedo starts from the configured pack's snow texture. The policy adds
restrained grain from the existing snow-decal texture, source height detail and
source rock luminance. Base material color is cooled to remove the strong beige
cast. Albedo changes do not regenerate terrain vertices or macro height fields.
In particular, do **not** switch a summer hill's geometry to the upstream snow
hill height map: that would move rocks, trees and roads between seasons.

Normals combine the current mapped normal with a snow micro-normal derived from
the snow height channel and the true geometric normal. Source rock normals retain
more influence than coated soil; winter tree normals retain more influence than
a flat canopy normal. Normalize safely after blending. Existing crevice, AO,
sun/moon directions and shadow providers remain authoritative. Snow coating
must not erase the forest's light/shade separation or flatten hill relief.

The layered recipe distinguishes cooler grassland, lighter pearl-gray plains,
pale ivory desert, teal-white floodplains and stronger ice-blue tundra. Ground
and decorative snow patches share one palette. Local jittered pillows add
material normals at a middle scale; deposition follows source material highs,
slope and shelter. Keep actual elevations, tree recipes and anchors fixed.
The Mac study's second shadow atlas omits ground decal carriers and uses actual
receiver-plane derivatives. Its water adds cached slope textures and restrained
bank frost. Those are diagnostic adapters: production should keep its existing
shared shadow/water providers and consume only the seasonal material policy.

The Lab corrects the source material's warm key response, cools skylight and
uses recipe-controlled winter display exposure. It holds the same noon sun
direction. In the real game, seasonal lighting must remain part of the shared
environment policy, governed by
[the existing lighting contract](../../../docs/environment_lighting_and_ambient_effects.md).
Apply any agreed correction consistently to terrain, foliage and reflections;
do not multiply it both in baked albedo and relighting, invent another sun angle,
or lose the existing moon/emissive behavior. Blue shadows should come from the
shared cool sky response and snow material, not painted shadow shapes.

## Spring: small authored flowers, placed programmatically

`asset_inputs.py` isolates four complete heads from the installed transparent
`TEXTURE_FX_Blossoms` payload. It retains petal light/dark detail and alpha, removes
the baked tint, adds transparent padding and generates a tiny mipmapped 128×32
atlas. The recipe then gives those shapes harmonious pink, blue, yellow and white
tints. Exact source particle/plant bindings remain unproven; using the isolated
heads as meadow flowers is a C3X adaptation.

The shader uses deterministic world cells with jitter, a broad meadow envelope,
finer cluster density and slower color variation. Thus flowers form irregular
patches with different colors, rather than an even confetti grid. It tests only
nearby eligible candidates, uses explicit texture gradients, and fades the
pattern as the projected footprint becomes too small. Derivatives are computed
before divergent eligibility branches. There is no particle emitter, moving
wind state, frame-time random seed or continuous redraw requirement.

Eligibility is grass `1.0`, plains `.24`, floodplains `.75`, tundra `.055`, desert
`0`, multiplied by `(1 - exposed_stone)`. Use the real road/river/building masks
in production too: flowers should not appear on paved roads, open water, roofs,
rock faces or inside a building footprint. Forest-floor treatment must keep the
same seasonal profile as the adjacent ground, with optional density attenuation
under dense canopy. The Lab's river coverage and stone exclusion are exercised;
its scene does not contain roads or buildings, so their exclusion is still a
production-port task.

Without an authored flower atlas the same placement policy draws small analytic
five-petal shapes. Without source snow art, a neutral generated snow material and
flat micro-height can supply the winter coat. Rich packs use height/normal/detail
channels; texture-only packs retain their own color and use geometric normals.
These are runtime design contracts. This Lab loader itself expects the current
compiled natural payload with 32 bodies and 35 recipes; it is not a universal
loader for arbitrary texture-only packs.

## Upstream art: what is actually useful

The broader installed-art census covers 13 relevant Base/DLC ArtDefs and 76
matching shared texture payloads, representing 45 distinct payload hashes.
Availability is not the same as a usable binding. The read-only source probe
additionally decoded **all 11 Base `CLUTTER_SNOW` decals**, including actual
texture classes, descriptor footprints, mesh UV bounds and ArtDef placement
parameters. This is stronger evidence than guessing from filenames.
The confirmed names, hashes, descriptors and UV findings are preserved in
[source-art-evidence.json](source-art-evidence.json), outside disposable output.

| Input | Evidence and current use | Decision |
| --- | --- | --- |
| Ground and mountain snow albedo/height/gloss | Already present in both terrain packs and declared by source terrain ArtDefs. Ground snow is sampled by the executable policy. | Reuse local channels; provide generic optional material slots. |
| Snow surface decal B/H/G/FOW channels | All 11 decoded source decals bind `TEXTURE_TER_Snow_Decal_*`; the existing water catalog already contains the normalized DDS files. Full authored meshes/UVs/placements are now preserved in the local `assets/snow-decals/` generic pack. | The Lab uses only selected granular texture regions as a material detail layer. Its four-cell crop recipe is an adaptation, **not** a recovered four-cell source placement rule. For richer decals, consume the preserved generic descriptors and meshes. |
| Five snowy pine bodies/clumps | Existing pair probes confirm equal positions, UVs and topology. Current foliage comes from the alternate vegetation pack; some snowy normals differ and extra opacity is present. | Reuse authored winter color/gloss while retaining original geometry, normals and opacity. Do not silently swap whole material descriptors. |
| Snow hill macro height | Already present in the local relief assets. | Preserve summer macro relief; use a surface coat instead of moving the landscape. |
| Blossom sprite | Preserved original payload and DDS; four complete alpha components isolated and used. | Tiny local atlas, generic runtime metadata, analytic fallback. No installed-game dependency for adaptation. |
| Colored/white flowered foliage atlases | Preserved 512×512 DDS payloads with carrier regions. | Do not repeat whole atlases over terrain. Use only with proven UV/material bindings; unnecessary for the first meadow pass. |
| Snowy rocks, floor marks and DLC variants | Source census confirms candidates; some are duplicate payloads or source-specific structures. | Use selectively when they improve a demonstrated gap. Existing rocks retain their shape/material detail under the coat; no need to import every snowy object. |
| Fall/spring four-season art | No established four-season terrain recipe was found in the inspected source data. | C3X tint/placement policy supplies these treatments. No original engine seasonal behavior is claimed. |

We are using the most useful confirmed channels, and have identified the next
useful decal route. Importing every asset would add storage/residency and often
bring unrelated carrier regions, snow boulders or duplicate textures. The
source probe leaves installed packages and existing normalized packs untouched.
The bounded `cache_assets.py` compiler preserves the next useful decal route and
flower inputs in `assets/`, outside disposable output. It adds only 5.08 MiB;
full source packages and raw compiler reports are not copied. Its manifest hashes
every retained file. `cache_assets.py --verify` validates all saved inputs and
the source-independent generic snow pack without opening the source installation.
Licensed inputs stay local. A distributable C3X pack can replace every optional
input with original art; runtime consumes DDS/material IDs/semantic roles, never
BLP/ArtDef data or installation paths.

## Concrete production port

This is the implementation sequence once the Lab policy is taken into Game
Integration. Coordinate with the performance work before changing its caches.

1. **Resolve active materials through the existing definition system.** Consume
   the final terrain, decal, mountain and foliage material IDs after scenario
   overrides and pack inheritance. Add optional generic seasonal metadata to
   those resolved materials: biome role, deciduous/evergreen/tropical role,
   leaf mask, exposed-stone/snow eligibility, winter override, atlas cells and
   recipe parameters. Preserve partial overrides. The Lab's `configured_pack`
   resolves the default grassland rule to one compiled pack; it does not emulate
   every mixed per-material rule. Do not carry that shortcut into runtime.
2. **Keep one generic policy implementation.** Move the tested policy into a
   shared renderer shader include, compiled by both Metal and D3D. Give it sampled
   linear albedo, mapped/geometric normals, gloss, canonical world position,
   authoritative biome/floodplain weights, source grain and semantic eligibility.
   Prefer an authored leaf/stone mask where available. The Lab's color-based
   mineral/chroma heuristics are fallback diagnostics, not sufficient metadata
   for arbitrary modded textures, mossy bark or unusual palettes.
3. **Apply it after all source material layers.** In
   `native/source_fidelity/terrain.hlsl`, insert after cliff/volcano composition
   and before `SANDBOX_TERRAIN_MATERIAL` output or direct lighting. This covers
   ordinary ground, hill decals, grass/plains/desert patches and vegetation
   floors. In `mountain.hlsl`, cover both the early flat-ground return and the
   final mountain-to-ground handoff before material output; preserve the common
   fringe and rock coverage. The Lab crop does not exercise the production early
   flat-return variant. In `objects.hlsl`, apply foliage policy after original
   color/opacity/AO/normal sampling and before lighting; apply coating only to
   explicitly eligible natural objects/cliffs. Preserve instanced/reflected
   variants and both color and depth coverage.
4. **Include the underlay and shoreline material branches.** The retained
   `sandbox/fresh_pipeline.h` compiles `PSSandboxMaterialAlbedo` and
   `PSSandboxMaterialNormalWorld` from `hydrology.hlsl`. Apply the same policy
   after `q3_shore_material`, `q2_material_form` and coherent rock composition,
   **before both `Q8_DEBUG_ALBEDO` and `Q8_DEBUG_NORMAL` returns**. Update the
   albedo and normal variants together. Use the original shore/river/dune masks;
   leave water shaders and the accepted surf/reflection graph in charge of water.
   This prevents a summer beach ring, floor or patch floating above winter snow.
5. **Use safe per-program bindings.** The Lab's b5 and texture slots 89–106 are
   diagnostic assignments, not a global ABI. `hydrology.hlsl` already uses slots
   89–99 for river rocks/features/roads; other shaders use different layouts.
   Allocate and validate each production shader's slots, preserve shadow-page b4
   and instance b9 contracts, and bind neutral fallbacks for missing optional
   art. The Lab's six `float4` state vectors show the data needed, not mandatory
   production slot numbers. Do not paste these register assignments wholesale.
6. **Keep retained material and lighting responsibilities separate.** Seasonal
   albedo/normal evaluation belongs in retained material construction. Hour,
   shared shadow sampling, local lights and final color response belong in
   relighting. The Lab's coat gloss is used by direct shading; the retained
   terrain properties currently encode cavity/occlusion/inland and relighting
   uses fixed roughness. Start with its matte response. To reproduce variable
   snow gloss exactly, add an explicitly budgeted auxiliary material channel or
   a cheap documented coverage reconstruction; do not steal alpha/AO/cavity
   channels or claim exact direct/deferred equivalence before testing it.
7. **Refresh all affected views on a season change.** Give
   `prepare_material_cache`, `prepare_terrain_material`, and
   `prepare_reflected_terrain_material` an identity containing their existing
   content/view identity plus season, recipe generation, resolved pack/material
   generation and semantic metadata. Invalidate dependent static reflection,
   static scene and published output caches too. Reuse resident meshes,
   placements, biome/river/coast pages and original-opacity shadow casters.
   Present one coherent generation, never mixed summer and winter cache pages.
8. **Audit legacy signatures before claiming geometry reuse.**
   `native/terrain_scene_runtime.cpp::terrain_frame_signature` computes an
   environment hash and currently folds it into `geometry`. `scene_revision()`
   in the fresh pipeline depends on that geometry identity. Several native tile
   keys also contain hour/season outside particular profiles. The existing
   separation is therefore incomplete. Split material/environment changes from
   pure geometry keys where correct, keeping complete frame/publication validity
   and shadow-light invalidation intact. A change of light direction still
   changes shadow output; a foliage tint alone need not rebuild its caster.
9. **Use the existing C3X clock without requiring legacy PCX art.**
   `prepare_custom_renderer_frame` already supplies `frame.season`:
   Summer=0, Fall=1, Winter=2, Spring=3. Existing
   `patch_perform_interturn_in_main_loop` gates clock advancement on
   `day_night_cycle_img_state == IS_OK` and suppresses redraw on native image
   reload failure. Existing `patch_Map_Renderer_load_images` initializes that
   legacy path. For custom rendering, separate clock advancement/renderer
   invalidation from native `DayNight/SEASON/HOUR` image availability. Preserve
   unchanged native handling and unrelated features when custom rendering is
   disabled. These are existing patch points; no new patch-table symbol is
   identified. Record existing dependencies and `required_user_action: none`
   in the patch ledger when actual integration changes are made.

Existing buildings, resources and infrastructure need coherent eligibility and
exclusion when the terrain policy is ported. Do not tint uniforms or owner colors.
Roof coating can follow as an existing-object material pass; it is not needed to
prove terrain feasibility. Natural wonders, constructed wonders and Districts
retain their deferred M9/M10/M11 contracts. This study begins none of that scope.

## Verification and practical limits

The executable probes exercise visible flowers rather than an empty/faded
pattern. They verify wrapped values and derivatives using actual map dimensions,
evergreen and wood protection, desert/stone flower exclusion, autumn desert
protection, unchanged autumn/spring ground normals, finite unit snow normals,
unchanged scene coverage and exact linear-target Summer round-trip/disabled
behavior within this harness. Source pack/art hashes are compared before and
after rendering. The BIQ exporter preserves actual WMAP wrapping and river bytes.

`compare.py` measures five controlled strips at noon and creates the gallery and
contact sheets. Its CIE76 distances and high-pass correlations are diagnostics,
not visual approval, guarantees for other packs, or a night-readability proof.
Texture correlation is expected to drop where spring flowers or winter snow add
new detail. Rock readability also needs the normal/height/relief checks and
full-resolution visual comparison. The selected concept images can change asset
shape and fine detail, so a literal pixel match is not a meaningful acceptance
test. Their luminous snow, inviting warm foliage and harmonious flowers remain
the aesthetic targets; further production-graph beauty matching is still needed.

Final noon diagnostics (CIE76 color distance; correlation ranges from -1 to 1):

| Measurement | Civ V-style pack | Civ VI pack |
| --- | ---: | ---: |
| Autumn grassland versus plains | 8.49 | 11.98 |
| Autumn smallest pair among five biomes | 8.49 | 11.98 |
| Winter smallest pair among five biomes | 6.73 | 6.73 |
| Spring smallest pair among five biomes | 7.29 | 9.65 |
| Autumn grassland fine-pattern correlation to Summer | 0.973 | 0.938 |

Winter's closest colors still rely on retained grain, value differences and
landform/river cues as well as hue. Verify that combination in the production
graph at reduced zoom and night; these noon numbers alone cannot establish it.

The harness expands the detailed source meshes, uses a 2× supersampled viewport,
and shades/readbacks full frames rather than timing retained gameplay. At
1600×900 it processes about 10 million vertices and reports roughly 1.4 GiB of
GPU allocations; this is not the incremental cost of a seasonal game feature.
Receipts contain per-pass GPU times. Autumn has little extra sampling; winter
adds snow/detail channels; spring has the largest material cost. Empty meadow
patches, ineligible surfaces and distant flowers skip atlas work, and only
admitted nearby candidates sample the sprite. The production goal is to pay this
material work when the material view changes and reuse it during ordinary
lighting updates. Do not infer FPS or zero cost from these measurements.

Before a real-game promotion, verify D3D/Metal material results and the actual
retained graph at normal/reduced zoom; scrolling and both wrap directions;
all five biomes in all seasons at noon/dusk/night; rock/detail preservation;
masked canopy coverage/caster parity; coast/river/road coherence; reflected
seasonal terrain and forests; cache-generation transitions and return to Summer;
native PCX independence; and config-off delegation. Use current category tests
and bounded scripted game evidence when the VM is available. Run the approved
injected compile smoke test only if injected files change. None of those Windows
or live-game passes is claimed by this Mac study.

## Repeat with bounded storage

From the repository root:

```sh
python3 Renderer/lab/studies/seasons/render.py --all-packs
python3 Renderer/lab/studies/seasons/render.py --all-packs --case biomes --width 1920
python3 Renderer/lab/studies/seasons/compare.py
```

Omit `--all-packs` to resolve the configured default terrain pack, or pass
`--pack Renderer/packs/NAME`. Scenario/custom definition files can be passed with
`--scenario-definitions` / `--custom-definitions`. `--hour` and camera arguments
allow focused diagnostics; they overwrite the same case outputs, avoiding an
unbounded archive. The source BIQ export uses the existing sibling editor parser.
Metal execution may require the desktop tool's normal local-process permission.
Shader tools are the existing Lab tools, not a new Windows build dependency.

Optional asset probes are `asset_inputs.census()` and
`asset_inputs.snow_decal_evidence()`; store their small JSON reports in the same
output directory. `C3X_CIV6_ASSETS` can point to another local installed Assets
tree. These optional source probes still require the installation; rendering
uses the preserved `assets/flowers/blossoms.source` first. Flower adaptation
occurs in memory; only the tiny adapted atlas is written into the temporary work
directory. A run refuses to start below 8 GiB free.
Builds, temporary DDS, raw BGRA frames and exported CSVs are removed on completion
or failure. Packs and the protected seasonal cache are read in place; ordinary
renders create no persistent extracted art or geometry.

Keep code/recipes, the selected concepts, **`Renderer/packs/` and this study's
ignored `assets/` directory**, even when removing Civ VI to reclaim disk space.
The cache preserves selected seasonal inputs, not the full source-art library;
new source research may require reinstalling the game. Generated outputs
under `Renderer/lab/out/seasons/programmatic/` are reproducible and disposable;
delete that exact directory when these comparisons are no longer useful.
Do not clean other agents' outputs or source packs to reclaim space.

The [source-independence receipt](source-independence.json) records a successful
Mac GPU `test.biq` run with `C3X_CIV6_ASSETS` set to an absent tree. All ten image
hashes across both packs matched the preceding renders, and the recorded pack
inputs were unchanged. This proves this Lab path's independence from the source
installation, not that every future source-art investigation is already cached.
