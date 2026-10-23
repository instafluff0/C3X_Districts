# Seasons in Renderer64: integration plan

Status: **first integration staged for in-game testing** (2026-10-22; decisions in
section 5, what landed in section 7).
Lead: the Autumn session.
Inputs: the three Lab studies (`lab/studies/autumn/`, `winter/`, `spring/`),
replies from the Winter and Spring sessions, and traces of the current C3X
season clock and renderer pipeline. File references are relative to `Renderer/`
unless they name `injected_code.c` / `C3X.h`.

Approved looks to reproduce: Autumn **v18**, Winter **v24-laden**, Spring
**v20**. The user first confirmed v28, then chose v20's flowers instead
(2026-10-22, section 5). The rejected posies (v23–v25) are not part of the port. Each study's private shader root is
the visual reference; the port must match it in the Lab before anything is
staged, and the user accepts each season's in-game look separately.

## 1. How seasons work today

**C3X clock.** The season is an integer 0–3 (Summer, Fall, Winter, Spring) in
`is->current_seasonal_cycle`. It only changes in steps: at end of turn
(`patch_perform_interturn_in_main_loop`, `injected_code.c` ~35373), or at native
map-image load for the timer/user-season/specified modes. There is no blend
state. The default config has `seasonal_cycle_mode = off` (Summer forever).
`prepare_custom_renderer_frame` copies the value into `frame.season` on every
frame (`injected_code.c` ~29830); units get the same value per draw. The
mapping is identical to the renderer's (`scene_export.cpp` `season_name`).

**One defect to fix first.** The end-of-turn block that advances both the hour
and the season is inside `if (is->day_night_cycle_img_state == IS_OK)`. If the
native `Art/DayNight/<Season>/<Hour>` PCX art is missing or fails to load, the
clock freezes, even with custom rendering on. The seasons playbook
(`lab/studies/seasons/PLAYBOOK.md`, step 9) already calls for separating them.

**Renderer.** Today the season only nudges ambient light
(`environment_runtime.cpp:103-109`). In the Renderer64 path an hour or season
change is one "lighting context" event (`sandbox/fresh_pipeline.h:3658-3700`):

- static rasters are marked stale but stay displayable, then are re-rasterised
  in time slices (`C3X_RENDERER_STATIC_BUDGET_MS`, default 8 ms per frame) and
  promoted when complete;
- reflection, reflected-terrain material and the water-lighting cache are
  dropped; the unit plan is reselected;
- **shadow pages survive** a season change (their basis depends on the hour);
- **resident geometry, forest instances and object compilation do not include
  the season** (`compile_context` zeroes it under the pickup profile), so a flip
  is a re-raster plus CPU reassembly, never a recompile.

So seasonal *appearance* fits the existing path at no steady-state cost. Only
two winter features are geometry today (snowy pines, flurry quads); section 3
moves both out of the compile path.

**Shaders.** Editable sources live in `lab/shared/shaders/` (relief
`beauty_terrain.hlsl`, `beauty_mountain.hlsl`, objects `beauty_objects.hlsl`,
hydrology, lighting, `objects/city_scene_material.hlsl`) and
`native/render_core/*.hlsl`. `renderer.py prepare` (`lab/preparation.py`)
generates `native/city_fidelity/*.hlsl`. The game reads shaders at runtime from
`packs/Renderer64ResidentRuntime`, which receives generated regions through the
`tools/prepare_renderer64_materials.py` / `tools/overlay_*_shading.py` chain.
`fresh_pipeline.h:185-300` and `sandbox/terrain_material_fast.h` also patch the
shader text at runtime, so their anchors must survive every edit.

**Lab vs. game path.** The three studies rendered with the 32-bit candidate DLL
and `native_preview.exe`. The game runs Renderer64 (x64, `C3X_RENDERER64_FRESH`,
the fresh pipeline with runtime text patches). Each port must also be checked
on the x64 path (sandbox `client_x64` capture or the scripted game test).

## 2. Architecture

**D1. One season state, two constant buffers.** Add `float4 Season =
{index, progress, reserved, reserved}`:

- integrated shaders (hydrology, feature, city, rigid_feature, resource_body):
  `TerrainSettings` b0, using the reserved last two floats of
  `TerrainShaderSettings::resource_look[4]` (`c3x_renderer.cpp:209-228`; HLSL
  `game_resource_look.zw`), so the layout does not change;
- relief, mountain and objects: a 7th `float4` in the natural `Frame`
  (`lab/shared/natural/data.h` `Frame`, `NaturalData::frame_settings()`,
  `source_fidelity/runtime.h` ByteWidth 96 → 112).

Both are filled where they are today, on the existing hour/season change
(`fresh_pipeline.h` `update_environment`, `c3x_renderer.cpp` ~9946). `progress`
is 1.0 in the first release (see D8). Units keep their per-draw lighting; they
get no seasonal material change.

**D2. One shared seasonal library, uniform branches, no variants.** New
`lab/shared/shaders/seasons/`:

- `season_common.hlsl`: OKLab/LCh helpers (one copy instead of the three
  `autumn_`/`winter_`/`spring_` copies), integer-hash periodic noise (D7), the
  calibrated snow white and sky fill per shader family, and the shared
  per-tree seed and crown normal. The relief and integrated shaders light snow
  differently in both gain and tint: relief gets a cool sky lift, integrated
  shaders do not. So each family carries its own gain and tint, chosen to
  reproduce Winter's matched values (roofs ×1.5 with 35% of the baked AO,
  river banks and shelf ×1.4).
- `season_autumn.hlsl`, `season_winter.hlsl`, `season_spring.hlsl`: each season's
  functions, owned by that season's session.
- `season_hooks.hlsl`: the hook functions below, each a `switch` on
  `Season.x`. Summer (0) returns its input untouched.

The generator inlines these into every target shader. Branches on a constant
are coherent per draw, so the per-pixel cost of the inactive seasons is
negligible, and one compile serves all four seasons (separate variants would
multiply the ~41 s `PSIntegrated` cold compile by four). Ship the compiled
`.cso` files in the pack as today.

**Summer parity is a hard gate:** with season 0, every shader must produce
byte-identical frames to the build before the seasons plumbing (an automated
test in the Lab and on the x64 path). Nothing in this plan changes Summer.

**D3. Hook points.** Every study patched the same places, so each becomes one
named hook:

| Surface | Shader / entry | Autumn | Winter | Spring |
| --- | --- | --- | --- | --- |
| Turf, plains, hills, hill stone | relief `shade()`, after biome composition | hue grade, turf pattern, khaki hills | snow cover, land types, drifts | green-up, painterly turf, meadow flowers |
| Forest floor | relief floor branch | leaf piles (broadleaf floors only) | snow | – |
| Ambient / sky fill | relief, mountain, objects | – | cool sky lift | – |
| Mountain-tile ground and rock | `mountain.hlsl` | ground grade | snow cap on every mountain (adds snow beyond Civ III's snow bit; hides no tile state), lower snow line, cool rock | ground grade |
| Trees | objects VS (seed, crown normal) and PS | crown palette, crown light, leaf glow | laden crowns, needle snow, hue lock | fresh greens, blossoms, crown light |
| Coast, beach, underlay | `q6_raw_main` | ground grade | snow texture | ground grade |
| Farms, rocks, resources | `q6_raw_feature` | ochre farm tint | snowy fields, snow on up-facing | – |
| Buildings, plates | `PSNativeCityBody` | – | roof snow, trodden plates | – |
| Roads, rails | hydrology route branches | darker dirt | trodden snow | – |
| Rivers | hydrology river branch | via water | shelf ice, floes, snowy banks | via water |
| Open water | game: `PSWaterSurface` (`sandbox/water_surface.hlsl`); Lab: `q3_water_material` | deeper blue (relative, D10) | winter blue (relative), ice | deeper blue (relative) |
| Shore and lake beds | underlay / `q3_water_material` beds (confirm the game layer) | – | bed ice | – |
| Coastal rocks | `q4_coastal_rock` | – | cool stone, snow caps | – |
| Mountain flurry | city effect layer, kind 6 | – | animated flakes | – |

Reflection entries (`PSReflection`) call the same hooks, so reflections follow
the season. The flurry is excluded from the mirror, and so is Spring's meadow:
flowers are invisible in the mirror at game zoom.

Hooks are not only pre-lighting albedo edits. Winter's approved look also
needs the following, so the hook set includes them from Phase 0:

- *post-lighting* hooks: the tree pass's final radiance (hue lock), and
  `PSNativeCityBody` after `q8_city_look`, where roof snow is relit outside the
  city look; that is the fix for the "dull grey roofs";
- a pre-lighting city-body hook that can change base colour, normal and
  metalness, and hand values to the post-look hook (roof weight, snow colour,
  normal, AO factor);
- ambient multipliers in relief, mountain and objects, which receive the snow
  weight;
- Spring's bloom pixels in the relief hook: in/out lighting normal (an upright
  head, the slope normal bent toward +z) and specular map; the frame light
  direction (`ShadowL.xy`, for petal top light and contact shadow); and the
  screen derivatives of world xy (the pixel footprint: below 1.4 px a bloom is
  exactly one solid pixel or a five-pixel plus); and the biome weights at the
  pixel, since the plains share picks drift palettes. Upright blooms also use
  screen-proportional
  units derived from the native projection (x = 0.7071(wx+wy),
  y = 0.35355(wx−wy) − 0.57983·z, from `relief_scale` tile/224·.82) and snap
  their centres to pixel centres. The hook takes these from the projection
  constants, not hard-coded numbers. This assumes the relief raster's pixel
  grid is the final screen grid at that tile width, which holds for per-camera
  composition;
- river and water hooks with their full context:
  - river: surface, specular, world, pixel distance, shoreline, water width,
    the bank noises and the input for illumination;
  - water: `real_terrain`, `hydrology_data.w`, `surface_kind`, `macro_uv`.

**D4. A season flip never recompiles geometry.** Two winter features are
geometry in the Lab. Both move out of the compile path:

- **Snowy pines: precomputed winter stands.** The approved v24 look uses the
  `snow_pine_forest` *variety*, which has its own recipe set: clump-heavy, 8
  recipes, no shrubs. The pine set has 10 recipes, more and smaller singles,
  and two shrubs. A plain draw-time body swap would keep the pine stands, so
  it would not match v24. Recommended: when a pine-forest tile compiles, also
  compute its winter placements with the snow set (same canopy clearing). In
  Winter, draw that instance list instead of the summer one. This gives the
  exact approved stands, no recompile on a flip, and a small memory cost (64 B
  per extra placement, pine tiles only). The snowy bodies already share the
  pine bodies' vertex data at runtime (`load_forest_varieties`); only the
  material differs, adding an opacity channel. Details:
  - *Tundra:* pine tiles on tundra already use the snow set in every season,
    so they need no second list.
  - *Hill forests:* a raised hill canopy takes the variety its neighbouring
    forests vote for (`variety_of` in `canopy_clearing()`). Its winter stand
    must re-run that vote with Winter's mapping (every pine maps to the snow
    set).
  - *Shadows:* the resident world keeps its own caster lists. The winter
    stands need their own caster entries; refreshing the shadow pages alone
    would still cast summer's stands in Winter. Refresh the pages when
    entering or leaving Winter.
  - *Downstream readers of the placements:* floor-decal selection, plant vs.
    animal gaps around resources, and the conifer floor marker. Check each
    one. Either the winter list drives it in Winter, or we accept summer's,
    which is likely invisible under snow; record which in the port.
  Fallbacks, which need the user's acceptance because they change the
  approved stands: (a) a draw-time material swap on the pine placements (no
  extra memory; shrubs keep shader needle snow); (b) rebuild the forest
  instances on entering and leaving Winter (twice per cycle). The Winter
  session will render the options against v24 in the Lab before this is
  locked.
- **Flurry: a season-gated effect list**, built from the visible mountain tiles
  in the fresh pipeline (as water animation is counted), not from object
  compilation. It counts toward `visible_city_effects` and continuous redraw
  only in Winter and only above its zoom threshold (section 4).

**D5. Static masks are baked always, independent of season.**

- *Conifer floor marker:* forest-floor quads of pine stands carry
  `material_plains = 3.25` (`lab/shared/natural/vegetation_floor_mesh_body.h`,
  one line; it is in the autumn study's private build and follows tile
  content, not the season). Autumn adds no leaves there.
- *Ground-use mask:* one R8 texel per tile with flags (forest/jungle, city,
  farm, road, river), sampled by world position. Tundra, desert and exposed
  hill rock are already excluded by biome weights and the relief rocky band;
  marsh and floodplain need no flag. Spring needs it for meadow
  placement ("grassland, plains, hills only", never under roads, farms, cities
  or woods), and Autumn and Winter can use it too. It is updated through the
  existing topology/content-revision path; a tile change already invalidates
  that region's raster. Size: map width × height bytes (17 KB at 130 × 130).

**D6. Seasonal roles from pack metadata, never names.** A new optional sidecar
`seasonal-roles.bin` beside `natural.bin` / `forest-varieties.bin`, written by
`native/source_fidelity/prepare.py`, holds for each body its role (broadleaf,
conifer, shrub, jungle, snowy) and its winter substitute. Older DLLs never open
it; without it, seasons leave trees unchanged. The role reaches the shader in
the free `InstanceMaterial.instance_secondary.zw` (`objects.hlsl`). This
replaces the studies' leaf-atlas average-colour tests.

**D7. Wrap-safe world noise.** All world-anchored noise (autumn patches and
litter, winter scour, drifts and ripples, spring drifts, palettes and bloom
cells) must be periodic in the map's wrap width (`C3XWorldSettings
c3x_world_dimensions`). The study looks use raw world coordinates and would
seam at the date line. The shared library provides lattice noise whose period
is an integer number of cells per map width. The same rule applies to
world-anchored texture lookups: snow-texture repeats on `q6_world`,
`macro_uv` and `q3_source_world` (rivers) and on roofs each need their own
integer repeat per map width.

**D8. Transitions: instant at the turn boundary.** C3X flips seasons at end
of turn, like the hour. The flip goes through the existing lighting-context
path: the stale frame stays on screen while the new season refines over a few
frames, as an hour change does today. An animated crossfade is rejected,
because each frame of it would invalidate every static raster (the
`C3X_SANDBOX_DAY_NIGHT` worst case). `progress` stays in the constant buffer
for an optional later stage:

- C3X could report season progress per turn (`turns_in_current_season /
  fixed_turns_per_season`). This needs a new frame field and an injected change;
- Winter would accumulate snow depth, Spring would bring blooms in by
  per-element thresholds (never alpha), and Autumn would deepen colour.

That still costs one re-raster per turn, as the hour does.

**D9. Clock independent of native art.** In `injected_code.c`, advance the
hour and season at end of turn when custom rendering is on, even if
`day_night_cycle_img_state != IS_OK`; keep native behaviour unchanged when it
is off. This uses existing hooks only (`perform_interturn_in_main_loop`,
`Map_Renderer_load_images`) and needs no new `civ_prog_objects.csv` entry.
Record it in `docs/civ3_patch_dependency_ledger.md` with
`required_user_action: none`. With the clock decoupled, C3X also skips loading
the native seasonal PCX sprites while custom rendering is on (user decision,
section 5). Keep the native load and swap unchanged when custom rendering is
off. Check that nothing else under custom rendering reads those sprites; units
and UI keep their base art.

**D10. Seasonal water must preserve the game's Coast/Sea/Ocean tint.** The
game already distinguishes the three water types. Renderer64 draws open water
with `PSWaterSurface` (`sandbox/water_surface.hlsl`), which tints by the
Civ III water family carried in `surface_coordinate` (coast .34, sea .68,
ocean 1.0, blended between tiles). Coast gets a teal lift, sea a smaller one,
and ocean keeps the deep base. On the Sep 28 x64 capture of the concept view,
`water_tiers.py` measures Coast–Sea ΔE 4.9, Sea–Ocean 2.1 (hue 229° vs 244°) and
Coast–Ocean 7.0.

The Lab preview, which every season study measured, draws open water with
`q3_water_material` instead. It has no family tint, which made Sea and Ocean
look identical there. That was a Lab artifact, so **Summer water does not
change** (user, 2026-10-22). Rules for the ports:

- Seasonal open-water recolours go into `PSWaterSurface`, the game's shader,
  not only `q3_water_material`. Rivers do reach the game through
  `PSRiverSurface` → `q6_raw_main`, so the river patches stand.
- A seasonal recolour must be relative: one hue rotation and chroma/lightness
  scale applied to every water type after the family tint, never a pull
  toward a single target hue. Autumn's v18 grade pulls every blue toward
  240° and would merge sea (229°) and ocean (244°), so it gets reworked.
  Winter's v11 rotation already keeps the differences.
- Check on the x64 path that each season's Coast–Sea and Sea–Ocean
  separations are no smaller than Summer's.

## 3. Performance

**Where the cost lands.**

| Pass | When it runs | Seasonal work |
| --- | --- | --- |
| Static rasters (relief, mountains, trees, coast, farms, cities, routes) | once after a flip; again for every newly exposed strip while scrolling, zooming or jumping | almost all seasonal shading |
| Water dispatch, waves, **rivers** | every frame while visible (most views); rivers draw live with the animated water (`fresh_pipeline.h` ~5480) | water recolour (all seasons); Winter river ice (shelf reach, scallops, floes, windows: ~6 noises, 1 fbm, 1 OKLab round trip, 2 snow-texture reads per river pixel); bed ice (check which layer draws `surface_kind` 4 beds) |
| City effect layer | every frame while effects are visible | flurry (Winter) |
| Units | every frame | none |

Retained rasters make steady-state idle nearly free, but the static shading
cost still shows up in scroll and zoom fill, the current bottleneck (GPU
5–6.5 ms per scrolling frame at 2×–3×). Heavier terrain shaders mean slower
reveal of new strips.

**Budgets** (VM, busy 1498 AD save and light save, `near` and `zoom-out`
scenarios, at least three alternated runs per side):

- Summer: byte-identical frames to the pre-plumbing build; timing within
  run-to-run noise.
- Each season, steady idle: within noise of Summer; with the flurry visible, at
  most +1 ms GPU per frame.
- Each season, static raster GPU time per pixel at most 1.15× Summer (measured
  per pass with `C3X_RENDERER_PROFILE=4` drained marks); scroll step time and
  zoom fps within noise.
- Season flip: the frame never freezes; full refinement takes at most
  1.5× an hour change on the same view.
- Memory: no extra geometry; at most 8 MB extra GPU for LUTs, masks and any
  precomputed fields.
- Cold shader compile: at most +15%; warm launches unchanged because `.cso` files ship.

**Cheapening, in order (apply until the budgets are met):**

1. Integer hashes instead of `sin`-based value noise. The value noise is most
   of the ALU in all three studies; Winter's relief block runs about 30 noise
   evaluations per pixel.
2. Remove duplicate work, such as Winter's dune ripple evaluated twice.
3. Hoist constants out of the per-pixel path: texture means read through
   `SampleLevel` (mip 11/15) and per-material targets go into metadata or
   constants.
4. Replace per-pixel OKLab round trips on hot paths with precomputed grades: a
   small 3D LUT per surface family and season (16³ RGBA8, 16 KB each). Do the
   per-frame water and river paths first, since they are paid every frame.
   Winter's river ice noise is the first candidate for steps 1–3.
5. Early-outs before per-cell work. Spring samples its drift field once per
   pixel and skips the whole bloom-cell loop where it is below threshold minus
   a margin: that covers the ~50–70% of grass that is bare. Without it, each
   grass pixel evaluates 2 layers × 4 cells, each with drift fbm and clump
   noise before its density test.
6. Only if measurement still demands it: a map-sized, wrap-periodic "season
   fields" texture (4–8 texels per tile, generated once per map) holding broad
   patch, drift, clump and palette fields, shared by Winter's scour and drifts,
   Spring's drifts and palettes, and Autumn's patches.

**Flurry budget** (Winter):

- Use an integer hash or a small flake texture.
- Three parallax layers at tile width ≥ 128 if that fits the +1 ms budget,
  otherwise two. One layer below 96, off below 64. Dropping the third (small,
  far, faint) layer thins the snowfall by about a third, so the user sees the
  two-layer clip before it is locked.
- Cap its screen area, exclude it from reflections, and optionally advance its
  time at 20 Hz.
- It keeps its occlusion: each quad stands on the tile's front corner at the
  natural height there and depth-tests against the retained depth, so the
  effect list carries that height.

Most views already redraw continuously because water is visible, so its cost
is fill, not an extra redraw trigger. Inland mountain views are the exception
and must be measured.

**Test harness.** Add a renderer-only `C3X_RENDERER_SEASON_OVERRIDE`
environment switch (like `C3X_RENDERER_PREVIEW_SEASON`) for pinned-season
captures, plus a scripted-test config with `seasonal_cycle_mode = every-turn`
and `fixed_turns_per_season = 1`, so each end turn flips the season. Use the
isolated A/B trio method for timing.

## 4. Phases, owners and order

| Phase | Work | Owner |
| --- | --- | --- |
| 0. Foundations | D1 constants and Summer-parity tests; Lab water parity (the preview draws open water with the game's `PSWaterSurface`, or season water is judged on the x64 path only); D2 library skeleton and hooks wired into the generator and pack overlay; D5 conifer marker and ground-use mask; D6 roles sidecar; D7 periodic noise; D9 clock decoupling (+ ledger); season override and harness | Autumn session (lead) |
| 1a. Autumn port | `season_autumn.hlsl` through the hooks; match v18 | Autumn session |
| 1b. Spring port | `season_spring.hlsl`; meadow on the ground-use mask and biome weights; wrap-safe drifts; drift early-out; match v28 (or its signed-off successor) | Spring session |
| 1c. Winter port | `season_winter.hlsl`; precomputed winter pine stands (after the Lab comparison against v24); flurry effect list. The Winter session drafts the pipeline changes for the lead to review. Must match v24-laden (stills: `lab/out/winter/final24`, regenerated with `flurry_tree.py prepare` / `build`, then `study.py render … --look v24-laden --tree flurry --winter-pines`) | Winter session |
| 2. Performance | budgets per season; cheapening steps 1–6 as needed; flurry A/B on a mountain-heavy save | lead, with each season's owner |
| 3. Verification and staging | Lab `seasons` category (four seasons × key views); x64 path check; in-game flips, save/load, config-off, scroll/zoom/wrap, reflection, dusk/night; user acceptance per season; then staging | lead; user accepts |

Phase 1 ports can run in parallel because each owner edits only their own season
file. The lead owns the hook file, the constant buffers, the generator and pack
overlay, and every C++ change in `c3x_renderer.cpp` / `fresh_pipeline.h`.
The Winter session drafts its pipeline changes for review instead of editing
those shared files directly. As always: one VM workload at a time
(`wait_for_vm` in the study harnesses), private trees for experimental builds,
no commits or staging without the user's word.

## 5. Decisions (user, 2026-10-22)

The user approved every recommended default:

1. **Transitions:** instant at the turn boundary (D8). The per-turn "season
   progress" stage is not planned.
2. **Water (D10):** the user first approved shared water tiers in every
   season, then pointed out that the game already distinguishes
   Coast/Sea/Ocean. The Lab path misled the measurement. Summer water does not
   change, and each season's water recolour must preserve the game's
   family tint.
3. **Flurry:** off at far zoom. Three layers at close zoom; two only if three
   exceed the budget, and only after the user has seen the two-layer clip.
4. **Winter pines:** precomputed winter stands that reproduce the approved v24
   look (D4); the cheaper options are dropped.
5. **Native seasonal art:** once the clock is decoupled (D9), C3X skips loading
   its native seasonal PCX sprites while custom rendering is on. They still
   load when custom rendering is off.

Spring reviewed and agreed (2026-10-22). Its look was v28 until the user chose
v20 for the look of its flowers (2026-10-22, before the first staging). v20
and v28 run the same code; they differ only in the study's constants.

## 6. Risks

- Shared shader files are edited by many sessions. The runtime text patches in
  `fresh_pipeline.h` and `terrain_material_fast.h` can silently miss an anchor,
  so every hook needs a test that fails when its anchor or call is absent.
- Register pressure from inactive-season code can slow every season. Measure
  Summer before and after Phase 0, not only the seasonal looks.
- The Lab's 32-bit path and Renderer64's x64 path diverge in places, so every
  season is accepted on the x64 path.
- Calibration: snow white, sky fill and water strength were tuned per study.
  The shared library must keep each approved look while unifying the constants.
- Spring is the season most likely to need the early-out and precomputed
  fields to meet the static-raster budget (8 cell evaluations per grass pixel
  in v28). Its port starts from its final committed look.
- The Lab preview and the game draw open water with different shaders (D10).
  Any water decision made from Lab renders alone can be wrong.

## 7. First integration (2026-10-22)

The user asked for all three looks in the game ("integrate seasons and stage so
I can test in the game after tests pass … make sure performance doesn't slip").
What landed, against the architecture above:

- **D1 constants:**
  - `EnvironmentState::season` / `season_progress`, with progress 1;
  - integrated shaders read `game_resource_look.zw`;
  - natural shaders read `Frame.season`, the 7th `float4` (`C3XSeason`; `Frame` is 112 B).
- **D2, simplified:** each study's own code is ported verbatim as marked
  insertions with uniform `[branch]` guards (`lab/shared/seasons/seasons.py`,
  libraries beside it). The tool is `tools/season_shading.py`; tests are in
  `tools/test_season_shading.py`. Summer runs the unchanged code (Lab parity:
  at most 1/255 in a few pixels, from compiler rounding). Every season matches
  its study look in the Lab to the same 1/255.
- **Water dispatch:** each season used to call `q3_water_material` itself,
  which made PSIntegrated inline it four times (406 s fxc compile). The dispatch
  now evaluates it once (`c3x_water`), each season recolours that, and the
  original return sits under a marked `#if 0`.
- **D4, Winter (by the Winter session):**
  - precomputed snowy-pine stands, which match v24 in the Lab;
  - the flurry as a per-view ambient layer like the waves: Winter only, tiles
    of 64 px or more, never in reflections.
  - The flurry's sin hash snapped flakes to a lattice at the game's large
    clock values; it now uses an integer hash with a bounded clock.
  - The user asked for far more randomness, so the Winter session redesigned
    it (Lab look v24-flurry4):
    - per-column speeds and offsets;
    - per-flake size, brightness and wide sway;
    - per-layer wind slant;
    - a ragged per-mountain plume with gusts.
    In game, Winter at 3× idle is 59.4–59.5 fps (Summer 58.6–58.9).
- **D5:** conifer floor marker 3.25 in `vegetation_floor_mesh_body.h`. No
  ground-use mask: v20 needs only biome weights, and roads, farms and cities
  draw over the meadow.
- **D6, partial:** jungle bodies (only the jungle recipes 25–34 place them) set
  `InstanceMaterial.instance_secondary.z`. Autumn and Spring leave jungle
  unchanged, like conifers (user). Other foliage classes still use the
  studies' atlas-average colour tests, which are texture data, not names.
- **D9:** the clock is decoupled and native PCX loading is skipped under
  custom rendering (`injected_code.c`; ledger entry).
- **D10, revised by the user:** the coast keeps Summer's translucent look in
  every season.
  - Shallow beds (q3 kind 4) never change; Winter's bed ice is removed.
  - Every season's water grade fades in from Coast to Sea
    (`smoothstep(.34, .63, surface_coordinate)`), in `water_surface.hlsl` and
    in the q3 dispatch. Rivers keep their seasonal grade.
- **Harness:** `C3X_RENDERER_SEASON_OVERRIDE=0..3` pins the rendered season for
  scripted captures. It must reach both environment uploads (fresh
  `update_environment` and the bridge's per-frame render); a test guards this.
- **Performance** (busy 1498 AD save, `near`, warm caches, no foreign VM work
  in the measured window):
  - every season on the final build is within the baseline's run-to-run spread
    on every segment, including Winter with stands and the three-layer flurry
    (3× idle 57–59 fps against 58–60);
  - cold compile is larger: game-blocking shader compiles after a shader change
    take about 3 bounded runs (PSIntegrated about 95 s against 50 s pristine
    with fxc). It is cached per shader version.
- **Not yet:**
  - D7 wrap-periodic noise: on wrapping maps the date line may show a faint
    seam in broad seasonal patches;
  - per-season compile variants (`#define C3X_SEASON_INDEX n`), which would
    restore the cold-compile budget.
