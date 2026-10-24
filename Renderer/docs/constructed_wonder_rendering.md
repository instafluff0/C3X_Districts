# Constructed Wonder Rendering

C3X's constructed wonders (Wonder Districts) drawn by Renderer64 with Civ VI art.
The approved look is the Lab study `Renderer/lab/studies/wonders/` (page:
`Renderer/lab/out/wonders/page/`). The user approved it for the game on
2026-10-24. The Autumn session leads integration and reviews each stage.

## Authoritative C3X State

Wonders are a C3X feature, so their state is C3X's own (`injected_code.c`, `C3X.h`):

- `district_tile_map` instances with `district_id == WONDER_DISTRICT_ID`;
- `wonder_info.state`: unused, under construction or completed;
- the completed wonder in `wonder_info.wonder_index`;
- the wonder under construction from `wonder_district_tile_under_construction`, the reserving city's build;
- the alternate facing from `wonder_should_use_alternative_direction_image`;
- the unused district's era, which C3X takes from the territory owner or the builder.

Under Renderer64, `draw_district_on_tile` never draws (Renderer64 owns m19), so a
wonder tile shows only its terrain today. Nothing needs suppressing. No
`civ_prog_objects.csv` entry is needed.

## Rules

- **No names in runtime code.** Scenarios define arbitrary wonders. C3X config
  names reach kits only through the pack's aliases. Seating and the fallback for
  a wonder without a kit are pack metadata, applied by generic rules (the user,
  2026-10-24). Placement rules C3X already holds come from C3X itself.
- **Facts only in C3X.** C3X reports the state; Renderer64 draws. C3X keeps no
  renderer logic.
- **Recompile on change only.** State, stage and facing are selected by facts, so
  a chunk recompiles when one of them changes, never per frame. Measure a
  busy-save scroll before and after.

## Why City Bodies

Civ VI lights buildings at night with emissive maps laid out by the third UV set.
Its AO and baked light maps (the Oracle, the Hanging Gardens) use the second.
Resource bodies carry one UV set; city bodies carry all three, along with normal,
gloss, metalness and opacity maps, point lights and attached effects. So wonders
are city bodies.

## Residency (the user, 2026-10-24)

Civ VI's build-movie stages are heavy (50–90k vertices per stage against 10–15k
for a finished wonder; about 370 MB for every stage of all 23 wonders). The user
chose to keep the movie stages and load them on demand, without slowing the game
when a wonder comes on screen:

- a wonder's geometry is loaded when its district appears in the captured world
  state (game load, a city starting it, completion), never because it scrolls
  into view;
- loading and preparation run in the world-preparation jobs, off the frame path;
- the pack is split per wonder and stage, so only the stage in use is resident;
  crossing into the next stage loads it and frees the previous one;
- once loaded, a wonder is an ordinary cached tile and recompiles only when its
  state, stage or facing changes.

## Runtime Pack `Renderer/packs/WonderRuntime`

`Renderer/tools/asset_compiler/build_wonder_pack.py` builds it from the approved
study. It holds no Civ VI identifiers.

- **`index.bin`** lists every composition group (authority, file, byte size) and
  the shared texture closure; **`groups/<n>.bin`** holds one group each in the
  city library format, wonders only. The renderer appends a loaded group to the
  city library; `complete_city_set` checks production alone. A composition's
  authority names what it is:
  `wonder|<alias>|<variant>|<season mask>`.
  - Variants are `complete` and `build:<progress>` (0.25, 0.5, 0.75). An `alt`
    suffix marks the turned composition for wonders with C3X's alternate
    direction.
  - The unused Wonder District is `district|<era>|<variant>|<season mask>`. Its
    plot is the composition's era paving.
- **Seasons:** foliage parts only are emitted per season.
  - Summer foliage is shown in Summer and Winter; recoloured copies in Autumn and
    Spring. Every other part has mask 0 (all seasons).
  - A season flip is a selection change through the season-stand path, never a
    recompile.
  - Conifers keep their green (the user's Autumn rule). The compiler marks slender
    foliage parts (more than twice as tall as wide) as conifers, from geometry.
  - Winter roof snow reaches city bodies through the city shaders.
- **Effects** are city effect records:
  - fires use the flame kind (0);
  - Civ VI lamps use the night-light kind (2), with a mode for colour (warm,
    white, red, green) and blinking. Kind 2's default stays bit-identical.
  - The 90–99 effect material band is unchanged.
- **Placement metadata, per kit:**
  - **Seating:** bodies stand on the ground at their centre, as in the approved
    Lab. A kit whose metadata asks for a foundation gets the city terrace (its
    skirt and top plates); parity showed the terrace laying plates even on gentle
    grassland relief, so it is never automatic.
  - **River:** not pack metadata. A wonder whose C3X config needs a river
    (`buildable_on_rivers`) is reported as `C3X_RENDERER_WONDER_ON_RIVER`; the
    renderer places it across the tile's primary river (C3X's direction rule) and,
    on a mountain, cuts the one generic river valley (`stamps.bin`, alias
    `wonder|river valley`) whose walls keep the host mountain's look and snow.
    `lab/studies/wonders/river_gallery.py` renders twelve sites (every river edge,
    bends, a fork, a source, ranges, hills and forest, facing, stages, snow).
  - **Fallback:** the era Wonder District for a wonder name with no kit.
- **Memory:** wonder textures are capped (1024, 512 for small props). The pack
  holds at most 64 MB of resident textures (BC-compressed, full mip chains),
  recorded in `pack_report.json` and enforced by a test. The city texture
  budget rises only by the pack's measured size.

## Renderer Design (stage 2)

World-preparation jobs read the city library while frames draw, so wonder groups
never grow it in place:

- **Loading:** a group a tile's facts call for is decoded on a preparation job
  into its own `city_fidelity::Library`, and its textures are uploaded there
  (Direct3D 11 device creation is free-threaded). The group stays resident while
  any explored tile uses it, and is freed after.
- **Compiling:** a wonder tile compiles against its group's library. Its chunk
  material indices are shifted into the shared material range, which the render
  thread extends in storage reserved at load, so no reader ever sees it move.
- **Selection:** by facts (alias, state, stage, facing, season mask); a tile whose
  group is not resident yet draws its terrain only until the job finishes.
- **While loading:** a tile whose stage is not resident yet shows its era's
  Wonder District (also the no-kit fallback), never an empty tile or native art.
  The next build stage is prefetched as progress nears a boundary. Eviction
  follows world state, not the camera.
- **Lamp modes:** kind 2 decodes mode = floor(seed / 64) and takes its flicker
  phase from fmod(seed, 64). A test pins that every production city effect seed
  stays below 64, with the bit-identity test of mode 0.
- **Season switches:** foliage variants make every season switch reselect
  (not only Winter), so variant chunks stay foliage-only, and the before/after
  measurement includes an Autumn→Spring switch.
- **Facts only:** `wonder_fact` reads the ABI 24 tile fields and nothing else;
  the alias is the C3X config name, lowercased. The Lab preview fills those
  fields from `C3X_LAB_TILE_WONDERS` (a Lab tile's resource name selects its
  line), so `lab/studies/wonders/prod_parity.py` renders the production path.
- **River valley reach:** a follow-river stamp moves to its river edge, so its
  reach is capped at 1.5 − max(|ox|, |oy|) (1 tile) to stay inside the owner's
  3x3, and its field slopes to zero within that reach. Overflowing constructed
  stamps are dropped (never natural ones) and traced as `wonder-stamps-dropped`.

## Facts (frame ABI 24, after natural wonders' 23)

The tile record gains:

- the wonder's C3X config name;
- the state: none, unused, under construction or completed;
- build progress 0–255 (the reserving city's shields over the wonder's cost);
- the alternate-direction bit and the river bit (`buildable_on_rivers`);
- the unused district's era.

All come from existing C3X state. Config-off delegation tests apply.

## Stages

Each stage went to the integration lead as a diff. Stages 1–3 have landed
(2026-10-24): stage 2 was developed as anchored patches in a private tree and
applied once approved; `lab/studies/wonders/test_port.py` now pins it in the
checkout.

1. Pack compiler and its tests (memory cap, authority format, conifer exemption,
   no Civ VI identifiers).
2. Renderer: the appended library, fact-based selection in the prepared and
   direct object paths, season masks, kind 2 lamp modes, foundations and river
   spans.
3. Injected facts and config-off delegation tests, with
   `TEST_INJECTED_CODE_COMPILE.bat`.
4. Build, stage, and a scripted game test on a disposable save with wonder
   districts, plus the busy-save scroll measurement.
