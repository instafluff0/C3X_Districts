# Winter on Renderer64 (Lab study)

Goal: make C3X's Winter season (frame season 2) beautiful before anything
reaches Renderer64. Target: the selected concept
`Renderer/lab/out/seasons/scene-concepts/winter-icy-blue.png` (an image edit of
the `test.biq` view at camera 20,75). The earlier Mac Metal study in
`../seasons/` (WINTER.md) is superseded; like `../autumn/`, this study renders
through the **current production renderer** on the Windows VM.

## How it works

- `study.py render LABEL --look LOOK --views ...` renders with the candidate DLL
  and `native_preview.exe`, frame season 2 (`--season 0` for Summer).
- A look is a **private shader root**: the flattened production shaders with
  this study's anchored patches (`winter.py`), selected with
  `C3X_RENDERER_SHADER_SOURCE_ROOT`. Production shaders, packs and staged
  binaries are never touched; a missing or ambiguous anchor fails loudly.
- `winter_common.hlsl` holds the shared snow functions (OKLab colour, snow
  cover over the authored material height, dormant ground).
- `study.py sheet A B --target [--crop l t r b]` builds side-by-side PNGs with
  ffmpeg + numpy (no Pillow on this Mac).
- Views: `concept` (the target's own framing; no tile overlays, so all forests
  draw broadleaf), `pines` (same view through the Lab tile overlays, which
  honour the BIQ pine bit as the game does), `village*` (ancient cities, roads,
  farms), `salamanca*` (1498 AD save window: cities, farms, rails, mines), and
  `*-sp` / `snow-pines` variants (see below).

## Where each surface is shaded

| Surface | Shader (city_fidelity/) | Snow texture available? |
| --- | --- | --- |
| Land, hills, hill stone, forest floor, ground decals | `terrain.hlsl` (relief) | No: procedural snow over material height + `SurfaceDetail` (t30) crust |
| Mountains | `mountain.hlsl` | Yes, t9-11 (authored mountain snow) |
| Trees (leaves and trunks are both kind 1) | `objects.hlsl` | No |
| Beaches, banks, terrain underlay | `q6_raw_main` in integrated shaders | Yes, t8 `mountain_snow_texture` |
| Farms, resources, rocks, small plants | `q6_raw_feature` (feature/city/rigid/resource_body) | Yes, t8 |
| City buildings, settlement plates | `PSNativeCityBody` (same files) | Yes, t8 |
| Roads, railways | `hydrology.hlsl` route branches | Yes, t8 |
| Water, rivers (surface_kind 9) | `q3_water_material` dispatch | n/a |

## Lab-only stand-ins that production must replace

- **Season selection**: the private shader root *is* the winter variant.
  Production needs the frame season (already in `frame.season`) in a constant
  buffer, with these functions behind it.
- **Relief snow**: the relief draw has no snow texture bound. Production can
  bind the pack's ground snow material (`snow_base_color/height/specular`
  exist in the terrain packs) for a richer crust than the procedural field.
- **Foliage classes** use each atlas's average colour (linear g/b: leafy 4.4,
  pine 2.2, shrub 1.8, jungle 1.7, snowy pine 1.3). Production should read a
  pack flag; never names.
- **Snowy pines**: production already has a `snow_pine_forest` variety (Civ VI
  snowy pine bodies, same topology as the pines) but selects it only for pine
  forests on tundra. The flurry tree's `C3X_LAB_WINTER_PINES` switch
  (`render --winter-pines`) selects it for every pine forest, previewing
  "Winter selects the snowy variety". The older `snow-pines` views marked the
  tiles tundra instead, which leaked tundra into the neighbouring ground.

## Current look (v24-laden) and what each layer does

Colour targets measured on the concept (OKLab): lit snow L .92 C .025 h 261,
deep water L .47 C .096 h 249, forests dark navy-green under white boughs.

| Layer | Where | Change |
| --- | --- | --- |
| Ground snow | relief | settles in the authored material-height lows, sheds on steep faces, thins in broad wind-scoured patches; crust grain from `SurfaceDetail`; soft drift normal replaces most micro relief |
| Land types (v17+) | relief (`RELIEF_SUBTLE`) | grassland fully white; plains near-white with a sparse scatter of straw-gold stubble flecks; desert a gentle cream snow shaped by wandering dune ripples (relief only, no bare patches); tundra a cooler blue-grey snow with a stronger wind crust and a little gloss. Mountains and their ground collars keep the uniform cover |
| Hills | relief | snow with grey stone on steep faces and hill-stone patches |
| Mountains | `mountain.hlsl` | every mountain uses the authored snow layer, lower and on steeper faces; exposed rock desaturated to cool grey |
| Shade | relief, mountains, trees | cool sky fill on snow (`WINTER_SKY`); never violet: a violet fill under the warm sun read pink |
| Pines | variety | the authored snowy-pine variety (user's pick; Lab: `--tree flurry --winter-pines`); shader snow on summer pines remains for shrubs and jungle |
| Broadleaf | `objects.hlsl` | snow-laden crowns (user's pick; the bare-canopy option is `WINTER_BARE` > 0) |
| Authored snowy pines | `objects.hlsl` | keep their own snow; cool fill only |
| Tree snow hue | `objects.hlsl` | hue lock: the canopy's sunlight is orange (grey probe 1 : .85 : .75 vs ground 1 : .95 : .86), so snow keeps the pass's luminance but takes the concept's cool white |
| Beaches, banks | `q6_raw_main` | authored snow texture (t8) over the underlay |
| Shore and lake beds | water dispatch | snow at the waterline, pale ice further out; shallow water keeps its tint |
| Water | water dispatch | deeper, saturated winter blue; v11+: all water hues turn together (Coast, Sea and Ocean keep their differences), lightness steps widened, smooth darkening with tiles from shore (depth inverted) plus a mild step from the tile's Civ III class |
| Rivers (v9; v12 dims at dusk/night; v24 banks) | `hydrology.hlsl` river branch | user, 2026-10-21: icy, matching the sea; open channel in the sea's winter blue; shelf ice in some reaches only, scalloped, snow-covered to a thin glassy edge; a few mid-channel floes; banks carry the ground's snow (gain-matched) to the water |
| Shoreline rocks | `q4_coastal_rock` | user, 2026-10-21: cool grey stone, thick snow caps on their tops |
| Farms | `q6_raw_feature` | rows against each plot's own blurred mean; per-plot state from that mean colour: fallow under snow, ploughed (dark furrows), straw stubble; field paths as dark hedgerow lines |
| Rocks, resources, small plants | `q6_raw_feature` | snow on up-facing surfaces, partly dormant colour; units excluded |
| City buildings | `PSNativeCityBody` | snow coat on roofs and ledges (normal z), lit after the city look so its whitewash shoulder does not grey it; walls keep their material; windows still glow at night |
| Settlement plates | `PSNativeCityBody` | trodden snow over streets and yards |
| Roads, railways | `hydrology.hlsl` | trodden grey-brown snow tracks; rail beds snowed, ballast and steel kept |

## Experiments and diagnostics

- `probe-trees`: foliage classes as flat colours (confirms `objects.hlsl` draws
  every tree; leafy atlases g/b 4.4 classify broadleaf, pines 2.2).
- `probe-grey`: neutral grey on trees and relief; found the orange canopy light.
- `-sp` views: pine forests drawn with the authored snowy-pine variety. It reads
  closer to the concept's distinct dark conifers than shader snow on summer pines.

## Coast, Sea and Ocean readability (user, 2026-10-21)

`water_classes.py` samples each Coast (11), Sea (12) and Ocean (13) tile's
centre in the `seas` view (test.biq 60,26, tile 96) and prints the OKLab
distance between class means (about 2 is barely noticeable, 5+ clear).

Production today, Summer and Winter alike: Coast–Sea 8.6, **Sea–Ocean 1.2**,
Coast–Ocean 7.6. The water's depth tint comes only from shore distance
(`lab/shared/hydrology/field.h`: `depth=.46*(1-exp(-offshore/.85))`), which
saturates about two tiles out, so Sea and Ocean share one body colour in every
season. The water shader does receive the tile class (`real_terrain`). v10 rotates
all water hues together (differences kept), widens the lightness steps and adds
a per-class step (Coast lighter teal, Ocean deeper navy): Sea–Ocean 6.4, but
the step shows the tiles' diamonds because a pixel cannot see its neighbours'
class. v11 darkens open water smoothly with tiles from shore (the interpolated
depth inverts exactly) and keeps a milder exact step: Coast–Sea 10.4,
Sea–Ocean 5.6, Coast–Ocean 15.9, with faint depth bands. A production fix
belongs in the shared water for all seasons: a class term in the hydrology
depth would also blend smoothly across tile borders.

## Mountain snow flurry (user, 2026-10-21)

The flurry is an animated effect. Terrain is retained, so no shader-only look can
animate it. It runs in a private Lab tree (`flurry_tree.py`, under
`out/winter/tree-flurry/`; the checkout is never written):
`city_fidelity/compiler.h` `mountain_flurry()` emits one rising quad (kind 6,
material 96) per mountain tile on the tile's front corner. Both object paths
emit it behind the Lab switch `C3X_LAB_MOUNTAIN_FLURRY`, and
`city_scene_material.hlsl` draws three parallax layers of hashed cells.
`biq_preview.cpp` gains `C3X_LAB_EFFECT_FRAMES`.
`study.py render LABEL --look v10 --tree flurry --flurry --frames 90 --views ...`
writes `effect.mp4` beside the still. `flurry.py` is the earlier Mac
compositing mock with the same equations.

Production port: gate on the Winter season (not a Lab switch); add the season
to the object compile key, or emit always and gate in the shader. Every view
with a mountain then redraws continuously, like a smoking city.

## Decisions (user)

- 2026-10-22, trees: broadleaf forests keep **snow-laden crowns** (`WINTER_BARE 0`,
  the `-laden` looks); pine forests use the **authored snowy-pine variety**
  (production change: select `snow_pine_forest` for pine forests in Winter; the
  Lab previews it with the `snow-pines` / `*-sp` views).
- 2026-10-22, land: Desert, Plains, Grassland and Tundra must stay
  distinguishable under snow (`water_classes.py --land` measures them), but
  beauty first: v13–v16 exposed ground in patterns and read "so splotchy".
  Grassland and mountains stay fully covered; other types may show only a few
  splotches. v17 separates the types mainly through the snow itself (tint,
  crust, dune relief) with sparse stubble/sand only on plains and desert.
- 2026-10-22, roofs: snow on buildings read "too dull a gray" and lacked
  texture. Cause: the city look's soft shoulder (~.42, meant for whitewash)
  squeezed roof snow to about half the ground snow's brightness; the coat also
  flattened the roof relief and sampled the snow texture at map scale. v22
  (`ROOFS_LIT`) lights roof snow like the ground, outside the city look, keeps
  most roof relief, adds building-scale crust and bluer, darker creases (AO).
  That still measured ~60% of the ground's brightness: the baked city AO
  darkens whole roofs, and the 2–3 px grain vanished in the downsample. v23
  (`ROOFS_TEXTURED`, the final look) lights roof snow with only 35% of the
  baked AO and a gain of 1.5, which matches the open ground snow (a tan roof
  RGB 203,211,222 vs ground 206,213,226); gain 2.1 blew roofs out and lost
  face shading. Roof faces keep their own light and shade, fbm mottling sits
  at ~10 px, and a few bare dabs (more on steep faces) show the roof through,
  edged with thinner, bluer snow.
- 2026-10-22, river banks: the grey bank "reads like an un-textured gray
  line". It was v9's shelf ice, a fixed ~2.2 px strip inside both edges of
  every reach, grey because integrated shaders light snow about 1/1.4 as bright
  as the relief (the roofs' gap again). v24 (`RIVER_BANKS`, the final look):
  shelf ice grows only in some reaches (low-frequency noise), with a scalloped
  edge; the snow on it and on the bank uses gain 1.4, which matches the ground
  (bank ~200 vs ground 207 in the village view); a thin band of bare glassy ice
  and a few bare windows edge the shelf; open reaches meet the snow with a
  broken, faintly wet rim. Waterline stones were tried and did not read at
  gameplay zoom, so they were dropped.
