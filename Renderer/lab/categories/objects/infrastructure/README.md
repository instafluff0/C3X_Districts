# Infrastructure

Current roads, railroads, mines and farms; preserve category ownership and fallback.

The fixtures supply two connected runs and a cross-branch, plus a separate mine
and irrigated farm, with and without surrounding relief/vegetation. The
`network` case stamps roads on most land around one railroad corridor, two
road eras, a city, hills, a small range, a wood and a river. All four custom
ownership flags are checked through the production API.

Roads and railroads (accepted for the game; the route shading ships in the
Renderer64 pack): when the local `Renderer/packs/RoutePatternsRuntime` pack
exists, each tile draws Civ III's own pattern for its 8-neighbor mask (bit k
is neighbor k+1: NE, E, SE, S, SW, W, NW, N). As in
`Map_Renderer::m16_Draw_Tile_Roads`, a road links road neighbors except where
both tiles carry a railroad, and an unconnected city or railroad tile draws no
road. As in `m14_Draw_Tile_Railroads`, a railroad links railroad neighbors and
draws `railroads.pcx`'s pattern; a fully connected railroad picks one of that
sheet's 17 full-junction variants by a stable hash of its position, and an
unconnected city draws none. Where roads and railroads meet they simply
overlap. `build_route_pattern_runtime.py import` copies the installed
`Art/Terrain/roads.pcx` and `railroads.pcx` into the ignored
`RoutePatternSources`; the `route-patterns` asset job thins each painted path
(the railroad ladder first closed into one band) into tile-local centerlines
whose ends land exactly on the shared edge midpoint or corner. Junction pixels
within 3 px form one junction (the pieces inside it are dropped), pieces
through two-piece nodes join into one path, and short spurs are dropped.

Runtime draws a terrain-draped strip along each centerline: Civ VI's ancient
dirt piece for every road era (3.5–4.5 px at a 128-pixel tile), or, for a
railroad (3.85–4.95 px, so a dense rail network does not outweigh the roads),
Civ VI's recipe of rail pieces over a dirt bed: a worn earthy bed reaching
1.65 stroke widths, the atlas's ballast-and-sleeper strip
fading in over it, and its two steel rails on top (unwrapped sleeper
coordinate). Near every shared join both tiles ease their path onto the
join's shared axis, so the halves meet tangent to each other. The stroke keeps that screen width on slopes,
so a path across a steep face narrows instead of smearing down it. The shader
blends a strip into the ground by height, as the source route material does
(its "height" texture is the base color's alpha): a worn profile lowers the
strip toward its edges and a wide blend against the owner tile's ground
height map (grass, plains, desert, hills, mountain or marsh, passed per
vertex) feathers the shoulders. Roads read as a light tan-brown worn path;
the blended height also shades the strip. Pattern routes follow the
whole rendered mountain surface, so the rock never cuts them along a ragged
contour, and each route point fades out between 35 and 65 height units of
mountain rise: routes show on a mountain's feet and lower flanks and dissolve
before its upper slopes, as Civ III draws routes on a mountain tile's lower art
but never over its peak. A railroad running into a mountain goes through
a tunnel at ground level instead (Lab, in review; `tunnel_route`). The user
asked for the portal at ground level, not up the mountain, with only the
entrance showing (2026-10-07). The rail is hidden where it lies in rock: at
least 4 units over the land with rock (35 units) within 0.2 tile. Between two
mountain tiles it is also hidden from their shared edge until it reaches
rock, so it never surfaces in a range's saddle. Where hidden rail meets
shown rail, Civ VI's tunnel portal entrance stands at the mountain's foot,
seated on the land and turned toward the shown rail. At an open tile edge
where a range's tunnel ends, it stands at that edge. The shown rail ends
inside the arch. The portal is 70% of the railroad bridge's calibrated scale,
as pattern bridges are. It comes from Gathering Storm's `IMP_Mountain_Tunnel`
(`route_tunnel_sets.json`, compound importer with
`--terrain-edits preserve_unresolved`, into `RouteTunnelsNormalized`).
Civ VI hides the block behind the facade, and the rock cap on it, under a
terrain edit; the mountain's foot cannot. So `build_route_bridge_runtime.py`
keeps only the entrance (facade, wing walls, floor and bore). It goes in the
unused medieval pillaged bridge's texture slot (1), so no shader change is
needed. The Lab `network` case runs a railroad through its small range (raw
row 16, fixed to the range whatever the view center). Neighboring tiles share one axis at every
join (a variant's neighbors compute its variant too), and route strips are a
depth-tested decal without depth writes. Renderer64 compiles its shaders from
`Renderer/packs/Renderer64ResidentRuntime`: after a Lab route-shader change,
run `python3 Renderer/tools/overlay_route_shading.py --backup <dated path>` to
overlay only the route branch (it writes `route-overlay.json`).

Only a link across a tile-diagonal river edge (NE, SE, SW, NW) gets a Civ VI
bridge: a road's by era (medieval for ancient/medieval, then industrial and
modern), a railroad's truss. The bridge lies on the tile diagonal, square to
its edge. The rendered river bows up to about a quarter tile off its edge
around hills, so the bridge moves along that axis onto the middle of the
water (both tiles find the same point). The authored meshes put their deck
ends at z=0, so the bridge rests on the lower of its two banks (one shared
rigid seat), and both halves end under its ends. Pattern bridges stand at
70% of the meshes' calibrated scale. Out of a bridge the network runs
straight along its axis through the center of the deck and about 0.12 tile
beyond (less when the bridge stands deep in the tile), then eases back onto
the pattern. Civ III rail patterns often fork just inside the edge, and a few
leave a loose stub short of the line it meets. A line closer to the join
along the network than the straight run's end is hidden up to there, and its
parts beyond start at the run's end. Rivers run in valleys about 15 units
below the land beyond ~0.6 tile from the water. A level deck on the valley
floor sits that far below the paths coming over the land, and on the fixed
oblique view a straight path descending to it is drawn bent into the deck's
side. So the deck stands at the top of the lower bank (the highest route
ground out to 0.75 tile on each side, the lower of the two sides, at most 20
units over its ends). Its paths are carried level to it where their ground lies
lower, out to 0.75 tile, then fade back onto their ground. The user chose this
over the higher bank, which looked silly with a high bridge and long raised
approach (2026-10-07).
Route strips draw 6.5 units over their height (world z +9 against the
features' +2.5), and the rails stand about 2.5 over the deck, so the carried
level is 4 under the deck's. Renderer64 gives bridge
materials the natural depth basis plus the river's layer bias, so decks show
over raised rivers. A link through a corner (N, S, E, W) that the river
separates gets no bridge. Elsewhere a stretch inside a river channel moves
toward its tile's center onto dry bank, a join touching a river bend moves
toward the open bank, and whatever still lies over water fades into the bank.
Without the pack the previous segment routes remain.
`Renderer/lab/studies/roads/civ3_patterns.py --before` renders overview,
gameplay and close zooms beside the segment roads; `--eras` renders each road
era and its bridge (`C3X_LAB_ROAD_ERA` overrides the `network` fixture era).

This category does not imply new support for pollution, craters, colonies or
other tile improvements that the current renderer does not replace.

`standard.json` identifies the shared implementation, dependencies, fixture recipe
and focused regression tests. The current checkout is authoritative.
Reference captures reproduce production using small synthetic diagnostic scenes;
they are not claimed to be live Civ III captures. References are optional review
aids. See `Renderer/docs/visual_fidelity_playbook.md`.

Farms (accepted 2026-10-06; plots promoted 2026-10-07): a farm kit drapes
green fields on the rendered ground and keeps its routes, resource and water
open. Production lays out rectangular plots along the routes, joined across
tiles within each small area the route network encloses. See
`Renderer/lab/studies/farms/README.md`; packs without a `farm_kit` group keep
the earlier quadrant fields.

Mines (user choice 2026-10-07): every era draws one building, Civ VI's Mine AN 03
main building (`IMP_MINE_ANC_Bld_C`, a timber headframe and ore house). It is shown
alone, at 1.2x Civ VI's size, with its Civ VI front turned toward the camera. The user
chose 1.2x after comparing 2.0x, 1.6x and 1.4x on the 1498 AD save.
- The choice lives in `improvement_render_strategy.json` (`mine.runtime_building`).
  `build_mine_runtime.py` writes it into `ImprovementsNormalized/mine_runtime.bin`.
  The previous file is kept beside it as `mine_runtime-before-bldc-20261007.bin`.
- Mines stand on the visible ground, as sites do: centred on flat land and on a
  hill's crown (0.9x), and at a mountain's camera-facing foot (0.75x).
- Their material fraction adds .0035, the farm kit props' natural height-depth
  marker, to the emissive code digit. With the feature basis, the hill or mountain
  under a mine hid its lower half.
- Tests: `Renderer/native/test_mine_placement.py`.
- Before/after study: `Renderer/lab/studies/mines/in_game.py`.
- Gallery of Civ VI mine and quarry kits and parts: `Renderer/lab/studies/mines/kit_gallery.py`.
