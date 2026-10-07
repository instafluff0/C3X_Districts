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
railroad (5–6 px), Civ VI's recipe of rail pieces over a dirt bed: a worn
earthy bed reaching two stroke widths, the atlas's ballast-and-sleeper strip
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
but never over its peak. Neighboring tiles share one axis at every
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
rigid seat), and both halves end under its ends. Renderer64 gives bridge
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

The farm study under `Renderer/lab/studies/farms/` renders deterministic irrigation
placements over the unchanged `test.biq` terrain. Its current candidate uses
source crop-atlas rows instead of shrinking the full atlas into each field,
samples field vertices against terrain relief, clips them at the shore and river
bank, and
varies palette, placement, and source tree/building composition by tile seed.
The atlas subregion choice and field layout are visual reconstruction decisions;
the source decal metadata confirms the materials and footprints, not this exact
Civ III tile arrangement. The compact runtime still omits the source crop height,
specular, and foliage opacity response. These examples have not been visually
accepted or staged for game use.
