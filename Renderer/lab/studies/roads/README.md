# Roads on `test.biq`

Run `python3 Renderer/lab/studies/roads/test_biq.py` from the project root. The
study exports the original BIQ to a local CSV, stamps a deterministic road
network on that preview only, builds an isolated native candidate, and writes
era, context, and isolated-road sheets under `Renderer/lab/out/roads/`. The BIQ
and normal game installation are unchanged. Generated packs remain local and
are not part of C3X runtime distribution.

The fixtures include four route eras on the same road graph, a matched no-road
control, dense late-game coverage, hills and mountains, river crossings,
forests, a coast, two sun angles, and a road with no neighbors. The study checks
that two identical modern captures have the same image hash. The existing
authored bridge arches sit at the river crossing. A raised roadbed
and narrow side fascia close their missing deck; both tile halves meet at the
arch crown, with approaches blending into the banks. River distance sampling
also places a bridge where the road crosses the channel inside a tile. The
ordinary dirt road remains narrow; only its bridge deck fills the arch gap.
Each tile contributes
its half of an incident edge so the path stays visible where adjoining terrain
meshes overlap. A road tile can connect in all eight directions, including
left-right and top-bottom pairs; a sparse diagonal stays connected, while
redundant diagonals in a dense network are omitted. Mountain tiles connect
their spokes through an interior skirt path that omits unused arcs and bends
toward low saddles. Where a rocky crown extends into another tile, the path
narrows into the occluding mass instead of painting across a sheer face.
Tile coordinates seed
junction positions, width, atlas-region, and bend variation; a shared edge
seed agrees at the seam and replay is stable.

## Source evidence and reconstruction choices

The local normalized route pack supplies separate ancient, medieval,
industrial, and modern road styles. Its `route_road_*.json` records an authored
`tiled_path`, `fadeout`, and `transition` piece, a 512 by 512 atlas, a nominal
width of 0.06 tiles, and terrain-conforming placement. The base-color textures
are sRGB with alpha; the pack also records linear height and fog-color channels.
The current runtime uses base color and alpha. It does not yet interpret the
height channel as a normal or bump map, because that channel's exact meaning
and scale have not been established for this renderer. The path UV subregion,
graph pruning, width range, and bend recipe are Civ III reconstruction choices,
not claims about Civ VI's own layout algorithm.

Following `Renderer/docs/visual_fidelity_playbook.md`, this study preserves the
authored atlas detail rather than sharpening the final image, samples the path
mesh against world terrain relief instead of painting a flat tile sprite, and
uses the same scene lighting and relief normal basis as the surrounding ground.
Roads use terrain depth so they sit above hills and the mountain mesh. They
receive scene lighting but do not cast a raised shadow. Bridge arches cast
their own shadows; the procedural deck has side normals and follows the road
material for its era. The study
does not claim complete source material parity: further source-channel work,
vegetation clearance, all-zoom review, and live Civ III capture remain for a
later visual promotion. These images are candidate Lab examples, not accepted
game integration evidence.

The isolated runtime root includes private declaration and local-name repairs
for unrelated in-progress terrain and vegetation edits; their shared source
files are left untouched by this study. The render script records candidate
and shader hashes with each capture and leaves any uncertain native fixture in
place for audit.
