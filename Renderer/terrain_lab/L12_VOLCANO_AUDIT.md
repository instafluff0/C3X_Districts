# L12 Volcano Promotion Audit

Status: approved by the user on 2026-09-05.

## Authoritative map input

The L11 `Scenarios/test.biq` contains no volcano terrain, so it cannot satisfy
L12 without inventing a tile. L12 instead uses the installed Firaxis
`Conquests/5 Mesoamerica.biq`, parsed read-only through the same C3X Editor BIQ
implementation.

- Windows source: `C:\Program Files (x86)\GOG Galaxy\Games\Civilization III Complete\Conquests\Conquests\5 Mesoamerica.biq`
- Source size: 23,019 bytes (compressed BIQ; 241,504 bytes after parser decompression)
- Source SHA-256: `8519e37a7de6749e99f9ebf8d99d4bf47451a287c61ec07e15a844a53e27cda6`
- Read-only parser result: 60x120 map dimensions, 3,600 TILE records

## Deterministic 192-tile selection

`PREPARE_L12_BIQ_VIEWPORT.bat` selects a volcano-bearing 16x12 true-adjacency
diamond. The two local axes follow raw BIQ steps `(1,1)` and `(1,-1)`.

- Selected raw BIQ origin: `(23,45)`
- Visible cells: exactly 192
- Non-rendered two-cell authoritative adjacency halo: 128 cells
- Export fixture SHA-256: `1a1c169527c2982b631fa276978c20e174541028e8489bd453dc58ee8985f7a5`
- Real-terrain counts: plains 20, grassland 1, hills 28, mountains 6, jungle 18, volcano 2, coast 57, sea 44, ocean 16
- Base-terrain counts: plains 20, grassland 55, coast 57, sea 44, ocean 16

This official scenario window supplies two genuine volcano cells, allowing the
complete render to show one deterministic active presentation and one dormant
presentation without changing BIQ ownership. It also exercises the accepted
shoreline, water, hills, mountains, and jungle layers over a substantially
larger footprint. The source window contains no marsh, forest, or desert cells;
their accepted code paths remain unchanged but are not visual witnesses in this
particular authoritative crop. Forest density and scale are separately checked
against the authoritative L11 viewport, which contains 21 forest cells beside
14 mountains.

## Source truth and presentation boundary

Confirmed normalized source data:

- `terrain/feature/volcano` provides 256x256 LOD0 authored height, blend, and
  region-ID channels. The height field drives the visible body; the blend field
  owns the terrain-integrated skirt. The renderer does not generate a cone.
- Dormant volcano base/height and active volcano base/specular textures are
  normalized generic DDS resources in `TerrainNormalized`.
- The exact engine rule that chooses active versus dormant art is not exposed
  by the normalized terrain element, so L12 assigns the two BIQ volcanoes a
  stable coordinate-derived state solely to compare both authored material
  presentations.

Presentation boundary:

- The normalized height field is sampled over its complete authored extent.
  A rigid coordinate mirror/rotation plus bounded aspect and vertical fits map
  it into the Civ III diamond. The normalized pack contains one ordinary
  volcano terrain element, not multiple sprites; deterministic fits keep
  repeated instances from appearing cloned without inventing a second shape.
- Edge-adjacent mountain and volcano ownership cells enlarge and overlap those
  same authored footprints in world space. The maximum authored displacement
  supplies shared shoulders and saddles, while the outside of the connected
  component tapers back to ordinary land. No connector mesh is generated.
- The outer range boundary applies one broadened envelope to both authored
  displacement and authored material blend. Mountain/volcano color therefore
  fades into the incident grassland on the same skirt where geometry flattens,
  instead of stopping at the hidden Civ III diamond edge.
- Hills and authored relief also use the continuous signed-shore field as a
  coastal-clearance envelope. Their geometry and material flatten before the
  beach/water contour, preventing elevated tile lips at coastal adjacencies.
- Mountain geometry may continue through a narrow edge band into an adjacent
  authored hill so the hill cannot slice off the rear of a peak. The two height
  fields use a smooth maximum; rock remains mountain-owned and only a restrained
  transition band crosses onto the grass-covered hill shoulder.
- Civ VI's active base map is a mostly dark surface map with a small bright lava
  area. It is composited over the dormant rock rather than replacing the whole
  albedo: the dark authored variation is restrained and the bright source texels
  remain fully visible. This corrects the nearly black first candidate without
  inventing a lava color.
- L12 does not synthesize smoke, glow, fire, a cone, crater geometry, or other
  decorative effects. Any visible active coloration comes directly from the
  normalized active base/specular material channels.
- BIQ forest and jungle use the existing normalized vegetation bodies and
  ArtDef-derived weighted placement records without proxy geometry. A
  deterministic jittered lattice prevents random interior holes: forest uses
  36 bodies per cell at `0.42` scene scale and jungle uses 49 at `0.40`. This
  makes connected cells read as closed canopies while keeping individual trees
  and palms substantially shorter than adjacent mountain and volcano relief.
- Static shallow-water detail now uses the normalized nonzero
  `CLUTTER_OCEAN` source set: five rock/contour decals plus four shoreline
  cracks. Deterministic occupancy preserves the authored `0.52` set density;
  projected edges fade before their cells end, and crack coverage is clipped by
  the same continuous shoreline field as the bed and water passes. The source
  footprints are calibrated to Civ III scale so the result reads as small
  underwater structure rather than the rejected dark clouds. Increasing
  whole-surface transparency exposed a sandy bed and remains rejected.
- Grassland, plains, and grassland-hill projected clutter are normalized in the
  same generic pack and blended by the continuous BIQ material weights. Their
  base/height contribution is deliberately restrained so it breaks up broad
  land surfaces without becoming a second relief system.

Key normalized local asset SHA-256 values:

- Terrain-element contract: `4197b2609deed0e46b4685310ab1ed1e7341827fd7250e11e82ed286519e3295`
- Height LOD0: `2a6a4bfd3621b7c4f7679f07c1b2a6eb2512d70bf95f36f6193ef3ae1c331462`
- Blend LOD0: `3ebae1f544bf21ee0f187a4f70c79898c4b60b139c45cccad87e04258932c18a`
- Region IDs LOD0: `71ab4d87d4b4445bbcc80f732d2e85b109bc2758bfc64e9fa49a5a5cdf55bd52`
- Dormant base/height: `0df103895e3b863dce886271a36d358c9e07966cae6a3bfd19ff448f1a3a1538` / `1bf210d121afa17fa89aadc17aaca3a00ebe56071a44e7f67d7d5f06aafaee4e`
- Active base/specular: `68d1b3617a82a473a3a854784697a867bd672f37a1bb34f311e3b51948b1f5d8` / `4a71c72eeabae0b3954a3c6fc009838523ef1cfa008aab005bc499ab57c4bc7d`

## Candidate outputs

- Complete SHA-256: `5b75e1573392130ba6fff1baf5b2ce12d5d109ac34e4c47561481a56cb37f56c`
- No-volcano SHA-256: `bfae617dedc008171ece35c11fc0af17ce2ea7c5ad5e9d38099059feb9d0d69a`
- Volcano-only SHA-256: `34234893fc87a822a02cadda9a28ec99d33d1f2d8dbd6c8ce60efe03fad0fbaf`
- Thumbnail SHA-256: `f01c29fe4f29f6a55b70f7cc88ab246f4e281863338ecc8eab48217e0c7912fb`

Two consecutive L12 batches produced these exact hashes. Automated contracts
also enforce the 16x12 count, deterministic halo, generic volcano channel use,
and separate complete/no-volcano/volcano-only modes. The user explicitly approved
the complete promotion render on 2026-09-05; the frozen integration record is
`handoffs/L12_volcano.json`.
