# Goody huts and Civ III colony stand-ins

## Result

Goody huts have a direct installed-source match: Civ VI's
`IMPROVEMENT_GOODY_HUT -> LM_GOODY_HUT` resolves to three tribal-thatch
compound landmarks. All three roots and their recursive static components now
compile into a source-independent local pack.

Civ III colonies have no equivalent Civ VI gameplay object. Their visual role
is frozen as a **small owned resource-logistics outpost**, not a city, fort,
barbarian camp, or duplicate resource extractor. The closest useful upstream
body is Civ VI's ordinary resource-camp family. Three preindustrial and three
industrial camp roots now compile strictly with their barrels, pots, racks,
tables, cabins, wells, tents, rocks, and related components. The earlier
industrial rejection was stale evidence from before the row-vector matrix-order
correction; remeasurement passes below `8e-7` without changing the shared
`2e-5` threshold. No static bake or generic replacement is needed.

This is offline intake only. No native art is suppressed, no runtime renderer
path is enabled, and the ready L13A gate is unchanged.

## Confirmed Civ III behavior

The installed executable evidence supports these rules:

- `Tile::m15_Check_Goody_Hut(viewer_civ_id)` reads bit 5 from the
  viewer-conditioned overlay value. Scene capture must use the active viewer;
  it must not reveal a hut merely because the underlying tile has one.
- `Map_Renderer::m12_Draw_Tile_Buildings` seeds its local random stream from
  map seed and canonical tile index, advances it four times, then selects one
  of eight goody-hut images. The replacement therefore uses a stable eight-
  bucket selection followed by the checked 8-to-3 source-variant mapping. It
  does not require mutable animation state.
- The same tile-building pass checks `Tile::has_colony` and calls the colony
  draw independently of barbarian camps, mines, radar, and outposts.
- The colony draw selects one of four native sprites from
  `leaders[colony_owner].Era`. Era is owner-driven, not tile-territory-driven.
- Existing C3X state already exposes `p_colonies`, tile-building ID, and
  `Tile_Building_Body.{X,Y,OwnerID}`. This is important because C3X can allow a
  colony owned by one civilization inside another civilization's territory.

Decompiler register recovery around `m12` is imperfect, but the invariant
facts above are independently visible in the tile accessor, deterministic
seed construction, eight-way `rand_int`, colony list use, and era-indexed
sprite address.

## Goody-hut rendering contract

### Selection and presentation

- Three normalized roots: `tile_object/goody_hut/variant_01..03`.
- Eight logical reference buckets preserve Civ III's authored variability.
  Buckets map `[0,1,2,0,1,2,0,1]` into the three source roots.
- Bucket selection is a stable hash of world seed and canonical tile index.
  Wrapped screen occurrences of one logical tile therefore show the same hut.
- Huts are culture-neutral and era-neutral. Civilization style, territory
  owner, nearby city culture, and current world era must not alter them.
- Source `FlattenTerrain` is evidence about Civ VI placement, not permission to
  flatten C3X terrain. The normalized compound is grounded on the renderer's
  accepted terrain surface.
- Facing uses stable diagonal choices compatible with the Civ III projection.
  The authored entrance/front keeps a clear screen-facing sector; random full
  yaw that hides the entrance is rejected.

### Night and ambient behavior

The converted graphs contain the static torch/campfire bodies and exact
attachment sockets. Three optional unresolved effect resources were found:
`FX_Campfire`, `FX_Bone_Campfire`, and `FX_Bone_Torch`. These line up with the
separately audited FireFX/light families, but their resource scripts are not
decoded here.

The lab presentation is therefore staged:

1. Static body and lit campfire/torch geometry can be rendered immediately.
2. At night, bind a restrained warm local light to the accepted fire/torch
   socket through the shared M6.4 environment contract.
3. Animated flame and smoke remain M7.5 attachments until the generic effect
   graph is decoded and calibrated. Their absence cannot block the static hut.
4. Animated attachments request ticks only while visible and active. A static
   hut adds no redraw work.

Popping a hut removes the body when the authoritative overlay bit clears.
Civ III continues to own the reward, gameplay mutation, message, and sound.
No result-specific art may predict a reward before it is rolled.

## Colony stand-in contract

### Why a logistics outpost

A Civ III colony is a worker-built, owned claim on a resource outside a city.
It can sit on any resource, so a fixed mine, pasture, plantation, or oil well
would communicate false gameplay. A city-sized village would imply population
and borders; a fort or barbarian camp would collide with separate Civ III
objects. The resource-camp kit supplies small shelter and handling props that
read as extraction/logistics without dictating the resource type.

### Composition

- Render one accepted camp compound at scale `0.62`, never the source family's
  resource-conditioned deer/furs/ivory/truffles attachments.
- Keep the independently rendered Civ III resource body visible. Choose a
  stable clear quadrant away from its silhouette; do not place a second copy
  of the resource in the colony kit.
- Face the outpost toward tile center, snapped to a projection-compatible
  diagonal. This yields a natural authored front while keeping the tile
  footprint deterministic.
- Add a generated slim pennant and a small awning-trim band from the effective
  Civ III owner-color lookup. The marker uses at most 14% of visible colored
  surface and must cover at least six reduced-view pixels, avoiding both the
  invisible and overwhelming unit-tint failure modes.
- Read color and era from `Colony_Body.OwnerID`. Never substitute
  `Tile::m38_Get_Territory_OwnerID`.

### Era policy

Civ III exposes four owner eras and the catalog keeps four distinct profiles.
The accepted source bodies now cover both preindustrial and industrial profiles.
Industrial and modern eras select among the three strict industrial roots;
promotion still decides their final scale and density.

The intended final progression is:

| Civ III era | Body | Owner marker | Night cue |
| --- | --- | --- | --- |
| 0 Ancient | canvas/wood resource camp | small cloth pennant | fire/torch |
| 1 Medieval | accepted camp with denser supply dressing | pennant plus trim | lantern/fire |
| 2 Industrial | strict industrial camp compound | sharper flag/painted trim | warm utility lamp, optional smoke |
| 3 Modern | reduced industrial camp compound with cleaner selection | compact high-contrast marker | electric task light |

The corrected strict converter measures `IMP_Camp_IND_01`, `_02`, and `_03` at
maximum bind-pose errors `3.3737868e-7`, `7.9908300e-7`, and `3.3737868e-7`.
These pass the unchanged threshold. The recursive build accepts all dependencies
with zero optional rejects, so a lower-fidelity static bake is explicitly not
used.

## Capture, ownership, and invalidation

No new Civ III patch symbol is needed for the planned implementation:

- the existing `Map_Renderer_m19_Draw_Tile_by_XY_and_Flags` capture supplies
  the logical tile, authoritative screen occurrence, clip, wrap, zoom, viewer,
  and terrain context;
- the tile's viewer-conditioned goody accessor supplies hut visibility;
- `Tile_has_colony`, tile-building ID, `p_colonies`, and the colony body supply
  presence, owner, and coordinates;
- the existing effective leader palette lookup supplies owner color; and
- the existing map-seed/tile identity supplies deterministic variants.

The tile/object scene record should carry `kind`, canonical tile identity,
screen occurrence, asset/profile ID, stable variant bucket, terrain-fit
transform, visibility/fog class, and optional attachment IDs. Colony records
also carry owner ID, resolved owner-color row, and owner era. Presence, owner,
era, viewer visibility, hour/season, asset-definition revision, and resource
body bounds participate in the fragment fingerprint. Movement is impossible;
creation/removal/era/color changes are handled on the next authoritative map
redraw and content-derived cache comparison.

Before promotion, each family needs isolated 192-tile lab evidence at both
zooms, day and night, slopes and vegetation, fog states, wrap duplication, and
same-tile resource coexistence. Custom-on ownership is all-or-nothing per
promoted family: a missing or failed replacement fails the custom frame and
never replays the native body.

## Reproducible intake

Run:

```text
python3 Renderer/tools/asset_compiler/tile_object_asset_importer.py
```

The checked strategy is
`tools/asset_compiler/tile_object_render_strategy.json`. The ignored local pack
contains 3 hut roots, 6 reusable colony roots, and—with the prepared
infrastructure families—91 recursively compiled components, 243 geometry parts,
179 materials, 79 textures, and 542 attachment points. The pack passes the
generic runtime-independence validator. Its build
report remains under `preview/out/tile_objects/build.json` because it contains
machine-local installed-source paths.
