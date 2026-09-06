# Remaining Tile Infrastructure Intake

Status: offline source intake prepared; no Lab promotion or runtime ownership is enabled.

## Why This Is A Separate Gate

The vanilla inventory includes several map bodies beyond roads, rails,
resources, cities, mines, farms, goody huts, and colonies: fortresses,
barricades, airfields, outposts, radar towers, pollution, craters, and victory
locations. They were classified but had no dedicated visual gate. L19B/I19B
now owns them so they cannot silently disappear between L19A and units.

## Confirmed And Converted Source Graphs

The installed Civ VI ArtDefs provide exact graph roots for two major families:

- `IMPROVEMENT_FORT -> LM_FORT` resolves Medieval, Industrial, and Modern
  source roots. Medieval and Industrial roots pass strict normalization.
- `IMPROVEMENT_AIRSTRIP -> LM_AIRSTRIP` resolves one complete airstrip root.

Fort bases are not single meshes. They are terrain decals with declared
attachments for walls/earthworks, banners and flags, barrels, rocks, buildings,
cannon, cannonballs, boxes, and wagons. Some attachment records use the generic
`ROAD` connection type. The compound adapter now preserves that as normalized
`connection_type: road` instead of rejecting a valid source graph.

The airstrip is likewise a decal-plus-components system. Its attachments
include a control/watch tower, windsock, tents, hangar, house, barrels, trucks,
lanterns, and runway lantern rows. This gives L19B real day/night inputs rather
than requiring invented runway lights.

Two tower bodies also normalize independently:

- `IMP_Airstrip_Tower`
- `VIL_BAR_IND_Tower`

They form the preindustrial and industrial/modern Civ III Outpost candidates.
The runtime IDs are generic and the outpost remains a small sight-post, not a
barbarian camp or colony.

The recursive tile-object build now accepts five infrastructure roots alongside
the already prepared huts and colonies. The combined ignored proof pack contains
77 components with dependencies, 223 geometry parts, 156 materials, 75 unique
textures, 71 emissive material bindings, and zero rejected optional
dependencies. These totals describe the combined tile-object proof pack; the
five new infrastructure roots account for the increase from the earlier
30-component hut/colony build.

## Presentation Mapping

- **Fortress:** use Medieval fort art in Civ III eras 0-1 and Industrial art in
  eras 2-3. Keep owner color to declared flags/banners.
- **Barricade:** reuse the era-correct Fort source graph but select a denser
  authored wall/earthwork perimeter, so it reads as a clear upgrade rather than
  an unrelated structure.
- **Airfield:** use the complete airstrip compound at one stable diagonal runway
  axis; activate only source-backed lantern/runway-light attachments at night.
- **Outpost:** use one small tower body with restrained owner pennant and at most
  one guard light. Do not render the full airstrip or barbarian camp.

The checked strategy covers all four Civ III eras for those four accepted
families. Rotation remains projection-aware and deterministic, terrain follows
the renderer surface, and Civ III state remains authoritative.

## Honest Deferred Families

- **Radar tower:** `IMP_SILO` remains rejected. A separately discovered
  observatory main body normalizes cleanly, including an emissive material, as
  `candidate/infrastructure/radar_observatory_body`. Automatic tile fitting and
  a 32-cell eight-facing/two-zoom/day-night sheet now prove that the isolated
  body requires 11.22191796x enlargement and reads as a flat ornamental plaza,
  not a tower or antenna. It remains available as evidence but is a weak L19B
  candidate rather than an approved mapping.
- **Pollution:** use the installed `NUCLEAR_FALLOUT -> FX_Radiation` family as
  the preferred visual direction. Five Base radiation color/alpha/atlas
  textures now normalize under generic `effect/pollution/*` IDs. L19B should
  show a restrained tile-local subset of the green ground wisps, with lifetime
  owned by Civ III's pollution flag; it must not inherit the source disaster
  radius, city-wide density, or detonation timing. A bounded generic profile now
  caps the tile-local presentation at seven particles across both zoom policies;
  visual calibration remains pending, so this is not yet a promoted body.
- **Crater:** all four `CLUTTER_CRATERBLASTS` crater decal roots now normalize
  as `infrastructure/crater/variant_01..04`. Lifetime must be owned by Civ III's
  crater flag, independently of the transient impact event that may have
  created it.
- **Victory location:** explicitly set aside by user direction. No candidate or
  fallback is advanced by this preparation pass.

Two rejected source records are preserved in the strategy rather than hidden:
the Modern Fort root has maximum bind-pose error `0.05620446733770823`, and the
missile-silo root has an invalid decoded mesh range. Neither threshold is
weakened to make the intake appear complete.

## Commands

Compile the complete recursive tile-object proof pack locally:

```powershell
py Renderer\tools\asset_compiler\tile_object_asset_importer.py
```

The smaller root-only source probe is
`tools/asset_compiler/infrastructure_source_sets.json` and can be passed to
`compound_landmark_importer.py`. Runtime selection is defined by the
`infrastructure` section of `tile_object_render_strategy.json`.

## L19B Acceptance Boundary

L19B must render fortress, barricade, airfield, and outpost at both zooms across
all reachable eras, owner-color cases, terrain/relief grounding, rotation,
clipping/wrap, and noon/night. Barricade must be visibly stronger than Fortress;
Airfield runway lights must be dark by day. Radar, pollution, crater, and
victory-location mappings must be closed with exact or explicitly approved
generic art before L19B can promote. I19B then owns capture, invalidation,
exclusive native suppression, retained layers, and failure behavior.
