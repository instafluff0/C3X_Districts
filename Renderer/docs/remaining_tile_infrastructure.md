# Remaining Tile Infrastructure Intake

Status: L19B Renderer Lab promotion complete and frozen; runtime ownership remains disabled until I19B.

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

## Closed L19B Families

- **Radar tower:** `IMP_SILO`, the Modern Fort, and the flat observatory plaza
  remain rejected. The accepted generic mapping uses the normalized
  `VIL_BAR_IND_Tower` source body's narrow watchtower/antenna silhouette at
  restrained strategic scale. It reads as a radar post without masquerading as
  a fort, city, or missile silo.
- **Pollution:** the accepted static ground state combines a low-opacity
  source crater-blast soil footprint with the installed
  `NUCLEAR_FALLOUT -> FX_Radiation` atlas. Both source layers feather before the
  tile-local quad edge; the radiation layer contributes only its authored alpha.
  Civ III's pollution flag owns lifetime. No particles, disaster radius,
  detonation timing, smoke, glow, or animation are inherited.
- **Crater:** all four `CLUTTER_CRATERBLASTS` crater decal roots now normalize
  as `infrastructure/crater/variant_01..04`. Lifetime must be owned by Civ III's
  crater flag, independently of the transient impact event that may have
  created it.
- **Victory location:** the accepted restrained marker uses source-authored fort
  pole and bunting components. It is deliberately smaller and simpler than a
  city, wonder, or fortification and carries no invented effect.

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

## L19B Acceptance Evidence

The frozen 192-tile alternate-skin scene renders every family across all four
eras/owners, routes, relief, stable rotations, noon, midnight, full scale, and
reduced scale. Family-isolation frames prove silhouettes and ground blending;
the no-infrastructure control is byte-identical to L19A. Two unchanged Lab runs
are byte-deterministic. See `terrain_lab/L19B_INFRASTRUCTURE_AUDIT.md` and
`handoffs/L19B_remaining_tile_infrastructure.json`. I19B owns later capture,
invalidation, exclusive native suppression, retained layers, and failure
behavior; none is enabled by the Lab promotion.
