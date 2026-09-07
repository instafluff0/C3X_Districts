# Barbarian Camp Import And Runtime Strategy

This is an offline intake contract. It does not enable custom camp rendering,
change Civ III gameplay, reopen the completed L19A handoff, or advance LQ0.
Barbarian camps need their own modular Lab v2 visual gate before Game
Integration may suppress the native camp sprite.

## Exact upstream result

The installed Civ VI chain is exact and distinct from both goody huts and the
resource-camp art used for the Civ III colony stand-in:

```text
IMPROVEMENT_BARBARIAN_CAMP
  -> LM_BARBARIAN_CAMP
  -> VIL_BAR_01       (preindustrial)
  -> VIL_BAR_IND      (later-era alternative)
  -> Base/Platforms/Windows/BLPs/landmarks/tilebases.blp
```

`barbarian_camp_source_probe.py` resolves that chain through the installed
ArtDefs and package index. The same probe confirms that Civ VI gameplay data
marks the improvement as a barbarian camp, removes it on entry, and awards 50
gold. Those values are source evidence only. Civ III remains authoritative for
camp placement, capture, rewards, messages, sounds, and units.

The recursive importer accepts both exact roots. `VIL_BAR_01` has 94 declared
attachments: 92 resolve, spanning 14 unique normalized child assets. Two
optional references to `VIL_BAR_Skull_Pile` are explicitly omitted because its
source material has no required base-color channel. The omission is bounded and
does not remove the camp perimeter or its primary silhouette. `VIL_BAR_IND` has
25 of 25 attachments resolved across 13 unique children. It includes exact
unlit/lit fire-barrel children. The preindustrial root exposes no equivalent
fire, light, smoke, or torch child, so a primitive campfire would be an authored
effect rather than decoded Civ VI behavior.

The full shared tile-object conversion now closes with 118 normalized
components, 268 geometry parts, 242 material records, 85 textures, and 665
attachment points. The compact source-independent runtime proof also builds.
Derived output stays in the ignored local pack; C3X runtime has no dependency
on Civ VI files or formats.

## Civ III authority

The decompiled Civ III path gives the renderer everything needed without a new
patch symbol:

- `Tile::m7_Check_Barbarian_Camp(viewer_civ_id)` reads overlay bit 7 through the
  visibility-conditioned overlay accessor. Capture must use the active viewer,
  not omniscient tile state.
- `Tile::m44_Get_Barbarian_TribeID` supplies stable camp identity. Native draw
  reads this value before its dedicated barbarian-camp call.
- Native spawn assigns the camp bit and tribe ID, then creates two ordinary
  units separately. Guards therefore remain authoritative unit-renderer
  instances; they must never be baked into camp geometry.
- Native capture clears bit `0x80`, releases the tribe ID, and owns the reward,
  message, sound, and unit consequences. The custom body disappears as soon as
  the captured camp bit clears.

Civ III's native visual reference is one `128x64` slice in
`Art/Terrain/TerrainBuildings.pcx`. It has no owner-color or era array. Tribe ID
is useful for deterministic cache/variation input, but does not authorize a
culture mapping or territorial tint.

## Default presentation

The fidelity default is `VIL_BAR_01` for every Civ III era. The later
`VIL_BAR_IND` root is retained as an optional pack profile, not selected
automatically by world or civilization era. This prevents a late-game camp from
reading as a colony, owned outpost, or industrial settlement.

One stable hash of world seed, canonical tile index, and barbarian tribe ID
selects a root when a pack supplies multiple roots within a stage. The same
inputs plus attachment ID resolve the source `OPTIONAL` child sockets. Wrapped
occurrences therefore compose identically, save/reload does not reshuffle the
camp, and different camps can still vary. Rotation snaps a source-authored
entrance to a stable Civ III diagonal with entrance clearance; it cannot change
on redraw.

The body is neutral and receives no owner color. Barbarian colors remain on the
independently rendered units. A future pack may offer an explicit scenario
override, but the default cannot infer style from nearby culture or territory.

Civ VI marks its camp as suppressing a resource, but that is not transferable
gameplay authority. The Civ III renderer already treats camp and resource as
separate map objects, so the default preserves authoritative Civ III resource
visibility and composes the camp around it. A visual gate must exercise a
same-tile resource witness before promotion.

## Night and effects

All camp bodies use the shared environment lighting. The primitive default may
optionally add one small pack-authored flame, restrained warm local light, and
smoke. The flame and light must activate together, remain tile-local, and tick
only while visible. Documentation and manifests must label this as authored
behavior. The optional industrial profile may use its exact lit fire-barrel
children, subject to the same visual calibration.

Barbarian Clans interaction effects are not imported. Civ III has no matching
incite, ransom, bribe, raid, or hire states. Camp dispersal may eventually bind
an approved generic destruction effect to the authoritative native removal
event, but importing a source effect must not invent a new gameplay state.

## Promotion checklist

A dedicated modular Lab v2 gate should cover at least:

- visible and viewer-hidden camps, fog transitions, capture/removal, save/reload,
  scrolling, wrap duplicates, and cache stability;
- primitive default at all four Civ III eras, plus an isolated optional
  industrial-pack witness;
- both zooms, all eight view facings, slopes and mixed terrain, city/route/forest
  clearance, same-tile resource coexistence, and nearby independent units;
- deterministic optional-child density and stable entrance rotation;
- noon/night shared lighting, authored flame/light matched controls, and no
  emissive or animation work while hidden;
- configuration-off native parity and configuration-on fail-closed ownership
  only after the family is approved.

## Repeatable commands

```bash
python3 Renderer/tools/asset_compiler/barbarian_camp_source_probe.py
python3 Renderer/tools/asset_compiler/tile_object_asset_importer.py
python3 Renderer/tools/asset_compiler/build_tile_object_runtime.py
python3 -m unittest Renderer.tools.asset_compiler.test_tile_object_asset_importer
```

The source probe writes only ignored local evidence. No new or changed
`civ_prog_objects.csv` entry is required by this preparation.
