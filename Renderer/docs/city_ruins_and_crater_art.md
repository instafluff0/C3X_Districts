# Crater and ruined-city art candidates

The current local `FutureGateCandidates` pack already contains four normalized
`CLUTTER_CRATERBLASTS` decals under `infrastructure/crater/variant_01..04`.
They are ground-damage art candidates for Civ III's authoritative crater tile
flag. An impact flash alone must not create a persistent crater.

The installed Civ VI Base `landmarks/tilebases.blp` package references
`Ruin_Debris_Decal`, its auxiliary `H` texture, and a lighter ground variant.
The color texture visibly contains broken masonry on dark soil. The source
package also names rubble components, but those are internal parts rather than
standalone landmark roots accepted by the current compound importer. Package
evidence does not prove that Civ VI uses this decal for a razed city.

`city_ruin_texture_sets.json` selects those three Base textures, and
`city_ruin_texture_importer.py` converts them to generic `city/ruins/*` DDS
assets in the ignored local `CityRuinsCandidates` pack:

```sh
python3 Renderer/tools/asset_compiler/city_ruin_texture_importer.py
```

This is a texture-only visual candidate, with no raze event capture, lifetime,
placement, or runtime binding. A future city-ruin presentation must use Civ III
state or a captured authoritative raze event to decide when it appears and
disappears. Derived Civ VI pixels stay local and are not redistributed.
