# C3X Visible Scene Contract v0

## Purpose And Boundary

`c3x.visible_scene.v0` is the source-independent handoff between future Civ III capture and the standalone renderer. A scene records what is visible and where Civ III placed it; it does not contain a live process pointer, a guessed camera, a selected source asset, or renderer API state.

The M5.2 native exporter writes this contract directly from the bounded M5 capture records. Its terrain metadata also preserves numeric `visibility_mask`, `fog_status`, and `territory_owner_id` context. Object records may preserve numeric city, tile-building, unit-owner, unit-state, unit-damage, and unit-direction fields alongside author-facing names. These are values copied from authoritative game state, never addresses or handles.

The canonical fixture is `samples/scenes/grassland_viewport.scene.json`. Validate or canonicalize it from the C3X root with:

```powershell
py -m Renderer.scenes.scene_contract Renderer\samples\scenes\grassland_viewport.scene.json
```

The command emits compact JSON with sorted object keys and one trailing newline. Parsing that output and serializing it again must produce identical UTF-8 bytes.

## Top-Level Record

Every field is required and unknown fields are rejected:

- `schema`: exactly `c3x.visible_scene.v0`.
- `scene_id`: a deterministic ID derived from world seed, scroll position, viewport size, hour, and season.
- `profile_id`: the case-sensitive M2 renderer profile ID used during replay.
- `world`: integer seed and tile dimensions plus independent X/Y wrapping flags.
- `viewport`: output dimensions, the map rectangle inside that output, and Civ III's captured pixel scroll position.
- `projection`: an explicit pixel basis rooted at a captured map coordinate and screen point.
- `environment`: renderer environment ID plus authoritative C3X hour and season.
- `tiles`: visible tile records with captured base-terrain metadata.
- `instances`: independently anchored non-terrain records such as resources, cities, and units.

The map rectangle must fit inside the viewport. Tile coordinates must fit the recorded world dimensions. A scene contains at least one tile; its object list may be empty.

## Pixel-Defined Isometric Projection

The only v0 projection type is `civ3-isometric-pixel`. It records:

- `origin_tile` and `origin_px`: one authoritative tile-to-screen correspondence.
- `tile_x_basis_px`: positive X and Y, mapping increasing tile X down/right.
- `tile_y_basis_px`: negative X and positive Y, mapping increasing tile Y down/left.
- `elevation_basis_px`: zero X and negative Y, mapping elevation upward.

For a tile `(x, y)`, the captured anchor must equal:

```text
origin_px
  + (x - origin_tile.x) * tile_x_basis_px
  + (y - origin_tile.y) * tile_y_basis_px
```

The validator checks this equality. M4 can therefore consume Civ III's actual pixel basis directly instead of reconstructing a camera angle.

## Tiles And Instances

A tile contains `id`, `map_x`, `map_y`, `anchor_px`, and a `terrain` renderable. Terrain uses the tile anchor. An object instance contains `id`, `category`, `tile_id`, `ordinal`, its own authoritative `anchor_px`, `variant_seed`, and `resolver_input`.

IDs are deterministic and validated:

```text
tile:<map_x>:<map_y>
terrain:<map_x>:<map_y>:0
<category>:<map_x>:<map_y>:<ordinal>
```

Ordinals start at zero and are contiguous per category and tile. IDs and coordinates must agree with the referenced tile. The 64-bit `variant_seed` is the first eight SHA-256 bytes of a fixed encoding of world seed, instance ID, map X, and map Y. This scene seed is available to future pack selection and procedural work; M2 may additionally produce a rule-specific coordinate-hash seed after a rule wins.

`resolver_input` contains only the typed M2 captured metadata vocabulary. `category`, `map_x`, and `map_y` are required. The offline inspector adds the scene's hour and season before calling the M2 resolver, ensuring every input needed for deterministic replay is present without duplicating environment state in every record.

## Source Independence And Validation

The validator rejects:

- Missing required or unknown fields, with JSON-style field paths.
- Duplicate JSON object keys.
- Wrong primitive types, invalid hours/seasons/categories, unresolved tile references, duplicate IDs, and noncontiguous ordinals.
- IDs, coordinates, variant seeds, or terrain anchors that disagree with their deterministic derivation.
- Process-specific pointer/address/handle fields.
- Civ VI, ArtDef, BLP, FGX, or installed Steam-source markers anywhere in a runtime scene.

Civ III metadata such as a selected PCX filename or sprite index is intentionally retained because it is a resolver input, not a renderer asset dependency.

## Planned Animated Scene Version

The strict v0 fixture records static unit selection inputs but is not the final M7.4 animation wire format. The animated scene version must add one frame-level monotonic presentation timestamp and stable per-unit/per-effect event records without weakening v0 validation.

Unit records will include stable unit/action/event IDs, current tile and authoritative anchor, movement start/end coordinates and anchors, path segment/progress, eight-direction facing, clip/action state, start time, duration, playback rate, normalized progress, loop/completion policy, target anchor, stack/display order, visibility, and current/active/selected flags. The selected-unit underlay is a separate instance linked to its target unit so its depth and anchor can be tested independently, but it and the native health/activity/status HUD remain Civ III-owned during M7.4; only the unit body is eligible for replacement.

Effect records will include stable event ID, source/target identities and anchors, spawn time, duration/progress, outcome, interruption, and cleanup state. Replay fixtures use explicit timestamps; neither schema derives progress from frame count or wall-clock time. Exact requirements and Civ III loop ownership are in `runtime_animation_and_frame_pacing.md`.

M7.2 resource instances carry the authoritative BIQ resource ID/name/class, selected map PCX index, tile/pixel anchor, terrain context, and player-specific visibility/fog result. These records describe only map presentation. No Civilopedia, city-screen, trade-network, advisor, diplomacy, notification, or other non-map icon ownership is transferred to the renderer.

M9/M10/M11 scene versions add natural-wonder, constructed-wonder, and district records without weakening source independence. Natural wonders carry stable C3X natural-wonder/instance identity, authoritative tile/pixel anchor, required terrain and adjacency/direction, native sprite/animation fallback metadata, visibility, and retained-label eligibility; Civ VI multipart/footprint conventions never enter runtime scenes. Constructed wonders carry stable BIQ improvement identity, Great/Small class, authoritative C3X Wonder District placement, construction/completion and orientation state. Districts carry stable instance/type identity, `render_strategy`, effective dependent-building IDs/count, construction/damage/abandonment state, culture/era, coastline orientation, and topology masks. A `by-building` district exposes its effective building set as semantic attachment inputs; a `by-count` district exposes the selected count stage. Exact composition and fallback rules are in `natural_wonder_rendering.md` and `wonder_and_district_rendering.md`.

## Offline Resolver Inspection

`inspect_scene` validates the scene, adds its environment state to every renderable's captured metadata, and sends each item through `definitions/rule_resolver.py`. Its `c3x.visible_scene_resolution.v0` output contains the source record path, stable instance ID, authoritative anchor, scene variant seed, complete resolver input, and the full M2 winner/loser or fallback diagnostic.

The command-line form accepts a merged M2 catalog JSON:

```powershell
py -m Renderer.scenes.scene_contract scene.json --catalog catalog.json
```

This path reads no Civ III process state and performs no model or texture payload loading. `--config-off` proves the replay path retains M2's no-access fallback behavior.
