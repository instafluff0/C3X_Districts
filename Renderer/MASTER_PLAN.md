# C3X Custom Renderer Master Plan

## Mission

Build an optional, scenario-aware 3D renderer for the Civilization III map while preserving Civ III and C3X as the authority for game state, configuration, time, seasons, screen coordinates, overlays, and UI.

The renderer is a new presentation path, not a new game engine. With custom rendering disabled, behavior and visuals must remain unchanged. With it enabled, any unsupported or unmapped object must be able to fall back to Civ III's normal 2D draw path.

## Product Goals

1. Let modders replace any map-rendered category with source-agnostic 3D assets: terrain, terrain features, roads, railroads, rivers, tile improvements, resources, cities, units, effects, and map decorations.
2. Let scenarios provide their own renderer definitions and art through Civ III's normal scenario search folders.
3. Preserve C3X user overrides and its existing `default -> scenario -> custom` configuration model.
4. Reuse C3X's authoritative day/night hour and season state. Use native 3D lighting and material changes where possible, while retaining current 2D day/night and seasonal art for categories still drawn by Civ III.
5. Keep runtime packs independent of Civ VI, Civ VII, Blender, or any particular source format. Source-specific conversion belongs only in offline tools.
6. Render the complete visible 3D scene off-screen, then insert a map-sized image into Civ III before Civ III draws retained overlays and UI.
7. Make every phase independently testable and keep the game usable throughout development.
8. Let alternate visual skins and mod source trees compile into separate, interchangeable packs with stable logical asset IDs.
9. Export replayable scenes from loaded saves/BIQs and evaluate rendered fixture matrices with deterministic metrics plus human and AI-assisted visual review.

## Non-Goals

- Replacing Civ III's window, input loop, game rules, UI, advisors, or presentation mechanism.
- Presenting a second D3D swap chain over Civ III's window.
- Redistributing Firaxis assets with C3X.
- Requiring every scenario to replace every visual category.
- Reconstructing gameplay state from pixels when structured Civ III state is available.

## Architectural Boundaries

The project has five explicit layers:

1. **Source importers** read Civ VI cooked assets, loose FGX, Blender files, or other authoring formats. They emit normalized C3X packs.
2. **C3X asset packs** contain models, textures, materials, animations, variants, and source-agnostic metadata.
3. **Renderer definitions** map Civ III/C3X metadata to pack asset IDs. These are layered text files and may be scenario-specific.
4. **Visible scene capture** converts authoritative Civ III map state and draw anchors into a compact source-agnostic scene.
5. **Off-screen renderer and bridge** render that scene into a map-sized target and copy it into Civ III's map surface at a controlled boundary.

No runtime code below layer 1 may branch on `civ6`, `fgx`, `blp`, or another source format.

Development is organized into two workstreams. The standalone **Renderer Lab**
owns asset use, rendering technique, and visual quality for every map-rendered
category—not only terrain. **Game Integration** owns authoritative Civ III
capture, caching/invalidation, redraws, compositing, and native draw ownership.
Every visual gate has a same-numbered delivery gate (`L#` -> `I#`). An approved
lab system is promoted through a versioned handoff and then integrated without
independently redesigning its look. Configuration off is fully vanilla;
configuration on owns the complete `m19` map plane and never replays native
terrain after a partial or failed custom frame. See
[docs/renderer_workstreams.md](docs/renderer_workstreams.md).

After the complete L9-L21 Lab v1 stack was frozen, the user authorized a
separate Renderer Lab v2 quality campaign before the remaining Integration
gates. Lab v2 preserves every historical handoff. The user subsequently assigned a sole lead implementer and visual integrator
to the complete current Lab v2 path. Q0-Q8 remain requirement categories and
historical namespaces; exclusive ownership and cross-owner start blockers are
superseded. Preserve their candidates and iterate on one combined gameplay view.
Windows D3D11 remains the promotion-parity target and Civ III
Integration remains Windows-only. See
[docs/renderer_lab_v2.md](docs/renderer_lab_v2.md).

## End-State Import And Mapping Workflow

The intended authoring experience is a single offline importer command once a source adapter and mapping profile exist. Conceptually:

```powershell
py Renderer\tools\import_civ6.py `
  --civ6-assets "Z:\...\Civ6.app\Contents\Assets" `
  --mapping Renderer\mappings\civ3_conquests_to_civ6.txt `
  --output Renderer\packs\Civ6Conquests
```

That command should discover and extract source art, normalize models/textures/materials/animations, apply authored semantic mappings, build a source-agnostic C3X pack, generate previews, and validate every reference. It runs offline when creating a pack, not once per game tile.

The mapping profile is necessary because Civ VI does not know the meaning of Civ III sprite indices, culture groups, city sizes, terrain topology, or scenario names. Import tooling may suggest mappings from metadata and names, but it must preserve authored decisions and report ambiguity rather than silently guessing.

The target is close visual and semantic correspondence, not necessarily one source file for every Civ III sprite. One Civ III map location is a composition of independently resolved facts. For example:

```text
tile (x, y)
  terrain: grassland, sheet/sprite variant 5
  west neighbor: mountains, sheet/sprite variant 2
  city: culture index 4, medium size, current era
  environment: current C3X hour and season
```

The resolver may select a grassland material, a grass-to-mountain transition, a deterministic mountain model, a culture/era/size city model, seasonal variants, and native day/night lighting. These selections form one visible scene; they are not required to collapse into one monolithic asset.

Civ III's low-level selected-art metadata must always remain available to rules even when the 3D renderer handles the same topology procedurally. A terrain sprite index may encode adjacency or transition information rather than identify a literal texture. The mapping layer decides whether to use that index for an exact asset match, a procedural transition, a deterministic variant seed, or only diagnostics.

Success for the mature pipeline means:

- A pack author performs repetitive conversion and validation with one command.
- Scenario authors map by friendly names where possible and exact numeric/sprite metadata where necessary.
- The same normalized pack can be reused by multiple Civ III scenarios through different mapping profiles.
- Runtime selection is deterministic and explainable for every rendered object.
- Unsupported or ambiguous mappings fall back to Civ III art instead of producing an incorrect replacement.

The adapter boundary, stable logical IDs, alternate-skin equivalence reports, and output layout are specified in [docs/source_adapter_contract.md](docs/source_adapter_contract.md).

Vanilla Civ VI and an alternate environment skin are compiled as different pack roots from the same logical-ID contract. The preferred first alternate-skin target is the Civ V Environment Skin for Civ VI, subject to documented conversion permission. C3X chooses the active result through the normal layered `#Pack path`; the runtime never needs a Civ V/Civ VI-specific switch.

## Runtime Data Flow

```text
Civ III scenario + map state
          |
          v
C3X config/time/season state
          |
          v
visible scene capture ---- renderer definitions ---- C3X asset packs
          |                         |                       |
          +-------------------------+-----------------------+
                                    |
                                    v
                         off-screen D3D renderer
                                    |
                                    v
                         map/dynamic texture or bitmap
                                    |
                                    v
               Civ III-owned map or unit surface at its insertion point
                                    |
                                    v
                Civ III fog, borders, labels, selection, UI
```

The renderer has no independent game or presentation loop. Static map work enters through normal Civ III `Map_Renderer::m71` redraws; M7.4 unit work enters through Civ III's later Animator-owned unit update and dirty-region path. C3X captures one immutable snapshot and timestamp per applicable pass. When visible continuous animation needs another frame, an audited Civ III timer may only mark the animator dirty and request ordinary Civ III work. It may not render, blit, sleep, busy-wait, or pump messages from the timer callback. Static scenes request no continuous updates. The audited loop and two-plane architecture are in [docs/civ3_render_loop_viability.md](docs/civ3_render_loop_viability.md); the full clock, dirty-state, pause, frame-skipping, telemetry, and degradation contract is in [docs/runtime_animation_and_frame_pacing.md](docs/runtime_animation_and_frame_pacing.md).

## Configuration Model

The small C3X INI controls whether the feature is active and which profile is selected. The large art catalog belongs in dedicated renderer definition files described in [docs/renderer_config_spec.md](docs/renderer_config_spec.md).

Planned C3X settings:

```ini
custom_rendering_mode = off
custom_rendering_profile = default
custom_rendering_debug = false
```

Planned definition layers, from lowest to highest precedence:

1. `default.custom_rendering.txt` in the C3X folder.
2. `scenario.custom_rendering.txt` resolved through the active scenario search path.
3. `custom.custom_rendering.txt` in the C3X folder.

Later layers may replace a named definition or disable it. Relative scenario assets use Civ III's asset-path resolution. The exact merge and rule-selection semantics are part of the config contract and must be tested before runtime integration.

When `enable_custom_rendering` is true, C3X's legacy `enable_custom_animations` tile-FLC overlays are ignored globally. Renderer-owned resource and terrain animation comes from renderer definitions and packs; an unavailable renderer mapping falls back to Civ III's normal static art path, not to a legacy custom FLC overlay. The configured legacy-animation value is preserved and becomes effective again when custom rendering is disabled.

## Civ III Metadata Contract

The asset resolver must accept semantic game data and low-level art-selection data. A category may use whichever fields are stable and useful.

Common fields include:

- Map coordinate, screen anchor, viewport, zoom, and world seed.
- Base and real terrain type, landmark state, `Tile.SquareParts`, sheet index, sprite index, and adjacency/topology masks.
- River, road, railroad, irrigation, mine, goody-hut, colony, pollution, crater, and terrain-building state.
- Resource ID/name/class and the actual PCX index Civ III selected.
- City owner, culture group, era, size, walls, capital status, and selected sprite metadata.
- Unit type, owner/civilization, era, direction, action, animation frame, hit points, and the draw anchor selected by Civ III.
- Current C3X day/night hour and season.

Names are preferred in author-facing files; stable numeric IDs and draw metadata are retained in captured scenes for exact matching and diagnostics.

## Day/Night And Seasons

C3X remains the clock and calendar. The renderer receives:

- `hour` in the existing 0..23 range.
- `season` using the existing summer/fall/winter/spring enum.
- A deterministic transition fraction only after interpolation is intentionally added.

The first 3D implementation should derive sun direction, color, ambient light, exposure, shadows, water response, and emissive intensity from the hour. It should not require 24 copies of each texture. Seasons may select material parameters, texture sets, vegetation variants, snow coverage, or entirely different assets.

Ownership must be explicit per category:

- A 3D-owned category uses renderer lighting/material variation and suppresses the equivalent Civ III base sprite.
- A 2D-owned category keeps the existing C3X proxy/image day-night and seasonal path.
- Civ III overlays and UI are never globally color-graded by the renderer.

Required visual fixtures are noon, sunset, midnight, and sunrise for each enabled season. Time and season changes must invalidate the map render without changing game state.

The advanced shared environment contract, Civ VI source evidence, emissive/attachment ownership, and redraw policy are in [docs/environment_lighting_and_ambient_effects.md](docs/environment_lighting_and_ambient_effects.md) and [docs/civ6_lighting_findings.md](docs/civ6_lighting_findings.md).

Reference screenshots are art-direction targets, not pixel-equality baselines across different engines. The save/BIQ scene-export path, fixture matrices, deterministic metrics, day/night rubric, and human/AI review responsibilities are specified in [docs/visual_validation_plan.md](docs/visual_validation_plan.md).

## Milestones And Gates

### M0: Project Contract And Tooling

Deliverables:

- Standalone `Renderer/` workspace.
- Master plan, config contract, roadmap, status file, and status checker.
- Source-agnostic pack manifest and dependency-free isometric preview.
- Pinned references to known Civ VI conversion tools.

Gate: the full renderer verification suite passes from the C3X root.

### M1: Civ VI Grassland Import

Deliverables:

- Discover ArtDefs, BLP packages, and shared cooked assets.
- Decode standalone `CIVBIG` textures into valid DDS/PNG.
- Associate `ART_DEF_TERRAIN_MATERIAL_GRASSLAND` with its material texture roles using evidence from `CIVBLP`, not filename assumptions.
- Recover or source enough terrain geometry/UV information to display one correct grassland patch.
- Emit only normalized pack data and provenance into the C3X pack.

Gate: a deterministic standalone image uses extracted Civ VI grassland art with documented material/UV selection. It renders nonblank at two viewport sizes and no import code is needed by the preview/runtime.

### M2: Renderer Definition Parser And Resolver

Deliverables:

- Parser for the renderer text format with file/line diagnostics.
- Layer merge implementation for default, scenario, and custom files.
- Pack and asset ID resolution with path-safety checks.
- Deterministic rule matcher with priority, specificity, layer, and declaration-order tie breaks.
- Coverage for terrain plus representative resource, city, and unit selectors.

Gate: table-driven tests prove override precedence, disabling, fallback, time/season filters, and Civ III sheet/sprite matching.

### M3: Visible Scene Contract

Deliverables:

- Versioned scene schema independent of Civ III pointers and source assets.
- Captured fields for viewport, environment, tiles, and object instances.
- Deterministic IDs/variant seeds.
- Recorded scene fixtures that can be replayed outside Civ III.

Gate: fixture validation catches missing required fields and a scene round trip is stable. Resolver output can be inspected without launching the game.

### M4: Standalone 3D Renderer

Deliverables:

- Whole-viewport orthographic renderer using Civ III's pixel-defined basis.
- Terrain continuity, depth testing, directional light, and material support.
- Native day/night lighting and initial seasonal material/asset switching.
- Resize and device-loss handling appropriate to the chosen D3D version.
- After the base renderer gate, a batch fixture matrix with deterministic metrics and reference contact sheets.

Gate: recorded scenes produce nonblank deterministic reference images at two sizes; anchors align to expected Civ III tile centers; noon/night and summer/winter results differ as specified.

### M5: Civ III Capture And Off-Screen Bridge

Deliverables:

- Confirmed insertion point after map preparation and before retained overlays/UI.
- Visible scene capture from existing hooks, initially terrain-only.
- Map-sized render target, readback, and bounded copy into Civ III's map surface.
- Config-off path that does not initialize or call the renderer.
- Automatic game-assisted export of a populated viewport into the versioned visible-scene contract, with strict deterministic replay tests and lightweight in-game health screenshots. Formal matched fixture sets remain optional for regression diagnosis and release review.
- M5.3 integration with Civ III's redraw/timer loop: one monotonic timestamp per immutable frame, dirty-driven redraw requests only while visible animation is active, idle/static suppression, deterministic frame skipping, pause/reset behavior, and bounded frame-time telemetry.

Bridge gate: `TEST_INJECTED_CODE_COMPILE.bat` passes; config-off screenshots match baseline; config-on terrain aligns while fog, borders, labels, highlights, and UI remain above it. The exporter passes strict synthetic coverage and deterministic replay gates; ordinary development does not require the user to administer paired fixture files.

M5.3 gate: normal Civ III map draws remain the only render entry point; static maps request no continuous redraws; animated scenes do not create reentrant draws, queued catch-up frames, a second loop, `Sleep`, or busy waiting; skipped frames and pauses do not alter simulation or gameplay animation timing.

### M6: Terrain Productionization

Deliverables:

- **M6.0 inventory prerequisite:** a generated, tested ledger of every vanilla Civ III map-visible art selection and every retained compositing layer. It combines layered Base/PTW/Conquests/scenario file discovery with BIQ semantics, atlas cell geometry, annotated index contact sheets, and runtime draw-census evidence. C3X natural wonders and Districts are deferred to M9 and M11. See [docs/vanilla_art_inventory.md](docs/vanilla_art_inventory.md).
- Grassland, plains, desert, tundra, coast, sea, ocean, hills, mountains, and terrain transitions.
- Deterministic variants, continuous geometry, water, forests/jungles/marsh, and landmark handling.
- Streaming/cache budgets and useful missing-asset diagnostics.
- First load the proven normalized grassland pack through the native off-screen renderer, then expand the same definition-driven path across the remaining terrain families.
- Keep the tracked per-family runtime coverage explicit: a family with incomplete geometry, materials, or selection dependencies stays an atomic Civ III fallback rather than receiving a procedural or semantically false stand-in. See [docs/definition_driven_terrain.md](docs/definition_driven_terrain.md).
- After real-terrain expansion, a shared M6.4 environment foundation for sun/moon curves, exposure, water specular/Fresnel response, emissive materials, analytic lights, and generic ambient-effect attachments. It consumes C3X time/season and remains source-independent.
- **M6.5 connected base materials:** remove per-tile checkerboarding, blend adjacent mapped terrain materials symmetrically, and feather mapped/fallback boundaries into a complete verified Civ III base underlay.
- **M6.6 complete vanilla terrain scene:** replace the flat-diamond bootstrap with connected relief, water, vegetation/clutter, transitions, polar ice, landmark handling, and all fourteen vanilla terrain families at both Civ III zoom levels. Partial families continue to fall back atomically until their full geometry/material dependency set is ready.
- **M6.7 approved-terrain game integration:** establish the shared delivery boundary, then track each approved visual system through the same-numbered I# gate. I9-I11 consume terrain, dunes, and marshes with faithful in-game placement, caching/invalidation, redraw behavior, scrolling, wrapping, both zooms, retained overlays, exclusive native suppression, and hard failure without native replay. Art-direction changes belong back in the lab.
- **M6.8 alternate visual skin:** after its own Renderer Lab pack/skin approval, compile a second interchangeable terrain skin to its own pack directory through the same source-adapter contract, with a logical-ID equivalence report against the baseline pack. Prefer the Civ V Environment Skin for Civ VI if permission is obtained; otherwise use a synthetic or permissioned skin to prove the contract.

M6.0 gate: every BIQ-defined or runtime-reachable visual is classified as mapped, vanilla fallback, not map-rendered, or unreachable; strict deterministic inventory generation has no unknown atlas layouts, semantic bindings, indices, or layer-ownership decisions. File and directory counts alone never satisfy this gate.

M6 production gate: representative maps and seams pass visual fixtures across scrolling, wrapping, zoom, time, and seasons.

M6.4 environment gate: deterministic noon/sunset/midnight/sunrise fixtures prove bounded moonlight on water, static night emissives, one attached animated light/effect pair, idle suppression, skipped-frame determinism, and retained Civ III overlays untouched. Civ VI extraction evidence must label confirmed ArtDef/package data separately from inferred attachment or engine behavior.

M6.5 gate: an executable native fixture proves that mapped/mapped boundaries converge to the same mixed edge from both tiles, mapped/fallback boundaries expose a feathered transparent edge over the complete Civ III base underlay, and repeated renders are deterministic without coordinate-seeded tile brightness blocks.

M6.6 gate: executable two-size mixed-terrain fixtures cover every vanilla base, relief, water, vegetation, transition, ice, and landmark family with explicit complete-replacement or atomic-fallback ownership. Flat material diamonds are not accepted as completed relief or feature terrain.

M6.7/I9-I11 gate: game output faithfully consumes the approved lab handoffs within documented integration tolerances; cache and invalidation traces cover camera, zoom, wrap, environment, pack/definition/handoff revision, and ownership changes; production output has zero native fallback; custom-on load/capture/render/validation/blit/device/reentrant failures never replay native map terrain; and retained post-`m19` overlays remain correctly anchored. One optional batched in-game checkpoint follows automated fixtures.

### M7: Map-Object Game Integration

M7 uses same-numbered per-system promotion rather than waiting for the complete L21 scene: L12 feeds I12, L13 feeds I13, and so on through L21/I21. Each map-object family becomes eligible only after its matching Renderer Lab gate and required visual approval. Integration consumes the frozen lab pack/definition/scene/reference handoff, then proves live capture, cache invalidation, redraw, anchoring, compositing, exclusive suppression, retained layers, and visible hard failure without native replay or redesigning the look. M7 retains the broader architectural grouping below, while I# is the canonical execution sequence.

1. **M7.1 Infrastructure and tile-bound overlays:** port the approved rivers, roads, railroads, irrigation/farms, mines, goody huts, colonies, fortresses, barricades, airfields, outposts, radar towers, pollution, craters, and victory locations into the source-agnostic pack/definition/scene and native compositing paths. Rivers, roads, and rails require deterministic connection/topology masks, wrap-stable neighbor selection, grade/crossing behavior, and terrain-conforming geometry. Farms and mines require era/terrain-aware variants, believable footprint blending, relief-aware placement, and preservation of gameplay ownership in Civ III. Goody huts preserve viewer-conditioned visibility and stable variation; colonies preserve authoritative owner/era, extraterritorial ownership, and same-tile resource readability. L19B is frozen and approved with source-backed raised infrastructure plus static feathered pollution/crater state distinct from transient M7.5 effects; I19B remains the separate runtime-ownership gate. Each family receives its own exclusive-ownership and hard-failure gate. See [docs/remaining_tile_infrastructure.md](docs/remaining_tile_infrastructure.md).
2. **M7.2 Resources:** strategic, luxury, and bonus map-resource bodies using player-specific visibility, resource ID/name/class, selected PCX index, anchor/calibration data, optional ambient animation, and fog ownership. Civ III retains Civilopedia, city-screen, trade-network, advisor, diplomacy, notification, and every other non-map resource icon. The editable 26-resource seed is documented in [docs/civ3_to_civ6_resource_mapping.md](docs/civ3_to_civ6_resource_mapping.md).
3. **M7.3 Cities:** use a lab matrix that first proves one static city, then expands across civilization/owner identity, culture group, era, and size before adding walls, capital status, city style, and visible buildings/wonders. Calibration must cover footprint, terrain grounding, occlusion envelope, and the city-window/lamp emissive groups and attachments. The production handoff preserves Civ III labels, population displays, fog, borders, and UI above the 3D city; unsupported custom-on combinations fail visibly without replaying the native city body.
4. **M7.4 Units:** begin in the lab with one static, owner-tinted unit and eight-direction turntable, then prove source-independent skeleton/clip intake, absolute-time idle/movement playback, authoritative anchor interpolation, and finally combat/victory/death/interruption fixtures. Production capture adds unit type and owner/civilization, stable unit/action/event IDs, current clip, fortification, stacking/display order, hit points, and visibility. Armies are one composite parent: the exact member selected by Civ III is resolved through the ordinary arbitrary-unit path beside a dedicated era-profiled Great General commander; the loaded roster is not baked or all rendered, and the parent HUD appears once. Initial replacement ownership applies only to the vanilla unit body sprite. Civ III continues to draw the selection cursor/ring, green health bar, left-side activity/status marks, stack indicators, and related unit HUD above and around the 3D body. Those native overlays remain unlit and are not transferred to `augment` or `replace` ownership during M7.4; changing that policy later requires an explicit follow-up milestone and gate.

The source-backed I20 execution handoff is
[`docs/i20_native_unit_animation_handoff.md`](docs/i20_native_unit_animation_handoff.md).
It preserves Civ III as the movement/combat/action director, keeps the native
FLC state machine advancing invisibly for gameplay waits, and narrows new patch
research to the body-only draw boundary.
The Army-specific composite, source-art, and member-selection decisions are in
[`docs/army_rendering_strategy.md`](docs/army_rendering_strategy.md).
5. **M7.5 Effects and transient animation:** particles, projectiles, impacts, combat effects, attached flames/smoke/steam/beacons, blending, stable event IDs, source/target anchors, spawn time, duration/progress, interruption, lifetime/cleanup, and deterministic replay. Effects follow Civ III outcomes and timing rather than inferring either from rendered frames.

Every integrated category has explicit custom ownership. Before its I# gate it is absent from the custom-on `m19` plane; after its I# gate it must render custom art or fail the frame visibly. It never silently substitutes Civ III map art while custom rendering is enabled.

Fog of war, unexplored shroud, territory borders, grid, selection and movement indicators, city labels, unit status, cursor, minimap, HUD, and editor-only markers are also first-class inventory responsibilities. They are not implicitly part of M7 merely because most begin as Civ III-owned layers; changing ownership later requires its own scene, layering, fallback, and visual gate.

Gate per substep: scene-contract coverage, metadata/rule mapping tests, representative source-independent assets, standalone fixture, calibration/reference images, day/night and seasonal behavior, in-game layering, missing-asset hard failure, and config-off regression. Unit and effect gates additionally require timestamped lifecycle, interruption, slow/skipped-frame, gameplay-timing-neutrality, and renderer-reset tests. M7.4 also tests movement endpoints, stack selection, and native selection/health/status HUD retention and Z-order across movement, scrolling, clipping, combat, and failure.

The complete lab ladder and promotion rules are maintained in [terrain_lab/PLAN.md](terrain_lab/PLAN.md). Lab success never transfers rendering ownership by itself. Each family becomes integrated only after its same-numbered I# scene-capture, cache/invalidation, exclusive-suppression, compositing, telemetry, and hard-failure gate passes. Natural wonders, constructed wonders, and districts remain deferred to M9, M10, and M11.

### M8: Authoring And Validation

Deliverables:

- Calibration mode with original-sprite ghost, anchor, envelope, scale, and offsets.
- Pack validator and preview generator.
- Seasonal material/texture authoring automation that preserves UVs, atlas layout, alpha semantics, and logical asset IDs.
- Batch visual review tooling that records deterministic metrics, reference metadata, AI observations/confidence, and human acceptance decisions.
- Modder documentation and legal/source provenance checks.
- Performance budgets, compatibility matrix, and graceful error handling.

Gate: a scenario can ship its own renderer file and pack without modifying C3X; missing assets fail validation clearly without native substitution; distributable C3X contains no Firaxis art.

### M9: Natural Wonders

Deliverables:

- An effective inventory of default, user, scenario, and dynamic C3X natural-wonder definitions plus every placed instance, using C3X's authoritative natural-wonder ID and tile anchor.
- A swappable mapping seed from C3X natural-wonder names to Civ VI Base/DLC Features, TerrainStyle NaturalWonders, Clutter, VFX, water, and cooked source records, with exact/approximate/unavailable/authored confidence.
- Source-agnostic natural-wonder kits combining terrain-integrated geometry, directional pieces, clutter, water, emissives/lights, and ambient effects while retaining C3X's one-tile gameplay identity.
- Exclusive custom body/animation ownership with visible hard failure, while preserving the native natural-wonder name label, fog, borders, minimap, HUD, and UI.

Gate: every effective natural-wonder definition and placed instance is classified; ordinary landmark terrain is never misclassified; representative land, mountain, geothermal/volcanic, coastal, and water fixtures pass environment, animation, source-multipart, layering, performance, and custom-on hard-failure tests. See [docs/natural_wonder_rendering.md](docs/natural_wonder_rendering.md).

### M10: Constructed Wonders

Deliverables:

- A generated inventory of every loaded BIQ Great and Small Wonder plus every effective default/user/scenario C3X wonder-district definition.
- A swappable Civ III-to-source mapping seed, beginning with Civ VI Wonder and Building ArtDefs, with exact, approximate, unavailable, and mapless classifications.
- Source-agnostic construction and completed models with alternate orientation, placement, calibration, destruction/abandonment state, emissive groups, ambient-effect attachments, day/night, seasons, and exclusive custom ownership.
- A scene contract that never invents a map object for a wonder without a C3X Wonder District tile.

Gate: every wonder is classified; every configured map-visible wonder has deterministic construction/completed fixtures and correct orientation/layering; config-off preserves existing C3X rendering, while custom-on missing assets fail visibly without native substitution.

### M11: Districts (Final Milestone)

Deliverables:

- An effective inventory of all built-in, dynamic, user, and scenario district definitions and every runtime-visible render state.
- A swappable mapping from C3X district/building identities to source-agnostic district kits, seeded from Civ VI District, Building, Improvement, Route, and Landmark ArtDefs where useful.
- Composite base-plus-attachment rendering for `by-building`, deterministic stage presets for `by-count`, component-owned lights/emissives/ambient effects, and exclusive per-instance custom ownership.
- Dedicated topology handling for Bridge, Canal, Great Wall, Port/coast alignment, Wonder District relationships, construction, pillage/destruction, abandonment, culture, era, and shared-district building additions.

Gate: every effective district definition and state is classified; scenario override, composite-building, count-stage, topology, environment, layering, custom-on hard failure, and config-off fixtures pass. M11 completes the currently planned renderer scope. See [docs/wonder_and_district_rendering.md](docs/wonder_and_district_rendering.md).

## Testing Policy

Every implementation step must add the smallest test that proves its new contract. Test layers are:

1. Unit tests for binary readers, config parsing, matching, math, and validation.
2. Golden-data tests for imported metadata and recorded visible scenes.
3. Image tests for nonblank output, anchor positions, bounds, deterministic variants, and environment differences.
4. Fixture-matrix tests across packs, viewports, zoom, hours, and seasons, with deterministic machine gates separated from qualitative review.
5. Injected-code compilation via `TEST_INJECTED_CODE_COMPILE.bat` whenever `C3X.h` or `injected_code.c` changes.
6. In-game save/BIQ export and visual fixtures only when the milestone reaches Civ III integration.
7. Timestamped animation traces that produce identical poses independently of update rate, plus idle/redraw, interruption, frame-skip, pause/reset, and gameplay-timing tests.

AI review may triage contact sheets and report likely seams, clipping, scale, readability, or style problems, but it is not the sole release gate. Human art-direction approval and reproducible structural tests remain authoritative.

Manual in-game evidence follows the interaction budget in [docs/visual_validation_plan.md](docs/visual_validation_plan.md): automated and replayable evidence comes first, still-valid captures are reused, and any user request is batched at a strategic milestone checkpoint rather than repeated for each implementation iteration. A pending manual checkpoint does not block unrelated engineering work and is never treated as implicit approval.

Tests must not require redistributed Civ VI assets. Local integration commands may use the installed game, but committed tests use synthetic fixtures.

Routine verification is track-specific. Use `python3 Renderer/tools/renderer_dev.py state` before ordinary work, `renderer_dev.py lab` or `renderer_dev.py integration` during iteration, and `renderer_dev.py full` only when closing a production step/milestone, changing a shared contract, or preparing a strategic checkpoint. On macOS the latter three commands dispatch to the documented `Windows 11` Parallels VM automatically. The workflow builds each native target once and avoids recompiling injected C when `C3X.h` and `injected_code.c` are unchanged.

## Agent Handoff Protocol

An agent asked to "do the next step" must:

1. Read `AGENTS.md`, this file, `ROADMAP.md`, `VERIFICATION.md`, and `project_status.json`.
2. Run `python3 Renderer/tools/renderer_dev.py state`, then use the one-command `lab` or `integration` workflow named by the step. The workflow dispatches Windows-only work to the documented VM. Do not manually reproduce its script chain.
3. Implement only the `next_step` marked `ready`, unless the user redirects scope.
4. Preserve config-off behavior and the architectural boundaries above.
5. Add tests, evidence, and named executable verification gates for the step's acceptance criteria.
6. Audit `docs/civ3_patch_dependency_ledger.md`. Reuse existing patch points where possible; if a new `civ_prog_objects.csv` entry is proven necessary, add an exact `required_user_action` record to the ledger and `project_status.json` and tell the user in the same turn. Never request a speculative symbol/address.
7. Mark the completed step `complete`, select exactly one new `ready` step, and update `ROADMAP.md` plus relevant findings.
8. Run `python3 Renderer/tools/renderer_dev.py full` when closing the step. The integration workflow and full verifier run `TEST_INJECTED_CODE_COMPILE.bat` when injected C changed; use `--with-injected` to force it when necessary.

`project_status.json` is the canonical machine-readable pointer. `ROADMAP.md` is its human-readable explanation. They must agree.
