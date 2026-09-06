# C3X Renderer Verification Contract

## Purpose

`project_status.json` says which work is complete and what comes next. Completion is valid only when it is backed by executable verification, not merely by prose or an existing file.

Run the fast state contract from the C3X root before ordinary renderer work:

```text
python3 Renderer/tools/renderer_dev.py state
```

Then use one track command during iteration:

```text
python3 Renderer/tools/renderer_dev.py lab
python3 Renderer/tools/renderer_dev.py integration
```

These commands replace manual script chains. They run focused contracts, build
the relevant native target once, write an ignored iteration report, and stop on
the first failed phase. Integration recompiles injected C only when `C3X.h` or
`injected_code.c` changed; `--with-injected` forces the smoke test.

On macOS the track commands dispatch automatically through `prlctl` to the
running Parallels VM named `Windows 11`. The one canonical Git working tree is
visible there at `Y:\fun\Civilization III Complete\Conquests\C3X_Districts`.
The installed
`C:\\...\\Conquests\\C3X_Districts` path is a Windows directory link to that
same checkout through the stable `\\Mac\Home\...` Parallels share, so Git,
builds, installation, and game loading all see the same files. Override the
defaults with `C3X_RENDERER_VM`, `C3X_RENDERER_WINDOWS_ROOT`,
`C3X_RENDERER_WINDOWS_LIVE_TARGET`, and `C3X_RENDERER_CIV3_CONQUESTS` if the VM,
share, link target, or install moves. On Windows the same commands run directly.

For injected compilation, the workflow creates/reuses a
`C3X_Shared_Verify` directory link beneath the VM's installed GOG `Conquests`
folder. The link points to the `Y:` checkout, allowing `ep.c` to see the installed
executable while compiling the current shared sources. Override the install
folder with `C3X_RENDERER_CIV3_CONQUESTS` when needed.

After a successful `integration` or `full` workflow, the live-check phase
verifies that the installed `C3X_Districts` path still links to the canonical
`Y:` checkout and that its Git metadata is accessible. There is no deployment
copy and therefore no second working tree that can become stale. Ignored custom
configuration and generated renderer packs live in the canonical checkout. The
user then runs `INSTALL.bat` interactively from the familiar installed path so
its success/error dialog is visible and no hidden `temp.exe` process is created
by automation.

Those workflows also render the approved L13 192-tile BIQ fixture and its
topology-only halo through the freshly built production DLL to ignored
`native_i13a_noon.bmp` and `native_i13a_sunset_zoom2.bmp` outputs. These are
replays of the game renderer, not the Lab executable: together they exercise all
terrain identities, authoritative river masks/topology, both Civ III zooms,
shared lighting phases, exact semantic shader inputs, the ordered terrain/river/
shadow/feature passes, and zero fallback before the interactive install.

The native build must also copy the verified DLL into `Renderer/bin`, the exact
path C3X loads. A sharing violation now fails the workflow even if an older DLL
already exists there; close Civ III and rerun the workflow before `INSTALL.bat`.
This prevents a passing build report from leaving the game on stale renderer
code.

Run the full historical suite when closing a production step/milestone, changing
a shared contract, or preparing a strategic checkpoint:

```text
python3 Renderer/tools/renderer_dev.py full
```

The full command validates every source-independent gate named by completed
milestones and steps, then runs one consolidated native build/smoke and one
injected compile on the Windows VM. It writes reports under
`Renderer/verification/`. Licensed local-asset probes are run when the current
lab or integration step names them rather than being repeated on every full
historical pass. Direct legacy verification still memoizes repeated native
builds and injected compiles within one process.

The underlying verifier remains directly available for diagnosis:

```powershell
py Renderer\tools\verify_project.py --require-local-assets
```

## Completion Rule

A milestone or step marked `complete` must contain:

- `evidence`: tracked files that implement or document the result.
- `verification`: one or more portable executable check IDs.
- Optional `local_verification`: integration checks requiring locally installed copyrighted assets or external software.

The status checker rejects complete items with no portable verification. The verification runner rejects unknown check IDs and fails when any required portable check fails. Local checks pass when prerequisites are available, otherwise they are skipped unless `--require-local-assets` is supplied.

The status checker also requires the Civ III patch-dependency ledger and a machine-readable dependency block. Any `required_user_action` request must contain the milestone/step, symbol, reason, patch capability, C signature, supported-build addresses, fallback, and verification it unlocks. Audit candidates with unknown necessity or addresses are not user requests.

An agent completing a step must add or extend a verification function when existing checks do not prove the new acceptance criteria. Reusing a broad unit-test gate is allowed only when the tests actually exercise the completed capability.

## Verification Levels

1. **Project contract**: required documents, exactly one ready next step, roadmap/status agreement, completion evidence, and gate declarations.
2. **Unit tests**: binary readers, configuration contracts, matching, scene math, and other source-independent behavior.
3. **Capability smoke tests**: direct end-to-end checks such as synthetic CIVBIG-to-DDS conversion or deterministic rendering at multiple sizes.
4. **Local asset integration**: installed Civ VI assets and pinned conversion tools. These never become requirements for redistributed tests.
5. **Injected-code compile**: `TEST_INJECTED_CODE_COMPILE.bat` after `C3X.h` or `injected_code.c` changes.
6. **In-game visual gates**: recorded/manual fixtures only at documented strategic integration or milestone checkpoints. Agents exhaust automation and reuse still-valid evidence; they batch any necessary user request instead of requiring screenshots after each iteration.
7. **Cross-pack equivalence**: stable logical-ID coverage, explicit inheritance, and deterministic difference reports for alternate visual skins.
8. **Perceptual review**: versioned reference metadata, deterministic image metrics, AI observations/confidence, and recorded human decisions. Perceptual review supplements but never replaces structural gates.
9. **Frame pacing and animation lifecycle**: timestamped event traces, idle/redraw accounting, interruption and reset behavior, gameplay-timing neutrality, and bounded frame-time telemetry through Civ III's normal loop.
10. **Environment and ambient effects**: deterministic sun/moon curves, bounded water response, emissive activation, attachment identity/transform, static-idle behavior, skipped-frame phase, fallback, and protection of retained Civ III layers.

## Current Completed Gates

- M0 project contract/tooling: state contract, complete renderer unit suite, preview smoke test, and renderer-config fixture contract.
- M1.1 standalone texture extraction: synthetic CIVBIG-to-DDS round trip, plus local extraction and PNG conversion of the installed Civ VI grassland texture.
- M1.2 grassland material binding: synthetic allocation/string/pointer-chain probes, plus an exact deterministic report comparison against the installed `TerrainMaterialSet_Base.blp` package.
- M1.3 material role resolution: synthetic role/format/mip-size/embedded-range tests, plus an exact deterministic binding comparison against all relevant records in the installed `TerrainMaterialSet_Base.blp` package.
- M1.4 geometry/UV selection: synthetic ArtDef-chain and normalized-mesh tests, plus an exact installed ArtDef/package-inventory comparison proving the generic flat-grid selection.
- M1.5 normalized grassland rendering: synthetic embedded extraction, BC3 DDS sampling, source-independent pack, path-safety, mesh rasterization, and two-size PNG tests, plus a local one-command build/render using the installed texture payload.
- M2.1 renderer-definition parsing: strict starter-fixture parsing plus table-driven typed-value, diagnostic, layer replacement/disable, reference-integrity, and cross-platform path-safety tests.
- M2.2 deterministic rule resolution: exhaustive shared/terrain selector coverage, representative resource/city/unit selection, exact four-stage ranking with loser explanations, wrapped hours, season aliases, coordinate variants, missing-asset fallback, and config-off no-load behavior.
- M3.1 visible-scene contract: strict field/type/reference validation, pointer and source-format exclusion, deterministic IDs/seeds/pixel anchors, byte-stable canonical round trips, and offline resolver replay of the recorded terrain/resource/city/unit fixture.
- M4.1 standalone whole-viewport renderer: source-independent logical-ID pack loading, authoritative pixel-basis projection, map-rectangle clipping, depth-tested overlap, deterministic hour/season lighting, two-size byte-stable PNG output, safe fallback, and explicit renderer recreation/teardown.
- M4.2 fixture matrix and reference validation: byte-stable two-size/four-hour/four-season PNG matrices, exact dependency/output hashes, deterministic structural metrics and thresholds, labeled contact sheets, and enforced separation between exact C3X regression baselines and qualitative cross-engine references.
- M5.1 native bridge: `native_civ3_bridge` builds and loads the 32-bit D3D11 DLL, exercises off-screen render/readback/resize/ABI/failure/bounded-blit behavior, and runs `TEST_INJECTED_CODE_COMPILE.bat`. Source contract tests enforce the configuration-off branch, bounded capture, proven `0x9 -> 0x1fef0` insertion sequence, vanilla fallback replay, source independence, and absence of a swap chain/presenter. The three `evidence/m5_1` captures document the normal configuration-off path, configuration-on layering, and anchor refresh after scrolling.

The registered M5.2 gate `m5_2_scene_export` builds the native exporter, validates its deterministic synthetic output through the strict `c3x.visible_scene.v0` parser, requires all nine renderer categories in that fixture, runs the offline batch tests, and recompiles injected code. A supplied full-screen game-health capture confirms that retained layers and UI remain present. Formal matched fixtures and explicit per-layer review remain available for regression diagnosis and release review, but are not a routine development prerequisite.

The registered M5.3 gate `m5_3_frame_scheduler` builds and executes the native absolute-time scheduler smoke, runs source-contract tests that keep D3D/render/blit/form-draw calls out of the timer hook, and recompiles injected code. It covers zero static redraws, one pending request, skipped-frame accounting, deterministic absolute event phase, loading/focus/modal/draw/pending suppression, large-pause rebasing, and bounded scalar telemetry. No renderer-owned animation is enabled by M5.3 itself.

Future gates should be narrow enough to identify the broken prerequisite from their name and failure message.

The M6.1 `m6_1_production_terrain` gate reconciles the closed BIQ and atlas ledgers, then renders the replayable terrain matrix through a connected shared-vertex surface. It covers all 14 terrain types, land/water transitions, polar ice, landmark state, two sizes, scrolling, horizontal wrap topology, relief/depth, hour/season lighting, clipping, missing/corrupt per-item fallback, reset, retained-layer ownership, a bounded visible-tile budget, and bounded LRU cache diagnostics. It requires no installed source assets or manual game capture.

The M6.2 `m6_2_native_terrain_art` gate is the first real-art native integration. It uses a generated synthetic generic pack to prove the 32-bit DLL validates normalized manifest/mesh/material contracts, loads a bounded BC3 mip chain, creates a D3D11 shader resource, samples grassland in the existing off-screen target, rejects malformed input, reports textured tiles, and preserves the bridge ABI/fallback compile. The local `m6_2_local_grassland_art` gate loads the ignored licensed-source grassland pack and requires a deterministic native pixel hash distinct from the synthetic pack.

The M6.3 `m6_3_definition_terrain` gate parses the tracked production definition layer, requires an explicit disposition for all 14 terrain types plus transitions, polar ice, and landmarks, builds the 32-bit DLL, and exercises synthetic default/scenario/custom merging, multiple BC3 material slots, deterministic continuous UVs, a custom override, and one corrupt per-item fallback. The local `m6_3_local_terrain_art` gate rebuilds the ignored six-family normalized pack from installed licensed assets and renders four representative families through the same definition-driven ABI with a distinct deterministic hash. Incomplete relief/feature families remain transparent Civ III fallbacks rather than procedural stand-ins.

The M6.4 `m6_4_environment_runtime` gate builds the native renderer and exercises two-size noon/sunset/midnight/sunrise output, continuous finite environment values, bounded directional moon response on water, static night emissive idle, and one state-gated flame/light attachment whose phase is derived from absolute time plus a stable seed. Visibility, missing resources, native-owner fallback, reset/replay, and retained-layer ownership are explicit. The local `civ6_lighting_metadata_local` gate reruns the metadata-only probe, requires the documented GameLighting/Light/VFX/water classes and resources, and checks conservative confirmed/inferred/unresolved evidence labels without redistributing cooked payloads.

The M6.5 `m6_5_connected_terrain` gate builds the native renderer and exercises an adjacent mapped-material pair plus a mapped/fallback pair. It requires a mixed opaque shared edge, a partial-alpha transition to the complete native base underlay, deterministic output, and source-contract evidence that coordinate-seeded per-tile brightness checkerboarding is disabled. It recompiles the injected bridge because complete fallback-underlay replay is part of this gate.

The M6.6 `m6_6_vanilla_terrain` gate accounts for every vanilla terrain identity, generic authored relief and depth, connected rendering, and explicit retained ownership. Its local `m6_6_local_biq_terrain` companion rebuilds the ignored normalized pack and renders the user-supplied 100x100 BIQ deterministically at both Civ III zoom bases.

The completed M6.7/I9-I12 gates freeze approved L9/L10/L11/L12 handoff and reference hashes, validate source-independent terrain/dune/marsh/volcano component mapping and zero-error anchor/clip/cache tolerances, and exercise the capture/composite bridge contract. With ignored normalized payloads available, `i12_local_approved_volcano_payload` runs the production definition through the 32-bit DLL. It materially verifies exact base/real terrain identity, geometry normals, independently transferred relief and active-effect state, the Lab-derived material/shore/depth fields, ordered underlay/land/bed/water passes, authored hills and mountains, dunes, BC1 forest/jungle bodies, `GrassMarsh` plus `CLUTTER_MARSH`, dormant/active volcanoes, both zooms, partial clipping, pixel scrolling, duplicate horizontal-wrap occurrences, exact static reuse, definition/environment/ownership invalidation, deterministic reset, all fourteen identities in the production replay, zero production fallback, and no native replay on custom-on failures. The workflow byte-checks the actual VM test root before the user's interactive `INSTALL.bat` run.

## M7 Category Gates

Infrastructure, resources, cities, units, and effects must use separate verification IDs and separate completion statuses. A broad renderer smoke test cannot mark multiple M7 categories complete.

Each category gate must verify its scene fields, selector resolution, standalone output, calibration, time/season behavior, in-game ownership/layering, missing-asset hard failure without native replay, and config-off behavior. Resource checks include exact BIQ/icon-index seed coverage, player-specific visibility, terrain variants, animation, complete custom map-body coverage, and proof that Civilopedia, city-screen, trade-network, advisor, diplomacy, notification, and other non-map resource icons remain native and unchanged. City checks include culture/era/size selection plus representative day/night window and lamp activation. Unit checks include stable event identity, all eight directions, authoritative-anchor movement, absolute clip timing, attack/victory/death transitions, stacking, interruption, runtime palette-row selection, and body-only replacement. Owner-color fixtures must cover multiple leaders sharing one converted unit, capture/alternate-color changes, effective partial scenario `ntpXX.pcx` overrides, barbarians, and viewer-conditioned hidden-nationality presentation without exposing the real owner or rebuilding unit art. They must also prove that Civ III's selection cursor/ring, health bar, left-side activity/status marks, stack indicators, and related unit HUD remain visible, correctly anchored, unduplicated, and above the replacement body during movement, combat, clipping, scrolling, and custom-pass failure. Effect checks include stable event identity, source/target anchors, spawn, timing, skipped frames, interruption, cleanup, and attached ambient flame/smoke/steam behavior.

M9 natural-wonder, M10 constructed-wonder, and M11 district gates remain separate from M7 and from each other. Natural-wonder verification parses effective scenario/user/default replacement sets, inventories placed C3X instances, distinguishes ordinary landmark terrain, and covers terrain/adjacency/direction, source multipart normalization, water/VFX, retained labels/fog, environment, animation timing, complete coverage, and hard failure. Constructed-wonder verification enumerates BIQ Great/Small Wonders and effective C3X wonder configs, preserves mapless wonders, and covers construction/completion, alternate orientation, placement, environment, destruction/abandonment, layering, and no-native-replay failures. District verification enumerates effective built-in/dynamic/user/scenario definitions and covers `by-count`, `by-building`, shared-building additions, construction/damage/abandonment, culture/era, coast alignment, Bridge/Canal/Great Wall topology, Wonder District relationships, deterministic composition, and whole-instance custom ownership. None may rely on repeated manual screenshots.

## M5.3 Frame Scheduling Gate

M5.3 tests the renderer as a guest in Civ III's UI loop. Static maps must produce zero renderer-requested continuous redraws. Visible animated scenes may use the audited Civ III timer path to mark dirty state and request a normal map update, but no timer callback may render, blit, wait, recurse into map drawing, or pump messages.

Timestamped traces must produce the same unit/effect pose regardless of rendered-frame count. Slow and skipped frames may reduce smoothness but cannot change movement endpoints, combat outcomes, action duration, removal timing, turn processing, or input responsiveness. Loading, modal dialogs, focus loss, minimize/restore, resize, scrolling, and renderer reset require explicit pause/rebase and stale-event tests.

Telemetry gates cover capture CPU time, renderer CPU/GPU time where available, readback/blit time, total map-pass time, requested/presented/skipped frames, and visible instance counts. The initial target is Civ III's existing approximately 66 ms/15 Hz cadence without queue growth; higher target rates require a separately audited and profiled UI-thread path.

## M6.0 Inventory Gate

The portable inventory tests prove layered path precedence, PCX/FLC metadata parsing, unit INI action discovery, deterministic serialization, explicit fog/retained-layer classification, and visible reporting of unknown files. These tests validate the tool, not vanilla coverage by themselves.

The final local M6.0 gate runs strict inventory generation against Base, Play the World, Conquests, and representative scenario search roots. It also consumes BIQ semantic extraction, annotated atlas-layout records, and runtime draw-census evidence. It fails for any unclassified effective file, unresolved atlas layout/index family, unresolved unit binding/action state, missing ownership decision, or runtime-observed file/index pair absent from the ledger.

Fog of war, unexplored shroud, borders, grid, selections, paths, labels, unit status, cursor, minimap, HUD, and editor-only markers must appear in the ledger even when their accepted ownership is `retained_civ3` or `out_of_runtime_scope`.
