# C3X Renderer Roadmap

This is the living, human-readable project status. Stable goals and gates live in `MASTER_PLAN.md`; the machine-readable current task lives in `project_status.json`.

All completed prerequisites are executable gates. Run
`python3 Renderer/tools/renderer_dev.py state` before ordinary work and use the
track command named by the current step. Use `renderer_dev.py full` when closing
a step or changing a shared contract. macOS automatically dispatches
Windows-only work to the documented VM. See `VERIFICATION.md` for the completion
rule.

The end state is a one-command offline source importer plus authored mapping profiles. Runtime C3X composes independently resolved terrain, adjacency, city, unit, resource, time, and season facts into one visible 3D scene. See `MASTER_PLAN.md` under "End-State Import And Mapping Workflow."

## Current Position

Milestone M0 is complete. The renderer workspace, source-agnostic pack shape, isometric preview, project contract, and local tool references exist.

Milestone M1 is complete. We can:

- Index Civ VI ArtDefs and cooked BLP trees.
- Resolve grassland ArtDef references to `TerrainMaterialSet_Base.blp`.
- Parse standalone 48-byte `CIVBIG` texture wrappers.
- Validate BC mip-chain sizes using the stored DXGI format.
- Produce a standard DDS and PNG for `TEXTURE_TER_Grass_Decal_B`.
- Map Civ III grassland `square_type`, sheet index, and sprite index selectors to a source-agnostic material entry.
- Parse the `TerrainMaterialSet_Base.blp` allocation table without reading its big-data payload.
- Resolve `ART_DEF_TERRAIN_MATERIAL_GRASSLAND` to a reproducible typed material record and four typed texture records.
- Resolve the four texture roles by class consistency across all 31 material records.
- Resolve and bounds-check all 79 texture records as embedded package resources with exact BC mip-chain sizes.
- Follow `TERRAIN_GRASS -> StandardFlat -> GrasslandMtl` through explicit ArtDef fields.
- Normalize the flat base to a tested generic unit grid with two triangles, +Z normals, and full-range UV0.
- Extract the validated embedded grassland base color to standard DDS and assemble a source-agnostic pack.
- Render that pack deterministically at 640x480 and 1024x768 through a generic BC3 textured-mesh preview.

The local build produces a 4096x4096 BC3 sRGB base-color DDS, a normalized pack with no source-specific runtime paths or formats, and deterministic textured PNGs at both required sizes. Generated Firaxis-derived artifacts remain ignored and untracked.

Milestone M2 is complete. The v0 renderer-definition parser produces a typed intermediate catalog with structured file/line diagnostics. It implements whole-section replacement and disabling across `default -> scenario -> custom`, validates rule/asset/pack/environment references, and rejects root-escaping or unauthorized development paths. The pure resolver covers the documented shared, terrain, resource, city, and unit selectors; ranks matches by priority, specificity, layer, then declaration; explains every winner and loser; applies wrapped hours and season aliases; generates stable coordinate variant seeds; and falls back without loading asset payloads.

Milestone M3 is complete. `c3x.visible_scene.v0` now records source-independent world, viewport, environment, tile, and object data with Civ III's authoritative pixel projection and anchors. Strict validation rejects missing/unknown fields, unstable IDs/seeds/anchors, bad references, process pointers, and source-specific asset markers. The recorded four-tile fixture survives a byte-stable canonical round trip and replays terrain, resource, city, and unit records through the M2 resolver without Civ III or asset payload loading.

Milestone M4 is complete. The source-independent standalone renderer consumes validated scenes, merged definitions, and normalized logical-ID packs; projects whole-scene meshes from authoritative Civ III pixel anchors; clips to the captured map rectangle; uses a real depth buffer; and applies deterministic hour-driven lighting plus an initial seasonal response. The fixture-matrix command now renders two viewport sizes across four hours and four seasons, hashes every effective input and output, records mapping/bounds/depth/anchor/color/luminance metrics, enforces time/season differences, and emits a labeled deterministic contact sheet. Exact C3X regression hashes remain separate from qualitative cross-engine art-direction references. No injected code changed during M4.

Milestone M5 is complete. M5.1 provides a 32-bit D3D11 off-screen bridge at the audited `m19` pass boundary while Civ III retains overlays and UI. The bridge is isolated under `Renderer/native/`, gated by `enable_custom_rendering`, bounded, and restores vanilla terrain on renderer failure. Live evidence covers configuration off, configuration on, and scrolling with retained rivers, resources, features, cities, units, borders, selection, labels, fog, HUD, and UI. No new `civ_prog_objects.csv` entry was required.

M5.2 is complete under the lightweight development-validation policy requested by the user. The first successfully composited frame now writes a bounded, atomic `c3x.visible_scene.v0` export automatically, with `Ctrl+Shift+F12` retained only as an optional recapture shortcut. The native gate proves deterministic strict-schema output for all nine renderer categories, the offline batch tests prove canonicalization/metrics/review-state behavior, and the supplied full-screen capture shows the game healthy with retained fog, borders, labels, selection, minimap, HUD, and UI. Users are not required to manage paired fixture files during routine development; formal paired captures remain available for regression diagnosis and release review.

M5.3 is complete. `Renderer/native/frame_scheduler.cpp` supplies a pure absolute-QPC scheduling decision and deterministic event phase. The existing `on_timer_0x9F6500` hook marks only Civ III's animator dirty, leaving capture, D3D, readback, and blit inside the normal `m71` map draw. A single pending bit prevents queue growth; focus, visibility, modal, loading, nested-draw, and long-pause states suppress or rebase work. Scalar telemetry records capture, native render/readback, blit, map-pass, requested, presented, and skipped values. Static maps remain idle because M5.3 claims no animated category.

Custom rendering also suppresses C3X's legacy custom terrain/resource FLC overlays at their existing load, match, draw-registration, scheduling, animation-load, and spawn guards. The user's `enable_custom_animations` setting remains intact and resumes when custom rendering is disabled; renderer fallback uses Civ III's normal static art rather than the legacy overlay path.

M6.0 is complete. The deterministic inventory resolves 112 effective map-related files and all 76 effective PCX atlases through explicit Civ III slice rectangles, not dimension guesses. It correlates 124 primary BIQ unit types with 144 selectable unit-art directories and their INI/FLC action, direction, smoke, shadow, and transparency metadata; catalogs 26 resources and 14 terrain types; generates local annotated PNG contact sheets; and closes all 21 replacement/retained/editor responsibilities with zero unknowns. Generated Firaxis pixels remain ignored and untracked. No new Civ III patch address was required.

M6.1 is complete. The closed semantic and atlas inventory now resolves to source-independent logical terrain assets and explicit fallback/retained dispositions. The standalone production reference renders all 14 BIQ terrain types, land/water transitions, polar ice, landmark state, and relief as one shared-vertex viewport; its replayable two-size fixtures prove deterministic wrap/scroll topology, authoritative anchors, clipping/depth, hour/season changes, atomic missing/corrupt fallback, reset, and bounded cache/tile budgets. M7 and retained Civ III instances are never drawn by this pass. No injected code or new Civ III address was required.

M6.2 is complete. The native 32-bit off-screen renderer now validates a generic normalized manifest, mesh, material, and bounded BC3 mip chain, creates a D3D11 shader resource, and samples the actual ignored local grassland texture instead of the flat green placeholder. A versioned pack-path ABI configures the DLL before rendering; missing or malformed packs reject initialization and restore vanilla terrain. Portable synthetic and local licensed-pack hashes prove texture consumption. No new Civ III function/address was required.

M6.3 is complete. `Renderer/default.custom_rendering.txt` now drives native terrain selection through default, scenario, and custom layers. The DLL resolves generic pack IDs and logical assets into separate bounded material resources, samples same-material tiles with continuous map-coordinate UVs, and leaves missing or corrupt assets transparent for per-item Civ III fallback. The ignored local pack contains six proven real material families: desert, plains, grassland, tundra, coast, and sea. Every incomplete relief, feature, ocean, ice, or landmark family is explicitly classified as vanilla fallback rather than receiving a colored or semantically false stand-in. No new Civ III function/address was required.

M6.4 is complete. The native renderer now consumes one source-independent environment state with continuous sun/moon, ambient, exposure, shadow, emissive activation, and bounded directional water response. Generic material and attachment fixtures define emissive channels, analytic lights, explicit local transforms/bounds/state requirements, stable phase seeds, and explicit missing-resource policies. The existing absolute-time scheduler remains the only animation cadence source; static night scenes stay idle. Structured local metadata confirms GameLighting and Water/Wave bindings plus Light/VFX/resource classes. A later offline intake now decodes six typed fields for each Base analytic light and normalizes twelve production-like resources; model attachments, VFX behavior, and visual calibration remain unresolved. No new Civ III function/address was required.

M6.5 is complete. The native material bootstrap now uses authoritative map coordinates to inspect all four valid neighbors, blends mapped terrain families through symmetric edge bands, and feathers mapped/fallback boundaries into Civ III's complete native base underlay. Whole-tile coordinate brightness variation is removed, eliminating the artificial checkerboard. The native smoke proves opaque mixed edges, partial-alpha fallback edges, and deterministic output; the bridge contract and injected compile protect the full-underlay fallback that prevents black relief wedges. This is a continuity correction, not a claim that flat material diamonds are production terrain.

M6.6 is complete. The bridge now distinguishes underlying material from visible real terrain, and the ignored local pack exposes all fourteen terrain identities. The compiler structurally extracts actual Civ VI R8 terrain-element height fields, including a nine-variant mountain atlas. The native renderer uses a 12x12 surface grid, map-continuous UVs, subpixel seam closure, source-material grading, distinct shallow/deep water, and a D24S8 depth target. Forest, jungle, marsh, ice, shoreline detail, and landmarks remain explicit complete Civ III overlays/fallbacks rather than false procedural replacements. The user-supplied 100x100 `test.biq` drives deterministic native screenshots at both Civ III zoom sizes with zero fallback, plus a closed all-fourteen-type fixture. No new Civ III function/address was required.

M6.7/I13A is complete. Build-time contracts consume the frozen approved L9
through L13A handoffs; the production DLL contains only deliberately integrated
paths and performs no runtime handoff-approval decision tree. It loads generic
terrain, dune, vegetation, marsh, volcano, clutter, river, shore-feature, and
shared-environment payloads and uses the approved formulas, topology, densities,
anchors, materials, shadows, and lighting. L14 road code remains absent.

Authoritative Civ III capture supplies real/base terrain, SquareParts,
visibility, canonical coordinates, and distinct screen occurrences. A bounded
terrain-only cache fingerprints target, zoom, anchors, wrap, environment,
content revision, ownership, and device generation independently of dirty
hints; the dirty clip and retained unit/UI state are deliberately excluded. For
the current user-directed diagnostic stage, an active custom frame
owns the complete Civ III `m19` map plane after using that function's
authoritative capture/composite boundary. Native roads, improvements, resources,
and cities are intentionally absent until their approved renderer
systems arrive. Configuration off is fully vanilla; configuration on never
replays native terrain after loader, capture, render, validation, blit, device,
or reentrant failure.
This exclusive custom-on policy supersedes the fallback behavior recorded for
historical M5.1-M6.6 bootstrap gates; those entries remain unchanged as an audit
trail of what their tests proved at the time.
The actual VM `C3X_Districts` test root is now a directory link to the single
macOS-hosted Git checkout. The integration workflow verifies that link before
the user's interactive `INSTALL.bat` run instead of copying selected files
between repositories. The former compile-only shared link did not replace the
separate live checkout and therefore allowed stale live files; linking the live
path itself removes that second source of truth.
The user's 2026-09-05 live retest first exposed a completely black custom map
plane. Native debug output identified reversed Civ III `m49`/`m50` capture
semantics; correcting underlying-ground versus visible-category capture produced
custom terrain in game. The next live report exposed cumulative brightening and
scroll ghosts caused by alpha-blending each new map over the prior surface. The
DLL now replaces the complete map bitmap with `BitBlt(..., SRCCOPY)` on every
composite. The user then confirmed stable camera movement and scrolling, closing
I11. The frozen L9 terrain through L13A shared-lighting handoffs load in the
standalone production fixture, and the Windows smoke
proves both zooms, clipping, scrolling, wrapping, cache invalidation, device
reset, exact terrain/feature/river ownership, authoritative lighting phases, and zero native
fallback. No new Civ III patch symbol was required.

Post-I13A integration maintenance preserves the approved hashes while reducing
the native 400-tile/800-record cold benchmark from 57.790 seconds to 5.241
seconds and an anchor-only scroll to 1.054 seconds through shared-grid sampling,
per-tile surface memoization, river-pass isolation, large-live shadow LOD, and a
content-fingerprinted anchor-independent shadow field. API v7 returns the exact
authoritative clip and blits only that dirty rectangle, preventing cleared
off-clip pixels from truncating retained neighboring tile art. Custom-on `m71`
now captures a complete visible traversal even for a partial unit/UI redraw, so
such a traversal cannot poison the full-viewport cache; partial and reordered
static-terrain traversals are zero-tick cache hits. Feature bodies use the frozen
Lab ground-plane depth rule rather than the formerly divergent lifted-screen-Y
formula that allowed neighboring ground diamonds to clip tall vegetation.

L11 is complete. The user explicitly approved the corrected marsh promotion
render on 2026-09-05. Its frozen 96-tile handoff preserves the authoritative
two-cell BIQ halo that corrected the misleading lower-right crop-edge water cell.
Beginning with L12, promotion viewports double to 192 visible cells in a 16x12
true-adjacency diamond; L9-L11 retain their historical accepted sizes.

The preserved L12 standalone candidate now also corrects the sparse vegetation
regression visible in its authoritative Mesoamerica crop. Forest and jungle use
more, smaller normalized source bodies on deterministic jittered lattices, so
connected tiles form closed canopies without making individual trees or palms
too large relative to mountains. The L12 crop witnesses jungle; the unchanged
authoritative L11 crop supplies the forest/mountain scale witness.

A subsequent Civ VI close-reference audit corrected the static-water approach.
The missing richness was the source `CLUTTER_OCEAN` projected seabed layer—not
extra transparency or stronger repeating water normals. L12 now normalizes and
renders its five nonzero rock/contour entries and four coast-crack entries with
the confirmed `0.52` density, deterministic selection, edge fades, and
shoreline clipping. The same generic pass adds restrained grassland, plains,
and grassland-hill surface clutter. The rejected transparency, whole-atlas,
oversized-cloud, and every-cell repetition candidates are not preserved.

## Next Step: L19

L18 is complete and approved under the user's explicit autonomous-review
authorization. Its frozen handoff adds twenty recursive source-backed mine
compositions across preindustrial/industrial eras and three variants, with
terrain-following excavation decals, one compound shadow, confirmed emissives,
and native/reduced readability. The no-mine control is byte-identical to L17.
Begin L19 with connected, topology-aware normalized farm and irrigation blocks
without altering the approved alternate-skin scene.
Use `python3 Renderer/tools/renderer_dev.py lab`.

Ahead of that visual gate, all 24 animated clutter bodies now compile with
model-aware pose caches, including single-animal elephant selection and one
documented invalid source-curve sentinel repair. Reusable future-gate tooling
also generates 192-tile/two-zoom Lab scaffolds, inactive Integration cache and
ownership contracts, content-addressed deduplicated pack bundles, and a
nine-case arbitrary-scenario proof. These artifacts do not constitute L16
rendering or approval; see `docs/offline_future_gate_acceleration.md`.

## Paired Workstream And Promotion Rule

**Renderer Lab** is the standalone visual-development path for terrain and every
later map object. **Game Integration** is the Civ III delivery path. Each system
moves through matching gates: L9 -> I9, L10 -> I10, L11 -> I11, and so on.
The L# gate owns standalone quality and freezes the handoff; the I# gate owns
live capture, cache/invalidation, compositing, exclusive native suppression,
telemetry, reset behavior, and visible hard failure without native replay.
L21/I21 are the combined release-level gates. See `docs/renderer_workstreams.md` and
`terrain_lab/PLAN.md`.

User-directed M6.8 preparation began on 2026-09-05 without advancing its formal
gate. The one-command terrain adapter, deterministic equivalence report,
external Renderer Lab pack selection, and production `#Pack` override are now
executable. A private ignored pack combines the alternate skin's materials with
the proven baseline relief and animated-water resources; all 14 current terrain
IDs are replaced and none are missing. Renderer Lab and the native integration
workflow both pass with the local override. Formal M6.8 remains blocked until
M6.7 and the alternate skin's own lab approval, and no converted payload is
tracked or distributed.

## Ordered Backlog

1. M6.0: Complete — the strict layered/BIQ/atlas/FLC/ownership inventory closes with zero unknowns; C3X natural wonders and Districts are deferred to M9 and final renderer milestone M11.
2. M6.1: Complete — connected source-independent terrain, transition, feature, water, landmark, fallback, and budget gates pass.
3. M6.2: Complete — actual normalized grassland DDS is sampled by the native in-game rendering path with safe pack failure.
4. M6.3: Complete — layered native definitions select six real-art terrain materials and every incomplete family falls back atomically to Civ III.
5. M6.4: Complete — shared sun/moon, water lighting, emissive-material, analytic-light, and deterministic ambient-attachment primitives pass portable and local evidence gates.
6. M6.5: Complete — connected mapped materials blend symmetrically, fallback boundaries feather into a complete Civ III underlay, and tile checkerboarding is disabled.
7. M6.6: Complete — all fourteen terrain identities use normalized source materials/relief or explicit complete retained ownership; `test.biq` renders deterministically at both zooms.
8. M6.7: Complete — automated Windows delivery passes, and the live I11 checkpoint confirms nonblack custom terrain, stable redraw brightness, and ghost-free scrolling.
9. L9 / I9: Complete — approved terrain foundation is integrated.
10. L10 / I10: Complete — approved source-backed tile-continuous dunes are integrated.
11. L11 / I11: Complete — approved marsh art is frozen and stable in the live integration boundary.
12. L12 / I12: Complete — the approved volcano and complete shared L12 terrain-stack delta are integrated without native terrain fallback.
13. L13 / I13: Complete — authoritative Civ III river masks drive the frozen approved shared-edge graph, valleys, water/material channels, mouth/source/junction topology, and normalized river rocks.
14. L13A / I13A: Complete — authoritative hour and season drive the frozen approved sun/moon/ambient/exposure/water response and raised-terrain/feature cast-shadow contract at both game zooms.

Production now compiles a frozen copy of the approved standalone renderer's
`PSMain` and `PSFeature` functions. Production retains only a Civ III
scene/input adapter and semantic terrain settings around that copy; it does not
include the live Lab shader, so in-progress visual work cannot enter the game
before its handoff. The current freeze ends at L13A; L14 road selectors and
textures are omitted at compile time and no road geometry is submitted.

The first live I12 checkpoint exposed an incomplete shader-only convergence:
production still averaged categorical terrain IDs, supplied flat normals,
overloaded authored relief with active-effect state, and used a tile-major
two-pass approximation. The corrected adapter now transfers exact Civ III base
and real terrain identities, computed normals, distinct relief/effect fields,
and the Lab-derived material/shore/depth values. It emits the same ordered
viewport-wide underlay, land, bed, water, and feature stack, fails custom-on
configuration atomically if any approved L9-L12 payload is missing, and creates
a production-DLL all-terrain replay during both `integration` and `full`.

A later live comparison exposed a reversed Civ III-to-Lab lattice basis in the
production adapter. The corrected source transform now drives material,
shoreline, relief, feature, wrap, and deterministic-seed lookups consistently.
The old generic height approximation was removed from the BIQ game path;
production now copies the approved hill macro/support, mountain/volcano chain,
dune envelope, vegetation placement, and 224-pixel/0.82 terrain projection
rules. The native preview accepts the exact Lab BIQ-window CSV, including its
halo, so future integration corrections can replay the same authoritative input
instead of comparing unrelated map regions.

13. L13 complete / I13 handoff available — the approved alternate-skin river gate covers canonical shared-edge topology, source-backed channels/banks/clutter, sources, junctions, coast mouths, relief, vegetation, and horizontal wrap in deterministic 192-tile fixtures.
13a. L13A complete / I13A handoff available — the approved alternate-skin lighting gate covers shared shadows, coherent face and cast direction, water response, non-visual emissive activation, and deterministic noon/sunset/midnight/sunrise fixtures at both Civ III zoom scales. Visible city lights remain owned by L17.
14. L14 complete / I14 handoff available — the approved 98-node / 109-edge Lab road graph uses gently curved exact-node centerlines, source-backed continuity coverage, normalized bridge bodies, wrap continuity, four styles, and pillage coverage without changing the approved L13A control.
15. L15 complete / I15 handoff available — the approved connected railroad subset uses narrow authored sleepers/ballast, paired source-colored rails, exact river bridges, road coexistence, relief, wrap, and deterministic both-zoom evidence without changing L14.
16. L16 complete / I16 handoff available — the approved normalized resource gate covers strategic, luxury, land-bonus, and aquatic-bonus bodies, corrected Civ III-scale clustering, shared lighting/grounding, visibility suppression, and deterministic both-zoom evidence without changing L15.
17. L17 complete / I17 handoff available — the approved normalized city gate covers four eras, all three Civ III size bands, culture/owner metadata, wall/capital states, compact source-backed compositions, retained-label clearance, shared lighting, and source-authored night emissives without changing L16.
18. L18 complete / I18 handoff available — the approved recursive normalized mine gate covers preindustrial/industrial families, three variants, terrain-following excavation, mineral and relief adjacency, coherent compound shadows, source emissives, and deterministic isolation without changing L17.
19. L19 ready: add connected topology-aware normalized farm and irrigation blocks over the approved alternate-skin 192-tile scene, then critically inspect and self-approve under the user's authorization.

Offline L14/L15 intake is prepared ahead without advancing either gate. The
source-independent route packs now cover four road stages, railroad ballast and
rails, worked/pillaged route recipes, four worked/pillaged bridge bodies, twelve
endpoint decals, and all fifteen Base/Expansion 2 bridge transition rules. L14
still owns graph construction, terrain conformance, visual rendering, and its
approval render; L15 remains ordered after L14.

16. L16: Add resources, then generate and obtain approval for the 192-tile running integration render.
17. L17: Add civilization-, size-, and era-specific cities, then generate and obtain approval for the 192-tile running integration render.

Offline L17 intake is prepared without advancing the gate. The complete source
graph resolves 2,690 component bindings / 975 unique components; a generic proof
pack converts 44 representative components across all twenty Civ III
culture-group/era pools, including 96 emissive material bindings and 35 exact
attachment sockets. Population controls deterministic composition density;
the palace marker is confirmed not to be a terminal asset, and a separate pack
converts 19 ancient/medieval/industrial wall pieces. Native informational layers
remain retained. L17 still owns composed-city and wall rendering, any separately
authored capital centerpiece, both-zoom readability, and promotion approval.

18. L18: Add mines, then generate and obtain approval for the 192-tile running integration render.
19. L19: Add farms/irrigation and, as an independent convenience-bundled task, close the Terrain Lab's missing tundra-material path; then generate and obtain approval for the 192-tile running integration render. Tundra is a base-terrain concern, not a farm or irrigation feature: the gate must bind Civ III tundra (`base == 3`) to the normalized tundra base-color, height, and specular material rather than grassland fallback regardless of irrigation state. Separate witnesses must prove the irrigation topology matrix across every irrigable terrain family and prove irrigated/unirrigated tundra plus mixed tundra/non-tundra material boundaries at both zooms.

Offline L18/L19 intake is prepared without advancing either gate. The closed
source graphs expose 18 distinct mine components and 204 farm components. The
representative pack remains available, while the full intake now accepts all
222 top-level roots with zero rejects and recursively normalizes 294 components,
including 114 confirmed emissive material bindings. Mine
era/variant/resource ownership and farm era/topology/terrain/crop policies are
checked in, while final adjacency recipes, visual rendering, the tundra Lab
material correction, and promotion approval remain owned by L18/L19. L19 must
exercise every one of the sixteen Civ III irrigation masks across every
irrigable terrain family. Its separate tundra track must prove the base material
with and without irrigation; tundra coverage is a required gate witness, not an
incidental result of the selected BIQ viewport or a dependent part of farms.

19a. L19A: Add goody huts and Civ III colony stand-ins, then generate and obtain approval for the 192-tile running integration render.

19b. L19B: Add fortresses, barricades, airfields, outposts, radar towers, pollution, craters, and victory locations, then generate and obtain approval for the 192-tile running integration render.

20. L20: Add units and animation, then generate and obtain approval for the 192-tile running integration render.

Offline L20 intake is prepared without advancing the gate. Archer, Swordsman,
Infantry, Fighter, and Galley now compile into a generic proof pack containing
19 components (12 skinned, seven rigid), 45 normalized textures, 37 unique
validated raw clips, and 44 logical bindings. Ninety-three deduplicated
model-aware component pose caches now serve 100 logical component/action
bindings, so the runtime never needs source curves. The basic contract covers idle,
fidget, move, fortify, attack, event-derived defend, victory, and death across
five archetypes, plus Fighter takeoff/landing/turns. Galley's three-mesh/two-
material body closes the generic multi-mesh component format gap. ATTACK1/2/3
intentionally alias one logical attack by default. Mounted, crewed siege, and
armored-with-crew source parts now pass one generic arbitrary-tree compiler:
Horseman, Classical Great General, Catapult, and Tank produce eight independently
animated nodes, four resolved parent sockets, 30 components, 50 textures, and
52 converted node/action clips serving 62 logical bindings across 31 actions.
Fifty-two deterministic model-aware pose caches eliminate runtime raw-curve
sampling, and four-phase CPU composition proves the child node frame remains on
its animated socket without unit-name runtime branches. Horseman, Tank, and the
Classical General now cover all eight basic actions; Catapult deliberately keeps
death as one explicit gap instead of mislabelling a reaction. A checked-in
eight-facing/two-zoom matrix freezes one rotated instance, shared pose data,
exact half-scale projection, single-body default, optional pack-authored
humanoid triad, and Army commander+member exception. L20 still owns actual
all-cell visual measurement, final scale/facing offsets, the Catapult death
decision, and promotion approval.
Runtime owner-color selection is also frozen without advancing L20. Converted
materials retain one neutral base plus a civ-color weight; one 64-by-32 lookup
is populated from Civ III's effective loaded scenario palettes, and each unit
selects `Leader.Color_Table_ID` using the native viewer-conditioned display
civilization rather than blindly exposing its owner. Captures or alternate
color assignments update only the instance selector, while scenario palette
changes rebuild the lookup, never the unit art.

Army presentation is now frozen without advancing L20. Civ III's dedicated
Army path confirms that the authoritative displayed member and the Army's own
general body animate side by side, with 40-pixel normal and 20-pixel reduced-
zoom offset references. The generic contract therefore composes the exact
ordinary member asset selected by Civ III with era-profiled dedicated Civ VI
Great General art, retains one parent HUD, supports empty and arbitrary mixed-
member Armies, and never bakes member combinations. The Modern foot General
recipe resolves directly; the Classical mounted General now passes the same
generic horse+rider socket and paired-animation proof as Horseman. L20 still
owns animation/scale calibration, the full Army visual matrix, and approval.

The future I20 unit-body boundary now has exact installed-GOG evidence without
enabling a patch. `Unit::tick_anim` calls one normal or reduced Sprite body
routine, then retained HUD work; its Army helper calls the same routine twice
for commander/member, then one retained HUD. The preferred design is a scoped
Unit context plus guarded normal/reduced Sprite inleads, allowing unrelated
Sprite draws to pass through and making custom success/fallback atomic for an
ordinary or compound body and for both Army bodies. Steam/PCGames addresses and
the reduced ABI remain unresolved, so no CSV request is made. See
`docs/i20_unit_body_replacement_spike.md`.

Offline M7.5 combat-effect intake is prepared without advancing the gate.
Civ III's exact target-effect boundary is now traced: bombard fire/bombing runs
before outcome resolution, and every presented damage roll creates one of four
hit effects, a land miss, or a water miss through the already-patched animated-
effect loader. The generic contract separates authored unit release markers
from authoritative native impact calls, supports ballistic shells, dropped
bombs, guided missiles, and a fail-closed nuclear family, preserves native
audio/timing, and forbids mixed custom/native pixels. Twenty-two upstream
muzzle/projectile/explosion/smoke/debris/water/nuclear textures convert to a
3,172,048-byte ignored generic pack. The final two native-boundary audits are
now closed: native FLCs tick and trigger sound before byte `0x184` gates their
pixel blit, and nuclear results enter `Unit::do_nuke_tile` (detonation) or
`Unit::get_intercepted_as_nuke` (interception), including multiplayer replay.
Ordinary effects need no new draw hook. M7.5 does require the standalone
animation loader upgraded to `inlead` for the SDI FLC and two new nuclear-
outcome inleads; exact supported-build requests are in the dependency ledger.
Particle behavior, runtime event implementation, and visual calibration remain
owned by M7.5, so no effect replacement is enabled early. See
`docs/bombardment_and_explosion_effects.md`.

Offline goody-hut and colony intake is also prepared without advancing L13A.
The exact goody-hut ArtDef chain resolves to three tribal-thatch compounds;
they and six ordinary resource-camp roots recursively normalize into the generic
tile-object pack. Huts retain viewer-conditioned visibility and eight
deterministic Civ III reference buckets, with culture/era neutrality and
optional night fire/light attachments. Colonies render as a reduced owned
resource-logistics outpost beside—not instead of—the resource, use the colony
body's owner and era even when extraterritorial, and apply restrained runtime
Civ III color only to a generated pennant/trim marker. The three former
industrial rejects now pass the corrected strict row-vector matrix proof below
`8e-7`, so eras 2-3 use the real industrial compounds without a static bake. No
new native patch symbol is required. See `docs/goody_huts_and_colonies.md`.

Offline L19B intake is prepared without advancing the gate. Exact Fort and
Airstrip ArtDef chains resolve and five roots recursively compile with their
walls, earthworks, cannon, flags, tower, windsock, hangar, vehicles, and runway
lanterns. The combined hut/colony/infrastructure proof pack now contains 91
components, 243 geometry parts, 179 materials, 79 textures, and 71 emissive
bindings with zero optional dependency rejects. Fortress, denser Barricade,
Airfield, and two-era Outpost policies are checked in. A later probe normalized
four persistent crater decals and an emissive observatory body as a Radar
readability candidate; the semantically wrong missile silo and invalid Modern
Fort remain rejected. Pollution now uses `NUCLEAR_FALLOUT -> FX_Radiation` as
its preferred direction with five normalized textures and a bounded generic
seven-particle tile-local profile, though L19B still owns visual calibration.
Victory Location is explicitly set aside. See
`docs/remaining_tile_infrastructure.md`.

The same broader source pass normalized two culture-specific palace compounds
as optional L17 capital centerpieces and consolidated 2,620 exact model
attachment identities (88 VFX candidates and six analytic-light candidates)
with socket transforms. Resource-script behavior remains undecoded, so these
joins do not enable effects early.

A further offline cross-cutting pass now supplies six bounded generic effect
profiles, automatic eight-facing/two-zoom tile fitting, 79/79 future-category
state provenance with eight bounded audits and no patch request, a verified
content-addressed reference loader ABI, and dependency-free visual-QA metrics.
The radar observatory's isolated 32-cell sheet makes it a weak semantic
candidate rather than an approved mapping. See
`docs/offline_crosscutting_preparation.md`.

21. L21: Generate the complete 192-tile beauty scene, including goody huts, colonies, and remaining tile infrastructure, and obtain final release-level visual approval; do not use it to delay prior per-system integrations.
22. M6.8: After the alternate skin's own lab approval, compile it into a separate selectable pack and compare stable logical-ID coverage; use only assets with documented conversion permission.
23. M7.1: As each matching lab gate is approved, port rivers, roads, railroads, farms/irrigation, mines, goody huts, colonies, and remaining tile-bound infrastructure through independent native handoff, exclusive-ownership, and hard-failure gates.
24. M7.2: Render map-resource bodies with player-specific visibility and optional ambient animation while retaining every native non-map resource icon.
25. M7.3: Port the approved city matrix across owner/civilization, culture group, era, and size, then walls, capital/style flags, buildings, windows, and lamps while retaining Civ III labels/population/UI.
26. M7.4: Port the approved owner-colored, eight-direction, animated unit path with movement, stacking, combat, victory/death, and interruption while retaining Civ III's native selection ring, health bar, activity/status marks, and related unit HUD.
27. M7.5: Render transient effects, projectiles, and attached flames/smoke/steam using stable event IDs, authoritative anchors, deterministic timing, interruption, and cleanup.
28. M8.1: Automate seasonal asset/material authoring and validation.
29. M8.2: Add reproducible human and AI-assisted visual review manifests.
30. M9.1: Inventory every effective C3X natural-wonder definition and placed instance, then map names and source parts across permissioned source art.
31. M9.2: Render terrain-integrated natural-wonder kits with direction, water/VFX, environment, retained labels, animation timing, and custom-on hard failure without native body replay.
32. M10.1: Inventory every BIQ Great/Small Wonder and effective C3X wonder definition, then seed swappable source mappings.
33. M10.2: Render constructed wonders through construction/completion, orientation, environment, lights/effects, destruction/abandonment, and exclusive custom-on ownership.
34. M11.1: Inventory every effective built-in, dynamic, user, and scenario C3X district definition and runtime render state.
35. M11.2: Render source-agnostic district kits with `by-building` attachments, component-owned lights/effects, and deterministic `by-count` stage presets.
36. M11.3: Complete Bridge/Canal/Great Wall topology, Port alignment, Wonder District relationships, shared-building state, custom-on hard failure, and final integration gates.

Do not jump to D3D injection while M1 through M4 contracts are still unproven. In-game integration should consume tested packs, definitions, scenes, and projection math.

The future source-adapter rules are in `docs/source_adapter_contract.md`. Save/BIQ fixture export and production human/AI review are in `docs/visual_validation_plan.md`. Exported retained categories remain descriptive capture facts and do not transfer rendering ownership away from Civ III.

Manual in-game screenshots are reserved for batched strategic checkpoints. Agents first complete automated/replay evidence, reuse still-valid captures, and carry a missing user review as `pending_manual_checkpoint` while continuing independent work; ordinary iterations must not generate repeated screenshot requests.

Frame pacing and animated-unit/effect ownership are specified in `docs/runtime_animation_and_frame_pacing.md`; bombard, bombing, impact, and nuclear event ownership is detailed in `docs/bombardment_and_explosion_effects.md`; the audited native call chain is in `docs/civ3_render_loop_viability.md`. Terrain uses Civ III's retained map redraw boundary, while M7.4 unit bodies must use the later Animator-owned dynamic canvas and dirty regions. Both derive animation phase from absolute monotonic time, request ordinary Civ III work only while visible animation is active, and skip late presentation frames instead of slowing or extending gameplay.

Shared environment ownership and source evidence are specified in `docs/environment_lighting_and_ambient_effects.md` and the lighting findings. M6.4 converts its conservative supported slice into generic runtime primitives without embedding source-specific concepts; later object gates bind real model-owned emissives and effects.

Civ III hook/address dependencies are tracked in `docs/civ3_patch_dependency_ledger.md` and mirrored in `project_status.json`. There is currently no user action: M6.6 consumed the existing base/real terrain identities, SquareParts, anchors, insertion boundary, and M5.3 scheduler without a new hook or address. Any later need for an object attachment selector must first be proven and recorded in the dependency ledger.

Renderer Lab gates promote systems independently into Game Integration. Each M7
family waits for its matching lab approval, handoff record, and integration
acceptance gate; it does not wait for unrelated future lab systems or L21. L21 is
the final combined visual/release gate. Unsupported categories continue using
Civ III independently.

M6.0 inventories fog of war/unexplored shroud, borders, grid, selections, paths, labels, status overlays, cursor, minimap, HUD, and editor markers. The completed deterministic census, contracts, and evidence are in `inventory/civ3_art_inventory.py`, `inventory/vanilla_atlas_layouts.json`, `inventory/vanilla_conquests_biq_semantics.json`, `inventory/runtime_selector_census.json`, `docs/vanilla_art_inventory.md`, and `evidence/m6_0/README.md`.

Natural wonders, constructed wonders, and C3X districts are late, separate categories described in `docs/natural_wonder_rendering.md` and `docs/wonder_and_district_rendering.md`. Natural wonders resolve from authoritative C3X natural-wonder instances rather than landmark terrain inference. Districts use composite kits: `by-building` maps a base plus independently keyed building attachments, while `by-count` maps the current count to a deterministic stage preset. Missing pieces preserve the complete existing C3X draw rather than producing a partial or duplicated instance.

## Known Risks

- `CIVBLP` is a proprietary serialized package; some associations may require Firaxis runtime metadata or an SDK/Pantry fallback.
- Civ III's selected sprite metadata may encode topology that must be preserved even when the replacement art is continuous 3D terrain.
- GPU readback may be slow, but it is acceptable for the first bridge and can be optimized after correctness.
- Layer boundaries for roads, rivers, fog, and other overlays must be verified in-game rather than assumed from decompiled names.
- Scenario and user override semantics must be deterministic before packs become large.
