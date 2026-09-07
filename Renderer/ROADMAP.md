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

The user-authorized production refresh from the pinned [C3X implementation pickup package](handoffs/candidates/lab_v2_terrain_lighting_r1/README.md) is ready for the user-run gameplay checkpoint. The [native port record](native/profile_v2/README.md) and [checkpoint](native/profile_v2/checkpoint.json) record passing cache/image/hardware gates, 18 combined scene cases, authoritative edit/reset checks, the full workflow and approved injected compile. The API 14 DLL is staged; the usual `INSTALL.bat` installs matching code, and normal game launches select pickup-r1 with detailed diagnostics by default. Latest cold and minimap measurements are recorded below; first exposure remains noticeable. Civ III was not launched; visual approval and LQ gates remain unchanged. The reported logging execute fault was mapped to a direct injected OutputDebugStringA import retaining an invalid installer address; it now uses the established game import pointer. Normal gameplay logs only through OutputDebugStringA, without automatic file output. The next user log proceeds past that logging call; no recurrence of that execute fault is reported.

The next in-game report isolated a black-map ownership failure: rendering succeeded,
but caster/prefetch-only tiles incorrectly claimed native replacement. The old DLL
reproduces the rejection headlessly; corrected publication keeps zero ownership
for those records while retaining their shadow and cache contributions. The actual
injected validator remains strict and is now exercised by an executable test;
headless replay also checks ownership on every cold/warm/scroll frame. An exact
inland flat-ground shortcut retains byte-identical output. Cold first-map geometry
remains a measured limitation, separate from the corrected missing-terrain cause.

The user's subsequent mountain/gradient screenshots exposed an unscoped material
JSON lookup and final-copy color loss. Root mountain/coast channels now bind the
correct selected-skin textures; headless pickup tests now include production custom
overrides (earlier measurements used default packs). A real RGB555 terrain copy
reproduces the water/sand bands. Known RGB555/RGB565 destinations now receive
world-anchored ordered rounding; full-color caches and existing budgets remain
unchanged. Fifty focused tests and Windows GDI smoke pass. The corrected copy
averages 2.505 ms at 960x640; corrected-skin prepared jumps build no meshes.
The DLL is staged for normal INSTALL.bat. Runtime destination-format diagnostics
will confirm the game's actual surface; full Lab parity and cold-load speed remain
open. This correction does not promote active Lab experiments.

The user then confirmed slow cold starts/minimap jumps and explicitly deferred
edge speckling. Exact prepared coastline queries, bounded allocation-reusing point
scratch and height-only normal samples reduce a controlled 960x640 cold render
from 11.019 to 5.358 seconds, with a byte-identical image. New distant views improve
from 5.813/1.831 seconds to 2.845/1.042 seconds; retained revisits take about 4 ms.
Existing cache budgets and the color/fog paths are unchanged; no disk cache was
added. Fifty focused tests, 18 production scene cases and authoritative edit/reset
checks pass. The exact-final-binary prepared-scroll witness passes with zero mesh builds/uploads;
the tested DLL is staged for normal INSTALL.bat.
An expanded older frozen-profile boundary test fails on both the previous and new
DLLs, so this is not a clean full-workflow pass; unchanged thresholds and that
existing limitation are recorded in the pickup checkpoint. Cold exposures still
hitch; no vanilla-speed claim or milestone advancement is made.

The active Lab goal is now **city quality across sizes, eras and cultures, including reflected night lights**. The user permits modest cross-tile footprints, especially for larger cities. The [city campaign](terrain_lab/v2/audits/beauty/CITY_QUALITY_CAMPAIGN.md) records complete cities composed with the current 100-tile water scene, a visible building-scale comparison, expanded source intake (132 components), corrected source-origin foundations and r8 night lights. Recovered UV2 light-atlas coordinates remove roof artifacts; brighter windows, GPU HDR glow and reflected lights have matched controls plus four passing standalone Metal/D3D11 comparisons. Full culture/growth coverage, local light pools, remaining materials and all acceptance gates remain open. The subsequent r11 colonial growth probe preserves building placements across all three sizes; r13 adds an explicitly mapped source palace with seven unchanged houses in its no-palace controls at both zooms/noon/night. The complete offline palace audit now resolves and normalizes all 47 distinct standard-game palace roots; L17 still owns generic profile mapping, visual calibration, the Gran Colombian root's four unresolved tree children, and native capital-state binding. The current Lab run has one L19A tile-object pack hash failure; that pack was not modified by this city pass.

The subsequent [single-era r18/r19 pass](terrain_lab/v2/audits/beauty/CITY_GROUND_AND_CAPITAL_PASS.md)
adds source paving beneath every selected modern/medieval building and clips
pad overhangs to the sampled shore. It also connects the American palace from
the existing 47-root pack. Eight standalone Windows comparisons pass; gains are
partial, and the palace's obscured facade/scattered city layout are still visible
limitations. Previous bests and all gates remain preserved.

The [r21/r22 capital composition pass](terrain_lab/v2/audits/beauty/CITY_COMPOSITION_PASS.md)
aligns visibility scoring with authored ground, makes the American palace facade
clearer, and preserves its placement plus existing buildings through all three
sizes. A region new to city tuning renders with the same recipe. Exact dry-cell
clipping lets source paving survive composition; ten Windows comparisons pass.
This is a provisional family-specific improvement, with full city/material and
milestone acceptance still open.

The [r28 city material pass](terrain_lab/v2/audits/beauty/CITY_AO_MATERIAL_PASS.md)
uses UV1 for the tested medieval source AO atlas while preserving diffuse UV0
and emissive UV2. Combined repeat addressing and paving improve eave/recess
separation; a new inland city witness and the ancient capital broaden checks.
Fourteen Windows comparisons pass. Tangent/LEAN/gloss, broad ground coverage,
other families and all existing city gates remain unfinished.

The user's subsequent reference review exposed omitted city-generator and grounding
metadata. The [generator findings](terrain_lab/v2/audits/beauty/CITY_GENERATOR_FINDINGS.md)
record mixed-era center ordering and growth/fill parameters, plus recovered city
paving triangles and atlas UVs. r16 ground projection works but remains mostly
occluded by the packed buildings; it is not a new visual best. The next Lab
pass will use city fabric and ground pieces together. The user rejected r17
historical-era mixing and selected one current era per city; generator-profile
composition now defaults to that policy. Roads are explicitly
user-deferred; native ownership and milestone gates remain unchanged.

The preceding Lab goal covered **more natural static water**, with animation and coastal surf deferred. The user said “Water looks great” about water-natural-r6 and requested object reflections. [water-reflection-r5](terrain_lab/v2/audits/beauty/WATER_OBJECT_REFLECTIONS.md) adds planar GPU prepasses in the standalone Metal and D3D11 Lab, preserving r6's open-water appearance. Twenty matched frames cover the fixed scenes and coastal holdout; eight offline/GPU comparisons and four exact disabled controls support the implementation. Four shifted-camera probes preserve offscreen-object reflections; sixteen focused Metal/D3D11 comparisons pass with identical Windows repeats. The extra local Metal GPU cost is 3.2–7.5 ms in eight-frame samples. Native delivery, provider halo/culling coverage, multiple river elevations and visual review remain open; implementation notes are ready. The [earlier water exploration](terrain_lab/v2/audits/beauty/WATER_EFFECTS_EXPLORATION.md), [surface-richness campaign](terrain_lab/v2/audits/beauty/SURFACE_RICHNESS_CAMPAIGN.md), source-normal/AO work and frozen pickup remain preserved. Milestone and integration gates are unchanged.

The [ground decal source pass](terrain_lab/v2/audits/beauty/GROUND_DECAL_PASS.md) recovers 19 exact grass/plains triangle/UV variants and the selected override textures. Twenty matched diagnostic frames include a wholly unseen 100-tile holdout. The visual gain is subtle and GPU cost excessive, so no new best or Integration promotion is recorded. Effective high-ground layering and complete mountain channel projection remain next; the all-applicable-texture audit is still incomplete.

The [mountain material pass](terrain_lab/v2/audits/beauty/ROCK_CHANNEL_PASS.md) aligns projected color/height/specular and adds eight snow/stripe channels in sixteen matched combined diagnostics. The gray rock relief improves modestly; normal-pipeline bindings, crop/wrap and cost remain open. All 72 test.biq mountains have grass base terrain, so desert coverage is an explicit synthetic material witness. Next, trace continental ground geometry together with the flat/high/hill material graph.

The [continental/source-baking pass](terrain_lab/v2/audits/beauty/CONTINENTAL_GROUND_PASS.md)
recovers continental height fields and rejects a broad pale high-material mask.
Installed bytecode confirms alpha-squared weighted material baking and separate
normalization; a bounded decoder matches 3,029 Microsoft-disassembled instructions.
Sixteen candidate frames and eight exact disabled controls are recorded. The
source-weighted diagnostic improves no overall grit and is not promoted. Cached
height-to-normal/AO processing and source layer contributions remain the next
investigation; the all-applicable-texture audit and LQ0 remain incomplete.

The preceding Lab goal covered river, forest and jungle quality.
The [campaign brief](terrain_lab/v2/audits/beauty/RIVER_VEGETATION_CAMPAIGN.md)
records the target appearance, actual displayed-code findings and the coupled
implementation sequence. `shadow-receiver-r1` remains the retained visual
baseline; the integration preparation package remains frozen. The first loop
now has 16 canopy comparisons and 16 combined `river-corridor-r2` frames across
the fixed coastal, inland and wilderness regions plus a new 100-tile forest/
jungle witness. Source-stable canopy ordering, connected source pools, relief-
aware curves and beach-crossing outlets are implemented as opt-in candidates.
Offshore pipe caps were rejected and corrected. Actual source terrain matches
across 3,888 shifted-crop samples. Banks, source-pool shape, corridor-aligned
clutter and broader coverage remain active work; see
[the first-loop findings](terrain_lab/v2/audits/beauty/RIVER_VEGETATION_PASS_r2.md).
No visual acceptance or milestone advancement is claimed. The subsequent
[bank-rock correction](terrain_lab/v2/audits/beauty/RIVER_BANK_ROCK_PASS_r3.md)
produces `river-corridor-r3`: the same 16-frame matrix with source rocks placed
from the actual corridor and all other source instances fixed. This is a small
placement improvement; bank outline and pool shape remain the larger gaps.

At the user's request, the retained Lab v2 findings are consolidated for native
implementation in [the terrain/lighting preparation package](handoffs/candidates/lab_v2_terrain_lighting_r1/README.md).
Its source snapshot, 32 frame hashes and local asset inventory verify. The
package maps the coupled changes to the evolving native renderer, including
vertex/binding differences, cache dependencies, rejected experiments and parity
requirements. This is preparation, not visual promotion or a native code change;
existing approval/refresh gates remain unchanged.

Lab v2's 2026-09-06 receiver correction retains `shadow-receiver-r1` as
incremental work in progress on top of the larger `relief-size-r3` bodies.
A bounded shadow-texel normal offset substantially reduces the thin false
shadow lines across dunes while preserving visible forest and mountain shadows.
Seven real-map regions, including a new 100-tile desert/forest/mountain holdout,
and a separate synthetic volcano witness have noon/midnight frames at both
zooms: 32 retained frames. Each matched pair has byte-identical geometry,
materials, placement and shadow-field packets. Previous bests and rejected
experiments remain preserved. Direct volcano source-channel inspection exposed
an incorrect height-to-normal interpretation; its first normal reconstruction
was not a convincing visual improvement and is not selected. Source dune
reconstruction, residual facet artifacts, relief projection, soft shallows and
cliff joins remain open. See `terrain_lab/v2/audits/beauty/CURRENT_VISUAL.md`
and `SHADOW_RECEIVER_PASS.md` there. LQ0 remains ready/unaccepted.
This does not close LQ0, assert Civ VI quality, request Integration promotion
or record human approval.

Production performance maintenance is consolidated as the API 13 Windows test
build at the user's 2026-09-06 request. It preserves production system coverage
and does not promote Lab v2, advance I19, or change LQ0. Indexed world-tile meshes,
validated bitmap overlap and 128-pixel image blocks reuse appearance independently
of native screen anchors. One worker prepares nearby authoritative meshes and
small off-screen image blocks with foreground priority and nonblocking readback.
Budgets are 192 MiB for GPU meshes (including a 64 MiB unused-prefetch sub-budget),
128 MiB for recent viewport bitmaps and 16 MiB for pixel blocks/metadata. Fixed-jump
replay with 400 visible meshes builds/uploads none after preparation and measures
12.451–24.604 ms per jump. Foreground redraw p95 during preparation is 2.356 ms;
a 54.392 ms maximum outlier and cold-area costs prevent a vanilla-speed claim.
Full Windows verification, both zoom previews and injected compilation pass.
The final test build emits per-map OutputDebugStringA capture/composite/whole-map
summaries; `Renderer/TEST_IN_GAME.bat` enables all DLL phases and a bounded log.
The user will launch Civ III and share logs and experience. Further optimization
is paused for that checkpoint; do not launch the game. See
`evidence/integration_cache_worker/README.md` for the build evidence, memory bounds,
remaining parity/performance limits, test launcher and log locations.

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

M6.7/I18 is complete. Build-time contracts consume the frozen approved L9
through L18 handoffs; the production DLL contains only deliberately integrated
paths and performs no runtime handoff-approval decision tree. It loads generic
terrain, dune, vegetation, marsh, volcano, clutter, river, shore-feature,
shared-environment, route, resource, city, wall, and mine payloads and uses the
approved formulas, topology, densities, anchors, materials, shadows, lighting,
and emissives. L19 farm/irrigation and tundra changes remain absent.

Authoritative Civ III capture supplies real/base terrain, SquareParts,
visibility, canonical coordinates, and distinct screen occurrences. A bounded
terrain-only cache fingerprints target, zoom, anchors, wrap, environment,
content revision, renderer-owned map state, ownership, and device generation
independently of dirty hints; the dirty clip and retained unit/UI state are
deliberately excluded. For
the current user-directed diagnostic stage, an active custom frame
owns the complete Civ III `m19` map plane after using that function's
authoritative capture/composite boundary. Configuration off is fully vanilla;
configuration on never
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
I11. The frozen L9 terrain through L18 mine handoffs load in the
standalone production fixture, and the Windows smoke
proves both zooms, clipping, scrolling, wrapping, cache invalidation, device
reset, exact terrain/feature/river/route/resource/city/mine ownership,
authoritative lighting phases, and zero native fallback. No new Civ III patch
symbol was required.

Post-I18 integration maintenance preserves the approved hashes while reducing
the native 400-tile/800-record cold benchmark from 57.790 seconds to roughly
5-7 seconds, a uniform anchor-only scroll to 0.026 seconds, and a one-pair
tile-boundary change to 1.948 seconds. The fast paths combine shared-grid
sampling, a 96 MiB anchor-independent world-tile sample/shadow LRU, river-pass
isolation, large-live shadow LOD, retained generated layer geometry with
vertex-adapter XY/depth translation, bounded immutable D3D vertex-buffer
regions, a 32-entry/128 MiB exact viewport LRU, and a persistent clip-only GDI
blit surface. One dedicated DLL worker owns immutable snapshot consumption,
renderer state, and D3D work; Civ III's thread still owns capture and the final
serialized GDI blit, and the synchronous ABI never presents a stale result.
API v9 returns the exact
authoritative clip and blits only that dirty rectangle, preventing cleared
off-clip pixels from truncating retained neighboring tile art. Custom-on `m71`
now captures a complete visible traversal even for a partial unit/UI redraw, so
such a traversal cannot poison the full-viewport cache; partial and reordered
static-terrain traversals are zero-tick cache hits. Visible native unit animation
and other retained selectors no longer invalidate static terrain, and an
memory-bounded exact-signature LRU makes recent unit-jump camera views zero-tick
hits while remaining deterministically bounded. Feature bodies use the frozen
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

## Next Step: LQ0

Current authorization (2026-09-06): one lead owns the complete Lab v2 path.
Former Q0-Q8 tasks were idle at takeover. Preserve their work and namespaces;
exclusive-owner/start blockers are superseded. The current combined checkpoint
and short defect list live in `terrain_lab/v2/audits/beauty/CURRENT_VISUAL.md`.
LQ0 remains ready/unaccepted; no acceptance or Integration gate advances merely
because ownership changed. Ordinary visual iterations run on Mac.

The current user-directed terrain pass fixes three 100-tile `test.biq` windows
(coastal, inland and previously untuned wilderness) with unchanged source tiles,
camera and output sizes. Combined mountain material projection, river optics,
selected-skin hill height, missing desert material bindings, desert boundaries
and shared night lighting are being compared at both gameplay zooms. Earlier
baselines and rejected shoreline experiments are preserved. Cities, units and
improvements are deferred for this pass. This is visible improvement work, not
source-fidelity or beauty acceptance; dunes, source cliffs, shallow-bed detail
and vegetation composition remain unresolved. Reproducible images and evidence
are linked from the current review; the earlier city checkpoint remains history.

Resource incident diagnosed: the campaign-test `_copy_campaign` helper copied
the entire v2 runtime tree into temporary directories, expanding cache hardlinks
and exhausting disk space. It now copies only declared campaign metadata,
prompts, reference catalog and policy documents, with a metadata-only regression
test. This was a test-harness defect, not evidence of renderer working-set size.

Q8 combined review is now active under the coordinator. The first actual Mac
coastal comparison is `terrain_lab/v2/audits/beauty/REVIEW.md`: matched verified
terrain/city/route recipe and camera, before vs current Q2/Q3/Q6 composition.
It is deliberately unaccepted: source cliff/shallows, current city assembly,
and route/vegetation clearance are the three visible priorities. Prefer bounded
fixes in this picture to expanded isolated matrices while these failures remain.
Report visible improvements separately from enabling engineering and unresolved
defects. Per-run evidence is not a waiver of full visual or promotion gates.

Shadow-shape requirement confirmed in the Q6 task: every cast shadow follows
the actual transformed/posed mesh and authored cutout alpha, with Q6 lighting
and actual receivers. Generic blob/oval/ribbon substitutes and omission cannot
close visual acceptance. Preserve frozen history; replace legacy approximations
only in versioned candidates. COMMON.md and Q8 review enforce this across owners.

Placement-clearance update: Q5 publishes final road/rail clearance envelopes,
Q3 river/bank exclusions, Q4 places forest/jungle around transport corridors,
and Q7 places city buildings/walls around river and transport corridors. Q0
supplies shared plumbing and Q8 checks composed results. See
`terrain_lab/v2/PLACEMENT_CLEARANCE.md`; local interface witnesses permit parallel
work without altering authoritative city anchors, routes, or river topology.

Source-art policy update: use Civ VI source art and the explicitly selected
source skin before inventing substitutes. Q4 must investigate source cliff/rock
pieces and source-height/material construction, not autogenerate dominant rock
faces as the beauty solution. All tracks distinguish source reuse/adaptation
from diagnostic proxies; original-art fallback selection needs explicit user
authorization. See `terrain_lab/v2/SOURCE_ART_POLICY.md`. Continue independent
source-backed work while missing asset/import capabilities are resolved.

Gameplay-first evidence update: Lab v2 beauty reports now lead with plausible
actual-gameplay-scale contextual views, followed by focused A/B diagnostics.
The frozen all-assets diorama remains regression/stress evidence, not the main
beauty target. Q8 first publishes small runnable shared placement recipes using
available inputs; all owners continue with labeled local contextual proxies
until available. See `terrain_lab/v2/GAMEPLAY_BENCHMARKS.md`. This introduces no
new launch gates or per-edit full-matrix/Windows requirement.

Scheduling override: the user has explicitly authorized Q0-Q8 to start and run
in parallel. LQ0 remains the next global closure checkpoint, while LQ1 is in
progress. Track `dependencies` are empty; `integration_inputs` name results to
adopt at convergence. Q0 completes the platform and verified test.biq registry
while owners use current Mac interfaces, frozen inputs, or declared local proxy
fixtures. Q6 supplies color/alpha semantics directly to Q0 without waiting for
Q0 acceptance. Missing inputs and parity remain pending evidence, not blanket
stop conditions. Final platform/real-map/composition gates are still required.

Renderer Lab v1 is complete through L21. Its final revision adds
single-main-color civilization territory ribbons with no same-owner seams, and
extends the shared authored-normal/self-face plus directionally consistent
source-mesh cast-shadow system to cities, walls, mines, farm buildings, goody
huts, colonies, fortresses, barricades, airfields, outposts, radar towers, and
victory locations. Flat borders, pollution, and crater decals correctly remain
non-casters. At the user's explicit request, the last full validation matrix was
waived and the visually inspected result was frozen immediately. Those L9-L21
handoffs are now the immutable Lab v1 baseline.

The user has explicitly reopened standalone appearance work as Renderer Lab v2
before I19. Integration is paused at I19 while LQ0 modularizes the Lab and
establishes a headless macOS Metal fast path with D3D11 promotion parity. LQ0 is
a zero-intended-visual-change step: it must preserve the frozen v1 results while
adding stable system ownership, render-packet and composition contracts,
namespaced outputs, content-addressed caches, microfixtures, and quick/check/
compose/promote workflows. The detailed campaign is in
`docs/renderer_lab_v2.md`; the machine-readable work packages and reusable
agent prompts are under `terrain_lab/v2/`.

Renderer Lab v2 has one recorded terrain-family audit prerequisite in
`docs/terrain_visual_direction.md`: revalidate grassland, plains, desert,
tundra, and flood plains first, then relief, vegetation/wetland, and water families plus all
cross-family transitions. Tile-shaped brown or dark smudges are explicitly
logged as invalid overlay/shadow leakage rather than legitimate terrain.

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
L21/I21 are the historical v1 combined release-level gates. Lab v2 produces
separately versioned replacement candidates and never overwrites them. See
`docs/renderer_workstreams.md`, `docs/renderer_lab_v2.md`, and
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
15. L14 / I14: Complete — authoritative road topology and visible-era style drive the approved connected route and bridge pass.
16. L15 / I15: Complete — authoritative railroad topology drives the approved sleeper, ballast, rail, coexistence, and river-crossing pass.
17. L16 / I16: Complete — visibility-conditioned Civ III resource identity drives the approved land and aquatic resource bodies while resource UI remains native.
18. L17 / I17: Complete — authoritative city owner/style/era/size/wall/capital state drives the approved city body while labels and status overlays remain native.
19. L18 / I18: Complete — authoritative mine state and visible era drive the approved terrain-following mine families, variants, shadows, and emissives.

Production now compiles a frozen copy of the approved standalone renderer's
`PSMain` and `PSFeature` functions. Production retains only a Civ III
scene/input adapter and semantic terrain settings around that copy; it does not
include the live Lab shader, so in-progress visual work cannot enter the game
before its handoff. The current integration freeze ends at L18; approved L19
farms, irrigation, and tundra changes remain Lab handoff input until I19.

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
14. L14 / I14 complete — the approved 98-node / 109-edge Lab road graph uses gently curved exact-node centerlines, source-backed continuity coverage, normalized bridge bodies, wrap continuity, four styles, and pillage coverage without changing the approved L13A control.
15. L15 / I15 complete — the approved connected railroad subset uses narrow authored sleepers/ballast, paired source-colored rails, exact river bridges, road coexistence, relief, wrap, and deterministic both-zoom evidence without changing L14.
16. L16 / I16 complete — the approved normalized resource gate covers strategic, luxury, land-bonus, and aquatic-bonus bodies, corrected Civ III-scale clustering, visibility suppression, and deterministic both-zoom evidence without changing L15. Its final L21 lighting revision gives every land resource authored-normal face separation and a readable source-scaled cast/contact shadow while preserving submerged fish without a false water-surface shadow.
17. L17 / I17 complete — the approved normalized city gate covers four eras, all three Civ III size bands, culture/owner metadata, wall/capital states, compact source-backed compositions, retained-label clearance, shared lighting, and source-authored night emissives without changing L16.
18. L18 / I18 complete — the approved recursive normalized mine gate covers preindustrial/industrial families, three variants, terrain-following excavation, mineral and relief adjacency, coherent compound shadows, source emissives, and deterministic isolation without changing L17.
19. L19 complete / I19 handoff available — the approved crop-first farm gate covers all sixteen irrigation masks, four eras, mixed terrain and adjacency cases, source-only sparse accents, and the independent alternate-skin tundra material path without changing L18 when farms are disabled.
19a. L19A complete / I19A handoff available — the approved source-backed hut/colony gate covers viewer-hidden huts, the deterministic eight-to-three variant map, all four colony eras and owners, extraterritorial ownership, resource coexistence, both zooms, and shared lighting without changing L19 when disabled.
19b. L19B complete / I19B handoff available — the approved pass covers all remaining persistent tile infrastructure, both zooms, shared lighting, and source-only night emission without changing L19A when disabled.
20. L20 complete / I20 handoff available — representative ordinary, mounted, crewed, vehicle, naval, air, worker, and Army bodies pass the 192-tile/two-zoom/action matrix without regressing L19B. Its final L21 lighting revision gives every visible formation member authored-normal face separation and a readable source-scaled cast/contact shadow using the shared environment direction.
21. L21 complete: the corrected complete alternate-skin beauty scene includes one-main-color territory borders and unified self/cast lighting for raised map objects. The final territory revision was visually inspected and frozen by explicit user direction; the remaining validation matrix was waived.

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

Worker specialty-action intake is now frozen without advancing L20. The native
job mapper proves that current FLC type is insufficient: irrigation and damage
cleanup share one slot, road and railroad share one slot, fortress and
barricade share one slot, and airfield/radar/outpost present DEFAULT. A checked
source-to-generic compiler therefore maps all 13 `Job_ID` values, state
fallbacks, CAPTURE, and generic BUILD; it also emits an exclusive action-
selected Tool group. Eleven installed Builder motions (three primary work,
four optional repair, four capture) successfully normalize with one Root group
and 41 tracks. Body/tool import, pose caches, all-facing/two-zoom rendering,
VFX calibration, and approval remain L20 work. See
`docs/worker_builder_animation_mapping.md`.

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

## Animation gameplay checkpoint ready (2026-09-07)

The tested animation DLL is staged for normal INSTALL.bat. The GOG unit inleads
and corrected reduced signature compile successfully. Ten resource families
animate facing southeast; nine custom unit families follow native cursors and
anchors, now including Warrior, Scout, Settler and Worker/Builder. Civilian
coverage is idle/move/fidget/fortify-stop/capture; specialty Worker jobs and
unsupported families retain native bodies. The extension fixes scientific
notation in material numbers (which hid Scout) and gives the staffed Settler
compatible staff-humanoid clips. All 852 source-pose samples and 576 movement /
912 action draws pass; existing cache budgets remain unchanged. Both zooms, day/night, action
interruptions, held endpoints, RGB555/RGB565 clipping, config-off, cache reuse
and terrain preservation pass. The agent did not launch Civ III and stops for
the batched user-run checkpoint in `docs/animation_integration_checkpoint.md`.
The full workflow was run but remains non-green on the unrelated frozen L19A
pack hash; the known legacy frozen-profile boundary failure also remains open.
No Lab gate was advanced or regression threshold changed.

## Animation integration foundation (2026-09-06)

The user authorized southeast-facing animated resources and enabled unit bodies,
with the renderer in the DLL and a later user-run gameplay checkpoint. The generic
validated skin-palette compiler/evaluator now covers 26 subjects in 10 resource
families (8.95 MB payloads); 104 authored samples match normalized CPU skinning
within 6.575e-7 tiles. Portable and Windows x86 playback/rejection checks pass.
Resource dynamic composition, facing calibration and the native Animator unit
bridge remain in progress; production animation is not enabled or staged yet.
See `docs/animation_integration_checkpoint.md`. Terrain caches, the installed
performance DLL, injected sources and the deferred fog-edge work are unchanged.

### Resource dynamic pass and importer correction

The animation candidate now renders ten resource families in the DLL, preserving
static terrain meshes/pixels and reusing exact MSAA4 color/depth behind moving
bodies. Its additional backdrop cache is capped at 24 MiB. Source-backed 1/12-to-
1/100 translation normalization and constant root-placement removal fix stretched
clutter poses discovered by actual renders. Timed normal/reduced noon/night
witnesses change poses with zero terrain builds/uploads; fixed-scroll and removal
comparisons match fresh output within existing pixel tolerances. Ordinary debug
output reports resource bindings, demand, timing and cache costs. See
`docs/animation_integration_checkpoint.md`. Unit movement integration is still
required; the resource candidate remains verification-only and is not staged.

### Complete unit clip export (2026-09-07)

The animation candidate now has a complete-clip unit export for six families and
48 actions, including skinned bodies and animated socket attachments. The actual
C++ evaluator matches 576 normalized source poses within 2.256e-7 tiles. Planar
root travel is removed consistently across the kit so Civ III alone positions
moving units. Native cursor/anchor guards pass the Windows x86 build; live body
rendering/hooks, marine heading correction and the combined gameplay checkpoint
remain unfinished. No animation DLL was staged and the game was not launched.

### Barbarian camp offline intake

Barbarian camps now have a separate source and runtime contract rather than
sharing goody-hut or colony semantics. The exact installed chain resolves
`IMPROVEMENT_BARBARIAN_CAMP -> LM_BARBARIAN_CAMP` to primitive `VIL_BAR_01`
and optional later `VIL_BAR_IND` roots. Both recursively normalize into the
generic tile-object pack and the compact proof bundle builds. The primitive
root remains the all-era Civ III default; viewer-conditioned presence, tribe-
stable optional-child selection, neutral color, independent barbarian units,
resource coexistence, immediate native removal, stable diagonal placement and
authored-vs-source night effects are checked in
`docs/barbarian_camp_import.md`. Two optional skull-pile child references are
explicitly omitted because their source material lacks a required base-color
channel; the primary camp composition is intact. This preparation neither
reopens the completed L19A handoff nor advances LQ0. A dedicated modular Lab v2
visual gate is still required before Game Integration may own camp pixels.

### Animation body/config and southeast calibration increment

The candidate DLL renders five complete unit families through a separate
8 MiB sprite cache. The thin native bridge and ordinary
`enable_custom_rendered_units` configuration forwarding compile successfully.
Real-DLL day/night witnesses cover both zooms, cached anchor translation,
16-bit clipped bodies, config-off preservation and unchanged retained terrain.
Resources now have per-sample whole-body southeast calibration, including all
12 fish and three whales without breaking cross-rig weights. The runtime
environment switches are removed; animation is still candidate-only.
The three existing GOG CSV definitions need the audited inlead/signature changes
in `handoffs/animation_unit_hooks_gog.md` before live unit activation. No animation
DLL has been staged and Civ III was not launched. See
`docs/animation_integration_checkpoint.md` for coverage and remaining gates.

### Lab city source-normal diagnostic

Opt-in r29/r30 decode the tested static source normal bytes without changing
geometry, coordinate sets, material bindings or lighting. Coastal and inland
pixels change subtly; no new visual best or promotion is claimed. Six standalone
Windows comparisons and ten focused tests pass. The next material work remains
tangent/LEAN/gloss interpretation, broad grounding and local night light transport.
See `terrain_lab/v2/audits/beauty/CITY_SOURCE_NORMAL_PASS.md`. Integration state
and all milestone gates are unchanged by this Lab experiment.

### Lab city source material restoration

The installed shader audit resolves authored tangent-frame and normal-Z decoding
and cooked dual-lobe roughness roles. r31–r33 apply the bounded material adapter
with modest visual gains and exact disabled controls; ten Windows comparisons
pass. The audit also identifies omitted source metalness/opacity slots. Normalize
those and recover variance/environment/local-light contributions next. City
quality and all milestone gates remain open; integration code is unchanged by
this Lab pass. See `terrain_lab/v2/audits/beauty/CITY_SOURCE_SURFACE_PASS.md`.

### Lab city opacity and wilderness clearance

r34 restores small roof openings with exact r36 disabled controls. Direct-only
metalness r35 is unselected because environment specular is missing. The fixed
wilderness site exposed forest overlap; a four-body stage now clears vegetation,
but seven-body growth failed 33 bounded alternatives and remains unresolved.
Eight Windows comparisons and twelve focused tests pass. One-era style, broader
palace intake, earlier candidates and all gates remain preserved. Next work is
coherent urban composition, environment lighting and a broader placement strategy.
See `terrain_lab/v2/audits/beauty/CITY_EXTRA_MATERIAL_PASS.md`.

### Lab city growth and shadow composition

A constrained-first layout search resolves the seven-body wilderness city while
preserving a four-body growth prefix. The eleven-body inland case fits; the
wilderness large stage remains unresolved. A previously untuned city region
(freshshadow) still looks crowded, so no general city recipe is accepted.
Matched comparison also exposed all-scene shadow-grid refitting: retaining a
checked prior grid removes unrelated terrain pixel changes. The selected local
candidate changes 8,515 day / 7,130 night pixels, with exact outside-city identity.
Ten Windows comparisons and seventeen focused tests pass. Native code and gates
remain unchanged. See `terrain_lab/v2/audits/beauty/CITY_GROWTH_PASS.md`.

### Lab city local night-light spill

Facade-light r3 adds bounded warm light around actual emissive lower facades,
using unchanged scene packets and the shared night activation. It is an authored
source-informed approximation, not recovered source light binding. Five combined
cases retain daylight within isolated 1/255 rounding; blocker/reflection controls
confirm local effects and preserved capital lake reflections. Fourteen Windows
comparisons and twenty focused tests pass. Layout, urban ground, environment
specular, broad city coverage and native delivery remain open. No gate advances.
See `terrain_lab/v2/audits/beauty/CITY_FACADE_LIGHT_PASS.md`.

### Lab city era grounding material

Ground-binding r1 replaces default dirt pads with source-era paving on the five
fixed facade-light candidates. Geometry, UVs, shadows and light shaders stay
matched; the gain is visible but small. Fourteen Windows comparisons pass, and
the corrected no-op packets/images match exactly. Broader urban ground, source
height/state behavior, city layouts and all gates remain open. See
`terrain_lab/v2/audits/beauty/CITY_GROUND_BINDING_PASS.md`.

### Lab connected modern city ground

Settlement-ground r2 joins the modern building bases with a projected footprint
union, reusing source paving and keeping its coordinates fixed through growth.
Twelve Windows comparisons, six analytic tests and sixteen packet isolation
checks pass. The medieval attempts are rejected; broad flat stone remains worse
than the prior best. Next, improve the small-city growth hierarchy and broaden
culture/era coverage. All gates remain open. See
`terrain_lab/v2/audits/beauty/CITY_SETTLEMENT_GROUND_PASS.md`.

### Lab single-era city growth hierarchy

Growth r51/r53-r57 keeps smaller modern cities low and reserves the tallest
compound for the large stage, with exact earlier-building prefixes and unchanged
source scale. The recipe also reduces freshshadow congestion without local growth
tuning. Twelve Windows gameplay-size comparisons and seven focused layout tests
pass; independent packet/frame controls preserve composition. The large wilderness
fit remains unresolved, and the new wilderness reflection witness is weak.
Next broaden single-era culture/era/capital coverage and facade response. Previous
candidates and all gates remain preserved. See
`terrain_lab/v2/audits/beauty/CITY_GROWTH_HIERARCHY_PASS.md`.

### Lab individual-house culture density

Asian medieval and ancient-brick medium cities now use sixteen source houses
at unchanged scale, producing more readable neighborhoods than their matched
seven-house controls. Twelve Windows frames complete; four baseline Metal
comparisons pass and eight denser-frame comparisons remain pending. Prolonged
Metal compilation persisted after reducing spill proxies; static shader data
needs a different implementation. The large scattered layout is unselected and
future large holdout placement remains unresolved. Twelve focused tests pass.
No native changes or gate advancement. See
`terrain_lab/v2/audits/beauty/CITY_CULTURE_DENSITY_PASS.md`.

### Lab buffered city lights

Generic b1 frame data resolves the dense-city Metal compile bottleneck without
backend changes. Six full replays take 2.58-5.24 seconds. Eight Windows transport
controls are byte-identical to their prior static-array frames, superseding the
pending dense-city comparisons; all twelve new Metal/D3D checks pass. Full
53-proxy spill improves local night readability on inland and holdout medium
cities while daylight remains exact. Geometry, material and shadow-prefix checks
pass. Large scattered growth remains unselected; next improve connected city
placement and broader culture/era/palace coverage. No native or gate advancement.
See `terrain_lab/v2/audits/beauty/CITY_LIGHT_BUFFER_PASS.md`.

### Lab connected city growth and river clearance

Connected Asian large growth preserves the original sixteen houses and joins
its last eight into a coherent neighborhood, reducing bounding area 24.5%.
The river holdout revealed missing corridor clearance in old and new candidates;
generic convex bank exclusions now produce a clear eight-house correction.
Sixteen-house river growth remains budget-exhausted. Ancient large growth and a
previously city-untuned 100-tile coastal case provide additional coverage. Twelve
Windows comparisons, independent terrain/light/frame checks and fifteen focused
tests pass. Next broaden single-era palace landmarks and culture coverage;
river-site medium/large growth and material richness remain open. No native or
milestone advancement. See
`terrain_lab/v2/audits/beauty/CITY_CONNECTED_GROWTH_PASS.md`.

### Lab single-era palace composition

The broader palace library now supplies East Asian medieval and ancient-brick
capitals. Staged house connectivity and shared palace frontage improve the
small capital and preserve exact 8/16/24 growth. Dry-land-centered placement
adds main coastal coverage without changing the inland layout. The separate
capital-untuned coastal holdout remains unresolved; its failures are retained.
Twelve Windows comparisons, 36 independent composition checks and 23 focused
tests pass. Removed 614.3 MiB of completed new linear readbacks, retaining all
images and replay inputs. Facade/material variety, broader city coverage and
all gates remain open; no native or milestone advancement. See
`terrain_lab/v2/audits/beauty/CITY_PALACE_COMPOSITION_PASS.md`.

### Lab city environment material trial

An authored sky/ground reflection fallback enables the existing modern
metalness maps without leaving facades starved of indirect light. The r2
inland-large and wilderness-medium cases provide modest local material gains.
Source-family roughness attenuation fixes part of the initial washout, but the
Asian dielectric roofs remain unselected. Eight current Windows comparisons,
four independent material-byte inspections and a pixel-exact disabled control
pass. The older American capital lacks the newer material bindings; restore
that complete intake next while preserving its lake reflection witness.
The analytic fallback is not source environment parity or a global default.
Removed 249.0 MiB of new completed readbacks; all images and replay inputs
remain. No native or milestone advancement. See
`terrain_lab/v2/audits/beauty/CITY_ENVIRONMENT_PASS.md`.
