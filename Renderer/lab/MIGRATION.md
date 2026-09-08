# Renderer migration: current state

The goal remains incomplete. The accepted baseline is commit `65b02cfc`
and the staged DLL recorded in `baseline.json`. Integration finished before writes.
The staged DLL and injected source are untouched. The candidate DLL has now been
rebuilt with shared CPU-data code; it is not byte-identical to the old DLL.
Preserve `docs/visual_fidelity_playbook.md` and the deferred wonder/District contracts.

## Working foundation

- All 20 categories have focused/context D3D11 references at approved/integrated
  revision 1. No new approval or live-game pass has been invented.
- `Renderer/renderer.py` supports list/show/affected, lab/compare/gallery, tests,
  explicit approval, candidate build and integration pending/verify/record.
  Complete Lab/compare runs select dependent categories and consumers whose
  source inputs changed, even when another category was requested. Asset read
  closures and current fixture content route city/resource/unit edits; unknown
  shared inputs remain conservative. One reviewed-input signature per category
  is tied to its approved revision. Only an exact complete comparison against
  approved images or explicit approval can update it. Neither partial nor
  different-pixel comparisons silently clear the affected review set.
  Approval rejects
  partial, stale or modified results and missing affected appearances.
- Native build receipts follow quoted include closures. The standalone witness
  rebuilds when its source or binary changes. The VM dispatcher uses the shared
  UNC checkout, not a drive-letter assumption. Builds/replays do not stage,
  install or launch Civ III.
- City asset intake, layout, coverage and facade lighting are under
  `lab/shared/cities`. Preserved inputs are in ignored
  `packs/CityFidelitySources/current`. Rebuilding 122 models/37 materials/72
  templates preserved the 7,244,120-byte city.bin exactly.
  City preparation is now part of the common command. Its observed read closure
  includes catalogs/meshes/materials/textures and shared Python helpers (941
  current files), excluding generated scratch frames. Component caches clear
  between builds; repeated reads of changing bytes fail instead of mixing inputs.
- The 24 currently used shader source modules are under `lab/shared/shaders`;
  shared hydrology code/data are under `lab/shared/hydrology`. Rebuilding
  the current generated shader/header/natural payloads was byte-identical.
- Category lab/test/build and integration verification now run the five shader
  adapters through one cached preparation step. A disposable source mirror
  prevents failed generation from partially replacing current files. Missing
  outputs regenerate; edited generated files are preserved as conflicts.
  Comparison, approval and integration recording reject unprepared inputs.
  Shared-shader iteration no longer rebuilds the natural asset pack. Its metadata
  now records actual pack inputs rather than unrelated shader/CPU versions.
- The same command now prepares natural and hill/cliff assets incrementally.
  Builders generate temporary output, validate all conflicts and source hashes,
  then publish changes. New texture paths introduced by a source bundle become
  tracked dependencies. Missing output regenerates; unknown edits and symlink
  output aliases are rejected. The 48 natural DDS files were detached from
  source hard links without changing their bytes. Future source edits therefore
  cannot silently mutate the generated runtime copies.
- Unit fidelity packaging now uses the same incremental workflow. It builds
  disposable output from the existing animation runtime and authored-normal
  catalog, preserving 78 units/94 native aliases. All 2,935 generated files match
  the prior runtime bytes; 497 hard-linked files were detached without content
  changes. The 3,689 observed inputs include the absent procedural source catalog
  whose presence would require imported-component normal authority.
- Static resources and resource animation also use incremental preparation.
  The current normalized sources reproduce the static bundle and 67 animation
  files; 24 clip translation scales now live in a small shared recipe rather
  than a historical preview report. Preparation tracks 34 static and 201 animated
  inputs. Obsolete top-level pending/enablement flags were removed from generated
  animation metadata; category approval remains authoritative. The old extraction
  report is retained because upstream source conversion still consumes it.
- Category renders and integration verification now automatically build missing
  or stale candidates using the actual compiled include closure. Current builds
  are reused; input/receipt errors do not trigger speculative rebuilds. A build
  must pass its freshness check before rendering. Nothing stages the DLL.
- `lab/shared/natural/data.h` now owns the production natural-pack decoder,
  height sampling and terrain/mountain/object lighting-frame calculations.
  The D3D runtime inherits this API-independent data and retains GPU ownership.
- `lab/shared/natural/vertex.h` and `ground.h` now own the 168-byte vertex,
  production projection, normal/material sampling and 16x16 terrain grid.
  The native compiler calls this code; a Mac executable exercises it without
  Windows or a GPU. `natural/queries.h` now supplies the production neighborhood,
  coast, material weights and combined height query. Native code retains its
  world/coast invalidation observations and reusable scratch cache.
  `natural/world.h` now owns the same 16-page river cache, while `relief.h`
  owns CPU source fields, normalized sampling and flat/ground-cache/height-only
  relief queries. D3D data inherits these shared CPU fields; native callbacks
  still supply authoritative river/activity observations and asset loading.
  `natural/mesh.h` and its shared statement bodies now own the production hill
  decals, mountain grids and forest body emission/clipping. The native compiler
  includes those bodies at their existing location; portable tooling uses typed
  adapters. Preserving the native statement context avoids observed x86 rounding
  drift from an extra mesh-function boundary. City-bound collection and cache ownership remain in
  the native adapter. `patterns.h` supplies the same placement hash/random and
  analytic dune field to both native and Mac callers. The full Mac render graph
  is not done.
- All 26 hill/cliff source inputs are preserved in ignored
  `packs/TerrainProfileSources/current`. The current asset builder reads them
  directly, without a historical handoff. Its rebuilt runtime payloads are
  byte-identical; only source-path metadata changed. Original source files remain
  until the wider dependency audit permits deletion.
- Reusable Metal/D3D11 backends, packet transport, compiler/cache, color response
  and tool bootstrap live under `lab/backends`, `lab/contracts` and `lab/shared`.
  Installed shader tools moved intact to `lab/.local/shader-tools`.
  Old imports/includes/build scripts are temporary compatibility for retained
  source probes, not independent renderer implementations.

## Verified evidence and limits

The latest full category suite passes 163 tests (126.4 seconds), including the
Mac pilot's input, scope, freshness and approval-isolation checks. Source-selection checks cover
actual city/resource/unit fixture consumers, global lighting/shadow/transition
changes, new/removed inputs and recipe/revision changes. Approval checks reject
missing, stale, partial or modified appearances. Exact complete comparisons may
record equivalent inputs but cannot change approvals, images or integration.

The current shared mesh/pattern candidate matches all 56 D3D11 views to approved
pixels exactly, across all 20 categories. The complete run includes city dawn
and desert gameplay, which exposed and verified the native statement-context
rounding constraint. All current category comparisons completed successfully.
Reference images and approval/integration revisions remain unchanged at r1.
A shared-folder error interrupted the forest capture. Windows confirmed no
native preview process remained; complete captures were retained and incomplete
categories resumed through the common renderer functions. The resumed run and
full image comparison completed successfully.
No DLL was staged and no injected source or approved reference was changed.

The inventory now uses cached directory-entry metadata and avoids rewriting an
unchanged hash cache. All 29,965 prior pack paths/hashes match exactly; local warm
inventory time fell from 1.26-1.51 seconds to about 0.40 seconds. This preserves
input coverage rather than dropping historical inputs before their audit.
Atomic JSON writes use unique temporary files. Focused affected runs apply the
requested case only to its category and use complete dependent recipes.
The focused workflow/dependency/preparation suite passes 72 tests, including
file/directory symlinks, added/removed inputs, unchanged cache reuse, concurrent
writers and differently named animation fixtures.

Shared-grid CPU verification passes 324 byte-identical full-vertex comparisons
at both tile zooms, multiple positions/heights and material/marsh weights. It
checks 289 corner evaluations, 512 triangle ordering, cancellation without
partial output and coastal clipping. The Windows candidate build passes and the
staged DLL remains unchanged. The complete native category comparison passes;
this is not a new live-game or full native behavior verification.

Shared surface-query verification compares the previous production expressions
with the common sampler over 18 owner scopes, including no wrapping, X/XY
wrapping and a terrain edit. Values, material weights, height/support, world and
coast dependency observations, and reusable-cache hit/miss statistics match
exactly. The complete category suite and all-category image comparison pass
with that shared-query candidate. The staged DLL remains the accepted baseline;
no live-game or full behavior-suite pass is inferred from image equivalence.

Shared river/relief CPU checks now compare 312 river samples, 1,440 normalized
source-field samples and 54,354 ordered query observations to the previous
production expressions. They cover world revision changes, wrapped maps,
16-page LRU eviction, flat certificates, cache hits/misses and uncached height
queries. Mesh extraction compares 7,807,710 vertices byte-for-byte and 320,578
ordered observations across 326 scopes against the previous production formulas,
including all five mountain variants, source-body clipping and cancellation.
Pattern checks compare 200,000 hash/random values and 132,612 dune samples exactly.
Both mesh and pattern comparisons also pass under production x86 MSVC, not only
the Mac compiler. The two views exposing the function-boundary rounding issue
(desert gameplay/noon and city gameplay/dawn) now reproduce approved pixels
exactly with the shared statement-body arrangement.
The candidate Windows build is current and all 163 category tests pass.

Native fixtures now write an invocation-bound exit record and a bounded log.
Parallels transport failures cannot certify old output. If completion is missing,
the dispatcher asks Windows for the actual preview process, rechecks for late
completion, and permits one retry only after the recognized no-process response.
Live, failed-query or unrecognized process state is not restarted. Focused tests
cover current/stale receipts, actual failures, delayed completion and bounded
retry. The current native resource gameplay check exercised an actual Parallels
failure, confirmed process absence, retried once and matched approved pixels.

The current preparation tests exercise actual adapters, all 50 generated shader
outputs and propagation of a shared terrain edit into production bindings.
Asset tests protect source/output separation, missing and conflicting output,
hard-link detachment, new/retired dependencies and changing source bytes.
The natural decoder tests include 14 malformed inputs and 24 lighting phases.
Portable C++ evaluation agrees with authored resource data for 26 subjects,
104 poses and 7,215 marine frames. The unit comparison covers 55 source-reference
units, 4,622 body poses and 1,611 attachment samples; the other 23 composed/original
kits are not included in that source-reference proof.

Packet wire 7 transports layered texture mips and b0-b7 vertex/pixel constants.
The Metal/D3D11 binding witness matches exact pixels and linear attachments.
All 15 production shader programs/30 stages compile on Metal. The actual Mac
grassland detail scene now runs through the common `lab ... --backend metal
--case detail` command and has a separate comparison path. It loads the current
natural pack (48 DDS textures plus an empty flat-scene shadow field), emits
170,496 shared production vertices for 111 visible tiles, binds the actual
168-byte CPU vertex fields, and uses the production terrain shader, mip bias,
MSAA and color response. It uses production relief queries and their flat-ground shortcut, and
rejects unsupported recipes, gameplay scenes, malformed worlds and unsupported
DDS layouts. Missing/stale/modified inputs cannot certify the comparison.
Against approved grassland detail, 97.078% of pixels match exactly and maximum
channel difference is 1/255. The common warm command took 12.26 seconds including
preparation, with 2.062 seconds in scene/render work. This is **not** complete
category parity: occluder shadows and the remaining render graph need wiring.
Results remain under `out/grassland/metal`, separate from
D3D candidates and all approval/integration records.
The internal context compiler now emits 239,082 production vertices in 128 draws
for the gameplay fixture's plains, hill, mountain and forest, with production
layer ordering and premultiplied object/decal blending. Its diagnostic image is
under `out/grassland/metal-context`; it has no approval record and is not exposed
as a completed category command. Maximum difference is currently 105/255, with
mean RGB differences of 0.987/0.888/0.406: its missing shadows are visible, not
accepted as parity. The next graph connection is the production world-aligned
source shadow atlas, including terrain/mountain cutouts and tree opacity.
Both opaque and cutout entry pairs from the current city-profile
`source_caster.hlsl` also compile on Metal. This is shader preparation, not proof
that the source atlas has been rendered or consumed.

Current baseline limitations must stay visible:

- Resource fixtures have black backdrop rectangles, road/rail fixtures have
  segment gaps, and the production renderer does not map Silks.
- The latest full native integration witness run passed six of seven behavior
  cases. Terrain-edit reuse fails with 387 built and zero reused, reproduced on
  both 32x32 and 100x100 worlds. Do not relax the assertion or change production
  behavior to make reorganization checks pass.
- The earlier candidate passed warm/cold scrolling, reduced zoom, wrapping and
  resource timing/scroll/removal. The unit matrix passed noon/midnight, 288 body
  draws and 564 interruption/held-endpoint draws per phase, both zooms,
  RGB555/RGB565 clipping/magenta, config-off, native cursor reuse and unchanged
  retained terrain. Nine matrix families are not the complete 78-unit roster.
- Integration is not an overall pass while edit reuse fails. Receipts remain
  under `lab/out/integration`; no live-game check was run. Category tests and
  exact images do not supersede that failed integration receipt.

## Cleanup already performed

Retired MASTER_PLAN/ROADMAP/project_status/VERIFICATION, state/campaign dispatchers
and their ledger-only tests, old Lab plans, ten Q-track prompts, 19 Q1 campaign
manifests, Q0 audit/guard/candidate files, 69 unused generated experiment shaders
and three historical pickup-verification wrappers. These deletions are
recoverable in Git. Approximately 1.1 GiB of regenerable old compiler caches was
removed; necessary local packs, source inputs and candidate references remain.
The profile helper README is now a compact implementation/operating guide rather
than a sequence of historical staging, screenshot and performance reports.
The unit fidelity guide now describes current inputs and contracts, replacing
obsolete profile-default and handoff claims. The unconsumed historical
`verification/environment_refresh/unit-fidelity.json` was removed; Git retains it.
The patch dependency ledger is now a current boundary/request record instead
of a sequence of campaign snapshots. Existing deferred effects requests and
wonder/District contracts remain intact. The unit contract now follows the
actual GOG CSV and injected wrappers, including native palette capture and
per-body Army fallback. Its old body-spike document was removed after preserving
the useful GOG address/call evidence in that contract. Frame-pacing and render-
loop references no longer describe the supplied GOG hooks as unresolved.
The animation checkpoint page is now a current delivery/calibration guide,
with obsolete nine-family roster claims, staging snapshots, campaign gates and
early import command history removed. Resource clip-unit/marine calibration,
unit/canvas/input safeguards and the batched game-check guidance remain.
The natural adapter guide now points at the category workflow and shared CPU
providers, corrects the obsolete hard-link claim and preserves the x86 calling-
context constraint alongside the existing source/material findings.

## Remaining implementation

1. Finish the Mac fast path using current production geometry, material inputs,
   shadows, reflection/composition and animation. Grassland detail now uses
   production geometry/materials; finish gameplay-context shadows and then every
   category. Old isolated Lab scenes are not baseline proof.
   The shared grid and WorldCoast-backed surface queries are available.
   The production river pages, relief source fields and cached/height-only
   sampler are now shared as well. Wire actual asset loading and the remaining
   remaining general river/activity and asset callbacks into the Mac scene compiler, preserving
   integration cache observations. Do not substitute flat terrain or separate
   approximate Lab blending.
   The next shadow connection must use the current city-profile caster shader,
   six-tile/1024-texel world-aligned R32 pages, maximum physical light depth and
   the existing b2/b4/t17 receiver bindings. Include offscreen casters, canonical
   world coordinates and wrap copies; visible packet geometry alone is not the
   full native caster set. Preserve alpha/mountain cutouts and bounded page
   selection. The current 111-visible-tile detail packet is not proof of that
   wider graph or its capacity.
2. Finish bounded iteration/freshness integration with the Mac fast path. Shader
   bindings and current natural/hill/cliff/city/unit/resource assets have
   incremental preparation; compiled changes select candidate builds. Changed
   source bytes now select actual fixture consumers in addition to declared
   category dependencies. Global stale-preview checks remain conservative;
   rerendering unaffected categories is not required for a localized review,
   but their older candidate previews are still reported stale after any input
   change. Audit delivery asset isolation as well as candidate DLL isolation.
   The detail pilot currently repeats whole-asset freshness checks before and
   after its work. Profiling a warm preparation alone measured 3.60 seconds and
   8,236 content digests; source/output reads dominate. Reduce repeated reads
   without dropping stale-source, edited-output or during-render change guards.
   The freshness inventory still scans 29,965 pack files, including historical
   candidates (8.4 MB of disposable hash metadata). Restrict it to established
   active runtime/source read closures as part of the dependency cleanup, while
   retaining necessary ignored art outside the hot iteration path.
   Audit both definition-selected packs and direct companion/runtime reads in
   `native/c3x_renderer.cpp`; definitions alone omit unit/resource runtime packs
   and hard-coded companion paths. Preserve configurable pack overrides.
3. Preserve useful unapproved candidates and migrate remaining shared providers/
   tests before deleting the beauty tree. City source evidence selects r111/r112;
   r113-r115 are failed coastal placements, while r116 is an unselected inland
   alternative. A larger experiment number is not a better or approved standard.
   Preserve the coastal fallback and audit useful alternatives and ignored data
   separately; Git does not protect local art.
4. Consolidate remaining native probes (including minimap/default logging) and
   operational docs into the common interface. Preserve capture, ownership,
   invalidation, scrolling/wrapping, compositing, animation and config-off checks.
   Report genuine baseline failures without turning them into migration passes.
5. Remove dependency-checked historical handoffs, experiments, reports, scripts,
   compatibility shims and generated output. Remove one-time city input migration
   tooling when its final preservation audit is complete.
6. Verify current commands/previews/build behavior and audit every requirement
   against the full goal. `check --complete` still lacks Metal reference coverage;
   passing narrow transport or shader tests is not completion.

This is temporary migration state, not a replacement historical ledger.
