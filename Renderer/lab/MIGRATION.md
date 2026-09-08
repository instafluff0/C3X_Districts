# Renderer migration: current state

The full goal is still incomplete. Finish the cleanup-first plan; do not restart
planning or turn individual internal refactors into new milestones. The user's
accepted production baseline is commit `65b02cfc` and the staged DLL recorded in
`baseline.json`. Integration completed before writes. Preserve the fidelity
playbook and deferred wonder/District contracts. Do not stage/install/launch
the game merely to reorganize files.

## Working foundation

- All 20 categories have focused/context D3D11 references at approved/integrated
  r1. The common interface renders, compares, selects affected tests, prepares
  assets, builds isolated candidates, records explicit approval and verifies
  integration. No new visual approval or live-game check has been invented.
- Shared providers under `lab/shared` own city layout/material/light inputs,
  production shaders, natural data/lighting, terrain grids, world/river queries,
  relief sampling, mesh emission and placement/dune patterns. Native retains
  game capture, cache invalidation, GPU resources and compositing.
- Keep native relief/forest statement bodies at their original compiler
  location. An extra x86 function boundary caused one-channel image differences
  despite matching standalone numeric tests. The shared statement-body approach
  restored exact pixels across all 56 category views.
- Preparation covers five shader adapters and six asset jobs. It observes real
  source reads, rejects edited generated outputs and source/output aliases,
  preserves independent texture copies, and builds missing/stale DLL candidates.
  It does not install a DLL or accept new visuals.
- Current city inputs are curated in `packs/CityFidelitySources/current`;
  hill/cliff inputs in `packs/TerrainProfileSources/current`. Unit and resource
  source packs remain intact. Neither newer experiment numbers nor stale
  promotion metadata override the accepted build.

## Cleanup progress and active operation

The original 61 GiB legacy Lab tree is being reduced before further Mac graph
work. Already removed in this cleanup:
- 52 original Lab executable/shader/runner/scenario/experiment-test files;
- 16 L-series handoffs, four obsolete fidelity ledgers, four historical evidence
  pages and the unused legacy deployment wrapper;
- 2,217 unreferenced generated geometry cache entries (about 20.5 GiB exclusively
  held data), retaining 93 referenced/unusual entries;
- 493 generated replay packets and their temporary resource copies;
- 3,456 GPU attachment readbacks and 2,022 BMPs whose decoded RGBA pixels exactly
  match retained PNGs (18.68 GB logical bytes in this batch).

Tracked deletions are recoverable in Git. Removed generated data can be rebuilt;
all PNG review images, current references and source packs remain. Earlier
cleanup retired milestone dispatch/status machinery and about 1.1 GiB of caches.
The old deployment and integration narrative has been replaced by the current
integration contract; historical source findings still need consolidation.

A deeper asset audit found seven *consumed* city DDS paths still embedded in
pack metadata under the legacy tree. Nine referenced DDS inputs (including two
currently unused channels) were copied byte-for-byte into
`packs/CityFidelitySources/current/textures`; the curated palace-material
mapping received only mechanical path replacements. The common preparation
command is rebuilding the city pack. Before proceeding, verify that its binary
diff consists solely of length-prefixed texture-path replacements and that every
DDS byte matches. The one-time verifier is
`/private/tmp/c3x_relocate_city_textures.py verify`; its temporary pre-migration
binary is outside the repository. Do not delete old textures until this passes
and the refreshed city input receipt has no legacy paths. Remove the temporary
verifier and backup after verification is recorded.

## Current verification

- Latest complete category run: 163 tests passed in 154.7 seconds, before the
  city texture-path relocation. Preparation reported zero refreshed inputs then.
- 12 unit/compound importer checks passed; the useful component-local owner-color
  regression was moved out of the retired Lab test into the existing importer
  suite.
- 39 bridge/scroll/unit-input/shadow checks now pass. Two source-inspection
  assertions were following old source locations; they now verify the actual
  shared dune call and included map-vertex layout, without dropping assertions.
- The last complete native candidate comparison matched all 56 approved D3D
  images exactly. This predates the latest Mac-only packet edits and cleanup;
  global preview freshness is overconservative, so do not call all metadata
  current without checking. Reference images and approval revisions are unchanged.
- The candidate DLL is separate from the unchanged staged DLL. No injected source
  changed during cleanup, so no injected compile was run.
- Source-payload comparisons cover 26 resources/104 poses/7,215 marine frames,
  and 55 source-reference units/449 actions/4,622 body poses/1,611 attachments.
  The other 23 composed/original units require separate evidence.
- Known baseline limitations remain: resource backdrop rectangles, route gaps,
  unmapped Silks, and terrain-edit reuse failing with 387 built/zero reused.
  The earlier native behavior suite passed six of seven cases. Do not weaken
  that failure or infer a live-game pass from screenshots.

## Remaining work, in execution order

1. Finish dependency-audited removal of the legacy tree, old preview/verification
   outputs, handoffs and compatibility wrappers. Preserve necessary ignored art,
   useful unapproved candidate inputs and source findings, not an archive copy of
   the entire old system. City selections are r111/r112; r113-r115 failed coastal
   placement, r116 is an unselected inland alternative. Preserve the coastal
   fallback and r116's useful recipe/data before retiring their directories.
   The old one-time city input migration script can go after its preservation
   audit. Other active read closures currently report no legacy-tree inputs.
2. Restrict freshness to current source/runtime read closures, eliminate repeated
   full-inventory hashing, separate backend freshness, and bound caches/outputs.
   Inventory still scans 29,965 pack paths including history. Audit direct native
   companion-pack reads and configurable overrides, not just definitions.
   Verify runtime asset isolation as well as candidate DLL isolation.
3. Complete production-faithful Mac scene rendering across every category.
   Grassland detail is the only public pilot: 97.078% exact pixels, maximum
   channel difference 1/255, unapproved. Its warm command measured 12.3 seconds,
   including 2.1 seconds of scene/render work; repeated preparation dominates.
   Internal gameplay context emits actual plains/hill/mountain/forest meshes,
   but missing source shadows caused a maximum difference of 105/255.
   Wire-8 packet metadata, `shadow_plan.h` and `metal_shadow.h` are partial
   source-shadow work, not a connected or verified render pass. Use the current
   city-profile caster shader, offscreen/wrap casters, six-tile R32 pages and
   b2/b4/t17 receivers. Do not substitute approximate legacy Lab visuals.
   Finish shadows, water/reflections, cities, units and animation on the shared
   scene pipeline, with focused/context comparisons for all categories.
4. Exercise edit → affected previews → explicit approval → pending integration
   → native/game verification → integrated revision, keeping candidate,
   approved and integrated states distinct. Preserve current behavior checks and
   fold remaining native probes into the common interface.
5. Audit every requirement in the original objective, run current verification,
   and deliver a short operating guide and storage-removal summary.
   `check --complete` still lacks full Metal category coverage; the goal must
   remain active until the actual requested end state is proven.

This is temporary migration state, not another historical ledger.
