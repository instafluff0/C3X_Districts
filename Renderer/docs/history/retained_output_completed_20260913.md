# Retained renderer implementation

## Objective and authority

Complete a representative path through persistent world/content ownership, local
invalidation, camera selection, compatible explicit passes, GPU reuse and caller-
driven publication. The 2026-09-13 user instruction replaces the single-next-
experiment queue and conflicting selection rules in the entry guides, execution
contract and benchmark workflow. Intermediate changes need not individually win.
AGENTS.md, assets/visual contracts, native ownership and deferred scope still apply.
Implementation initially excluded staging, installation and game launch. The
subsequent user request, "Please add so I can test in Civ 3," authorizes evaluation
staging below. Installation and game launch remain unperformed.

## Starting implementation and control

Canonical captured appearance/topology, generation-checked content bindings,
resident tile meshes, shared animated meshes and occurrence records already exist.
The worker already accepts immutable caller requests and publishes exact identity;
ambient adoption and exact queued-camera joining preserve native overlays/picking.
Regional raster reuse and local finishing are useful controls. Geometry dependencies
still contain broad revisions; selection and passes repeatedly traverse regions.

An unfinished full scene surface is also present in the starting worktree. Its
existing dense result is 202.66 ms versus 145.23 ms control; stationary results are
50.07 versus 45.12 ms. It copies/restores full MSAA color/depth repeatedly. It is
not a completed architecture or a speedup. Preserve it as evidence, not a new run.
Starting worktree patch/source and exact build receipts are retained under
`Renderer/native/build/retained-architecture-20260913/`. The existing local-finishing
control remains the performance reference. Earlier findings and rejected approaches
are preserved in [the previous execution record](retained_execution_before_architecture_20260913.md).
Parallels GPU timestamps are unreliable; CPU completion endpoints remain authoritative.

## Implementation design

1. Retain the existing world and compiled-content owners. Extend actual dependency
   validity where necessary; do not replace the scene compiler or duplicate assets.
   View occurrences borrow protected content and retain captured ownership/anchors.
2. Select ordered pass inputs once for exposed/static damage and dynamic bodies/
   shadows. Batch within real material and shadow-page limits; preserve layer order.
3. Keep a persistent circular scene color/depth surface and one sparse static
   backup. Restore old animated damage; camera movement changes up to four physical
   spans, without translating samples. Draw exposed/invalidated inputs, save only
   current animated damage, then draw dynamic passes. World-relative projection
   removes the former translated-bitmap clipping collar. Raster damage selects
   scissors; it does not reconstruct little scenes.
4. Keep MSAA4, 2x reconstruction, HDR/glow, common depth and display transfer.
   Finish changed output with one resolve and one readback completion. Bound target
   ownership explicitly, invalidate on failure/reset and retain >=512 MiB sampled
   contiguous process headroom. Do not infer transient GPU residency from VA samples.
5. Use the existing caller-driven worker/publication bridge. Exact compatible
   pixels and ownership are consumed only on Civ III demand. Camera calls without
   a compatible ready result must finish exact work; stale-camera display cannot
   become the means of claiming a latency improvement.

## Checkpoints and completion

Design: owner/repeated-work audit above. Implementation: independent boundary,
pose/removal, edit, pan, wrap, zoom and reset witnesses through existing harness.
Performance: matched stationary animation, dense scrolling and local-content edit,
including setup, complete request latency, actual reuse, completion waits, copy
and bounded ownership. Compare saved pixels with both independent current redraws
and the control. Existing accepted depth rounding is distinct from new differences.
If performance fails, correct the dominant architectural cost within this path.

The circular-surface implementation passes the independent six-case boundary,
local edit and zoom-return witnesses. Correctly matched normal-tier comparisons
use production defaults, the synthetic 100x100 world at 2240x1192/width 128, and
waves/reflections disabled in both paths. Endpoints include capture through the
completed result and ownership check; setup/cold redraws are reported separately.

| Workload | Control | Retained scene | Whole-request saving |
| --- | ---: | ---: | ---: |
| Dense scrolling pair 1, 28 requests each | 150.64 ms | 129.51 ms | 14.03% |
| Dense scrolling pair 2, 28 requests each | 151.68 ms | 127.66 ms | 15.84% |
| Final automatic-selection dense pair, 28 requests each | 152.76 ms | 125.19 ms | 18.05% |
| Stationary animation, 30 requests after warmup | 41.90 ms | 24.24 ms | 42.14% |
| Distant local edits and reversals, 2 requests each | 96.55 ms | 89.27 ms | 7.54% |
| Visible local edits and reversals, 2 requests each | 1,872.30 ms | 1,474.71 ms | 21.24% |

Setup remains substantial: assets-loaded initial scene preparation is 12.7–12.9 s
candidate versus 13.2–13.6 s control. Dense requests compile the same average 26.86
tile entries and reuse 1,030.29 entries in both paths; the gain comes from scene
execution, not claiming existing compilation reuse as a new speedup. There is one
resolve/readback boundary and no full-surface color/depth copies. The complete
viewport is finished during scrolling, so fewer processed pixels is not the claim.
Stationary finishing is limited to the disjoint old/new animated damage.

In the first matched dense candidate, CPU static submission averages 6.03 ms,
dynamic submission 1.66 ms, finishing submission 0.03 ms, completion wait 69.28 ms
and CPU copy 0.82 ms. Completion includes pending GPU execution and readback; it
is not a calibrated transfer-only duration. Full composition accounting leaves
less than 0.1 ms unexplained inside worker rendering. GPU timestamps stay invalid.

The earlier full-surface translation implementation measured 180–182 ms and is
superseded. Its source and receipts remain under the implementation evidence
folder. A later comparison accidentally disabled production defaults and is
excluded from performance claims; use only `production-*` and final `auto-*` receipts. The tested
shader is now generated by the production adapter, byte-identical to the first
circular shader witness. Category tests pass 135 checks (one existing skip), with
28 separate architecture/publication checks and the independent GPU witnesses.
Final current-code production integration passes 255 tests (one existing skip),
resource playback, animation, scroll/removal parity and common-depth witnesses.

Evidence: `Renderer/native/build/retained-architecture-20260913/comparison.json`,
`automatic-comparison.json`, `production-validation.log`, `final-delivery.log`,
`resources-tests.log`, `architecture-final-tests.log`, and the named
`retained-architecture-production-*` / `retained-architecture-auto-*` run directories. Dense local edits
also match four independent full redraws: distant edits average 96.55 → 89.27 ms
(7.54%); visible edits average 1,872.30 → 1,474.71 ms (21.24%). Visible edit geometry
still costs about 1.2 s and is the remaining dominant edit cost. The explicit-
identity caller-driven ambient boundary passes. The control's saved dense image
is byte-identical to the preserved starting control. No live-game result is claimed.

Eligible city-profile views now select this path automatically when waves and
reflections are configured off. `C3X_RENDERER_SHARED_SCENE_SURFACE=0` retains the
reproducible regional control; `1` explicitly selects the retained profile for
witnesses. Unsupported extents use the existing custom renderer. The new path
replaces static/animation regional execution for eligible requests; it is not a
separate presenter or renderer service. Automatic-selection boundary replay and
the final matched dense pair pass; the non-oracle production candidate also passes
`python3 Renderer/renderer.py integration resources --renderer-only`.

The same existing publication bridge consumes this path. No injected source or
patch-table change is needed. Unmatched-camera calls still finish exact work;
returning old pixels would violate native overlay/picking ownership. General
camera anticipation and native presented cadence are not established by replay.

Minimum sampled contiguous VA across matched dense candidates is 1,474.92 MiB;
transient driver residency remains unmeasured. Resource ownership: one RGBA16F/D24 MSAA4 scene at 2x, one color/depth backup,
one resolved HDR target, one native reconstruction and BGRA target. Max target
payload is 1,165,363,200 bytes at 2240x1192 including guards, within the existing
1,152 MiB surface cap. Damage metadata has a bounded 8-pixel union grid; it does
not own additional raster surfaces. No full-surface copies run during scrolling.
Reference resources still use separate pose/placement on shared resident meshes;
static/dynamic pass submissions preserve material order and the 32-page shadow cap.

New differences from the regional control (8,055 pixels, 0.302% of the initial dense
view; RGB mean absolute error 0.0121, maximum channel delta 62) remain separate
from the previously authorized common-depth change. Saved side-by-side and focused
comparisons are `comparison.png` and `edge-comparison.png` in the evidence folder.
Independent current warm/cold redraws are exact; that does not constitute acceptance
of differences from the control. No new
visual acceptance or live-game result is claimed. Evaluation staging was
subsequently authorized and completed below; no installer or game launch was run.
Dense requests remain above the existing 100 ms average target. Completion wait
is the largest measured dense stage; calibrated GPU busy time and driver residency
remain unknown. Waves/reflections retain the existing path, and no generalized
hardware instancing or speculative camera prediction was introduced.

## Initial evaluation staging

The initial verified non-oracle production candidate was staged in `Renderer/bin/C3XRenderer.dll`:
SHA-256 `6440ea60a3d1fcab8810e0026db078107900994163a6af24058b3aa2e1df7138`.
Its current compiled inputs and passing resource integration receipt were checked,
and candidate/staged hashes match. The previous DLL and exact staged candidate,
build/integration receipts and staging receipt are preserved under
`Renderer/native/build/retained-architecture-20260913/evaluation-staging/`.
Rollback DLL: `previous-C3XRenderer.dll`, SHA-256
`b28d8f13676066445086bf2a4fe887e1e7fdf5bc6e3ea4a0a32446262cfd78e8`.
Existing local configuration already enables custom rendering/cache and disables
waves/reflections; no configuration changes or fixed-reference replacements were
made. Restarting the existing configured game loads the new DLL. Unsupported
viewport extents still select the existing path.

## Incremental output continuation

The user authorized this implementation after evaluation staging. The initial staged DLL
above was preserved throughout validation and is the rollback for the new evaluation
build below. Continuation builds and benchmarks used isolated candidates. Starting
source, receipts and rejected mechanisms are preserved under
`Renderer/native/build/incremental-output-20260913/`.

The finished circular HDR/native targets now persist across camera translations.
Static damage expands by the glow kernel's four-native-pixel support, including
across physical seams. Animated bounds already include that support. Disjoint
workgroup-aligned damage selects reconstruction, HDR glow and display conversion;
it does not reconstruct miniature scenes. A changed view still remaps the complete
finished image into the current camera bitmap with one readback completion.
`C3X_RENDERER_INCREMENTAL_OUTPUT=0` preserves the prior circular full-finishing
control; the default uses incremental finishing. No additional render textures,
publication buffers, native patches or redraw requests are introduced.

A bounded diagnostic showed that Parallels event queries do not establish useful
host GPU completion here: their waits were 0.14/0.04 ms, followed by a 66.45 ms
**one-pixel** readback and only 0.85 ms for the subsequent full-image readback.
This attributes most of the original wait to pending GPU work, not transfer
volume; it does not separate scene drawing from finishing or establish GPU busy
time. These serialized runs are excluded from performance acceptance, including
when a receipt's quality label is incorrectly changed. The existing harness now
also rejects an observed Civ III process during exclusive benchmark cases.

Full hardware MSAA4 resolve remains. Compute and render-target local resolves
failed exact half-float parity with hardware resolve; an explicit rounding variant
still differed. Their shader/witness snapshots and failures are preserved in the
same evidence folder. Do not repeat those mechanisms without a materially different
precision-preserving approach. Shared art, reconstruction, glow and display shaders
are unchanged. The stronger GPU witness compares circular retained finishing with
full reconstruction at seams, thin edits, bright bloom and sample-varying coverage.

Two matched intermediate dense pairs improved whole requests by 10.65% and 12.00%,
with all five saved images per pair exact. A redundant second glow margin initially
increased stationary work; the final correction retains the existing animated
margin and expands only exposed/static damage.

Final delivery uses the same normal-tier, production-default profile and synthetic
100x100 dense world as its matched circular full-finishing control: 2240x1192,
tile width 128, waves/reflections off. These are complete requests through checked
result ownership; cold setup and independent verification remain separate.

| Workload | Circular control | Incremental output | Effect |
| --- | ---: | ---: | --- |
| Dense scrolling, 28 requests each | 119.40 ms | 103.28 ms | 13.50% lower latency |
| Stationary animation, 30 after warmup | 22.61 ms | 22.98 ms | +0.37 ms; no claimed improvement |
| Distant edits/reversals, 2 each | 78.43 ms | 68.99 ms | 12.04% lower measured latency |
| Visible edits/reversals, 2 each | 1,432.13 ms | 1,386.83 ms | 3.16% lower measured latency |

Dense finishing falls from 2,697,600 to 365,952 pixels/request. Both paths still
resolve 10,790,400 high-resolution pixels and remap the image in four readback
copies before one completion wait. GPU completion/readback wait falls from
65.52 to 49.27 ms; CPU copy stays about 0.66 ms. Compiled/reused entries and upload
bytes are identical (26.86 / 1,030.29 / 421,114 per dense request). Initial scene
preparation remains 12.5–12.7 seconds. Stationary finishing is identical at 174,037
pixels/request; there is no eliminated stationary work or claimed stationary win.
Visible edit geometry still averages 1.17 seconds; the small edit samples do not
establish a general edit-speed guarantee.

Target payload remains 1,165,363,200 bytes; no render texture or publication buffer
was added. Minimum sampled contiguous VA in the final dense candidate is 1,477.20
MiB. Transient driver residency and native presented cadence remain unmeasured.
Dense requests remain slightly above the 100 ms average target.

Validation: 255 current-code integration tests pass (one existing skip), with
resource animation, scrolling/removal and zoom-return witnesses. The independent
six-case boundary and four-edit redraw checks pass. All five dense, 32 stationary
and two edit saved images match their paired control exactly. The GPU fixture
passes nine regional and nine circular HDR/MSAA finishing cases; portable circular
convolution/damage tests and 17 analysis tests also pass. Earlier zoom and explicit
caller-driven ambient witnesses passed; the final margin correction preserves
that publication code. No new visual differences from the staged circular control
were observed; its earlier differences from the regional renderer remain pending.

Receipts: `comparison.json`, `attribution.json`, `delivery-validation.log`,
`final-output-tests.log` and `analysis-tests.log` under the continuation evidence
folder. Exact final benchmark build: `Renderer/native/build/incremental-output-delivery-build/`.
The rejected local resolves, intermediate runs and source snapshots remain there
as evidence; they are not the delivered path.

The verified non-oracle production DLL is now staged for the user's existing game
test request: SHA-256 `b5cd1b08213d4b4d6ccc50aab07d3a1d5ebf63cdf5afb1162467a76fd59c1f4d`.
Candidate and staged hashes match. The exact DLL, build/integration/comparison
receipts and rollback `previous-C3XRenderer.dll` are under the continuation evidence
folder's `evaluation-staging/`. Restart Civ III to load this update. Configuration,
fixed references and injected code were unchanged; no installer or game launch
was run by this task.
