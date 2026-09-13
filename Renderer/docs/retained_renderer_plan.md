# Retained renderer program

Current status and preserved evidence for the retained renderer. Planned work
is not evidence that implementation or performance targets pass.

## Current status — native identity retained; shared animated resource geometry next

Local region validity passes the dense edit comparison: the four-edit sequence
is 16.8–16.9% faster, with exact independent full redraws. Distant edits average
425–465 ms control versus 103–116 ms candidate. Visible edits remain slow at
1.85–1.90 seconds, dominated by geometry preparation. A separate navigation check
is unchanged within one matched pair (193.4 / 193.8 ms); the below-100 ms navigation
gate and native presentation remain unmet.

The previously accepted shadow-only optimization remains retained: the user
explicitly accepted its repeatable 13.3–13.6 ms navigation saving despite the
previous 10%/20 ms selection rule. These are different workloads; do not add the
percentages. Both implementations remain scoped opt-in paths, with no staging,
installation or live-game claim. Tooling is complete and failed raster/array
experiments remain closed. The 100x100 fixture uses realistic synthetic density,
not captured live state.

[The architecture](renderer_architecture.md) defines the destination;
[the execution contract](autonomous_renderer_execution.md) governs continuation;
[the benchmark workflow](benchmark_workflow.md) specifies validation. This status
is the single active task record.

### Destination checkpoints

The user's clarified objective keeps all six capabilities in scope. This table
records architectural gaps, not a second task queue or completion percentage.

| Destination | Current foothold and remaining gap |
| --- | --- |
| Complete persistent scene/world-instance database | A bounded canonical captured-appearance owner now feeds compiler lookups and retains local revisions; resident compiled-content bindings are now consumed; complete appearance capture and sparse object/action ownership remain unfinished. |
| General spatially selected draw lists | Static draw/receiver/caster/region consumers now read compact resident occurrence records; general spatial selection feeding compatible batches remains unfinished. |
| Systematic local revisions and invalidation | Local topology/dependency observations and retained-region validity now preserve unaffected work; compiled-input owners still have broad revision invalidation. |
| Broad batching/instancing with explicit passes | Shared natural buffers and an explicit resource pass exist; general asset/instance batching remains unfinished. |
| Completed asynchronous Civ III presentation | The injected compositor now supplies lifecycle/visibility identity and consumes exact queued or compatible ambient work through the existing owner; general asynchronous camera handoff and live presentation verification remain unfinished. |
| Camera movement with bounded reconstruction/rendering/readback | Retained overlap and region reuse exist; latest accepted-control dense navigation is 161–164 ms and remains above the first 100 ms gate. |

The publication handoff, captured-appearance owner, resident-content bindings and
compact occurrence records are retained. The first static terrain batch representation
was rejected after exact but unhelpful whole-workload results. Native request identity
is now connected and verified in source/replay. Shared animated resource geometry
is the current selection. Other gaps remain explicit dependencies,
not tasks to pursue concurrently.

### Prioritization correction

The user explicitly requested established RTS rendering approaches to guide
priorities rather than a chain of nearby experiments. Apply the destination's
persistent world content/local revisions, spatial selection with batching and
instancing, explicit static/dynamic passes and bounded GPU ownership as the
basis for implementation selection. Measurements choose among these mechanisms
and validate complete changes; a large phase does not justify successively
trimming every nearby operation. Preserve useful existing content reuse rather
than adding more completed-image cache policy. Keep readback/publication at the
Civ III bridge, with its exact-camera/overlay constraints.

The current upstream [0 A.D. source review](0ad_renderer_review.md) supports
shared immutable meshes, locally dirty terrain patches, material variants and
preparation shared across passes. Those are concrete mechanisms for the existing
architectural destination. It also distinguishes world-content sharing from our
unsuccessful screen-region batching. No external engine code was imported and no
new performance claim follows from that review.

### Architectural selection after the user's course correction

The source review against all six capabilities finds persistent compact topology
and shared tile meshes, but no complete world appearance/instance owner; view
assembly still rebuilds captured occurrence/layer lists. The contributor index
serves cache/shadow queries rather than a general selected draw-list owner.
`submit_geometry` combines selection, recursive passes and GPU bindings, limiting
compatible batching. Local validity improvements are useful but do not complete
these responsibilities or make newly exposed content cheap to draw.

The first architectural boundary selected after that review was **caller-driven
publication selection**. Civ III owns render demand; the renderer prepares work and supplies a
compatible ready result at the next native render call. The user's clarification
explicitly rules out worker-completion notification/redraw requests. The proposed
notification code was removed before any build or native changes. Existing game
animation scheduling is preserved, not converted into a completion channel.

### Single next implementation task — shared animated resource geometry

The native bridge now carries identity into ordinary publication selection; its
strategic live checkpoint is explicit below. The remaining scene/database, spatial
selection and batching gaps still share a missing responsibility: **animated mesh
assets and instance placement are not separated at GPU submission**. Static
occurrence records already provide that separation for resident static content.
Resources still skin every instance on the CPU, project every posed vertex into
body and shadow streams, and upload both streams before clipping/submission.
`ResourceAnimation` already owns the source mesh/palettes/texture, `ResourceAnchor`
already supplies placement and deterministic phase, and the explicit resource pass
already separates compatible body/shadow bindings. Extend these owners.

**Next task:** implement shared animated resource mesh inputs with per-instance
pose/placement for the existing resource body and projected-shadow passes. Evaluate
GPU skinning/projection as the bounded implementation, preserving the authored
interpolated pose and normal transform. Reuse the existing asset, dynamic-buffer
and pass owners; replace the superseded per-vertex CPU projection/upload path once
validated, rather than keeping another optional pose cache or renderer.

- **Capability afterward:** resource instances draw resident shared mesh content
  using their own pose and authoritative occurrence transform; both body and shadow
  use the same pose contract. This is a bounded animated-asset/instance capability,
  not a claim of a complete world database or general batching.
- **Repeated work removed:** CPU skinning, camera projection and full body/shadow
  vertex uploads for every occurrence on each animation composition. Existing
  dense trace samples put pose preparation around 15 ms and uploads around
  5.6–6.2 MB per composition. The whole animation phase is much larger; do not
  attribute its readback/GPU wait to CPU skinning or promise to remove all of it.
- **Connection to the destination:** compatible instances can subsequently share
  asset/material bindings and instance submissions without first reconstructing
  individual screen-space meshes. This also lets newly exposed views render
  resident animated assets when raster reuse misses. Unit action ownership,
  waves and deferred wonders/Districts stay under their established contracts.
- **Architectural acceptance:** independently preserve posed positions, inverse-
  transpose normals (including collapsed bones), alpha coverage, depth, phase,
  source facing, captured anchors and projected shadows. Keep unchanged budgets,
  explicit failure/reset/asset-replacement cleanup and >=512 MiB sampled contiguous
  headroom. Account for shared source/palette and temporary storage, not only the
  replaced uploads. Start with the existing small resource boundary and supported
  zoom/phase checks; image equality remains required for unchanged appearance.
- **Performance acceptance and stop:** after exact focused checks, use the existing
  matched short dense navigation sequence with identical production defaults and
  reset policy. Retain a useful whole-request improvement; a structurally necessary
  representation can remain without a large speedup only if it avoids material
  regression and actually exposes the shared mesh/instance input needed by the
  compatible instance submission above. Reject extra machinery that merely moves
  CPU work into more expensive repeated GPU work. Close after the bounded result;
  do not expand into a skinning framework or another baseline campaign.

### Shared resource geometry — implementation evaluation

The first candidate uses resident source meshes and per-instance bone/placement
constants in vertex shaders. It builds and passes 133 category tests (one existing
skip), nine focused pose/bounds checks, and the small native five-publication
witness. An initial dense comparison found two changed wheat-edge pixels. Exact
CPU bounds did not remove them. A bounded GPU probe isolated one-bit world-coordinate
rounding differences; precise world arithmetic restored exact dense images without
changing the shared compiler or relaxing pixel equality. The temporary CPU-bounds
diagnostic was removed. Its source and evidence remain under
`Renderer/native/build/resource-instances-20260913/`.

That corrected vertex-stage version is not retained as the final implementation:
whole-request evidence is mixed. Candidate means were 154.843 and 168.703 ms;
the intervening two `assets_loaded` CPU control cases were 187.090 and 159.364 ms.
All four 14-offset cases preserve saved images, inputs, ownership/accounting and
exclusive-GPU checks. CPU pose preparation falls, but vertex skinning repeats for
body/shadow and regional submissions, and GPU/readback variation offsets the saving.
There is no reliable whole-workload speedup claim from these pairs. The earlier
incorrect-image pair and shader-cache warmups remain diagnostic evidence only.

The same selected task is now evaluating **one GPU skinning dispatch per instance,
with a shared posed mesh consumed by body and shadow passes**. Source meshes,
instance constants and posed buffers remain under the existing asset/buffer owners
and unchanged 32 MiB resource-storage cap. Camera projection remains per occurrence;
no pose cache, extra queue or new optional rendering mode is introduced. Existing
legacy profiles keep their established CPU path. The revised isolated build and
focused pose tests pass; native rendering and whole-request acceptance are pending.
Do not describe this candidate as accepted or stage it while evaluation is open.

### Native request identity — structurally retained

The optional `c3x_renderer_render_view` ordinary-demand export uses the existing
versioned camera request, worker queue and publication owner. Injected capture now
supplies scenario lifetime, native viewer, observed visibility and topology scene
revision; ordered local tile/object/anchor data still participates in exact request
matching. Changing any identity rejects old work. Identical requests consume/join
existing work rather than cancelling and executing it again. Same-view ambient
completion remains passive until Civ III next calls. Legacy API-17 callers and DLLs
remain compatible; no second presenter, completion callback or redraw hook exists.

Architectural acceptance passes: seven executable native-identity/publication tests
cover epoch changes, exact adoption, reset/failure/unit takeover, visibility-only
edits, mixed-viewer rejection and allocation failure. A final review caught a partial
allocation hazard when shrinking the world observations; both owners now commit
only after allocation succeeds, with regression coverage for failure at either
allocation followed by returning to the old map. Another 38 bridge/fixture tests
pass. The final approved injected compile and the isolated normal-tier native DLL/
preview build pass. The restored pre-change renderer also passed 73 focused checks.

The native `native-view-ambient-small-20260913` witness passes all five independent
exact publications with nonzero epochs and verified source/binary/input identity.
Three stationary calls took 0.501, 0.228 and 0.210 ms; their publication completion
polls observed 32, 15 and 31 ms. An explicitly queued current request was adopted
by ordinary demand in 15.143 ms (15.253 ms including begin), with no second render.
The changed-camera request still waited 640.167 ms for exact output. These sparse
640x480 standalone samples validate the contract, not dense navigation or native
frame rate. Minimum sampled contiguous headroom was 2038.8 MiB; no fallback or
recovery occurred. Tests/compile hashes are under `native-view-identity-20260913`,
and the complete native build is `native-view-complete-build-20260913`, all beneath
`Renderer/native/build/`.

The native visibility observation adds eight bytes per parity tile to the existing
world scan: 40,000 bytes for 100x100, at most 16 MiB at the supported 2048x2048 cap.
Resize temporarily overlaps old/new visibility allocations; neither publication
nor geometry budgets increase. When full-world topology capture is disabled,
visibility epoch remains zero and exact ordered local visibility still governs
reuse. This is not complete world appearance capture or measured live capture cost.

**Remaining native limitation and checkpoint:** same-view asynchronous ambient
rendering is still the existing opt-in mode. A new camera without compatible output
must finish exactly before native overlays/picking proceed. Current source is ready
for a strategic staged/installed game check of scenario/viewer/visibility changes,
stationary ambient demand, camera/zoom/overlay/picking and unit takeover, but no
staging, installation or game launch was performed or authorized here. That check
is required before enabling/claiming ordinary native async presentation or native
frame-rate gains; general asynchronous camera changes cannot bypass the exact-view
constraint. Independent resident-content work can proceed while it remains pending.
No new executable patch-table entry is required.

### Fixed-topology terrain batch experiment — rejected and restored

The bounded candidate packed immutable natural terrain vertices into pages keyed
by exact index bytes/format and vertex size. Consecutive eligible occurrences used
indexed instanced draws with their original projection; shadow/reflection consumers
used the same allocations at byte offsets. Pages were owned by compiled-content
leases, with slack/index/instance storage charged to the unchanged geometry budget.
This was a real submission mechanism, not the earlier bounds-only index experiment.

Architectural checks passed at the small scope: 144 selected terrain occurrences
became 54 submissions in the initial boundary frame, preserving alpha/triangle
order. `terrain-batch-defaults-boundary-20260912` passed all six independent redraws
and exactly matched the accepted occurrence renderer's saved images with full
input/source/binary verification. The page ownership/failure/bounds test and 251
selected resource/integration tests passed (one existing skip). These do not certify
broader zoom, edit or live-game behavior; those campaigns were not earned by the
performance result.

The deciding dense 100x100, 14-offset navigation cases used the same production
defaults, normal budgets, `assets_loaded` reset and exclusive-GPU watchdog:

| Order | Control whole request | Candidate whole request |
| --- | ---: | ---: |
| control, candidate | 163.522 ms | 164.519 ms |
| candidate, control | 160.968 ms | 200.267 ms |

All 56 playback endpoints passed ownership/accounting checks; saved images, runtime
inputs and request manifests matched. All watchdogs were clear. Candidate contiguous
headroom remained at least 1559.098 MiB (controls at least 1701.105 MiB). Initial
preparation was 13.414/13.566 seconds control and 13.574/13.701 seconds candidate.
The second candidate had broad phase variation; preserve it without claiming a
proven cause or excluding it. Neither pair demonstrated useful whole-request gain.
The first pair's main draw phase was 6.646/6.760 ms; animation composition remained
91.755/92.197 ms. Fewer terrain submissions did not remove the dominant workload.

Reject this representation rather than retaining its paging/raw-vertex complexity
on an unspecified promise of future gains. The compact occurrence records and all
preceding accepted improvements remain; no live batch capability is claimed.
The candidate sources/shaders and comparison receipt are preserved under
`Renderer/native/build/terrain-batch-navigation-candidate2-20260912/`.
The accepted pre-batch runtime, shadow owner and natural shader runtime were restored.
Obsolete batch implementation/tests were removed from the active source tree.
Three stale native source assertions were updated to name the already accepted
captured-appearance/content owners; their executable owner tests remain.

Earlier setup receipts are preserved: the first shader include form failed the
existing flattened-source loader; a warm-cache run cannot pass input stability.
An initial replay command omitted inherited production defaults and produced five
one-level color differences in both control and candidate; restoring the documented
switches produced exact accepted images. Neither setup issue is performance evidence.
The final candidate generated its batch vertex entry separately so existing shaders
remained unchanged. No art/reference acceptance, staging, install or game launch occurred.

### Resident occurrence draw records — structurally retained

The user explicitly directed retaining this step if it is a dependency for
architectural pieces likely to produce decisive performance gains. Retain the
**revised borrowed-reference implementation** on that basis: general selected
draw lists need shared content handles and occurrence projection across drawing,
shadow selection and region validity without cloning mesh/material ownership. Those
consumers now share that interface. The rejected terrain page allocator was one
consumer of it, not its only architectural purpose. No speedup or
completed general spatial-selection capability is claimed for the representation.

`occurrence-draw-reference-complete-build-20260912` is the accepted build. Static
view assembly and pixel prefetch now keep source references plus projected bounds,
translation and natural projection; immutable mesh/material metadata stays in
`CachedTileGeometry`. Static drawing, receiver/caster selection and region keys use
one read adapter. Active epoch protection preserves source lifetimes through
compilation/prefetch, with view clearing before eviction/reset destruction. The
view takes no COM/material references; dynamic poses/waves keep their existing
owners. Read traversal copies two pointers, not the full projection record.

Focused owner/mesh/region tests pass (30), including borrowed versus independently
traversed region keys, local dependencies, order and supported projected bounds.
The assembly test works with non-copyable source chunks and verifies unchanged
resource reference counts. A focused caster test additionally matches borrowed
and owned traversal for every wrapping combination with different occurrence
projection; it passes. Resource category tests pass (132, one existing skip).
The original build caught one remaining owned-container traversal; the reference
revision required an explicit array-reference return for the existing MSVC mode.
Both were corrected before the accepted build, with unchanged verified sources.

`occurrence-reference-boundary-20260912` passes all six independent small redraw
checks. `occurrence-reference-zoom-small-20260912/comparison.json` records exact
128/192/160 repeats and saved-image parity with accepted content bindings, using
identical runtime inputs. Profiled static-view metadata peaks at 80,160 bytes in
that 640x480 fixture and returns to the fixed 564-byte layer-array metadata after
reset. Sampled contiguous headroom exceeds 2030 MiB. The source enforces occurrence
records at most half the owned chunk size; allocator/driver transients are not
fully measured. `occurrence-reference-edit-small-20260912` passes four distant/
visible edits and reversals against independent full redraws. The visible calls
remain 1260.039/1239.929 ms; these single-arm correctness timings are not a speedup.

`occurrence-reference-navigation-candidate3-20260912/comparison.json` preserves
three matched 14-offset cases per arm, first control/candidate then candidate/
control twice, with `assets_loaded` reset and identical initial preparation. The
2240x1192, width-128, waves/reflections-off dense 100x100 synthetic world retains
1921 initial occurrences, 5 cities, 349 roads, 80 rails, 344 improvements and 150
resources. All 84 playback endpoints pass ownership/accounting checks; saved images
match exactly. Full input/source/binary verification and all watchdogs pass. Dense
saved-image equality is not an independent full dense redraw oracle.

| Mean whole navigation request | Case 1 (ms) | Case 2 (ms) | Case 3 (ms) |
| --- | --- | --- | --- |
| Accepted content-binding control | 157.147 | 159.723 | 158.803 |
| Borrowed occurrence records | 154.171 | 168.342 | 156.666 |

**Architectural result:** retained as the direct input to the selected batch pass.
**Performance result:** inconclusive for speedup; aggregate means are 158.558 ms
control / 159.726 ms candidate (about 0.7% slower). The slower second candidate
repeat remains in the evidence; it did not recur in the deciding third pair.
This closes the comparison without extending a tuning campaign. Initial preparation
is 13.035/13.228/13.235 seconds control and 13.167/13.169/13.424 seconds candidate.
Animation composition remains dominant (86.716/91.570/86.790 ms candidate);
geometry is 31.499/37.168/33.520 ms. Headroom across the six runs stays above
1701 MiB at sampled host boundaries. The below-100 ms navigation gate is unmet.

The first implementation is rejected: its read adapter returned projection records
by value on every traversal. Its exact small boundary/zoom output did not justify
159.495/158.517 ms controls versus 174.732/176.786 ms candidates. Draw means rose
from 6.524/6.547 to 8.343/8.581 ms, and animation composition from 86.425/86.436 to
95.794/95.024 ms. All 56 endpoints and verification checks passed with no watchdog
conflicts. The first source remains under
`occurrence-draw-complete-build-20260912/first-candidate-source/`; its comparison
is `occurrence-draw-navigation-candidate2-20260912/comparison.json`. The revision
removed the added per-traversal copying; it is not claimed to explain every timing
variation. No optional switch, GPU budget, shader appearance, native hook or ownership
category changed. No staging, installation, game launch or visual promotion occurred.

### Resident compiled-content bindings — structurally retained

`content-binding-complete-build-20260912` builds the retained implementation. The
existing mesh cache remains the sole buffer owner. A slot table, bounded by that
cache's existing entry limit, supplies non-owning generation handles; eviction/reset
invalidate handles without retaining buffers. World records, cached-view arrays,
projection entries and pixel preparation consume these handles. World-bound reuse
still checks the exact compiled signature and every existing dependency. The keyed
owner lookup remains for a missing binding/new context. No mode or GPU budget grew.

Host owner/mesh tests pass (20 tests), including stale slots after eviction/reuse,
reset, failed capacity admission, no lifetime extension, real cache eviction,
viewport restoration and prefetch rejection. Resource category tests pass (132,
one existing skip). The initial build found a remaining iterator-style prefetch
access; it was converted before the successful build.

`content-binding-boundary-20260912` passes all six small independent full-redraw
resource/ownership/clock/removal checks. `content-binding-zoom-small-20260912/comparison.json`
records two cycles at widths 128/192/160: all repeat comparisons and saved images
against the preceding accepted control are exact. These are profiled 640x480
correctness checks, not a zoom speedup claim. The handle table reaches 25,584 bytes
there and returns to zero on reset; sampled contiguous headroom exceeds 2036 MiB.

`content-binding-edit-small-20260912` additionally passes all four distant/visible
terrain edits and reversals against independent full redraws at 640x480, with
unchanged inputs/sources/binaries and no watchdog conflicts. This closes local
appearance/dependency invalidation coverage for the binding change; its single-arm
timings do not establish an edit speedup.

`content-binding-navigation-candidate-20260912/comparison.json` records two dense
14-offset cases per arm, controls then candidates, each with `assets_loaded` reset
and identical initial preparation. The 100x100 synthetic world supplies 1921 initial
captured occurrences, 5 cities, 349 roads, 80 rails, 344 improvements and 150 resources
at 2240x1192, width 128, waves/reflections off. All saved images match across arms;
all 56 playback endpoints pass ownership/accounting checks. Dense saved-image parity
is not an independent full dense redraw oracle.

| Mean whole navigation request | Case 1 (ms) | Case 2 (ms) |
| --- | --- | --- |
| Preceding accepted control | 165.582 | 183.783 |
| Resident content bindings | 160.334 | 183.937 |

**Architectural acceptance:** retained. Lifetime-checked content association replaces
key/version rediscovery in real consumers without another buffer owner. **Performance
acceptance:** no material regression in these repeats, but no reliable speedup or
below-100 ms pass. Initial preparation is separately 14.230/13.449 seconds control
and 13.002/13.205 seconds candidate. Animation composition remains dominant at
88.561/99.504 ms candidate; geometry is 34.071/42.459 ms. This does not select another
animation helper as the next task.

Up to 2425 world records occupy 2,635,448 tracked bytes versus 2,596,648 control.
Host-boundary contiguous headroom stays above 1699 MiB across both arms; internal
transient/driver residency and dense handle-table peaks remain unmeasured. Existing
GPU budgets are unchanged. Inputs/request manifests match; source differences are
the renderer and the captured-scene/resident-content headers. Full source/input/
binary verification passes, and both dense watchdogs report no competing GPU/build
process. Small boundary/zoom runs are one-shot without watchdog coverage.

Shared content is still copied/projected into per-view chunks; complete appearance
capture, general spatial draw lists, compatible batching and native presented-frame
performance remain unfinished. No staging, installation, game launch, injected
changes, visual promotion or live-game performance claim occurred.

### Persistent captured appearance — structurally retained

`SceneTopology` now extends the worker-owned `CapturedScene` record table instead
of owning six rebuilt maps and borrowed appearance pointers. Records keep exact
current occurrence data, canonical static appearance and a local revision. Camera,
clock, visibility and unit-body changes do not change static appearance identity;
actual tile-attached changes do. Lightweight halos preserve remembered full
appearance but retain their own current lookup data. The existing last-ordered
canonical occurrence rule is unchanged. Absent records cannot serve current
lookups, and an incomplete/cancelled update exposes no partial current lookup.

The bounded observed subset retains at most 8192 records, matching the API's
maximum occurrence count. Capacity pressure evicts absent records only, after
protecting every current coordinate regardless of traversal order. Eviction and
reinsertion receive a new revision within the owner lifetime. World basis changes
clear remembered records; the existing geometry/reset lifecycle still clears this
CPU owner too. Payload is statically bounded below 16 MiB; tracked bytes include
estimated node overhead and actual bucket count, with allocator/transient limits
reported separately. This is not complete full-world capture, durable object/action
identity, GPU-reset-independent CPU retention, or general spatial selection.

The compiler's terrain, river/route, natural/city-exclusion and dependency lookups
now consume this single owner. Combined neighborhood queries perform one lookup
instead of separate ground/surface/relief lookups. The current geometry compiler,
GPU caches, raster reuse, pass ordering and public capture ABI are preserved.
There is no new opt-in mode or parallel lookup implementation.

Owner and retained mesh tests pass (19 tests), covering owned-copy lifetime,
revision stability, wrapped ordering, full/halo precedence, removal, cancellation,
world replacement, bounded admission/eviction and existing mesh dependencies. The
old mesh test fixture's map lookup was adapted to the new record lookup without
removing dependency assertions. `renderer.py test resources` passes 132 tests with
one existing skip. The new owner tests are included in ordinary integration checks.

`captured-scene-complete-build-20260912` built the measured candidate after the
initial build identified one remaining forest/city lookup on the deleted map;
that lookup was converted. `captured-scene-boundary-20260912` passes all six exact
resource/ownership/clock/removal checks. `captured-scene-edit-small-20260912` passes
four distant/visible edits and reversals against independent full redraws. The
640x480 edit calls were 20.496/20.256 ms distant and 1207.540/1251.307 ms visible;
these single-arm figures are correctness context, not an edit speedup claim.

`captured-scene-navigation-candidate-20260912/comparison.json` records two dense
14-offset cases per arm, controls then candidates, each with `assets_loaded`
reset and identical initial preparation policy. The complete 100x100 world has
1921 initial captured occurrences, 5 cities, 349 roads, 80 rails, 344 improvements
and 150 resources at 2240x1192, width 128, waves/reflections off. All saved images
match across arms; all 56 playback endpoints pass ownership/accounting checks.
Small independent redraw checks establish correctness beyond revisit equality
at their small scope; dense saved-image parity is not a full dense redraw oracle.

| Mean whole navigation request | Case 1 (ms) | Case 2 (ms) |
| --- | --- | --- |
| Control | 158.736 | 183.991 |
| Persistent scene owner | 168.875 | 154.988 |

Variation is larger than a reliable speedup claim. Retain the structural ownership
replacement with no material regression established across these repeated cases;
the below-100 ms gate remains unmet. Animation composition remains the largest
measured component (candidate means 89.605/83.218 ms); geometry means are
38.600/33.322 ms. The next task follows the missing content association, not the
largest helper. Up to 2425 scene records occupy 2596648 tracked bytes (about
2.5 MiB), and sampled contiguous headroom stays at least 1696.2 MiB across both
arms. Internal transient/driver residency is unmeasured. Inputs/request manifests
match; source differences are the renderer, captured-scene header and one included
geometry lookup. Full source/input/binary verification passes and the session
watchdog observes no competing GPU/compiler process. The small six-case boundary
is one-shot without watchdog coverage; the edit/dense cases use it.

The final `captured-scene-documented-build-20260912` only clarifies the header's
bucket-accounting comment relative to the measured candidate; runtime code is
unchanged. No staging, installation, game launch, injected changes, visual promotion
or native presented-frame claim occurred.

The pull contract is resolved: completion never requests a native redraw. Native
map/viewer/visibility epochs and exact displayed-view compatibility remain required
before enabling general asynchronous camera handoff in Civ III. Until that bridge
is implemented, a missing exact camera blocks for its result; old-view pixels do
not become acceptable because preparation is asynchronous. This boundary must be
resolved at the native handoff implementation, before its strategic live checkpoint,
not inferred from upstream scene speedups or deferred behind a frame-rate claim.

### Exact queued-request consumption — structurally retained

`RendererWorker::render` now joins an identical pending/active camera request or
consumes its completed publication through `camera_poll_locked`. It reuses the
existing immutable snapshots and condition variable. Legacy calls adopt only zero
caller-owned lifecycle epochs; the optional ambient path also rejects ambiguous
nonzero epochs. Different camera, clock, visibility, order or topology still takes
the existing synchronous path. An adopted failure returns its error without
publishing an older front as success. No owner, allocation, byte cap, ABI layout,
mode default or injected code changed. API comments document passive consumption.

The host C++ worker tests establish one execution for active, ready and pending-
behind-obsolete-copy requests, exact pixels/ownership, all four epoch rejections,
changed input rejection, failure atomicity/recovery and preserved reset/unit
interruption. Publication, navigation fixture and endpoint analysis checks pass
(20 tests). The actual worker test uses host clang; the build and replay below use
Windows x86 D3D11. No injected compile is required for these Renderer-only changes.

`pull-adoption-contract-build-20260912` is the successful current build.
`pull-adoption-ambient-small2-20260912/comparison.json` preserves two matched
640x480, width-128 boundary pairs in candidate/control then control/candidate order.
Each fresh process runs the same three stationary clocks, changed camera and queued
return to resident home content, with independent synchronous references. All 20
publication checks are exact with no fallback or recovery. Candidate traces show
the queued request finishing through the camera owner and being adopted; controls
replace it with an ordinary synchronous job. The deterministic host test, rather
than an assumed native scheduling race, proves avoidance of duplicate execution.

| Endpoint | Control repeats (ms) | Candidate repeats (ms) |
| --- | --- | --- |
| Queue begin through ordinary exact return | 15.244 / 15.111 | 14.731 / 14.057 |
| Changed camera through exact return | 652.196 / 629.174 | 630.295 / 626.545 |
| Maximum stationary ordinary call | 0.549 / 0.367 | 0.425 / 0.526 |

This supports structural retention with no material regression in the small
workload, not a meaningful speedup or native cadence claim. Queue timing includes
submission and waiting over a prebuilt immutable capture; live capture and actual
presentation are unmeasured. Sampled contiguous headroom is at least 2031.2 MiB;
internal transient/driver residency is unmeasured. Inputs match across arms, the
only native source difference is `c3x_renderer.cpp`, and all source/input/binary
verification passes. Runs were serial one-shot fixtures without session
exclusivity-watchdog coverage. Control source was restored to the candidate after
each run. Changed-camera reconstruction remains expensive; this handoff does not
solve it. No staging, installation, game launch or live-game acceptance occurred.

### Publication-owned front validity — structurally retained

`PublishedMapFrame::matches_static_view` now checks the front's own exact captured
occurrences, camera/world basis and topology revision before the ambient legacy
path can return it. The existing immutable request comparison separately checks
full topology bytes. An active job matching the caller no longer certifies a front
from a different view. No allocation, budget, native code, shader or mode default
changed for this correction.

The six publication/worker tests pass on the host C++ path. The new mixed-interface
case fails against the previous guard (`!returned.load()`), reproducing old-camera
return while the new camera is in flight. It passes with the correction, along
with passive next-call completion consumption and no repeated execution of an
unchanged clock. The initial test timeout came from reusing the earlier fixture's
two-record array instead of the intended changed tile; the fixture was corrected.

`pull-publication-build-20260912` is the successful isolated native build.
`pull-publication-ambient-small-20260912/comparison.json` compares it with the
accepted control on the existing 640x480, width-128 ambient boundary: three
stationary clocks plus a changed camera, all exact against independent synchronous
references. Runtime inputs match; the only native source difference is this guard;
all run inputs/sources/binaries verify unchanged. Maximum measured stationary
calls were 0.394 ms candidate / 0.359 ms control. Changed-camera exact completion
was 610.993 / 614.073 ms. Sampled contiguous headroom was at least 2034.8 MiB.
One pair supports no material regression at this scope, not a speedup/tail claim.
Both runs were serial one-shot fixtures without session exclusivity-watchdog
coverage; internal transient/driver residency is unmeasured. The earlier attempt
to reuse an old build directly was correctly rejected by source-identity checks;
the valid control was rebuilt/reverified, and the candidate source was restored.
The wrapper only exposes and validates the existing ambient boundary fixture.
No Civ III launch, staging, injected build or native presented-frame claim occurred.

The helper-led natural-mesh task was not implemented. Its final small profiling
probe, `natural-mesh-edit-phase-small-20260912`, passed four exact edit/full-redraw
checks and input verification. Visible edits spent about 357 ms in natural ground
and 355–356 ms in relief, within a 757–758 ms natural phase. This is preserved
attribution, not a task selection. The river-field experiment is closed below.

### Local river-field inputs — rejected after the small exact comparison

`local-river-input-build-20260912` implemented an opt-in dependency snapshot for
the existing 16-page owner, including height callback inputs beyond its nominal
halo. The candidate reserved at most 256 KiB of snapshot payload across the cache;
page ownership and immutable held handles remained bounded. CPU parity covered
distant edits, river removal/return, relief beyond the source halo, coast, wrapping,
owner/dimension changes and reset, alongside the existing 312-sample/LRU witness.

`local-river-edit-small-candidate-20260912/comparison.json` records one matched
640x480 pair over the complete 100x100 world, width 128, fixed clock, waves and
reflections off, with accepted shadow/local-region paths enabled in both arms.
All eight edit results passed independent exact full redraws, ownership and
animation counts. Full input/source/binary verification matched and no competing
GPU/build process was observed. Candidate retained 10/10 pages on distant changes
and 6/10 on visible changes, versus zero for control.

Despite that reuse, mean visible edit latency was 1274.791 ms candidate versus
1256.385 ms control; mean complete four-edit latency was 648.291 / 639.794 ms.
Distant edits were 21.792 / 23.203 ms. These single-pair differences do not establish
a regression or tail behavior, but show no useful small-workload gain and do not
earn dense repetition. Geometry remained about 1.1 seconds on visible edits.
The compiled river-field work was not a material part of that cost in this fixture.
The candidate, its switch and candidate-only tests were removed. Source snapshots
and the worktree patch are preserved under the build's `rejected-source/`;
the accepted shadow/local-region implementation is unchanged. No broad category
or larger GPU campaign was run for this rejected candidate. Host-boundary memory
samples remain in the receipt; internal transient/GPU-query coverage is unmeasured.

### Local region validity — retained after dense exact/performance validation

`local-region-dense-edit-build-20260912` is the validated isolated build. Its
renderer is unchanged from the small fixture; the preview adds host-boundary
address-space samples outside the measured edit interval.

`local-region-edit-dense-candidate2-20260912/comparison.json` records two matched
pairs in alternating arm order: 2240x1192, width 128, fixed clock, waves/reflections
off, normal budgets and the accepted shadow path in both arms. The complete
100x100 world supplies 1,921 captured occurrences, including 5 cities, 349 roads,
80 rails, 344 improvements and 150 resources. Every distant/visible edit and
reversal matches independent full-redraw pixels, ownership and animation counts.
Inputs, sources and binaries match across all four runs; full verification passed
and no competing GPU/build process was observed.

| Mean edit latency | Control repeats (ms) | Candidate repeats (ms) |
| --- | --- | --- |
| Complete four-edit sequence, per edit | 1206.060 / 1181.455 | 1002.073 / 983.028 |
| Distant edits | 465.394 / 424.929 | 102.641 / 115.685 |
| Visible edits | 1946.727 / 1937.980 | 1901.506 / 1850.370 |

Whole-sequence paired savings are 203.987 / 198.427 ms per edit, or 16.9 / 16.8%.
For distant edits, 66 unchanged animation backdrops are reused instead of rebuilt.
Visible edits reuse 21 backdrops and rebuild 45, while both arms rebuild 81 geometry
tiles. These counts explain the measured result; they are not the retention gate.
The minimum sampled contiguous address-space headroom is 1,613.375 MiB. Sampling
is at host request boundaries; internal transient/GPU-query coverage is unmeasured.
The fixed mix of two distant and two visible edits is a diagnostic workload,
not an estimate of their frequency in real gameplay.

`local-region-navigation-candidate-20260912/comparison.json` is one matched
unaffected-navigation regression pair: 193.386 ms control / 193.833 ms candidate
on the 14-offset dense reversal, with identical saved images/exact revisits,
matched inputs/sources/binaries and no observed GPU conflict. Minimum sampled
contiguous headroom is 1,692.859 MiB. This establishes no material regression in
that pair, not a new navigation speedup or a long-tail claim. Current resource
checks pass (132 tests, one existing skip). Retain the local-region implementation
under its existing control; broader coast/river/effect/lifecycle pixel coverage
and native presentation remain unpassed scopes, not implied by flat-land edits.

`local-region-edit-isolated-build-20260912` built the initial opt-in
`C3X_RENDERER_LOCAL_REGION_REVISIONS` candidate. Geometry still validates each
topology change; only the retained-region context substitutes local contributor/
shadow dependency validity for the global topology revision. Device/content,
projection, lighting, wrap and reset contracts remain in place. No cache cap or
native ownership changed. The same context governs animation backdrop reuse.

The existing preview now has a focused topology-edit fixture: one distant and
one visible flat-land edit, each reversed, at fixed view/time, with independent
reset/full-redraw pixels and ownership for every step. Its receipt separates
edit-through-copied-result timing from untimed full-redraw verification and
explicitly leaves native presentation/internal transient memory unmeasured.
The first setup did not launch a case; the next accidentally included the generic
animation prelude and is invalid. Both fixture issues are corrected. The region/
fixture checks pass (11), using the bundled Python for the Pillow-dependent test;
the existing fixture extraction was corrected for its changed animation condition.

`local-region-edit-small-base-isolated-20260912` and
`local-region-edit-small-candidate-20260912` pass all four exact checks at 640x480,
width 128, over the complete 100x100 world. Inputs, sources and binaries match;
all runtime verification passes and no competing GPU/build process was observed.
The accepted shadow path is enabled in both arms. Their `comparison.json` is
one small matched pair, not repeated dense acceptance.

Distant edit/reversal latency is 45.636 / 46.296 ms control versus
24.434 / 24.973 ms candidate. Main-map drawing was already avoided in the control;
the eliminated work is rebuilding six unaffected animation backdrops. Candidate
backdrop hits/misses are 6/0 versus control 0/6. Visible edit/reversal is
1347.946 / 1359.989 ms versus 1317.981 / 1338.462 ms. Both build 80 geometry tiles;
geometry alone takes roughly 1.12–1.14 seconds. Those counts explain remaining
work, not a performance pass. That small comparison alone was provisional. The dense repeated edit and
unaffected-navigation results above now close this experiment at their stated
scope. Broader dependency/effect/lifecycle coverage remains to be earned.

### Resource shadow material variant — retained by user decision

`resource-shadow-material-build-20260912` is the final shadow-only isolated build.
`resource-shadow-material-boundary-20260912` passed all six independent full-
redraw comparisons with confirmed variant execution and unchanged inputs/sources/
binaries. Camera/guard movement, clock advance, removal and reappearance were exact.
The earlier resource checks passed (132 tests, one existing skip). The corrected
shader changes no source coverage, sample/depth order or rendering ownership.

`resource-shadow-material-reversal-candidate2-20260912/comparison.json` records
two profiling-off dense 14-request pairs, alternating arm order, fixed clock,
2240x1192, width 128, waves/reflections off and existing normal budgets. Controls
205.245 / 204.636 ms versus candidates 191.985 / 191.015 ms saved 13.260 / 13.621 ms.
All saved endpoint pixels/revisits were exact; complete inputs, sources and binaries
matched across the comparison; no competing GPU/build process was observed.
Sampled contiguous headroom was at least 1,690.660 MiB at host request boundaries;
internal transient/GPU-query coverage remains unmeasured with profiling off.

The analyzer classified this below its 20.494 ms usefulness threshold. The user
explicitly overrode that retention decision for this result. Keep the prepared
resource pass plus shadow variant under its existing scoped control and use it
as the candidate baseline in subsequent relevant experiments. This does not
retroactively accept failed body pixels, arrays, or unrelated historical probes,
or imply staging, install, launch or broader dynamic/effect acceptance.

The initial constant-material body/shadow candidate failed its first independent
full redraw: 77 pixels differed by up to two channel values.
`resource-material-boundary-first-20260912` preserves the mismatching images.
Restoring ordinary body shading isolated the issue: the shadow-only diagnostic
passed six checks. First shader-compiling runs changed runtime inputs and are
not stable-input acceptance. The body variant is removed, with no tolerance change.

### GPU region batch — rejected and removed

The candidate grouped up to four independently guarded texture-array layers for
body/shadow draws and reconstruction. It preserved each 136-pixel projection,
MSAA4 depth/sample contract and output crop, reserving 17,756,160 scratch bytes
inside the existing backdrop allowance. A revision removed a redundant cached-
background copy by restoring directly into the array slice. Neither version
established a useful whole-workload improvement.

`resource-gpu-batch-boundary-20260912` and the revised
`resource-gpu-batch-direct-boundary-20260912` passed six exact independent full-
redraw comparisons with actual batch execution and unchanged inputs/sources/
binaries. Resource checks passed (132 tests, one existing skip); focused region,
publication and budget checks passed (17). The first boundary run was exact but
changed shader cache inputs and is not acceptance evidence.

The initial four-column pair averaged 352.908 ms control / 381.435 ms candidate.
The deciding direct-copy 14-request repeats are in
`resource-gpu-batch-direct-reversal-candidate2-20260912/comparison.json`:
controls 196.547 / 194.388 ms, candidates 537.880 / 194.020 ms, paired savings
-341.333 / 0.368 ms. Saved images and revisits were exact, inputs/sources/binaries
verified unchanged, and no competing GPU/build process was observed. Sampled
contiguous headroom was at least 1,688.898 MiB at host request boundaries;
internal transient/GPU-query coverage was not measured with profiling off.

The first candidate sequence included a 3,754.416 ms readback wait; it did not
recur in the second sequence. Its cause is unestablished, so it is not discarded
as an environment fault. Even the unstalled repeat failed the usefulness gate.
Array code, adapter and test changes were removed; original shader outputs were
regenerated. The rejected source patch and new source files are preserved under
`resource-gpu-batch-direct-build-20260912/rejected-source/`. No production
optimization, navigation gate or live integration is claimed.

### Prepared resource pass — retained with the accepted shadow variant

The candidate now prepares ordered, region-specific body/shadow draw lists and
submits them through an explicit GPU pass using the existing shader/depth/guard
contract. Consecutive regions share exact source-shadow page demand in batches
within the existing 32-page atlas. A static backdrop miss invalidates the active
batch before animation resumes. The synchronous pass borrows pinned pose buffers;
its retained draw/page descriptions have a conservative 4 MiB allowance and fall
back to ordinary submission when exceeded. Wave frames retain ordinary submission
until their scene/depth pass contract is explicitly covered. No cache budget,
rendering ownership, staged binary or asset changes are involved.

`prepared-resource-batched-build-20260912` compiled successfully.
`prepared-resource-batched-boundary-20260912` passed all six exact comparisons
against independent ordinary-path full redraws, including camera/guard movement,
advancing clock, removal and reappearance; inputs/sources/binaries were unchanged.
`test resources` passed 132 tests with one existing skip. These establish the
initial correctness checkpoint, not dense navigation performance.

`prepared-resource-reversal-candidate2-20260912/comparison.json` closes the
preparation-only candidate: two profiling-off 14-request controls averaged
193.713 / 194.173 ms; candidates 198.170 / 193.230 ms. Paired savings were
-4.457 / 0.943 ms, below the 20 ms criterion. All four runs had exact saved
images/revisits, unchanged inputs/sources/binaries and no observed competing GPU
or compiler process. Host-boundary contiguous headroom was at least 1,691.730 MiB;
internal transient/GPU-query coverage remains unmeasured. The first four-column
candidate was exact but suffered a 3,841.913 ms readback wait (4,058.441 ms whole
request); the matched control was approximately 336 ms. That stall did not recur
in either reversal, but its cause is unestablished and it is not discarded as a
known environment fault. No useful performance pass or advancement gate is claimed.

The remaining dominant animation cost is GPU completion of regional rendering and
reconstruction, with per-request pose/selection work also present. Current preparation code
reduces setup calls but still executes the same serial region passes. The GPU-array
revision also failed, as recorded above. The preparation pass is now retained with the accepted shadow variant above,
under the same opt-in control. Its preparation-only organization is not separately
claimed as a speedup.
Focused region/publication/analysis checks passed 29 tests; the Pillow-dependent
check was rerun successfully using the bundled Python after system Python lacked
Pillow.

### Session tooling evidence (deliverable 2 complete)

Per-translation-unit include hashing and compiler/SDK stamps now reuse intact
objects. `session-incremental-seed-retry-20260912` compiled seven units in 37.345 s
wrapper time; `session-incremental-warm-20260912` compiled none in 3.108 s. Changing
only the preview compiled that unit alone. Recipe, dependency and object changes
reject reuse.

`session-four-verified-20260912/comparison.json` records four assets-loaded cases:
39.943 s wrapper total, 9.986 s per case versus 14.285 s for the matched fresh
one-shot. All initial/final images exactly match that independent fresh process;
full inputs, sources and binaries verified unchanged. This is about 30% less
iteration wait in this short comparison, not a renderer speedup or tail estimate.
Resets preserve budgets and clear mutable content/publication while retaining
assets/device. `session-prepared-verified-20260912/comparison.json` independently
matches the same fresh images after explicit warmup (5.056 s through its final
check), retaining 84.174 MB geometry and 30.628 MB ground content with no evictions.
Its single 146.326 ms small-view transition is not a navigation performance pass.

An earlier four-case run failed at teardown and entered Windows Error Reporting;
it remains invalid evidence. A subsequent watchdog exit-code reporting fault was
fixed and verified against success, nonzero exit and owned-process timeout cases.
The accepted runs above completed teardown under the bounded watchdog. The initial
transient teardown fault was not reproduced or localized; preserve that limitation
without reopening completed session validation as an unbounded investigation.

### Current evidence and limits

Ignored evidence under `Renderer/native/build/`:

- Bounded animation reconstruction is **rejected and removed**.
  `bounded-resource-boundary-20260912` passed six exact independent full-redraw
  checks; focused regressions passed (16), resource category tests passed (132,
  one existing skip), and navigation analysis/causal checks passed (16).
  Four-column time was 354.472 ms baseline versus 339.654 ms candidate, exact
  saved images but only 14.818 ms saved. Deciding profiling-off reversal repeats
  in `bounded-resource-reversal-candidate2-20260912/comparison.json` were
  195.805 / 198.330 ms baseline versus 189.115 / 190.932 ms candidate: only
  6.690 / 7.398 ms saved, below the 20 ms threshold. All four runs had identical
  saved endpoint pixels, exact revisits, unchanged inputs/sources/binaries and
  no observed GPU/build conflicts. Host-boundary contiguous headroom was at least
  1,689.332 MiB; internal transient/GPU-query coverage was unmeasured with profiling
  off. Reduced postprocessing area was not a sufficient architectural improvement.
  Advancing-clock expansion was not earned. The existing analyzer now supports
  explicitly requested profiling-off cases with constructor/host-only coverage
  labels and rejects accidental mixing with profiled cases.


- Static-pass sharing is **rejected and removed**. The preserved
  `shared-static-build-20260912` / `shared-static-boundary-20260912` passed six
  exact independent full-redraw comparisons (pixels, replacement flags and
  animation counts). Four-column paired repeats were 392.812 / 405.994 ms baseline
  versus 329.058 / 332.316 ms candidate, a 16.2–18.1% saving. However,
  `shared-static-reversal-candidate2-20260912/comparison.json` records the deciding
  14-request sequence: baseline 227.392 / 249.731 ms, candidate 244.933 / 246.393 ms.
  Paired savings were -17.541 / 3.338 ms, below the 23.856 ms usefulness threshold.
  All four sequence runs matched saved endpoint pixels, exact revisits, unchanged
  inputs/sources/binaries and zero observed GPU/build conflicts; minimum sampled
  contiguous headroom was 1,694.574 MiB. No budget was increased. Handoffs were
  often unused (114 of 180 initial captures); reduced duplicate submission did
  not yield a whole-sequence win. Advancing-clock dense expansion was not earned.
  The earlier failed baseline dispatch produced no child and remains invalid;
  its completed retry is `shared-static-four-base2-retry-20260912`.
- `retained-production-cost-20260912/production-cost.json`: one profiling-off
  14-request sequence against the cleaned renderer averages 193.617 ms. Phase
  means are geometry 34.906, submission 6.646, static readback 27.236 and animation
  composition 114.019 ms. Revisit geometry is 11.474–12.106 ms; revisit animation
  is 68.844–80.381 ms. All saved images match the profiled baseline and revisits
  are exact. This isolates production work; turning off diagnostic memory scans
  is not a renderer improvement. It is a single scoped cost check, not a repeated
  performance gate, independent full-redraw oracle or live-game measurement.
  Profiling-off omits per-frame address-space/GPU-query coverage; the separate
  profiled runs above preserve headroom evidence for the same budgets/workload.
- `retained-boundary-clean-20260912`: cleaned renderer build reuses the original
  DLL translation unit and rebuilds the retained preview fixture. Region/cache/
  causal tests pass (30); resource category tests previously passed (132, one
  existing skip). `retained-boundary-clean-check-20260912` passes all six exact
  reset/full-redraw comparisons with unchanged inputs/sources/binaries. The
  temporary handoff retirement test was removed with its
  rejected owner, preserving the original budget/LRU behavior test. No injected
  edit, staging, installation, game launch or reference replacement.

- `dense-route-current-20260912/results.json`: completed automatic batch, two
  repetitions in alternating arm order, 2240×1192, tile width 128, fixed clock,
  waves/reflections off, normal retained budgets. The 100×100 world supplies
  1,921 captured occurrences with 5 cities, 349 roads, 80 rails, 344 improvements
  and 150 resources. All 20 cases verified unchanged inputs/sources/binaries and
  no observed competing GPU/build process; minimum sampled contiguous headroom
  1,681.852 MiB. Repeated endpoints are exact; prepared-content endpoints match
  fresh-content rendering. Intermediate independent full-redraw parity is still
  unmeasured, and omitted-pixel arms cannot establish production correctness.

  | Arm | Four-column moves (ms) | Reversal mean per request (ms) | Decision |
  | --- | --- | --- | --- |
  | Full | 391.963 / 402.839 | 251.355 / 249.466 | Target unmet |
  | Route draws omitted | 394.056 / 396.714 | 243.751 / 250.359 | Reject primary target |
  | Route surfaces omitted | 405.555 / 406.747 | 245.846 / 250.916 | Reject primary target |
  | Prepared content | 345.472 / 309.586 | 224.613 / 227.262 | Useful four-column effect; reversal inconclusive at 10% threshold |
  | Half geometry pixels | 341.074 / 347.185 | 223.917 / 224.520 | Useful four-column diagnostic; reversal inconclusive at 10% threshold |

  Baseline reversal phase means: geometry 62.799 ms, submission 20.311 ms,
  readback 27.963 ms, animation composition 114.886 ms. Existing traces and source
  inspection show animated-region backdrop misses resubmit static geometry after
  main-map rendering. That observation led to the now-rejected sharing candidate above; the
  diagnostic batch itself established neither correctness nor speedup.


- `dense-route-quiet-20260912`: two uncontested, fully verified four-column cases
  completed on the 2240×1192 dense fixture, tile width 128, waves/reflections off,
  existing retained defaults and normal budgets. The 100×100 world supplied 1,921
  captured occurrences: 5 cities, 349 roads, 80 rails, 344 improvements and 150
  resources. Full rendering took 398.012 ms; route surface draws omitted took
  395.704 ms. Their pixels differ as intended. These are single samples, not a
  matched-repeat decision: route drawing showed no large effect in this pair.
  Full-render phases were geometry 114.525, submission 31.905, readback 71.509
  and animation composition 150.163 ms; sampled contiguous headroom exceeded
  1.8 GiB. Initial setup remains separately recorded. Runtime shader changes and
  another `native_preview` interrupted the third arm; it was rejected, and no
  performance conclusion is drawn from that arm or the incomplete matrix.
- `dense-route-diag2-20260912` contains the earlier overlapping batch, explicitly
  invalid for causal timing. The overlap watchdog was then added and correctly
  rejected `dense-route-serial-20260912` before rendering. It monitors known Lab
  renderer/compiler processes once per second, records conflicts and terminates
  only its own child. All diagnostic children are terminal. An initial startup
  exit in `dense-route-diagnostic-20260912` was not reproduced; the preview now
  reports the exact setup precondition on any recurrence.
- `dense-diag-setup-20260912`: isolated x86 build passed, with benchmark-only route
  surface and half-pixel controls. The category dispatcher exposes the fixed
  batch and the existing analyzer owns its comparisons. Focused causal/endpoint/
  build/damage tests passed (19); fixture/trace/worker checks passed (8).
  `test infrastructure` passed 123 tests with one existing skip. No injected edit,
  staged binary, reference replacement, installation or game launch.
- `endpoint-short-20260912`: 640×480, width 128, waves off, fixed-clock 14-offset
  reversal. All 14 requests and exact revisits passed, with zero fallback/recovery.
  Capture median 0.371 ms; request-to-checked-result median 164.303 ms, range
  31.628–858.733 ms. Initial render 5.550 s; before/after input verification 7.010 s;
  wrapper total 18.389 s. Revisit equality is repeatability, not full-redraw parity.
- `endpoint-four-column-20260912/analysis.json`: completed final endpoint layout,
  same small viewport and one four-column move. 759 captured occurrences, 728.304 ms
  request-to-checked-result: capture 0.447, worker rendering 725.935, queue 0.053,
  snapshot/drain 0.773, publication/preparation 0.905 ms. Within worker rendering,
  geometry 607.059, submission 14.240, readback 57.834, animation composition 39.297,
  unaccounted remainder 7.505 ms. The final delayed GPU sample is unmeasured; no
  blocking query was added. Native capture/presentation and independent pixel
  parity remain outside this tooling check. Full runtime input/binary verification
  passed; both builds produced the same initial image hash.
- `endpoint-accounting-complete-20260912/build-evidence.json`: isolated x86 `/O2`
  build passed. Compiler setup 2.110 s, DLL compile/link 22.620 s, preview 4.860 s;
  outer compile/dispatch 30.462 s. No staging, install, game launch or injected edit.
- Endpoint ordering/missing-query/invalid-span analysis, bounded trace behavior,
  dense fixture identity and delayed GPU telemetry tests pass (15 tests). Category
  dispatcher `test infrastructure`: 123 tests passed, one existing skip. The
  system Python 3.9 cannot run the telemetry test's newer tempfile option; the
  bundled workspace Python ran it successfully without changing that test.

**Dominant measured costs:** animation completion remains the largest phase in
ordinary dense navigation (about 114 ms in the earlier profiling-off trace).
Visible topology edits are dominated by geometry preparation: a representative
accepted-candidate trace reports 262 ms ground preparation, 773 ms natural-
geometry preparation and 100 ms upload for 81 rebuilt tiles. The river-field
hypothesis was subsequently rejected above; these phase counts do not select the
next task. Persistent appearance, compiled-content bindings and compact occurrence records are
retained. A compatible static batch pass is selected to consume those records and
reduce actual submission work; whole-request performance acceptance remains separate.
Local region validity now preserves unaffected backdrops on world edits; it does
not eliminate geometry rebuilding or establish fast native presentation.

The selected rendering migration remains retained route/object composition:
translate the valid static front, compose exposed strips/dirty bounds and keep
ambient/unit invalidation independent. Extend persistent content and local
dependencies where the diagnostics justify them. This is a step toward the
architecture, not its permanent definition. If evidence rejects the selected
causal target, replace this record's next task with one supported alternative;
do not start a competing sequence or the architecture's entire mechanism list.

| Capability / target | Evidence carried forward | Remaining limit |
| --- | --- | --- |
| Stationary retained front | Opt-in same-view ambient publication and independent cached-pose CPU composition are implemented; prior standalone minute-long 24-unit run reports 1.720 ms p95 render entry and 7.42 Hz fresh maps. | Still captures full native view; this is not live presentation or fast navigation. |
| Useful preparation | Existing `prefetch-guard2-replay-20260910/benchmark.log` records 176 prepared tiles before its first jump and exact return parity. | First three jumps were 707.534, 828.214 and 785.799 ms; short diagnostic, not a sustained gate. |
| Dense resident navigation | Retained overlap and bounded working sets already exist; preserve them. | Below-100 ms gate unmet; route/object composition and synchronous completion remain the selected causal target. |
| Regional geometry / zoom | Some structural sharing exists. | General regional batching shared across 128/160/192 is unfinished. |
| Native asynchronous presentation | Versioned publication and stale-ticket rejection exist in the DLL. | Native coordination, current-camera presentation and input-to-visible performance remain unverified. |
| Iteration tooling | Timing, incremental compilation, persistent sessions and the fixed dense causal batch are complete. | Additional tooling requires a specific missing correctness/measurement question for the selected rendering change. |

The preparation receipt is under `Renderer/native/build/`; older stationary
measurements are preserved in the linked experiment archive. These are scoped inherited
results, not a newly matched baseline/candidate comparison or current-binary
certification. No universal percentage-complete or overall speedup is claimed.

Preserve the staged DLL, rollback, licensed assets, source findings and existing
uncommitted renderer/build changes. Do not promote experimental options or alter
cache budgets, visual ownership or references during tooling work. Record future
progress here as capability, workload, evidence, unmet target and one next action.

## Preserved evidence

The [experiment archive](history/retained_experiments_20260910.md) retains the
original protocols, measurements and failed hypotheses. Read only the evidence
needed for the current mechanism; its embedded next-step instructions are historical.
